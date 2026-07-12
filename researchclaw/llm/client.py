"""Lightweight OpenAI-compatible LLM client — stdlib only.

Features:
  - Model fallback chain (gpt-5.2 → gpt-5.1 → gpt-4.1 → gpt-4o)
  - Auto-detect max_tokens vs max_completion_tokens per model
  - Cloudflare User-Agent bypass
  - Exponential backoff retry with jitter
  - JSON mode support
  - Streaming disabled (sync only)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens
_NEW_PARAM_MODELS = frozenset(
    {
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3",
        "gpt-5.4",
    }
)

_NO_TEMPERATURE_MODELS = frozenset(
    {
        "o3",
        "o3-mini",
        "o4-mini",
    }
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_MAX_BACKOFF_SEC = 300  # 5-minute ceiling for retry delays


@dataclass
class LLMResponse:
    """Parsed response from the LLM API."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    base_url: str
    api_key: str
    wire_api: str = "chat_completions"
    primary_model: str = "gpt-4o"
    fallback_models: list[str] = field(
        default_factory=lambda: ["gpt-4.1", "gpt-4o-mini"]
    )
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 2.0
    timeout_sec: int = 300
    user_agent: str = _DEFAULT_USER_AGENT
    # MetaClaw bridge: extra headers for proxy requests
    extra_headers: dict[str, str] = field(default_factory=dict)
    # MetaClaw bridge: fallback URL if primary (proxy) is unreachable
    fallback_url: str = ""
    fallback_api_key: str = ""


class LLMClient:
    """Stateless OpenAI-compatible chat completion client."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model_chain = [config.primary_model] + list(config.fallback_models)
        self._anthropic = None  # Will be set by from_rc_config if needed

    @staticmethod
    def _normalize_wire_api(wire_api: str) -> str:
        return wire_api.strip().lower()

    def _endpoint_url(self, base_url: str) -> str:
        wire_api = self._normalize_wire_api(self.config.wire_api)
        base = base_url.rstrip("/")
        if wire_api == "responses":
            return f"{base}/responses"
        return f"{base}/chat/completions"

    @staticmethod
    def _supports_temperature(model: str) -> bool:
        return not any(model.startswith(prefix) for prefix in _NO_TEMPERATURE_MODELS)

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> LLMClient:
        """Build LLMClient from a ResearchClaw RCConfig object."""
        from researchclaw.config import RCConfig

        if not isinstance(rc_config, RCConfig):
            raise TypeError(f"Expected RCConfig, got {type(rc_config)}")

        llm_cfg = rc_config.llm
        api_key = llm_cfg.api_key or os.environ.get(llm_cfg.api_key_env, "")

        # Detect Anthropic provider and use adapter
        provider = llm_cfg.provider.lower()
        if provider == "anthropic":
            from researchclaw.llm.anthropic_adapter import AnthropicAdapter

            client = cls(config=LLMConfig(
                base_url="",
                api_key=api_key,
                primary_model=llm_cfg.primary_model,
                fallback_models=list(llm_cfg.fallback_models),
                max_tokens=4096,
                temperature=0.7,
                max_retries=3,
                timeout_sec=300,
            ))
            client._anthropic = AnthropicAdapter(api_key=api_key)
            return client

        return cls(config=LLMConfig(
            base_url=llm_cfg.base_url,
            api_key=api_key,
            wire_api=llm_cfg.wire_api,
            primary_model=llm_cfg.primary_model,
            fallback_models=list(llm_cfg.fallback_models),
            max_tokens=4096,
            temperature=0.7,
            max_retries=3,
            timeout_sec=300,
        ))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        strip_thinking: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request with model fallback.

        Parameters
        ----------
        messages:
            List of message dicts with 'role' and 'content'.
        system:
            Optional system prompt (prepended to messages).
        json_mode:
            Request JSON response format.
        max_tokens:
            Override configured max_tokens.
        temperature:
            Override configured temperature.
        strip_thinking:
            If True, strip <think>…</think> reasoning blocks from the response.
        """
        _messages = list(messages)
        if system:
            _messages.insert(0, {"role": "system", "content": system})

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        last_exc = None
        for model in self._model_chain:
            try:
                resp = self._call_with_retry(m, messages, max_tok, temp, json_mode)
                if strip_thinking:
                    from researchclaw.utils.thinking_tags import strip_thinking_tags

                    return LLMResponse(
                        content=strip_thinking_tags(resp.content),
                        model=resp.model,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        total_tokens=resp.total_tokens,
                        finish_reason=resp.finish_reason,
                        truncated=resp.truncated,
                        raw=resp.raw,
                    )
                return resp
            except Exception as exc:
                last_exc = exc
                logger.warning(f"Model {model} failed: {exc}. Trying next.")
                continue

        raise RuntimeError(f"All models failed. Last error: {last_exc}")

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Call API with exponential backoff retry, falling back through model chain."""
        last_exc = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._single_call(
                    model, messages, max_tokens, temperature, json_mode
                )
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_base_delay ** (attempt + 1),
                        _MAX_BACKOFF_SEC,
                    )
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): "
                        f"{exc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError(f"All retries exhausted. Last error: {last_exc}")

    def _single_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Make a single API call."""

        # Use Anthropic adapter if configured
        if self._anthropic:
            data = self._anthropic.chat_completion(
                model, messages, max_tokens, temperature, json_mode
            )
        else:
            # Original OpenAI logic
            # Copy messages to avoid mutating the caller's list (important for
            # retries and model-fallback — each attempt must start from the
            # original, un-modified messages).
            msgs = [dict(m) for m in messages]

            # MiniMax API requires temperature in [0, 1.0]
            _temp = temperature
            if "api.minimaxi.com" in self.config.base_url or "api.minimax.io" in self.config.base_url:
                _temp = max(0.0, min(_temp, 1.0))

            if self._normalize_wire_api(self.config.wire_api) == "responses":
                body = self._build_responses_body(model, msgs, max_tokens, _temp)
            else:
                body = {
                    "model": model,
                    "messages": msgs,
                }
                if self._supports_temperature(model):
                    body["temperature"] = _temp

                # Use correct token parameter based on model
                # DeepSeek v4-pro is a reasoning model and may need
                # max_completion_tokens to properly control output length.
                if any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS) or "v4-pro" in model:
                    reasoning_min = 32768
                    body["max_completion_tokens"] = max(max_tokens, reasoning_min)
                else:
                    body["max_tokens"] = max_tokens

            if json_mode:
                # Many OpenAI-compatible providers don't support the
                # response_format parameter and return HTTP 400.
                # Fall back to system-prompt injection for known-incompatible
                # models (Claude, DeepSeek, Qwen, etc.) and the responses API.
                _model_lower = model.lower()
                _no_response_format = (
                    _model_lower.startswith("claude")
                    or _model_lower.startswith("deepseek")
                    or _model_lower.startswith("qwen")
                    or _model_lower.startswith("yi-")
                    or _model_lower.startswith("glm")
                    or _model_lower.startswith("moonshot")
                    or _model_lower.startswith("minimax")
                    or _model_lower.startswith("doubao")
                    or _model_lower.startswith("abab")
                    or _model_lower.startswith("hunyuan")
                    or _model_lower.startswith("ernie")
                    or _model_lower.startswith("spark")
                    or _model_lower.startswith("gemma")
                    or self._normalize_wire_api(self.config.wire_api) == "responses"
                )
                if _no_response_format:
                    _json_hint = (
                        "You MUST respond with valid JSON only. "
                        "Do not include any text outside the JSON object."
                    )
                    # Prepend to existing system message or add as new one
                    if msgs and msgs[0]["role"] == "system":
                        msgs[0]["content"] = _json_hint + "\n\n" + msgs[0]["content"]
                    else:
                        msgs.insert(0, {"role": "system", "content": _json_hint})
                else:
                    body["response_format"] = {"type": "json_object"}

            payload = json.dumps(body).encode("utf-8")
            url = self._endpoint_url(self.config.base_url)

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            }
            # MetaClaw bridge: inject extra headers (session ID, stage info, etc.)
            headers.update(self.config.extra_headers)

            req = urllib.request.Request(url, data=payload, headers=headers)

            try:
                with urllib.request.urlopen(
                    req, timeout=self.config.timeout_sec
                ) as resp:
                    data = json.loads(resp.read())
            except (urllib.error.URLError, OSError) as exc:
                # MetaClaw bridge: fallback to direct LLM if proxy unreachable
                if self.config.fallback_url:
                    logger.warning(
                        "Primary endpoint unreachable, falling back to %s: %s",
                        self.config.fallback_url,
                        exc,
                    )
                    fallback_url = self._endpoint_url(self.config.fallback_url)
                    fallback_key = self.config.fallback_api_key or self.config.api_key
                    fallback_headers = {
                        "Authorization": f"Bearer {fallback_key}",
                        "Content-Type": "application/json",
                        "User-Agent": self.config.user_agent,
                    }
                    fallback_req = urllib.request.Request(
                        fallback_url, data=payload, headers=fallback_headers
                    )
                    with urllib.request.urlopen(
                        fallback_req, timeout=self.config.timeout_sec
                    ) as resp:
                        data = json.loads(resp.read())
                else:
                    raise

        if not isinstance(data, dict):
            raise ValueError(
                f"Malformed API response: expected JSON object, got {type(data).__name__}: {data}"
            )

        # Handle API error responses
        if "error" in data and data["error"] is not None:
            error_info = data["error"]
            if isinstance(error_info, dict):
                error_msg = str(error_info.get("message", str(error_info)))
                error_type = str(error_info.get("type", "api_error"))
            else:
                error_msg = str(error_info)
                error_type = "api_error"
            import io

            raise urllib.error.HTTPError(
                "",
                500,
                f"{error_type}: {error_msg}",
                None,
                io.BytesIO(error_msg.encode()),
            )

        if self._normalize_wire_api(self.config.wire_api) == "responses":
            return self._parse_responses_response(data, model)
        return self._parse_chat_completions_response(data, model)

    def _build_responses_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": self._messages_to_responses_input(messages),
        }
        if self._supports_temperature(model):
            body["temperature"] = temperature
        body["max_output_tokens"] = max_tokens
        return body

    def _messages_to_responses_input(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user") or "user")
            content = str(message.get("content", "") or "")
            items.append({"role": role, "content": content})
        return items

    def _parse_chat_completions_response(
        self, data: dict[str, Any], model: str
    ) -> LLMResponse:
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Malformed API response: missing choices. Got: {data}")

        choice = data["choices"][0]
        usage = data.get("usage", {})

        message = choice.get("message", {})
        content = message.get("content") or ""
        # DeepSeek v4 Pro may return empty content when reasoning consumes
        # all output tokens. Try to salvage JSON from reasoning_content.
        if not content:
            reasoning = message.get("reasoning_content") or ""
            if reasoning:
                # Look for JSON object/array in the reasoning tail
                import re as _re
                _json_candidates = _re.findall(r'\{[^{}]*\{[^}]*\}[^{}]*\}|\[[^\[\]]*\[[^\]]*\][^\[\]]*\]|\{[^}]*\}|\[[^\]]*\]', reasoning)
                for _c in reversed(_json_candidates):
                    try:
                        json.loads(_c)
                        content = _c
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
                if not content:
                    # Last resort: try balanced-brace extraction from the tail
                    _brace_depth = 0
                    _start = -1
                    for _i in range(len(reasoning) - 1, -1, -1):
                        if reasoning[_i] == '}':
                            if _brace_depth == 0:
                                _end = _i
                            _brace_depth += 1
                        elif reasoning[_i] == '{':
                            _brace_depth -= 1
                            if _brace_depth == 0:
                                content = reasoning[_i:_end + 1]
                                break

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            truncated=(choice.get("finish_reason", "") == "length"),
            raw=data,
        )

    def _parse_responses_response(
        self, data: dict[str, Any], model: str
    ) -> LLMResponse:
        output_items = data.get("output")
        if not isinstance(output_items, list):
            raise ValueError(
                f"Malformed responses API payload: missing output. Got: {data}"
            )
        if not output_items:
            # Empty output list — API returned no content (e.g. reasoning-only
            # response, empty completion).  Return empty response instead of
            # crashing so the model-fallback loop can try the next model.
            return LLMResponse(content="", model=model)

        chunks: list[str] = []
        finish_reason = str(data.get("status", "") or "")
        truncated = False

        for item in output_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)

        incomplete_details = data.get("incomplete_details")
        if isinstance(incomplete_details, dict):
            reason = incomplete_details.get("reason")
            if reason == "max_output_tokens":
                truncated = True
                finish_reason = "length"

        return LLMResponse(
            content="".join(chunks),
            model=model,
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            prompt_tokens=data.get("usage", {}).get("input_tokens", 0),
            completion_tokens=data.get("usage", {}).get("output_tokens", 0),
            finish_reason=finish_reason,
            truncated=truncated,
            raw=data,
        )