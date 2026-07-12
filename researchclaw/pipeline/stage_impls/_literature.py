"""Stages 3-6: Search strategy, literature collection, screening, and knowledge extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_fallback_queries,
    _chat_with_prompt,
    _extract_topic_keywords,
    _extract_yaml_block,
    _get_evolution_overlay,
    _parse_jsonl_rows,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_jsonl,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            # --- Format A: new pure-JSON search_plan (preferred) ---
            raw_plan = payload.get("search_plan")
            if isinstance(raw_plan, dict):
                plan = raw_plan
                logger.info("Stage 3: parsed search_plan as pure JSON (new format)")
            # --- Format B: legacy YAML-inside-JSON search_plan_yaml ---
            if plan is None:
                yaml_text = str(payload.get("search_plan_yaml", "")).strip()
                if yaml_text:
                    parsed = None
                    try:
                        parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                    except yaml.YAMLError:
                        # Auto-fix: unquoted topic string with colon+space breaks YAML.
                        # Wrap the topic value in quotes and retry.
                        _fixed = re.sub(
                            r'^(topic:\s*)(.+)$',
                            r'\1"\2"',
                            _extract_yaml_block(yaml_text),
                            count=1,
                            flags=re.MULTILINE,
                        )
                        try:
                            parsed = yaml.safe_load(_fixed)
                            logger.info(
                                "Stage 3: auto-fixed unquoted topic in YAML, retry succeeded"
                            )
                        except yaml.YAMLError:
                            parsed = None
                    if isinstance(parsed, dict):
                        plan = parsed
                        logger.info("Stage 3: parsed search_plan from legacy YAML format")
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    model_plan_parsed = plan is not None
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        _fallback_queries = _build_fallback_queries(topic)
        n = len(_fallback_queries)
        # Distribute queries across 3 strategy categories
        chunk_size = max(3, min(5, n // 3 + 1))
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "core_methods",
                    "queries": _fallback_queries[:chunk_size],
                    "sources": ["arxiv", "semantic_scholar", "openalex"],
                    "max_results_per_query": 40,
                },
                {
                    "name": "application_domain",
                    "queries": _fallback_queries[chunk_size:chunk_size * 2],
                    "sources": ["arxiv", "semantic_scholar", "openalex"],
                    "max_results_per_query": 40,
                },
                {
                    "name": "surveys_and_theory",
                    "queries": _fallback_queries[chunk_size * 2:],
                    "sources": ["semantic_scholar", "arxiv"],
                    "max_results_per_query": 40,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search
    queries_list: list[str] = []
    year_min = 2020
    if isinstance(plan, dict):
        strategies = plan.get("search_strategies", [])
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        queries_list.extend(str(q) for q in qs if q)
        # Also accept the alternate schema where queries live under
        # query_strategies.<sub_question>.{boolean_seeds, queries}.
        if not queries_list:
            qstrats = plan.get("query_strategies", {})
            if isinstance(qstrats, dict):
                for sub in qstrats.values():
                    if not isinstance(sub, dict):
                        continue
                    for key in ("boolean_seeds", "queries"):
                        qs = sub.get(key, [])
                        if isinstance(qs, list):
                            queries_list.extend(str(q) for q in qs if q)
        filters = plan.get("filters", {})
        if isinstance(filters, dict) and filters.get("min_year"):
            try:
                year_min = int(filters["min_year"])
            except (ValueError, TypeError):
                pass

    # --- Sanitize queries: shorten overly long queries ---
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
        "challenge", "challenges", "gaps", "gap", "critical", "survey", "review",
    }

    def _extract_search_terms(text: str) -> list[str]:
        """Extract meaningful search terms from text, removing stop words."""
        return [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]

    _MAX_QUERY_LEN = 60
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        kws = _extract_search_terms(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    def _build_default_search_queries(topic_text: str) -> list[str]:
        """Generate concept-style search queries from the topic."""
        _words = _extract_search_terms(topic_text)
        if not _words:
            return [topic_text[:60]]
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        kw_alt = " ".join(_words[1:5]) if len(_words) > 4 else kw_short
        return [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
            kw_alt,
            f"{kw_short} recent advances",
        ]

    fell_back_to_defaults = False
    if not queries_list:
        queries_list = _build_default_search_queries(topic)
        fell_back_to_defaults = True

    _all_kw = _extract_search_terms(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    silent_fallback = fell_back_to_defaults and model_plan_parsed
    if silent_fallback:
        logger.warning(
            "Stage 3: model plan parsed but no queries harvested; "
            "queries.json fell back to topic-derived defaults"
        )
    queries_meta = {
        "queries": queries_list,
        "year_min": year_min,
        "model_queries_extracted": model_plan_parsed and not fell_back_to_defaults,
        "fallback_reason": (
            "model_plan_used_unknown_schema" if silent_fallback else None
        ),
    }
    (stage_dir / "queries.json").write_text(
        json.dumps(queries_meta, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )