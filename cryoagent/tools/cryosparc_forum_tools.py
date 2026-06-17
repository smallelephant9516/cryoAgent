"""Search CryoSPARC Discuss (discuss.cryosparc.com) for troubleshooting guidance."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set

FORUM_BASE_URL = "https://discuss.cryosparc.com"
SEARCH_API_URL = f"{FORUM_BASE_URL}/search.json"
DEFAULT_TIMEOUT_SECONDS = 20
MAX_QUERY_LENGTH = 120

# Patterns that yield useful forum search terms when matched in job logs.
_ERROR_TOKEN_PATTERNS = [
    re.compile(r"CUDA_[A-Z_]+", re.IGNORECASE),
    re.compile(r"AssertionError[^\n]{0,80}", re.IGNORECASE),
    re.compile(r"failed to converge", re.IGNORECASE),
    re.compile(r"convergence failed", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"symmetry error", re.IGNORECASE),
    re.compile(r"invalid (?:parameter|symmetry)", re.IGNORECASE),
    re.compile(r"no particles", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
]


def _normalize_query(query: str) -> str:
    """Collapse whitespace and cap length for Discourse search."""
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    if len(cleaned) > MAX_QUERY_LENGTH:
        cleaned = cleaned[:MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
    return cleaned


def extract_search_queries_from_log(
    log_content: str,
    error_analysis: Optional[Dict[str, Any]] = None,
    max_queries: int = 3,
) -> List[str]:
    """
    Build short search queries from a CryoSPARC job log and optional error analysis.

    Args:
        log_content: Raw job.log text.
        error_analysis: Optional output from CryoSPARCTools._analyze_job_log.
        max_queries: Maximum number of distinct queries to return.

    Returns:
        List of search query strings (may be empty).
    """
    queries: List[str] = []
    seen: Set[str] = set()

    def _add(candidate: str) -> None:
        normalized = _normalize_query(candidate)
        key = normalized.lower()
        if not normalized or key in seen:
            return
        seen.add(key)
        queries.append(normalized)

    if error_analysis:
        for line in error_analysis.get("critical_errors", [])[:5]:
            for pattern in _ERROR_TOKEN_PATTERNS:
                match = pattern.search(line)
                if match:
                    _add(match.group(0))
                    break
            else:
                if "error" in line.lower() or "failed" in line.lower():
                    _add(line)

    if log_content:
        for pattern in _ERROR_TOKEN_PATTERNS:
            for match in pattern.finditer(log_content):
                _add(match.group(0))
                if len(queries) >= max_queries:
                    return queries[:max_queries]

    if not queries and error_analysis and error_analysis.get("has_errors"):
        _add("cryosparc job failed troubleshooting")

    return queries[:max_queries]


def search_cryosparc_forum(
    query: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Search CryoSPARC Discuss via the public Discourse API.

    Args:
        query: Search terms (error message, job type, symptom).
        max_results: Maximum topics to return.
        timeout: HTTP timeout in seconds.

    Returns:
        Dictionary with success flag and result entries (title, url, blurb).
    """
    normalized = _normalize_query(query)
    if not normalized:
        return {
            "success": False,
            "error": "Empty search query",
            "query": query,
            "results": [],
        }

    params = urllib.parse.urlencode({"q": normalized, "page": 1})
    url = f"{SEARCH_API_URL}?{params}"

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CryoAgent/1.0 (troubleshooting)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")

        import json

        data = json.loads(payload)
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "query": normalized,
            "results": [],
        }

    topics = data.get("topics") or []
    posts = data.get("posts") or []
    blurb_by_topic: Dict[int, str] = {}
    for post in posts:
        topic_id = post.get("topic_id")
        if topic_id is None or topic_id in blurb_by_topic:
            continue
        blurb = (post.get("blurb") or "").strip()
        if blurb:
            blurb_by_topic[int(topic_id)] = blurb

    results: List[Dict[str, str]] = []
    for topic in topics[:max_results]:
        topic_id = topic.get("id")
        slug = topic.get("slug") or ""
        title = (topic.get("title") or topic.get("fancy_title") or "Untitled").strip()
        if topic_id is None:
            continue
        topic_url = f"{FORUM_BASE_URL}/t/{slug}/{topic_id}" if slug else f"{FORUM_BASE_URL}/t/{topic_id}"
        results.append(
            {
                "title": title,
                "url": topic_url,
                "blurb": blurb_by_topic.get(int(topic_id), ""),
            }
        )

    return {
        "success": True,
        "query": normalized,
        "results": results,
        "result_count": len(results),
    }


def search_cryosparc_forum_multi(
    queries: List[str],
    max_results_per_query: int = 3,
    max_total_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run multiple forum searches and merge unique topics."""
    merged: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()
    errors: List[str] = []
    queries_used: List[str] = []

    for raw_query in queries:
        query = _normalize_query(raw_query)
        if not query:
            continue
        queries_used.append(query)
        result = search_cryosparc_forum(
            query,
            max_results=max_results_per_query,
            timeout=timeout,
        )
        if not result.get("success"):
            if result.get("error"):
                errors.append(f"{query}: {result['error']}")
            continue
        for entry in result.get("results", []):
            url = entry.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            entry_with_query = dict(entry)
            entry_with_query["matched_query"] = query
            merged.append(entry_with_query)
            if len(merged) >= max_total_results:
                break
        if len(merged) >= max_total_results:
            break

    return {
        "success": bool(merged),
        "queries": queries_used,
        "results": merged[:max_total_results],
        "result_count": len(merged[:max_total_results]),
        "errors": errors,
    }


def format_forum_search_response(search_payload: Dict[str, Any]) -> str:
    """Format forum search results for LLM consumption."""
    if not search_payload.get("success"):
        errors = search_payload.get("errors") or []
        error = search_payload.get("error")
        if errors:
            error = "; ".join(errors)
        return (
            "❌ CryoSPARC forum search did not return results. "
            f"Error: {error or 'unknown'}. "
            "Fall back to get_job_log suggestions and reason_about_workflow."
        )

    queries = search_payload.get("queries") or [search_payload.get("query")]
    query_text = ", ".join(q for q in queries if q) or "n/a"
    lines = [f"🔍 CryoSPARC forum search ({query_text}):"]

    for index, entry in enumerate(search_payload.get("results", []), start=1):
        title = entry.get("title", "Untitled")
        url = entry.get("url", "")
        blurb = entry.get("blurb", "")
        matched = entry.get("matched_query")
        header = f"\n{index}. {title}"
        if matched:
            header += f" (matched: {matched})"
        lines.append(header)
        if url:
            lines.append(f"   URL: {url}")
        if blurb:
            lines.append(f"   Excerpt: {blurb}")

    if len(lines) == 1:
        return (
            f"🔍 CryoSPARC forum search ({query_text}) returned no relevant topics. "
            "Try a more specific query from the job log error line, or use get_job_log suggestions."
        )

    lines.append(
        "\nUse these forum discussions to inform your retry parameters. "
        "Cite the reasoning in your Thought before re-queueing the job."
    )
    return "\n".join(lines)


# Heuristic themes mined from common CryoSPARC Discuss troubleshooting threads.
_CONCLUSION_RULES = [
    {
        "id": "gpu_memory",
        "patterns": [
            "cuda_error_out_of_memory",
            "out of memory",
            "could not allocate gpu",
            "cudamemory",
        ],
        "likely_cause": "GPU ran out of memory during the job.",
        "suggested_actions": [
            "Reduce batch size, number of classes, or particle count.",
            "Enable low-memory mode when the job type supports it.",
            "Use binning / smaller box size, or run on a GPU with more VRAM.",
            "Ensure no other jobs are occupying the same GPU.",
        ],
    },
    {
        "id": "convergence",
        "patterns": [
            "failed to converge",
            "do not converge",
            "does not converge",
            "not converge",
            "poor ab initio",
            "half map of noise",
        ],
        "likely_cause": "Ab initio / refinement did not converge to a stable structure.",
        "suggested_actions": [
            "Increase initial_resolution (e.g. 20–30 Å) and use C1 symmetry.",
            "Reduce num_classes to 1 for a first attempt.",
            "Check 2D class quality; re-extract with a larger box if classes are poor.",
            "Try homogeneous reconstruction as an alternative initial model.",
        ],
    },
    {
        "id": "box_size",
        "patterns": [
            "box size",
            "box too small",
            "re-extract",
            "larger box",
        ],
        "likely_cause": "Extraction box size may be too small for the particle.",
        "suggested_actions": [
            "Re-extract particles with box size ~1.5–2× particle diameter.",
            "Re-run 2D classification after re-extraction.",
        ],
    },
    {
        "id": "ctf_refinement",
        "patterns": [
            "ctf",
            "ctfrefine",
            "defocus",
            "ctf refinement",
            "refine_defocus",
        ],
        "likely_cause": "CTF-related refinement may be unstable for this dataset.",
        "suggested_actions": [
            "Disable local/global CTF refinement on the first retry "
            "(refine_defocus_refine=false, refine_ctf_global_refine=false).",
            "Confirm micrograph CTF estimates are reasonable before re-enabling CTF refine.",
            "Use a more conservative refinement_resolution before CTF refinement.",
        ],
    },
    {
        "id": "symmetry",
        "patterns": [
            "symmetry",
            "c1 symmetry",
            "wrong symmetry",
        ],
        "likely_cause": "Symmetry setting may not match the true particle symmetry.",
        "suggested_actions": [
            "Retry with C1 (no symmetry) unless symmetry is known with confidence.",
            "Verify symmetry visually before applying CN/DN symmetry.",
        ],
    },
    {
        "id": "heterogeneity",
        "patterns": [
            "heterogeneous",
            "heterogeneity",
            "multiple classes",
            "junk classes",
        ],
        "likely_cause": "Structural heterogeneity or junk particles may be affecting the job.",
        "suggested_actions": [
            "Use multiple ab initio classes and discard junk volumes.",
            "Consider heterogeneous refinement if distinct conformations are present.",
            "Tighten 2D class selection before reconstruction.",
        ],
    },
]


def _collect_search_corpus(
    search_payload: Dict[str, Any],
    log_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Combine query, excerpts, and log analysis into one lowercase corpus."""
    parts: List[str] = []
    queries = search_payload.get("queries") or []
    if search_payload.get("query"):
        queries = list(queries) + [search_payload["query"]]
    parts.extend(queries)
    for entry in search_payload.get("results", []):
        parts.append(entry.get("title", ""))
        parts.append(entry.get("blurb", ""))
    if log_analysis:
        parts.extend(log_analysis.get("critical_errors", []))
    return " ".join(parts).lower()


def derive_forum_conclusions(
    search_payload: Dict[str, Any],
    log_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Derive heuristic troubleshooting conclusions from forum search results.

    Returns a structured summary (not LLM-generated). The reconstruction agent
    still uses the LLM to reason over raw forum hits at runtime.
    """
    if not search_payload.get("success") or not search_payload.get("results"):
        return {
            "has_conclusions": False,
            "summary": "No forum results to analyze.",
            "matched_themes": [],
            "suggested_actions": [],
            "supporting_threads": [],
        }

    corpus = _collect_search_corpus(search_payload, log_analysis)
    matched_themes: List[Dict[str, Any]] = []
    actions: List[str] = []
    seen_actions: Set[str] = set()

    for rule in _CONCLUSION_RULES:
        if not any(pattern in corpus for pattern in rule["patterns"]):
            continue
        matched_themes.append(
            {
                "id": rule["id"],
                "likely_cause": rule["likely_cause"],
            }
        )
        for action in rule["suggested_actions"]:
            if action not in seen_actions:
                seen_actions.add(action)
                actions.append(action)

    supporting = [
        {
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
        }
        for entry in search_payload.get("results", [])[:3]
        if entry.get("url")
    ]

    if not matched_themes:
        return {
            "has_conclusions": True,
            "summary": (
                "Forum threads were found, but no known error theme matched automatically. "
                "Read the excerpts above and adjust parameters manually."
            ),
            "matched_themes": [],
            "suggested_actions": [
                "Review the top forum thread excerpts for dataset-specific advice.",
                "Cross-check with get_job_log built-in suggestions.",
            ],
            "supporting_threads": supporting,
        }

    primary = matched_themes[0]["likely_cause"]
    theme_count = len(matched_themes)
    summary = (
        f"Likely issue: {primary} "
        f"({theme_count} theme{'s' if theme_count != 1 else ''} detected from forum excerpts)."
    )

    return {
        "has_conclusions": True,
        "summary": summary,
        "matched_themes": matched_themes,
        "suggested_actions": actions,
        "supporting_threads": supporting,
    }


def format_forum_conclusions(conclusions: Dict[str, Any]) -> str:
    """Format heuristic conclusions for terminal or LLM display."""
    if not conclusions.get("has_conclusions"):
        return f"\n📌 Conclusions: {conclusions.get('summary', 'No analysis available.')}"

    lines = ["\n📌 Conclusions (heuristic, from forum excerpts):"]
    lines.append(f"   {conclusions.get('summary', '')}")

    themes = conclusions.get("matched_themes", [])
    if themes:
        lines.append("\n   Detected themes:")
        for theme in themes:
            lines.append(f"   - {theme.get('likely_cause', theme.get('id', ''))}")

    actions = conclusions.get("suggested_actions", [])
    if actions:
        lines.append("\n   Suggested retry actions:")
        for action in actions:
            lines.append(f"   - {action}")

    threads = conclusions.get("supporting_threads", [])
    if threads:
        lines.append("\n   Most relevant threads:")
        for thread in threads:
            title = thread.get("title", "Untitled")
            url = thread.get("url", "")
            lines.append(f"   - {title}")
            if url:
                lines.append(f"     {url}")

    lines.append(
        "\n   Note: These are rule-based hints for testing. "
        "The live agent uses the LLM to reason over the full forum output."
    )
    return "\n".join(lines)
