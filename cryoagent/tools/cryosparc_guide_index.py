"""Cached whole-guide discovery and lightweight BM25 ranking.

The CryoSPARC GitBook publishes ``llms.txt`` containing every public guide page,
its title, URL, and summary. We rank that complete catalog instead of relying on
the small curated catalog used by the original advisor. No third-party retrieval
dependency is required.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

GUIDE_BASE = "https://guide.cryosparc.com"
GUIDE_INDEX_URL = f"{GUIDE_BASE}/llms.txt"
GUIDE_SITEMAP_URL = f"{GUIDE_BASE}/sitemap.xml"
INDEX_TTL_SECONDS = 24 * 60 * 60
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "cryoagent" / "cryosparc_guide_index.json"

_STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "both", "but", "can", "could",
    "did", "does", "doing", "from", "had", "has", "have", "having", "how",
    "into", "its", "just", "more", "most", "not", "now", "only", "other",
    "our", "out", "over", "same", "should", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "too", "under", "use", "using", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "a", "an", "as", "at", "be", "by", "for", "i", "in", "is", "it", "of",
    "on", "or", "so", "to", "we", "you",
}

# Expand common scientific phrasing before matching guide terminology.
_QUERY_EXPANSIONS = (
    (r"\bdeep(?:er)?[ -]learning picking\b", "topaz deep picking"),
    (r"\bdeep picking\b", "topaz deep picking"),
    (r"\bper[ -]particle (?:beam[ -]induced )?motion\b", "local motion correction"),
    (r"\bparticle trajector(?:y|ies)\b", "local motion correction"),
    (r"\bresolution (?:plateau|ceiling)\b", "high resolution refinement"),
)


def tokenize(text: str) -> List[str]:
    """Tokenize prose for ranking, dropping search-language stopwords."""
    q = (text or "").lower()
    for pattern, replacement in _QUERY_EXPANSIONS:
        q = re.sub(pattern, replacement, q)
    return [
        tok for tok in re.findall(r"[a-z0-9]+", q)
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _display_url(url: str) -> str:
    """Keep canonical guide URLs and reject look-alike hosts."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.netloc != "guide.cryosparc.com":
        return ""
    path = parsed.path[:-3] if parsed.path.endswith(".md") else parsed.path
    return GUIDE_BASE + path


def parse_llms_index(text: str) -> List[Dict[str, str]]:
    """Parse GitBook's ``- [title](url): summary`` whole-site catalog."""
    entries: List[Dict[str, str]] = []
    seen = set()
    # Descriptions may continue on following indented lines; the title + URL is
    # sufficient for discovery, and the first description line adds useful text.
    for match in re.finditer(r"^-\s+\[([^]]+)]\((https://guide\.cryosparc\.com/[^)]+)\)(?::\s*(.*))?$",
                             text or "", flags=re.MULTILINE):
        title, raw_url, description = match.groups()
        url = _display_url(raw_url)
        if not url or url in seen:
            continue
        seen.add(url)
        entries.append({
            "title": title.strip(),
            "url": url,
            "description": (description or "").strip(),
        })
    return entries


def _entries_from_sitemap(fetcher: Callable[[str], Optional[str]]) -> List[Dict[str, str]]:
    """Fallback to the public sitemap if llms.txt is unavailable."""
    root = fetcher(GUIDE_SITEMAP_URL) or ""
    locations = re.findall(r"<loc>\s*([^<]+)\s*</loc>", root)
    urls: List[str] = []
    for loc in locations:
        if loc.endswith(".xml"):
            child = fetcher(loc) or ""
            urls.extend(re.findall(r"<loc>\s*([^<]+)\s*</loc>", child))
        else:
            urls.append(loc)
    entries = []
    seen = set()
    for raw_url in urls:
        url = _display_url(raw_url)
        if not url or url in seen:
            continue
        seen.add(url)
        slug = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
        title = re.sub(r"[-_]+", " ", slug).strip().title() or url
        entries.append({"title": title, "url": url, "description": ""})
    return entries


def _read_cache(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data.get("entries"), list) else {}
    except Exception:
        return {}


def load_guide_index(
    fetcher: Callable[[str], Optional[str]],
    *,
    refresh: bool = False,
    cache_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Load the complete guide catalog, using a 24-hour disk cache.

    A stale cache is preferable to an empty index during a network outage.
    """
    path = Path(cache_path or os.environ.get("CRYOAGENT_GUIDE_CACHE", DEFAULT_CACHE_PATH))
    cached = _read_cache(path)
    age = time.time() - float(cached.get("fetched_at", 0) or 0)
    if not refresh and cached and age < INDEX_TTL_SECONDS:
        return cached["entries"]

    entries = parse_llms_index(fetcher(GUIDE_INDEX_URL) or "")
    if not entries:
        entries = _entries_from_sitemap(fetcher)
    if entries:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"fetched_at": time.time(), "entries": entries}, handle)
        except Exception:
            pass  # Retrieval still works when the cache is read-only.
        return entries
    return cached.get("entries", [])


def bm25_rank(
    entries: Iterable[Dict[str, str]],
    query: str,
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Rank title + summary metadata with BM25; titles receive triple weight."""
    docs = list(entries)
    query_tokens = tokenize(query)
    if not docs or not query_tokens:
        return []

    doc_tokens = [
        tokenize(f"{e.get('title', '')} {e.get('title', '')} {e.get('title', '')} "
                 f"{e.get('description', '')} {e.get('url', '').replace('/', ' ').replace('-', ' ')}")
        for e in docs
    ]
    df: Dict[str, int] = {}
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    avg_len = sum(map(len, doc_tokens)) / max(1, len(doc_tokens))
    n_docs = len(docs)
    k1, b = 1.5, 0.75
    ranked = []
    for entry, tokens in zip(docs, doc_tokens):
        frequencies: Dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            freq = frequencies.get(token, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
            denominator = freq + k1 * (1.0 - b + b * len(tokens) / max(1.0, avg_len))
            score += idf * freq * (k1 + 1.0) / denominator
        if score > 0:
            item = dict(entry)
            item["score"] = score
            ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]
