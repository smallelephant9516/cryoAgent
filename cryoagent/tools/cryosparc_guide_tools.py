"""Consult the official CryoSPARC guide (https://guide.cryosparc.com/) as an advisor.

Lightweight, no-RAG retrieval over two corpora:
  * job-reference pages (what a job does + its params), and
  * the tutorials & case-studies library (worked, problem-solving examples —
    preferred orientation, pseudosymmetry, membrane proteins, 3D classification,
    3DVA, CTF refinement, end-to-end GPCR/ferritin processing, etc.).

Used by the dynamic improvement agent to ground decisions in the official guide
rather than guessing. Two access modes:
  * AUTO  — pass a `question`; the tool scores it against both corpora, fetches
    the best match(es), and lists other relevant tutorials to drill into.
  * BROWSE — pass `slug` to fetch a specific tutorial/page, or `list_tutorials`
    to see the whole library catalog (title + slug + when-to-use).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

GUIDE_BASE = "https://guide.cryosparc.com"
TUTORIALS_BASE = "/processing-data/tutorials-and-case-studies"
DEFAULT_TIMEOUT = 20
# Ceiling on the full page body returned for the deep-read/condense path, so a
# pathologically long case study can't blow up the side-LLM call.
MAX_FULL_CHARS = 50000

# Curated topic → job-reference page map (what a job does + params).
_TOPIC_PAGES: Dict[str, str] = {
    "ctf": "/processing-data/all-job-types-in-cryosparc/ctf-estimation/job-patch-ctf-estimation",
    "motion": "/processing-data/all-job-types-in-cryosparc/motion-correction/job-patch-motion-correction",
    "2d classification": "/processing-data/all-job-types-in-cryosparc/particle-curation/job-2d-classification",
    "class 2d": "/processing-data/all-job-types-in-cryosparc/particle-curation/job-2d-classification",
    "ab initio": "/processing-data/all-job-types-in-cryosparc/3d-reconstruction/job-ab-initio-reconstruction",
    "ab-initio": "/processing-data/all-job-types-in-cryosparc/3d-reconstruction/job-ab-initio-reconstruction",
    "heterogeneous": "/processing-data/all-job-types-in-cryosparc/3d-refinement/job-heterogeneous-refinement",
    "homogeneous refinement": "/processing-data/all-job-types-in-cryosparc/3d-refinement/job-homogeneous-refinement",
    "non-uniform": "/processing-data/all-job-types-in-cryosparc/3d-refinement/job-non-uniform-refinement-new",
    "nonuniform": "/processing-data/all-job-types-in-cryosparc/3d-refinement/job-non-uniform-refinement-new",
    "refinement": "/processing-data/all-job-types-in-cryosparc/3d-refinement/job-homogeneous-refinement",
    "picking": "/processing-data/all-job-types-in-cryosparc/particle-picking",
    "blob": "/processing-data/all-job-types-in-cryosparc/particle-picking/job-blob-picker",
    "template pick": "/processing-data/all-job-types-in-cryosparc/particle-picking/job-template-picker",
    "extract": "/processing-data/all-job-types-in-cryosparc/extraction/job-extract-from-micrographs",
}

# The tutorials & case-studies library. Each entry: (slug, title, when, tags).
# `slug` is the final path segment under TUTORIALS_BASE (verified live). `tags`
# are the keywords matched against a question. Curated to map the improvement
# agent's symptoms (preferred orientation, heterogeneity, membrane, ...) to the
# right worked example.
_TUTORIALS: List[Dict[str, Any]] = [
    {"slug": "tutorial-orientation-diagnostics",
     "title": "Tutorial: Orientation Diagnostics (cFAR / preferred orientation)",
     "when": "Diagnose preferred orientation / low cFAR and quantify orientation bias.",
     "tags": ["orientation", "cfar", "preferred", "bias", "conical", "diagnostics", "scf"]},
    {"slug": "case-study-picking-induced-orientation-bias-in-ha-trimer-empiar-10096-and-10097",
     "title": "Case study: picking-induced orientation bias (HA trimer)",
     "when": "Preferred orientation caused by biased picking; how to recover lost views.",
     "tags": ["orientation", "preferred", "bias", "picking", "trimer", "ha", "tilt"]},
    {"slug": "case-study-pseudosymmetry-in-trpv5-and-calmodulin-empiar-10256",
     "title": "Case study: pseudosymmetry in TRPV5 + calmodulin",
     "when": "Pseudosymmetry / symmetry mismatch breaking refinement.",
     "tags": ["pseudosymmetry", "symmetry", "trpv5", "calmodulin", "relaxation"]},
    {"slug": "tutorial-symmetry-relaxation",
     "title": "Tutorial: Symmetry Relaxation",
     "when": "A pseudo-symmetric complex needs symmetry relaxed to resolve true asymmetry.",
     "tags": ["symmetry", "relaxation", "pseudosymmetry", "asymmetry"]},
    {"slug": "tutorial-tips-for-membrane-protein-structures",
     "title": "Tutorial: Tips for membrane protein structures",
     "when": "Membrane proteins, micelle/detergent density, small anisotropic particles.",
     "tags": ["membrane", "protein", "micelle", "detergent", "nanodisc", "nonuniform", "mask"]},
    {"slug": "tutorial-3d-classification",
     "title": "Tutorial: 3D Classification",
     "when": "Resolve DISCRETE conformational/compositional states without re-aligning.",
     "tags": ["classification", "heterogeneity", "discrete", "classes", "states", "focus"]},
    {"slug": "tutorial-3d-variability-analysis-part-one",
     "title": "Tutorial: 3D Variability Analysis (part 1)",
     "when": "Probe CONTINUOUS heterogeneity / flexibility (3DVA setup).",
     "tags": ["variability", "3dva", "continuous", "motion", "flexibility", "modes"]},
    {"slug": "tutorial-3d-variability-analysis-part-two",
     "title": "Tutorial: 3D Variability Analysis (part 2)",
     "when": "Interpret and display 3DVA results (clusters, trajectories).",
     "tags": ["variability", "3dva", "continuous", "motion", "display", "clusters"]},
    {"slug": "tutorial-3d-flexible-refinement",
     "title": "Tutorial: 3D Flexible Refinement (3DFlex)",
     "when": "Model continuous flexible motion to improve resolution of flexible regions.",
     "tags": ["flex", "flexible", "continuous", "motion", "deformation"]},
    {"slug": "tutorial-3d-flex-mesh-preparation",
     "title": "Tutorial: 3DFlex mesh preparation",
     "when": "Prepare the mesh required for 3D Flexible Refinement.",
     "tags": ["flex", "mesh", "flexible", "preparation"]},
    {"slug": "tutorial-ctf-refinement",
     "title": "Tutorial: CTF Refinement",
     "when": "Refine global aberrations / per-particle defocus to push resolution.",
     "tags": ["ctf", "defocus", "aberration", "tilt", "trefoil", "refinement"]},
    {"slug": "tutorial-patch-motion-and-patch-ctf",
     "title": "Tutorial: Patch Motion & Patch CTF",
     "when": "Preprocessing: patch-based motion correction and CTF estimation.",
     "tags": ["motion", "ctf", "patch", "preprocessing"]},
    {"slug": "tutorial-dynamic-masking-in-refinements-v5.0",
     "title": "Tutorial: Dynamic masking in refinements",
     "when": "Masking behavior during refinement; fix mask-related artifacts.",
     "tags": ["mask", "dynamic", "masking", "refinement"]},
    {"slug": "mask-selection-and-generation-in-ucsf-chimera",
     "title": "Mask selection & generation in UCSF Chimera",
     "when": "Build a focus/solvent mask for local refinement or classification.",
     "tags": ["mask", "chimera", "focus", "solvent", "generation"]},
    {"slug": "tutorial-blob-picker-tuner",
     "title": "Tutorial: Blob Picker Tuner",
     "when": "Tune blob-picker parameters to pick more/cleaner particles.",
     "tags": ["blob", "picker", "picking", "tuning", "diameter"]},
    {"slug": "tutorial-particle-picking-calibration",
     "title": "Tutorial: Particle picking calibration",
     "when": "Calibrate particle diameter / picking thresholds.",
     "tags": ["picking", "calibration", "diameter", "threshold"]},
    {"slug": "tutorial-common-cryosparc-plots",
     "title": "Tutorial: Common CryoSPARC plots",
     "when": "Interpret FSC, viewing-direction, and diagnostic plots.",
     "tags": ["plots", "fsc", "diagnostics", "interpretation", "viewing"]},
    {"slug": "performance-metrics",
     "title": "Performance metrics",
     "when": "Understand speed/throughput benchmarks.",
     "tags": ["performance", "benchmark", "speed", "metrics"]},
    {"slug": "negative-stain-data",
     "title": "Negative stain data",
     "when": "Process negative-stain (non-cryo) data.",
     "tags": ["negative", "stain"]},
    {"slug": "phase-plate-data",
     "title": "Phase plate data",
     "when": "Process Volta phase-plate data.",
     "tags": ["phase", "plate", "volta"]},
    {"slug": "tutorial-eer-file-support",
     "title": "Tutorial: EER file support",
     "when": "Import and handle EER-format movies.",
     "tags": ["eer", "import", "movies", "format"]},
    {"slug": "tutorial-float16-support",
     "title": "Tutorial: float16 support",
     "when": "Use 16-bit storage to save disk/memory.",
     "tags": ["float16", "storage", "disk", "memory"]},
    {"slug": "tutorial-epu-afis-beam-shift-import",
     "title": "Tutorial: EPU AFIS beam-shift import (exposure groups)",
     "when": "Import AFIS beam-shift groups for per-group CTF/optics refinement.",
     "tags": ["afis", "beam", "shift", "epu", "exposure", "groups", "import"]},
    {"slug": "tutorial-ewald-sphere-correction",
     "title": "Tutorial: Ewald sphere correction",
     "when": "Large particles at high resolution limited by Ewald sphere curvature.",
     "tags": ["ewald", "sphere", "large", "high-resolution", "curvature"]},
    {"slug": "tutorial-bild-files",
     "title": "Tutorial: BILD files",
     "when": "Generate/visualize BILD geometry files.",
     "tags": ["bild", "visualization", "geometry"]},
    {"slug": "case-study-discrete-and-continuous-heterogeneity-in-fanac1-empiar-11631-and-11632",
     "title": "Case study: discrete + continuous heterogeneity (FANCA-C1)",
     "when": "A sample with BOTH discrete states and continuous motion.",
     "tags": ["heterogeneity", "discrete", "continuous", "classification", "variability"]},
    {"slug": "case-study-discrete-heterogeneity-in-a-sample-of-acetogenin-bound-complex-i-empiar-10927",
     "title": "Case study: discrete heterogeneity (ligand-bound Complex I)",
     "when": "Separate discrete ligand-bound/unbound states.",
     "tags": ["heterogeneity", "discrete", "ligand", "classification", "complex"]},
    {"slug": "case-study-dktx-bound-trpv1-empiar-10059",
     "title": "Case study: DkTx-bound TRPV1",
     "when": "Membrane channel with a bound toxin/ligand.",
     "tags": ["ligand", "trpv1", "membrane", "toxin", "channel"]},
    {"slug": "case-study-empiar-10031-mavs",
     "title": "Case study: helical processing (MAVS, EMPIAR-10031)",
     "when": "Helical/filamentous specimens.",
     "tags": ["helical", "filament", "mavs", "helix"]},
    {"slug": "case-study-end-to-end-and-exploratory-processing-of-a-motor-bound-nucleosome-empiar-10739",
     "title": "Case study: end-to-end motor-bound nucleosome",
     "when": "Full exploratory end-to-end workflow on a challenging complex.",
     "tags": ["end-to-end", "exploratory", "nucleosome", "workflow"]},
    {"slug": "case-study-processing-of-a-novel-motor-bound-nucleosome-state-empiar-10739-part-2",
     "title": "Case study: motor-bound nucleosome (part 2)",
     "when": "Resolve a novel state via heterogeneity analysis.",
     "tags": ["heterogeneity", "nucleosome", "state", "classification"]},
    {"slug": "case-study-end-to-end-processing-of-a-ligand-bound-gpcr-empiar-10853",
     "title": "Case study: end-to-end ligand-bound GPCR",
     "when": "Small membrane GPCR with a bound ligand, start to finish.",
     "tags": ["gpcr", "membrane", "ligand", "end-to-end", "small"]},
    {"slug": "case-study-end-to-end-processing-of-an-inactive-gpcr-empiar-10668",
     "title": "Case study: end-to-end inactive GPCR",
     "when": "Small membrane GPCR processing, start to finish.",
     "tags": ["gpcr", "membrane", "end-to-end", "small", "inactive"]},
    {"slug": "case-study-end-to-end-processing-of-encapsulated-ferritin-empiar-10716",
     "title": "Case study: end-to-end ferritin (high resolution)",
     "when": "High-resolution, high-symmetry benchmark workflow.",
     "tags": ["ferritin", "end-to-end", "high-resolution", "octahedral", "symmetry"]},
    {"slug": "case-study-exploratory-data-processing-by-oliver-clarke",
     "title": "Case study: exploratory data processing (O. Clarke)",
     "when": "General exploratory processing strategy / workflow decisions.",
     "tags": ["exploratory", "workflow", "strategy", "general"]},
    {"slug": "case-study-yeast-u4-u6.u5-tri-snrnp",
     "title": "Case study: yeast U4/U6.U5 tri-snRNP",
     "when": "Large flexible RNA-protein complex with heterogeneity.",
     "tags": ["heterogeneity", "large", "complex", "rna", "snrnp", "flexible"]},
]



def _strip_html(html: str) -> str:
    """HTML → text. The guide is a GitBook site; the real page body lives inside
    the <main> element, while everything else is the nav sidebar/boilerplate
    ("CryoSPARC Guide About... Changelog Licensing..."). Extract <main> first so
    excerpts are the actual tutorial content, not the navigation menu."""
    mains = re.findall(r"<main\b[^>]*>(.*?)</main>", html, flags=re.S | re.I)
    if mains:
        html = max(mains, key=len)  # the largest <main> is the page body
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # Drop the common GitBook page-header preamble ("...llms.txt ... On this page")
    # that precedes the real content, keeping the body that follows it.
    m = re.search(r"On this page\b", text)
    if m and m.end() < len(text) - 50:
        text = text[m.end():].strip()
    return text


def _fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CryoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="ignore")
    except Exception:
        return None


def _relevant_excerpt(text: str, question: str, window: int = 1200) -> str:
    """Return the text window most relevant to the question's keywords."""
    if not text:
        return ""
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", question.lower())]
    if not words:
        return text[:window]
    low = text.lower()
    best_pos, best_score = 0, -1
    # Slide over candidate anchor positions (first occurrence of each keyword).
    for w in words:
        pos = low.find(w)
        if pos == -1:
            continue
        seg = low[max(0, pos - 100): pos + window]
        score = sum(seg.count(k) for k in words)
        if score > best_score:
            best_score, best_pos = score, max(0, pos - 100)
    return text[best_pos: best_pos + window]


def _score_tutorial(entry: Dict[str, Any], question: str) -> int:
    """Keyword-overlap score of a tutorial against the question."""
    q = question.lower()
    qwords = set(re.findall(r"[a-z0-9]{3,}", q))
    score = 0
    for tag in entry["tags"]:
        if tag in q:
            score += 3                      # whole-phrase tag hit
        elif tag in qwords:
            score += 2
    # Title words give a weaker signal.
    for w in re.findall(r"[a-z0-9]{4,}", entry["title"].lower()):
        if w in qwords:
            score += 1
    return score


def _rank_tutorials(question: str, limit: int = 5) -> List[Dict[str, Any]]:
    scored = [(s, e) for e in _TUTORIALS if (s := _score_tutorial(e, question)) > 0]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:limit]]


def list_cryosparc_tutorials() -> Dict[str, object]:
    """Return the full tutorials & case-studies catalog (title + slug + when)."""
    items = [{"slug": e["slug"], "title": e["title"], "when": e["when"]} for e in _TUTORIALS]
    return {"success": True, "count": len(items), "tutorials": items,
            "message": f"{len(items)} tutorials/case-studies available. "
                       "Fetch one by passing slug=<slug>."}


def fetch_guide_page(path_or_slug: str, question: str = "", full: bool = False) -> Dict[str, object]:
    """Fetch a specific guide page by tutorial slug or full guide path.

    By default returns the excerpt most relevant to `question` (or the page start).
    When `full=True`, also returns the ENTIRE cleaned page body under `full_text`
    (capped at MAX_FULL_CHARS) — used by the deep-read/condense path so a separate
    LLM call can read the whole tutorial without dumping it into the main context.
    """
    p = path_or_slug.strip()
    if p.startswith("http"):
        path = p[len(GUIDE_BASE):] if p.startswith(GUIDE_BASE) else None
        if path is None:
            return {"success": False, "pages": [], "message": "URL must be on guide.cryosparc.com."}
    elif p.startswith("/"):
        path = p
    else:
        # bare slug -> a tutorials/case-studies page
        path = f"{TUTORIALS_BASE}/{p}"
    url = GUIDE_BASE + path
    html = _fetch(url)
    if not html:
        return {"success": False, "pages": [], "message": f"Could not fetch {url} (unknown slug or network?)."}
    text = _strip_html(html)
    excerpt = _relevant_excerpt(text, question) if question else text[:1200]
    page: Dict[str, Any] = {"url": url, "excerpt": excerpt}
    if full:
        page["full_text"] = text[:MAX_FULL_CHARS]
    return {"success": True, "pages": [page], "message": f"Fetched {url}"}


def consult_cryosparc_guide(question: str = "", max_pages: int = 2, *,
                            slug: Optional[str] = None,
                            list_tutorials: bool = False,
                            full: bool = False) -> Dict[str, object]:
    """
    Advisor over the CryoSPARC guide. Three modes:

    * list_tutorials=True  -> return the full tutorials/case-studies catalog.
    * slug="..."           -> fetch that specific tutorial/page. With full=True
                              also returns the WHOLE cleaned body under
                              pages[0]['full_text'] (deep-read/condense path).
    * question="..."       -> AUTO: fetch the best matching job-reference page
                              and/or tutorial, and also list other relevant
                              tutorials to drill into via slug.

    Advisor only — does not decide anything. Returns {success, pages, message,
    and (auto mode) related_tutorials}.
    """
    if list_tutorials:
        return list_cryosparc_tutorials()
    if slug:
        return fetch_guide_page(slug, question, full=full)

    # Empty/whitespace question with no slug: there is nothing to match on.
    # Return the browsable catalog so the agent can pick a slug, rather than a
    # dead "no match" (the agent sometimes calls with question="" expecting the
    # tool to infer intent).
    if not (question or "").strip():
        res = list_cryosparc_tutorials()
        res["message"] = ("No question provided. Browse the catalog below and "
                          "re-call with slug=<slug>, or pass a specific question. "
                          + str(res.get("message", "")))
        return res

    q = (question or "").lower()
    pages: List[Dict[str, str]] = []

    # 1) Best job-reference page(s) by keyword.
    matched_paths: List[str] = []
    for kw, path in _TOPIC_PAGES.items():
        if kw in q and path not in matched_paths:
            matched_paths.append(path)

    # 2) Best matching tutorials/case-studies.
    ranked = _rank_tutorials(question)
    tutorial_paths = [f"{TUTORIALS_BASE}/{e['slug']}" for e in ranked]

    # Interleave: take the top tutorial first (worked examples are high-signal for
    # problem solving), then a job-reference page, up to max_pages.
    fetch_order: List[str] = []
    if tutorial_paths:
        fetch_order.append(tutorial_paths[0])
    fetch_order.extend(matched_paths)
    fetch_order.extend(tutorial_paths[1:])
    # de-dup preserving order
    seen = set()
    fetch_order = [p for p in fetch_order if not (p in seen or seen.add(p))]

    if not fetch_order:
        return {
            "success": False, "pages": [], "related_tutorials": [],
            "message": (
                "No curated page or tutorial matched. Pass list_tutorials=true to "
                "browse the case-study library, or rephrase with a topic like "
                "orientation/cFAR, heterogeneity, membrane protein, CTF refinement, "
                "symmetry, 3D classification, 3DVA."
            ),
        }

    for path in fetch_order[:max_pages]:
        html = _fetch(GUIDE_BASE + path)
        if html:
            excerpt = _relevant_excerpt(_strip_html(html), question)
            if excerpt:
                pages.append({"url": GUIDE_BASE + path, "excerpt": excerpt})

    related = [{"slug": e["slug"], "title": e["title"], "when": e["when"]} for e in ranked]
    if not pages:
        return {"success": False, "pages": [], "related_tutorials": related,
                "message": f"Matched pages but could not fetch them (network?): {fetch_order[:max_pages]}"}
    return {"success": True, "pages": pages, "related_tutorials": related,
            "message": f"Fetched {len(pages)} page(s); {len(related)} related tutorial(s) listed."}

