"""Static data for the About tab — algorithms, metrics, datasets.

Kept separate from app.py for clarity and maintainability.
Algorithms are ordered: CPD/Distribution-based, CPD/Subsequence-based,
SD/Integrated, SD/Decomposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoEntry:
    """One row of the algorithm table."""

    name: str
    task: str  # "CPD" or "SD"
    family: str  # Distribution-based, Subsequence-based, Integrated, Decomposed
    complexity: str  # Big-O complexity (math notation, no $ wrappers)
    license: str  # SPDX-like short id, or "—" if unknown
    source: str  # upstream project / author


# Ordered: CPD Distribution-based → CPD Subsequence-based → SD Integrated → SD Decomposed
ALGO_ENTRIES: Sequence[AlgoEntry] = (
    # ── CPD · Distribution-based ──────────────────────────────────────────
    AlgoEntry("AMOC", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("BinSeg", "CPD", "Distribution-based", "O(dn\\log n)", "BSD-2", "ruptures"),
    AlgoEntry("BottomUp", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("BOCD", "CPD", "Distribution-based", "O(dn^2)", "Apache-2.0", "hildensia"),
    AlgoEntry("DynP", "CPD", "Distribution-based", "O(n_{cp}\\,dn^2)", "BSD-2", "ruptures"),
    AlgoEntry("EAgglo", "CPD", "Distribution-based", "O(dn^2)", "BSD-3", "aeon"),
    AlgoEntry("GreedyGaussian", "CPD", "Distribution-based", "O(n_{cp}^2 d^3 n)", "BSD-3", "aeon"),
    AlgoEntry("ICID", "CPD", "Distribution-based", "O(dn)", "GPLv3", "IsolationKernel"),
    AlgoEntry("KCPD", "CPD", "Distribution-based", "O(n_{cp}\\,dn^2)", "BSD-2", "ruptures"),
    AlgoEntry("Pelt", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("Prophet", "CPD", "Distribution-based", "O(n_{cp}\\,dn)", "MIT", "Facebook"),
    AlgoEntry("Window", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("Beast", "CPD", "Distribution-based", "-", "MIT", "Rbeast"),
    # ── CPD · Subsequence-based ───────────────────────────────────────────
    AlgoEntry("CLaSP", "CPD", "Subsequence-based", "O(n_{cp}\\,dn^2)", "BSD-3", "claspy"),
    AlgoEntry("ESPRESSO", "CPD", "Subsequence-based", "O(dn^2 + n_{cp}\\,n)", "—", "cruiseresearchgroup"),
    AlgoEntry("FLUSS", "CPD", "Subsequence-based", "O(dn^2 + n_{cp}\\,n)", "BSD-3", "aeon"),
    AlgoEntry("InformationGain", "CPD", "Subsequence-based", "O(n_{cp}\\,dn^2)", "BSD-3", "aeon"),
    AlgoEntry("TGLAD", "CPD", "Subsequence-based", "O(d^3 n)", "Non-Commercial", "Harshs27"),
    AlgoEntry("Tire", "CPD", "Subsequence-based", "O(dn)", "—", "KU Leuven"),
    AlgoEntry("TSCP2", "CPD", "Subsequence-based", "O(dn)", "—", "cruiseresearchgroup"),
    # ── SD · Integrated ───────────────────────────────────────────────────
    AlgoEntry("AutoPlait", "SD", "Integrated", "O(dn)", "—", "—"),
    AlgoEntry("HdpHsmm", "SD", "Integrated", "O(dn + n_s \\min(n, 2000)^2)", "—", "mattjj/pyhsmm (reimpl.)"),
    AlgoEntry("Hidalgo", "SD", "Integrated", "O(dn^2)", "BSD-3", "aeon"),
    AlgoEntry("HMM", "SD", "Integrated", "-", "BSD-3", "hmmlearn"),
    # ── SD · Decomposed ───────────────────────────────────────────────────
    AlgoEntry("TICC", "SD", "Decomposed", "O(n_s(dn + (dl)^3))", "BSD-2", "davidhallac"),
    AlgoEntry("Time2State", "SD", "Decomposed", "O(dn)", "MIT", "Lab-ANT"),
    AlgoEntry("E2USD", "SD", "Decomposed", "O(dn\\log l)", "—", "AI4CTS"),
    AlgoEntry("Clap", "SD", "Decomposed", "O(n_{cp}\\,n(nd + n_{cp}\\log n_{cp}))", "BSD-3", "claspy"),
)

# Quick lookup by name (case-insensitive)
ALGO_BY_NAME: dict[str, AlgoEntry] = {a.name.lower(): a for a in ALGO_ENTRIES}


def get_algo_entry(name: str) -> AlgoEntry | None:
    """Lookup an algorithm entry by name (case-insensitive)."""
    return ALGO_BY_NAME.get(name.lower())


# ---------------------------------------------------------------------------
# Complexity classification (for color-coding in the UI)
# ---------------------------------------------------------------------------


def classify_complexity(complexity: str) -> str:
    """Categorize a complexity string into a speed tier.

    Returns one of: ``linear``, ``log_linear``, ``quadratic``, ``cubic``.
    """
    s = complexity.lower().replace(" ", "")
    if "n^3" in s or "n³" in s:
        return "cubic"
    if "n^2" in s or "n²" in s:
        return "quadratic"
    if "logn" in s or "log(n)" in s or "\\logn" in s:
        return "log_linear"
    return "linear"


COMPLEXITY_COLORS = {
    "linear": "#1e7d32",  # green
    "log_linear": "#1565c0",  # blue
    "quadratic": "#ef6c00",  # orange
    "cubic": "#c62828",  # red
}

COMPLEXITY_LABELS = {
    "linear": "linear",
    "log_linear": "log-linear",
    "quadratic": "quadratic",
    "cubic": "cubic",
}


# ---------------------------------------------------------------------------
# Variable glossary (for tooltips)
# ---------------------------------------------------------------------------

COMPLEXITY_GLOSSARY = {
    "n": "number of points in the time series",
    "d": "number of dimensions (channels)",
    "l": "window size",
    "n_cp": "number of change points",
    "n_s": "number of distinct states",
}

COMPLEXITY_GLOSSARY_MD = "**Complexity notation:** " + " · ".join(
    f"`{k}` = {v}" for k, v in COMPLEXITY_GLOSSARY.items()
)


def format_complexity_plain(s: str) -> str:
    """Render a LaTeX-ish complexity string as plain text.

    The complexity strings in :data:`ALGO_ENTRIES` use LaTeX fragments such as
    ``\\log``, ``n^2`` and ``n_{cp}``. Gradio's Markdown renderer on the
    Hugging Face Space runtime does not consistently process ``$...$`` math,
    so we convert these strings into a readable Unicode form (e.g.
    ``O(dn log l)``, ``O(n_cp dn²)``) instead of relying on KaTeX.
    """
    import re

    if not s:
        return s
    out = s
    out = out.replace("\\log", " log").replace("\\cdot", "·").replace("\\,", " ")
    out = (
        out.replace("n^2", "n²")
        .replace("n^3", "n³")
        .replace("d^2", "d²")
        .replace("d^3", "d³")
        .replace("n_s^2", "n_s²")
        .replace("n_{cp}^2", "n_cp²")
    )
    # Subscripts: n_{cp} -> n_cp
    out = re.sub(r"_\{([^}]+)\}", r"_\1", out)
    # Strip any leftover backslashes
    out = out.replace("\\", "")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricEntry:
    name: str
    task: str  # "CPD" or "SD"
    description: str


METRIC_ENTRIES: Sequence[MetricEntry] = (
    # CPD
    MetricEntry(
        "F1 Score", "CPD", "Classic precision / recall / F1 with a hard margin tolerance around each true change point"
    ),
    MetricEntry(
        "Gaussian F1",
        "CPD",
        "Gaussian-weighted soft F1 — predictions are rewarded via a Gaussian kernel "
        "that decays with distance, no hard margin",
    ),
    MetricEntry("Covering", "CPD", "Segment-wise Intersection over Union (IoU) weighted by segment duration"),
    MetricEntry(
        "Bidirectional Covering",
        "CPD",
        "Evaluates coverage in both directions (truth→pred and pred→truth) aggregated by harmonic mean",
    ),
    MetricEntry(
        "Hausdorff Distance",
        "CPD",
        "Maximum over all true CPs of the distance to the nearest predicted CP (and vice-versa)",
    ),
    # SD
    MetricEntry(
        "Adjusted Rand Index", "SD", "Chance-adjusted agreement between true and predicted label partitions (sklearn)"
    ),
    MetricEntry(
        "Normalized Mutual Information", "SD", "Information-theoretic agreement normalised to [0, 1] (sklearn)"
    ),
    MetricEntry("Adjusted Mutual Information", "SD", "Chance-adjusted variant of NMI (sklearn)"),
    MetricEntry("Weighted ARI", "SD", "ARI variant that gives more weight to samples near segment boundaries"),
    MetricEntry("Weighted NMI", "SD", "NMI variant that gives more weight to samples near segment boundaries"),
    MetricEntry(
        "State Matching Score",
        "SD",
        "Hungarian matching of predicted to true states with fine-grained error classification",
    ),
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    dtype: str  # "Generated" or "Real-world"
    description: str
    dimensions: str
    segments: str
    source: str


DATASET_ENTRIES: Sequence[DatasetEntry] = (
    DatasetEntry(
        "Sample dataset",
        "Real-world",
        "A single labelled trial from CMU MoCap 86 (humeral & femoral angles), served as the built-in example",
        "4",
        "3–8",
        "[CMU MoCap](https://mocap.cs.cmu.edu/)",
    ),
    DatasetEntry(
        "Synthetic",
        "Generated",
        "Configurable generator with 7 intra-segment patterns "
        "(Gaussian processes, sinusoidal, constant, step, stairs, linear, mixed), noise control, recurring states",
        "1–8",
        "2–12",
        "tsseg",
    ),
)


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def build_algo_table(runtime_detectors: dict | None = None) -> str:
    """Return a Markdown table of algorithms (no group separator rows)."""
    rows: list[str] = []
    for a in ALGO_ENTRIES:
        rows.append(
            f"| {a.name} | {a.task} | {a.family} | `{format_complexity_plain(a.complexity)}` | {a.license} | {a.source} |"
        )

    header = (
        "| Algorithm | Task | Family | Complexity | License | Source |\n"
        "|-----------|------|--------|------------|---------|--------|\n"
    )
    return header + "\n".join(rows)


def build_metric_table() -> str:
    rows: list[str] = []
    for m in METRIC_ENTRIES:
        rows.append(f"| {m.name} | {m.task} | {m.description} |")
    header = "| Metric | Task | Description |\n|--------|------|-------------|\n"
    return header + "\n".join(rows)


def build_dataset_table() -> str:
    rows: list[str] = []
    for d in DATASET_ENTRIES:
        rows.append(f"| {d.name} | {d.dtype} | {d.description} | {d.dimensions} | {d.segments} | {d.source} |")
    header = (
        "| Dataset | Type | Description | Dimensions | Segments | Source |\n"
        "|---------|------|-------------|------------|----------|--------|\n"
    )
    return header + "\n".join(rows)


DISCLAIMER_MD = (
    "> **⚠️ Disclaimer:** Some algorithms were adapted from research paper artifacts "
    'published without an explicit license (marked "—" above). These are provided '
    "for **research and educational purposes only**. If you are the author of one of "
    "these works and wish to specify licensing terms, please "
    "[open an issue](https://github.com/fchavelli/tsseg/issues)."
)

MOCAP_NOTE_MD = (
    "> The sample dataset uses data from the **CMU Graphics Lab Motion Capture Database** "
    "(NSF EIA-0196217). Labels were sourced from Time2State (Wang et al., 2023). "
    "Only the first rotation axis of left/right humerus and femur is kept, following "
    "the protocol of AutoPlait (Matsubara et al., 2014)."
)

LINKS_MD = """- [tsseg Documentation](https://fchavelli.github.io/tsseg/)
- [GitHub Repository](https://github.com/fchavelli/tsseg)
- [License: AGPLv3](https://github.com/fchavelli/tsseg/blob/main/LICENSE)

Built with [Gradio](https://gradio.app) · Powered by [tsseg](https://github.com/fchavelli/tsseg)"""
