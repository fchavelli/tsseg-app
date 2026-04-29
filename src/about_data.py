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
    task: str        # "CPD" or "SD"
    family: str      # Distribution-based, Subsequence-based, Integrated, Decomposed
    license: str     # SPDX-like short id, or "—" if unknown
    source: str      # upstream project / author
    status: str      # "stable" or "experimental"


# Ordered: CPD Distribution-based → CPD Subsequence-based → SD Integrated → SD Decomposed
ALGO_ENTRIES: Sequence[AlgoEntry] = (
    # ── CPD · Distribution-based ──────────────────────────────────────────
    AlgoEntry("AMOC",             "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("BinSeg",           "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("BottomUp",         "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("BOCD",             "CPD", "Distribution-based", "Apache-2.0",    "hildensia",            "stable"),
    AlgoEntry("DynP",             "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("EAgglo",           "CPD", "Distribution-based", "BSD-3",         "aeon",                 "stable"),
    AlgoEntry("ICID",             "CPD", "Distribution-based", "GPLv3",         "IsolationKernel",      "stable"),
    AlgoEntry("KCPD",             "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("Pelt",             "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    AlgoEntry("Prophet",          "CPD", "Distribution-based", "MIT",           "Facebook",             "stable"),
    AlgoEntry("Window",           "CPD", "Distribution-based", "BSD-2",         "ruptures",             "stable"),
    # ── CPD · Subsequence-based ───────────────────────────────────────────
    AlgoEntry("CLaSP",            "CPD", "Subsequence-based",  "BSD-3",         "claspy",               "stable"),
    AlgoEntry("ESPRESSO",         "CPD", "Subsequence-based",  "—",             "cruiseresearchgroup",  "stable"),
    AlgoEntry("FLUSS",            "CPD", "Subsequence-based",  "BSD-3",         "aeon",                 "stable"),
    AlgoEntry("IGTS",             "CPD", "Subsequence-based",  "BSD-3",         "aeon",                 "stable"),
    AlgoEntry("TGLAD",            "CPD", "Subsequence-based",  "Non-Commercial","Harshs27",             "stable"),
    AlgoEntry("TiRE",             "CPD", "Subsequence-based",  "—",             "KU Leuven",            "stable"),
    AlgoEntry("TSCP2",            "CPD", "Subsequence-based",  "—",             "cruiseresearchgroup",  "stable"),
    AlgoEntry("GreedyGaussian",   "CPD", "Distribution-based", "BSD-3",         "aeon",                 "stable"),
    # ── SD · Integrated ───────────────────────────────────────────────
    AlgoEntry("AutoPlait",        "SD",  "Integrated",         "—",             "—",                    "experimental"),
    AlgoEntry("HdpHsmm",          "SD",  "Integrated",         "—",             "zhd96",                "stable"),
    AlgoEntry("Hidalgo",          "SD",  "Integrated",         "BSD-3",         "aeon",                 "stable"),
    # ── SD · Decomposed ──────────────────────────────────────────────────
    AlgoEntry("TICC",             "SD",  "Decomposed",         "BSD-2",         "davidhallac",          "stable"),
    AlgoEntry("Time2State",       "SD",  "Decomposed",         "MIT",           "Lab-ANT",              "stable"),
    AlgoEntry("E2USD",            "SD",  "Decomposed",         "—",             "AI4CTS",               "stable"),
    AlgoEntry("CLaP",             "SD",  "Decomposed",         "BSD-3",         "claspy",               "stable"),
)

# Quick lookup by name
ALGO_BY_NAME: dict[str, AlgoEntry] = {a.name: a for a in ALGO_ENTRIES}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricEntry:
    name: str
    task: str       # "CPD" or "SD"
    description: str


METRIC_ENTRIES: Sequence[MetricEntry] = (
    # CPD
    MetricEntry("F1 Score",              "CPD", "Classic precision / recall / F1 with a hard margin tolerance around each true change point"),
    MetricEntry("Gaussian F1",           "CPD", "Gaussian-weighted soft F1 — predictions are rewarded via a Gaussian kernel that decays with distance, no hard margin"),
    MetricEntry("Covering",              "CPD", "Segment-wise Intersection over Union (IoU) weighted by segment duration"),
    MetricEntry("Bidirectional Covering","CPD", "Evaluates coverage in both directions (truth→pred and pred→truth) aggregated by harmonic mean"),
    MetricEntry("Hausdorff Distance",    "CPD", "Maximum over all true CPs of the distance to the nearest predicted CP (and vice-versa)"),
    # SD
    MetricEntry("Adjusted Rand Index",               "SD", "Chance-adjusted agreement between true and predicted label partitions (sklearn)"),
    MetricEntry("Normalized Mutual Information",      "SD", "Information-theoretic agreement normalised to [0, 1] (sklearn)"),
    MetricEntry("Adjusted Mutual Information",        "SD", "Chance-adjusted variant of NMI (sklearn)"),
    MetricEntry("Weighted ARI",                       "SD", "ARI variant that gives more weight to samples near segment boundaries"),
    MetricEntry("Weighted NMI",                       "SD", "NMI variant that gives more weight to samples near segment boundaries"),
    MetricEntry("State Matching Score",               "SD", "Hungarian matching of predicted to true states with fine-grained error classification"),
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetEntry:
    name: str
    dtype: str       # "Generated" or "Real-world"
    description: str
    dimensions: str
    segments: str
    source: str


DATASET_ENTRIES: Sequence[DatasetEntry] = (
    DatasetEntry(
        "Synthetic", "Generated",
        "Configurable generator with 6 intra-segment patterns "
        "(constant, sinusoidal, step, stairs, linear, mixed), noise control, recurring states",
        "1–8", "2–12", "tsseg",
    ),
    DatasetEntry(
        "CMU MoCap 86", "Real-world",
        "Motion capture from CMU subject 86 — humeral & femoral angles, 9 labelled trials",
        "4", "3–8", "[CMU MoCap](https://mocap.cs.cmu.edu/)",
    ),
)


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------

def build_algo_table(runtime_detectors: dict | None = None) -> str:
    """Return a Markdown table of algorithms.

    If *runtime_detectors* is provided (name→DetectorInfo mapping from
    ``get_all_detectors()``), a runtime availability column is appended.
    """
    rows: list[str] = []
    current_group = ""
    for a in ALGO_ENTRIES:
        group_label = f"{a.task} · {a.family}"
        if group_label != current_group:
            current_group = group_label
            # Visible group separator row
            rows.append(f"| **{group_label}** | | | | | |")

        status_icon = "🟢" if a.status == "stable" else "🟠"
        status_label = f"{status_icon} {a.status}"
        rows.append(
            f"| {a.name} | {a.task} | {a.family} | {a.license} | {a.source} | {status_label} |"
        )

    header = (
        "| Algorithm | Task | Family | License | Source | Status |\n"
        "|-----------|------|--------|---------|--------|--------|\n"
    )
    return header + "\n".join(rows)


def build_metric_table() -> str:
    rows: list[str] = []
    current_task = ""
    for m in METRIC_ENTRIES:
        if m.task != current_task:
            current_task = m.task
        rows.append(f"| {m.name} | {m.task} | {m.description} |")
    header = (
        "| Metric | Task | Description |\n"
        "|--------|------|-------------|\n"
    )
    return header + "\n".join(rows)


def build_dataset_table() -> str:
    rows: list[str] = []
    for d in DATASET_ENTRIES:
        rows.append(
            f"| {d.name} | {d.dtype} | {d.description} | {d.dimensions} | {d.segments} | {d.source} |"
        )
    header = (
        "| Dataset | Type | Description | Dimensions | Segments | Source |\n"
        "|---------|------|-------------|------------|----------|--------|\n"
    )
    return header + "\n".join(rows)


DISCLAIMER_MD = (
    "> **⚠️ Disclaimer :** Some algorithms were adapted from research paper artifacts "
    "published without an explicit license (marked \"—\" above). These are provided "
    "for **research and educational purposes only**. If you are the author of one of "
    "these works and wish to specify licensing terms, please "
    "[open an issue](https://github.com/fchavelli/tsseg/issues)."
)

MOCAP_NOTE_MD = (
    "> The MoCap subset uses data from the **CMU Graphics Lab Motion Capture Database** "
    "(NSF EIA-0196217). Labels were sourced from Time2State (Wang et al., 2023). "
    "Only the first rotation axis of left/right humerus and femur is kept, following "
    "the protocol of AutoPlait (Matsubara et al., 2014)."
)

LINKS_MD = """- [tsseg Documentation](https://fchavelli.github.io/tsseg/)
- [GitHub Repository](https://github.com/fchavelli/tsseg)
- [License: AGPLv3](https://github.com/fchavelli/tsseg/blob/main/LICENSE)

Built with [Gradio](https://gradio.app) · Powered by [tsseg](https://github.com/fchavelli/tsseg)"""
