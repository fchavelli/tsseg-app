"""Curated metadata for the About tab; the runtime detector registry supplies its rows."""

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
    AlgoEntry("BOCD", "CPD", "Distribution-based", "O(dn^2)", "MIT", "hildensia"),
    AlgoEntry("ChangeFinder", "CPD", "Distribution-based", "O(dn)", "AGPLv3", "tsseg"),
    AlgoEntry("ChangeForest", "CPD", "Distribution-based", "—", "BSD-3", "changeforest"),
    AlgoEntry("DynP", "CPD", "Distribution-based", "O(n_{cp}\\,dn^2)", "BSD-2", "ruptures"),
    AlgoEntry("EAgglo", "CPD", "Distribution-based", "O(dn^2)", "BSD-3", "aeon"),
    AlgoEntry("GreedyGaussian", "CPD", "Distribution-based", "O(n_{cp}^2 d^3 n)", "BSD-2", "GGS"),
    AlgoEntry("ICID", "CPD", "Distribution-based", "O(dn)", "GPLv3", "IsolationKernel"),
    AlgoEntry("KCPD", "CPD", "Distribution-based", "O(n_{cp}\\,dn^2)", "BSD-2", "ruptures"),
    AlgoEntry("Pelt", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("Prophet", "CPD", "Distribution-based", "O(n_{cp}\\,dn)", "MIT", "Facebook"),
    AlgoEntry("Random", "CPD", "Baseline", "O(n)", "AGPLv3", "tsseg"),
    AlgoEntry("SeededBinSeg", "CPD", "Distribution-based", "O(dn\\log n)", "AGPLv3", "tsseg"),
    AlgoEntry("Window", "CPD", "Distribution-based", "O(dn)", "BSD-2", "ruptures"),
    AlgoEntry("Beast", "CPD", "Distribution-based", "-", "GPL-2.0+", "Rbeast"),
    # ── CPD · Subsequence-based ───────────────────────────────────────────
    AlgoEntry("CLaSP", "CPD", "Subsequence-based", "O(n_{cp}\\,dn^2)", "BSD-3", "claspy"),
    AlgoEntry("ESPRESSO", "CPD", "Subsequence-based", "O(dn^2 + n_{cp}\\,n)", "—", "cruiseresearchgroup"),
    AlgoEntry("FLUSS", "CPD", "Subsequence-based", "O(dn^2 + n_{cp}\\,n)", "BSD-3", "aeon"),
    AlgoEntry("InformationGain", "CPD", "Subsequence-based", "O(n_{cp}\\,dn^2)", "BSD-3", "aeon"),
    AlgoEntry("RuLSIF", "CPD", "Distribution-based", "—", "MIT", "Liu et al."),
    AlgoEntry("TGLAD", "CPD", "Subsequence-based", "O(d^3 n)", "Non-Commercial", "Harshs27"),
    AlgoEntry("Tire", "CPD", "Subsequence-based", "O(dn)", "—", "KU Leuven"),
    AlgoEntry("TSCP2", "CPD", "Subsequence-based", "O(dn)", "—", "cruiseresearchgroup"),
    # ── SD · Integrated ───────────────────────────────────────────────────
    AlgoEntry("AutoPlait", "SD", "Integrated", "O(dn)", "—", "—"),
    AlgoEntry("HdpHsmm", "SD", "Integrated", "O(dn + n_s \\min(n, 2000)^2)", "—", "mattjj/pyhsmm (reimpl.)"),
    AlgoEntry("GaussianHMM", "SD", "Integrated", "—", "AGPLv3", "tsseg"),
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

# Search strategy and comparison axis of the tsseg taxonomy. Keys are the
# detector names shown in the app; candidates are intentionally omitted.
TAXONOMY: dict[str, tuple[str, str]] = {
    "Amoc": ("Optimal DP", "Observation model"),
    "AutoPlait": ("State sequence DP", "Observation model"),
    "Beast": ("Bayesian DP", "Observation model"),
    "BinSeg": ("Binary segmentation", "Selected cost"),
    "BOCD": ("Bayesian DP", "Observation model"),
    "BottomUp": ("Bottom-up", "Selected cost"),
    "ChangeFinder": ("Profile + peaks", "Observation model"),
    "ChangeForest": ("Binary segmentation", "Model-free distribution"),
    "Clap": ("Segment, then group", "Subsequence shapes"),
    "Clasp": ("Binary segmentation", "Subsequence shapes"),
    "Dynp": ("Optimal DP", "Selected cost"),
    "E2USD": ("Cluster windows", "Learned representation"),
    "EAgglo": ("Bottom-up", "Model-free distribution"),
    "Espresso": ("Profile + peaks", "Subsequence shapes"),
    "FLUSS": ("Profile + peaks", "Subsequence shapes"),
    "GaussianHMM": ("State sequence DP", "Observation model"),
    "GreedyGaussian": ("Binary segmentation", "Observation model"),
    "HMM": ("State sequence DP", "Observation model"),
    "HdpHsmm": ("State sequence DP", "Observation model"),
    "Hidalgo": ("Cluster windows", "Observation model"),
    "ICID": ("Profile + peaks", "Model-free distribution"),
    "InformationGain": ("Binary segmentation", "Observation model"),
    "KCPD": ("Optimal DP", "Model-free distribution"),
    "Pelt": ("Optimal DP", "Selected cost"),
    "Prophet": ("Convex relaxation", "Observation model"),
    "Random": ("Baseline", "—"),
    "RuLSIF": ("Profile + peaks", "Model-free distribution"),
    "SeededBinSeg": ("Binary segmentation", "Observation model"),
    "TGLAD": ("Profile + peaks", "Observation model"),
    "Ticc": ("State sequence DP", "Observation model"),
    "Time2State": ("Cluster windows", "Learned representation"),
    "Tire": ("Profile + peaks", "Learned representation"),
    "Window": ("Profile + peaks", "Selected cost"),
}

COMPLEXITY_BY_NAME = {
    "Beast": r"O(dn)",
    "BOCD": r"O(dn+n^3)",
    "BottomUp": r"O(dn\log n)",
    "ChangeForest": "",  # depends on classifier and segmentation tree
    "ChangeFinder": r"O(dn)",
    "Clap": r"O(n_{\mathrm{CP}}n(nd+n_{\mathrm{CP}}\log n_{\mathrm{CP}}))",
    "Dynp": r"O(n_{\mathrm{CP}}dn^2)",
    "E2USD": r"O(dn\ell\log\ell)",
    "Espresso": r"O(dn^2\log n)",
    "GaussianHMM": r"O(n n_{\mathrm{S}}(n_{\mathrm{S}}+d))",
    "GreedyGaussian": r"O(n_{\mathrm{CP}}d^3n)",
    "HMM": r"O(n_{\mathrm{S}}^2n)",
    "HdpHsmm": r"O(dn+n_{\mathrm{S}}(n_{\mathrm{S}}+d^2))",
    "ICID": r"O(dn+n^2)",
    "InformationGain": r"O(n_{\mathrm{CP}}^2dn)",
    "KCPD": "",  # depends on kernel cost and PELT/DynP search mode
    "Prophet": r"O(dn)",
    "RuLSIF": r"O(nb^3)",
    "SeededBinSeg": r"O(dn\log n)",
    "Ticc": r"O(n_{\mathrm{S}}(n(d\ell)^2+(d\ell)^3))",
    "Time2State": r"O(dn\ell)",
    "Tire": r"O(dn\ell)",
    "Window": r"O(dn\ell)",
}

SOURCE_URLS = {
    "Amoc": "https://github.com/deepcharles/ruptures",
    "AutoPlait": "https://www.dm.sanken.osaka-u.ac.jp/~yasuko/software.html",
    "Beast": "https://github.com/zhaokg/Rbeast",
    "BinSeg": "https://github.com/deepcharles/ruptures",
    "BOCD": "https://github.com/hildensia/bayesian_changepoint_detection",
    "BottomUp": "https://github.com/deepcharles/ruptures",
    "ChangeForest": "https://github.com/mlondschien/changeforest",
    "Clap": "https://github.com/ermshaua/clasp",
    "Clasp": "https://github.com/ermshaua/clasp",
    "Dynp": "https://github.com/deepcharles/ruptures",
    "EAgglo": "https://github.com/aeon-toolkit/aeon",
    "Espresso": "https://github.com/cruiseresearchgroup/ESPRESSO",
    "FLUSS": "https://github.com/aeon-toolkit/aeon",
    "GreedyGaussian": "https://github.com/cvxgrp/GGS",
    "HMM": "https://github.com/aeon-toolkit/aeon",
    "HdpHsmm": "https://github.com/mattjj/pyhsmm",
    "Hidalgo": "https://github.com/aeon-toolkit/aeon",
    "ICID": "https://github.com/IsolationKernel/iCID",
    "InformationGain": "https://github.com/aeon-toolkit/aeon",
    "KCPD": "https://github.com/deepcharles/ruptures",
    "Pelt": "https://github.com/deepcharles/ruptures",
    "Prophet": "https://github.com/facebook/prophet",
    "RuLSIF": "https://github.com/anewgithubname/change_detection",
    "TGLAD": "https://github.com/Harshs27/tGLAD",
    "Ticc": "https://github.com/davidhallac/TICC",
    "Time2State": "https://github.com/Lab-ANT/Time2State",
    "Tire": "https://github.com/deryckt/TIRE",
    "TSCP2": "https://github.com/cruiseresearchgroup/TSCP2",
    "E2USD": "https://github.com/AI4CTS/E2Usd",
}

DOC_SLUGS = {
    "Clap": "clap",
    "Clasp": "clap",
    "Dynp": "dynp",
    "EAgglo": "eagglo",
    "GaussianHMM": "gaussian_hmm",
    "GreedyGaussian": "ggs",
    "HdpHsmm": "hdp_hsmm",
    "InformationGain": "igts",
    "SeededBinSeg": "seeded_binseg",
}


def get_complexity(name: str) -> str:
    entry = get_algo_entry(name)
    return COMPLEXITY_BY_NAME.get(name, entry.complexity if entry else "")


def get_algo_entry(name: str) -> AlgoEntry | None:
    """Lookup an algorithm entry by name (case-insensitive)."""
    return ALGO_BY_NAME.get(name.lower())


# ---------------------------------------------------------------------------
# Mathematical complexity notation
# ---------------------------------------------------------------------------

COMPLEXITY_TOOLTIP = (
    "n: time points; d: channels; ℓ: window or subsequence length; "
    "nCP: change points; nS: states; b: kernel basis size. Typical cost per run; "
    "training iterations and classifier costs may add work."
)

LATEX_DELIMITERS = [{"left": "$", "right": "$", "display": False}]


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


def build_algo_table(runtime_detectors: dict) -> str:
    """Return a table of detectors exposed by the running app.

    Curated entries supply metadata, while the runtime registry decides which
    algorithms exist. New detectors without an entry still appear in the table.
    """
    rows: list[str] = []
    for name, info in sorted(runtime_detectors.items(), key=lambda item: item[0].lower()):
        a = get_algo_entry(name)
        task = "CPD" if info.detector_type == "change_point_detection" else "SD"
        search, compared = TAXONOMY.get(name, ("Other", "—"))
        complexity = get_complexity(name)
        license_name = a.license if a else "—"
        source_label = a.source if a and a.source != "—" else "tsseg"
        if name == "AutoPlait":
            source_label = "Original software"
        source_url = SOURCE_URLS.get(name)
        if not source_url:
            module_path = info.cls.__module__.replace(".", "/")
            source_url = f"https://github.com/fchavelli/tsseg/blob/main/{module_path}.py"
        doc_slug = DOC_SLUGS.get(name, name.lower())
        doc_url = f"https://fchavelli.github.io/tsseg/detectors/{doc_slug}.html"
        complexity_text = f"${complexity}$" if complexity and complexity not in ("-", "—") else "—"
        rows.append(
            f"| [{name}]({doc_url}) | {task} | {search} | {compared} | "
            f"{complexity_text} | {license_name} | [{source_label}]({source_url}) |"
        )

    header = (
        '| Algorithm | Task | Search strategy | Compared | Complexity <sup class="complexity-help" '
        f'title="{COMPLEXITY_TOOLTIP}">?</sup> | License | Source |\n'
        "|-----------|------|-----------------|----------|------------|---------|--------|\n"
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
