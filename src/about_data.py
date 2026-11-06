"""Curated metadata for the About tab; the runtime detector registry supplies its rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoEntry:
    """Licence and upstream source of a detector (placement: ``src/taxonomy.py``)."""

    name: str  # detector name in the app (class name without "Detector")
    license: str  # SPDX-like short id, or "—" if unknown
    source: str  # upstream project / author


ALGO_ENTRIES: Sequence[AlgoEntry] = (
    AlgoEntry("AMOC", "BSD-2", "ruptures"),
    AlgoEntry("AutoPlait", "—", "—"),
    AlgoEntry("Beast", "GPL-2.0+", "Rbeast"),
    AlgoEntry("BinSeg", "BSD-2", "ruptures"),
    AlgoEntry("BOCD", "MIT", "hildensia"),
    AlgoEntry("BottomUp", "BSD-2", "ruptures"),
    AlgoEntry("ChangeFinder", "AGPLv3", "tsseg"),
    AlgoEntry("ChangeForest", "BSD-3", "changeforest"),
    AlgoEntry("Clap", "BSD-3", "claspy"),
    AlgoEntry("CLaSP", "BSD-3", "claspy"),
    AlgoEntry("DynP", "BSD-2", "ruptures"),
    AlgoEntry("E2USD", "—", "AI4CTS"),
    AlgoEntry("EAgglo", "BSD-3", "aeon"),
    AlgoEntry("ESPRESSO", "—", "cruiseresearchgroup"),
    AlgoEntry("FLUSS", "BSD-3", "aeon"),
    AlgoEntry("GaussianHMM", "AGPLv3", "tsseg"),
    AlgoEntry("GreedyGaussian", "BSD-2", "GGS"),
    AlgoEntry("HdpHsmm", "—", "mattjj/pyhsmm (reimpl.)"),
    AlgoEntry("Hidalgo", "BSD-3", "aeon"),
    AlgoEntry("HMM", "BSD-3", "hmmlearn"),
    AlgoEntry("ICID", "GPLv3", "IsolationKernel"),
    AlgoEntry("InformationGain", "BSD-3", "aeon"),
    AlgoEntry("KCPD", "BSD-2", "ruptures"),
    AlgoEntry("Pelt", "BSD-2", "ruptures"),
    AlgoEntry("Prophet", "MIT", "Facebook"),
    AlgoEntry("Random", "AGPLv3", "tsseg"),
    AlgoEntry("RuLSIF", "MIT", "Liu et al."),
    AlgoEntry("SeededBinSeg", "AGPLv3", "tsseg"),
    AlgoEntry("SST", "AGPLv3", "tsseg"),
    AlgoEntry("TGLAD", "Non-Commercial", "Harshs27"),
    AlgoEntry("TICC", "BSD-2", "davidhallac"),
    AlgoEntry("Time2State", "MIT", "Lab-ANT"),
    AlgoEntry("Tire", "—", "KU Leuven"),
    AlgoEntry("TSCP2", "—", "cruiseresearchgroup"),
    AlgoEntry("Window", "BSD-2", "ruptures"),
)

# Quick lookup by name (case-insensitive)
ALGO_BY_NAME: dict[str, AlgoEntry] = {a.name.lower(): a for a in ALGO_ENTRIES}

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


def get_algo_entry(name: str) -> AlgoEntry | None:
    """Lookup an algorithm entry by name (case-insensitive)."""
    return ALGO_BY_NAME.get(name.lower())


# ---------------------------------------------------------------------------
# Mathematical complexity notation
# ---------------------------------------------------------------------------

COMPLEXITY_TOOLTIP = (
    "n: time points; d: channels; ℓ: window or subsequence length; "
    "nCP: change points; nS: states; b: kernel basis size. Time of one run as implemented, "
    "default settings; epochs and sampler or EM iterations omitted. "
    "*: typical case of a search that prunes or stops early."
)
COMPLEXITY_HELP_HTML = (
    '<span class="complexity-help" tabindex="0" aria-label="Complexity notation">?'
    f'<span class="complexity-tooltip" role="tooltip">{COMPLEXITY_TOOLTIP}</span></span>'
)

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
        "Gaussian-weighted soft F1: predictions are rewarded via a Gaussian kernel "
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
    """HTML table of the detectors exposed by the running app.

    The runtime registry decides which detectors exist; their placement,
    complexity and documentation page come from the taxonomy of the
    documentation (``src/taxonomy.py``), licence and source from
    ``ALGO_ENTRIES``. Rows follow the taxonomy: task, then search procedure.
    """
    from html import escape

    from . import taxonomy as tx

    order = list(tx.SEARCHES)

    def key(item):
        name, info = item
        m = tx.method(info.cls.__name__)
        task = 0 if info.detector_type == "change_point_detection" else 1
        search = order.index(m["search"]) if m else len(order)
        return task, search, tx.display_name(info.cls.__name__, name).lower()

    rows: list[str] = []
    for name, info in sorted(runtime_detectors.items(), key=key):
        cls_name = info.cls.__name__
        m = tx.method(cls_name)
        a = get_algo_entry(name)
        is_cpd = info.detector_type == "change_point_detection"
        task = '<span class="ts-tag is-cpd">CPD</span>' if is_cpd else '<span class="ts-tag is-sd">SD</span>'
        search, criterion = tx.placement(cls_name) or ("—", "—")
        complexity = tx.complexity_html(cls_name) or "—"
        license_name = a.license if a else "—"
        source_label = a.source if a and a.source != "—" else "tsseg"
        if name == "AutoPlait":
            source_label = "Original software"
        source_url = SOURCE_URLS.get(name)
        if not source_url:
            module_path = info.cls.__module__.replace(".", "/")
            source_url = f"https://github.com/fchavelli/tsseg/blob/main/{module_path}.py"
        doc = tx.doc_url(m["page"]) if m else tx.DOCS_URL + "detectors/index.html"
        label = escape(tx.display_name(cls_name, name))
        where = (
            f'<a class="ts-grid-link" href="{tx.taxonomy_link(cls_name)}" target="_blank" rel="noopener" '
            f'title="Show it in the taxonomy of the documentation">{escape(search)}</a>'
            if m
            else escape(search)
        )
        rows.append(
            "<tr>"
            f'<td><a href="{doc}" target="_blank" rel="noopener"><b>{label}</b></a>'
            f"<code>{escape(cls_name)}</code></td>"
            f"<td>{task}</td><td>{where}</td><td>{escape(criterion)}</td>"
            f'<td class="ts-cx">{complexity}</td><td>{escape(license_name)}</td>'
            f'<td><a href="{source_url}" target="_blank" rel="noopener">{escape(source_label)}</a></td>'
            "</tr>"
        )
    head = (
        "<thead><tr><th>Detector</th><th>Task</th><th>Search procedure</th>"
        f"<th>Homogeneity criterion</th><th>Complexity {COMPLEXITY_HELP_HTML}</th>"
        "<th>Licence</th><th>Source</th></tr></thead>"
    )
    return f'<div class="ts-table-wrap"><table class="ts-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'


def build_metric_table() -> str:
    from html import escape

    rows = "".join(
        f"<tr><td><b>{escape(m.name)}</b></td>"
        f'<td><span class="ts-tag {"is-cpd" if m.task == "CPD" else "is-sd"}">{m.task}</span></td>'
        f"<td>{escape(m.description)}</td></tr>"
        for m in METRIC_ENTRIES
    )
    head = "<thead><tr><th>Metric</th><th>Task</th><th>Description</th></tr></thead>"
    return f'<div class="ts-table-wrap"><table class="ts-table">{head}<tbody>{rows}</tbody></table></div>'


def build_dataset_table() -> str:
    import re
    from html import escape

    def link(md: str) -> str:
        m = re.fullmatch(r"\[(.+)\]\((.+)\)", md)
        return f'<a href="{m[2]}" target="_blank" rel="noopener">{escape(m[1])}</a>' if m else escape(md)

    rows = "".join(
        f"<tr><td><b>{escape(d.name)}</b></td><td>{escape(d.dtype)}</td><td>{escape(d.description)}</td>"
        f"<td>{escape(d.dimensions)}</td><td>{escape(d.segments)}</td><td>{link(d.source)}</td></tr>"
        for d in DATASET_ENTRIES
    )
    head = (
        "<thead><tr><th>Dataset</th><th>Type</th><th>Description</th><th>Channels</th>"
        "<th>Segments</th><th>Source</th></tr></thead>"
    )
    return f'<div class="ts-table-wrap"><table class="ts-table">{head}<tbody>{rows}</tbody></table></div>'


DISCLAIMER_HTML = (
    '<p class="ts-note"><b>Licences.</b> Some detectors are adapted from research code published without '
    "a licence (marked “—”). They are provided for research and educational purposes only. If you are the "
    "author of one of these works and wish to specify licensing terms, please "
    '<a href="https://github.com/fchavelli/tsseg/issues" target="_blank" rel="noopener">open an issue</a>.</p>'
)

MOCAP_NOTE_HTML = (
    '<p class="ts-note">The sample dataset uses data from the <b>CMU Graphics Lab Motion Capture Database</b> '
    "(NSF EIA-0196217). Labels were sourced from Time2State (Wang et al., 2023). Only the first rotation axis "
    "of the left and right humerus and femur is kept, following the protocol of AutoPlait (Matsubara et al., "
    '2014). Built with <a href="https://gradio.app" target="_blank" rel="noopener">Gradio</a>, powered '
    'by <a href="https://github.com/fchavelli/tsseg" target="_blank" rel="noopener">tsseg</a> '
    '(<a href="https://github.com/fchavelli/tsseg/blob/main/LICENSE" target="_blank" rel="noopener">AGPL-3.0</a>).</p>'
)
