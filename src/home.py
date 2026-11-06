"""HTML of the top bar, the Home tab and the tab headings, in the style of the documentation."""

from __future__ import annotations

from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

DOCS = "https://fchavelli.github.io/tsseg/"
GITHUB = "https://github.com/fchavelli/tsseg"
PAPER = "https://doi.org/10.1007/978-3-032-37685-5_41"

_ARROW_OUT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5h6.5V10M12.5 3.5L4 12"/></svg>'
_GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 '
    "7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28"
    "-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64"
    "-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68"
    " 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75"
    "-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58"
    '-8-8-8z"/></svg>'
)


def _a(href: str, body: str, cls: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f'<a{klass} href="{href}" target="_blank" rel="noopener">{body}</a>'


@lru_cache(maxsize=None)
def svg_data_uri(name: str) -> str:
    data = (ASSETS / name).read_bytes()
    return "data:image/svg+xml;base64," + b64encode(data).decode()


def header_html() -> str:
    """Top bar: logo, name and links to the documentation, paper and code."""
    return (
        '<nav class="ts-nav">'
        + _a(
            DOCS,
            f'<img class="ts-logo-light" src="{svg_data_uri("logo-mark.svg")}" alt="">'
            f'<img class="ts-logo-dark" src="{svg_data_uri("logo-mark-dark.svg")}" alt="">'
            "<b>tsseg</b><span>demo</span>",
            "ts-brand",
        )
        + '<div class="ts-nav-links">'
        + _a(DOCS, "Documentation")
        + _a(DOCS + "taxonomy.html", "Taxonomy")
        + _a(DOCS + "evaluation.html", "Metrics")
        + _a(PAPER, f"Paper {_ARROW_OUT}")
        + _a(GITHUB, f"{_GITHUB_ICON}GitHub", "ts-gh")
        + "</div></nav>"
    )


def hero_text_html(n_detectors: int) -> str:
    return (
        _a(PAPER, "<b>ECML-PKDD 2026</b><span>Read the demo paper</span>", "ts-pill")
        + '<p class="ts-title">Time series <span class="ts-grad">segmentation</span>, in your browser</p>'
        + f'<p class="ts-sub">Load a series, run any of {n_detectors} detectors of tsseg, compare their '
        "change points and states, and score them against the truth. Nothing to install.</p>"
    )


def hero_art_html() -> str:
    """The animated segmented series of the documentation's home page."""
    svg = (ASSETS / "hero.svg").read_text(encoding="utf-8")
    return (
        '<div class="ts-art"><div class="ts-art-bar"><i></i><i></i><i></i>'
        f"<span>detector.fit_predict(X)</span></div>{svg}</div>"
    )


def stats_html(n_detectors: int, n_metrics: int) -> str:
    stats = [
        (str(n_detectors), "detectors, classic to deep"),
        ("2", "tasks: change points and states"),
        (str(n_metrics), "evaluation metrics"),
        ("9", "labelled MoCap trials, plus a generator"),
    ]
    return (
        '<div class="ts-stats">'
        + "".join(f'<div class="ts-stat"><b>{escape(v)}</b><span>{escape(t)}</span></div>' for v, t in stats)
        + "</div>"
    )


_TASKS = (ASSETS / "tasks.html").read_text(encoding="utf-8")

_STEPS = [
    (
        "Data",
        "Load a motion-capture recording with its ground truth, generate a synthetic series with "
        "recurring states, or upload a CSV.",
    ),
    (
        "Segmentation",
        "Pick a detector and tune its parameters. <i>Use GT</i> passes the true number of change points or states.",
    ),
    ("Comparison", "Add runs to the comparison to see several detectors on the same series."),
    (
        "Evaluation",
        "Score every run against the truth, and split the State Matching Score into its four error types.",
    ),
]


def home_body_html() -> str:
    """Two tasks, four steps, the figure of the sample dataset and the paper."""
    steps = "".join(f'<div class="ts-step"><h3>{t}</h3><p>{d}</p></div>' for t, d in _STEPS)
    figure = (
        '<figure class="ts-figure">'
        f'<img src="{svg_data_uri("demo.svg")}" alt="The sample dataset of the demo, a four-channel '
        "motion-capture recording: the ground truth (seven segments in four states, state A coming back "
        "three times), the change points found by BottomUp with an RBF cost without being told how many "
        '(Covering 0.75), and the states found by Time2State told that there are four (State Matching Score 0.93)">'
        "<figcaption>The sample dataset, segmented in this demo: change points of BottomUp (unsupervised) "
        "and states of Time2State (told the number of states), each scored against the truth.</figcaption>"
        "</figure>"
    )
    cite = (
        '<div class="ts-cite"><div>'
        '<span class="ts-venue">ECML-PKDD 2026 · Demo track · LNCS</span>'
        "<h3>tsseg: An Interactive Toolkit for Time Series Segmentation</h3>"
        "<p>Félix Chavelli, Arik Ermshaus, Fan Yang, Patrick Schäfer, John Paparrizos, Paul Boniol</p>"
        '</div><div class="ts-links">'
        + _a(PAPER, f"Read the paper {_ARROW_OUT}", "ts-btn")
        + _a(GITHUB + "#citation", "Cite tsseg", "ts-btn")
        + "</div></div>"
    )
    return (
        _TASKS
        + '<div class="ts-section"><span class="ts-kicker">How it works</span>'
        + '<p class="ts-h2">Four tabs, from a series to a score</p></div>'
        + f'<div class="ts-steps">{steps}</div>'
        + '<div class="ts-section"><span class="ts-kicker">Example</span>'
        + '<p class="ts-h2">What a run looks like</p></div>'
        + figure
        + '<div class="ts-section"></div>'
        + cite
    )


def tab_head_html(kicker: str, title: str, lead: str = "") -> str:
    """Heading of a tool tab: kicker, title and one sentence."""
    lead_html = f'<p class="ts-lead">{lead}</p>' if lead else ""
    return f'<div class="ts-head"><span class="ts-kicker">{kicker}</span><p class="ts-h2">{title}</p>{lead_html}</div>'
