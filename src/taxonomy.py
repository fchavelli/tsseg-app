"""The detector taxonomy of the tsseg documentation, for the app.

Every detector is placed by its search procedure (how a segmentation is searched
for) and its homogeneity criterion (what makes a segment homogeneous), as on
https://fchavelli.github.io/tsseg/taxonomy.html and in the TSB-SEG benchmark.

The declarations come from ``taxonomy_spec.json``, a copy of the documentation's
``docs/_ext/tsseg_taxonomy.py`` made by ``scripts/sync_taxonomy.py``. The
capabilities (unsupervised or guided, univariate or multivariate, optional
extra) are read from the estimator tags of the installed tsseg, as the
documentation does at build time. ``assets/taxonomy/taxonomy.js`` renders the
interactive grid from :func:`taxonomy_data`.
"""

from __future__ import annotations

import html
import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS_URL = "https://fchavelli.github.io/tsseg/"
TAXONOMY_URL = DOCS_URL + "taxonomy.html"
TSB_SEG_URL = "https://github.com/fchavelli/TSB-SEG"

SPEC: dict = json.loads(Path(__file__).with_name("taxonomy_spec.json").read_text(encoding="utf-8"))
SEARCHES = {s["key"]: s for s in SPEC["searches"]}
CRITERIA = {c["key"]: c for c in SPEC["criteria"]}
COST_ROWS = [c["row"] for c in SPEC["costs"]]
METHODS = {m["cls"]: m for m in SPEC["methods"]}
TASKS = {"change_point_detection": "cpd", "state_detection": "sd"}
TASK_NAMES = {"cpd": "Change point detection", "sd": "State detection"}


def doc_url(page: str) -> str:
    return f"{DOCS_URL}detectors/{page}.html"


def method(cls_name: str) -> dict | None:
    """Declaration of a detector class (``"PeltDetector"``), or None."""
    return METHODS.get(cls_name)


def display_name(cls_name: str, fallback: str) -> str:
    """Name used by the documentation (``"PELT"``, ``"Gaussian HMM"``)."""
    m = method(cls_name)
    return m["name"] if m else fallback


def rows_of(m: dict) -> list[str]:
    return COST_ROWS if m["row"] == "bar" else [m["row"]]


def criterion_label(m: dict) -> str:
    """Homogeneity criterion as the documentation writes it."""
    if m["row"] == "bar":
        label = " or ".join(CRITERIA[r]["tag"] for r in COST_ROWS) + ", by cost"
        return label[0].upper() + label[1:]
    return CRITERIA[m["row"]]["name"]


def placement(cls_name: str) -> tuple[str, str] | None:
    """``(search procedure, homogeneity criterion)`` of a detector class."""
    m = method(cls_name)
    if m is None:
        return None
    return SEARCHES[m["search"]]["name"], criterion_label(m)


def complexity_html(cls_name: str) -> str:
    """Time complexity as HTML (``O(<i>d</i><i>n</i>)``), empty when unknown."""
    m = method(cls_name)
    if not m or not m["cx_html"]:
        return ""
    return m["cx_html"] + ("*" if m["typical"] else "")


def taxonomy_link(cls_name: str) -> str:
    return f"{TAXONOMY_URL}#m={cls_name}"


def _extras_of(deps, extras_table) -> list[str]:
    """Smallest set of extras installing ``deps`` (as the documentation does)."""
    if not deps:
        return []
    deps = set(deps)
    single = [e for e, mods in extras_table.items() if deps <= set(mods)]
    if single:
        return [min(single, key=lambda e: len(extras_table[e]))]
    found = []
    for dep in sorted(deps):
        owners = [e for e, mods in extras_table.items() if dep in mods]
        name = min(owners, key=lambda e: len(extras_table[e])) if owners else dep
        if name not in found:
            found.append(name)
    return found


def _tags(cls) -> dict:
    get = cls.get_class_tag
    return {
        "task": TASKS.get(get("detector_type")),
        "unsup": bool(get("capability:unsupervised", False)),
        "guided": bool(get("capability:semi_supervised", False)),
        "uni": bool(get("capability:univariate", False)),
        "mv": bool(get("capability:multivariate", False)),
        "deps": list(get("python_dependencies", None) or []),
    }


@lru_cache(maxsize=1)
def taxonomy_data() -> dict:
    """Data of the interactive grid, in the format of the documentation."""
    import tsseg.algorithms as algorithms

    try:
        from tsseg.algorithms._dependencies import EXTRAS
    except ImportError:  # older tsseg
        EXTRAS = {}

    methods = []
    for spec in SPEC["methods"]:
        cls = getattr(algorithms, spec["cls"], None) if spec["cls"] in algorithms.__all__ else None
        if cls is None:
            logger.info("Taxonomy: %s is not in the installed tsseg", spec["cls"])
            continue
        tags = _tags(cls)
        guidance = ("opt" if tags["unsup"] else "req") if tags["guided"] else None
        methods.append(
            {
                "id": spec["cls"],
                "name": spec["name"],
                "cls": spec["cls"],
                "url": doc_url(spec["page"]),
                "task": tags["task"] or SEARCHES[spec["search"]]["task"],
                "search": spec["search"],
                "rows": rows_of(spec),
                "bar": spec["row"] == "bar",
                "cx": spec["cx_html"],
                "typical": spec["typical"],
                "guidance": guidance,
                "unsup": tags["unsup"],
                "guided": tags["guided"],
                "uni": tags["uni"],
                "mv": tags["mv"],
                "extras": _extras_of(tags["deps"], EXTRAS),
                "deep": spec["deep"],
                "bench": spec["bench"],
                "status": spec["status"],
                "approach": spec["approach"],
                "ref": spec["ref"],
            }
        )
    for i, lit in enumerate(SPEC["literature"]):
        methods.append(
            {
                "id": f"lit-{i}",
                "name": lit["name"],
                "task": SEARCHES[lit["search"]]["task"],
                "search": lit["search"],
                "rows": [lit["row"]],
                "bar": False,
                "status": "literature",
                "ref": lit["ref"],
            }
        )
    return {
        "searches": SPEC["searches"],
        "criteria": SPEC["criteria"],
        "costs": SPEC["costs"],
        "methods": methods,
        "excluded": [{"cls": x["cls"], "url": doc_url(x["page"]), "why": x["why"]} for x in SPEC["excluded"]],
        "dpNote": SPEC["dpNote"],
        "tsbseg": TSB_SEG_URL,
    }


def _fallback_html(data: dict) -> str:
    """Plain nested list of links, shown until the script renders the grid."""
    crit = {c["key"]: c["name"] for c in data["criteria"]}
    parts = ['<div class="tx-fallback">']
    for task, title in TASK_NAMES.items():
        parts.append(f"<h3>{title}</h3><dl>")
        for s in (s for s in data["searches"] if s["task"] == task):
            items = [m for m in data["methods"] if m["search"] == s["key"] and m["status"] != "literature"]
            links = ", ".join(
                f'<a href="{html.escape(m["url"])}" target="_blank" rel="noopener">{html.escape(m["name"])}</a>'
                f" ({html.escape(' / '.join(crit[r] for r in m['rows']))})"
                for m in items
            )
            parts.append(f"<dt>{html.escape(s['name'])}</dt><dd>{links}</dd>")
        parts.append("</dl>")
    parts.append("</div>")
    return "".join(parts)


def taxonomy_html() -> str:
    """Mount point and data of the interactive grid (rendered by taxonomy.js)."""
    data = taxonomy_data()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<div class="tx-app" id="tsseg-taxonomy" data-state="loading">'
        f"{_fallback_html(data)}</div>"
        f'<script type="application/json" id="tsseg-taxonomy-data">{payload}</script>'
    )
