"""Copy the detector taxonomy of the tsseg documentation into the app.

The documentation declares the taxonomy (search procedure x homogeneity
criterion, complexity, reference of every detector) in
``docs/_ext/tsseg_taxonomy.py`` and renders it with ``docs/_static/js/taxonomy.js``
and ``docs/_static/css/taxonomy.css``. The app shows the same grid: run this
script after the taxonomy changes in tsseg, then commit the three outputs.

Usage, from the root of tsseg-app (needs Sphinx, as the documentation build)::

    python scripts/sync_taxonomy.py [path/to/tsseg]     # default: ../tsseg

Writes:

* ``src/taxonomy_spec.json``: the declarations (placement, complexity,
  approach, reference); capabilities are read at runtime from the estimator
  tags of the installed tsseg, as the documentation does at build time;
* ``assets/taxonomy/taxonomy.js`` and ``assets/taxonomy/taxonomy.css``: copies,
  with one change to the script: the Enter key opens the only match in a new
  tab instead of the app's frame (clicks are redirected by ``src/style.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JS_PATCHES = [
    ("window.location.href = on[0].url;", 'window.open(on[0].url, "_blank", "noopener");'),
]


def load_extension(tsseg: Path):
    path = tsseg / "docs" / "_ext" / "tsseg_taxonomy.py"
    spec = importlib.util.spec_from_file_location("tsseg_taxonomy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_spec(ext) -> dict:
    methods = []
    for m in ext.METHODS:
        methods.append(
            {
                "name": m["name"],
                "cls": m["cls"],
                "page": m["page"],
                "search": m["search"],
                "row": "bar" if m["row"] == ext.BAR else m["row"],
                "approach": m["approach"],
                "ref": m["ref"],
                "cx": m["cx"],
                "cx_html": ext.complexity_html(m["cx"]) if m["cx"] else None,
                "deep": m["deep"],
                "status": m["status"],
                "bench": m["name"] in ext.TSB_SEG,
                "typical": m["name"] in ext.TYPICAL,
            }
        )
    return {
        "source": "tsseg docs/_ext/tsseg_taxonomy.py (scripts/sync_taxonomy.py)",
        "searches": ext.SEARCHES,
        "criteria": ext.CRITERIA,
        "costs": [{"row": r, "label": label} for r, label in ext.COSTS],
        "dpNote": ext.DP_NOTE,
        "methods": methods,
        "literature": [{"name": n, "search": s, "row": r, "ref": ref} for n, s, r, ref in ext.LITERATURE],
        "excluded": [{"cls": c, "page": p, "why": w} for c, (p, w) in ext.EXCLUDED.items()],
    }


def main() -> None:
    tsseg = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "tsseg"
    ext = load_extension(tsseg)
    spec_path = ROOT / "src" / "taxonomy_spec.json"
    spec_path.write_text(json.dumps(build_spec(ext), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    out = ROOT / "assets" / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    header = "Copied from tsseg {} by scripts/sync_taxonomy.py: do not edit here."
    js = (tsseg / "docs" / "_static" / "js" / "taxonomy.js").read_text(encoding="utf-8")
    for old, new in JS_PATCHES:
        if js.count(old) != 1:
            raise SystemExit(f"taxonomy.js changed: cannot apply the patch {old!r}")
        js = js.replace(old, new)
    (out / "taxonomy.js").write_text(f"/* {header.format('docs/_static/js/taxonomy.js')} */\n" + js, encoding="utf-8")
    css = (tsseg / "docs" / "_static" / "css" / "taxonomy.css").read_text(encoding="utf-8")
    (out / "taxonomy.css").write_text(
        f"/* {header.format('docs/_static/css/taxonomy.css')} */\n" + css, encoding="utf-8"
    )
    print(f"{spec_path.relative_to(ROOT)}: {len(ext.METHODS)} detectors, {len(ext.LITERATURE)} from the literature")


if __name__ == "__main__":
    main()
