"""Look of the app: the palette, type and components of the tsseg documentation.

The documentation (https://fchavelli.github.io/tsseg/) uses the PyData Sphinx
theme with Inter, a blue primary, blue for change point detection, orange for
state detection and green as a third accent, in light and dark modes. The
Gradio theme below maps the same values on Gradio's variables; ``APP_CSS``
adds the landing components (hero, stats, cards) and the shims the vendored
taxonomy needs; ``build_head`` loads the taxonomy script.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# --------------------------------------------------------------------------
# Gradio theme
# --------------------------------------------------------------------------

BLUE = gr.themes.Color(
    name="tsseg_blue",
    c50="#eef3fb",
    c100="#e6eefa",
    c200="#c9daf3",
    c300="#a3bfea",
    c400="#6f97d8",
    c500="#3f73c2",
    c600="#2d5fb4",
    c700="#234c94",
    c800="#1b3f80",
    c900="#162f5e",
    c950="#0f1f3d",
)

APP_THEME = gr.themes.Base(
    primary_hue=BLUE,
    secondary_hue=BLUE,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f5f7fa",
    body_background_fill_dark="#0e1217",
    body_text_color="#1f2a3a",
    body_text_color_dark="#dbe2ec",
    body_text_color_subdued="#5b6b80",
    body_text_color_subdued_dark="#94a1b4",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#161c24",
    background_fill_secondary="#f5f7fa",
    background_fill_secondary_dark="#141920",
    border_color_primary="#e3e8ef",
    border_color_primary_dark="#2b3340",
    border_color_accent="#2d5fb4",
    border_color_accent_dark="#86acf0",
    color_accent="#2d5fb4",
    color_accent_soft="#e6eefa",
    color_accent_soft_dark="#1c2a44",
    link_text_color="#2d5fb4",
    link_text_color_dark="#86acf0",
    link_text_color_hover="#1b3f80",
    link_text_color_hover_dark="#b8cff7",
    link_text_color_visited="#2d5fb4",
    link_text_color_visited_dark="#86acf0",
    block_background_fill="#ffffff",
    block_background_fill_dark="#161c24",
    block_border_color="#e3e8ef",
    block_border_color_dark="#2b3340",
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 1px 2px rgb(15 23 42 / 4%)",
    block_shadow_dark="none",
    block_title_text_weight="600",
    block_title_text_color="#1f2a3a",
    block_title_text_color_dark="#dbe2ec",
    block_label_text_weight="600",
    block_label_text_color="#5b6b80",
    block_label_text_color_dark="#94a1b4",
    block_label_background_fill="#ffffff",
    block_label_background_fill_dark="#161c24",
    panel_background_fill="#ffffff",
    panel_background_fill_dark="#161c24",
    panel_border_color="#e3e8ef",
    panel_border_color_dark="#2b3340",
    input_background_fill="#ffffff",
    input_background_fill_dark="#0f141a",
    input_background_fill_focus="#ffffff",
    input_background_fill_focus_dark="#0f141a",
    input_border_color="#d5dce5",
    input_border_color_dark="#2b3340",
    input_border_color_focus="#2d5fb4",
    input_border_color_focus_dark="#86acf0",
    input_radius="10px",
    input_shadow_focus="0 0 0 3px rgb(45 95 180 / 18%)",
    input_shadow_focus_dark="0 0 0 3px rgb(134 172 240 / 22%)",
    checkbox_background_color_selected="#2d5fb4",
    checkbox_background_color_selected_dark="#86acf0",
    checkbox_border_color_selected="#2d5fb4",
    checkbox_border_color_selected_dark="#86acf0",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_dark="#161c24",
    checkbox_label_border_color="#e3e8ef",
    checkbox_label_border_color_dark="#2b3340",
    checkbox_label_background_fill_selected="#e6eefa",
    checkbox_label_background_fill_selected_dark="#1c2a44",
    checkbox_label_border_color_selected="#2d5fb4",
    checkbox_label_border_color_selected_dark="#86acf0",
    checkbox_label_text_color_selected="#1b3f80",
    checkbox_label_text_color_selected_dark="#dbe2ec",
    slider_color="#2d5fb4",
    slider_color_dark="#86acf0",
    button_large_radius="11px",
    button_medium_radius="10px",
    button_small_radius="9px",
    button_large_text_weight="650",
    button_medium_text_weight="600",
    button_primary_background_fill="#2d5fb4",
    button_primary_background_fill_dark="#86acf0",
    button_primary_background_fill_hover="#1b3f80",
    button_primary_background_fill_hover_dark="#b8cff7",
    button_primary_border_color="#2d5fb4",
    button_primary_border_color_dark="#86acf0",
    button_primary_border_color_hover="#1b3f80",
    button_primary_border_color_hover_dark="#b8cff7",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#0e1217",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#1d242e",
    button_secondary_background_fill_hover="#f5f7fa",
    button_secondary_background_fill_hover_dark="#232b36",
    button_secondary_border_color="#e3e8ef",
    button_secondary_border_color_dark="#2b3340",
    button_secondary_text_color="#0f172a",
    button_secondary_text_color_dark="#e8edf4",
    button_cancel_background_fill="#ffffff",
    button_cancel_background_fill_dark="#1d242e",
    button_cancel_border_color="#f1c6a8",
    button_cancel_border_color_dark="#5a3320",
    button_cancel_text_color="#a7461a",
    button_cancel_text_color_dark="#f39b63",
    code_background_fill="#f5f7fa",
    code_background_fill_dark="#141920",
    table_border_color="#e3e8ef",
    table_border_color_dark="#2b3340",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#161c24",
    table_odd_background_fill="#f9fafb",
    table_odd_background_fill_dark="#141920",
    shadow_drop="0 1px 2px rgb(15 23 42 / 6%)",
    shadow_spread="3px",
)

# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

APP_CSS = (ASSETS / "app.css").read_text(encoding="utf-8")

# --------------------------------------------------------------------------
# Head: theme attribute, taxonomy script, links in a new tab
# --------------------------------------------------------------------------

_HEAD_SCRIPT = r"""
(function () {
  "use strict";
  // The taxonomy styles follow html[data-theme], as in the documentation;
  // Gradio marks dark mode with a "dark" class on <body>.
  function syncTheme() {
    if (!document.body) return;
    document.documentElement.dataset.theme = document.body.classList.contains("dark") ? "dark" : "light";
  }
  // Tabs render lazily: mount the taxonomy when its tab first appears.
  function mountTaxonomy() {
    const app = document.getElementById("tsseg-taxonomy");
    if (!app || app.dataset.state !== "loading" || !document.getElementById("tsseg-taxonomy-data")) return;
    const old = document.getElementById("tx-card");
    if (old) old.remove();
    try {
      window.__tssegTaxonomy();
    } catch (err) {
      app.dataset.state = "error";
      console.error("tsseg taxonomy:", err);
    }
  }
  function start() {
    syncTheme();
    new MutationObserver(syncTheme).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    new MutationObserver(mountTaxonomy).observe(document.body, { childList: true, subtree: true });
    mountTaxonomy();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
  // Links of the taxonomy point to the documentation: open them in a new tab,
  // not in the frame of the Space.
  document.addEventListener(
    "click",
    (e) => {
      const a = e.target.closest && e.target.closest("#tsseg-taxonomy a[href], .tx-card a[href]");
      if (a && a.host && a.host !== location.host) {
        a.target = "_blank";
        a.rel = "noopener";
      }
    },
    true,
  );
  // The taxonomy focuses its search box on "/": only while its tab is shown.
  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "/") return;
      const t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      const app = document.getElementById("tsseg-taxonomy");
      if (!app || !app.offsetParent) e.stopPropagation();
    },
    true,
  );
})();
"""


def build_head() -> str:
    """``<head>`` additions: the taxonomy script and its loader."""
    taxonomy_js = (ASSETS / "taxonomy" / "taxonomy.js").read_text(encoding="utf-8")
    return (
        f"<script>window.__tssegTaxonomy = function () {{\n{taxonomy_js}\n}};</script>\n"
        f"<script>{_HEAD_SCRIPT}</script>\n"
    )


def taxonomy_css() -> str:
    """Stylesheet of the vendored taxonomy (``assets/taxonomy/taxonomy.css``)."""
    return (ASSETS / "taxonomy" / "taxonomy.css").read_text(encoding="utf-8")
