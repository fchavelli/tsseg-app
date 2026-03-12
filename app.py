"""tsseg-app: Interactive Time Series Segmentation — Gradio Application.

Launch with:
    python app.py
    # or
    gradio app.py
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd

from src.algorithms import (
    DetectorInfo,
    get_all_detectors,
    get_constructor_signature,
    get_detectors_by_type,
    instantiate_detector,
)
from src.data import (
    TimeSeriesData,
    generate_synthetic_series,
    get_builtin_datasets,
    load_csv,
    load_mocap_dataset,
)
from src.metrics import compute_metrics, get_metrics_for_task
from src.params import build_data_context, get_param_info, validate_params_for_detector
from src.plotting import (
    build_cpd_color_map,
    build_segment_color_map,
    make_annotation_figure,
    make_comparison_figure,
    make_metrics_bar_chart,
    make_multi_comparison_figure,
    make_signal_figure,
)
from src.utils import (
    ExecutionTimeoutError,
    align_states_via_hungarian,
    change_points_to_states,
    interpret_predictions,
    normalize_zscore,
    parse_user_value,
    run_with_timeout,
    states_to_change_points,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU for torch-based detectors
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ============================================================================
# Constants
# ============================================================================

TASK_LABELS = {
    "Change Point Detection": "change_point_detection",
    "State Detection": "state_detection",
}
TASK_LABELS_INV = {v: k for k, v in TASK_LABELS.items()}

MAX_EXECUTION_TIME = 300  # seconds

# ============================================================================
# Helpers
# ============================================================================


def _format_params(params: dict) -> str:
    """Format a params dict for display in the comparison table.

    Instead of raw JSON like ``{"n_cps": "6", "model": "rbf"}``,
    produce a readable string like ``n_cps=6, model=rbf``.
    """
    if not params:
        return "—"
    parts = []
    for k, v in params.items():
        # Strip quotes from string values
        v_str = str(v).strip("'\"")
        parts.append(f"{k}={v_str}")
    text = ", ".join(parts)
    # Truncate if too long
    if len(text) > 100:
        text = text[:97] + "…"
    return text


def _get_detector_choices(task_label: str) -> list[str]:
    """Get sorted list of detector names for a given task label.

    For Change Point Detection, also include State Detection algorithms
    (they infer change points from detected states).
    """
    task = TASK_LABELS.get(task_label, "change_point_detection")
    detectors = get_detectors_by_type(task)
    # CPD task: also include state detectors (they derive CPs from states)
    if task == "change_point_detection":
        state_detectors = get_detectors_by_type("state_detection")
        detectors.update(state_detectors)
    return sorted(detectors.keys())


def _get_all_detector_choices() -> dict[str, DetectorInfo]:
    return get_all_detectors()


def _run_detector(
    detector_info: DetectorInfo,
    params: Dict[str, Any],
    signal: np.ndarray,
) -> Any:
    """Run a detector on a signal and return raw output (with timeout)."""
    instance = instantiate_detector(detector_info.cls, params)
    signal_norm = normalize_zscore(signal)

    def _execute():
        try:
            if hasattr(instance, "fit_predict"):
                return instance.fit_predict(signal_norm, axis=0)
            else:
                instance.fit(signal_norm, axis=0)
                if hasattr(instance, "predict"):
                    return instance.predict(signal_norm, axis=0)
                elif hasattr(instance, "transform"):
                    return instance.transform(signal_norm)
                else:
                    raise RuntimeError("Detector does not expose predict/transform")
        except Exception as exc:
            raise RuntimeError(f"Execution failed: {exc}") from exc

    return run_with_timeout(_execute, timeout=MAX_EXECUTION_TIME)


# ============================================================================
# Theme & Style
# ============================================================================

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1600px !important; }
.main-header { text-align: center; margin-bottom: 0.3em; }
.main-header h1 { font-weight: 600; font-size: 1.6em; letter-spacing: -0.02em; color: #1a1a2e; }
.main-header p { color: #555; font-weight: 300; font-size: 0.95em; }
.main-header a { color: #3b5998; text-decoration: none; font-weight: 400; }
.main-header a:hover { text-decoration: underline; }
.run-btn { min-height: 40px !important; font-weight: 500 !important; }
.param-section { max-height: 55vh; overflow-y: auto; padding-right: 4px; }
footer { display: none !important; }
"""

APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="#2d3748",
    button_primary_background_fill_hover="#4a5568",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#edf2f7",
    button_secondary_background_fill_hover="#e2e8f0",
    button_secondary_text_color="#2d3748",
    block_title_text_weight="500",
    block_label_text_weight="400",
)


# ============================================================================
# Main Gradio App
# ============================================================================

def create_app() -> gr.Blocks:
    all_detectors = _get_all_detector_choices()

    with gr.Blocks(
        title="tsseg — Time Series Segmentation",
    ) as app:
        # -- Shared state --
        data_state = gr.State(value=None)          # TimeSeriesData
        data_ctx_state = gr.State(value=None)       # dict {n_samples, n_channels}
        runs_state = gr.State(value=[])             # list of run dicts
        last_run_state = gr.State(value=None)       # last run result dict (for add-to-comparison)

        # -- Header --
        gr.Markdown(
            """
            # tsseg — Time Series Segmentation
            Segment time series with 30+ algorithms. Upload data or use built-in datasets, compare results, evaluate with standard metrics.

            <p style="text-align:center; margin-top:2px">
                <a href="https://fchavelli.github.io/tsseg/" target="_blank">Documentation</a> &nbsp;·&nbsp;
                <a href="https://github.com/fchavelli/tsseg" target="_blank">GitHub</a>
            </p>
            """,
            elem_classes="main-header",
        )

        with gr.Tabs():
            # ==================================================================
            # TAB 1 — DATA
            # ==================================================================
            with gr.Tab("Data", id="tab_data"):
                gr.Markdown("#### Load or generate a time series")

                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        data_source = gr.Radio(
                            choices=["Synthetic", "MoCap (built-in)", "Upload CSV"],
                            value="Synthetic",
                            label="Data source",
                        )

                        # -- Synthetic controls --
                        with gr.Group(visible=True) as synth_controls:
                            synth_length = gr.Slider(200, 10000, value=1000, step=50, label="Length")
                            synth_segments = gr.Slider(2, 12, value=3, step=1, label="Segments")
                            synth_dims = gr.Slider(1, 8, value=2, step=1, label="Dimensions")
                            synth_pattern = gr.Dropdown(
                                choices=["constant", "sinusoidal", "step", "stairs", "linear", "mixed"],
                                value="sinusoidal",
                                label="Pattern",
                                info="Intra-segment waveform shape",
                            )
                            synth_recurring = gr.Checkbox(
                                value=False,
                                label="Recurring states",
                                info="Allow state labels to repeat across segments",
                            )
                            synth_min_seg = gr.Slider(5, 200, value=20, step=5,
                                                      label="Min segment length",
                                                      info="Minimum number of timestamps per segment")
                            synth_noise = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="Noise level")
                            synth_seed = gr.Number(value=42, label="Random seed", precision=0)

                        # -- MoCap controls --
                        with gr.Group(visible=False) as mocap_controls:
                            mocap_trial = gr.Slider(0, 8, value=0, step=1, label="Trial index")

                        # -- CSV Upload controls --
                        with gr.Group(visible=False) as csv_controls:
                            csv_file = gr.File(label="Upload CSV", file_types=[".csv", ".tsv", ".txt"])
                            csv_separator = gr.Dropdown(
                                choices=[",", ";", "\\t", " "],
                                value=",",
                                label="Separator",
                            )
                            csv_header = gr.Checkbox(value=True, label="Has header row")
                            csv_label_col = gr.Textbox(
                                value="",
                                label="Ground truth column (optional)",
                                placeholder="e.g., label, state, ground_truth",
                            )

                        load_btn = gr.Button("Load Data", variant="primary", elem_classes="run-btn")
                        load_example_btn = gr.Button("Load Example (MoCap)", variant="secondary")

                    with gr.Column(scale=3):
                        data_info = gr.Markdown("*No data loaded yet.*")
                        data_plot = gr.Plot(label="Data preview")

                # -- Data source visibility toggle --
                def toggle_source(source):
                    return (
                        gr.update(visible=source == "Synthetic"),
                        gr.update(visible=source == "MoCap (built-in)"),
                        gr.update(visible=source == "Upload CSV"),
                    )

                data_source.change(
                    toggle_source,
                    inputs=[data_source],
                    outputs=[synth_controls, mocap_controls, csv_controls],
                )

                # -- Load data handler --
                def load_data(source, length, segments, dims, pattern, recurring, min_seg, noise, seed,
                              trial, file, separator, header, label_col):
                    try:
                        if source == "Synthetic":
                            ts_data = generate_synthetic_series(
                                length=int(length),
                                n_segments=int(segments),
                                n_dims=int(dims),
                                pattern=pattern,
                                noise_level=noise,
                                recurring_states=recurring,
                                min_segment_length=int(min_seg),
                                random_state=int(seed) if seed is not None else None,
                            )
                        elif source == "MoCap (built-in)":
                            ts_data = load_mocap_dataset(trial=int(trial))
                        elif source == "Upload CSV":
                            if file is None:
                                return None, None, "*Please upload a CSV file.*", None
                            sep = "\t" if separator == "\\t" else separator
                            ts_data = load_csv(
                                file_path=file.name if hasattr(file, 'name') else file,
                                separator=sep,
                                has_header=header,
                                label_column=label_col if label_col.strip() else None,
                            )
                        else:
                            return None, None, "*Unknown data source.*", None

                        # Build info markdown
                        if ts_data.has_ground_truth:
                            gt_info = f"Ground truth available ({ts_data.n_states} states)"
                        else:
                            gt_info = "No ground truth"
                        info_md = (
                            f"**{ts_data.name}** — "
                            f"{ts_data.length} timestamps, {ts_data.n_dims} dim. "
                            f"| {gt_info}"
                        )

                        # Build plot
                        if ts_data.has_ground_truth:
                            n_states = ts_data.n_states
                            color_map = build_segment_color_map(n_states, ts_data.n_dims)
                            fig = make_annotation_figure(
                                ts_data.signal, ts_data.change_points or [],
                                ts_data.ground_truth, ts_data.name, color_map,
                            )
                        else:
                            fig = make_signal_figure(ts_data.signal, ts_data.name)

                        # Build data context for parameter constraints
                        ctx = build_data_context(ts_data.signal, n_channels=ts_data.n_dims)

                        return ts_data, ctx, info_md, fig

                    except Exception as e:
                        return None, None, f"**Error:** {e}", None

                load_btn.click(
                    load_data,
                    inputs=[
                        data_source, synth_length, synth_segments, synth_dims,
                        synth_pattern, synth_recurring, synth_min_seg, synth_noise, synth_seed,
                        mocap_trial, csv_file, csv_separator, csv_header, csv_label_col,
                    ],
                    outputs=[data_state, data_ctx_state, data_info, data_plot],
                )

                # -- Load Example handler (MoCap trial 0) --
                def load_example():
                    try:
                        ts_data = load_mocap_dataset(trial=0)
                        ctx = build_data_context(ts_data.signal, n_channels=ts_data.n_dims)
                        n_states = ts_data.n_states
                        color_map = build_segment_color_map(n_states, ts_data.n_dims)
                        fig = make_annotation_figure(
                            ts_data.signal, ts_data.change_points or [],
                            ts_data.ground_truth, ts_data.name, color_map,
                        )
                        info_md = (
                            f"**{ts_data.name}** — "
                            f"{ts_data.length} timestamps, {ts_data.n_dims} dim. "
                            f"| Ground truth available ({ts_data.n_states} states)"
                        )
                        return ts_data, ctx, info_md, fig
                    except Exception as e:
                        return None, None, f"**Error:** {e}", None

                load_example_btn.click(
                    load_example,
                    outputs=[data_state, data_ctx_state, data_info, data_plot],
                )

            # ==================================================================
            # TAB 2 — SEGMENTATION (landscape: controls left, results right)
            # ==================================================================
            with gr.Tab("Segmentation", id="tab_seg"):
                with gr.Row(equal_height=False):
                    # -- LEFT: algorithm selection + parameters --
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("#### Algorithm")
                        task_selector = gr.Radio(
                            choices=list(TASK_LABELS.keys()),
                            value="Change Point Detection",
                            label="Task",
                        )
                        detector_dropdown = gr.Dropdown(
                            choices=_get_detector_choices("Change Point Detection"),
                            label="Algorithm",
                            interactive=True,
                        )

                        # Update detector list when task changes
                        def update_detectors(task_label):
                            choices = _get_detector_choices(task_label)
                            return gr.update(choices=choices, value=choices[0] if choices else None)

                        task_selector.change(update_detectors, inputs=[task_selector], outputs=[detector_dropdown])

                        # Dynamic parameters zone
                        @gr.render(inputs=[detector_dropdown, data_ctx_state])
                        def render_params(detector_name, data_ctx):
                            if not detector_name or detector_name not in all_detectors:
                                gr.Markdown("*Select an algorithm to configure.*")
                                return

                            info = all_detectors[detector_name]

                            task_type = "State Detection" if info.detector_type == "state_detection" else "CPD"
                            gr.Markdown(f"**{detector_name}** · {task_type} · {info.category}")

                            param_infos = get_param_info(info.cls, data_ctx=data_ctx)
                            if not param_infos:
                                gr.Markdown("*No configurable parameters.*")
                            else:
                                gr.Markdown("#### Parameters")

                            # Show warnings for auto-adjusted defaults
                            adjustments = [
                                p["adjust_reason"]
                                for p in param_infos
                                if p.get("adjusted")
                            ]
                            if adjustments:
                                adj_md = "\n".join(f"- {a}" for a in adjustments)
                                gr.Markdown(
                                    f"⚠️ **Parameters auto-adjusted:**\n{adj_md}",
                                )

                            components = {}
                            current_group = ""
                            for p in param_infos:
                                # Group headers (from param_schema groups)
                                grp = p.get("group", "")
                                if grp and grp != current_group:
                                    current_group = grp
                                    gr.Markdown(f"**{grp.replace('_', ' ').title()}**")

                                key = f"p_{detector_name}_{p['name']}"
                                desc = p.get("description", "") or None
                                bounds = ""
                                if p.get("min") is not None or p.get("max") is not None:
                                    lo = p.get("min", "")
                                    hi = p.get("max", "")
                                    bounds = f" [{lo}–{hi}]"
                                info_text = (desc + bounds) if desc else (bounds.strip() or None)

                                if p["type"] == "bool":
                                    components[p["name"]] = gr.Checkbox(
                                        value=p["default"], label=p["name"], key=key,
                                        info=desc,
                                        interactive=True,
                                    )
                                elif p["type"] == "int":
                                    is_n_cps = p["name"] in ("n_cps", "n_bkps", "n_components")
                                    if is_n_cps:
                                        with gr.Row():
                                            num_comp = gr.Number(
                                                value=p["default"],
                                                label=p["name"],
                                                info=info_text,
                                                key=key,
                                                precision=0,
                                                interactive=True,
                                                scale=3,
                                            )
                                            gt_btn = gr.Button(
                                                "Use GT", variant="secondary",
                                                scale=1, min_width=70,
                                                key=f"gt_{detector_name}_{p['name']}",
                                            )

                                        def _set_gt_value(ts_data, _pname=p["name"]):
                                            if ts_data is None or not ts_data.has_ground_truth:
                                                return gr.update()
                                            n_gt = len(ts_data.change_points) if ts_data.change_points else 0
                                            return gr.update(value=n_gt)

                                        gt_btn.click(
                                            _set_gt_value,
                                            inputs=[data_state],
                                            outputs=[num_comp],
                                        )
                                        components[p["name"]] = num_comp
                                    else:
                                        components[p["name"]] = gr.Number(
                                            value=p["default"],
                                            label=p["name"],
                                            info=info_text,
                                            key=key,
                                            precision=0,
                                            interactive=True,
                                        )
                                elif p["type"] == "float":
                                    components[p["name"]] = gr.Number(
                                        value=p["default"],
                                        label=p["name"],
                                        info=info_text,
                                        key=key,
                                        interactive=True,
                                    )
                                elif p["type"] == "choice":
                                    components[p["name"]] = gr.Dropdown(
                                        choices=p["choices"],
                                        value=p["default"],
                                        label=p["name"],
                                        info=desc,
                                        key=key,
                                        interactive=True,
                                    )
                                elif p["type"] == "str":
                                    components[p["name"]] = gr.Textbox(
                                        value=str(p["default"]) if p["default"] is not None else "",
                                        label=p["name"],
                                        info=desc,
                                        key=key,
                                        interactive=True,
                                    )
                                else:  # text / complex types
                                    default_str = str(p["default"]) if p["default"] is not None else "None"
                                    components[p["name"]] = gr.Textbox(
                                        value=default_str,
                                        label=p["name"],
                                        info=desc or "Python literal syntax (e.g., [1,2,3] or None)",
                                        key=key,
                                        interactive=True,
                                    )

                            with gr.Row():
                                run_btn = gr.Button("Run", variant="primary", elem_classes="run-btn")
                                add_btn = gr.Button("Add to comparison", variant="secondary")

                            seg_feedback = gr.Markdown("", key=f"fb_{detector_name}")

                            def run_segmentation(*args):
                                ts_data = args[0]
                                cur_data_ctx = args[1]
                                ui_task_label = args[2]
                                param_vals = args[3:]

                                if ts_data is None:
                                    return "Load data first (Data tab).", None, None

                                # Resolve the UI task (may differ from detector_type)
                                ui_task = TASK_LABELS.get(ui_task_label, "change_point_detection")

                                params = {}
                                for i, p in enumerate(param_infos):
                                    val = param_vals[i] if i < len(param_vals) else p["default"]
                                    if p["type"] == "text" and isinstance(val, str):
                                        val = parse_user_value(val, p["default"])
                                    if p["type"] == "int" and val is not None:
                                        try:
                                            val = int(val)
                                        except (ValueError, TypeError):
                                            pass
                                    if p["type"] == "float" and val is not None:
                                        try:
                                            val = float(val)
                                        except (ValueError, TypeError):
                                            pass
                                    params[p["name"]] = val

                                # --- Validate & auto-adjust params via schema ---
                                params, v_warnings, v_errors = validate_params_for_detector(
                                    info.cls, params, data_ctx=cur_data_ctx,
                                )
                                if v_errors:
                                    err_md = "\n".join(f"- {e}" for e in v_errors)
                                    warn_md = ""
                                    if v_warnings:
                                        warn_md = "\n\n⚠️ Adjustments:\n" + "\n".join(f"- {w}" for w in v_warnings)
                                    return f"**Validation errors:**\n{err_md}{warn_md}", None, None

                                expect_states = (info.detector_type == "state_detection")

                                try:
                                    t0 = time.time()
                                    output = _run_detector(info, params, ts_data.signal)
                                    elapsed = time.time() - t0

                                    pred_states, pred_cps = interpret_predictions(
                                        output, ts_data.length, expect_states
                                    )

                                    if expect_states and pred_states is not None and ts_data.has_ground_truth:
                                        pred_states = align_states_via_hungarian(
                                            ts_data.ground_truth, pred_states
                                        )
                                        pred_cps = states_to_change_points(pred_states)

                                    n_states_max = max(
                                        ts_data.n_states if ts_data.has_ground_truth else 1,
                                        int(np.max(pred_states)) + 1 if pred_states is not None else 1,
                                    )
                                    color_map = build_segment_color_map(n_states_max, ts_data.n_dims)

                                    fig = make_comparison_figure(
                                        ts_data.signal,
                                        truth_states=ts_data.ground_truth,
                                        truth_cps=ts_data.change_points,
                                        pred_states=pred_states,
                                        pred_cps=pred_cps,
                                        color_map=color_map,
                                        task=ui_task,
                                    )

                                    n_cps = len(pred_cps) if pred_cps else 0
                                    n_st = int(np.max(pred_states) + 1) if pred_states is not None else "—"
                                    result_md = (
                                        f"**{detector_name}** — {elapsed:.2f}s — "
                                        f"{n_cps} change points, {n_st} states"
                                    )
                                    if v_warnings:
                                        result_md += "\n\n⚠️ " + " · ".join(v_warnings)

                                    # Store result for add-to-comparison
                                    last_run = {
                                        "name": detector_name,
                                        "params": {k: str(v) for k, v in params.items()},
                                        "pred_states": pred_states.tolist() if pred_states is not None else None,
                                        "pred_cps": pred_cps if pred_cps else [],
                                        "task": ui_task,
                                    }
                                    return result_md, fig, last_run

                                except Exception as e:
                                    tb = traceback.format_exc()
                                    logger.error(f"Detector failed: {tb}")
                                    return f"**Error:** {e}", None, None

                            run_btn.click(
                                run_segmentation,
                                inputs=[data_state, data_ctx_state, task_selector] + list(components.values()),
                                outputs=[seg_feedback, seg_result_plot, last_run_state],
                            ).then(
                                lambda fig: gr.update(visible=fig is None),
                                inputs=[seg_result_plot],
                                outputs=[seg_data_preview],
                            )

                            def add_to_comparison(last_run, current_runs):
                                """Add the last run result to the comparison list.

                                Reuses the stored result from the last 'Run' click
                                instead of re-running the detector (which would give
                                different results for non-deterministic algorithms).
                                """
                                if last_run is None:
                                    return current_runs or [], "Run the algorithm first before adding to comparison."

                                if current_runs is None:
                                    current_runs = []
                                updated = current_runs + [last_run]
                                name = last_run.get("name", "?")
                                return updated, f"**{name}** added ({len(updated)} runs total)"

                            add_btn.click(
                                add_to_comparison,
                                inputs=[last_run_state, runs_state],
                                outputs=[runs_state, seg_feedback],
                            )

                    # -- RIGHT: results plot (always visible, updates on Run / data load) --
                    with gr.Column(scale=2):
                        seg_result_info = gr.Markdown("*Run an algorithm to see results here.*")
                        seg_data_preview = gr.Plot(label="Data preview", visible=True)
                        seg_result_plot = gr.Plot(label="Segmentation result")

                # Auto-display data in Segmentation tab when data changes
                # Also reset the segmentation result plot so preview is visible again
                def update_seg_on_data_change(ts_data):
                    if ts_data is None:
                        return gr.update(visible=False), gr.update(value=None)
                    if ts_data.has_ground_truth:
                        n_states = ts_data.n_states
                        color_map = build_segment_color_map(n_states, ts_data.n_dims)
                        fig = make_annotation_figure(
                            ts_data.signal, ts_data.change_points or [],
                            ts_data.ground_truth, ts_data.name, color_map,
                        )
                    else:
                        fig = make_signal_figure(ts_data.signal, ts_data.name)
                    return gr.update(value=fig, visible=True), gr.update(value=None)

                data_state.change(
                    update_seg_on_data_change,
                    inputs=[data_state],
                    outputs=[seg_data_preview, seg_result_plot],
                )

            # ==================================================================
            # TAB 3 — COMPARISON
            # ==================================================================
            with gr.Tab("Comparison", id="tab_compare"):
                gr.Markdown("#### Compare algorithm runs")

                with gr.Row():
                    refresh_compare_btn = gr.Button("Refresh", variant="primary")
                    clear_runs_btn = gr.Button("Clear all runs", variant="stop")

                compare_info = gr.Markdown("*Add runs from the Segmentation tab.*")
                compare_plot = gr.Plot(label="Multi-algorithm comparison")
                compare_table = gr.Dataframe(label="Run summary", interactive=False)

                def refresh_comparison(ts_data, runs):
                    if not runs:
                        return "No runs to compare. Add runs from the Segmentation tab.", None, None
                    if ts_data is None:
                        return "No data loaded.", None, None

                    plot_runs = []
                    table_data = []
                    for i, run in enumerate(runs):
                        ps = np.array(run["pred_states"]) if run.get("pred_states") else None
                        pc = run.get("pred_cps", [])
                        plot_runs.append({
                            "name": f"{i+1}. {run['name']}",
                            "pred_states": ps,
                            "pred_cps": pc,
                            "task": run.get("task", "change_point_detection"),
                        })
                        table_data.append({
                            "Run": i + 1,
                            "Algorithm": run["name"],
                            "Task": TASK_LABELS_INV.get(run.get("task", ""), run.get("task", "")),
                            "# CPs": len(pc),
                            "# States": int(np.max(ps) + 1) if ps is not None else "—",
                            "Params": _format_params(run.get("params", {})),
                        })

                    fig = make_multi_comparison_figure(
                        ts_data.signal,
                        plot_runs,
                        truth_states=ts_data.ground_truth,
                        truth_cps=ts_data.change_points,
                    )

                    info_md = f"**{len(runs)} runs** loaded."
                    df = pd.DataFrame(table_data)
                    return info_md, fig, df

                refresh_compare_btn.click(
                    refresh_comparison,
                    inputs=[data_state, runs_state],
                    outputs=[compare_info, compare_plot, compare_table],
                )

                def clear_runs():
                    return [], "All runs cleared.", None, None

                clear_runs_btn.click(
                    clear_runs,
                    outputs=[runs_state, compare_info, compare_plot, compare_table],
                )

            # ==================================================================
            # TAB 4 — EVALUATION
            # ==================================================================
            with gr.Tab("Evaluation", id="tab_eval"):
                gr.Markdown("#### Evaluate results with standard metrics")

                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        eval_task = gr.Radio(
                            choices=list(TASK_LABELS.keys()),
                            value="Change Point Detection",
                            label="Evaluation mode",
                        )

                        cpd_metric_names = list(get_metrics_for_task("change_point_detection").keys())
                        state_metric_names = list(get_metrics_for_task("state_detection").keys())

                        cpd_metrics_cb = gr.CheckboxGroup(
                            choices=cpd_metric_names,
                            value=cpd_metric_names,
                            label="CPD Metrics",
                            visible=True,
                        )
                        state_metrics_cb = gr.CheckboxGroup(
                            choices=state_metric_names,
                            value=[m for m in state_metric_names if m != "Adjusted Mutual Information"],
                            label="State Metrics",
                            visible=False,
                        )

                        f1_margin = gr.Slider(1, 50, value=5, step=1, label="F1 Score margin", visible=True)

                        evaluate_btn = gr.Button("Evaluate All Runs", variant="primary", elem_classes="run-btn")

                        def toggle_eval_metrics(task_label):
                            is_cpd = task_label == "Change Point Detection"
                            return (
                                gr.update(visible=is_cpd),
                                gr.update(visible=not is_cpd),
                                gr.update(visible=is_cpd),
                            )

                        eval_task.change(
                            toggle_eval_metrics,
                            inputs=[eval_task],
                            outputs=[cpd_metrics_cb, state_metrics_cb, f1_margin],
                        )

                    with gr.Column(scale=3):
                        eval_results_md = gr.Markdown("")
                        eval_table = gr.Dataframe(label="Evaluation results", interactive=False)
                        eval_chart = gr.Plot(label="Scores comparison")
                        with gr.Row():
                            export_csv_btn = gr.Button("Export CSV", variant="secondary")
                            export_file = gr.File(label="Download", visible=False)

                def evaluate_runs(ts_data, runs, task_label, cpd_metrics, state_metrics, margin):
                    if ts_data is None:
                        return "Load data first.", None, None
                    if not ts_data.has_ground_truth:
                        return "No ground truth available. Upload data with labels or use a built-in dataset.", None, None
                    if not runs:
                        return "No runs to evaluate. Add runs from the Segmentation tab.", None, None

                    task = TASK_LABELS[task_label]
                    selected = cpd_metrics if task == "change_point_detection" else state_metrics

                    metric_params = {}
                    if "F1 Score" in selected:
                        metric_params["F1 Score"] = {"margin": int(margin)}

                    all_results = []
                    skipped = []
                    for i, run in enumerate(runs):
                        run_task = run.get("task", "change_point_detection")
                        pred_cps = run.get("pred_cps", [])
                        pred_states = np.array(run["pred_states"]) if run.get("pred_states") else None

                        # Block incompatible task/metric combinations
                        if task == "state_detection" and run_task == "change_point_detection":
                            # CPD-only algo cannot be evaluated with SD metrics
                            skipped.append(
                                f"{i+1}. {run['name']} (CPD-only — incompatible with State Detection metrics)"
                            )
                            continue

                        if task == "change_point_detection":
                            y_pred = change_points_to_states(pred_cps, ts_data.length)
                        else:
                            # State Detection evaluation
                            y_pred = pred_states if pred_states is not None else change_points_to_states(pred_cps, ts_data.length)

                        if y_pred is None:
                            skipped.append(f"{i+1}. {run['name']} (no valid predictions)")
                            continue

                        scores = compute_metrics(
                            ts_data.ground_truth, y_pred, task,
                            selected_metrics=selected,
                            metric_params=metric_params,
                        )

                        # Filter out Bidirectional Covering sub-scores
                        scores = {
                            k: v for k, v in scores.items()
                            if not (
                                k.startswith("Bidirectional Covering (")
                                and k != "Bidirectional Covering"
                            )
                        }

                        scores["Algorithm"] = f"{i+1}. {run['name']}"
                        all_results.append(scores)

                    if not all_results:
                        return "No valid results.", None, None

                    df = pd.DataFrame(all_results)
                    cols = ["Algorithm"] + [c for c in df.columns if c != "Algorithm"]
                    df = df[cols]

                    chart = make_metrics_bar_chart(df)
                    info = f"Evaluated **{len(all_results)}** runs with **{len(selected)}** metrics."
                    if skipped:
                        info += f"\n\nSkipped (no valid predictions): {', '.join(skipped)}"

                    return info, df, chart

                evaluate_btn.click(
                    evaluate_runs,
                    inputs=[data_state, runs_state, eval_task, cpd_metrics_cb, state_metrics_cb, f1_margin],
                    outputs=[eval_results_md, eval_table, eval_chart],
                )

                def export_results(ts_data, runs, task_label, cpd_metrics, state_metrics, margin):
                    _, df, _ = evaluate_runs(ts_data, runs, task_label, cpd_metrics, state_metrics, margin)
                    if df is None:
                        return gr.update(visible=False)
                    path = "/tmp/tsseg_evaluation_results.csv"
                    df.to_csv(path, index=False)
                    return gr.update(value=path, visible=True)

                export_csv_btn.click(
                    export_results,
                    inputs=[data_state, runs_state, eval_task, cpd_metrics_cb, state_metrics_cb, f1_margin],
                    outputs=[export_file],
                )

            # ==================================================================
            # TAB 5 — ABOUT
            # ==================================================================
            with gr.Tab("About", id="tab_about"):
                from src.about_data import (
                    build_algo_table,
                    build_dataset_table,
                    build_metric_table,
                    DISCLAIMER_MD,
                    LINKS_MD,
                    MOCAP_NOTE_MD,
                )

                algo_table = build_algo_table()
                metric_table = build_metric_table()
                dataset_table = build_dataset_table()

                gr.Markdown(f"""
#### About tsseg-app

**tsseg** is a Python library for **Time Series Segmentation**, covering both
**Change Point Detection (CPD)** and **State Detection (SD)**.
This app exposes 25+ algorithms through an interactive interface.
State detection algorithms are also available under the CPD task, since change points
can be inferred from detected states.

---

#### Algorithms

{algo_table}

{DISCLAIMER_MD}

---

#### Metrics

{metric_table}

---

#### Datasets

{dataset_table}

{MOCAP_NOTE_MD}

---

#### Links

{LINKS_MD}
                """)

    return app


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,
        css=APP_CSS,
        theme=APP_THEME,
    )
