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
from src.params import get_param_info
from src.plotting import (
    build_segment_color_map,
    make_annotation_figure,
    make_comparison_figure,
    make_metrics_bar_chart,
    make_multi_comparison_figure,
    make_signal_figure,
)
from src.utils import (
    align_states_via_hungarian,
    change_points_to_states,
    interpret_predictions,
    normalize_zscore,
    parse_user_value,
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


def _get_detector_choices(task_label: str) -> list[str]:
    """Get sorted list of detector names for a given task label."""
    task = TASK_LABELS.get(task_label, "change_point_detection")
    detectors = get_detectors_by_type(task)
    return sorted(detectors.keys())


def _get_all_detector_choices() -> dict[str, DetectorInfo]:
    return get_all_detectors()


def _run_detector(
    detector_info: DetectorInfo,
    params: Dict[str, Any],
    signal: np.ndarray,
) -> Any:
    """Run a detector on a signal and return raw output."""
    instance = instantiate_detector(detector_info.cls, params)
    signal_norm = normalize_zscore(signal)

    try:
        if hasattr(instance, "fit_predict"):
            output = instance.fit_predict(signal_norm, axis=0)
        else:
            instance.fit(signal_norm, axis=0)
            if hasattr(instance, "predict"):
                output = instance.predict(signal_norm, axis=0)
            elif hasattr(instance, "transform"):
                output = instance.transform(signal_norm)
            else:
                raise RuntimeError("Detector does not expose predict/transform")
    except Exception as exc:
        raise RuntimeError(f"Execution failed: {exc}") from exc

    return output


# ============================================================================
# App state management
# ============================================================================

def _empty_runs_state():
    return []


# ============================================================================
# Main Gradio App
# ============================================================================

def create_app() -> gr.Blocks:
    all_detectors = _get_all_detector_choices()

    # CSS for a polished look
    custom_css = """
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .info-box { padding: 10px; border-radius: 8px; background: #f0f4ff; margin: 5px 0; }
    .run-btn { min-height: 45px !important; }
    """

    with gr.Blocks(
        title="tsseg — Time Series Segmentation",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        css=custom_css,
    ) as app:
        # -- Shared state --
        data_state = gr.State(value=None)          # TimeSeriesData
        runs_state = gr.State(value=[])             # list of run dicts

        # -- Header --
        gr.Markdown(
            """
            # 📈 tsseg — Interactive Time Series Segmentation
            **Segment your time series** with 30+ algorithms. Upload your data or use built-in datasets.
            Compare algorithms, tune parameters, and evaluate with standard metrics.

            <p style="text-align:center">
                <a href="https://fchavelli.github.io/tsseg/" target="_blank">📖 Documentation</a> ·
                <a href="https://github.com/fchavelli/tsseg" target="_blank">💻 GitHub</a>
            </p>
            """,
            elem_classes="main-title",
        )

        with gr.Tabs():
            # ==================================================================
            # TAB 1 — DATA
            # ==================================================================
            with gr.Tab("📊 Data", id="tab_data"):
                gr.Markdown("### Load or generate a time series")

                with gr.Row():
                    with gr.Column(scale=1):
                        data_source = gr.Radio(
                            choices=["Synthetic", "MoCap (built-in)", "Upload CSV"],
                            value="Synthetic",
                            label="Data source",
                        )

                        # -- Synthetic controls --
                        with gr.Group(visible=True) as synth_controls:
                            synth_length = gr.Slider(200, 10000, value=1000, step=50, label="Length")
                            synth_segments = gr.Slider(2, 12, value=4, step=1, label="Number of segments")
                            synth_dims = gr.Slider(1, 8, value=2, step=1, label="Dimensions")
                            synth_complexity = gr.Dropdown(
                                choices=["simple", "medium", "complex"],
                                value="medium",
                                label="Complexity",
                            )
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
                                label="Ground truth column name (optional)",
                                placeholder="e.g., label, state, ground_truth",
                            )

                        load_btn = gr.Button("Load Data", variant="primary", elem_classes="run-btn")

                    with gr.Column(scale=2):
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
                def load_data(source, length, segments, dims, complexity, noise, seed,
                              trial, file, separator, header, label_col):
                    try:
                        if source == "Synthetic":
                            ts_data = generate_synthetic_series(
                                length=int(length),
                                n_segments=int(segments),
                                n_dims=int(dims),
                                complexity=complexity,
                                noise_level=noise,
                                random_state=int(seed) if seed is not None else None,
                            )
                        elif source == "MoCap (built-in)":
                            ts_data = load_mocap_dataset(trial=int(trial))
                        elif source == "Upload CSV":
                            if file is None:
                                return None, "*Please upload a CSV file.*", None
                            sep = "\t" if separator == "\\t" else separator
                            ts_data = load_csv(
                                file_path=file.name if hasattr(file, 'name') else file,
                                separator=sep,
                                has_header=header,
                                label_column=label_col if label_col.strip() else None,
                            )
                        else:
                            return None, "*Unknown data source.*", None

                        # Build info markdown
                        gt_info = f"✅ Ground truth available ({ts_data.n_states} states)" if ts_data.has_ground_truth else "❌ No ground truth"
                        info_md = f"""
**{ts_data.name}**
- **Shape:** {ts_data.length} timestamps × {ts_data.n_dims} dimensions
- **Ground truth:** {gt_info}
"""
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

                        return ts_data, info_md, fig

                    except Exception as e:
                        return None, f"**Error:** {e}", None

                load_btn.click(
                    load_data,
                    inputs=[
                        data_source, synth_length, synth_segments, synth_dims,
                        synth_complexity, synth_noise, synth_seed,
                        mocap_trial, csv_file, csv_separator, csv_header, csv_label_col,
                    ],
                    outputs=[data_state, data_info, data_plot],
                )

            # ==================================================================
            # TAB 2 — SEGMENTATION
            # ==================================================================
            with gr.Tab("🔬 Segmentation", id="tab_seg"):
                gr.Markdown("### Run a segmentation algorithm")

                with gr.Row():
                    with gr.Column(scale=1):
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
                        param_zone = gr.Markdown("*Select an algorithm to see its parameters.*")

                        @gr.render(inputs=[detector_dropdown])
                        def render_params(detector_name):
                            if not detector_name or detector_name not in all_detectors:
                                gr.Markdown("*Select an algorithm.*")
                                return

                            info = all_detectors[detector_name]
                            deps = info.dependencies
                            if deps:
                                gr.Markdown(f"⚠️ Requires: **{', '.join(deps)}** (running on CPU)")

                            param_infos = get_param_info(info.cls)
                            if not param_infos:
                                gr.Markdown("*No configurable parameters.*")
                                return

                            gr.Markdown(f"#### Parameters for {detector_name}")
                            components = {}
                            for p in param_infos:
                                key = f"p_{detector_name}_{p['name']}"
                                if p["type"] == "bool":
                                    components[p["name"]] = gr.Checkbox(
                                        value=p["default"], label=p["name"], key=key
                                    )
                                elif p["type"] == "int":
                                    components[p["name"]] = gr.Slider(
                                        minimum=p.get("min", 0),
                                        maximum=p.get("max", 100),
                                        value=p["default"],
                                        step=p.get("step", 1),
                                        label=p["name"],
                                        key=key,
                                    )
                                elif p["type"] == "float":
                                    components[p["name"]] = gr.Slider(
                                        minimum=p.get("min", 0.0),
                                        maximum=p.get("max", 10.0),
                                        value=p["default"],
                                        step=p.get("step", 0.01),
                                        label=p["name"],
                                        key=key,
                                    )
                                elif p["type"] == "choice":
                                    components[p["name"]] = gr.Dropdown(
                                        choices=p["choices"],
                                        value=p["default"],
                                        label=p["name"],
                                        key=key,
                                    )
                                elif p["type"] == "str":
                                    components[p["name"]] = gr.Textbox(
                                        value=str(p["default"]) if p["default"] is not None else "",
                                        label=p["name"],
                                        key=key,
                                    )
                                else:  # text / complex types
                                    default_str = str(p["default"]) if p["default"] is not None else "None"
                                    components[p["name"]] = gr.Textbox(
                                        value=default_str,
                                        label=p["name"],
                                        info="Use Python literal syntax (e.g., [1,2,3] or None)",
                                        key=key,
                                    )

                            run_btn = gr.Button("▶ Run Algorithm", variant="primary", elem_classes="run-btn")
                            add_btn = gr.Button("➕ Add to Comparison", variant="secondary")

                            seg_output_md = gr.Markdown("")
                            seg_output_plot = gr.Plot(label="Segmentation result")

                            def run_segmentation(*args):
                                # args = (data_state_value, param1, param2, ...)
                                ts_data = args[0]
                                param_vals = args[1:]

                                if ts_data is None:
                                    return "**⚠️ Load data first** (go to Data tab).", None

                                # Reconstruct params dict
                                params = {}
                                for i, p in enumerate(param_infos):
                                    val = param_vals[i] if i < len(param_vals) else p["default"]
                                    # Parse text inputs
                                    if p["type"] == "text" and isinstance(val, str):
                                        val = parse_user_value(val, p["default"])
                                    params[p["name"]] = val

                                task = TASK_LABELS.get(task_selector.value, info.detector_type)
                                expect_states = (info.detector_type == "state_detection")

                                try:
                                    t0 = time.time()
                                    output = _run_detector(info, params, ts_data.signal)
                                    elapsed = time.time() - t0

                                    pred_states, pred_cps = interpret_predictions(
                                        output, ts_data.length, expect_states
                                    )

                                    # Align states if ground truth available
                                    if expect_states and pred_states is not None and ts_data.has_ground_truth:
                                        pred_states = align_states_via_hungarian(
                                            ts_data.ground_truth, pred_states
                                        )
                                        pred_cps = states_to_change_points(pred_states)

                                    # Build figure
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
                                    )

                                    n_cps = len(pred_cps) if pred_cps else 0
                                    n_st = int(np.max(pred_states) + 1) if pred_states is not None else "N/A"
                                    result_md = (
                                        f"✅ **{detector_name}** completed in **{elapsed:.2f}s**\n\n"
                                        f"- Change points detected: **{n_cps}**\n"
                                        f"- States found: **{n_st}**\n"
                                        f"- CPs: `{pred_cps}`"
                                    )
                                    return result_md, fig

                                except Exception as e:
                                    tb = traceback.format_exc()
                                    logger.error(f"Detector failed: {tb}")
                                    return f"**❌ Error:** {e}", None

                            run_btn.click(
                                run_segmentation,
                                inputs=[data_state] + list(components.values()),
                                outputs=[seg_output_md, seg_output_plot],
                            )

                            def add_to_comparison(*args):
                                ts_data = args[0]
                                current_runs = args[1]
                                param_vals = args[2:]

                                if ts_data is None:
                                    return current_runs, "**⚠️ Load data first.**"

                                params = {}
                                for i, p in enumerate(param_infos):
                                    val = param_vals[i] if i < len(param_vals) else p["default"]
                                    if p["type"] == "text" and isinstance(val, str):
                                        val = parse_user_value(val, p["default"])
                                    params[p["name"]] = val

                                expect_states = (info.detector_type == "state_detection")

                                try:
                                    output = _run_detector(info, params, ts_data.signal)
                                    pred_states, pred_cps = interpret_predictions(
                                        output, ts_data.length, expect_states
                                    )
                                    if expect_states and pred_states is not None and ts_data.has_ground_truth:
                                        pred_states = align_states_via_hungarian(
                                            ts_data.ground_truth, pred_states
                                        )
                                        pred_cps = states_to_change_points(pred_states)

                                    run_entry = {
                                        "name": detector_name,
                                        "params": {k: str(v) for k, v in params.items()},
                                        "pred_states": pred_states.tolist() if pred_states is not None else None,
                                        "pred_cps": pred_cps if pred_cps else [],
                                        "task": info.detector_type,
                                    }

                                    if current_runs is None:
                                        current_runs = []
                                    updated = current_runs + [run_entry]
                                    return updated, f"✅ **{detector_name}** added ({len(updated)} runs total)"

                                except Exception as e:
                                    return current_runs, f"**❌ Error:** {e}"

                            add_btn.click(
                                add_to_comparison,
                                inputs=[data_state, runs_state] + list(components.values()),
                                outputs=[runs_state, seg_output_md],
                            )

                    with gr.Column(scale=2):
                        gr.Markdown("*Results will appear here after running an algorithm.*")

            # ==================================================================
            # TAB 3 — COMPARISON
            # ==================================================================
            with gr.Tab("📊 Comparison", id="tab_compare"):
                gr.Markdown("### Compare multiple algorithm runs")

                with gr.Row():
                    refresh_compare_btn = gr.Button("🔄 Refresh Comparison", variant="primary")
                    clear_runs_btn = gr.Button("🗑️ Clear All Runs", variant="stop")

                compare_info = gr.Markdown("*Add runs from the Segmentation tab using the 'Add to Comparison' button.*")
                compare_plot = gr.Plot(label="Multi-algorithm comparison")
                compare_table = gr.Dataframe(label="Run summary", interactive=False)

                def refresh_comparison(ts_data, runs):
                    if not runs:
                        return "**No runs to compare.** Use the Segmentation tab to add runs.", None, None
                    if ts_data is None:
                        return "**⚠️ No data loaded.**", None, None

                    # Build runs list for plotting
                    plot_runs = []
                    table_data = []
                    for i, run in enumerate(runs):
                        ps = np.array(run["pred_states"]) if run.get("pred_states") else None
                        pc = run.get("pred_cps", [])
                        plot_runs.append({
                            "name": f"{i+1}. {run['name']}",
                            "pred_states": ps,
                            "pred_cps": pc,
                        })
                        table_data.append({
                            "Run": i + 1,
                            "Algorithm": run["name"],
                            "Task": TASK_LABELS_INV.get(run.get("task", ""), run.get("task", "")),
                            "# CPs": len(pc),
                            "# States": int(np.max(ps) + 1) if ps is not None else "N/A",
                            "Params": json.dumps(run.get("params", {}), default=str)[:80],
                        })

                    fig = make_multi_comparison_figure(
                        ts_data.signal,
                        plot_runs,
                        truth_states=ts_data.ground_truth,
                        truth_cps=ts_data.change_points,
                    )

                    info_md = f"**{len(runs)} runs** loaded for comparison."
                    df = pd.DataFrame(table_data)
                    return info_md, fig, df

                refresh_compare_btn.click(
                    refresh_comparison,
                    inputs=[data_state, runs_state],
                    outputs=[compare_info, compare_plot, compare_table],
                )

                def clear_runs():
                    return [], "**All runs cleared.**", None, None

                clear_runs_btn.click(
                    clear_runs,
                    outputs=[runs_state, compare_info, compare_plot, compare_table],
                )

            # ==================================================================
            # TAB 4 — EVALUATION
            # ==================================================================
            with gr.Tab("📏 Evaluation", id="tab_eval"):
                gr.Markdown("### Evaluate segmentation results with standard metrics")

                eval_task_info = gr.Markdown("*Metrics will be computed for all runs stored in the Comparison tab.*")

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_task = gr.Radio(
                            choices=list(TASK_LABELS.keys()),
                            value="Change Point Detection",
                            label="Evaluation mode",
                        )

                        # CPD metrics
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
                            value=state_metric_names,
                            label="State Metrics",
                            visible=False,
                        )

                        # F1 margin parameter
                        f1_margin = gr.Slider(1, 50, value=5, step=1, label="F1 Score margin")

                        evaluate_btn = gr.Button("📏 Evaluate All Runs", variant="primary", elem_classes="run-btn")

                        def toggle_eval_metrics(task_label):
                            is_cpd = task_label == "Change Point Detection"
                            return gr.update(visible=is_cpd), gr.update(visible=not is_cpd)

                        eval_task.change(
                            toggle_eval_metrics,
                            inputs=[eval_task],
                            outputs=[cpd_metrics_cb, state_metrics_cb],
                        )

                    with gr.Column(scale=2):
                        eval_results_md = gr.Markdown("")
                        eval_table = gr.Dataframe(label="Evaluation results", interactive=False)
                        eval_chart = gr.Plot(label="Scores comparison")
                        export_csv_btn = gr.Button("⬇️ Export Results as CSV")
                        export_file = gr.File(label="Download", visible=False)

                def evaluate_runs(ts_data, runs, task_label, cpd_metrics, state_metrics, margin):
                    if ts_data is None:
                        return "**⚠️ Load data first.**", None, None
                    if not ts_data.has_ground_truth:
                        return "**❌ No ground truth available.** Upload data with labels or use a built-in dataset.", None, None
                    if not runs:
                        return "**No runs to evaluate.** Add runs from the Segmentation tab.", None, None

                    task = TASK_LABELS[task_label]
                    selected = cpd_metrics if task == "change_point_detection" else state_metrics

                    metric_params = {}
                    if "F1 Score" in selected:
                        metric_params["F1 Score"] = {"margin": int(margin)}

                    all_results = []
                    for i, run in enumerate(runs):
                        pred_cps = run.get("pred_cps", [])
                        pred_states = np.array(run["pred_states"]) if run.get("pred_states") else None

                        if task == "change_point_detection":
                            y_pred = change_points_to_states(pred_cps, ts_data.length)
                        else:
                            y_pred = pred_states if pred_states is not None else change_points_to_states(pred_cps, ts_data.length)

                        if y_pred is None:
                            continue

                        scores = compute_metrics(
                            ts_data.ground_truth, y_pred, task,
                            selected_metrics=selected,
                            metric_params=metric_params,
                        )
                        scores["Algorithm"] = f"{i+1}. {run['name']}"
                        all_results.append(scores)

                    if not all_results:
                        return "**No valid results.**", None, None

                    df = pd.DataFrame(all_results)
                    # Reorder columns
                    cols = ["Algorithm"] + [c for c in df.columns if c != "Algorithm"]
                    df = df[cols]

                    chart = make_metrics_bar_chart(df)
                    info = f"✅ Evaluated **{len(all_results)}** runs with **{len(selected)}** metrics."

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
            with gr.Tab("ℹ️ About", id="tab_about"):
                # List all available algorithms in a nice table
                detector_rows = []
                for name, info in sorted(all_detectors.items()):
                    task = "CPD" if info.detector_type == "change_point_detection" else "State"
                    deps = ", ".join(info.dependencies) if info.dependencies else "—"
                    avail = "✅" if info.is_available else "❌"
                    detector_rows.append(f"| {name} | {task} | {info.category} | {deps} | {avail} |")

                detectors_table = "\n".join(detector_rows)

                gr.Markdown(f"""
### About tsseg-app

**tsseg** is a Python library for Time Series Segmentation, covering both Change Point Detection
and State Detection. This app provides an interactive interface to all its algorithms.

#### Available Algorithms

| Algorithm | Task | Category | Dependencies | Available |
|-----------|------|----------|-------------|-----------|
{detectors_table}

#### Metrics

**Change Point Detection:** F1 Score, Covering, Bidirectional Covering, Hausdorff Distance

**State Detection:** Adjusted Rand Index, Normalized Mutual Information, Adjusted Mutual Information,
Weighted ARI, Weighted NMI, State Matching Score

#### Links

- [tsseg Documentation](https://fchavelli.github.io/tsseg/)
- [GitHub Repository](https://github.com/fchavelli/tsseg)
- [License: AGPLv3](https://github.com/fchavelli/tsseg/blob/main/LICENSE)

*Built with [Gradio](https://gradio.app) · Powered by [tsseg](https://github.com/fchavelli/tsseg)*
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
    )
