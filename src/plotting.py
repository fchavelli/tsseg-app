"""Plotting utilities using Plotly for the Gradio app."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _adjust_color_hex(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_segment_color_map(n_segments: int, n_dims: int) -> list[list[str]]:
    colors = []
    for idx in range(n_segments):
        base = _BASE_COLORS[idx % len(_BASE_COLORS)]
        segment_colors = [_adjust_color_hex(base, 0.15 * dim) for dim in range(n_dims)]
        colors.append(segment_colors)
    return colors


def _iter_segments(states: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    start = 0
    current = states[0]
    for idx in range(1, len(states)):
        if states[idx] != current:
            yield start, idx, current
            start = idx
            current = states[idx]
    yield start, len(states), current


def make_annotation_figure(
    signal: np.ndarray,
    change_points: Sequence[int],
    states: Sequence[int],
    title: str,
    color_map: list[list[str]],
) -> go.Figure:
    """Build a Plotly figure highlighting segments and change points."""
    n_samples, n_dims = signal.shape
    fig = go.Figure()

    for start, stop, state in _iter_segments(states):
        for dim in range(n_dims):
            color = color_map[state % len(color_map)][dim % len(color_map[state % len(color_map)])]
            fig.add_trace(go.Scatter(
                x=np.arange(start, stop),
                y=signal[start:stop, dim],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"State {state} · Dim {dim}",
                showlegend=False,
            ))

    for cp in change_points:
        fig.add_vline(x=cp, line=dict(color="#333333", width=1, dash="dash"))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=300 + 30 * n_dims,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def make_signal_figure(signal: np.ndarray, title: str = "Signal") -> go.Figure:
    """Simple signal visualization without state coloring."""
    n_samples, n_dims = signal.shape
    fig = go.Figure()
    for dim in range(n_dims):
        fig.add_trace(go.Scatter(
            x=np.arange(n_samples),
            y=signal[:, dim],
            mode="lines",
            name=f"Dim {dim}",
            line=dict(color=_BASE_COLORS[dim % len(_BASE_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def make_comparison_figure(
    signal: np.ndarray,
    truth_states: Optional[Sequence[int]],
    truth_cps: Optional[Sequence[int]],
    pred_states: Optional[Sequence[int]],
    pred_cps: Optional[Sequence[int]],
    color_map: list[list[str]],
    *,
    fallback_states: Optional[Sequence[int]] = None,
) -> go.Figure:
    """Two-row figure contrasting ground-truth and predictions."""
    has_truth = truth_states is not None
    n_rows = 2 if has_truth else 1
    titles = (["Ground Truth", "Prediction"] if has_truth else ["Prediction"])

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=titles,
    )

    if has_truth:
        truth_fig = make_annotation_figure(signal, truth_cps or [], truth_states, "", color_map)
        for trace in truth_fig.data:
            fig.add_trace(trace, row=1, col=1)
        for shape in truth_fig.layout.shapes or []:
            fig.add_shape(shape, row=1, col=1)

    pred_row = 2 if has_truth else 1
    if pred_states is not None:
        pred_fig = make_annotation_figure(signal, pred_cps or [], pred_states, "", color_map)
        for trace in pred_fig.data:
            fig.add_trace(trace, row=pred_row, col=1)
        for shape in pred_fig.layout.shapes or []:
            fig.add_shape(shape, row=pred_row, col=1)
    elif fallback_states is not None:
        pred_fig = make_annotation_figure(signal, pred_cps or [], fallback_states, "", color_map)
        for trace in pred_fig.data:
            fig.add_trace(trace, row=pred_row, col=1)
        for shape in pred_fig.layout.shapes or []:
            fig.add_shape(shape, row=pred_row, col=1)
    else:
        n_dims = signal.shape[1]
        for dim in range(n_dims):
            fig.add_trace(go.Scatter(
                x=np.arange(signal.shape[0]),
                y=signal[:, dim],
                mode="lines",
                name=f"Dim {dim}",
                showlegend=False,
            ), row=pred_row, col=1)
        if pred_cps:
            for cp in pred_cps:
                fig.add_vline(x=cp, line=dict(color="#333333", width=1, dash="dash"), row=pred_row, col=1)

    height = 350 * n_rows
    fig.update_layout(height=height, template="plotly_white", margin=dict(t=50, b=40, l=50, r=20))
    fig.update_yaxes(title_text="Value", row=1, col=1)
    if n_rows > 1:
        fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=n_rows, col=1)
    return fig


def make_multi_comparison_figure(
    signal: np.ndarray,
    runs: list[dict],
    truth_states: Optional[Sequence[int]] = None,
    truth_cps: Optional[Sequence[int]] = None,
) -> go.Figure:
    """Multi-row figure comparing several algorithm runs.

    Parameters
    ----------
    signal : np.ndarray
        The input signal, shape (n_timestamps, n_dims).
    runs : list of dict
        Each dict has keys: 'name', 'pred_states', 'pred_cps'.
    truth_states, truth_cps : optional ground truth.
    """
    has_truth = truth_states is not None
    n_rows = len(runs) + (1 if has_truth else 0)
    if n_rows == 0:
        return go.Figure()

    n_states_max = 1
    if has_truth:
        n_states_max = max(n_states_max, int(np.max(truth_states)) + 1)
    for run in runs:
        if run.get("pred_states") is not None:
            n_states_max = max(n_states_max, int(np.max(run["pred_states"])) + 1)

    color_map = build_segment_color_map(n_states_max, signal.shape[1])

    titles = []
    if has_truth:
        titles.append("Ground Truth")
    for run in runs:
        titles.append(run["name"])

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=titles,
    )

    row = 1
    if has_truth:
        truth_fig = make_annotation_figure(signal, truth_cps or [], truth_states, "", color_map)
        for trace in truth_fig.data:
            fig.add_trace(trace, row=row, col=1)
        for shape in truth_fig.layout.shapes or []:
            fig.add_shape(shape, row=row, col=1)
        row += 1

    for run in runs:
        ps = run.get("pred_states")
        pc = run.get("pred_cps", [])
        if ps is not None:
            run_fig = make_annotation_figure(signal, pc, ps, "", color_map)
            for trace in run_fig.data:
                fig.add_trace(trace, row=row, col=1)
            for shape in run_fig.layout.shapes or []:
                fig.add_shape(shape, row=row, col=1)
        else:
            for dim in range(signal.shape[1]):
                fig.add_trace(go.Scatter(
                    x=np.arange(signal.shape[0]),
                    y=signal[:, dim],
                    mode="lines",
                    showlegend=False,
                ), row=row, col=1)
            for cp in (pc or []):
                fig.add_vline(x=cp, line=dict(color="#333333", width=1, dash="dash"), row=row, col=1)
        row += 1

    fig.update_layout(
        height=max(250 * n_rows, 400),
        template="plotly_white",
        margin=dict(t=50, b=40, l=50, r=20),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time", row=n_rows, col=1)
    return fig


def make_metrics_bar_chart(scores_df) -> go.Figure:
    """Bar chart comparing metric scores across algorithms.

    Colors are grouped semantically:
    - F1-related metrics → blue palette
    - Covering metrics → green palette
    - State detection metrics → purple/orange palette
    """
    import pandas as pd
    if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
        return go.Figure()

    # Semantic color mapping for metrics
    _METRIC_COLORS = {
        # F1-related → blues
        "F1 Score": "#1f77b4",
        "F1 Score (precision)": "#4a9ecf",
        "F1 Score (recall)": "#7ec4e8",
        # Covering → greens
        "Covering": "#2ca02c",
        "Bidirectional Covering": "#5cc85c",
        # State metrics → warm tones
        "Adjusted Rand Index": "#d62728",
        "Normalized Mutual Information": "#e87d2e",
        "Adjusted Mutual Information": "#f0a050",
        "Weighted ARI": "#9467bd",
        "Weighted NMI": "#b694d4",
        "State Matching Score": "#8c564b",
    }

    fig = go.Figure()
    metric_cols = [c for c in scores_df.columns if c != "Algorithm"]

    for i, metric in enumerate(metric_cols):
        color = _METRIC_COLORS.get(metric, _BASE_COLORS[i % len(_BASE_COLORS)])
        fig.add_trace(go.Bar(
            x=scores_df["Algorithm"],
            y=scores_df[metric],
            name=metric,
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        title="Metric Scores Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Score",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig
