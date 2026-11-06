"""Plotly figures of the app, in the visual language of the tsseg documentation.

States are soft bands behind the series (the colours of the tsseg logo, as in
the README figures), with their letter; change points are thick ink strokes;
the channels are drawn in shades of ink so that the bands carry the colour.
A change point prediction has no states: its segments alternate two greys, and
the true change points are dotted.
"""

from __future__ import annotations

import colorsys
from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
INK = "#1f2328"
MUTED = "#5b6b80"
FAINT = "#8c959f"
LINE = "#e3e8ef"

# States: the three colours of the logo, then hues of the same lightness
STATE_COLORS = [
    "#a5d0f3",  # blue
    "#faae75",  # orange
    "#acdca8",  # green
    "#cdb8ee",  # purple
    "#f3d27a",  # yellow
    "#f4b3c8",  # pink
    "#9fdcd6",  # teal
    "#d9c2a5",  # sand
    "#c3cad4",  # grey
    "#cfe39a",  # lime
]
# Channels: ink to grey, the eye follows the darkest
CHANNEL_COLORS = ["#1f2328", "#424a53", "#59636e", "#8c959f", "#6e7781", "#afb8c1", "#2f3b4e", "#9aa4b1"]
# Change point predictions: two neutral greys, no state meaning
_CPD_BANDS = ["#eaeef2", "#f6f8fa"]
BAND_OPACITY = 0.45


def _shade(hex_color: str, light: float, sat: float | None = None) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, light, s if sat is None else sat)
    return "#" + "".join(f"{round(v * 255):02x}" for v in (r, g, b))


def state_letter(state: int) -> str:
    """A, B, …, Z, then S26, S27, …"""
    return chr(ord("A") + state) if 0 <= state < 26 else f"S{state}"


def build_segment_color_map(n_segments: int, n_dims: int) -> list[list[str]]:
    """Colour of each state (repeated per channel, for backward compatibility)."""
    return [[STATE_COLORS[i % len(STATE_COLORS)]] * max(1, n_dims) for i in range(n_segments)]


def build_cpd_color_map(n_segments: int, n_dims: int) -> list[list[str]]:
    """Alternating neutral greys for change point predictions (no state identity)."""
    return [[_CPD_BANDS[i % 2]] * max(1, n_dims) for i in range(n_segments)]


def _iter_segments(states: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    states = np.asarray(states)
    if states.size == 0:
        return
    edges = [0, *(np.flatnonzero(np.diff(states)) + 1).tolist(), len(states)]
    for a, b in zip(edges[:-1], edges[1:]):
        yield a, b, int(states[a])


def _refs(row: int | None) -> tuple[str, str]:
    suffix = "" if row in (None, 1) else str(row)
    return f"x{suffix}", f"y{suffix}"


def _band(fig: go.Figure, row: int | None, x0: float, x1: float, color: str, opacity: float = 1.0) -> None:
    xref, yref = _refs(row)
    fig.add_shape(
        type="rect",
        xref=xref,
        yref=f"{yref} domain",
        x0=x0,
        x1=x1,
        y0=0,
        y1=1,
        fillcolor=color,
        opacity=opacity,
        line_width=0,
        layer="below",
    )


def _vline(fig: go.Figure, row: int | None, x: float, color: str = INK, width: float = 2.4, dash: str = "solid"):
    xref, yref = _refs(row)
    fig.add_shape(
        type="line",
        xref=xref,
        yref=f"{yref} domain",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        line=dict(color=color, width=width, dash=dash),
        layer="above",
    )


def _series(fig: go.Figure, signal: np.ndarray, row: int | None, legend: bool = False) -> None:
    n, d = signal.shape
    x = np.arange(n)
    for dim in reversed(range(d)):
        trace = go.Scatter(
            x=x,
            y=signal[:, dim],
            mode="lines",
            name=f"Dim {dim}",
            line=dict(color=CHANNEL_COLORS[dim % len(CHANNEL_COLORS)], width=1.5),
            opacity=0.95 if dim == 0 else 0.8,
            showlegend=legend,
            hovertemplate=f"dim {dim}: %{{y:.3g}}<extra></extra>",
        )
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=1)


def _state_panel(fig, signal, states, change_points, color_map, row=None) -> None:
    n = signal.shape[0]
    xref, yref = _refs(row)
    for a, b, s in _iter_segments(states):
        color = color_map[s % len(color_map)][0] if color_map else STATE_COLORS[s % len(STATE_COLORS)]
        _band(fig, row, a - 0.5, b - 0.5, color, BAND_OPACITY)
        if b - a >= 0.03 * n:
            fig.add_annotation(
                x=a,
                y=1,
                xref=xref,
                yref=f"{yref} domain",
                text=f"<b>{state_letter(s)}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                xshift=4,
                yshift=-2,
                font=dict(size=12, color=_shade(color, 0.36, 0.55), family=FONT),
            )
    _series(fig, signal, row)
    for cp in change_points or []:
        _vline(fig, row, cp - 0.5)


def _cpd_panel(fig, signal, pred_cps, truth_cps=None, row=None) -> None:
    n = signal.shape[0]
    edges = [0, *sorted(int(c) for c in pred_cps), n]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        _band(fig, row, a - 0.5, b - 0.5, _CPD_BANDS[i % 2])
    _series(fig, signal, row)
    for cp in truth_cps or []:
        _vline(fig, row, cp - 0.5, color=FAINT, width=1.6, dash="dot")
    for cp in pred_cps:
        _vline(fig, row, cp - 0.5)


def _style(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=12, color=MUTED),
        margin=dict(t=46, b=36, l=48, r=14),
        showlegend=legend,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font=dict(color="#ffffff", family=FONT, size=12)),
        title=dict(
            x=0,
            xref="paper",
            xanchor="left",
            y=1,
            yref="paper",
            yanchor="bottom",
            pad=dict(b=14),
            font=dict(size=14, color=INK, family=FONT),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=LINE,
        ticks="outside",
        tickcolor=LINE,
        ticklen=4,
        showspikes=False,
    )
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, nticks=4, tickfont=dict(size=11, color=FAINT))
    return fig


def _left_titles(fig: go.Figure, titles: Sequence[str]) -> None:
    """Subplot titles flush left above each row: bold name, then muted detail."""
    for row, text in enumerate(titles, start=1):
        axis = fig.layout["yaxis" if row == 1 else f"yaxis{row}"]
        top = axis.domain[1] if axis.domain else 1
        fig.add_annotation(
            text=text,
            x=0,
            xref="paper",
            xanchor="left",
            y=top,
            yref="paper",
            yanchor="bottom",
            yshift=4,
            showarrow=False,
            font=dict(size=13.5, color=INK, family=FONT),
        )


def _title(name: str, detail: str = "") -> str:
    detail_html = f'  <span style="color:{MUTED};font-weight:400">{detail}</span>' if detail else ""
    return f"<b>{name}</b>{detail_html}"


def _describe_states(states) -> str:
    segs = list(_iter_segments(states))
    k = len({s for _, _, s in segs})
    return f"{len(segs)} segments, {k} state{'s' if k != 1 else ''}"


# --------------------------------------------------------------------------
# Public figures
# --------------------------------------------------------------------------


def make_annotation_figure(
    signal: np.ndarray,
    change_points: Sequence[int],
    states: Sequence[int],
    title: str,
    color_map: list[list[str]],
) -> go.Figure:
    """A series with its states as bands and its change points as strokes."""
    n_samples, n_dims = signal.shape
    fig = go.Figure()
    _state_panel(fig, signal, states, change_points, color_map)
    _style(fig, height=min(300 + 18 * n_dims, 440))
    if title:
        fig.update_layout(title=dict(text=_title(title, "ground truth · " + _describe_states(states))))
    else:
        fig.update_layout(margin=dict(t=16))
    return fig


def make_signal_figure(signal: np.ndarray, title: str = "Signal") -> go.Figure:
    """A series without ground truth: channels only."""
    n_samples, n_dims = signal.shape
    fig = go.Figure()
    _series(fig, signal, None, legend=n_dims > 1)
    _style(fig, height=320, legend=n_dims > 1)
    fig.update_layout(title=dict(text=_title(title, "no ground truth")))
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
    task: str = "change_point_detection",
) -> go.Figure:
    """Ground truth above the prediction, on a shared time axis.

    A change point prediction (task ``change_point_detection``) is drawn with
    neutral bands, so that it does not suggest states, and the true change
    points are dotted on it.
    """
    has_truth = truth_states is not None
    n_rows = 2 if has_truth else 1
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    titles = []
    if has_truth:
        _state_panel(fig, signal, truth_states, truth_cps or [], color_map, row=1)
        titles.append(_title("Ground truth", _describe_states(truth_states)))

    row = 2 if has_truth else 1
    is_cpd = task == "change_point_detection"
    if pred_states is not None and not is_cpd:
        _state_panel(fig, signal, pred_states, pred_cps or [], color_map, row=row)
        titles.append(_title("Prediction", "states · " + _describe_states(pred_states)))
    elif pred_cps is not None and (len(pred_cps) or fallback_states is None):
        _cpd_panel(fig, signal, list(pred_cps), truth_cps if has_truth else None, row=row)
        detail = f"{len(pred_cps)} change point{'s' if len(pred_cps) != 1 else ''}"
        if has_truth:
            detail += " · dotted: true change points"
        titles.append(_title("Prediction", detail))
    elif fallback_states is not None:
        _state_panel(fig, signal, fallback_states, pred_cps or [], color_map, row=row)
        titles.append(_title("Prediction", _describe_states(fallback_states)))
    else:
        _cpd_panel(fig, signal, [], truth_cps if has_truth else None, row=row)
        titles.append(_title("Prediction", "no change point"))

    _left_titles(fig, titles)
    _style(fig, height=300 * n_rows)
    fig.update_xaxes(title_text="time", title_font=dict(size=12, color=FAINT), row=n_rows, col=1)
    return fig


def make_multi_comparison_figure(
    signal: np.ndarray,
    runs: list[dict],
    truth_states: Optional[Sequence[int]] = None,
    truth_cps: Optional[Sequence[int]] = None,
) -> go.Figure:
    """One row per run (and the ground truth first), on a shared time axis.

    Parameters
    ----------
    signal : np.ndarray
        The input signal, shape (n_timestamps, n_dims).
    runs : list of dict
        Each dict has keys: 'name', 'pred_states', 'pred_cps', 'task'.
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

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.3 / n_rows))
    titles = []
    row = 1
    if has_truth:
        _state_panel(fig, signal, truth_states, truth_cps or [], color_map, row=row)
        titles.append(_title("Ground truth", _describe_states(truth_states)))
        row += 1

    for run in runs:
        ps = run.get("pred_states")
        pc = list(run.get("pred_cps", []) or [])
        if ps is not None and run.get("task", "change_point_detection") == "state_detection":
            _state_panel(fig, signal, ps, pc, color_map, row=row)
            titles.append(_title(run["name"], "states · " + _describe_states(ps)))
        else:
            _cpd_panel(fig, signal, pc, truth_cps if has_truth else None, row=row)
            titles.append(_title(run["name"], f"{len(pc)} change point{'s' if len(pc) != 1 else ''}"))
        row += 1

    _left_titles(fig, titles)
    _style(fig, height=max(230 * n_rows, 420))
    fig.update_xaxes(title_text="time", title_font=dict(size=12, color=FAINT), row=n_rows, col=1)
    return fig


_METRIC_COLORS = {
    # change point metrics: blues and greens
    "F1 Score": "#2d5fb4",
    "F1 Score (precision)": "#6f97d8",
    "F1 Score (recall)": "#a3bfea",
    "Gaussian F1": "#3f73c2",
    "Covering": "#23804f",
    "Bidirectional Covering": "#5fb184",
    # state metrics: oranges and purples
    "Adjusted Rand Index": "#c65a22",
    "Normalized Mutual Information": "#e08a4f",
    "Adjusted Mutual Information": "#f0b48a",
    "Weighted ARI": "#7a3f9d",
    "Weighted NMI": "#a87bc6",
    "State Matching Score": "#0f172a",
}
_FALLBACK = ["#3f73c2", "#23804f", "#c65a22", "#7a3f9d", "#5b6b80", "#e0a23a"]


def make_metrics_bar_chart(scores_df) -> go.Figure:
    """Grouped bars: one group per run, one bar per metric."""
    import pandas as pd

    if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
        return go.Figure()

    fig = go.Figure()
    metric_cols = [c for c in scores_df.columns if c != "Algorithm"]
    for i, metric in enumerate(metric_cols):
        fig.add_trace(
            go.Bar(
                x=scores_df["Algorithm"],
                y=scores_df[metric],
                name=metric,
                marker=dict(color=_METRIC_COLORS.get(metric, _FALLBACK[i % len(_FALLBACK)]), cornerradius=4),
                hovertemplate=f"{metric}: %{{y:.3f}}<extra></extra>",
            )
        )
    _style(fig, height=420, legend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        hovermode="closest",
        title=dict(text=_title("Scores", "one bar per metric, higher is better")),
        legend=dict(y=-0.18, yanchor="top", x=0, xanchor="left"),
        margin=dict(t=46, b=90),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5", rangemode="tozero")
    return fig


def make_sms_breakdown_chart(breakdown_df) -> go.Figure:
    """Stack the contributions of each SMS error type to the total score loss."""
    colors = {"Delay": "#3f73c2", "Transition": "#e0a23a", "Isolation": "#c65a22", "Missing": "#7a3f9d"}
    fig = go.Figure()
    for error_type, color in colors.items():
        subset = breakdown_df[breakdown_df["Error type"] == error_type]
        fig.add_bar(
            x=subset["Algorithm"],
            y=subset["SMS loss"],
            name=error_type,
            marker=dict(color=color),
            hovertemplate=f"{error_type}: %{{y:.4f}}<extra></extra>",
        )
    _style(fig, height=380, legend=True)
    fig.update_layout(
        barmode="stack",
        bargap=0.45,
        hovermode="closest",
        title=dict(text=_title("SMS loss by error type", "the four parts add up to 1 − SMS")),
        legend=dict(y=-0.18, yanchor="top", x=0, xanchor="left", traceorder="normal"),
        margin=dict(t=46, b=80),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5", rangemode="tozero")
    return fig
