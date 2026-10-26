"""Data loading utilities: synthetic generation, built-in datasets, CSV upload."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

PatternType = Literal["constant", "sinusoidal", "step", "stairs", "linear", "mixed", "gp"]

MAX_UPLOAD_POINTS = 200_000  # safety limit for HF Spaces


@dataclass
class TimeSeriesData:
    """Container for a loaded time series with optional ground truth."""

    signal: np.ndarray  # shape (n_timestamps, n_dims)
    ground_truth: Optional[np.ndarray] = None  # state labels, shape (n_timestamps,)
    change_points: Optional[list[int]] = None
    name: str = "Unknown"
    n_dims: int = 1

    def __post_init__(self):
        if self.signal.ndim == 1:
            self.signal = self.signal[:, np.newaxis]
        self.n_dims = self.signal.shape[1]
        if self.ground_truth is not None and self.change_points is None:
            self.change_points = _states_to_cps(self.ground_truth)

    @property
    def length(self) -> int:
        return self.signal.shape[0]

    @property
    def n_states(self) -> int:
        if self.ground_truth is not None:
            return int(self.ground_truth.max()) + 1
        return 0

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None


def _states_to_cps(states: np.ndarray) -> list[int]:
    diffs = np.diff(states)
    return list(np.nonzero(diffs)[0] + 1)


# ---------------------------------------------------------------------------
# Synthetic Data
# ---------------------------------------------------------------------------


@dataclass
class SyntheticSeries:
    signal: np.ndarray
    change_points: list[int]
    change_points_one_hot: np.ndarray
    states: np.ndarray


def _generate_boundaries(
    length: int,
    n_segments: int,
    rng: np.random.Generator,
    min_segment_length: int = 20,
) -> list[int]:
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if min_segment_length < 1:
        raise ValueError("min_segment_length must be >= 1")
    if min_segment_length * n_segments > length:
        raise ValueError("length must be at least segments × minimum segment length")
    proportions = rng.dirichlet(alpha=np.ones(n_segments))
    segment_lengths = min_segment_length + rng.multinomial(length - min_segment_length * n_segments, proportions)
    return np.cumsum(segment_lengths)[:-1].astype(int).tolist()


def _sine_segment(start, stop, n_dims, mean_offset, amplitude, freq, phase, noise_level, rng):
    t = np.linspace(0, 1, stop - start, endpoint=False)
    base = amplitude * np.sin(2 * math.pi * (freq * t + phase)) + mean_offset
    noise_scale = (0.05 * amplitude if amplitude else 0.05) * noise_level
    noise = rng.normal(scale=max(noise_scale, 1e-12), size=(stop - start, n_dims))
    return np.tile(base[:, None], (1, n_dims)) + noise


def _complex_segment(start, stop, n_dims, trend, curvature, freq, noise_level, rng):
    t = np.linspace(0, 1, stop - start, endpoint=False)
    base = trend * t + curvature * (t - 0.5) ** 2
    seasonal = np.sin(2 * math.pi * freq * t)
    features = base + 0.6 * seasonal
    out = np.empty((stop - start, n_dims))
    for dim in range(n_dims):
        noise = rng.normal(scale=max(0.1 * noise_level, 1e-12), size=features.shape)
        drift = rng.normal(scale=max(0.2 * noise_level, 1e-12)) * t
        out[:, dim] = features + drift + noise
    return out


# -- Pattern-based segment generators --


def _gen_constant(length: int, n_dims: int, level: float, noise_level: float, rng) -> np.ndarray:
    """Flat segment at a given level."""
    noise = rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
    return np.full((length, n_dims), level) + noise


def _gen_sinusoidal(
    length: int, n_dims: int, amplitude: float, freq: float, phase: float, offset: float, noise_level: float, rng
) -> np.ndarray:
    """Sinusoidal segment."""
    t = np.linspace(0, 1, length, endpoint=False)
    base = amplitude * np.sin(2 * math.pi * (freq * t + phase)) + offset
    noise = rng.normal(scale=max(0.05 * amplitude * noise_level, 1e-12), size=(length, n_dims))
    return np.tile(base[:, None], (1, n_dims)) + noise


def _gen_step(length: int, n_dims: int, levels: Sequence[float], noise_level: float, rng) -> np.ndarray:
    """Piecewise-constant (step function) within a single segment."""
    n_steps = len(levels)
    step_len = length // n_steps
    out = np.empty((length, n_dims))
    for i, lvl in enumerate(levels):
        start = i * step_len
        stop = (i + 1) * step_len if i < n_steps - 1 else length
        out[start:stop] = lvl
    noise = rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
    return out + noise


def _gen_stairs(
    length: int, n_dims: int, start_level: float, step_size: float, n_stairs: int, noise_level: float, rng
) -> np.ndarray:
    """Staircase pattern — monotonically increasing within the segment."""
    stair_len = length // max(n_stairs, 1)
    out = np.empty((length, n_dims))
    for i in range(n_stairs):
        s = i * stair_len
        e = (i + 1) * stair_len if i < n_stairs - 1 else length
        out[s:e] = start_level + i * step_size
    noise = rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
    return out + noise


def _gen_linear(length: int, n_dims: int, start_val: float, end_val: float, noise_level: float, rng) -> np.ndarray:
    """Linear trend."""
    t = np.linspace(start_val, end_val, length)
    noise = rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
    return np.tile(t[:, None], (1, n_dims)) + noise


_PATTERN_GENERATORS = {
    "constant": "constant",
    "sinusoidal": "sinusoidal",
    "step": "step",
    "stairs": "stairs",
    "linear": "linear",
    "gp": "gp",
}


# -- Gaussian-process based segment (smooth, state-dependent) --


def _rbf_kernel(t: np.ndarray, lengthscale: float, variance: float = 1.0) -> np.ndarray:
    """Squared-exponential covariance matrix on a 1-D grid."""
    diff = t[:, None] - t[None, :]
    return variance * np.exp(-0.5 * (diff / max(lengthscale, 1e-3)) ** 2)


def _sample_gp(length: int, n_dims: int, lengthscale: float, variance: float, rng: np.random.Generator) -> np.ndarray:
    """Sample on at most 128 knots and interpolate, keeping large series responsive."""
    if length <= 1:
        return rng.normal(size=(length, n_dims))
    n_knots = min(length, 128)
    t = np.linspace(0.0, 1.0, n_knots)
    K = _rbf_kernel(t, lengthscale=lengthscale, variance=variance)
    K = K + 1e-6 * np.eye(n_knots)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition if Cholesky fails
        w, V = np.linalg.eigh(K)
        w = np.clip(w, 1e-10, None)
        L = V * np.sqrt(w)
    z = rng.normal(size=(n_knots, n_dims))
    knots = L @ z
    if n_knots == length:
        return knots
    full_t = np.linspace(0.0, 1.0, length)
    return np.column_stack([np.interp(full_t, t, knots[:, dim]) for dim in range(n_dims)])


def _gen_gp(
    length: int,
    n_dims: int,
    lengthscale: float,
    variance: float,
    offset: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth Gaussian-process segment with state-dependent statistics."""
    base = _sample_gp(length, n_dims, lengthscale=lengthscale, variance=variance, rng=rng)
    noise = rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
    return base + offset + noise


def _generate_segment_by_pattern(
    pattern: str,
    state_idx: int,
    length: int,
    n_dims: int,
    noise_level: float,
    shape_rng: np.random.Generator,
    noise_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one segment according to the chosen pattern type.

    Parameters
    ----------
    shape_rng : numpy Generator
        Drives all *shape* draws (waveform parameters, GP hyper-parameters,
        mixed-pattern choice, GP base realization). Seed it from the state id
        so that recurring segments of the same state share the same generative
        process.
    noise_rng : numpy Generator, optional
        Drives the additive observation noise. If ``None``, ``shape_rng`` is
        reused (pixel-identical recurring states). Pass a per-segment RNG to
        keep the shape consistent across recurrences while varying noise.
    """
    if noise_rng is None:
        noise_rng = shape_rng
    if pattern == "constant":
        level = shape_rng.uniform(-2, 2)
        return _gen_constant(length, n_dims, level, noise_level, noise_rng)
    elif pattern == "sinusoidal":
        amp = shape_rng.uniform(0.5, 2.0)
        freq = shape_rng.uniform(1.0, 4.0)
        phase = shape_rng.uniform(0, 1)
        offset = shape_rng.uniform(-1, 1)
        return _gen_sinusoidal(length, n_dims, amp, freq, phase, offset, noise_level, noise_rng)
    elif pattern == "step":
        n_steps = int(shape_rng.integers(2, 5))
        levels = shape_rng.uniform(-2, 2, size=n_steps).tolist()
        return _gen_step(length, n_dims, levels, noise_level, noise_rng)
    elif pattern == "stairs":
        start = shape_rng.uniform(-2, 1)
        step_size = shape_rng.uniform(0.3, 1.0) * shape_rng.choice([-1, 1])
        n_stairs = int(shape_rng.integers(3, 7))
        return _gen_stairs(length, n_dims, start, step_size, n_stairs, noise_level, noise_rng)
    elif pattern == "linear":
        s = shape_rng.uniform(-2, 2)
        e = shape_rng.uniform(-2, 2)
        return _gen_linear(length, n_dims, s, e, noise_level, noise_rng)
    elif pattern == "gp":
        # All shape draws (hyper-parameters AND base GP sample) use shape_rng,
        # so two segments sharing the same state produce the same realization.
        lengthscale = float(shape_rng.uniform(0.04, 0.20))
        variance = float(shape_rng.uniform(0.4, 1.8))
        offset = float(shape_rng.uniform(-1.5, 1.5))
        base = _sample_gp(length, n_dims, lengthscale=lengthscale, variance=variance, rng=shape_rng)
        noise = noise_rng.normal(scale=max(0.05 * noise_level, 1e-12), size=(length, n_dims))
        return base + offset + noise
    elif pattern == "mixed":
        # Pick a sub-pattern deterministically from shape_rng so two segments
        # of the same recurring state pick the SAME sub-pattern.
        choice = shape_rng.choice(["constant", "sinusoidal", "step", "stairs", "linear", "gp"])
        return _generate_segment_by_pattern(choice, state_idx, length, n_dims, noise_level, shape_rng, noise_rng)
    else:
        # Fallback to constant
        return _gen_constant(length, n_dims, float(state_idx), noise_level, noise_rng)


def generate_synthetic_series(
    length: int = 1000,
    n_segments: int = 3,
    n_dims: int = 2,
    pattern: PatternType = "gp",
    noise_level: float = 0.5,
    recurring_states: bool = False,
    min_segment_length: int = 20,
    continuous: bool = False,
    random_state: int | None = None,
) -> TimeSeriesData:
    """Generate a synthetic multivariate time series with labelled segments.

    Parameters
    ----------
    pattern : str
        Intra-segment waveform: 'constant', 'sinusoidal', 'step', 'stairs',
        'linear', 'gp' (Gaussian process, smooth and state-dependent),
        or 'mixed' (random per segment).
    recurring_states : bool
        If True, state labels can repeat (fewer distinct states than segments).
        Useful for testing state-detection algorithms.
    min_segment_length : int
        Minimum length of each segment (default 20).
    continuous : bool
        If True, shift each segment so its first value matches the last value
        of the previous segment. This can hide level changes and make recurring
        states less recognisable, so the default is False.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    rng = np.random.default_rng(random_state)
    change_points = _generate_boundaries(
        length=length,
        n_segments=n_segments,
        rng=rng,
        min_segment_length=min_segment_length,
    )
    boundaries = [0, *change_points, length]

    # Assign state labels
    if recurring_states and n_segments >= 3:
        # Use fewer distinct states than segments (at least 2)
        n_distinct = max(2, n_segments // 2)
        state_ids: list[int] = []
        for i in range(n_segments):
            # Avoid same state as previous
            candidates = list(range(n_distinct))
            if state_ids:
                candidates = [c for c in candidates if c != state_ids[-1]]
            state_ids.append(int(rng.choice(candidates)))
    else:
        state_ids = list(range(n_segments))

    states = np.zeros(length, dtype=int)
    signal = np.zeros((length, n_dims), dtype=float)

    # One seed per *distinct state* drives all shape draws (waveform params,
    # GP hyper-params, mixed sub-pattern choice). Two segments sharing the
    # same state thus share the same generative process. A separate per-segment
    # seed drives only the additive observation noise, so recurring segments
    # look alike without being pixel-identical.
    distinct_states = sorted(set(state_ids))
    state_seeds: dict[int, int] = {s: int(rng.integers(0, 2**31)) for s in distinct_states}
    noise_seeds: list[int] = [int(rng.integers(0, 2**31)) for _ in range(len(state_ids))]
    channel_scales = {}
    channel_offsets = {}
    for sid in distinct_states:
        channel_rng = np.random.default_rng(state_seeds[sid] ^ 0x5EED)
        channel_scales[sid] = channel_rng.uniform(0.7, 1.3, size=n_dims)
        channel_offsets[sid] = channel_rng.uniform(-0.6, 0.6, size=n_dims)

    for seg_idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        sid = state_ids[seg_idx]
        seg_len = stop - start

        shape_rng = np.random.default_rng(state_seeds[sid])
        noise_rng = np.random.default_rng(noise_seeds[seg_idx])

        segment = _generate_segment_by_pattern(pattern, sid, seg_len, n_dims, noise_level, shape_rng, noise_rng)
        segment = segment * channel_scales[sid] + channel_offsets[sid]

        # Stitch segment to previous endpoint to suppress visual discontinuities
        # at the boundary (does not change the state label or boundary index).
        if continuous and seg_idx > 0 and seg_len > 0:
            prev_end = signal[start - 1]  # shape (n_dims,)
            shift = prev_end - segment[0]
            segment = segment + shift

        signal[start:stop] = segment
        states[start:stop] = sid

    recur_label = ", recurring" if recurring_states else ""
    continuity_label = ", level-matched" if continuous else ""
    return TimeSeriesData(
        signal=signal,
        ground_truth=states,
        change_points=change_points,
        name=f"Synthetic ({pattern}{recur_label}{continuity_label}, {n_segments} seg, {n_dims}D)",
    )


# ---------------------------------------------------------------------------
# Built-in Datasets
# ---------------------------------------------------------------------------

MOCAP_TRIALS = list(range(9))


def load_mocap_dataset(trial: int = 0) -> TimeSeriesData:
    """Load a sample CMU MoCap 86 trial from tsseg built-in datasets."""
    from tsseg.data.datasets import load_mocap

    X, y = load_mocap(trial=trial, return_X_y=True)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    return TimeSeriesData(
        signal=X,
        ground_truth=y,
        name=f"Sample dataset (CMU MoCap 86, trial {trial})",
    )


def get_builtin_datasets() -> dict[str, dict]:
    """Return available built-in datasets metadata."""
    return {
        "Sample dataset": {
            "description": (
                "A real-world trial from CMU MoCap 86 — humeral & femoral angles, 4 dimensions, 9 labelled trials."
            ),
            "loader": load_mocap_dataset,
            "params": {"trial": {"type": "int", "choices": MOCAP_TRIALS, "default": 0}},
        },
    }


# ---------------------------------------------------------------------------
# CSV Upload
# ---------------------------------------------------------------------------


def load_csv(
    file_path: str,
    separator: str = ",",
    has_header: bool = True,
    label_column: str | None = None,
) -> TimeSeriesData:
    """Load a time series from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    separator : str
        Column separator.
    has_header : bool
        Whether the first row is a header.
    label_column : str or None
        Name of the ground truth column (if any).
    """
    header = 0 if has_header else None
    df = pd.read_csv(file_path, sep=separator, header=header)

    if len(df) > MAX_UPLOAD_POINTS:
        raise ValueError(
            f"File too large: {len(df)} rows (max {MAX_UPLOAD_POINTS}). Please truncate or subsample your data."
        )

    ground_truth = None
    if label_column and label_column in df.columns:
        ground_truth = df[label_column].values.astype(int)
        df = df.drop(columns=[label_column])
    elif label_column and label_column not in df.columns:
        # Try auto-detect common names
        for candidate in [
            "label",
            "labels",
            "ground_truth",
            "gt",
            "state",
            "states",
            "class",
            "segment",
            "is_anomaly",
            "anomaly",
            "target",
            "y",
        ]:
            if candidate in df.columns:
                ground_truth = df[candidate].values.astype(int)
                df = df.drop(columns=[candidate])
                break
    else:
        # No label_column specified — still try auto-detect
        for candidate in [
            "label",
            "labels",
            "ground_truth",
            "gt",
            "state",
            "states",
            "class",
            "segment",
            "is_anomaly",
            "anomaly",
            "target",
            "y",
        ]:
            if candidate in df.columns:
                ground_truth = df[candidate].values.astype(int)
                df = df.drop(columns=[candidate])
                break

    # Drop common index / timestamp columns that are not signal data
    _INDEX_COLS = {
        "timestamp",
        "timestamps",
        "time",
        "date",
        "datetime",
        "index",
        "idx",
        "id",
        "row",
        "step",
    }
    cols_to_drop = [c for c in df.columns if str(c).lower().strip() in _INDEX_COLS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the CSV file.")

    signal = numeric_df.values.astype(float)

    # Check for NaN
    nan_ratio = np.isnan(signal).mean()
    if nan_ratio > 0.1:
        raise ValueError(f"Too many NaN values ({nan_ratio:.1%}). Please clean your data.")
    if nan_ratio > 0:
        # Forward-fill NaN
        df_filled = pd.DataFrame(signal).ffill().bfill()
        signal = df_filled.values

    return TimeSeriesData(
        signal=signal,
        ground_truth=ground_truth,
        name="Uploaded CSV",
    )
