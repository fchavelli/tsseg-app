"""Data loading utilities: synthetic generation, built-in datasets, CSV upload."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

ComplexityLevel = Literal["simple", "medium", "complex"]

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


def _generate_boundaries(length: int, n_segments: int, rng: np.random.Generator) -> list[int]:
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if length <= n_segments:
        raise ValueError("length must exceed number of segments")
    proportions = rng.dirichlet(alpha=np.ones(n_segments))
    segment_lengths = np.maximum(5, (proportions * length).astype(int))
    delta = length - int(segment_lengths.sum())
    segment_lengths[0] += delta
    boundaries: list[int] = []
    cursor = 0
    for seg_len in segment_lengths[:-1]:
        cursor += int(seg_len)
        boundaries.append(min(cursor, length - 1))
    return [cp for cp in boundaries if cp < length]


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


def generate_synthetic_series(
    length: int = 1000,
    n_segments: int = 3,
    n_dims: int = 1,
    complexity: ComplexityLevel = "medium",
    noise_level: float = 0.5,
    random_state: int | None = None,
) -> TimeSeriesData:
    """Generate a synthetic multivariate time series with labelled segments."""
    if length <= 0:
        raise ValueError("length must be positive")
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    rng = np.random.default_rng(random_state)
    change_points = _generate_boundaries(length=length, n_segments=n_segments, rng=rng)
    states = np.zeros(length, dtype=int)
    signal = np.zeros((length, n_dims), dtype=float)
    boundaries = [0, *change_points, length]

    for idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if complexity == "simple":
            level = np.full((n_dims,), float(idx))
            level += rng.normal(scale=max(0.2 * noise_level, 1e-12), size=(n_dims,))
            noise = rng.normal(scale=max((0.05 + 0.05 * idx) * noise_level, 1e-12), size=(stop - start, n_dims))
            segment = level + noise
        elif complexity == "medium":
            amplitude = 1.0 + 0.5 * idx
            segment = _sine_segment(start, stop, n_dims, idx * 0.5, amplitude, 1 + 0.5 * idx, rng.uniform(0, 1), noise_level, rng)
        elif complexity == "complex":
            segment = _complex_segment(start, stop, n_dims, rng.normal(loc=0.5 * idx, scale=0.2), rng.uniform(-1.0, 1.0), rng.uniform(1.0, 3.0), noise_level, rng)
        else:
            raise ValueError(f"Unknown complexity level: {complexity}")
        signal[start:stop] = segment
        states[start:stop] = idx

    return TimeSeriesData(
        signal=signal,
        ground_truth=states,
        change_points=change_points,
        name=f"Synthetic ({complexity}, {n_segments} segments, {n_dims}D)",
    )


# ---------------------------------------------------------------------------
# Built-in Datasets
# ---------------------------------------------------------------------------

MOCAP_TRIALS = list(range(9))


def load_mocap_dataset(trial: int = 0) -> TimeSeriesData:
    """Load a MoCap trial from tsseg built-in datasets."""
    from tsseg.data.datasets import load_mocap

    X, y = load_mocap(trial=trial, return_X_y=True)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    return TimeSeriesData(
        signal=X,
        ground_truth=y,
        name=f"MoCap (trial {trial})",
    )


def get_builtin_datasets() -> dict[str, dict]:
    """Return available built-in datasets metadata."""
    return {
        "MoCap": {
            "description": "CMU MoCap 86 — humeral/femoral angles, 4 dimensions, 9 trials",
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
            f"File too large: {len(df)} rows (max {MAX_UPLOAD_POINTS}). "
            "Please truncate or subsample your data."
        )

    ground_truth = None
    if label_column and label_column in df.columns:
        ground_truth = df[label_column].values.astype(int)
        df = df.drop(columns=[label_column])
    elif label_column and label_column not in df.columns:
        # Try auto-detect common names
        for candidate in ["label", "labels", "ground_truth", "gt", "state", "states", "class", "segment"]:
            if candidate in df.columns:
                ground_truth = df[candidate].values.astype(int)
                df = df.drop(columns=[candidate])
                break

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
