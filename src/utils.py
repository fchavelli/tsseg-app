"""Helper utilities for the app."""

from __future__ import annotations

import ast
import concurrent.futures
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Execution timeout
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 300  # seconds


class ExecutionTimeoutError(Exception):
    """Raised when a detector exceeds the allowed execution time."""


def run_with_timeout(fn: Callable, *args, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Any:
    """Run *fn* in a thread with a timeout.

    Raises ``ExecutionTimeoutError`` if the function does not return in
    *timeout* seconds.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise ExecutionTimeoutError(
                f"Execution timed out after {timeout}s. "
                f"Try a smaller dataset or simpler algorithm parameters."
            ) from None


def states_to_change_points(states: Sequence[int]) -> list[int]:
    """Derive change points from state labels."""
    diffs = np.diff(states)
    return list(np.nonzero(diffs)[0] + 1)


def change_points_to_states(change_points: Sequence[int], length: int) -> np.ndarray:
    """Convert sorted change-point list to per-sample state labels."""
    labels = np.zeros(length, dtype=int)
    boundaries = [0, *sorted(cp for cp in change_points if 0 < cp < length), length]
    for idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[start:stop] = idx
    return labels


def one_hot_from_change_points(change_points: Sequence[int], length: int) -> np.ndarray:
    vec = np.zeros(length, dtype=int)
    for cp in change_points:
        if 0 <= cp < length:
            vec[cp] = 1
    return vec


def align_states_via_hungarian(true_states: Sequence[int], pred_states: Sequence[int]) -> np.ndarray:
    """Relabel predictions to maximise agreement with ground truth."""
    true_states = np.asarray(true_states, dtype=int)
    pred_states = np.asarray(pred_states, dtype=int)
    n_true = true_states.max() + 1
    n_pred = pred_states.max() + 1
    size = max(n_true, n_pred)
    cost = np.zeros((size, size), dtype=int)
    for t, p in zip(true_states, pred_states):
        cost[t, p] -= 1
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in pred_states], dtype=int)


def parse_user_value(raw: str, default: Any) -> Any:
    """Parse a string input using ``ast.literal_eval`` with fallbacks."""
    if raw.strip() == "":
        return default
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Apply z-normalisation along time for each dimension."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (signal - mean) / std


def interpret_predictions(output: Any, length: int, expect_states: bool):
    """Normalise detector output into (pred_states, pred_cps).

    Returns
    -------
    pred_states : np.ndarray or None
    pred_cps : list[int] or None
    """
    pred_states = None
    pred_cps = None
    raw_state_series = None

    if output is None:
        return pred_states, pred_cps

    if isinstance(output, dict) and "change_points" in output:
        pred_cps = [int(x) for x in output["change_points"]]
    elif isinstance(output, (list, tuple)):
        primary = output[0] if output and isinstance(output[0], (list, tuple, np.ndarray)) else output
        arr = np.asarray(primary)
        if arr.ndim == 1 and arr.size == length:
            raw_state_series = arr.astype(int)
        else:
            pred_cps = [int(x) for x in arr]
    else:
        arr = np.asarray(output)
        if arr.ndim == 1 and arr.size == length:
            raw_state_series = arr.astype(int)
        else:
            pred_cps = [int(x) for x in np.ravel(arr)]

    if expect_states:
        pred_states = raw_state_series
        if pred_states is None and pred_cps is not None:
            pred_states = change_points_to_states(pred_cps, length)
        if pred_states is not None and pred_cps is None:
            pred_cps = states_to_change_points(pred_states)
        return pred_states, pred_cps

    if pred_cps is None and raw_state_series is not None:
        pred_cps = states_to_change_points(raw_state_series)
    return None, pred_cps
