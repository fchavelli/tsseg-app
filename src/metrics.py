"""Metrics wrapper for evaluating segmentation results."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from tsseg.metrics import (
    AdjustedMutualInformation,
    AdjustedRandIndex,
    BidirectionalCovering,
    Covering,
    F1Score,
    GaussianF1Score,
    NormalizedMutualInformation,
    StateMatchingScore,
    WeightedAdjustedRandIndex,
    WeightedNormalizedMutualInformation,
)

# -- Registry of metrics by task type --

# NOTE: The app always passes per-timestep state labels (not change point
# index lists) to metric.compute().  Every CPD metric therefore needs
# ``convert_labels_to_segments=True`` so that labels are automatically
# converted to the [0, cp1, cp2, …, T] format that the metrics expect.

CPD_METRICS = {
    "F1 Score": {
        "class": F1Score,
        "default_params": {"margin": 5, "convert_labels_to_segments": True},
        "description": "F1, precision, recall with margin tolerance",
    },
    "Covering": {
        "class": Covering,
        "default_params": {"convert_labels_to_segments": True},
        "description": "Segment-wise IoU weighted by segment duration",
    },
    "Bidirectional Covering": {
        "class": BidirectionalCovering,
        "default_params": {"aggregation": "harmonic", "convert_labels_to_segments": True},
        "description": "Bidirectional covering (truth→pred + pred→truth)",
    },
    "Gaussian F1": {
        "class": GaussianF1Score,
        "default_params": {"sigma_fraction": 0.01, "convert_labels_to_segments": True},
        "description": "Gaussian-weighted F1 with soft matching (no hard margin)",
    },
}

STATE_METRICS = {
    "Adjusted Rand Index": {
        "class": AdjustedRandIndex,
        "default_params": {},
        "description": "Adjusted Rand Index (sklearn)",
    },
    "Normalized Mutual Information": {
        "class": NormalizedMutualInformation,
        "default_params": {},
        "description": "Normalized Mutual Information (sklearn)",
    },
    "Adjusted Mutual Information": {
        "class": AdjustedMutualInformation,
        "default_params": {},
        "description": "Adjusted Mutual Information (sklearn)",
    },
    "Weighted ARI": {
        "class": WeightedAdjustedRandIndex,
        "default_params": {},
        "description": "Weighted ARI (segment boundary aware)",
    },
    "Weighted NMI": {
        "class": WeightedNormalizedMutualInformation,
        "default_params": {},
        "description": "Weighted NMI (segment boundary aware)",
    },
    "State Matching Score": {
        "class": StateMatchingScore,
        "default_params": {},
        "description": "SMS with Hungarian matching and error classification",
    },
}


def get_metrics_for_task(task: str) -> dict:
    """Return metric registry for the given task type."""
    if task == "change_point_detection":
        return CPD_METRICS
    elif task == "state_detection":
        return STATE_METRICS
    return {}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    selected_metrics: Optional[list[str]] = None,
    metric_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Compute selected metrics and return a flat dict of scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth state labels (one integer per timestep).
    y_pred : np.ndarray
        Predicted state labels (one integer per timestep).
    task : str
        "change_point_detection" or "state_detection".
    selected_metrics : list of str, optional
        Which metrics to compute. If None, compute all available.
    metric_params : dict, optional
        Override default parameters for specific metrics.
    """
    registry = get_metrics_for_task(task)
    if selected_metrics is None:
        selected_metrics = list(registry.keys())

    metric_params = metric_params or {}
    results: Dict[str, float] = {}

    for metric_name in selected_metrics:
        if metric_name not in registry:
            continue
        info = registry[metric_name]
        params = {**info["default_params"], **metric_params.get(metric_name, {})}
        try:
            metric_instance = info["class"](**params)
            output = metric_instance.compute(y_true, y_pred)

            score = output.get("score", None)
            if score is not None:
                results[metric_name] = round(float(score), 4)
            # Also capture sub-scores
            for key, val in output.items():
                if key != "score" and isinstance(val, (int, float)):
                    results[f"{metric_name} ({key})"] = round(float(val), 4)
        except Exception as e:
            results[metric_name] = f"Error: {e}"

    return results
