"""Basic tests for tsseg-app modules."""

from __future__ import annotations

import numpy as np
import pytest


class TestData:
    def test_generate_synthetic(self):
        from src.data import generate_synthetic_series

        ts = generate_synthetic_series(length=500, n_segments=3, n_dims=2, random_state=42)
        assert ts.signal.shape == (500, 2)
        assert ts.ground_truth is not None
        assert len(ts.change_points) == 2

    def test_generate_synthetic_complexities(self):
        from src.data import generate_synthetic_series

        for c in ("simple", "medium", "complex"):
            ts = generate_synthetic_series(length=300, n_segments=2, complexity=c, random_state=0)
            assert ts.length == 300

    def test_mocap_dataset(self):
        from src.data import load_mocap_dataset

        ts = load_mocap_dataset(trial=0)
        assert ts.signal.ndim == 2
        assert ts.has_ground_truth


class TestAlgorithms:
    def test_get_all_detectors(self):
        from src.algorithms import get_all_detectors

        detectors = get_all_detectors()
        assert len(detectors) > 0
        for name, info in detectors.items():
            assert info.name == name
            assert info.detector_type in ("change_point_detection", "state_detection")

    def test_get_detectors_by_type(self):
        from src.algorithms import get_detectors_by_type

        cpd = get_detectors_by_type("change_point_detection")
        sd = get_detectors_by_type("state_detection")
        assert len(cpd) > 0
        assert len(sd) > 0
        # No overlap
        assert set(cpd.keys()).isdisjoint(set(sd.keys()))

    def test_constructor_signature(self):
        from src.algorithms import get_all_detectors, get_constructor_signature

        for name, info in list(get_all_detectors().items())[:5]:
            sig = get_constructor_signature(info.cls)
            assert "self" not in [p.name for p in sig.parameters.values()]


class TestUtils:
    def test_states_to_change_points(self):
        from src.utils import states_to_change_points

        states = [0, 0, 0, 1, 1, 2, 2, 2]
        cps = states_to_change_points(states)
        assert cps == [3, 5]

    def test_change_points_to_states(self):
        from src.utils import change_points_to_states

        states = change_points_to_states([3, 5], 8)
        assert list(states) == [0, 0, 0, 1, 1, 2, 2, 2]

    def test_normalize_zscore(self):
        from src.utils import normalize_zscore

        signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = normalize_zscore(signal)
        assert np.allclose(normed.mean(axis=0), 0, atol=1e-10)

    def test_interpret_predictions_cps(self):
        from src.utils import interpret_predictions

        output = [10, 20, 30]
        states, cps = interpret_predictions(output, 50, expect_states=False)
        assert states is None
        assert cps == [10, 20, 30]

    def test_interpret_predictions_states(self):
        from src.utils import interpret_predictions

        output = np.array([0, 0, 1, 1, 2, 2])
        states, cps = interpret_predictions(output, 6, expect_states=True)
        assert states is not None
        assert cps == [2, 4]


class TestParams:
    def test_get_param_info(self):
        from src.algorithms import get_all_detectors
        from src.params import get_param_info

        detectors = get_all_detectors()
        # Test with a known simple detector
        if "PeltDetector" in detectors:
            params = get_param_info(detectors["PeltDetector"].cls)
            assert len(params) > 0
            names = [p["name"] for p in params]
            assert "model" in names

    def test_tunable_ranges(self):
        from src.params import get_tunable_ranges

        ranges = get_tunable_ranges("PeltDetector")
        assert "model" in ranges
        assert "l2" in ranges["model"]


class TestMetrics:
    def test_compute_cpd_metrics(self):
        from src.metrics import compute_metrics

        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        scores = compute_metrics(y_true, y_pred, "change_point_detection")
        assert "F1 Score" in scores

    def test_compute_state_metrics(self):
        from src.metrics import compute_metrics

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        scores = compute_metrics(y_true, y_pred, "state_detection")
        assert "Adjusted Rand Index" in scores


class TestPlotting:
    def test_signal_figure(self):
        from src.plotting import make_signal_figure

        signal = np.random.randn(100, 2)
        fig = make_signal_figure(signal)
        assert fig is not None

    def test_annotation_figure(self):
        from src.plotting import build_segment_color_map, make_annotation_figure

        signal = np.random.randn(100, 1)
        states = np.array([0] * 50 + [1] * 50)
        cps = [50]
        color_map = build_segment_color_map(2, 1)
        fig = make_annotation_figure(signal, cps, states, "Test", color_map)
        assert fig is not None
