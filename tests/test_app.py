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

    def test_generate_synthetic_patterns(self):
        from src.data import generate_synthetic_series

        for p in ("constant", "sinusoidal", "step", "stairs", "linear", "mixed"):
            ts = generate_synthetic_series(length=300, n_segments=2, pattern=p, random_state=0)
            assert ts.length == 300

    def test_generate_synthetic_recurring(self):
        from src.data import generate_synthetic_series

        ts = generate_synthetic_series(length=500, n_segments=5, pattern="sinusoidal",
                                       recurring_states=True, random_state=0)
        assert ts.length == 500
        # Fewer distinct states than segments
        assert ts.n_states < 5

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

    def test_param_info_uses_schema(self):
        """Schema-based detectors return descriptions and groups."""
        from src.algorithms import get_all_detectors
        from src.params import get_param_info

        detectors = get_all_detectors()
        if "BOCDDetector" in detectors:
            params = get_param_info(detectors["BOCDDetector"].cls)
            assert len(params) > 0
            # Schema provides descriptions
            p_map = {p["name"]: p for p in params}
            if "hazard_lambda" in p_map:
                assert p_map["hazard_lambda"].get("description")

    def test_param_info_data_ctx_adjusts_bounds(self):
        """When data_ctx is provided, bounds reflect data dimensions."""
        from src.algorithms import get_all_detectors
        from src.params import get_param_info

        detectors = get_all_detectors()
        # E2USD has DataDependent: window_size * 2 <= n_samples
        if "E2USDDetector" in detectors:
            ctx = {"n_samples": 100, "n_channels": 1}
            params = get_param_info(detectors["E2USDDetector"].cls, data_ctx=ctx)
            p_map = {p["name"]: p for p in params}
            ws = p_map.get("window_size")
            if ws and "max" in ws:
                assert ws["max"] <= 100, f"window_size max should be <= n_samples, got {ws['max']}"

    def test_param_info_auto_clamp_default(self):
        """Default values are clamped when they exceed data bounds."""
        from src.algorithms import get_all_detectors
        from src.params import get_param_info

        detectors = get_all_detectors()
        if "E2USDDetector" in detectors:
            # E2USD has window_size=256 by default, which is too big for 100 samples
            ctx = {"n_samples": 100, "n_channels": 1}
            params = get_param_info(detectors["E2USDDetector"].cls, data_ctx=ctx)
            p_map = {p["name"]: p for p in params}
            ws = p_map.get("window_size")
            if ws:
                assert ws["default"] <= 100, f"Expected window_size default clamped, got {ws['default']}"
                assert ws.get("adjusted") is True

    def test_build_data_context(self):
        from src.params import build_data_context

        signal = np.random.randn(500, 3)
        ctx = build_data_context(signal)
        assert ctx["n_samples"] == 500
        assert ctx["n_channels"] == 3

    def test_validate_params_for_detector(self):
        from src.algorithms import get_all_detectors
        from src.params import validate_params_for_detector

        detectors = get_all_detectors()
        if "PeltDetector" in detectors:
            cls = detectors["PeltDetector"].cls
            params = {"model": "l2", "penalty": 10.0}
            adj, warnings, errors = validate_params_for_detector(cls, params)
            # Should not crash, may or may not have errors
            assert isinstance(adj, dict)
            assert isinstance(warnings, list)
            assert isinstance(errors, list)


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
