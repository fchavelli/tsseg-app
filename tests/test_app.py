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

        ts = generate_synthetic_series(
            length=500, n_segments=5, pattern="sinusoidal", recurring_states=True, random_state=0
        )
        assert ts.length == 500
        # Fewer distinct states than segments
        assert ts.n_states < 5

    def test_synthetic_boundaries_and_recurring_shapes(self):
        from src.data import generate_synthetic_series

        ts = generate_synthetic_series(
            length=500,
            n_segments=5,
            n_dims=3,
            pattern="constant",
            noise_level=0,
            recurring_states=True,
            min_segment_length=80,
            random_state=3,
        )
        bounds = [0, *ts.change_points, ts.length]
        assert min(np.diff(bounds)) >= 80
        segments = [ts.signal[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        labels = [int(ts.ground_truth[a]) for a in bounds[:-1]]
        for i, label in enumerate(labels):
            for j in range(i + 1, len(labels)):
                if labels[j] == label:
                    assert np.allclose(segments[i][0], segments[j][0])
        assert not np.allclose(segments[0][0, 0], segments[0][0, 1])

    def test_synthetic_rejects_impossible_minimum_and_gp_scales(self):
        from src.data import generate_synthetic_series

        with pytest.raises(ValueError, match="minimum segment length"):
            generate_synthetic_series(length=200, n_segments=12, min_segment_length=20)
        ts = generate_synthetic_series(length=5000, n_segments=2, pattern="gp", random_state=2)
        assert ts.signal.shape == (5000, 2)
        assert np.isfinite(ts.signal).all()

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

    def test_catalogue_tracks_runtime_detectors(self):
        from src.about_data import TAXONOMY, build_algo_table
        from src.algorithms import get_all_detectors

        detectors = get_all_detectors()
        assert {"AutoPlait", "ChangeFinder", "ChangeForest", "GaussianHMM", "RuLSIF", "SeededBinSeg"} <= set(detectors)
        assert "Patss" not in detectors
        assert set(detectors) <= set(TAXONOMY)
        rows = build_algo_table(detectors).splitlines()[2:]
        assert {row.split("|")[1].strip().split("](")[0][1:] for row in rows} == set(detectors)
        assert all("https://" in row for row in rows)
        assert any("$O(" in row for row in rows)
        assert "complexity-help" in build_algo_table(detectors).splitlines()[0]

    def test_task_menu_marks_state_detectors_in_cpd(self):
        from app import _get_detector_choices

        cpd = dict(_get_detector_choices("Change Point Detection"))
        sd = dict(_get_detector_choices("State Detection"))
        assert "GaussianHMM · SD (change points inferred)" in cpd
        assert cpd["GaussianHMM · SD (change points inferred)"] == "GaussianHMM"
        assert "GaussianHMM" in sd
        assert "Pelt" in cpd and "Pelt" not in sd

    def test_source_discovery_includes_lazy_exports(self):
        import importlib.util
        from pathlib import Path

        from src.algorithms import _parse_algorithms_init_exports

        spec = importlib.util.find_spec("tsseg.algorithms")
        assert spec is not None and spec.origin is not None
        exports = dict((name, module) for module, name in _parse_algorithms_init_exports(Path(spec.origin)))
        assert exports["BeastDetector"] == "beast.detector"
        assert exports["ChangeForestDetector"] == "changeforest.detector"
        assert exports["PeltDetector"] == "pelt.detector"

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
    def test_sms_error_decomposition_matches_score(self):
        from tsseg.metrics import StateMatchingScore

        from src.metrics import summarize_sms_errors

        truth = np.array([0] * 5 + [1] * 5 + [2] * 5)
        pred = np.array([0] * 6 + [1] * 4 + [2] * 5)
        rows = summarize_sms_errors(truth, pred)
        score = StateMatchingScore().compute(truth, pred)["score"]
        assert {row["Error type"] for row in rows} == {"Delay", "Transition", "Isolation", "Missing"}
        assert sum(row["Points"] for row in rows) > 0
        assert abs(sum(row["SMS loss"] for row in rows) - (1 - score)) < 0.0003

    def test_sd_run_from_cpd_menu_can_be_evaluated_as_sd(self):
        from app import create_app
        from src.data import TimeSeriesData

        ui = create_app()
        evaluate = next(item.fn for item in ui.fns.values() if item.fn and item.fn.__name__ == "evaluate_runs")
        truth = np.array([0] * 50 + [1] * 50)
        data = TimeSeriesData(signal=np.ones((100, 2)), ground_truth=truth)
        # Older sessions recorded the selected CPD task even for an SD detector.
        run = {
            "name": "GaussianHMM",
            "task": "change_point_detection",
            "pred_states": [0] * 55 + [1] * 45,
            "pred_cps": [55],
        }
        _, scores, _, breakdown, chart = evaluate(data, [run], "State Detection", [], ["State Matching Score"], 5)
        assert scores["State Matching Score"].iloc[0] < 1
        assert breakdown["Points"].sum() == 5
        assert chart is not None

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
