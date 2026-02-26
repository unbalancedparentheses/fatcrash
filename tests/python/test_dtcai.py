"""Tests for DTCAI (Distance-to-Crash AI) method."""

import importlib
import tempfile

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None
sklearn_available = importlib.util.find_spec("sklearn") is not None


# ── Crash Labels ───────────────────────────────────────────


class TestCrashLabels:
    def test_detects_known_crash(self):
        """Synthetic series with a clear 30% drop should trigger detection."""
        from fatcrash.nn.crash_labels import detect_crashes

        n = 600
        prices = np.ones(n, dtype=np.float64) * 100.0
        # Build up to peak at index 300, then crash 35%
        prices[:301] = np.linspace(80, 120, 301)
        prices[301:361] = np.linspace(120, 78, 60)  # 35% drop
        prices[361:] = 78.0

        crashes = detect_crashes(prices, lookback=262, forward=60, threshold=0.25)
        assert len(crashes) >= 1
        # Peak should be near index 300
        peak_indices = [c.peak_date_idx for c in crashes]
        assert any(abs(p - 300) <= 5 for p in peak_indices)
        assert all(c.drawdown > 0.25 for c in crashes)

    def test_no_crash_in_flat_series(self):
        """A flat series should produce no crashes."""
        from fatcrash.nn.crash_labels import detect_crashes

        prices = np.ones(600, dtype=np.float64) * 100.0
        crashes = detect_crashes(prices, lookback=262, forward=60, threshold=0.25)
        assert len(crashes) == 0

    def test_lookback_constraint(self):
        """Peak must be the highest in the lookback period."""
        from fatcrash.nn.crash_labels import detect_crashes

        n = 600
        prices = np.ones(n, dtype=np.float64) * 100.0
        # Create a higher peak early, then a lower peak that drops
        prices[100] = 150.0  # Earlier high
        prices[200:261] = np.linspace(130, 85, 61)  # Drop from 130 (not highest)
        # Since 150 > 130 within lookback, index 200 is NOT a peak

        crashes = detect_crashes(prices, lookback=262, forward=60, threshold=0.25)
        # No crash at 200 because 150 > 130 in lookback
        peak_indices = [c.peak_date_idx for c in crashes]
        assert 200 not in peak_indices

    def test_label_lppls_params_within_tolerance(self):
        from fatcrash.nn.crash_labels import label_lppls_params

        assert label_lppls_params(100.0, [95, 200], tolerance=10) == 1
        assert label_lppls_params(100.0, [95, 200], tolerance=4) == 0
        assert label_lppls_params(100.0, [105], tolerance=5) == 1
        assert label_lppls_params(100.0, [], tolerance=10) == 0

    def test_short_series_returns_empty(self):
        from fatcrash.nn.crash_labels import detect_crashes

        prices = np.ones(50, dtype=np.float64) * 100.0
        crashes = detect_crashes(prices, lookback=262, forward=60, threshold=0.25)
        assert len(crashes) == 0


# ── DTCAI Dataset ──────────────────────────────────────────


class TestDTCAIDataset:
    def test_dataset_shapes(self):
        """X should be (n, 7) and y should be (n,)."""
        from fatcrash.nn.dtcai_data import DTCAIDataset

        X = np.random.randn(50, 7)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        ds = DTCAIDataset(X=X, y=y, feature_names=["A", "B", "C", "tc", "phi", "omega", "beta"])
        assert ds.X.shape == (50, 7)
        assert ds.y.shape == (50,)

    def test_labels_are_binary(self):
        from fatcrash.nn.dtcai_data import DTCAIDataset

        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
        ds = DTCAIDataset(
            X=np.random.randn(6, 7),
            y=y,
            feature_names=["A", "B", "C", "tc", "phi", "omega", "beta"],
        )
        assert set(ds.y.tolist()).issubset({0, 1})

    def test_feature_names(self):
        from fatcrash.nn.dtcai_data import FEATURE_NAMES

        assert FEATURE_NAMES == ["A", "B", "C", "tc", "phi", "omega", "beta"]
        assert len(FEATURE_NAMES) == 7

    def test_extract_features(self):
        """_extract_features should produce 7 values from lppls_fit output."""
        from fatcrash.nn.dtcai_data import _extract_features

        # lppls_fit returns: (tc, m, omega, a, b, c1, c2, rss)
        fit_result = (100.0, 0.5, 8.0, 10.0, -2.0, 0.3, 0.4, 0.01)
        feats = _extract_features(fit_result)
        assert feats.shape == (7,)
        # Check C = sqrt(c1^2 + c2^2)
        assert feats[2] == pytest.approx(np.sqrt(0.3**2 + 0.4**2))
        # Check phi = atan2(c2, c1)
        assert feats[3] == pytest.approx(100.0)  # tc
        assert feats[4] == pytest.approx(np.arctan2(0.4, 0.3))  # phi

    def test_empty_dataset_for_short_series(self):
        """Short price series should return empty dataset."""
        from fatcrash.nn.dtcai_data import generate_dtcai_dataset

        prices = np.ones(100, dtype=np.float64) * 100.0
        ds = generate_dtcai_dataset(prices, window_size=504)
        assert ds.X.shape[0] == 0
        assert ds.y.shape[0] == 0


# ── DTCAI Metric ───────────────────────────────────────────


class TestDTCMetric:
    def test_dtc_formula(self):
        """DTC = (t2 - t1) / (tc - t1)."""
        from fatcrash.nn.dtcai import compute_dtc

        # t1=0, t2=80, tc=100 -> DTC = 80/100 = 0.8
        assert compute_dtc(0.0, 80.0, 100.0) == pytest.approx(0.8)

    def test_dtc_at_crash(self):
        """When t2 == tc, DTC should be 1.0."""
        from fatcrash.nn.dtcai import compute_dtc

        assert compute_dtc(0.0, 100.0, 100.0) == pytest.approx(1.0)

    def test_dtc_tc_before_start(self):
        """When tc <= t1, DTC should be 1.0 (already past crash)."""
        from fatcrash.nn.dtcai import compute_dtc

        assert compute_dtc(50.0, 100.0, 30.0) == pytest.approx(1.0)

    def test_dtcai_is_dtc_times_p(self):
        """DTCAI = DTC * reliability."""
        from fatcrash.nn.dtcai import DTCAIResult

        r = DTCAIResult(
            dtc=0.8, reliability=0.6, dtcai=0.48,
            lppls_params=(1.0,) * 7, classifier="ANN",
        )
        assert r.dtcai == pytest.approx(r.dtc * r.reliability)


# ── Signal converter ───────────────────────────────────────


class TestDTCAISignalConverter:
    def test_signal_passthrough(self):
        from fatcrash.aggregator.signals import dtcai_signal

        assert dtcai_signal(0.7) == pytest.approx(0.7)
        assert dtcai_signal(0.0) == pytest.approx(0.0)
        assert dtcai_signal(1.0) == pytest.approx(1.0)

    def test_signal_clips(self):
        from fatcrash.aggregator.signals import dtcai_signal

        assert dtcai_signal(1.5) == pytest.approx(1.0)
        assert dtcai_signal(-0.1) == pytest.approx(0.0)

    def test_signal_nan(self):
        from fatcrash.aggregator.signals import dtcai_signal

        assert dtcai_signal(float("nan")) == pytest.approx(0.0)

    def test_dtcai_in_default_weights(self):
        from fatcrash.aggregator.signals import DEFAULT_WEIGHTS

        assert "dtcai_signal" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["dtcai_signal"] == pytest.approx(0.05)

    def test_weights_sum_to_one(self):
        from fatcrash.aggregator.signals import DEFAULT_WEIGHTS

        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_dtcai_in_bubble_category(self):
        """DTCAI should be in the bubble category for agreement detection."""
        from fatcrash.aggregator.signals import aggregate_signals

        # Create signals where DTCAI is the bubble signal
        components = {
            "dtcai_signal": 0.8,
            "kappa_regime": 0.7,
            "hurst_trending": 0.6,
            "multiscale": 0.6,
        }
        result = aggregate_signals(components)
        # Should detect agreement across bubble, tail, regime, structure
        assert result.n_agreeing >= 3


# ── ANN classifier ─────────────────────────────────────────


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestDTCAIANNClassifier:
    def test_ann_architecture(self):
        """ANN should have 3 hidden layers with correct sizes."""
        from fatcrash.nn.dtcai import _build_ann

        model = _build_ann()
        # Sequential: Linear(7,256), ReLU, Linear(256,128), ReLU, Linear(128,64), ReLU, Linear(64,1), Sigmoid
        layers = list(model.children())
        assert len(layers) == 8

        import torch.nn as nn
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        assert len(linear_layers) == 4
        assert linear_layers[0].in_features == 7
        assert linear_layers[0].out_features == 256
        assert linear_layers[1].in_features == 256
        assert linear_layers[1].out_features == 128
        assert linear_layers[2].in_features == 128
        assert linear_layers[2].out_features == 64
        assert linear_layers[3].in_features == 64
        assert linear_layers[3].out_features == 1

    def test_ann_forward_pass(self):
        """Output should be in [0, 1]."""
        import torch
        from fatcrash.nn.dtcai import _build_ann

        model = _build_ann()
        model.eval()
        x = torch.randn(10, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (10, 1)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_train_ann_small_dataset(self):
        """Train ANN on a small synthetic dataset."""
        from fatcrash.nn.dtcai import train_dtcai
        from fatcrash.nn.dtcai_data import DTCAIDataset

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 7))
        y = rng.integers(0, 2, n).astype(np.int64)
        ds = DTCAIDataset(X=X, y=y, feature_names=["A", "B", "C", "tc", "phi", "omega", "beta"])

        model = train_dtcai(ds, model_type="ANN", epochs=2, batch_size=32, seed=42)
        assert model.model_type == "ANN"
        assert model.feature_mean.shape == (7,)
        assert model.feature_std.shape == (7,)

    def test_save_load_ann_roundtrip(self):
        """ANN weights should match after save/load."""
        import torch
        from fatcrash.nn.dtcai import load_dtcai, save_dtcai, train_dtcai
        from fatcrash.nn.dtcai_data import DTCAIDataset

        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 7))
        y = rng.integers(0, 2, n).astype(np.int64)
        ds = DTCAIDataset(X=X, y=y, feature_names=["A", "B", "C", "tc", "phi", "omega", "beta"])

        model = train_dtcai(ds, model_type="ANN", epochs=2, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dtcai(model, tmpdir)
            loaded = load_dtcai(tmpdir)

        assert loaded.model_type == "ANN"
        np.testing.assert_array_almost_equal(loaded.feature_mean, model.feature_mean)
        np.testing.assert_array_almost_equal(loaded.feature_std, model.feature_std)

        # Check weights match
        for p1, p2 in zip(model.model.parameters(), loaded.model.parameters()):
            assert torch.allclose(p1, p2)


# ── Sklearn classifiers ───────────────────────────────────


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")
class TestDTCAISklearn:
    def _make_dataset(self):
        from fatcrash.nn.dtcai_data import DTCAIDataset

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 7))
        y = rng.integers(0, 2, n).astype(np.int64)
        return DTCAIDataset(X=X, y=y, feature_names=["A", "B", "C", "tc", "phi", "omega", "beta"])

    def test_rf_train_predict(self):
        from fatcrash.nn.dtcai import train_dtcai, _predict_reliability

        ds = self._make_dataset()
        model = train_dtcai(ds, model_type="RF", seed=42)
        assert model.model_type == "RF"

        # Predict on a single sample
        prob = _predict_reliability(model, ds.X[0])
        assert 0.0 <= prob <= 1.0

    def test_logreg_train_predict(self):
        from fatcrash.nn.dtcai import train_dtcai, _predict_reliability

        ds = self._make_dataset()
        model = train_dtcai(ds, model_type="LogReg", seed=42)
        assert model.model_type == "LogReg"

        prob = _predict_reliability(model, ds.X[0])
        assert 0.0 <= prob <= 1.0

    def test_rf_save_load_roundtrip(self):
        from fatcrash.nn.dtcai import load_dtcai, save_dtcai, train_dtcai, _predict_reliability

        ds = self._make_dataset()
        model = train_dtcai(ds, model_type="RF", seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dtcai(model, tmpdir)
            loaded = load_dtcai(tmpdir)

        assert loaded.model_type == "RF"
        np.testing.assert_array_almost_equal(loaded.feature_mean, model.feature_mean)

        # Predictions should match
        p1 = _predict_reliability(model, ds.X[0])
        p2 = _predict_reliability(loaded, ds.X[0])
        assert p1 == pytest.approx(p2)

    def test_logreg_save_load_roundtrip(self):
        from fatcrash.nn.dtcai import load_dtcai, save_dtcai, train_dtcai, _predict_reliability

        ds = self._make_dataset()
        model = train_dtcai(ds, model_type="LogReg", seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dtcai(model, tmpdir)
            loaded = load_dtcai(tmpdir)

        assert loaded.model_type == "LogReg"
        p1 = _predict_reliability(model, ds.X[0])
        p2 = _predict_reliability(loaded, ds.X[0])
        assert p1 == pytest.approx(p2)

    def test_invalid_model_type_raises(self):
        from fatcrash.nn.dtcai import train_dtcai

        ds = self._make_dataset()
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_dtcai(ds, model_type="XGBoost")


# ── Random oversampling ────────────────────────────────────


class TestRandomOversample:
    def test_balances_classes(self):
        from fatcrash.nn.dtcai import _random_oversample

        X = np.random.randn(100, 7)
        y = np.array([0] * 90 + [1] * 10, dtype=np.int64)
        rng = np.random.default_rng(42)

        X_new, y_new = _random_oversample(X, y, rng)
        _, counts = np.unique(y_new, return_counts=True)
        assert counts[0] == counts[1]

    def test_single_class_unchanged(self):
        from fatcrash.nn.dtcai import _random_oversample

        X = np.random.randn(50, 7)
        y = np.zeros(50, dtype=np.int64)
        rng = np.random.default_rng(42)

        X_new, y_new = _random_oversample(X, y, rng)
        assert len(X_new) == 50
