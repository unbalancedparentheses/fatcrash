"""Tests for P-LNN (Poly-LPPLS Neural Network)."""

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not installed")


class TestPLNNArchitecture:
    def test_4_hidden_layers(self):
        """P-LNN should have 4 hidden layers (paper spec)."""
        from fatcrash.nn.plnn import PLNNNetwork

        net = PLNNNetwork(input_dim=252)
        # Count Linear layers in sequential
        linear_layers = [m for m in net.net if isinstance(m, __import__("torch").nn.Linear)]
        assert len(linear_layers) == 5  # 4 hidden + 1 output

    def test_layer_sizes(self):
        """Verify layer dimensions match paper: 252->512->256->128->64->3."""
        from fatcrash.nn.plnn import PLNNNetwork

        net = PLNNNetwork(input_dim=252)
        linear_layers = [m for m in net.net if isinstance(m, __import__("torch").nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linear_layers]
        assert sizes == [(252, 512), (512, 256), (256, 128), (128, 64), (64, 3)]

    def test_output_shape(self):
        import torch
        from fatcrash.nn.plnn import PLNNNetwork

        net = PLNNNetwork(input_dim=100)
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 3)


class TestPLNNTraining:
    def test_train_small_dataset(self):
        """Train on tiny dataset to verify pipeline works."""
        from fatcrash.nn.plnn import train_plnn

        model = train_plnn(
            variant="P-LNN-test",
            n_samples=100,
            n_obs=50,
            batch_size=10,
            epochs=2,
            lr=1e-4,
            seed=42,
        )
        assert model.variant == "P-LNN-test"
        assert model.input_dim == 50

    def test_save_load_roundtrip(self):
        """Model should survive save/load."""
        import torch
        from fatcrash.nn.plnn import PLNNModel, PLNNNetwork, load_plnn

        net = PLNNNetwork(input_dim=50)
        model = PLNNModel(network=net, input_dim=50, variant="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)

            loaded = load_plnn(path)
            assert loaded.input_dim == 50
            assert loaded.variant == "test"

            # Check weights match
            x = torch.randn(1, 50)
            with torch.no_grad():
                orig_out = model.network(x)
                loaded_out = loaded.network(x)
            torch.testing.assert_close(orig_out, loaded_out)


class TestPLNNPrediction:
    def test_predict_output(self):
        """predict_plnn should return valid LPPLSNNResult."""
        from fatcrash.nn.plnn import PLNNModel, PLNNNetwork, predict_plnn

        net = PLNNNetwork(input_dim=50)
        model = PLNNModel(network=net, input_dim=50)

        times = np.arange(50, dtype=np.float64)
        log_prices = 10.0 - 0.3 * np.power(70 - times, 0.5)

        result = predict_plnn(model, times, log_prices, window_size=50)
        assert result.method == "P-LNN"
        assert isinstance(result.tc, float)
        assert 0.1 <= result.m <= 1.0
        assert 6.0 <= result.omega <= 13.0
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_shorter_input(self):
        """Should handle input shorter than model input_dim via padding."""
        from fatcrash.nn.plnn import PLNNModel, PLNNNetwork, predict_plnn

        net = PLNNNetwork(input_dim=100)
        model = PLNNModel(network=net, input_dim=100)

        times = np.arange(50, dtype=np.float64)
        log_prices = np.linspace(10, 11, 50)

        result = predict_plnn(model, times, log_prices, window_size=50)
        assert not np.isnan(result.tc)
