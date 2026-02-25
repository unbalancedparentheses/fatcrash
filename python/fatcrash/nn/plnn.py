"""Poly-LPPLS Neural Network (P-LNN).

Pre-trained on synthetic data, ~700x faster than CMA-ES at inference.
The network maps a 252-observation min-max normalized price window
directly to (tc, m, omega) in a single forward pass.

Reference: Nielsen, Sornette, Raissi (2024) "Deep Learning for LPPLS", arXiv:2405.12803
Section 3.2: P-LNN architecture, synthetic data generation, and training.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from pathlib import Path

from fatcrash.nn import _require_torch

_require_torch()

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fatcrash.nn.lppls_torch import (
    LPPLSNNResult,
    min_max_normalize,
    passes_nielsen_filter,
)
from fatcrash.nn.synthetic import generate_dataset


class PLNNNetwork(nn.Module):
    """P-LNN: 4-hidden-layer network for fast LPPLS parameter prediction.

    Architecture (per paper Section 3.2):
        Linear(252, 512) -> ReLU
        Linear(512, 256) -> ReLU
        Linear(256, 128) -> ReLU
        Linear(128, 64) -> ReLU
        Linear(64, 3)
    """

    def __init__(self, input_dim: int = 252):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 3) raw predictions for (tc_norm, m, omega)."""
        return self.net(x)


class PLNNModel:
    """Wrapper around a trained P-LNN for convenient inference."""

    def __init__(self, network: PLNNNetwork, input_dim: int = 252, variant: str = "P-LNN"):
        self.network = network
        self.input_dim = input_dim
        self.variant = variant

    def save(self, path: str | Path) -> None:
        """Save model weights to disk."""
        torch.save(
            {"state_dict": self.network.state_dict(), "input_dim": self.input_dim, "variant": self.variant},
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> PLNNModel:
        """Load a saved P-LNN model from disk."""
        checkpoint = torch.load(str(path), weights_only=False)
        input_dim = checkpoint.get("input_dim", 252)
        variant = checkpoint.get("variant", "P-LNN")
        network = PLNNNetwork(input_dim=input_dim)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        return cls(network=network, input_dim=input_dim, variant=variant)


def train_plnn(
    variant: str = "P-LNN-100K",
    n_samples: int = 100_000,
    n_obs: int = 252,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-5,
    seed: int = 42,
) -> PLNNModel:
    """Train a P-LNN on synthetic LPPLS data.

    Args:
        variant: name for this model variant
        n_samples: number of synthetic training series
        n_obs: observations per series
        batch_size: mini-batch size (default 8 per paper)
        epochs: training epochs (default 20 per paper)
        lr: learning rate (default 1e-5 per paper)
        seed: random seed

    Returns:
        Trained PLNNModel ready for inference
    """
    # Determine noise type from variant name
    if "AR1" in variant.upper():
        noise_type = "ar1"
    elif "BOTH" in variant.upper():
        noise_type = "both"
    else:
        noise_type = "white"

    dataset = generate_dataset(n_samples=n_samples, n_obs=n_obs, noise_type=noise_type, seed=seed)

    X_tensor = torch.from_numpy(dataset.X)
    y_tensor = torch.from_numpy(dataset.y)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    torch.manual_seed(seed)
    network = PLNNNetwork(input_dim=n_obs)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    network.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            pred = network(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    network.eval()
    return PLNNModel(network=network, input_dim=n_obs, variant=variant)


def load_plnn(path: str | Path) -> PLNNModel:
    """Load a pre-trained P-LNN model from disk."""
    return PLNNModel.load(path)


def predict_plnn(
    model: PLNNModel,
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    window_size: int = 252,
) -> LPPLSNNResult:
    """Run P-LNN inference on a price window.

    Uses the last `window_size` observations. Prices are min-max normalized
    before feeding to the network.

    Args:
        model: trained PLNNModel
        times: observation times
        log_prices: corresponding log prices
        window_size: number of observations to use (must match model input_dim)

    Returns:
        LPPLSNNResult with predicted parameters
    """
    n = min(window_size, len(times))
    t = times[-n:]
    lp = log_prices[-n:]
    t2 = float(t[-1])
    n_obs = len(t)

    # Min-max normalize
    lp_tensor = torch.tensor(lp, dtype=torch.float32)
    lp_norm = min_max_normalize(lp_tensor)

    # Pad or truncate to match model input dimension
    if len(lp_norm) < model.input_dim:
        padded = torch.zeros(model.input_dim)
        padded[-len(lp_norm):] = lp_norm
        lp_norm = padded
    elif len(lp_norm) > model.input_dim:
        lp_norm = lp_norm[-model.input_dim:]

    model.network.eval()
    with torch.no_grad():
        raw = model.network(lp_norm.unsqueeze(0))  # (1, 3)
        tc_norm_pred = raw[0, 0].item()
        m_pred = raw[0, 1].item()
        omega_pred = raw[0, 2].item()

    # De-normalize tc: tc_norm was scaled to [0,1] relative to (n_obs + 50)
    tc_pred = tc_norm_pred * (n_obs + 50)

    # Map tc to actual time coordinates
    t_start = float(t[0])
    t_step = (t2 - t_start) / max(n_obs - 1, 1)
    tc_actual = t_start + tc_pred * t_step

    # Clamp m and omega to valid ranges
    m_pred = float(np.clip(m_pred, 0.1, 1.0))
    omega_pred = float(np.clip(omega_pred, 6.0, 13.0))

    # Solve linear params via Rust for speed, fallback to numpy
    try:
        from fatcrash._core import lppls_solve_linear

        result = lppls_solve_linear(
            np.asarray(t, dtype=np.float64),
            np.asarray(lp, dtype=np.float64),
            tc_actual,
            m_pred,
            omega_pred,
        )
        if result is not None:
            a, b, c1, c2, rss = result
        else:
            a, b, c1, c2, rss = 0.0, 0.0, 0.0, 0.0, float("inf")
    except Exception:
        a, b, c1, c2, rss = 0.0, 0.0, 0.0, 0.0, float("inf")

    is_bubble = passes_nielsen_filter(tc_actual, m_pred, omega_pred, t2, b)
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + rss * 100)))

    return LPPLSNNResult(
        tc=tc_actual, m=m_pred, omega=omega_pred,
        a=a, b=b, c1=c1, c2=c2,
        rss=rss,
        is_bubble=is_bubble,
        confidence=confidence,
        method=model.variant,
    )
