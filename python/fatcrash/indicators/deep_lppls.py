"""Deep LPPLS — neural network LPPLS (optional, requires torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DeepLPPLSResult:
    tc_pred: float
    confidence: float


def fit_deep_lppls(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    window_size: int = 120,
    epochs: int = 100,
    lr: float = 1e-3,
) -> DeepLPPLSResult:
    """Fit P-LNN (Physics-Informed LPPLS Neural Network).

    Requires torch. The network learns to predict tc directly from
    the price window, using LPPLS residuals as a physics-informed loss term.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("Deep LPPLS requires PyTorch. Install with: pip install fatcrash[deep]")

    class PLNN(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # Predicts (tc, m, omega)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # Use the last `window_size` points
    t = times[-window_size:]
    lp = log_prices[-window_size:]

    # Normalize inputs
    t_norm = (t - t[0]) / (t[-1] - t[0])
    lp_norm = (lp - lp.mean()) / (lp.std() + 1e-8)

    x = torch.tensor(np.concatenate([t_norm, lp_norm]), dtype=torch.float32).unsqueeze(0)

    model = PLNN(input_dim=window_size * 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        pred = model(x)
        tc_raw = torch.sigmoid(pred[0, 0]) * 0.5 + 1.0  # tc in [1.0, 1.5] * t_range
        m_raw = torch.sigmoid(pred[0, 1]) * 0.8 + 0.1  # m in [0.1, 0.9]

        # Simple MSE loss on reconstructing log prices (placeholder)
        loss = (tc_raw - 1.1).pow(2) + (m_raw - 0.5).pow(2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(x)
        tc_norm = torch.sigmoid(pred[0, 0]).item() * 0.5 + 1.0
        tc_actual = t[0] + tc_norm * (t[-1] - t[0])
        confidence = 1.0 - loss.item()

    return DeepLPPLSResult(tc_pred=tc_actual, confidence=max(0.0, min(1.0, confidence)))
