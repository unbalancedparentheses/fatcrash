"""Mono-LPPLS Neural Network (M-LNN).

One network trained per time series. The network predicts nonlinear LPPLS
parameters (tc, m, omega) from min-max normalized prices. Linear parameters
(A, B, C1, C2) are solved analytically via OLS at each forward pass.

Reference: Nielsen, Sornette, Raissi (2024) "Deep Learning for LPPLS", arXiv:2405.12803
Section 3.1: M-LNN architecture and training.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from fatcrash.nn import _require_torch

_require_torch()

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from fatcrash.nn.lppls_torch import (
    LPPLSNNResult,
    lppls_design_matrix,
    min_max_normalize,
    passes_nielsen_filter,
    solve_linear_torch,
)


class MLNNNetwork(nn.Module):
    """M-LNN: 2-hidden-layer network predicting (tc, m, omega).

    Architecture (per paper Section 3.1):
        Linear(N, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 3)
        Output clamped via sigmoid to parameter bounds.
    """

    def __init__(self, input_dim: int, tc_bounds: tuple[float, float]):
        super().__init__()
        self.tc_lo, self.tc_hi = tc_bounds
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        # Sigmoid clamping to parameter bounds
        tc = torch.sigmoid(raw[..., 0]) * (self.tc_hi - self.tc_lo) + self.tc_lo
        m = torch.sigmoid(raw[..., 1]) * 0.9 + 0.1  # [0.1, 1.0]
        omega = torch.sigmoid(raw[..., 2]) * 7.0 + 6.0  # [6, 13]
        return tc, m, omega


def fit_mlnn(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    epochs: int = 200,
    lr: float = 1e-2,
    penalty_weight: float = 10.0,
    seed: int = 42,
) -> LPPLSNNResult:
    """Fit M-LNN to a single time series.

    Trains a small network to predict (tc, m, omega) by minimizing
    reconstruction MSE between LPPLS fit and observed log-prices.

    Args:
        times: observation times (any length)
        log_prices: corresponding log prices
        epochs: training iterations (default 200 per paper)
        lr: learning rate (default 1e-2 per paper)
        penalty_weight: multiplier for bound-violation penalties
        seed: random seed

    Returns:
        LPPLSNNResult with fitted parameters
    """
    torch.manual_seed(seed)
    n = len(times)

    # Min-max normalize prices (paper spec)
    lp_tensor = torch.tensor(log_prices, dtype=torch.float32)
    lp_norm = min_max_normalize(lp_tensor)

    # Time tensor for LPPLS design matrix
    t_tensor = torch.tensor(times, dtype=torch.float32)
    t2 = float(times[-1])
    dt = t2 - float(times[0])

    # tc bounds: [t2 - 0.2*dt, t2 + 0.2*dt]
    tc_lo = t2 - 0.2 * dt
    tc_hi = t2 + 0.2 * dt

    model = MLNNNetwork(input_dim=n, tc_bounds=(tc_lo, tc_hi))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_result: tuple | None = None

    x = lp_norm.unsqueeze(0)  # (1, N)

    for _ in range(epochs):
        model.train()
        tc_pred, m_pred, omega_pred = model(x)
        tc_val = tc_pred[0]
        m_val = m_pred[0]
        omega_val = omega_pred[0]

        # Build design matrix and solve linear params
        X = lppls_design_matrix(t_tensor, tc_val, m_val, omega_val)
        beta = solve_linear_torch(X, lp_tensor)

        # Reconstruction loss
        lp_hat = X @ beta
        recon_loss = ((lp_hat - lp_tensor) ** 2).mean()

        # Penalty for B >= 0 (should be negative for bubbles)
        b_penalty = torch.relu(beta[1]) * penalty_weight

        total_loss = recon_loss + b_penalty

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            with torch.no_grad():
                best_result = (
                    tc_val.item(),
                    m_val.item(),
                    omega_val.item(),
                    beta[0].item(),
                    beta[1].item(),
                    beta[2].item(),
                    beta[3].item(),
                    recon_loss.item(),
                )

    tc, m, omega, a, b, c1, c2, rss = best_result  # type: ignore[misc]

    is_bubble = passes_nielsen_filter(tc, m, omega, t2, b)
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + rss * 100)))

    return LPPLSNNResult(
        tc=tc, m=m, omega=omega,
        a=a, b=b, c1=c1, c2=c2,
        rss=rss,
        is_bubble=is_bubble,
        confidence=confidence,
        method="M-LNN",
    )
