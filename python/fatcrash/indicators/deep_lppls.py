"""Deep LPPLS — Physics-Informed LPPLS Neural Network (optional, requires torch).

Based on the P-LNN approach: the network predicts LPPLS nonlinear parameters
(tc, m, omega) from a price window, and the loss function penalizes deviation
from the LPPLS equation (physics-informed loss) in addition to reconstruction error.

Reference: Barza et al. (2023) "Deep Learning for Detecting Financial Bubbles"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DeepLPPLSResult:
    tc_pred: float
    m_pred: float
    omega_pred: float
    confidence: float
    reconstruction_loss: float
    physics_loss: float


def _lppls_reconstruct(t: "torch.Tensor", tc: "torch.Tensor", m: "torch.Tensor",
                        omega: "torch.Tensor", a: "torch.Tensor", b: "torch.Tensor",
                        c1: "torch.Tensor", c2: "torch.Tensor") -> "torch.Tensor":
    """Differentiable LPPLS reconstruction for physics-informed loss."""
    import torch
    dt = tc - t
    dt = torch.clamp(dt, min=1e-6)
    dt_m = dt.pow(m)
    log_dt = dt.log()
    return a + b * dt_m + dt_m * (c1 * (omega * log_dt).cos() + c2 * (omega * log_dt).sin())


def fit_deep_lppls(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    window_size: int = 120,
    epochs: int = 200,
    lr: float = 1e-3,
    physics_weight: float = 1.0,
) -> DeepLPPLSResult:
    """Fit P-LNN (Physics-Informed LPPLS Neural Network).

    The network predicts (tc, m, omega) from the price window.
    Linear params (A, B, C1, C2) are solved analytically via OLS
    for each predicted (tc, m, omega), making training more stable.

    Loss = reconstruction_MSE + physics_weight * LPPLS_residual_penalty
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("Deep LPPLS requires PyTorch. Install with: pip install fatcrash[deep]")

    class PLNN(nn.Module):
        """Physics-informed LPPLS neural network."""

        def __init__(self, input_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
            )
            # Separate heads for each parameter
            self.tc_head = nn.Linear(32, 1)
            self.m_head = nn.Linear(32, 1)
            self.omega_head = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.encoder(x)
            # tc: predicted as fraction beyond window end, in [0, 0.5] * t_range
            tc_raw = torch.sigmoid(self.tc_head(h)) * 0.5
            # m: in [0.1, 0.9] per Sornette constraints
            m_raw = torch.sigmoid(self.m_head(h)) * 0.8 + 0.1
            # omega: in [2, 25] per Sornette constraints
            omega_raw = torch.sigmoid(self.omega_head(h)) * 23.0 + 2.0
            return tc_raw, m_raw, omega_raw

    # Prepare data
    n = min(window_size, len(times))
    t = times[-n:].copy()
    lp = log_prices[-n:].copy()

    # Normalize time to [0, 1]
    t_start, t_end = t[0], t[-1]
    t_range = t_end - t_start
    t_norm = (t - t_start) / t_range
    lp_mean, lp_std = lp.mean(), lp.std() + 1e-8
    lp_norm = (lp - lp_mean) / lp_std

    t_tensor = torch.tensor(t_norm, dtype=torch.float32)
    lp_tensor = torch.tensor(lp_norm, dtype=torch.float32)
    x = torch.cat([t_tensor, lp_tensor]).unsqueeze(0)

    model = PLNN(input_dim=n * 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_params = None

    for epoch in range(epochs):
        model.train()
        tc_frac, m_pred, omega_pred = model(x)

        # tc in normalized time: slightly beyond 1.0
        tc_norm = 1.0 + tc_frac[0, 0]
        m_val = m_pred[0, 0]
        omega_val = omega_pred[0, 0]

        # Compute LPPLS basis functions for OLS
        dt = tc_norm - t_tensor
        dt = torch.clamp(dt, min=1e-6)
        dt_m = dt.pow(m_val)
        log_dt = dt.log()

        # Design matrix for OLS: [1, dt^m, dt^m*cos(w*log(dt)), dt^m*sin(w*log(dt))]
        ones = torch.ones_like(dt)
        cos_term = dt_m * (omega_val * log_dt).cos()
        sin_term = dt_m * (omega_val * log_dt).sin()
        X = torch.stack([ones, dt_m, cos_term, sin_term], dim=1)

        # OLS solve for linear params: beta = (X'X)^-1 X'y
        XtX = X.T @ X + 1e-6 * torch.eye(4)  # Ridge for stability
        Xty = X.T @ lp_tensor
        beta = torch.linalg.solve(XtX, Xty)

        # Reconstruction
        lp_hat = X @ beta
        recon_loss = ((lp_hat - lp_tensor) ** 2).mean()

        # Physics loss: penalize Sornette violations
        # B should be negative for bubbles
        b_penalty = torch.relu(beta[1]) * 10.0
        # Damping ratio constraint
        c_amp = (beta[2] ** 2 + beta[3] ** 2).sqrt()
        damping = (m_val * beta[1]).abs() / (omega_val * c_amp + 1e-8)
        damping_penalty = torch.relu(damping - 1.0)

        physics_loss = b_penalty + damping_penalty
        total_loss = recon_loss + physics_weight * physics_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = (
                tc_norm.item(),
                m_val.item(),
                omega_val.item(),
                recon_loss.item(),
                physics_loss.item(),
            )

    tc_n, m_n, omega_n, recon_l, phys_l = best_params
    tc_actual = t_start + tc_n * t_range

    # Confidence: inverse of normalized reconstruction loss
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + recon_l * 100)))

    return DeepLPPLSResult(
        tc_pred=tc_actual,
        m_pred=m_n,
        omega_pred=omega_n,
        confidence=confidence,
        reconstruction_loss=recon_l,
        physics_loss=phys_l,
    )
