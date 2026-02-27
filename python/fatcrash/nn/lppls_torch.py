"""Differentiable LPPLS utilities for PyTorch-based methods.

Provides a differentiable design matrix, linear solver, normalization,
and filtering — shared by M-LNN and P-LNN.

Reference: Nielsen, Sornette, Raissi (2024) "Deep Learning for LPPLS", arXiv:2405.12803

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from fatcrash.nn import _require_torch

_require_torch()

import torch  # noqa: E402


@dataclass
class LPPLSNNResult:
    """Result from any neural-network LPPLS method."""

    tc: float
    m: float
    omega: float
    a: float
    b: float
    c1: float
    c2: float
    rss: float
    is_bubble: bool
    confidence: float
    method: str  # "M-LNN" or "P-LNN"


def lppls_design_matrix(
    t: torch.Tensor,
    tc: torch.Tensor,
    m: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Build differentiable LPPLS design matrix.

    Returns (N, 4) tensor: [1, (tc-t)^m, (tc-t)^m*cos(w*ln(tc-t)), (tc-t)^m*sin(w*ln(tc-t))]
    """
    dt = tc - t
    dt = torch.clamp(dt, min=1e-6)
    dt_m = dt.pow(m)
    log_dt = dt.log()
    ones = torch.ones_like(dt)
    cos_term = dt_m * (omega * log_dt).cos()
    sin_term = dt_m * (omega * log_dt).sin()
    return torch.stack([ones, dt_m, cos_term, sin_term], dim=-1)


def solve_linear_torch(
    X: torch.Tensor,
    y: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Differentiable OLS: beta = (X'X + rI)^-1 X'y.

    Args:
        X: (N, 4) design matrix
        y: (N,) target vector
        ridge: regularization for numerical stability

    Returns:
        (4,) parameter vector [A, B, C1, C2]
    """
    XtX = X.T @ X + ridge * torch.eye(4, device=X.device, dtype=X.dtype)
    Xty = X.T @ y
    return torch.linalg.solve(XtX, Xty)


def min_max_normalize(prices: torch.Tensor) -> torch.Tensor:
    """Per-series [0,1] min-max normalization (paper spec, NOT z-score)."""
    p_min = prices.min()
    p_max = prices.max()
    denom = p_max - p_min
    if denom < 1e-10:
        return torch.zeros_like(prices)
    return (prices - p_min) / denom


def passes_nielsen_filter(
    tc: float,
    m: float,
    omega: float,
    t2: float,
    b: float,
) -> bool:
    """Nielsen et al. (2024) qualifying filter for LPPLS fits.

    Constraints:
    - m in [0.1, 1.0]
    - omega in [6, 13]
    - b < 0 (super-exponential growth)
    - tc > t2 (crash in the future)
    """
    return (
        0.1 <= m <= 1.0
        and 6.0 <= omega <= 13.0
        and b < 0
        and tc > t2
    )
