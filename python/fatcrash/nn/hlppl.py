"""HLPPL — Hyped LPPL with dual-stream transformer for bubble detection.

Fuses price/LPPLS features with hype (sentiment/volume) features via two
parallel TransformerEncoder streams, then combines for crash probability.

Reference: Cao, Shao, Yan, Geman (2025) "HLPPL", arXiv:2510.10878

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fatcrash.nn import _require_torch

_require_torch()

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

from fatcrash.nn.sentiment import SentimentFeatures, compute_sentiment_proxy


@dataclass
class HLPPLResult:
    """Output from HLPPL model."""

    bubble_score: float  # [0, 1] probability of impending crash
    temporal_features: npt.NDArray[np.float32]  # encoded price features
    sentiment_features: npt.NDArray[np.float32]  # encoded hype features


class TemporalEncoder(nn.Module):
    """Stream 1: Price + LPPLS features -> TransformerEncoder.

    Architecture (per paper):
        Input projection -> TransformerEncoder(2 layers, 4 heads, d_model=64)
    """

    def __init__(self, input_dim: int = 5, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, d_model)"""
        h = self.input_proj(x)
        h = self.transformer(h)
        return h.mean(dim=1)  # Pool over sequence


class SentimentEncoder(nn.Module):
    """Stream 2: Hype features -> TransformerEncoder.

    Architecture (per paper):
        Input projection -> TransformerEncoder(1 layer, 2 heads, d_model=32)
    """

    def __init__(self, input_dim: int = 4, d_model: int = 32, nhead: int = 2, num_layers: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, d_model)"""
        h = self.input_proj(x)
        h = self.transformer(h)
        return h.mean(dim=1)


class HLPPLNetwork(nn.Module):
    """Dual-stream HLPPL model fusing temporal and sentiment encoders.

    Concatenates outputs from both streams, then projects to bubble score.
    """

    def __init__(
        self,
        temporal_input_dim: int = 5,
        sentiment_input_dim: int = 4,
        temporal_d: int = 64,
        sentiment_d: int = 32,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim=temporal_input_dim, d_model=temporal_d)
        self.sentiment_encoder = SentimentEncoder(input_dim=sentiment_input_dim, d_model=sentiment_d)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_d + sentiment_d, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        temporal_x: torch.Tensor,
        sentiment_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning bubble score and intermediate features.

        Args:
            temporal_x: (batch, seq_len, temporal_input_dim)
            sentiment_x: (batch, seq_len, sentiment_input_dim)

        Returns:
            (bubble_score, temporal_feats, sentiment_feats)
        """
        t_feats = self.temporal_encoder(temporal_x)
        s_feats = self.sentiment_encoder(sentiment_x)
        fused = torch.cat([t_feats, s_feats], dim=-1)
        score = self.classifier(fused).squeeze(-1)
        return score, t_feats, s_feats


def _prepare_features(
    df: pd.DataFrame,
    window: int = 60,
    close_col: str = "close",
    volume_col: str = "volume",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare temporal and sentiment feature tensors from OHLCV data.

    Temporal features (5): log_price, returns, volatility, volume_change, hype_index
    Sentiment features (4): volume_z, volume_momentum, abs_return_z, hype_index

    Returns:
        (temporal_tensor, sentiment_tensor) each of shape (1, window, dim)
    """
    close = df[close_col].values.astype(np.float64)
    log_price = np.log(np.maximum(close, 1e-10))

    # Normalize log prices
    lp_min, lp_max = log_price.min(), log_price.max()
    lp_denom = lp_max - lp_min if (lp_max - lp_min) > 1e-10 else 1.0
    log_price_norm = (log_price - lp_min) / lp_denom

    # Returns
    returns = np.diff(log_price, prepend=log_price[0])

    # Rolling volatility
    vol_series = pd.Series(returns).rolling(20, min_periods=1).std().values

    # Volume change
    volume = df[volume_col].values.astype(np.float64)
    vol_lag = pd.Series(volume).shift(1).bfill().values
    vol_lag = np.where(vol_lag < 1e-10, 1.0, vol_lag)
    vol_change = (volume - vol_lag) / vol_lag

    # Sentiment proxy
    sent = compute_sentiment_proxy(df, volume_col=volume_col, close_col=close_col)

    # Take last `window` observations
    n = min(window, len(df))
    sl = slice(-n, None)

    temporal = np.stack([
        log_price_norm[sl],
        returns[sl],
        vol_series[sl],
        vol_change[sl],
        sent.hype_index[sl],
    ], axis=-1).astype(np.float32)

    sentiment = np.stack([
        sent.volume_z[sl],
        sent.volume_momentum[sl],
        sent.abs_return_z[sl],
        sent.hype_index[sl],
    ], axis=-1).astype(np.float32)

    # Replace NaN/inf
    temporal = np.nan_to_num(temporal, nan=0.0, posinf=1.0, neginf=-1.0)
    sentiment = np.nan_to_num(sentiment, nan=0.0, posinf=1.0, neginf=-1.0)

    return (
        torch.from_numpy(temporal).unsqueeze(0),
        torch.from_numpy(sentiment).unsqueeze(0),
    )


def train_hlppl(
    train_data: list[pd.DataFrame],
    crash_labels: list[int],
    window: int = 60,
    epochs: int = 50,
    lr: float = 1e-3,
    close_col: str = "close",
    volume_col: str = "volume",
    seed: int = 42,
) -> HLPPLNetwork:
    """Train HLPPL on labeled data.

    Crash labels: 1 if >15% drawdown within 30 days, 0 otherwise.

    Args:
        train_data: list of OHLCV DataFrames (one per sample)
        crash_labels: binary labels [0, 1] for each sample
        window: sequence length for transformer input
        epochs: training epochs
        lr: learning rate
        close_col: column name for close prices
        volume_col: column name for volume
        seed: random seed

    Returns:
        Trained HLPPLNetwork
    """
    torch.manual_seed(seed)

    model = HLPPLNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Pre-compute features for all samples
    temporal_list = []
    sentiment_list = []
    for df in train_data:
        t_feat, s_feat = _prepare_features(df, window=window, close_col=close_col, volume_col=volume_col)
        temporal_list.append(t_feat)
        sentiment_list.append(s_feat)

    labels = torch.tensor(crash_labels, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for i in range(len(train_data)):
            score, _, _ = model(temporal_list[i], sentiment_list[i])
            loss = loss_fn(score, labels[i:i+1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


def predict_hlppl(
    model: HLPPLNetwork,
    df: pd.DataFrame,
    window: int = 60,
    close_col: str = "close",
    volume_col: str = "volume",
) -> HLPPLResult:
    """Run HLPPL inference on a single OHLCV DataFrame.

    Args:
        model: trained HLPPLNetwork
        df: OHLCV DataFrame with at least close and volume columns
        window: sequence length
        close_col: column name for close prices
        volume_col: column name for volume

    Returns:
        HLPPLResult with bubble probability and features
    """
    t_feat, s_feat = _prepare_features(df, window=window, close_col=close_col, volume_col=volume_col)

    model.eval()
    with torch.no_grad():
        score, t_enc, s_enc = model(t_feat, s_feat)

    return HLPPLResult(
        bubble_score=score[0].item(),
        temporal_features=t_enc[0].numpy(),
        sentiment_features=s_enc[0].numpy(),
    )


def save_hlppl(model: HLPPLNetwork, path: str | Path) -> None:
    """Save HLPPL model weights."""
    torch.save(model.state_dict(), str(path))


def load_hlppl(path: str | Path) -> HLPPLNetwork:
    """Load HLPPL model from disk."""
    model = HLPPLNetwork()
    model.load_state_dict(torch.load(str(path), weights_only=True))
    model.eval()
    return model
