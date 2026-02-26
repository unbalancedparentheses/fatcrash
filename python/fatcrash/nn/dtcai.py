"""DTCAI — Distance-to-Crash AI reliability classifier.

Trains a classifier (ANN, Random Forest, or Logistic Regression) on LPPLS
parameters to predict whether an LPPLS fit is "reliable" (tc within 10 days
of an actual crash). The DTCAI metric combines this reliability score with
the Distance-to-Crash (DTC) ratio.

Reference: Lee, G., Jeong, M., Park, T. & Ahn, K. (2025).
"More Than Ex-Post Fitting: LPPL and Its AI-Based Classification."
Humanities and Social Sciences Communications, 12, 236.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fatcrash._core import lppls_fit
from fatcrash.nn.dtcai_data import DTCAIDataset, _extract_features

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@dataclass
class DTCAIResult:
    """Output from DTCAI prediction."""

    dtc: float  # distance-to-crash [0, 1]
    reliability: float  # AI reliability score [0, 1]
    dtcai: float  # dtc * reliability [0, 1]
    lppls_params: tuple[float, ...]  # (A, B, C, tc, phi, omega, beta)
    classifier: str  # "ANN", "RF", or "LogReg"


@dataclass
class DTCAIModel:
    """Wrapper for a trained DTCAI classifier."""

    model: Any  # nn.Module, RandomForestClassifier, or LogisticRegression
    model_type: str  # "ANN", "RF", or "LogReg"
    feature_mean: npt.NDArray[np.float64]  # for standardization
    feature_std: npt.NDArray[np.float64]  # for standardization


# ── ANN (PyTorch) ──────────────────────────────────────────


def _build_ann() -> Any:
    """Build the ANN classifier: Input(7) -> 256 -> 128 -> 64 -> 1."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(7, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def _random_oversample(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Simple random oversampling of the minority class."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    max_count = counts.max()
    X_parts = [X]
    y_parts = [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            deficit = max_count - cnt
            indices = np.where(y == cls)[0]
            extra_idx = rng.choice(indices, size=deficit, replace=True)
            X_parts.append(X[extra_idx])
            y_parts.append(y[extra_idx])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def train_dtcai(
    dataset: DTCAIDataset,
    model_type: str = "ANN",
    oversample: bool = True,
    lr: float = 0.01,
    batch_size: int = 512,
    epochs: int = 50,
    val_split: float = 0.2,
    seed: int = 42,
) -> DTCAIModel:
    """Train a DTCAI reliability classifier.

    Args:
        dataset: DTCAIDataset with X (n, 7) and y (n,) arrays.
        model_type: "ANN" (PyTorch), "RF" (sklearn), or "LogReg" (sklearn).
        oversample: Apply random oversampling to balance classes.
        lr: Learning rate (ANN only).
        batch_size: Batch size (ANN only).
        epochs: Training epochs (ANN only).
        val_split: Validation split fraction.
        seed: Random seed.

    Returns:
        Trained DTCAIModel.
    """
    rng = np.random.default_rng(seed)

    X = dataset.X.copy()
    y = dataset.y.copy()

    # Standardize features
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)
    X = (X - feature_mean) / feature_std

    # Train/val split
    n = len(X)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if oversample:
        X_train, y_train = _random_oversample(X_train, y_train, rng)

    if model_type == "ANN":
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "ANN classifier requires PyTorch. Install with: pip install fatcrash[deep]"
            )
        model = _train_ann(X_train, y_train, lr, batch_size, epochs, seed)
    elif model_type == "RF":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "RF classifier requires scikit-learn. Install with: pip install fatcrash[deep]"
            )
        model = _train_rf(X_train, y_train, seed)
    elif model_type == "LogReg":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "LogReg classifier requires scikit-learn. Install with: pip install fatcrash[deep]"
            )
        model = _train_logreg(X_train, y_train, seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'ANN', 'RF', or 'LogReg'.")

    return DTCAIModel(
        model=model,
        model_type=model_type,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _train_ann(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> Any:
    """Train ANN classifier."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    model = _build_ann()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    n = len(X_t)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(X_t[idx]).squeeze(-1)
            loss = loss_fn(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _train_rf(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    seed: int,
) -> Any:
    """Train Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def _train_logreg(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    seed: int,
) -> Any:
    """Train Logistic Regression classifier."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Prediction ─────────────────────────────────────────────


def compute_dtc(t1: float, t2: float, tc: float) -> float:
    """Compute Distance-to-Crash ratio.

    DTC = (t2 - t1) / (tc - t1)

    Args:
        t1: Window start time.
        t2: Window end time.
        tc: Predicted critical time.

    Returns:
        DTC in [0, 1]. Returns 1.0 if tc <= t1.
    """
    denom = tc - t1
    if denom <= 0:
        return 1.0
    return float(np.clip((t2 - t1) / denom, 0.0, 1.0))


def _predict_reliability(model: DTCAIModel, features: npt.NDArray[np.float64]) -> float:
    """Get reliability probability from the trained classifier."""
    # Standardize
    x = (features - model.feature_mean) / model.feature_std
    x = x.reshape(1, -1)

    if model.model_type == "ANN":
        import torch

        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            prob = model.model(x_t).squeeze().item()
        return float(prob)
    else:
        # sklearn: predict_proba returns [[p0, p1]]
        proba = model.model.predict_proba(x)
        return float(proba[0, 1])


def predict_dtcai(
    model: DTCAIModel,
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    t1: int | None = None,
    t2: int | None = None,
    seed: int = 42,
) -> DTCAIResult:
    """Compute DTCAI score for a price window.

    Pipeline: fit LPPLS -> extract 7 params -> classifier predicts
    reliability -> compute DTC -> DTCAI = DTC * P.

    Args:
        model: Trained DTCAIModel.
        times: Observation time indices.
        log_prices: Log prices corresponding to times.
        t1: Window start index (default: first element of times).
        t2: Window end index (default: last element of times).
        seed: Seed for LPPLS fit.

    Returns:
        DTCAIResult with dtc, reliability, dtcai, and parameters.
    """
    if t1 is None:
        t1 = int(times[0])
    if t2 is None:
        t2 = int(times[-1])

    # Fit LPPLS
    result = lppls_fit(times, log_prices, seed=seed)
    features = _extract_features(result)
    tc = result[0]

    # Compute components
    dtc = compute_dtc(float(t1), float(t2), tc)
    reliability = _predict_reliability(model, features)
    dtcai = dtc * reliability

    return DTCAIResult(
        dtc=dtc,
        reliability=reliability,
        dtcai=dtcai,
        lppls_params=tuple(features.tolist()),
        classifier=model.model_type,
    )


# ── Save/Load ──────────────────────────────────────────────


def save_dtcai(model: DTCAIModel, path: str | Path) -> None:
    """Save a trained DTCAI model to disk.

    ANN models are saved as PyTorch state dicts + metadata.
    Sklearn models are saved via pickle + metadata.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "model_type": model.model_type,
        "feature_mean": model.feature_mean.tolist(),
        "feature_std": model.feature_std.tolist(),
    }
    (path / "meta.json").write_text(json.dumps(meta))

    if model.model_type == "ANN":
        import torch

        torch.save(model.model.state_dict(), str(path / "model.pt"))
    else:
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(model.model, f)


def load_dtcai(path: str | Path) -> DTCAIModel:
    """Load a trained DTCAI model from disk."""
    path = Path(path)

    with open(path / "meta.json") as f:
        meta = json.loads(f.read())

    model_type = meta["model_type"]
    feature_mean = np.array(meta["feature_mean"], dtype=np.float64)
    feature_std = np.array(meta["feature_std"], dtype=np.float64)

    if model_type == "ANN":
        import torch

        ann = _build_ann()
        ann.load_state_dict(torch.load(str(path / "model.pt"), weights_only=True))
        ann.eval()
        model_obj = ann
    else:
        with open(path / "model.pkl", "rb") as f:
            model_obj = pickle.load(f)  # noqa: S301

    return DTCAIModel(
        model=model_obj,
        model_type=model_type,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
