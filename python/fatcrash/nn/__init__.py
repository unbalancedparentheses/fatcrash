"""Neural network methods for LPPLS bubble detection.

Implements Deep LPPLS (Nielsen, Sornette, Raissi 2024) and HLPPL (Cao et al. 2025).

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

import importlib

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Neural network modules require PyTorch. Install with: pip install fatcrash[deep]"
        )
