"""FastAPI service for crash monitoring and webhook alerts."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    asset: str = "BTC"
    source: str = "coingecko"
    days: int = 365


class DetectionResponse(BaseModel):
    asset: str
    probability: float
    level: str
    horizon_days: float
    components: dict[str, float]


class WebhookConfig(BaseModel):
    url: str
    threshold: float = 0.5
    assets: list[str] = ["BTC"]


def create_app() -> FastAPI:
    app = FastAPI(title="fatcrash", description="Crash detection via fat-tail statistics")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/detect", response_model=DetectionResponse)
    async def detect(req: DetectionRequest):
        """Run crash detection on an asset."""
        import numpy as np
        from fatcrash.data import ingest, transforms
        from fatcrash.indicators.tail_indicator import estimate_tail_index, estimate_kappa
        from fatcrash.indicators.evt_indicator import compute_var_es
        from fatcrash.aggregator.signals import (
            aggregate_signals,
            kappa_regime_signal,
            var_exceedance_signal,
        )

        # Load data
        if req.source == "coingecko":
            coin_map = {"BTC": "bitcoin", "ETH": "ethereum"}
            coin_id = coin_map.get(req.asset.upper(), req.asset.lower())
            df = ingest.from_coingecko(coin_id=coin_id, days=req.days)
        else:
            df = ingest.from_yahoo(ticker=f"{req.asset}-USD")

        returns = transforms.log_returns(df)
        components = {}

        try:
            tail = estimate_tail_index(returns)
            kappa = estimate_kappa(returns)
            components["hill_thinning"] = max(0, (4.0 - tail.alpha) / 4.0)
            components["kappa_regime"] = kappa_regime_signal(kappa.kappa, kappa.gaussian_benchmark)
        except Exception:
            pass

        try:
            risk = compute_var_es(returns)
            latest_loss = -returns[-1]
            components["gpd_var_exceedance"] = var_exceedance_signal(returns[-1], risk.var)
        except Exception:
            pass

        signal = aggregate_signals(components)
        return DetectionResponse(
            asset=req.asset,
            probability=signal.probability,
            level=signal.level,
            horizon_days=signal.horizon_days,
            components=signal.components,
        )

    return app
