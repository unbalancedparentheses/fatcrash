"""CLI interface for fatcrash."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(name="fatcrash", help="Crash detection via fat-tail statistics.")
console = Console()

TICKER_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GOLD": "GC=F",
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "SILVER": "SI=F",
    "OIL": "CL=F",
}

COINGECKO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
}


def _load_data(asset: str, source: str, days: int = 365, csv_path: str | None = None, start: str | None = None, end: str | None = None):
    from fatcrash.data import ingest

    if source == "csv" and csv_path:
        return ingest.from_csv(csv_path)
    elif source == "sample":
        return ingest.from_sample(asset)
    elif source == "coingecko":
        coin_id = COINGECKO_MAP.get(asset.upper(), asset.lower())
        return ingest.from_coingecko(coin_id=coin_id, days=days)
    elif source == "yahoo":
        ticker = TICKER_MAP.get(asset.upper(), asset)
        return ingest.from_yahoo(ticker=ticker, start=start or "2015-01-01", end=end)
    elif source == "ccxt":
        return ingest.from_ccxt(symbol=f"{asset}/USDT", limit=days)
    else:
        console.print(f"[red]Unknown source: {source}. Use: yahoo, sample, coingecko, ccxt, csv[/]")
        raise typer.Exit(1)


@app.command()
def detect(
    asset: str = typer.Option("BTC", help="Asset symbol (BTC, GOLD, SPX, etc.)"),
    source: str = typer.Option("yahoo", help="Data source: yahoo, coingecko, ccxt, csv"),
    days: int = typer.Option(365, help="Number of days of history"),
    csv_path: str | None = typer.Option(None, help="Path to CSV file (if source=csv)"),
) -> None:
    """Run crash detection on an asset."""
    import numpy as np
    from fatcrash.data import transforms

    console.print(f"[bold cyan]fatcrash[/] — analyzing {asset}...\n")

    df = _load_data(asset, source, days, csv_path)
    returns = transforms.log_returns(df)
    log_p = transforms.log_prices(df)
    times = transforms.time_index(df)

    console.print(f"Loaded {len(df)} data points\n")

    table = Table(title=f"Crash Detection Report — {asset}")
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Signal", justify="right")
    table.add_column("Status", style="bold")

    # Tail analysis
    try:
        from fatcrash.indicators.tail_indicator import estimate_tail_index, estimate_kappa, estimate_taleb_kappa

        tail = estimate_tail_index(returns)
        table.add_row(
            "Hill alpha",
            f"{tail.alpha:.2f}",
            "Fat tail" if tail.is_fat_tail else "Normal",
            "[red]!!![/]" if tail.alpha < 2 else "[yellow]![/]" if tail.is_fat_tail else "[green]OK[/]",
        )

        kappa = estimate_kappa(returns)
        table.add_row(
            "Max-Stab Kappa",
            f"{kappa.kappa:.3f}",
            f"vs {kappa.gaussian_benchmark:.3f}",
            "[red]!!![/]" if kappa.is_fat_tail else "[green]OK[/]",
        )

        tk = estimate_taleb_kappa(returns)
        table.add_row(
            "Taleb Kappa",
            f"{tk.kappa:.3f}",
            f"vs {tk.gaussian_benchmark:.3f}",
            "[red]!!![/]" if tk.is_fat_tail else "[green]OK[/]",
        )
    except Exception as e:
        table.add_row("Tail", "error", str(e), "[red]ERR[/]")

    # EVT
    try:
        from fatcrash.indicators.evt_indicator import compute_var_es

        risk = compute_var_es(returns)
        table.add_row("VaR 99%", f"{risk.var:.4f}", "", "")
        table.add_row("ES 99%", f"{risk.es:.4f}", "", "")

        latest_loss = -returns[-1]
        if latest_loss > risk.var:
            table.add_row("VaR breach", "YES", f"loss={latest_loss:.4f}", "[red]!!![/]")
    except Exception as e:
        table.add_row("EVT", "error", str(e), "[red]ERR[/]")

    # LPPLS
    try:
        from fatcrash.indicators.lppls_indicator import fit_lppls

        result = fit_lppls(times, log_p)
        status = "[red]BUBBLE[/]" if result.is_bubble else "[green]OK[/]"
        table.add_row("LPPLS bubble", "Yes" if result.is_bubble else "No", f"tc={result.tc:.1f}", status)
        table.add_row("LPPLS m", f"{result.m:.2f}", f"w={result.omega:.1f}", "")
    except Exception as e:
        table.add_row("LPPLS", "error", str(e), "[red]ERR[/]")

    console.print(table)


@app.command()
def backtest(
    asset: str = typer.Option("BTC", help="Asset symbol"),
    start: str = typer.Option("2017-01-01", help="Start date"),
    end: str = typer.Option("2018-06-01", help="End date"),
    source: str = typer.Option("yahoo", help="Data source"),
    window: int = typer.Option(252, help="Rolling window size"),
) -> None:
    """Run historical backtest against known crash periods."""
    import numpy as np
    from fatcrash.data import transforms
    from fatcrash.indicators.tail_indicator import rolling_tail_index, rolling_kappa, rolling_taleb_kappa
    from fatcrash.indicators.evt_indicator import rolling_var_es
    from fatcrash.aggregator.signals import (
        aggregate_signals,
        hill_thinning_signal,
        kappa_regime_signal,
        taleb_kappa_signal,
        var_exceedance_signal,
    )

    console.print(f"[bold cyan]fatcrash backtest[/] — {asset} from {start} to {end}\n")

    df = _load_data(asset, source, start=start, end=end)
    returns = transforms.log_returns(df)

    console.print(f"Loaded {len(df)} data points")

    # Rolling indicators
    console.print("Computing rolling Hill estimator...")
    hill_alpha = rolling_tail_index(returns, window=window)

    console.print("Computing rolling max-stability kappa...")
    kappa_arr, kappa_bench = rolling_kappa(returns, window=window)

    console.print("Computing rolling Taleb kappa...")
    tk_arr, tk_bench = rolling_taleb_kappa(returns, window=window)

    console.print("Computing rolling VaR/ES...")
    var_arr, es_arr = rolling_var_es(returns, window=window)

    # Build aggregate signals
    console.print("Aggregating signals...\n")
    n = len(returns)
    probabilities = np.full(n, np.nan)

    for i in range(window + 1, n):
        components = {}

        if not np.isnan(hill_alpha[i]) and not np.isnan(hill_alpha[i - 1]):
            components["hill_thinning"] = hill_thinning_signal(hill_alpha[i], hill_alpha[i - 1])

        if not np.isnan(kappa_arr[i]):
            components["kappa_regime"] = kappa_regime_signal(kappa_arr[i], kappa_bench)

        if not np.isnan(tk_arr[i]):
            components["taleb_kappa"] = taleb_kappa_signal(tk_arr[i], tk_bench)

        if not np.isnan(var_arr[i]):
            components["gpd_var_exceedance"] = var_exceedance_signal(returns[i], var_arr[i])

        if components:
            signal = aggregate_signals(components)
            probabilities[i] = signal.probability

    # Summary stats
    valid = probabilities[~np.isnan(probabilities)]
    if len(valid) > 0:
        table = Table(title=f"Backtest Summary — {asset} ({start} to {end})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Mean crash probability", f"{valid.mean():.3f}")
        table.add_row("Max crash probability", f"{valid.max():.3f}")
        table.add_row("Days above 0.5", f"{(valid > 0.5).sum()}")
        table.add_row("Days above 0.7", f"{(valid > 0.7).sum()}")
        console.print(table)
    else:
        console.print("[yellow]Not enough data for backtest.[/]")


@app.command()
def plot(
    asset: str = typer.Option("BTC", help="Asset symbol"),
    indicator: str = typer.Option("hill", help="Indicator: hill, kappa, evt, qq"),
    source: str = typer.Option("yahoo", help="Data source"),
    days: int = typer.Option(365, help="Number of days"),
) -> None:
    """Generate visualization plots."""
    from fatcrash.data import transforms

    console.print(f"[bold cyan]fatcrash plot[/] — {indicator} for {asset}\n")

    df = _load_data(asset, source, days)
    returns = transforms.log_returns(df)

    if indicator == "hill":
        from fatcrash.viz.tail_dashboard import hill_plot
        hill_plot(returns, title=f"Hill Plot — {asset}")

    elif indicator == "kappa":
        from fatcrash.indicators.tail_indicator import rolling_kappa
        from fatcrash.viz.tail_dashboard import kappa_evolution
        kappa_arr, bench = rolling_kappa(returns, window=252)
        kappa_evolution(df.index[1:], kappa_arr, bench, title=f"Kappa — {asset}")

    elif indicator == "evt":
        from fatcrash.indicators.evt_indicator import rolling_var_es
        from fatcrash.viz.evt_plots import rolling_var_es_plot
        var_arr, es_arr = rolling_var_es(returns, window=252)
        rolling_var_es_plot(df.index[1:], returns, var_arr, es_arr, title=f"VaR/ES — {asset}")

    elif indicator == "qq":
        from fatcrash.viz.tail_dashboard import qq_plot
        qq_plot(returns, title=f"QQ Plot — {asset}")

    else:
        console.print(f"[red]Unknown indicator: {indicator}. Choose from: hill, kappa, evt, qq[/]")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
) -> None:
    """Start the FastAPI monitoring service."""
    import uvicorn
    from fatcrash.service.server import create_app

    api = create_app()
    console.print(f"[bold cyan]fatcrash serve[/] — starting on {host}:{port}")
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
