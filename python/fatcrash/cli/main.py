"""CLI interface for fatcrash."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="fatcrash", help="Crash detection via fat-tail statistics.")
console = Console()


@app.command()
def detect(
    asset: str = typer.Option("BTC", help="Asset symbol (BTC, GOLD, SPX, etc.)"),
    source: str = typer.Option("coingecko", help="Data source: coingecko, ccxt, yahoo, csv"),
    days: int = typer.Option(365, help="Number of days of history"),
    csv_path: str | None = typer.Option(None, help="Path to CSV file (if source=csv)"),
) -> None:
    """Run crash detection on an asset."""
    import numpy as np
    from fatcrash.data import ingest, transforms

    console.print(f"[bold cyan]fatcrash[/] — analyzing {asset}...\n")

    # Load data
    if source == "csv" and csv_path:
        df = ingest.from_csv(csv_path)
    elif source == "coingecko":
        coin_map = {"BTC": "bitcoin", "ETH": "ethereum", "GOLD": "gold"}
        coin_id = coin_map.get(asset.upper(), asset.lower())
        df = ingest.from_coingecko(coin_id=coin_id, days=days)
    elif source == "yahoo":
        ticker_map = {"BTC": "BTC-USD", "GOLD": "GC=F", "SPX": "^GSPC", "SP500": "^GSPC"}
        ticker = ticker_map.get(asset.upper(), asset)
        df = ingest.from_yahoo(ticker=ticker)
    elif source == "ccxt":
        df = ingest.from_ccxt(symbol=f"{asset}/USDT", limit=days)
    else:
        console.print(f"[red]Unknown source: {source}[/]")
        raise typer.Exit(1)

    returns = transforms.log_returns(df)
    log_p = transforms.log_prices(df)
    times = transforms.time_index(df)

    console.print(f"Loaded {len(df)} data points\n")

    # Run indicators
    table = Table(title=f"Crash Detection Report — {asset}")
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Signal", justify="right")
    table.add_column("Status", style="bold")

    # Tail analysis
    try:
        from fatcrash.indicators.tail_indicator import estimate_tail_index, estimate_kappa

        tail = estimate_tail_index(returns)
        table.add_row(
            "Hill α",
            f"{tail.alpha:.2f}",
            "Fat tail" if tail.is_fat_tail else "Normal",
            "[red]!!![/]" if tail.alpha < 2 else "[yellow]![/]" if tail.is_fat_tail else "[green]OK[/]",
        )

        kappa = estimate_kappa(returns)
        table.add_row(
            "Kappa",
            f"{kappa.kappa:.3f}",
            f"vs {kappa.gaussian_benchmark:.3f}",
            "[red]!!![/]" if kappa.is_fat_tail else "[green]OK[/]",
        )
    except Exception as e:
        table.add_row("Tail", "error", str(e), "[red]ERR[/]")

    # EVT
    try:
        from fatcrash.indicators.evt_indicator import compute_var_es

        risk = compute_var_es(returns)
        table.add_row("VaR 99%", f"{risk.var:.4f}", "", "")
        table.add_row("ES 99%", f"{risk.es:.4f}", "", "")

        # Check if latest return exceeds VaR
        latest_loss = -returns[-1]
        if latest_loss > risk.var:
            table.add_row("VaR breach", "YES", f"loss={latest_loss:.4f}", "[red]!!![/]")
    except Exception as e:
        table.add_row("EVT", "error", str(e), "[red]ERR[/]")

    # LPPLS (skip if too few data points)
    try:
        from fatcrash.indicators.lppls_indicator import fit_lppls

        result = fit_lppls(times, log_p)
        status = "[red]BUBBLE[/]" if result.is_bubble else "[green]OK[/]"
        table.add_row("LPPLS bubble", "Yes" if result.is_bubble else "No", f"tc={result.tc:.1f}", status)
        table.add_row("LPPLS m", f"{result.m:.2f}", f"ω={result.omega:.1f}", "")
    except Exception as e:
        table.add_row("LPPLS", "error", str(e), "[red]ERR[/]")

    console.print(table)


@app.command()
def backtest(
    asset: str = typer.Option("BTC", help="Asset symbol"),
    start: str = typer.Option("2017-01-01", help="Start date"),
    end: str = typer.Option("2018-06-01", help="End date"),
    source: str = typer.Option("yahoo", help="Data source"),
) -> None:
    """Run historical backtest."""
    console.print(f"[bold cyan]fatcrash backtest[/] — {asset} from {start} to {end}")
    console.print("[yellow]Backtest not yet implemented. Coming in Phase 5.[/]")


@app.command()
def plot(
    asset: str = typer.Option("BTC", help="Asset symbol"),
    indicator: str = typer.Option("lppls", help="Indicator to plot: lppls, hill, kappa, evt"),
) -> None:
    """Generate visualization plots."""
    console.print(f"[bold cyan]fatcrash plot[/] — {indicator} for {asset}")
    console.print("[yellow]Plotting command not yet implemented. Coming in Phase 5.[/]")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
) -> None:
    """Start the FastAPI monitoring service."""
    import uvicorn
    from fatcrash.service.server import create_app

    app = create_app()
    console.print(f"[bold cyan]fatcrash serve[/] — starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
