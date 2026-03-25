"""Click CLI for the trade simulator."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import click

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.engine import run_simulation
from trade_simulator.plotting import build_chart
from trade_simulator.report import generate_summary, print_summary
from trade_simulator.strategy import Strategy


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Trade Simulator — Backtesting with Jup.ag OHLCV data."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--feed", required=True, type=click.Choice(["SOLUSD", "BTCUSD", "ETHUSD"], case_sensitive=False), help="Trading pair feed.")
@click.option("--type", "candle_type", required=True, type=click.Choice(["1", "5", "15", "1h", "1D"]), help="Candle timescale.")
@click.option("--range", "time_range", required=True, help="Time range, e.g. '1d', '7d', '30d', '6m', '1y'.")
@click.option("--strategy", "strategy_file", type=click.Path(exists=True), default=None, help="Path to a JSON strategy file.")
@click.option("--entry-drop", type=float, default=None, help="Simple mode: entry drop percentage.")
@click.option("--entry-window", default="30m", help="Simple mode: lookback window (e.g. '30m', '1h').")
@click.option("--take-profit", type=float, default=None, help="Simple mode: take profit percentage.")
@click.option("--stop-loss", type=float, default=None, help="Simple mode: stop loss percentage.")
@click.option("--capital", type=float, default=1000.0, help="Initial capital in USD.")
@click.option("--fee", type=float, default=0.1, help="Trading fee percentage.")
@click.option("--max-positions", type=int, default=3, help="Max concurrent open positions.")
@click.option("--position-size", type=float, default=100.0, help="Position size in USD (fixed mode).")
@click.option("--position-sizing", type=click.Choice(["fixed", "percentage"]), default="fixed", help="Position sizing method.")
@click.option("--output-dir", type=click.Path(), default="output", help="Output directory for results.")
def run(
    feed: str,
    candle_type: str,
    time_range: str,
    strategy_file: str | None,
    entry_drop: float | None,
    entry_window: str,
    take_profit: float | None,
    stop_loss: float | None,
    capital: float,
    fee: float,
    max_positions: int,
    position_size: float,
    position_sizing: str,
    output_dir: str,
) -> None:
    """Run a trading simulation."""
    # Build or load strategy
    if strategy_file:
        click.echo(f"Loading strategy from {strategy_file}...")
        strategy = Strategy.from_file(strategy_file)
    elif entry_drop is not None and take_profit is not None:
        strategy = Strategy.from_simple_params(
            entry_drop=entry_drop,
            entry_window=entry_window,
            take_profit=take_profit,
            stop_loss=stop_loss,
            capital=capital,
            fee_pct=fee,
            max_positions=max_positions,
            position_size=position_size,
            position_sizing_type=position_sizing,
        )
    else:
        raise click.UsageError(
            "Provide either --strategy FILE or both --entry-drop and --take-profit."
        )

    # Parse time range and fetch data
    feed = validate_feed(feed)
    candle_type = validate_type(candle_type)
    from_ms, till_ms = parse_time_range(time_range)

    click.echo(f"Fetching {feed} {candle_type} candles for {time_range}...")

    def _progress(fetched: int, chunk: int) -> None:
        click.echo(f"  chunk {chunk}: {fetched} candles so far...")

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=_progress)
    click.echo(f"Received {len(candles)} candles.")

    # Run simulation
    click.echo("Running simulation...")
    result = run_simulation(candles, strategy, candle_type, feed, time_range)

    # Generate outputs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)

    chart_path = out_dir / f"{feed}_{timestamp}_chart.html"
    summary_path = out_dir / f"{feed}_{timestamp}_summary.json"

    click.echo("Generating chart...")
    build_chart(candles, result, chart_path)
    click.echo(f"Chart saved to {chart_path}")

    click.echo("Generating summary...")
    generate_summary(result, summary_path)
    click.echo(f"Summary saved to {summary_path}")

    # Print console summary
    print_summary(result)
