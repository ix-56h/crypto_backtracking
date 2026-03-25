#!/usr/bin/env python3
"""Batch backtest: fetch data once, run all strategies, compare results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.engine import run_simulation
from trade_simulator.strategy import Strategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRATEGY_DIR = PROJECT_ROOT / "strategies"


def fetch_data(feed: str, candle_type: str, time_range: str):
    """Fetch OHLCV data once."""
    feed = validate_feed(feed)
    candle_type = validate_type(candle_type)
    from_ms, till_ms = parse_time_range(time_range)

    print(f"Fetching {feed} {candle_type}m candles for {time_range}...")

    def _progress(fetched: int, chunk: int):
        print(f"  chunk {chunk}: {fetched} candles...", flush=True)

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=_progress)
    print(f"Total: {len(candles)} candles")
    print(f"Period: {candles[0].time.date()} to {candles[-1].time.date()}")
    return candles


def find_strategies(strategy_dir: Path) -> list[Path]:
    """Find all .json strategy files, recursively if top-level strategies/ dir."""
    return sorted(strategy_dir.rglob("*.json"))


def _run_one(strategy_path: Path, strategy_dir: Path, candles, candle_type: str, feed: str, time_range: str):
    """Run a single strategy and return (name, result). Used by process pool."""
    rel = strategy_path.relative_to(strategy_dir)
    name = str(rel.with_suffix(""))
    strategy = Strategy.from_file(strategy_path)
    result = run_simulation(candles, strategy, candle_type, feed, time_range)
    return name, result


def run_all_strategies(candles, strategy_dir: Path, candle_type: str, feed: str, time_range: str, workers: int):
    """Run all strategy files in parallel and collect results."""
    strategy_files = find_strategies(strategy_dir)
    if not strategy_files:
        print(f"No strategy files found in {strategy_dir}")
        return []

    total = len(strategy_files)
    print(f"\nFound {total} strategies in {strategy_dir}")
    print(f"Running with {workers} worker process{'es' if workers > 1 else ''}...\n")

    results: list[tuple[str, object]] = []
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, sf, strategy_dir, candles, candle_type, feed, time_range): sf
            for sf in strategy_files
        }
        for i, future in enumerate(as_completed(futures), 1):
            name, result = future.result()
            results.append((name, result))

            wr_mark = "✓" if result.win_rate >= 60 else "✗"
            pnl_mark = "✓" if result.total_pnl > 0 else "✗"
            print(
                f"  [{i:>{len(str(total))}}/{total}] {name:<35} "
                f"trades={result.total_trades:<4} WR={result.win_rate:>5.1f}% {wr_mark}  "
                f"PnL=${result.total_pnl:>10,.2f} {pnl_mark}  "
                f"Sharpe={result.sharpe_ratio:>7.4f}  PF={result.profit_factor:>6.2f}"
            )

    elapsed = time.monotonic() - t0
    print(f"\nCompleted {total} backtests in {elapsed:.1f}s")

    # Sort by name for consistent table ordering
    results.sort(key=lambda x: x[0])
    return results


def print_comparison_table(results):
    """Print a side-by-side comparison table."""
    print(f"\n\n{'='*120}")
    print(f"{'STRATEGY COMPARISON TABLE':^120}")
    print(f"{'='*120}")

    header = (
        f"{'Strategy':<28} {'Trades':>6} {'WR%':>6} {'PnL$':>10} "
        f"{'PnL%':>8} {'Expect':>8} {'Sharpe':>7} {'PF':>6} "
        f"{'MaxDD%':>7} {'AvgHrs':>7} {'W-Str':>5} {'L-Str':>5} {'RecF':>6}"
    )
    print(header)
    print("-" * 120)

    for name, r in results:
        wr_flag = " ★" if r.win_rate >= 60 else ""
        pf = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
        print(
            f"{name:<28} {r.total_trades:>6} {r.win_rate:>5.1f}%{wr_flag} "
            f"{r.total_pnl:>10,.2f} {r.total_pnl_pct:>+7.2f}% "
            f"{r.expectancy:>8.4f} {r.sharpe_ratio:>7.4f} {pf:>6} "
            f"{r.max_drawdown_pct:>6.2f}% {r.avg_duration_minutes / 60:>7.1f} "
            f"{r.max_consecutive_wins:>5} {r.max_consecutive_losses:>5} "
            f"{r.recovery_factor:>6.2f}"
        )

    print("-" * 120)

    # Highlight best performers
    if results:
        best_wr = max(results, key=lambda x: x[1].win_rate)
        best_pnl = max(results, key=lambda x: x[1].total_pnl)
        best_sharpe = max(results, key=lambda x: x[1].sharpe_ratio)
        best_pf = max(results, key=lambda x: x[1].profit_factor if x[1].profit_factor != float("inf") else 0)

        print(f"\nBest Win Rate:    {best_wr[0]} ({best_wr[1].win_rate:.1f}%)")
        print(f"Best PnL:         {best_pnl[0]} (${best_pnl[1].total_pnl:,.2f})")
        print(f"Best Sharpe:      {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.4f})")
        print(f"Best Profit Factor: {best_pf[0]} ({best_pf[1].profit_factor:.2f})")

        winners = [n for n, r in results if r.win_rate >= 60]
        if winners:
            print(f"\n★ Strategies with WR >= 60%: {', '.join(winners)}")
        else:
            print("\n⚠ No strategy reached 60% win rate yet. Iteration needed.")

    # Save results to JSON
    output_path = Path(__file__).resolve().parent.parent / "output" / "batch_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = []
    for name, r in results:
        comparison.append({
            "strategy": name,
            "trades": r.total_trades,
            "win_rate": round(r.win_rate, 2),
            "total_pnl": round(r.total_pnl, 2),
            "total_pnl_pct": round(r.total_pnl_pct, 2),
            "expectancy": round(r.expectancy, 4),
            "sharpe_ratio": round(r.sharpe_ratio, 4),
            "profit_factor": round(r.profit_factor, 4) if r.profit_factor != float("inf") else "inf",
            "max_drawdown_pct": round(r.max_drawdown_pct, 2),
            "avg_duration_hours": round(r.avg_duration_minutes / 60, 1),
            "max_consecutive_wins": r.max_consecutive_wins,
            "max_consecutive_losses": r.max_consecutive_losses,
            "recovery_factor": round(r.recovery_factor, 4),
            "per_rule": {
                rule_id: {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
                for rule_id, stats in r.per_rule.items()
            },
        })
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch backtest: fetch data once, run all strategies, compare results."
    )
    parser.add_argument("--feed", default="BTCUSD", choices=["BTCUSD", "SOLUSD", "ETHUSD"],
                        help="Trading pair (default: BTCUSD)")
    parser.add_argument("--type", dest="candle_type", default="15", choices=["5", "15"],
                        help="Candle width in minutes (default: 15)")
    parser.add_argument("--range", dest="time_range", default="1y",
                        help="Data range (default: 1y)")
    parser.add_argument("--strategies", default=None,
                        help="Path to strategy directory (default: strategies/ recursive)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help=f"Parallel worker processes (default: {os.cpu_count() or 4})")
    args = parser.parse_args()

    if args.strategies:
        strategy_dir = Path(args.strategies).resolve()
    else:
        strategy_dir = DEFAULT_STRATEGY_DIR

    if not strategy_dir.is_dir():
        print(f"Error: strategy directory not found: {strategy_dir}")
        sys.exit(1)

    candles = fetch_data(args.feed, args.candle_type, args.time_range)
    results = run_all_strategies(candles, strategy_dir, args.candle_type, args.feed, args.time_range, args.workers)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
