#!/usr/bin/env python3
"""Batch backtest: fetch data once, run all strategies, compare results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.engine import run_simulation
from trade_simulator.strategy import Strategy


FEED = "BTCUSD"
CANDLE_TYPE = "15"
TIME_RANGE = "1y"

# Default to champions; override with env or arg
import os
_dir_name = os.environ.get("STRATEGY_SET", "champions")
STRATEGY_DIR = Path(__file__).resolve().parent.parent / "strategies" / _dir_name


def fetch_data():
    """Fetch OHLCV data once."""
    feed = validate_feed(FEED)
    candle_type = validate_type(CANDLE_TYPE)
    from_ms, till_ms = parse_time_range(TIME_RANGE)

    print(f"Fetching {feed} {candle_type}m candles for {TIME_RANGE}...")

    def _progress(fetched: int, chunk: int):
        print(f"  chunk {chunk}: {fetched} candles...", flush=True)

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=_progress)
    print(f"Total: {len(candles)} candles")
    print(f"Period: {candles[0].time.date()} to {candles[-1].time.date()}")
    return candles


def run_all_strategies(candles):
    """Run all strategy files and collect results."""
    strategy_files = sorted(STRATEGY_DIR.glob("*.json"))
    if not strategy_files:
        print(f"No strategy files found in {STRATEGY_DIR}")
        return []

    results = []
    for sf in strategy_files:
        name = sf.stem
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        strategy = Strategy.from_file(sf)
        result = run_simulation(candles, strategy, CANDLE_TYPE, FEED, TIME_RANGE)
        results.append((name, result))

        # Quick summary
        wr_mark = "✓" if result.win_rate >= 60 else "✗"
        pnl_mark = "✓" if result.total_pnl > 0 else "✗"
        print(f"  Trades: {result.total_trades:>5}")
        print(f"  WinRate: {result.win_rate:>6.1f}% {wr_mark}")
        print(f"  PnL:    ${result.total_pnl:>10,.2f} ({result.total_pnl_pct:+.2f}%) {pnl_mark}")
        print(f"  Expect: ${result.expectancy:>8,.4f}/trade")
        print(f"  Sharpe: {result.sharpe_ratio:>8.4f}")
        print(f"  PF:     {result.profit_factor:>8.2f}")
        print(f"  MaxDD:  {result.max_drawdown_pct:>6.2f}%")
        print(f"  AvgDur: {result.avg_duration_minutes / 60:>6.1f}h")
        print(f"  Streaks: W={result.max_consecutive_wins} L={result.max_consecutive_losses}")

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
    candles = fetch_data()
    results = run_all_strategies(candles)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
