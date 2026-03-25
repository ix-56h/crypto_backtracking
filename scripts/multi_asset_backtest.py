#!/usr/bin/env python3
"""Multi-asset backtest: test champion strategies on BTC, SOL, and ETH."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.engine import run_simulation
from trade_simulator.strategy import Strategy


FEEDS = ["BTCUSD", "SOLUSD", "ETHUSD"]
CANDLE_TYPE = "15"
TIME_RANGE = "1y"
STRATEGY_SET = os.environ.get("STRATEGY_SET", "champions")
STRATEGY_DIR = Path(__file__).resolve().parent.parent / "strategies" / STRATEGY_SET


def fetch_data(feed: str):
    """Fetch OHLCV data for a given feed."""
    feed = validate_feed(feed)
    candle_type = validate_type(CANDLE_TYPE)
    from_ms, till_ms = parse_time_range(TIME_RANGE)

    print(f"\nFetching {feed} {candle_type}m candles for {TIME_RANGE}...")

    def _progress(fetched: int, chunk: int):
        print(f"  chunk {chunk}: {fetched} candles...", flush=True)

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=_progress)
    print(f"Total: {len(candles)} candles ({candles[0].time.date()} to {candles[-1].time.date()})")
    return candles


def main():
    strategy_files = sorted(STRATEGY_DIR.glob("*.json"))
    if not strategy_files:
        print(f"No strategy files found in {STRATEGY_DIR}")
        return

    # Fetch data for all feeds
    data = {}
    for feed in FEEDS:
        data[feed] = fetch_data(feed)

    # Run all strategies on all feeds
    all_results = []
    for sf in strategy_files:
        sname = sf.stem
        strategy = Strategy.from_file(sf)
        for feed in FEEDS:
            result = run_simulation(data[feed], strategy, CANDLE_TYPE, feed, TIME_RANGE)
            all_results.append((sname, feed, result))

    # Print comparison table
    print(f"\n\n{'='*140}")
    print(f"{'MULTI-ASSET CHAMPION COMPARISON':^140}")
    print(f"{'='*140}")

    header = (
        f"{'Strategy':<30} {'Feed':<8} {'Trades':>6} {'WR%':>6} {'PnL$':>10} "
        f"{'PnL%':>8} {'Expect':>8} {'Sharpe':>7} {'PF':>6} "
        f"{'MaxDD%':>7} {'AvgHrs':>7} {'W-Str':>5} {'L-Str':>5}"
    )
    print(header)
    print("-" * 140)

    for sname, feed, r in all_results:
        wr_flag = " ★" if r.win_rate >= 60 else ""
        pf = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
        pnl_flag = " ✓" if r.total_pnl > 0 else ""
        print(
            f"{sname:<30} {feed:<8} {r.total_trades:>6} {r.win_rate:>5.1f}%{wr_flag} "
            f"{r.total_pnl:>10,.2f}{pnl_flag} {r.total_pnl_pct:>+7.2f}% "
            f"{r.expectancy:>8.4f} {r.sharpe_ratio:>7.4f} {pf:>6} "
            f"{r.max_drawdown_pct:>6.2f}% {r.avg_duration_minutes / 60:>7.1f} "
            f"{r.max_consecutive_wins:>5} {r.max_consecutive_losses:>5}"
        )

    print("-" * 140)

    # Summary by strategy
    print(f"\n{'AGGREGATE BY STRATEGY':^140}")
    print("-" * 80)
    strats = sorted(set(s for s, _, _ in all_results))
    for sname in strats:
        sresults = [(f, r) for s, f, r in all_results if s == sname]
        avg_wr = sum(r.win_rate for _, r in sresults) / len(sresults)
        total_pnl = sum(r.total_pnl for _, r in sresults)
        total_trades = sum(r.total_trades for _, r in sresults)
        profitable_count = sum(1 for _, r in sresults if r.total_pnl > 0)
        avg_pf = sum(r.profit_factor for _, r in sresults if r.profit_factor != float("inf")) / len(sresults)
        print(
            f"  {sname:<30} AvgWR={avg_wr:>5.1f}%  "
            f"TotalPnL=${total_pnl:>10,.2f}  Trades={total_trades:>5}  "
            f"Profitable={profitable_count}/{len(sresults)} feeds  "
            f"AvgPF={avg_pf:.2f}"
        )

    # Save detailed results
    output_path = Path(__file__).resolve().parent.parent / "output" / "champion_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = []
    for sname, feed, r in all_results:
        comparison.append({
            "strategy": sname,
            "feed": feed,
            "trades": r.total_trades,
            "win_rate": round(r.win_rate, 2),
            "total_pnl": round(r.total_pnl, 2),
            "total_pnl_pct": round(r.total_pnl_pct, 2),
            "expectancy": round(r.expectancy, 4),
            "sharpe_ratio": round(r.sharpe_ratio, 4),
            "profit_factor": round(r.profit_factor, 4) if r.profit_factor != float("inf") else "inf",
            "max_drawdown_pct": round(r.max_drawdown_pct, 2),
            "avg_duration_hours": round(r.avg_duration_minutes / 60, 1),
            "per_rule": r.per_rule,
        })
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
