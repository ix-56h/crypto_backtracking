#!/usr/bin/env python3
"""Batch optimize: fetch data once, optimize all strategies, compare results.

Discovers strategy JSON files, uses each as a seed for Bayesian or GA
optimization, and outputs a comparison table of original vs optimized
performance.

Usage:
  uv run python -B scripts/batch_optimize.py --feed BTCUSD --type 15
  uv run python -B scripts/batch_optimize.py --feed BTCUSD --type 15 --method ga --generations 30
  uv run python -B scripts/batch_optimize.py --feed BTCUSD --type 15 --strategies strategies/champions --trials 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent / "src"))
sys.path.insert(0, str(_scripts_dir))

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

from optim_shared import (  # noqa: E402
    build_strategy,
    compute_score,
    extract_params_from_strategy,
    fetch_candles,
    split_candles,
    validate_feed,
    validate_type,
    run_simulation,
)
from optimize import create_objective  # noqa: E402
from optimize_ga import run_ga  # noqa: E402
from trade_simulator.strategy import Strategy  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = _scripts_dir.parent
DEFAULT_STRATEGY_DIR = PROJECT_ROOT / "strategies"


def find_strategies(strategy_dir: Path) -> list[Path]:
    """Find all .json strategy files recursively."""
    return sorted(strategy_dir.rglob("*.json"))


def _optimize_one(
    strategy_path: Path,
    strategy_dir: Path,
    candles,
    split: float,
    candle_type: str,
    feed: str,
    time_range: str,
    method: str,
    trials: int,
    pop_size: int,
    generations: int,
    objective: str,
    seed: int | None,
) -> dict:
    """Optimize a single strategy. Returns results dict."""
    rel = strategy_path.relative_to(strategy_dir)
    name = str(rel.with_suffix(""))

    try:
        strategy = Strategy.from_file(strategy_path)
        seed_params, n_rules, entry_type = extract_params_from_strategy(strategy)
        capital = strategy.initial_capital

        train, val = split_candles(candles, split)

        # Original performance on full data
        orig_result = run_simulation(candles, strategy, candle_type, feed, time_range)
        orig_score = compute_score(orig_result, objective)

        # Run optimization (suppress console output, warnings, and optuna logs)
        optuna_logger = logging.getLogger("optuna")
        prev_level = optuna_logger.level
        optuna_logger.setLevel(logging.CRITICAL)
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            if method == "bayesian":
                sampler = TPESampler(seed=seed, n_startup_trials=min(20, trials // 2))
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.enqueue_trial(seed_params)
                obj_fn = create_objective(
                    train, candle_type, feed, time_range,
                    n_rules, entry_type, capital, objective,
                )
                study.optimize(obj_fn, n_trials=trials)
                best_params = study.best_trial.params
                train_score = study.best_value
            else:  # ga
                best_params, train_score, _ = run_ga(
                    train, candle_type, feed, time_range,
                    n_rules, entry_type, capital, objective,
                    pop_size, generations,
                    0.8, 0.15, 3, 2,
                    seed, seed_params,
                )

        optuna_logger.setLevel(prev_level)

        # Optimized performance on full data
        opt_strat = build_strategy(best_params, n_rules, entry_type, capital)
        opt_result = run_simulation(candles, opt_strat, candle_type, feed, time_range)
        opt_score = compute_score(opt_result, objective)

        # Save optimized strategy
        out_dir = PROJECT_ROOT / "output" / "batch_optimized"
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        desc = (
            f"Optimized from {rel.name} via {method} "
            f"({trials if method == 'bayesian' else f'{generations}gen×{pop_size}pop'}, "
            f"score={train_score:.4f})"
        )
        strat_dict = {
            "_description": desc,
            "initial_capital": capital,
            "fee_pct": strategy.fee_pct,
            "max_concurrent_positions": opt_strat.max_concurrent_positions,
            "position_sizing": {
                "type": opt_strat.position_sizing.type.value,
                "value": opt_strat.position_sizing.value,
            },
            "rules": [
                {
                    "id": rule.id,
                    "entry": {
                        "type": rule.entry.type.value,
                        "percentage": rule.entry.percentage,
                        "window": rule.entry.window,
                    },
                    "exit": {
                        "take_profit_pct": rule.exit.take_profit_pct,
                        "stop_loss_pct": rule.exit.stop_loss_pct,
                    },
                    "max_positions": rule.max_positions,
                }
                for rule in opt_strat.rules
            ],
        }
        with open(out_path, "w") as f:
            json.dump(strat_dict, f, indent=2)
            f.write("\n")

        return {
            "name": name,
            "error": None,
            "n_rules": n_rules,
            "entry_type": entry_type,
            "train_score": round(train_score, 4),
            "orig_trades": orig_result.total_trades,
            "orig_wr": round(orig_result.win_rate, 1),
            "orig_pnl": round(orig_result.total_pnl_pct, 2),
            "orig_score": round(orig_score, 4),
            "opt_trades": opt_result.total_trades,
            "opt_wr": round(opt_result.win_rate, 1),
            "opt_pnl": round(opt_result.total_pnl_pct, 2),
            "opt_score": round(opt_score, 4),
            "opt_pf": round(opt_result.profit_factor, 2) if opt_result.profit_factor != float("inf") else 99.99,
            "opt_dd": round(opt_result.max_drawdown_pct, 2),
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


def print_comparison(results: list[dict], method: str, elapsed: float) -> None:
    """Print the original vs optimized comparison table."""
    valid = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    if not valid:
        print("\nNo strategies were optimized successfully.")
        return

    print(f"\n{'=' * 130}")
    print(f"{'BATCH OPTIMIZATION RESULTS':^130}")
    print(f"{'=' * 130}")

    header = (
        f"{'Strategy':<30} │ {'Orig PnL%':>9} │ {'Opt PnL%':>9} │ {'Δ PnL%':>8} │ "
        f"{'Orig WR':>7} │ {'Opt WR':>7} │ {'Score':>8} │ {'MaxDD%':>7} │ {'PF':>6} │ {'Trades':>6}"
    )
    print(header)
    print("─" * 130)

    improved = 0
    for r in valid:
        delta_pnl = r["opt_pnl"] - r["orig_pnl"]
        mark = "✓" if delta_pnl > 0 else " "
        if delta_pnl > 0:
            improved += 1

        pf_str = f"{r['opt_pf']:6.2f}" if r["opt_pf"] < 99.99 else "   inf"
        print(
            f"{r['name']:<30} │ {r['orig_pnl']:>+8.2f}% │ {r['opt_pnl']:>+8.2f}% │ {delta_pnl:>+7.2f}% │ "
            f"{r['orig_wr']:>6.1f}% │ {r['opt_wr']:>6.1f}% │ {r['opt_score']:>8.2f} │ "
            f"{r['opt_dd']:>6.2f}% │ {pf_str} │ {r['opt_trades']:>6d} {mark}"
        )

    print("─" * 130)

    # Summary
    best_score = max(valid, key=lambda r: r["opt_score"])
    best_pnl = max(valid, key=lambda r: r["opt_pnl"])
    best_delta = max(valid, key=lambda r: r["opt_pnl"] - r["orig_pnl"])

    avg_orig_pnl = sum(r["orig_pnl"] for r in valid) / len(valid)
    avg_opt_pnl = sum(r["opt_pnl"] for r in valid) / len(valid)

    print(f"\n  Optimized {len(valid)} strategies via {method} in {elapsed:.1f}s")
    print(f"  Improved: {improved}/{len(valid)} ({improved / len(valid) * 100:.0f}%)")
    print(f"  Avg PnL: {avg_orig_pnl:+.2f}% → {avg_opt_pnl:+.2f}%")
    print(f"  Best Score:      {best_score['name']} ({best_score['opt_score']:.2f})")
    print(f"  Best PnL:        {best_pnl['name']} ({best_pnl['opt_pnl']:+.2f}%)")
    delta = best_delta["opt_pnl"] - best_delta["orig_pnl"]
    print(f"  Best Improvement: {best_delta['name']} ({delta:+.2f}%)")

    if errors:
        print(f"\n  ⚠ {len(errors)} strategies failed:")
        for r in errors:
            print(f"    {r['name']}: {r['error']}")

    print(f"\n  Optimized strategies saved to output/batch_optimized/")


def save_results(results: list[dict]) -> Path:
    """Save comparison results to JSON."""
    out_path = PROJECT_ROOT / "output" / "batch_optimize_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export = [r for r in results if r["error"] is None]
    # Remove non-serializable fields
    for r in export:
        r.pop("best_params", None)

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
        f.write("\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch optimize: fetch data once, optimize all strategies, compare results."
    )
    parser.add_argument("--feed", default="BTCUSD", choices=["BTCUSD", "SOLUSD", "ETHUSD"],
                        help="Trading pair (default: BTCUSD)")
    parser.add_argument("--type", dest="candle_type", default="15", choices=["5", "15"],
                        help="Candle width in minutes (default: 15)")
    parser.add_argument("--range", dest="time_range", default="1y",
                        help="Data range (default: 1y)")
    parser.add_argument("--strategies", default=None,
                        help="Path to strategy directory (default: strategies/ recursive)")
    parser.add_argument("--method", default="bayesian", choices=["bayesian", "ga"],
                        help="Optimization method (default: bayesian)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Bayesian trials per strategy (default: 50)")
    parser.add_argument("--pop-size", type=int, default=30,
                        help="GA population size (default: 30)")
    parser.add_argument("--generations", type=int, default=50,
                        help="GA generations (default: 50)")
    parser.add_argument("--objective", default="composite",
                        choices=["pnl", "sharpe", "composite", "profit_factor", "expectancy"],
                        help="Optimization objective (default: composite)")
    parser.add_argument("--split", type=float, default=0.75,
                        help="Train/validation split ratio (default: 0.75)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help=f"Parallel worker processes (default: {os.cpu_count() or 4})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    strategy_dir = Path(args.strategies).resolve() if args.strategies else DEFAULT_STRATEGY_DIR
    if not strategy_dir.is_dir():
        print(f"Error: strategy directory not found: {strategy_dir}")
        sys.exit(1)

    strategy_files = find_strategies(strategy_dir)
    if not strategy_files:
        print(f"No strategy files found in {strategy_dir}")
        sys.exit(1)

    feed = validate_feed(args.feed)
    candle_type = validate_type(args.candle_type)

    # Fetch data once
    candles = fetch_candles(feed, candle_type, args.time_range)

    total = len(strategy_files)
    method_desc = (
        f"{args.trials} trials" if args.method == "bayesian"
        else f"{args.generations}gen × {args.pop_size}pop"
    )
    print(f"\nFound {total} strategies in {strategy_dir}")
    print(f"Method: {args.method} ({method_desc}) │ Objective: {args.objective}")
    print(f"Workers: {args.workers}")
    print()

    results: list[dict] = []
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, sf in enumerate(strategy_files):
            strategy_seed = (args.seed + i) if args.seed is not None else None
            future = pool.submit(
                _optimize_one, sf, strategy_dir, candles, args.split,
                candle_type, feed, args.time_range,
                args.method, args.trials, args.pop_size, args.generations,
                args.objective, strategy_seed,
            )
            futures[future] = sf

        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)

            if r["error"]:
                print(f"  [{i:>{len(str(total))}}/{total}] {r['name']:<30} ERROR: {r['error']}")
            else:
                delta = r["opt_pnl"] - r["orig_pnl"]
                mark = "✓" if delta > 0 else "✗"
                print(
                    f"  [{i:>{len(str(total))}}/{total}] {r['name']:<30} "
                    f"PnL: {r['orig_pnl']:>+7.2f}% → {r['opt_pnl']:>+7.2f}% "
                    f"(Δ {delta:>+6.2f}%) "
                    f"Score={r['opt_score']:>8.2f} {mark}"
                )

    elapsed = time.monotonic() - t0

    # Sort by name for consistent table
    results.sort(key=lambda r: r["name"])

    print_comparison(results, args.method, elapsed)
    json_path = save_results(results)
    print(f"  Results saved to {json_path}")
    print()


if __name__ == "__main__":
    main()
