#!/usr/bin/env python3
"""Bayesian Optimization for trading strategy hyperparameters.

Uses Optuna (TPE sampler) to search for the best combination of:
- Entry trigger percentage and lookback window
- Take-profit and stop-loss percentages
- Max concurrent positions and max per rule
- Position sizing

Walk-forward validation splits data into train/test to guard against overfitting.

Usage:
  uv run python -B scripts/optimize.py --feed BTCUSD --type 15
  uv run python -B scripts/optimize.py --feed SOLUSD --type 15 --trials 300 --rules 2
  uv run python -B scripts/optimize.py --feed ETHUSD --type 15 --objective sharpe --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.engine import run_simulation
from trade_simulator.models import Candle, TradeResult
from trade_simulator.strategy import Strategy

# Suppress optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Search space bounds ──────────────────────────────────────────────────────

ENTRY_PCT_RANGE = (0.5, 15.0)
ENTRY_PCT_STEP = 0.25
ENTRY_WINDOWS = ["1h", "2h", "4h", "1d"]
TP_PCT_RANGE = (0.5, 15.0)
TP_PCT_STEP = 0.25
SL_PCT_RANGE = (2.0, 25.0)
SL_PCT_STEP = 0.5
MAX_CONCURRENT_RANGE = (1, 10)
MAX_PER_RULE_RANGE = (1, 10)
POSITION_SIZE_RANGE = (100, 1000)
POSITION_SIZE_STEP = 50

MIN_TRADES = 10  # Discard trials with too few trades


def fetch_candles(feed: str, candle_type: str, time_range: str) -> list[Candle]:
    """Fetch OHLCV data with progress display."""
    from_ms, till_ms = parse_time_range(time_range)
    print(f"Fetching {feed} {candle_type}m candles for {time_range}...")

    def progress(fetched: int, chunk: int) -> None:
        print(f"  chunk {chunk}: {fetched} candles...")

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=progress)
    print(f"Total: {len(candles)} candles")
    print(f"Period: {candles[0].time.date()} to {candles[-1].time.date()}")
    return candles


def split_candles(
    candles: list[Candle], train_ratio: float = 0.75
) -> tuple[list[Candle], list[Candle]]:
    """Walk-forward split: first N% for training, rest for validation."""
    split_idx = int(len(candles) * train_ratio)
    return candles[:split_idx], candles[split_idx:]


def build_strategy(
    params: dict,
    n_rules: int,
    entry_type: str,
    capital: float,
) -> Strategy:
    """Build a Strategy from trial parameters."""
    rules = []
    for i in range(n_rules):
        suffix = f"_{i}" if n_rules > 1 else ""
        rules.append({
            "id": f"rule{suffix}" if n_rules == 1 else f"rule_{i}",
            "entry": {
                "type": entry_type,
                "percentage": params[f"entry_pct{suffix}"],
                "window": params[f"entry_window{suffix}"],
            },
            "exit": {
                "take_profit_pct": params[f"tp_pct{suffix}"],
                "stop_loss_pct": params[f"sl_pct{suffix}"],
            },
            "max_positions": params[f"max_per_rule{suffix}"],
        })

    strategy_dict = {
        "initial_capital": capital,
        "fee_pct": 0.1,
        "max_concurrent_positions": params["max_concurrent"],
        "position_sizing": {"type": "fixed", "value": params["position_size"]},
        "rules": rules,
    }
    return Strategy.model_validate(strategy_dict)


def compute_score(result: TradeResult, objective: str) -> float:
    """Compute the optimization objective score."""
    if result.total_trades < MIN_TRADES:
        return -999.0

    if objective == "pnl":
        return result.total_pnl_pct
    elif objective == "sharpe":
        return result.sharpe_ratio
    elif objective == "profit_factor":
        pf = result.profit_factor
        return min(pf, 20.0) if pf != float("inf") else 20.0
    elif objective == "expectancy":
        return result.expectancy
    else:  # composite (default)
        pf = result.profit_factor
        pf = min(pf, 20.0) if pf != float("inf") else 20.0
        dd = max(result.max_drawdown_pct, 0.01)
        return result.total_pnl_pct * (1 + pf) / (1 + dd / 10)


def create_objective(
    candles: list[Candle],
    candle_type: str,
    feed: str,
    time_range: str,
    n_rules: int,
    entry_type: str,
    capital: float,
    objective: str,
):
    """Create the optuna objective function (closure over data)."""
    trial_count = [0]
    best_score = [float("-inf")]
    start_time = [time.time()]

    def objective_fn(trial: optuna.Trial) -> float:
        trial_count[0] += 1

        # ── Sample parameters ────────────────────────────────────────
        params: dict = {}

        prev_entry_pct = 0.0
        for i in range(n_rules):
            suffix = f"_{i}" if n_rules > 1 else ""

            # For multi-rule, enforce increasing entry percentage (DCA ladder)
            low = max(ENTRY_PCT_RANGE[0], prev_entry_pct + 0.5) if i > 0 else ENTRY_PCT_RANGE[0]
            if low >= ENTRY_PCT_RANGE[1]:
                return -999.0  # Can't fit more rules

            entry_pct = trial.suggest_float(
                f"entry_pct{suffix}", low, ENTRY_PCT_RANGE[1], step=ENTRY_PCT_STEP
            )
            prev_entry_pct = entry_pct

            params[f"entry_pct{suffix}"] = entry_pct
            params[f"entry_window{suffix}"] = trial.suggest_categorical(
                f"entry_window{suffix}", ENTRY_WINDOWS
            )
            params[f"tp_pct{suffix}"] = trial.suggest_float(
                f"tp_pct{suffix}", TP_PCT_RANGE[0], TP_PCT_RANGE[1], step=TP_PCT_STEP
            )
            params[f"sl_pct{suffix}"] = trial.suggest_float(
                f"sl_pct{suffix}", SL_PCT_RANGE[0], SL_PCT_RANGE[1], step=SL_PCT_STEP
            )
            params[f"max_per_rule{suffix}"] = trial.suggest_int(
                f"max_per_rule{suffix}", MAX_PER_RULE_RANGE[0], MAX_PER_RULE_RANGE[1]
            )

        params["max_concurrent"] = trial.suggest_int(
            "max_concurrent", MAX_CONCURRENT_RANGE[0], MAX_CONCURRENT_RANGE[1]
        )
        params["position_size"] = trial.suggest_float(
            "position_size", POSITION_SIZE_RANGE[0], POSITION_SIZE_RANGE[1], step=POSITION_SIZE_STEP
        )

        # Constraint: sum of max_per_rule <= max_concurrent
        total_per_rule = sum(
            params[f"max_per_rule{'_' + str(i) if n_rules > 1 else ''}"]
            for i in range(n_rules)
        )
        if total_per_rule > params["max_concurrent"]:
            return -999.0

        # ── Build strategy and run simulation ────────────────────────
        strategy = build_strategy(params, n_rules, entry_type, capital)
        result = run_simulation(candles, strategy, candle_type, feed, time_range)

        score = compute_score(result, objective)

        # Store extra metrics as user attributes for later analysis
        trial.set_user_attr("total_trades", result.total_trades)
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("pnl_pct", result.total_pnl_pct)
        trial.set_user_attr("max_drawdown_pct", result.max_drawdown_pct)
        trial.set_user_attr("profit_factor", result.profit_factor)
        trial.set_user_attr("sharpe_ratio", result.sharpe_ratio)
        trial.set_user_attr("expectancy", result.expectancy)

        # Progress update
        if score > best_score[0]:
            best_score[0] = score
        elapsed = time.time() - start_time[0]
        if trial_count[0] % 25 == 0 or trial_count[0] <= 5:
            print(
                f"  Trial {trial_count[0]:4d} │ Score: {score:8.2f} │ "
                f"Best: {best_score[0]:8.2f} │ "
                f"Trades: {result.total_trades:4d} │ "
                f"PnL: {result.total_pnl_pct:+6.2f}% │ "
                f"WR: {result.win_rate:5.1f}% │ "
                f"Elapsed: {elapsed:.0f}s"
            )

        return score

    return objective_fn


def print_result_metrics(result: TradeResult, label: str) -> None:
    """Print a compact metrics table for a simulation result."""
    print(f"\n  {label}:")
    print("  ┌──────────────────────┬──────────────┐")
    print(f"  │ Trades               │ {result.total_trades:>12d} │")
    print(f"  │ Win Rate             │ {result.win_rate:>11.1f}% │")
    print(f"  │ PnL ($)              │ {result.total_pnl:>+12.2f} │")
    print(f"  │ PnL (%)              │ {result.total_pnl_pct:>+11.2f}% │")
    print(f"  │ Expectancy ($/trade) │ {result.expectancy:>+12.4f} │")
    print(f"  │ Sharpe Ratio         │ {result.sharpe_ratio:>12.4f} │")
    pf_str = f"{result.profit_factor:>12.2f}" if result.profit_factor != float("inf") else "         inf"
    print(f"  │ Profit Factor        │ {pf_str} │")
    print(f"  │ Max Drawdown (%)     │ {result.max_drawdown_pct:>11.2f}% │")
    h, m = divmod(int(result.avg_duration_minutes), 60)
    print(f"  │ Avg Duration         │ {h:>5d}h {m:02d}m    │")
    print(f"  │ Max Win Streak       │ {result.max_consecutive_wins:>12d} │")
    print(f"  │ Max Loss Streak      │ {result.max_consecutive_losses:>12d} │")
    print("  └──────────────────────┴──────────────┘")


def print_top_trials(study: optuna.Study, n: int = 10) -> None:
    """Print top N trials sorted by score."""
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)
    top = [t for t in trials if t.value is not None and t.value > -999][:n]

    if not top:
        print("\n  No valid trials found.")
        return

    print(f"\n  TOP {len(top)} TRIALS:")
    print("  ┌───────┬──────────┬────────┬─────────┬──────────┬──────────┬──────────┐")
    print("  │ Trial │   Score  │ Trades │   WR%   │   PnL%   │  MaxDD%  │    PF    │")
    print("  ├───────┼──────────┼────────┼─────────┼──────────┼──────────┼──────────┤")

    for t in top:
        attrs = t.user_attrs
        print(
            f"  │ {t.number:5d} │ {t.value:8.2f} │ {attrs.get('total_trades', 0):6d} │ "
            f"{attrs.get('win_rate', 0):6.1f}% │ {attrs.get('pnl_pct', 0):+7.2f}% │ "
            f"{attrs.get('max_drawdown_pct', 0):7.2f}% │ "
            f"{min(attrs.get('profit_factor', 0), 99.99):8.2f} │"
        )

    print("  └───────┴──────────┴────────┴─────────┴──────────┴──────────┴──────────┘")


def print_best_params(study: optuna.Study, n_rules: int) -> None:
    """Print the best parameters in a neat table."""
    best = study.best_trial
    params = best.params

    print(f"\n  BEST PARAMETERS (Trial #{best.number}, Score: {best.value:.4f}):")
    print("  ┌──────────────────────────────────┬──────────────┐")

    for i in range(n_rules):
        suffix = f"_{i}" if n_rules > 1 else ""
        label = f" (rule {i})" if n_rules > 1 else ""
        print(f"  │ Entry %{label:24s} │ {params[f'entry_pct{suffix}']:>11.2f}% │")
        print(f"  │ Entry Window{label:19s} │ {params[f'entry_window{suffix}']:>12s} │")
        print(f"  │ Take Profit %{label:18s} │ {params[f'tp_pct{suffix}']:>11.2f}% │")
        print(f"  │ Stop Loss %{label:20s} │ {params[f'sl_pct{suffix}']:>11.2f}% │")
        print(f"  │ Max Positions/Rule{label:13s} │ {params[f'max_per_rule{suffix}']:>12d} │")
        if i < n_rules - 1:
            print("  ├──────────────────────────────────┼──────────────┤")

    print("  ├──────────────────────────────────┼──────────────┤")
    print(f"  │ Max Concurrent Positions         │ {params['max_concurrent']:>12d} │")
    print(f"  │ Position Size ($)                │ {params['position_size']:>12.0f} │")
    print("  └──────────────────────────────────┴──────────────┘")


def export_strategy(
    study: optuna.Study,
    n_rules: int,
    entry_type: str,
    capital: float,
    feed: str,
    candle_type: str,
) -> Path:
    """Save the best strategy as a JSON file."""
    params = study.best_trial.params
    strategy = build_strategy(params, n_rules, entry_type, capital)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"optimized_{feed}_{candle_type}m.json"

    strategy_dict = {
        "_description": (
            f"Optimized via Bayesian Optimization ({len(study.trials)} trials, "
            f"score={study.best_value:.4f})"
        ),
        "initial_capital": strategy.initial_capital,
        "fee_pct": strategy.fee_pct,
        "max_concurrent_positions": strategy.max_concurrent_positions,
        "position_sizing": {
            "type": strategy.position_sizing.type.value,
            "value": strategy.position_sizing.value,
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
            for rule in strategy.rules
        ],
    }

    with open(path, "w") as f:
        json.dump(strategy_dict, f, indent=2)
        f.write("\n")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for trading strategy hyperparameters."
    )
    parser.add_argument("--feed", required=True, choices=["BTCUSD", "SOLUSD", "ETHUSD"])
    parser.add_argument("--type", dest="candle_type", required=True, choices=["5", "15"])
    parser.add_argument("--range", dest="time_range", default="1y", help="Data range (default: 1y)")
    parser.add_argument("--trials", type=int, default=200, help="Number of optimization trials (default: 200)")
    parser.add_argument("--rules", type=int, default=1, choices=[1, 2, 3], help="Number of rules (default: 1)")
    parser.add_argument(
        "--entry-type", default="price_drop", choices=["price_drop", "price_rise"],
        help="Entry condition type (default: price_drop)"
    )
    parser.add_argument(
        "--objective", default="composite",
        choices=["pnl", "sharpe", "composite", "profit_factor", "expectancy"],
        help="Optimization objective (default: composite)"
    )
    parser.add_argument("--capital", type=float, default=5000, help="Initial capital (default: 5000)")
    parser.add_argument("--split", type=float, default=0.75, help="Train/validation split ratio (default: 0.75)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    feed = validate_feed(args.feed)
    candle_type = validate_type(args.candle_type)

    # ── Fetch data ───────────────────────────────────────────────────────
    candles = fetch_candles(feed, candle_type, args.time_range)
    train_candles, val_candles = split_candles(candles, args.split)

    print(f"\nTrain set: {len(train_candles)} candles ({train_candles[0].time.date()} to {train_candles[-1].time.date()})")
    print(f"Val   set: {len(val_candles)} candles ({val_candles[0].time.date()} to {val_candles[-1].time.date()})")

    # ── Run optimization ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  BAYESIAN OPTIMIZATION — {feed} {candle_type}m — {args.trials} trials")
    print(f"  Objective: {args.objective} │ Rules: {args.rules} │ Entry: {args.entry_type}")
    print(f"{'=' * 80}")
    print()

    sampler = TPESampler(seed=args.seed, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective_fn = create_objective(
        train_candles, candle_type, feed, args.time_range,
        args.rules, args.entry_type, args.capital, args.objective,
    )

    t0 = time.time()
    study.optimize(objective_fn, n_trials=args.trials)
    elapsed = time.time() - t0

    print(f"\n  Optimization complete: {len(study.trials)} trials in {elapsed:.1f}s")
    print(f"  ({elapsed / len(study.trials):.3f}s per trial)")

    # ── Results ──────────────────────────────────────────────────────────
    print_top_trials(study)
    print_best_params(study, args.rules)

    # ── Validate on held-out data ────────────────────────────────────────
    best_params = study.best_trial.params
    best_strategy = build_strategy(best_params, args.rules, args.entry_type, args.capital)

    train_result = run_simulation(train_candles, best_strategy, candle_type, feed, args.time_range)
    val_result = run_simulation(val_candles, best_strategy, candle_type, feed, args.time_range)

    print(f"\n{'=' * 80}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'=' * 80}")

    print_result_metrics(train_result, f"TRAIN ({train_candles[0].time.date()} → {train_candles[-1].time.date()})")
    print_result_metrics(val_result, f"VALIDATION ({val_candles[0].time.date()} → {val_candles[-1].time.date()})")

    # ── Overfitting detection ────────────────────────────────────────────
    train_score = compute_score(train_result, args.objective)
    val_score = compute_score(val_result, args.objective)

    print(f"\n  Train Score:      {train_score:+.4f}")
    print(f"  Validation Score: {val_score:+.4f}")

    if train_score > 0 and val_score > 0:
        ratio = val_score / train_score
        print(f"  Val/Train Ratio:  {ratio:.2f}")
        if ratio < 0.5:
            print("\n  ⚠️  OVERFITTING WARNING: Validation score < 50% of train score.")
            print("      The optimized parameters may not generalize well.")
            print("      Consider reducing --trials or widening parameter ranges.")
    elif train_score > 0 > val_score:
        print("\n  ⚠️  OVERFITTING WARNING: Strategy is profitable on train but loses on validation.")
        print("      The parameters are likely overfit to the training period.")

    # ── Full-data run ────────────────────────────────────────────────────
    full_result = run_simulation(candles, best_strategy, candle_type, feed, args.time_range)
    print_result_metrics(full_result, f"FULL DATA ({candles[0].time.date()} → {candles[-1].time.date()})")

    # ── Export ────────────────────────────────────────────────────────────
    path = export_strategy(study, args.rules, args.entry_type, args.capital, feed, candle_type)
    print(f"\n  Strategy saved to {path}")
    print()


if __name__ == "__main__":
    main()
