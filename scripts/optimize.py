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
import time

import optuna
from optuna.samplers import TPESampler

from optim_shared import (
    ENTRY_PCT_RANGE,
    ENTRY_PCT_STEP,
    ENTRY_WINDOWS,
    MAX_CONCURRENT_RANGE,
    MAX_PER_RULE_RANGE,
    MIN_TRADES,
    POSITION_SIZE_RANGE,
    POSITION_SIZE_STEP,
    SL_PCT_RANGE,
    SL_PCT_STEP,
    TP_PCT_RANGE,
    TP_PCT_STEP,
    add_common_args,
    build_strategy,
    compute_score,
    export_strategy,
    fetch_candles,
    load_seed_strategy,
    print_best_params,
    print_result_metrics,
    split_candles,
    validate_feed,
    validate_type,
    run_simulation,
)

# Suppress optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for trading strategy hyperparameters."
    )
    add_common_args(parser)
    parser.add_argument("--trials", type=int, default=200, help="Number of optimization trials (default: 200)")
    parser.add_argument("--split", type=float, default=0.75, help="Train/validation split ratio (default: 0.75)")
    args = parser.parse_args()

    seed_params = load_seed_strategy(args)

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

    if seed_params:
        study.enqueue_trial(seed_params)

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

    best_params = study.best_trial.params
    print_best_params(best_params, args.rules, study.best_value)

    # ── Validate on held-out data ────────────────────────────────────────
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
    desc = (
        f"Optimized via Bayesian Optimization ({len(study.trials)} trials, "
        f"score={study.best_value:.4f})"
    )
    path = export_strategy(
        best_params, args.rules, args.entry_type, args.capital, feed, candle_type, desc,
    )
    print(f"\n  Strategy saved to {path}")
    print()


if __name__ == "__main__":
    main()
