#!/usr/bin/env python3
"""Walk-Forward Analysis (WFA) for trading strategy optimization.

Divides historical data into multiple overlapping train/test folds, runs
Bayesian optimization on each training window, and evaluates out-of-sample
on the test window. Aggregates all OOS results for robust performance
estimates and parameter stability analysis.

Modes:
  rolling  — Fixed-size train window slides forward
  anchored — Train window grows from start (anchored beginning)

Usage:
  uv run python -B scripts/optimize_wfa.py --feed BTCUSD --type 15
  uv run python -B scripts/optimize_wfa.py --feed SOLUSD --type 15 --mode anchored --train-months 8
  uv run python -B scripts/optimize_wfa.py --feed ETHUSD --type 15 --trials 150 --test-months 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime, timedelta

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
    Candle,
    add_common_args,
    build_strategy,
    compute_score,
    export_strategy,
    fetch_candles,
    load_seed_strategy,
    print_best_params,
    print_result_metrics,
    run_simulation,
    validate_feed,
    validate_type,
)

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Fold generation ──────────────────────────────────────────────────────────


def _add_months(dt: datetime, months: int) -> datetime:
    """Add calendar months to a datetime (stdlib only)."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    # Clamp day to valid range
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return dt.replace(year=year, month=month, day=day)


def generate_folds(
    candles: list[Candle],
    train_months: int,
    test_months: int,
    mode: str,
) -> list[tuple[list[Candle], list[Candle]]]:
    """Generate train/test folds for walk-forward analysis.

    Returns list of (train_candles, test_candles) tuples.
    """
    first_time = candles[0].time
    last_time = candles[-1].time
    folds: list[tuple[list[Candle], list[Candle]]] = []

    if mode == "rolling":
        cursor = first_time
        while True:
            train_end = _add_months(cursor, train_months)
            test_end = _add_months(train_end, test_months)
            if test_end > last_time:
                break

            train = [c for c in candles if cursor <= c.time < train_end]
            test = [c for c in candles if train_end <= c.time < test_end]

            if len(train) > 100 and len(test) > 50:
                folds.append((train, test))

            cursor = _add_months(cursor, test_months)  # step by test_months

    elif mode == "anchored":
        anchor = first_time
        cursor = _add_months(anchor, train_months)
        while True:
            test_end = _add_months(cursor, test_months)
            if test_end > last_time:
                break

            train = [c for c in candles if anchor <= c.time < cursor]
            test = [c for c in candles if cursor <= c.time < test_end]

            if len(train) > 100 and len(test) > 50:
                folds.append((train, test))

            cursor = _add_months(cursor, test_months)

    return folds


# ── Inner optimizer (per fold) ───────────────────────────────────────────────


def inner_optimize(
    train_candles: list[Candle],
    candle_type: str,
    feed: str,
    time_range: str,
    n_rules: int,
    entry_type: str,
    capital: float,
    objective: str,
    n_trials: int,
    seed: int | None,
    seed_params: dict | None = None,
) -> tuple[dict, float]:
    """Run a small Bayesian optimization on one fold's training data.

    Returns (best_params, best_score).
    """
    sampler = TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if seed_params:
        study.enqueue_trial(seed_params)

    def objective_fn(trial: optuna.Trial) -> float:
        params: dict = {}
        prev_entry_pct = 0.0

        for i in range(n_rules):
            suffix = f"_{i}" if n_rules > 1 else ""
            low = max(ENTRY_PCT_RANGE[0], prev_entry_pct + 0.5) if i > 0 else ENTRY_PCT_RANGE[0]
            if low >= ENTRY_PCT_RANGE[1]:
                return -999.0

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
            "position_size", POSITION_SIZE_RANGE[0], POSITION_SIZE_RANGE[1],
            step=POSITION_SIZE_STEP,
        )

        # Constraint
        total_per_rule = sum(
            params[f"max_per_rule{'_' + str(i) if n_rules > 1 else ''}"]
            for i in range(n_rules)
        )
        if total_per_rule > params["max_concurrent"]:
            return -999.0

        strategy = build_strategy(params, n_rules, entry_type, capital)
        result = run_simulation(train_candles, strategy, candle_type, feed, time_range)
        return compute_score(result, objective)

    study.optimize(objective_fn, n_trials=n_trials)

    return study.best_trial.params, study.best_value


# ── Output formatters ────────────────────────────────────────────────────────


def print_fold_table(fold_results: list[dict]) -> None:
    """Print the fold-by-fold breakdown table."""
    print("\n  FOLD RESULTS:")
    print("  ┌──────┬──────────────────────────┬──────────────────────────┬────────┬────────┬──────────┬──────────┐")
    print("  │ Fold │       Train Period       │       Test Period        │ Trades │  WR%   │   PnL%   │    PF    │")
    print("  ├──────┼──────────────────────────┼──────────────────────────┼────────┼────────┼──────────┼──────────┤")

    for fr in fold_results:
        pf = fr["profit_factor"]
        pf_str = f"{pf:8.2f}" if pf != float("inf") else "     inf"
        print(
            f"  │ {fr['fold']:4d} │ {fr['train_start']} → {fr['train_end']} │ "
            f"{fr['test_start']} → {fr['test_end']} │ "
            f"{fr['trades']:6d} │ {fr['win_rate']:5.1f}% │ "
            f"{fr['pnl_pct']:+7.2f}% │ {pf_str} │"
        )

    print("  └──────┴──────────────────────────┴──────────────────────────┴────────┴────────┴──────────┴──────────┘")


def print_param_stability(all_params: list[dict], n_rules: int) -> None:
    """Show how much best parameters vary across folds."""
    if len(all_params) < 2:
        return

    print("\n  PARAMETER STABILITY (std dev across folds):")
    print("  ┌──────────────────────────────────┬──────────┬──────────┐")
    print("  │ Parameter                        │     Mean │   StdDev │")
    print("  ├──────────────────────────────────┼──────────┼──────────┤")

    # Collect numeric keys
    for key in all_params[0]:
        values = [p[key] for p in all_params]
        if isinstance(values[0], (int, float)):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            print(f"  │ {key:32s} │ {mean:8.2f} │ {std:8.2f} │")
        else:
            # Categorical — show most common
            from collections import Counter
            counts = Counter(values)
            most = counts.most_common(1)[0]
            pct = most[1] / len(values) * 100
            print(f"  │ {key:32s} │ {most[0]:>8s} │ {pct:6.0f}%  │")

    print("  └──────────────────────────────────┴──────────┴──────────┘")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Analysis for trading strategy optimization."
    )
    add_common_args(parser)
    parser.add_argument(
        "--mode", default="rolling", choices=["rolling", "anchored"],
        help="WFA mode (default: rolling)"
    )
    parser.add_argument("--train-months", type=int, default=6, help="Training window in months (default: 6)")
    parser.add_argument("--test-months", type=int, default=2, help="Test window in months (default: 2)")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials per fold (default: 100)")
    args = parser.parse_args()

    seed_params = load_seed_strategy(args)

    feed = validate_feed(args.feed)
    candle_type = validate_type(args.candle_type)

    # ── Fetch data ───────────────────────────────────────────────────────
    candles = fetch_candles(feed, candle_type, args.time_range)

    # ── Generate folds ───────────────────────────────────────────────────
    folds = generate_folds(candles, args.train_months, args.test_months, args.mode)
    if not folds:
        print("\n  ERROR: Not enough data for the requested fold configuration.")
        print(f"  Data spans {candles[0].time.date()} to {candles[-1].time.date()}")
        print(f"  Need at least {args.train_months + args.test_months} months for one fold.")
        return

    print(f"\n{'=' * 80}")
    print(f"  WALK-FORWARD ANALYSIS — {feed} {candle_type}m — {args.mode.upper()}")
    print(f"  Train: {args.train_months}mo │ Test: {args.test_months}mo │ Folds: {len(folds)} │ Trials/fold: {args.trials}")
    print(f"  Objective: {args.objective} │ Rules: {args.rules} │ Entry: {args.entry_type}")
    print(f"{'=' * 80}")

    # ── Run each fold ────────────────────────────────────────────────────
    fold_results: list[dict] = []
    all_best_params: list[dict] = []
    oos_total_trades = 0
    oos_total_wins = 0
    oos_total_pnl = 0.0
    oos_scores: list[float] = []

    t0 = time.time()

    for fold_idx, (train, test) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n  ── Fold {fold_num}/{len(folds)} ──")
        print(f"     Train: {train[0].time.date()} → {train[-1].time.date()} ({len(train)} candles)")
        print(f"     Test:  {test[0].time.date()} → {test[-1].time.date()} ({len(test)} candles)")

        # Inner optimization on training data
        fold_seed = (args.seed + fold_idx) if args.seed is not None else None
        best_params, train_score = inner_optimize(
            train, candle_type, feed, args.time_range,
            args.rules, args.entry_type, args.capital, args.objective,
            args.trials, fold_seed, seed_params,
        )
        all_best_params.append(best_params)

        print(f"     Train score: {train_score:+.4f}")

        # Out-of-sample evaluation
        strategy = build_strategy(best_params, args.rules, args.entry_type, args.capital)
        oos_result = run_simulation(test, strategy, candle_type, feed, args.time_range)
        oos_score = compute_score(oos_result, args.objective)

        print(
            f"     OOS: {oos_result.total_trades} trades │ WR: {oos_result.win_rate:.1f}% │ "
            f"PnL: {oos_result.total_pnl_pct:+.2f}% │ Score: {oos_score:+.4f}"
        )

        fold_results.append({
            "fold": fold_num,
            "train_start": str(train[0].time.date()),
            "train_end": str(train[-1].time.date()),
            "test_start": str(test[0].time.date()),
            "test_end": str(test[-1].time.date()),
            "trades": oos_result.total_trades,
            "win_rate": oos_result.win_rate,
            "pnl_pct": oos_result.total_pnl_pct,
            "profit_factor": oos_result.profit_factor,
            "train_score": train_score,
            "oos_score": oos_score,
        })

        oos_total_trades += oos_result.total_trades
        oos_total_wins += oos_result.winning_trades
        oos_total_pnl += oos_result.total_pnl
        oos_scores.append(oos_score)

    elapsed = time.time() - t0
    print(f"\n  WFA complete: {len(folds)} folds in {elapsed:.1f}s")

    # ── Fold table ───────────────────────────────────────────────────────
    print_fold_table(fold_results)

    # ── Aggregate OOS metrics ────────────────────────────────────────────
    profitable_folds = sum(1 for fr in fold_results if fr["pnl_pct"] > 0)
    consistency = profitable_folds / len(folds) * 100

    print(f"\n  AGGREGATE OUT-OF-SAMPLE METRICS:")
    print("  ┌──────────────────────────┬──────────────┐")
    print(f"  │ Total OOS Trades         │ {oos_total_trades:>12d} │")
    if oos_total_trades > 0:
        oos_wr = oos_total_wins / oos_total_trades * 100
        print(f"  │ OOS Win Rate             │ {oos_wr:>11.1f}% │")
    pnl_pcts = [fr["pnl_pct"] for fr in fold_results]
    total_oos_pnl_pct = sum(pnl_pcts)
    avg_oos_pnl_pct = statistics.mean(pnl_pcts) if pnl_pcts else 0
    print(f"  │ Total OOS PnL ($)        │ {oos_total_pnl:>+12.2f} │")
    print(f"  │ Total OOS PnL (%)        │ {total_oos_pnl_pct:>+11.2f}% │")
    print(f"  │ Avg OOS PnL/Fold (%)     │ {avg_oos_pnl_pct:>+11.2f}% │")
    if len(oos_scores) > 1:
        avg_score = statistics.mean(oos_scores)
        std_score = statistics.stdev(oos_scores)
        print(f"  │ Avg OOS Score            │ {avg_score:>+12.4f} │")
        print(f"  │ StdDev OOS Score         │ {std_score:>12.4f} │")
    print(f"  │ Profitable Folds         │ {profitable_folds}/{len(folds):>3d} ({consistency:4.0f}%) │")
    print("  └──────────────────────────┴──────────────┘")

    # ── Consistency assessment ───────────────────────────────────────────
    if consistency >= 75:
        print("\n  ✅ HIGH CONSISTENCY: Strategy generalizes well across time periods.")
    elif consistency >= 50:
        print("\n  ⚠️  MODERATE CONSISTENCY: Strategy works in some periods but not all.")
    else:
        print("\n  ❌ LOW CONSISTENCY: Strategy likely overfit — performs poorly out-of-sample.")

    # ── Parameter stability ──────────────────────────────────────────────
    print_param_stability(all_best_params, args.rules)

    # ── Export best fold's strategy ──────────────────────────────────────
    # Use the params from the fold with the best OOS score
    best_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]["oos_score"])
    best_fold = fold_results[best_fold_idx]
    best_params = all_best_params[best_fold_idx]

    print_best_params(
        best_params, args.rules, best_fold["oos_score"],
        label=f"BEST FOLD #{best_fold['fold']} PARAMETERS",
    )

    desc = (
        f"Walk-Forward Analysis ({args.mode}, {len(folds)} folds, "
        f"best OOS fold #{best_fold['fold']}, score={best_fold['oos_score']:.4f})"
    )
    path = export_strategy(
        best_params, args.rules, args.entry_type, args.capital, feed, candle_type, desc,
    )
    print(f"\n  Strategy saved to {path}")
    print()


if __name__ == "__main__":
    main()
