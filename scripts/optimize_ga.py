#!/usr/bin/env python3
"""Genetic Algorithm optimization for trading strategy hyperparameters.

Uses a population-based evolutionary search with tournament selection,
BLX-α crossover, gaussian mutation, and elitism.

Usage:
  uv run python -B scripts/optimize_ga.py --feed BTCUSD --type 15
  uv run python -B scripts/optimize_ga.py --feed SOLUSD --type 15 --pop-size 80 --generations 150
  uv run python -B scripts/optimize_ga.py --feed ETHUSD --type 15 --objective sharpe --seed 42
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time

from optim_shared import (
    ENTRY_PCT_RANGE,
    ENTRY_PCT_STEP,
    ENTRY_WINDOWS,
    MAX_CONCURRENT_RANGE,
    MAX_PER_RULE_RANGE,
    POSITION_SIZE_RANGE,
    POSITION_SIZE_STEP,
    SL_PCT_RANGE,
    SL_PCT_STEP,
    TP_PCT_RANGE,
    TP_PCT_STEP,
    add_common_args,
    build_strategy,
    compute_score,
    evaluate_params,
    export_strategy,
    fetch_candles,
    print_best_params,
    print_result_metrics,
    run_simulation,
    split_candles,
    validate_feed,
    validate_type,
)

# ── Parameter metadata ───────────────────────────────────────────────────────

# Maps param key prefixes to (low, high, step, type)
# type: "float", "int", "categorical"
PARAM_META: dict[str, tuple] = {
    "entry_pct": (*ENTRY_PCT_RANGE, ENTRY_PCT_STEP, "float"),
    "entry_window": (ENTRY_WINDOWS, None, None, "categorical"),
    "tp_pct": (*TP_PCT_RANGE, TP_PCT_STEP, "float"),
    "sl_pct": (*SL_PCT_RANGE, SL_PCT_STEP, "float"),
    "max_per_rule": (*MAX_PER_RULE_RANGE, 1, "int"),
    "max_concurrent": (*MAX_CONCURRENT_RANGE, 1, "int"),
    "position_size": (*POSITION_SIZE_RANGE, POSITION_SIZE_STEP, "float"),
}


def _get_meta(key: str) -> tuple:
    """Get parameter metadata by matching key prefix."""
    for prefix, meta in PARAM_META.items():
        if key == prefix or key.startswith(prefix + "_"):
            return meta
    raise KeyError(f"Unknown parameter key: {key}")


def _snap(value: float, low: float, high: float, step: float) -> float:
    """Snap a float to the nearest step within bounds."""
    value = max(low, min(high, value))
    return round(round((value - low) / step) * step + low, 10)


def _snap_int(value: int, low: int, high: int) -> int:
    """Clamp an int within bounds."""
    return max(low, min(high, value))


# ── Individual operations ────────────────────────────────────────────────────


def random_individual(n_rules: int, rng: random.Random) -> dict:
    """Create a random individual (parameter set)."""
    params: dict = {}
    prev_entry = 0.0

    for i in range(n_rules):
        suffix = f"_{i}" if n_rules > 1 else ""

        # Enforce increasing entry_pct for DCA ladder
        low = max(ENTRY_PCT_RANGE[0], prev_entry + 0.5) if i > 0 else ENTRY_PCT_RANGE[0]
        entry = _snap(
            rng.uniform(low, ENTRY_PCT_RANGE[1]),
            low, ENTRY_PCT_RANGE[1], ENTRY_PCT_STEP,
        )
        prev_entry = entry

        params[f"entry_pct{suffix}"] = entry
        params[f"entry_window{suffix}"] = rng.choice(ENTRY_WINDOWS)
        params[f"tp_pct{suffix}"] = _snap(
            rng.uniform(*TP_PCT_RANGE), *TP_PCT_RANGE, TP_PCT_STEP,
        )
        params[f"sl_pct{suffix}"] = _snap(
            rng.uniform(*SL_PCT_RANGE), *SL_PCT_RANGE, SL_PCT_STEP,
        )
        params[f"max_per_rule{suffix}"] = rng.randint(*MAX_PER_RULE_RANGE)

    params["max_concurrent"] = rng.randint(*MAX_CONCURRENT_RANGE)
    params["position_size"] = _snap(
        rng.uniform(*POSITION_SIZE_RANGE), *POSITION_SIZE_RANGE, POSITION_SIZE_STEP,
    )

    _repair(params, n_rules)
    return params


def _repair(params: dict, n_rules: int) -> None:
    """Fix constraint violations in-place."""
    # Ensure sum(max_per_rule) <= max_concurrent
    total = sum(
        params[f"max_per_rule{'_' + str(i) if n_rules > 1 else ''}"]
        for i in range(n_rules)
    )
    if total > params["max_concurrent"]:
        params["max_concurrent"] = min(total, MAX_CONCURRENT_RANGE[1])

    # Ensure entry_pct is strictly increasing for multi-rule
    if n_rules > 1:
        prev = 0.0
        for i in range(n_rules):
            key = f"entry_pct_{i}"
            low = max(ENTRY_PCT_RANGE[0], prev + 0.5)
            if params[key] <= prev:
                params[key] = _snap(low, low, ENTRY_PCT_RANGE[1], ENTRY_PCT_STEP)
            prev = params[key]


def crossover(p1: dict, p2: dict, n_rules: int, alpha: float, rng: random.Random) -> dict:
    """BLX-α blend crossover for floats, uniform swap for categorical/int."""
    child: dict = {}

    for key in p1:
        meta = _get_meta(key)
        ptype = meta[3]

        if ptype == "categorical":
            child[key] = rng.choice([p1[key], p2[key]])
        elif ptype == "int":
            low, high, step, _ = meta
            # Uniform swap with occasional average
            if rng.random() < 0.5:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
        else:  # float
            low, high, step, _ = meta
            lo_val = min(p1[key], p2[key])
            hi_val = max(p1[key], p2[key])
            d = hi_val - lo_val
            new_val = rng.uniform(lo_val - alpha * d, hi_val + alpha * d)
            child[key] = _snap(new_val, low, high, step)

    _repair(child, n_rules)
    return child


def mutate(individual: dict, n_rules: int, mutation_rate: float, rng: random.Random) -> None:
    """Gaussian mutation for floats, random reset for categorical, ±step for int."""
    for key in individual:
        if rng.random() >= mutation_rate:
            continue

        meta = _get_meta(key)
        ptype = meta[3]

        if ptype == "categorical":
            individual[key] = rng.choice(meta[0])
        elif ptype == "int":
            low, high, _, _ = meta
            individual[key] = _snap_int(
                individual[key] + rng.choice([-2, -1, 1, 2]), low, high,
            )
        else:  # float
            low, high, step, _ = meta
            range_size = high - low
            noise = rng.gauss(0, 0.1 * range_size)
            individual[key] = _snap(individual[key] + noise, low, high, step)

    _repair(individual, n_rules)


def tournament_select(
    population: list[dict],
    fitness: list[float],
    k: int,
    rng: random.Random,
) -> dict:
    """Tournament selection: pick k random, return the fittest."""
    indices = rng.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitness[i])
    return copy.deepcopy(population[best_idx])


# ── Main GA loop ─────────────────────────────────────────────────────────────


def run_ga(
    candles,
    candle_type: str,
    feed: str,
    time_range: str,
    n_rules: int,
    entry_type: str,
    capital: float,
    objective: str,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_size: int,
    elite_count: int,
    seed: int | None,
) -> tuple[dict, float, list[dict]]:
    """Run the genetic algorithm. Returns (best_params, best_score, history)."""
    rng = random.Random(seed)

    # Initialize population
    print(f"  Initializing population of {pop_size}...")
    population = [random_individual(n_rules, rng) for _ in range(pop_size)]

    # Evaluate initial population
    fitness: list[float] = []
    for ind in population:
        score, _ = evaluate_params(
            ind, candles, candle_type, feed, time_range,
            n_rules, entry_type, capital, objective,
        )
        fitness.append(score)

    best_ever_score = max(fitness)
    best_ever_idx = fitness.index(best_ever_score)
    best_ever = copy.deepcopy(population[best_ever_idx])
    history: list[dict] = []

    total_evals = pop_size
    t0 = time.time()

    valid_fitness = [f for f in fitness if f > -999]
    avg = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0
    print(
        f"  Gen   0 │ Best: {best_ever_score:8.2f} │ "
        f"Avg: {avg:8.2f} │ Evals: {total_evals:5d} │ "
        f"Elapsed: {time.time() - t0:.0f}s"
    )
    history.append({
        "gen": 0, "best": best_ever_score, "avg": avg, "evals": total_evals,
    })

    for gen in range(1, generations + 1):
        new_population: list[dict] = []
        new_fitness: list[float] = []

        # Elitism: carry over top individuals
        ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
        for i in ranked[:elite_count]:
            new_population.append(copy.deepcopy(population[i]))
            new_fitness.append(fitness[i])

        # Fill rest via selection + crossover + mutation
        while len(new_population) < pop_size:
            parent1 = tournament_select(population, fitness, tournament_size, rng)
            parent2 = tournament_select(population, fitness, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = crossover(parent1, parent2, n_rules, 0.5, rng)
            else:
                child = copy.deepcopy(parent1 if rng.random() < 0.5 else parent2)

            mutate(child, n_rules, mutation_rate, rng)

            score, _ = evaluate_params(
                child, candles, candle_type, feed, time_range,
                n_rules, entry_type, capital, objective,
            )
            new_population.append(child)
            new_fitness.append(score)
            total_evals += 1

        population = new_population
        fitness = new_fitness

        gen_best = max(fitness)
        if gen_best > best_ever_score:
            best_ever_score = gen_best
            best_ever_idx = fitness.index(gen_best)
            best_ever = copy.deepcopy(population[best_ever_idx])

        valid_fitness = [f for f in fitness if f > -999]
        avg = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0
        history.append({
            "gen": gen, "best": best_ever_score, "avg": avg, "evals": total_evals,
        })

        if gen % 5 == 0 or gen == generations:
            print(
                f"  Gen {gen:3d} │ Best: {best_ever_score:8.2f} │ "
                f"Avg: {avg:8.2f} │ Evals: {total_evals:5d} │ "
                f"Elapsed: {time.time() - t0:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\n  GA complete: {generations} generations, {total_evals} evaluations in {elapsed:.1f}s")
    print(f"  ({elapsed / total_evals:.3f}s per evaluation)")

    return best_ever, best_ever_score, history


def print_convergence(history: list[dict]) -> None:
    """Print a compact convergence summary."""
    print("\n  CONVERGENCE:")
    print("  ┌──────┬──────────┬──────────┬────────┐")
    print("  │  Gen │     Best │      Avg │  Evals │")
    print("  ├──────┼──────────┼──────────┼────────┤")

    # Show first, last, and every ~20% of generations
    total = len(history)
    indices = {0, total - 1}
    for pct in [0.2, 0.4, 0.6, 0.8]:
        indices.add(int(total * pct))

    for i in sorted(indices):
        h = history[i]
        print(
            f"  │ {h['gen']:4d} │ {h['best']:8.2f} │ {h['avg']:8.2f} │ {h['evals']:6d} │"
        )

    print("  └──────┴──────────┴──────────┴────────┘")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm optimization for trading strategy hyperparameters."
    )
    add_common_args(parser)
    parser.add_argument("--pop-size", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover probability (default: 0.8)")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Mutation probability per gene (default: 0.15)")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size (default: 3)")
    parser.add_argument("--elite-count", type=int, default=2, help="Elitism: keep top N unchanged (default: 2)")
    parser.add_argument("--split", type=float, default=0.75, help="Train/validation split ratio (default: 0.75)")
    args = parser.parse_args()

    feed = validate_feed(args.feed)
    candle_type = validate_type(args.candle_type)

    # ── Fetch data ───────────────────────────────────────────────────────
    candles = fetch_candles(feed, candle_type, args.time_range)
    train_candles, val_candles = split_candles(candles, args.split)

    print(f"\nTrain set: {len(train_candles)} candles ({train_candles[0].time.date()} to {train_candles[-1].time.date()})")
    print(f"Val   set: {len(val_candles)} candles ({val_candles[0].time.date()} to {val_candles[-1].time.date()})")

    # ── Run GA ───────────────────────────────────────────────────────────
    total_evals = args.pop_size + args.pop_size * args.generations
    print(f"\n{'=' * 80}")
    print(f"  GENETIC ALGORITHM — {feed} {candle_type}m")
    print(f"  Pop: {args.pop_size} │ Gens: {args.generations} │ ~{total_evals} evals")
    print(f"  Objective: {args.objective} │ Rules: {args.rules} │ Entry: {args.entry_type}")
    print(f"  Crossover: {args.crossover_rate} │ Mutation: {args.mutation_rate} │ Tournament: {args.tournament_size} │ Elite: {args.elite_count}")
    print(f"{'=' * 80}")
    print()

    best_params, best_score, history = run_ga(
        train_candles, candle_type, feed, args.time_range,
        args.rules, args.entry_type, args.capital, args.objective,
        args.pop_size, args.generations, args.crossover_rate,
        args.mutation_rate, args.tournament_size, args.elite_count,
        args.seed,
    )

    # ── Results ──────────────────────────────────────────────────────────
    print_convergence(history)
    print_best_params(best_params, args.rules, best_score)

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
    elif train_score > 0 > val_score:
        print("\n  ⚠️  OVERFITTING WARNING: Strategy is profitable on train but loses on validation.")

    # ── Full-data run ────────────────────────────────────────────────────
    full_result = run_simulation(candles, best_strategy, candle_type, feed, args.time_range)
    print_result_metrics(full_result, f"FULL DATA ({candles[0].time.date()} → {candles[-1].time.date()})")

    # ── Export ────────────────────────────────────────────────────────────
    desc = (
        f"Optimized via Genetic Algorithm ({args.generations} gens × "
        f"{args.pop_size} pop, score={best_score:.4f})"
    )
    path = export_strategy(
        best_params, args.rules, args.entry_type, args.capital, feed, candle_type, desc,
    )
    print(f"\n  Strategy saved to {path}")
    print()


if __name__ == "__main__":
    main()
