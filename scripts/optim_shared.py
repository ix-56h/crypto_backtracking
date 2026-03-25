"""Shared utilities for strategy optimization scripts.

Provides common search-space constants, strategy builders, scoring functions,
data fetching, and output formatters used by optimize.py, optimize_ga.py,
and optimize_wfa.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type  # noqa: E402
from trade_simulator.engine import run_simulation  # noqa: E402
from trade_simulator.models import Candle, TradeResult  # noqa: E402
from trade_simulator.strategy import Strategy  # noqa: E402

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


# ── Data helpers ─────────────────────────────────────────────────────────────


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


# ── Strategy builders ────────────────────────────────────────────────────────


def build_strategy(
    params: dict,
    n_rules: int,
    entry_type: str,
    capital: float,
) -> Strategy:
    """Build a Strategy from a parameter dictionary."""
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


# ── Scoring ──────────────────────────────────────────────────────────────────


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


def evaluate_params(
    params: dict,
    candles: list[Candle],
    candle_type: str,
    feed: str,
    time_range: str,
    n_rules: int,
    entry_type: str,
    capital: float,
    objective: str,
) -> tuple[float, TradeResult]:
    """Build strategy from params, simulate, return (score, result)."""
    strategy = build_strategy(params, n_rules, entry_type, capital)
    result = run_simulation(candles, strategy, candle_type, feed, time_range)
    score = compute_score(result, objective)
    return score, result


# ── Output formatters ────────────────────────────────────────────────────────


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


def print_best_params(params: dict, n_rules: int, score: float, label: str = "BEST PARAMETERS") -> None:
    """Print the best parameters in a neat table."""
    print(f"\n  {label} (Score: {score:.4f}):")
    print("  ┌──────────────────────────────────┬──────────────┐")

    for i in range(n_rules):
        suffix = f"_{i}" if n_rules > 1 else ""
        rlabel = f" (rule {i})" if n_rules > 1 else ""
        print(f"  │ Entry %{rlabel:24s} │ {params[f'entry_pct{suffix}']:>11.2f}% │")
        print(f"  │ Entry Window{rlabel:19s} │ {params[f'entry_window{suffix}']:>12s} │")
        print(f"  │ Take Profit %{rlabel:18s} │ {params[f'tp_pct{suffix}']:>11.2f}% │")
        print(f"  │ Stop Loss %{rlabel:20s} │ {params[f'sl_pct{suffix}']:>11.2f}% │")
        print(f"  │ Max Positions/Rule{rlabel:13s} │ {params[f'max_per_rule{suffix}']:>12d} │")
        if i < n_rules - 1:
            print("  ├──────────────────────────────────┼──────────────┤")

    print("  ├──────────────────────────────────┼──────────────┤")
    print(f"  │ Max Concurrent Positions         │ {params['max_concurrent']:>12d} │")
    print(f"  │ Position Size ($)                │ {params['position_size']:>12.0f} │")
    print("  └──────────────────────────────────┴──────────────┘")


def export_strategy(
    params: dict,
    n_rules: int,
    entry_type: str,
    capital: float,
    feed: str,
    candle_type: str,
    description: str,
) -> Path:
    """Save a strategy parameter dict as a JSON file."""
    strategy = build_strategy(params, n_rules, entry_type, capital)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"optimized_{feed}_{candle_type}m.json"

    strategy_dict = {
        "_description": description,
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


def _snap_to_range(value: float, low: float, high: float, step: float) -> float:
    """Clamp value to [low, high] and snap to nearest step."""
    value = max(low, min(high, value))
    return round(round((value - low) / step) * step + low, 10)


def _nearest_window(window: str) -> str:
    """Map an arbitrary window string to the nearest valid ENTRY_WINDOWS value."""
    if window in ENTRY_WINDOWS:
        return window
    # Parse window to minutes for comparison
    _units = {"m": 1, "h": 60, "d": 1440}
    def _to_min(w: str) -> int:
        for u, m in _units.items():
            if w.endswith(u):
                return int(w[:-len(u)]) * m
        return 60  # fallback to 1h
    target = _to_min(window)
    return min(ENTRY_WINDOWS, key=lambda w: abs(_to_min(w) - target))


def extract_params_from_strategy(strategy: Strategy) -> tuple[dict, int, str]:
    """Extract optimizer param dict from a Strategy object.

    Returns (params_dict, n_rules, entry_type).
    Values are clamped to the optimizer search space bounds.
    """
    params: dict = {}
    n_rules = len(strategy.rules)
    entry_type = strategy.rules[0].entry.type.value

    for i, rule in enumerate(strategy.rules):
        suffix = f"_{i}" if n_rules > 1 else ""
        params[f"entry_pct{suffix}"] = _snap_to_range(
            rule.entry.percentage, *ENTRY_PCT_RANGE, ENTRY_PCT_STEP,
        )
        params[f"entry_window{suffix}"] = _nearest_window(rule.entry.window)
        params[f"tp_pct{suffix}"] = _snap_to_range(
            rule.exit.take_profit_pct, *TP_PCT_RANGE, TP_PCT_STEP,
        )
        params[f"sl_pct{suffix}"] = _snap_to_range(
            rule.exit.stop_loss_pct or SL_PCT_RANGE[0], *SL_PCT_RANGE, SL_PCT_STEP,
        )
        params[f"max_per_rule{suffix}"] = max(
            MAX_PER_RULE_RANGE[0], min(MAX_PER_RULE_RANGE[1], rule.max_positions),
        )

    params["max_concurrent"] = max(
        MAX_CONCURRENT_RANGE[0],
        min(MAX_CONCURRENT_RANGE[1], strategy.max_concurrent_positions),
    )
    params["position_size"] = _snap_to_range(
        strategy.position_sizing.value, *POSITION_SIZE_RANGE, POSITION_SIZE_STEP,
    )

    return params, n_rules, entry_type


def load_seed_strategy(args) -> dict | None:
    """Load a strategy file as seed params and override args accordingly.

    Returns the seed params dict, or None if --strategy was not provided.
    Mutates args in-place to set rules, entry_type, and capital from the strategy.
    """
    if not args.strategy:
        return None

    from pathlib import Path

    strategy = Strategy.from_file(args.strategy)
    seed_params, n_rules, entry_type = extract_params_from_strategy(strategy)

    args.rules = n_rules
    args.entry_type = entry_type
    args.capital = strategy.initial_capital

    name = Path(args.strategy).name
    print(f"\n  Seed strategy: {name} ({n_rules} rule{'s' if n_rules > 1 else ''}, {entry_type})")
    for i in range(n_rules):
        suffix = f"_{i}" if n_rules > 1 else ""
        print(
            f"    Rule {i}: entry={seed_params[f'entry_pct{suffix}']:.2f}% "
            f"window={seed_params[f'entry_window{suffix}']} "
            f"TP={seed_params[f'tp_pct{suffix}']:.2f}% "
            f"SL={seed_params[f'sl_pct{suffix}']:.2f}%"
        )
    print(
        f"    Global: max_concurrent={seed_params['max_concurrent']} "
        f"position_size=${seed_params['position_size']:.0f} "
        f"capital=${args.capital:.0f}"
    )

    return seed_params


def add_common_args(parser) -> None:
    """Add CLI arguments shared across all optimizer scripts."""
    parser.add_argument("--feed", required=True, choices=["BTCUSD", "SOLUSD", "ETHUSD"])
    parser.add_argument("--type", dest="candle_type", required=True, choices=["5", "15"])
    parser.add_argument("--range", dest="time_range", default="1y", help="Data range (default: 1y)")
    parser.add_argument("--strategy", default=None,
                        help="Path to strategy JSON to use as optimization seed")
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
