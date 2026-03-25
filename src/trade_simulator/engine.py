"""Core simulation engine for trading backtesting."""

from __future__ import annotations

import logging
from datetime import datetime

from trade_simulator.api import get_candle_minutes
from trade_simulator.models import Candle, Position, PositionStatus, TradeResult
from trade_simulator.strategy import (
    DependsOnCondition,
    DropFrom,
    EntryType,
    ExitRelativeTo,
    PositionSizingType,
    Rule,
    Strategy,
)

logger = logging.getLogger(__name__)


def _compute_price_change_pct(
    candles: list[Candle],
    current_idx: int,
    lookback_candles: int,
) -> float | None:
    """Compute the percentage price change over a lookback window.

    Returns the percentage change from the high of the lookback window
    to the current close (for drops) or from the low to the current close
    (for rises).
    """
    start_idx = max(0, current_idx - lookback_candles)
    if start_idx >= current_idx:
        return None

    window = candles[start_idx : current_idx + 1]
    if len(window) < 2:
        return None

    # For drop detection: compare current close to highest point in window
    highest = max(c.high for c in window[:-1])
    current_close = candles[current_idx].close
    return ((current_close - highest) / highest) * 100


def _compute_price_rise_pct(
    candles: list[Candle],
    current_idx: int,
    lookback_candles: int,
) -> float | None:
    """Compute the rise percentage from the lowest point in the window."""
    start_idx = max(0, current_idx - lookback_candles)
    if start_idx >= current_idx:
        return None

    window = candles[start_idx : current_idx + 1]
    if len(window) < 2:
        return None

    lowest = min(c.low for c in window[:-1])
    current_close = candles[current_idx].close
    return ((current_close - lowest) / lowest) * 100


def _get_open_positions_for_rule(positions: list[Position], rule_id: str) -> list[Position]:
    """Return open positions belonging to a specific rule."""
    return [p for p in positions if p.rule_id == rule_id and p.status == PositionStatus.OPEN]


def _get_all_open_positions(positions: list[Position]) -> list[Position]:
    """Return all open positions."""
    return [p for p in positions if p.status == PositionStatus.OPEN]


def _check_entry_condition(
    rule: Rule,
    candles: list[Candle],
    current_idx: int,
    candle_minutes: int,
    positions: list[Position],
) -> bool:
    """Check whether the entry condition for a rule is met."""
    window_minutes = rule.entry.window_minutes
    lookback = max(1, window_minutes // candle_minutes)

    if rule.entry.type == EntryType.PRICE_DROP:
        change_pct = _compute_price_change_pct(candles, current_idx, lookback)
        if change_pct is None:
            return False
        # change_pct is negative for drops, threshold is positive
        return change_pct <= -rule.entry.percentage

    elif rule.entry.type == EntryType.PRICE_RISE:
        rise_pct = _compute_price_rise_pct(candles, current_idx, lookback)
        if rise_pct is None:
            return False
        return rise_pct >= rule.entry.percentage

    return False


def _check_linked_entry(
    rule: Rule,
    candles: list[Candle],
    current_idx: int,
    candle_minutes: int,
    positions: list[Position],
) -> bool:
    """Check linked (depends_on) entry condition."""
    dep = rule.depends_on
    if dep is None:
        return True  # No dependency, always passes

    parent_positions = [
        p for p in positions if p.rule_id == dep.rule_id
    ]

    if dep.condition == DependsOnCondition.HAS_OPEN_POSITION:
        parent_open = [p for p in parent_positions if p.status == PositionStatus.OPEN]
        if not parent_open:
            return False
    elif dep.condition == DependsOnCondition.HAS_CLOSED_POSITION:
        parent_closed = [p for p in parent_positions if p.status == PositionStatus.CLOSED]
        if not parent_closed:
            return False

    # Check additional drop from parent's entry price
    if dep.drop_from == DropFrom.PARENT_ENTRY:
        parent_open = [p for p in parent_positions if p.status == PositionStatus.OPEN]
        if not parent_open:
            return False
        parent_entry_price = parent_open[-1].entry_price  # Most recent parent
        current_price = candles[current_idx].close
        drop_pct = ((current_price - parent_entry_price) / parent_entry_price) * 100

        if rule.entry.type == EntryType.PRICE_DROP:
            return drop_pct <= -rule.entry.percentage
        elif rule.entry.type == EntryType.PRICE_RISE:
            return drop_pct >= rule.entry.percentage

    # drop_from == CURRENT_PRICE: use the normal window-based check
    return _check_entry_condition(rule, candles, current_idx, candle_minutes, positions)


def _compute_position_size(
    strategy: Strategy,
    current_capital: float,
) -> float | None:
    """Compute the USD size for a new position. Returns None if insufficient capital."""
    sizing = strategy.position_sizing

    if sizing.type == PositionSizingType.FIXED:
        size = sizing.value
    else:
        size = current_capital * (sizing.value / 100)

    if size <= 0 or size > current_capital:
        return None
    return size


def _check_exit_conditions(
    position: Position,
    candle: Candle,
    rule: Rule,
    positions: list[Position],
) -> tuple[bool, float]:
    """Check if an open position should be closed.

    Returns (should_close, exit_price).
    Uses candle high/low for intra-candle exit detection.
    """
    exit_cond = rule.exit

    # Determine reference price for the exit
    if exit_cond.exit_relative_to == ExitRelativeTo.PARENT_ENTRY and rule.depends_on:
        parent_positions = [
            p for p in positions
            if p.rule_id == rule.depends_on.rule_id and p.status == PositionStatus.OPEN
        ]
        if parent_positions:
            ref_price = parent_positions[-1].entry_price
        else:
            ref_price = position.entry_price
    else:
        ref_price = position.entry_price

    # Take profit check (using candle high)
    tp_price = ref_price * (1 + exit_cond.take_profit_pct / 100)
    if candle.high >= tp_price:
        return True, tp_price

    # Stop loss check (using candle low)
    if exit_cond.stop_loss_pct is not None:
        sl_price = ref_price * (1 - exit_cond.stop_loss_pct / 100)
        if candle.low <= sl_price:
            return True, sl_price

    return False, 0.0


def run_simulation(
    candles: list[Candle],
    strategy: Strategy,
    candle_type: str,
    feed: str,
    time_range: str,
) -> TradeResult:
    """Run the backtesting simulation.

    Args:
        candles: List of OHLCV candles sorted by time ascending.
        strategy: The trading strategy to simulate.
        candle_type: The candle type string (e.g., '15', '1h').
        feed: The feed name (e.g., 'BTCUSD').
        time_range: The human-readable time range (e.g., '30d').

    Returns:
        TradeResult with all simulation statistics.
    """
    candle_minutes = get_candle_minutes(candle_type)
    positions: list[Position] = []
    capital = strategy.initial_capital
    peak_capital = capital
    max_drawdown = 0.0

    # Build a rule lookup
    rules_by_id: dict[str, Rule] = {r.id: r for r in strategy.rules}

    logger.info(
        "Starting simulation: %s %s, %d candles, capital=$%.2f",
        feed, candle_type, len(candles), capital,
    )

    for idx, candle in enumerate(candles):
        # --- Phase 1: Check exits on all open positions ---
        open_positions = _get_all_open_positions(positions)
        for pos in open_positions:
            rule = rules_by_id[pos.rule_id]
            should_close, exit_price = _check_exit_conditions(
                pos, candle, rule, positions
            )
            if should_close:
                pos.close(exit_price, candle.time, strategy.fee_pct)
                capital += pos.pnl + pos.size_usd + pos.entry_fee  # Return principal + pnl
                logger.debug(
                    "SELL %s @ %.4f (rule=%s, pnl=%.2f)",
                    feed, exit_price, pos.rule_id, pos.pnl,
                )

        # --- Phase 2: Check entries ---
        open_count = len(_get_all_open_positions(positions))

        for rule in strategy.rules:
            # Global position limit
            if open_count >= strategy.max_concurrent_positions:
                break

            # Per-rule position limit
            rule_open = _get_open_positions_for_rule(positions, rule.id)
            if len(rule_open) >= rule.max_positions:
                continue

            # Check linked dependency first
            if rule.depends_on:
                if not _check_linked_entry(
                    rule, candles, idx, candle_minutes, positions
                ):
                    continue
            else:
                # Normal entry check
                if not _check_entry_condition(
                    rule, candles, idx, candle_minutes, positions
                ):
                    continue

            # Compute position size
            size = _compute_position_size(strategy, capital)
            if size is None:
                continue

            entry_price = candle.close
            entry_fee = size * (strategy.fee_pct / 100)
            quantity = (size - entry_fee) / entry_price

            pos = Position(
                rule_id=rule.id,
                entry_price=entry_price,
                entry_time=candle.time,
                size_usd=size,
                quantity=quantity,
                entry_fee=entry_fee,
            )
            positions.append(pos)
            capital -= size
            open_count += 1

            logger.debug(
                "BUY %s @ %.4f (rule=%s, size=$%.2f)",
                feed, entry_price, pos.rule_id, size,
            )

        # --- Track drawdown ---
        # Compute equity: capital + value of open positions
        equity = capital + sum(
            p.quantity * candle.close
            for p in _get_all_open_positions(positions)
        )
        if equity > peak_capital:
            peak_capital = equity
        drawdown = peak_capital - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # --- Force-close any remaining open positions at last candle's close ---
    if candles:
        last_candle = candles[-1]
        for pos in _get_all_open_positions(positions):
            pos.close(last_candle.close, last_candle.time, strategy.fee_pct)
            capital += pos.pnl + pos.size_usd + pos.entry_fee

    # --- Compute statistics ---
    closed = [p for p in positions if p.status == PositionStatus.CLOSED]
    wins = [p for p in closed if p.pnl > 0]
    losses = [p for p in closed if p.pnl <= 0]

    total_pnl = sum(p.pnl for p in closed)
    total_fees = sum(p.entry_fee + p.exit_fee for p in closed)
    win_pnls = [p.pnl for p in wins]
    loss_pnls = [p.pnl for p in losses]

    gross_profit = sum(p.pnl for p in wins) if wins else 0.0
    gross_loss = abs(sum(p.pnl for p in losses)) if losses else 0.0

    # Per-rule breakdown
    per_rule: dict[str, dict] = {}
    for rule in strategy.rules:
        rule_positions = [p for p in closed if p.rule_id == rule.id]
        rule_wins = [p for p in rule_positions if p.pnl > 0]
        rule_losses = [p for p in rule_positions if p.pnl <= 0]
        per_rule[rule.id] = {
            "total_trades": len(rule_positions),
            "winning_trades": len(rule_wins),
            "losing_trades": len(rule_losses),
            "win_rate": (len(rule_wins) / len(rule_positions) * 100) if rule_positions else 0.0,
            "total_pnl": sum(p.pnl for p in rule_positions),
        }

    max_dd_pct = (max_drawdown / peak_capital * 100) if peak_capital > 0 else 0.0

    return TradeResult(
        feed=feed,
        timeframe=candle_type,
        time_range=time_range,
        initial_capital=strategy.initial_capital,
        final_capital=capital,
        total_trades=len(closed),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=(len(wins) / len(closed) * 100) if closed else 0.0,
        total_pnl=total_pnl,
        total_pnl_pct=(total_pnl / strategy.initial_capital * 100) if strategy.initial_capital > 0 else 0.0,
        total_fees=total_fees,
        avg_win=(sum(win_pnls) / len(win_pnls)) if win_pnls else 0.0,
        avg_loss=(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0,
        best_trade=max(win_pnls) if win_pnls else 0.0,
        worst_trade=min(loss_pnls) if loss_pnls else 0.0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_dd_pct,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
        per_rule=per_rule,
        positions=closed,
    )
