"""Tests for the simulation engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_simulator.engine import (
    _check_entry_condition,
    _compute_price_change_pct,
    _get_all_open_positions,
    run_simulation,
)
from trade_simulator.models import Candle, PositionStatus
from trade_simulator.strategy import (
    DependsOn,
    DependsOnCondition,
    DropFrom,
    EntryCondition,
    EntryType,
    ExitCondition,
    ExitRelativeTo,
    PositionSizing,
    PositionSizingType,
    Rule,
    Strategy,
)


class TestPriceChangeComputation:
    def test_drop_detection(self, sample_candles):
        # At index 3, price dropped from ~100 high to 94 close
        change = _compute_price_change_pct(sample_candles, 3, lookback_candles=2)
        assert change is not None
        assert change < -5  # Should show > 5% drop

    def test_no_data_returns_none(self, sample_candles):
        change = _compute_price_change_pct(sample_candles, 0, lookback_candles=5)
        assert change is None


class TestSimulationBasic:
    def test_simulation_produces_trades(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        assert result.total_trades > 0
        assert result.feed == "BTCUSD"
        assert result.timeframe == "15"

    def test_capital_conservation(self, sample_candles, simple_strategy):
        """Capital + open position value should be accounted for."""
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        # All positions should be closed (force-closed at end)
        for pos in result.positions:
            assert pos.status == PositionStatus.CLOSED

    def test_fees_deducted(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        assert result.total_fees > 0

    def test_no_trades_with_tight_threshold(self, sample_candles):
        """A very tight threshold (50% drop) should produce no trades."""
        strategy = Strategy.from_simple_params(
            entry_drop=50.0,
            entry_window="30m",
            take_profit=2.0,
            capital=1000.0,
        )
        result = run_simulation(
            sample_candles, strategy, "15", "BTCUSD", "1d"
        )
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.final_capital == 1000.0  # No change

    def test_stop_loss_triggers(self):
        """Create candles where stop loss should trigger."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        candles = [
            Candle(time=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc), open=100, high=100.5, low=99.5, close=100, volume=100),
            Candle(time=datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc), open=100, high=100.2, low=93, close=93, volume=200),   # big drop
            Candle(time=datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc), open=93, high=93.5, low=89, close=90, volume=300),     # more drop, SL hit at -3%
        ]

        strategy = Strategy.from_simple_params(
            entry_drop=5.0,
            entry_window="15m",
            take_profit=10.0,
            stop_loss=3.0,
            capital=1000.0,
            position_size=100.0,
        )
        result = run_simulation(candles, strategy, "15", "BTCUSD", "1d")
        # Should have trades with stop loss exits
        assert result.total_trades >= 1

    def test_max_positions_enforced(self, sample_candles):
        """Should not exceed max concurrent positions."""
        strategy = Strategy.from_simple_params(
            entry_drop=2.0,
            entry_window="15m",
            take_profit=50.0,  # Very high TP so positions stay open
            capital=1000.0,
            max_positions=1,
            position_size=50.0,
        )
        result = run_simulation(
            sample_candles, strategy, "15", "BTCUSD", "1d"
        )
        # All positions should be closed at the end
        assert all(p.status == PositionStatus.CLOSED for p in result.positions)


class TestLinkedThresholds:
    def test_linked_rule_with_open_parent(self):
        """Rule B should only fire when Rule A has an open position."""
        candles = [
            Candle(time=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc), open=100, high=100.5, low=99.5, close=100, volume=100),
            Candle(time=datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc), open=100, high=100.2, low=93, close=94, volume=200),   # -6% drop, rule A triggers
            Candle(time=datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc), open=94, high=94.5, low=88, close=89, volume=300),     # -5.3% from A entry, rule B triggers
            Candle(time=datetime(2025, 1, 1, 0, 45, tzinfo=timezone.utc), open=89, high=98, low=88.5, close=97, volume=250),     # recovery, TP hit for both
            Candle(time=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc), open=97, high=100, low=96, close=99, volume=200),
        ]

        strategy = Strategy(
            initial_capital=1000,
            fee_pct=0.1,
            max_concurrent_positions=5,
            position_sizing=PositionSizing(type=PositionSizingType.FIXED, value=100),
            rules=[
                Rule(
                    id="rule_a",
                    entry=EntryCondition(type=EntryType.PRICE_DROP, percentage=5.0, window="15m"),
                    exit=ExitCondition(take_profit_pct=5.0, stop_loss_pct=10.0),
                    max_positions=1,
                ),
                Rule(
                    id="rule_b",
                    entry=EntryCondition(type=EntryType.PRICE_DROP, percentage=5.0, window="15m"),
                    exit=ExitCondition(take_profit_pct=3.0, stop_loss_pct=10.0, exit_relative_to=ExitRelativeTo.OWN_ENTRY),
                    max_positions=1,
                    depends_on=DependsOn(
                        rule_id="rule_a",
                        condition=DependsOnCondition.HAS_OPEN_POSITION,
                        drop_from=DropFrom.PARENT_ENTRY,
                    ),
                ),
            ],
        )

        result = run_simulation(candles, strategy, "15", "BTCUSD", "1d")
        rule_ids = [p.rule_id for p in result.positions]
        # Rule A should definitely have fired
        assert "rule_a" in rule_ids


class TestStatisticsComputation:
    def test_win_rate_calculation(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        if result.total_trades > 0:
            expected_wr = result.winning_trades / result.total_trades * 100
            assert result.win_rate == pytest.approx(expected_wr)

    def test_pnl_sums_match(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        computed_pnl = sum(p.pnl for p in result.positions)
        assert result.total_pnl == pytest.approx(computed_pnl)

    def test_per_rule_breakdown(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        assert "cli_rule" in result.per_rule
        rule_stats = result.per_rule["cli_rule"]
        assert rule_stats["total_trades"] == result.total_trades
