"""Tests for the strategy schema and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trade_simulator.strategy import (
    EntryCondition,
    EntryType,
    ExitCondition,
    PositionSizing,
    PositionSizingType,
    Rule,
    Strategy,
    parse_window_minutes,
)


class TestParseWindowMinutes:
    def test_standard_windows(self):
        assert parse_window_minutes("1m") == 1
        assert parse_window_minutes("5m") == 5
        assert parse_window_minutes("15m") == 15
        assert parse_window_minutes("30m") == 30
        assert parse_window_minutes("1h") == 60
        assert parse_window_minutes("4h") == 240
        assert parse_window_minutes("1d") == 1440

    def test_custom_windows(self):
        assert parse_window_minutes("45m") == 45
        assert parse_window_minutes("2h") == 120
        assert parse_window_minutes("3d") == 4320

    def test_case_insensitive(self):
        assert parse_window_minutes("1H") == 60
        assert parse_window_minutes("30M") == 30

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="Invalid window format"):
            parse_window_minutes("abc")


class TestStrategyValidation:
    def test_minimal_strategy(self):
        s = Strategy(
            rules=[
                Rule(
                    id="test",
                    entry=EntryCondition(type=EntryType.PRICE_DROP, percentage=5, window="30m"),
                    exit=ExitCondition(take_profit_pct=2),
                )
            ]
        )
        assert s.initial_capital == 1000.0
        assert s.fee_pct == 0.1
        assert len(s.rules) == 1

    def test_duplicate_rule_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate rule IDs"):
            Strategy(
                rules=[
                    Rule(id="dup", entry=EntryCondition(percentage=5, window="30m"), exit=ExitCondition(take_profit_pct=2)),
                    Rule(id="dup", entry=EntryCondition(percentage=3, window="1h"), exit=ExitCondition(take_profit_pct=1)),
                ]
            )

    def test_self_referencing_depends_on_rejected(self):
        with pytest.raises(ValueError, match="cannot depend on itself"):
            Strategy(
                rules=[
                    Rule(
                        id="self_ref",
                        entry=EntryCondition(percentage=5, window="30m"),
                        exit=ExitCondition(take_profit_pct=2),
                        depends_on={"rule_id": "self_ref", "condition": "has_open_position", "drop_from": "parent_entry"},
                    )
                ]
            )

    def test_unknown_depends_on_rejected(self):
        with pytest.raises(ValueError, match="depends on unknown rule_id"):
            Strategy(
                rules=[
                    Rule(
                        id="child",
                        entry=EntryCondition(percentage=5, window="30m"),
                        exit=ExitCondition(take_profit_pct=2),
                        depends_on={"rule_id": "nonexistent", "condition": "has_open_position", "drop_from": "parent_entry"},
                    )
                ]
            )

    def test_no_rules_rejected(self):
        with pytest.raises(ValueError):
            Strategy(rules=[])


class TestStrategyFromFile:
    def test_load_example_strategy(self, example_strategy_path):
        path = Path(example_strategy_path)
        if path.exists():
            s = Strategy.from_file(path)
            assert len(s.rules) >= 1
            assert s.initial_capital > 0

    def test_load_from_dict(self):
        data = {
            "initial_capital": 500,
            "fee_pct": 0.05,
            "max_concurrent_positions": 2,
            "position_sizing": {"type": "percentage", "value": 50},
            "rules": [
                {
                    "id": "r1",
                    "entry": {"type": "price_drop", "percentage": 3, "window": "15m"},
                    "exit": {"take_profit_pct": 1.5, "stop_loss_pct": 2},
                    "max_positions": 1,
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            s = Strategy.from_file(f.name)

        assert s.initial_capital == 500
        assert s.fee_pct == 0.05
        assert s.position_sizing.type == PositionSizingType.PERCENTAGE
        assert s.position_sizing.value == 50


class TestStrategySimpleParams:
    def test_from_simple_params(self):
        s = Strategy.from_simple_params(
            entry_drop=5,
            entry_window="1h",
            take_profit=3,
            stop_loss=2,
            capital=2000,
        )
        assert s.initial_capital == 2000
        assert len(s.rules) == 1
        assert s.rules[0].id == "cli_rule"
        assert s.rules[0].entry.percentage == 5
        assert s.rules[0].entry.window == "1h"
        assert s.rules[0].exit.take_profit_pct == 3
        assert s.rules[0].exit.stop_loss_pct == 2
