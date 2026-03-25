"""Tests for the report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trade_simulator.engine import run_simulation
from trade_simulator.report import generate_summary, print_summary
from trade_simulator.strategy import Strategy


class TestGenerateSummary:
    def test_summary_json_created(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            output = generate_summary(result, path)
            assert output.exists()

            with open(output) as f:
                data = json.load(f)

            assert data["feed"] == "BTCUSD"
            assert data["timeframe"] == "15"
            assert "performance" in data
            assert "trades" in data
            assert "risk" in data
            assert "per_rule" in data
            assert isinstance(data["trades_log"], list)

    def test_summary_values_match_result(self, sample_candles, simple_strategy):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            generate_summary(result, path)

            with open(path) as f:
                data = json.load(f)

            assert data["capital"]["initial"] == result.initial_capital
            assert data["trades"]["total"] == result.total_trades
            assert data["trades"]["win_rate"] == round(result.win_rate, 2)


class TestPrintSummary:
    def test_print_summary_no_crash(self, sample_candles, simple_strategy, capsys):
        result = run_simulation(
            sample_candles, simple_strategy, "15", "BTCUSD", "1d"
        )
        print_summary(result)
        captured = capsys.readouterr()
        assert "TRADE SIMULATION SUMMARY" in captured.out
        assert "BTCUSD" in captured.out

    def test_print_zero_trades(self, sample_candles):
        strategy = Strategy.from_simple_params(
            entry_drop=99.0,
            entry_window="30m",
            take_profit=2.0,
            capital=1000,
        )
        result = run_simulation(
            sample_candles, strategy, "15", "BTCUSD", "1d"
        )
        # Should not crash even with zero trades
        from trade_simulator.report import print_summary as ps
        ps(result)
