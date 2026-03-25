"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_simulator.models import Candle
from trade_simulator.strategy import Strategy


@pytest.fixture
def sample_candles() -> list[Candle]:
    """Generate a sequence of candles simulating a price drop then recovery.

    Timeline (15-minute candles):
    0: 100 → 100 (baseline)
    1: 100 → 99
    2: 99  → 97   (-3% from peak)
    3: 97  → 94   (-6% from peak, triggers 5% drop)
    4: 94  → 92   (-8% from peak)
    5: 92  → 95   (recovery start)
    6: 95  → 97   (recovery continues)
    7: 97  → 99   (close to TP)
    8: 99  → 102  (exceeds 2% TP from ~94 entry → ~95.88)
    9: 102 → 101
    """
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    data = [
        # (open, high, low, close, volume)
        (100.0, 100.5, 99.5, 100.0, 1000),  # 0
        (100.0, 100.2, 98.8, 99.0, 1200),   # 1
        (99.0, 99.5, 96.5, 97.0, 1500),     # 2
        (97.0, 97.5, 93.5, 94.0, 2000),     # 3 - drop triggers
        (94.0, 94.5, 91.5, 92.0, 2500),     # 4
        (92.0, 95.5, 91.0, 95.0, 1800),     # 5
        (95.0, 97.5, 94.5, 97.0, 1600),     # 6
        (97.0, 99.5, 96.5, 99.0, 1400),     # 7
        (99.0, 102.5, 98.5, 102.0, 1300),   # 8 - TP hit
        (102.0, 103.0, 100.5, 101.0, 1100), # 9
    ]

    candles = []
    for i, (o, h, l, c, v) in enumerate(data):
        total_minutes = i * 15
        hour = base_time.hour + total_minutes // 60
        minute = total_minutes % 60
        t = datetime(
            base_time.year, base_time.month, base_time.day,
            hour, minute, tzinfo=timezone.utc,
        )
        candles.append(Candle(time=t, open=o, high=h, low=l, close=c, volume=v))
    return candles


@pytest.fixture
def simple_strategy() -> Strategy:
    """A basic strategy: buy on 5% drop in 30min, sell at 2% profit or 3% loss."""
    return Strategy.from_simple_params(
        entry_drop=5.0,
        entry_window="30m",
        take_profit=2.0,
        stop_loss=3.0,
        capital=1000.0,
        fee_pct=0.1,
        max_positions=3,
        position_size=100.0,
    )


@pytest.fixture
def example_strategy_path() -> str:
    """Path to the example strategy file."""
    return "strategies/example.json"


@pytest.fixture
def sample_api_response() -> dict:
    """Mock API response matching the real format."""
    return {
        "result": [
            {
                "time": 1774180800000,
                "open": 68209.2,
                "high": 68329.16,
                "low": 68189.15,
                "close": 68304.58,
                "volume": 278463.253779,
            },
            {
                "time": 1774181700000,
                "open": 68304.58,
                "high": 68749.04,
                "low": 68302.03,
                "close": 68650.49,
                "volume": 1253822.107914,
            },
            {
                "time": 1774182600000,
                "open": 68650.49,
                "high": 68662.53,
                "low": 68582.33,
                "close": 68599.37,
                "volume": 1755474.828286,
            },
        ]
    }
