"""Data models for the trade simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Candle:
    """Single OHLCV candle from the API."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_api(cls, data: dict) -> Candle:
        """Create a Candle from an API response dict."""
        return cls(
            time=datetime.fromtimestamp(data["time"] / 1000, tz=timezone.utc),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
        )


@dataclass
class Position:
    """A single trading position."""

    rule_id: str
    entry_price: float
    entry_time: datetime
    size_usd: float
    quantity: float
    entry_fee: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_fee: float = 0.0
    pnl: float = 0.0

    def close(self, exit_price: float, exit_time: datetime, fee_pct: float) -> None:
        """Close the position at the given price and compute PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = PositionStatus.CLOSED
        exit_value = self.quantity * exit_price
        self.exit_fee = exit_value * (fee_pct / 100)
        self.pnl = exit_value - self.exit_fee - self.size_usd - self.entry_fee


@dataclass
class TradeResult:
    """Aggregated results from a simulation run."""

    feed: str
    timeframe: str
    time_range: str
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    total_fees: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    avg_duration_minutes: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    per_rule: dict[str, dict] = field(default_factory=dict)
    positions: list[Position] = field(default_factory=list)
