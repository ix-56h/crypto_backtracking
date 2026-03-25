"""Strategy schema and validation using Pydantic."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PositionSizingType(str, Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"


class EntryType(str, Enum):
    PRICE_DROP = "price_drop"
    PRICE_RISE = "price_rise"


class ExitRelativeTo(str, Enum):
    OWN_ENTRY = "own_entry"
    PARENT_ENTRY = "parent_entry"


class DependsOnCondition(str, Enum):
    HAS_OPEN_POSITION = "has_open_position"
    HAS_CLOSED_POSITION = "has_closed_position"


class DropFrom(str, Enum):
    PARENT_ENTRY = "parent_entry"
    CURRENT_PRICE = "current_price"


VALID_WINDOWS = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


def parse_window_minutes(window: str) -> int:
    """Convert a window string like '30m' or '1h' to minutes."""
    w = window.lower().strip()
    if w in VALID_WINDOWS:
        return VALID_WINDOWS[w]
    # Try parsing numeric suffixes
    if w.endswith("m"):
        return int(w[:-1])
    if w.endswith("h"):
        return int(w[:-1]) * 60
    if w.endswith("d"):
        return int(w[:-1]) * 1440
    raise ValueError(f"Invalid window format: {window!r}. Use e.g. '30m', '1h', '1d'.")


class PositionSizing(BaseModel):
    type: PositionSizingType = PositionSizingType.FIXED
    value: float = Field(gt=0, description="Dollar amount (fixed) or percentage (1-100)")


class EntryCondition(BaseModel):
    type: EntryType = EntryType.PRICE_DROP
    percentage: float = Field(gt=0, description="Required price change percentage")
    window: str = Field(description="Lookback window e.g. '30m', '1h'")

    @property
    def window_minutes(self) -> int:
        return parse_window_minutes(self.window)


class ExitCondition(BaseModel):
    take_profit_pct: float = Field(gt=0, description="Take profit percentage from entry")
    stop_loss_pct: float | None = Field(default=None, gt=0, description="Stop loss percentage from entry")
    exit_relative_to: ExitRelativeTo = ExitRelativeTo.OWN_ENTRY


class DependsOn(BaseModel):
    rule_id: str = Field(description="ID of the parent rule")
    condition: DependsOnCondition = DependsOnCondition.HAS_OPEN_POSITION
    drop_from: DropFrom = DropFrom.PARENT_ENTRY


class Rule(BaseModel):
    id: str = Field(description="Unique identifier for this rule")
    entry: EntryCondition
    exit: ExitCondition
    max_positions: int = Field(default=1, ge=1, description="Max concurrent positions for this rule")
    depends_on: DependsOn | None = None


class Strategy(BaseModel):
    initial_capital: float = Field(default=1000.0, gt=0)
    fee_pct: float = Field(default=0.1, ge=0, description="Trading fee percentage (applied on entry and exit)")
    max_concurrent_positions: int = Field(default=5, ge=1, description="Global max concurrent open positions")
    position_sizing: PositionSizing = Field(default_factory=lambda: PositionSizing(type=PositionSizingType.FIXED, value=100))
    rules: list[Rule] = Field(min_length=1, description="At least one trading rule is required")

    @model_validator(mode="after")
    def validate_rule_references(self) -> Strategy:
        """Ensure all depends_on.rule_id references point to existing rules."""
        rule_ids = {r.id for r in self.rules}
        for rule in self.rules:
            if rule.depends_on and rule.depends_on.rule_id not in rule_ids:
                raise ValueError(
                    f"Rule '{rule.id}' depends on unknown rule_id "
                    f"'{rule.depends_on.rule_id}'. Available: {rule_ids}"
                )
            if rule.depends_on and rule.depends_on.rule_id == rule.id:
                raise ValueError(f"Rule '{rule.id}' cannot depend on itself.")
        # Check for duplicate rule IDs
        if len(rule_ids) != len(self.rules):
            raise ValueError("Duplicate rule IDs found.")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Strategy:
        """Load a strategy from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_simple_params(
        cls,
        *,
        entry_drop: float,
        entry_window: str = "30m",
        take_profit: float,
        stop_loss: float | None = None,
        capital: float = 1000.0,
        fee_pct: float = 0.1,
        max_positions: int = 3,
        position_size: float = 100.0,
        position_sizing_type: str = "fixed",
    ) -> Strategy:
        """Build a simple single-rule strategy from CLI parameters."""
        return cls(
            initial_capital=capital,
            fee_pct=fee_pct,
            max_concurrent_positions=max_positions,
            position_sizing=PositionSizing(
                type=PositionSizingType(position_sizing_type),
                value=position_size,
            ),
            rules=[
                Rule(
                    id="cli_rule",
                    entry=EntryCondition(
                        type=EntryType.PRICE_DROP,
                        percentage=entry_drop,
                        window=entry_window,
                    ),
                    exit=ExitCondition(
                        take_profit_pct=take_profit,
                        stop_loss_pct=stop_loss,
                    ),
                    max_positions=max_positions,
                )
            ],
        )
