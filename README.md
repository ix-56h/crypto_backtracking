# Trade Simulator

A trading backtesting simulator that fetches OHLCV data from the [Jup.ag Oracle](https://history.oraclesecurity.org) and replays JSON-defined strategies against historical price action. Supports multi-rule entries, DCA ladders, cascade (linked) rules, and configurable risk management.

## Installation

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the `trade-sim` CLI and all dependencies.

## Quick Start

### Run with a strategy file

```bash
uv run trade-sim run \
  --feed BTCUSD --type 15 --range 1y \
  --strategy strategies/champions/02_deep_guarded.json
```

### Run with inline parameters

```bash
uv run trade-sim run \
  --feed SOLUSD --type 15 --range 30d \
  --entry-drop 5.0 --entry-window 4h \
  --take-profit 3.0 --stop-loss 12.0 \
  --capital 5000 --fee 0.1 --max-positions 5 --position-size 250
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--feed` | Trading pair: `BTCUSD`, `SOLUSD`, `ETHUSD` | *required* |
| `--type` | Candle width: `1`, `5`, `15`, `1h`, `1D` | *required* |
| `--range` | Lookback: `1d`, `7d`, `30d`, `6m`, `1y` | *required* |
| `--strategy` | Path to JSON strategy file | — |
| `--entry-drop` | Entry drop % (simple mode) | — |
| `--entry-window` | Lookback window for entry (simple mode) | `30m` |
| `--take-profit` | Take profit % (simple mode) | — |
| `--stop-loss` | Stop loss % (simple mode) | — |
| `--capital` | Initial capital in USD | `1000` |
| `--fee` | Fee per side (%) | `0.1` |
| `--max-positions` | Max concurrent open positions | `3` |
| `--position-size` | Position size in USD | `100` |
| `--position-sizing` | `fixed` or `percentage` | `fixed` |
| `--output-dir` | Directory for results | `output/` |
| `-v` / `--verbose` | Enable debug logging | off |

### Output

Each run produces two files in the output directory:

- **`<FEED>_<timestamp>_chart.html`** — Interactive Plotly chart (candlestick + volume + equity curve with buy/sell markers).
- **`<FEED>_<timestamp>_summary.json`** — Full results: strategy config, performance metrics, per-rule breakdown, and trade log.

## Strategy Files

Strategies are JSON files with entry/exit rules, position sizing, and risk parameters. The simulator supports three patterns: **single-rule**, **DCA ladder** (independent multi-level), and **cascade** (linked dependent rules).

### Schema Reference

```jsonc
{
  // Optional description (ignored by engine)
  "_description": "My strategy",

  // Account settings
  "initial_capital": 5000,    // Starting balance in USD
  "fee_pct": 0.1,             // Fee per side (0.1% = 0.2% round trip)
  "max_concurrent_positions": 5,

  // Position sizing
  "position_sizing": {
    "type": "fixed",           // "fixed" (USD) or "percentage" (of capital)
    "value": 250
  },

  // One or more trading rules
  "rules": [
    {
      "id": "rule_1",         // Unique identifier
      "entry": {
        "type": "price_drop",  // "price_drop" or "price_rise"
        "percentage": 5.0,     // Required price change %
        "window": "4h"         // Lookback window (1m–1d)
      },
      "exit": {
        "take_profit_pct": 3.0,  // TP from entry price (required)
        "stop_loss_pct": 12.0    // SL from entry price (optional)
      },
      "max_positions": 5,       // Per-rule position limit

      // Optional: link to a parent rule (cascade/DCA)
      "depends_on": {
        "rule_id": "parent_rule",        // Must match another rule's id
        "condition": "has_open_position", // or "has_closed_position"
        "drop_from": "parent_entry"      // or "current_price"
      }
    }
  ]
}
```

### Entry & Exit

| Field | Values | Description |
|-------|--------|-------------|
| `entry.type` | `price_drop`, `price_rise` | Direction of price move that triggers entry |
| `entry.percentage` | `> 0` | Required magnitude of move (%) |
| `entry.window` | `1m`–`1d` | Lookback period to measure the move |
| `exit.take_profit_pct` | `> 0` | Close position at this % profit from entry |
| `exit.stop_loss_pct` | `> 0` or omit | Close position at this % loss from entry |

Valid windows: `1m`, `5m`, `10m`, `15m`, `30m`, `1h`, `2h`, `4h`, `1d`.

### Position Sizing

- **`fixed`**: Each position uses a fixed USD amount (e.g., `250` USD).
- **`percentage`**: Each position uses a percentage of current capital (e.g., `5` for 5%).

### Example: Single Rule

Buy on a 5% dip over 4 hours, take profit at +3%, stop loss at -12%:

```json
{
  "initial_capital": 5000,
  "fee_pct": 0.1,
  "max_concurrent_positions": 5,
  "position_sizing": { "type": "fixed", "value": 250 },
  "rules": [
    {
      "id": "deep_guarded",
      "entry": { "type": "price_drop", "percentage": 5.0, "window": "4h" },
      "exit": { "take_profit_pct": 3.0, "stop_loss_pct": 12.0 },
      "max_positions": 5
    }
  ]
}
```

### Example: DCA Ladder

Three independent levels that trigger at progressively deeper dips:

```json
{
  "initial_capital": 5000,
  "fee_pct": 0.1,
  "max_concurrent_positions": 15,
  "position_sizing": { "type": "fixed", "value": 100 },
  "rules": [
    {
      "id": "dca_level_1",
      "entry": { "type": "price_drop", "percentage": 1.5, "window": "30m" },
      "exit": { "take_profit_pct": 0.8 },
      "max_positions": 5
    },
    {
      "id": "dca_level_2",
      "entry": { "type": "price_drop", "percentage": 3.0, "window": "1h" },
      "exit": { "take_profit_pct": 1.5 },
      "max_positions": 5
    },
    {
      "id": "dca_level_3",
      "entry": { "type": "price_drop", "percentage": 5.0, "window": "2h" },
      "exit": { "take_profit_pct": 2.5 },
      "max_positions": 5
    }
  ]
}
```

### Example: Cascade (Linked Rules)

The second rule only triggers when the base rule has an open position, measuring the additional drop from the parent's entry price:

```json
{
  "initial_capital": 5000,
  "fee_pct": 0.1,
  "max_concurrent_positions": 10,
  "position_sizing": { "type": "fixed", "value": 150 },
  "rules": [
    {
      "id": "cascade_base",
      "entry": { "type": "price_drop", "percentage": 3.0, "window": "1h" },
      "exit": { "take_profit_pct": 2.0 },
      "max_positions": 5
    },
    {
      "id": "cascade_deep",
      "entry": { "type": "price_drop", "percentage": 3.0, "window": "2h" },
      "exit": { "take_profit_pct": 3.0 },
      "max_positions": 5,
      "depends_on": {
        "rule_id": "cascade_base",
        "condition": "has_open_position",
        "drop_from": "parent_entry"
      }
    }
  ]
}
```

#### `depends_on` fields

| Field | Values | Description |
|-------|--------|-------------|
| `rule_id` | string | ID of the parent rule |
| `condition` | `has_open_position`, `has_closed_position` | When the child rule is eligible |
| `drop_from` | `parent_entry`, `current_price` | Reference price for measuring the child's entry drop |

## Batch Testing

### Single-asset batch

Run all strategies in a directory against one feed:

```bash
uv run python -B scripts/batch_backtest.py
```

By default runs the `strategies/champions/` folder on BTCUSD 15m 1y. Override with environment variables:

```bash
STRATEGY_SET=scalping_v2 uv run python -B scripts/batch_backtest.py
```

Outputs a comparison table and saves results to `output/batch_comparison.json`.

### Multi-asset batch

Test champion strategies across BTC, SOL, and ETH simultaneously:

```bash
uv run python -B scripts/multi_asset_backtest.py
```

Outputs per-asset tables plus an aggregated cross-asset summary to `output/champion_comparison.json`.

## Metrics

Each backtest reports:

| Metric | Description |
|--------|-------------|
| Win Rate | % of trades closed at take profit |
| Profit Factor | Gross profits / gross losses |
| Expectancy | Average expected $ per trade |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Max Drawdown | Largest peak-to-trough equity decline |
| Recovery Factor | Total PnL / max drawdown |
| Avg Duration | Mean trade holding time |
| Max Consecutive Wins/Losses | Longest winning and losing streaks |

## Pre-built Strategies

The `strategies/` folder contains 26 ready-to-use strategy files organized by research iteration:

| Directory | Count | Description |
|-----------|-------|-------------|
| `strategies/` | 2 | `example.json`, `dip_buyer.json` — basic examples |
| `strategies/scalping/` | 8 | Round 1 — initial scalping approaches |
| `strategies/scalping_v2/` | 8 | Round 2 — deep dip mean reversion |
| `strategies/scalping_v3/` | 8 | Round 3 — optimized DCA + protective SL |
| `strategies/champions/` | 8 | Finals — best performers tested cross-asset |

**Top performer:** `champions/02_deep_guarded.json` — 5% dip entry, 3% TP, 12% SL. Averaged 82% win rate and PF 1.07 across BTC, SOL, and ETH over 1 year of 15-minute data.

## Testing

```bash
uv run pytest tests/ -q
```

## Project Structure

```
src/trade_simulator/
├── api.py        # OHLCV data fetching with pagination
├── cli.py        # Click CLI interface
├── engine.py     # Backtesting simulation loop
├── models.py     # Candle, Position, TradeResult dataclasses
├── plotting.py   # Interactive Plotly charts
├── report.py     # JSON summary and console output
└── strategy.py   # Pydantic strategy schema and validation
scripts/
├── batch_backtest.py         # Single-asset batch runner
└── multi_asset_backtest.py   # Multi-asset batch runner
strategies/                   # Pre-built strategy JSON files
tests/                        # pytest test suite
```

## License

MIT