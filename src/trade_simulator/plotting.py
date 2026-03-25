"""Interactive Plotly chart generation for trade simulation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trade_simulator.models import Candle, Position, PositionStatus, TradeResult


def build_chart(
    candles: list[Candle],
    result: TradeResult,
    output_path: str | Path,
) -> Path:
    """Build an interactive OHLCV candlestick chart with buy/sell markers.

    Args:
        candles: List of OHLCV candles.
        result: The TradeResult containing closed positions.
        output_path: Path to write the HTML file.

    Returns:
        The path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build DataFrame for candlestick
    df = pd.DataFrame([
        {
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=("", "Volume", "Equity Curve"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = [
        "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        for _, row in df.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.5,
        ),
        row=2, col=1,
    )

    # Buy markers
    buys = [
        p for p in result.positions
        if p.entry_time is not None
    ]
    if buys:
        fig.add_trace(
            go.Scatter(
                x=[p.entry_time for p in buys],
                y=[p.entry_price for p in buys],
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#00e676",
                    line=dict(width=1, color="black"),
                ),
                text=[
                    f"BUY {p.rule_id}<br>"
                    f"Price: ${p.entry_price:,.2f}<br>"
                    f"Size: ${p.size_usd:,.2f}"
                    for p in buys
                ],
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # Sell markers
    sells = [
        p for p in result.positions
        if p.exit_time is not None
    ]
    if sells:
        pnl_colors = ["#00e676" if p.pnl >= 0 else "#ff1744" for p in sells]
        fig.add_trace(
            go.Scatter(
                x=[p.exit_time for p in sells],
                y=[p.exit_price for p in sells],
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=pnl_colors,
                    line=dict(width=1, color="black"),
                ),
                text=[
                    f"SELL {p.rule_id}<br>"
                    f"Price: ${p.exit_price:,.2f}<br>"
                    f"PnL: ${p.pnl:,.2f}"
                    for p in sells
                ],
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # Equity curve
    equity_times, equity_values = _compute_equity_curve(candles, result)
    if equity_times:
        fig.add_trace(
            go.Scatter(
                x=equity_times,
                y=equity_values,
                name="Equity",
                line=dict(color="#42a5f5", width=2),
                fill="tozeroy",
                fillcolor="rgba(66, 165, 245, 0.1)",
            ),
            row=3, col=1,
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"{result.feed} — {result.timeframe} candles — {result.time_range} | "
                f"PnL: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%) | "
                f"Win Rate: {result.win_rate:.1f}% | "
                f"Trades: {result.total_trades}"
            ),
            font=dict(size=14),
        ),
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=3, col=1)

    fig.write_html(str(output_path), include_plotlyjs=True)
    return output_path


def _compute_equity_curve(
    candles: list[Candle],
    result: TradeResult,
) -> tuple[list, list]:
    """Compute an equity curve over time based on trade history.

    Simple approach: capital changes at each buy/sell event.
    """
    if not result.positions:
        return [], []

    # Build events sorted by time
    events: list[tuple] = []
    for pos in result.positions:
        events.append((pos.entry_time, "buy", -pos.size_usd))
        if pos.exit_time:
            events.append((pos.exit_time, "sell", pos.pnl + pos.size_usd + pos.entry_fee))
    events.sort(key=lambda e: e[0])

    times = []
    values = []
    equity = result.initial_capital

    # Start point
    times.append(candles[0].time)
    values.append(equity)

    for t, _, delta in events:
        equity += delta
        times.append(t)
        values.append(equity)

    # End point
    times.append(candles[-1].time)
    values.append(equity)

    return times, values
