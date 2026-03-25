"""Summary report generation."""

from __future__ import annotations

import json
from pathlib import Path

import click

from trade_simulator.models import TradeResult


def generate_summary(result: TradeResult, output_path: str | Path) -> Path:
    """Write the simulation summary to a JSON file.

    Args:
        result: The TradeResult from the simulation.
        output_path: Path to write the summary JSON file.

    Returns:
        The path to the generated JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "feed": result.feed,
        "timeframe": result.timeframe,
        "time_range": result.time_range,
        "capital": {
            "initial": round(result.initial_capital, 2),
            "final": round(result.final_capital, 2),
        },
        "performance": {
            "total_pnl": round(result.total_pnl, 2),
            "total_pnl_pct": round(result.total_pnl_pct, 2),
            "total_fees": round(result.total_fees, 2),
            "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float("inf") else "inf",
        },
        "trades": {
            "total": result.total_trades,
            "winning": result.winning_trades,
            "losing": result.losing_trades,
            "win_rate": round(result.win_rate, 2),
        },
        "averages": {
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "best_trade": round(result.best_trade, 2),
            "worst_trade": round(result.worst_trade, 2),
        },
        "risk": {
            "max_drawdown": round(result.max_drawdown, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        },
        "per_rule": {
            rule_id: {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in stats.items()
            }
            for rule_id, stats in result.per_rule.items()
        },
        "trades_log": [
            {
                "rule_id": p.rule_id,
                "entry_price": round(p.entry_price, 4),
                "entry_time": p.entry_time.isoformat(),
                "exit_price": round(p.exit_price, 4) if p.exit_price else None,
                "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                "size_usd": round(p.size_usd, 2),
                "pnl": round(p.pnl, 2),
                "entry_fee": round(p.entry_fee, 4),
                "exit_fee": round(p.exit_fee, 4),
            }
            for p in result.positions
        ],
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def print_summary(result: TradeResult) -> None:
    """Print a formatted summary to the console."""
    click.echo()
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo(click.style("  TRADE SIMULATION SUMMARY", fg="cyan", bold=True))
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo()

    click.echo(f"  Feed:           {result.feed}")
    click.echo(f"  Timeframe:      {result.timeframe}")
    click.echo(f"  Range:          {result.time_range}")
    click.echo()

    click.echo(click.style("  Capital", bold=True))
    click.echo(f"    Initial:      ${result.initial_capital:,.2f}")
    click.echo(f"    Final:        ${result.final_capital:,.2f}")
    click.echo()

    # PnL with color
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    click.echo(click.style("  Performance", bold=True))
    click.echo(f"    Total PnL:    " + click.style(f"${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)", fg=pnl_color))
    click.echo(f"    Total Fees:   ${result.total_fees:,.2f}")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
    click.echo(f"    Profit Factor: {pf_str}")
    click.echo()

    click.echo(click.style("  Trades", bold=True))
    click.echo(f"    Total:        {result.total_trades}")
    click.echo(f"    Winning:      {result.winning_trades}")
    click.echo(f"    Losing:       {result.losing_trades}")
    wr_color = "green" if result.win_rate >= 50 else "red"
    click.echo(f"    Win Rate:     " + click.style(f"{result.win_rate:.1f}%", fg=wr_color))
    click.echo()

    click.echo(click.style("  Averages", bold=True))
    click.echo(f"    Avg Win:      ${result.avg_win:,.2f}")
    click.echo(f"    Avg Loss:     ${result.avg_loss:,.2f}")
    click.echo(f"    Best Trade:   ${result.best_trade:,.2f}")
    click.echo(f"    Worst Trade:  ${result.worst_trade:,.2f}")
    click.echo()

    click.echo(click.style("  Risk", bold=True))
    click.echo(f"    Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    click.echo()

    # Per-rule breakdown
    if result.per_rule:
        click.echo(click.style("  Per-Rule Breakdown", bold=True))
        for rule_id, stats in result.per_rule.items():
            rule_pnl = stats["total_pnl"]
            rule_color = "green" if rule_pnl >= 0 else "red"
            click.echo(f"    [{rule_id}]  trades={stats['total_trades']}  "
                        f"wins={stats['winning_trades']}  "
                        f"wr={stats['win_rate']:.1f}%  "
                        f"pnl=" + click.style(f"${rule_pnl:,.2f}", fg=rule_color))
        click.echo()

    click.echo(click.style("=" * 60, fg="cyan"))
