#!/usr/bin/env python3
"""Analyze dip depth and bounce height from historical OHLCV data.

Detects every dip (consecutive red candles), measures how deep it goes,
then measures how high the price bounces before the next dip.

A dip ends when:
  - A single green candle gains >= 1%, OR
  - 3 consecutive green candles appear.

Usage:
  uv run python -B scripts/dip_analyzer.py --feed BTCUSD --type 15
  uv run python -B scripts/dip_analyzer.py --feed SOLUSD --type 5 --min-dip 1.0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_simulator.api import fetch_ohlcv, parse_time_range, validate_feed, validate_type
from trade_simulator.models import Candle


@dataclass
class DipEvent:
    """A single dip-and-bounce event."""

    start_time: datetime
    bottom_time: datetime
    end_time: datetime
    peak_before: float       # highest price before the dip
    bottom: float            # lowest low during the dip
    dip_pct: float           # (peak_before - bottom) / peak_before * 100
    bounce_peak: float       # highest high after dip before next dip
    bounce_pct: float        # (bounce_peak - bottom) / bottom * 100
    dip_candles: int         # number of candles in the dip
    bounce_candles: int      # number of candles in the bounce
    duration_minutes: float  # dip duration from start to end in minutes


def detect_dips(candles: list[Candle], min_dip_pct: float = 0.5) -> list[DipEvent]:
    """Detect all dip events using a state machine.

    A dip starts on a red candle and ends when a single green candle
    gains >= 1% or 3 consecutive green candles appear.
    """
    events: list[DipEvent] = []

    # State tracking
    in_dip = False
    peak_since_reset = candles[0].high
    dip_start_peak = 0.0
    dip_bottom = float("inf")
    dip_start_time = candles[0].time
    dip_bottom_time = candles[0].time
    dip_start_idx = 0
    green_streak = 0

    for i, c in enumerate(candles):
        is_green = c.close >= c.open

        if not in_dip:
            peak_since_reset = max(peak_since_reset, c.high)
            if not is_green:
                # Red candle — start a potential dip
                in_dip = True
                dip_start_peak = peak_since_reset
                dip_bottom = c.low
                dip_bottom_time = c.time
                dip_start_time = c.time
                dip_start_idx = i
                green_streak = 0
        else:
            dip_bottom = min(dip_bottom, c.low)
            if c.low <= dip_bottom:
                dip_bottom = c.low
                dip_bottom_time = c.time

            if is_green:
                green_streak += 1
                candle_gain = (c.close - c.open) / c.open if c.open > 0 else 0
                if candle_gain >= 0.01 or green_streak >= 3:
                    # Dip ended
                    dip_pct = (dip_start_peak - dip_bottom) / dip_start_peak * 100 if dip_start_peak > 0 else 0
                    if dip_pct >= min_dip_pct:
                        dur_min = (c.time - dip_start_time).total_seconds() / 60
                        events.append(DipEvent(
                            start_time=dip_start_time,
                            bottom_time=dip_bottom_time,
                            end_time=c.time,
                            peak_before=dip_start_peak,
                            bottom=dip_bottom,
                            dip_pct=dip_pct,
                            bounce_peak=0.0,     # filled in next pass
                            bounce_pct=0.0,      # filled in next pass
                            dip_candles=i - dip_start_idx + 1,
                            bounce_candles=0,     # filled in next pass
                            duration_minutes=dur_min,
                        ))
                    in_dip = False
                    peak_since_reset = c.high
                    green_streak = 0
            else:
                green_streak = 0

    # Compute bounce peaks between consecutive dips using index lookup
    # Build a time→index map for fast lookup
    time_to_idx: dict[datetime, int] = {c.time: i for i, c in enumerate(candles)}

    for idx, event in enumerate(events):
        bounce_start = time_to_idx.get(event.end_time)
        if bounce_start is None:
            continue

        if idx + 1 < len(events):
            bounce_end = time_to_idx.get(events[idx + 1].start_time, len(candles))
        else:
            bounce_end = len(candles)

        if bounce_start >= bounce_end:
            continue

        highest = max(c.high for c in candles[bounce_start:bounce_end])
        event.bounce_peak = highest
        event.bounce_pct = (highest - event.bottom) / event.bottom * 100 if event.bottom > 0 else 0
        event.bounce_candles = bounce_end - bounce_start

    return events


def print_summary(events: list[DipEvent], feed: str, candle_type: str) -> None:
    """Print dip/bounce statistics."""
    if not events:
        print("No dips found matching criteria.")
        return

    dip_pcts = [e.dip_pct for e in events]
    bounce_pcts = [e.bounce_pct for e in events if e.bounce_pct > 0]
    durations = [e.duration_minutes for e in events]

    print()
    print("=" * 80)
    print(f"  DIP ANALYSIS — {feed} {candle_type}m candles — 1 year")
    print("=" * 80)

    # Overall stats
    print(f"\n  Total dips found: {len(events)}")
    print(f"  Dips with bounce data: {len(bounce_pcts)}")
    print()

    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    DIP DEPTH (%)                        │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Average:    {statistics.mean(dip_pcts):8.2f}%                           │")
    print(f"  │  Median:     {statistics.median(dip_pcts):8.2f}%                           │")
    print(f"  │  Std Dev:    {statistics.stdev(dip_pcts):8.2f}%                           │" if len(dip_pcts) > 1 else "")
    print(f"  │  Min:        {min(dip_pcts):8.2f}%                           │")
    print(f"  │  Max:        {max(dip_pcts):8.2f}%                           │")
    print(f"  │  25th pctl:  {percentile(dip_pcts, 25):8.2f}%                           │")
    print(f"  │  75th pctl:  {percentile(dip_pcts, 75):8.2f}%                           │")
    print("  └─────────────────────────────────────────────────────────┘")

    # Duration stats
    def _fmt_dur(minutes: float) -> str:
        """Format minutes as Xh Ym."""
        h, m = divmod(int(minutes), 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                   DIP DURATION                         │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Average:    {_fmt_dur(statistics.mean(durations)):>10s}                          │")
    print(f"  │  Median:     {_fmt_dur(statistics.median(durations)):>10s}                          │")
    if len(durations) > 1:
        print(f"  │  Std Dev:    {_fmt_dur(statistics.stdev(durations)):>10s}                          │")
    print(f"  │  Min:        {_fmt_dur(min(durations)):>10s}                          │")
    print(f"  │  Max:        {_fmt_dur(max(durations)):>10s}                          │")
    print(f"  │  25th pctl:  {_fmt_dur(percentile(durations, 25)):>10s}                          │")
    print(f"  │  75th pctl:  {_fmt_dur(percentile(durations, 75)):>10s}                          │")
    print("  └─────────────────────────────────────────────────────────┘")

    if bounce_pcts:
        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │                  BOUNCE HEIGHT (%)                      │")
        print("  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  Average:    {statistics.mean(bounce_pcts):8.2f}%                           │")
        print(f"  │  Median:     {statistics.median(bounce_pcts):8.2f}%                           │")
        print(f"  │  Std Dev:    {statistics.stdev(bounce_pcts):8.2f}%                           │" if len(bounce_pcts) > 1 else "")
        print(f"  │  Min:        {min(bounce_pcts):8.2f}%                           │")
        print(f"  │  Max:        {max(bounce_pcts):8.2f}%                           │")
        print(f"  │  25th pctl:  {percentile(bounce_pcts, 25):8.2f}%                           │")
        print(f"  │  75th pctl:  {percentile(bounce_pcts, 75):8.2f}%                           │")
        print("  └─────────────────────────────────────────────────────────┘")

    # Bucketed analysis
    buckets = [
        ("Small   (0.5–1%)", 0.5, 1.0),
        ("Medium  (1–2%)",   1.0, 2.0),
        ("Large   (2–3%)",   2.0, 3.0),
        ("Big     (3–5%)",   3.0, 5.0),
        ("Crash   (5–10%)",  5.0, 10.0),
        ("Extreme (10%+)",  10.0, 100.0),
    ]

    print()
    print("  ┌───────────────────────┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Dip Bucket            │ Count │ Avg Dip% │ Med Dip% │Avg Bnc%  │Med Bnc%  │ Avg Dur  │ Med Dur  │")
    print("  ├───────────────────────┼───────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for label, lo, hi in buckets:
        bucket_events = [e for e in events if lo <= e.dip_pct < hi]
        if not bucket_events:
            print(f"  │ {label:21s} │ {0:5d} │     —    │     —    │     —    │     —    │     —    │     —    │")
            continue

        b_dips = [e.dip_pct for e in bucket_events]
        b_bounces = [e.bounce_pct for e in bucket_events if e.bounce_pct > 0]
        b_durs = [e.duration_minutes for e in bucket_events]

        avg_dip = statistics.mean(b_dips)
        med_dip = statistics.median(b_dips)
        avg_bnc = statistics.mean(b_bounces) if b_bounces else 0
        med_bnc = statistics.median(b_bounces) if b_bounces else 0
        avg_dur = _fmt_dur(statistics.mean(b_durs))
        med_dur = _fmt_dur(statistics.median(b_durs))

        print(f"  │ {label:21s} │ {len(bucket_events):5d} │ {avg_dip:7.2f}% │ {med_dip:7.2f}% │ {avg_bnc:7.2f}% │ {med_bnc:7.2f}% │ {avg_dur:>8s} │ {med_dur:>8s} │")

    print("  └───────────────────────┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # Monthly breakdown
    monthly: dict[str, list[DipEvent]] = {}
    for e in events:
        key = e.start_time.strftime("%Y-%m")
        monthly.setdefault(key, []).append(e)

    print()
    print("  ┌────────────┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Month      │ Count │ Avg Dip% │ Med Dip% │Avg Bnc%  │Med Bnc%  │ Avg Dur  │ Med Dur  │")
    print("  ├────────────┼───────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for month in sorted(monthly):
        m_events = monthly[month]
        m_dips = [e.dip_pct for e in m_events]
        m_bounces = [e.bounce_pct for e in m_events if e.bounce_pct > 0]
        m_durs = [e.duration_minutes for e in m_events]
        avg_d = statistics.mean(m_dips)
        med_d = statistics.median(m_dips)
        avg_b = statistics.mean(m_bounces) if m_bounces else 0
        med_b = statistics.median(m_bounces) if m_bounces else 0
        avg_dur = _fmt_dur(statistics.mean(m_durs))
        med_dur = _fmt_dur(statistics.median(m_durs))
        print(f"  │ {month:10s} │ {len(m_events):5d} │ {avg_d:7.2f}% │ {med_d:7.2f}% │ {avg_b:7.2f}% │ {med_b:7.2f}% │ {avg_dur:>8s} │ {med_dur:>8s} │")

    print("  └────────────┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # Top 10 biggest dips
    sorted_events = sorted(events, key=lambda e: e.dip_pct, reverse=True)
    top_n = sorted_events[:10]

    print()
    print(f"  TOP {len(top_n)} BIGGEST DIPS:")
    print("  ┌─────┬─────────────────────┬──────────┬──────────┬──────────────┬──────────────┬──────────┐")
    print("  │  #  │ Bottom Time         │  Dip %   │ Bounce % │ Peak Before  │   Bottom $   │ Duration │")
    print("  ├─────┼─────────────────────┼──────────┼──────────┼──────────────┼──────────────┼──────────┤")

    for rank, e in enumerate(top_n, 1):
        ts = e.bottom_time.strftime("%Y-%m-%d %H:%M")
        dur = _fmt_dur(e.duration_minutes)
        print(f"  │ {rank:3d} │ {ts:19s} │ {e.dip_pct:7.2f}% │ {e.bounce_pct:7.2f}% │ {e.peak_before:>12,.2f} │ {e.bottom:>12,.2f} │ {dur:>8s} │")

    print("  └─────┴─────────────────────┴──────────┴──────────┴──────────────┴──────────────┴──────────┘")

    # Strategy recommendation
    if bounce_pcts:
        print()
        print("  ╔═════════════════════════════════════════════════════════╗")
        print("  ║              STRATEGY PARAMETER SUGGESTIONS            ║")
        print("  ╠═════════════════════════════════════════════════════════╣")
        med_dip = statistics.median(dip_pcts)
        med_bounce = statistics.median(bounce_pcts)
        p25_bounce = percentile(bounce_pcts, 25)
        conservative_tp = p25_bounce * 0.7  # 70% of 25th percentile bounce
        moderate_tp = med_bounce * 0.6       # 60% of median bounce
        print(f"  ║  Median dip:    {med_dip:.2f}%  → entry trigger           ║")
        print(f"  ║  Median bounce: {med_bounce:.2f}%  → max TP potential        ║")
        print(f"  ║                                                         ║")
        print(f"  ║  Conservative TP: {conservative_tp:.2f}% (70% of P25 bounce)  ║")
        print(f"  ║  Moderate TP:     {moderate_tp:.2f}% (60% of median bounce)  ║")
        print(f"  ║                                                         ║")
        print(f"  ║  Suggested SL:    {med_dip * 3:.1f}% (3x median dip)           ║")
        print("  ╚═════════════════════════════════════════════════════════╝")


def percentile(data: list[float], pct: float) -> float:
    """Compute percentile using sorted interpolation."""
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def save_json(events: list[DipEvent], feed: str, candle_type: str) -> Path:
    """Save raw dip data to JSON."""
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"dip_analysis_{feed}_{candle_type}m.json"

    data = {
        "feed": feed,
        "candle_type": candle_type,
        "total_dips": len(events),
        "dips": [
            {
                "start": e.start_time.isoformat(),
                "bottom_time": e.bottom_time.isoformat(),
                "end": e.end_time.isoformat(),
                "peak_before": e.peak_before,
                "bottom": e.bottom,
                "dip_pct": round(e.dip_pct, 4),
                "bounce_peak": e.bounce_peak,
                "bounce_pct": round(e.bounce_pct, 4),
                "dip_candles": e.dip_candles,
                "bounce_candles": e.bounce_candles,
                "duration_minutes": round(e.duration_minutes, 1),
            }
            for e in events
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dip depth and bounce height from OHLCV data.")
    parser.add_argument("--feed", required=True, choices=["BTCUSD", "SOLUSD", "ETHUSD"], help="Trading pair")
    parser.add_argument("--type", dest="candle_type", required=True, choices=["5", "15"], help="Candle width in minutes")
    parser.add_argument("--min-dip", type=float, default=0.5, help="Minimum dip %% to include (default: 0.5)")
    parser.add_argument("--save-json", action="store_true", help="Save raw dip data to output/ as JSON")
    args = parser.parse_args()

    feed = validate_feed(args.feed)
    candle_type = validate_type(args.candle_type)
    from_ms, till_ms = parse_time_range("1y")

    print(f"Fetching {feed} {candle_type}m candles for 1y...")

    def progress(fetched: int, chunk: int) -> None:
        print(f"  chunk {chunk}: {fetched} candles...")

    candles = fetch_ohlcv(feed, candle_type, from_ms, till_ms, on_progress=progress)
    print(f"Total: {len(candles)} candles")
    print(f"Period: {candles[0].time.date()} to {candles[-1].time.date()}")

    print(f"\nDetecting dips (min {args.min_dip}%)...")
    events = detect_dips(candles, min_dip_pct=args.min_dip)

    print_summary(events, feed, args.candle_type)

    if args.save_json:
        path = save_json(events, feed, args.candle_type)
        print(f"\nRaw data saved to {path}")


if __name__ == "__main__":
    main()
