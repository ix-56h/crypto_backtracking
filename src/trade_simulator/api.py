"""API client for fetching OHLCV data from the Jup.ag oracle."""

from __future__ import annotations

import re
import time

import httpx

from trade_simulator.models import Candle

BASE_URL = "https://history.oraclesecurity.org/trading-view/data"

VALID_FEEDS = {"SOLUSD", "BTCUSD", "ETHUSD"}
VALID_TYPES = {"1", "5", "15", "1h", "1D"}

_RANGE_PATTERN = re.compile(r"^(\d+)([dmyh])$", re.IGNORECASE)


def parse_time_range(range_str: str) -> tuple[int, int]:
    """Parse a human-readable time range string into (from_ms, till_ms).

    Supported formats: '1d', '7d', '30d', '6m', '1y', '12h'
    till = now, from = now - duration.
    """
    match = _RANGE_PATTERN.match(range_str.strip())
    if not match:
        raise ValueError(
            f"Invalid range format: {range_str!r}. "
            "Use e.g. '1d', '30d', '6m', '1y'."
        )

    amount = int(match.group(1))
    unit = match.group(2).lower()

    now_ms = int(time.time() * 1000)

    multipliers = {
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "m": 30 * 24 * 60 * 60 * 1000,
        "y": 365 * 24 * 60 * 60 * 1000,
    }

    delta_ms = amount * multipliers[unit]
    from_ms = now_ms - delta_ms
    return from_ms, now_ms


def validate_feed(feed: str) -> str:
    """Validate and normalize the feed name."""
    feed_upper = feed.upper().strip()
    if feed_upper not in VALID_FEEDS:
        raise ValueError(f"Invalid feed: {feed!r}. Choose from: {VALID_FEEDS}")
    return feed_upper


def validate_type(type_str: str) -> str:
    """Validate the candle type/timescale."""
    t = type_str.strip()
    if t not in VALID_TYPES:
        raise ValueError(f"Invalid type: {type_str!r}. Choose from: {VALID_TYPES}")
    return t


def get_candle_minutes(type_str: str) -> int:
    """Return the number of minutes per candle for a given type string."""
    mapping = {
        "1": 1,
        "5": 5,
        "15": 15,
        "1h": 60,
        "1D": 1440,
    }
    return mapping[type_str]


# The API caps results at 4000 candles per request.
# Use 3500 candles worth of time per chunk to stay safely within the limit.
API_MAX_CANDLES = 4000
_CHUNK_CANDLES = 3500


def _chunk_ms(type_str: str) -> int:
    """Return the time span (in ms) for one pagination chunk."""
    candle_min = get_candle_minutes(type_str)
    return _CHUNK_CANDLES * candle_min * 60 * 1000


def _fetch_single_page(
    client: httpx.Client,
    feed: str,
    type_str: str,
    from_ms: int,
    till_ms: int,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch a single page of candle data with retries."""
    params = {
        "feed": feed,
        "type": type_str,
        "from": str(from_ms),
        "till": str(till_ms),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("result", [])
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise ConnectionError(
        f"Failed to fetch data after {max_retries} attempts: {last_error}"
    ) from last_error


def fetch_ohlcv(
    feed: str,
    type_str: str,
    from_ms: int,
    till_ms: int,
    *,
    max_retries: int = 3,
    on_progress: callable | None = None,
) -> list[Candle]:
    """Fetch OHLCV candle data from the oracle API with automatic pagination.

    The API returns at most 4000 candles per request. For long time ranges,
    this function fetches in chunks and concatenates the results.

    Args:
        feed: Trading pair (e.g. 'BTCUSD').
        type_str: Candle timescale (e.g. '15', '1h').
        from_ms: Start timestamp in milliseconds.
        till_ms: End timestamp in milliseconds.
        max_retries: Max retries per request.
        on_progress: Optional callback(fetched_so_far, chunk_num) for progress.

    Returns:
        List of Candle objects sorted by time ascending.
    """
    feed = validate_feed(feed)
    type_str = validate_type(type_str)

    chunk_size = _chunk_ms(type_str)
    all_raw: list[dict] = []
    cursor = from_ms
    chunk_num = 0

    with httpx.Client(timeout=30.0) as client:
        while cursor < till_ms:
            chunk_end = min(cursor + chunk_size, till_ms)
            page = _fetch_single_page(client, feed, type_str, cursor, chunk_end, max_retries)
            chunk_num += 1

            if not page:
                # No data in this chunk; advance past it
                cursor = chunk_end + 1
                continue

            all_raw.extend(page)

            if on_progress:
                on_progress(len(all_raw), chunk_num)

            # Advance cursor past the last returned candle to avoid duplicates
            last_time = max(c["time"] for c in page)
            new_cursor = last_time + 1
            if new_cursor <= cursor:
                # No progress; break to avoid infinite loop
                break
            cursor = new_cursor

    if not all_raw:
        raise ValueError(
            f"No candle data returned for {feed} ({type_str}) "
            f"in the specified time range."
        )

    # Deduplicate by timestamp
    seen: set[int] = set()
    unique: list[dict] = []
    for c in all_raw:
        if c["time"] not in seen:
            seen.add(c["time"])
            unique.append(c)

    candles = [Candle.from_api(c) for c in unique]
    candles.sort(key=lambda c: c.time)
    return candles
