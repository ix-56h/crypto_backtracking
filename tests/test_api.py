"""Tests for the API client."""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from trade_simulator.api import (
    BASE_URL,
    fetch_ohlcv,
    get_candle_minutes,
    parse_time_range,
    validate_feed,
    validate_type,
)


class TestParseTimeRange:
    """Test the human-readable time range parser."""

    def test_parse_days(self):
        from_ms, till_ms = parse_time_range("7d")
        expected_delta = 7 * 24 * 60 * 60 * 1000
        assert till_ms - from_ms == expected_delta
        assert till_ms == pytest.approx(int(time.time() * 1000), abs=2000)

    def test_parse_months(self):
        from_ms, till_ms = parse_time_range("6m")
        expected_delta = 6 * 30 * 24 * 60 * 60 * 1000
        assert till_ms - from_ms == expected_delta

    def test_parse_year(self):
        from_ms, till_ms = parse_time_range("1y")
        expected_delta = 365 * 24 * 60 * 60 * 1000
        assert till_ms - from_ms == expected_delta

    def test_parse_hours(self):
        from_ms, till_ms = parse_time_range("12h")
        expected_delta = 12 * 60 * 60 * 1000
        assert till_ms - from_ms == expected_delta

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid range format"):
            parse_time_range("abc")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Invalid range format"):
            parse_time_range("5x")


class TestValidateFeed:
    def test_valid_feeds(self):
        assert validate_feed("btcusd") == "BTCUSD"
        assert validate_feed("SOLUSD") == "SOLUSD"
        assert validate_feed("EthUSD") == "ETHUSD"

    def test_invalid_feed(self):
        with pytest.raises(ValueError, match="Invalid feed"):
            validate_feed("DOGEUSD")


class TestValidateType:
    def test_valid_types(self):
        for t in ("1", "5", "15", "1h", "1D"):
            assert validate_type(t) == t

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid type"):
            validate_type("2h")


class TestGetCandleMinutes:
    def test_all_types(self):
        assert get_candle_minutes("1") == 1
        assert get_candle_minutes("5") == 5
        assert get_candle_minutes("15") == 15
        assert get_candle_minutes("1h") == 60
        assert get_candle_minutes("1D") == 1440


class TestFetchOhlcv:
    """Tests for fetch_ohlcv with mocked internals.

    We patch both httpx.Client (to avoid real network connections that
    conflict with pytest-asyncio/anyio) and _fetch_single_page.
    """

    def test_successful_fetch(self, sample_api_response):
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = [sample_api_response["result"], []]
            candles = fetch_ohlcv("BTCUSD", "15", 1774180000000, 1774183000000)
        assert len(candles) == 3
        assert candles[0].open == 68209.2
        assert candles[0].close == 68304.58

    def test_empty_result_raises(self):
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.return_value = []
            with pytest.raises(ValueError, match="No candle data"):
                fetch_ohlcv("BTCUSD", "15", 1774180000000, 1774183000000)

    def test_http_error_retries(self):
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = ConnectionError(
                "Failed to fetch data after 3 attempts: server error"
            )
            with pytest.raises(ConnectionError, match="Failed to fetch"):
                fetch_ohlcv("BTCUSD", "15", 1774180000000, 1774183000000, max_retries=3)

    def test_pagination_multiple_chunks(self):
        """Test that pagination fetches multiple chunks for large ranges."""
        chunk1 = [
            {"time": 1000000, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 50},
            {"time": 2000000, "open": 100.5, "high": 102, "low": 100, "close": 101, "volume": 60},
        ]
        chunk2 = [
            {"time": 3000000, "open": 101, "high": 103, "low": 100.5, "close": 102, "volume": 70},
        ]
        pages = iter([chunk1, chunk2])
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = lambda *a, **kw: next(pages, [])
            candles = fetch_ohlcv("BTCUSD", "1", 1000000, 220_000_000)
        assert len(candles) == 3
        assert candles[0].time.timestamp() * 1000 == 1000000
        assert candles[-1].time.timestamp() * 1000 == 3000000
        assert mock_fetch.call_count >= 2

    def test_progress_callback(self, sample_api_response):
        """Test that the progress callback is invoked."""
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = [sample_api_response["result"], []]
            progress_calls = []
            fetch_ohlcv(
                "BTCUSD", "15", 1774180000000, 1774183000000,
                on_progress=lambda fetched, chunk: progress_calls.append((fetched, chunk)),
            )
        assert len(progress_calls) >= 1
        assert progress_calls[0][0] == 3  # 3 candles fetched

    def test_deduplication(self):
        """Test that duplicate candles across chunks are deduplicated."""
        chunk1 = [
            {"time": 1000000, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 50},
            {"time": 2000000, "open": 100.5, "high": 102, "low": 100, "close": 101, "volume": 60},
        ]
        chunk2 = [
            {"time": 2000000, "open": 100.5, "high": 102, "low": 100, "close": 101, "volume": 60},
            {"time": 3000000, "open": 101, "high": 103, "low": 100.5, "close": 102, "volume": 70},
        ]
        pages = iter([chunk1, chunk2])
        with patch("trade_simulator.api.httpx.Client"), \
             patch("trade_simulator.api._fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = lambda *a, **kw: next(pages, [])
            candles = fetch_ohlcv("BTCUSD", "1", 1000000, 220_000_000)
        assert len(candles) == 3  # deduplicated from 4 raw entries
