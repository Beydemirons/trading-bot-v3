"""Data loader – fetch, cache (SQLite) and update OHLCV candles.

Handles both backward (historical gap) and forward (new candles) fills.
Saves incrementally to DB so interrupted downloads preserve progress.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config
from logger_setup import get_logger

log = get_logger("data_loader")

# ─────────────────────────────────────────────────────────────────────
# Timeframe -> milliseconds lookup
# ─────────────────────────────────────────────────────────────────────

_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
}

# ─────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
"""


def _get_conn() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    db = os.path.join(config.DATA_DIR, config.DB_PATH)
    conn = sqlite3.connect(db)
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _load_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM candles WHERE symbol=? AND timeframe=? ORDER BY timestamp",
        conn,
        params=(symbol, timeframe),
    )
    conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
    return df


def _get_cache_range(
    symbol: str, timeframe: str
) -> tuple[Optional[int], Optional[int], int]:
    """Return (earliest_ms, latest_ms, count) in cache."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) "
        "FROM candles WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    conn.close()
    if row and row[0] is not None:
        return int(row[0]), int(row[1]), int(row[2])
    return None, None, 0


def _save_batch(records: list[tuple]) -> None:
    if not records:
        return
    conn = _get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?)", records
    )
    conn.commit()
    conn.close()


def _ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# ─────────────────────────────────────────────────────────────────────
# ccxt fetcher — saves incrementally to DB
# ─────────────────────────────────────────────────────────────────────

def _get_exchange_start_ms(symbol: str) -> int:
    return config.EXCHANGE_START_MS.get(
        symbol, config.EXCHANGE_START_MS["DEFAULT"]
    )


def _fetch_and_store(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    limit: int = 1000,
) -> int:
    """Fetch candles [since_ms, until_ms) and save to DB in chunks.
    Returns total number of new candles fetched."""
    import ccxt

    exchange_cls = getattr(ccxt, config.EXCHANGE_ID)
    exchange: ccxt.Exchange = exchange_cls({
        "apiKey": config.API_KEY or None,
        "secret": config.API_SECRET or None,
        "enableRateLimit": True,
    })
    exchange.load_markets()

    end_ms = until_ms or int(time.time() * 1000)
    tf_duration = _TF_MS.get(timeframe, 3_600_000)
    estimated_total = max(1, (end_ms - since_ms) // tf_duration)

    log.info(
        "Fetching %s %s  %s -> %s  (~%d candles expected)",
        symbol, timeframe,
        _ms_to_str(since_ms), _ms_to_str(end_ms), estimated_total,
    )

    fetch_since = since_ms
    total_fetched = 0
    buffer: list[tuple] = []
    consecutive_errors = 0

    while fetch_since < end_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe, since=fetch_since, limit=limit
            )
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            wait = min(60, 5 * consecutive_errors)
            log.warning("API error #%d (retry in %ds): %s", consecutive_errors, wait, e)
            if consecutive_errors >= 10:
                log.error("Too many consecutive errors — aborting this segment.")
                break
            time.sleep(wait)
            continue

        if not batch:
            break

        for row in batch:
            ts = int(row[0])
            if until_ms and ts >= until_ms:
                continue
            buffer.append(
                (symbol, timeframe, ts, row[1], row[2], row[3], row[4], row[5])
            )

        total_fetched += len(batch)
        fetch_since = int(batch[-1][0]) + 1

        if len(buffer) >= 5000:
            _save_batch(buffer)
            buffer = []

        pct = min(100.0, (total_fetched / estimated_total) * 100)
        if total_fetched % 5000 < limit:
            log.info(
                "  ✓ %d / ~%d candles  (%.1f%%)  last: %s",
                total_fetched, estimated_total, pct,
                _ms_to_str(int(batch[-1][0])),
            )

        if len(batch) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if buffer:
        _save_batch(buffer)

    log.info("Segment complete — %d candles stored.", total_fetched)
    return total_fetched


# ─────────────────────────────────────────────────────────────────────
# yfinance fetcher
# ─────────────────────────────────────────────────────────────────────

_YF_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo",
}


def _fetch_yfinance(symbol: str, timeframe: str) -> pd.DataFrame:
    import yfinance as yf

    interval = _YF_INTERVAL_MAP.get(timeframe, "1d")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max", interval=interval)
    if df.empty:
        return df

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df.index = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    df["timestamp"] = (df.index.astype("int64") // 10**6).astype(int)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def load_data(
    symbol: str | None = None,
    timeframe: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Load candles: check cache, fill backward & forward gaps, return full set."""
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or config.TIMEFRAME
    source = source or config.DATA_SOURCE

    if source == "yfinance":
        fresh = _fetch_yfinance(symbol, timeframe)
        if fresh.empty:
            raise RuntimeError(f"No data found for {symbol}/{timeframe}")
        return fresh

    # ── ccxt: detect and fill gaps ───────────────────────────────────
    start_ms = _get_exchange_start_ms(symbol)
    now_ms = int(time.time() * 1000)
    cached_min, cached_max, cached_count = _get_cache_range(symbol, timeframe)

    if cached_min is None:
        # ── DB empty: full download ──────────────────────────────────
        log.info(
            "Database is empty — downloading full history for %s …", symbol
        )
        _fetch_and_store(symbol, timeframe, since_ms=start_ms)

    else:
        log.info(
            "Cache: %d candles  [%s -> %s]",
            cached_count, _ms_to_str(cached_min), _ms_to_str(cached_max),
        )

        # ── Backward gap: DB starts AFTER desired start date ─────────
        if cached_min > start_ms:
            log.info(
                "Backward gap: need data from %s to %s",
                _ms_to_str(start_ms), _ms_to_str(cached_min),
            )
            _fetch_and_store(
                symbol, timeframe,
                since_ms=start_ms,
                until_ms=cached_min,
            )

        # ── Forward gap: new candles since last cached ───────────────
        tf_duration = _TF_MS.get(timeframe, 3_600_000)
        gap_ms = now_ms - cached_max
        if gap_ms > tf_duration:
            log.info(
                "Forward gap: fetching from %s to now …",
                _ms_to_str(cached_max),
            )
            _fetch_and_store(
                symbol, timeframe,
                since_ms=cached_max + 1,
            )
        else:
            log.info("Cache is up-to-date — no new candles to fetch.")

    # ── Load complete dataset from DB ────────────────────────────────
    combined = _load_cached(symbol, timeframe)
    if combined.empty:
        raise RuntimeError(f"No data found for {symbol}/{timeframe}")

    log.info(
        "Dataset ready: %d candles  [%s -> %s]",
        len(combined), combined.index[0], combined.index[-1],
    )
    return combined
