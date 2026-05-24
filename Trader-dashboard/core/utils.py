import os
import io
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import streamlit as st
import requests

logger = logging.getLogger(__name__)

# --- Disk Cache ---
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 6


def _cache_key(ticker, interval, period):
    """Generate a unique cache key for a data request."""
    raw = f"{ticker}_{interval}_{period}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(cache_key):
    """Read from disk cache if valid."""
    cache_path = CACHE_DIR / f"{cache_key}.parquet"
    meta_path = CACHE_DIR / f"{cache_key}.meta"

    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        cached_time = datetime.fromisoformat(meta["cached_at"])
        if datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS):
            df = pd.read_parquet(cache_path)
            logger.debug(f"Cache HIT: {meta.get('ticker', cache_key)}")
            return df
        else:
            logger.debug(f"Cache EXPIRED: {meta.get('ticker', cache_key)}")
            # Stale cache - remove
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
    except Exception:
        return None


def _write_cache(cache_key, df, ticker="unknown"):
    """Write data to disk cache."""
    try:
        cache_path = CACHE_DIR / f"{cache_key}.parquet"
        meta_path = CACHE_DIR / f"{cache_key}.meta"
        df.to_parquet(cache_path)
        with open(meta_path, "w") as f:
            json.dump({
                "ticker": ticker,
                "cached_at": datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns)
            }, f)
        logger.debug(f"Cache WRITE: {ticker}")
    except Exception as e:
        logger.warning(f"Cache write failed for {ticker}: {e}")


def clean_columns(df):
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
    use_cache: bool = True,
):
    """
    Load stock data with disk caching + Streamlit caching.

    Falls back gracefully if Yahoo Finance fails.
    Returns a clean DataFrame with 'Close', 'Open', 'High', 'Low', 'Volume'.
    """
    # Auto-append .NS for NSE stocks (no need to type .NS everywhere)
    ticker = ticker.strip().upper()
    if ticker and not ticker.endswith(".NS") and "^" not in ticker and ".BO" not in ticker:
        ticker += ".NS"

    cache_key = _cache_key(ticker, interval, period)

    # Try disk cache first
    if use_cache:
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached

    # Fetch from Yahoo Finance with exponential backoff retries
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is not None and not df.empty:
                df = clean_columns(df)
                df.dropna(inplace=True)

                # Ensure all required columns exist
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = 0

                # Cache to disk
                if use_cache:
                    _write_cache(cache_key, df, ticker)

                return df

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate limited" in err_str or "too many requests" in err_str or "429" in err_str

            if is_rate_limit:
                delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32 seconds
                logger.warning(f"Rate limited for {ticker}, retry {attempt + 1}/{max_retries} after {delay}s")
            else:
                delay = base_delay * (attempt + 1)
                logger.warning(f"Yahoo fetch attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")

            if attempt < max_retries - 1:
                time.sleep(delay)

    # Last resort: try with .NS suffix for Indian stocks
    if not ticker.endswith(".NS") and not ticker.endswith(".BO") and "^" not in ticker:
        try:
            ns_ticker = f"{ticker}.NS"
            logger.info(f"Retrying {ticker} as {ns_ticker}")
            df = yf.download(ns_ticker, interval=interval, period=period, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                df = clean_columns(df)
                df.dropna(inplace=True)
                if use_cache:
                    _write_cache(cache_key, df, ticker)
                return df
        except Exception:
            pass

    return pd.DataFrame()


def fetch_batch(tickers, period="1y"):
    """Fetch multiple tickers in a single yfinance download (faster)."""
    try:
        df = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return df
    except Exception as e:
        logger.error(f"Batch fetch failed: {e}")
        return pd.DataFrame()


def get_cached_tickers():
    """List all tickers currently in cache."""
    tickers = set()
    for f in CACHE_DIR.glob("*.meta"):
        try:
            with open(f) as fh:
                meta = json.load(fh)
                tickers.add(meta.get("ticker", f.stem))
        except Exception:
            pass
    return sorted(tickers)


def clear_cache():
    """Clear all cached data."""
    count = 0
    for f in CACHE_DIR.glob("*"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def pct_change(a, b):
    if b == 0:
        return 0
    return ((a - b) / b) * 100


def safe_round(v, d=2):
    try:
        return round(float(v), d)
    except Exception:
        return None
