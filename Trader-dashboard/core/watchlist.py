"""
Watchlist Manager — Sniper Terminal
====================================
Persistent watchlist management using JSON file storage.
Supports multiple named watchlists with live price refresh.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / ".data"
DATA_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = DATA_DIR / "watchlists.json"


def _load_watchlists() -> dict:
    """Load all watchlists from disk."""
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Watchlist load failed: {e}")
    return {"Default": []}


def _save_watchlists(data: dict):
    """Save all watchlists to disk."""
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Watchlist save failed: {e}")


def get_watchlists() -> list:
    """Get list of all watchlist names."""
    data = _load_watchlists()
    return list(data.keys())


def get_watchlist(name: str = "Default") -> list:
    """Get stocks in a named watchlist."""
    data = _load_watchlists()
    return data.get(name, [])


def create_watchlist(name: str):
    """Create a new watchlist."""
    data = _load_watchlists()
    if name not in data:
        data[name] = []
        _save_watchlists(data)
        return True
    return False


def delete_watchlist(name: str):
    """Delete a watchlist."""
    data = _load_watchlists()
    if name in data and len(data) > 1:
        del data[name]
        _save_watchlists(data)
        return True
    return False


def rename_watchlist(old_name: str, new_name: str):
    """Rename a watchlist."""
    data = _load_watchlists()
    if old_name in data and new_name not in data:
        data[new_name] = data.pop(old_name)
        _save_watchlists(data)
        return True
    return False


def add_to_watchlist(symbol: str, watchlist: str = "Default"):
    """Add a stock to a watchlist."""
    data = _load_watchlists()
    if watchlist not in data:
        data[watchlist] = []
    if symbol not in data[watchlist]:
        data[watchlist].append(symbol)
        _save_watchlists(data)
        return True
    return False


def remove_from_watchlist(symbol: str, watchlist: str = "Default"):
    """Remove a stock from a watchlist."""
    data = _load_watchlists()
    if watchlist in data and symbol in data[watchlist]:
        data[watchlist].remove(symbol)
        _save_watchlists(data)
        return True
    return False


def get_watchlist_with_prices(name: str = "Default") -> Optional[pd.DataFrame]:
    """
    Get watchlist stocks with live price data.

    Returns DataFrame with Symbol, Close, Change%, RSI, Signal, Score.
    """
    from core.utils import load_data
    from core.indicators import compute_indicators, compute_institutional_score, generate_swing_signal

    stocks = get_watchlist(name)
    if not stocks:
        return None

    rows = []
    for symbol in stocks:
        try:
            df = load_data(symbol, period="6mo")
            if df.empty or len(df) < 50:
                continue

            df = compute_indicators(df)
            latest = df.iloc[-1]
            change = (float(latest["Close"]) / float(df.iloc[-2]["Close"]) - 1) * 100 if len(df) > 1 else 0
            score = compute_institutional_score(df)
            signal = generate_swing_signal(df)

            rows.append({
                "Symbol": symbol.replace(".NS", ""),
                "Close": round(float(latest["Close"]), 2),
                "Change%": round(change, 2),
                "RSI": round(float(latest["RSI"]), 1) if "RSI" in latest else None,
                "Score": score,
                "Signal": signal["Signal"],
                "Confidence": signal["Confidence"]
            })
        except Exception as e:
            logger.debug(f"Watchlist price failed for {symbol}: {e}")
            continue

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("Score", ascending=False)
        return df
    return None


def export_watchlist(name: str = "Default") -> Optional[str]:
    """Export a watchlist as CSV string."""
    df = get_watchlist_with_prices(name)
    if df is not None:
        return df.to_csv(index=False)
    return None
