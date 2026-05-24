"""
Shoonya API (Finvasia) Bridge — Sniper Terminal
================================================
Integrates Shoonya trading API for:
  - Fast Indian market data (NSE, BSE, F&O)
  - Real-time quotes
  - Historical data retrieval
  - Order execution (place, modify, cancel)
  - Portfolio & position tracking

Install: pip install shoonya-api

API Docs: https://github.com/algobitz/shoonya-api-py
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config storage
# ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / ".data"
DATA_DIR.mkdir(exist_ok=True)
CRED_FILE = DATA_DIR / "shoonya_creds.json"


def save_credentials(uid: str, password: str, twofa: str, api_key: str = ""):
    """Save Shoonya credentials to encrypted-ish local storage."""
    creds = {
        "uid": uid,
        "password": password,
        "twofa": twofa,
        "api_key": api_key,
        "saved_at": datetime.now().isoformat()
    }
    with open(CRED_FILE, "w") as f:
        json.dump(creds, f)


def load_credentials() -> Optional[dict]:
    """Load saved Shoonya credentials."""
    if CRED_FILE.exists():
        try:
            with open(CRED_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def clear_credentials():
    """Clear saved credentials."""
    if CRED_FILE.exists():
        CRED_FILE.unlink()


# ──────────────────────────────────────────────
# Shoonya API Wrapper
# ──────────────────────────────────────────────

class ShoonyaClient:
    """
    Wrapper around the Shoonya API for market data and trading.

    Handles connection lifecycle, reconnection, and data caching.
    """

    # Exchange mapping
    EXCHANGE_MAP = {
        "NSE": "NSE",
        "BSE": "BSE",
        "NFO": "NFO",  # NSE F&O
        "BFO": "BFO",  # BSE F&O
        "CDS": "CDS",  # Currency
        "MCX": "MCX",  # Commodity
    }

    # Interval mapping for historical data
    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "10m": "10",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "1d": "D",
        "1wk": "W",
        "1mo": "M",
    }

    def __init__(self):
        self.api = None
        self.connected = False
        self.uid = None
        self._connection_lock = threading.Lock()

    def connect(self, uid: str, password: str, twofa: str) -> bool:
        """
        Connect to Shoonya API.

        Args:
            uid: Finvasia user ID
            password: Finvasia password
            twofa: Two-factor auth code (usually PAN + DOB or TOTP)

        Returns:
            True if connected successfully
        """
        try:
            import shoonya_api as shoonya
        except ImportError:
            try:
                import shoonya as shoonya
            except ImportError:
                logger.error("shoonya-api not installed. Run: pip install shoonya-api")
                return False

        with self._connection_lock:
            try:
                self.api = shoonya.Shoonya()
                ret = self.api.login(userid=uid, password=password, twofa=twofa)
                if ret and ret.get("stat") == "Ok":
                    self.connected = True
                    self.uid = uid
                    logger.info(f"Shoonya connected: {uid}")
                    return True
                else:
                    err = ret.get("emsg", "Unknown error") if ret else "No response"
                    logger.error(f"Shoonya login failed: {err}")
                    return False
            except Exception as e:
                logger.error(f"Shoonya connection error: {e}")
                return False

    def disconnect(self):
        """Disconnect from Shoonya API."""
        with self._connection_lock:
            if self.api and self.connected:
                try:
                    self.api.logout()
                except Exception:
                    pass
                self.connected = False
                self.api = None
                logger.info("Shoonya disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected and self.api is not None

    def _ensure_connected(self):
        """Ensure we're connected before making API calls."""
        if not self.is_connected:
            creds = load_credentials()
            if creds:
                self.connect(creds["uid"], creds["password"], creds["twofa"])
            else:
                raise ConnectionError("Shoonya not connected and no saved credentials")

    # ──────────────────────────────────────
    # MARKET DATA
    # ──────────────────────────────────────

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[dict]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE, BSE, NFO, etc.)

        Returns:
            Dict with price data, or None on failure
        """
        self._ensure_connected()
        try:
            token = self._symbol_to_token(symbol, exchange)
            ret = self.api.get_quotes(exchange, token)
            if ret and ret.get("stat") == "Ok":
                return {
                    "symbol": symbol,
                    "ltp": float(ret.get("lp", 0)),
                    "open": float(ret.get("o", 0)),
                    "high": float(ret.get("h", 0)),
                    "low": float(ret.get("l", 0)),
                    "close": float(ret.get("c", 0)),
                    "volume": int(ret.get("v", 0)),
                    "change": float(ret.get("c", 0)) - float(ret.get("o", 0)),
                    "change_pct": round(
                        ((float(ret.get("c", 0)) / float(ret.get("o", 1))) - 1) * 100
                        if float(ret.get("o", 0)) > 0 else 0, 2
                    ),
                    "bid": float(ret.get("bp", 0)),
                    "ask": float(ret.get("sp", 0)),
                    "timestamp": ret.get("tk", ""),
                }
            return None
        except Exception as e:
            logger.warning(f"Shoonya quote failed for {symbol}: {e}")
            return None

    def get_history(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1d",
        days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data from Shoonya.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE, BSE)
            interval: Candle interval
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        self._ensure_connected()
        try:
            interval_code = self.INTERVAL_MAP.get(interval, "D")
            token = self._symbol_to_token(symbol, exchange)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            ret = self.api.get_history(
                exchange=exchange,
                token=token,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval_code,
            )

            if ret and isinstance(ret, list) and len(ret) > 0:
                rows = []
                for bar in ret:
                    rows.append({
                        "Date": bar.get("time", ""),
                        "Open": float(bar.get("into", 0)),
                        "High": float(bar.get("inth", 0)),
                        "Low": float(bar.get("intl", 0)),
                        "Close": float(bar.get("intc", 0)),
                        "Volume": int(bar.get("intv", 0)),
                    })

                df = pd.DataFrame(rows)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").set_index("Date")
                df.dropna(inplace=True)
                return df

            logger.warning(f"Shoonya history returned no data for {symbol}")
            return None

        except Exception as e:
            logger.warning(f"Shoonya history failed for {symbol}: {e}")
            return None

    def get_bulk_quotes(self, symbols: list, exchange: str = "NSE") -> dict:
        """
        Get quotes for multiple symbols at once.

        Args:
            symbols: List of stock symbols
            exchange: Exchange

        Returns:
            Dict of symbol -> quote dict
        """
        results = {}
        for symbol in symbols:
            try:
                q = self.get_quote(symbol, exchange)
                if q:
                    results[symbol] = q
            except Exception:
                continue
        return results

    # ──────────────────────────────────────
    # ORDER MANAGEMENT
    # ──────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        buy_sell: str,
        quantity: int,
        price: float = 0,
        trigger_price: float = 0,
        order_type: str = "LIMIT",
        exchange: str = "NSE",
        validity: str = "DAY",
    ) -> Optional[dict]:
        """
        Place an order.

        Args:
            symbol: Stock symbol
            buy_sell: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Limit price (0 for market orders)
            trigger_price: Trigger price for SL orders
            order_type: 'LIMIT', 'MARKET', 'SL', 'SL-M'
            exchange: Exchange
            validity: 'DAY' or 'IOC'

        Returns:
            Order response dict, or None on failure
        """
        self._ensure_connected()
        try:
            token = self._symbol_to_token(symbol, exchange)

            # Map order type
            dtype_map = {
                "LIMIT": "L",
                "MARKET": "MKT",
                "SL": "SL",
                "SL-M": "SL-M",
            }
            dtype = dtype_map.get(order_type.upper(), "L")

            ret = self.api.place_order(
                buy_or_sell=buy_shell.upper() if buy_sell.upper() in ("BUY", "SELL") else "B",
                exchange=exchange,
                tradingsymbol=token,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                dtype=dtype,
                validity=validity.upper() if validity.upper() in ("DAY", "IOC") else "DAY",
            )

            if ret and ret.get("stat") == "Ok":
                logger.info(f"Order placed: {buy_sell} {quantity} {symbol} @ {price}")
                return {
                    "order_id": ret.get("norenordno", ""),
                    "status": ret.get("status", ""),
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "type": buy_sell,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                err = ret.get("emsg", "Unknown error") if ret else "No response"
                logger.error(f"Order failed: {err}")
                return {"error": err}

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {"error": str(e)}

    def modify_order(self, order_id: str, quantity: int, price: float) -> Optional[dict]:
        """Modify an existing order."""
        self._ensure_connected()
        try:
            ret = self.api.modify_order(
                ordno=order_id,
                quantity=quantity,
                price=price,
            )
            if ret and ret.get("stat") == "Ok":
                return {"order_id": order_id, "status": "MODIFIED"}
            return None
        except Exception as e:
            logger.error(f"Order modify error: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()
        try:
            ret = self.api.cancel_order(ordno=order_id)
            return ret and ret.get("stat") == "Ok"
        except Exception as e:
            logger.error(f"Order cancel error: {e}")
            return False

    def get_order_book(self) -> Optional[pd.DataFrame]:
        """Get all orders."""
        self._ensure_connected()
        try:
            ret = self.api.get_order_book()
            if ret and isinstance(ret, list):
                return pd.DataFrame(ret)
            return None
        except Exception as e:
            logger.error(f"Order book error: {e}")
            return None

    def get_positions(self) -> Optional[pd.DataFrame]:
        """Get current positions."""
        self._ensure_connected()
        try:
            ret = self.api.get_positions()
            if ret and isinstance(ret, list):
                return pd.DataFrame(ret)
            return None
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return None

    def get_trade_book(self) -> Optional[pd.DataFrame]:
        """Get trade history."""
        self._ensure_connected()
        try:
            ret = self.api.get_trade_book()
            if ret and isinstance(ret, list):
                return pd.DataFrame(ret)
            return None
        except Exception as e:
            logger.error(f"Trade book error: {e}")
            return None

    # ──────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────

    def _symbol_to_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Convert a symbol to Shoonya token format."""
        # Remove .NS, .BO suffixes
        clean = symbol.replace(".NS", "").replace(".BO", "").replace("-", "")
        return clean

    def search_symbols(self, query: str) -> Optional[list]:
        """Search for symbols matching query."""
        self._ensure_connected()
        try:
            ret = self.api.searchscrip(exchange="NSE", searchtext=query)
            if ret and isinstance(ret, list):
                return ret
            return None
        except Exception:
            return None


# ──────────────────────────────────────────────
# Global client instance (singleton)
# ──────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def get_client() -> ShoonyaClient:
    """Get or create the global Shoonya client instance."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = ShoonyaClient()
    return _client


def connect(uid: str = None, password: str = None, twofa: str = None) -> bool:
    """
    Convenience function to connect to Shoonya.

    Uses saved credentials if no args provided.
    """
    if not uid or not password or not twofa:
        creds = load_credentials()
        if creds:
            uid = creds.get("uid")
            password = creds.get("password")
            twofa = creds.get("twofa")

    if not all([uid, password, twofa]):
        logger.error("Shoonya credentials not provided")
        return False

    client = get_client()
    success = client.connect(uid, password, twofa)
    if success:
        save_credentials(uid, password, twofa)
    return success


def disconnect():
    """Disconnect from Shoonya."""
    global _client
    if _client:
        _client.disconnect()
        _client = None


# ──────────────────────────────────────────────
# Data source integration (compatible with utils.py)
# ──────────────────────────────────────────────

def load_data_shoonya(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
) -> pd.DataFrame:
    """
    Load stock data from Shoonya API (drop-in replacement for utils.load_data).

    Args:
        ticker: Stock symbol (e.g., "RELIANCE.NS")
        interval: Candle interval
        period: Data period

    Returns:
        DataFrame with OHLCV data, or empty DataFrame on failure
    """
    client = get_client()
    if not client.is_connected:
        logger.warning("Shoonya not connected, trying auto-connect")
        if not connect():
            return pd.DataFrame()

    # Parse period to days
    period_map = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
        "6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825
    }
    days = period_map.get(period, 365)

    # Clean ticker for Shoonya
    clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
    exchange = "NSE" if ".BO" not in ticker else "BSE"

    df = client.get_history(clean_symbol, exchange, interval, days)
    if df is not None and not df.empty:
        # Ensure consistent column format
        df.index.name = "Date"
        df = df.reset_index()
        df.set_index("Date", inplace=True)
        return df

    return pd.DataFrame()
