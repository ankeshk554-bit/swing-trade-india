"""
Shoonya API (Finvasia) Bridge — Sniper Terminal
================================================
Integrates Shoonya trading API (v0.1.4) for:
  - Order execution (place, modify, cancel)
  - Portfolio & position tracking
  - Instrument search & quotes

Install: pip install shoonya

IMPORTANT - SEBI Static IP / URL Whitelisting:
  SEBI mandates that trading APIs must use whitelisted IPs/URLs.
  Shoonya API will ONLY work from your registered IP address.

  If you have an Oracle Cloud static IP:
    1. Contact Finvasia support: support@finvasia.com
    2. Ask them to whitelist your Oracle Cloud static IP for Shoonya API
    3. Provide them: User ID, Static IP address, Purpose (API trading)
    4. Once whitelisted, deploy this app on your Oracle Cloud VM
    5. Connect from the VM - it will work

  To use locally (no whitelisting needed):
    streamlit run app.py
    Then connect Shoonya from the Settings panel - works from your home IP
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / ".data"
DATA_DIR.mkdir(exist_ok=True)
CRED_FILE = DATA_DIR / "shoonya_creds.json"


def save_credentials(uid: str, password: str, panno: str):
    creds = {"uid": uid, "password": password, "panno": panno,
             "saved_at": datetime.now().isoformat()}
    with open(CRED_FILE, "w") as f:
        json.dump(creds, f)


def load_credentials() -> Optional[dict]:
    if CRED_FILE.exists():
        try:
            with open(CRED_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def clear_credentials():
    if CRED_FILE.exists():
        CRED_FILE.unlink()


class ShoonyaClient:
    """Wrapper around the Shoonya API (v0.1.4 by Finvasia)."""

    def __init__(self):
        self.api = None
        self.connected = False
        self.uid = None
        self._lock = threading.Lock()

    def connect(self, uid: str, password: str, panno: str) -> dict:
        """Connect to Shoonya API.

        Args:
            uid: Finvasia user ID (e.g., FA123456)
            password: Trading password
            panno: PAN card (UPPERCASE, e.g., ABCDE1234F)

        Returns:
            {"success": True} or {"success": False, "error": "..."}
        """
        try:
            import shoonya
        except ImportError:
            return {"success": False,
                    "error": "shoonya package not installed. Run: pip install shoonya"}

        with self._lock:
            try:
                self.api = shoonya.Shoonya()
                result = self.api.login_and_get_authorizations(uid, password, panno)
                if result:
                    self.connected = True
                    self.uid = uid
                    save_credentials(uid, password, panno)
                    logger.info(f"Shoonya connected: {uid}")
                    return {"success": True, "error": None}
                return {"success": False, "error": "Login failed. Check credentials."}
            except Exception as e:
                err = str(e)
                logger.error(f"Shoonya connection error: {err}")
                if any(kw in err.lower() for kw in ["ip", "url", "host", "whitelist"]):
                    return {"success": False,
                            "error": "IP not whitelisted. Contact Finvasia support to whitelist your Oracle Cloud static IP."}
                return {"success": False, "error": f"Connection failed: {err}"}

    def disconnect(self):
        with self._lock:
            self.connected = False
            self.api = None

    @property
    def is_connected(self) -> bool:
        return self.connected and self.api is not None

    def _ensure_connected(self):
        if not self.is_connected:
            creds = load_credentials()
            if creds:
                self.connect(creds["uid"], creds["password"], creds["panno"])
            else:
                raise ConnectionError("Shoonya not connected")

    # ──────────────────────────────
    # QUOTE
    # ──────────────────────────────

    def search_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[dict]:
        self._ensure_connected()
        try:
            return self.api.get_instrument_by_symbol(exchange, symbol.upper())
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return None

    # ──────────────────────────────
    # ORDER
    # ──────────────────────────────

    def place_order(self, symbol: str, buy_sell: str, quantity: int,
                    price: float = 0.0, trigger_price: float = 0.0,
                    order_type: str = "MARKET",
                    product_type: str = "INTRADAY") -> dict:
        """Place an order.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            buy_sell: "BUY" or "SELL"
            quantity: Number of shares
            price: Limit price (0 for MARKET)
            trigger_price: Trigger for SL orders
            order_type: "MARKET", "LIMIT", "SL", "SL-M"
            product_type: "INTRADAY" or "DELIVERY"

        Returns:
            {"success": True, "order_id": "..."} or {"success": False, "error": "..."}
        """
        self._ensure_connected()
        try:
            import shoonya

            bs = shoonya.TransactionType.Buy if buy_sell.upper() == "BUY" else shoonya.TransactionType.Sell
            ot_map = {"MARKET": shoonya.OrderType.Market, "LIMIT": shoonya.OrderType.Limit,
                       "SL": shoonya.OrderType.StopLoss, "SL-M": shoonya.OrderType.StopLossMarket}
            ot = ot_map.get(order_type.upper(), shoonya.OrderType.Market)
            pt_map = {"INTRADAY": shoonya.ProductType.Intraday,
                       "DELIVERY": shoonya.ProductType.Delivery,
                       "MARGIN": shoonya.ProductType.Margin}
            pt = pt_map.get(product_type.upper(), shoonya.ProductType.Intraday)

            inst = self.api.get_instrument_by_symbol("NSE", symbol.upper())
            if not inst:
                return {"success": False, "error": f"Symbol {symbol} not found"}

            result = self.api.place_order(
                transaction_type=bs, instrument=inst,
                InstrumentType=shoonya.InstrumentType.Equity,
                quantity=quantity, order_type=ot, product_type=pt,
                price=price,
                trigger_price=trigger_price if trigger_price > 0 else None)

            if result:
                logger.info(f"Order placed: {buy_sell} {quantity} {symbol}")
                return {"success": True, "order_id": str(result), "error": None}
            return {"success": False, "error": "Order placement failed"}
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {"success": False, "error": str(e)}

    def modify_order(self, order_id: str, order_type: str = "LIMIT",
                     quantity: int = None, price: float = 0.0,
                     trigger_price: float = 0.0) -> dict:
        self._ensure_connected()
        try:
            import shoonya
            ot_map = {"MARKET": shoonya.OrderType.Market, "LIMIT": shoonya.OrderType.Limit,
                       "SL": shoonya.OrderType.StopLoss, "SL-M": shoonya.OrderType.StopLossMarket}
            ot = ot_map.get(order_type.upper(), shoonya.OrderType.Limit)
            result = self.api.modify_order(
                order_id=order_id, order_type=ot,
                quantity=quantity, price=price,
                trigger_price=trigger_price if trigger_price > 0 else 0.0)
            return {"success": True, "order_id": order_id} if result else {"success": False, "error": "Modify failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        self._ensure_connected()
        try:
            result = self.api.cancel_order(order_id)
            return {"success": True, "order_id": order_id} if result else {"success": False, "error": "Cancel failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ──────────────────────────────
    # PORTFOLIO
    # ──────────────────────────────

    def get_orders(self) -> Optional[pd.DataFrame]:
        self._ensure_connected()
        try:
            data = self.api.get_orders()
            return pd.DataFrame(data) if data and isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Orders error: {e}")
            return None

    def get_positions(self) -> Optional[pd.DataFrame]:
        self._ensure_connected()
        try:
            data = self.api.get_positions()
            return pd.DataFrame(data) if data and isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return None

    def get_trade_book(self) -> Optional[pd.DataFrame]:
        self._ensure_connected()
        try:
            data = self.api.get_trade_book()
            return pd.DataFrame(data) if data and isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Trade book error: {e}")
            return None

    def get_holdings(self) -> Optional[pd.DataFrame]:
        self._ensure_connected()
        try:
            data = self.api.get_holdings()
            return pd.DataFrame(data) if data and isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Holdings error: {e}")
            return None

    def get_limits(self) -> Optional[dict]:
        self._ensure_connected()
        try:
            return self.api.get_limits()
        except Exception as e:
            logger.error(f"Limits error: {e}")
            return None


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def get_client() -> ShoonyaClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = ShoonyaClient()
    return _client


def connect(uid: str = None, password: str = None, panno: str = None) -> dict:
    """Connect to Shoonya. Returns {"success": True/False, "error": "..."}"""
    if not uid or not password or not panno:
        creds = load_credentials()
        if creds:
            uid, password, panno = creds.get("uid"), creds.get("password"), creds.get("panno")
    if not all([uid, password, panno]):
        return {"success": False, "error": "User ID, Password, and PAN are all required"}
    return get_client().connect(uid, password, panno)


def disconnect():
    global _client
    if _client:
        _client.disconnect()
        _client = None
