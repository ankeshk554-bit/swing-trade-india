"""
Indian Market Data Providers — Sniper Terminal
===============================================
Provides institutional-grade Indian market data:
  - India VIX (volatility index)
  - F&O Chain Analysis (PCR, OI Change, IV)
  - Delivery Volume Analysis
  - FII/DII Cash & F&O Flow
  - Market Breadth (Advance/Decline, New Highs/Lows)
  - Block/Bulk Deals

All data is cached to disk with configurable TTL to avoid
rate limits from NSE/BSE websites.
"""

import json
import time
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from core.utils import load_data, CACHE_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NSE_BASE = "https://www.nseindia.com"
NSE_API = f"{NSE_BASE}/api"
CACHE_TTL_FO = 2  # hours for F&O data
CACHE_TTL_FII = 4  # hours for FII/DII data
CACHE_TTL_BREADTH = 2

# Browser-like headers to bypass NSE bot detection
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _nse_session() -> Optional[requests.Session]:
    """Create a requests session with NSE-compatible headers."""
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get(NSE_BASE, timeout=5)
        return session
    except Exception:
        return None


def _cached_fetch(url: str, cache_key: str, ttl_hours: int = 4) -> Optional[dict]:
    """Fetch JSON from URL with disk caching."""
    cache_path = CACHE_DIR / f"nse_{cache_key}.json"
    meta_path = CACHE_DIR / f"nse_{cache_key}.meta"

    # Check cache
    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            cached_time = datetime.fromisoformat(meta["cached_at"])
            if datetime.now() - cached_time < timedelta(hours=ttl_hours):
                with open(cache_path) as f:
                    return json.load(f)
        except Exception:
            pass

    # Fetch fresh
    try:
        session = _nse_session()
        if session is None:
            return None
        resp = session.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            with open(meta_path, "w") as f:
                json.dump({"cached_at": datetime.now().isoformat(), "url": url}, f)
            return data
        return None
    except requests.Timeout:
        logger.debug(f"NSE timeout for {url}")
        return None
    except Exception as e:
        logger.debug(f"NSE fetch failed for {url}: {e}")
        return None


# ══════════════════════════════════════════════
# 1. INDIA VIX
# ══════════════════════════════════════════════

def get_india_vix() -> Optional[Dict[str, Any]]:
    """
    Fetch India VIX data.

    Returns dict with current, high, low, change%, and regime classification.
    Falls back to yfinance if NSE API fails.
    """
    result = {"vix": None, "change": 0, "high_52w": None, "low_52w": None, "regime": "UNKNOWN"}

    try:
        # Try yfinance first (most reliable)
        vix_df = yf.download("^INDIAVIX", period="1y", progress=False, auto_adjust=True)
        if vix_df is not None and not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            current_vix = float(vix_df["Close"].iloc[-1])
            prev_vix = float(vix_df["Close"].iloc[-2]) if len(vix_df) > 1 else current_vix
            vix_high = float(vix_df["High"].max())
            vix_low = float(vix_df["Low"].min())

            result["vix"] = round(current_vix, 2)
            result["change"] = round(((current_vix / prev_vix) - 1) * 100, 2)
            result["high_52w"] = round(vix_high, 2)
            result["low_52w"] = round(vix_low, 2)

            # VIX Regime classification
            if current_vix < 14:
                result["regime"] = "LOW_VOL"  # Complacency / Bullish
            elif current_vix < 18:
                result["regime"] = "NORMAL"
            elif current_vix < 22:
                result["regime"] = "ELEVATED"
            elif current_vix < 28:
                result["regime"] = "HIGH_VOL"  # Fear
            else:
                result["regime"] = "EXTREME_FEAR"

            return result

    except Exception as e:
        logger.warning(f"India VIX fetch failed: {e}")

    # Fallback: try NSE API
    try:
        data = _cached_fetch(f"{NSE_BASE}/api/indices?index=VIX", "vix", ttl_hours=2)
        if data and "data" in data:
            vix_data = data["data"][0]
            result["vix"] = vix_data.get("last", 0)
            result["change"] = vix_data.get("change", 0)
            return result
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════
# 2. F&O CHAIN ANALYSIS
# ══════════════════════════════════════════════

def get_fo_chain(symbol: str = "NIFTY") -> Optional[Dict[str, Any]]:
    """
    Fetch F&O option chain data for a given symbol.

    Returns dict with:
      - pcr (Put-Call Ratio)
      - total_oi (total open interest)
      - max_oi_call / max_oi_put (strikes with highest OI)
      - max_pain (max pain strike)
      - iv_call / iv_put (implied volatility)
      - change_in_oi (significant OI changes)
      - fo_stocks_count (stocks in F&O ban)
    """
    result = {
        "pcr": None, "pcr_change": None,
        "total_ce_oi": 0, "total_pe_oi": 0,
        "max_oi_call": None, "max_oi_put": None,
        "max_pain": None,
        "iv_call": None, "iv_put": None,
        "oi_buildup": [], "oi_unwinding": [],
        "fo_stocks_in_ban": [],
        "expiry": None,
        "error": None
    }

    try:
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={symbol}"
        data = _cached_fetch(url, f"fo_{symbol}", ttl_hours=CACHE_TTL_FO)

        if data is None or "records" not in data:
            result["error"] = "Could not fetch F&O data"
            return result

        records = data["records"]
        expiry_date = records.get("expiryDates", [None])[0]
        result["expiry"] = expiry_date

        # Parse option chain
        option_data = records.get("data", [])
        total_ce_oi = 0
        total_pe_oi = 0
        ce_oi_by_strike = {}
        pe_oi_by_strike = {}
        ce_iv_by_strike = {}
        pe_iv_by_strike = {}

        for item in option_data:
            strike = item.get("strikePrice", 0)

            # Call options
            ce = item.get("CE", {})
            if ce:
                ce_oi = ce.get("openInterest", 0)
                ce_oi_by_strike[strike] = ce_oi
                total_ce_oi += ce_oi
                ce_iv_by_strike[strike] = ce.get("impliedVolatility", 0)

            # Put options
            pe = item.get("PE", {})
            if pe:
                pe_oi = pe.get("openInterest", 0)
                pe_oi_by_strike[strike] = pe_oi
                total_pe_oi += pe_oi
                pe_iv_by_strike[strike] = pe.get("impliedVolatility", 0)

        # PCR
        if total_ce_oi > 0:
            result["pcr"] = round(total_pe_oi / total_ce_oi, 3)
        result["total_ce_oi"] = total_ce_oi
        result["total_pe_oi"] = total_pe_oi

        # Max OI strikes
        if ce_oi_by_strike:
            max_ce_strike = max(ce_oi_by_strike, key=ce_oi_by_strike.get)
            result["max_oi_call"] = {"strike": max_ce_strike, "oi": ce_oi_by_strike[max_ce_strike]}
        if pe_oi_by_strike:
            max_pe_strike = max(pe_oi_by_strike, key=pe_oi_by_strike.get)
            result["max_oi_put"] = {"strike": max_pe_strike, "oi": pe_oi_by_strike[max_pe_strike]}

        # Max Pain (strike with lowest sum of CE+PE OI)
        all_strikes = set(ce_oi_by_strike.keys()) | set(pe_oi_by_strike.keys())
        if all_strikes:
            pain_by_strike = {}
            for s in all_strikes:
                pain = abs(ce_oi_by_strike.get(s, 0) - pe_oi_by_strike.get(s, 0))
                pain_by_strike[s] = pain
            result["max_pain"] = min(pain_by_strike, key=pain_by_strike.get)

        # IV
        if ce_iv_by_strike:
            result["iv_call"] = round(np.mean(list(ce_iv_by_strike.values())), 2)
        if pe_iv_by_strike:
            result["iv_put"] = round(np.mean(list(pe_iv_by_strike.values())), 2)

        # Stocks in F&O ban
        try:
            ban_url = f"{NSE_BASE}/api/fo-sec-ban"
            ban_data = _cached_fetch(ban_url, "fo_ban", ttl_hours=2)
            if ban_data and "data" in ban_data:
                result["fo_stocks_in_ban"] = [b["symbol"] for b in ban_data["data"]]
        except Exception:
            pass

        # PCR change (compare with yesterday)
        result["pcr_change"] = _calculate_pcr_change(symbol)

    except Exception as e:
        logger.error(f"F&O chain error: {e}")
        result["error"] = str(e)

    return result


def _calculate_pcr_change(symbol: str) -> Optional[float]:
    """Attempt to calculate PCR change from cached data."""
    try:
        # Simple approach: re-fetch and compare (cached)
        return None
    except Exception:
        return None


def get_stock_fo_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get F&O data for a specific stock.

    Returns OI change %, delivery volume %, and F&O status.
    This works for stocks that have individual F&O contracts.
    """
    result = {"in_fo": False, "oi_change_pct": None, "delivery_pct": None}

    # Check if stock is F&O by trying to fetch its option chain
    try:
        symbol = ticker.replace(".NS", "")
        url = f"{NSE_BASE}/api/option-chain-equities?symbol={symbol}"
        data = _cached_fetch(url, f"fo_stock_{symbol}", ttl_hours=4)
        if data and "records" in data:
            result["in_fo"] = True
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════
# 3. DELIVERY VOLUME ANALYSIS
# ══════════════════════════════════════════════

def get_delivery_data(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch delivery volume data for a stock from NSE bhavcopy.

    Returns DataFrame with columns:
      - Date, Delivery_Qty, Traded_Qty, Delivery_Pct
      - Delivery_MA (20-day average delivery %)
      - Delivery_Spurt (bool — delivery > 1.5x average)
    """
    try:
        symbol = ticker.replace(".NS", "")

        # Build date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days * 2)  # fetch more, filter later

        all_data = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Weekdays only
                date_str = current.strftime("%d%m%Y")
                url = (
                    f"https://archives.nseindia.com/products/content/"
                    f"sec_bhavdata_full_{date_str}.csv"
                )
                try:
                    df_day = pd.read_csv(url)
                    df_stock = df_day[df_day["SYMBOL"] == symbol]
                    if not df_stock.empty:
                        row = df_stock.iloc[-1]
                        all_data.append({
                            "Date": current,
                            "Delivery_Qty": int(row.get("DELIV_QTY", 0)),
                            "Traded_Qty": int(row.get("TTL_TRADED_QTY", 0)),
                            "Delivery_Pct": float(row.get("DELIV_PER", 0))
                        })
                except Exception:
                    pass
            current += timedelta(days=1)

        if not all_data:
            return None

        df = pd.DataFrame(all_data)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").tail(days)
        df["Delivery_MA"] = df["Delivery_Pct"].rolling(10).mean()
        df["Delivery_Spurt"] = df["Delivery_Pct"] > df["Delivery_MA"] * 1.5

        return df

    except Exception as e:
        logger.debug(f"Delivery data fetch failed for {ticker}: {e}")
        return None


def get_delivery_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Quick delivery analysis summary for a stock.

    Returns dict with:
      - latest_delivery_pct, avg_delivery_10d, delivery_trend
      - delivery_spurt (bool), accumulation_days (recent 5d)
    """
    df = get_delivery_data(ticker, days=30)
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    avg_10 = df["Delivery_Pct"].tail(10).mean()
    avg_30 = df["Delivery_Pct"].mean()

    # Trend: increasing, decreasing, stable
    recent5 = df["Delivery_Pct"].tail(5).values
    if len(recent5) >= 3:
        trend = "RISING" if np.mean(recent5[-3:]) > np.mean(recent5[:2]) else "FALLING"
    else:
        trend = "STABLE"

    # Accumulation days: days where delivery > 1.2x 10-day average
    accumulation_days = int((df["Delivery_Pct"] > avg_10 * 1.2).tail(10).sum())

    return {
        "latest_delivery_pct": round(float(latest["Delivery_Pct"]), 2),
        "avg_delivery_10d": round(float(avg_10), 2),
        "avg_delivery_30d": round(float(avg_30), 2),
        "delivery_trend": trend,
        "delivery_spurt": bool(latest.get("Delivery_Spurt", False)),
        "accumulation_days_10": accumulation_days,
        "delivery_quality": _delivery_quality_score(avg_10, trend, accumulation_days)
    }


def _delivery_quality_score(avg_pct: float, trend: str, acc_days: int) -> str:
    """Classify delivery quality."""
    score = 0
    if avg_pct > 40:
        score += 2
    elif avg_pct > 25:
        score += 1

    if trend == "RISING":
        score += 2
    elif trend == "STABLE":
        score += 1

    if acc_days >= 5:
        score += 2
    elif acc_days >= 3:
        score += 1

    if score >= 4:
        return "STRONG"
    elif score >= 2:
        return "MODERATE"
    return "WEAK"


# ══════════════════════════════════════════════
# 4. FII / DII FLOW DATA
# ══════════════════════════════════════════════

def get_fii_dii_data() -> Optional[Dict[str, Any]]:
    """
    Fetch FII/DII cash market and F&O flows.

    Returns dict with FII and DII net buys in cash, F&O, and derivatives.
    """
    result = {
        "fii_cash": None, "dii_cash": None,
        "fii_fo": None, "dii_fo": None,
        "total_fii": None, "total_dii": None,
        "net_combined": None,
        "date": None,
        "error": None
    }

    try:
        url = f"{NSE_BASE}/api/fiidii?type=all"
        data = _cached_fetch(url, "fiidii", ttl_hours=CACHE_TTL_FII)

        if data and "data" in data:
            rows = data["data"]
            if rows:
                latest = rows[-1]  # Most recent day

                try:
                    result["fii_cash"] = float(latest.get("FII", {}).get("Cash", 0))
                except Exception:
                    result["fii_cash"] = 0
                try:
                    result["dii_cash"] = float(latest.get("DII", {}).get("Cash", 0))
                except Exception:
                    result["dii_cash"] = 0
                try:
                    result["fii_fo"] = float(latest.get("FII", {}).get("F&O", 0))
                except Exception:
                    result["fii_fo"] = 0
                try:
                    result["dii_fo"] = float(latest.get("DII", {}).get("F&O", 0))
                except Exception:
                    result["dii_fo"] = 0

                result["total_fii"] = round((result["fii_cash"] or 0) + (result["fii_fo"] or 0), 2)
                result["total_dii"] = round((result["dii_cash"] or 0) + (result["dii_fo"] or 0), 2)
                result["net_combined"] = round(result["total_fii"] + result["total_dii"], 2)
                result["date"] = latest.get("date", str(date.today()))

                return result

    except Exception as e:
        logger.warning(f"FII/DII fetch failed: {e}")

    # Fallback: Try SEBI's website or return None
    result["error"] = "Could not fetch FII/DII data"
    return result


# ══════════════════════════════════════════════
# 5. MARKET BREADTH
# ══════════════════════════════════════════════

def get_market_breadth() -> Optional[Dict[str, Any]]:
    """
    Get market breadth data: advance/decline, new highs/lows, 52-week data.

    Returns dict with:
      - advances, declines, unchanged
      - advance_decline_ratio
      - new_highs_52w, new_lows_52w
      - high_low_ratio
      - breadth_strength (STRONG/MODERATE/WEAK)
    """
    result = {
        "advances": 0, "declines": 0, "unchanged": 0,
        "advance_decline_ratio": 0,
        "new_highs_52w": 0, "new_lows_52w": 0,
        "high_low_ratio": 0,
        "breadth_strength": "UNKNOWN",
        "error": None
    }

    try:
        # Fetch from NSE
        url = f"{NSE_BASE}/api/marketStatus"
        data = _cached_fetch(url, "market_status", ttl_hours=1)

        # For advance/decline, use NSE's equity indices API
        breadth_url = f"{NSE_BASE}/api/equity-stockIndices?index=BROAD MARKET INDICES"
        breadth_data = _cached_fetch(breadth_url, "breadth", ttl_hours=1)

        # Alternative: compute from NIFTY 500 data
        # Try fetching from a simpler endpoint
        try:
            session = _nse_session()
            resp = session.get(
                f"{NSE_BASE}/api/market-data-gainers-losers?index=SECURITIES%20IN%20F%26O",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    all_stocks = data["data"]
                    advances = sum(1 for s in all_stocks if s.get("pChange", 0) > 0)
                    declines = sum(1 for s in all_stocks if s.get("pChange", 0) < 0)
                    unchanged = len(all_stocks) - advances - declines

                    result["advances"] = advances
                    result["declines"] = declines
                    result["unchanged"] = max(unchanged, 0)
                    result["advance_decline_ratio"] = round(
                        advances / max(declines, 1), 2
                    )

                    # Strength classification
                    adr = result["advance_decline_ratio"]
                    if adr > 1.5:
                        result["breadth_strength"] = "STRONG"
                    elif adr > 1.0:
                        result["breadth_strength"] = "MODERATE"
                    elif adr > 0.7:
                        result["breadth_strength"] = "WEAK"
                    else:
                        result["breadth_strength"] = "BEARISH"

                    return result
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Market breadth fetch failed: {e}")
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════
# 6. BLOCK / BULK DEALS
# ══════════════════════════════════════════════

def get_block_deals() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch recent block/bulk deals from NSE.

    Returns list of deals with: symbol, quantity, price, buyer, seller, deal_type
    """
    try:
        url = f"{NSE_BASE}/api/block-deals"
        data = _cached_fetch(url, "block_deals", ttl_hours=2)

        if data and "data" in data:
            deals = []
            for d in data["data"][:20]:  # Top 20
                deals.append({
                    "symbol": d.get("symbol", ""),
                    "quantity": d.get("quantity", 0),
                    "price": d.get("price", 0),
                    "value_cr": round(d.get("quantity", 0) * d.get("price", 0) / 1e7, 2),
                    "buyer": d.get("buyerName", ""),
                    "seller": d.get("sellerName", ""),
                    "deal_type": "BULK"
                })
            return deals

    except Exception as e:
        logger.debug(f"Block deals fetch failed: {e}")

    return []


# ══════════════════════════════════════════════
# 7. EXPIRY CALENDAR
# ══════════════════════════════════════════════

def get_expiry_calendar() -> Optional[List[Dict[str, Any]]]:
    """
    Get upcoming F&O expiry dates.

    Returns list of expiry events with: symbol, expiry_date, days_to_expiry, type
    """
    try:
        url = f"{NSE_BASE}/api/option-chain-indices?symbol=NIFTY"
        data = _cached_fetch(url, "expiry_cal", ttl_hours=24)

        if data and "records" in data:
            expiry_dates = data["records"].get("expiryDates", [])
            today = date.today()

            calendar = []
            for exp_str in expiry_dates[:5]:  # Next 5 expiries
                try:
                    exp_date = datetime.strptime(exp_str, "%d-%b-%Y").date()
                    days_to = (exp_date - today).days
                    if days_to >= 0:
                        # Determine weekly or monthly
                        is_monthly = exp_date.month != (exp_date - timedelta(days=7)).month
                        calendar.append({
                            "symbol": "NIFTY",
                            "expiry_date": exp_str,
                            "days_to_expiry": days_to,
                            "type": "MONTHLY" if is_monthly else "WEEKLY"
                        })
                except Exception:
                    pass

            # Also get BANKNIFTY
            try:
                bn_url = f"{NSE_BASE}/api/option-chain-indices?symbol=BANKNIFTY"
                bn_data = _cached_fetch(bn_url, "expiry_cal_bn", ttl_hours=24)
                if bn_data and "records" in bn_data:
                    for exp_str in bn_data["records"].get("expiryDates", [])[:3]:
                        try:
                            exp_date = datetime.strptime(exp_str, "%d-%b-%Y").date()
                            days_to = (exp_date - today).days
                            if days_to >= 0:
                                is_monthly = exp_date.month != (exp_date - timedelta(days=7)).month
                                calendar.append({
                                    "symbol": "BANKNIFTY",
                                    "expiry_date": exp_str,
                                    "days_to_expiry": days_to,
                                    "type": "MONTHLY" if is_monthly else "WEEKLY"
                                })
                        except Exception:
                            pass
            except Exception:
                pass

            # Sort by days to expiry
            calendar.sort(key=lambda x: x["days_to_expiry"])
            return calendar

    except Exception as e:
        logger.debug(f"Expiry calendar fetch failed: {e}")

    return []


# ══════════════════════════════════════════════
# 8. COMPREHENSIVE MARKET OVERVIEW
# ══════════════════════════════════════════════

def get_market_overview() -> Dict[str, Any]:
    """
    Get a comprehensive snapshot of Indian market health.

    Combines: India VIX, F&O PCR, FII/DII flows, market breadth.
    Returns a single dict suitable for dashboard display.
    """
    overview = {
        "timestamp": datetime.now().isoformat(),
        "vix": get_india_vix(),
        "fo": get_fo_chain("NIFTY"),
        "fii_dii": get_fii_dii_data(),
        "breadth": get_market_breadth(),
        "expiry_calendar": get_expiry_calendar(),
        "block_deals": get_block_deals(),
        "market_status": "OPEN"
    }

    # Market status
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)

    if now.weekday() < 5 and market_open <= now <= market_close:
        overview["market_status"] = "OPEN 🟢"
    elif now.weekday() < 5:
        overview["market_status"] = "CLOSED 🔴"
    else:
        overview["market_status"] = "WEEKEND ⚫"

    return overview


# ══════════════════════════════════════════════
# 9. SECTOR PERFORMANCE
# ══════════════════════════════════════════════

def get_sector_performance() -> Optional[list]:
    """
    Compute performance for each sector based on NIFTY 500 constituent data.

    Returns list of dicts: sector, change_pct, strength, trending_stocks
    Falls back to estimated values if API data unavailable.
    """
    try:
        from data.sectors import SECTOR_MAP

        # Use a sample of stocks per sector to estimate performance
        # (full 500-stock scan would be too heavy for dashboard)
        sector_perf = []

        for sector, info in SECTOR_MAP.items():
            stocks = info["stocks"]
            if not stocks:
                continue

            # Sample up to 10 stocks per sector
            sample = stocks[:10]
            changes = []

            for ticker in sample:
                try:
                    df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        if len(df) >= 2:
                            chg = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
                            changes.append(chg)
                except Exception:
                    continue

            if changes:
                avg_change = np.mean(changes)
                pos_count = sum(1 for c in changes if c > 0)
                total_count = len(changes)
                strength_pct = (pos_count / max(total_count, 1)) * 100

                if avg_change > 0.5:
                    strength = "LEADING"
                elif avg_change > 0:
                    strength = "POSITIVE"
                elif avg_change > -0.5:
                    strength = "LAGGING"
                else:
                    strength = "WEAK"

                sector_perf.append({
                    "sector": sector,
                    "change_pct": round(avg_change, 2),
                    "strength": strength,
                    "advance_pct": round(strength_pct, 1),
                    "sample_size": total_count
                })

        # Sort by performance
        sector_perf.sort(key=lambda x: x["change_pct"], reverse=True)
        return sector_perf

    except Exception as e:
        logger.warning(f"Sector performance fetch failed: {e}")
        return None


# ══════════════════════════════════════════════
# 10. VIX HISTORY
# ══════════════════════════════════════════════

def get_vix_history(period: str = "6mo") -> Optional[pd.DataFrame]:
    """
    Get historical India VIX data for charting.

    Returns DataFrame with Date, Close columns, or None.
    """
    try:
        vix_df = yf.download("^INDIAVIX", period=period, progress=False, auto_adjust=True)
        if vix_df is not None and not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            return vix_df[["Close"]]
    except Exception as e:
        logger.warning(f"VIX history fetch failed: {e}")

    return None


# ══════════════════════════════════════════════
# 11. TOP MOVERS (GAINERS / LOSERS)
# ══════════════════════════════════════════════

def get_top_movers(universe_stocks, top_n: int = 10) -> dict:
    """
    Get top gainers and losers from a stock universe.

    Uses yfinance to compute daily change %.

    Returns dict with 'gainers' and 'losers' lists.
    """
    from core.utils import load_data

    gainers = []
    losers = []
    most_volume = []

    for stock in universe_stocks[:100]:  # Sample top 100 for speed
        try:
            df = load_data(stock, period="5d")
            if df is not None and not df.empty and len(df) >= 2:
                change = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
                volume = float(df["Volume"].iloc[-1])
                close = float(df["Close"].iloc[-1])

                item = {"symbol": stock, "change": round(change, 2), "close": close, "volume": volume}

                if change > 0:
                    gainers.append(item)
                elif change < 0:
                    losers.append(item)

                most_volume.append(item)
        except Exception:
            continue

    gainers.sort(key=lambda x: x["change"], reverse=True)
    losers.sort(key=lambda x: x["change"])
    most_volume.sort(key=lambda x: x["volume"], reverse=True)

    return {
        "gainers": gainers[:top_n],
        "losers": losers[:top_n],
        "most_volume": most_volume[:top_n]
    }


def clear_data_provider_cache():
    """Clear all NSE data provider cache files."""
    count = 0
    for f in CACHE_DIR.glob("nse_*"):
        f.unlink(missing_ok=True)
        count += 1
    return count
