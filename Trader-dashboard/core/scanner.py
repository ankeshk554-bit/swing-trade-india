import pandas as pd
import numpy as np
import logging
from core.utils import load_data
from core.indicators import (
    compute_indicators, compute_institutional_score,
    generate_swing_signal, detect_vcp,
    compute_delivery_enhanced_score, generate_fo_enhanced_signal
)
from core.patterns import run_all_patterns, get_best_pattern
from core.data_providers import (
    get_delivery_summary, get_stock_fo_data, get_india_vix
)

logger = logging.getLogger(__name__)

# Cache for delivery data to avoid repeated API calls in a single scan
_delivery_cache = {}
_fo_cache = {}


def scan_stock(symbol, include_fundamental=False):
    """
    Scan a single stock and return enriched scan result.

    Now includes F&O status, delivery volume analysis, and enhanced signals.

    Args:
        symbol: Ticker symbol (e.g., "RELIANCE.NS")
        include_fundamental: If True, includes delivery and F&O data (slower)

    Returns dict with technical + pattern + score data, or None on failure.
    """
    try:
        df = load_data(symbol, period="2y")
        if df.empty or len(df) < 220:
            return None

        df = compute_indicators(df)
        latest = df.iloc[-1]

        # Base scoring
        score = compute_institutional_score(df)
        signal_info = generate_swing_signal(df)
        vcp = detect_vcp(df)

        # Run pattern detection
        patterns = run_all_patterns(df)
        best_pattern = get_best_pattern(patterns)
        near_52w_high = patterns.get("52w", {}).get("near_high", False)
        uptrend_structure = patterns.get("structure", {}).get("type") == "UPTREND_HH_HL"

        result = {
            "Symbol": symbol,
            "Close": round(float(latest["Close"]), 2),
            "Change%": round(float(latest["Close"] / df.iloc[-2]["Close"] - 1) * 100, 2) if len(df) > 1 else 0,
            "RSI": round(float(latest["RSI"]), 2),
            "RVOL": round(float(latest["RVOL"]), 2),
            "EMA50": round(float(latest["EMA50"]), 2),
            "EMA200": round(float(latest["EMA200"]), 2),
            "ATR%": round(float(latest["ATR_PCT"]), 2),
            "MACD": round(float(latest["MACD_HIST"]), 4),
            "Supertrend": "UP" if latest["SUPERTREND_DIR"] == 1 else "DOWN",
            "BB_Pos": round(float(latest["BB_POS"]), 2),
            "Score": score,
            "Signal": signal_info["Signal"],
            "Confidence": signal_info["Confidence"],
            "Reason": signal_info["Reasons"],
            "VCP": vcp["VCP_Flag"],
            "VCP_Stage": vcp["Stage"],
            # Pattern detection
            "Pattern": best_pattern["name"],
            "Pattern_Conf": best_pattern["confidence"],
            "Near_52W_High": near_52w_high,
            "Uptrend_HH_HL": uptrend_structure,
            # F&O & Delivery (will be filled if available)
            "In_FO": False,
            "Delivery%": None,
            "Delivery_Quality": None,
            "Delivery_Trend": None
        }

        # Enhanced data (slower — fetched on demand)
        if include_fundamental:
            # Delivery data
            del_key = symbol.replace(".NS", "")
            if del_key not in _delivery_cache:
                _delivery_cache[del_key] = get_delivery_summary(symbol)
            delivery = _delivery_cache[del_key]

            if delivery:
                result["Delivery%"] = delivery.get("latest_delivery_pct")
                result["Delivery_Quality"] = delivery.get("delivery_quality")
                result["Delivery_Trend"] = delivery.get("delivery_trend")

            # F&O data
            fo_key = symbol.replace(".NS", "")
            if fo_key not in _fo_cache:
                _fo_cache[fo_key] = get_stock_fo_data(symbol)
            fo_data = _fo_cache[fo_key]

            if fo_data and fo_data.get("in_fo", False):
                result["In_FO"] = True

            # Enhanced signal with delivery + F&O context
            vix_info = get_india_vix()
            enhanced = generate_fo_enhanced_signal(
                df,
                delivery_summary=delivery,
                fo_data=fo_data if fo_data else None,
                vix_info=vix_info
            )
            result["Enhanced_Signal"] = enhanced["Signal"]
            result["Enhanced_Confidence"] = enhanced["Confidence"]
            result["Enhanced_Reason"] = enhanced["Reasons"]

        return result

    except Exception as e:
        logger.warning(f"Scan failed for {symbol}: {e}")
        return None


def scan_universe(stocks, progress_callback=None, include_fundamental=False):
    """
    Scan an entire universe of stocks with optional fundamental data.

    Args:
        stocks: List of ticker symbols
        progress_callback: Optional callable(current, total) for UI updates
        include_fundamental: Include delivery volume & F&O data (slower)

    Returns:
        List of scan result dicts (sorted by Score descending)
    """
    global _delivery_cache, _fo_cache
    _delivery_cache = {}
    _fo_cache = {}

    results = []
    total = len(stocks)

    for idx, stock in enumerate(stocks):
        result = scan_stock(stock, include_fundamental=include_fundamental)
        if result is not None:
            results.append(result)

        if progress_callback:
            progress_callback(idx + 1, total)

    results.sort(key=lambda x: x.get("Score", 0), reverse=True)
    return results




    results.sort(key=lambda x: x["Score"], reverse=True)
    return results


def filter_by_strategy(results, strategy="ALL"):
    """
    Filter scan results by predefined strategy themes.

    Now uses the scan_engine STRATEGIES definitions for consistent filtering.
    Falls back to simple signal/supertrend/score matching.
    """
    if strategy == "ALL":
        return results

    # Legacy/simple strategy filters (quick match without full re-evaluation)
    strategy_map = {
        "MOMENTUM": lambda r: r.get("Signal") == "BUY" and r.get("Confidence", 0) >= 3,
        "VCP_BREAKOUT": lambda r: r.get("VCP", False) is True and r.get("Signal") == "BUY",
        "MEAN_REVERSION": lambda r: r.get("BB_Pos") is not None and r.get("BB_Pos", 1) < 0.15,
        "STRONG_TREND": lambda r: r.get("Supertrend") == "UP" and r.get("Score", 0) >= 75,
        "WEAK": lambda r: r.get("Signal") == "SELL",
        "MOMENTUM_RUNNER": lambda r: r.get("Signal") == "BUY" and r.get("Confidence", 0) >= 3 and r.get("RVOL", 0) > 1.3,
        "DELIVERY_SPURT": lambda r: r.get("Delivery_Quality") == "STRONG" and r.get("Delivery%", 0) or 0 > 35,
        "BREAKOUT_52W": lambda r: r.get("Score", 0) >= 75 and r.get("Signal") == "BUY" and r.get("RVOL", 0) > 1.5,
        "GOLDEN_CROSS": lambda r: r.get("Signal") == "BUY" and r.get("Confidence", 0) >= 3 and r.get("Supertrend") == "UP",
        "BULL_FLAG": lambda r: r.get("Signal") == "BUY" and r.get("VCP", False) is True,
        "CONSOLIDATION_BREAKOUT": lambda r: r.get("Signal") == "BUY" and r.get("RVOL", 0) > 1.5 and r.get("Score", 0) >= 60,
        "MACD_MOMENTUM": lambda r: r.get("Signal") == "BUY" and r.get("Confidence", 0) >= 2,
    }

    filter_fn = strategy_map.get(strategy)
    if filter_fn:
        filtered = [r for r in results if filter_fn(r)]
        return filtered if filtered else results

    return results