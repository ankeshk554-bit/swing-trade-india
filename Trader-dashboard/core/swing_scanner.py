"""
Swing Trade Setup Scanner
Screens stocks on weekly/monthly data for high-probability swing trades.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from core.utils import load_data
from core.indicators import compute_indicators, detect_vcp


# ── Scoring Weights ──
W = {
    "trend_stack": 15,
    "adx": 10,
    "rsi": 10,
    "macd": 10,
    "volume": 10,
    "vcp": 12,
    "supertrend": 8,
    "ema_slope": 8,
    "consolidation": 7,
    "relative_strength": 10,
}


def scan_swing_setups(
    symbols,
    interval="1wk",
    period="1y",
    min_price=20,
    max_price=50000,
    top_n=6,
    min_score=50,
):
    """
    Screen stocks for swing trade setups on weekly/monthly timeframe.
    
    Parameters
    ----------
    symbols : list
        Stock symbols to scan
    interval : str
        "1wk" (weekly) or "1mo" (monthly)
    period : str
        Lookback period ("6mo", "1y", "2y", "5y")
    min_price : float
        Minimum price filter
    max_price : float
        Maximum price filter
    top_n : int
        Max setups to return
    min_score : int
        Minimum composite score (0-100)
    
    Returns
    -------
    list[dict]
        Top swing setups sorted by score descending
    """
    results = []
    
    for symbol in symbols:
        try:
            df = load_data(symbol, interval=interval, period=period)
            if df is None or df.empty or len(df) < 30:
                continue
            
            df = compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            prev4 = df.iloc[-5] if len(df) > 4 else prev
            
            close = latest.get("Close", 0)
            if close < min_price or close > max_price:
                continue
            
            score = 0
            reasons = []
            concerns = []
            
            # ── 1. EMA Stack (P > EMA20 > EMA50 > EMA200) ──
            ema20 = latest.get("EMA20", 0)
            ema50 = latest.get("EMA50", 0)
            ema200 = latest.get("EMA200", 0)
            
            if ema20 > 0 and ema50 > 0 and ema200 > 0:
                if close > ema20 > ema50 > ema200:
                    score += W["trend_stack"]
                    reasons.append("Perfect EMA stack (P>20>50>200)")
                elif close > ema20 > ema50:
                    score += W["trend_stack"] * 0.6
                    reasons.append("Partial EMA stack (P>20>50)")
                elif close > ema20:
                    score += W["trend_stack"] * 0.3
                    reasons.append("Above EMA20")
                else:
                    concerns.append("Below EMA20")
            
            # ── 2. ADX Trend Strength ──
            adx = latest.get("ADX", 0)
            if adx > 30:
                score += W["adx"]
                reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 22:
                score += W["adx"] * 0.7
                reasons.append(f"Trending (ADX {adx:.1f})")
            elif adx < 15:
                concerns.append(f"Low trend (ADX {adx:.1f})")
            
            # ── 3. RSI ──
            rsi = latest.get("RSI", 50)
            if 55 < rsi < 75:
                score += W["rsi"]
                reasons.append(f"RSI {rsi:.1f} (sweet spot)")
            elif 45 < rsi <= 55:
                score += W["rsi"] * 0.5
                reasons.append(f"RSI {rsi:.1f} (neutral)")
            elif rsi >= 75:
                score += W["rsi"] * 0.3
                concerns.append(f"RSI {rsi:.1f} (overbought)")
            else:
                concerns.append(f"RSI {rsi:.1f} (weak)")
            
            # ── 4. MACD ──
            macd_hist = latest.get("MACD_Hist", 0)
            prev_macd = prev.get("MACD_Hist", 0)
            macd_line = latest.get("MACD", 0)
            signal = latest.get("Signal", 0)
            
            if macd_hist > 0 and macd_hist > prev_macd:
                score += W["macd"]
                reasons.append("MACD histogram rising positive")
            elif macd_hist > 0:
                score += W["macd"] * 0.6
                reasons.append("MACD positive")
            elif macd_line > signal:
                score += W["macd"] * 0.3
                reasons.append("MACD above signal line")
            
            # ── 5. Volume ──
            rvol = latest.get("RVOL", 1.0)
            if rvol > 2.0:
                score += W["volume"]
                reasons.append(f"High volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.5:
                score += W["volume"] * 0.7
                reasons.append(f"Good volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.2:
                score += W["volume"] * 0.4
                reasons.append(f"Above avg volume (RVOL {rvol:.2f}x)")
            
            # ── 6. VCP Pattern ──
            vcp = detect_vcp(df)
            if vcp and vcp.get("is_vcp", False):
                score += W["vcp"]
                reasons.append(f"VCP Stage {vcp.get('stage', '?')} (tight: {vcp.get('tightness', 0):.1f}%)")
            
            # ── 7. Supertrend ──
            st_dir = latest.get("SuperTrend_Dir", 1)
            prev_st = prev.get("SuperTrend_Dir", 1)
            if st_dir == 1:
                score += W["supertrend"]
                if prev_st == -1:
                    score += 5
                    reasons.append("Fresh SuperTrend BUY flip (weekly)")
                else:
                    reasons.append("SuperTrend UP (weekly)")
            else:
                score -= 5
                concerns.append("SuperTrend DOWN (weekly)")
            
            # ── 8. EMA Slope (momentum) ──
            if ema20 > 0 and len(df) > 10:
                ema20_5bars_ago = df["EMA20"].iloc[-5]
                ema20_slope = (ema20 - ema20_5bars_ago) / ema20_5bars_ago * 100
                if ema20_slope > 1.0:
                    score += W["ema_slope"]
                    reasons.append(f"EMA20 rising ({ema20_slope:.2f}%/5w)")
                elif ema20_slope > 0:
                    score += W["ema_slope"] * 0.5
                    reasons.append(f"EMA20 flat-positive ({ema20_slope:.2f}%/5w)")
            
            # ── 9. Consolidation Breakout Check ──
            if len(df) > 20:
                high_10 = df["High"].rolling(10).max()
                low_10 = df["Low"].rolling(10).min()
                range_10 = (high_10 - low_10) / low_10 * 100
                recent_range = range_10.iloc[-1]
                
                if recent_range < 8 and close >= high_10.iloc[-1] * 0.99:
                    score += W["consolidation"]
                    reasons.append(f"Consolidation breakout ({recent_range:.1f}% range)")
                elif recent_range < 12:
                    score += W["consolidation"] * 0.4
                    reasons.append(f"Tight range ({recent_range:.1f}%)")
            
            # ── 10. Price vs 52-week High ──
            if len(df) > 50:
                high_52w = df["High"].rolling(50).max().iloc[-1]
                dist_from_high = (high_52w - close) / high_52w * 100
                if dist_from_high < 5:
                    score += 5
                    reasons.append(f"Near 52w high ({dist_from_high:.1f}% below)")
                elif dist_from_high < 15:
                    score += 3
                    reasons.append(f"Within 15% of 52w high")
            
            # Clamp score
            score = max(0, min(100, int(score)))
            
            if score < min_score:
                continue
            
            # ── Calculate levels ──
            atr = latest.get("ATR", close * 0.02)
            entry_price = close
            stop_loss = entry_price - (2.5 * atr)
            target_1 = entry_price + (2.0 * atr)
            target_2 = entry_price + (4.0 * atr)
            
            risk = entry_price - stop_loss
            rr_1 = (target_1 - entry_price) / risk if risk > 0 else 0
            
            # Determine timeframe label
            tf_label = "Weekly" if interval == "1wk" else "Monthly"
            
            setup = {
                "Symbol": symbol,
                "Score": score,
                "Timeframe": tf_label,
                "Interval": interval,
                "Entry": round(entry_price, 2),
                "StopLoss": round(stop_loss, 2),
                "Target1": round(target_1, 2),
                "Target2": round(target_2, 2),
                "R:R": round(rr_1, 2),
                "ATR%": round(atr / close * 100, 2),
                "LastPrice": round(close, 2),
                "RSI": round(rsi, 1),
                "ADX": round(adx, 1),
                "RVOL": round(rvol, 2),
                "SuperTrend": "UP" if st_dir == 1 else "DOWN",
                "EMA_Stack": f"{'✅' if close > ema20 > ema50 > ema200 else '⚠️'}",
                "VCP": vcp.get("is_vcp", False) if vcp else False,
                "Reasons": reasons[:4],
                "Concerns": concerns[:3],
            }
            results.append(setup)
            
        except Exception:
            continue
    
    results.sort(key=lambda x: x["Score"], reverse=True)
    return results[:top_n]


def get_weekly_picks(nifty_stocks, top_n=6):
    """Get weekly swing trade setups."""
    return scan_swing_setups(
        symbols=nifty_stocks,
        interval="1wk",
        period="1y",
        top_n=top_n,
    )


def get_monthly_picks(nifty_stocks, top_n=4):
    """Get monthly swing trade setups."""
    return scan_swing_setups(
        symbols=nifty_stocks,
        interval="1mo",
        period="2y",
        top_n=top_n,
    )
