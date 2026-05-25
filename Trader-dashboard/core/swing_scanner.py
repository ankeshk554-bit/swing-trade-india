"""
Swing Trade Setup Scanner
Screens stocks on weekly/monthly data for high-probability swing trades.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from core.utils import load_data
from core.indicators import compute_indicators, detect_vcp


def scan_swing_setups(
    symbols,
    interval="1wk",
    period="1y",
    min_price=20,
    max_price=50000,
    top_n=6,
    min_score=55,
):
    """
    Screen stocks for swing trade setups on weekly/monthly timeframe.
    
    Key improvements:
    - Stop: 2.0x ATR (tighter than 2.5x) + recent swing low floor
    - Target 1: 3.0x ATR (3:1 R:R minimum)
    - Target 2: 5.0x ATR (5:1 R:R)
    - Hard gates: SuperTrend UP, ADX > 20, RSI > 50
    - Minimum R:R >= 2.0
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
            
            close = latest.get("Close", 0)
            if close < min_price or close > max_price:
                continue
            
            # ----- HARD GATES -----
            st_dir = latest.get("SuperTrend_Dir", -1)
            adx = latest.get("ADX", 0)
            rsi = latest.get("RSI", 50)
            
            if st_dir != 1:   continue   # SuperTrend UP
            if adx < 20:      continue   # ADX trending
            if rsi <= 50:     continue   # RSI bullish
            
            score = 0
            reasons = []
            concerns = []
            vcp_data = None
            
            # 1. EMA Stack
            ema20 = latest.get("EMA20", 0)
            ema50 = latest.get("EMA50", 0)
            ema200 = latest.get("EMA200", 0)
            
            if ema20 > 0 and ema50 > 0 and ema200 > 0:
                if close > ema20 > ema50 > ema200:
                    score += 18; reasons.append("Perfect EMA stack (P>20>50>200)")
                elif close > ema20 > ema50:
                    score += 12; reasons.append("Partial EMA stack (P>20>50)")
                elif close > ema20:
                    score += 6;  reasons.append("Above EMA20")
                else:
                    concerns.append("Below EMA20")
            
            # 2. ADX
            if adx > 35:
                score += 14; reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 25:
                score += 10; reasons.append(f"Trending (ADX {adx:.1f})")
            else:
                score += 5;  reasons.append(f"Mild trend (ADX {adx:.1f})")
            
            # 3. RSI
            if 58 <= rsi <= 72:
                score += 14; reasons.append(f"RSI {rsi:.1f} (sweet spot)")
            elif 55 <= rsi < 58:
                score += 10; reasons.append(f"RSI {rsi:.1f} (building)")
            elif 72 < rsi <= 78:
                score += 6;  concerns.append(f"RSI {rsi:.1f} (overbought)")
            elif 50 < rsi < 55:
                score += 5;  reasons.append(f"RSI {rsi:.1f} (neutral-bullish)")
            else:
                concerns.append(f"RSI {rsi:.1f}")
            
            # 4. MACD
            prev_macd = prev.get("MACD_Hist", 0)
            macd_hist = latest.get("MACD_Hist", 0)
            macd_line = latest.get("MACD", 0)
            signal_line = latest.get("Signal", 0)
            if macd_hist > 0 and macd_hist > prev_macd:
                score += 14; reasons.append("MACD rising positive")
            elif macd_hist > 0:
                score += 8;  reasons.append("MACD positive")
            elif macd_line > signal_line:
                score += 4;  reasons.append("MACD above signal")
            
            # 5. Volume
            rvol = latest.get("RVOL", 1.0)
            if rvol > 2.0:
                score += 14; reasons.append(f"High volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.5:
                score += 10; reasons.append(f"Good volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.2:
                score += 6;  reasons.append(f"Above avg volume (RVOL {rvol:.2f}x)")
            else:
                score += 2
            
            # 6. VCP Pattern
            try:
                vcp_data = detect_vcp(df)
                if vcp_data and vcp_data.get("is_vcp", False):
                    score += 14
                    reasons.append(f"VCP Stage {vcp_data.get('stage', '?')} (tight: {vcp_data.get('tightness', 0):.1f}%)")
            except Exception:
                pass
            
            # 7. SuperTrend quality
            prev_st = prev.get("SuperTrend_Dir", 1)
            if prev_st == -1:
                score += 10; reasons.append("Fresh SuperTrend BUY flip (weekly)")
            else:
                score += 5;  reasons.append("SuperTrend UP (established)")
            
            # 8. EMA slope
            if ema20 > 0 and len(df) > 10:
                ema20_5 = df["EMA20"].iloc[-5]
                slope = (ema20 - ema20_5) / ema20_5 * 100
                if slope > 1.0:
                    score += 8; reasons.append(f"EMA20 rising ({slope:.2f}%/period)")
                elif slope > 0:
                    score += 4; reasons.append(f"EMA20 flat-positive ({slope:.2f}%/period)")
            
            # 9. Consolidation breakout
            if len(df) > 20:
                high_10 = df["High"].rolling(10).max()
                low_10 = df["Low"].rolling(10).min()
                rng_10 = (high_10 - low_10) / low_10 * 100
                if rng_10.iloc[-1] < 8 and close >= high_10.iloc[-1] * 0.99:
                    score += 8; reasons.append("Consolidation breakout")
                elif rng_10.iloc[-1] < 12:
                    score += 4; reasons.append("Tight range")
            
            # 10. Near 52w high
            if len(df) > 50:
                high_52w = df["High"].rolling(50).max().iloc[-1]
                dist_52 = (high_52w - close) / high_52w * 100
                if dist_52 < 5:
                    score += 6; reasons.append(f"Near 52w high ({dist_52:.1f}% below)")
                elif dist_52 < 15:
                    score += 3; reasons.append(f"Within 15% of 52w high")
            
            # ----- Calculate Stop & Targets -----
            atr = latest.get("ATR", close * 0.02)
            
            # Dynamic stop: 2.0x ATR below entry, but not above recent swing low
            lookback = min(14, len(df) - 1)
            recent_low = df["Low"].iloc[-lookback:].min()
            stop_atr = close - (2.0 * atr)
            stop_level = max(stop_atr, recent_low - 0.5 * atr)
            stop_level = max(stop_level, close - (3.0 * atr))
            
            stop_loss = stop_level
            target_1 = close + (3.0 * atr)   # 3.0R
            target_2 = close + (5.0 * atr)   # 5.0R
            
            risk = close - stop_loss
            reward_1 = target_1 - close
            rr = reward_1 / risk if risk > 0 else 0
            
            # ----- FINAL GATE: R:R >= 2.0 -----
            if rr < 2.0:
                continue
            
            if rr >= 4.0:
                score += 8; reasons.append(f"Excellent R:R ({rr:.2f})")
            elif rr >= 3.0:
                score += 5; reasons.append(f"Great R:R ({rr:.2f})")
            
            score = max(0, min(100, score))
            if score < min_score:
                continue
            
            tf_label = "Weekly" if interval == "1wk" else "Monthly"
            
            setup = {
                "Symbol": symbol,
                "Score": score,
                "Timeframe": tf_label,
                "Interval": interval,
                "Entry": round(close, 2),
                "StopLoss": round(stop_loss, 2),
                "Target1": round(target_1, 2),
                "Target2": round(target_2, 2),
                "R:R": round(rr, 2),
                "ATR%": round(atr / close * 100, 2),
                "LastPrice": round(close, 2),
                "RSI": round(rsi, 1),
                "ADX": round(adx, 1),
                "RVOL": round(rvol, 2),
                "SuperTrend": "UP",
                "EMA_Stack": "✅" if close > ema20 > ema50 > ema200 else "⚠️",
                "VCP": vcp_data.get("is_vcp", False) if vcp_data else False,
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
    return scan_swing_setups(nifty_stocks, interval="1wk", period="1y", top_n=top_n)


def get_monthly_picks(nifty_stocks, top_n=4):
    """Get monthly swing trade setups."""
    return scan_swing_setups(nifty_stocks, interval="1mo", period="2y", top_n=top_n)
