"""
Intraday High-Probability Trade Scanner
Screens stocks on 15min data for high-confidence intraday setups.
Returns top 3-4 daily picks with strict entry/stop/target levels.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from core.utils import load_data
from core.indicators import compute_indicators


def scan_intraday_setups(
    symbols,
    min_price=50,
    max_price=10000,
    top_n=4,
    interval="15m",
    period="5d"
):
    """
    Screen a universe of stocks for high-probability intraday setups.
    Key improvements over v1:
    - Tighter stops and wider targets for R:R >= 2.0
    - Stricter composite scoring with minimum score 60/100
    - Hard gates: SuperTrend UP, RVOL >= 1.2, above VWAP, RSI > 50, ADX > 18
    """
    results = []
    
    for symbol in symbols:
        try:
            df = load_data(symbol, interval=interval, period=period)
            if df is None or df.empty or len(df) < 26:
                continue
            
            df = compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            close = latest.get("Close", 0)
            if close < min_price or close > max_price:
                continue
            
            # ----- HARD GATES: ALL must pass -----
            st_dir = latest.get("SuperTrend_Dir", -1)
            rvol = latest.get("RVOL", 0.0)
            rsi = latest.get("RSI", 50)
            vwap_val = latest.get("VWAP", None)
            adx = latest.get("ADX", 0)
            
            if st_dir != 1:          continue   # Gate 1: SuperTrend UP
            if rvol < 1.2:           continue   # Gate 2: Volume above avg
            if vwap_val is None or close <= vwap_val: continue  # Gate 3: Above VWAP
            if rsi <= 50:            continue   # Gate 4: RSI bullish bias
            if adx < 18:             continue   # Gate 5: Some trend
            
            # ----- COMPOSITE SCORE (0-100) -----
            score = 0
            reasons = []
            concerns = []
            
            # 1. VWAP proximity (0-2% above = ideal)
            vwap_dist = (close - vwap_val) / vwap_val * 100
            if vwap_dist < 1.5:
                score += 20; reasons.append(f"At VWAP (+{vwap_dist:.2f}%)")
            elif vwap_dist < 3.0:
                score += 14; reasons.append(f"Near VWAP (+{vwap_dist:.2f}%)")
            elif vwap_dist < 5.0:
                score += 8;  reasons.append(f"Above VWAP (+{vwap_dist:.2f}%)")
            else:
                score += 4;  concerns.append(f"Extended from VWAP (+{vwap_dist:.2f}%)")
            
            # 2. Volume
            if rvol > 2.5:
                score += 18; reasons.append(f"Massive vol (RVOL {rvol:.2f}x)")
            elif rvol > 1.8:
                score += 14; reasons.append(f"Strong vol (RVOL {rvol:.2f}x)")
            elif rvol > 1.4:
                score += 10; reasons.append(f"Good vol (RVOL {rvol:.2f}x)")
            else:
                score += 5;  reasons.append(f"Moderate vol (RVOL {rvol:.2f}x)")
            
            # 3. RSI zone
            if 58 <= rsi <= 72:
                score += 16; reasons.append(f"RSI {rsi:.1f} (ideal momentum)")
            elif 55 <= rsi < 58:
                score += 12; reasons.append(f"RSI {rsi:.1f} (building)")
            elif 72 < rsi <= 78:
                score += 8;  concerns.append(f"RSI {rsi:.1f} (running hot)")
            elif 50 < rsi < 55:
                score += 6;  reasons.append(f"RSI {rsi:.1f} (neutral-bullish)")
            else:
                concerns.append(f"RSI {rsi:.1f}")
            
            # 4. MACD
            prev_macd = prev.get("MACD_Hist", 0)
            macd_hist = latest.get("MACD_Hist", 0)
            if macd_hist > 0 and macd_hist > prev_macd:
                score += 14; reasons.append("MACD rising & positive")
            elif macd_hist > 0:
                score += 8;  reasons.append("MACD positive")
            elif macd_hist < 0 and macd_hist > prev_macd:
                score += 3;  reasons.append("MACD improving")
            else:
                concerns.append("MACD weakening")
            
            # 5. ADX
            if adx > 35:
                score += 12; reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 25:
                score += 8;  reasons.append(f"Trending (ADX {adx:.1f})")
            else:
                score += 4;  reasons.append(f"Mild trend (ADX {adx:.1f})")
            
            # 6. SuperTrend quality
            prev_st = prev.get("SuperTrend_Dir", 1)
            if prev_st == -1:
                score += 8;  reasons.append("Fresh SuperTrend BUY flip")
            else:
                score += 4;  reasons.append("SuperTrend UP (established)")
            
            # 7. EMA structure
            ema20 = latest.get("EMA20", close)
            ema50 = latest.get("EMA50", close)
            if close > ema20 > ema50:
                score += 12; reasons.append("P > EMA20 > EMA50")
            elif close > ema20:
                score += 6;  reasons.append("Above EMA20")
            else:
                score += 2
            
            # 8. Recent price action
            high_10 = df["High"].rolling(10).max()
            if close >= high_10.iloc[-1] * 0.998:
                score += 8; reasons.append("At/near 10-bar high")
            recent_rets = df["Close"].pct_change().tail(3)
            pos = (recent_rets > 0).sum()
            if pos >= 2:
                score += 6; reasons.append(f"{pos}/3 green candles")
            elif pos == 1:
                score += 2
            
            # ----- Calculate Stop & Targets -----
            atr = latest.get("ATR", close * 0.005)
            lookback = min(10, len(df) - 1)
            recent_low = df["Low"].iloc[-lookback:].min()
            
            # Dynamic stop: between 1.0x ATR and 1.8x ATR below entry,
            # but at least below the recent swing low
            stop_atr = close - (1.0 * atr)
            stop_level = max(stop_atr, recent_low - 0.3 * atr)
            stop_level = max(stop_level, close - (1.8 * atr))
            
            stop_loss = stop_level
            target_1 = close + (2.0 * atr)   # 2.0R
            target_2 = close + (4.0 * atr)   # 4.0R
            
            risk = close - stop_loss
            reward_1 = target_1 - close
            rr = reward_1 / risk if risk > 0 else 0
            
            # ----- FINAL GATE: R:R >= 1.8 -----
            if rr < 1.8:
                continue
            
            # Bonus for high R:R
            if rr >= 3.0:
                score += 10; reasons.append(f"Excellent R:R ({rr:.2f})")
            elif rr >= 2.5:
                score += 6;  reasons.append(f"Great R:R ({rr:.2f})")
            elif rr >= 2.0:
                score += 3;  reasons.append(f"Good R:R ({rr:.2f})")
            
            score = max(0, min(100, score))
            if score < 60:
                continue
            
            atr_pct = (atr / close * 100) if close > 0 else 1.0
            
            setup = {
                "Symbol": symbol,
                "Score": score,
                "Direction": "BUY",
                "Entry": round(close, 2),
                "StopLoss": round(stop_loss, 2),
                "Target1": round(target_1, 2),
                "Target2": round(target_2, 2),
                "R:R": round(rr, 2),
                "ATR%": round(atr_pct, 2),
                "LastPrice": round(close, 2),
                "Change%": round(latest.get("Change%", 0), 2),
                "RSI": round(rsi, 1),
                "RVOL": round(rvol, 2),
                "Volume": int(latest.get("Volume", 0)),
                "ADX": round(adx, 1),
                "VWAP_Dist%": round(vwap_dist, 2),
                "Reasons": reasons[:4],
                "Concerns": concerns[:3],
                "Indicators": {
                    "RSI": round(rsi, 1), "ADX": round(adx, 1),
                    "RVOL": round(rvol, 2), "MACD_Hist": round(macd_hist, 6),
                    "SuperTrend": "UP", "ATR": round(atr, 2),
                    "VWAP": round(vwap_val, 2),
                    "Entry_vs_VWAP": f"+{vwap_dist:.2f}%",
                },
                "Time": datetime.now().strftime("%H:%M"),
            }
            results.append(setup)
        except Exception:
            continue
    
    results.sort(key=lambda x: x["Score"], reverse=True)
    return results[:top_n]


def get_daily_picks(nifty_stocks, top_n=4):
    """Convenience wrapper: runs intraday scan on Nifty stocks."""
    return scan_intraday_setups(
        symbols=nifty_stocks,
        min_price=50, max_price=10000,
        top_n=top_n,
        interval="15m", period="5d"
    )
