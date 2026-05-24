"""
Intraday High-Probability Trade Scanner
Screens stocks on 15min data for high-confidence intraday setups.
Returns top 3-4 daily picks with entry/stop/target levels.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    
    Parameters
    ----------
    symbols : list
        Stock symbols to scan (e.g. ["RELIANCE.NS", "HDFCBANK.NS"])
    min_price : float
        Minimum stock price filter
    max_price : float
        Maximum stock price filter
    top_n : int
        Number of top picks to return
    interval : str
        Data interval ("5m", "15m", "30m")
    period : str
        Lookback period ("2d", "5d", "1mo")
    
    Returns
    -------
    list[dict]
        Top setups sorted by composite score, each with:
        Symbol, Score, Direction, Entry, StopLoss, Targets,
        Reasoning, Indicators dict
    """
    results = []
    
    for symbol in symbols:
        try:
            df = load_data(symbol, interval=interval, period=period)
            if df is None or df.empty or len(df) < 20:
                continue
            
            df = compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            prev2 = df.iloc[-3] if len(df) > 2 else prev
            
            # Basic price filter
            close = latest.get("Close", 0)
            if close < min_price or close > max_price:
                continue
            
            # ---- Long Setup Score (0-100) ----
            score = 50  # baseline
            
            reasons = []
            concerns = []
            
            # 1. VWAP position (most important intraday)
            vwap = latest.get("VWAP", None)
            if vwap is not None and close > vwap:
                vwap_dist = (close - vwap) / vwap * 100
                if vwap_dist < 2:  # Within 2% above VWAP = good entry
                    score += 12
                    reasons.append(f"Above VWAP (+{vwap_dist:.2f}%)")
                elif vwap_dist < 5:
                    score += 6
                    reasons.append(f"Well above VWAP (+{vwap_dist:.2f}%)")
                else:
                    score += 3
                    concerns.append(f"Extended from VWAP (+{vwap_dist:.2f}%)")
            else:
                score -= 15
                concerns.append("Below VWAP")
            
            # 2. Volume confirmation
            rvol = latest.get("RVOL", 1.0)
            if rvol > 2.0:
                score += 15
                reasons.append(f"High volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.5:
                score += 10
                reasons.append(f"Good volume (RVOL {rvol:.2f}x)")
            elif rvol > 1.2:
                score += 5
                reasons.append(f"Above avg volume (RVOL {rvol:.2f}x)")
            else:
                score -= 5
                concerns.append(f"Low volume (RVOL {rvol:.2f}x)")
            
            # 3. RSI momentum
            rsi = latest.get("RSI", 50)
            if 55 < rsi < 75:
                score += 12
                reasons.append(f"RSI {rsi:.1f} (bullish momentum)")
            elif rsi >= 75:
                score += 5
                concerns.append(f"RSI {rsi:.1f} (overbought)")
            elif 45 < rsi <= 55:
                score += 2
                reasons.append(f"RSI {rsi:.1f} (neutral)")
            else:
                score -= 10
                concerns.append(f"RSI {rsi:.1f} (weak)")
            
            # 4. MACD check
            macd_hist = latest.get("MACD_Hist", 0)
            prev_macd = prev.get("MACD_Hist", 0)
            if macd_hist > 0 and macd_hist > prev_macd:
                score += 10
                reasons.append("MACD rising positive")
            elif macd_hist > 0:
                score += 5
                reasons.append("MACD positive")
            elif macd_hist < 0 and macd_hist > prev_macd:
                score += 2
                reasons.append("MACD improving")
            else:
                score -= 8
                concerns.append("MACD negative")
            
            # 5. Supertrend
            st_dir = latest.get("SuperTrend_Dir", 1)
            if st_dir == 1:
                score += 8
                # Bonus for fresh flip
                prev_st = prev.get("SuperTrend_Dir", 1)
                if prev_st == -1 and st_dir == 1:
                    score += 5
                    reasons.append("Fresh SuperTrend BUY flip")
                else:
                    reasons.append("SuperTrend UP")
            else:
                score -= 10
                concerns.append("SuperTrend DOWN")
            
            # 6. ADX trend strength
            adx = latest.get("ADX", 15)
            if adx > 30:
                score += 8
                reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 22:
                score += 5
                reasons.append(f"Trending (ADX {adx:.1f})")
            elif adx < 15:
                score -= 5
                concerns.append(f"Low trend (ADX {adx:.1f})")
            
            # 7. Price action — recent high breakout
            high_10 = df["High"].rolling(10).max()
            if close >= high_10.iloc[-1] * 0.995:  # Within 0.5% of 10-bar high
                score += 8
                reasons.append("Near 10-bar high")
            
            # 8. EMA position
            ema20 = latest.get("EMA20", close)
            ema50 = latest.get("EMA50", close)
            if close > ema20 > ema50:
                score += 10
                reasons.append("P > EMA20 > EMA50")
            elif close > ema20:
                score += 5
                reasons.append("Above EMA20")
            elif close < ema20:
                score -= 5
                concerns.append("Below EMA20")
            
            # 9. Consecutive candles — recent momentum
            recent_returns = df["Close"].pct_change().tail(3)
            pos_candles = (recent_returns > 0).sum()
            if pos_candles >= 2:
                score += 5
                reasons.append(f"{pos_candles}/3 recent green candles")
            
            # ---- Calculate Target & Stop ----
            atr = latest.get("ATR", close * 0.01)
            atr_pct = (atr / close * 100) if close > 0 else 1.0
            
            entry_price = close
            stop_loss = entry_price - (2.0 * atr)
            target_1 = entry_price + (1.5 * atr)  # 1.5R
            target_2 = entry_price + (3.0 * atr)  # 3R
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward_1 = target_1 - entry_price
            rr_ratio = reward_1 / risk if risk > 0 else 0
            
            if rr_ratio < 1.0:
                score -= 10
                concerns.append(f"Low R:R ({rr_ratio:.2f})")
            elif rr_ratio > 2.0:
                score += 5
                reasons.append(f"Good R:R ({rr_ratio:.2f})")
            
            # ---- Direction ----
            direction = "BUY" if score >= 50 else ("NEUTRAL" if score >= 35 else "AVOID")
            
            # Clamp score
            score = max(0, min(100, score))
            
            setup = {
                "Symbol": symbol,
                "Score": score,
                "Direction": direction,
                "Entry": round(entry_price, 2),
                "StopLoss": round(stop_loss, 2),
                "Target1": round(target_1, 2),
                "Target2": round(target_2, 2),
                "R:R": round(rr_ratio, 2),
                "ATR%": round(atr_pct, 2),
                "LastPrice": round(close, 2),
                "Change%": round(latest.get("Change%", 0), 2),
                "RSI": round(rsi, 1),
                "RVOL": round(rvol, 2),
                "Volume": int(latest.get("Volume", 0)),
                "Reasons": reasons,
                "Concerns": concerns,
                "Indicators": {
                    "RSI": round(rsi, 1),
                    "ADX": round(adx, 1),
                    "RVOL": round(rvol, 2),
                    "MACD_Hist": round(macd_hist, 6),
                    "SuperTrend": "UP" if st_dir == 1 else "DOWN",
                    "ATR": round(atr, 2),
                    "VWAP": round(vwap, 2) if vwap else None,
                },
                "Time": datetime.now().strftime("%H:%M"),
            }
            results.append(setup)
            
        except Exception as e:
            continue
    
    # Sort by score descending, take top N
    results.sort(key=lambda x: x["Score"], reverse=True)
    return results[:top_n]


def get_daily_picks(nifty_stocks, top_n=4):
    """
    Convenience wrapper: runs intraday scan on Nifty stocks.
    
    Parameters
    ----------
    nifty_stocks : list
        List of NSE stock symbols
    top_n : int
        Number of picks to return
    
    Returns
    -------
    list[dict]
        Top intraday picks for the day
    """
    return scan_intraday_setups(
        symbols=nifty_stocks,
        min_price=50,
        max_price=10000,
        top_n=top_n,
        interval="15m",
        period="5d"
    )


def format_pick_card(setup, index=1):
    """
    Format a single intraday pick as a readable string (for display).
    """
    emoji = "🟢" if setup["Direction"] == "BUY" else ("🟡" if setup["Direction"] == "NEUTRAL" else "🔴")
    reasons_str = " | ".join(setup["Reasons"][:3])
    concerns_str = " | ".join(setup["Concerns"][:2]) if setup["Concerns"] else "None"
    
    return f"""
{emoji} **#{index} — {setup['Symbol']}** (Score: {setup['Score']}/100)
- Entry: **₹{setup['Entry']:,.2f}** | Stop: ₹{setup['StopLoss']:,.2f}
- Target 1: ₹{setup['Target1']:,.2f} (1.5R) | Target 2: ₹{setup['Target2']:,.2f} (3R)
- R:R: {setup['R:R']} | ATR%: {setup['ATR%']}%
- RSI: {setup['RSI']} | RVOL: {setup['RVOL']}x | MACD: {setup['Indicators']['MACD_Hist']:.6f}
- ✅ {reasons_str}
- ⚠️ {concerns_str}
"""
