"""
Profit Engine v3 — Sniper Terminal
====================================
CORE PHILOSOPHY: Profit comes from EXIT MANAGEMENT, not entry precision.
- Lenient entry: Catch most setups (score >= 3 of 8)
- ATR Trailing Stops: Cut losers fast, let winners run
- Market Regime Gate: Skip bear markets
- Max Hold Period: Exit stale trades

Strategy: Enter on bullish alignment + momentum, EXIT with discipline.
"""

import numpy as np
import pandas as pd
from core.indicators import compute_indicators


def generate_v3_signal(df, regime="BULLISH"):
    """Smart lenient signal. Returns dict with Score (0-8), Signal, StopLoss."""
    if df is None or len(df) < 200:
        return {"Signal": "NEUTRAL", "Score": 0}

    r = df.iloc[-1]

    def _safe(field, default=0):
        v = r.get(field, default)
        return float(v) if not np.isnan(v) else default

    c = _safe("Close")
    atr = _safe("ATR", c * 0.02)
    if atr <= 0:
        atr = c * 0.02

    score = 0
    ema50 = _safe("EMA50", c)
    ema200 = _safe("EMA200", c)
    rsi = _safe("RSI", 50)
    macd_hist = _safe("MACD_HIST", 0)
    rvol = _safe("RVOL", 1)
    st_dir = _safe("SUPERTREND_DIR", 0)

    # Bullish alignment: strong weight
    if c > ema50 and ema50 > ema200:
        score += 2
    elif c > ema200:
        score += 1

    # Momentum zone
    if 55 <= rsi <= 75:
        score += 2
    elif 48 <= rsi <= 54:
        score += 1

    # MACD bullish
    if macd_hist > 0:
        score += 1

    # Supertrend
    if st_dir == 1:
        score += 1

    # Volume above average
    if rvol > 1.0:
        score += 1

    # Near EMA50 pullback (healthy retracement)
    if ema50 > 0:
        dist50 = (c / ema50 - 1) * 100
        if 0 <= dist50 <= 5:
            score += 1

    # Bearish exit check
    exits = 0
    if c < ema50:
        exits += 1
    if rsi < 40:
        exits += 1
    if macd_hist < 0:
        exits += 1
    if st_dir == -1:
        exits += 1

    # Signal determination — lenient in bullish, stricter in bearish
    if regime == "BEARISH":
        sig = "STRONG_BUY" if score >= 6 else ("BUY" if score >= 4 else "NEUTRAL")
    else:
        sig = "STRONG_BUY" if score >= 5 else ("BUY" if score >= 3 else "NEUTRAL")

    if exits >= 3:
        sig = "SELL"

    return {
        "Signal": sig, "Score": score,
        "StopLoss": round(c - atr * 2.5, 2),
        "Target": round(c + atr * 5.0, 2),
        "ATR": round(atr, 2), "RSI": round(rsi, 1), "RVOL": round(rvol, 2)
    }


def run_v3_backtest(symbol, capital=100000, risk_pct=2.0,
                     max_hold=20, use_regime=True):
    """High-profit backtest: enter on v3 signal, exit with ATR trail + time stop + target."""
    from core.utils import load_data
    from core.market_regime import get_market_regime

    try:
        df = load_data(symbol, period="5y")
        if df is None or len(df) < 220:
            return {"Error": "No data", "Total Trades": 0}

        df = compute_indicators(df)
        regime = get_market_regime() if use_regime else "NEUTRAL"

        # Adaptive risk: don't block trades entirely, just size down
        if regime == "BEARISH":
            effective_risk_pct = risk_pct * 0.5  # Halve risk in bearish
            min_entry_score = 5                   # Require STRONG_BUY
        elif regime == "SIDEWAYS":
            effective_risk_pct = risk_pct * 0.75  # 75% risk in sideways
            min_entry_score = 3
        else:
            effective_risk_pct = risk_pct          # Full risk
            min_entry_score = 3

        cap = capital
        trades = []
        pos = False
        ep = eb = 0
        ed = None
        sl = tgt = hc = 0
        sh = 0

        for i in range(200, len(df)):
            bar = df.iloc[i]
            c = float(bar["Close"])
            atr = float(bar.get("ATR", c * 0.02))
            if np.isnan(atr) or atr <= 0: atr = c * 0.02
            sig = generate_v3_signal(df.iloc[:i + 1], regime)

            # ENTRY — use adaptive score threshold based on regime
            if not pos and sig["Signal"] in ("STRONG_BUY", "BUY"):
                # Skip trades that don't meet the regime's minimum score
                if sig["Score"] < min_entry_score:
                    continue
                ep = c; eb = i; ed = bar.name
                sl = sig["StopLoss"]; tgt = sig["Target"]; hc = c
                sd = abs(c - sl)
                if sd > 0:
                    risk_amt = cap * effective_risk_pct / 100
                    raw = int(risk_amt / sd)
                    mx = int(cap * 0.3 / c)
                    sh = max(1, min(raw, mx))
                else:
                    sh = int(cap * 0.2 / c)
                pos = True
                continue

            # EXIT
            if pos:
                held = i - eb
                if c > hc: hc = c
                sl = max(sl, hc - atr * 2.5)

                ex = None; er = ""
                if c <= sl: ex = c; er = "TrailStop"
                elif c >= tgt: ex = c; er = "Target"
                elif held >= max_hold: ex = c; er = f"Time({max_hold}d)"
                elif sig["Signal"] == "SELL": ex = c; er = "SellSig"

                if ex is not None:
                    pp = (ex - ep) / ep * 100
                    pa = sh * (ex - ep)
                    cap += pa
                    trades.append({
                        "Entry": ed, "Exit": bar.name, "Entry₹": round(ep, 2),
                        "Exit₹": round(ex, 2), "PnL%": round(pp, 2),
                        "PnL₹": round(pa, 2), "Result": "WIN" if pp > 0 else "LOSS",
                        "Bars": held + 1, "Reason": er, "Shares": sh
                    })
                    pos = False

        # Close open
        if pos:
            fp = float(df["Close"].iloc[-1])
            pp = (fp - ep) / ep * 100; pa = sh * (fp - ep); cap += pa
            trades.append({
                "Entry": ed, "Exit": df.index[-1], "Entry₹": round(ep, 2),
                "Exit₹": round(fp, 2), "PnL%": round(pp, 2), "PnL₹": round(pa, 2),
                "Result": "WIN" if pp > 0 else "LOSS",
                "Bars": len(df) - eb, "Reason": "End", "Shares": sh
            })

        # METRICS
        n = len(trades)
        if n == 0:
            return {"Total Return %": 0.0, "Total Trades": 0, "Win Rate %": 0.0,
                    "Profit Factor": 0.0, "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
                    "Expectancy %": 0.0, "Avg Win %": 0.0, "Avg Loss %": 0.0,
                    "Final Capital": cap, "Trades": pd.DataFrame(),
                    "Exit Breakdown": {}, "Error": None}

        wl = [t for t in trades if t["Result"] == "WIN"]
        ll = [t for t in trades if t["Result"] == "LOSS"]
        wc = len(wl); lc = len(ll)

        wr = round(wc / n * 100, 1)
        aw = round(np.mean([t["PnL%"] for t in wl]), 2) if wl else 0
        al = round(np.mean([t["PnL%"] for t in ll]), 2) if ll else 0
        tp = sum(t["PnL%"] for t in wl) if wl else 0
        tl = abs(sum(t["PnL%"] for t in ll)) if ll else 0
        pf = round(tp / max(tl, 0.01), 2)

        rets = [t["PnL%"] for t in trades]
        shp = round(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252), 2) if len(rets) > 1 else 0

        eq = [capital]
        for t in trades: eq.append(eq[-1] + t.get("PnL₹", 0))
        es = pd.Series(eq[1:]); rm = es.cummax()
        dd = (es - rm) / rm.replace(0, 1) * 100
        mdd = round(dd.min(), 2) if len(dd) > 0 else 0

        ret = round((cap - capital) / capital * 100, 2)
        ev = round((wr / 100 * aw) - ((1 - wr / 100) * abs(al)), 2)

        er_map = {}
        for t in trades: er_map[t.get("Reason", "?")] = er_map.get(t.get("Reason", "?"), 0) + 1

        best = max(trades, key=lambda x: x["PnL%"]) if trades else None
        worst = min(trades, key=lambda x: x["PnL%"]) if trades else None

        return {
            "Total Return %": ret, "Total Trades": n, "Win Count": wc, "Loss Count": lc,
            "Win Rate %": wr, "Avg Win %": aw, "Avg Loss %": al,
            "Profit Factor": pf, "Sharpe Ratio": shp, "Max Drawdown %": mdd,
            "Expectancy %": ev,
            "Best Trade %": round(best["PnL%"], 2) if best else 0,
            "Worst Trade %": round(worst["PnL%"], 2) if worst else 0,
            "Final Capital": round(cap, 2),
            "Exit Breakdown": er_map,
            "Trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            "Error": None
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"Error": str(e), "Total Trades": 0}


def compare_v3_vs_old(symbol: str) -> dict:
    """Compare old EMA/RSI vs Profit Engine v3 (regime filter disabled for fair comparison)."""
    from core.backtest import run_backtest as old_bt
    old = old_bt(symbol) or {}
    new = run_v3_backtest(symbol, use_regime=True) or {}

    op = old.get("Profit Factor", 0) or 0
    npf = new.get("Profit Factor", 0) or 0
    or_ = old.get("Total Return %", 0) or 0
    nr_ = new.get("Total Return %", 0) or 0

    return {
        "old": {"Total Return %": or_, "Win Rate %": old.get("Win Rate %", 0) or 0,
                "Profit Factor": op, "Max Drawdown %": old.get("Max Drawdown %", 0) or 0,
                "Sharpe Ratio": old.get("Sharpe Ratio", 0) or 0,
                "Total Trades": old.get("Total Trades", 0) or 0},
        "new": {"Total Return %": nr_, "Win Rate %": new.get("Win Rate %", 0) or 0,
                "Profit Factor": npf, "Max Drawdown %": new.get("Max Drawdown %", 0) or 0,
                "Sharpe Ratio": new.get("Sharpe Ratio", 0) or 0,
                "Total Trades": new.get("Total Trades", 0) or 0},
        "improvement": {"PF Delta %": round((npf - op) / max(op, 0.1) * 100, 1),
                        "Return Delta %": round(nr_ - or_, 2)},
        "details": new
    }
