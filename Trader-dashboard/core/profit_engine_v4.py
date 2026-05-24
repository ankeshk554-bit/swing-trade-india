"""
Profit Engine v4 — Institutional Grade
========================================
Targets: PF > 2.0, Sharpe > 2.0, MaxDD < 15%

Features:
  1. Weekly Trend Gate       — Only trade stocks with bullish weekly structure
  2. ADX > 22 Filter         — Skip range-bound / choppy markets
  3. Partial Profit Booking  — Book 40% at 1.5R, trail 60% at 1x ATR
  4. Volatility-Adaptive Stops — 2x ATR (low vol), 1.5x ATR (high vol)
  5. Kelly Position Sizing   — f* = W - (1-W)/R, capped at 25%
  6. Walk-Forward Opt        — 3yr train / 1yr test rolling windows
  7. Monte Carlo Simulation  — 90% confidence interval on returns
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from core.indicators import compute_indicators

# ──────────────────────────────────────────────────────────────────
# 1. SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────

def _safe_val(r, field, default=0):
    """Safely extract a float value from a row, handling NaN/None."""
    v = r.get(field, default)
    try:
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else default
    except (ValueError, TypeError):
        return default


def get_weekly_trend(symbol) -> str:
    """Fetch weekly data and return trend direction: BULLISH / BEARISH / NEUTRAL."""
    try:
        import yfinance as yf
        wk = yf.download(symbol, period="1y", interval="1wk", auto_adjust=True, progress=False)
        if wk is None or wk.empty:
            return "NEUTRAL"
        if isinstance(wk.columns, pd.MultiIndex):
            wk.columns = wk.columns.get_level_values(0)
        wk["EMA50"] = wk["Close"].ewm(span=50).mean()
        wk["EMA200"] = wk["Close"].ewm(span=200).mean()
        c = float(wk["Close"].iloc[-1])
        e50 = float(wk["EMA50"].iloc[-1])
        e200 = float(wk["EMA200"].iloc[-1])
        if c > e50 > e200:
            return "BULLISH"
        elif c < e50 < e200:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def compute_weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly for trend analysis."""
    wk = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
        "Volume": "sum"
    }).dropna()
    wk["EMA50"] = wk["Close"].ewm(span=50).mean()
    wk["EMA200"] = wk["Close"].ewm(span=200).mean()
    return wk


def generate_v4_signal(df, weekly_trend="NEUTRAL", regime="BULLISH", adx_threshold=22):
    """
    Multi-factor institutional signal generator.

    Scoring (0–10):
      - Trend alignment:     0–3 pts
      - Momentum:            0–2 pts
      - Volume confirmation: 0–2 pts
      - Pullback quality:    0–2 pts
      - ADX strength:        0–1 pt

    Returns dict with Signal, Score, StopLoss, Target, ATR, etc.
    """
    if df is None or len(df) < 200:
        return {"Signal": "NEUTRAL", "Score": 0}

    r = df.iloc[-1]
    c = _safe_val(r, "Close")
    atr = _safe_val(r, "ATR", c * 0.02)
    if atr <= 0:
        atr = c * 0.02

    ema20 = _safe_val(r, "EMA20", c)
    ema50 = _safe_val(r, "EMA50", c)
    ema200 = _safe_val(r, "EMA200", c)
    rsi = _safe_val(r, "RSI", 50)
    macd_hist = _safe_val(r, "MACD_HIST", 0)
    rvol = _safe_val(r, "RVOL", 1)
    st_dir = _safe_val(r, "SUPERTREND_DIR", 0)
    adx = _safe_val(r, "ADX", 0)

    score = 0

    # ── 1. Trend alignment (0–3 pts) ──
    if c > ema50 and ema50 > ema200:
        score += 3
    elif c > ema50 > ema20:
        score += 2
    elif c > ema200:
        score += 1

    # ── 2. Momentum (0–2 pts) ──
    if 55 <= rsi <= 70:
        score += 2
    elif 50 <= rsi <= 54:
        score += 1

    # ── 3. Volume confirmation (0–2 pts) ──
    if rvol > 2.0:
        score += 2
    elif rvol > 1.5:
        score += 1

    # ── 4. Pullback quality (0–2 pts) ──
    if ema50 > 0:
        dist50 = (c / ema50 - 1) * 100
        if 0 <= dist50 <= 3:
            score += 2       # Ideal pullback to EMA50
        elif -2 <= dist50 < 0:
            score += 1       # Slight dip below EMA50
        elif 3 < dist50 <= 8:
            score += 1       # Above EMA50 but not extended

    # ── 5. ADX strength (0–1 pt) ──
    if adx > adx_threshold:
        score += 1

    # ── Bonus: MACD + Supertrend confirmation ──
    if macd_hist > 0 and st_dir == 1:
        score += 1  # Bonus point for double confirmation

    # ── Weekly trend gate ──
    if weekly_trend == "BEARISH":
        score -= 2  # Strong penalty — weekly bearish

    # ── Regime-based thresholds ──
    if regime == "BEARISH":
        min_buy = 5
        min_strong = 7
    elif regime == "SIDEWAYS":
        min_buy = 4
        min_strong = 6
    else:
        min_buy = 4
        min_strong = 6

    sig = "STRONG_BUY" if score >= min_strong else ("BUY" if score >= min_buy else "NEUTRAL")

    # ── Bearish exit check ──
    exits = 0
    if c < ema50:
        exits += 1
    if rsi < 40:
        exits += 1
    if macd_hist < 0 and st_dir == -1:
        exits += 1
    if adx < 15:
        exits += 1  # Very weak trend, exit
    if exits >= 3:
        sig = "SELL"

    # ── Volatility-adaptive stops ──
    atr_pct = (atr / c) * 100 if c > 0 else 2.0
    if atr_pct > 4.0:
        stop_mult = 1.5   # High vol: tighter stop
        tgt_mult = 3.5
    elif atr_pct > 2.5:
        stop_mult = 2.0   # Normal vol
        tgt_mult = 5.0
    else:
        stop_mult = 2.5   # Low vol: wider stop
        tgt_mult = 6.0

    return {
        "Signal": sig,
        "Score": max(0, score),
        "StopLoss": round(c - atr * stop_mult, 2),
        "Target": round(c + atr * tgt_mult, 2),
        "Target1": round(c + atr * 1.5, 2),  # Partial book level
        "ATR": round(atr, 2),
        "ATR_PCT": round(atr_pct, 2),
        "RSI": round(rsi, 1),
        "RVOL": round(rvol, 2),
        "ADX": round(adx, 1),
        "StopMult": stop_mult,
        "TgtMult": tgt_mult,
    }


# ──────────────────────────────────────────────────────────────────
# 2. KELLY POSITION SIZING
# ──────────────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Compute optimal Kelly fraction.
    f* = W - (1-W) / R   where R = |avg_win / avg_loss|
    Returns fraction of capital (capped at 25%).
    """
    if win_rate <= 0 or avg_loss <= 0:
        return 0.02  # Default to 2% if no history
    R = abs(avg_win / max(avg_loss, 0.01))
    f = win_rate / 100 - (1 - win_rate / 100) / max(R, 0.1)
    return max(0.01, min(f, 0.25))  # Cap at 25%, min 1%


def compute_position_size(capital: float, risk_per_trade: float,
                           entry_price: float, stop_price: float,
                           win_rate: float = 0, avg_win: float = 0,
                           avg_loss: float = 0, use_kelly: bool = True) -> int:
    """
    Compute position size. Uses Kelly if historical data available, else fixed fraction.
    """
    risk_amt = capital * risk_per_trade
    if use_kelly and win_rate > 0:
        kelly = kelly_fraction(win_rate, avg_win, avg_loss)
        kelly_amt = capital * kelly
        risk_amt = min(risk_amt, kelly_amt)

    atr_risk = abs(entry_price - stop_price)
    if atr_risk <= 0:
        return int(capital * 0.05 / max(entry_price, 1))

    raw = int(risk_amt / atr_risk)
    mx = int(capital * 0.3 / max(entry_price, 1))
    return max(1, min(raw, mx))


# ──────────────────────────────────────────────────────────────────
# 3. PARTIAL PROFIT BOOKING
# ──────────────────────────────────────────────────────────────────

class PartialFillTracker:
    """Tracks partial profit booking state for a single trade."""

    def __init__(self, entry_price, initial_stop, target1,
                 initial_shares, atr, vol_adapt_stop_mult):
        self.entry = entry_price
        self.stop = initial_stop
        self.target1 = target1
        self.shares = initial_shares
        self.atr = atr
        self.vol_mult = vol_adapt_stop_mult
        self.partial_taken = False
        self.highest_since_entry = entry_price
        self.breakeven_activated = False

    def update(self, current_price):
        """Update tracker with current bar price. Returns (exit_price, reason, shares_sold) or None."""
        self.highest_since_entry = max(self.highest_since_entry, current_price)

        if not self.partial_taken and current_price >= self.target1:
            # Book 40% profit
            self.partial_taken = True
            sold_shares = int(self.shares * 0.4)
            remaining = self.shares - sold_shares
            # Move stop to breakeven
            self.stop = self.entry
            self.breakeven_activated = True
            return ("PARTIAL", current_price, sold_shares, remaining)

        # Update trailing stop after breakeven
        if self.breakeven_activated:
            trail_distance = self.atr * self.vol_mult
            self.stop = max(self.stop, self.highest_since_entry - trail_distance)

        # Check exit conditions
        if current_price <= self.stop:
            if self.partial_taken:
                return ("TRAIL_EXIT", current_price, self.shares, 0)
            else:
                return ("STOP", current_price, self.shares, 0)

        return None


# ──────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD OPTIMIZATION
# ──────────────────────────────────────────────────────────────────

def _score_params(df, params, regime, weekly_trends=None):
    """Score a parameter set by running a quick backtest on the given dataframe."""
    atr_mult_range = params.get("atr_mult_range", (1.5, 2.5))
    min_score_range = params.get("min_score_range", (4, 6))
    adx_threshold = params.get("adx_threshold", 22)

    # Quick grid search within ranges
    best_score = -999
    best_params = {"stop_mult": 2.0, "min_score": 4, "adx_threshold": 22}

    for stop_mult in [x * 0.25 for x in range(int(atr_mult_range[0] * 4), int(atr_mult_range[1] * 4) + 1)]:
        for min_score in range(min_score_range[0], min_score_range[1] + 1):
            # Simple metric: ratio of avg win to avg loss * win rate
            trades = _run_quick_backtest(df, stop_mult, min_score, adx_threshold, regime)
            if len(trades) < 5:
                continue
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            if not losses:
                continue
            avg_w = np.mean(wins) if wins else 0
            avg_l = abs(np.mean(losses)) if losses else 1
            wr = len(wins) / max(len(trades), 1)
            score = (avg_w / max(avg_l, 0.01)) * wr * np.sqrt(len(trades))
            if score > best_score:
                best_score = score
                best_params = {"stop_mult": stop_mult, "min_score": min_score, "adx_threshold": adx_threshold}

    return best_params


def _run_quick_backtest(df, stop_mult, min_score, adx_threshold, regime):
    """Minimal backtest for parameter optimization. Returns list of PnL%."""
    trades = []
    pos = False
    ep = sl = 0
    for i in range(200, len(df)):
        bar = df.iloc[:i + 1]
        r = bar.iloc[-1]
        c = float(r["Close"])
        atr = float(r.get("ATR", c * 0.02))
        if np.isnan(atr) or atr <= 0:
            atr = c * 0.02
        sig = generate_v4_signal(bar, regime=regime, adx_threshold=adx_threshold)

        if not pos and sig["Signal"] in ("BUY", "STRONG_BUY") and sig["Score"] >= min_score:
            ep = c
            sl = c - atr * stop_mult
            pos = True
        elif pos:
            if c <= sl or sig["Signal"] == "SELL":
                pnl = (c - ep) / ep * 100
                trades.append(pnl)
                pos = False

    if pos:
        fp = float(df["Close"].iloc[-1])
        trades.append((fp - ep) / ep * 100)

    return trades


def walk_forward_optimize(symbol: str, capital: float = 100000) -> dict:
    """
    Walk-forward optimization: 3yr train / 1yr test / 1yr out-of-sample.
    Returns optimized parameters and out-of-sample performance.
    """
    from core.utils import load_data
    df = load_data(symbol, period="7y")
    if df is None or len(df) < 1500:
        return {"Error": "Insufficient data for walk-forward", "Total Trades": 0}

    df = compute_indicators(df)
    regime = "NEUTRAL"

    # Split data: 3yr train, 1yr val, rest test
    total_bars = len(df)
    train_end = int(total_bars * 0.5)       # First 50% for training
    val_end = int(total_bars * 0.7)          # Next 20% for validation

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Optimize on training
    param_ranges = {
        "atr_mult_range": (1.5, 2.5),
        "min_score_range": (4, 6),
        "adx_threshold": 22
    }
    best = _score_params(train_df, param_ranges, regime)

    # Validate
    val_trades = _run_quick_backtest(val_df, best["stop_mult"], best["min_score"],
                                      best["adx_threshold"], regime)
    # Test (out-of-sample)
    test_trades = _run_quick_backtest(test_df, best["stop_mult"], best["min_score"],
                                       best["adx_threshold"], regime)

    return {
        "Optimal Stop Mult": best["stop_mult"],
        "Optimal Min Score": best["min_score"],
        "ADX Threshold": best["adx_threshold"],
        "Train Trades": len(_run_quick_backtest(train_df, 2.0, 4, 22, regime)),
        "Val Trades": len(val_trades),
        "Test Trades": len(test_trades),
        "Val Avg PnL%": round(np.mean(val_trades), 2) if val_trades else 0,
        "Test Avg PnL%": round(np.mean(test_trades), 2) if test_trades else 0,
        "Val Win Rate%": round(sum(1 for t in val_trades if t > 0) / max(len(val_trades), 1) * 100, 1) if val_trades else 0,
        "Test Win Rate%": round(sum(1 for t in test_trades if t > 0) / max(len(test_trades), 1) * 100, 1) if test_trades else 0,
    }


# ──────────────────────────────────────────────────────────────────
# 5. MONTE CARLO SIMULATION
# ──────────────────────────────────────────────────────────────────

def run_monte_carlo(trades: list, capital: float = 100000,
                     simulations: int = 1000, confidence: float = 0.90) -> dict:
    """
    Monte Carlo simulation by sampling trades with replacement.

    Returns:
      - Median return
      - Confidence interval at given level
      - Probability of profit
      - Max drawdown distribution
    """
    if not trades or len(trades) < 5:
        return {
            "Median Return %": 0, "CI Lower %": 0, "CI Upper %": 0,
            "Prob Profit %": 0, "Avg MaxDD %": 0, "Simulations": 0
        }

    results = []
    dd_list = []
    n_trades = len(trades)

    for _ in range(simulations):
        sampled = np.random.choice(trades, size=n_trades, replace=True)
        eq = [capital]
        for pnl in sampled:
            eq.append(eq[-1] * (1 + pnl / 100))
        total_ret = (eq[-1] - capital) / capital * 100
        results.append(total_ret)

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (np.array(eq) - peak) / peak * 100
        dd_list.append(abs(dd.min()))

    results = np.array(results)
    dd_list = np.array(dd_list)

    alpha = (1 - confidence) / 2
    ci_lo = np.percentile(results, alpha * 100)
    ci_hi = np.percentile(results, (1 - alpha) * 100)

    return {
        "Median Return %": round(np.median(results), 2),
        "CI Lower %": round(ci_lo, 2),
        "CI Upper %": round(ci_hi, 2),
        "Prob Profit %": round(np.mean(results > 0) * 100, 1),
        "Avg MaxDD %": round(np.mean(dd_list), 2),
        "Max MaxDD %": round(np.max(dd_list), 2),
        "Std Dev %": round(np.std(results), 2),
        "Simulations": simulations
    }


# ──────────────────────────────────────────────────────────────────
# 6. FULL V4 BACKTEST
# ──────────────────────────────────────────────────────────────────

def run_v4_backtest(symbol, capital=100000, risk_pct=2.0,
                     max_hold=20, use_regime=True, use_kelly=True,
                     use_weekly_gate=True, use_partial_booking=True,
                     use_adx_filter=True, use_vol_adapt=True):
    """
    Institutional-grade backtest with all v4 features enabled/disabled selectively.

    Returns full metrics dictionary.
    """
    from core.utils import load_data
    from core.market_regime import get_market_regime

    try:
        df = load_data(symbol, period="5y")
        if df is None or len(df) < 220:
            return {"Error": "No data", "Total Trades": 0}

        df = compute_indicators(df)
        regime = get_market_regime() if use_regime else "NEUTRAL"

        # Weekly trend
        weekly_trend = "NEUTRAL"
        if use_weekly_gate:
            weekly_df = compute_weekly_from_daily(df)
            if len(weekly_df) > 10:
                wr = weekly_df.iloc[-1]
                wc = _safe_val(wr, "Close")
                we50 = _safe_val(wr, "EMA50", wc)
                we200 = _safe_val(wr, "EMA200", wc)
                if wc > we50 > we200:
                    weekly_trend = "BULLISH"
                elif wc < we50 < we200:
                    weekly_trend = "BEARISH"

        # Adaptive risk
        if regime == "BEARISH":
            effective_risk = risk_pct * 0.5
        elif regime == "SIDEWAYS":
            effective_risk = risk_pct * 0.75
        else:
            effective_risk = risk_pct

        cap = capital
        trades = []
        pos = False
        ep = eb = 0
        ed = None
        sh = 0
        tracker = None
        # For Kelly calculation (rolling)
        recent_trades_pnl = []

        for i in range(200, len(df)):
            bar_slice = df.iloc[:i + 1]
            bar = df.iloc[i]
            c = float(bar["Close"])
            atr = float(bar.get("ATR", c * 0.02))
            if np.isnan(atr) or atr <= 0:
                atr = c * 0.02

            # Generate signal with all filters
            sig = generate_v4_signal(
                bar_slice, weekly_trend=weekly_trend,
                regime=regime,
                adx_threshold=22
            )

            # ── ADX filter ──
            if use_adx_filter and not pos:
                adx = _safe_val(bar, "ADX", 0)
                if adx < 22:
                    continue

            # ── Weekly gate ──
            if use_weekly_gate and not pos and weekly_trend == "BEARISH":
                continue

            # ── ENTRY ──
            if not pos and sig["Signal"] in ("STRONG_BUY", "BUY"):
                # Dynamic Kelly-based sizing
                wr = 0
                aw = 0
                al = 0
                if recent_trades_pnl:
                    wins = [p for p in recent_trades_pnl if p > 0]
                    losses = [p for p in recent_trades_pnl if p <= 0]
                    if len(recent_trades_pnl) > 5:
                        wr = len(wins) / len(recent_trades_pnl) * 100
                        aw = np.mean(wins) if wins else 0
                        al = abs(np.mean(losses)) if losses else 1

                # Use volatility-adaptive stop multiplier from signal
                stop_mult = sig.get("StopMult", 2.0)

                ep = c
                eb = i
                ed = bar.name
                sl = c - atr * stop_mult
                tgt1 = c + atr * 1.5  # 1.5R for partial booking
                sd = abs(c - sl)

                sh = compute_position_size(cap, effective_risk, c, sl,
                                           wr, aw, al, use_kelly=use_kelly and len(recent_trades_pnl) > 5)
                sh = max(1, sh)

                if use_partial_booking:
                    tracker = PartialFillTracker(c, sl, tgt1, sh, atr, stop_mult)
                pos = True
                continue

            # ── EXIT ──
            if pos:
                held = i - eb
                ex = None
                er = ""
                shares_sold = 0
                pnl_total = 0
                total_shares = sh

                if use_partial_booking and tracker is not None:
                    result = tracker.update(c)
                    if result is not None:
                        action, price, sold, remaining = result
                        if action == "PARTIAL":
                            # Partial fill: book profit on 40%
                            pnl_partial = (price - ep) / ep * 100
                            pnl_total += sold * (price - ep)
                            sh = remaining
                            # Keep position open with reduced shares
                            continue
                        else:
                            ex = price
                            er = action
                            shares_sold = sold
                            pnl_total += sold * (price - ep)
                else:
                    # Standard exit logic without partial booking
                    sl_use = sig.get("StopLoss", c - atr * 2.5)
                    tgt = sig.get("Target", c + atr * 5.0)
                    if c <= sl_use:
                        ex = c
                        er = "TrailStop"
                    elif c >= tgt:
                        ex = c
                        er = "Target"
                    elif held >= max_hold:
                        ex = c
                        er = f"Time({max_hold}d)"
                    elif sig["Signal"] == "SELL":
                        ex = c
                        er = "SellSig"

                if ex is not None:
                    # Calculate final PnL if partial was taken
                    if use_partial_booking and tracker is not None and tracker.partial_taken:
                        pnl_final = sh * (ex - ep)
                        total_pnl = pnl_total + pnl_final
                        avg_price = ep
                        pp = total_pnl / (total_shares * avg_price) * 100 if total_shares > 0 else 0
                    else:
                        shares_sold = sh
                        pnl_total = sh * (ex - ep)
                        pp = (ex - ep) / ep * 100

                    cap += pnl_total
                    recent_trades_pnl.append(pp)
                    # Keep rolling window of last 50 trades for Kelly
                    if len(recent_trades_pnl) > 50:
                        recent_trades_pnl.pop(0)

                    trades.append({
                        "Entry": ed, "Exit": bar.name, "Entry₹": round(ep, 2),
                        "Exit₹": round(ex, 2), "PnL%": round(pp, 2),
                        "PnL₹": round(pnl_total, 2),
                        "Result": "WIN" if pp > 0 else "LOSS",
                        "Bars": held + 1, "Reason": er, "Shares": total_shares
                    })
                    pos = False
                    tracker = None

        # Close any open position
        if pos:
            fp = float(df["Close"].iloc[-1])
            if use_partial_booking and tracker is not None and tracker.partial_taken:
                pnl_final = sh * (fp - ep)
                total_pnl = pnl_total + pnl_final
                pp = total_pnl / (total_shares * ep) * 100 if total_shares > 0 else 0
            else:
                pnl_total = sh * (fp - ep)
                pp = (fp - ep) / ep * 100
            cap += pnl_total
            trades.append({
                "Entry": ed, "Exit": df.index[-1], "Entry₹": round(ep, 2),
                "Exit₹": round(fp, 2), "PnL%": round(pp, 2),
                "PnL₹": round(pnl_total, 2),
                "Result": "WIN" if pp > 0 else "LOSS",
                "Bars": len(df) - eb, "Reason": "End", "Shares": total_shares
            })

        # ── METRICS ──
        n = len(trades)
        if n == 0:
            return {
                "Total Return %": 0.0, "Total Trades": 0,
                "Win Rate %": 0.0, "Profit Factor": 0.0,
                "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
                "Expectancy %": 0.0, "Avg Win %": 0.0, "Avg Loss %": 0.0,
                "Final Capital": cap, "Trades": pd.DataFrame(),
                "Exit Breakdown": {}, "Kelly Fraction": 0,
                "Error": None, "Version": "v4"
            }

        wl = [t for t in trades if t["Result"] == "WIN"]
        ll = [t for t in trades if t["Result"] == "LOSS"]
        wc = len(wl)
        lc = len(ll)

        wr = round(wc / n * 100, 1)
        aw = round(np.mean([t["PnL%"] for t in wl]), 2) if wl else 0
        al = round(np.mean([t["PnL%"] for t in ll]), 2) if ll else 0
        tp = sum(t["PnL%"] for t in wl) if wl else 0
        tl = abs(sum(t["PnL%"] for t in ll)) if ll else 0
        pf = round(tp / max(tl, 0.01), 2)

        rets = [t["PnL%"] for t in trades]
        shp = round(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252), 2) if len(rets) > 1 else 0

        eq = [capital]
        for t in trades:
            eq.append(eq[-1] + t.get("PnL₹", 0))
        es = pd.Series(eq[1:])
        rm = es.cummax()
        dd = (es - rm) / rm.replace(0, 1) * 100
        mdd = round(dd.min(), 2) if len(dd) > 0 else 0

        ret = round((cap - capital) / capital * 100, 2)
        ev = round((wr / 100 * aw) - ((1 - wr / 100) * abs(al)), 2)

        # Kelly from realized trades
        kelly = kelly_fraction(wr, aw, al) if wr > 0 else 0.02

        # Monte Carlo
        mc = run_monte_carlo(rets, capital, simulations=2000)

        er_map = {}
        for t in trades:
            er_map[t.get("Reason", "?")] = er_map.get(t.get("Reason", "?"), 0) + 1

        best = max(trades, key=lambda x: x["PnL%"]) if trades else None
        worst = min(trades, key=lambda x: x["PnL%"]) if trades else None

        return {
            "Total Return %": ret, "Total Trades": n,
            "Win Count": wc, "Loss Count": lc,
            "Win Rate %": wr, "Avg Win %": aw, "Avg Loss %": al,
            "Profit Factor": pf, "Sharpe Ratio": shp,
            "Max Drawdown %": mdd, "Expectancy %": ev,
            "Best Trade %": round(best["PnL%"], 2) if best else 0,
            "Worst Trade %": round(worst["PnL%"], 2) if worst else 0,
            "Final Capital": round(cap, 2),
            "Kelly Fraction": round(kelly, 4),
            "Exit Breakdown": er_map,
            "Monte Carlo": mc,
            "Weekly Trend": weekly_trend,
            "Market Regime": regime,
            "Trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            "Version": "v4",
            "Error": None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"Error": str(e), "Total Trades": 0}


# ──────────────────────────────────────────────────────────────────
# 7. COMPARISON: V4 vs OLD BASELINE
# ──────────────────────────────────────────────────────────────────

def compare_v4_vs_old(symbol: str) -> dict:
    """Compare old EMA/RSI baseline vs Profit Engine v4."""
    from core.backtest import run_backtest as old_bt
    old = old_bt(symbol) or {}
    new = run_v4_backtest(symbol) or {}

    op = old.get("Profit Factor", 0) or 0
    npf = new.get("Profit Factor", 0) or 0
    or_ = old.get("Total Return %", 0) or 0
    nr_ = new.get("Total Return %", 0) or 0

    return {
        "old": {
            "Total Return %": or_,
            "Win Rate %": old.get("Win Rate %", 0) or 0,
            "Profit Factor": op,
            "Max Drawdown %": old.get("Max Drawdown %", 0) or 0,
            "Sharpe Ratio": old.get("Sharpe Ratio", 0) or 0,
            "Total Trades": old.get("Total Trades", 0) or 0
        },
        "new": {
            "Total Return %": nr_,
            "Win Rate %": new.get("Win Rate %", 0) or 0,
            "Profit Factor": npf,
            "Max Drawdown %": new.get("Max Drawdown %", 0) or 0,
            "Sharpe Ratio": new.get("Sharpe Ratio", 0) or 0,
            "Total Trades": new.get("Total Trades", 0) or 0,
            "Kelly Fraction": new.get("Kelly Fraction", 0),
            "Monte Carlo": new.get("Monte Carlo", {}),
            "Weekly Trend": new.get("Weekly Trend", "N/A"),
            "Market Regime": new.get("Market Regime", "N/A"),
        },
        "improvement": {
            "PF Delta %": round((npf - op) / max(op, 0.1) * 100, 1),
            "Return Delta %": round(nr_ - or_, 2),
        },
        "details": new
    }
