"""
Profit Engine v5 — Institutional Precision
============================================
Targets: PF > 2.5, Sharpe > 2.5, MaxDD < 10%, Calmar > 1.5, Payoff > 2.0

SECTIONS:
  1. Entry Filters (8)   — Volume, RS, 52w High, EMA Stack, VWAP, Earnings, Sector, Breadth
  2. Exit Rules   (5)   — TimeStop, Chandelier, ParSAR, VolExhaust, GapUp
  3. Risk Mgmt    (7)   — Portfolio Heat, Correlation, CircuitBreaker, DynamicKelly, AntiMartingale, DailyLoss, Pyramid
  4. Analytics    (16)  — Calmar, Sortino, Omega, Payoff, Expectancy, MAR, Recovery, Ulcer, K-Ratio, TailRatio, etc.
  5. Robustness   (5)   — Anchored WFA, Param Stability, HMM Regime, OOS Ratio, Deflated Sharpe
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import timedelta
from typing import Optional, Callable
from core.indicators import compute_indicators

# ─────────────────────────────────────────────────────────────────────
# 0. HELPERS
# ─────────────────────────────────────────────────────────────────────

def _sf(r, field, default=0):
    """Safely extract a float."""
    v = r.get(field, default)
    try:
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else default
    except (ValueError, TypeError):
        return default

def _get_nifty_data(period="1y"):
    """Fetch Nifty 50 benchmark data."""
    import yfinance as yf
    try:
        n = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
        if n is not None and not n.empty:
            if isinstance(n.columns, pd.MultiIndex):
                n.columns = n.columns.get_level_values(0)
            return n
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────
# 1. ENTRY FILTERS
# ─────────────────────────────────────────────────────────────────────

class EntryFilters:
    """Collection of entry filter checks. Each returns (passed: bool, reason: str or None)."""

    @staticmethod
    def volume_confirmation(df, i, min_rvol=1.5):
        """Volume > 1.5x 20-day average. Avoids low-liquidity breakouts."""
        r = df.iloc[i]
        rvol = _sf(r, "RVOL", 1)
        if rvol >= min_rvol:
            return True, None
        return False, f"RVOL {rvol:.2f} < {min_rvol}"

    @staticmethod
    def relative_strength(symbol, lookback=63):
        """3-month RS vs Nifty 50. RS > 1.0 = outperforming."""
        try:
            import yfinance as yf
            end = pd.Timestamp.now()
            start = end - timedelta(days=lookback + 20)
            stock = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            nifty = _get_nifty_data("6mo")
            if stock is None or stock.empty or nifty is None or nifty.empty:
                return False, "No RS data"
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)
            sr = (float(stock["Close"].iloc[-1]) / max(float(stock["Close"].iloc[0]), 0.01)) - 1
            nr = (float(nifty["Close"].iloc[-1]) / max(float(nifty["Close"].iloc[0]), 0.01)) - 1
            rs = (1 + sr) / (1 + nr) if nr > -1 else 1
            if rs >= 1.0:
                return True, None
            return False, f"RS {rs:.3f} < 1.0"
        except Exception:
            return True, None  # Fail open if can't fetch

    @staticmethod
    def week_52_proximity(df, i, max_distance_pct=25):
        """Price within 25% of 52-week high. Momentum factor."""
        lookback = min(252, i)
        if lookback < 50:
            return False, "Insufficient history"
        window = df.iloc[i - lookback:i + 1]
        high_52w = window["High"].max()
        c = float(df.iloc[i]["Close"])
        dist = (high_52w - c) / max(high_52w, 0.01) * 100
        if dist <= max_distance_pct:
            return True, None
        return False, f"{dist:.1f}% below 52w high"

    @staticmethod
    def ema_stack(df, i):
        """Price > EMA20 > EMA50 > EMA200. Confirms trend structure."""
        r = df.iloc[i]
        c = _sf(r, "Close")
        e20 = _sf(r, "EMA20", c)
        e50 = _sf(r, "EMA50", c)
        e200 = _sf(r, "EMA200", c)
        if c > e20 > e50 > e200:
            return True, None
        return False, f"Stack fail: C{c:.0f} E20{e20:.0f} E50{e50:.0f} E200{e200:.0f}"

    @staticmethod
    def vwap_filter(df, i):
        """Price above VWAP. Institutional bias."""
        c = float(df.iloc[i]["Close"])
        # Compute rolling VWAP over the lookback
        lookback = min(20, i)
        seg = df.iloc[i - lookback:i + 1]
        tp = (seg["High"] + seg["Low"] + seg["Close"]) / 3
        vwap = (tp * seg["Volume"]).sum() / max(seg["Volume"].sum(), 1)
        if c >= vwap:
            return True, None
        return False, f"C {c:.0f} < VWAP {vwap:.0f}"

    @staticmethod
    def earnings_blackout(df, i, days_before=5, days_after=2):
        """No entries 5d before / 2d after earnings. Placeholder — needs earnings date source."""
        # Without an earnings calendar API, this is approximated.
        # We check for abnormal volume + price gaps as earnings proxy.
        # This filter is soft-fail by default.
        return True, None  # Disabled — requires earnings calendar data source

    @staticmethod
    def sector_momentum(symbol, sector_map: dict = None, top_n=3):
        """Only trade stocks in top 3 performing sectors (30d)."""
        if sector_map is None:
            return True, None  # Disabled without sector mapping
        sec = sector_map.get(symbol.replace(".NS", ""))
        if sec is None:
            return True, None
        # Sector momentum would require fetching sector ETF/index data
        # For now, pass-through
        return True, None

    @staticmethod
    def market_breadth_gate(min_ad_ratio=1.2):
        """Nifty 500 advance/decline ratio > 1.2. Confirms broad market health."""
        # Approximate using Nifty 50 A/D
        try:
            import yfinance as yf
            nifty = yf.download("^NSEI", period="5d", auto_adjust=True, progress=False)
            if nifty is None or nifty.empty:
                return True, None  # Fail open
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            # Use Nifty return as breadth proxy: positive = more advancers
            ret = (float(nifty["Close"].iloc[-1]) / max(float(nifty["Close"].iloc[-2]), 0.01) - 1) * 100
            if ret > -0.3:  # Not a strongly negative day
                return True, None
            return False, f"Nifty {ret:.2f}% — breadth weak"
        except Exception:
            return True, None  # Fail open


# ─────────────────────────────────────────────────────────────────────
# 2. EXIT RULES
# ─────────────────────────────────────────────────────────────────────

class ExitRules:
    """Collection of exit rule triggers. Each returns (exit: bool, reason: str or None)."""

    @staticmethod
    def time_stop(entry_bar, current_bar, max_hold, entry_price, current_price, atr):
        """Exit if held > max_hold without hitting 1R profit."""
        held = current_bar - entry_bar
        if held >= max_hold:
            pnl_r = (current_price - entry_price) / max(atr, 0.01)
            if pnl_r < 1.0:
                return True, f"Time({held}d, R:{pnl_r:.1f})"
        return False, None

    @staticmethod
    def chandelier_exit(df, current_bar, entry_price, current_price, atr, mult=3.0, period=22):
        """Trail stop at HH(22) - 3x ATR. Better than fixed trail."""
        lookback = min(period, current_bar)
        seg = df.iloc[current_bar - lookback:current_bar + 1]
        hh = seg["High"].max()
        trail = hh - atr * mult
        if current_price <= trail:
            return True, f"Chandelier({mult}xATR)"
        return False, None

    @staticmethod
    def parabolic_sar_exit(df, current_bar, current_price):
        """Parabolic SAR as trailing exit."""
        sar = _sf(df.iloc[current_bar], "SAR", None)
        if sar is None or np.isnan(sar):
            return False, None
        if current_price <= sar:
            return True, "ParSAR"
        return False, None

    @staticmethod
    def volume_exhaustion(df, current_bar):
        """3 consecutive bars: price up but volume declining (distribution)."""
        if current_bar < 3:
            return False, None
        seg = df.iloc[current_bar - 2:current_bar + 1]
        price_up = all(seg.iloc[j]["Close"] > seg.iloc[j - 1]["Close"] for j in range(1, 3))
        vol_down = all(seg.iloc[j]["Volume"] < seg.iloc[j - 1]["Volume"] for j in range(1, 3))
        if price_up and vol_down:
            return True, "VolExhaust"
        return False, None

    @staticmethod
    def gap_up_exit(df, current_bar, entry_price, pct_threshold=3.0):
        """Book full profit on gap-up > 3% above previous close."""
        if current_bar < 1:
            return False, None
        prev_close = float(df.iloc[current_bar - 1]["Close"])
        curr_open = float(df.iloc[current_bar]["Open"])
        gap_pct = (curr_open / max(prev_close, 0.01) - 1) * 100
        if gap_pct > pct_threshold:
            pnl = (curr_open - entry_price) / max(entry_price, 0.01) * 100
            if pnl > 0:
                return True, f"GapUp({gap_pct:.1f}%)"
        return False, None


# ─────────────────────────────────────────────────────────────────────
# 3. RISK MANAGER
# ─────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Tracks portfolio-level risk across multiple positions.

    State:
      - open_positions: list of active position dicts
      - trade_history: list of closed trade PnL%
      - consecutive_wins: int
      - consecutive_losses: int
      - daily_pnl: float (intraday)
      - drawdown_circuit_active: bool
      - halt_until_bar: int (bar index when circuit breaker activates)
    """

    def __init__(self, capital: float, max_portfolio_heat: float = 0.20,
                 max_daily_loss_pct: float = 2.0, dd_circuit_pct: float = 8.0,
                 dd_circuit_bars: int = 5):
        self.initial_capital = capital
        self.capital = capital
        self.max_heat = max_portfolio_heat
        self.max_daily_loss = max_daily_loss_pct
        self.dd_circuit_pct = dd_circuit_pct
        self.dd_circuit_bars = dd_circuit_bars

        self.open_positions: list = []
        self.trade_pnls: list = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.daily_pnl_start = capital
        self.daily_bar = -1
        self.circuit_halt_until = -1
        self.last_10_pnls: list = []

    @property
    def portfolio_heat(self) -> float:
        """Total capital at risk across all open positions."""
        total_risk = sum(p.get("risk_amt", 0) for p in self.open_positions)
        return total_risk / max(self.capital, 1)

    @property
    def in_circuit_breaker(self) -> bool:
        return self.circuit_halt_until >= 0

    def can_trade(self, current_bar: int, sector: str = None) -> tuple:
        """Check if new entry is allowed. Returns (allowed: bool, reason: str)."""
        if self.in_circuit_breaker:
            if current_bar <= self.circuit_halt_until:
                return False, f"Circuit breaker ({self.circuit_halt_until - current_bar} bars left)"
            self.circuit_halt_until = -1

        # Portfolio heat
        if self.portfolio_heat >= self.max_heat:
            return False, f"Portfolio heat {self.portfolio_heat:.1%} >= {self.max_heat:.0%}"

        # Max daily loss
        daily_loss = (self.daily_pnl - self.daily_pnl_start) / max(self.daily_pnl_start, 1) * 100
        if daily_loss <= -self.max_daily_loss:
            return False, f"Daily loss {daily_loss:.1f}% < -{self.max_daily_loss:.0f}%"

        return True, None

    def check_correlation(self, new_sector: str) -> tuple:
        """Max 2 stocks per sector. Returns (allowed: bool, reason: str)."""
        if new_sector is None:
            return True, None
        sector_count = sum(1 for p in self.open_positions if p.get("sector") == new_sector)
        if sector_count >= 2:
            return False, f"Sector {new_sector} already has {sector_count} positions"
        return True, None

    def open_position(self, pos: dict):
        """Register a new position."""
        self.open_positions.append(pos)
        self.daily_bar = -1  # Reset daily tracking

    def close_position(self, pos_id: str, pnl_pct: float, pnl_amt: float, current_bar: int):
        """Close a position and update risk state."""
        self.open_positions = [p for p in self.open_positions if p.get("id") != pos_id]
        self.trade_pnls.append(pnl_pct)
        self.last_10_pnls.append(pnl_pct)
        if len(self.last_10_pnls) > 10:
            self.last_10_pnls.pop(0)

        self.capital += pnl_amt
        self.daily_pnl += pnl_amt

        # Consecutive tracking
        if pnl_pct > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Drawdown circuit breaker
        if len(self.last_10_pnls) >= 5:
            rolling_dd = sum(p for p in self.last_10_pnls if p < 0)
            if abs(rolling_dd) >= self.dd_circuit_pct:
                self.circuit_halt_until = current_bar + self.dd_circuit_bars

    def get_kelly_fraction(self, default: float = 0.02) -> float:
        """Dynamic Kelly from rolling trade history."""
        if len(self.trade_pnls) < 10:
            return default
        wins = [p for p in self.trade_pnls[-20:] if p > 0]
        losses = [p for p in self.trade_pnls[-20:] if p <= 0]
        if not losses:
            return 0.25  # Cap
        wr = len(wins) / max(len(self.trade_pnls[-20:]), 1)
        aw = np.mean(wins) if wins else 0
        al = abs(np.mean(losses)) if losses else 1
        R = abs(aw / max(al, 0.01))
        f = wr - (1 - wr) / max(R, 0.1)
        return max(0.01, min(f, 0.25))

    def get_anti_martingale_mult(self) -> float:
        """Increase 10% after 2 consecutive wins, decrease 10% after 2 losses."""
        if self.consecutive_wins >= 2:
            return 1.0 + 0.1 * min(self.consecutive_wins, 5)
        if self.consecutive_losses >= 2:
            return 1.0 - 0.1 * min(self.consecutive_losses, 5)
        return 1.0

    def update_daily(self, current_bar: int):
        """Track daily bar changes for daily loss limit."""
        if current_bar != self.daily_bar:
            self.daily_bar = current_bar
            self.daily_pnl_start = self.daily_pnl

    def pyramid_allowed(self, current_price: float, entry_price: float,
                         atr: float, r_mult: float = 1.0) -> bool:
        """Allow adding to winner only if > 1R above entry and heat < 15%."""
        if self.portfolio_heat >= 0.15:
            return False
        r_distance = (current_price - entry_price) / max(atr, 0.01)
        return r_distance > r_mult


# ─────────────────────────────────────────────────────────────────────
# 4. ADVANCED PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────

def compute_advanced_metrics(trades: list, initial_capital: float) -> dict:
    """
    Compute 16 institutional-grade performance metrics from a trade list.
    Each trade: {"PnL%": float, "PnL₹": float, "Result": "WIN"/"LOSS",
                 "Entry": date, "Exit": date, "Bars": int}
    """
    if not trades or len(trades) < 3:
        return {}

    n = len(trades)
    rets = np.array([t["PnL%"] for t in trades])
    rets_rupee = np.array([t.get("PnL₹", 0) for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    wc = len(wins)
    lc = len(losses)
    wr = wc / n * 100 if n > 0 else 0
    aw = float(np.mean(wins)) if wc > 0 else 0
    al = float(abs(np.mean(losses))) if lc > 0 else 0
    net_profit = float(np.sum(rets_rupee))
    total_return = (net_profit + initial_capital) / max(initial_capital, 1) - 1

    # ── 1. Profit Factor ──
    gross_profit = float(np.sum(wins)) if wc > 0 else 0
    gross_loss = float(abs(np.sum(losses))) if lc > 0 else 0
    pf = round(gross_profit / max(gross_loss, 0.01), 2)

    # ── 2. Payoff Ratio ──
    payoff = round(aw / max(al, 0.01), 2)

    # ── 3. Expectancy ──
    expectancy = round((wr / 100 * aw) - ((1 - wr / 100) * al), 2)

    # ── 4. Sharpe Ratio ──
    sharpe = round(float(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252)), 2) if len(rets) > 1 else 0

    # ── 5. Sortino Ratio (only downside deviation) ──
    downside = rets[rets < 0]
    dd_dev = float(np.std(downside)) if len(downside) > 1 else 0.01
    sortino = round(float(np.mean(rets) / max(dd_dev, 0.01) * np.sqrt(252)), 2)

    # ── Equity Curve ──
    eq = [initial_capital]
    for t in trades:
        eq.append(eq[-1] + t.get("PnL₹", 0))
    eq_series = pd.Series(eq[1:])
    peak = eq_series.cummax()
    dd_vals = (eq_series - peak) / peak.replace(0, 1) * 100
    max_dd = float(dd_vals.min()) if len(dd_vals) > 0 else 0

    # ── 6. Calmar Ratio ──
    cagr = total_return * 100  # Approximate
    calmar = round(cagr / max(abs(max_dd), 0.1), 2)

    # ── 7. MAR Ratio ──
    mar = round(cagr / max(abs(max_dd), 0.1), 2)  # Same as Calmar

    # ── 8. Recovery Factor ──
    recovery = round(net_profit / max(abs(max_dd) * initial_capital / 100, 1), 2)

    # ── 9. Omega Ratio ──
    threshold = 0
    excess = rets - threshold
    gains = excess[excess > 0].sum()
    losses_abs = abs(excess[excess < 0].sum())
    omega = round(gains / max(losses_abs, 0.01), 2)

    # ── 10. Ulcer Index (depth + duration of drawdowns) ──
    dd_squared = (dd_vals ** 2).sum() / max(len(dd_vals), 1)
    ulcer = round(float(np.sqrt(dd_squared)), 2)

    # ── 11. K-Ratio ──
    eq_vals = np.array(eq) / initial_capital
    x = np.arange(len(eq_vals))
    if len(eq_vals) > 1 and np.std(x) > 0:
        from numpy import polyfit
        slope, _ = polyfit(x, eq_vals, 1)
        residuals = eq_vals - (slope * x + _)
        se_slope = np.std(residuals) / (np.std(x) * np.sqrt(len(x)))
        k_ratio = round(slope / max(se_slope, 0.0001), 2) if se_slope > 0 else 0
    else:
        k_ratio = 0

    # ── 12. Tail Ratio ──
    if len(rets) >= 10:
        p95 = np.percentile(rets, 95)
        p05 = np.percentile(rets, 5)
        tail = round(abs(p95 / max(p05, 0.01)), 2) if p05 != 0 else 0
    else:
        tail = 0

    # ── 13. Consecutive Loss Streak ──
    max_loss_streak = 0
    curr_streak = 0
    streaks = []
    for t in trades:
        if t["Result"] == "LOSS":
            curr_streak += 1
            max_loss_streak = max(max_loss_streak, curr_streak)
        else:
            if curr_streak > 0:
                streaks.append(curr_streak)
            curr_streak = 0
    if curr_streak > 0:
        streaks.append(curr_streak)
    avg_loss_streak = round(np.mean(streaks), 1) if streaks else 0

    # ── 14. Trade Duration Analysis ──
    win_durations = []
    loss_durations = []
    for t in trades:
        bars = t.get("Bars", 0)
        if t["Result"] == "WIN":
            win_durations.append(bars)
        else:
            loss_durations.append(bars)
    avg_win_bars = round(np.mean(win_durations), 1) if win_durations else 0
    avg_loss_bars = round(np.mean(loss_durations), 1) if loss_durations else 0

    # ── 15. MFE/MAE approximation ──
    # Uses exit reason to classify
    mfe_mae = {}
    for t in trades:
        reason = t.get("Reason", "?")
        mfe_mae[reason] = mfe_mae.get(reason, 0) + 1

    # ── 16. Rolling Sharpe (3-month ≈ 63 bars window) ──
    rolling_sharpes = []
    window = min(63, len(rets))
    if window >= 20:
        for j in range(window, len(rets) + 1):
            chunk = rets[j - window:j]
            rs = np.mean(chunk) / max(np.std(chunk), 0.01) * np.sqrt(252)
            rolling_sharpes.append(rs)
    avg_rolling_sharpe = round(float(np.mean(rolling_sharpes)), 2) if rolling_sharpes else sharpe
    rolling_sharpe_std = round(float(np.std(rolling_sharpes)), 2) if len(rolling_sharpes) > 1 else 0

    return {
        "Profit Factor": pf,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "MAR Ratio": mar,
        "Omega Ratio": omega,
        "Payoff Ratio": payoff,
        "Expectancy %": expectancy,
        "Recovery Factor": recovery,
        "Ulcer Index": ulcer,
        "K-Ratio": k_ratio,
        "Tail Ratio": tail,
        "Max Drawdown %": round(max_dd, 2),
        "Max Loss Streak": max_loss_streak,
        "Avg Loss Streak": avg_loss_streak,
        "Avg Win Bars": avg_win_bars,
        "Avg Loss Bars": avg_loss_bars,
        "Rolling Sharpe (Avg)": avg_rolling_sharpe,
        "Rolling Sharpe (Std)": rolling_sharpe_std,
        "Exit Breakdown": mfe_mae,
    }


# ─────────────────────────────────────────────────────────────────────
# 5. REGIME DETECTION (HMM-like)
# ─────────────────────────────────────────────────────────────────────

def detect_regime_hmm(df: pd.DataFrame, lookback: int = 252) -> str:
    """
    Two-state regime detection: TRENDING vs RANGING.
    Uses rolling ADX + volatility ratio instead of full HMM (avoids sklearn dep).
    """
    if df is None or len(df) < 50:
        return "RANGING"
    recent = df.tail(lookback)
    adx_vals = recent["ADX"].dropna()
    if len(adx_vals) < 20:
        return "RANGING"
    avg_adx = adx_vals.mean()
    # Volatility regime
    atr_pct = (recent["ATR"] / recent["Close"] * 100).dropna()
    if len(atr_pct) < 20:
        return "RANGING"
    recent_vol = atr_pct.tail(20).mean()
    older_vol = atr_pct.tail(60).head(40).mean()
    vol_ratio = recent_vol / max(older_vol, 0.01)

    if avg_adx > 22 and vol_ratio < 1.3:
        return "TRENDING"
    return "RANGING"


# ─────────────────────────────────────────────────────────────────────
# 6. WALK-FORWARD & ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────

def anchored_walk_forward(symbol: str, folds: int = 4) -> dict:
    """
    Anchored walk-forward: first fold always starts from day 1 (expanding window).
    Each fold: train on expanding window, test on next 1/fold of data.
    """
    from core.utils import load_data
    df = load_data(symbol, period="7y")
    if df is None or len(df) < 1000:
        return {"Error": "Insufficient data"}

    df = compute_indicators(df)
    total = len(df)
    fold_size = total // (folds + 1)

    results = []
    for f in range(1, folds + 1):
        train_end = f * fold_size * 2  # Expanding
        test_start = train_end
        test_end = min(test_start + fold_size, total)

        if test_start >= total or test_end <= test_start:
            break

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        # Quick IS/OOS performance
        from core.profit_engine_v4 import kelly_fraction as kf
        is_trades = _run_quick_test(train)
        oos_trades = _run_quick_test(test)

        is_ret = np.mean(is_trades) if is_trades else 0
        oos_ret = np.mean(oos_trades) if oos_trades else 0
        is_sharpe = (np.mean(is_trades) / max(np.std(is_trades), 0.01) * np.sqrt(252)) if len(is_trades) > 1 else 0
        oos_sharpe = (np.mean(oos_trades) / max(np.std(oos_trades), 0.01) * np.sqrt(252)) if len(oos_trades) > 1 else 0

        results.append({
            "Fold": f, "Train Bars": train_end, "Test Bars": test_end - test_start,
            "IS Trades": len(is_trades), "OOS Trades": len(oos_trades),
            "IS Sharpe": round(is_sharpe, 2), "OOS Sharpe": round(oos_sharpe, 2),
            "IS Avg Ret%": round(is_ret, 2), "OOS Avg Ret%": round(oos_ret, 2),
        })

    # Summary
    is_sharpes = [r["IS Sharpe"] for r in results]
    oos_sharpes = [r["OOS Sharpe"] for r in results]
    is_avg = np.mean(is_sharpes) if is_sharpes else 0
    oos_avg = np.mean(oos_sharpes) if oos_sharpes else 0
    degradation = (oos_avg / max(is_avg, 0.01)) * 100 if is_avg > 0 else 0

    return {
        "Folds": results,
        "Avg IS Sharpe": round(is_avg, 2),
        "Avg OOS Sharpe": round(oos_avg, 2),
        "OOS/IS Ratio %": round(degradation, 1),
        "Overfit Flag": degradation < 50,
    }


def _run_quick_test(df):
    """Minimal backtest for walk-forward. Returns list of PnL%."""
    trades = []
    pos = False
    ep = 0
    for i in range(200, len(df)):
        bar = df.iloc[:i + 1]
        r = bar.iloc[-1]
        c = float(r["Close"])
        atr = float(r.get("ATR", c * 0.02))
        if np.isnan(atr) or atr <= 0:
            atr = c * 0.02
        ema50 = _sf(r, "EMA50", c)
        ema200 = _sf(r, "EMA200", c)
        rsi = _sf(r, "RSI", 50)
        score = 0
        if c > ema50 and ema50 > ema200: score += 2
        if 55 <= rsi <= 75: score += 2
        if _sf(r, "MACD_HIST", 0) > 0: score += 1
        if _sf(r, "SUPERTREND_DIR", 0) == 1: score += 1
        if _sf(r, "RVOL", 1) > 1.5: score += 1

        if not pos and score >= 4:
            ep = c
            pos = True
        elif pos:
            if c <= ep - atr * 2 or c >= ep + atr * 5 or score < 2:
                trades.append((c - ep) / max(ep, 0.01) * 100)
                pos = False
    if pos:
        trades.append((float(df["Close"].iloc[-1]) - ep) / max(ep, 0.01) * 100)
    return trades


def parameter_stability_test(symbol: str, base_params: dict, ranges: dict) -> dict:
    """
    Test parameter stability by varying each param ±20%.
    Flags any parameter where performance degrades > 30%.
    """
    from core.utils import load_data
    df = load_data(symbol, period="5y")
    if df is None or len(df) < 300:
        return {"Error": "Insufficient data"}
    df = compute_indicators(df)

    base_pf = _calc_pf_quick(df, base_params)
    results = {}

    for param, values in ranges.items():
        degraded = False
        for val in values:
            params = base_params.copy()
            params[param] = val
            pf = _calc_pf_quick(df, params)
            if base_pf > 0:
                change = abs(pf - base_pf) / max(base_pf, 0.01) * 100
                if change > 30:
                    degraded = True
            results[f"{param}={val}"] = {"PF": pf, "Degraded": degraded}
        results[f"{param}_stable"] = not degraded

    return {
        "Base PF": base_pf,
        "Parameters": results,
        "Pass": all(not v.get("Degraded", False) for k, v in results.items() if isinstance(v, dict)),
    }


def _calc_pf_quick(df, params):
    """Quick PF calculation for parameter testing."""
    trades = []
    pos = False
    ep = 0
    min_score = params.get("min_score", 4)
    stop_mult = params.get("stop_mult", 2.0)
    tgt_mult = params.get("tgt_mult", 5.0)
    for i in range(200, len(df)):
        bar = df.iloc[:i + 1]
        r = bar.iloc[-1]
        c = float(r["Close"])
        atr = float(r.get("ATR", c * 0.02))
        if np.isnan(atr) or atr <= 0:
            atr = c * 0.02
        score = 0
        if c > _sf(r, "EMA50", c) and _sf(r, "EMA50", c) > _sf(r, "EMA200", c): score += 2
        if 55 <= _sf(r, "RSI", 50) <= 75: score += 2
        if _sf(r, "MACD_HIST", 0) > 0: score += 1
        if _sf(r, "SUPERTREND_DIR", 0) == 1: score += 1
        if _sf(r, "RVOL", 1) > 1.5: score += 1
        if not pos and score >= min_score:
            ep = c
            pos = True
        elif pos:
            if c <= ep - atr * stop_mult or c >= ep + atr * tgt_mult or score < 2:
                trades.append((c - ep) / max(ep, 0.01) * 100)
                pos = False
    if pos:
        trades.append((float(df["Close"].iloc[-1]) - ep) / max(ep, 0.01) * 100)
    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    if not trades:
        return 0
    gw = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t <= 0))
    return round(gw / max(gl, 0.01), 2) if gl > 0 else gw


def deflated_sharpe_ratio(observed_sharpe: float, num_trials: int,
                           num_observations: int) -> float:
    """
    Adjust Sharpe for multiple testing bias.
    Uses simple approximation when scipy is unavailable.
    """
    if num_trials <= 1 or observed_sharpe <= 0:
        return observed_sharpe
    try:
        from scipy.stats import norm
        e_max = norm.ppf(1 - 1 / num_trials) * (1 - 0.5772 * norm.ppf(1 - 1 / num_trials) ** (-2))
        adjustment = e_max / np.sqrt(num_observations / 252) if num_observations > 0 else 0
        return round(max(observed_sharpe - adjustment, 0), 2)
    except ImportError:
        # Simple approximation without scipy
        adjustment = np.sqrt(2 * np.log(num_trials)) / np.sqrt(num_observations / max(252, 1))
        return round(max(observed_sharpe - adjustment, 0), 2)


# ─────────────────────────────────────────────────────────────────────
# 7. MAIN V5 BACKTEST
# ─────────────────────────────────────────────────────────────────────

def _get_sector(symbol: str) -> Optional[str]:
    """Get sector from sectors module if available."""
    try:
        from data.sectors import get_sector
        return get_sector(symbol.replace(".NS", ""))
    except Exception:
        return None


def run_v5_backtest(
    symbol,
    capital=100000,
    risk_pct=2.0,
    max_hold=15,
    use_regime=True,
    # ── Section 1: Entry Filters ──
    use_volume_filter=True,
    use_rs_filter=True,
    use_52w_filter=True,
    use_ema_stack=True,
    use_vwap_filter=True,
    use_earnings_blackout=False,
    use_sector_momentum=False,
    use_breadth_gate=True,
    # ── Section 2: Exit Rules ──
    use_time_stop=True,
    use_chandelier=True,
    use_par_sar=False,
    use_vol_exhaust=True,
    use_gap_up_exit=True,
    # ── Section 3: Risk Upgrades ──
    use_portfolio_heat=True,
    use_correlation_filter=True,
    use_circuit_breaker=True,
    use_dynamic_kelly=True,
    use_anti_martingale=True,
    use_daily_loss_limit=False,
    use_pyramid=False,
):
    """Institutional-grade backtest with all v5 features."""
    from core.utils import load_data
    from core.market_regime import get_market_regime

    try:
        df = load_data(symbol, period="5y")
        if df is None or len(df) < 220:
            return {"Error": "No data", "Total Trades": 0}

        df = compute_indicators(df)
        regime = get_market_regime() if use_regime else "NEUTRAL"

        # Add Parabolic SAR column if needed
        if use_par_sar and "SAR" not in df.columns:
            _add_par_sar(df)

        # Weekly trend
        weekly_trend = _get_weekly_trend(df)

        # Risk manager
        rm = RiskManager(capital, max_portfolio_heat=0.20)

        cap = capital
        trades = []
        pos = False
        ep = eb = 0
        ed = None
        sh = 0
        pos_id = None
        sector = _get_sector(symbol)

        for i in range(200, len(df)):
            bar = df.iloc[i]
            c = float(bar["Close"])
            atr = float(bar.get("ATR", c * 0.02))
            if np.isnan(atr) or atr <= 0:
                atr = c * 0.02

            rm.update_daily(i)

            # ── Generate v4-style base signal (used for scoring) ──
            sig = _generate_v5_signal(df.iloc[:i + 1], weekly_trend=weekly_trend, regime=regime)

            # ── ENTRY ──
            if not pos:
                # 1. Base signal check
                if sig["Signal"] not in ("STRONG_BUY", "BUY"):
                    continue
                if sig["Score"] < (5 if regime == "BEARISH" else 4):
                    continue

                # 2. Entry filters
                filters_ok = True
                filter_reasons = []

                if use_volume_filter:
                    ok, r = EntryFilters.volume_confirmation(df, i)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_ema_stack:
                    ok, r = EntryFilters.ema_stack(df, i)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_vwap_filter:
                    ok, r = EntryFilters.vwap_filter(df, i)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_52w_filter:
                    ok, r = EntryFilters.week_52_proximity(df, i)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_rs_filter:
                    ok, r = EntryFilters.relative_strength(symbol)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_breadth_gate:
                    ok, r = EntryFilters.market_breadth_gate()
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if use_earnings_blackout:
                    ok, r = EntryFilters.earnings_blackout(df, i)
                    if not ok:
                        filters_ok = False
                        filter_reasons.append(r)

                if not filters_ok:
                    continue

                # 3. Risk manager checks
                if use_portfolio_heat or use_daily_loss_limit or use_circuit_breaker:
                    allowed, reason = rm.can_trade(i, sector)
                    if not allowed:
                        continue

                if use_correlation_filter:
                    allowed, reason = rm.check_correlation(sector)
                    if not allowed:
                        continue

                # 4. Position sizing
                if use_dynamic_kelly:
                    kelly = rm.get_kelly_fraction(risk_pct / 100)
                    effective_risk = kelly
                else:
                    effective_risk = risk_pct / 100

                if use_anti_martingale:
                    am_mult = rm.get_anti_martingale_mult()
                    effective_risk *= am_mult

                # Regime adjustment
                if regime == "BEARISH":
                    effective_risk *= 0.5
                elif regime == "SIDEWAYS":
                    effective_risk *= 0.75

                stop_mult = sig.get("StopMult", 2.0)
                sl = c - atr * stop_mult
                sd = abs(c - sl)
                risk_amt = cap * effective_risk
                raw = int(risk_amt / max(sd, 0.01)) if sd > 0 else 0
                mx = int(cap * 0.3 / max(c, 1))
                sh = max(1, min(raw, mx))

                # Open position
                ep = c
                eb = i
                ed = bar.name
                pos = True
                pos_id = f"{symbol}_{i}"
                rm.open_position({
                    "id": pos_id, "entry": ep, "entry_bar": eb, "shares": sh,
                    "sector": sector, "risk_amt": risk_amt,
                    "stop_mult": stop_mult, "atr": atr,
                })
                continue

            # ── EXIT ──
            if pos:
                ex = None
                er = ""
                sl_use = sig.get("StopLoss", c - atr * 2.5)
                tgt = sig.get("Target", c + atr * 5.0)

                # Chandelier Exit (overrides basic trail)
                if use_chandelier:
                    exit_ch, reason_ch = ExitRules.chandelier_exit(df, i, ep, c, atr)
                    if exit_ch:
                        ex, er = c, reason_ch

                # Parabolic SAR
                if use_par_sar and ex is None:
                    exit_sar, reason_sar = ExitRules.parabolic_sar_exit(df, i, c)
                    if exit_sar:
                        ex, er = c, reason_sar

                # Volume Exhaustion
                if use_vol_exhaust and ex is None:
                    exit_ve, reason_ve = ExitRules.volume_exhaustion(df, i)
                    if exit_ve:
                        ex, er = c, reason_ve

                # Gap-Up Exit
                if use_gap_up_exit and ex is None:
                    exit_gap, reason_gap = ExitRules.gap_up_exit(df, i, ep)
                    if exit_gap:
                        ex, er = c, reason_gap

                # Time Stop (dead money filter)
                if use_time_stop and ex is None:
                    exit_ts, reason_ts = ExitRules.time_stop(eb, i, max_hold, ep, c, atr)
                    if exit_ts:
                        ex, er = c, reason_ts

                # Basic trail/target (fallback)
                if ex is None:
                    if c <= sl_use:
                        ex, er = c, "TrailStop"
                    elif c >= tgt:
                        ex, er = c, "Target"
                    elif sig["Signal"] == "SELL":
                        ex, er = c, "SellSig"

                # Pyramid: add to winner
                if use_pyramid and ex is None:
                    if rm.pyramid_allowed(c, ep, atr):
                        add_shares = int(sh * 0.3)
                        sh += add_shares
                        rm.open_position({
                            "id": f"{pos_id}_pyra", "entry": c, "entry_bar": i,
                            "shares": add_shares, "sector": sector,
                            "risk_amt": add_shares * atr * 2, "atr": atr,
                        })

                if ex is not None:
                    pp = (ex - ep) / max(ep, 0.01) * 100
                    pa = sh * (ex - ep)
                    cap += pa
                    rm.close_position(pos_id, pp, pa, i)

                    trades.append({
                        "Entry": ed, "Exit": bar.name, "Entry₹": round(ep, 2),
                        "Exit₹": round(ex, 2), "PnL%": round(pp, 2),
                        "PnL₹": round(pa, 2),
                        "Result": "WIN" if pp > 0 else "LOSS",
                        "Bars": i - eb + 1, "Reason": er, "Shares": sh,
                    })
                    pos = False
                    pos_id = None

        # Close open
        if pos:
            fp = float(df["Close"].iloc[-1])
            pp = (fp - ep) / max(ep, 0.01) * 100
            pa = sh * (fp - ep)
            cap += pa
            trades.append({
                "Entry": ed, "Exit": df.index[-1], "Entry₹": round(ep, 2),
                "Exit₹": round(fp, 2), "PnL%": round(pp, 2),
                "PnL₹": round(pa, 2), "Result": "WIN" if pp > 0 else "LOSS",
                "Bars": len(df) - eb, "Reason": "End", "Shares": sh,
            })

        # ── METRICS ──
        n = len(trades)
        if n == 0:
            return {
                "Total Return %": 0.0, "Total Trades": 0,
                "Win Rate %": 0.0, "Profit Factor": 0.0,
                "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
                "Final Capital": cap, "Trades": pd.DataFrame(),
                "Error": None, "Version": "v5"
            }

        wl = [t for t in trades if t["Result"] == "WIN"]
        ll = [t for t in trades if t["Result"] == "LOSS"]
        wc = len(wl)
        lc = len(ll)
        wr = round(wc / n * 100, 1)
        aw = round(np.mean([t["PnL%"] for t in wl]), 2) if wl else 0
        al = round(np.mean([t["PnL%"] for t in ll]), 2) if ll else 0

        # Standard metrics
        rets = [t["PnL%"] for t in trades]
        tp = sum(t["PnL%"] for t in wl) if wl else 0
        tl = abs(sum(t["PnL%"] for t in ll)) if ll else 0
        pf = round(tp / max(tl, 0.01), 2)
        shp = round(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252), 2) if len(rets) > 1 else 0

        eq = [capital]
        for t in trades:
            eq.append(eq[-1] + t.get("PnL₹", 0))
        es = pd.Series(eq[1:])
        rm_ = es.cummax()
        dd = (es - rm_) / rm_.replace(0, 1) * 100
        mdd = round(dd.min(), 2) if len(dd) > 0 else 0

        ret = round((cap - capital) / capital * 100, 2)
        ev = round((wr / 100 * aw) - ((1 - wr / 100) * abs(al)), 2)

        # ── Advanced Analytics ──
        adv = compute_advanced_metrics(trades, capital)
        kelly = rm.get_kelly_fraction(0.02)
        er_map = {}
        for t in trades:
            er_map[t.get("Reason", "?")] = er_map.get(t.get("Reason", "?"), 0) + 1

        # Deflated Sharpe
        dsr = deflated_sharpe_ratio(shp, num_trials=20, num_observations=n)

        # HMM Regime
        hmm_regime = detect_regime_hmm(df)

        # Walk-forward
        wf = anchored_walk_forward(symbol)

        return {
            # Standard
            "Total Return %": ret, "Total Trades": n,
            "Win Rate %": wr, "Avg Win %": aw, "Avg Loss %": al,
            "Profit Factor": pf, "Sharpe Ratio": shp,
            "Max Drawdown %": mdd, "Expectancy %": ev,
            "Final Capital": round(cap, 2),
            "Exit Breakdown": er_map,
            "Trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            # v5 Advanced
            "Advanced": adv,
            "Kelly Fraction": round(kelly, 4),
            "Deflated Sharpe": dsr,
            "HMM Regime": hmm_regime,
            "Weekly Trend": weekly_trend,
            "Market Regime": regime,
            "WalkForward": wf,
            "Version": "v5",
            "Error": None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"Error": str(e), "Total Trades": 0}


def _get_weekly_trend(df: pd.DataFrame) -> str:
    """Compute weekly trend from daily df."""
    try:
        wk = df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()
        if len(wk) < 10:
            return "NEUTRAL"
        wr = wk.iloc[-1]
        wc = _sf(wr, "Close")
        we50 = wk["Close"].ewm(span=50).mean().iloc[-1] if len(wk) >= 50 else wc
        we200 = wk["Close"].ewm(span=200).mean().iloc[-1] if len(wk) >= 200 else we50
        if wc > we50 > we200:
            return "BULLISH"
        elif wc < we50 < we200:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _generate_v5_signal(df, weekly_trend="NEUTRAL", regime="BULLISH"):
    """Signal generator for v5 (adapted from v4 with stricter scoring)."""
    from core.profit_engine_v4 import generate_v4_signal
    return generate_v4_signal(df, weekly_trend=weekly_trend, regime=regime)


def _add_par_sar(df, acceleration=0.02, max_acc=0.2):
    """Add Parabolic SAR column to dataframe."""
    high, low = df["High"].values, df["Low"].values
    close = df["Close"].values
    sar = np.zeros(len(df))
    ep = np.zeros(len(df))
    trend = np.zeros(len(df))
    af = np.zeros(len(df))

    if len(df) < 2:
        df["SAR"] = close
        return

    # Initialize
    trend[1] = 1 if close[1] > close[0] else -1
    ep[1] = high[1] if trend[1] == 1 else low[1]
    af[1] = acceleration
    if trend[1] == 1:
        sar[1] = min(low[0], low[1])
    else:
        sar[1] = max(high[0], high[1])

    for i in range(2, len(df)):
        sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

        if trend[i - 1] == 1:
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = acceleration
            else:
                trend[i] = 1
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + acceleration, max_acc)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                sar[i] = min(sar[i], low[i - 1], low[i])
        else:
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = acceleration
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + acceleration, max_acc)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                sar[i] = max(sar[i], high[i - 1], high[i])

    df["SAR"] = sar
    df["SAR_TREND"] = trend


# ─────────────────────────────────────────────────────────────────────
# 8. COMPARISON
# ─────────────────────────────────────────────────────────────────────

def compare_v5_vs_old(symbol: str) -> dict:
    """Compare old baseline vs v5."""
    from core.backtest import run_backtest as old_bt
    old = old_bt(symbol) or {}
    new = run_v5_backtest(symbol) or {}

    op = old.get("Profit Factor", 0) or 0
    npf = new.get("Profit Factor", 0) or 0
    or_ = old.get("Total Return %", 0) or 0
    nr_ = new.get("Total Return %", 0) or 0

    adv = new.get("Advanced", {})

    return {
        "old": {
            "Total Return %": or_,
            "Win Rate %": old.get("Win Rate %", 0) or 0,
            "Profit Factor": op,
            "Max Drawdown %": old.get("Max Drawdown %", 0) or 0,
            "Sharpe Ratio": old.get("Sharpe Ratio", 0) or 0,
            "Total Trades": old.get("Total Trades", 0) or 0,
        },
        "new": {
            "Total Return %": nr_,
            "Win Rate %": new.get("Win Rate %", 0) or 0,
            "Profit Factor": npf,
            "Max Drawdown %": new.get("Max Drawdown %", 0) or 0,
            "Sharpe Ratio": new.get("Sharpe Ratio", 0) or 0,
            "Total Trades": new.get("Total Trades", 0) or 0,
            "Calmar": adv.get("Calmar Ratio", 0),
            "Sortino": adv.get("Sortino Ratio", 0),
            "Omega": adv.get("Omega Ratio", 0),
            "Payoff": adv.get("Payoff Ratio", 0),
            "Recovery Factor": adv.get("Recovery Factor", 0),
            "Ulcer Index": adv.get("Ulcer Index", 0),
            "K-Ratio": adv.get("K-Ratio", 0),
            "Kelly Fraction": new.get("Kelly Fraction", 0),
            "Deflated Sharpe": new.get("Deflated Sharpe", 0),
            "HMM Regime": new.get("HMM Regime", "N/A"),
            "Weekly Trend": new.get("Weekly Trend", "N/A"),
            "Market Regime": new.get("Market Regime", "N/A"),
        },
        "improvement": {
            "PF Delta %": round((npf - op) / max(op, 0.1) * 100, 1),
            "Return Delta %": round(nr_ - or_, 2),
        },
        "details": new
    }
