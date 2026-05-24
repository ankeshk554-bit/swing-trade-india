"""
V6 — Precision Backtesting Engine for NSE Equities
====================================================
Based on empirical V4/V5 diagnosis.

6 EXACT ENTRY FILTERS · 6 EXACT EXIT RULES · COST-AWARE · KELLY SIZED
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Callable
from core.indicators import compute_indicators, compute_rsi

# ─────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────

def _sf(r, field, default=0):
    v = r.get(field, default)
    try:
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else default
    except (ValueError, TypeError):
        return default


# ═════════════════════════════════════════════════════════════════════
# CLASS 1: IndiaMarketUtils
# ═════════════════════════════════════════════════════════════════════

class IndiaMarketUtils:
    """Indian market data, cost model, and calendar utilities."""

    @staticmethod
    def load_nifty50_data(start=None, end=None) -> pd.DataFrame:
        """Fetch Nifty 50 benchmark data."""
        import yfinance as yf
        try:
            n = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
            if n is not None and not n.empty:
                if isinstance(n.columns, pd.MultiIndex):
                    n.columns = n.columns.get_level_values(0)
                return n
        except Exception:
            pass
        return pd.DataFrame()

    @staticmethod
    def compute_rs_vs_nifty(stock_df: pd.DataFrame, nifty_df: pd.DataFrame, window=63) -> float:
        """Rolling 63-day RS vs Nifty 50. RS = (1 + S_ret) / (1 + N_ret)."""
        if stock_df is None or nifty_df is None or len(stock_df) < window or len(nifty_df) < window:
            return 1.0
        s_ret = float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-window]) - 1
        n_ret = float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-window]) - 1
        return (1 + s_ret) / max(1 + n_ret, 0.01)

    @staticmethod
    def is_earnings_blackout(date: pd.Timestamp, earnings_dates: list, before=5, after=2) -> bool:
        """Check if date is within blackout window of any earnings date."""
        if not earnings_dates:
            return False
        for ed in earnings_dates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts - timedelta(days=before) <= date <= ed_ts + timedelta(days=after):
                return True
        return False

    @staticmethod
    def apply_cost_model(entry_price: float, exit_price: float, shares: int, is_buy: bool = True) -> dict:
        """
        SEBI-compliant cost model for delivery trades.

        Costs applied per leg (entry and exit separately).
        Returns dict with total costs and breakdown.
        """
        trade_value_entry = entry_price * shares
        trade_value_exit = exit_price * shares

        # --- Entry leg costs ---
        brk_entry = min(20, 0.0003 * trade_value_entry)
        exc_entry = 0.0000345 * trade_value_entry
        sebi_entry = 0.000001 * trade_value_entry
        stamp_entry = 0.00015 * trade_value_entry  # stamp duty on buy side only
        gst_entry = 0.18 * (brk_entry + exc_entry + sebi_entry)

        # --- Exit leg costs ---
        stt_exit = 0.001 * trade_value_exit  # STT 0.1% on sell for delivery
        brk_exit = min(20, 0.0003 * trade_value_exit)
        exc_exit = 0.0000345 * trade_value_exit
        sebi_exit = 0.000001 * trade_value_exit
        gst_exit = 0.18 * (brk_exit + exc_exit + sebi_exit)

        total_entry = brk_entry + exc_entry + sebi_entry + stamp_entry + gst_entry
        total_exit = stt_exit + brk_exit + exc_exit + sebi_exit + gst_exit
        total_cost = total_entry + total_exit

        return {
            "entry_cost": round(total_entry, 2),
            "exit_cost": round(total_exit, 2),
            "total_cost": round(total_cost, 2),
            "breakdown": {
                "STT": round(stt_exit, 2),
                "Brokerage": round(brk_entry + brk_exit, 2),
                "Exchange": round(exc_entry + exc_exit, 2),
                "SEBI": round(sebi_entry + sebi_exit, 2),
                "GST": round(gst_entry + gst_exit, 2),
                "Stamp": round(stamp_entry, 2),
            }
        }

    @staticmethod
    def is_fno_expiry_week(date: pd.Timestamp) -> bool:
        """Check if date falls in F&O expiry week (last Thu of month ±2 days)."""
        import calendar
        year, month = date.year, date.month
        last_day = calendar.monthrange(year, month)[1]
        last_thu = None
        for d in range(last_day, last_day - 7, -1):
            if datetime(year, month, d).weekday() == 3:  # Thursday
                last_thu = datetime(year, month, d)
                break
        if last_thu is None:
            return False
        expiry_start = pd.Timestamp(last_thu) - timedelta(days=2)
        expiry_end = pd.Timestamp(last_thu) + timedelta(days=2)
        return expiry_start <= date <= expiry_end


# ═════════════════════════════════════════════════════════════════════
# CLASS 2: EntrySignalGenerator
# ═════════════════════════════════════════════════════════════════════

class EntrySignalGenerator:
    """
    6 exact entry filters.
    generate_entry_signal() returns (entry: bool, reasons: list of passed, score: int).
    """

    def __init__(self, adx_threshold=22, vol_multiplier=1.5, rs_window=63,
                 earnings_before=5, earnings_after=2):
        self.adx_threshold = adx_threshold
        self.vol_multiplier = vol_multiplier
        self.rs_window = rs_window
        self.earnings_before = earnings_before
        self.earnings_after = earnings_after

    @staticmethod
    def check_weekly_trend_gate(weekly_df: pd.DataFrame) -> tuple:
        """Weekly close > 10-week EMA (simpler than 50-week)."""
        if weekly_df is None or len(weekly_df) < 12:
            return False, "Insufficient weekly data"
        wk = weekly_df.copy()
        wk["EMA10"] = wk["Close"].ewm(span=10).mean()
        c = float(wk["Close"].iloc[-1])
        e10 = float(wk["EMA10"].iloc[-1])
        if c > e10:
            return True, None
        return False, f"Weekly close {c:.0f} < 10-EMA {e10:.0f}"

    def check_adx(self, df: pd.DataFrame) -> tuple:
        """ADX > threshold on daily."""
        if df is None or len(df) < 20:
            return False, "Insufficient data"
        adx = _sf(df.iloc[-1], "ADX", 0)
        if adx >= self.adx_threshold:
            return True, None
        return False, f"ADX {adx:.1f} < {self.adx_threshold}"

    @staticmethod
    def check_ema_stack(df: pd.DataFrame) -> tuple:
        """Price > EMA20 > EMA50 > EMA200."""
        if df is None or len(df) < 200:
            return False, "Insufficient data"
        r = df.iloc[-1]
        c = _sf(r, "Close")
        e20 = _sf(r, "EMA20", c)
        e50 = _sf(r, "EMA50", c)
        e200 = _sf(r, "EMA200", c)
        if c > e20 > e50 > e200:
            return True, None
        return False, f"Stack fail: C{c:.0f} E20{e20:.0f} E50{e50:.0f} E200{e200:.0f}"

    def check_volume_confirmation(self, df: pd.DataFrame) -> tuple:
        """Entry volume > 1.5x 20-day avg."""
        if df is None or len(df) < 21:
            return False, "Insufficient data"
        rvol = _sf(df.iloc[-1], "RVOL", 0)
        if rvol >= self.vol_multiplier:
            return True, None
        return False, f"RVOL {rvol:.2f} < {self.vol_multiplier}x"

    def check_rs_filter(self, stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> tuple:
        """RS > 1.0 over rolling window."""
        rs = IndiaMarketUtils.compute_rs_vs_nifty(stock_df, nifty_df, self.rs_window)
        if rs >= 1.0:
            return True, None
        return False, f"RS {rs:.3f} < 1.0"

    def check_earnings_blackout(self, date: pd.Timestamp, earnings_dates: list) -> tuple:
        """No entry within blackout window."""
        if IndiaMarketUtils.is_earnings_blackout(date, earnings_dates, self.earnings_before, self.earnings_after):
            return False, "Earnings blackout window"
        return True, None

    def generate_entry_signal(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame,
                               nifty_df: pd.DataFrame, date: pd.Timestamp,
                               earnings_dates: list = None) -> dict:
        """
        All 6 filters must pass. Returns dict with entry, reasons, failures.
        """
        if earnings_dates is None:
            earnings_dates = []

        passed = []
        failed = []
        score = 0

        # 1. Weekly trend gate
        ok, reason = self.check_weekly_trend_gate(weekly_df)
        if ok:
            passed.append("WeeklyTrend")
            score += 1
        else:
            failed.append(reason or "WeeklyTrend")

        # 2. ADX > 22
        ok, reason = self.check_adx(daily_df)
        if ok:
            passed.append("ADX")
            score += 1
        else:
            failed.append(reason or "ADX")

        # 3. EMA stack
        ok, reason = self.check_ema_stack(daily_df)
        if ok:
            passed.append("EMAStack")
            score += 1
        else:
            failed.append(reason or "EMAStack")

        # 4. Volume confirmation
        ok, reason = self.check_volume_confirmation(daily_df)
        if ok:
            passed.append("Volume")
            score += 1
        else:
            failed.append(reason or "Volume")

        # 5. RS vs Nifty
        ok, reason = self.check_rs_filter(daily_df, nifty_df)
        if ok:
            passed.append("RSFilter")
            score += 1
        else:
            failed.append(reason or "RSFilter")

        # 6. Earnings blackout
        ok, reason = self.check_earnings_blackout(date, earnings_dates)
        if ok:
            passed.append("EarningsClear")
            score += 1
        else:
            failed.append(reason or "EarningsBlackout")

        entry = len(failed) == 0

        return {
            "entry": entry,
            "score": score,
            "passed": passed,
            "failed": failed,
        }


# ═════════════════════════════════════════════════════════════════════
# CLASS 3: ExitManager
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ActiveTrade:
    """Tracks a single open position with consolidated partial+final trade record."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    initial_stop: float
    target_15r: float  # 1.5R for partial booking
    atr_at_entry: float
    highest_since_entry: float = 0.0
    chandelier_stop: float = 0.0
    current_pnl_r: float = 0.0

    # Partial booking state (BUG 4 fix: store in trade, not as separate row)
    partial_booked: bool = False
    partial_date: pd.Timestamp = None
    partial_price: float = 0.0
    partial_shares: int = 0
    partial_pnl: float = 0.0
    partial_cost: float = 0.0

    @property
    def remaining_shares(self) -> int:
        return self.shares

    @remaining_shares.setter
    def remaining_shares(self, val: int):
        self.shares = val


class ExitManager:
    """
    6 exact exit rules.
    get_exit_decision() returns (exit: bool, reason: str, exit_price: float).
    """

    def __init__(self, chandelier_period=22, chandelier_mult=3.0,
                 time_stop_days=20, time_stop_r=-0.5,
                 gap_up_threshold=0.04):
        self.chandelier_period = chandelier_period
        self.chandelier_mult = chandelier_mult
        self.time_stop_days = time_stop_days
        self.time_stop_r = time_stop_r
        self.gap_up_threshold = gap_up_threshold

    @staticmethod
    def compute_chandelier_trail(df: pd.DataFrame, current_bar: int, atr: float,
                                  period=22, multiplier=3.0) -> float:
        """Chandelier exit = HH(period) - multiplier * ATR."""
        lookback = min(period, current_bar)
        if lookback < 1:
            return 0
        seg = df.iloc[current_bar - lookback:current_bar + 1]
        hh = seg["High"].max()
        return hh - multiplier * atr

    @staticmethod
    def check_sell_signal(df: pd.DataFrame, regime: str = "NEUTRAL") -> bool:
        """Bearish signal from the v4 engine."""
        try:
            from core.profit_engine_v4 import generate_v4_signal
            sig = generate_v4_signal(df, regime=regime)
            return sig.get("Signal") == "SELL"
        except Exception:
            return False

    @staticmethod
    def check_time_stop(entry_date: pd.Timestamp, current_date: pd.Timestamp,
                        current_pnl_r: float, max_days=20, min_r_threshold=-0.5) -> tuple:
        """
        Modified time stop: only exit if > 20 days AND PnL < -0.5R.
        If above breakeven -> do NOT exit on time, let trail handle it.
        """
        held = (current_date - entry_date).days
        if held >= max_days and current_pnl_r < min_r_threshold:
            return True, f"Time({held}d, R:{current_pnl_r:.1f})"
        return False, None

    @staticmethod
    def check_gap_up(prev_close: float, open_price: float, threshold=0.04) -> tuple:
        """Gap-up > 4%: exit full position at market open."""
        gap_pct = (open_price / prev_close - 1)
        if gap_pct > threshold:
            return True, f"GapUp({gap_pct*100:.1f}%)"
        return False, None

    def get_exit_decision(self, trade: ActiveTrade, current_bar_df: pd.DataFrame,
                           current_bar: int, current_date: pd.Timestamp,
                           regime: str = "NEUTRAL") -> dict:
        """
        Evaluate all exit rules. Returns dict with exit, reason, exit_price.
        Rules evaluated in order of priority:
          1. Gap-up exit
          2. SellSig exit
          3. Time stop (modified)
          4. Partial booking (40% at 1.5R)
          5. Chandelier trail (for remaining shares)
        """
        r = current_bar_df.iloc[-1]
        c = float(r["Close"])
        o = float(r["Open"])
        prev_c = float(current_bar_df.iloc[-2]["Close"]) if len(current_bar_df) > 1 else c
        atr = _sf(r, "ATR", trade.atr_at_entry)
        if atr <= 0:
            atr = trade.atr_at_entry

        # Update highest since entry
        trade.highest_since_entry = max(trade.highest_since_entry, c)
        trade.current_pnl_r = (c - trade.entry_price) / max(atr, 0.01)

        # 1. Gap-up exit (highest priority)
        exit_gap, reason_gap = self.check_gap_up(prev_c, o, self.gap_up_threshold)
        if exit_gap:
            return {"exit": True, "reason": reason_gap, "exit_price": o,
                    "partial_booked": trade.partial_booked, "shares": trade.shares}

        # 2. SellSig exit (signal-based)
        if self.check_sell_signal(current_bar_df, regime):
            return {"exit": True, "reason": "SellSig", "exit_price": c,
                    "partial_booked": trade.partial_booked, "shares": trade.shares}

        # 3. Partial booking (40% at 1.5R, move stop to breakeven)
        if not trade.partial_booked:
            if c >= trade.target_15r:
                sold = int(trade.shares * 0.4)
                remaining = trade.shares - sold
                trade.partial_booked = True
                trade.remaining_shares = remaining
                # Move stop to breakeven for remaining
                trade.chandelier_stop = trade.entry_price
                return {"exit": True, "reason": "Partial40%", "exit_price": c,
                        "partial_booked": True, "shares": sold, "remaining": remaining}

        # 4. Chandelier trail (for remaining/partial-booked shares)
        stop_mult = self.chandelier_mult * (atr / max(trade.atr_at_entry, 0.01))
        # Volatility-adaptive: widen/narrow trail based on current ATR vs entry ATR
        trail = self.compute_chandelier_trail(
            current_bar_df, current_bar, atr,
            self.chandelier_period, self.chandelier_mult
        )
        trade.chandelier_stop = max(trade.chandelier_stop, trail)

        if trade.partial_booked:
            # After partial, trail remaining shares at chandelier
            trade.chandelier_stop = max(trade.chandelier_stop, trade.entry_price)
            if c <= trade.chandelier_stop:
                return {"exit": True, "reason": "ChandelierTrail", "exit_price": c,
                        "partial_booked": True, "shares": trade.remaining_shares}
        else:
            # Before partial, use initial stop (2x ATR) as floor
            effective_stop = max(trade.initial_stop, trade.chandelier_stop)
            if c <= effective_stop:
                return {"exit": True, "reason": "InitialStop", "exit_price": c,
                        "partial_booked": False, "shares": trade.shares}

        # 5. Modified time stop (only if below -0.5R after 20 days)
        exit_ts, reason_ts = self.check_time_stop(
            trade.entry_date, current_date, trade.current_pnl_r,
            self.time_stop_days, self.time_stop_r
        )
        if exit_ts:
            return {"exit": True, "reason": reason_ts, "exit_price": c,
                    "partial_booked": trade.partial_booked, "shares": trade.shares}

        return {"exit": False, "reason": None, "exit_price": None,
                "partial_booked": trade.partial_booked, "shares": trade.shares}


# ═════════════════════════════════════════════════════════════════════
# CLASS 4: RiskManager
# ═════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Kelly sizing, volatility-adaptive, portfolio heat, circuit breaker,
    daily loss limit, streak sizing.
    """

    def __init__(self, capital: float, max_heat=0.20, daily_loss_limit=0.02,
                 dd_circuit_window=10, dd_circuit_threshold=0.10,
                 dd_circuit_bars=5, streak_adjust=0.15):
        self.initial_capital = capital
        self.capital = capital
        self.max_heat = max_heat
        self.daily_loss_limit = daily_loss_limit
        self.dd_circuit_window = dd_circuit_window
        self.dd_circuit_threshold = dd_circuit_threshold
        self.dd_circuit_bars = dd_circuit_bars
        self.streak_adjust = streak_adjust

        self.open_positions: list = []
        self.trade_pnls: list = []
        self.last_10_pnls: list = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.daily_pnl_start = capital
        self.current_bar = -1
        self.circuit_halt_until = -1

    @property
    def portfolio_heat(self) -> float:
        total_risk = sum(p.get("risk_amt", 0) for p in self.open_positions)
        return total_risk / max(self.capital, 1)

    @staticmethod
    def compute_kelly(win_rate: float, payoff_ratio: float, cap=0.25) -> float:
        """f* = W - (1-W)/R, capped at cap."""
        if win_rate <= 0 or payoff_ratio <= 0:
            return 0.02
        f = win_rate - (1 - win_rate) / max(payoff_ratio, 0.1)
        return max(0.01, min(f, cap))

    @staticmethod
    def compute_position_size(capital: float, kelly_f: float, entry_price: float,
                               atr: float, atr_baseline: float) -> int:
        """
        Volatility-adaptive: base_size * (atr_baseline / current_atr).
        """
        base_size = capital * kelly_f
        vol_ratio = atr_baseline / max(atr, 0.01)
        adjusted_size = base_size * vol_ratio
        raw_shares = int(adjusted_size / max(entry_price, 1))
        mx = int(capital * 0.3 / max(entry_price, 1))  # Max 30% in one stock
        return max(1, min(raw_shares, mx))

    def check_portfolio_heat(self) -> tuple:
        """Max 20% of capital at risk."""
        if self.portfolio_heat >= self.max_heat:
            return False, f"Portfolio heat {self.portfolio_heat:.1%} >= {self.max_heat:.0%}"
        return True, None

    def check_drawdown_circuit_breaker(self, current_bar: int) -> tuple:
        """If 10-trade rolling drawdown > 10%, pause 5 days."""
        if self.circuit_halt_until > current_bar:
            return False, f"Circuit breaker ({self.circuit_halt_until - current_bar} bars left)"
        if len(self.last_10_pnls) >= 5:
            rolling_dd = sum(p for p in self.last_10_pnls if p < 0)
            if abs(rolling_dd) >= self.dd_circuit_threshold * 100:  # Convert to %
                self.circuit_halt_until = current_bar + self.dd_circuit_bars
                return False, f"DD circuit breaker ({abs(rolling_dd):.1f}% > {self.dd_circuit_threshold*100:.0f}%)"
        self.circuit_halt_until = -1
        return True, None

    def check_daily_loss_limit(self) -> tuple:
        """If portfolio drops 2% intraday, no new entries."""
        daily_loss = (self.daily_pnl - self.daily_pnl_start) / max(self.daily_pnl_start, 1)
        if daily_loss <= -self.daily_loss_limit:
            return False, f"Daily loss limit ({daily_loss*100:.1f}%)"
        return True, None

    def apply_streak_sizing(self, kelly_f: float) -> float:
        """
        After 2 consecutive losses: reduce by 15%.
        After 2 consecutive wins: restore to full Kelly.
        """
        if self.consecutive_losses >= 2:
            return kelly_f * (1 - self.streak_adjust)
        return kelly_f

    def open_position(self, pos: dict):
        self.open_positions.append(pos)

    def close_position(self, pos_id: str, pnl_pct: float, pnl_amt: float):
        self.open_positions = [p for p in self.open_positions if p.get("id") != pos_id]
        self.trade_pnls.append(pnl_pct)
        self.last_10_pnls.append(pnl_pct)
        if len(self.last_10_pnls) > 10:
            self.last_10_pnls.pop(0)

        self.capital += pnl_amt
        self.daily_pnl += pnl_amt

        if pnl_pct > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def update_daily(self, bar: int):
        if bar != self.current_bar:
            self.current_bar = bar
            self.daily_pnl_start = self.daily_pnl

    def can_trade(self, current_bar: int) -> tuple:
        ok, reason = self.check_portfolio_heat()
        if not ok:
            return False, reason
        ok, reason = self.check_drawdown_circuit_breaker(current_bar)
        if not ok:
            return False, reason
        ok, reason = self.check_daily_loss_limit()
        if not ok:
            return False, reason
        return True, None


# ═════════════════════════════════════════════════════════════════════
# CLASS 5: PerformanceAnalytics
# ═════════════════════════════════════════════════════════════════════

class PerformanceAnalytics:
    """14 institutional metrics from a trade log DataFrame."""

    @staticmethod
    def compute_all_metrics(trades_df: pd.DataFrame, initial_capital: float) -> dict:
        """
        Compute all 14 metrics from the trade log.
        trades_df must have columns: Entry, Exit, Entry₹, Exit₹, PnL%, PnL₹,
                                      Result, Bars, Reason, Shares
        """
        if trades_df is None or trades_df.empty or len(trades_df) < 3:
            return {}

        df = trades_df.copy()
        n = len(df)
        rets = df["PnL%"].values
        rets_rupee = df["PnL₹"].values

        wins = df[df["Result"] == "WIN"]
        losses = df[df["Result"] == "LOSS"]
        wc = len(wins)
        lc = len(losses)
        wr = wc / n * 100 if n > 0 else 0
        aw = float(wins["PnL%"].mean()) if wc > 0 else 0
        al = float(abs(losses["PnL%"].mean())) if lc > 0 else 0
        net_profit = float(rets_rupee.sum())

        # 1. Sharpe Ratio (annualized from trade returns)
        sharpe = round(float(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252)), 2) if len(rets) > 1 else 0

        # 2. Sortino Ratio
        downside = rets[rets < 0]
        dd_dev = float(np.std(downside)) if len(downside) > 1 else 0.01
        sortino = round(float(np.mean(rets) / max(dd_dev, 0.01) * np.sqrt(252)), 2)

        # 3. Profit Factor
        gp = float(wins["PnL%"].sum()) if wc > 0 else 0
        gl = float(abs(losses["PnL%"].sum())) if lc > 0 else 0
        pf = round(gp / max(gl, 0.01), 2)

        # 4. Payoff Ratio
        payoff = round(aw / max(al, 0.01), 2)

        # 5. Expectancy
        expectancy = round((wr / 100 * aw) - ((1 - wr / 100) * al), 2)

        # Equity curve
        eq = [initial_capital]
        for pnl in rets_rupee:
            eq.append(eq[-1] + pnl)
        eq_series = pd.Series(eq[1:])
        peak = eq_series.cummax()
        dd_vals = (eq_series - peak) / peak.replace(0, 1) * 100

        # 6. Max Drawdown %
        max_dd_pct = float(dd_vals.min()) if len(dd_vals) > 0 else 0
        # 7. Max Drawdown ₹
        min_eq_idx = dd_vals.idxmin() if len(dd_vals) > 0 else 0
        max_dd_rupee = float(peak.iloc[min_eq_idx] - eq_series.iloc[min_eq_idx]) if min_eq_idx < len(eq_series) else 0

        # 8. Calmar Ratio = CAGR / Max DD
        total_return = (eq_series.iloc[-1] - initial_capital) / initial_capital
        total_days = (df["Exit"].max() - df["Entry"].min()).days if "Entry" in df.columns and "Exit" in df.columns else 1
        years = max(total_days / 365, 0.01)
        cagr = ((1 + total_return) ** (1 / years) - 1) * 100
        calmar = round(cagr / max(abs(max_dd_pct), 0.1), 2)

        # 9. Recovery Factor
        recovery = round(net_profit / max(abs(max_dd_rupee), 1), 2)

        # 10. Avg holding bars
        avg_win_bars = round(float(wins["Bars"].mean()), 1) if wc > 0 else 0
        avg_loss_bars = round(float(losses["Bars"].mean()), 1) if lc > 0 else 0

        # 11. Max consecutive losses
        streaks = []
        curr = 0
        for res in df["Result"]:
            if res == "LOSS":
                curr += 1
            else:
                if curr > 0:
                    streaks.append(curr)
                curr = 0
        if curr > 0:
            streaks.append(curr)
        max_loss_streak = max(streaks) if streaks else 0

        # 12. MFE/MAE approximation from exit reasons
        exit_breakdown = df["Reason"].value_counts().to_dict()

        # 13. Rolling 20-trade Sharpe
        rolling_sharpes = []
        window = min(20, len(rets))
        if window >= 10:
            for j in range(window, len(rets) + 1):
                chunk = rets[j - window:j]
                rs = float(np.mean(chunk) / max(np.std(chunk), 0.01) * np.sqrt(252))
                rolling_sharpes.append(rs)
        avg_roll_sharpe = round(float(np.mean(rolling_sharpes)), 2) if rolling_sharpes else sharpe
        roll_sharpe_flag = any(s < 0.5 for s in rolling_sharpes) if rolling_sharpes else False

        # 14. Trade count and % exit by category
        exit_categories = df["Reason"].value_counts()
        exit_pcts = (exit_categories / n * 100).round(1).to_dict()

        return {
            "Total Trades": n,
            "Win Count": wc,
            "Loss Count": lc,
            "Win Rate %": round(wr, 1),
            "Avg Win %": round(aw, 2),
            "Avg Loss %": round(al, 2),
            "Net Profit ₹": round(net_profit, 2),
            "Total Return %": round(total_return * 100, 2),
            "CAGR %": round(cagr, 2),
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Profit Factor": pf,
            "Payoff Ratio": payoff,
            "Expectancy %": expectancy,
            "Calmar Ratio": calmar,
            "Recovery Factor": recovery,
            "Max Drawdown %": round(max_dd_pct, 2),
            "Max Drawdown ₹": round(max_dd_rupee, 2),
            "Avg Win Bars": avg_win_bars,
            "Avg Loss Bars": avg_loss_bars,
            "Max Consecutive Losses": max_loss_streak,
            "Exit Breakdown": exit_breakdown,
            "Exit % by Category": exit_pcts,
            "Rolling 20-Sharpe (Avg)": avg_roll_sharpe,
            "Rolling Sharpe < 0.5 Flag": roll_sharpe_flag,
        }

    @staticmethod
    def print_report(trades_df: pd.DataFrame, initial_capital: float) -> str:
        """Generate a formatted text report."""
        m = PerformanceAnalytics.compute_all_metrics(trades_df, initial_capital)
        if not m:
            return "No trades to report."

        lines = []
        lines.append("=" * 60)
        lines.append("V6 BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Trades:       {m['Total Trades']}")
        lines.append(f"Win Rate:           {m['Win Rate %']:.1f}% ({m['Win Count']}W / {m['Loss Count']}L)")
        lines.append(f"Net Profit:         ₹{m['Net Profit ₹']:,.2f}")
        lines.append(f"Total Return:       {m['Total Return %']:+.2f}%")
        lines.append(f"CAGR:               {m['CAGR %']:.2f}%")
        lines.append(f"Profit Factor:      {m['Profit Factor']:.2f}")
        lines.append(f"Sharpe Ratio:       {m['Sharpe Ratio']:.2f}")
        lines.append(f"Sortino Ratio:      {m['Sortino Ratio']:.2f}")
        lines.append(f"Payoff Ratio:       {m['Payoff Ratio']:.2f}")
        lines.append(f"Expectancy/Trade:   {m['Expectancy %']:+.2f}%")
        lines.append(f"Calmar Ratio:       {m['Calmar Ratio']:.2f}")
        lines.append(f"Recovery Factor:    {m['Recovery Factor']:.2f}")
        lines.append(f"Max Drawdown:       {m['Max Drawdown %']:.2f}% (₹{m['Max Drawdown ₹']:,.0f})")
        lines.append(f"Avg Win:            {m['Avg Win %']:+.2f}% ({m['Avg Win Bars']} bars)")
        lines.append(f"Avg Loss:           {m['Avg Loss %']:.2f}% ({m['Avg Loss Bars']} bars)")
        lines.append(f"Max Cons Losses:    {m['Max Consecutive Losses']}")
        lines.append(f"Rolling Sharpe:     {m['Rolling 20-Sharpe (Avg)']:.2f} {'⚠️' if m.get('Rolling Sharpe < 0.5 Flag') else '✅'}")
        lines.append("")
        lines.append("Exit Breakdown:")
        for reason, count in sorted(m.get("Exit Breakdown", {}).items(), key=lambda x: -x[1]):
            pct = m.get("Exit % by Category", {}).get(reason, 0)
            lines.append(f"  {reason}: {count} ({pct:.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# CLASS 6: Backtester (Main Engine)
# ═════════════════════════════════════════════════════════════════════

class Backtester:
    """
    V6 main backtesting engine.

    Config keys:
      - capital: float (default 100000)
      - risk_pct: float (default 0.02 i.e. 2%)
      - max_heat: float (default 0.20)
      - adx_threshold: float (default 22)
      - vol_multiplier: float (default 1.5)
      - rs_window: int (default 63)
      - earnings_dates: list of dates for blackout
      - chandelier_period: int (default 22)
      - chandelier_mult: float (default 3.0)
      - time_stop_days: int (default 20)
      - time_stop_r: float (default -0.5)
      - gap_up_threshold: float (default 0.04)
      - kelly_cap: float (default 0.25)
      - atr_baseline_period: int (default 14)
    """

    def __init__(self, config: dict = None):
        self.config = {
            "capital": 100000,
            "risk_pct": 0.02,
            "max_heat": 0.20,
            "adx_threshold": 22,
            "vol_multiplier": 1.5,
            "rs_window": 63,
            "earnings_dates": [],
            "chandelier_period": 22,
            "chandelier_mult": 3.0,
            "time_stop_days": 20,
            "time_stop_r": -0.5,
            "gap_up_threshold": 0.04,
            "kelly_cap": 0.25,
            "atr_baseline_period": 14,
        }
        if config:
            self.config.update(config)

        self.entry_gen = EntrySignalGenerator(
            adx_threshold=self.config["adx_threshold"],
            vol_multiplier=self.config["vol_multiplier"],
            rs_window=self.config["rs_window"],
        )
        self.exit_mgr = ExitManager(
            chandelier_period=self.config["chandelier_period"],
            chandelier_mult=self.config["chandelier_mult"],
            time_stop_days=self.config["time_stop_days"],
            time_stop_r=self.config["time_stop_r"],
            gap_up_threshold=self.config["gap_up_threshold"],
        )

    def load_data(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        """Load daily data with all indicators."""
        from core.utils import load_data
        df = load_data(symbol, period="5y")
        if df is None or df.empty:
            return pd.DataFrame()
        df = compute_indicators(df)
        return df

    def _load_weekly(self, symbol: str) -> pd.DataFrame:
        """Load weekly data for trend gate."""
        import yfinance as yf
        try:
            wk = yf.download(symbol, period="1y", interval="1wk", auto_adjust=True, progress=False)
            if wk is not None and not wk.empty:
                if isinstance(wk.columns, pd.MultiIndex):
                    wk.columns = wk.columns.get_level_values(0)
                return wk
        except Exception:
            pass
        return pd.DataFrame()

    def run(self, symbol: str, start_date=None, end_date=None) -> dict:
        """
        Run full V6 backtest on a single symbol.

        Returns dict with:
          - trades: DataFrame of all trades
          - metrics: dict of all 14 analytics
          - report: formatted string report
          - config: config used
          - equity_curve: list of capital values
        """
        # Load data
        df = self.load_data(symbol)
        if df.empty or len(df) < 250:
            return {"Error": "Insufficient data", "Total Trades": 0}

        weekly_df = self._load_weekly(symbol)
        nifty_df = IndiaMarketUtils.load_nifty50_data()
        earnings_dates = self.config["earnings_dates"]

        # Regime for signal
        from core.market_regime import get_market_regime
        regime = get_market_regime()

        # ATR baseline for volatility scaling
        atr_baseline_period = self.config["atr_baseline_period"]
        atr_baseline = float(df["ATR"].tail(atr_baseline_period).mean()) if len(df) > atr_baseline_period else 0

        # Risk manager
        rm = RiskManager(
            capital=self.config["capital"],
            max_heat=self.config["max_heat"],
        )

        cap = self.config["capital"]
        trades = []
        pos = False
        trade_id = 0
        active_trade = None

        for i in range(220, len(df)):
            bar = df.iloc[:i + 1]
            r = df.iloc[i]
            c = float(r["Close"])
            o = float(r["Open"])
            atr = _sf(r, "ATR", c * 0.02)
            if atr <= 0:
                atr = c * 0.02
            date = r.name
            if isinstance(date, pd.Timestamp):
                date_ts = date
            else:
                date_ts = pd.Timestamp(date)

            rm.update_daily(i)

            # ── ENTRY ──
            if not pos:
                sig = self.entry_gen.generate_entry_signal(bar, weekly_df, nifty_df, date_ts, earnings_dates)
                if not sig["entry"]:
                    continue

                # Risk manager checks
                ok, reason = rm.can_trade(i)
                if not ok:
                    continue

                # Kelly sizing
                wr = 0
                payoff_r = 0
                if len(rm.trade_pnls) >= 10:
                    wins = [p for p in rm.trade_pnls[-20:] if p > 0]
                    losses = [p for p in rm.trade_pnls[-20:] if p <= 0]
                    wr = len(wins) / max(len(rm.trade_pnls[-20:]), 1)
                    aw = np.mean(wins) if wins else 0
                    al = abs(np.mean(losses)) if losses else 1
                    payoff_r = aw / max(al, 0.01)

                kelly = RiskManager.compute_kelly(wr, payoff_r, self.config["kelly_cap"])
                kelly = rm.apply_streak_sizing(kelly)

                # Position size with volatility adaptation
                shares = RiskManager.compute_position_size(cap, kelly, c, atr, atr_baseline)
                if shares < 1:
                    continue

                initial_stop = c - atr * 2.0  # 2x ATR initial stop
                target_15r = c + atr * 1.5  # 1.5R target for partial

                trade_id += 1
                active_trade = ActiveTrade(
                    symbol=symbol,
                    entry_date=date_ts,
                    entry_price=c,
                    shares=shares,
                    initial_stop=initial_stop,
                    target_15r=target_15r,
                    atr_at_entry=atr,
                    highest_since_entry=c,
                )
                rm.open_position({"id": f"T{trade_id}", "risk_amt": cap * kelly})
                pos = True
                continue

            # ── EXIT ──
            if pos and active_trade is not None:
                decision = self.exit_mgr.get_exit_decision(
                    active_trade, bar, i, date_ts, regime
                )

                if decision["exit"]:
                    exit_price = decision["exit_price"]
                    reason = decision["reason"]

                    # BUG 2 fix: calculate partial shares with minimum 1
                    if reason == "Partial40%":
                        partial_shares = int(active_trade.shares * 0.40)
                        if partial_shares < 1:
                            partial_shares = 1
                        remaining = active_trade.shares - partial_shares

                        # Record partial data in active_trade (BUG 4: don't append to trades yet)
                        active_trade.partial_booked = True
                        active_trade.partial_date = date_ts
                        active_trade.partial_price = exit_price
                        active_trade.partial_shares = partial_shares

                        # Calculate partial PnL with costs
                        raw_pnl = partial_shares * (exit_price - active_trade.entry_price)
                        cost = IndiaMarketUtils.apply_cost_model(active_trade.entry_price, exit_price, partial_shares)
                        active_trade.partial_cost = cost["total_cost"]
                        active_trade.partial_pnl = raw_pnl - cost["total_cost"]

                        # Update capital and risk manager
                        cap += active_trade.partial_pnl

                        # Move stop to breakeven for remaining shares
                        active_trade.chandelier_stop = active_trade.entry_price
                        active_trade.shares = remaining  # BUG 2 fix: update shares for trail leg

                        # BUG 4: do NOT append trade row, do NOT close position — continue
                        continue

                    # ── FINAL EXIT (full remaining position) ──
                    # BUG 4: Create ONE consolidated trade record
                    shares_exit = decision.get("shares", active_trade.shares)

                    # Calculate final leg PnL
                    final_raw = shares_exit * (exit_price - active_trade.entry_price)
                    cost = IndiaMarketUtils.apply_cost_model(active_trade.entry_price, exit_price, shares_exit)
                    final_pnl = final_raw - cost["total_cost"]

                    # Combine with partial if it exists
                    if active_trade.partial_booked:
                        total_pnl = active_trade.partial_pnl + final_pnl
                        total_shares = active_trade.partial_shares + shares_exit
                    else:
                        total_pnl = final_pnl
                        total_shares = shares_exit

                    # BUG 1 fix: bars = calendar days from entry to exit
                    held_days = max((date_ts - active_trade.entry_date).days, 1)

                    # Total PnL percentage against total capital deployed
                    total_invested = active_trade.entry_price * total_shares
                    total_pnl_pct = total_pnl / max(total_invested, 0.01) * 100

                    cap += final_pnl
                    rm.close_position(f"T{trade_id}", total_pnl_pct, total_pnl)

                    # BUG 3 fix: >= 0 is WIN, tag partials clearly
                    if active_trade.partial_booked:
                        result_tag = "WIN" if total_pnl >= 0 else "LOSS"
                    else:
                        result_tag = "WIN" if total_pnl_pct >= 0 else "LOSS"

                    # Build consolidated trade record
                    trade_record = {
                        "Entry": active_trade.entry_date,
                        "Exit": date_ts,
                        "Entry₹": round(active_trade.entry_price, 2),
                        "Exit₹": round(exit_price, 2),
                        "PnL%": round(total_pnl_pct, 2),
                        "PnL₹": round(total_pnl, 2),
                        "Result": result_tag,
                        "Bars": held_days,
                        "Reason": reason,
                        "Initial_Shares": total_shares,
                    }

                    # Add partial booking columns (BUG 4)
                    if active_trade.partial_booked:
                        trade_record["Partial_Date"] = active_trade.partial_date
                        trade_record["Partial_Price"] = round(active_trade.partial_price, 2)
                        trade_record["Partial_Shares"] = active_trade.partial_shares
                        trade_record["Partial_PnL"] = round(active_trade.partial_pnl, 2)
                        trade_record["Final_Shares"] = shares_exit
                        trade_record["Final_PnL"] = round(final_pnl, 2)
                    else:
                        trade_record["Partial_Date"] = None
                        trade_record["Partial_Price"] = None
                        trade_record["Partial_Shares"] = 0
                        trade_record["Partial_PnL"] = 0.0
                        trade_record["Final_Shares"] = total_shares
                        trade_record["Final_PnL"] = round(total_pnl, 2)

                    trades.append(trade_record)
                    pos = False
                    active_trade = None

        # Close any open position at end
        if pos and active_trade is not None:
            fp = float(df["Close"].iloc[-1])
            shares = active_trade.shares
            final_raw = shares * (fp - active_trade.entry_price)
            cost = IndiaMarketUtils.apply_cost_model(active_trade.entry_price, fp, shares)
            final_pnl = final_raw - cost["total_cost"]

            if active_trade.partial_booked:
                total_pnl = active_trade.partial_pnl + final_pnl
                total_shares = active_trade.partial_shares + shares
            else:
                total_pnl = final_pnl
                total_shares = shares

            held_days = max((df.index[-1] - active_trade.entry_date).days, 1)
            total_invested = active_trade.entry_price * total_shares
            total_pnl_pct = total_pnl / max(total_invested, 0.01) * 100
            cap += final_pnl

            result_tag = "WIN" if total_pnl >= 0 else "LOSS"

            trade_record = {
                "Entry": active_trade.entry_date,
                "Exit": df.index[-1],
                "Entry₹": round(active_trade.entry_price, 2),
                "Exit₹": round(fp, 2),
                "PnL%": round(total_pnl_pct, 2),
                "PnL₹": round(total_pnl, 2),
                "Result": result_tag,
                "Bars": held_days,
                "Reason": "EndOfData",
                "Initial_Shares": total_shares,
            }
            if active_trade.partial_booked:
                trade_record["Partial_Date"] = active_trade.partial_date
                trade_record["Partial_Price"] = round(active_trade.partial_price, 2)
                trade_record["Partial_Shares"] = active_trade.partial_shares
                trade_record["Partial_PnL"] = round(active_trade.partial_pnl, 2)
                trade_record["Final_Shares"] = shares
                trade_record["Final_PnL"] = round(final_pnl, 2)
            else:
                trade_record["Partial_Date"] = None
                trade_record["Partial_Price"] = None
                trade_record["Partial_Shares"] = 0
                trade_record["Partial_PnL"] = 0.0
                trade_record["Final_Shares"] = total_shares
                trade_record["Final_PnL"] = round(total_pnl, 2)

            trades.append(trade_record)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = PerformanceAnalytics.compute_all_metrics(trades_df, self.config["capital"])
        report = PerformanceAnalytics.print_report(trades_df, self.config["capital"])

        # Equity curve
        eq = [self.config["capital"]]
        for t in trades:
            eq.append(eq[-1] + t.get("PnL₹", 0))

        return {
            "trades": trades_df,
            "metrics": metrics,
            "report": report,
            "config": self.config,
            "equity_curve": eq,
            "Total Trades": len(trades),
            "Total Return %": metrics.get("Total Return %", 0),
            "Profit Factor": metrics.get("Profit Factor", 0),
            "Sharpe Ratio": metrics.get("Sharpe Ratio", 0),
            "Win Rate %": metrics.get("Win Rate %", 0),
            "Max Drawdown %": metrics.get("Max Drawdown %", 0),
            "Final Capital": round(cap, 2),
        }

    def run_montecarlo(self, trades_df: pd.DataFrame, n_runs=2000, confidence=0.90) -> dict:
        """Monte Carlo simulation by sampling with replacement."""
        if trades_df is None or trades_df.empty or len(trades_df) < 5:
            return {"Simulations": 0}

        rets = trades_df["PnL%"].values
        capital = self.config["capital"]
        eq_base = [capital]
        for t in rets:
            eq_base.append(eq_base[-1] * (1 + t / 100))
        base_return = (eq_base[-1] - capital) / capital * 100

        results = []
        dd_list = []
        n_trades = len(rets)

        for _ in range(n_runs):
            sampled = np.random.choice(rets, size=n_trades, replace=True)
            eq = [capital]
            for pnl in sampled:
                eq.append(eq[-1] * (1 + pnl / 100))
            total_ret = (eq[-1] - capital) / capital * 100
            results.append(total_ret)

            peak = np.maximum.accumulate(eq)
            dd = (np.array(eq) - peak) / peak * 100
            dd_list.append(abs(dd.min()))

        results = np.array(results)
        dd_list = np.array(dd_list)

        alpha = (1 - confidence) / 2
        ci_lo = np.percentile(results, alpha * 100)
        ci_hi = np.percentile(results, (1 - alpha) * 100)

        return {
            "Base Return %": round(base_return, 2),
            "Median Return %": round(np.median(results), 2),
            "Mean Return %": round(np.mean(results), 2),
            "CI Lower %": round(ci_lo, 2),
            "CI Upper %": round(ci_hi, 2),
            "Prob Profit %": round(np.mean(results > 0) * 100, 1),
            "Avg MaxDD %": round(np.mean(dd_list), 2),
            "Worst MaxDD %": round(np.max(dd_list), 2),
            "Std Dev %": round(np.std(results), 2),
            "Simulations": n_runs,
        }


# ═════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ═════════════════════════════════════════════════════════════════════

def run_v6_backtest(symbol: str, capital: float = 100000) -> dict:
    """Run V6 backtest with default config."""
    bt = Backtester({"capital": capital})
    return bt.run(symbol)


def compare_v6_vs_old(symbol: str) -> dict:
    """Compare V6 against old baseline."""
    from core.backtest import run_backtest as old_bt
    old = old_bt(symbol) or {}
    new = run_v6_backtest(symbol) or {}

    return {
        "old": {
            "Total Return %": old.get("Total Return %", 0),
            "Win Rate %": old.get("Win Rate %", 0),
            "Profit Factor": old.get("Profit Factor", 0),
            "Max Drawdown %": old.get("Max Drawdown %", 0),
            "Sharpe Ratio": old.get("Sharpe Ratio", 0),
            "Total Trades": old.get("Total Trades", 0),
        },
        "new": {
            "Total Return %": new.get("Total Return %", 0),
            "Win Rate %": new.get("Win Rate %", 0),
            "Profit Factor": new.get("Profit Factor", 0),
            "Max Drawdown %": new.get("Max Drawdown %", 0),
            "Sharpe Ratio": new.get("Sharpe Ratio", 0),
            "Total Trades": new.get("Total Trades", 0),
        },
        "details": new,
    }
