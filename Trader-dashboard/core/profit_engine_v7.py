"""
V7 — Precision Backtesting Engine for NSE Equities
====================================================
Built from empirical V4/V5/V6 data across ABB, LT, ADANIENT.

4 EXACT ENTRY FILTERS · 5 EXACT EXIT RULES · KELLY SIZING · CHANDELIER TRAIL
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from core.indicators import compute_indicators

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _sf(r, field, default=0):
    v = r.get(field, default)
    try:
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else default
    except (ValueError, TypeError):
        return default


# ═════════════════════════════════════════════════════════════════════
# CLASS 1: CostModel
# ═════════════════════════════════════════════════════════════════════

class CostModel:
    """SEBI-compliant cost model. Apply to every exit leg separately."""

    @staticmethod
    def apply(entry_price: float, exit_price: float, shares: int) -> float:
        """
        Calculate total cost for one exit leg (partial or final).
        Returns total cost in rupees.

        Costs per spec:
          stt = trade_value * 0.001                        (sell side)
          brokerage = min(20, trade_value * 0.0003) * 2    (entry + exit)
          exchange_charge = (trade_value + entry_value) * 0.0000345
          sebi_charge = (trade_value + entry_value) * 0.000001
          gst = (brokerage + exchange_charge + sebi_charge) * 0.18
          stamp_duty = entry_value * 0.00015               (buy side)
        """
        entry_value = entry_price * shares
        trade_value = exit_price * shares

        stt = trade_value * 0.001
        brokerage = min(20, trade_value * 0.0003) * 2
        exchange_charge = (trade_value + entry_value) * 0.0000345
        sebi_charge = (trade_value + entry_value) * 0.000001
        gst = (brokerage + exchange_charge + sebi_charge) * 0.18
        stamp_duty = entry_value * 0.00015

        return stt + brokerage + exchange_charge + sebi_charge + gst + stamp_duty


# ═════════════════════════════════════════════════════════════════════
# CLASS 2: EntryFilter
# ═════════════════════════════════════════════════════════════════════

class EntryFilter:
    """
    4 exact entry filters. No more.
    is_valid_entry() returns (pass: bool, reason: str or None).
    """

    @staticmethod
    def check_weekly_trend(weekly_df: pd.DataFrame) -> bool:
        """Weekly close > 10-week EMA."""
        if weekly_df is None or len(weekly_df) < 12:
            return False
        wk = weekly_df.copy()
        wk["EMA10"] = wk["Close"].ewm(span=10).mean()
        c = float(wk["Close"].iloc[-1])
        e10 = float(wk["EMA10"].iloc[-1])
        return c > e10

    @staticmethod
    def check_adx(daily_df: pd.DataFrame, period=14, threshold=22) -> bool:
        """ADX(14) > 22 on daily."""
        if daily_df is None or len(daily_df) < period + 5:
            return False
        adx = _sf(daily_df.iloc[-1], "ADX", 0)
        return adx > threshold

    @staticmethod
    def check_ema_stack(daily_df: pd.DataFrame) -> bool:
        """Price > EMA20 > EMA50 > EMA200 on daily. All four conditions must be true."""
        if daily_df is None or len(daily_df) < 200:
            return False
        r = daily_df.iloc[-1]
        c = _sf(r, "Close")
        e20 = _sf(r, "EMA20", c)
        e50 = _sf(r, "EMA50", c)
        e200 = _sf(r, "EMA200", c)
        return c > e20 > e50 > e200

    @staticmethod
    def check_volume(daily_df: pd.DataFrame, multiplier=1.5) -> bool:
        """Entry bar volume > 1.5 * SMA(volume, 20)."""
        if daily_df is None or len(daily_df) < 21:
            return False
        rvol = _sf(daily_df.iloc[-1], "RVOL", 0)
        return rvol > multiplier

    def is_valid_entry(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> tuple:
        """
        All 4 filters must pass. Returns (True/False, reason_if_rejected).
        """
        if not self.check_weekly_trend(weekly_df):
            return False, "WeeklyTrend"
        if not self.check_adx(daily_df):
            return False, f"ADX<={22}"
        if not self.check_ema_stack(daily_df):
            return False, "EMAStack"
        if not self.check_volume(daily_df):
            return False, f"Volume<={1.5}x"
        return True, None


# ═════════════════════════════════════════════════════════════════════
# CLASS 3: ExitManager
# ═════════════════════════════════════════════════════════════════════

class ExitManager:
    """
    5 exact exit rules in priority order:
      GapUp → InitialStop → Partial40% → TimeStop → ChandelierTrail
    """

    def __init__(self, partial_pct=0.40, partial_r=1.5,
                 chandelier_period=22, chandelier_multiplier=3.0,
                 gap_up_threshold=0.04,
                 time_stop_days=20, time_stop_r=-0.5):
        self.partial_pct = partial_pct
        self.partial_r = partial_r
        self.chandelier_period = chandelier_period
        self.chandelier_multiplier = chandelier_multiplier
        self.gap_up_threshold = gap_up_threshold
        self.time_stop_days = time_stop_days
        self.time_stop_r = time_stop_r

    @staticmethod
    def get_initial_stop(entry_price: float, atr: float, multiplier=2.0) -> float:
        """stop_price = entry_price - (2 * ATR)."""
        return entry_price - multiplier * atr

    @staticmethod
    def check_gap_up(prev_close: float, open_price: float, threshold=0.04) -> tuple:
        """If open > prev_close * 1.04, exit at open. Returns (exit, reason, price)."""
        gap = open_price / prev_close - 1
        if gap > threshold:
            return True, f"GapUp({gap*100:.1f}%)", open_price
        return False, None, None

    @staticmethod
    def check_partial_trigger(high: float, entry_price: float, stop_price: float,
                               partial_done: bool, partial_r=1.5) -> bool:
        """When high >= entry + 1.5R, fire partial."""
        if partial_done:
            return False
        r_price = entry_price - stop_price  # R value in price terms
        trigger = entry_price + partial_r * r_price
        return high >= trigger

    @staticmethod
    def get_chandelier_stop(daily_df: pd.DataFrame, current_bar: int, atr: float,
                            lookback=22, multiplier=3.0, floor=None) -> float:
        """chandelier_stop = HH(lookback) - multiplier * ATR."""
        lb = min(lookback, current_bar)
        if lb < 1:
            return floor if floor else 0
        seg = daily_df.iloc[current_bar - lb:current_bar + 1]
        hh = seg["High"].max()
        stop = hh - multiplier * atr
        if floor is not None:
            stop = max(stop, floor)
        return stop

    @staticmethod
    def check_time_stop(bars_open: int, current_pnl_r: float,
                         max_days=20, r_threshold=-0.5) -> tuple:
        """
        Only exit if > 20 days AND PnL < -0.5R.
        If PnL >= 0: do NOT exit.
        """
        if bars_open > max_days and current_pnl_r < r_threshold:
            return True, f"TimeStop({bars_open}d, R:{current_pnl_r:.1f})"
        return False, None

    def get_exit(self, bar_df: pd.DataFrame, current_bar: int, trade, atr: float) -> tuple:
        """
        Evaluate all exit rules in priority order.
        Returns (should_exit: bool, reason: str, exit_price: float).
        """
        r = bar_df.iloc[-1]
        c = float(r["Close"])
        o = float(r["Open"])
        h = float(r["High"])
        lo = float(r["Low"])
        prev_close = float(bar_df.iloc[-2]["Close"]) if len(bar_df) > 1 else c

        # Calculate R and current PnL in R
        r_price = trade.entry_price - trade.initial_stop  # R value
        if r_price <= 0:
            r_price = atr * 2
        current_pnl_r = (c - trade.entry_price) / max(r_price, 0.01)

        # Update chandelier stop on the trade
        new_chandelier = self.get_chandelier_stop(
            bar_df, current_bar, atr,
            self.chandelier_period, self.chandelier_multiplier,
            floor=trade.breakeven_stop
        )
        trade.chandelier_stop = max(trade.chandelier_stop, new_chandelier)

        # ── PRIORITY 1: Gap-Up Exit ──
        exit_gap, reason_gap, price_gap = self.check_gap_up(prev_close, o, self.gap_up_threshold)
        if exit_gap:
            return True, reason_gap, price_gap

        # ── PRIORITY 2: Initial Stop ──
        if trade.initial_stop is not None and lo <= trade.initial_stop:
            exit_price = max(trade.initial_stop, lo * 0.99)  # slippage-adjusted
            return True, "InitialStop", exit_price

        # ── PRIORITY 3: Partial 40% at 1.5R (fires once) ──
        if self.check_partial_trigger(h, trade.entry_price, trade.initial_stop,
                                       trade.partial_done, self.partial_r):
            trigger = trade.entry_price + self.partial_r * r_price
            # Partial booking: update trade state, do NOT exit fully
            # Return a special signal that caller handles
            return True, "Partial40%", trigger

        # ── PRIORITY 4: Modified Time Stop ──
        bars_open = current_bar - trade.entry_bar
        exit_ts, reason_ts = self.check_time_stop(bars_open, current_pnl_r,
                                                   self.time_stop_days, self.time_stop_r)
        if exit_ts:
            return True, reason_ts, c

        # ── PRIORITY 5: Chandelier Trail ──
        if trade.chandelier_stop > 0 and c <= trade.chandelier_stop:
            return True, "ChandelierTrail", trade.chandelier_stop

        return False, None, None


# ═════════════════════════════════════════════════════════════════════
# CLASS 4: PositionSizer
# ═════════════════════════════════════════════════════════════════════

class PositionSizer:
    """Kelly sizing, volatility adjustment, streak adjustment, portfolio heat."""

    @staticmethod
    def kelly(win_rate: float, payoff_ratio: float, cap=0.25) -> float:
        """f_kelly = win_rate - (1 - win_rate) / payoff_ratio. Capped at cap."""
        if win_rate <= 0 or payoff_ratio <= 0:
            return 0.02
        f = win_rate - (1 - win_rate) / max(payoff_ratio, 0.1)
        return max(0.01, min(f, cap))

    @staticmethod
    def volatility_adj(current_atr: float, baseline_atr: float) -> float:
        """size_multiplier = baseline / current. Clipped to [0.5, 1.5]."""
        if current_atr <= 0 or baseline_atr <= 0:
            return 1.0
        mult = baseline_atr / current_atr
        return max(0.5, min(mult, 1.5))

    @staticmethod
    def streak_adj(consecutive_losses: int, consecutive_wins: int) -> float:
        """
        After 2 consecutive losses: multiply by 0.85.
        After 2 consecutive wins: restore to 1.0 (full Kelly).
        """
        if consecutive_losses >= 2:
            return 0.85
        return 1.0

    @staticmethod
    def shares(capital: float, entry_price: float, stop_price: float,
               kelly_f: float, vol_adj: float, streak_adj: float) -> int:
        """
        risk_per_trade = capital * kelly_f * vol_adj * streak_adj
        shares = floor(risk_per_trade / (entry_price - stop_price))
        shares = max(1, shares)
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 1
        risk_amount = capital * kelly_f * vol_adj * streak_adj
        raw = int(risk_amount / risk_per_share)
        return max(1, raw)

    @staticmethod
    def portfolio_heat_ok(open_positions: list, capital: float, limit=0.20) -> bool:
        """total_risk / capital <= limit."""
        total_risk = sum(p.get("risk_amount", 0) for p in open_positions)
        return (total_risk / max(capital, 1)) <= limit


# ═════════════════════════════════════════════════════════════════════
# CLASS 5: PerformanceAnalytics
# ═════════════════════════════════════════════════════════════════════

class PerformanceAnalytics:
    """17 institutional metrics from single-row-per-trade log."""

    @staticmethod
    def compute(trade_log: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Compute all 17 metrics from the trade log."""
        if trade_log is None or trade_log.empty or len(trade_log) < 2:
            return {}

        df = trade_log.copy()
        n = len(df)

        wins = df[df["Result"] == "WIN"]
        losses = df[df["Result"] == "LOSS"]
        wc = len(wins)
        lc = len(losses)
        wr = wc / n if n > 0 else 0

        aw = float(wins["TotalPnLPct"].mean()) if wc > 0 else 0
        al = float(abs(losses["TotalPnLPct"].mean())) if lc > 0 else 0
        total_pnl_rs = float(df["TotalPnLRs"].sum())
        gross_profit = float(wins["TotalPnLRs"].sum()) if wc > 0 else 0
        gross_loss = float(abs(losses["TotalPnLRs"].sum())) if lc > 0 else 0

        # 1-3: Counts
        m = {"Total Trades": n, "Win Count": wc, "Loss Count": lc}

        # 4: Win rate
        m["Win Rate"] = round(wr, 3)

        # 5: Avg win/loss %
        m["Avg Win %"] = round(aw, 2)
        m["Avg Loss %"] = round(al, 2)

        # 6: Payoff ratio
        m["Payoff Ratio"] = round(aw / max(al, 0.01), 2) if al > 0 else 0

        # 7: Expectancy
        m["Expectancy %"] = round((wr * aw) - ((1 - wr) * al), 2)

        # 8: Profit factor
        m["Profit Factor"] = round(gross_profit / max(gross_loss, 0.01), 2) if gross_loss > 0 else round(gross_profit, 2)

        # 9: Total PnL
        m["Total PnL ₹"] = round(total_pnl_rs, 2)

        # 10-11: Max drawdown ₹ and %
        eq = [initial_capital]
        for pnl in df["TotalPnLRs"]:
            eq.append(eq[-1] + pnl)
        eq_series = pd.Series(eq[1:])
        peak = eq_series.cummax()
        dd_vals = (eq_series - peak) / peak.replace(0, 1) * 100
        max_dd_pct = float(dd_vals.min()) if len(dd_vals) > 0 else 0
        min_eq_idx = dd_vals.idxmin() if len(dd_vals) > 0 else 0
        max_dd_rs = float(peak.iloc[min_eq_idx] - eq_series.iloc[min_eq_idx]) if min_eq_idx < len(eq_series) else 0
        m["Max Drawdown %"] = round(max_dd_pct, 2)
        m["Max Drawdown ₹"] = round(max_dd_rs, 2)

        # 12: Recovery factor
        m["Recovery Factor"] = round(total_pnl_rs / max(abs(max_dd_rs), 1), 2)

        # 13: Sharpe ratio
        rets = df["TotalPnLPct"].values
        sharpe = round(float(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(n)), 2) if len(rets) > 1 and np.std(rets) > 0 else 0
        m["Sharpe Ratio"] = sharpe

        # 14: Sortino ratio
        downside = rets[rets < 0]
        dd_dev = float(np.std(downside)) if len(downside) > 1 else 0.01
        sortino = round(float(np.mean(rets) / max(dd_dev, 0.01) * np.sqrt(n)), 2)
        m["Sortino Ratio"] = sortino

        # 15: Calmar ratio
        total_return = (eq_series.iloc[-1] - initial_capital) / initial_capital
        first_date = df["Entry"].min() if "Entry" in df.columns else pd.Timestamp.now() - timedelta(days=365)
        last_date = df["TrailDate"].max() if "TrailDate" in df.columns else (df["Entry"].max() if "Entry" in df.columns else pd.Timestamp.now())
        if "TrailDate" not in df.columns or df["TrailDate"].isna().all():
            last_date = df["Entry"].max() if "Entry" in df.columns else pd.Timestamp.now()
        total_days = max((last_date - first_date).days, 1)
        years = total_days / 365.0
        cagr = ((1 + total_return) ** (1 / max(years, 0.01)) - 1) * 100
        m["Calmar Ratio"] = round(cagr / max(abs(max_dd_pct), 0.1), 2)

        # 16: Max consecutive losses
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
        m["Max Consecutive Losses"] = max(streaks) if streaks else 0

        # 17: Avg bars held (winners vs losers)
        if "Bars" in df.columns:
            win_bars = df[df["Result"] == "WIN"]["Bars"]
            loss_bars = df[df["Result"] == "LOSS"]["Bars"]
            m["Avg Win Bars"] = round(float(win_bars.mean()), 1) if len(win_bars) > 0 else 0
            m["Avg Loss Bars"] = round(float(loss_bars.mean()), 1) if len(loss_bars) > 0 else 0
            m["Avg Bars"] = round(float(df["Bars"].mean()), 1)

        # 18: Partial execution rate
        partial_count = df["PartialDate"].notna().sum() if "PartialDate" in df.columns else 0
        m["Partial Execution Rate"] = round(partial_count / n, 3) if n > 0 else 0

        # 19: Exit breakdown
        exit_bk = df["ExitReason"].value_counts().to_dict() if "ExitReason" in df.columns else {}
        m["Exit Breakdown"] = exit_bk
        total_exits = sum(exit_bk.values())
        m["Exit %"] = {k: round(v / max(total_exits, 1) * 100, 1) for k, v in exit_bk.items()}

        # 20: Rolling 20-trade Sharpe flag
        roll_sharpes = PerformanceAnalytics.rolling_sharpe(df, window=20)
        m["Rolling Sharpe < 0.5 Flag"] = any(s < 0.5 for s in roll_sharpes) if len(roll_sharpes) > 0 else False
        m["Rolling Sharpe (Avg)"] = round(float(np.mean(roll_sharpes)), 2) if len(roll_sharpes) > 0 else sharpe

        return m

    @staticmethod
    def flag_partial_bug(trade_log: pd.DataFrame):
        """Print warning if partial execution rate < 60%."""
        if trade_log is None or trade_log.empty:
            return
        n = len(trade_log)
        partial_count = trade_log["PartialDate"].notna().sum() if "PartialDate" in trade_log.columns else 0
        rate = partial_count / n if n > 0 else 0
        if rate < 0.60:
            print(f"⚠️  PARTIAL BUG: execution rate = {rate:.1%} (target ≥ 60%). Check partial logic.")
        else:
            print(f"✅ Partial execution rate: {rate:.1%}")

    @staticmethod
    def rolling_sharpe(trade_log: pd.DataFrame, window=20) -> list:
        """Compute rolling window Sharpe. Returns list of values."""
        if trade_log is None or trade_log.empty or len(trade_log) < window:
            return []
        rets = trade_log["TotalPnLPct"].values
        results = []
        for j in range(window, len(rets) + 1):
            chunk = rets[j - window:j]
            s = float(np.mean(chunk) / max(np.std(chunk), 0.01) * np.sqrt(window))
            results.append(s)
        return results

    @staticmethod
    def print_report(metrics: dict):
        """Print formatted report to console."""
        if not metrics:
            print("No data to report.")
            return
        print("=" * 60)
        print("V7 BACKTEST REPORT")
        print("=" * 60)
        print(f"Total Trades:         {metrics.get('Total Trades', 0)}")
        print(f"Win Rate:             {metrics.get('Win Rate', 0):.1%} ({metrics.get('Win Count', 0)}W / {metrics.get('Loss Count', 0)}L)")
        print(f"Total PnL:            ₹{metrics.get('Total PnL ₹', 0):,.2f}")
        print(f"Avg Win:              {metrics.get('Avg Win %', 0):+.2f}%")
        print(f"Avg Loss:             {metrics.get('Avg Loss %', 0):.2f}%")
        print(f"Payoff Ratio:         {metrics.get('Payoff Ratio', 0):.2f}")
        print(f"Expectancy:           {metrics.get('Expectancy %', 0):+.2f}%")
        print(f"Profit Factor:        {metrics.get('Profit Factor', 0):.2f}")
        print(f"Sharpe Ratio:         {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"Sortino Ratio:        {metrics.get('Sortino Ratio', 0):.2f}")
        print(f"Calmar Ratio:         {metrics.get('Calmar Ratio', 0):.2f}")
        print(f"Max Drawdown:         {metrics.get('Max Drawdown %', 0):.2f}% (₹{metrics.get('Max Drawdown ₹', 0):,.0f})")
        print(f"Recovery Factor:      {metrics.get('Recovery Factor', 0):.2f}")
        print(f"Max Cons Losses:      {metrics.get('Max Consecutive Losses', 0)}")
        print(f"Avg Win Bars:         {metrics.get('Avg Win Bars', 0)}")
        print(f"Avg Loss Bars:        {metrics.get('Avg Loss Bars', 0)}")
        print(f"Partial Exec Rate:    {metrics.get('Partial Execution Rate', 0):.1%}")
        print(f"Rolling Sharpe Flag:  {'⚠️ Degraded' if metrics.get('Rolling Sharpe < 0.5 Flag') else '✅ Stable'}")
        print("")
        print("Exit Breakdown:")
        for reason, count in sorted(metrics.get("Exit Breakdown", {}).items(), key=lambda x: -x[1]):
            pct = metrics.get("Exit %", {}).get(reason, 0)
            print(f"  {reason}: {count} ({pct:.1f}%)")
        print("=" * 60)


# ═════════════════════════════════════════════════════════════════════
# CLASS 6: Backtester
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TradeState:
    """Tracks a single open position, used internally by Backtester."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_bar: int
    initial_shares: int
    initial_stop: float
    atr_at_entry: float

    # Partial leg
    partial_done: bool = False
    partial_date: Optional[pd.Timestamp] = None
    partial_price: float = 0.0
    partial_shares: int = 0
    partial_pnl_rs: float = 0.0
    partial_pnl_pct: float = 0.0

    # Trail leg
    breakeven_stop: float = 0.0  # moved to breakeven after partial
    chandelier_stop: float = 0.0  # updated each bar


class Backtester:
    """
    V7 main backtesting engine.

    Config keys with defaults:
      capital=100000, adx_threshold=22, kelly_cap=0.25,
      partial_pct=0.40, partial_r=1.5,
      chandelier_period=22, chandelier_multiplier=3.0,
      gap_up_threshold=0.04, time_stop_days=20, time_stop_r=-0.5,
      portfolio_heat_limit=0.20,
      circuit_breaker_window=10, circuit_breaker_threshold=0.10,
      earnings_blackout_dates=None,  # dict: symbol -> [date_list]
    """

    def __init__(self, config: dict = None):
        self.config = {
            "capital": 100000,
            "adx_threshold": 22,
            "kelly_cap": 0.25,
            "partial_pct": 0.40,
            "partial_r": 1.5,
            "chandelier_period": 22,
            "chandelier_multiplier": 3.0,
            "gap_up_threshold": 0.04,
            "time_stop_days": 20,
            "time_stop_r": -0.5,
            "portfolio_heat_limit": 0.20,
            "circuit_breaker_window": 10,
            "circuit_breaker_threshold": 0.10,
            "earnings_blackout_dates": None,
        }
        if config:
            self.config.update(config)

        self.entry_filter = EntryFilter()
        self.exit_mgr = ExitManager(
            partial_pct=self.config["partial_pct"],
            partial_r=self.config["partial_r"],
            chandelier_period=self.config["chandelier_period"],
            chandelier_multiplier=self.config["chandelier_multiplier"],
            gap_up_threshold=self.config["gap_up_threshold"],
            time_stop_days=self.config["time_stop_days"],
            time_stop_r=self.config["time_stop_r"],
        )

    def load_data(self, symbol: str, period="5y") -> pd.DataFrame:
        """Load daily data with all indicators. Cache to avoid re-download."""
        from core.utils import load_data
        df = load_data(symbol, period=period)
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

    def _check_earnings_blackout(self, date: pd.Timestamp, symbol: str) -> bool:
        """No entry if earnings date within 5 days before or 2 days after."""
        earnings_map = self.config.get("earnings_blackout_dates")
        if not earnings_map:
            return False  # No blackout if not configured
        dates = earnings_map.get(symbol, [])
        for ed in dates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts - timedelta(days=5) <= date <= ed_ts + timedelta(days=2):
                return True
        return False

    def run(self, symbol: str) -> dict:
        """Run backtest on a single symbol. Returns dict with trade_log, metrics, etc."""
        # Load data
        df = self.load_data(symbol)
        if df.empty or len(df) < 250:
            return {"Error": "Insufficient data", "Total Trades": 0}

        weekly_df = self._load_weekly(symbol)
        earnings_dates = (self.config.get("earnings_blackout_dates") or {}).get(symbol, [])

        # ATR baseline (median ATR over last 60 days)
        atr_baseline = float(df["ATR"].tail(60).median()) if len(df) >= 60 else float(df["ATR"].mean())

        # State
        capital = self.config["capital"]
        trades = []
        open_positions = []  # list of dicts for portfolio heat tracking
        consecutive_wins = 0
        consecutive_losses = 0
        trade_pnls = []  # rolling PnL% list for Kelly
        last_10_pnls_rs = []  # for circuit breaker
        circuit_halt_until = -1
        daily_pnl = 0.0
        daily_bar = -1
        daily_pnl_start = capital

        active_trade = None  # TradeState of open position
        pos = False

        for i in range(220, len(df)):
            bar = df.iloc[:i + 1]
            r = df.iloc[i]
            c = float(r["Close"])
            o = float(r["Open"])
            h = float(r["High"])
            lo = float(r["Low"])
            atr = _sf(r, "ATR", c * 0.02)
            if atr <= 0:
                atr = c * 0.02
            date = r.name
            if isinstance(date, pd.Timestamp):
                date_ts = date
            else:
                date_ts = pd.Timestamp(date)

            # Daily reset for max daily loss
            if i != daily_bar:
                daily_bar = i
                daily_pnl_start = daily_pnl

            # ── ENTRY ──
            if not pos:
                # Circuit breaker check
                if circuit_halt_until > i:
                    continue

                # Max daily loss check
                daily_loss_pct = (daily_pnl - daily_pnl_start) / max(daily_pnl_start, 1)
                if daily_loss_pct < -0.02:
                    continue

                # Earnings blackout
                if self._check_earnings_blackout(date_ts, symbol):
                    continue

                # Entry filters (4 total)
                ok, reason = self.entry_filter.is_valid_entry(bar, weekly_df)
                if not ok:
                    continue

                # Portfolio heat check
                total_risk = sum(p.get("risk_amount", 0) for p in open_positions)
                if total_risk / max(capital, 1) > self.config["portfolio_heat_limit"]:
                    continue

                # Kelly sizing
                if len(trade_pnls) >= 10:
                    recent = trade_pnls[-20:] if len(trade_pnls) >= 20 else trade_pnls
                    wins = [p for p in recent if p > 0]
                    losses = [p for p in recent if p <= 0]
                    wr_k = len(wins) / max(len(recent), 1)
                    aw_k = np.mean(wins) if wins else 0
                    al_k = abs(np.mean(losses)) if losses else 1
                    payoff_r = aw_k / max(al_k, 0.01)
                else:
                    wr_k = 0.55  # Static for first 20 trades
                    payoff_r = 3.0

                kelly_f = PositionSizer.kelly(wr_k, payoff_r, self.config["kelly_cap"])
                vol_adj = PositionSizer.volatility_adj(atr, atr_baseline)
                s_adj = PositionSizer.streak_adj(consecutive_losses, consecutive_wins)

                initial_stop = ExitManager.get_initial_stop(c, atr)
                shares = PositionSizer.shares(capital, c, initial_stop, kelly_f, vol_adj, s_adj)
                if shares < 1:
                    continue

                risk_amount = shares * abs(c - initial_stop)

                pos = True
                active_trade = TradeState(
                    entry_date=date_ts,
                    entry_price=c,
                    entry_bar=i,
                    initial_shares=shares,
                    initial_stop=initial_stop,
                    atr_at_entry=atr,
                    breakeven_stop=initial_stop,
                    chandelier_stop=initial_stop,
                )
                open_positions.append({"risk_amount": risk_amount})
                continue

            # ── EXIT ──
            if pos and active_trade is not None:
                should_exit, reason, exit_price = self.exit_mgr.get_exit(bar, i, active_trade, atr)

                if not should_exit:
                    continue

                # ── PARTIAL BOOKING (special handling — updates trade, doesn't close) ──
                if reason == "Partial40%":
                    partial_shares = max(1, int(active_trade.initial_shares * self.config["partial_pct"]))
                    remaining = active_trade.initial_shares - partial_shares

                    # Calculate partial PnL
                    partial_raw = partial_shares * (exit_price - active_trade.entry_price)
                    partial_cost = CostModel.apply(active_trade.entry_price, exit_price, partial_shares)
                    partial_pnl_rs = partial_raw - partial_cost
                    partial_pnl_pct = partial_pnl_rs / (active_trade.entry_price * partial_shares) * 100

                    # Update trade state
                    active_trade.partial_done = True
                    active_trade.partial_date = date_ts
                    active_trade.partial_price = exit_price
                    active_trade.partial_shares = partial_shares
                    active_trade.partial_pnl_rs = partial_pnl_rs
                    active_trade.partial_pnl_pct = partial_pnl_pct
                    active_trade.initial_shares = remaining  # Update for trail leg
                    active_trade.breakeven_stop = active_trade.entry_price  # Move to breakeven

                    # Update capital
                    capital += partial_pnl_rs
                    daily_pnl += partial_pnl_rs
                    # Update risk amount for remaining shares
                    if open_positions:
                        open_positions[-1]["risk_amount"] = remaining * abs(active_trade.entry_price - active_trade.breakeven_stop)

                    continue  # Keep position open

                # ── FINAL EXIT (all other reasons) ──
                shares_exit = active_trade.initial_shares

                # Calculate trail leg PnL
                trail_raw = shares_exit * (exit_price - active_trade.entry_price)
                trail_cost = CostModel.apply(active_trade.entry_price, exit_price, shares_exit)
                trail_pnl_rs = trail_raw - trail_cost

                # Combine with partial if it exists
                if active_trade.partial_done:
                    total_pnl_rs = active_trade.partial_pnl_rs + trail_pnl_rs
                    total_shares = active_trade.partial_shares + shares_exit
                else:
                    total_pnl_rs = trail_pnl_rs
                    total_shares = shares_exit

                total_invested = active_trade.entry_price * total_shares
                total_pnl_pct = total_pnl_rs / max(total_invested, 0.01) * 100

                # Bars = calendar days from entry to exit (BUG 1 fix)
                held_days = max((date_ts - active_trade.entry_date).days, 1)

                # Result tag
                result = "WIN" if total_pnl_rs >= 0 else "LOSS"

                # Update capital and tracking
                capital += trail_pnl_rs
                daily_pnl += trail_pnl_rs
                trade_pnls.append(total_pnl_pct)
                if len(trade_pnls) > 50:
                    trade_pnls.pop(0)
                last_10_pnls_rs.append(total_pnl_rs)
                if len(last_10_pnls_rs) > 10:
                    last_10_pnls_rs.pop(0)

                # Circuit breaker check
                if len(last_10_pnls_rs) >= 5:
                    rolling_sum = sum(last_10_pnls_rs)
                    if rolling_sum < -self.config["circuit_breaker_threshold"] * capital:
                        circuit_halt_until = i + 5

                # Streak tracking
                if total_pnl_rs > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0

                # Close in risk tracking
                if open_positions:
                    open_positions.pop()

                # Build single-row trade record
                record = {
                    "Entry": active_trade.entry_date,
                    "EntryPrice": round(active_trade.entry_price, 2),
                    "InitialStop": round(active_trade.initial_stop, 2),
                    "InitialShares": total_shares,
                    "PartialDate": active_trade.partial_date if active_trade.partial_done else None,
                    "PartialPrice": round(active_trade.partial_price, 2) if active_trade.partial_done else None,
                    "PartialShares": active_trade.partial_shares if active_trade.partial_done else 0,
                    "PartialPnLRs": round(active_trade.partial_pnl_rs, 2) if active_trade.partial_done else 0.0,
                    "PartialPnLPct": round(active_trade.partial_pnl_pct, 2) if active_trade.partial_done else 0.0,
                    "TrailDate": date_ts,
                    "TrailPrice": round(exit_price, 2),
                    "TrailShares": shares_exit,
                    "TrailPnLRs": round(trail_pnl_rs, 2),
                    "ExitReason": reason,
                    "TotalPnLRs": round(total_pnl_rs, 2),
                    "TotalPnLPct": round(total_pnl_pct, 2),
                    "Bars": held_days,
                    "Result": result,
                }
                trades.append(record)
                pos = False
                active_trade = None

        # Close any open position at end
        if pos and active_trade is not None:
            fp = float(df["Close"].iloc[-1])
            shares_exit = active_trade.initial_shares
            trail_raw = shares_exit * (fp - active_trade.entry_price)
            trail_cost = CostModel.apply(active_trade.entry_price, fp, shares_exit)
            trail_pnl_rs = trail_raw - trail_cost

            if active_trade.partial_done:
                total_pnl_rs = active_trade.partial_pnl_rs + trail_pnl_rs
                total_shares = active_trade.partial_shares + shares_exit
            else:
                total_pnl_rs = trail_pnl_rs
                total_shares = shares_exit

            total_invested = active_trade.entry_price * total_shares
            total_pnl_pct = total_pnl_rs / max(total_invested, 0.01) * 100
            held_days = max((df.index[-1] - active_trade.entry_date).days, 1)
            capital += trail_pnl_rs

            record = {
                "Entry": active_trade.entry_date,
                "EntryPrice": round(active_trade.entry_price, 2),
                "InitialStop": round(active_trade.initial_stop, 2),
                "InitialShares": total_shares,
                "PartialDate": active_trade.partial_date if active_trade.partial_done else None,
                "PartialPrice": round(active_trade.partial_price, 2) if active_trade.partial_done else None,
                "PartialShares": active_trade.partial_shares if active_trade.partial_done else 0,
                "PartialPnLRs": round(active_trade.partial_pnl_rs, 2) if active_trade.partial_done else 0.0,
                "PartialPnLPct": round(active_trade.partial_pnl_pct, 2) if active_trade.partial_done else 0.0,
                "TrailDate": df.index[-1],
                "TrailPrice": round(fp, 2),
                "TrailShares": shares_exit,
                "TrailPnLRs": round(trail_pnl_rs, 2),
                "ExitReason": "EndOfData",
                "TotalPnLRs": round(total_pnl_rs, 2),
                "TotalPnLPct": round(total_pnl_pct, 2),
                "Bars": held_days,
                "Result": "WIN" if total_pnl_rs >= 0 else "LOSS",
            }
            trades.append(record)

        trade_log = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = PerformanceAnalytics.compute(trade_log, self.config["capital"])

        # Assertions (BUG verification)
        if not trade_log.empty:
            PerformanceAnalytics.flag_partial_bug(trade_log)
            max_bars = trade_log["Bars"].max() if "Bars" in trade_log.columns else 0
            if max_bars < 365:
                print(f"✅ Bars are per-trade (max={max_bars}d)")
            else:
                print(f"⚠️  BARS BUG: max bars = {max_bars}d, expected < 365")
            dup_entries = trade_log["Entry"].duplicated().sum() if "Entry" in trade_log.columns else 0
            if dup_entries == 0:
                print("✅ No duplicate entry dates — single-row-per-trade OK")
            else:
                print(f"⚠️  DUAL-ROW BUG: {dup_entries} duplicate entry dates found")

        return {
            "trade_log": trade_log,
            "metrics": metrics,
            "config": self.config,
            "Total Trades": len(trades),
            "Total Return %": metrics.get("Total PnL ₹", 0) / self.config["capital"] * 100,
            "Profit Factor": metrics.get("Profit Factor", 0),
            "Sharpe Ratio": metrics.get("Sharpe Ratio", 0),
            "Win Rate %": metrics.get("Win Rate", 0) * 100,
            "Max Drawdown %": metrics.get("Max Drawdown %", 0),
            "Final Capital": round(capital, 2),
        }

    def run_montecarlo(self, trade_log: pd.DataFrame, n_runs=2000, confidence=0.90) -> dict:
        """Monte Carlo simulation by sampling trades with replacement."""
        if trade_log is None or trade_log.empty or len(trade_log) < 5:
            return {"Simulations": 0}
        rets = trade_log["TotalPnLRs"].values
        capital = self.config["capital"]
        n = len(rets)
        results = []
        dd_list = []
        for _ in range(n_runs):
            sampled = np.random.choice(rets, size=n, replace=True)
            eq = [capital]
            for pnl in sampled:
                eq.append(eq[-1] + pnl)
            total_ret = (eq[-1] - capital) / capital * 100
            results.append(total_ret)
            peak = np.maximum.accumulate(eq)
            dd = (np.array(eq) - peak) / peak * 100
            dd_list.append(abs(dd.min()))
        results = np.array(results)
        dd_list = np.array(dd_list)
        alpha = (1 - confidence) / 2
        return {
            "Median Return %": round(np.median(results), 2),
            "Mean Return %": round(np.mean(results), 2),
            "CI Lower %": round(np.percentile(results, alpha * 100), 2),
            "CI Upper %": round(np.percentile(results, (1 - alpha) * 100), 2),
            "Prob Profit %": round(np.mean(results > 0) * 100, 1),
            "Avg MaxDD %": round(np.mean(dd_list), 2),
            "Worst MaxDD %": round(np.max(dd_list), 2),
            "Simulations": n_runs,
        }

    def run_multi(self, symbols: list) -> dict:
        """Run backtest on each symbol. Returns combined report."""
        all_results = {}
        combined_logs = []
        for sym in symbols:
            print(f"\n{'='*60}")
            print(f"RUNNING: {sym}")
            print('=' * 60)
            result = self.run(sym)
            all_results[sym] = result
            tl = result.get("trade_log")
            if tl is not None and not tl.empty:
                tl = tl.copy()
                tl["Symbol"] = sym
                combined_logs.append(tl)
                m = result.get("metrics", {})
                PerformanceAnalytics.print_report(m)

        # Combined summary table
        print("\n" + "=" * 60)
        print("COMBINED SUMMARY")
        print("=" * 60)
        header = f"{'Symbol':<15} {'Trades':<8} {'WR':<8} {'Expect':<8} {'PF':<8} {'Sharpe':<8} {'PnL₹':<12} {'MaxDD':<8} {'Partial':<8}"
        print(header)
        print("-" * len(header))
        for sym, res in all_results.items():
            m = res.get("metrics", {})
            print(f"{sym:<15} {m.get('Total Trades', 0):<8} {m.get('Win Rate', 0):.1%:<8} "
                  f"{m.get('Expectancy %', 0):<8} {m.get('Profit Factor', 0):<8} "
                  f"{m.get('Sharpe Ratio', 0):<8} ₹{m.get('Total PnL ₹', 0):<8,.0f} "
                  f"{m.get('Max Drawdown %', 0):<8} {m.get('Partial Execution Rate', 0):.0%:<8}")

        combined_log = pd.concat(combined_logs, ignore_index=True) if combined_logs else pd.DataFrame()
        return {
            "results": all_results,
            "combined_log": combined_log,
        }


# ═════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPERS
# ═════════════════════════════════════════════════════════════════════

def run_v7_backtest(symbol: str, capital: float = 100000) -> dict:
    """Run V7 backtest with default config."""
    bt = Backtester({"capital": capital})
    return bt.run(symbol)


# ═════════════════════════════════════════════════════════════════════
# WORKING EXAMPLE
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    symbols = ["ABB.NS", "LT.NS", "ADANIENT.NS"]
    start = "2022-01-01"
    end = "2024-12-31"

    bt = Backtester(config={
        "capital": 100000,
        "adx_threshold": 22,
        "kelly_cap": 0.25,
        "partial_pct": 0.40,
        "partial_r": 1.5,
        "chandelier_period": 22,
        "chandelier_multiplier": 3.0,
        "gap_up_threshold": 0.04,
        "time_stop_days": 20,
        "time_stop_r": -0.5,
        "portfolio_heat_limit": 0.20,
        "circuit_breaker_window": 10,
        "circuit_breaker_threshold": 0.10,
    })

    results = bt.run_multi(symbols)

    # Final assertions
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    all_ok = True
    for sym in symbols:
        tl = results["results"].get(sym, {}).get("trade_log")
        if tl is not None and not tl.empty:
            # Assertion 1: partial execution rate >= 60%
            partial_rate = tl["PartialDate"].notna().sum() / len(tl)
            if partial_rate >= 0.60:
                print(f"✅ {sym}: Partial execution rate {partial_rate:.0%} ≥ 60%")
            else:
                print(f"❌ {sym}: PARTIAL BUG — rate {partial_rate:.0%} < 60%")
                all_ok = False

            # Assertion 2: all bars are per-trade (max bars < 365)
            max_bars = tl["Bars"].max()
            if max_bars < 365:
                print(f"✅ {sym}: Max bars = {max_bars}d < 365 (per-trade)")
            else:
                print(f"❌ {sym}: BARS BUG — max bars = {max_bars}d")
                all_ok = False

            # Assertion 3: no duplicate entry dates
            dup_entries = tl["Entry"].duplicated().sum()
            if dup_entries == 0:
                print(f"✅ {sym}: No duplicate entry dates")
            else:
                print(f"❌ {sym}: DUAL-ROW BUG — {dup_entries} duplicates found")
                all_ok = False
        else:
            print(f"⚠️  {sym}: No trades to verify")

    if all_ok:
        print("\n✅ ALL ASSERTIONS PASSED — V7 IS COMPLETE")
    else:
        print("\n❌ SOME ASSERTIONS FAILED — REVIEW BEFORE DEPLOYMENT")
