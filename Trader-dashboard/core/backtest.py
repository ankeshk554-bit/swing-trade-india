import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from core.indicators import compute_indicators


def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_data(symbol, period="5y"):
    """Fetch and clean stock data."""
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = clean_columns(df)
    return df


def net_pnl(entry: float, exit_price: float, shares: int) -> float:
    """SEBI cost model: STT, brokerage, GST, stamp."""
    gross = (exit_price - entry) * shares
    ev = exit_price * shares
    en = entry * shares
    stt = ev * 0.001
    brk = min(20, en * 0.0003) + min(20, ev * 0.0003)
    gst = brk * 0.18
    stamp = en * 0.00015 if gross > 0 else 0
    return gross - (stt + brk + gst + stamp)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """f* = W - (1-W)/R, capped at 0.25."""
    if win_rate <= 0 or avg_loss <= 0:
        return 0.02
    R = abs(avg_win / max(avg_loss, 0.01))
    f = win_rate / 100 - (1 - win_rate / 100) / max(R, 0.1)
    return max(0.01, min(f, 0.25))


def run_backtest_simple(symbol):
    """Simple EMA crossover backtest (legacy)."""
    try:
        df = fetch_data(symbol)
        if df is None:
            return None

        df = compute_indicators(df)

        df["Signal"] = 0
        df.loc[df["EMA50"] > df["EMA200"], "Signal"] = 1

        df["Returns"] = df["Close"].pct_change()
        df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

        total_return = ((1 + df["Strategy"]).cumprod().iloc[-1] - 1) * 100

        return {"Total Return": round(total_return, 2)}

    except Exception as e:
        print(f"Backtest Error: {e}")
        return None


def _sf(r, field, default=0):
    v = r.get(field, default)
    try:
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else default
    except (ValueError, TypeError):
        return default


def run_backtest(symbol, initial_capital=100000):
    """
    V8-powered swing strategy backtest.

    ATR stop · Kelly sizing · Partial 1.5R · Chandelier trail ·
    Breakeven migration · SEBI costs · Circuit breaker · SellSig gate

    Returns same dict structure as legacy backtest for dashboard compat.
    """
    try:
        df = fetch_data(symbol)
        if df is None or len(df) < 220:
            return None

        df = compute_indicators(df)
        cap = initial_capital
        trades = []
        recent_pnls = []

        # State
        pos = False
        ep = eb = ed = 0
        sh = 0
        sl = 0.0
        R_price = 0.0
        highest_since_entry = 0.0
        partial_done = False
        partial_shares_log = 0
        partial_pnl_log = 0.0
        partial_price_log = 0.0
        consecutive_losses = 0
        circuit_bk_fired = 0
        circuit_until = -1

        for i in range(200, len(df)):
            bar_slice = df.iloc[:i + 1]
            bar = df.iloc[i]
            c = float(bar["Close"])
            h = float(bar["High"])
            atr = float(bar.get("ATR", c * 0.02))
            if np.isnan(atr) or atr <= 0:
                atr = c * 0.02

            # Entry signal: EMA50 > EMA200 + RSI > 50 + volume > avg
            rsi = _sf(bar, "RSI", 50)
            ema50 = _sf(bar, "EMA50", c)
            ema200 = _sf(bar, "EMA200", c)
            rvol = _sf(bar, "RVOL", 1)
            macd_h = _sf(bar, "MACD_HIST", 0)
            st_dir = _sf(bar, "SUPERTREND_DIR", 0)

            entry_signal = (c > ema50 > ema200 and rsi > 50 and
                            macd_h > 0 and st_dir == 1 and rvol > 1.2)

            # ── ENTRY ──
            if not pos and entry_signal and circuit_until <= i:
                # Kelly sizing
                wr = 0; aw = 0; al = 0
                if len(recent_pnls) > 5:
                    wins = [p for p in recent_pnls if p > 0]
                    losses = [p for p in recent_pnls if p <= 0]
                    wr = len(wins) / max(len(recent_pnls), 1) * 100
                    aw = np.mean(wins) if wins else 0
                    al = abs(np.mean(losses)) if losses else 1

                initial_stop = c - 2.0 * atr
                ep = c
                eb = i
                ed = bar.name
                sl = initial_stop
                R_price = abs(c - initial_stop)
                risk_rs = cap * kelly_fraction(wr, aw, al)
                sh = max(1, int(risk_rs / max(R_price, 0.01)))

                highest_since_entry = c
                partial_done = False
                partial_shares_log = 0
                partial_pnl_log = 0.0
                partial_price_log = 0.0
                pos = True
                continue

            # ── EXIT ──
            if pos:
                ex = None
                er = ""
                total_shares = sh

                # Track highest high
                highest_since_entry = max(highest_since_entry, h)

                # 6. Breakeven stop: after 1R profit, stop = max(stop, entry)
                if R_price > 0 and highest_since_entry >= ep + 1.0 * R_price:
                    sl = max(sl, ep)

                # 3. Partial exit at 1.5R (40%)
                if not partial_done and R_price > 0:
                    partial_trigger = ep + 1.5 * R_price
                    if h >= partial_trigger:
                        p_shares = max(1, int(total_shares * 0.40))
                        remaining = total_shares - p_shares
                        p_pnl = net_pnl(ep, partial_trigger, p_shares)
                        partial_shares_log = p_shares
                        partial_pnl_log = p_pnl
                        partial_price_log = partial_trigger
                        cap += p_pnl
                        sh = remaining
                        sl = max(sl, ep)  # Move to breakeven — SAME BAR (CHANGE 2)
                        partial_done = True
                        # Log partial exit
                        trades.append({
                            "EntryDate": ed, "ExitDate": bar.name,
                            "EntryPrice": round(ep, 2),
                            "ExitPrice": round(partial_trigger, 2),
                            "PnL%": round(p_pnl / max(ep * p_shares, 0.01) * 100, 2),
                            "Result": "WIN",
                            "BarsHeld": max((bar.name - ed).days, 1) if hasattr(bar.name, 'days') else i - eb + 1,
                            "Reason": "Partial40%"
                        })
                        continue

                # 4. Chandelier trail: HH(22) - 3*ATR
                lb = min(22, i)
                seg = df.iloc[i - lb:i + 1]
                hh = seg["High"].max()
                chandelier = hh - 3.0 * atr
                trail_stop = max(chandelier, sl)

                # 5. SellSig exit — subordinated to trail (CHANGE 1)
                # Only fires when price is within 3% of chandelier stop floored at entry.
                # If price is far above trail, signal is ignored — trail runs on.
                sell_sig = (st_dir == -1 and macd_h < 0 and rsi < 45)
                if sell_sig and c > 0:
                    sell_stop = max(trail_stop, ep)  # never below breakeven
                    dist_to_trail = (c - sell_stop) / max(c, 0.01)
                    if dist_to_trail < 0.03:
                        ex = c
                        er = "SellSig"

                # Trail stop check
                if ex is None and c <= trail_stop:
                    ex = trail_stop
                    er = "ChandelierTrail"

                # Initial stop (floor)
                if ex is None and c <= sl:
                    ex = sl
                    er = "InitialStop"

                if ex is not None:
                    trail_pnl = net_pnl(ep, ex, sh)
                    total_pnl = trail_pnl + partial_pnl_log
                    total_shares_f = sh + partial_shares_log
                    pp = total_pnl / max(ep * total_shares_f, 0.01) * 100
                    cap += trail_pnl
                    recent_pnls.append(pp)
                    if len(recent_pnls) > 50:
                        recent_pnls.pop(0)

                    trades.append({
                        "EntryDate": ed, "ExitDate": bar.name,
                        "EntryPrice": round(ep, 2),
                        "ExitPrice": round(ex, 2),
                        "PnL%": round(pp, 2),
                        "Result": "WIN" if total_pnl > 0 else "LOSS",
                        "BarsHeld": max((bar.name - ed).days, 1) if hasattr(bar.name, 'days') else i - eb + 1,
                        "Reason": er,
                    })

                    # 8. Circuit breaker
                    if pp > 0:
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        if consecutive_losses >= 4:
                            circuit_until = i + 5
                            circuit_bk_fired += 1

                    pos = False
                    partial_shares_log = 0
                    partial_pnl_log = 0

        # Close open at end
        if pos:
            fp = float(df["Close"].iloc[-1])
            trail_pnl = net_pnl(ep, fp, sh)
            total_pnl = trail_pnl + partial_pnl_log
            total_shares_f = sh + partial_shares_log
            pp = total_pnl / max(ep * total_shares_f, 0.01) * 100
            cap += trail_pnl
            trades.append({
                "EntryDate": ed, "ExitDate": df.index[-1],
                "EntryPrice": round(ep, 2),
                "ExitPrice": round(fp, 2),
                "PnL%": round(pp, 2),
                "Result": "WIN" if total_pnl > 0 else "LOSS",
                "BarsHeld": (df.index[-1] - ed).days if hasattr(ed, 'days') else len(df) - eb,
                "Reason": "End",
            })

        # ── METRICS ──
        n = len(trades)
        if n == 0:
            trades_df = pd.DataFrame()
            return {
                "Total Return %": 0.0, "Total Trades": 0,
                "Win Rate %": 0.0, "Avg Win %": 0.0, "Avg Loss %": 0.0,
                "Profit Factor": 0.0, "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0, "Max Drawdown %": 0.0,
                "Expectancy": 0.0, "Final Capital": cap,
                "Trades": trades_df,
                "PartialRate": 0.0, "AvgBarsWin": 0, "AvgBarsLoss": 0,
                "CircuitBreakerFired": circuit_bk_fired,
            }

        wins = [t for t in trades if t["Result"] == "WIN"]
        losses = [t for t in trades if t["Result"] == "LOSS"]
        wc = len(wins)
        lc = len(losses)
        wr = round(wc / n * 100, 1)
        aw = round(np.mean([t["PnL%"] for t in wins]), 2) if wins else 0
        al = round(np.mean([t["PnL%"] for t in losses]), 2) if losses else 0
        tp = sum(t["PnL%"] for t in wins) if wins else 0
        tl = abs(sum(t["PnL%"] for t in losses)) if losses else 0
        pf = round(tp / max(tl, 0.01), 2)

        rets = [t["PnL%"] for t in trades]
        shp = round(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252), 2) if len(rets) > 1 else 0

        # Drawdown
        eq = [initial_capital]
        for t in trades:
            eq.append(eq[-1] + (t["PnL%"] / 100 * initial_capital))
        es = pd.Series(eq[1:])
        rm = es.cummax()
        dd = (es - rm) / rm.replace(0, 1) * 100
        mdd = round(dd.min(), 2) if len(dd) > 0 else 0

        ret = round((cap - initial_capital) / initial_capital * 100, 2)
        expectancy = round((wr / 100 * aw) - ((1 - wr / 100) * abs(al)), 2)

        # Partial rate
        partial_count = sum(1 for t in trades if t.get("Reason") == "Partial40%")
        partial_rate = round(partial_count / max(n, 1), 3)

        # Avg bars
        win_bars = [t.get("BarsHeld", 0) for t in wins]
        loss_bars = [t.get("BarsHeld", 0) for t in losses]
        avg_bw = round(np.mean(win_bars), 1) if win_bars else 0
        avg_bl = round(np.mean(loss_bars), 1) if loss_bars else 0

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "Total Return %": ret,
            "Total Trades": n,
            "Win Rate %": wr,
            "Avg Win %": aw,
            "Avg Loss %": al,
            "Profit Factor": pf,
            "Sharpe Ratio": shp,
            "Sortino Ratio": 0.0,  # simplified
            "Max Drawdown %": mdd,
            "Expectancy": expectancy,
            "Final Capital": round(cap, 2),
            "Trades": trades_df,
            "PartialRate": partial_rate,
            "AvgBarsWin": avg_bw,
            "AvgBarsLoss": avg_bl,
            "CircuitBreakerFired": circuit_bk_fired,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Backtest Error: {e}")
        return None