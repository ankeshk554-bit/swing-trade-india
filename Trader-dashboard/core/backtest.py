import yfinance as yf
import pandas as pd
import numpy as np
from core.indicators import compute_indicators, generate_swing_signal


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


def run_backtest(symbol, initial_capital=100000):
    """
    Full swing strategy backtest with complete metrics.

    Uses the swing signal engine from indicators.py for entry/exit logic.
    Returns detailed performance metrics.
    """
    try:
        df = fetch_data(symbol)
        if df is None or len(df) < 220:
            return None

        df = compute_indicators(df)

        capital = initial_capital
        position = 0  # shares held
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None

        for i in range(200, len(df)):
            row = df.iloc[i]
            signal_info = generate_swing_signal(df.iloc[:i + 1])

            if signal_info["Signal"] == "BUY" and not in_position:
                entry_price = float(row["Close"])
                entry_date = row.name
                position = capital / entry_price
                in_position = True

            elif signal_info["Signal"] == "SELL" and in_position:
                exit_price = float(row["Close"])
                exit_date = row.name
                pnl = (exit_price - entry_price) / entry_price * 100
                capital = position * exit_price
                trades.append({
                    "EntryDate": entry_date,
                    "ExitDate": exit_date,
                    "EntryPrice": round(entry_price, 2),
                    "ExitPrice": round(exit_price, 2),
                    "PnL%": round(pnl, 2),
                    "Result": "WIN" if pnl > 0 else "LOSS",
                    "BarsHeld": (exit_date - entry_date).days
                })
                position = 0
                in_position = False

        # Close open position at end
        if in_position:
            exit_price = float(df["Close"].iloc[-1])
            exit_date = df.index[-1]
            pnl = (exit_price - entry_price) / entry_price * 100
            capital = position * exit_price
            trades.append({
                "EntryDate": entry_date,
                "ExitDate": exit_date,
                "EntryPrice": round(entry_price, 2),
                "ExitPrice": round(exit_price, 2),
                "PnL%": round(pnl, 2),
                "Result": "WIN" if pnl > 0 else "LOSS",
                "BarsHeld": (exit_date - entry_date).days
            })

        # --- Compute Metrics ---
        total_return_pct = ((capital - initial_capital) / initial_capital) * 100
        total_trades = len(trades)
        wins = [t for t in trades if t["Result"] == "WIN"]
        losses = [t for t in trades if t["Result"] == "LOSS"]
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean([t["PnL%"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["PnL%"] for t in losses]) if losses else 0
        profit_factor = abs(sum(t["PnL%"] for t in wins) / sum(t["PnL%"] for t in losses)) if losses and sum(t["PnL%"] for t in losses) != 0 else float("inf")

        # Equity curve
        df["Strategy_Returns"] = 0.0
        for t in trades:
            exit_loc = df.index.get_loc(t["ExitDate"])
            df.loc[df.index[exit_loc], "Strategy_Returns"] = t["PnL%"] / 100

        df["Equity"] = initial_capital * (1 + df["Strategy_Returns"]).cumprod()
        running_max = df["Equity"].cummax()
        df["Drawdown"] = (df["Equity"] - running_max) / running_max * 100
        max_drawdown = round(df["Drawdown"].min(), 2)

        # Sharpe ratio (assuming 252 trading days)
        daily_returns = df["Strategy_Returns"]
        sharpe = 0
        if daily_returns.std() > 0:
            sharpe = round((daily_returns.mean() / daily_returns.std()) * np.sqrt(252), 2)

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        sortino = 0
        if len(downside) > 0 and downside.std() > 0:
            sortino = round((daily_returns.mean() / downside.std()) * np.sqrt(252), 2)

        # Expectancy
        expectancy = round((win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss)), 2) if total_trades > 0 else 0

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "Total Return %": round(total_return_pct, 2),
            "Total Trades": total_trades,
            "Win Rate %": round(win_rate, 1),
            "Avg Win %": round(avg_win, 2),
            "Avg Loss %": round(avg_loss, 2),
            "Profit Factor": round(profit_factor, 2),
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown %": max_drawdown,
            "Expectancy": expectancy,
            "Final Capital": round(capital, 2),
            "Trades": trades_df
        }

    except Exception as e:
        print(f"Backtest Error: {e}")
        return None