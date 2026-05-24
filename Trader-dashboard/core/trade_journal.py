"""
Trade Journal & Portfolio Tracker — Sniper Terminal
====================================================
Persistent trade journal with SQLite-like JSON storage.
Tracks open positions, closed trades, and portfolio metrics.

Data stored in .data/trade_journal.json
"""

import json
import uuid
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / ".data"
DATA_DIR.mkdir(exist_ok=True)
JOURNAL_FILE = DATA_DIR / "trade_journal.json"


def _load_trades() -> list:
    """Load all trades from disk."""
    if JOURNAL_FILE.exists():
        try:
            with open(JOURNAL_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Journal load failed: {e}")
    return []


def _save_trades(trades: list):
    """Save all trades to disk."""
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Journal save failed: {e}")


def _get_live_price(symbol: str) -> Optional[float]:
    """Get latest price for a symbol."""
    from core.utils import load_data
    try:
        df = load_data(symbol, period="5d")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# TRADE CRUD
# ──────────────────────────────────────────────

def add_trade(
    symbol: str,
    entry_price: float,
    quantity: int,
    direction: str = "LONG",
    entry_date: str = None,
    stop_loss: float = None,
    target: float = None,
    tags: list = None,
    notes: str = "",
) -> dict:
    """Add a new trade to the journal."""
    trades = _load_trades()
    trade = {
        "id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "entry_date": entry_date or str(date.today()),
        "exit_date": None,
        "entry_price": entry_price,
        "exit_price": None,
        "quantity": quantity,
        "stop_loss": stop_loss,
        "target": target,
        "direction": direction.upper(),
        "tags": tags or [],
        "notes": notes,
        "result": "OPEN",
        "pnl": None,
        "pnl_pct": None,
        "created_at": datetime.now().isoformat()
    }
    trades.append(trade)
    _save_trades(trades)
    return trade


def close_trade(trade_id: str, exit_price: float, exit_date: str = None) -> bool:
    """Close an open trade."""
    trades = _load_trades()
    for t in trades:
        if t["id"] == trade_id and t["result"] == "OPEN":
            t["exit_date"] = exit_date or str(date.today())
            t["exit_price"] = exit_price
            t["pnl"] = round(
                (exit_price - t["entry_price"]) * t["quantity"]
                if t["direction"] == "LONG"
                else (t["entry_price"] - exit_price) * t["quantity"],
                2
            )
            t["pnl_pct"] = round(
                ((exit_price / t["entry_price"]) - 1) * 100
                if t["direction"] == "LONG"
                else ((t["entry_price"] / exit_price) - 1) * 100,
                2
            )
            t["result"] = "WIN" if t["pnl"] > 0 else "LOSS"
            _save_trades(trades)
            return True
    return False


def delete_trade(trade_id: str) -> bool:
    """Delete a trade from the journal."""
    trades = _load_trades()
    filtered = [t for t in trades if t["id"] != trade_id]
    if len(filtered) < len(trades):
        _save_trades(filtered)
        return True
    return False


def update_trade(trade_id: str, updates: dict) -> bool:
    """Update fields on an existing trade."""
    trades = _load_trades()
    for t in trades:
        if t["id"] == trade_id:
            for k, v in updates.items():
                if k in t and k != "id":
                    t[k] = v
            _save_trades(trades)
            return True
    return False


# ──────────────────────────────────────────────
# QUERIES
# ──────────────────────────────────────────────

def get_all_trades() -> pd.DataFrame:
    """Get all trades as a DataFrame."""
    trades = _load_trades()
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


def get_open_trades() -> pd.DataFrame:
    """Get all open trades with live P&L."""
    trades = _load_trades()
    open_trades = [t for t in trades if t["result"] == "OPEN"]

    if not open_trades:
        return pd.DataFrame()

    for t in open_trades:
        live_price = _get_live_price(t["symbol"])
        if live_price:
            if t["direction"] == "LONG":
                t["live_pnl"] = round((live_price - t["entry_price"]) * t["quantity"], 2)
                t["live_pnl_pct"] = round(((live_price / t["entry_price"]) - 1) * 100, 2)
            else:
                t["live_pnl"] = round((t["entry_price"] - live_price) * t["quantity"], 2)
                t["live_pnl_pct"] = round(((t["entry_price"] / live_price) - 1) * 100, 2)
            t["live_price"] = round(live_price, 2)
        else:
            t["live_price"] = t["entry_price"]
            t["live_pnl"] = 0
            t["live_pnl_pct"] = 0

    return pd.DataFrame(open_trades)


def get_closed_trades() -> pd.DataFrame:
    """Get all closed trades."""
    trades = _load_trades()
    closed = [t for t in trades if t["result"] in ("WIN", "LOSS")]
    if not closed:
        return pd.DataFrame()
    return pd.DataFrame(closed)


# ──────────────────────────────────────────────
# PORTFOLIO METRICS
# ──────────────────────────────────────────────

def get_portfolio_summary(initial_capital: float = 0) -> dict:
    """
    Get comprehensive portfolio summary.

    Returns dict with:
      - total_trades, win_rate, profit_factor, avg_win, avg_loss
      - open_pnl, realized_pnl, total_pnl
      - total_exposure, position_count
      - monthly_returns, equity_curve
    """
    all_trades = _load_trades()
    open_trades = [t for t in all_trades if t["result"] == "OPEN"]
    closed_trades = [t for t in all_trades if t["result"] in ("WIN", "LOSS")]

    # Open positions
    total_exposure = 0
    total_open_pnl = 0
    for t in open_trades:
        entry_val = t["entry_price"] * t["quantity"]
        total_exposure += entry_val
        live_price = _get_live_price(t["symbol"]) or t["entry_price"]
        if t["direction"] == "LONG":
            pnl = (live_price - t["entry_price"]) * t["quantity"]
        else:
            pnl = (t["entry_price"] - live_price) * t["quantity"]
        total_open_pnl += pnl

    # Closed trades
    wins = [t for t in closed_trades if t["result"] == "WIN"]
    losses = [t for t in closed_trades if t["result"] == "LOSS"]
    total_trades = len(closed_trades)
    win_count = len(wins)
    loss_count = len(losses)

    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = np.mean([t.get("pnl_pct", 0) for t in wins]) if wins else 0
    avg_loss = np.mean([t.get("pnl_pct", 0) for t in losses]) if losses else 0

    total_profit = sum(t.get("pnl", 0) for t in wins) if wins else 0
    total_loss = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 0
    profit_factor = round(total_profit / max(total_loss, 1), 2)
    realized_pnl = round(sum(t.get("pnl", 0) for t in closed_trades), 2)

    # Max consecutive wins/losses
    consec_wins = consec_losses = 0
    max_consec_wins = max_consec_losses = 0
    for t in closed_trades:
        if t["result"] == "WIN":
            consec_wins += 1
            consec_losses = 0
            max_consec_wins = max(max_consec_wins, consec_wins)
        else:
            consec_losses += 1
            consec_wins = 0
            max_consec_losses = max(max_consec_losses, consec_losses)

    # Best / Worst trade
    best_trade = max(closed_trades, key=lambda t: t.get("pnl_pct", 0)) if closed_trades else None
    worst_trade = min(closed_trades, key=lambda t: t.get("pnl_pct", 0)) if closed_trades else None

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": profit_factor,
        "realized_pnl": realized_pnl,
        "open_pnl": round(total_open_pnl, 2),
        "total_pnl": round(realized_pnl + total_open_pnl, 2),
        "total_exposure": round(total_exposure, 2),
        "open_positions": len(open_trades),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "best_trade_pct": round(best_trade.get("pnl_pct", 0), 2) if best_trade else 0,
        "best_trade_symbol": best_trade["symbol"].replace(".NS", "") if best_trade else "—",
        "worst_trade_pct": round(worst_trade.get("pnl_pct", 0), 2) if worst_trade else 0,
        "worst_trade_symbol": worst_trade["symbol"].replace(".NS", "") if worst_trade else "—",
    }


def get_monthly_returns() -> pd.DataFrame:
    """Get monthly P&L breakdown."""
    closed = get_closed_trades()
    if closed.empty:
        return pd.DataFrame()

    closed["exit_date"] = pd.to_datetime(closed["exit_date"])
    closed["month"] = closed["exit_date"].dt.to_period("M")
    monthly = closed.groupby("month").agg(
        trades=("id", "count"),
        wins=("result", lambda x: sum(1 for r in x if r == "WIN")),
        pnl=("pnl", "sum")
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly["win_rate"] = round(monthly["wins"] / monthly["trades"] * 100, 1)
    return monthly


def export_journal() -> Optional[str]:
    """Export all trades as CSV."""
    df = get_all_trades()
    if not df.empty:
        return df.to_csv(index=False)
    return None
