"""Test v5 and save results to JSON"""
import json, sys
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.profit_engine_v5 import run_v5_backtest

r = run_v5_backtest("ABB.NS", capital=100000)

output = {
    "Version": r.get("Version"),
    "Trades": r.get("Total Trades", 0),
    "Return": r.get("Total Return %", 0),
    "PF": r.get("Profit Factor", 0),
    "Sharpe": r.get("Sharpe Ratio", 0),
    "WinRate": r.get("Win Rate %", 0),
    "MaxDD": r.get("Max Drawdown %", 0),
    "WeeklyTrend": r.get("Weekly Trend"),
    "Regime": r.get("Market Regime"),
    "HMM": r.get("HMM Regime"),
    "Kelly": r.get("Kelly Fraction", 0),
    "DeflatedSharpe": r.get("Deflated Sharpe", 0),
    "Advanced": r.get("Advanced", {}),
    "WalkForward": r.get("WalkForward", {}),
}

with open("_v5_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("Results saved to _v5_results.json")
