"""v5 with all filters - write to file"""
import sys, json, datetime
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.profit_engine_v5 import run_v5_backtest

r = run_v5_backtest(
    "ABB.NS", capital=100000,
    use_volume_filter=True, use_rs_filter=True,
    use_52w_filter=True, use_ema_stack=True,
    use_vwap_filter=True, use_breadth_gate=True,
    use_chandelier=True, use_vol_exhaust=True,
    use_gap_up_exit=True, use_time_stop=True,
    use_portfolio_heat=True, use_correlation_filter=True,
    use_circuit_breaker=True, use_dynamic_kelly=True,
    use_anti_martingale=True,
)

out = {
    "Trades": r.get("Total Trades", 0),
    "Return": r.get("Total Return %", 0),
    "PF": r.get("Profit Factor", 0),
    "Sharpe": r.get("Sharpe Ratio", 0),
    "WinRate": r.get("Win Rate %", 0),
    "MaxDD": r.get("Max Drawdown %", 0),
    "DeflSharpe": r.get("Deflated Sharpe", 0),
    "HMM": r.get("HMM Regime"),
    "Advanced": {k: v for k, v in r.get("Advanced", {}).items() if k != "Exit Breakdown"},
}
with open("_v5_all_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("SAVED")
