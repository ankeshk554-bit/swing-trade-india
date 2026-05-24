"""v5 quick test with all entry filters, skip walk-forward"""
import sys, json
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.profit_engine_v5 import run_v5_backtest

# Just the entry/exit filters - no portfolio heat/circuit breaker (those need walk-forward)
r = run_v5_backtest(
    "ABB.NS", capital=100000,
    use_volume_filter=True,
    use_rs_filter=True,
    use_52w_filter=True,
    use_ema_stack=True,
    use_vwap_filter=True,
    use_breadth_gate=True,
    use_chandelier=True,
    use_vol_exhaust=True,
    use_gap_up_exit=True,
    use_time_stop=True,
    use_portfolio_heat=False,
    use_correlation_filter=False,
    use_circuit_breaker=False,
    use_dynamic_kelly=False,
    use_anti_martingale=False,
)

out = {
    "trades": r.get("Total Trades", 0),
    "return": round(r.get("Total Return %", 0), 2),
    "pf": r.get("Profit Factor", 0),
    "sharpe": r.get("Sharpe Ratio", 0),
    "winrate": r.get("Win Rate %", 0),
    "maxdd": r.get("Max Drawdown %", 0),
    "hmm": r.get("HMM Regime"),
    "weekly": r.get("Weekly Trend"),
}
with open("_v5_res.json", "w") as f:
    json.dump(out, f)
print("DONE")
