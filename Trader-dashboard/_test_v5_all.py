"""Test v5 with ALL filters ON"""
import sys
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
print(f"ALL FILTERS ON:")
print(f"TRADES: {r.get('Total Trades', 0)}")
print(f"RETURN: {r.get('Total Return %', 0):+.2f}%")
print(f"PF: {r.get('Profit Factor', 0):.2f}")
print(f"SHARPE: {r.get('Sharpe Ratio', 0):.2f}")
print(f"WINRATE: {r.get('Win Rate %', 0):.1f}%")
print(f"MAXDD: {r.get('Max Drawdown %', 0):.2f}%")
print(f"ERROR: {r.get('Error', 'None')}")
print(f"HMM: {r.get('HMM Regime', '?')}")
print(f"Defl Sharpe: {r.get('Deflated Sharpe', 0)}")

adv = r.get("Advanced", {})
if adv:
    for k, v in adv.items():
        if k != "Exit Breakdown":
            print(f"  {k}: {v}")
print("DONE")
