"""Quick v5 smoke test - just entry/exit, skip walk-forward"""
import sys
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.profit_engine_v5 import run_v5_backtest, compute_advanced_metrics

# Run with minimal features to test quickly
r = run_v5_backtest(
    "ABB.NS",
    capital=100000,
    use_volume_filter=True,
    use_rs_filter=False,
    use_52w_filter=True,
    use_ema_stack=True,
    use_vwap_filter=True,
    use_breadth_gate=False,
    use_chandelier=True,
    use_vol_exhaust=False,
    use_gap_up_exit=False,
    use_time_stop=True,
    use_portfolio_heat=False,
    use_correlation_filter=False,
    use_circuit_breaker=False,
    use_dynamic_kelly=False,
    use_anti_martingale=False,
)
print(f"VERSION: {r.get('Version', '?')}")
print(f"TRADES: {r.get('Total Trades', 0)}")
print(f"RETURN: {r.get('Total Return %', 0):+.2f}%")
print(f"PF: {r.get('Profit Factor', 0):.2f}")
print(f"SHARPE: {r.get('Sharpe Ratio', 0):.2f}")
print(f"WINRATE: {r.get('Win Rate %', 0):.1f}%")
print(f"MAXDD: {r.get('Max Drawdown %', 0):.2f}%")
print(f"ERROR: {r.get('Error', 'None')}")

adv = r.get("Advanced", {})
if adv:
    print(f"\nADVANCED METRICS:")
    for k, v in adv.items():
        if k != "Exit Breakdown":
            print(f"  {k}: {v}")

print("\nDONE")
