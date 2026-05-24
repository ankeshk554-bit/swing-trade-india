"""Debug v5 - test entry filters individually"""
import sys
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.profit_engine_v5 import run_v5_backtest, EntryFilters
from core.utils import load_data
from core.indicators import compute_indicators

# Test with minimal filters to see if engine works at all
r = run_v5_backtest(
    "ABB.NS",
    capital=100000,
    use_volume_filter=True,
    use_rs_filter=False,
    use_52w_filter=False,
    use_ema_stack=False,
    use_vwap_filter=False,
    use_breadth_gate=False,
    use_chandelier=False,
    use_vol_exhaust=False,
    use_gap_up_exit=False,
    use_time_stop=True,
    use_portfolio_heat=False,
    use_correlation_filter=False,
    use_circuit_breaker=False,
    use_dynamic_kelly=False,
    use_anti_martingale=False,
)
print(f"V5 TRADES: {r.get('Total Trades', 0)}")
print(f"V5 RETURN: {r.get('Total Return %', 0):+.2f}%")
print(f"V5 PF: {r.get('Profit Factor', 0):.2f}")
print(f"V5 Sharpe: {r.get('Sharpe Ratio', 0):.2f}")
print(f"V5 WinRate: {r.get('Win Rate %', 0):.1f}%")
print(f"V5 MaxDD: {r.get('Max Drawdown %', 0):.2f}%")
print(f"V5 Error: {r.get('Error', 'None')}")
print(f"V5 Regime: {r.get('Market Regime', '?')}")
print(f"V5 Weekly: {r.get('Weekly Trend', '?')}")

# Test with all filters ON
r2 = run_v5_backtest(
    "ABB.NS",
    capital=100000,
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
    use_portfolio_heat=True,
    use_correlation_filter=True,
    use_circuit_breaker=True,
    use_dynamic_kelly=True,
    use_anti_martingale=True,
)
print(f"\n--- ALL FILTERS ON ---")
print(f"V5 TRADES: {r2.get('Total Trades', 0)}")
print(f"V5 RETURN: {r2.get('Total Return %', 0):+.2f}%")
print(f"V5 PF: {r2.get('Profit Factor', 0):.2f}")
print(f"V5 Error: {r2.get('Error', 'None')}")
print("DONE")
