"""Test v5 profit engine on ABB.NS"""
from core.profit_engine_v5 import run_v5_backtest, compute_advanced_metrics

r = run_v5_backtest("ABB.NS", capital=100000)
print(f"Version: {r.get('Version', '?')}")
print(f"Trades: {r.get('Total Trades', 0)}")
print(f"Return: {r.get('Total Return %', 0):+.2f}%")
print(f"PF: {r.get('Profit Factor', 0):.2f}")
print(f"Sharpe: {r.get('Sharpe Ratio', 0):.2f}")
print(f"Win Rate: {r.get('Win Rate %', 0):.1f}%")
print(f"MaxDD: {r.get('Max Drawdown %', 0):.2f}%")
print(f"Weekly Trend: {r.get('Weekly Trend', '?')}")
print(f"Regime: {r.get('Market Regime', '?')}")
print(f"HMM Regime: {r.get('HMM Regime', '?')}")
print(f"Kelly: {r.get('Kelly Fraction', 0)*100:.1f}%")
print(f"Deflated Sharpe: {r.get('Deflated Sharpe', 0)}")
adv = r.get("Advanced", {})
if adv:
    print(f"\nAdvanced Metrics:")
    for k, v in adv.items():
        if k != "Exit Breakdown":
            print(f"  {k}: {v}")
wf = r.get("WalkForward", {})
if wf and "Folds" in wf:
    print(f"\nWalk-Forward: {len(wf['Folds'])} folds")
    print(f"  Avg IS Sharpe: {wf.get('Avg IS Sharpe', '?')}")
    print(f"  Avg OOS Sharpe: {wf.get('Avg OOS Sharpe', '?')}")
    print(f"  OOS/IS Ratio: {wf.get('OOS/IS Ratio %', '?')}%")
    print(f"  Overfit: {wf.get('Overfit Flag', '?')}")
print("\nDone!")
