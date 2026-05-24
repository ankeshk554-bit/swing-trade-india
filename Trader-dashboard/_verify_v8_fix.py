"""V8 Fix Verification — ABB.NS + LT.NS"""
import sys, json
sys.path.insert(0, "c:/Users/Ankes/OneDrive/Desktop/AI projects/trader dashboard/Trader-dashboard")

from core.backtest import run_backtest

results = {}
for sym in ["ABB.NS", "LT.NS"]:
    print(f"\nRunning {sym}...")
    r = run_backtest(sym, initial_capital=100000)
    if r is None or r.get("Total Trades", 0) == 0:
        print(f"  No trades for {sym}")
        continue

    trades = r.get("Trades", [])
    if hasattr(trades, 'to_dict'):
        trades = trades.to_dict('records')

    wins = [t for t in trades if t["Result"] == "WIN"]
    losses = [t for t in trades if t["Result"] == "LOSS"]
    wc = len(wins); lc = len(losses); n = len(trades)
    wr = wc / n * 100 if n > 0 else 0
    aw = sum(t["PnL%"] for t in wins) / max(wc, 1) if wins else 0
    al = abs(sum(t["PnL%"] for t in losses) / max(lc, 1)) if losses else 0
    payoff = aw / max(al, 0.01)
    expectancy = (wr / 100 * aw) - ((1 - wr / 100) * al)
    pf = sum(t["PnL%"] for t in wins) / max(abs(sum(t["PnL%"] for t in losses)), 0.01) if losses else float('inf')

    partial_trades = [t for t in trades if t.get("Reason") == "Partial40%"]
    sellsig_trades = [t for t in trades if t.get("Reason") == "SellSig"]
    trail_trades = [t for t in trades if t.get("Reason") in ("ChandelierTrail", "TrailStop", "InitialStop")]

    # Bug check: trades with partial that still lost money
    partial_entry_dates = set()
    for t in partial_trades:
        partial_entry_dates.add(t.get("EntryDate"))
    be_bug_trades = []
    for t in trades:
        if t.get("EntryDate") in partial_entry_dates and t["Result"] == "LOSS" and t.get("Reason") != "Partial40%":
            be_bug_trades.append(t)

    sellsig_avg = sum(t["PnL%"] for t in sellsig_trades) / max(len(sellsig_trades), 1) if sellsig_trades else 0
    trail_avg = sum(t["PnL%"] for t in trail_trades) / max(len(trail_trades), 1) if trail_trades else 0
    partial_rate = len(partial_trades) / max(n, 1)

    results[sym] = {
        "n": n, "wr": round(wr, 1), "aw": round(aw, 2), "al": round(al, 2),
        "payoff": round(payoff, 2), "expectancy": round(expectancy, 2),
        "pf": round(pf, 2), "total_pnl": round(r.get("Total Return %", 0), 2),
        "maxdd": round(r.get("Max Drawdown %", 0), 2),
        "sharpe": round(r.get("Sharpe Ratio", 0), 2),
        "partial_rate": round(partial_rate, 3),
        "sellsig_count": len(sellsig_trades), "sellsig_avg": round(sellsig_avg, 2),
        "trail_count": len(trail_trades), "trail_avg": round(trail_avg, 2),
        "be_bug_count": len(be_bug_trades),
    }

# Print report
print("\n" + "=" * 70)
print("  V8 FIXED REPORT")
print("=" * 70)
all_pass = True
for sym, res in results.items():
    print(f"\n  Stock          : {sym}")
    print(f"  Trades         : {res['n']}")
    print(f"  Win Rate       : {res['wr']}%")
    print(f"  Avg Win %      : {res['aw']}%     [target: >20%]")
    print(f"  Avg Loss %     : {res['al']}%")
    print(f"  Payoff Ratio   : {res['payoff']}")
    print(f"  Expectancy     : {res['expectancy']}%     [target: >10%]")
    print(f"  Profit Factor  : {res['pf']}      [target: >3.0]")
    print(f"  Total Return % : {res['total_pnl']}%")
    print(f"  Max Drawdown   : {res['maxdd']}%")
    print(f"  Sharpe         : {res['sharpe']}      [target: >2.0]")
    print(f"  Partial Rate   : {res['partial_rate']*100:.0f}%     [target: >50%]")
    print(f"  SellSig exits  : {res['sellsig_count']} trades, avg {res['sellsig_avg']}%")
    print(f"  TrailStop exits: {res['trail_count']} trades, avg {res['trail_avg']}%")
    print(f"  BE Bug trades  : {res['be_bug_count']}     [target: 0]")

# Assertions
print("\n" + "-" * 70)
print("  ASSERTIONS")
print("-" * 70)
if "ABB.NS" in results:
    r = results["ABB.NS"]
    a1 = r["aw"] > 20.0; print(f"  1. AvgWin% > 20% : {'PASS' if a1 else 'FAIL'} ({r['aw']}%)")
    a2 = r["pf"] > 3.0;  print(f"  2. PF > 3.0      : {'PASS' if a2 else 'FAIL'} ({r['pf']})")
    a3 = r["sharpe"] > 2.0; print(f"  3. Sharpe > 2.0  : {'PASS' if a3 else 'FAIL'} ({r['sharpe']})")
    a4 = r["be_bug_count"] == 0; print(f"  4. No BE bugs   : {'PASS' if a4 else 'FAIL'} ({r['be_bug_count']} trades)")
    a5 = r["partial_rate"] >= 0.50; print(f"  5. Partial > 50% : {'PASS' if a5 else 'FAIL'} ({r['partial_rate']*100:.0f}%)")
    a6 = r["sellsig_avg"] > 15.0 if r["sellsig_count"] > 0 else True; print(f"  6. SellSig > 15% : {'PASS' if a6 else 'FAIL'} ({r['sellsig_avg']}%)")
    if not all([a1, a2, a3, a4, a5, a6]):
        all_pass = False

if "LT.NS" in results:
    r = results["LT.NS"]
    a4b = r["be_bug_count"] == 0
    a5b = r["partial_rate"] >= 0.50
    print(f"  4b. LT No BE bugs: {'PASS' if a4b else 'FAIL'} ({r['be_bug_count']} trades)")
    print(f"  5b. LT Partial>50%: {'PASS' if a5b else 'FAIL'} ({r['partial_rate']*100:.0f}%)")
    if not a4b or not a5b:
        all_pass = False

print("\n" + "=" * 70)
print(f"  {'✅ ALL ASSERTIONS PASS' if all_pass else '❌ SOME ASSERTIONS FAILED'}")
print("=" * 70)
