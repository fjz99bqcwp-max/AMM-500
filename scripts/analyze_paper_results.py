#!/usr/bin/env python3
"""
Analyze Paper Trading Results - 7 Day Performance Evaluation

This script analyzes the results of a 7-day paper trading session and compares
them against target metrics for HFT market making.

Usage:
    python scripts/analyze_paper_results.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATE_FILE = PROJECT_ROOT / "logs" / "autonomous_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / f"bot_{datetime.now().strftime('%Y-%m-%d')}.log"


def load_state() -> Dict:
    """Load autonomous state from JSON."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ State file not found: {STATE_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in state file")
        sys.exit(1)


def calculate_sharpe_ratio(pnl_series: List[float]) -> float:
    """Calculate Sharpe ratio from PnL series."""
    if not pnl_series or len(pnl_series) < 2:
        return 0.0
    
    returns = np.diff(pnl_series)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Annualized Sharpe (assuming daily PnL)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0
    
    return sharpe


def analyze_results():
    """Analyze 7-day paper trading results."""
    state = load_state()
    
    # Extract metrics
    total_trades = state.get("total_trades", 0)
    session_pnl = state.get("cumulative_pnl", 0)
    total_fees = state.get("total_fees_paid", 0)
    session_start = state.get("session_start_time", 0)
    
    # Trade breakdown
    session_trades = state.get("session_trades", [])
    buy_trades = sum(1 for t in session_trades if t.get("side") == "B")
    sell_trades = sum(1 for t in session_trades if t.get("side") == "S")
    
    # Maker/Taker
    maker_trades = sum(1 for t in session_trades if not t.get("crossed", False))
    taker_trades = sum(1 for t in session_trades if t.get("crossed", False))
    
    # Calculate duration
    if session_start:
        duration_ms = (datetime.now().timestamp() * 1000) - session_start
        duration_days = duration_ms / (1000 * 3600 * 24)
    else:
        duration_days = 7  # Assume 7 days if not tracked
    
    # Calculated metrics
    trades_per_day = total_trades / duration_days if duration_days > 0 else 0
    maker_ratio = maker_trades / total_trades if total_trades > 0 else 0
    net_pnl = session_pnl - total_fees
    
    # ROI calculation (assuming $1000 collateral)
    collateral = 1000
    roi_7d = (net_pnl / collateral) * 100
    roi_annual = roi_7d * (365 / duration_days)
    
    # Max drawdown (from peak equity)
    wallet_history = state.get("wallet_history", [])
    if wallet_history:
        equities = [w.get("equity", 0) for w in wallet_history]
        peak = max(equities) if equities else collateral
        trough = min(equities) if equities else collateral
        max_dd_pct = ((peak - trough) / peak) * 100 if peak > 0 else 0
    else:
        max_dd_pct = 0
    
    # Sharpe ratio (if we have daily PnL data)
    pnl_history = state.get("pnl_history", [])
    sharpe = calculate_sharpe_ratio(pnl_history) if pnl_history else 0
    
    # Spread metrics
    avg_spread = state.get("avg_spread_bps", 0)
    
    # Print results
    print("=" * 80)
    print("📊 7-DAY PAPER TRADING RESULTS")
    print("=" * 80)
    print()
    
    print("⏱️  DURATION")
    print(f"  Days Trading: {duration_days:.1f}")
    print(f"  Start: {datetime.fromtimestamp(session_start/1000).strftime('%Y-%m-%d %H:%M') if session_start else 'Unknown'}")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    print("📈 TRADING ACTIVITY")
    print(f"  Total Trades: {total_trades} ({trades_per_day:.0f}/day)")
    print(f"  Buy Trades: {buy_trades}")
    print(f"  Sell Trades: {sell_trades}")
    print(f"  Maker Trades: {maker_trades} ({maker_ratio:.1%})")
    print(f"  Taker Trades: {taker_trades} ({(1-maker_ratio):.1%})")
    print()
    
    print("💰 PROFIT & LOSS")
    print(f"  Gross PnL: ${session_pnl:+.2f}")
    print(f"  Fees Paid: ${total_fees:.2f}")
    print(f"  Net PnL: ${net_pnl:+.2f}")
    print(f"  ROI (7d): {roi_7d:+.2f}%")
    print(f"  ROI (Annual): {roi_annual:+.1f}%")
    print()
    
    print("📊 RISK METRICS")
    print(f"  Max Drawdown: {max_dd_pct:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Avg Spread: {avg_spread:.2f} bps")
    print()
    
    # Target comparison
    print("=" * 80)
    print("🎯 TARGET COMPARISON")
    print("=" * 80)
    print()
    
    targets = {
        "Trades/Day": (trades_per_day, 1000, ">"),
        "Maker Ratio": (maker_ratio * 100, 90, ">"),
        "ROI (7d)": (roi_7d, 5, ">"),
        "Max Drawdown": (max_dd_pct, 0.5, "<"),
        "Sharpe Ratio": (sharpe, 1.5, ">")
    }
    
    for metric, (actual, target, comparison) in targets.items():
        if comparison == ">":
            status = "✅" if actual >= target else "❌"
            comp_str = ">="
        else:  # "<"
            status = "✅" if actual <= target else "❌"
            comp_str = "<="
        
        print(f"  {status} {metric:15s}: {actual:8.2f} (target: {comp_str} {target})")
    
    print()
    
    # Overall assessment
    print("=" * 80)
    print("📋 OVERALL ASSESSMENT")
    print("=" * 80)
    print()
    
    passed = sum(1 for _, (a, t, c) in targets.items() if (a >= t if c == ">" else a <= t))
    total = len(targets)
    
    if passed == total:
        grade = "🏆 EXCELLENT - All targets met!"
    elif passed >= total * 0.8:
        grade = "✅ GOOD - Most targets met"
    elif passed >= total * 0.6:
        grade = "⚠️  ACCEPTABLE - Some targets missed"
    else:
        grade = "❌ NEEDS IMPROVEMENT - Many targets missed"
    
    print(f"  {grade}")
    print(f"  Targets Met: {passed}/{total}")
    print()
    
    # Recommendations
    print("=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if trades_per_day < 1000:
        print("  ⚠️  Low trade volume - Consider:")
        print("     - Tightening spreads (reduce min_spread_bps)")
        print("     - Increasing order levels")
        print("     - Reducing quote refresh interval")
    
    if maker_ratio < 0.9:
        print("  ⚠️  High taker ratio - Consider:")
        print("     - Using ALO (add liquidity only) orders exclusively")
        print("     - Widening spreads slightly")
        print("     - Improving quote placement logic")
    
    if roi_7d < 5:
        print("  ⚠️  Low profitability - Consider:")
        print("     - Increasing leverage (if drawdown is low)")
        print("     - Optimizing spread capture")
        print("     - Reducing fees through better maker ratio")
    
    if max_dd_pct > 0.5:
        print("  ⚠️  High drawdown - Consider:")
        print("     - Reducing leverage")
        print("     - Tightening risk management (faster rebalancing)")
        print("     - Implementing stop-loss earlier")
    
    if sharpe < 1.5:
        print("  ⚠️  Low risk-adjusted returns - Consider:")
        print("     - Reducing position size during high volatility")
        print("     - Improving spread adaptation")
        print("     - Better inventory management")
    
    if passed == total:
        print("  ✅ Performance is excellent - Ready for live trading!")
        print("     Next steps:")
        print("     1. Review logs for any anomalies")
        print("     2. Start with $1000 live (10x leverage)")
        print("     3. Monitor for first week actively")
        print("     4. Scale up gradually after 2-4 weeks")
    
    print()
    print("=" * 80)
    
    # Return summary for programmatic access
    return {
        "total_trades": total_trades,
        "trades_per_day": trades_per_day,
        "maker_ratio": maker_ratio,
        "net_pnl": net_pnl,
        "roi_7d": roi_7d,
        "roi_annual": roi_annual,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "targets_met": passed,
        "total_targets": total,
        "grade": grade
    }


if __name__ == "__main__":
    try:
        results = analyze_results()
        sys.exit(0 if results["targets_met"] >= results["total_targets"] * 0.8 else 1)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
