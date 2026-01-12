#!/usr/bin/env python3
"""
Auto-Optimization Engine

Automatically adjusts strategy parameters based on performance
"""

import os
import sys
import re
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

STRATEGY_FILE = "/Users/nheosdisplay/VSC/MMB/MMB-1/src/strategy.py"


class StrategyOptimizer:
    """Automatically optimize strategy parameters."""

    def __init__(self):
        self.optimizations_applied = []

    def read_strategy_file(self) -> str:
        """Read current strategy code."""
        with open(STRATEGY_FILE, "r") as f:
            return f.read()

    def write_strategy_file(self, content: str):
        """Write updated strategy code."""
        # Backup first
        import shutil
        from datetime import datetime

        backup_file = f"{STRATEGY_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(STRATEGY_FILE, backup_file)
        print(f"📦 Backup created: {backup_file}")

        # Write new content
        with open(STRATEGY_FILE, "w") as f:
            f.write(content)
        print(f"✅ Strategy file updated")

    def adjust_defensive_distance(self, current_spread_bps: float) -> Dict:
        """Adjust defensive distance based on spread performance."""
        content = self.read_strategy_file()

        # Find current defensive_distance assignments
        pattern = r"defensive_distance = ([\d.]+)"

        if current_spread_bps < -2.0:
            # Severe adverse selection - increase to $6
            new_distance = 6.0
            reason = f"Adverse selection detected ({current_spread_bps:.2f} bps)"
        elif current_spread_bps < 0:
            # Mild adverse selection - increase to $4
            new_distance = 4.0
            reason = f"Negative spread ({current_spread_bps:.2f} bps)"
        elif current_spread_bps < 2.0:
            # Low profitability - moderate distance
            new_distance = 3.5
            reason = f"Low profitability ({current_spread_bps:.2f} bps)"
        else:
            # Good performance - keep normal
            new_distance = 2.0
            reason = f"Good performance ({current_spread_bps:.2f} bps)"

        # Update the base defensive_distance (currently in 3 places)
        # 1. Good profitability case
        content = re.sub(
            r"# Good profitability or no data yet\s+defensive_distance = ([\d.]+)",
            f"# Good profitability or no data yet\n            defensive_distance = {new_distance}",
            content,
        )

        self.write_strategy_file(content)

        return {"parameter": "defensive_distance", "new_value": new_distance, "reason": reason}

    def adjust_min_spread(self, fill_rate_per_hour: int, current_spread_bps: float) -> Dict:
        """Adjust minimum spread based on fill rate and profitability."""
        content = self.read_strategy_file()

        # Find min_spread_bps in _calculate_spread method
        pattern = r"min_spread_bps=max\(self\.min_spread_bps, ([\d.]+)\)"

        current_min = 8.0  # OPT#14 default

        if current_spread_bps < 0 and fill_rate_per_hour > 20:
            # Losing money with high fill rate - widen spreads
            new_min = 10.0
            reason = f"High losses ({current_spread_bps:.2f} bps, {fill_rate_per_hour} fills/hr)"
        elif fill_rate_per_hour < 5:
            # Very low fill rate - tighten spreads
            new_min = 7.0
            reason = f"Low fill rate ({fill_rate_per_hour} fills/hr)"
        elif current_spread_bps > 10.0 and fill_rate_per_hour > 15:
            # Excellent performance - can tighten slightly for more volume
            new_min = 7.5
            reason = f"Excellent performance ({current_spread_bps:.2f} bps, {fill_rate_per_hour} fills/hr)"
        else:
            # Keep current
            new_min = current_min
            reason = "No change needed"

        if new_min != current_min:
            content = re.sub(
                pattern, f"min_spread_bps=max(self.min_spread_bps, {new_min})", content
            )
            self.write_strategy_file(content)

        return {
            "parameter": "min_spread_bps",
            "new_value": new_min,
            "reason": reason,
            "changed": new_min != current_min,
        }

    def adjust_order_levels(self, current_spread_bps: float, volatility: float) -> Dict:
        """Adjust order levels based on performance."""
        # This is already adaptive in OPT#14 code, but we can tune thresholds

        if current_spread_bps < -2.0:
            recommendation = "1 level (minimal exposure)"
        elif current_spread_bps < 2.0:
            recommendation = "2 levels (moderate)"
        else:
            recommendation = "3 levels (full exposure)"

        return {
            "parameter": "order_levels",
            "recommendation": recommendation,
            "note": "Already adaptive in OPT#14",
        }

    def generate_optimization_report(self, metrics: Dict) -> str:
        """Generate optimization recommendations."""
        spread_bps = metrics.get("spread_bps", 0)
        fill_rate = metrics.get("fills_per_hour", 0)

        report = []
        report.append("=" * 80)
        report.append("🔧 AUTO-OPTIMIZATION ANALYSIS")
        report.append("=" * 80)
        report.append(f"\nCurrent Metrics:")
        report.append(f"  Spread: {spread_bps:+.2f} bps")
        report.append(f"  Fill Rate: {fill_rate} fills/hour")
        report.append(f"\nRecommended Actions:")

        if spread_bps < -2.0:
            report.append("  ❌ CRITICAL: Severe adverse selection")
            report.append("     → Increase defensive_distance to $6-7")
            report.append("     → Increase min_spread_bps to 10")
            report.append("     → Reduce order_levels to 1")
        elif spread_bps < 0:
            report.append("  ⚠️  WARNING: Negative spread")
            report.append("     → Increase defensive_distance to $4-5")
            report.append("     → Increase min_spread_bps to 9")
            report.append("     → Reduce order_levels to 2")
        elif fill_rate < 5:
            report.append("  ⚠️  WARNING: Low fill rate")
            report.append("     → Decrease defensive_distance to $2")
            report.append("     → Decrease min_spread_bps to 7")
        elif spread_bps > 10 and fill_rate > 15:
            report.append("  ✅ EXCELLENT: High profitability")
            report.append("     → Can slightly tighten for more volume")
            report.append("     → Consider min_spread_bps 7.5")
        else:
            report.append("  ✅ OPTIMAL: No changes needed")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """Run optimization analysis."""
    import requests

    WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
    URL = "https://api.hyperliquid.xyz/info"

    # Get recent fills
    resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
    fills = resp.json()

    if not fills:
        print("No fills to analyze")
        return

    # Analyze last 20 fills
    recent_20 = fills[:20]
    buys = [f for f in recent_20 if f.get("side") == "B"]
    sells = [f for f in recent_20 if f.get("side") == "A"]

    if not buys or not sells:
        print("Insufficient fill data")
        return

    avg_buy = sum(float(f.get("px", 0)) for f in buys) / len(buys)
    avg_sell = sum(float(f.get("px", 0)) for f in sells) / len(sells)
    mid = (avg_buy + avg_sell) / 2
    spread_bps = ((avg_sell - avg_buy) / mid) * 10000 if mid > 0 else 0

    # Calculate fill rate
    import time

    now = time.time() * 1000
    one_hour_ago = now - (60 * 60 * 1000)
    fills_last_hour = [f for f in fills if f.get("time", 0) > one_hour_ago]
    fills_per_hour = len(fills_last_hour)

    metrics = {"spread_bps": spread_bps, "fills_per_hour": fills_per_hour}

    optimizer = StrategyOptimizer()
    report = optimizer.generate_optimization_report(metrics)
    print(report)

    print("\n⚠️  To apply optimizations automatically, uncomment the adjustment calls below")
    print("    and re-run this script with caution.\n")

    # UNCOMMENT THESE TO APPLY OPTIMIZATIONS (USE WITH CAUTION!)
    # result1 = optimizer.adjust_defensive_distance(spread_bps)
    # print(f"Applied: {result1}")
    #
    # result2 = optimizer.adjust_min_spread(fills_per_hour, spread_bps)
    # print(f"Applied: {result2}")


if __name__ == "__main__":
    main()
