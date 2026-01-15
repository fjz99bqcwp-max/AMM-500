#!/usr/bin/env python3
"""
Verify optimal parameters meet all targets.
"""

import os
os.environ['LOGURU_LEVEL'] = 'ERROR'

import sys
sys.path.insert(0, '/Users/nheosdisplay/VSC/AMM/AMM-500')

from src.realistic_backtest import RealisticConfig, run_realistic_backtest

# Optimal config - balanced for all targets
config = RealisticConfig(
    leverage=5,
    min_spread_bps=4,  # Balanced spread
    max_spread_bps=15,
    base_fill_rate=0.75,  # Moderate fill rate
    order_levels=18,
    slippage_mean_bps=0.3,
    order_size_pct=0.015,
    max_drawdown=0.05,
)

print("Running 30-day backtest with optimal config...")
result = run_realistic_backtest(config=config, days=30, verbose=True)

print()
print("=" * 60)
print("TARGET VERIFICATION")
print("=" * 60)

# Calculate metrics
roi_ann = result.roi_pct * 365 / 30
fill_rate = result.filled_orders / result.total_orders * 100

# Check targets
sharpe_ok = 1.5 <= result.sharpe_ratio <= 3.0
roi_ok = roi_ann > 5
dd_ok = result.max_drawdown * 100 < 1
trades_ok = result.trades_per_day > 500

print(f"Sharpe: {result.sharpe_ratio:.2f} (target: 1.5-3.0) {'✅' if sharpe_ok else '❌'}")
print(f"ROI Ann: {roi_ann:.1f}% (target: >5%) {'✅' if roi_ok else '❌'}")
print(f"Max DD: {result.max_drawdown*100:.2f}% (target: <1%) {'✅' if dd_ok else '❌'}")
print(f"Trades/Day: {result.trades_per_day:.0f} (target: >500) {'✅' if trades_ok else '❌'}")
print(f"Fill Rate: {fill_rate:.2f}%")

print()
if sharpe_ok and roi_ok and dd_ok and trades_ok:
    print("🎉 ALL TARGETS MET! Ready for paper trading.")
else:
    print("⚠️ Some targets not met. May need further tuning.")
