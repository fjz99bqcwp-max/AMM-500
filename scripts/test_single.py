#!/usr/bin/env python3
"""
Single fast backtest to verify the system works.
"""

import sys
import os

os.environ['LOGURU_LEVEL'] = 'ERROR'
sys.path.insert(0, '/Users/nheosdisplay/VSC/AMM/AMM-500')

from src.realistic_backtest import RealisticConfig, run_realistic_backtest

# Test with conservative settings for DD < 1%
config = RealisticConfig(
    leverage=7,                 # Lower leverage
    min_spread_bps=3,           # Wider spreads for safety
    max_spread_bps=10,
    base_fill_rate=0.85,
    order_levels=12,            # Fewer levels
    slippage_mean_bps=0.2,
    order_size_pct=0.015,       # Smaller orders
    max_drawdown=0.05,
    rebalance_interval=5,       # Standard rebalancing
    inventory_vol_bps=3.0,      # Lower inventory vol
)

print("Running 7-day backtest...")
result = run_realistic_backtest(config=config, days=7, verbose=True)
print(f"\nQuick Result:")
print(f"  Trades: {result.total_trades}")
print(f"  Trades/Day: {result.trades_per_day:.0f}")
print(f"  ROI: {result.roi_pct:.2f}%")
print(f"  ROI Annualized: {result.roi_pct * 365 / 7:.1f}%")
print(f"  Max DD: {result.max_drawdown*100:.2f}%")
print(f"  Sharpe: {result.sharpe_ratio:.2f}")
