#!/usr/bin/env python3
"""Grid search for optimal realistic backtest parameters."""

import sys
sys.path.insert(0, '/Users/nheosdisplay/VSC/AMM/AMM-500')

from src.realistic_backtest import RealisticConfig, run_realistic_backtest
import numpy as np

print("Starting grid search for optimal parameters...")

results = []
total = 3 * 3 * 3 * 3
count = 0

for leverage in [5, 8, 10]:
    for min_spread in [3, 5, 7]:
        for fill_rate in [0.75, 0.85, 0.90]:
            for order_levels in [8, 12, 15]:
                count += 1
                print(f"  [{count}/{total}] Testing lev={leverage}, spread={min_spread}, fill={fill_rate}, levels={order_levels}")
                
                config = RealisticConfig(
                    leverage=leverage,
                    min_spread_bps=min_spread,
                    max_spread_bps=min_spread * 3,
                    base_fill_rate=fill_rate,
                    order_levels=order_levels,
                    slippage_mean_bps=0.3,
                )
                try:
                    result = run_realistic_backtest(config=config, days=30, verbose=False)
                    results.append({
                        'leverage': leverage,
                        'min_spread': min_spread,
                        'fill_rate': fill_rate,
                        'order_levels': order_levels,
                        'sharpe': result.sharpe_ratio,
                        'roi': result.roi_pct,
                        'trades': result.total_trades,
                        'max_dd': result.max_drawdown * 100
                    })
                except Exception as e:
                    print(f"    Error: {e}")

# Sort by Sharpe
results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
print('\n' + '='*60)
print('TOP 10 PARAMETER COMBINATIONS')
print('='*60)
for i, r in enumerate(results[:10]):
    print(f"{i+1}. Sharpe: {r['sharpe']:.2f}, ROI: {r['roi']:.2f}%, DD: {r['max_dd']:.2f}%, Trades: {r['trades']}")
    print(f"   Lev={r['leverage']}, Spread={r['min_spread']}, Fill={r['fill_rate']}, Levels={r['order_levels']}")
