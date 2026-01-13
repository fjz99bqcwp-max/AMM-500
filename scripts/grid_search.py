#!/usr/bin/env python3
"""
Grid Search for Optimal Realistic Backtest Parameters
======================================================

Searches for parameters that achieve:
- Sharpe: 1.5-3.0
- Ann ROI: >5%
- Max DD: <1%
- Trades/Day: >500
- Fill Rate: 0.5-1%

Uses multiprocessing for M4 Mac (10 cores).
"""

import sys
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realistic_backtest import RealisticConfig, run_realistic_backtest

# Target metrics
TARGET_SHARPE_MIN = 1.5
TARGET_SHARPE_MAX = 3.0
TARGET_ROI_MIN = 5.0  # Annualized %
TARGET_DD_MAX = 1.0   # %
TARGET_TRADES_MIN = 500  # per day
TARGET_FILL_RATE_MIN = 0.005  # 0.5%
TARGET_FILL_RATE_MAX = 0.01   # 1%


@dataclass
class GridResult:
    """Result from a single grid point."""
    leverage: int
    min_spread: float
    fill_rate: float
    order_levels: int
    rebalance_interval: int
    sharpe: float
    roi: float
    roi_ann: float
    max_dd: float
    trades: int
    trades_per_day: float
    fill_rate_actual: float
    win_rate: float
    meets_targets: bool
    score: float  # Combined score for ranking


def evaluate_config(params: Tuple) -> Optional[GridResult]:
    """Evaluate a single configuration."""
    leverage, min_spread, fill_rate, order_levels, rebalance = params
    
    try:
        config = RealisticConfig(
            leverage=leverage,
            min_spread_bps=min_spread,
            max_spread_bps=min_spread * 3,
            base_fill_rate=fill_rate,
            order_levels=order_levels,
            rebalance_interval=rebalance,
            slippage_mean_bps=0.3,
            order_size_pct=0.02,
        )
        
        result = run_realistic_backtest(config=config, days=60, verbose=False)
        
        # Calculate annualized ROI
        days = result.duration_days if result.duration_days > 0 else 60
        roi_ann = result.roi_pct * (365 / days)
        
        # Calculate actual fill rate
        fill_rate_actual = result.filled_orders / max(result.total_orders, 1)
        
        # Check if meets targets
        meets_targets = (
            TARGET_SHARPE_MIN <= result.sharpe_ratio <= TARGET_SHARPE_MAX and
            roi_ann >= TARGET_ROI_MIN and
            result.max_drawdown * 100 <= TARGET_DD_MAX and
            result.trades_per_day >= TARGET_TRADES_MIN and
            TARGET_FILL_RATE_MIN <= fill_rate_actual <= TARGET_FILL_RATE_MAX
        )
        
        # Calculate combined score (higher is better)
        # Penalize being outside target ranges
        sharpe_score = min(result.sharpe_ratio, TARGET_SHARPE_MAX) / TARGET_SHARPE_MAX
        roi_score = min(roi_ann, 20) / 20
        dd_score = max(0, 1 - result.max_drawdown * 100 / TARGET_DD_MAX)
        trades_score = min(result.trades_per_day, 1000) / 1000
        
        score = sharpe_score * 0.3 + roi_score * 0.3 + dd_score * 0.2 + trades_score * 0.2
        
        return GridResult(
            leverage=leverage,
            min_spread=min_spread,
            fill_rate=fill_rate,
            order_levels=order_levels,
            rebalance_interval=rebalance,
            sharpe=result.sharpe_ratio,
            roi=result.roi_pct,
            roi_ann=roi_ann,
            max_dd=result.max_drawdown * 100,
            trades=result.total_trades,
            trades_per_day=result.trades_per_day,
            fill_rate_actual=fill_rate_actual,
            win_rate=result.win_rate * 100,
            meets_targets=meets_targets,
            score=score
        )
    except Exception as e:
        return None


def run_grid_search():
    """Run parallel grid search."""
    print("=" * 70)
    print("GRID SEARCH FOR OPTIMAL PARAMETERS")
    print("=" * 70)
    print(f"\nTarget Metrics:")
    print(f"  Sharpe: {TARGET_SHARPE_MIN} - {TARGET_SHARPE_MAX}")
    print(f"  Ann ROI: >{TARGET_ROI_MIN}%")
    print(f"  Max DD: <{TARGET_DD_MAX}%")
    print(f"  Trades/Day: >{TARGET_TRADES_MIN}")
    print(f"  Fill Rate: {TARGET_FILL_RATE_MIN*100:.1f}% - {TARGET_FILL_RATE_MAX*100:.1f}%")
    print()
    
    # Parameter grid - based on your recommendations
    leverages = [5, 7, 10]
    min_spreads = [1, 2, 3, 5]
    fill_rates = [0.5, 0.7, 0.85, 0.9]
    order_levels = [10, 15, 20]
    rebalance_intervals = [1, 3, 5]
    
    # Generate all combinations
    params_list = []
    for lev in leverages:
        for spread in min_spreads:
            for fill in fill_rates:
                for levels in order_levels:
                    for rebalance in rebalance_intervals:
                        params_list.append((lev, spread, fill, levels, rebalance))
    
    total = len(params_list)
    print(f"Testing {total} parameter combinations...")
    print(f"Using {cpu_count()} CPU cores\n")
    
    # Run in parallel
    start_time = time.time()
    
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(evaluate_config, params_list)):
            if result is not None:
                results.append(result)
            if (i + 1) % 20 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                remaining = elapsed / (i + 1) * (total - i - 1)
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) - "
                      f"ETA: {remaining:.0f}s - Valid: {len(results)}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    
    if not results:
        print("ERROR: No valid results!")
        return []
    
    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)
    
    # Print results
    print("\n" + "=" * 70)
    print("TOP 15 PARAMETER COMBINATIONS")
    print("=" * 70)
    
    for i, r in enumerate(results[:15]):
        target_str = "✅ MEETS TARGETS" if r.meets_targets else ""
        print(f"\n{i+1}. Score: {r.score:.3f} {target_str}")
        print(f"   Params: Lev={r.leverage}, Spread={r.min_spread}bps, "
              f"Fill={r.fill_rate}, Levels={r.order_levels}, Rebal={r.rebalance_interval}s")
        print(f"   Metrics: Sharpe={r.sharpe:.2f}, ROI={r.roi:.2f}% ({r.roi_ann:.1f}% ann), "
              f"DD={r.max_dd:.2f}%")
        print(f"   Trades: {r.trades} ({r.trades_per_day:.0f}/day), "
              f"Fill={r.fill_rate_actual*100:.2f}%, Win={r.win_rate:.1f}%")
    
    # Find configs that meet targets
    meeting_targets = [r for r in results if r.meets_targets]
    
    print("\n" + "=" * 70)
    if meeting_targets:
        print(f"FOUND {len(meeting_targets)} CONFIGURATIONS MEETING ALL TARGETS!")
        print("=" * 70)
        for r in meeting_targets[:5]:
            print(f"  Lev={r.leverage}, Spread={r.min_spread}bps, "
                  f"Fill={r.fill_rate}, Levels={r.order_levels}")
    else:
        print("NO CONFIGURATIONS MEET ALL TARGETS YET")
        print("Closest configuration:")
        r = results[0]
        print(f"  Lev={r.leverage}, Spread={r.min_spread}bps, "
              f"Fill={r.fill_rate}, Levels={r.order_levels}, Rebal={r.rebalance_interval}s")
        print(f"  Sharpe: {r.sharpe:.2f} (target: {TARGET_SHARPE_MIN}-{TARGET_SHARPE_MAX})")
        print(f"  ROI Ann: {r.roi_ann:.1f}% (target: >{TARGET_ROI_MIN}%)")
        print(f"  Max DD: {r.max_dd:.2f}% (target: <{TARGET_DD_MAX}%)")
        print(f"  Trades/Day: {r.trades_per_day:.0f} (target: >{TARGET_TRADES_MIN})")
        print(f"  Fill Rate: {r.fill_rate_actual*100:.2f}% (target: {TARGET_FILL_RATE_MIN*100:.1f}-{TARGET_FILL_RATE_MAX*100:.1f}%)")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_grid_search()
