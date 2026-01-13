#!/usr/bin/env python3
"""
Quick Grid Search - 7 day backtest for fast iteration.
"""

import sys
import os

# Suppress loguru entirely
os.environ['LOGURU_LEVEL'] = 'ERROR'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.realistic_backtest import RealisticConfig, run_realistic_backtest
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# Target metrics
TARGETS = {
    'sharpe_min': 1.5,
    'sharpe_max': 3.0,
    'roi_ann_min': 5.0,
    'dd_max': 1.0,
    'trades_per_day_min': 500,
}


def test_config(params):
    """Test a single config."""
    leverage, min_spread, fill_rate, order_levels = params
    
    try:
        config = RealisticConfig(
            leverage=leverage,
            min_spread_bps=min_spread,
            max_spread_bps=min_spread * 3,
            base_fill_rate=fill_rate,
            order_levels=order_levels,
            slippage_mean_bps=0.3,
            order_size_pct=0.02,
            max_drawdown=0.05,  # 5% max DD before stop loss
        )
        
        result = run_realistic_backtest(config=config, days=7, verbose=False)
        
        # Annualize
        roi_ann = result.roi_pct * (365 / 7)
        
        # Score: prioritize Sharpe in range, then ROI, then low DD
        sharpe_score = min(result.sharpe_ratio, 3) / 3 if result.sharpe_ratio > 0 else 0
        roi_score = min(roi_ann, 20) / 20 if roi_ann > 0 else 0
        dd_score = max(0, 1 - result.max_drawdown * 100 / 5)
        
        score = sharpe_score * 0.4 + roi_score * 0.4 + dd_score * 0.2
        
        return {
            'params': params,
            'sharpe': result.sharpe_ratio,
            'roi': result.roi_pct,
            'roi_ann': roi_ann,
            'dd': result.max_drawdown * 100,
            'trades': result.total_trades,
            'tpd': result.trades_per_day,
            'score': score,
        }
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("QUICK GRID SEARCH (7-day backtest)")
    print("=" * 60)
    
    # Parameter grid
    params_list = []
    for lev in [5, 7, 10]:
        for spread in [1, 2, 3, 5]:
            for fill in [0.7, 0.85, 0.95]:
                for levels in [10, 15, 20]:
                    params_list.append((lev, spread, fill, levels))
    
    print(f"Testing {len(params_list)} combinations...")
    
    results = []
    start = time.time()
    
    # Sequential for stability (faster with the adjustments made)
    for i, params in enumerate(params_list):
        r = test_config(params)
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(params_list)} - Valid: {len(results)}")
    
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({len(results)} valid)")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "=" * 60)
    print("TOP 10 RESULTS")
    print("=" * 60)
    
    for i, r in enumerate(results[:10]):
        lev, spread, fill, levels = r['params']
        print(f"\n{i+1}. Score: {r['score']:.3f}")
        print(f"   Params: Lev={lev}, Spread={spread}bps, Fill={fill}, Levels={levels}")
        print(f"   Sharpe={r['sharpe']:.2f}, ROI={r['roi']:.2f}% ({r['roi_ann']:.1f}% ann)")
        print(f"   DD={r['dd']:.2f}%, Trades/Day={r['tpd']:.0f}")
    
    print("\n" + "=" * 60)
    
    # Check if any meet targets
    meeting = [r for r in results if (
        TARGETS['sharpe_min'] <= r['sharpe'] <= TARGETS['sharpe_max'] and
        r['roi_ann'] >= TARGETS['roi_ann_min'] and
        r['dd'] <= TARGETS['dd_max'] and
        r['tpd'] >= TARGETS['trades_per_day_min']
    )]
    
    if meeting:
        print(f"✅ {len(meeting)} configurations meet ALL targets!")
        r = meeting[0]
        print(f"Best: Lev={r['params'][0]}, Spread={r['params'][1]}, "
              f"Fill={r['params'][2]}, Levels={r['params'][3]}")
    else:
        print("❌ No configurations meet all targets yet.")
        print("Best attempt:")
        r = results[0]
        print(f"  Sharpe: {r['sharpe']:.2f} (target: {TARGETS['sharpe_min']}-{TARGETS['sharpe_max']})")
        print(f"  ROI ann: {r['roi_ann']:.1f}% (target: >{TARGETS['roi_ann_min']}%)")
        print(f"  DD: {r['dd']:.2f}% (target: <{TARGETS['dd_max']}%)")
        print(f"  Trades/Day: {r['tpd']:.0f} (target: >{TARGETS['trades_per_day_min']})")


if __name__ == "__main__":
    main()
