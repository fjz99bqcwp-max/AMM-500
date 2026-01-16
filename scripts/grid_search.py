#!/usr/bin/env python3
"""
Grid Search for Parameter Optimization
========================================
Runs 108 parameter combinations to find optimal settings.

Usage:
    python scripts/grid_search.py
"""

import asyncio
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.data_fetcher import DataFetcher
from src.core.backtest import BacktestEngine, BacktestConfig


# Parameter grid
PARAM_GRID = {
    "leverage": [5, 10, 15],
    "min_spread_bps": [1, 2, 3],
    "order_levels": [10, 15, 20],
    "partial_fill_rate": [0.2, 0.3, 0.4],
}


async def run_grid_search():
    """Run grid search over parameter combinations."""
    logger.info("Starting grid search...")
    
    # Load config and data
    config = load_config()
    fetcher = DataFetcher(config)
    
    data = fetcher.load_cached()
    if data is None:
        logger.info("Fetching data...")
        data = await fetcher.fetch(30)
    
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    logger.info(f"Testing {len(combinations)} combinations...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create backtest config
        bt_config = BacktestConfig(
            initial_capital=1000.0,
            leverage=params["leverage"],
            partial_fill_rate=params["partial_fill_rate"],
        )
        
        # Run backtest
        engine = BacktestEngine(bt_config)
        result = await engine.run(data)
        
        # Record results
        results.append({
            **params,
            "sharpe": result.sharpe_ratio,
            "roi": result.total_return,
            "max_dd": result.max_drawdown,
            "trades_day": result.trades_per_day,
            "maker_ratio": result.maker_ratio,
            "final_equity": result.final_equity,
        })
        
        logger.info(
            f"[{i+1}/{len(combinations)}] "
            f"Lev={params['leverage']}, Spread={params['min_spread_bps']}, "
            f"Levels={params['order_levels']} → "
            f"Sharpe={result.sharpe_ratio:.2f}, ROI={result.total_return:.2%}"
        )
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("data/grid_search_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Find best by Sharpe
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    print("\n" + "=" * 60)
    print("BEST BY SHARPE RATIO:")
    print("=" * 60)
    print(best_sharpe)
    
    # Find best by ROI with low DD
    low_dd = df[df["max_dd"] < 0.01]
    if len(low_dd) > 0:
        best_safe = low_dd.loc[low_dd["roi"].idxmax()]
        print("\n" + "=" * 60)
        print("BEST ROI WITH DD < 1%:")
        print("=" * 60)
        print(best_safe)


if __name__ == "__main__":
    asyncio.run(run_grid_search())
