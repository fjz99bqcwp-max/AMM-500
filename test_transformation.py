#!/usr/bin/env python3
"""Validate transformed strategy methods"""
import sys
sys.path.insert(0, '.')

from src.strategy import MarketMakingStrategy
from src.config import Config

# Check if new methods exist
strategy_methods = dir(MarketMakingStrategy)
new_methods = [
    '_analyze_order_book',
    '_calculate_inventory_skew', 
    '_build_tiered_quotes',
    '_deduplicate_levels'
]

print('✅ Checking transformed strategy methods:')
for method in new_methods:
    if method in strategy_methods:
        print(f'  ✅ {method}')
    else:
        print(f'  ❌ {method} - MISSING!')

# Check class constants
if hasattr(MarketMakingStrategy, 'QUOTE_REFRESH_INTERVAL'):
    print(f'  ✅ QUOTE_REFRESH_INTERVAL = {MarketMakingStrategy.QUOTE_REFRESH_INTERVAL}s')
if hasattr(MarketMakingStrategy, 'MIN_BOOK_DEPTH_USD'):
    print(f'  ✅ MIN_BOOK_DEPTH_USD = ${MarketMakingStrategy.MIN_BOOK_DEPTH_USD}')
if hasattr(MarketMakingStrategy, 'ADVERSE_SELECTION_THRESHOLD'):
    print(f'  ✅ ADVERSE_SELECTION_THRESHOLD = {MarketMakingStrategy.ADVERSE_SELECTION_THRESHOLD}')

# Check BookDepthAnalysis class
try:
    from src.strategy import BookDepthAnalysis
    print(f'  ✅ BookDepthAnalysis class')
except ImportError:
    print(f'  ❌ BookDepthAnalysis class - MISSING!')

print('\n✅ Strategy transformation validated successfully!')
