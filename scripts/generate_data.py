#!/usr/bin/env python3
"""
Generate realistic US500 synthetic data for backtesting
Based on historical S&P 500 characteristics
"""
import os
import sys
import json
import random
import math
from datetime import datetime, timedelta

def generate_realistic_us500_data(days: int = 180, interval_minutes: int = 1) -> list:
    """
    Generate realistic US500 price data
    
    S&P 500 characteristics:
    - Average daily volatility: ~1% 
    - Per-minute volatility: ~0.01% (very small)
    - Mean reversion tendency
    """
    print(f"Generating {days} days of US500 data at {interval_minutes}m intervals...")
    
    # Starting price around current S&P 500 levels
    current_price = 5800.0
    base_price = current_price
    
    # Realistic per-minute volatility for S&P 500
    # Daily vol ~1%, so per-minute vol = 1% / sqrt(1440) = ~0.026%
    minute_volatility = 0.0001  # 0.01% per minute standard deviation
    
    # Very slight upward drift
    minute_drift = 0.10 / (365 * 24 * 60)  # 10% annual / minutes in year
    
    candles = []
    total_candles = days * 24 * 60 // interval_minutes
    
    # Start timestamp (180 days of data ending Jan 1, 2025)
    end_ts = 1735689600000  # Jan 1, 2025 00:00 UTC
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    current_ts = start_ts
    
    for i in range(total_candles):
        # Time-based volatility adjustment (simulate market sessions)
        hour_of_day = (current_ts // 3600000) % 24
        
        # Higher volatility during US market hours (14:30-21:00 UTC)
        if 14 <= hour_of_day <= 21:
            session_vol_mult = 1.3
        elif 8 <= hour_of_day <= 14:  # European session
            session_vol_mult = 1.1
        else:  # Asian session / off hours
            session_vol_mult = 0.7
        
        # Generate price movement - small random walk
        vol = minute_volatility * session_vol_mult * current_price
        price_change = minute_drift * current_price + random.gauss(0, vol)
        
        # Generate OHLC
        open_price = current_price
        close_price = open_price + price_change
        
        # Small intra-candle movement
        intra_range = abs(random.gauss(0, vol * 0.5))
        high_price = max(open_price, close_price) + intra_range
        low_price = min(open_price, close_price) - intra_range
        
        # Slight mean reversion to prevent drift too far
        if abs(close_price - base_price) / base_price > 0.15:
            close_price = close_price * 0.999 if close_price > base_price else close_price * 1.001
        
        # Volume
        base_volume = 1000000
        volume = base_volume * session_vol_mult * random.uniform(0.5, 1.5)
        
        candle = {
            't': current_ts,
            'o': round(open_price, 2),
            'h': round(high_price, 2),
            'l': round(low_price, 2),
            'c': round(close_price, 2),
            'v': round(volume, 0)
        }
        candles.append(candle)
        
        # Update state
        current_price = close_price
        current_ts += interval_minutes * 60 * 1000
        
        # Progress update
        if (i + 1) % 100000 == 0:
            print(f"  Generated {i+1}/{total_candles} candles ({100*(i+1)/total_candles:.1f}%)")
    
    print(f"Generated {len(candles)} candles")
    return candles


def add_market_events(candles: list) -> list:
    """Add realistic market events (flash crashes, rallies, etc.)"""
    print("Adding market events...")
    
    total = len(candles)
    
    # Add a few flash crash events (sudden drops followed by recovery)
    crash_events = random.sample(range(1000, total - 1000), min(5, (total - 2000) // 50000))
    
    for start_idx in crash_events:
        crash_size = random.uniform(0.02, 0.05)  # 2-5% drop
        recovery_duration = random.randint(60, 240)  # 1-4 hours
        
        base_price = candles[start_idx]['o']
        crash_bottom = base_price * (1 - crash_size)
        
        # Sharp drop (5-15 candles)
        drop_duration = random.randint(5, 15)
        for j in range(drop_duration):
            idx = start_idx + j
            if idx >= total:
                break
            progress = j / drop_duration
            target = base_price - (base_price - crash_bottom) * progress
            candles[idx]['c'] = round(target, 2)
            candles[idx]['l'] = round(min(candles[idx]['l'], target * 0.998), 2)
            candles[idx]['h'] = round(max(candles[idx]['o'], target), 2)
        
        # Slow recovery
        for j in range(recovery_duration):
            idx = start_idx + drop_duration + j
            if idx >= total:
                break
            progress = j / recovery_duration
            target = crash_bottom + (base_price - crash_bottom) * (progress ** 0.5)
            candles[idx]['o'] = round(candles[idx-1]['c'], 2)
            candles[idx]['c'] = round(target * random.uniform(0.998, 1.002), 2)
    
    # Add a few rally events
    rally_events = random.sample(range(1000, total - 1000), min(3, (total - 2000) // 80000))
    
    for start_idx in rally_events:
        rally_size = random.uniform(0.015, 0.03)  # 1.5-3% gain
        rally_duration = random.randint(120, 480)  # 2-8 hours
        
        base_price = candles[start_idx]['o']
        rally_top = base_price * (1 + rally_size)
        
        for j in range(rally_duration):
            idx = start_idx + j
            if idx >= total:
                break
            progress = j / rally_duration
            target = base_price + (rally_top - base_price) * (progress ** 0.7)
            candles[idx]['c'] = round(target * random.uniform(0.999, 1.001), 2)
            candles[idx]['h'] = round(max(candles[idx]['h'], target * 1.001), 2)
    
    print(f"  Added {len(crash_events)} crash events and {len(rally_events)} rally events")
    return candles


def save_data(candles: list, filename: str):
    """Save candles to JSON file"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(candles, f)
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved {len(candles)} candles to {filepath} ({size_mb:.1f} MB)")
    return filepath


def main():
    print("=" * 60)
    print("US500 Synthetic Data Generator")
    print("=" * 60)
    
    # Generate 180 days of 1-minute data
    candles = generate_realistic_us500_data(days=180, interval_minutes=1)
    
    # Add market events for realism
    candles = add_market_events(candles)
    
    # Save data
    save_data(candles, 'us500_synthetic_180d.json')
    
    # Print stats
    prices = [c['c'] for c in candles]
    print(f"\nUS500 Synthetic Data Stats:")
    print(f"  Candles: {len(candles)}")
    print(f"  Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"  Average: ${sum(prices)/len(prices):.2f}")
    print(f"  Start: {datetime.fromtimestamp(candles[0]['t']/1000)}")
    print(f"  End: {datetime.fromtimestamp(candles[-1]['t']/1000)}")
    
    # Calculate volatility
    returns = [(candles[i+1]['c'] - candles[i]['c']) / candles[i]['c'] 
               for i in range(len(candles)-1)]
    daily_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (24 * 60) ** 0.5
    print(f"  Daily Volatility: {daily_vol*100:.2f}%")
    print(f"  Annualized Vol: {daily_vol * (252**0.5) * 100:.1f}%")
    
    print("\nData generation complete!")
    return 0


if __name__ == '__main__':
    random.seed(42)  # Reproducible results
    sys.exit(main())
