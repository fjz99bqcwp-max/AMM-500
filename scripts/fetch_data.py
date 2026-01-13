#!/usr/bin/env python3
"""
Fetch real historical data from Hyperliquid SDK
Using BTC as proxy for US500 (scaled)
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperliquid.info import Info

def fetch_btc_data(days: int = 180) -> list:
    """Fetch BTC 1m candles from Hyperliquid"""
    print(f"Fetching {days} days of BTC data...")
    
    info = Info(skip_ws=True)
    
    # First get current server time by fetching most recent candle
    print("Getting server time...")
    try:
        # Get a recent 1h candle to determine server time
        test_end = 1736726400000  # Jan 13, 2025 00:00 UTC (known past time)
        test_start = test_end - (24 * 60 * 60 * 1000)  # 24h before
        test_candles = info.candles_snapshot(
            name="BTC",
            interval="1h", 
            startTime=test_start,
            endTime=test_end
        )
        if test_candles:
            # Use the most recent candle time as reference
            latest_ts = max(c.get('t', c.get('T', 0)) for c in test_candles)
            print(f"  Reference time: {datetime.fromtimestamp(latest_ts/1000)}")
            end_time = test_end
        else:
            print("  Using hardcoded end time: Jan 12, 2025")
            end_time = 1736640000000  # Jan 12, 2025
    except Exception as e:
        print(f"  Error getting server time: {e}")
        end_time = 1736640000000  # Fallback to Jan 12, 2025
    
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    print(f"Fetching from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
    
    all_candles = []
    
    # Fetch in chunks (max ~5000 candles per request = ~3.5 days of 1m data)
    chunk_ms = 5000 * 60 * 1000  # 5000 minutes in ms
    
    current_start = start_time
    chunk_num = 0
    
    while current_start < end_time:
        current_end = min(current_start + chunk_ms, end_time)
        chunk_num += 1
        
        print(f"  Chunk {chunk_num}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d %H:%M')}")
        
        try:
            # Use correct signature: name, interval, startTime, endTime
            candles = info.candles_snapshot(
                name="BTC",
                interval="1m",
                startTime=current_start,
                endTime=current_end
            )
            
            if candles:
                all_candles.extend(candles)
                print(f"    Got {len(candles)} candles (total: {len(all_candles)})")
            else:
                print(f"    No candles returned")
                
        except Exception as e:
            print(f"    Error: {e}")
            # Try to continue with next chunk
        
        current_start = current_end
        time.sleep(0.2)  # Rate limiting
    
    print(f"\nTotal candles fetched: {len(all_candles)}")
    return all_candles


def scale_to_us500(btc_candles: list, target_price: float = 5800.0) -> list:
    """
    Scale BTC data to look like US500
    - Scale price to ~5800 range
    - Compress volatility to 30% of BTC
    """
    if not btc_candles:
        return []
    
    # Get BTC price range for scaling
    btc_prices = [float(c['c']) for c in btc_candles if 'c' in c]
    if not btc_prices:
        return []
        
    btc_avg = sum(btc_prices) / len(btc_prices)
    scale_factor = target_price / btc_avg
    volatility_compression = 0.30  # US500 is less volatile than BTC
    
    print(f"Scaling: BTC avg={btc_avg:.2f}, target={target_price}, factor={scale_factor:.6f}")
    
    us500_candles = []
    for c in btc_candles:
        try:
            btc_open = float(c['o'])
            btc_high = float(c['h'])
            btc_low = float(c['l'])
            btc_close = float(c['c'])
            
            # Scale with volatility compression
            center = btc_avg * scale_factor
            
            us500_open = center + (btc_open - btc_avg) * scale_factor * volatility_compression
            us500_high = center + (btc_high - btc_avg) * scale_factor * volatility_compression
            us500_low = center + (btc_low - btc_avg) * scale_factor * volatility_compression
            us500_close = center + (btc_close - btc_avg) * scale_factor * volatility_compression
            
            us500_candles.append({
                't': c.get('t', c.get('T', 0)),  # timestamp
                'o': us500_open,
                'h': us500_high,
                'l': us500_low,
                'c': us500_close,
                'v': float(c.get('v', 0)) * scale_factor  # scaled volume
            })
        except (KeyError, ValueError) as e:
            continue
    
    return us500_candles


def save_data(candles: list, filename: str):
    """Save candles to JSON file"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(candles, f)
    
    print(f"Saved {len(candles)} candles to {filepath}")
    return filepath


def main():
    print("=" * 60)
    print("AMM-500 Data Fetcher")
    print("=" * 60)
    
    # Fetch 180 days of BTC data
    btc_candles = fetch_btc_data(days=180)
    
    if not btc_candles:
        print("ERROR: Failed to fetch BTC data")
        return 1
    
    # Save raw BTC data
    save_data(btc_candles, 'btc_raw_180d.json')
    
    # Scale to US500
    us500_candles = scale_to_us500(btc_candles)
    
    if us500_candles:
        save_data(us500_candles, 'us500_proxy_180d.json')
        
        # Print stats
        prices = [c['c'] for c in us500_candles]
        print(f"\nUS500 Proxy Stats:")
        print(f"  Candles: {len(us500_candles)}")
        print(f"  Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"  Average: ${sum(prices)/len(prices):.2f}")
        print(f"  Time Range: {datetime.fromtimestamp(us500_candles[0]['t']/1000)} to {datetime.fromtimestamp(us500_candles[-1]['t']/1000)}")
    
    print("\nData fetch complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
