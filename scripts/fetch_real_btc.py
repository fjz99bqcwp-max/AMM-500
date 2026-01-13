#!/usr/bin/env python3
"""
Fetch Real BTC Historical Data from Hyperliquid
Uses SDK to download 12 months of 1-minute candles for realistic backtesting.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperliquid.info import Info

def fetch_btc_candles(months: int = 6) -> pd.DataFrame:
    """
    Fetch BTC 1-minute candles from Hyperliquid.
    
    Args:
        months: Number of months of history to fetch (default 6 for realistic dataset)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {months} months of BTC 1-minute candles from Hyperliquid...")
    
    info = Info(base_url='https://api.hyperliquid.xyz', skip_ws=True)
    
    # Calculate time range
    end_time = int(time.time() * 1000)
    start_time = end_time - (months * 30 * 24 * 60 * 60 * 1000)
    
    print(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
    
    all_candles = []
    
    # Fetch in chunks (Hyperliquid limits response size)
    # Request ~7 days at a time to avoid timeouts
    chunk_size = 7 * 24 * 60 * 60 * 1000  # 7 days in ms
    current_start = start_time
    
    chunk_num = 0
    total_chunks = (end_time - start_time) // chunk_size + 1
    
    while current_start < end_time:
        chunk_num += 1
        current_end = min(current_start + chunk_size, end_time)
        
        print(f"Fetching chunk {chunk_num}/{total_chunks}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}...")
        
        try:
            # Use candles_snapshot (plural) for historical data
            candles = info.candles_snapshot(
                name='BTC',  # Asset name
                interval='1m',
                startTime=current_start,
                endTime=current_end
            )
            
            if candles:
                all_candles.extend(candles)
                print(f"  ✓ Fetched {len(candles)} candles")
            else:
                print(f"  ⚠ No data for this range")
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error fetching chunk: {e}")
            # Continue with next chunk
        
        current_start = current_end
    
    print(f"\nTotal candles fetched: {len(all_candles)}")
    
    if not all_candles:
        raise ValueError("No candles fetched - API may be unavailable or symbol incorrect")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles)
    
    # Hyperliquid candle format: {'t': timestamp_ms, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}
    df = df.rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })
    
    # Keep only necessary columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset='timestamp', keep='last')
    
    print(f"\nProcessed data:")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total candles: {len(df)}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Avg volume: {df['volume'].mean():.2f}")
    
    return df


def save_data(df: pd.DataFrame, output_dir: Path):
    """Save data in multiple formats for compatibility."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save as CSV (human-readable)
    csv_path = output_dir / 'btc_historical.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    # Save as JSON (for backtest compatibility)
    json_path = output_dir / 'btc_historical.json'
    
    # Convert to JSON-friendly format
    data = {
        'symbol': 'BTC',
        'interval': '1m',
        'start_time': df['timestamp'].min().isoformat(),
        'end_time': df['timestamp'].max().isoformat(),
        'candles': df.to_dict('records')
    }
    
    # Convert timestamps to ISO format in candles
    for candle in data['candles']:
        candle['timestamp'] = pd.to_datetime(candle['timestamp']).isoformat()
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved JSON: {json_path}")
    
    # Save metadata
    meta_path = output_dir / 'btc_metadata.json'
    metadata = {
        'symbol': 'BTC',
        'source': 'Hyperliquid API',
        'fetch_date': datetime.now().isoformat(),
        'total_candles': len(df),
        'date_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
        },
        'price_stats': {
            'min': float(df['close'].min()),
            'max': float(df['close'].max()),
            'mean': float(df['close'].mean()),
            'std': float(df['close'].std()),
        },
        'volume_stats': {
            'total': float(df['volume'].sum()),
            'mean': float(df['volume'].mean()),
            'std': float(df['volume'].std()),
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: {meta_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("BTC Historical Data Fetcher")
    print("=" * 70)
    print()
    
    # Fetch data
    try:
        df = fetch_btc_candles(months=6)
    except Exception as e:
        print(f"\n✗ Failed to fetch data: {e}")
        print("\nTrying with 3 months instead...")
        try:
            df = fetch_btc_candles(months=3)
        except Exception as e2:
            print(f"✗ Failed again: {e2}")
            return 1
    
    # Save
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    save_data(df, data_dir)
    
    print()
    print("=" * 70)
    print("✓ BTC Data Fetch Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Verify data: head -20 data/btc_historical.csv")
    print("  2. Run backtest: python scripts/verify_targets.py")
    print("  3. Optimize: python scripts/grid_search.py")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
