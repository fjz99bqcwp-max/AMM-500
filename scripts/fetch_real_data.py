#!/usr/bin/env python3
"""
Fetch real historical data for backtesting.
Attempts:
1. Hyperliquid US500 KM market
2. Hyperliquid BTC as proxy (most liquid)
3. External S&P 500 data from Yahoo Finance

Stores data in data/us500_historical.csv
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def fetch_hyperliquid_candles(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch candles from Hyperliquid API."""
    print(f"Fetching {symbol} candles for {days} days...")
    
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        # Fetch in chunks (max 5000 candles per request)
        chunk_end = min(current_start + (5000 * 60 * 1000), end_ts)
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": chunk_end
            }
        }
        
        try:
            resp = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=30)
            candles = resp.json()
            
            if candles and isinstance(candles, list):
                all_candles.extend(candles)
                print(f"  Fetched {len(candles)} candles from {datetime.fromtimestamp(current_start/1000)}")
            
            current_start = chunk_end
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles)
    df.columns = ['timestamp', 'T', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'n']
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    
    return df


def fetch_btc_as_proxy(days: int = 180) -> pd.DataFrame:
    """Fetch BTC data as proxy for backtesting (most liquid market)."""
    return fetch_hyperliquid_candles("BTC", "1m", days)


def fetch_yahoo_sp500(days: int = 180) -> pd.DataFrame:
    """Fetch S&P 500 data from Yahoo Finance as fallback."""
    print(f"Fetching S&P 500 from Yahoo Finance for {days} days...")
    
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use ES=F (S&P 500 E-mini futures) or ^GSPC (S&P 500 index)
        ticker = yf.Ticker("ES=F")
        df = ticker.history(start=start_date, end=end_date, interval="1m")
        
        if df.empty:
            # Try hourly if 1m not available
            df = ticker.history(start=start_date, end=end_date, interval="1h")
            if not df.empty:
                # Upsample to 1m with forward fill
                df = df.resample('1min').ffill()
        
        if not df.empty:
            df = df.reset_index()
            df['timestamp'] = df['Datetime'].apply(lambda x: int(x.timestamp() * 1000))
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return df
            
    except ImportError:
        print("  yfinance not installed, skipping Yahoo Finance")
    except Exception as e:
        print(f"  Yahoo Finance error: {e}")
    
    return pd.DataFrame()


def normalize_to_index_scale(df: pd.DataFrame, target_price: float = 6000.0) -> pd.DataFrame:
    """
    Normalize price data to US500-like scale.
    S&P 500 trades around 5000-6000, BTC around 90000.
    """
    if df.empty:
        return df
    
    current_price = df['close'].iloc[-1]
    scale_factor = target_price / current_price
    
    df_scaled = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        df_scaled[col] = df[col] * scale_factor
    
    print(f"  Scaled from {current_price:.2f} to {target_price:.2f} (factor: {scale_factor:.4f})")
    return df_scaled


def generate_realistic_synthetic(days: int = 180) -> pd.DataFrame:
    """
    Generate realistic synthetic US500 data with:
    - Proper volatility (15-20% annual vs crypto 80-100%)
    - Market hours effects (higher vol at open/close)
    - Trend with mean reversion
    - Jump events (earnings, fed, etc.)
    """
    import numpy as np
    
    print(f"Generating realistic synthetic US500 data for {days} days...")
    
    np.random.seed(42)
    
    candles_per_day = 1440
    total_candles = days * candles_per_day
    
    # Start price around current S&P 500 level
    start_price = 5900.0
    
    # Annual volatility 18% -> minute volatility
    annual_vol = 0.18
    minute_vol = annual_vol / np.sqrt(252 * 1440)
    
    # Generate returns with realistic properties
    timestamps = []
    prices = [start_price]
    
    start_time = int((time.time() - days * 24 * 60 * 60) * 1000)
    
    for i in range(total_candles):
        timestamps.append(start_time + i * 60 * 1000)
        
        # Time-of-day volatility adjustment
        minute_of_day = i % candles_per_day
        hour = minute_of_day // 60
        
        # Higher vol at US market open (9:30) and close (16:00)
        vol_mult = 1.0
        if 570 <= minute_of_day <= 630:  # 9:30-10:30 AM
            vol_mult = 1.5
        elif 900 <= minute_of_day <= 960:  # 3:00-4:00 PM
            vol_mult = 1.3
        elif 960 <= minute_of_day <= 1020:  # After hours
            vol_mult = 0.5
        
        # Jump events (rare but significant)
        jump = 0
        if np.random.random() < 0.0001:  # 0.01% chance per minute
            jump = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.02)
        
        # Mean reversion component
        deviation = (prices[-1] - start_price) / start_price
        mean_reversion = -deviation * 0.00001
        
        # Generate return
        ret = np.random.normal(0.00001 + mean_reversion, minute_vol * vol_mult) + jump
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate OHLC from prices
    data = []
    for i in range(0, len(prices) - 1):
        close = prices[i + 1]
        open_p = prices[i]
        
        # High/Low with realistic range
        range_pct = abs(np.random.normal(0.0003, 0.0002))
        if close > open_p:
            high = max(open_p, close) * (1 + range_pct * 0.7)
            low = min(open_p, close) * (1 - range_pct * 0.3)
        else:
            high = max(open_p, close) * (1 + range_pct * 0.3)
            low = min(open_p, close) * (1 - range_pct * 0.7)
        
        # Volume with time-of-day pattern
        minute_of_day = i % candles_per_day
        base_vol = 1000000
        if 570 <= minute_of_day <= 960:  # Market hours
            vol = base_vol * np.random.uniform(0.8, 1.5)
        else:
            vol = base_vol * np.random.uniform(0.1, 0.3)
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_p, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(vol, 0)
        })
    
    df = pd.DataFrame(data)
    print(f"  Generated {len(df)} candles, price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    return df


def main():
    """Fetch and save historical data."""
    DATA_DIR.mkdir(exist_ok=True)
    
    days = 180
    df = pd.DataFrame()
    source = "none"
    
    # Try 1: BTC from Hyperliquid (most liquid, good proxy for backtest)
    print("\n=== Attempting to fetch BTC data from Hyperliquid ===")
    df = fetch_btc_as_proxy(days)
    if not df.empty and len(df) > 10000:
        print(f"  Success! Got {len(df)} BTC candles")
        # Normalize to US500 scale for realistic pricing
        df = normalize_to_index_scale(df, target_price=5950.0)
        source = "BTC_proxy"
    
    # Try 2: Yahoo Finance S&P 500
    if df.empty or len(df) < 10000:
        print("\n=== Attempting Yahoo Finance S&P 500 ===")
        df = fetch_yahoo_sp500(days)
        if not df.empty and len(df) > 10000:
            source = "yahoo_sp500"
    
    # Try 3: Realistic synthetic
    if df.empty or len(df) < 10000:
        print("\n=== Generating realistic synthetic data ===")
        df = generate_realistic_synthetic(days)
        source = "synthetic_realistic"
    
    if df.empty:
        print("ERROR: Could not obtain any data!")
        return
    
    # Save as CSV
    csv_path = DATA_DIR / "us500_historical.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=== Data saved to {csv_path} ===")
    print(f"  Source: {source}")
    print(f"  Candles: {len(df)}")
    print(f"  Days: {len(df) / 1440:.1f}")
    print(f"  Date range: {datetime.fromtimestamp(df['timestamp'].iloc[0]/1000)} to {datetime.fromtimestamp(df['timestamp'].iloc[-1]/1000)}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Also save as JSON for compatibility
    json_path = DATA_DIR / "us500_historical.json"
    df.to_json(json_path, orient='records')
    print(f"  Also saved as JSON: {json_path}")


if __name__ == "__main__":
    main()
