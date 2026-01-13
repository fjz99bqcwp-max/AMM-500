#!/usr/bin/env python3
"""Generate realistic US500 synthetic data for backtesting."""

import numpy as np
import pandas as pd
import time
from pathlib import Path

def main():
    print("Generating realistic US500 synthetic data...")
    
    np.random.seed(42)
    
    days = 180
    candles_per_day = 1440
    total_candles = days * candles_per_day
    
    start_price = 5900.0
    annual_vol = 0.18
    minute_vol = annual_vol / np.sqrt(252 * 1440)
    
    start_time = int((time.time() - days * 24 * 60 * 60) * 1000)
    timestamps = [start_time + i * 60 * 1000 for i in range(total_candles)]
    
    prices = [start_price]
    for i in range(total_candles):
        minute_of_day = i % candles_per_day
        vol_mult = 1.0
        if 570 <= minute_of_day <= 630:
            vol_mult = 1.5
        elif 900 <= minute_of_day <= 960:
            vol_mult = 1.3
        elif minute_of_day > 960 or minute_of_day < 570:
            vol_mult = 0.5
        
        deviation = (prices[-1] - start_price) / start_price
        mean_reversion = -deviation * 0.00001
        
        jump = 0
        if np.random.random() < 0.0001:
            jump = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.02)
        
        ret = np.random.normal(0.00001 + mean_reversion, minute_vol * vol_mult) + jump
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for i in range(total_candles):
        close = prices[i + 1]
        open_p = prices[i]
        range_pct = abs(np.random.normal(0.0003, 0.0002))
        
        if close > open_p:
            high = max(open_p, close) * (1 + range_pct * 0.7)
            low = min(open_p, close) * (1 - range_pct * 0.3)
        else:
            high = max(open_p, close) * (1 + range_pct * 0.3)
            low = min(open_p, close) * (1 - range_pct * 0.7)
        
        minute_of_day = i % candles_per_day
        base_vol = 1000000
        if 570 <= minute_of_day <= 960:
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
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    df.to_csv(data_dir / "us500_historical.csv", index=False)
    df.to_json(data_dir / "us500_historical.json", orient="records")
    
    print(f"Generated {len(df)} candles ({len(df)/1440:.1f} days)")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Final price: ${df['close'].iloc[-1]:.2f}")
    print(f"Saved to data/us500_historical.csv and .json")

if __name__ == "__main__":
    main()
