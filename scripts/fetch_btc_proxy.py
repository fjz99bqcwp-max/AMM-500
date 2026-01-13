#!/usr/bin/env python3
"""
Direct BTC data fetcher for US500 proxy backtesting.
Fetches BTC candles and scales to US500 characteristics.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import HyperliquidDataFetcher
import pandas as pd
from loguru import logger

# US500 scaling factors
US500_PRICE = 5800  # Target US500 price
BTC_PRICE = 95000   # Approximate BTC price
VOL_SCALE = 0.3     # US500 vol is ~30% of BTC


async def fetch_btc_as_us500_proxy(days: int = 180):
    """Fetch BTC data and scale to US500 proxy."""
    
    logger.info(f"Fetching {days} days of BTC data as US500 proxy...")
    
    fetcher = HyperliquidDataFetcher(use_testnet=False)
    
    try:
        # Fetch BTC candles
        btc_candles = await fetcher.fetch_candles("BTC", "1m", days)
        
        if btc_candles.empty:
            logger.error("Failed to fetch BTC data")
            return None
        
        logger.info(f"Fetched {len(btc_candles)} BTC candles")
        
        # Scale to US500
        price_scale = US500_PRICE / BTC_PRICE
        df = btc_candles.copy()
        
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * price_scale
        
        # Compress volatility (returns toward mean)
        if len(df) > 1:
            df["return"] = df["close"].pct_change()
            df["return_scaled"] = df["return"] * VOL_SCALE
            
            initial_price = df["close"].iloc[0]
            df["close_new"] = initial_price * (1 + df["return_scaled"]).cumprod()
            
            # Scale OHLC proportionally
            ratio = df["close_new"] / df["close"]
            ratio = ratio.fillna(1)
            
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * ratio
            
            df = df.drop(columns=["return", "return_scaled", "close_new"], errors="ignore")
        
        # Save to data directory
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save proxy data
        proxy_path = data_dir / f"US500_proxy_candles_1m_{days}d.csv"
        df.to_csv(proxy_path, index=False)
        logger.info(f"Saved US500 proxy data to {proxy_path}")
        
        # Also save original BTC for reference
        btc_path = data_dir / f"BTC_candles_1m_{days}d.csv"
        btc_candles.to_csv(btc_path, index=False)
        logger.info(f"Saved BTC data to {btc_path}")
        
        # Stats
        logger.info(f"\n=== US500 Proxy Data Stats ===")
        logger.info(f"Period: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Candles: {len(df)}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"Avg price: ${df['close'].mean():.2f}")
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std() * (1440 ** 0.5)  # 1440 minutes per day
        annual_vol = daily_vol * (252 ** 0.5)
        logger.info(f"Annualized volatility: {annual_vol*100:.1f}%")
        
        return df
        
    finally:
        await fetcher.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="Days of data to fetch")
    args = parser.parse_args()
    
    await fetch_btc_as_us500_proxy(args.days)


if __name__ == "__main__":
    asyncio.run(main())
