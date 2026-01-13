#!/usr/bin/env python3
"""
Direct SDK data fetcher - fetches from Hyperliquid using the SDK directly.
Bypasses the data_fetcher module issues.
"""
import asyncio
import time
from pathlib import Path
import pandas as pd
from loguru import logger

# SDK import
try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("Hyperliquid SDK not available")


def fetch_candles_sdk(coin: str = "BTC", interval: str = "1m", days: int = 30) -> pd.DataFrame:
    """
    Fetch candles using the Hyperliquid SDK directly.
    
    Args:
        coin: Asset symbol (BTC, ETH, etc.)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        days: Number of days to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    if not SDK_AVAILABLE:
        raise ImportError("hyperliquid-python-sdk not installed")
    
    logger.info(f"Fetching {days} days of {interval} candles for {coin} via SDK...")
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    # Calculate time range
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    # Interval to milliseconds for chunking
    interval_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }[interval]
    
    all_candles = []
    current_start = start_time
    max_candles_per_request = 5000
    chunk_ms = max_candles_per_request * interval_ms
    chunk_count = 0
    
    while current_start < end_time:
        chunk_end = min(current_start + chunk_ms, end_time)
        chunk_count += 1
        
        try:
            candles = info.candles_snapshot(
                coin=coin,
                interval=interval,
                startTime=current_start,
                endTime=chunk_end
            )
            
            if candles:
                for c in candles:
                    all_candles.append({
                        "timestamp": c["t"],
                        "open": float(c["o"]),
                        "high": float(c["h"]),
                        "low": float(c["l"]),
                        "close": float(c["c"]),
                        "volume": float(c["v"]),
                    })
                
                if chunk_count % 10 == 0:
                    logger.info(f"  Chunk {chunk_count}: {len(all_candles)} candles so far...")
                    
        except Exception as e:
            logger.warning(f"Error fetching chunk {chunk_count}: {e}")
        
        current_start = chunk_end
        time.sleep(0.15)  # Rate limiting
    
    if not all_candles:
        logger.warning(f"No candle data retrieved for {coin}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    logger.info(f"Fetched {len(df)} candles from {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def scale_btc_to_us500(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Scale BTC data to approximate US500 characteristics."""
    if btc_df.empty:
        return btc_df
    
    df = btc_df.copy()
    
    # Price scaling: US500 ~5800, BTC ~95000
    price_scale = 5800 / 95000  # ~0.061
    
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * price_scale
    
    # Volatility compression (US500 ~15% vs BTC ~60%)
    vol_scale = 0.30  # US500 vol is ~30% of BTC
    
    if len(df) > 1:
        df["return"] = df["close"].pct_change()
        df["return_scaled"] = df["return"] * vol_scale
        
        initial_price = df["close"].iloc[0]
        df["close_new"] = initial_price * (1 + df["return_scaled"]).cumprod()
        
        ratio = df["close_new"] / df["close"]
        ratio = ratio.fillna(1)
        
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * ratio
        
        df = df.drop(columns=["return", "return_scaled", "close_new"], errors="ignore")
    
    logger.info(f"Scaled to US500 proxy: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch historical data via SDK")
    parser.add_argument("--coin", type=str, default="BTC", help="Asset to fetch")
    parser.add_argument("--days", type=int, default=180, help="Days of data")
    parser.add_argument("--interval", type=str, default="1m", help="Candle interval")
    parser.add_argument("--proxy", action="store_true", help="Scale BTC to US500 proxy")
    args = parser.parse_args()
    
    # Fetch data
    df = fetch_candles_sdk(args.coin, args.interval, args.days)
    
    if df.empty:
        logger.error("Failed to fetch data")
        return
    
    # Save data
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save original
    original_path = data_dir / f"{args.coin}_candles_{args.interval}_{args.days}d.csv"
    df.to_csv(original_path, index=False)
    logger.info(f"Saved to {original_path}")
    
    # Create US500 proxy if requested
    if args.proxy and args.coin == "BTC":
        proxy_df = scale_btc_to_us500(df)
        proxy_path = data_dir / f"US500_proxy_candles_{args.interval}_{args.days}d.csv"
        proxy_df.to_csv(proxy_path, index=False)
        logger.info(f"Saved US500 proxy to {proxy_path}")
        
        # Stats
        returns = proxy_df['close'].pct_change().dropna()
        daily_vol = returns.std() * (1440 ** 0.5)
        annual_vol = daily_vol * (252 ** 0.5)
        
        print("\n" + "="*50)
        print("US500 PROXY DATA SUMMARY")
        print("="*50)
        print(f"Period: {proxy_df['datetime'].min()} to {proxy_df['datetime'].max()}")
        print(f"Candles: {len(proxy_df):,}")
        print(f"Price range: ${proxy_df['close'].min():.2f} - ${proxy_df['close'].max():.2f}")
        print(f"Annualized volatility: {annual_vol*100:.1f}%")
        print("="*50)


if __name__ == "__main__":
    main()
