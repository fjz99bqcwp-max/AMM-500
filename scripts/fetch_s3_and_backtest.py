#!/usr/bin/env python3
"""
Fetch S3 archive data and run backtest.

Hyperliquid S3 archive format:
s3://hyperliquid-archive/market_data/[date]/[hour]/trades/BTC.lz4

This script:
1. Downloads trade data from S3 (public, no auth required)
2. Decompresses LZ4 files
3. Resamples to OHLCV format
4. Runs backtest with real historical data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import lz4.frame
import pandas as pd
import numpy as np
from loguru import logger


# S3 configuration
S3_BUCKET = "hyperliquid-archive"
S3_PREFIX = "market_data"


def download_s3_data(
    coin: str = "BTC",
    start_date: str = None,
    end_date: str = None,
    days: int = 7,
    output_dir: str = "data/s3_cache"
) -> list:
    """
    Download trade data from Hyperliquid S3 archive.
    
    The S3 bucket is public, no authentication required.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure boto3 for public access (no auth)
    s3 = boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED),
        region_name='us-east-1'
    )
    
    # Calculate date range
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()
    
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start = end - timedelta(days=days)
    
    logger.info(f"Fetching S3 data from {start.date()} to {end.date()}")
    
    downloaded_files = []
    current = start
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        for hour in range(24):
            s3_key = f"{S3_PREFIX}/{date_str}/{hour:02d}/trades/{coin}.lz4"
            local_file = output_path / f"{coin}_{date_str}_{hour:02d}.lz4"
            
            # Skip if already cached
            if local_file.exists():
                logger.debug(f"Using cached: {local_file}")
                downloaded_files.append(local_file)
                continue
            
            try:
                logger.info(f"Downloading: s3://{S3_BUCKET}/{s3_key}")
                s3.download_file(S3_BUCKET, s3_key, str(local_file))
                downloaded_files.append(local_file)
            except Exception as e:
                logger.warning(f"Failed to download {s3_key}: {e}")
        
        current += timedelta(days=1)
    
    logger.info(f"Downloaded {len(downloaded_files)} files")
    return downloaded_files


def decompress_and_parse(lz4_files: list) -> pd.DataFrame:
    """Decompress LZ4 files and parse trade data."""
    import json
    
    all_trades = []
    
    for lz4_file in lz4_files:
        try:
            with open(lz4_file, 'rb') as f:
                data = lz4.frame.decompress(f.read())
            
            lines = data.decode('utf-8').strip().split('\n')
            
            for line in lines:
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                    all_trades.append({
                        'timestamp': trade.get('time', trade.get('t', 0)),
                        'price': float(trade.get('px', trade.get('p', 0))),
                        'size': float(trade.get('sz', trade.get('s', 0))),
                        'side': trade.get('side', 'unknown'),
                    })
                except json.JSONDecodeError:
                    continue
            
            logger.debug(f"Parsed {len(lines)} trades from {lz4_file.name}")
            
        except Exception as e:
            logger.warning(f"Error processing {lz4_file}: {e}")
    
    if not all_trades:
        logger.warning("No trades found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Total trades: {len(df)}")
    return df


def resample_to_ohlcv(trades: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    """Resample tick trades to OHLCV format."""
    if trades.empty:
        return pd.DataFrame()
    
    trades['datetime'] = pd.to_datetime(trades['timestamp'], unit='ms')
    trades = trades.set_index('datetime')
    
    ohlcv = trades['price'].resample(interval).ohlc()
    ohlcv['volume'] = trades['size'].resample(interval).sum()
    
    ohlcv = ohlcv.reset_index()
    ohlcv['timestamp'] = (ohlcv['datetime'].astype(np.int64) / 1e6).astype(np.int64)
    ohlcv = ohlcv.dropna()
    
    logger.info(f"Resampled to {len(ohlcv)} OHLCV bars")
    return ohlcv


def run_backtest_with_data(ohlcv_data: pd.DataFrame):
    """Run backtest with the S3 data."""
    from src.backtest import BacktestEngine, BacktestConfig, MonteCarloSimulator
    
    # Configure backtest per recommendations
    config = BacktestConfig(
        initial_capital=2000.0,
        leverage=10,  # Target x10 minimum
        min_spread_bps=3.0,
        max_spread_bps=40.0,
        order_levels=12,
        rebalance_interval=15,
        max_drawdown=0.05,
        stop_loss_pct=0.015,
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(ohlcv_data)
    
    # Print summary
    print(result.summary())
    
    # Save plot
    plot_path = Path(__file__).parent.parent / "logs" / "backtest_s3_results.png"
    engine.plot_results(result, save_path=plot_path)
    
    # Run Monte Carlo with volatility scenarios
    mc = MonteCarloSimulator(config)
    mc_results = mc.run(result, num_simulations=10000, horizon_days=30)
    
    print("\n========== Monte Carlo Analysis ==========")
    print(f"Simulations: {mc_results.get('num_simulations', 'N/A')}")
    print(f"Horizon: {mc_results.get('horizon_days', 'N/A')} days")
    print(f"Liquidation Probability: {mc_results.get('liquidation_probability', 0):.2%}")
    print(f"Expected Return: {mc_results.get('expected_return', 0):.1f}%")
    print(f"Probability of Profit: {mc_results.get('probability_of_profit', 0):.2%}")
    print(f"Meets x10 Target (<15% liq): {mc_results.get('meets_x10_target', 'N/A')}")
    print(f"Meets x25 Target (<5% liq): {mc_results.get('meets_x25_target', 'N/A')}")
    print(f"Sharpe Estimate: {mc_results.get('sharpe_estimate', 0):.2f}")
    print("==========================================")
    
    # Print volatility scenario results
    if 'volatility_scenarios' in mc_results:
        print("\n========== Volatility Scenarios ==========")
        for scenario, results in mc_results['volatility_scenarios'].items():
            print(f"{scenario}: Liq Prob={results['liquidation_probability']:.2%}, "
                  f"Return={results['expected_return']:.1f}%")
        print("==========================================")
    
    return result


def main():
    """Main function to fetch S3 data and run backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch S3 data and run backtest")
    parser.add_argument("--coin", default="BTC", help="Coin to fetch")
    parser.add_argument("--days", type=int, default=7, help="Days of data")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       Hyperliquid HFT Bot - S3 Data Backtest                 ║
╠══════════════════════════════════════════════════════════════╣
║  Fetching {args.days} days of {args.coin} trade data from S3 archive       ║
║  Source: s3://hyperliquid-archive/market_data/               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Download S3 data
    print("Step 1: Downloading S3 archive data...")
    lz4_files = download_s3_data(
        coin=args.coin,
        days=args.days,
        start_date=args.start,
        end_date=args.end
    )
    
    if not lz4_files:
        print("ERROR: No S3 data found. The archive may not have data for the requested dates.")
        print("Try using API data instead: python mmb-1.py --backtest --days 30")
        return
    
    # Step 2: Decompress and parse
    print("\nStep 2: Decompressing and parsing trade data...")
    trades = decompress_and_parse(lz4_files)
    
    if trades.empty:
        print("ERROR: No trades found in downloaded files.")
        return
    
    # Step 3: Resample to OHLCV
    print("\nStep 3: Resampling to 1-minute OHLCV...")
    ohlcv = resample_to_ohlcv(trades, "1min")
    
    if ohlcv.empty:
        print("ERROR: Failed to resample data.")
        return
    
    # Save OHLCV data
    cache_path = Path("data") / f"{args.coin}_s3_ohlcv_{args.days}d.csv"
    cache_path.parent.mkdir(exist_ok=True)
    ohlcv.to_csv(cache_path, index=False)
    print(f"Saved OHLCV data to {cache_path}")
    
    # Step 4: Run backtest
    print("\nStep 4: Running backtest with S3 data...")
    run_backtest_with_data(ohlcv)


if __name__ == "__main__":
    main()
