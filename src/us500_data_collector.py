"""
US500 & XYZ100 Data Collector for AMM-500

Continuously collects US500 1-minute candle data while the bot runs
and provides XYZ100 historical data as fallback proxy when US500 history is insufficient.

Features:
- Real-time US500 candle collection (saved to data/us500_candles_1m.csv)
- XYZ100 historical data fetching for backtests
- Automatic fallback: Use XYZ100 when US500 history < 30 days
- Append-only CSV format for efficient storage
"""

import asyncio
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from loguru import logger

# Hyperliquid API endpoint
MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"

# Data file paths
DATA_DIR = Path(__file__).parent.parent / "data"
US500_CANDLES_FILE = DATA_DIR / "us500_candles_1m.csv"
XYZ100_HISTORICAL_FILE = DATA_DIR / "xyz100_historical.csv"
XYZ100_METADATA_FILE = DATA_DIR / "xyz100_metadata.json"


class US500DataCollector:
    """Continuously collect US500 candle data and provide XYZ100 fallback."""

    def __init__(self):
        """Initialize data collector."""
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_candle_time = 0
        self._collecting = False
        
        # Ensure data directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        # Initialize US500 candles file with header if not exists
        if not US500_CANDLES_FILE.exists():
            with open(US500_CANDLES_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            logger.info(f"Created US500 candles file: {US500_CANDLES_FILE}")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _fetch_latest_candle(self, symbol: str = "km:US500") -> Optional[Dict]:
        """Fetch the latest 1-minute candle for symbol."""
        session = await self._ensure_session()
        
        # Get current time in milliseconds
        end_time = int(time.time() * 1000)
        start_time = end_time - 120000  # 2 minutes ago
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,  # Use full km:US500 format
                "interval": "1m",
                "startTime": start_time,
                "endTime": end_time
            }
        }
        
        try:
            async with session.post(
                MAINNET_INFO_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        # Return the most recent candle
                        latest = data[-1]
                        return {
                            'timestamp': latest['t'],
                            'open': float(latest['o']),
                            'high': float(latest['h']),
                            'low': float(latest['l']),
                            'close': float(latest['c']),
                            'volume': float(latest['v'])
                        }
                else:
                    logger.warning(f"API error fetching {symbol} candle: {response.status}")
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} candle: {e}")
        
        return None
    
    async def save_candle(self, candle: Dict) -> None:
        """Save candle to CSV file (append mode)."""
        try:
            with open(US500_CANDLES_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    candle['timestamp'],
                    candle['open'],
                    candle['high'],
                    candle['low'],
                    candle['close'],
                    candle['volume']
                ])
            # Debug log removed - candles saved silently
        except Exception as e:
            logger.error(f"Failed to save candle: {e}")
    
    async def collect_continuously(self, interval: int = 60) -> None:
        """
        Continuously collect US500 candles every minute.
        
        Args:
            interval: Collection interval in seconds (default 60 for 1-minute candles)
        """
        self._collecting = True
        logger.info("Starting continuous US500 candle collection...")
        
        while self._collecting:
            try:
                # Fetch latest candle
                candle = await self._fetch_latest_candle("km:US500")
                
                if candle:
                    # Only save if it's a new candle (different timestamp)
                    if candle['timestamp'] != self._last_candle_time:
                        await self.save_candle(candle)
                        self._last_candle_time = candle['timestamp']
                
                # Wait for next collection interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(interval)
    
    def stop_collection(self) -> None:
        """Stop continuous collection."""
        self._collecting = False
        logger.info("Stopped US500 candle collection")
    
    async def fetch_xyz100_historical(self, days: int = 180) -> Optional[pd.DataFrame]:
        """
        Fetch XYZ100 historical data for backtests (fallback proxy).
        
        Args:
            days: Number of days of history to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching XYZ100 historical data ({days} days)...")
        
        session = await self._ensure_session()
        
        # Calculate time range
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_candles = []
        
        # Fetch in chunks (max ~4 days per request based on API limitation)
        chunk_size = 4 * 24 * 60 * 60 * 1000  # 4 days in milliseconds
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": "xyz:XYZ100",  # Full deployer prefix format
                    "interval": "1m",
                    "startTime": current_start,
                    "endTime": current_end
                }
            }
            
            try:
                async with session.post(
                    MAINNET_INFO_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            for candle in data:
                                all_candles.append({
                                    'timestamp': candle['t'],
                                    'open': float(candle['o']),
                                    'high': float(candle['h']),
                                    'low': float(candle['l']),
                                    'close': float(candle['c']),
                                    'volume': float(candle['v'])
                                })
                            logger.info(f"Fetched {len(data)} XYZ100 candles")
                    else:
                        logger.warning(f"API error: {response.status}")
                        break
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to fetch XYZ100 chunk: {e}")
                break
            
            current_start = current_end
        
        if all_candles:
            df = pd.DataFrame(all_candles)
            
            # Save to file
            df.to_csv(XYZ100_HISTORICAL_FILE, index=False)
            logger.info(f"Saved {len(df)} XYZ100 candles to {XYZ100_HISTORICAL_FILE}")
            
            # Save metadata
            import json
            metadata = {
                'symbol': 'XYZ100',
                'candles': len(df),
                'days': days,
                'start_time': int(df['timestamp'].min()),  # Convert to int
                'end_time': int(df['timestamp'].max()),  # Convert to int
                'fetched_at': int(time.time() * 1000)
            }
            with open(XYZ100_METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return df
        else:
            logger.error("No XYZ100 data fetched")
            return None
    
    def get_us500_data(self) -> Optional[pd.DataFrame]:
        """
        Load US500 candles from file.
        
        Returns:
            DataFrame with US500 candles or None if file doesn't exist
        """
        if US500_CANDLES_FILE.exists():
            try:
                df = pd.read_csv(US500_CANDLES_FILE)
                logger.info(f"Loaded {len(df)} US500 candles from file")
                return df
            except Exception as e:
                logger.error(f"Failed to load US500 data: {e}")
        return None
    
    def get_xyz100_data(self) -> Optional[pd.DataFrame]:
        """
        Load XYZ100 historical data from file.
        
        Returns:
            DataFrame with XYZ100 candles or None if file doesn't exist
        """
        if XYZ100_HISTORICAL_FILE.exists():
            try:
                df = pd.read_csv(XYZ100_HISTORICAL_FILE)
                logger.info(f"Loaded {len(df)} XYZ100 candles from file")
                return df
            except Exception as e:
                logger.error(f"Failed to load XYZ100 data: {e}")
        return None
    
    def get_data_for_backtest(self, min_days: int = 30) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Get data for backtesting with automatic fallback.
        
        Strategy:
        1. Check US500 data - if >= min_days, use it
        2. Otherwise, check XYZ100 data - if >= min_days, use it as proxy
        3. If both insufficient, return None
        
        Args:
            min_days: Minimum days of data required
            
        Returns:
            Tuple of (DataFrame, source_name) where source is "US500" or "XYZ100" or None
        """
        # Try US500 first
        us500_df = self.get_us500_data()
        if us500_df is not None and len(us500_df) > 0:
            # Calculate days of data
            time_range = (us500_df['timestamp'].max() - us500_df['timestamp'].min()) / 1000 / 86400
            if time_range >= min_days:
                logger.info(f"Using US500 data: {time_range:.1f} days available")
                return us500_df, "US500"
            else:
                logger.warning(f"US500 data insufficient: only {time_range:.1f} days (need {min_days})")
        
        # Fallback to XYZ100
        xyz100_df = self.get_xyz100_data()
        if xyz100_df is not None and len(xyz100_df) > 0:
            time_range = (xyz100_df['timestamp'].max() - xyz100_df['timestamp'].min()) / 1000 / 86400
            if time_range >= min_days:
                logger.info(f"Using XYZ100 as fallback proxy: {time_range:.1f} days available")
                return xyz100_df, "XYZ100"
            else:
                logger.warning(f"XYZ100 data insufficient: only {time_range:.1f} days (need {min_days})")
        
        logger.error(f"Insufficient data for backtest (need {min_days} days)")
        return None, "NONE"


# Global collector instance
_collector: Optional[US500DataCollector] = None


def get_collector() -> US500DataCollector:
    """Get or create global collector instance."""
    global _collector
    if _collector is None:
        _collector = US500DataCollector()
    return _collector


async def start_collection() -> None:
    """Start continuous US500 data collection in background."""
    collector = get_collector()
    await collector.collect_continuously()


async def fetch_xyz100_for_backtest(days: int = 180) -> None:
    """Fetch XYZ100 historical data for backtesting."""
    collector = get_collector()
    await collector.fetch_xyz100_historical(days)


if __name__ == "__main__":
    # Test the collector
    async def test():
        collector = US500DataCollector()
        
        # Fetch XYZ100 historical
        print("Fetching XYZ100 historical data...")
        await collector.fetch_xyz100_historical(days=30)
        
        # Test getting data for backtest
        df, source = collector.get_data_for_backtest(min_days=7)
        if df is not None:
            print(f"\nBacktest data source: {source}")
            print(f"Candles: {len(df)}")
            print(f"Time range: {datetime.fromtimestamp(df['timestamp'].min()/1000)} to {datetime.fromtimestamp(df['timestamp'].max()/1000)}")
        
        await collector.close()
    
    asyncio.run(test())
