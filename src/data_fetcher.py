"""
US500 Historical Data Fetcher for Hyperliquid
Fetches actual market data for US500 or uses BTC as a proxy when insufficient.

This module handles:
- Fetching US500 candle data via /info endpoint
- Fetching US500 funding rate history
- Automatic fallback to BTC data as proxy when US500 history is insufficient
- Periodic checking for sufficient US500 data to switch from proxy

Data sources:
1. API: Candlestick and funding rate data for US500 (km:US500)
2. Fallback: BTC data as proxy (scaled for volatility differences)

WARNING: Backtesting with real data is essential for strategy validation.
Synthetic data can give misleading results.
"""

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from loguru import logger

# Try to import lz4 for S3 archive decompression
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 not installed. S3 archive decompression will not be available.")

# Hyperliquid API endpoints
MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"
TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"

# S3 archive bucket
S3_ARCHIVE_BUCKET = "hyperliquid-archive"
S3_MARKET_DATA_PREFIX = "market_data"


@dataclass
class CandleData:
    """Single candlestick data point."""
    timestamp: int  # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FundingData:
    """Funding rate data point."""
    timestamp: int
    coin: str
    funding_rate: float
    premium: float


class HyperliquidDataFetcher:
    """
    Fetches real historical data from Hyperliquid for backtesting.
    
    Supports both standard assets (BTC, ETH) and KM deployer assets (US500).
    
    Data sources:
    1. Candlestick data via POST /info with action "candleSnapshot"
    2. Funding rates via POST /info with action "fundingHistory"
    
    Usage:
        fetcher = HyperliquidDataFetcher()
        candles = await fetcher.fetch_candles("US500", days=30)
        funding = await fetcher.fetch_funding_history("US500", days=30)
    """
    
    # Available intervals for candles
    VALID_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    def __init__(self, use_testnet: bool = False):
        """Initialize data fetcher."""
        self.base_url = TESTNET_INFO_URL if use_testnet else MAINNET_INFO_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        self._last_request_time = 0
        
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _rate_limit_wait(self) -> None:
        """Simple rate limiting - max 10 requests per second."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.1:  # 100ms between requests
            await asyncio.sleep(0.1 - elapsed)
        self._last_request_time = time.time()
    
    async def _post(self, payload: Dict) -> Optional[Dict]:
        """Make a POST request to the info endpoint."""
        session = await self._ensure_session()
        await self._rate_limit_wait()
        
        try:
            async with session.post(
                self.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error: {response.status} - {await response.text()}")
                    return None
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def _get_coin_identifier(self, coin: str) -> str:
        """
        Get the correct coin identifier for API calls.
        
        For KM deployer assets like US500, we need to use the bare symbol
        in most API calls, not the full km:US500 format.
        """
        # Strip km: prefix if present for candle/funding requests
        if coin.startswith("km:"):
            return coin[3:]
        return coin
    
    async def fetch_candles(
        self,
        coin: str = "US500",
        interval: str = "1m",
        days: int = 30,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data from Hyperliquid.
        
        Args:
            coin: Trading pair (e.g., "US500", "BTC")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            days: Number of days of history to fetch
            end_time: End timestamp in ms (default: now)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of: {self.VALID_INTERVALS}")
        
        coin_id = self._get_coin_identifier(coin)
        logger.info(f"Fetching {days} days of {interval} candles for {coin_id}...")
        
        # Calculate time range
        if end_time is None:
            end_time = int(time.time() * 1000)
        
        # Interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }[interval]
        
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_candles = []
        current_start = start_time
        
        # Fetch in chunks (API typically limits to 5000 candles per request)
        max_candles_per_request = 5000
        chunk_ms = max_candles_per_request * interval_ms
        
        while current_start < end_time:
            chunk_end = min(current_start + chunk_ms, end_time)
            
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin_id,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": chunk_end
                }
            }
            
            result = await self._post(payload)
            
            if result:
                for candle in result:
                    all_candles.append({
                        "timestamp": candle["t"],
                        "open": float(candle["o"]),
                        "high": float(candle["h"]),
                        "low": float(candle["l"]),
                        "close": float(candle["c"]),
                        "volume": float(candle["v"]),
                    })
                logger.debug(f"Fetched {len(result)} candles from {datetime.fromtimestamp(current_start/1000)}")
            
            current_start = chunk_end
            
            # Small delay between chunks
            await asyncio.sleep(0.2)
        
        if not all_candles:
            logger.warning(f"No candle data retrieved for {coin_id}, returning empty DataFrame")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Create DataFrame and remove duplicates
        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Convert timestamp to datetime for easier use
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        logger.info(f"Fetched {len(df)} candles from {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    async def fetch_funding_history(
        self,
        coin: str = "US500",
        days: int = 30,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical funding rate data.
        
        Args:
            coin: Trading pair (e.g., "US500", "BTC")
            days: Number of days of history
            end_time: End timestamp in ms (default: now)
            
        Returns:
            DataFrame with columns: timestamp, coin, funding_rate, premium
        """
        coin_id = self._get_coin_identifier(coin)
        logger.info(f"Fetching {days} days of funding history for {coin_id}...")
        
        if end_time is None:
            end_time = int(time.time() * 1000)
        
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        payload = {
            "type": "fundingHistory",
            "coin": coin_id,
            "startTime": start_time,
            "endTime": end_time
        }
        
        result = await self._post(payload)
        
        if not result:
            logger.warning(f"No funding data retrieved for {coin_id}")
            return pd.DataFrame(columns=["timestamp", "coin", "funding_rate", "premium"])
        
        funding_data = []
        for item in result:
            funding_data.append({
                "timestamp": item["time"],
                "coin": item["coin"],
                "funding_rate": float(item["fundingRate"]),
                "premium": float(item.get("premium", 0)),
            })
        
        df = pd.DataFrame(funding_data)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} funding rate records")
        
        return df
    
    async def fetch_meta(self) -> Optional[Dict]:
        """Fetch market metadata (asset info, leverage limits, etc.)."""
        payload = {"type": "meta"}
        return await self._post(payload)
    
    async def fetch_all_mids(self) -> Optional[Dict]:
        """Fetch current mid prices for all assets."""
        payload = {"type": "allMids"}
        return await self._post(payload)
    
    async def fetch_l2_book(self, coin: str = "US500") -> Optional[Dict]:
        """Fetch current L2 order book snapshot."""
        coin_id = self._get_coin_identifier(coin)
        payload = {
            "type": "l2Book",
            "coin": coin_id
        }
        return await self._post(payload)
    
    async def check_asset_exists(self, coin: str) -> bool:
        """Check if an asset exists and is tradable on Hyperliquid."""
        coin_id = self._get_coin_identifier(coin)
        
        # Try to fetch current price
        mids = await self.fetch_all_mids()
        if mids and coin_id in mids:
            return True
        
        # Try L2 book
        book = await self.fetch_l2_book(coin_id)
        if book and book.get("levels"):
            return True
        
        return False


class US500DataManager:
    """
    Manages US500 historical data with automatic BTC proxy fallback.
    
    This class handles:
    1. Checking available US500 historical data
    2. Falling back to BTC data as proxy if insufficient
    3. Scaling BTC data to approximate US500 volatility
    4. Periodically checking for sufficient US500 data
    
    Usage:
        manager = US500DataManager()
        candles, funding, is_proxy = await manager.get_trading_data(days=180)
    """
    
    # US500 volatility is typically 20-30% of BTC volatility
    VOL_SCALE_FACTOR = 0.3
    
    # Minimum candles needed for reliable backtesting (6 months at 1m)
    MIN_CANDLES_FOR_TRADING = 259200  # 180 days * 24 * 60
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.fetcher = HyperliquidDataFetcher(use_testnet=False)
        
    async def close(self) -> None:
        """Close the data fetcher."""
        await self.fetcher.close()
    
    def _get_cache_path(self, coin: str, data_type: str, days: int) -> Path:
        """Get path for cached data file."""
        return self.data_dir / f"{coin}_{data_type}_{days}d.csv"
    
    def _load_cached(self, path: Path) -> Optional[pd.DataFrame]:
        """Load cached data if available and fresh."""
        if not path.exists():
            return None
        
        # Check file age (invalidate after 1 day)
        file_age = time.time() - path.stat().st_mtime
        if file_age > 86400:  # 24 hours
            logger.debug(f"Cache expired: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            logger.info(f"Loaded {len(df)} records from cache: {path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
            return None
    
    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save data to cache."""
        try:
            df.to_csv(path, index=False)
            logger.debug(f"Saved {len(df)} records to cache: {path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {path}: {e}")
    
    def _scale_btc_to_us500(self, btc_candles: pd.DataFrame) -> pd.DataFrame:
        """
        Scale BTC price data to approximate US500 characteristics.
        
        US500 (S&P 500):
        - Current price ~5000-6000 (vs BTC ~90000)
        - Volatility ~15% annualized (vs BTC ~50-80%)
        - More mean-reverting behavior
        
        This creates a proxy that:
        1. Scales price to US500 range
        2. Reduces volatility to match US500
        """
        if btc_candles.empty:
            return btc_candles
        
        df = btc_candles.copy()
        
        # Current US500 is ~5800, BTC is ~90000
        # Scale factor: 5800 / 90000 ≈ 0.064
        price_scale = 0.064
        
        # Scale prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] * price_scale
        
        # Reduce volatility by compressing returns toward mean
        # This makes the proxy more representative of index behavior
        if len(df) > 1:
            # Calculate returns
            df["return"] = df["close"].pct_change()
            
            # Compress returns (reduce vol)
            df["return_scaled"] = df["return"] * self.VOL_SCALE_FACTOR
            
            # Reconstruct prices from compressed returns
            initial_price = df["close"].iloc[0]
            df["close_scaled"] = initial_price * (1 + df["return_scaled"]).cumprod()
            
            # Scale OHLC proportionally
            scale_ratio = df["close_scaled"] / df["close"]
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * scale_ratio.fillna(1)
            
            # Clean up temp columns
            df = df.drop(columns=["return", "return_scaled", "close_scaled"], errors="ignore")
        
        # Volume doesn't need scaling for backtesting purposes
        
        logger.info(f"Scaled BTC data to US500 proxy: price range {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        return df
    
    async def get_trading_data(
        self,
        days: int = 180,
        interval: str = "1m",
        min_required_days: int = 180
    ) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Get trading data for US500, falling back to BTC proxy if needed.
        
        Args:
            days: Days of data to fetch
            interval: Candle interval
            min_required_days: Minimum days needed before using real data
            
        Returns:
            Tuple of (candles_df, funding_df, is_proxy)
            is_proxy is True if BTC data was used as proxy
        """
        min_required_candles = min_required_days * 24 * 60  # For 1m interval
        
        # Try to fetch US500 data
        logger.info("Attempting to fetch US500 historical data...")
        
        us500_candles = await self.fetcher.fetch_candles("US500", interval, days)
        
        # Check if we have enough US500 data
        if len(us500_candles) >= min_required_candles:
            logger.info(f"Sufficient US500 data available: {len(us500_candles)} candles")
            
            # Fetch funding rates
            us500_funding = await self.fetcher.fetch_funding_history("US500", days)
            
            # Cache the data
            self._save_cache(us500_candles, self._get_cache_path("US500", "candles_1m", days))
            self._save_cache(us500_funding, self._get_cache_path("US500", "funding", days))
            
            return us500_candles, us500_funding, False
        
        # Insufficient US500 data - use BTC as proxy
        logger.warning(f"Insufficient US500 data ({len(us500_candles)} candles), using BTC as proxy")
        
        # Try cache first
        btc_cache_path = self._get_cache_path("BTC", "candles_1m", days)
        btc_candles = self._load_cached(btc_cache_path)
        
        if btc_candles is None or len(btc_candles) < min_required_candles:
            # Fetch fresh BTC data
            logger.info(f"Fetching {days} days of BTC data as proxy...")
            btc_candles = await self.fetcher.fetch_candles("BTC", interval, days)
            
            if not btc_candles.empty:
                self._save_cache(btc_candles, btc_cache_path)
        
        # Scale BTC data to US500 characteristics
        us500_proxy = self._scale_btc_to_us500(btc_candles)
        
        # Fetch BTC funding as proxy (or use zeros)
        btc_funding_path = self._get_cache_path("BTC", "funding", days)
        btc_funding = self._load_cached(btc_funding_path)
        
        if btc_funding is None:
            btc_funding = await self.fetcher.fetch_funding_history("BTC", days)
            if not btc_funding.empty:
                self._save_cache(btc_funding, btc_funding_path)
        
        # Scale funding rates (index funding tends to be lower)
        if not btc_funding.empty:
            btc_funding["funding_rate"] = btc_funding["funding_rate"] * 0.5  # Reduce by 50%
            btc_funding["coin"] = "US500"  # Rename for consistency
        
        # Save proxy data
        self._save_cache(us500_proxy, self._get_cache_path("US500_proxy", "candles_1m", days))
        
        return us500_proxy, btc_funding, True
    
    async def check_data_availability(self) -> Dict:
        """
        Check current data availability for US500.
        
        Returns:
            Dict with status information
        """
        # Check if US500 asset exists
        exists = await self.fetcher.check_asset_exists("US500")
        
        if not exists:
            return {
                "asset_exists": False,
                "candles_available": 0,
                "funding_available": 0,
                "sufficient_for_trading": False,
                "using_proxy": True,
            }
        
        # Fetch recent data to count
        candles = await self.fetcher.fetch_candles("US500", "1m", days=7)
        funding = await self.fetcher.fetch_funding_history("US500", days=7)
        
        candles_per_day = len(candles) / 7 if len(candles) > 0 else 0
        estimated_total = candles_per_day * 180  # Estimate 6 months
        
        return {
            "asset_exists": True,
            "candles_available": len(candles),
            "candles_per_day": candles_per_day,
            "estimated_6mo_candles": estimated_total,
            "funding_available": len(funding),
            "sufficient_for_trading": estimated_total >= self.MIN_CANDLES_FOR_TRADING,
            "using_proxy": estimated_total < self.MIN_CANDLES_FOR_TRADING,
        }


class S3ArchiveFetcher:
    """
    Fetches raw trade data from Hyperliquid S3 archives.
    
    Archive format: s3://hyperliquid-archive/market_data/[date]/[hour]/trades/[COIN].lz4
    
    Note: US500 may not have S3 archives if it's a newer asset.
    Falls back to BTC data if US500 archives don't exist.
    """
    
    def __init__(self, cache_dir: str = "data/s3_cache"):
        """Initialize S3 archive fetcher."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._check_aws_cli()
    
    def _check_aws_cli(self) -> bool:
        """Check if AWS CLI is available."""
        try:
            result = subprocess.run(
                ["aws", "--version"],
                capture_output=True,
                text=True
            )
            self._aws_available = result.returncode == 0
            if not self._aws_available:
                logger.warning("AWS CLI not available. S3 archive fetching disabled.")
            return self._aws_available
        except FileNotFoundError:
            self._aws_available = False
            logger.warning("AWS CLI not installed. S3 archive fetching disabled.")
            return False
    
    def _get_s3_path(self, coin: str, date: str, hour: int) -> str:
        """Get S3 path for a specific date/hour."""
        return f"s3://{S3_ARCHIVE_BUCKET}/{S3_MARKET_DATA_PREFIX}/{date}/{hour:02d}/trades/{coin}.lz4"
    
    def _get_cache_path(self, coin: str, date: str, hour: int) -> Path:
        """Get local cache path for downloaded data."""
        return self.cache_dir / f"{coin}_{date}_{hour:02d}.lz4"
    
    def download_hour(self, coin: str, date: str, hour: int) -> Optional[Path]:
        """Download trade data for a specific hour."""
        if not self._aws_available:
            logger.error("AWS CLI not available")
            return None
        
        s3_path = self._get_s3_path(coin, date, hour)
        local_path = self._get_cache_path(coin, date, hour)
        
        # Check cache
        if local_path.exists():
            logger.debug(f"Using cached: {local_path}")
            return local_path
        
        logger.info(f"Downloading: {s3_path}")
        
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, str(local_path), "--no-sign-request"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return local_path
            else:
                logger.warning(f"Failed to download {s3_path}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {s3_path}: {e}")
            return None
    
    def decompress_lz4(self, lz4_path: Path) -> Optional[bytes]:
        """Decompress an LZ4 file."""
        if not LZ4_AVAILABLE:
            logger.error("lz4 package not installed")
            return None
        
        try:
            with open(lz4_path, 'rb') as f:
                return lz4.frame.decompress(f.read())
        except Exception as e:
            logger.error(f"Error decompressing {lz4_path}: {e}")
            return None
    
    def parse_trades(self, data: bytes) -> pd.DataFrame:
        """Parse trade data from decompressed bytes."""
        try:
            import json
            lines = data.decode('utf-8').strip().split('\n')
            trades = []
            
            for line in lines:
                if not line:
                    continue
                trade = json.loads(line)
                trades.append({
                    'timestamp': trade.get('time', trade.get('t')),
                    'price': float(trade.get('px', trade.get('p', 0))),
                    'size': float(trade.get('sz', trade.get('s', 0))),
                    'side': trade.get('side', 'unknown'),
                })
            
            return pd.DataFrame(trades)
            
        except Exception as e:
            logger.error(f"Error parsing trades: {e}")
            return pd.DataFrame()
    
    def fetch_trades_for_date(self, coin: str, date: str) -> pd.DataFrame:
        """Fetch all trades for a specific date."""
        all_trades = []
        
        for hour in range(24):
            lz4_path = self.download_hour(coin, date, hour)
            if lz4_path is None:
                continue
            
            data = self.decompress_lz4(lz4_path)
            if data is None:
                continue
            
            trades_df = self.parse_trades(data)
            if not trades_df.empty:
                all_trades.append(trades_df)
        
        if not all_trades:
            return pd.DataFrame()
        
        combined = pd.concat(all_trades, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(combined)} trades for {coin} on {date}")
        return combined


async def download_historical_data(
    coin: str = "US500",
    days: int = 30,
    interval: str = "1m",
    output_dir: str = "data"
) -> Tuple[str, str]:
    """
    Download and save historical data for backtesting.
    
    Args:
        coin: Trading pair ("US500" or "BTC")
        days: Days of history
        interval: Candle interval
        output_dir: Directory to save files
        
    Returns:
        Tuple of (candles_path, funding_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fetcher = HyperliquidDataFetcher(use_testnet=False)
    
    try:
        # Fetch candles
        candles_df = await fetcher.fetch_candles(coin, interval, days)
        candles_file = output_path / f"{coin}_candles_{interval}_{days}d.csv"
        candles_df.to_csv(candles_file, index=False)
        logger.info(f"Saved candles to {candles_file}")
        
        # Fetch funding rates
        funding_df = await fetcher.fetch_funding_history(coin, days)
        funding_file = output_path / f"{coin}_funding_{days}d.csv"
        funding_df.to_csv(funding_file, index=False)
        logger.info(f"Saved funding rates to {funding_file}")
        
        return str(candles_file), str(funding_file)
        
    finally:
        await fetcher.close()


def load_cached_data(
    coin: str = "US500",
    days: int = 30,
    interval: str = "1m",
    data_dir: str = "data"
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load cached historical data if available.
    
    Returns:
        Tuple of (candles_df, funding_df) or (None, None) if not found
    """
    data_path = Path(data_dir)
    
    # Try US500 first, then proxy
    for coin_prefix in [coin, f"{coin}_proxy"]:
        candles_file = data_path / f"{coin_prefix}_candles_{interval}_{days}d.csv"
        funding_file = data_path / f"{coin}_funding_{days}d.csv"
        
        if candles_file.exists():
            candles_df = pd.read_csv(candles_file)
            candles_df["datetime"] = pd.to_datetime(candles_df["datetime"])
            logger.info(f"Loaded {len(candles_df)} cached candles from {candles_file}")
            
            funding_df = None
            if funding_file.exists():
                funding_df = pd.read_csv(funding_file)
                if "datetime" in funding_df.columns:
                    funding_df["datetime"] = pd.to_datetime(funding_df["datetime"])
                logger.info(f"Loaded {len(funding_df)} cached funding records")
            
            return candles_df, funding_df
    
    return None, None


# Standalone script for downloading data
if __name__ == "__main__":
    import sys
    
    coin = sys.argv[1] if len(sys.argv) > 1 else "US500"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    interval = sys.argv[3] if len(sys.argv) > 3 else "1m"
    
    print(f"Downloading {days} days of {interval} data for {coin}...")
    print("(Will use BTC as proxy if US500 data insufficient)")
    
    async def main():
        manager = US500DataManager()
        try:
            candles, funding, is_proxy = await manager.get_trading_data(days=days)
            print(f"Downloaded {len(candles)} candles")
            print(f"Using proxy: {is_proxy}")
        finally:
            await manager.close()
    
    asyncio.run(main())
    
    print("Done!")
