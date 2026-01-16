"""
Data Fetcher for AMM-500
=========================
Fetches historical data from multiple sources:
- xyz100 (^OEX S&P 100) via yfinance - Primary source
- BTC via Hyperliquid SDK - Fallback

Data is scaled to target US500 volatility (~12% annual).
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import time

import pandas as pd
import numpy as np
from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed - xyz100 data unavailable")

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False


class DataFetcher:
    """
    Multi-source data fetcher with volatility scaling.
    
    Primary: xyz100 (^OEX) via yfinance
    Fallback: BTC via Hyperliquid SDK
    
    Data is scaled to match US500 target volatility (~12% annual).
    """
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.target_vol = config.data.target_volatility
        self.use_xyz100 = config.data.use_xyz100_primary
        self.btc_fallback = config.data.btc_fallback_enabled
    
    async def fetch(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch and combine data from available sources.
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with OHLCV data scaled to target volatility
        """
        data = None
        
        # Try xyz100 first
        if self.use_xyz100 and YFINANCE_AVAILABLE:
            logger.info("Fetching xyz100 (^OEX) data...")
            data = await self._fetch_xyz100(days)
            if data is not None and len(data) > 0:
                logger.info(f"Fetched {len(data)} rows from xyz100")
        
        # Fallback to BTC
        if (data is None or len(data) < 100) and self.btc_fallback and HYPERLIQUID_AVAILABLE:
            logger.info("Fetching BTC fallback data...")
            btc_data = await self._fetch_btc(days)
            if btc_data is not None and len(btc_data) > 0:
                logger.info(f"Fetched {len(btc_data)} rows from BTC")
                
                # Scale BTC volatility to target
                btc_data = self._scale_volatility(btc_data)
                
                if data is None:
                    data = btc_data
                else:
                    # Combine: use xyz100 where available, fill gaps with BTC
                    data = self._combine_data(data, btc_data)
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to fetch any data")
        
        # Save to cache
        cache_path = self.data_dir / f"combined_{days}d.csv"
        data.to_csv(cache_path)
        logger.info(f"Saved {len(data)} rows to {cache_path}")
        
        return data
    
    async def _fetch_xyz100(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch xyz100 (^OEX) data via yfinance."""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            ticker = yf.Ticker("^OEX")
            
            # Fetch 1-minute data (max 7 days at a time)
            all_data = []
            end = datetime.now()
            
            for i in range(0, days, 5):
                chunk_end = end - timedelta(days=i)
                chunk_start = chunk_end - timedelta(days=5)
                
                try:
                    df = ticker.history(
                        start=chunk_start.strftime("%Y-%m-%d"),
                        end=chunk_end.strftime("%Y-%m-%d"),
                        interval="1m"
                    )
                    if len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error fetching xyz100 chunk: {e}")
                
                await asyncio.sleep(0.5)  # Rate limit
            
            if not all_data:
                return None
            
            data = pd.concat(all_data)
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            
            # Rename columns
            data.columns = [c.lower() for c in data.columns]
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching xyz100: {e}")
            return None
    
    async def _fetch_btc(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch BTC data via Hyperliquid SDK."""
        if not HYPERLIQUID_AVAILABLE:
            return None
        
        try:
            info = Info(constants.MAINNET_API_URL, skip_ws=True)
            
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # Fetch candles
            candles = info.candles_snapshot("BTC", "1m", start_time, end_time)
            
            if not candles:
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame(candles)
            data['timestamp'] = pd.to_datetime(data['t'], unit='ms')
            data = data.set_index('timestamp')
            
            data = data.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching BTC: {e}")
            return None
    
    def _scale_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data volatility to target."""
        if len(data) < 100:
            return data
        
        # Calculate realized volatility
        returns = data['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized from 1-min
        
        if realized_vol == 0:
            return data
        
        # Scale factor
        scale = self.target_vol / realized_vol
        
        # Apply scaling (scale returns, rebuild prices)
        scaled_returns = returns * scale
        
        # Rebuild OHLC from scaled returns
        initial_price = data['close'].iloc[0]
        scaled_close = initial_price * (1 + scaled_returns).cumprod()
        
        # Scale ratio for OHLC
        ratio = scaled_close / data['close'].iloc[1:]
        
        scaled_data = data.copy()
        scaled_data.loc[scaled_data.index[1:], 'close'] = scaled_close.values
        scaled_data.loc[scaled_data.index[1:], 'open'] = (data['open'].iloc[1:] * ratio).values
        scaled_data.loc[scaled_data.index[1:], 'high'] = (data['high'].iloc[1:] * ratio).values
        scaled_data.loc[scaled_data.index[1:], 'low'] = (data['low'].iloc[1:] * ratio).values
        
        logger.info(f"Scaled volatility from {realized_vol:.1%} to {self.target_vol:.1%}")
        
        return scaled_data
    
    def _combine_data(self, primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
        """Combine primary and fallback data."""
        # Use primary where available
        combined = primary.copy()
        
        # Fill gaps with fallback
        primary_start = primary.index.min()
        primary_end = primary.index.max()
        
        # Add fallback data before primary
        before = fallback[fallback.index < primary_start]
        if len(before) > 0:
            combined = pd.concat([before, combined])
        
        # Add fallback data after primary
        after = fallback[fallback.index > primary_end]
        if len(after) > 0:
            combined = pd.concat([combined, after])
        
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        
        return combined
    
    def load_cached(self, filename: str = None) -> Optional[pd.DataFrame]:
        """Load cached data if available."""
        if filename is None:
            # Find most recent cache
            cache_files = list(self.data_dir.glob("combined_*.csv"))
            if not cache_files:
                return None
            filename = max(cache_files, key=lambda p: p.stat().st_mtime).name
        
        cache_path = self.data_dir / filename
        if cache_path.exists():
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} rows from cache {filename}")
            return data
        
        return None


async def fetch_data(config, days: int = 30) -> pd.DataFrame:
    """Convenience function to fetch data."""
    fetcher = DataFetcher(config)
    return await fetcher.fetch(days)
