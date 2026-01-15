"""
XYZ100 FALLBACK DATA FETCHER
Add to /Users/nheosdisplay/VSC/AMM/AMM-500/src/data_fetcher.py

Fetches S&P 100 (^OEX) data via yfinance as US500 proxy when:
- US500-USDH historical data insufficient (<30 days)
- Hyperliquid API unavailable
- Backtesting on synthetic data

Scales volatility to match US500 characteristics.
"""

import yfinance as yf
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class XYZ100FallbackFetcher:
    """
    Fetch S&P 100 (^OEX) data as US500 proxy.
    
    Why ^OEX:
    - S&P 100 is subset of S&P 500 (100 largest stocks)
    - High correlation with S&P 500 (>0.98)
    - Available via yfinance with good data quality
    - 1-minute bars available for HFT backtesting
    
    Scaling:
    - Volatility: Scale OEX vol to match US500 target (typically same)
    - Price: Convert OEX levels (~1800) to US500 levels (~6900) via ratio
    """
    
    def __init__(self):
        self.symbol = "^OEX"  # S&P 100 Index
        self.us500_symbol = "US500"
        # Price scaling factor (US500 ~= OEX * 3.83 as of Jan 2026)
        # US500 ~6900, OEX ~1800 → ratio ~3.83
        self.price_scale_factor = 3.83
    
    async def fetch_xyz100_data(
        self, 
        days: int = 30, 
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch S&P 100 historical data via yfinance.
        
        Args:
            days: Number of days to fetch (max 30 for 1m, 60 for 5m)
            interval: yfinance interval (1m, 5m, 15m, 1h, 1d)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            logger.info(f"Fetching S&P 100 (^OEX) data: {days} days, {interval} interval")
            
            # Calculate start/end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch from yfinance
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                back_adjust=False
            )
            
            if df.empty:
                logger.error("No data received from yfinance")
                return None
            
            # Rename columns to match our format
            df = df.reset_index()
            df = df.rename(columns={
                "Datetime": "timestamp" if interval in ["1m", "5m", "15m", "1h"] else "Date",
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Convert timestamp to milliseconds
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**6
            
            # Scale prices to US500 levels (OEX ~1800 → US500 ~6900)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * self.price_scale_factor
            
            logger.info(f"Fetched {len(df)} bars from ^OEX (scaled to US500 levels)")
            logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        except Exception as e:
            logger.error(f"Failed to fetch ^OEX data: {e}")
            return None
    
    def scale_volatility(
        self, 
        df: pd.DataFrame, 
        target_vol: float = 0.12
    ) -> pd.DataFrame:
        """
        Scale volatility to match US500 target.
        
        Args:
            df: DataFrame with OHLCV data
            target_vol: Target annualized volatility (default 12% for US500)
        
        Returns:
            Scaled DataFrame
        """
        if df.empty:
            return df
        
        # Calculate current volatility
        returns = np.diff(np.log(df["close"]))
        current_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
        
        if current_vol == 0:
            logger.warning("Zero volatility detected - skipping scaling")
            return df
        
        # Scaling factor
        vol_scale = target_vol / current_vol
        
        logger.info(f"Scaling volatility: {current_vol:.2%} → {target_vol:.2%} (factor: {vol_scale:.3f})")
        
        # Scale prices around mean
        mean_price = df["close"].mean()
        for col in ["open", "high", "low", "close"]:
            df[col] = mean_price + (df[col] - mean_price) * vol_scale
        
        return df
    
    async def fetch_with_fallback(
        self,
        primary_fetcher,  # HyperliquidDataFetcher instance
        symbol: str = "US500",
        days: int = 30,
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with automatic fallback to ^OEX.
        
        Tries:
        1. Hyperliquid US500 data (primary)
        2. ^OEX proxy data (fallback)
        
        Returns:
            DataFrame with US500 data (real or proxy)
        """
        # Try primary (Hyperliquid US500)
        try:
            logger.info(f"Attempting to fetch {symbol} data from Hyperliquid...")
            df_primary = await primary_fetcher.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            )
            
            if df_primary is not None and len(df_primary) >= days * 24 * 60 * 0.5:
                # Sufficient data (at least 50% of expected)
                logger.info(f"Using Hyperliquid {symbol} data: {len(df_primary)} bars")
                return df_primary
            else:
                logger.warning(f"Insufficient {symbol} data from Hyperliquid ({len(df_primary) if df_primary is not None else 0} bars)")
        
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Hyperliquid: {e}")
        
        # Fallback to ^OEX
        logger.warning(f"Falling back to ^OEX proxy data for {symbol}")
        df_fallback = await self.fetch_xyz100_data(days=days, interval=interval)
        
        if df_fallback is not None:
            # Scale volatility to match US500
            df_fallback = self.scale_volatility(df_fallback, target_vol=0.12)
            logger.info(f"Using ^OEX proxy: {len(df_fallback)} bars (scaled to US500)")
            return df_fallback
        
        logger.error("All data sources failed - no data available")
        return None


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def fetch_us500_data_with_fallback(days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch US500 data with automatic ^OEX fallback.
    
    Usage in backtest.py or data_fetcher.py:
        df = await fetch_us500_data_with_fallback(days=30)
        if df is not None:
            # Run backtest on df
            ...
    """
    from .data_fetcher import HyperliquidDataFetcher
    
    config = Config.load()
    primary_fetcher = HyperliquidDataFetcher(config)
    fallback_fetcher = XYZ100FallbackFetcher()
    
    df = await fallback_fetcher.fetch_with_fallback(
        primary_fetcher=primary_fetcher,
        symbol="US500",
        days=days,
        interval="1m"
    )
    
    return df


# =============================================================================
# INTEGRATION TO data_fetcher.py
# =============================================================================

# Example integration code (commented out - add to HyperliquidDataFetcher class):
"""
async def fetch_candles_with_xyz100_fallback(
    self,
    symbol: str = "US500",
    interval: str = "1m",
    days: int = 30
) -> Optional[List[CandleData]]:
    # Fetch candles with automatic ^OEX fallback
    
    from .xyz100_fallback import XYZ100FallbackFetcher
    from .data_fetcher import CandleData
    
    # Try primary
    candles = await self.fetch_candles(
        symbol=symbol,
        interval=interval,
        start_time=int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    )
    
    if candles and len(candles) >= days * 24 * 60 * 0.5:
        logger.info(f"Using Hyperliquid {symbol}: {len(candles)} candles")
        return candles
    
    # Fallback
    logger.warning(f"Insufficient {symbol} data - using ^OEX fallback")
    fallback = XYZ100FallbackFetcher()
    df_fallback = await fallback.fetch_xyz100_data(days=days, interval=interval)
    
    if df_fallback is None:
        return None
    
    # Scale volatility
    df_fallback = fallback.scale_volatility(df_fallback, target_vol=0.12)
    
    # Convert to CandleData
    candles = []
    for _, row in df_fallback.iterrows():
        candles.append(CandleData(
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"])
        ))
    
    logger.info(f"Using ^OEX proxy: {len(candles)} candles (scaled to US500)")
    return candles
"""
