# Data Directory

This directory contains historical data for backtesting the AMM-500 US500 trading bot.

## Data Files

### US500 Data (Primary)
- `US500_candles_1m_*d.csv` - US500 1-minute candle data
- `US500_funding_*d.csv` - US500 funding rate history

### BTC Proxy Data (Fallback)
- `US500_proxy_candles_1m_*d.csv` - BTC data scaled to US500 characteristics
- `BTC_candles_1m_*d.csv` - Raw BTC candle data (for scaling)
- `BTC_funding_*d.csv` - BTC funding rates (scaled for proxy use)

## Data Sources

1. **Hyperliquid API** - Primary source for US500 and BTC candle/funding data
2. **S3 Archives** - Historical trade data (if available)

## Fetching Data

```bash
# Fetch US500 data (uses BTC proxy if insufficient)
python amm-500.py --fetch-data --fetch-days 180

# Or use the data fetcher directly
python -m src.data_fetcher US500 180 1m
```

## Proxy Data Explanation

US500 was launched on Hyperliquid via the KM deployer and may not have
sufficient historical data (6+ months) for reliable backtesting. When
this is the case, the bot automatically uses BTC data as a proxy:

1. BTC prices are scaled to US500 range (~5800 vs ~90000)
2. Volatility is compressed to match US500 characteristics (30% of BTC vol)
3. Funding rates are scaled down by 50%

The bot will automatically switch to real US500 data once sufficient
history is available.

## Cache Expiration

Cached data files expire after 24 hours and are re-fetched on next use.
