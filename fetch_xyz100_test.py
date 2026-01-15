#!/usr/bin/env python3
"""
Quick test script to fetch xyz100 (S&P100) data
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from xyz100_fallback import XYZ100FallbackFetcher

async def main():
    print("=" * 60)
    print("FETCHING XYZ100 (S&P100) DATA")
    print("=" * 60)
    print()
    
    fetcher = XYZ100FallbackFetcher()
    
    # Fetch 30 days of 1-minute data
    print("Fetching 30 days of ^OEX (S&P100) data...")
    print("Note: yfinance 1m data limited to 7 days, using 5m instead")
    print()
    
    df = await fetcher.fetch_xyz100_data(days=30, interval="5m")
    
    if df is not None:
        print("\n✅ SUCCESS!")
        print(f"Fetched {len(df)} bars")
        print(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print()
        print("Sample data (first 5 rows):")
        print(df.head())
        print()
        
        # Save to CSV
        output_file = "data/xyz100_proxy.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")
        
        # Scale volatility test
        print("\nScaling volatility to 12% target...")
        df_scaled = fetcher.scale_volatility(df.copy(), target_vol=0.12)
        print(f"✅ Volatility scaled")
        print(f"Price range after scaling: ${df_scaled['low'].min():.2f} - ${df_scaled['high'].max():.2f}")
        
        output_scaled = "data/xyz100_scaled.csv"
        df_scaled.to_csv(output_scaled, index=False)
        print(f"✅ Saved scaled data to {output_scaled}")
    else:
        print("\n❌ FAILED to fetch data")
        return 1
    
    print("\n" + "=" * 60)
    print("COMPLETE - Ready for backtesting")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
