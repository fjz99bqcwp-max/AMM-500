#!/usr/bin/env python3
"""Debug order sizing issue."""
import sys
sys.path.insert(0, '/Users/nheosdisplay/VSC/AMM/AMM-500')

from src.config import Config
from src.risk import RiskManager
from src.exchange import Exchange

async def main():
    # Load config
    config = Config.load()
    print(f"Symbol: {config.trading.symbol}")
    print(f"Collateral: ${config.trading.collateral}")
    print(f"Max exposure: ${config.trading.max_net_exposure}")
    
    # Create exchange and risk manager
    client = Exchange(config)
    await client.connect()
    
    # Get current price
    book = await client.get_orderbook()
    if not book:
        print("Could not get orderbook!")
        await client.disconnect()
        return
    
    price = book.mid_price
    print(f"\nCurrent price: ${price:.2f}")
    
    # Simulate risk metrics
    risk_mgr = RiskManager(config, client)
    await risk_mgr.initialize()
    
    # Calculate order size
    size = risk_mgr.calculate_order_size(price, "both", None)
    print(f"\nCalculated order size: {size:.8f} contracts")
    print(f"Notional value: ${size * price:.2f}")
    print(f"Minimum requirement: $10.00")
    
    if size * price < 10.0:
        print(f"ERROR: Size too small! Need at least {10.0 / price:.8f} contracts")
    else:
        print(f"OK: Size meets minimum requirement")
    
    await client.disconnect()

import asyncio
asyncio.run(main())
