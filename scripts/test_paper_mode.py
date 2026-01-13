#!/usr/bin/env python3
"""
Test paper trading mode to debug order placement.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.exchange import HyperliquidClient
from src.risk import RiskManager
from src.strategy import MarketMakingStrategy


async def test_paper_mode():
    """Test paper mode with debug output."""
    print("Loading config...")
    config = Config.load()
    
    print(f"Config loaded: {config.trading.symbol}")
    print(f"Paper trading: {config.execution.paper_trading}")
    print(f"Collateral: ${config.trading.collateral:,.2f}")
    print(f"Quote refresh interval: {config.execution.quote_refresh_interval}s")
    
    print("\nConnecting to exchange...")
    client = HyperliquidClient(config)
    await client.connect()
    
    print("Creating risk manager...")
    risk = RiskManager(config, client)
    await risk.initialize()
    
    print(f"Risk manager equity: ${risk._equity:,.2f}")
    
    print("\nCreating strategy...")
    strategy = MarketMakingStrategy(config, client, risk)
    await strategy.start()
    
    print("Strategy started!")
    print(f"Quote interval: {strategy.quote_interval}s")
    print(f"Order levels: {strategy.order_levels}")
    print(f"Min spread: {strategy.min_spread_bps} bps")
    print(f"Max spread: {strategy.max_spread_bps} bps")
    
    print("\nRunning 10 iterations...")
    for i in range(10):
        print(f"\n--- Iteration {i+1} ---")
        await strategy.run_iteration()
        
        status = strategy.get_status()
        print(f"Position: {status.position}")
        print(f"Active bids: {len(status.active_bids)}")
        print(f"Active asks: {len(status.active_asks)}")
        print(f"Actions: {status.actions_today}")
        
        await asyncio.sleep(2)
    
    print("\nStopping...")
    await strategy.stop()
    await client.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(test_paper_mode())
