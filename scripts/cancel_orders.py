#!/usr/bin/env python3
"""Cancel all open orders on Hyperliquid"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from dotenv import load_dotenv
from src.core.exchange import HyperliquidClient
from src.utils.config import Config

async def main():
    try:
        # Load configuration from .env file
        config = Config.load()
        
        print(f'🔑 Wallet: {config.wallet_address}\n')
        
        client = HyperliquidClient(config)
        await client.connect()
        
        # Check internal open orders tracking
        open_orders_count = len(client._open_orders)
        
        if open_orders_count > 0:
            print(f'⚠️  Found {open_orders_count} open order(s) in internal tracking\n')
            for order_id, order in list(client._open_orders.items())[:10]:  # Show first 10
                print(f'  - {order.side.value} {order.size} @ ${order.price:.2f} (ID: {order_id[:8]}...)')
        
        # Cancel all orders (this will also fetch from exchange)
        print(f'\n🗑️  Cancelling all orders...')
        cancelled_count = await client.cancel_all_orders(config.trading.symbol)
        
        if cancelled_count > 0:
            print(f'✅ Cancelled {cancelled_count} order(s) successfully')
        else:
            print('✅ No open orders to cancel')
        
        await client.disconnect()
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
