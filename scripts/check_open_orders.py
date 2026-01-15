"""Check open orders on Hyperliquid."""
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.exchange import HyperliquidClient
from src.utils.config import Config


async def main():
    """Check open orders."""
    config = Config()
    client = HyperliquidClient(config)
    
    try:
        await client.connect()
        
        # Get open orders
        orders = await client.get_open_orders("US500")
        
        print(f"\n{'='*60}")
        print(f"OPEN ORDERS CHECK - US500")
        print(f"{'='*60}\n")
        print(f"Total Open Orders: {len(orders)}")
        
        if orders:
            print(f"\nOrder Details:")
            for i, order in enumerate(orders[:10], 1):  # Show first 10
                side = "BID" if order.get('side') == 'buy' else "ASK"
                price = order.get('limitPx', 'N/A')
                size = order.get('sz', 'N/A')
                oid = order.get('oid', 'N/A')
                print(f"  {i}. {side} @ ${price} x {size} (oid: {oid})")
            
            if len(orders) > 10:
                print(f"  ... and {len(orders) - 10} more")
        
        print(f"\n{'='*60}\n")
        
        # If too many orders, offer to cancel
        if len(orders) > 30:
            print(f"⚠️  WARNING: {len(orders)} orders open (max recommended: 30)")
            response = input("Cancel all orders? (yes/no): ")
            if response.lower() == 'yes':
                print("\nCancelling all orders...")
                result = await client.cancel_all_orders("US500")
                print(f"Result: {result}")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
