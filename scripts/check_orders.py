#!/usr/bin/env python3
"""Check and analyze all open orders on the account."""

from hyperliquid.info import Info
from hyperliquid.utils import constants
import json

def main():
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
    
    # Get user state 
    state = info.user_state(wallet)
    print('=== Account State ===')
    margin = state.get('marginSummary', {})
    print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
    print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
    print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
    print()
    
    # Get open orders via different methods
    print('=== Checking Open Orders ===')
    
    # Method 1: Regular open_orders
    try:
        orders = info.open_orders(wallet)
        print(f'open_orders(): {len(orders)} orders')
        if orders:
            for o in orders[:5]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')}")
            if len(orders) > 5:
                print(f"  ... and {len(orders) - 5} more")
    except Exception as e:
        print(f'open_orders error: {e}')
    
    # Method 2: Frontend open orders
    print()
    try:
        frontend = info.frontend_open_orders(wallet)
        print(f'frontend_open_orders(): {len(frontend)} orders')
        if frontend:
            # Count by symbol
            by_symbol = {}
            for o in frontend:
                sym = o.get('coin', 'unknown')
                by_symbol[sym] = by_symbol.get(sym, 0) + 1
            print(f'By symbol: {by_symbol}')
            
            # Show sample orders
            for o in frontend[:5]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(frontend) > 5:
                print(f"  ... and {len(frontend) - 5} more")
    except Exception as e:
        print(f'frontend_open_orders error: {e}')
    
    # Method 3: User state open orders
    print()
    print('=== User State Analysis ===')
    
    # Check for perp_dex orders
    perp_dexs = state.get('perpDexs', [])
    print(f'perp_dexs in state: {perp_dexs}')
    
    # Check asset positions
    print()
    print('=== Positions ===')
    positions = state.get('assetPositions', [])
    for pos in positions:
        p = pos.get('position', {})
        szi = float(p.get('szi', 0))
        if szi != 0:
            print(f"{p.get('coin')}: size={p.get('szi')}, entry={p.get('entryPx')}, pnl={p.get('unrealizedPnl')}")
    
    if not any(float(pos.get('position', {}).get('szi', 0)) != 0 for pos in positions):
        print('No open positions')
    
    # Method 4: Query perp dex orders directly via POST
    print()
    print('=== Checking km:US500 Orders via API ===')
    import requests
    
    # Try the perp_dex open orders query
    try:
        payload = {
            "type": "frontendOpenOrders",
            "user": wallet
        }
        resp = requests.post(constants.MAINNET_API_URL + '/info', json=payload)
        data = resp.json()
        
        us500_orders = [o for o in data if 'US500' in str(o.get('coin', ''))]
        print(f'US500 orders from API: {len(us500_orders)}')
        for o in us500_orders[:10]:
            print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
        if len(us500_orders) > 10:
            print(f"  ... and {len(us500_orders) - 10} more")
            
    except Exception as e:
        print(f'API query error: {e}')

if __name__ == '__main__':
    main()
