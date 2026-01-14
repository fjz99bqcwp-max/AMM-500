#!/usr/bin/env python3
"""Check perp dex orders using the SDK properly."""

from hyperliquid.info import Info
from hyperliquid.utils import constants
import requests
import json

def main():
    wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
    api_url = constants.MAINNET_API_URL
    
    print(f'Wallet: {wallet}')
    print()
    
    # Create Info with perp_dexs parameter
    print('=== Using SDK with perp_dexs ===')
    try:
        info = Info(api_url, skip_ws=True, perp_dexs=['km'])
        
        # Get user state
        state = info.user_state(wallet)
        print(f'State keys: {list(state.keys())}')
        
        margin = state.get('marginSummary', {})
        print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
        print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
        
        # Open orders
        orders = info.open_orders(wallet)
        print(f'\nOpen orders: {len(orders)}')
        if orders:
            for o in orders[:30]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(orders) > 30:
                print(f"  ... and {len(orders) - 30} more")
                
        # Frontend open orders
        frontend = info.frontend_open_orders(wallet)
        print(f'\nFrontend orders: {len(frontend)}')
        if frontend:
            bids = [o for o in frontend if o.get('side') == 'B']
            asks = [o for o in frontend if o.get('side') == 'A']
            print(f'Bids: {len(bids)}, Asks: {len(asks)}')
            
            for o in frontend[:30]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(frontend) > 30:
                print(f"  ... and {len(frontend) - 30} more")
        
        # Positions
        print(f'\nPositions:')
        for pos in state.get('assetPositions', []):
            p = pos.get('position', {})
            szi = float(p.get('szi', 0))
            if szi != 0:
                print(f"  {p.get('coin')}: size={p.get('szi')}, entry={p.get('entryPx')}, pnl={p.get('unrealizedPnl')}")
                
    except Exception as e:
        import traceback
        print(f'Error: {e}')
        traceback.print_exc()
    
    # Also get meta
    print()
    print('=== km Market Meta ===')
    try:
        info = Info(api_url, skip_ws=True, perp_dexs=['km'])
        meta = info.meta()
        print(f'Universe has {len(meta.get("universe", []))} assets')
        for asset in meta.get('universe', []):
            if 'US500' in asset.get('name', ''):
                print(f"  US500: szDecimals={asset.get('szDecimals')}, maxLeverage={asset.get('maxLeverage')}")
    except Exception as e:
        print(f'Meta error: {e}')

if __name__ == '__main__':
    main()
