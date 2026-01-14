#!/usr/bin/env python3
"""Check perp dex orders and state using the correct API."""

from hyperliquid.info import Info
from hyperliquid.utils import constants
import requests
import json

def main():
    wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
    api_url = constants.MAINNET_API_URL
    
    print(f'Wallet: {wallet}')
    print(f'API URL: {api_url}')
    print()
    
    # Query perp dex state directly
    print('=== Perp Dex User State ===')
    try:
        payload = {
            "type": "perpClearinghouseState",
            "user": wallet,
            "perpDex": "km"
        }
        resp = requests.post(api_url + '/info', json=payload)
        data = resp.json()
        print(f'Response type: {type(data)}')
        
        if isinstance(data, dict):
            margin = data.get('marginSummary', {})
            print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
            print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
            print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
            
            # Check positions
            positions = data.get('assetPositions', [])
            print(f'\nPositions ({len(positions)}):')
            for pos in positions:
                p = pos.get('position', {})
                szi = float(p.get('szi', 0))
                if szi != 0:
                    print(f"  {p.get('coin')}: size={p.get('szi')}, entry={p.get('entryPx')}")
        else:
            print(f'Unexpected response: {data}')
    except Exception as e:
        print(f'Error: {e}')
    
    # Query perp dex open orders
    print()
    print('=== Perp Dex Open Orders ===')
    try:
        payload = {
            "type": "openOrders",
            "user": wallet,
            "perpDex": "km"
        }
        resp = requests.post(api_url + '/info', json=payload)
        data = resp.json()
        
        print(f'Open orders: {len(data)} orders')
        if data:
            for o in data[:20]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(data) > 20:
                print(f"  ... and {len(data) - 20} more")
    except Exception as e:
        print(f'Error: {e}')
    
    # Also try frontend open orders with perpDex
    print()
    print('=== Frontend Open Orders (perpDex) ===')
    try:
        payload = {
            "type": "frontendOpenOrders", 
            "user": wallet,
            "perpDex": "km"
        }
        resp = requests.post(api_url + '/info', json=payload)
        data = resp.json()
        
        print(f'Frontend orders: {len(data)} orders')
        if data:
            # Group by side
            bids = [o for o in data if o.get('side') == 'B']
            asks = [o for o in data if o.get('side') == 'A']
            print(f'Bids: {len(bids)}, Asks: {len(asks)}')
            
            for o in data[:20]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(data) > 20:
                print(f"  ... and {len(data) - 20} more")
    except Exception as e:
        print(f'Error: {e}')

    # Also check meta for US500 specs
    print()
    print('=== US500 Market Info ===')
    try:
        payload = {"type": "perpMeta", "perpDex": "km"}
        resp = requests.post(api_url + '/info', json=payload)
        meta = resp.json()
        
        for asset in meta.get('universe', []):
            if 'US500' in asset.get('name', ''):
                print(f"Name: {asset.get('name')}")
                print(f"szDecimals: {asset.get('szDecimals')}")
                print(f"maxLeverage: {asset.get('maxLeverage')}")
                break
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
