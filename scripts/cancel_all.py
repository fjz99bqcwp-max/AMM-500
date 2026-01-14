#!/usr/bin/env python3
"""Cancel all orders and check balance for km:US500."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('config/.env')

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
import time

def main():
    private_key = os.getenv('PRIVATE_KEY')
    wallet_address = os.getenv('WALLET_ADDRESS', '').strip()
    
    if not private_key:
        print("ERROR: PRIVATE_KEY not found in config/.env")
        return
        
    wallet = Account.from_key(private_key)
    api_wallet = wallet.address
    
    print(f"API Wallet: {api_wallet}")
    print(f"Account Wallet: {wallet_address}")
    print()
    
    # Create clients with perp_dexs for HIP-3
    info = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
    
    if wallet_address and wallet_address.lower() != api_wallet.lower():
        exchange = Exchange(wallet, constants.MAINNET_API_URL, account_address=wallet_address, perp_dexs=['km'])
        target_wallet = wallet_address
    else:
        exchange = Exchange(wallet, constants.MAINNET_API_URL, perp_dexs=['km'])
        target_wallet = api_wallet
    
    print(f"Target wallet for queries: {target_wallet}")
    print()
    
    # Get user state
    print("=== Account State ===")
    try:
        state = info.user_state(target_wallet)
        margin = state.get('marginSummary', {})
        print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
        print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
        print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
    except Exception as e:
        print(f"State error: {e}")
    
    # Get open orders
    print()
    print("=== Open Orders ===")
    try:
        orders = info.open_orders(target_wallet)
        print(f"Total orders: {len(orders)}")
        
        if orders:
            # Group by side
            bids = [o for o in orders if o.get('side') == 'B']
            asks = [o for o in orders if o.get('side') == 'A']
            print(f"Bids: {len(bids)}, Asks: {len(asks)}")
            
            # Show all orders
            for o in orders[:50]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(orders) > 50:
                print(f"  ... and {len(orders) - 50} more")
    except Exception as e:
        print(f"Orders error: {e}")
    
    # Get frontend open orders
    print()
    print("=== Frontend Open Orders ===")
    try:
        frontend = info.frontend_open_orders(target_wallet)
        print(f"Frontend orders: {len(frontend)}")
        
        if frontend:
            bids = [o for o in frontend if o.get('side') == 'B']
            asks = [o for o in frontend if o.get('side') == 'A']
            print(f"Bids: {len(bids)}, Asks: {len(asks)}")
            
            for o in frontend[:50]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')} (oid: {o.get('oid')})")
            if len(frontend) > 50:
                print(f"  ... and {len(frontend) - 50} more")
    except Exception as e:
        print(f"Frontend error: {e}")
    
    # Get positions
    print()
    print("=== Positions ===")
    try:
        state = info.user_state(target_wallet)
        positions = state.get('assetPositions', [])
        has_position = False
        for pos in positions:
            p = pos.get('position', {})
            szi = float(p.get('szi', 0))
            if szi != 0:
                has_position = True
                print(f"  {p.get('coin')}: size={p.get('szi')}, entry={p.get('entryPx')}, pnl={p.get('unrealizedPnl')}")
        if not has_position:
            print("  No open positions")
    except Exception as e:
        print(f"Positions error: {e}")
    
    # Cancel all orders
    print()
    print("=== Cancelling All Orders ===")
    try:
        # Cancel all orders on US500
        result = exchange.cancel('km:US500', None)  # oid=None cancels all
        print(f"Cancel result: {result}")
    except Exception as e:
        print(f"Cancel error: {e}")
    
    # Wait and check again
    print()
    print("Waiting 3 seconds...")
    time.sleep(3)
    
    print()
    print("=== Orders After Cancel ===")
    try:
        orders = info.open_orders(target_wallet)
        print(f"Remaining orders: {len(orders)}")
        if orders:
            for o in orders[:20]:
                print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')}")
    except Exception as e:
        print(f"Check error: {e}")

if __name__ == '__main__':
    main()
