#!/usr/bin/env python3
"""
Cancel all open orders
"""
import os
import sys
from dotenv import load_dotenv
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Load environment
load_dotenv('config/.env')
private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

if not private_key or not wallet_address:
    print("ERROR: Missing PRIVATE_KEY or WALLET_ADDRESS in config/.env")
    sys.exit(1)

# Initialize exchange
account = eth_account.Account.from_key(private_key)
exchange = Exchange(account, constants.MAINNET_API_URL, account_address=wallet_address, perp_dexs=['km'])

try:
    # Get all open orders using info
    print("Checking open orders...")
    info = exchange.info
    orders = info.open_orders(wallet_address)
    print(f"Found {len(orders)} open orders")
    
    if orders:
        # Show some details
        us500_orders = [o for o in orders if 'US500' in o.get('coin', '')]
        km_us500_orders = [o for o in orders if o.get('coin') == 'km:US500']
        
        print(f"US500 orders: {len(us500_orders)}")
        print(f"km:US500 orders: {len(km_us500_orders)}")
        
        # Cancel all orders
        cancel_requests = []
        for order in orders:
            cancel_requests.append({
                'coin': order['coin'],
                'oid': order['oid']
            })
        
        print(f"Cancelling {len(cancel_requests)} orders...")
        result = exchange.bulk_cancel(cancel_requests)
        print(f"Cancel result: {result}")
        
        # Verify cancellation
        orders_after = exchange.open_orders(wallet_address)
        print(f"Orders remaining: {len(orders_after)}")
        
        if orders_after:
            print("Remaining orders:")
            for order in orders_after:
                print(f"  - {order['coin']}: {order.get('sz', 'N/A')} @ {order.get('limitPx', 'N/A')}")
    else:
        print("No orders to cancel")
        
    # Check account balances
    print("\n=== BALANCE CHECK ===")
    info = exchange.info
    
    # Get clearinghouse state
    state = info.user_state(wallet_address)
    margin_summary = state.get("marginSummary", {})
    
    print(f"Account Value: ${float(margin_summary.get('accountValue', 0)):.2f}")
    print(f"Total Margin Used: ${float(margin_summary.get('totalMarginUsed', 0)):.2f}")
    print(f"Total Notional Pos: ${float(margin_summary.get('totalNtlPos', 0)):.2f}")
    
    # Check Spot USDH (for HIP-3)
    try:
        spot_state = info.spot_user_state(wallet_address)
        balances = spot_state.get("balances", [])
        
        total_usdh = 0
        for balance in balances:
            if balance.get("coin") == "USDH":
                total = float(balance.get("total", 0))
                hold = float(balance.get("hold", 0))
                entryNotional = float(balance.get("entryNotional", 0))
                
                print(f"\nSpot USDH Balance:")
                print(f"  Total: ${total:.2f}")
                print(f"  Hold: ${hold:.2f}")
                print(f"  Available: ${total - hold:.2f}")
                print(f"  Entry Notional: ${entryNotional:.2f}")
                
                total_usdh = total
                break
        
        if total_usdh == 0:
            print("No USDH found in Spot account")
            
    except Exception as e:
        print(f"Could not check Spot balances: {e}")
    
    # Check positions
    asset_positions = state.get("assetPositions", [])
    print(f"\nPositions: {len(asset_positions)}")
    
    for pos in asset_positions:
        pos_info = pos.get("position", {})
        coin = pos_info.get("coin", "")
        size = float(pos_info.get("szi", 0))
        unrealized = float(pos_info.get("unrealizedPnl", 0))
        
        if abs(size) > 0.001:  # Only show significant positions
            print(f"  {coin}: {size:.4f} (PnL: ${unrealized:.2f})")
    
    print("\n=== CLEANUP COMPLETE ===")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)