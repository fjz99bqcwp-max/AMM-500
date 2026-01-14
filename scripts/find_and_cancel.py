#!/usr/bin/env python3
"""Find and cancel all 140+ open orders on km:US500"""
import os
import sys
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv('config/.env')

from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

MAINNET = 'https://api.hyperliquid.xyz/info'
wallet_address = os.getenv('WALLET_ADDRESS')
private_key = os.getenv('PRIVATE_KEY')

print(f"Wallet: {wallet_address}")
print("=" * 60)

# Step 1: Get OID range from fills
print("\n1. Getting OID range from recent fills...")
resp = requests.post(MAINNET, json={'type': 'userFills', 'user': wallet_address})
fills = resp.json()
oids = [f.get('oid') for f in fills if f.get('oid')]
min_oid = min(oids) if oids else 0
max_oid = max(oids) if oids else 0
print(f"   OID range: {min_oid} to {max_oid}")

# Step 2: Search for open orders in expanded range
print("\n2. Searching for open orders (this may take a while)...")
search_start = max_oid
search_end = max_oid + 50000  # Search 50k OIDs forward

open_orders = []
checked = 0

for oid in range(search_start, search_end):
    resp = requests.post(MAINNET, json={
        'type': 'orderStatus', 'user': wallet_address, 'oid': oid
    })
    data = resp.json()
    checked += 1
    
    if data.get('status') == 'order':
        order = data.get('order', {}).get('order', {})
        if order and 'US500' in str(order.get('coin', '')):
            open_orders.append(oid)
            side = 'BUY' if order.get('side') == 'B' else 'SELL'
            print(f"   Found: OID {oid} - {side} {order.get('sz')} @ ${order.get('limitPx')}")
    
    if checked % 1000 == 0:
        print(f"   Checked {checked} OIDs, found {len(open_orders)} open orders...")
    
    # Stop if we found 150+ or checked 10k without finding new ones
    if len(open_orders) >= 150:
        break
    if checked > 5000 and len(open_orders) == 0:
        print("   No orders found in first 5000 OIDs, stopping search")
        break

print(f"\n   Total open orders found: {len(open_orders)}")

# Step 3: Cancel all found orders
if open_orders:
    print(f"\n3. Cancelling {len(open_orders)} orders...")
    
    account = Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL, 
                       account_address=wallet_address, perp_dexs=['km'])
    
    # Cancel in batches of 50
    for i in range(0, len(open_orders), 50):
        batch = open_orders[i:i+50]
        cancels = [{'coin': 'km:US500', 'oid': oid} for oid in batch]
        try:
            result = exchange.bulk_cancel(cancels)
            success = sum(1 for s in result.get('response', {}).get('data', {}).get('statuses', []) 
                         if s == 'success')
            print(f"   Batch {i//50 + 1}: cancelled {success}/{len(batch)}")
        except Exception as e:
            print(f"   Batch {i//50 + 1} error: {e}")
    
    print("\nDone! Check Hyperliquid UI to verify.")
else:
    print("\n3. No orders to cancel.")

print("=" * 60)
