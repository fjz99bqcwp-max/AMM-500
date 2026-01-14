#!/usr/bin/env python3
"""Cancel all km:US500 orders using the Exchange SDK."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('config/.env')

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from eth_account import Account
import requests
import time

MAINNET_API = "https://api.hyperliquid.xyz/info"

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    return None

private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS', '').strip()

wallet = Account.from_key(private_key)
api_wallet = wallet.address

print(f"API Wallet: {api_wallet}")
print(f"Account Wallet: {wallet_address}")
print()

# Create exchange client with perp_dexs
if wallet_address and wallet_address.lower() != api_wallet.lower():
    exchange = Exchange(wallet, constants.MAINNET_API_URL, account_address=wallet_address, perp_dexs=['km'])
else:
    exchange = Exchange(wallet, constants.MAINNET_API_URL, perp_dexs=['km'])
    wallet_address = api_wallet

# Find all open orders by checking known OID range
print("=== Finding all open orders ===")
known_oids = [293827435961, 293827435962, 293827435963, 293827435964]
open_orders = []

for oid in known_oids:
    data = query({"type": "orderStatus", "user": wallet_address, "oid": oid})
    if data and data.get('status') == 'order':
        order_data = data.get('order', {})
        if order_data.get('status') == 'open':
            order = order_data.get('order', {})
            open_orders.append({
                'oid': oid,
                'coin': order.get('coin'),
                'side': order.get('side'),
                'sz': order.get('sz'),
                'px': order.get('limitPx')
            })
            print(f"  Found: {order.get('coin')} {order.get('side')} {order.get('sz')} @ {order.get('limitPx')} (oid: {oid})")

print(f"\nTotal open orders: {len(open_orders)}")

if len(open_orders) == 0:
    print("No orders to cancel!")
    sys.exit(0)

# Cancel each order
print()
print("=== Cancelling orders ===")
for order in open_orders:
    oid = order['oid']
    coin = order['coin']
    try:
        # The coin should be "km:US500" for HIP-3 perps
        result = exchange.cancel(coin, oid)
        print(f"  Cancelled OID {oid}: {result}")
    except Exception as e:
        print(f"  Error cancelling OID {oid}: {e}")

# Wait and verify
print()
print("Waiting 3 seconds...")
time.sleep(3)

print()
print("=== Verifying cancellation ===")
for oid in [o['oid'] for o in open_orders]:
    data = query({"type": "orderStatus", "user": wallet_address, "oid": oid})
    if data and data.get('status') == 'order':
        status = data.get('order', {}).get('status')
        print(f"  OID {oid}: {status}")
    else:
        print(f"  OID {oid}: not found or cancelled")

print()
print("Done!")
