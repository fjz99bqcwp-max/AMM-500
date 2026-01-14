#!/usr/bin/env python3
"""Quick check of specific OIDs."""

import requests
import os
from dotenv import load_dotenv
load_dotenv('config/.env')

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = os.getenv('WALLET_ADDRESS', '').strip()

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    return None

print(f"Wallet: {wallet}")
print()

# Check specific OIDs from the log
oids_to_check = [293845852848, 293845852849, 293845852850, 293845852851]

print("=== Checking new order OIDs ===")
for oid in oids_to_check:
    data = query({"type": "orderStatus", "user": wallet, "oid": oid})
    if data and data.get('status') == 'order':
        order_data = data.get('order', {})
        status = order_data.get('status')
        order = order_data.get('order', {})
        print(f"OID {oid}: {status} - {order.get('coin')} {order.get('side')} {order.get('sz')} @ {order.get('limitPx')}")
    elif data and data.get('status') == 'unknownOid':
        print(f"OID {oid}: unknown")
    else:
        print(f"OID {oid}: {data}")
