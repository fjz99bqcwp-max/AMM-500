#!/usr/bin/env python3
"""Search for open orders in the correct OID range."""

import requests
import os
from dotenv import load_dotenv
load_dotenv('config/.env')

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = os.getenv('WALLET_ADDRESS', '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C').strip()

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    return None

print(f"Wallet: {wallet}")
print()

# Known open OIDs from earlier
known_oids = [293827435961, 293827435962, 293827435963, 293827435964]

print("=== Checking known OIDs ===")
for oid in known_oids:
    data = query({"type": "orderStatus", "user": wallet, "oid": oid})
    if data and data.get('status') == 'order':
        order_data = data.get('order', {})
        status = order_data.get('status')
        order = order_data.get('order', {})
        print(f"OID {oid}: {status} - {order.get('coin')} {order.get('side')} {order.get('sz')} @ {order.get('limitPx')}")
    elif data and data.get('status') == 'unknownOid':
        print(f"OID {oid}: unknown (order may have been cancelled or filled)")
    else:
        print(f"OID {oid}: {data}")

# Search around this range for more orders
print()
print("=== Searching for all open orders (OID range 293827400000 to 293828000000) ===")
open_orders = []
search_start = 293827400000
search_end = 293828000000

for oid in range(search_start, search_end, 100):  # Sample every 100th OID
    data = query({"type": "orderStatus", "user": wallet, "oid": oid})
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

print(f"Found {len(open_orders)} open orders in sampled range")
for o in open_orders:
    print(f"  {o['coin']}: {o['side']} {o['sz']} @ {o['px']} (oid: {o['oid']})")

# Now do a dense search around the known OIDs
print()
print("=== Dense search around known OIDs ===")
open_orders = []
for oid in range(293827435950, 293827436000):
    data = query({"type": "orderStatus", "user": wallet, "oid": oid})
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

print(f"Found {len(open_orders)} open orders in dense range")
for o in open_orders:
    print(f"  {o['coin']}: {o['side']} {o['sz']} @ {o['px']} (oid: {o['oid']})")
