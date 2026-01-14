#!/usr/bin/env python3
"""Check HIP-3 perp dex state properly."""

import requests
import json

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    return None

# Get ALL clearinghouse states (main + all perp dexs)
print("=== Querying all perp dex states ===")

# Try different API patterns for HIP-3 perp dex state
api_patterns = [
    {"type": "clearinghouseState", "user": wallet},
    {"type": "perpDexClearinghouseState", "user": wallet, "perpDex": "km"},
    {"type": "perpClearinghouseState", "user": wallet, "perpDex": "km"},
]

for p in api_patterns:
    try:
        resp = requests.post(MAINNET_API, json=p)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                margin = data.get('marginSummary', {})
                val = float(margin.get('accountValue', 0))
                print(f"{p['type']}: Account Value = ${val:,.2f}")
        else:
            print(f"{p['type']}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"{p['type']}: Error {e}")

print()
print("=== Checking individual order statuses to find all open orders ===")

# Query orderStatus for a range of recent OIDs to find all open orders
# Start from the known OIDs and search around them
base_oid = 293827435961
open_orders = []

# Search recent orders (last 1000)
for oid in range(base_oid - 500, base_oid + 500):
    try:
        data = query({"type": "orderStatus", "user": wallet, "oid": oid})
        if data and data.get('status') == 'order':
            order_data = data.get('order', {})
            status = order_data.get('status')
            if status == 'open':
                order = order_data.get('order', {})
                open_orders.append({
                    'oid': oid,
                    'coin': order.get('coin'),
                    'side': order.get('side'),
                    'sz': order.get('sz'),
                    'px': order.get('limitPx')
                })
    except:
        pass

print(f"Found {len(open_orders)} open orders in range")
for o in open_orders[:20]:
    print(f"  {o['coin']}: {o['side']} {o['sz']} @ {o['px']} (oid: {o['oid']})")
if len(open_orders) > 20:
    print(f"  ... and {len(open_orders) - 20} more")

print()
print("=== Spot Balance ===")
data = query({"type": "spotClearinghouseState", "user": wallet})
for bal in data.get('balances', []):
    coin = bal.get('coin')
    total = float(bal.get('total', 0))
    if total > 0:
        print(f"  {coin}: {total:,.4f}")

print()
print("=== Recent Fills (last 10) ===")
data = query({"type": "userFills", "user": wallet})
if data:
    print(f"Total fills: {len(data)}")
    for f in data[:10]:
        print(f"  {f.get('coin')}: {f.get('side')} {f.get('sz')} @ {f.get('px')} | time: {f.get('time')}")
