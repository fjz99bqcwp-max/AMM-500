#!/usr/bin/env python3
"""Quick search for recent open orders."""

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

# Search multiple OID ranges - recent activity
# From earlier session: OIDs around 293845852848 were placed
# And earlier: 293827435961

oid_ranges = [
    (293845852840, 293845852900),   # Most recent batch from this session
    (293827435950, 293827436000),   # Earlier batch
    (293850000000, 293860000000),   # Extended recent range (sample every 1000)
]

all_open = []

for start, end in oid_ranges[:2]:  # Dense search first two ranges
    print(f"Dense search OID range {start} to {end}...")
    for oid in range(start, end):
        data = query({"type": "orderStatus", "user": wallet, "oid": oid})
        if data and data.get('status') == 'order':
            order_data = data.get('order', {})
            if order_data.get('status') == 'open':
                order = order_data.get('order', {})
                all_open.append({
                    'oid': oid,
                    'coin': order.get('coin'),
                    'side': order.get('side'),
                    'sz': order.get('sz'),
                    'px': order.get('limitPx')
                })

# Sparse search for the extended range
start, end = oid_ranges[2]
print(f"Sparse search OID range {start} to {end} (every 1000)...")
for oid in range(start, end, 1000):
    data = query({"type": "orderStatus", "user": wallet, "oid": oid})
    if data and data.get('status') == 'order':
        order_data = data.get('order', {})
        if order_data.get('status') == 'open':
            order = order_data.get('order', {})
            all_open.append({
                'oid': oid,
                'coin': order.get('coin'),
                'side': order.get('side'),
                'sz': order.get('sz'),
                'px': order.get('limitPx')
            })
            # Found one - do dense search around it
            print(f"  Found order at {oid}, searching ±50...")
            for oid2 in range(oid-50, oid+50):
                if oid2 == oid:
                    continue
                data = query({"type": "orderStatus", "user": wallet, "oid": oid2})
                if data and data.get('status') == 'order':
                    order_data = data.get('order', {})
                    if order_data.get('status') == 'open':
                        order = order_data.get('order', {})
                        all_open.append({
                            'oid': oid2,
                            'coin': order.get('coin'),
                            'side': order.get('side'),
                            'sz': order.get('sz'),
                            'px': order.get('limitPx')
                        })

print()
print(f"Total open orders found: {len(all_open)}")

if all_open:
    # Group by coin
    by_coin = {}
    for o in all_open:
        coin = o['coin']
        if coin not in by_coin:
            by_coin[coin] = {'B': [], 'A': []}
        by_coin[coin][o['side']].append(o)
    
    for coin, sides in by_coin.items():
        print(f"\n{coin}:")
        for side, orders in sides.items():
            if orders:
                print(f"  {side} ({len(orders)} orders):")
                for o in orders[:10]:
                    print(f"    OID {o['oid']}: {o['sz']} @ {o['px']}")
                if len(orders) > 10:
                    print(f"    ... and {len(orders) - 10} more")
