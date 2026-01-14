#!/usr/bin/env python3
"""Quick status check for AMM-500 bot"""
import requests
from datetime import datetime

MAINNET = 'https://api.hyperliquid.xyz/info'
wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

print('=' * 60)
print(f'AMM-500 Bot Status Check - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 60)

# Check balance
r = requests.post(MAINNET, json={'type': 'spotClearinghouseState', 'user': wallet})
spot = r.json()
for bal in spot.get('balances', []):
    if float(bal.get('total', 0)) > 0.001:
        print(f'Spot {bal["coin"]}: {float(bal.get("total", 0)):.4f}')

# Check km perps
r = requests.post(MAINNET, json={'type': 'clearinghouseState', 'user': wallet, 'perpDex': 'km'})
km = r.json()
margin = km.get('marginSummary', {})
print(f'km Perps Account Value: ${float(margin.get("accountValue", 0)):.4f}')

# Check known orders from most recent bot log
oids = [293866058501, 293866058502, 293866058503, 293866058504]
print(f'\nChecking {len(oids)} known orders...')
live_count = 0
for oid in oids:
    r = requests.post(MAINNET, json={'type': 'orderStatus', 'user': wallet, 'oid': oid})
    data = r.json()
    if data.get('status') == 'order':
        order = data.get('order', {}).get('order', {})
        if order:
            live_count += 1
            side = 'BUY' if order.get('side') == 'B' else 'SELL'
            print(f'  {side} {order.get("sz")} @ ${order.get("limitPx")}')

print(f'\nLive orders: {live_count}')
print('=' * 60)
