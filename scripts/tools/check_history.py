#!/usr/bin/env python3
"""Check wallet history for HIP-3 perps (like US500)."""

import requests

wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
url = 'https://api.hyperliquid.xyz/info'

# Check perp user state for HIP-3 (US500)
print('=== PERPS STATE ===')
resp = requests.post(url, json={"type": "clearinghouseState", "user": wallet}, timeout=10)
state = resp.json()
print(f'Account Value: ${float(state.get("marginSummary", {}).get("accountValue", 0)):,.2f}')
print(f'Withdrawable: ${float(state.get("withdrawable", 0)):,.2f}')

positions = state.get('assetPositions', [])
print(f'\nPositions: {len(positions)}')
for p in positions:
    pos = p.get('position', {})
    coin = pos.get('coin', '?')
    size = float(pos.get('szi', 0))
    if size != 0:
        entry = float(pos.get('entryPx', 0))
        pnl = float(pos.get('unrealizedPnl', 0))
        print(f'  {coin}: {size} @ ${entry:,.2f} (PnL: ${pnl:,.2f})')

# Check spot balances
print('\n=== SPOT STATE ===')
resp = requests.post(url, json={"type": "spotClearinghouseState", "user": wallet}, timeout=10)
spot = resp.json()
balances = spot.get('balances', [])
for b in balances:
    token = b.get('coin', '?')
    total = float(b.get('total', 0))
    if total > 0:
        print(f'  {token}: {total:,.2f}')

# Check user fills (recent trades) - with timeout
print('\n=== CHECKING FILLS (may be slow) ===')
try:
    resp = requests.post(url, json={"type": "userFills", "user": wallet}, timeout=30)
    fills = resp.json()
    if fills:
        print(f'Total fills: {len(fills)}')
        for f in fills[:10]:
            coin = f.get('coin', '?')
            side = f.get('side', '?')
            sz = f.get('sz', '?')
            px = f.get('px', '?')
            print(f'  {coin}: {side} {sz} @ ${px}')
    else:
        print('No fills found')
except Exception as e:
    print(f'Error getting fills: {e}')

# Check order history
print('\n=== ORDER HISTORY ===')
try:
    resp = requests.post(url, json={"type": "historicalOrders", "user": wallet}, timeout=30)
    orders = resp.json()
    if orders:
        print(f'Total orders: {len(orders)}')
        for o in orders[:10]:
            coin = o.get('coin', '?')
            side = o.get('side', '?')
            sz = o.get('sz', '?')
            px = o.get('limitPx', '?')
            status = o.get('status', '?')
            print(f'  {coin}: {side} {sz} @ ${px} - {status}')
    else:
        print('No order history')
except Exception as e:
    print(f'Error: {e}')
