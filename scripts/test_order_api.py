#!/usr/bin/env python3
"""Test all order visibility APIs for HIP-3"""
from hyperliquid.info import Info
from hyperliquid.utils import constants
import requests

wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

print('=== Testing Order Visibility APIs ===')
print()

# 1. SDK with perp_dexs
print('1. SDK info.open_orders() with perp_dexs=["km"]:')
info = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
orders = info.open_orders(wallet)
print(f'   Result: {len(orders)} orders')

# 2. SDK frontend_open_orders
print('2. SDK info.frontend_open_orders():')
frontend = info.frontend_open_orders(wallet)
print(f'   Result: {len(frontend)} orders')

# 3. SDK user_state openOrders field
print('3. SDK user_state().openOrders:')
state = info.user_state(wallet)
open_orders = state.get('openOrders', [])
print(f'   Result: {len(open_orders)} orders')

# 4. Direct API with perpDex parameter
print('4. Direct API openOrders with perpDex=km:')
r = requests.post('https://api.hyperliquid.xyz/info', json={
    'type': 'openOrders', 'user': wallet, 'perpDex': 'km'
})
print(f'   Result: {len(r.json()) if isinstance(r.json(), list) else r.json()}')

# 5. clearinghouseState openOrders with perpDex
print('5. clearinghouseState.openOrders with perpDex=km:')
r = requests.post('https://api.hyperliquid.xyz/info', json={
    'type': 'clearinghouseState', 'user': wallet, 'perpDex': 'km'
})
state = r.json()
open_orders = state.get('openOrders', [])
print(f'   Result: {len(open_orders)} orders')

print()
print('CONCLUSION: HIP-3 orders are NOT returned by standard APIs.')
print('The only way to track them is via orderStatus by OID.')
