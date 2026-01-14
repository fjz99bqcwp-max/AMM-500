#!/usr/bin/env python3
"""Test order placement and persistence on Hyperliquid."""

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account
import os
from dotenv import load_dotenv
import time

load_dotenv('config/.env')
private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

# Use the main wallet private key
account = eth_account.Account.from_key(private_key)
print(f'Using wallet for signing: {account.address}')
print(f'Account address (trading account): {wallet_address}')

info = Info(constants.MAINNET_API_URL, skip_ws=True)

# Use vault address if set, otherwise main wallet
vault_address = wallet_address
exchange = Exchange(account, constants.MAINNET_API_URL, account_address=vault_address, perp_dexs=['km'])

# Check current orders on main wallet
orders = info.open_orders(wallet_address)
print(f'Current open orders on {wallet_address}: {len(orders)}')
for o in orders:
    print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')}")

# Place a test order - bid at 690 (below best bid)
print('\nPlacing test order: BID 0.1 @ 690 (GTC)')
result = exchange.order(
    name='km:US500',
    is_buy=True,
    sz=0.1,
    limit_px=690.0,
    order_type={'limit': {'tif': 'Gtc'}},
    reduce_only=False
)
print('Result:', result)

# Get the order ID
if result.get('status') == 'ok':
    oid = result['response']['data']['statuses'][0].get('resting', {}).get('oid')
    print(f'Order ID: {oid}')
    
    # Wait and check order status by OID
    import requests
    time.sleep(1)
    
    # Check order status
    url = 'https://api.hyperliquid.xyz/info'
    response = requests.post(url, json={
        'type': 'orderStatus',
        'user': wallet_address,
        'oid': oid
    })
    print(f'Order status by OID: {response.text}')

# Wait and check again
time.sleep(2)
api_wallet_address = account.address
print(f'\nChecking orders on API wallet ({api_wallet_address}):')
orders_api = info.open_orders(api_wallet_address)
print(f'  Open orders: {len(orders_api)}')
for o in orders_api:
    print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')}")

print(f'\nChecking orders on Main wallet ({wallet_address}):')
orders = info.open_orders(wallet_address)
print(f'  Open orders: {len(orders)}')
for o in orders:
    print(f"  {o.get('coin')}: {o.get('side')} {o.get('sz')} @ {o.get('limitPx')}")
