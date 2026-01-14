#!/usr/bin/env python3
"""Test order placement with perp_dexs parameter."""

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from eth_account import Account
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('/Users/nheosdisplay/VSC/AMM/AMM-500/config/.env'))

private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

print(f"Wallet: {wallet_address}")

# Create exchange with perp_dexs for HIP-3
account = Account.from_key(private_key)
print(f"API wallet: {account.address}")

perp_dexs = ['km']  # For km:US500

exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    account_address=wallet_address,
    perp_dexs=perp_dexs
)

# Check if km:US500 is in the mapping now
print(f"\nkm:US500 in name_to_coin: {'km:US500' in exchange.info.name_to_coin}")

# Try to set leverage with km:US500
print("\n=== Testing Set Leverage ===")
try:
    result = exchange.update_leverage(leverage=5, name='km:US500', is_cross=False)  # HIP-3 uses isolated
    print(f'Set leverage result: {result}')
except Exception as e:
    print(f'Set leverage error: {e}')

# Try to place a small test order - $10 minimum
print("\n=== Testing Order Placement ===")
try:
    # US500 is ~$695, so 0.02 contracts = $13.90 > $10 minimum
    result = exchange.order(
        name='km:US500',  # Use km: prefix
        is_buy=True,
        sz=0.02,  # 0.02 * 695 = $13.90
        limit_px=690.0,  # Below market for limit order
        order_type={'limit': {'tif': 'Gtc'}},
        reduce_only=False
    )
    print(f'Order result: {result}')
    
    # If order placed, cancel it
    if result.get('status') == 'ok':
        statuses = result.get('response', {}).get('data', {}).get('statuses', [])
        if statuses and statuses[0].get('resting'):
            oid = statuses[0]['resting']['oid']
            print(f'Order placed with oid: {oid}, canceling...')
            cancel_result = exchange.cancel('km:US500', oid)
            print(f'Cancel result: {cancel_result}')
except Exception as e:
    print(f'Order error: {e}')
