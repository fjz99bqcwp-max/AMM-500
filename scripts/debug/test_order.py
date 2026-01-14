#!/usr/bin/env python3
"""Test order placement with km:US500."""

from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('/Users/nheosdisplay/VSC/AMM/AMM-500/config/.env'))

private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

print(f"Private key (first 10): {private_key[:10] if private_key else 'None'}...")
print(f"Wallet: {wallet_address}")

# Create exchange
account = Account.from_key(private_key)
print(f"API wallet: {account.address}")

exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    account_address=wallet_address
)

# Try to set leverage
print("\n=== Testing Set Leverage ===")
try:
    result = exchange.update_leverage(leverage=5, name='km:US500', is_cross=True)
    print(f'Set leverage result: {result}')
except Exception as e:
    print(f'Set leverage error: {e}')

# Try to place a small test order
print("\n=== Testing Order Placement ===")
try:
    result = exchange.order(
        name='km:US500',  # SDK uses 'name' not 'coin'
        is_buy=True,
        sz=0.01,
        limit_px=694.0,  # Below market
        order_type={'limit': {'tif': 'Gtc'}},
        reduce_only=False
    )
    print(f'Order result: {result}')
except Exception as e:
    print(f'Order error: {e}')
