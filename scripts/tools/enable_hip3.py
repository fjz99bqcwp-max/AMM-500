#!/usr/bin/env python3
"""Enable HIP-3 DEX abstraction for the wallet."""

from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
import os
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('/Users/nheosdisplay/VSC/AMM/AMM-500/config/.env'))

private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

print(f"Wallet: {wallet_address}")

# Create exchange
account = Account.from_key(private_key)
print(f"API wallet: {account.address}")

exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    account_address=wallet_address
)

# Enable HIP-3 DEX abstraction using SDK method
print("\n=== Enabling HIP-3 DEX Abstraction ===")

# Try agent_enable_dex_abstraction (for API wallets - takes no args)
try:
    print("Trying agent_enable_dex_abstraction()...")
    result = exchange.agent_enable_dex_abstraction()
    print(f'Result: {result}')
except Exception as e:
    print(f'agent_enable_dex_abstraction error: {e}')

# Try user_dex_abstraction(user, enabled)
try:
    print(f"\nTrying user_dex_abstraction('{wallet_address}', True)...")
    result = exchange.user_dex_abstraction(wallet_address, True)
    print(f'Result: {result}')
except Exception as e:
    print(f'user_dex_abstraction error: {e}')

# Check what methods the exchange has
print("\n=== Available Exchange Methods ===")
methods = [m for m in dir(exchange) if not m.startswith('_') and callable(getattr(exchange, m))]
hip3_methods = [m for m in methods if 'dex' in m.lower() or 'hip' in m.lower() or 'abstraction' in m.lower()]
print(f"HIP-3 related: {hip3_methods}")
print(f"All methods: {methods[:20]}...")
