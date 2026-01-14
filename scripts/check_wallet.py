#!/usr/bin/env python3
"""Check which wallet address corresponds to the private key."""

from eth_account import Account
import os
from dotenv import load_dotenv

# Load config
load_dotenv('config/.env')
private_key = os.getenv('PRIVATE_KEY')
configured_wallet = os.getenv('WALLET_ADDRESS')
api_wallet = os.getenv('API_WALLET_ADDRESS')

# Derive address from private key
wallet = Account.from_key(private_key)
derived_address = wallet.address

print("=" * 60)
print("WALLET ADDRESS CHECK")
print("=" * 60)
print(f"Private key derives to:  {derived_address}")
print(f"Configured WALLET_ADDRESS:  {configured_wallet}")
print(f"Configured API_WALLET_ADDRESS: {api_wallet}")
print()

if derived_address.lower() == configured_wallet.lower():
    print("✓ Private key matches WALLET_ADDRESS (main wallet)")
    print("Orders should be queried from:", configured_wallet)
elif derived_address.lower() == api_wallet.lower():
    print("✓ Private key matches API_WALLET_ADDRESS")
    print("PROBLEM: Orders are placed from API wallet but queried from main wallet!")
    print("Solution: Query openOrders for API wallet address:", api_wallet)
else:
    print("⚠ Private key doesn't match either address!")
    print("This is the actual signing address:", derived_address)
    print("Orders are placed from this address, not from WALLET_ADDRESS")
