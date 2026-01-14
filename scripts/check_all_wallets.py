#!/usr/bin/env python3
"""Check open orders for all wallet addresses."""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from eth_account import Account

# Load environment
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / "config" / ".env")

wallet_address = os.getenv("WALLET_ADDRESS")
api_wallet_address = os.getenv("API_WALLET_ADDRESS")
private_key = os.getenv("PRIVATE_KEY")

# Derive signing wallet
signing_wallet = Account.from_key(private_key)
signing_address = signing_wallet.address

print("=" * 60)
print("OPEN ORDERS CHECK - ALL WALLETS")
print("=" * 60)
print(f"Main wallet: {wallet_address}")
print(f"API wallet: {api_wallet_address}")
print(f"Signing wallet: {signing_address}")
print()

# Query each address
for name, addr in [
    ("Main Wallet", wallet_address),
    ("API Wallet", api_wallet_address),
    ("Signing Wallet", signing_address)
]:
    print(f"{name} ({addr}):")
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "openOrders", "user": addr},
            timeout=10
        )
        orders = resp.json()
        print(f"  Total orders: {len(orders)}")
        
        # Filter US500
        us500 = [o for o in orders if 'US500' in str(o.get('coin', '')).upper()]
        if us500:
            print(f"  US500 orders: {len(us500)}")
            for o in us500[:5]:
                print(f"    OID {o.get('oid')}: {o.get('side')} {o.get('sz')}@${o.get('limitPx')} [{o.get('coin')}]")
        else:
            print(f"  No US500 orders")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
