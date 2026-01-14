#!/usr/bin/env python3
"""Check recent trades/fills on all wallets."""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from eth_account import Account
from datetime import datetime

# Load environment
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / "config" / ".env")

wallet_address = os.getenv("WALLET_ADDRESS")
private_key = os.getenv("PRIVATE_KEY")

# Derive signing wallet
signing_wallet = Account.from_key(private_key)
signing_address = signing_wallet.address

print("=" * 60)
print("RECENT TRADES CHECK")
print("=" * 60)

# Query each address
for name, addr in [
    ("Main Wallet", wallet_address),
    ("Signing Wallet", signing_address)
]:
    print(f"{name} ({addr}):")
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "userFills", "user": addr},
            timeout=10
        )
        fills = resp.json()
        
        if not fills:
            print(f"  No fills")
        else:
            print(f"  Total: {len(fills)}")
            us500 = [f for f in fills if 'US500' in str(f.get('coin', '')).upper()]
            if us500:
                print(f"  US500 fills (last 5):")
                for f in us500[:5]:
                    ts = f.get('time', 0) / 1000
                    dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    print(f"    {dt}: {f.get('side')} {f.get('sz')}@${f.get('px')}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
