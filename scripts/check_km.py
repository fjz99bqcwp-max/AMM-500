#!/usr/bin/env python3
"""Check the HIP-3 perp dex margin via the correct API endpoint."""

import requests
import json

MAINNET_API = "https://api.hyperliquid.xyz"
wallet = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

# First, get all perp metas to understand the structure
print("=== Getting perpMetas for km ===")
try:
    # The SDK source shows perpMetas requires perpDex parameter in the exchange init
    # but the info API uses a different pattern
    resp = requests.post(f"{MAINNET_API}/info", json={"type": "meta"})
    if resp.status_code == 200:
        meta = resp.json()
        print(f"Main perp universe size: {len(meta.get('universe', []))}")
except Exception as e:
    print(f"Error: {e}")

# Check the perp dex meta endpoint
print()
print("=== Getting km perp dex meta ===")
try:
    # Try different API endpoints that might work
    endpoints = [
        {"type": "meta", "perpDex": "km"},
        {"type": "perpDexMeta", "perpDex": "km"},
    ]
    for ep in endpoints:
        resp = requests.post(f"{MAINNET_API}/info", json=ep)
        print(f"  {ep}: HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and 'universe' in data:
                for a in data['universe']:
                    if 'US500' in str(a.get('name', '')):
                        print(f"    US500: {a}")
except Exception as e:
    print(f"Error: {e}")

# Now check the user's km perp state
print()
print("=== Checking km perp dex user state ===")

# Try to find the right API call format from SDK source
# Looking at hyperliquid-python-sdk, it uses:
# "clearinghouseState" for main perps
# The perp_dexs parameter changes the meta query

# Let's check what the SDK actually sends
print()
print("=== SDK-style query with perp_dexs ===")
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Create Info with perp_dexs
info_km = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
info_main = Info(constants.MAINNET_API_URL, skip_ws=True)

print("Main perp state:")
try:
    state = info_main.user_state(wallet)
    margin = state.get('marginSummary', {})
    print(f"  Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("km perp dex state:")
try:
    state = info_km.user_state(wallet)
    margin = state.get('marginSummary', {})
    print(f"  Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
    print(f"  Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
    print(f"  Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
except Exception as e:
    print(f"  Error: {e}")

# Check open orders with km
print()
print("=== km perp dex open orders ===")
try:
    orders = info_km.open_orders(wallet)
    print(f"Open orders: {len(orders)}")
    for o in orders[:10]:
        print(f"  {o}")
except Exception as e:
    print(f"  Error: {e}")

# Check frontend open orders
print()
print("=== km frontend open orders ===")
try:
    orders = info_km.frontend_open_orders(wallet)
    print(f"Frontend orders: {len(orders)}")
    for o in orders[:10]:
        print(f"  {o}")
except Exception as e:
    print(f"  Error: {e}")

# Get the raw user state to see all fields
print()
print("=== Raw km user state ===")
try:
    state = info_km.user_state(wallet)
    print(json.dumps(state, indent=2)[:2000])
except Exception as e:
    print(f"  Error: {e}")
