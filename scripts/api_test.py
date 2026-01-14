#!/usr/bin/env python3
"""
Check orders using alternative methods - the user claims 135+ open orders.
Try different API endpoints and parameters.
"""

import requests
import json
import os
from dotenv import load_dotenv
load_dotenv('config/.env')

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    print(f"  Query: {payload.get('type')} -> HTTP {resp.status_code}")
    if resp.status_code == 200:
        return resp.json()
    print(f"    Error: {resp.text[:200]}")
    return None

print(f"Wallet: {wallet}")
print("=" * 70)

# Try ALL possible order query methods
print("\n=== Testing ALL order query methods ===\n")

# 1. openOrders (standard)
print("1. openOrders (standard):")
data = query({"type": "openOrders", "user": wallet})
if data:
    print(f"   Result: {len(data)} orders")
    if data:
        print(f"   Sample: {data[0] if data else 'none'}")

# 2. frontendOpenOrders (standard)  
print("\n2. frontendOpenOrders (standard):")
data = query({"type": "frontendOpenOrders", "user": wallet})
if data:
    print(f"   Result: {len(data)} orders")

# 3. Try with perpDex parameter
print("\n3. openOrders with perpDex='km':")
data = query({"type": "openOrders", "user": wallet, "perpDex": "km"})
if data:
    print(f"   Result: {len(data) if isinstance(data, list) else data}")

# 4. Try perpDexOpenOrders
print("\n4. perpDexOpenOrders:")
data = query({"type": "perpDexOpenOrders", "user": wallet, "perpDex": "km"})
if data:
    print(f"   Result: {len(data) if isinstance(data, list) else data}")

# 5. allMids to confirm US500 exists
print("\n5. Checking allMids for km:US500:")
data = query({"type": "allMids"})
if data:
    us500_mid = data.get("km:US500")
    print(f"   km:US500 mid price: {us500_mid}")

# 6. meta for km
print("\n6. meta for perpDex='km':")
data = query({"type": "meta", "perpDex": "km"})
if data and isinstance(data, dict):
    universe = data.get("universe", [])
    print(f"   Universe size: {len(universe)}")
    for asset in universe:
        if "US500" in str(asset.get("name", "")):
            print(f"   US500 found: {asset}")

# 7. l2Book for km:US500
print("\n7. l2Book for km:US500:")
data = query({"type": "l2Book", "coin": "km:US500"})
if data and isinstance(data, dict):
    levels = data.get("levels", [[], []])
    bids = levels[0] if len(levels) > 0 else []
    asks = levels[1] if len(levels) > 1 else []
    print(f"   Bids: {len(bids)}, Asks: {len(asks)}")
    if bids:
        print(f"   Best bid: {bids[0]}")
    if asks:
        print(f"   Best ask: {asks[0]}")

# 8. userState to find openOrders field
print("\n8. userState checking openOrders field:")
data = query({"type": "clearinghouseState", "user": wallet})
if data and isinstance(data, dict):
    open_orders = data.get("openOrders", [])
    print(f"   openOrders in state: {len(open_orders)}")

# 9. Try clearinghouseState with perpDex
print("\n9. clearinghouseState with perpDex='km':")
data = query({"type": "clearinghouseState", "user": wallet, "perpDex": "km"})
if data:
    if isinstance(data, dict):
        print(f"   marginSummary: {data.get('marginSummary', {})}")
        open_orders = data.get("openOrders", [])
        print(f"   openOrders in state: {len(open_orders)}")
    else:
        print(f"   Response: {data}")

# 10. Use SDK properly
print("\n10. Using SDK with perp_dexs=['km']:")
from hyperliquid.info import Info
from hyperliquid.utils import constants

try:
    info = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
    
    # open_orders
    orders = info.open_orders(wallet)
    print(f"   info.open_orders: {len(orders)}")
    
    # frontend_open_orders
    frontend = info.frontend_open_orders(wallet)
    print(f"   info.frontend_open_orders: {len(frontend)}")
    
    # user_state
    state = info.user_state(wallet)
    print(f"   user_state margin: {state.get('marginSummary', {})}")
    
except Exception as e:
    print(f"   SDK Error: {e}")

# 11. Check spot state for USDH and margin details
print("\n11. spotClearinghouseState:")
data = query({"type": "spotClearinghouseState", "user": wallet})
if data and isinstance(data, dict):
    for bal in data.get("balances", []):
        coin = bal.get("coin")
        total = float(bal.get("total", 0))
        if total > 0.001:
            print(f"   {coin}: {total}")

# Summary
print("\n" + "=" * 70)
print("CONCLUSION:")
print("If the API shows 0 orders but user sees 135+ in UI,")
print("the orders might be on a different wallet or the UI is showing")
print("order history, not open orders.")
print("=" * 70)
