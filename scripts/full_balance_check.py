#!/usr/bin/env python3
"""
Comprehensive check of all Hyperliquid account balances and orders.
Check both main perps, HIP-3 perps (km), spot, and subaccounts.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('config/.env')

import requests
import json

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = os.getenv('WALLET_ADDRESS', '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C').strip()

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
    return None

print(f"Wallet: {wallet}")
print("=" * 60)

# 1. Main Perps (USDC margin)
print("\n=== 1. MAIN PERPS (USDC) ===")
data = query({"type": "clearinghouseState", "user": wallet})
if data:
    margin = data.get('marginSummary', {})
    val = float(margin.get('accountValue', 0))
    print(f"Account Value: ${val:,.2f}")
    print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
    print(f"Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
    
    # Positions
    for pos in data.get('assetPositions', []):
        p = pos.get('position', {})
        szi = float(p.get('szi', 0))
        if szi != 0:
            print(f"  Position: {p.get('coin')} = {szi}")

# 2. Spot Balances
print("\n=== 2. SPOT BALANCES ===")
data = query({"type": "spotClearinghouseState", "user": wallet})
if data:
    for bal in data.get('balances', []):
        total = float(bal.get('total', 0))
        if total > 0.001:
            print(f"  {bal.get('coin')}: {total:,.4f}")

# 3. Check all subaccounts
print("\n=== 3. SUBACCOUNTS ===")
data = query({"type": "subAccounts", "user": wallet})
if data:
    print(f"Found {len(data)} subaccounts")
    for sub in data[:10]:
        print(f"  {sub}")
else:
    print("No subaccounts or not supported")

# 4. HIP-3 Perp Dex (km) State
# The km perp dex uses a different clearinghouse
print("\n=== 4. HIP-3 PERPS (km:US500) ===")

# Try perpDex-specific query
# Based on SDK source, the perp_dexs param changes how queries work
from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])

# Get meta first
try:
    meta = info.meta()
    us500_found = False
    for asset in meta.get('universe', []):
        if 'US500' in str(asset.get('name', '')):
            print(f"US500 found in km meta: szDecimals={asset.get('szDecimals')}, maxLeverage={asset.get('maxLeverage')}")
            us500_found = True
            break
    if not us500_found:
        print("US500 not found in km meta!")
except Exception as e:
    print(f"Meta error: {e}")

# Get km perp state
try:
    state = info.user_state(wallet)
    margin = state.get('marginSummary', {})
    val = float(margin.get('accountValue', 0))
    print(f"km Perps Account Value: ${val:,.2f}")
    print(f"km Perps Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
    
    # Positions
    for pos in state.get('assetPositions', []):
        p = pos.get('position', {})
        szi = float(p.get('szi', 0))
        if szi != 0:
            print(f"  Position: {p.get('coin')} = {szi}")
except Exception as e:
    print(f"km state error: {e}")

# 5. Check open orders on km
print("\n=== 5. OPEN ORDERS ===")
try:
    # Regular orders
    orders = info.open_orders(wallet)
    print(f"km open_orders: {len(orders)}")
    
    # Frontend orders
    frontend = info.frontend_open_orders(wallet)
    print(f"km frontend_open_orders: {len(frontend)}")
except Exception as e:
    print(f"Orders error: {e}")

# 6. Search for open orders via orderStatus
print("\n=== 6. SEARCHING FOR OPEN ORDERS (last 2000 OIDs) ===")
# Get most recent fills to find order range
fills = query({"type": "userFills", "user": wallet})
if fills and len(fills) > 0:
    # Find the OID range from fills
    oids_from_fills = [int(f.get('oid', 0)) for f in fills if f.get('oid')]
    if oids_from_fills:
        max_oid = max(oids_from_fills)
        min_oid = min(oids_from_fills)
        print(f"OID range from fills: {min_oid} to {max_oid}")
        
        # Search around the max OID for open orders
        open_orders = []
        search_range = range(max(0, max_oid - 1000), max_oid + 200)
        print(f"Searching {len(list(search_range))} OIDs...")
        
        for oid in search_range:
            data = query({"type": "orderStatus", "user": wallet, "oid": oid})
            if data and data.get('status') == 'order':
                order_data = data.get('order', {})
                if order_data.get('status') == 'open':
                    order = order_data.get('order', {})
                    open_orders.append({
                        'oid': oid,
                        'coin': order.get('coin'),
                        'side': order.get('side'),
                        'sz': order.get('sz'),
                        'px': order.get('limitPx')
                    })
        
        print(f"\nFound {len(open_orders)} open orders:")
        for o in open_orders:
            print(f"  {o['coin']}: {o['side']} {o['sz']} @ {o['px']} (oid: {o['oid']})")

# 7. Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Get spot USDH
data = query({"type": "spotClearinghouseState", "user": wallet})
spot_usdh = 0
if data:
    for bal in data.get('balances', []):
        if bal.get('coin') == 'USDH':
            spot_usdh = float(bal.get('total', 0))

# Get main perps
data = query({"type": "clearinghouseState", "user": wallet})
main_perps = float(data.get('marginSummary', {}).get('accountValue', 0)) if data else 0

# Get km perps
state = info.user_state(wallet)
km_perps = float(state.get('marginSummary', {}).get('accountValue', 0))

print(f"Spot USDH: ${spot_usdh:,.2f}")
print(f"Main Perps: ${main_perps:,.2f}")
print(f"km Perps: ${km_perps:,.2f}")
print(f"Total Visible: ${spot_usdh + main_perps + km_perps:,.2f}")
