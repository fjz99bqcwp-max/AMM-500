#!/usr/bin/env python3
"""
Comprehensive check of ALL balances and orders for the wallet.
This script checks every possible location for USDH and orders.
"""

import requests
import json
import os
from dotenv import load_dotenv
load_dotenv('config/.env')

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = os.getenv('WALLET_ADDRESS', '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C').strip()

def query(payload):
    resp = requests.post(MAINNET_API, json=payload)
    if resp.status_code == 200:
        return resp.json()
    return {"error": f"HTTP {resp.status_code}", "body": resp.text[:200]}

print(f"Wallet: {wallet}")
print("=" * 70)

# ============================================================================
# 1. SPOT BALANCES
# ============================================================================
print("\n" + "=" * 70)
print("1. SPOT BALANCES")
print("=" * 70)

spot_data = query({"type": "spotClearinghouseState", "user": wallet})
spot_usdh = 0
if isinstance(spot_data, dict) and "balances" in spot_data:
    for bal in spot_data.get("balances", []):
        coin = bal.get("coin", "")
        total = float(bal.get("total", 0))
        hold = float(bal.get("hold", 0))
        if total > 0.001:
            print(f"  {coin}: total={total:.4f}, hold={hold:.4f}")
            if coin == "USDH":
                spot_usdh = total
else:
    print(f"  Error: {spot_data}")

# ============================================================================
# 2. MAIN PERPS (USDC margin)
# ============================================================================
print("\n" + "=" * 70)
print("2. MAIN PERPS (USDC margin)")
print("=" * 70)

perp_data = query({"type": "clearinghouseState", "user": wallet})
main_perp_value = 0
if isinstance(perp_data, dict):
    margin = perp_data.get("marginSummary", {})
    main_perp_value = float(margin.get("accountValue", 0))
    print(f"  Account Value: ${main_perp_value:.2f}")
    print(f"  Withdrawable: ${float(margin.get('withdrawable', 0)):.2f}")
    print(f"  Margin Used: ${float(margin.get('totalMarginUsed', 0)):.2f}")
    
    # Check positions
    for pos in perp_data.get("assetPositions", []):
        p = pos.get("position", {})
        szi = float(p.get("szi", 0))
        if szi != 0:
            print(f"  Position: {p.get('coin')} = {szi} @ entry {p.get('entryPx')}")

# ============================================================================
# 3. HIP-3 PERPS (km deployer) - Try multiple query methods
# ============================================================================
print("\n" + "=" * 70)
print("3. HIP-3 PERPS (km deployer)")
print("=" * 70)

# Method A: Using SDK with perp_dexs
from hyperliquid.info import Info
from hyperliquid.utils import constants

info_km = Info(constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])

km_perp_value = 0
try:
    km_state = info_km.user_state(wallet)
    margin = km_state.get("marginSummary", {})
    km_perp_value = float(margin.get("accountValue", 0))
    print(f"  [SDK perp_dexs=['km']]")
    print(f"  Account Value: ${km_perp_value:.2f}")
    print(f"  Withdrawable: ${float(margin.get('withdrawable', 0)):.2f}")
    print(f"  Raw margin data: {margin}")
    
    # Check positions in km
    for pos in km_state.get("assetPositions", []):
        p = pos.get("position", {})
        szi = float(p.get("szi", 0))
        if szi != 0:
            print(f"  Position: {p.get('coin')} = {szi}")
except Exception as e:
    print(f"  SDK Error: {e}")

# ============================================================================
# 4. ALL OPEN ORDERS - Multiple methods
# ============================================================================
print("\n" + "=" * 70)
print("4. OPEN ORDERS")
print("=" * 70)

# Method A: Standard API
orders_main = query({"type": "openOrders", "user": wallet})
print(f"  [Main API openOrders]: {len(orders_main) if isinstance(orders_main, list) else 'error'}")

# Method B: Frontend open orders
frontend_orders = query({"type": "frontendOpenOrders", "user": wallet})
print(f"  [Frontend openOrders]: {len(frontend_orders) if isinstance(frontend_orders, list) else 'error'}")

# Method C: SDK with km perp_dexs
try:
    km_orders = info_km.open_orders(wallet)
    print(f"  [SDK km open_orders]: {len(km_orders)}")
    if km_orders:
        # Count by coin
        by_coin = {}
        for o in km_orders:
            coin = o.get('coin', 'unknown')
            by_coin[coin] = by_coin.get(coin, 0) + 1
        print(f"    By coin: {by_coin}")
except Exception as e:
    print(f"  SDK km orders error: {e}")

# Method D: SDK frontend orders with km
try:
    km_frontend = info_km.frontend_open_orders(wallet)
    print(f"  [SDK km frontend_orders]: {len(km_frontend)}")
except Exception as e:
    print(f"  SDK km frontend error: {e}")

# ============================================================================
# 5. SEARCH FOR ORDERS BY OID RANGE
# ============================================================================
print("\n" + "=" * 70)
print("5. SEARCHING FOR ALL OPEN ORDERS (by OID)")
print("=" * 70)

# Get recent fills to find OID range
fills = query({"type": "userFills", "user": wallet})
if isinstance(fills, list) and len(fills) > 0:
    # Find OID ranges from fills
    oids_from_fills = []
    for f in fills:
        oid = f.get('oid')
        if oid:
            try:
                oids_from_fills.append(int(oid))
            except:
                pass
    
    if oids_from_fills:
        min_oid = min(oids_from_fills)
        max_oid = max(oids_from_fills)
        print(f"  OID range from {len(fills)} fills: {min_oid} to {max_oid}")
        
        # Search around the max for open orders
        open_orders = []
        search_ranges = [
            (max_oid - 100, max_oid + 500),  # Recent orders
            (max_oid + 500, max_oid + 2000),  # Very recent
        ]
        
        for start, end in search_ranges:
            print(f"  Searching OID range {start} to {end}...")
            for oid in range(start, end):
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
        
        print(f"\n  Found {len(open_orders)} open orders via OID search:")
        # Group by coin and side
        by_coin_side = {}
        for o in open_orders:
            key = f"{o['coin']} {o['side']}"
            if key not in by_coin_side:
                by_coin_side[key] = []
            by_coin_side[key].append(o)
        
        for key, orders in by_coin_side.items():
            prices = [float(o['px']) for o in orders]
            print(f"    {key}: {len(orders)} orders @ ${min(prices):.2f} - ${max(prices):.2f}")
            for o in orders[:5]:
                print(f"      OID {o['oid']}: {o['sz']} @ {o['px']}")
            if len(orders) > 5:
                print(f"      ... and {len(orders) - 5} more")

# ============================================================================
# 6. CHECK SUBACCOUNTS
# ============================================================================
print("\n" + "=" * 70)
print("6. SUBACCOUNTS")
print("=" * 70)

subaccounts = query({"type": "subAccounts", "user": wallet})
if isinstance(subaccounts, list):
    print(f"  Found {len(subaccounts)} subaccounts")
    for sub in subaccounts[:5]:
        print(f"    {sub}")
else:
    print(f"  No subaccounts or error: {subaccounts}")

# ============================================================================
# 7. VAULT CHECK
# ============================================================================
print("\n" + "=" * 70)
print("7. VAULT/BUILDER VAULTS")
print("=" * 70)

# Check if wallet has any vault deposits
try:
    vaults = query({"type": "userVaultEquities", "user": wallet})
    if isinstance(vaults, list) and vaults:
        print(f"  Vault deposits: {len(vaults)}")
        for v in vaults:
            print(f"    {v}")
    else:
        print("  No vault deposits")
except Exception as e:
    print(f"  Vault check error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Spot USDH: ${spot_usdh:.2f}")
print(f"  Main Perps: ${main_perp_value:.2f}")
print(f"  km Perps: ${km_perp_value:.2f}")
print(f"  Total Found: ${spot_usdh + main_perp_value + km_perp_value:.2f}")
print(f"  User Claims: $1469 USDH")
print(f"  Difference: ${1469 - (spot_usdh + main_perp_value + km_perp_value):.2f}")
