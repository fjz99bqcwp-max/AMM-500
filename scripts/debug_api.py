#!/usr/bin/env python3
"""Debug API calls for HIP-3 perp dex."""

import requests
import json

MAINNET_API = "https://api.hyperliquid.xyz/info"
wallet = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

def query(payload):
    """Make API query."""
    resp = requests.post(MAINNET_API, json=payload)
    return resp.json()

print("=== Method 1: Regular clearinghouseState ===")
try:
    data = query({"type": "clearinghouseState", "user": wallet})
    margin = data.get('marginSummary', {})
    print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
    print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):,.2f}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 2: spotClearinghouseState ===")
try:
    data = query({"type": "spotClearinghouseState", "user": wallet})
    print(f"Balances: {data.get('balances', [])}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 3: Check all user fills ===")
try:
    data = query({"type": "userFills", "user": wallet})
    if data:
        print(f"Total fills: {len(data)}")
        for f in data[:5]:
            print(f"  {f.get('coin')}: {f.get('side')} {f.get('sz')} @ {f.get('px')}")
    else:
        print("No fills")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 4: All open orders ===")
try:
    data = query({"type": "openOrders", "user": wallet})
    print(f"Open orders: {len(data)}")
    for o in data[:10]:
        print(f"  {o}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 5: frontendOpenOrders ===")
try:
    data = query({"type": "frontendOpenOrders", "user": wallet})
    print(f"Frontend orders: {len(data)}")
    for o in data[:10]:
        print(f"  {o}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 6: Check order status for known OIDs ===")
# These are OIDs from earlier in the conversation
oids = [293827435961, 293827435962, 293827435963, 293827435964]
for oid in oids:
    try:
        data = query({"type": "orderStatus", "user": wallet, "oid": oid})
        print(f"OID {oid}: {data}")
    except Exception as e:
        print(f"OID {oid} error: {e}")
print()

print("=== Method 7: Get meta for km perp dex ===")
try:
    # Try to find the right API call for HIP-3
    # The SDK source shows perpMetas for perp_dexs
    data = query({"type": "perpMetas"})
    print(f"perpMetas type: {type(data)}")
    if isinstance(data, list):
        for i, m in enumerate(data):
            if isinstance(m, dict) and m.get('universe'):
                for asset in m.get('universe', [])[:3]:
                    print(f"  [{i}] {asset.get('name')}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 8: Check user state via perpDexs array ===")
try:
    # Try clearinghouseStates (plural) for multiple dexs
    data = query({"type": "clearinghouseStates", "user": wallet})
    print(f"clearinghouseStates type: {type(data)}")
    if isinstance(data, list):
        for i, state in enumerate(data):
            if isinstance(state, dict):
                margin = state.get('marginSummary', {})
                val = float(margin.get('accountValue', 0))
                if val > 0:
                    print(f"  State {i}: Account Value = ${val:,.2f}")
    print()
except Exception as e:
    print(f"Error: {e}")

print("=== Method 9: Check spot balance ===")
try:
    data = query({"type": "spotClearinghouseState", "user": wallet})
    for bal in data.get('balances', []):
        print(f"  {bal.get('coin')}: {bal.get('hold')} hold, {bal.get('total')} total")
    print()
except Exception as e:
    print(f"Error: {e}")
