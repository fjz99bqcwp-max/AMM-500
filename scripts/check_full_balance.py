#!/usr/bin/env python3
"""Check full account balance including spot and isolated margins."""

from hyperliquid.info import Info
import hyperliquid.utils.constants as hl_constants
import json

info = Info(hl_constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
wallet = '0x1ccc14e287a18e5f76c88e8ab8cf2e11f08ff5d4'

# Full state
state = info.user_state(wallet)

print('=== MARGIN SUMMARY ===')
print(f"Account Value: ${float(state['marginSummary']['accountValue']):.2f}")
print(f"Total Raw USD: ${float(state['marginSummary']['totalRawUsd']):.2f}")

# Cross margin
cross = state.get('crossMarginSummary', {})
print()
print('=== CROSS MARGIN ===')
print(f"Account Value: ${float(cross.get('accountValue', 0)):.2f}")

# Isolated margins
print()
print('=== ISOLATED MARGINS ===')
for iso in state.get('isolatedMarginSummaries', []):
    print(json.dumps(iso, indent=2))

# Positions
print()
print('=== POSITIONS ===')
for pos in state.get('assetPositions', []):
    p = pos.get('position', {})
    if float(p.get('szi', 0)) != 0:
        print(json.dumps(p, indent=2))

# Check spot balances via spot API
print()
print('=== SPOT STATE ===')
try:
    spot_state = info.spot_user_state(wallet)
    balances = spot_state.get('balances', [])
    for b in balances:
        if float(b.get('hold', 0)) > 0 or float(b.get('total', 0)) > 0:
            print(f"  {b.get('coin')}: total={b.get('total')}, hold={b.get('hold')}")
except Exception as e:
    print(f"Error getting spot state: {e}")

# Check HIP-3 margin
print()
print('=== HIP-3 USDH MARGIN ===')
try:
    km_state = info.user_hip3_state(wallet)
    print(f"Spot USDH: ${float(km_state.get('spotUsdh', {}).get('total', 0)):.2f}")
    print(f"USDH Hold: ${float(km_state.get('spotUsdh', {}).get('hold', 0)):.2f}")
    
    # Isolated balances per asset
    for iso in km_state.get('isolatedBalances', []):
        print(f"Asset {iso.get('asset')}: equity=${float(iso.get('equity', 0)):.2f}, available=${float(iso.get('available', 0)):.2f}")
except Exception as e:
    print(f"Error: {e}")
