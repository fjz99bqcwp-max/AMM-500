#!/usr/bin/env python3
"""Check all balances across Hyperliquid accounts."""

import requests
from hyperliquid.info import Info
import hyperliquid.utils.constants as hl_constants

wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

# Check main clearinghouse (USDC perps)
r = requests.post('https://api.hyperliquid.xyz/info', json={'type': 'clearinghouseState', 'user': wallet})
main = r.json()
print('=== MAIN PERPS (USDC) ===')
print(f"Account Value: ${float(main.get('marginSummary', {}).get('accountValue', 0)):.2f}")

# Check spot balances
r = requests.post('https://api.hyperliquid.xyz/info', json={'type': 'spotClearinghouseState', 'user': wallet})
spot = r.json()
print()
print('=== SPOT BALANCES ===')
for b in spot.get('balances', []):
    total = float(b.get('total', 0))
    if total > 0.001:
        print(f"  {b.get('coin')}: {total:.4f}")

# Check HIP-3 specific state
print()
print('=== HIP-3 (km) USDH PERPS ===')
info = Info(hl_constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
km_state = info.user_state(wallet)
print(f"km Account Value: ${float(km_state.get('marginSummary', {}).get('accountValue', 0)):.2f}")

# Show where funds need to be
print()
print('='*60)
print('NOTE: To trade US500-USDH (HIP-3), you need USDH in the')
print('HIP-3 margin account, NOT in spot or main USDC perps.')
print()
print('Transfer USDH to HIP-3 margin via:')
print('  1. Go to app.hyperliquid.xyz')
print('  2. Navigate to Trade > US500')
print('  3. Click "Transfer" to move USDH from spot to margin')
print('='*60)
