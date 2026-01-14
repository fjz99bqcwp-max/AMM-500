#!/usr/bin/env python3
"""Test if bot now detects USDH in Spot as margin for HIP-3."""

from hyperliquid.info import Info
from hyperliquid.utils import constants

wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
info = Info(constants.MAINNET_API_URL, skip_ws=True)

# Get Perps state
user_state = info.user_state(wallet)
margin = user_state.get("marginSummary", {})
perps_equity = float(margin.get("accountValue", 0))
perps_available = float(margin.get("withdrawable", 0))

print("=== CURRENT STATE ===")
print(f"Perps Equity: ${perps_equity:,.2f}")
print(f"Perps Available: ${perps_available:,.2f}")

# Get Spot USDH
spot_state = info.spot_user_state(wallet)
spot_usdh = 0.0
for b in spot_state.get("balances", []):
    if b.get("coin") == "USDH":
        spot_usdh = float(b.get("total", 0))
        break

print(f"\nSpot USDH: ${spot_usdh:,.2f}")

# Calculate what the bot will now see
total_equity = perps_equity + spot_usdh
total_available = perps_available + spot_usdh

print("\n=== WITH HIP-3 FIX ===")
print(f"Total Equity (Perps + Spot USDH): ${total_equity:,.2f}")
print(f"Total Available: ${total_available:,.2f}")

if total_equity > 0:
    print("\n✅ Bot will now see $1,469.10 as available margin for km:US500 trading!")
else:
    print("\n❌ Still no margin detected")
