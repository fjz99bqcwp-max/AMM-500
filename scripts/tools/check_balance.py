#!/usr/bin/env python3
"""Quick script to verify account balance on Hyperliquid"""
import requests
import json

wallet = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"
url = "https://api.hyperliquid.xyz/info"

# Check Perps (Clearinghouse)
perps_payload = {
    "type": "clearinghouseState",
    "user": wallet
}
perps_response = requests.post(url, json=perps_payload)
perps_data = perps_response.json()

# Check Spot
spot_payload = {
    "type": "spotClearinghouseState",
    "user": wallet
}
spot_response = requests.post(url, json=spot_payload)
spot_data = spot_response.json()

print(f"=== ACCOUNT VERIFICATION ===")
print(f"Wallet: {wallet}")
print(f"Network: Hyperliquid Mainnet")
print()

# Perps Account
print("💹 PERPS ACCOUNT (for trading perpetuals):")
if 'marginSummary' in perps_data:
    ms = perps_data['marginSummary']
    perps_value = float(ms.get('accountValue', 0))
    
    print(f"   Account Value: ${perps_value:,.2f}")
    print(f"   Withdrawable: ${float(ms.get('withdrawable', 0)):,.2f}")
    print(f"   Margin Used: ${float(ms.get('totalMarginUsed', 0)):,.2f}")
else:
    perps_value = 0
    print(f"   Account Value: $0.00")
print()

# Spot Account
print("💰 SPOT ACCOUNT (needs transfer to Perps):")
spot_value = 0
if 'balances' in spot_data:
    balances = spot_data['balances']
    for balance in balances:
        coin = balance['coin']
        total = float(balance['total'])
        if total > 0:
            spot_value += total
            print(f"   {coin}: ${total:,.2f}")
    if spot_value == 0:
        print(f"   Balance: $0.00")
else:
    print(f"   Balance: $0.00")
print()

# Summary
total_value = perps_value + spot_value
print("📊 SUMMARY:")
print(f"   Total Balance: ${total_value:,.2f}")
print(f"   - Spot: ${spot_value:,.2f}")
print(f"   - Perps: ${perps_value:,.2f}")
print()

if perps_value >= 1400:
    print("✅ PERPS ACCOUNT FUNDED - Ready to trade!")
elif spot_value >= 1400:
    print("⚠️  FUNDS IN SPOT - Need to transfer to Perps!")
    print()
    print("ACTION REQUIRED:")
    print("1. Go to: https://app.hyperliquid.xyz")
    print("2. Click 'Transfer' between Spot and Perps")
    print(f"3. Transfer ${spot_value:,.2f} from Spot → Perps")
    print("4. Re-run: python3 check_balance.py")
elif total_value > 0:
    print(f"⚠️  INSUFFICIENT FUNDS - ${total_value:,.2f}")
    print(f"   Expected: ~$1469 USDC")
else:
    print("❌ NO FUNDS DETECTED")

print()
print("=== POSITIONS ===")
if 'assetPositions' in perps_data:
    positions = [p for p in perps_data['assetPositions'] if abs(float(p['position']['szi'])) > 0]
    if positions:
        print(f"Open Positions: {len(positions)}")
        for pos in positions:
            coin = pos['position']['coin']
            size = float(pos['position']['szi'])
            entry = float(pos['position']['entryPx'])
            print(f"  {coin}: {size:+.4f} @ ${entry:.2f}")
    else:
        print("No open positions")
else:
    print("No positions")
