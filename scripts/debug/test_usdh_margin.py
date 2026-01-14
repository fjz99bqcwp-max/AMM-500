#!/usr/bin/env python3
"""Test if bot can see USDH and trade"""
import os
import sys
from dotenv import load_dotenv

load_dotenv('config/.env')
sys.path.insert(0, os.getcwd())

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange

wallet = os.getenv('WALLET_ADDRESS')
private_key = os.getenv('PRIVATE_KEY')

print('=== USDH MARGIN CHECK ===')
print(f'Wallet: {wallet}\n')

info = Info(skip_ws=True)

# Check spot
spot = info.spot_user_state(wallet)
print('1. SPOT (USDH):')
if 'balances' in spot:
    for bal in spot['balances']:
        total = float(bal.get('total', 0))
        if total > 0.01:
            print(f'   {bal["coin"]}: ${total:,.2f}')
print()

# Check perps
perps = info.user_state(wallet)
print('2. PERPS MARGIN:')
if 'marginSummary' in perps:
    ms = perps['marginSummary']
    account_value = float(ms.get('accountValue', 0))
    withdrawable = float(ms.get('withdrawable', 0))
    print(f'   Account Value: ${account_value:,.2f}')
    print(f'   Withdrawable: ${withdrawable:,.2f}')
    
    # Check cross/isolated margin
    if 'crossMarginSummary' in perps:
        cms = perps['crossMarginSummary']
        print(f'   Cross Margin Value: ${float(cms.get("accountValue", 0)):,.2f}')
print()

print('3. ANALYSIS:')
if account_value > 1400:
    print('   ✅ Perps account funded - ready to trade!')
else:
    print('   ⚠️  Perps account shows $0')
    print('   USDH in spot might need to be "activated" for perps')
    print()
    print('   SOLUTION: Place a small manual trade on Hyperliquid UI')
    print('   1. Go to app.hyperliquid.xyz')
    print('   2. Trade any perp (even $1 worth)')
    print('   3. This will "activate" your USDH for perps trading')
    print('   4. Then the bot will see your balance')
