#!/usr/bin/env python3
"""Enhanced balance check with Perp USDH detection"""

from dotenv import load_dotenv
from pathlib import Path
import os
import requests
import json

# Load config
load_dotenv(Path('config/.env'))
wallet = os.getenv('WALLET_ADDRESS', '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C')

print(f'Checking wallet: {wallet}')

# Check spot balance
spot_resp = requests.post('https://api.hyperliquid.xyz/info', 
                         json={'type': 'spotClearinghouseState', 'user': wallet}, 
                         timeout=10)
spot_data = spot_resp.json()

print('\n=== SPOT BALANCES ===')
spot_usdh = 0
for balance in spot_data.get('balances', []):
    coin = balance.get('coin', '')
    total = float(balance.get('total', '0'))
    hold = float(balance.get('hold', '0'))
    available = total - hold
    print(f'{coin}: Total ${total:.2f}, Hold ${hold:.2f}, Available ${available:.2f}')
    if coin == 'USDH':
        spot_usdh = total

# Check perp balance  
perp_resp = requests.post('https://api.hyperliquid.xyz/info', 
                         json={'type': 'clearinghouseState', 'user': wallet}, 
                         timeout=10)
perp_data = perp_resp.json()

print('\n=== PERP ACCOUNT ===')
margin_summary = perp_data.get('marginSummary', {})
account_value = float(margin_summary.get('accountValue', '0'))
print(f'Account Value: ${account_value:.2f}')
print(f'Total Margin Used: ${float(margin_summary.get("totalMarginUsed", "0")):.2f}')
print(f'Withdrawable: ${float(perp_data.get("withdrawable", "0")):.2f}')

# Check for USDH in perp asset positions
print('\n=== PERP ASSET POSITIONS ===')
perp_usdh = 0
asset_positions = perp_data.get('assetPositions', [])
for asset in asset_positions:
    position_info = asset.get('position', {})
    coin = position_info.get('coin', '')
    szi = float(position_info.get('szi', 0))
    print(f'{coin}: Size {szi:.8f}')
    if coin == 'USDH':
        perp_usdh = szi

# If no explicit USDH position, account value might be USDH
if perp_usdh == 0 and account_value > 0:
    perp_usdh = account_value
    print(f'Using account value as Perp USDH: ${perp_usdh:.2f}')

# Check for USDH in open orders
print('\n=== OPEN ORDERS ===')
orders_resp = requests.post('https://api.hyperliquid.xyz/info', 
                           json={'type': 'openOrders', 'user': wallet}, 
                           timeout=10)
orders_data = orders_resp.json()

total_held_in_orders = 0
if orders_data:
    for order in orders_data:
        if order.get('coin') == 'USDH':
            sz = float(order.get('sz', 0))
            px = float(order.get('limitPx', 0))
            notional = sz * px
            total_held_in_orders += notional
            print(f'USDH Order: {sz:.4f} @ ${px:.2f} = ${notional:.2f}')

print(f'Total USDH held in orders: ${total_held_in_orders:.2f}')

# Check all user state
print('\n=== USER STATE ===')
user_state_resp = requests.post('https://api.hyperliquid.xyz/info', 
                                json={'type': 'userState', 'user': wallet}, 
                                timeout=10)
user_state_data = user_state_resp.json()
print(f'User State Response: {json.dumps(user_state_data, indent=2)[:1000]}...')

# Check withdrawal status
print('\n=== WITHDRAWAL STATUS ===')
try:
    withdrawal_resp = requests.post('https://api.hyperliquid.xyz/info', 
                                   json={'type': 'userFunding', 'user': wallet}, 
                                   timeout=10)
    withdrawal_data = withdrawal_resp.json()
    print(f'Funding Data: {json.dumps(withdrawal_data, indent=2)[:1000]}...')
except Exception as e:
    print(f'Could not get funding data: {e}')

# Calculate total balance
total_balance = spot_usdh + perp_usdh + total_held_in_orders

print(f'\n=== SUMMARY ===')
print(f'Spot USDH: ${spot_usdh:.2f}')  
print(f'Perp USDH: ${perp_usdh:.2f}')
print(f'Orders USDH: ${total_held_in_orders:.2f}')
print(f'TOTAL BALANCE: ${total_balance:.2f}')
print(f'Expected: ~$1464.40')
print(f'Difference: ${1464.40 - total_balance:+.2f}')

# Important note
if total_balance < 1400:
    print(f'\n⚠️  SIGNIFICANT DISCREPANCY DETECTED!')
    print(f'   API shows: ${total_balance:.2f}')
    print(f'   UI shows:  $1464.40')
    print(f'   Missing:   ${1464.40 - total_balance:.2f}')
    print(f'\n   Possible causes:')
    print(f'   - Funds in subaccount or different wallet')
    print(f'   - USDH locked in complex positions')
    print(f'   - Recent transfers not reflected in API')
    print(f'   - UI caching different values')

# Print raw data for debugging
print(f'\n=== RAW DATA DEBUG ===')
print(f'Raw Spot Response: {json.dumps(spot_data, indent=2)}')
print(f'Raw Perp Response: {json.dumps(perp_data, indent=2)}')
