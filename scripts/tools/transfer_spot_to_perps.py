#!/usr/bin/env python3
"""Transfer USDH from Spot to Perps account"""
import os
import sys
from dotenv import load_dotenv

load_dotenv('config/.env')
sys.path.insert(0, os.getcwd())

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

# Main wallet (has funds) and API wallet (signs transactions)
wallet_address = os.getenv('WALLET_ADDRESS')  # 0x1cCC... (main wallet with USDH)
api_wallet_address = os.getenv('API_WALLET_ADDRESS')  # 0x02E0... (API wallet)
private_key = os.getenv('PRIVATE_KEY')  # API wallet's private key

print('=== TRANSFER USDH: SPOT → PERPS ===')
print(f'Main Wallet (with funds): {wallet_address}')
print(f'API Wallet (signing): {api_wallet_address}\n')

# Initialize with account_address set to main wallet
info = Info(skip_ws=True)
account = eth_account.Account.from_key(private_key)

# Key fix: specify account_address as the main wallet
# The API wallet signs, but actions are performed on behalf of main wallet
exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    account_address=wallet_address  # This tells SDK to act on behalf of main wallet
)

# Check current balances
print('1. CHECKING CURRENT BALANCES...')
spot = info.spot_user_state(wallet_address)
perps = info.user_state(wallet_address)

spot_usdh = 0
if 'balances' in spot:
    for bal in spot['balances']:
        if bal['coin'] == 'USDH':
            spot_usdh = float(bal.get('total', 0))
            print(f'   Spot USDH: ${spot_usdh:,.2f}')

perps_value = 0
if 'marginSummary' in perps:
    perps_value = float(perps['marginSummary'].get('accountValue', 0))
    print(f'   Perps Value: ${perps_value:,.2f}')

if spot_usdh < 1:
    print('\n❌ No USDH in Spot to transfer')
    sys.exit(1)

print(f'\n2. TRANSFERRING ${spot_usdh:,.2f} USDH...')

# Try send_asset first (uses different signing method)
# From SDK: send_asset(destination, source_dex, destination_dex, token, amount)
# source_dex="spot", destination_dex="" (empty = default perp)

try:
    # Method 1: send_asset (works with API wallets for HIP-3 perps)
    print('   Attempting send_asset method...')
    result = exchange.send_asset(
        destination=wallet_address,  # Send to same wallet
        source_dex="spot",           # From spot
        destination_dex="",          # To default perps (empty string = main perps)
        token="USDH",                # USDH token
        amount=spot_usdh
    )
    
    print(f'   Result: {result}')
    
    if result.get('status') == 'ok':
        print('\n✅ TRANSFER SUCCESSFUL!')
        print(f'   Transferred: ${spot_usdh:,.2f} USDH')
        print('\n3. VERIFYING NEW BALANCE...')
        
        import time
        time.sleep(3)
        
        perps_new = info.user_state(wallet_address)
        if 'marginSummary' in perps_new:
            new_value = float(perps_new['marginSummary'].get('accountValue', 0))
            print(f'   Perps Value: ${new_value:,.2f}')
            
            if new_value > perps_value:
                print(f'\n✅ SUCCESS! Perps balance: ${new_value:,.2f}')
                print('\nThe bot can now trade on km:US500!')
            else:
                print('\n⚠️  Balance not updated yet. Wait and check:')
                print('   python check_balance.py')
    else:
        print(f'\n❌ send_asset failed: {result}')
        print('\nTrying alternative method...')
        
        # Method 2: usd_class_transfer (requires main wallet key)
        print('   Attempting usd_class_transfer...')
        result2 = exchange.usd_class_transfer(
            amount=spot_usdh,
            to_perp=True
        )
        print(f'   Result: {result2}')
        
        if result2.get('status') == 'ok':
            print('\n✅ TRANSFER SUCCESSFUL!')
        else:
            print(f'\n❌ Both methods failed.')
            print('\nManual transfer required via Hyperliquid UI.')
        
except Exception as e:
    print(f'\n❌ Error: {e}')
    print('\nManual transfer required:')
    print('1. Go to: https://app.hyperliquid.xyz')
    print('2. Connect wallet: 0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C')
    print('3. Click "Transfer" → Spot → Perps')
    print(f'4. Transfer {spot_usdh:.2f} USDH')
