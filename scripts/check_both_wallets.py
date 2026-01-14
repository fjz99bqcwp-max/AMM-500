#!/usr/bin/env python3
"""Check both wallets for balances and orders"""
import requests
import json

MAINNET = 'https://api.hyperliquid.xyz/info'

# Check BOTH wallets
wallets = {
    'Main': '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C',
    'API': '0x02E02FBC90a195BF82dDc4d43FAfd8449B518805'
}

grand_total = 0

for name, wallet in wallets.items():
    print(f'\n{"="*50}')
    print(f'{name} Wallet: {wallet}')
    print("="*50)
    
    wallet_total = 0
    
    # Spot
    r = requests.post(MAINNET, json={'type': 'spotClearinghouseState', 'user': wallet})
    spot = r.json()
    print("\nSpot Balances:")
    for bal in spot.get('balances', []):
        amt = float(bal.get('total', 0))
        if amt > 0.001:
            print(f"  {bal['coin']}: {amt:.4f}")
            if bal['coin'] in ['USDC', 'USDH']:
                wallet_total += amt
    
    # km Perps - check margin and positions
    r = requests.post(MAINNET, json={'type': 'clearinghouseState', 'user': wallet, 'perpDex': 'km'})
    km = r.json()
    margin = km.get('marginSummary', {})
    acct_val = float(margin.get('accountValue', 0))
    wallet_total += acct_val
    print(f"\nkm Perps:")
    print(f"  Account Value: ${acct_val:.4f}")
    
    positions = km.get('assetPositions', [])
    for pos in positions:
        p = pos.get('position', {})
        if float(p.get('szi', 0)) != 0:
            print(f"  Position: {p.get('coin')} size={p.get('szi')} entry={p.get('entryPx')}")
    
    open_orders = km.get('openOrders', [])
    print(f"  Open Orders: {len(open_orders)}")
    
    # Main perps too
    r = requests.post(MAINNET, json={'type': 'clearinghouseState', 'user': wallet})
    main = r.json()
    main_margin = main.get('marginSummary', {})
    main_acct = float(main_margin.get('accountValue', 0))
    if main_acct > 0.01:
        print(f"\nMain Perps Account Value: ${main_acct:.4f}")
        wallet_total += main_acct
    
    print(f"\n>>> {name} Wallet Total: ${wallet_total:.2f}")
    grand_total += wallet_total

print(f"\n{'='*50}")
print(f"GRAND TOTAL ACROSS ALL WALLETS: ${grand_total:.2f}")
print("="*50)

# Also check userFills to see recent activity
print("\n\nRecent Fills (last 10):")
r = requests.post(MAINNET, json={'type': 'userFills', 'user': wallets['Main']})
fills = r.json()[-10:] if r.json() else []
for fill in fills:
    print(f"  {fill.get('time')[:16]} {fill.get('coin')} {fill.get('side')} {fill.get('sz')} @ {fill.get('px')}")
