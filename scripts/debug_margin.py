#!/usr/bin/env python3
"""Check HIP-3 isolated margin for US500."""
import json
from hyperliquid.info import Info

info = Info(skip_ws=True, perp_dexs=['km'])
addr = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

print('=== Full HIP-3 User State ===')
state = info.user_state(addr)
print(json.dumps(state, indent=2)[:3000])

print('\n=== Checking user_fills for today ===')
import time
end_time = int(time.time() * 1000)
start_time = end_time - 24*60*60*1000  # Last 24 hours

try:
    fills = info.user_fills(addr)
    print(f'Total fills: {len(fills)}')
    if fills:
        print('Last 5 fills:')
        for f in fills[-5:]:
            print(f"  {f.get('coin')} {f.get('side')} {f.get('sz')}@{f.get('px')} = ${float(f.get('sz', 0))*float(f.get('px', 0)):.2f}")
except Exception as e:
    print(f'Error: {e}')

print('\n=== Checking spot USDH balance ===')
spot_info = Info(skip_ws=True)  # Without perp_dexs for spot
spot_state = spot_info.spot_user_state(addr)
for b in spot_state.get('balances', []):
    if b.get('coin') == 'USDH':
        print(f"USDH: total={b.get('total')}, hold={b.get('hold')}")

print('\n=== Checking Historical Orders for km:US500 ===')
try:
    from hyperliquid.exchange import Exchange
    from eth_account import Account
    import os
    from dotenv import load_dotenv
    load_dotenv('/Users/nheosdisplay/VSC/AMM/AMM-500/config/.env')
    
    # Get filled orders to understand where margin went
    historical = info.historical_orders(addr)
    km_orders = [o for o in historical if o.get('coin', '').startswith('km:')]
    print(f'Total km: orders: {len(km_orders)}')
    if km_orders:
        print('Last 3 km orders:')
        for o in km_orders[-3:]:
            print(f"  {o.get('coin')} {o.get('side')} {o.get('sz')}@{o.get('limitPx')} status={o.get('status')}")
except Exception as e:
    print(f'Error: {e}')
