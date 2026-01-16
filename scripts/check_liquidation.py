#!/usr/bin/env python3
"""Check for liquidation or recent closures."""
from hyperliquid.info import Info
from datetime import datetime

info = Info(skip_ws=True, perp_dexs=['km'])
addr = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

fills = info.user_fills(addr)
km_fills = [f for f in fills if f.get('coin', '').startswith('km:')]

print(f'Total km:US500 fills: {len(km_fills)}')
print()

# Show last 20 fills with timestamps
print('Last 20 fills:')
for f in km_fills[-20:]:
    ts = f.get('time', 0)
    dt = datetime.fromtimestamp(ts/1000)
    side = 'BUY' if f.get('side') == 'B' else 'SELL'
    sz = f.get('sz')
    px = f.get('px')
    fee = f.get('fee', 0)
    cpnl = f.get('closedPnl', 0)
    print(f'{dt.strftime("%H:%M:%S")} {side:4} {sz}@{px} fee=${fee} closedPnl=${cpnl}')

# Check if any fills might be liquidations (usually have liquidation flag)
print()
print('Checking for liquidation fills...')
for f in km_fills:
    if f.get('liquidation'):
        print(f'LIQUIDATION: {f}')
