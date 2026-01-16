#!/usr/bin/env python3
"""Full fill analysis for km:US500."""
from hyperliquid.info import Info
from datetime import datetime

info = Info(skip_ws=True, perp_dexs=['km'])
addr = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

fills = info.user_fills(addr)
km_fills = [f for f in fills if f.get('coin', '').startswith('km:')]

print(f'Total km:US500 fills: {len(km_fills)}')

# Running position calculation
pos = 0
total_fees = 0
total_closed_pnl = 0
margin_used = 0

for f in km_fills:
    sz = float(f.get('sz', 0))
    px = float(f.get('px', 0))
    fee = float(f.get('fee', 0))
    cpnl = float(f.get('closedPnl', 0))
    side = f.get('side')
    
    if side == 'B':
        pos += sz
    else:
        pos -= sz
    
    total_fees += fee
    total_closed_pnl += cpnl

print(f'Final calculated position: {pos:.4f}')
print(f'Total fees: ${total_fees:.4f}')
print(f'Total closed PnL: ${total_closed_pnl:.4f}')

# Check first and last timestamps
if km_fills:
    first = datetime.fromtimestamp(km_fills[0].get('time', 0)/1000)
    last = datetime.fromtimestamp(km_fills[-1].get('time', 0)/1000)
    print(f'First fill: {first}')
    print(f'Last fill: {last}')

# Show fills by date
from collections import defaultdict
by_date = defaultdict(lambda: {'buys': 0, 'sells': 0, 'fees': 0, 'pnl': 0})
for f in km_fills:
    dt = datetime.fromtimestamp(f.get('time', 0)/1000)
    date = dt.strftime('%Y-%m-%d')
    sz = float(f.get('sz', 0))
    if f.get('side') == 'B':
        by_date[date]['buys'] += sz
    else:
        by_date[date]['sells'] += sz
    by_date[date]['fees'] += float(f.get('fee', 0))
    by_date[date]['pnl'] += float(f.get('closedPnl', 0))

print()
print('By date:')
for date in sorted(by_date.keys()):
    d = by_date[date]
    net = d['buys'] - d['sells']
    print(f'{date}: bought {d["buys"]:.2f}, sold {d["sells"]:.2f}, net={net:.2f}, fees=${d["fees"]:.2f}, pnl=${d["pnl"]:.2f}')
