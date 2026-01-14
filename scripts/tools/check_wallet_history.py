#!/usr/bin/env python3
"""Check wallet trading history and positions."""

from hyperliquid.info import Info
from hyperliquid.utils import constants

wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'
info = Info(constants.MAINNET_API_URL, skip_ws=True)

# Get user state (perps)
print('=== PERPS STATE ===')
state = info.user_state(wallet)
print(f'Account Value: ${float(state.get("marginSummary", {}).get("accountValue", 0)):,.2f}')
print(f'Margin Used: ${float(state.get("marginSummary", {}).get("totalMarginUsed", 0)):,.2f}')
print(f'Withdrawable: ${float(state.get("withdrawable", 0)):,.2f}')

positions = state.get('assetPositions', [])
if positions:
    print('\nPositions:')
    for p in positions:
        pos = p.get('position', {})
        coin = pos.get('coin', '?')
        size = float(pos.get('szi', 0))
        entry = float(pos.get('entryPx', 0))
        pnl = float(pos.get('unrealizedPnl', 0))
        if size != 0:
            print(f'  {coin}: {size} @ ${entry:,.2f} (PnL: ${pnl:,.2f})')
else:
    print('No positions')

# Get recent fills/trades
print('\n=== RECENT TRADES (Last 10) ===')
fills = info.user_fills(wallet)
if fills:
    for f in fills[:10]:
        coin = f.get('coin', '?')
        side = f.get('side', '?')
        sz = f.get('sz', '?')
        px = f.get('px', '?')
        time = f.get('time', '?')
        print(f'  {coin}: {side} {sz} @ ${px} ({time})')
else:
    print('No recent trades')

# Check spot state
print('\n=== SPOT STATE ===')
spot = info.spot_user_state(wallet)
balances = spot.get('balances', [])
for b in balances:
    token = b.get('coin', '?')
    total = float(b.get('total', 0))
    if total > 0:
        print(f'  {token}: {total:,.2f}')

# Check for US500 specifically
print('\n=== US500 INFO ===')
us500_trades = [f for f in fills if 'US500' in f.get('coin', '').upper() or 'US500' in str(f)]
if us500_trades:
    print(f'Found {len(us500_trades)} US500 trades')
    for t in us500_trades[:5]:
        print(f'  {t}')
else:
    print('No US500 trades found in history')
