#!/usr/bin/env python3
"""Query Hyperliquid for real-time trading metrics."""

from hyperliquid.info import Info
import hyperliquid.utils.constants as hl_constants
from datetime import datetime

info = Info(hl_constants.MAINNET_API_URL, skip_ws=True, perp_dexs=['km'])
wallet = '0x1ccc14e287a18e5f76c88e8ab8cf2e11f08ff5d4'

# Get current state
state = info.user_state(wallet)
equity = float(state['marginSummary']['accountValue'])

# Get recent fills  
fills = info.user_fills(wallet)
today_fills = [f for f in fills if f.get('time', 0) > 1768368000000]

print('='*60)
print('REAL-TIME HYPERLIQUID US500-USDH ANALYSIS')
print('='*60)
print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Wallet: {wallet[:10]}...{wallet[-6:]}')
print()
print(f'💰 Equity: ${equity:.2f}')
print(f'📈 Profit: ${equity - 1000:.2f} ({(equity/1000 - 1)*100:.2f}%)')
print()
print(f'📊 Fills Today: {len(today_fills)}')

if today_fills:
    # Count maker/taker
    maker = sum(1 for f in today_fills if not str(f.get('fee', '')).startswith('-'))
    taker = len(today_fills) - maker
    maker_ratio = (maker/len(today_fills)*100)
    
    print(f'   Maker: {maker} ({maker_ratio:.1f}%)')
    print(f'   Taker: {taker} ({100-maker_ratio:.1f}%)')
    
    # Calculate PnL from fills
    total_pnl = sum(float(f.get('closedPnl', 0)) for f in today_fills)
    total_fees = sum(abs(float(f.get('fee', 0))) for f in today_fills)
    
    print(f'   PnL: ${total_pnl:.2f}')
    print(f'   Fees: ${total_fees:.2f}')
    print(f'   Net: ${total_pnl - total_fees:.2f}')

# Get open orders
orders = info.open_orders(wallet)
print()
print(f'📋 Open Orders: {len(orders)}')

if orders:
    bids = sum(1 for o in orders if o.get('side') == 'B')
    asks = sum(1 for o in orders if o.get('side') == 'A')
    print(f'   Bids: {bids}')
    print(f'   Asks: {asks}')

print('='*60)
