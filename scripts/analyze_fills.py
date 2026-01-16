#!/usr/bin/env python3
"""Analyze km:US500 trading history."""
from hyperliquid.info import Info

info = Info(skip_ws=True, perp_dexs=['km'])
addr = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

fills = info.user_fills(addr)
km_fills = [f for f in fills if f.get('coin', '').startswith('km:')]
print(f'Total km:US500 fills: {len(km_fills)}')

# Count buys and sells
buys = sum(1 for f in km_fills if f.get('side') == 'B')
sells = sum(1 for f in km_fills if f.get('side') == 'A')
buy_qty = sum(float(f.get('sz', 0)) for f in km_fills if f.get('side') == 'B')
sell_qty = sum(float(f.get('sz', 0)) for f in km_fills if f.get('side') == 'A')
print(f'Buys: {buys} fills, {buy_qty:.4f} total')
print(f'Sells: {sells} fills, {sell_qty:.4f} total')
print(f'Net position: {buy_qty - sell_qty:.4f}')

# Sum realized PnL
total_pnl = sum(float(f.get('closedPnl', 0)) for f in km_fills)
total_fees = sum(float(f.get('fee', 0)) for f in km_fills)
print(f'Closed PnL: ${total_pnl:.2f}')
print(f'Total Fees: ${total_fees:.2f}')

# Net notional traded
buy_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in km_fills if f.get('side') == 'B')
sell_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in km_fills if f.get('side') == 'A')
print(f'Buy Notional: ${buy_notional:.2f}')
print(f'Sell Notional: ${sell_notional:.2f}')
print(f'Net Notional: ${buy_notional - sell_notional:.2f}')
