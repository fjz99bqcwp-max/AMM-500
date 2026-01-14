#!/usr/bin/env python3
"""Analyze HIP-3 fills for US500"""
import requests
import json

WALLET = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

resp = requests.post(
    "https://api.hyperliquid.xyz/info",
    json={
        "type": "userFillsByTime",
        "user": WALLET,
        "startTime": 1768385000000,
        "endTime": 1768500000000,
        "perp_dexs": ["km"]
    }
)
fills = resp.json()

buys = [f for f in fills if f['side'] == 'B']
sells = [f for f in fills if f['side'] == 'A']

buy_qty = sum(float(f['sz']) for f in buys)
sell_qty = sum(float(f['sz']) for f in sells)
buy_fees = sum(float(f['fee']) for f in buys)
sell_fees = sum(float(f['fee']) for f in sells)
buy_avg = sum(float(f['px'])*float(f['sz']) for f in buys) / buy_qty if buy_qty else 0
sell_avg = sum(float(f['px'])*float(f['sz']) for f in sells) / sell_qty if sell_qty else 0

print(f"=== FILL SUMMARY ({len(fills)} fills) ===")
print(f"BUYS:  {len(buys)} fills, {buy_qty:.1f} contracts @ avg ${buy_avg:.2f}, fees ${buy_fees:.4f}")
print(f"SELLS: {len(sells)} fills, {sell_qty:.1f} contracts @ avg ${sell_avg:.2f}, fees ${sell_fees:.4f}")
print()
print(f"Net Position: {buy_qty - sell_qty:.1f} contracts")
if sell_qty and buy_qty:
    spread_captured = sell_avg - buy_avg
    print(f"Avg Spread Captured: ${spread_captured:.2f} ({spread_captured/buy_avg*10000:.1f} bps)")
    
    # PnL calculation
    matched_qty = min(buy_qty, sell_qty)
    gross_pnl = matched_qty * spread_captured
    total_fees = buy_fees + sell_fees
    net_pnl = gross_pnl - total_fees
    print(f"Gross PnL (matched): ${gross_pnl:.4f}")
    print(f"Net PnL: ${net_pnl:.4f}")
    
print(f"Total Fees: ${buy_fees + sell_fees:.4f}")

# Show individual fills
print("\n=== INDIVIDUAL FILLS ===")
for i, f in enumerate(fills):
    side = "BUY " if f['side'] == 'B' else "SELL"
    print(f"{i+1}. {side} {f['sz']} @ ${f['px']} | fee ${f['fee']}")
