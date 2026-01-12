#!/usr/bin/env python3
"""Detailed trade-by-trade analysis."""

import requests
import time
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
URL = "https://api.hyperliquid.xyz/info"

resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
fills = resp.json()

# Last 20 fills
recent = fills[:20]

print("=" * 70)
print("LAST 20 FILLS - DETAILED ANALYSIS")
print("=" * 70)

for i, f in enumerate(recent, 1):
    ts = datetime.fromtimestamp(f.get("time", 0) / 1000).strftime("%H:%M:%S")
    side = "BUY " if f.get("side") == "B" else "SELL"
    px = float(f.get("px", 0))
    sz = float(f.get("sz", 0))
    fee = float(f.get("fee", 0))
    print(f"{i:2}. {ts} | {side:4} {sz:.5f} BTC @ ${px:,.2f} | fee: ${fee:.4f}")

# Analyze pairs
buys = [f for f in recent if f.get("side") == "B"]
sells = [f for f in recent if f.get("side") == "A"]

print(f"\n{'='*70}")
print(f"SUMMARY (Last 20 fills)")
print(f"{'='*70}")
print(f"Buys:  {len(buys)}")
print(f"Sells: {len(sells)}")

if len(buys) > 0:
    avg_buy = sum(float(f.get("px", 0)) for f in buys) / len(buys)
    print(f"Avg Buy Price: ${avg_buy:.2f}")

if len(sells) > 0:
    avg_sell = sum(float(f.get("px", 0)) for f in sells) / len(sells)
    print(f"Avg Sell Price: ${avg_sell:.2f}")

if len(buys) > 0 and len(sells) > 0:
    spread = avg_sell - avg_buy
    mid = (avg_buy + avg_sell) / 2
    spread_bps = spread / mid * 10000
    print(f"\nSpread: ${spread:.2f} ({spread_bps:.2f} bps)")

    if spread_bps < 0:
        print(f"❌ ADVERSE SELECTION: Buying HIGHER than selling!")
        print(f"   We're getting filled at WORSE prices than mid")
    else:
        print(f"✅ Positive spread")

# Check if we're getting picked off on one side
print(f"\n{'='*70}")
print("MARKET DIRECTION ANALYSIS")
print(f"{'='*70}")

# Compare first and last fills
if len(recent) >= 2:
    first_px = float(recent[-1].get("px", 0))
    last_px = float(recent[0].get("px", 0))
    move = last_px - first_px
    move_pct = move / first_px * 100

    print(f"Price moved from ${first_px:.2f} to ${last_px:.2f}")
    print(f"Change: ${move:.2f} ({move_pct:.2f}%)")

    if move > 0:
        print("Market moving UP - We likely bought BEFORE and sold AFTER the move")
    else:
        print("Market moving DOWN - We likely sold BEFORE and bought AFTER the move")
