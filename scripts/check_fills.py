#!/usr/bin/env python3
"""Quick script to check recent fills."""
from hyperliquid.info import Info
from datetime import datetime

info = Info(skip_ws=True)
fills = info.user_fills("0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283")

print("=" * 60)
print("RECENT FILLS (Last 15)")
print("=" * 60)

for f in fills[:15]:
    side = f["side"]
    side_str = "BUY " if side == "B" else "SELL"
    px = float(f["px"])
    sz = float(f["sz"])
    fee = float(f.get("fee", 0))
    ts = int(f.get("time", 0))
    time_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"
    print(f"{time_str} | {side_str} {sz:.5f} BTC @ ${px:,.2f} (fee: ${fee:.4f})")
