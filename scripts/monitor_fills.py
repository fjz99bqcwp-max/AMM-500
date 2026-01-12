#!/usr/bin/env python3
import time
from hyperliquid.info import Info

info = Info(skip_ws=True)
wallet = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"

# Get current fills
fills = info.user_fills(wallet)
btc = [f for f in fills if f.get("coin") == "BTC"]
START = len(btc)
TARGET = 10

print(f"=== MONITORING FOR {TARGET} NEW FILLS ===")
print(f"Starting from: {START} fills")
print("-" * 40)

last_count = START
while True:
    fills = info.user_fills(wallet)
    btc = [f for f in fills if f.get("coin") == "BTC"]
    current = len(btc)
    new_fills = current - START

    # Show new fill immediately
    if current > last_count:
        for f in btc[: (current - last_count)]:
            side = "BUY " if f.get("side") == "B" else "SELL"
            px = float(f["px"])
            sz = float(f["sz"])
            pnl = float(f.get("closedPnl", 0))
            print(f"  NEW: {side} {sz:.5f} BTC @ ${px:.2f} | PnL: ${pnl:.4f}")
        last_count = current

    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] Fills: {current} | New: {new_fills}/{TARGET}")

    if new_fills >= TARGET:
        print(f"\n✅ TARGET REACHED! {new_fills} new fills!")

        # Show stats
        recent = btc[:new_fills]
        total_pnl = sum(float(f.get("closedPnl", 0)) for f in recent)
        total_fees = sum(float(f.get("fee", 0)) for f in recent)
        print(f"Gross PnL: ${total_pnl:.4f}")
        print(f"Fees: ${total_fees:.4f}")
        print(f"Net PnL: ${total_pnl - total_fees:.4f}")
        break

    time.sleep(15)
