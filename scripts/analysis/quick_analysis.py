#!/usr/bin/env python3
"""Quick trade analysis for Hyperliquid fills"""
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict

wallet = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
url = "https://api.hyperliquid.xyz/info"

print("Fetching trade data from Hyperliquid...")
response = requests.post(url, json={"type": "userFills", "user": wallet})
fills = response.json()

# Filter BTC fills only
btc_fills = [f for f in fills if f["coin"] == "BTC"]
print(f"Total BTC fills: {len(btc_fills)}")

# Get last 24h fills
now = datetime.now()
fills_24h = []
for f in btc_fills:
    ts = int(f["time"]) / 1000
    dt = datetime.fromtimestamp(ts)
    if now - dt < timedelta(hours=24):
        fills_24h.append(f)

print(f"Fills in last 24h: {len(fills_24h)}")

print("=" * 70)
print("TRADE ANALYSIS (Last 24 hours)")
print("=" * 70)

buys = [f for f in fills_24h if f["side"] == "B"]
sells = [f for f in fills_24h if f["side"] == "A"]

avg_buy = 0.0
avg_sell = 0.0

if buys:
    avg_buy = sum(float(f["px"]) for f in buys) / len(buys)
    total_buy_sz = sum(float(f["sz"]) for f in buys)
    print(
        f"BUY trades: {len(buys)}, Avg Price: ${avg_buy:,.2f}, Total Size: {total_buy_sz:.4f} BTC"
    )

if sells:
    avg_sell = sum(float(f["px"]) for f in sells) / len(sells)
    total_sell_sz = sum(float(f["sz"]) for f in sells)
    print(
        f"SELL trades: {len(sells)}, Avg Price: ${avg_sell:,.2f}, Total Size: {total_sell_sz:.4f} BTC"
    )

if buys and sells:
    spread = avg_sell - avg_buy
    print(f"\nSPREAD CAPTURED: ${spread:+.2f} per round-trip")
    if spread < 0:
        print("⚠️  ADVERSE SELECTION: Selling LOWER than buying!")
    else:
        print("✅ POSITIVE SPREAD: Strategy is capturing the bid-ask spread")

# Calculate PnL
total_fee = sum(float(f["fee"]) for f in fills_24h)
total_closed_pnl = sum(float(f.get("closedPnl", "0")) for f in fills_24h)
net_pnl = total_closed_pnl - total_fee

print(f"\n--- PnL Summary (24h) ---")
print(f"Closed PnL: ${total_closed_pnl:+.4f}")
print(f"Total Fees: ${total_fee:.4f}")
print(f"Net PnL: ${net_pnl:+.4f}")

# Per-trade analysis
print("\n--- Per-Trade Analysis ---")
winning = 0
losing = 0
for f in fills_24h:
    pnl = float(f.get("closedPnl", "0"))
    fee = float(f["fee"])
    net = pnl - fee
    if net > 0:
        winning += 1
    elif net < 0:
        losing += 1

print(f"Winning trades: {winning}")
print(f"Losing trades: {losing}")
if winning + losing > 0:
    print(f"Win rate: {winning/(winning+losing)*100:.1f}%")

# Show last 30 trades
print("\n--- Last 30 Trades ---")
print("-" * 80)
for f in fills_24h[:30]:
    ts = int(f["time"]) / 1000
    dt = datetime.fromtimestamp(ts)
    side = "BUY " if f["side"] == "B" else "SELL"
    px = float(f["px"])
    sz = float(f["sz"])
    fee = float(f["fee"])
    pnl = float(f.get("closedPnl", "0"))
    net = pnl - fee
    status = "✅" if net > 0 else "❌" if net < 0 else "⚪"
    print(
        f'{dt.strftime("%H:%M:%S")} | {side} {sz:.4f} @ ${px:,.2f} | Fee: ${fee:.4f} | PnL: ${pnl:+.4f} | Net: ${net:+.4f} {status}'
    )

# Analyze trade patterns - are we getting picked off?
print("\n--- Trade Pattern Analysis ---")
# Group trades into round trips
print("Checking for adverse selection patterns...")

# Look at price movement after our fills
for i, f in enumerate(fills_24h[:10]):
    ts = int(f["time"]) / 1000
    dt = datetime.fromtimestamp(ts)
    side = f["side"]
    px = float(f["px"])

    # Find next trade in opposite direction
    for j in range(i + 1, min(i + 20, len(fills_24h))):
        next_f = fills_24h[j]
        if next_f["side"] != side:
            next_px = float(next_f["px"])
            if side == "B":  # We bought, then sold
                profit = next_px - px
            else:  # We sold, then bought
                profit = px - next_px
            print(
                f"Trade {i}: {side} @ ${px:,.2f} -> {'A' if side=='B' else 'B'} @ ${next_px:,.2f} = ${profit:+.2f}"
            )
            break
