#!/usr/bin/env python3
"""
Calculate TRUE market making performance: spread captured
This measures buy price vs sell price, ignoring legacy position PnL
"""

import requests
from datetime import datetime, timedelta
from collections import defaultdict

wallet = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
url = "https://api.hyperliquid.xyz/info"

response = requests.post(url, json={"type": "userFills", "user": wallet})
fills = response.json()

# Look at last 24 hours
cutoff = datetime.now() - timedelta(hours=24)
cutoff_ts = cutoff.timestamp() * 1000

fills_24h = [f for f in fills if f["time"] >= cutoff_ts and f["coin"] == "BTC"]
fills_24h.reverse()  # Oldest first for FIFO matching

print(f"=== SPREAD CAPTURE ANALYSIS (Last 24h) ===")
print(f"Total fills: {len(fills_24h)}")
print()

# Track buys and sells for FIFO matching
buys = []  # [(price, size, timestamp)]
sells = []  # [(price, size, timestamp)]

total_buy_value = 0
total_sell_value = 0
total_buy_size = 0
total_sell_size = 0

for f in fills_24h:
    px = float(f["px"])
    sz = float(f["sz"])

    if f["side"] == "B":
        buys.append((px, sz, f["time"]))
        total_buy_value += px * sz
        total_buy_size += sz
    else:
        sells.append((px, sz, f["time"]))
        total_sell_value += px * sz
        total_sell_size += sz

# Match trades FIFO-style to calculate spread captured
matched_trades = []
buy_idx = 0
sell_idx = 0
matched_size = 0

while buy_idx < len(buys) and sell_idx < len(sells):
    buy_px, buy_sz, buy_ts = buys[buy_idx]
    sell_px, sell_sz, sell_ts = sells[sell_idx]

    # Match the smaller of the two sizes
    match_sz = min(buy_sz, sell_sz)
    spread_captured = (sell_px - buy_px) * match_sz

    matched_trades.append(
        {
            "buy_px": buy_px,
            "sell_px": sell_px,
            "size": match_sz,
            "spread_captured": spread_captured,
            "bps": (sell_px - buy_px) / buy_px * 10000,
        }
    )

    matched_size += match_sz

    # Reduce the sizes
    remaining_buy = buy_sz - match_sz
    remaining_sell = sell_sz - match_sz

    if remaining_buy <= 0.000001:
        buy_idx += 1
    else:
        buys[buy_idx] = (buy_px, remaining_buy, buy_ts)

    if remaining_sell <= 0.000001:
        sell_idx += 1
    else:
        sells[sell_idx] = (sell_px, remaining_sell, sell_ts)

# Calculate statistics
print(f"=== MATCHED ROUND-TRIPS ===")
print(f"Total matched trades: {len(matched_trades)}")
print(f"Matched size: {matched_size:.6f} BTC")
print()

if matched_trades:
    total_spread = sum(t["spread_captured"] for t in matched_trades)
    avg_bps = sum(t["bps"] for t in matched_trades) / len(matched_trades)

    positive_spreads = [t for t in matched_trades if t["spread_captured"] > 0]
    negative_spreads = [t for t in matched_trades if t["spread_captured"] < 0]

    print(f"Total spread captured: ${total_spread:.4f}")
    print(f"Average spread: {avg_bps:.2f} bps")
    print(
        f"Positive round-trips: {len(positive_spreads)} (${sum(t['spread_captured'] for t in positive_spreads):.4f})"
    )
    print(
        f"Negative round-trips: {len(negative_spreads)} (${sum(t['spread_captured'] for t in negative_spreads):.4f})"
    )
    print()

    print("=== SAMPLE ROUND-TRIPS (last 10) ===")
    for t in matched_trades[-10:]:
        status = "✅" if t["spread_captured"] > 0 else "❌"
        print(
            f"BUY @${t['buy_px']:.0f} → SELL @${t['sell_px']:.0f} = ${t['spread_captured']:+.4f} ({t['bps']:+.1f} bps) {status}"
        )

# Summary stats
print()
print("=== VOLUME-WEIGHTED PRICES ===")
avg_buy = 0.0
avg_sell = 0.0
if total_buy_size > 0:
    avg_buy = total_buy_value / total_buy_size
    print(f"Average BUY price:  ${avg_buy:.2f}")
if total_sell_size > 0:
    avg_sell = total_sell_value / total_sell_size
    print(f"Average SELL price: ${avg_sell:.2f}")
if total_buy_size > 0 and total_sell_size > 0:
    spread = avg_sell - avg_buy
    spread_bps = spread / avg_buy * 10000
    print(f"Spread captured: ${spread:.2f} ({spread_bps:.2f} bps)")
    print()
    if spread < 0:
        print("⚠️  ADVERSE SELECTION: Selling LOWER than buying!")
        print("This is the fundamental problem to fix.")
    else:
        print("✅ Positive spread - capturing the bid-ask correctly")

# Fee analysis
total_fees = sum(float(f.get("fee", 0)) for f in fills_24h)
print()
print(f"=== FEES ===")
print(f"Total fees paid: ${total_fees:.4f}")
print(f"Number of trades: {len(fills_24h)}")
if len(fills_24h) > 0:
    print(f"Average fee per trade: ${total_fees/len(fills_24h):.4f}")

# Net profitability
if total_buy_size > 0 and total_sell_size > 0:
    min_size = min(total_buy_size, total_sell_size)
    gross_spread = (avg_sell - avg_buy) * min_size
    net_after_fees = gross_spread - total_fees
    print()
    print(f"=== NET PROFITABILITY ===")
    print(f"Gross spread on matched: ${gross_spread:.4f}")
    print(f"Total fees: ${total_fees:.4f}")
    print(f"Net P&L: ${net_after_fees:.4f}")
