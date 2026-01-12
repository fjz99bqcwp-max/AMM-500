#!/usr/bin/env python3
"""Check REAL Hyperliquid data to verify actual performance."""

import requests
import json
from datetime import datetime, timedelta

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"

# Fetch fills
resp = requests.post("https://api.hyperliquid.xyz/info", json={"type": "userFills", "user": WALLET})
fills = resp.json()

# Get recent fills (last 2 hours)
now_ms = int(datetime.now().timestamp() * 1000)
two_hours_ago = now_ms - (2 * 60 * 60 * 1000)
recent_fills = [f for f in fills if int(f["time"]) > two_hours_ago]

print("=" * 80)
print("REAL HYPERLIQUID DATA ANALYSIS")
print("=" * 80)
print(f"Total fills ever: {len(fills)}")
print(f"Fills in last 2 hours: {len(recent_fills)}")
print()

if len(recent_fills) == 0:
    print("⚠️  NO RECENT ACTIVITY IN LAST 2 HOURS!")
    print("   Bot may not be trading actively")
    print()

    # Check last 24 hours
    day_ago = now_ms - (24 * 60 * 60 * 1000)
    day_fills = [f for f in fills if int(f["time"]) > day_ago]
    print(f"Fills in last 24 hours: {len(day_fills)}")

    if len(day_fills) > 0:
        recent_fills = day_fills[-20:]  # Last 20 fills
        print("Using last 20 fills for analysis...")
        print()

if len(recent_fills) > 0:
    buys = [f for f in recent_fills if f["side"] == "B"]
    sells = [f for f in recent_fills if f["side"] == "A"]

    print(f"Buy fills: {len(buys)}")
    print(f"Sell fills: {len(sells)}")
    print()

    if buys and sells:
        # Weighted average calculation
        buy_notional = sum(float(f["px"]) * float(f["sz"]) for f in buys)
        buy_volume = sum(float(f["sz"]) for f in buys)
        avg_buy = buy_notional / buy_volume

        sell_notional = sum(float(f["px"]) * float(f["sz"]) for f in sells)
        sell_volume = sum(float(f["sz"]) for f in sells)
        avg_sell = sell_notional / sell_volume

        spread_bps = ((avg_sell - avg_buy) / avg_buy) * 10000
        total_fees = sum(float(f["fee"]) for f in recent_fills)

        # Estimate PnL
        min_volume = min(buy_volume, sell_volume)
        gross_pnl = (avg_sell - avg_buy) * min_volume
        net_pnl = gross_pnl - total_fees

        print(f"Average Buy:  ${avg_buy:.2f}")
        print(f"Average Sell: ${avg_sell:.2f}")
        print()
        print(f"📊 REAL SPREAD: {spread_bps:+.2f} bps")
        print(f"💰 Gross PnL:   ${gross_pnl:+.4f}")
        print(f"💸 Fees Paid:   ${total_fees:.4f}")
        print(f"💵 Net PnL:     ${net_pnl:+.4f}")
        print()

        if spread_bps < 0:
            print("❌ LOSING TRADES - Adverse selection detected!")
        elif spread_bps < 2:
            print("⚠️  LOW SPREAD - Barely profitable")
        elif spread_bps < 5:
            print("✅ POSITIVE - Marginal profit")
        else:
            print("✅ PROFITABLE - Good performance")
    else:
        print("⚠️  Not enough buy/sell pairs to calculate spread")

    # Show last 10 fills
    print()
    print("Last 10 fills:")
    for f in recent_fills[-10:]:
        ts = datetime.fromtimestamp(int(f["time"]) / 1000)
        print(
            f"  {ts.strftime('%H:%M:%S')} | {f['side']:1s} {f['sz']:>8s} @ ${float(f['px']):.2f} | fee: ${f['fee']} | pnl: ${f.get('closedPnl', '0')}"
        )

print()
print("=" * 80)
