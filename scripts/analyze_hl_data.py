#!/usr/bin/env python3
"""
Analyze Hyperliquid data directly and compare with bot logs.
"""

import requests
import json
from datetime import datetime, timedelta

# Configuration
URL = "https://api.hyperliquid.xyz/info"
WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"


def fetch_account_state():
    """Fetch current account state from Hyperliquid."""
    resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET})
    return resp.json()


def fetch_fills():
    """Fetch user fills from Hyperliquid."""
    resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
    return resp.json()


def analyze():
    """Main analysis function."""
    print("=" * 60)
    print("📊 HYPERLIQUID REAL DATA ANALYSIS")
    print("=" * 60)

    # Fetch data
    state = fetch_account_state()
    fills = fetch_fills()

    # Account State
    print("\n💰 ACCOUNT STATE:")
    account_value = float(state["marginSummary"]["accountValue"])
    margin_used = float(state["marginSummary"]["totalMarginUsed"])
    print(f"  Account Value: ${account_value:,.2f}")
    print(f"  Total Margin: ${margin_used:,.2f}")
    print(f"  PnL from $1000: ${account_value - 1000:+.2f} ({(account_value/1000-1)*100:+.2f}%)")

    # Position
    if state.get("assetPositions"):
        for pos in state["assetPositions"]:
            if pos["position"]["coin"] == "BTC":
                p = pos["position"]
                size = float(p["szi"])
                entry = float(p["entryPx"])
                pnl = float(p["unrealizedPnl"])
                lev = p.get("leverage", {}).get("value", "N/A")

                print(f"\n📈 BTC POSITION:")
                print(f"  Size: {size:+.6f} BTC ({'LONG' if size > 0 else 'SHORT'})")
                print(f"  Entry: ${entry:,.2f}")
                print(f"  Unrealized PnL: ${pnl:+.2f}")
                print(f"  Leverage: {lev}x")

    # Recent fills analysis
    print(f"\n📋 FILLS ANALYSIS:")
    print(f"  Total fills in history: {len(fills)}")

    if not fills:
        print("  No fills found!")
        return

    # Last 50 fills
    recent = fills[:50] if len(fills) >= 50 else fills
    buys = [f for f in recent if f["side"] == "B"]
    sells = [f for f in recent if f["side"] == "A"]

    print(f"  Last {len(recent)} fills: {len(buys)} buys, {len(sells)} sells")

    # Calculate weighted spread
    if buys and sells:
        buy_notional = sum(float(f["px"]) * float(f["sz"]) for f in buys)
        buy_volume = sum(float(f["sz"]) for f in buys)
        sell_notional = sum(float(f["px"]) * float(f["sz"]) for f in sells)
        sell_volume = sum(float(f["sz"]) for f in sells)

        avg_buy = buy_notional / buy_volume if buy_volume > 0 else 0
        avg_sell = sell_notional / sell_volume if sell_volume > 0 else 0
        mid = (avg_buy + avg_sell) / 2 if avg_buy and avg_sell else 0
        spread_bps = (avg_sell - avg_buy) / mid * 10000 if mid > 0 else 0

        print(f"  Avg Buy: ${avg_buy:,.2f}")
        print(f"  Avg Sell: ${avg_sell:,.2f}")
        print(f"  Spread: {spread_bps:+.2f} bps", end="")
        if spread_bps > 0:
            print(" ✅ PROFITABLE")
        else:
            print(" ❌ LOSING")

    # Time analysis
    times = [datetime.fromtimestamp(int(f["time"]) / 1000) for f in recent]
    oldest = min(times)
    newest = max(times)
    span = (newest - oldest).total_seconds() / 3600
    fill_rate = len(recent) / max(span, 0.1)

    print(f"  Time span: {span:.1f} hours")
    print(f"  Fill rate: {fill_rate:.1f} fills/hour")

    # Last 10 fills detail
    print(f"\n📝 LAST 10 FILLS:")
    for f in fills[:10]:
        side = "BUY " if f["side"] == "B" else "SELL"
        ts = datetime.fromtimestamp(int(f["time"]) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        size = float(f["sz"])
        price = float(f["px"])
        notional = size * price
        print(f"  {ts} | {side} | {size:8.5f} BTC @ ${price:,.2f} (${notional:.2f})")

    # Hourly performance breakdown
    print(f"\n📊 HOURLY PERFORMANCE (last 24h):")
    now = datetime.now()
    for hours_ago in range(0, 24, 4):
        start = now - timedelta(hours=hours_ago + 4)
        end = now - timedelta(hours=hours_ago)

        period_fills = [
            f for f in fills if start <= datetime.fromtimestamp(int(f["time"]) / 1000) < end
        ]
        if not period_fills:
            continue

        period_buys = [f for f in period_fills if f["side"] == "B"]
        period_sells = [f for f in period_fills if f["side"] == "A"]

        if period_buys and period_sells:
            buy_not = sum(float(f["px"]) * float(f["sz"]) for f in period_buys)
            buy_vol = sum(float(f["sz"]) for f in period_buys)
            sell_not = sum(float(f["px"]) * float(f["sz"]) for f in period_sells)
            sell_vol = sum(float(f["sz"]) for f in period_sells)

            if buy_vol > 0 and sell_vol > 0:
                avg_b = buy_not / buy_vol
                avg_s = sell_not / sell_vol
                m = (avg_b + avg_s) / 2
                spr = (avg_s - avg_b) / m * 10000
                status = "✅" if spr > 0 else "❌"
                print(
                    f"  {hours_ago}-{hours_ago+4}h ago: {len(period_fills):3} fills, spread: {spr:+6.2f} bps {status}"
                )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze()
