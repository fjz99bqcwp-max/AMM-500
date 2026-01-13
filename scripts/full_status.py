#!/usr/bin/env python3
"""Full analysis of Hyperliquid account state and performance."""

import requests
import time
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
URL = "https://api.hyperliquid.xyz/info"


def main():
    # Get account state
    resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET})
    state = resp.json()

    print("=" * 60)
    print("HYPERLIQUID ACCOUNT STATE")
    print("=" * 60)

    if "marginSummary" in state:
        ms = state["marginSummary"]
        print(f"Account Value: ${float(ms.get('accountValue', 0)):.2f}")
        print(f"Total Margin Used: ${float(ms.get('totalMarginUsed', 0)):.2f}")
        print(f"Withdrawable: ${float(ms.get('withdrawable', 0)):.2f}")

    # Position
    print("\n=== POSITION ===")
    if "assetPositions" in state:
        for pos in state["assetPositions"]:
            p = pos.get("position", {})
            coin = p.get("coin", "N/A")
            if coin == "BTC":
                size = float(p.get("szi", 0))
                entry = float(p.get("entryPx", 0))
                upnl = float(p.get("unrealizedPnl", 0))
                print(f"  {coin}: {size:.6f} @ ${entry:.2f} | uPnL: ${upnl:.4f}")

    # Open orders
    resp = requests.post(URL, json={"type": "openOrders", "user": WALLET})
    orders = resp.json()
    print(f"\n=== OPEN ORDERS: {len(orders)} ===")
    buy_orders = [o for o in orders if o.get("side") == "B"]
    sell_orders = [o for o in orders if o.get("side") == "A"]
    print(f"  Bids: {len(buy_orders)}, Asks: {len(sell_orders)}")

    # Recent fills
    resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
    fills = resp.json()

    # Last 30 minutes
    now = int(time.time() * 1000)
    thirty_min = now - (30 * 60 * 1000)
    recent = [f for f in fills if f.get("time", 0) > thirty_min]

    print(f"\n=== FILLS (Last 30 min): {len(recent)} ===")
    buys = [f for f in recent if f.get("side") == "B"]
    sells = [f for f in recent if f.get("side") == "A"]
    print(f"  Buys: {len(buys)}, Sells: {len(sells)}")

    avg_buy = 0.0
    avg_sell = 0.0
    if len(buys) > 0:
        avg_buy = sum(float(f.get("px", 0)) for f in buys) / len(buys)
        print(f"  Avg Buy: ${avg_buy:.2f}")

    if len(sells) > 0:
        avg_sell = sum(float(f.get("px", 0)) for f in sells) / len(sells)
        print(f"  Avg Sell: ${avg_sell:.2f}")

    if len(buys) > 0 and len(sells) > 0:
        spread = avg_sell - avg_buy
        mid = (avg_buy + avg_sell) / 2
        spread_bps = spread / mid * 10000
        print(f"  Spread Captured: ${spread:.2f} ({spread_bps:.2f} bps)")

        total_fees = sum(float(f.get("fee", 0)) for f in recent)
        print(f"  Total Fees: ${total_fees:.4f}")

        if spread_bps > 0:
            print("\n  ✅ POSITIVE SPREAD - OPT#12 WORKING!")
        else:
            print("\n  ❌ NEGATIVE SPREAD - ADVERSE SELECTION!")

    # Last 24h stats
    day_ago = now - (24 * 60 * 60 * 1000)
    day_fills = [f for f in fills if f.get("time", 0) > day_ago]
    day_buys = [f for f in day_fills if f.get("side") == "B"]
    day_sells = [f for f in day_fills if f.get("side") == "A"]
    print(f"\n=== 24H STATS ===")
    print(f"  Total Fills: {len(day_fills)}")
    print(f"  Buys: {len(day_buys)}, Sells: {len(day_sells)}")
    if len(day_buys) > 0 and len(day_sells) > 0:
        avg_buy_24h = sum(float(f.get("px", 0)) for f in day_buys) / len(day_buys)
        avg_sell_24h = sum(float(f.get("px", 0)) for f in day_sells) / len(day_sells)
        spread_24h = avg_sell_24h - avg_buy_24h
        print(f"  24h Avg Buy: ${avg_buy_24h:.2f}")
        print(f"  24h Avg Sell: ${avg_sell_24h:.2f}")
        print(f"  24h Spread: ${spread_24h:.2f} ({spread_24h/avg_buy_24h*10000:.2f} bps)")

    # Check bot status
    print("\n=== BOT STATUS ===")
    import subprocess

    result = subprocess.run(["pgrep", "-f", "amm-500.py"], capture_output=True, text=True)
    if result.returncode == 0:
        pids = result.stdout.strip().split("\n")
        print(f"  Bot running: YES (PIDs: {', '.join(pids)})")
    else:
        print("  Bot running: NO")


if __name__ == "__main__":
    main()
