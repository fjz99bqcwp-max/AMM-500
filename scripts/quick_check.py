#!/usr/bin/env python3
"""Quick order analysis."""
from hyperliquid.info import Info
from dotenv import load_dotenv
import os

load_dotenv("config/.env")
wallet = os.getenv("WALLET_ADDRESS")
info = Info(base_url="https://api.hyperliquid.xyz", skip_ws=True)

# Get orders
orders = info.open_orders(wallet)
bids = sorted([float(o["limitPx"]) for o in orders if o.get("side") == "B"], reverse=True)
asks = sorted([float(o["limitPx"]) for o in orders if o.get("side") == "A"])

# Get L2
l2 = info.l2_snapshot("BTC")
levels = l2.get("levels", []) if isinstance(l2, dict) else l2

print("=" * 50)
print("ORDER ANALYSIS")
print("=" * 50)

if levels and len(levels) >= 2:
    market_bids = levels[0]
    market_asks = levels[1]
    bb = float(market_bids[0]["px"])
    ba = float(market_asks[0]["px"])
    print(f"Market BBO: ${bb:.2f} / ${ba:.2f} (spread: ${ba-bb:.2f})")
else:
    bb = ba = 0
    print("Could not get market L2")

print(f"Total orders: {len(orders)} ({len(bids)} bids, {len(asks)} asks)")
if bids:
    print(f"Our best bid: ${bids[0]:.2f}")
if asks:
    print(f"Our best ask: ${asks[0]:.2f}")
if bids and asks:
    print(f"Our spread: ${asks[0]-bids[0]:.2f}")
if bids and asks and bb > 0:
    print(f"Gap from BBO: bid=${bb-bids[0]:.2f}, ask=${asks[0]-ba:.2f}")

# Get position & PnL
user_state = info.user_state(wallet)
if user_state:
    margin = user_state.get("marginSummary", {})
    print(f"\nEquity: ${float(margin.get('accountValue', 0)):,.2f}")

    positions = user_state.get("assetPositions", [])
    for pos in positions:
        pd = pos.get("position", {})
        if pd.get("coin") == "BTC":
            print(f"Position: {float(pd.get('szi', 0)):+.6f} BTC")
            print(f"Unrealized PnL: ${float(pd.get('unrealizedPnl', 0)):+.2f}")

# Get recent fills
fills = info.user_fills(wallet)
if fills:
    btc_fills = [f for f in fills if f.get("coin") == "BTC"][:5]
    print(f"\nLast {len(btc_fills)} fills:")
    for f in btc_fills:
        print(f"  {f.get('dir')} {f.get('sz')} @ ${float(f.get('px', 0)):,.2f}")

print("=" * 50)
