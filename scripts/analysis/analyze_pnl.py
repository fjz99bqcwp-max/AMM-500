#!/usr/bin/env python3
"""Analyze PnL from Hyperliquid fills."""
import json
import urllib.request
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"


def fetch_fills():
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "userFills", "user": WALLET, "aggregateByTime": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_account():
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "clearinghouseState", "user": WALLET}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def analyze():
    print("=" * 60)
    print("HYPERLIQUID PnL ANALYSIS")
    print("=" * 60)

    # Fetch account state
    account = fetch_account()
    account_value = float(account["marginSummary"]["accountValue"])
    print(f"\nAccount Value: ${account_value:.2f}")

    # Get position info
    positions = account.get("assetPositions", [])
    for pos in positions:
        p = pos.get("position", {})
        coin = p.get("coin")
        size = float(p.get("szi", 0))
        entry = float(p.get("entryPx", 0))
        upnl = float(p.get("unrealizedPnl", 0))
        funding = p.get("cumFunding", {})
        all_time_funding = float(funding.get("allTime", 0))
        print(f"\nPosition: {coin}")
        print(f"  Size: {size}")
        print(f"  Entry: ${entry:.2f}")
        print(f"  Unrealized PnL: ${upnl:.2f}")
        print(f"  Cumulative Funding (all-time): ${all_time_funding:.6f}")

    # Fetch fills
    print("\nFetching fills...")
    fills = fetch_fills()
    print(f"Total fills: {len(fills)}")

    # Analyze different time periods
    now_ms = fills[0]["time"] if fills else 0
    periods = {
        "1h": 3600000,
        "4h": 14400000,
        "24h": 86400000,
        "7d": 604800000,
    }

    for period_name, period_ms in periods.items():
        cutoff = now_ms - period_ms
        period_fills = [f for f in fills if f["time"] >= cutoff]

        if not period_fills:
            continue

        closed_pnl = sum(float(f.get("closedPnl", 0)) for f in period_fills)
        fees = sum(float(f.get("fee", 0)) for f in period_fills)
        volume = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in period_fills)

        # Maker vs taker
        makers = [f for f in period_fills if not f.get("crossed", False)]
        takers = [f for f in period_fills if f.get("crossed", False)]
        maker_vol = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in makers)
        taker_vol = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in takers)

        # Maker rebates (0.003% = 3 bps)
        maker_rebate = maker_vol * 0.00003

        # Net PnL
        net_pnl = closed_pnl - fees + maker_rebate

        buys = len([f for f in period_fills if f["side"] == "B"])
        sells = len([f for f in period_fills if f["side"] == "A"])

        print(f"\n--- {period_name.upper()} ---")
        print(f"  Fills: {len(period_fills)} (buys: {buys}, sells: {sells})")
        print(f"  Volume: ${volume:,.2f}")
        print(f"  Maker: {len(makers)} fills (${maker_vol:,.2f})")
        print(f"  Taker: {len(takers)} fills (${taker_vol:,.2f})")
        print(f"  Closed PnL: ${closed_pnl:.4f}")
        print(f"  Fees: ${fees:.4f}")
        print(f"  Maker rebate: ${maker_rebate:.4f}")
        print(f"  NET PnL: ${net_pnl:.4f}")

        # Annualized
        if period_name == "24h" and volume > 0:
            daily_return = net_pnl / account_value
            ann_return = daily_return * 365
            print(f"  Daily return: {daily_return*100:.3f}%")
            print(f"  Annualized: {ann_return*100:.1f}%")

    # Analyze adverse selection
    print("\n" + "=" * 60)
    print("ADVERSE SELECTION ANALYSIS")
    print("=" * 60)

    day_fills = [f for f in fills if f["time"] >= now_ms - 86400000]

    # Group by direction
    opens = [f for f in day_fills if "Open" in f.get("dir", "")]
    closes = [f for f in day_fills if "Close" in f.get("dir", "") or ">" in f.get("dir", "")]

    print(f"\nOpening trades: {len(opens)}")
    print(f"Closing trades: {len(closes)}")

    # Check for negative closed PnL trades
    negative_pnl = [f for f in day_fills if float(f.get("closedPnl", 0)) < 0]
    positive_pnl = [f for f in day_fills if float(f.get("closedPnl", 0)) > 0]

    total_neg = sum(float(f.get("closedPnl", 0)) for f in negative_pnl)
    total_pos = sum(float(f.get("closedPnl", 0)) for f in positive_pnl)

    print(f"\nTrades with positive PnL: {len(positive_pnl)} (${total_pos:.4f})")
    print(f"Trades with negative PnL: {len(negative_pnl)} (${total_neg:.4f})")

    if negative_pnl:
        print("\nSample negative trades:")
        for f in negative_pnl[:5]:
            print(f"  {f['dir']}: {f['side']} {f['sz']} @ ${f['px']} = ${f['closedPnl']}")


def analyze_adverse_selection():
    """Deep analysis of adverse selection patterns."""
    print("\n" + "=" * 60)
    print("TRADE SEQUENCE ANALYSIS")
    print("=" * 60)

    fills = fetch_fills()
    now_ms = fills[0]["time"]
    day_fills = [f for f in fills if f["time"] >= now_ms - 86400000]

    # Look at consecutive buy-sell pairs
    buys = []
    sells = []

    for f in reversed(day_fills):  # oldest first
        if f["side"] == "B":
            buys.append(float(f["px"]))
        else:
            sells.append(float(f["px"]))

    # Compare average buy vs sell price
    avg_buy = sum(buys) / len(buys) if buys else 0
    avg_sell = sum(sells) / len(sells) if sells else 0

    print(f"\nAverage BUY price: ${avg_buy:.2f}")
    print(f"Average SELL price: ${avg_sell:.2f}")
    print(f"Spread captured: ${avg_sell - avg_buy:.2f}")

    # The problem: If sells < buys, we're losing on the round-trip
    if avg_sell < avg_buy:
        print("\n🔴 PROBLEM: Selling LOWER than buying = ADVERSE SELECTION")
        print(f"   Expected spread: ~$1-2 (2 bps at $90k)")
        print(f"   Actual spread: ${avg_sell - avg_buy:.2f}")
        print(f"   Loss per round-trip: ${avg_buy - avg_sell:.2f}")
    else:
        print("\n✅ OK: Selling higher than buying")

    # Check how quickly the bot is getting filled on opposite sides
    print("\n=== TIME BETWEEN FILLS ===")

    last_buy_time = None
    last_sell_time = None
    quick_reversals = 0

    for f in day_fills[:100]:  # Last 100 fills
        if f["side"] == "B":
            if last_sell_time:
                gap_ms = abs(last_sell_time - f["time"])
                if gap_ms < 10000:  # Within 10 seconds
                    quick_reversals += 1
            last_buy_time = f["time"]
        else:
            if last_buy_time:
                gap_ms = abs(last_buy_time - f["time"])
                if gap_ms < 10000:  # Within 10 seconds
                    quick_reversals += 1
            last_sell_time = f["time"]

    print(f"Quick reversals (< 10s): {quick_reversals} in last 100 fills")

    if quick_reversals > 20:
        print("🔴 Too many quick reversals = getting picked off")

    # Analyze price movement around fills
    print("\n=== ROOT CAUSE HYPOTHESIS ===")
    print(
        """
The bot is experiencing ADVERSE SELECTION because:

1. OPTIMIZATION #7 places quotes AT the BBO (best bid/ask)
   - This makes us first in line to get filled
   - BUT we get filled when informed traders cross the spread
   
2. When an informed trader sees a directional move:
   - They HIT our bid (we buy) right before price drops
   - They LIFT our ask (we sell) right before price rises
   
3. We're the "dumb liquidity" that smart traders trade against

SOLUTIONS:
A. WIDEN SPREADS: Quote INSIDE the BBO, not AT it
   - bid = best_bid - 1 (instead of = best_bid)
   - ask = best_ask + 1 (instead of = best_ask)
   
B. REDUCE QUOTE FREQUENCY: Don't refresh every 30 seconds
   - Stale quotes = less adverse selection
   
C. ADD FLOW TOXICITY DETECTION:
   - Skip quoting when large imbalance in order flow
   - Widen spreads when volatility spikes
"""
    )


if __name__ == "__main__":
    analyze()
    analyze_adverse_selection()
