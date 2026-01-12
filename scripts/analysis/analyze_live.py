#!/usr/bin/env python3
"""
Live Trading Analysis Script
Queries Hyperliquid SDK for real-time performance metrics and diagnostics.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperliquid.info import Info
from hyperliquid.utils import constants


def load_config():
    """Load config from .env file."""
    from dotenv import load_dotenv
    import os

    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)

    return {
        "wallet_address": os.getenv("WALLET_ADDRESS"),
        "testnet": os.getenv("TESTNET", "False").lower() == "true",
    }


def get_base_url(testnet: bool) -> str:
    """Get API URL based on network."""
    return "https://api.hyperliquid-testnet.xyz" if testnet else "https://api.hyperliquid.xyz"


def analyze_trading_performance():
    """Analyze trading performance from Hyperliquid API."""
    config = load_config()
    base_url = get_base_url(config["testnet"])
    wallet = config["wallet_address"]

    print("=" * 70)
    print("HYPERLIQUID LIVE TRADING ANALYSIS")
    print(f"Wallet: {wallet}")
    print(f"Network: {'Testnet' if config['testnet'] else 'Mainnet'}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    info = Info(base_url=base_url, skip_ws=True)

    # 1. Account State
    print("\n📊 ACCOUNT STATE")
    print("-" * 40)
    try:
        user_state = info.user_state(wallet)
        if user_state:
            margin_summary = user_state.get("marginSummary", {})
            account_value = float(margin_summary.get("accountValue", 0))
            total_margin = float(margin_summary.get("totalMarginUsed", 0))

            print(f"  Account Value: ${account_value:,.2f}")
            print(f"  Margin Used: ${total_margin:,.2f}")
            print(
                f"  Margin Ratio: {total_margin/account_value*100:.2f}%"
                if account_value > 0
                else "  Margin Ratio: N/A"
            )

            # Withdrawable balance
            withdrawable = user_state.get("withdrawable", 0)
            print(f"  Withdrawable: ${float(withdrawable):,.2f}")
    except Exception as e:
        print(f"  Error getting account state: {e}")

    # 2. Open Positions
    print("\n📈 OPEN POSITIONS")
    print("-" * 40)
    try:
        if user_state and "assetPositions" in user_state:
            positions = user_state["assetPositions"]
            if positions:
                for pos in positions:
                    pos_data = pos.get("position", {})
                    coin = pos_data.get("coin", "?")
                    size = float(pos_data.get("szi", 0))
                    entry = float(pos_data.get("entryPx", 0))
                    unrealized = float(pos_data.get("unrealizedPnl", 0))
                    leverage = pos_data.get("leverage", {})
                    lev_value = leverage.get("value", "?")
                    liq_px = pos_data.get("liquidationPx")

                    print(f"  {coin}:")
                    print(f"    Size: {size:+.6f}")
                    print(f"    Entry: ${entry:,.2f}")
                    print(f"    Unrealized PnL: ${unrealized:+,.2f}")
                    print(f"    Leverage: {lev_value}x")
                    if liq_px:
                        print(f"    Liquidation: ${float(liq_px):,.2f}")
            else:
                print("  No open positions")
    except Exception as e:
        print(f"  Error getting positions: {e}")

    # 3. Open Orders
    print("\n📋 OPEN ORDERS")
    print("-" * 40)
    try:
        open_orders = info.open_orders(wallet)
        if open_orders:
            bid_count = sum(1 for o in open_orders if o.get("side") == "B")
            ask_count = sum(1 for o in open_orders if o.get("side") == "A")
            print(f"  Total: {len(open_orders)} orders (Bids: {bid_count}, Asks: {ask_count})")

            # Show order book spread
            bids = [float(o["limitPx"]) for o in open_orders if o.get("side") == "B"]
            asks = [float(o["limitPx"]) for o in open_orders if o.get("side") == "A"]

            if bids and asks:
                best_bid = max(bids)
                best_ask = min(asks)
                our_spread = best_ask - best_bid
                print(f"  Our Best Bid: ${best_bid:,.2f}")
                print(f"  Our Best Ask: ${best_ask:,.2f}")
                print(f"  Our Spread: ${our_spread:,.2f}")
        else:
            print("  No open orders")
    except Exception as e:
        print(f"  Error getting open orders: {e}")

    # 4. Get BTC market data for comparison
    print("\n💹 BTC MARKET DATA")
    print("-" * 40)
    try:
        meta = info.meta()
        all_mids = info.all_mids()

        btc_mid = float(all_mids.get("BTC", 0))
        print(f"  BTC Mid Price: ${btc_mid:,.2f}")

        # Get BTC asset info
        for asset in meta.get("universe", []):
            if asset.get("name") == "BTC":
                print(f"  szDecimals: {asset.get('szDecimals')}")
                break

        # L2 orderbook for spread analysis
        l2 = info.l2_snapshot("BTC")
        if l2 and len(l2) >= 2:
            bids = l2[0]  # [[price, size], ...]
            asks = l2[1]
            if bids and asks:
                bb = float(bids[0]["px"])
                ba = float(asks[0]["px"])
                market_spread = ba - bb
                market_spread_bps = (market_spread / btc_mid) * 10000
                print(f"  Market Best Bid: ${bb:,.2f}")
                print(f"  Market Best Ask: ${ba:,.2f}")
                print(f"  Market Spread: ${market_spread:.2f} ({market_spread_bps:.2f} bps)")

                # Compare our orders to market
                if open_orders:
                    our_bids = [float(o["limitPx"]) for o in open_orders if o.get("side") == "B"]
                    our_asks = [float(o["limitPx"]) for o in open_orders if o.get("side") == "A"]

                    if our_bids:
                        our_best_bid = max(our_bids)
                        bid_distance = bb - our_best_bid
                        print(
                            f"\n  ⚠️  Our bid ${our_best_bid:,.2f} is ${bid_distance:.2f} below market"
                        )

                    if our_asks:
                        our_best_ask = min(our_asks)
                        ask_distance = our_best_ask - ba
                        print(
                            f"  ⚠️  Our ask ${our_best_ask:,.2f} is ${ask_distance:.2f} above market"
                        )
    except Exception as e:
        print(f"  Error getting market data: {e}")

    # 5. Recent Fills
    print("\n🔄 RECENT FILLS (Last 24h)")
    print("-" * 40)
    try:
        fills = info.user_fills(wallet)
        if fills:
            # Filter to last 24h
            now = datetime.now()
            recent_fills = [f for f in fills if f.get("coin") == "BTC"][:20]  # Last 20 BTC fills

            if recent_fills:
                total_volume = sum(float(f.get("sz", 0)) for f in recent_fills)
                total_pnl = sum(float(f.get("closedPnl", 0)) for f in recent_fills)

                print(f"  Recent Fills: {len(recent_fills)}")
                print(f"  Total Volume: {total_volume:.6f} BTC")
                print(f"  Realized PnL: ${total_pnl:+,.2f}")

                # Show last few fills
                print("\n  Last 5 fills:")
                for f in recent_fills[:5]:
                    side = "BUY" if f.get("dir") == "Buy" else "SELL"
                    px = float(f.get("px", 0))
                    sz = float(f.get("sz", 0))
                    ts = f.get("time", 0)
                    pnl = float(f.get("closedPnl", 0))
                    dt = datetime.fromtimestamp(ts / 1000) if ts else "?"
                    print(f"    {dt}: {side} {sz:.6f} @ ${px:,.2f} | PnL: ${pnl:+.2f}")
            else:
                print("  No recent BTC fills")
        else:
            print("  No fills found")
    except Exception as e:
        print(f"  Error getting fills: {e}")

    # 6. Funding Rate
    print("\n💰 FUNDING RATE")
    print("-" * 40)
    try:
        funding_history = info.funding_history(
            "BTC", start_time=int((datetime.now() - timedelta(hours=8)).timestamp() * 1000)
        )
        if funding_history:
            latest = funding_history[-1]
            rate = float(latest.get("fundingRate", 0))
            print(f"  Current Rate: {rate*100:.4f}% (hourly)")
            print(f"  Annualized: {rate*100*24*365:.2f}%")

            if rate > 0:
                print("  Direction: Longs pay shorts (bullish sentiment)")
            else:
                print("  Direction: Shorts pay longs (bearish sentiment)")
    except Exception as e:
        print(f"  Error getting funding: {e}")

    # 7. Recommendations
    print("\n🎯 ANALYSIS & RECOMMENDATIONS")
    print("-" * 40)

    try:
        # Calculate our spread vs market spread
        l2 = info.l2_snapshot("BTC")
        if l2 and len(l2) >= 2 and open_orders:
            bb = float(l2[0][0]["px"])
            ba = float(l2[1][0]["px"])
            market_spread = ba - bb

            our_bids = [float(o["limitPx"]) for o in open_orders if o.get("side") == "B"]
            our_asks = [float(o["limitPx"]) for o in open_orders if o.get("side") == "A"]

            if our_bids and our_asks:
                our_best_bid = max(our_bids)
                our_best_ask = min(our_asks)
                our_spread = our_best_ask - our_best_bid

                # Issue detection
                issues = []

                # Check if we're too far from BBO
                bid_distance = bb - our_best_bid
                ask_distance = our_best_ask - ba

                if bid_distance > 10:
                    issues.append(f"❌ Bid too far from BBO (${bid_distance:.0f} gap)")
                if ask_distance > 10:
                    issues.append(f"❌ Ask too far from BBO (${ask_distance:.0f} gap)")

                if our_spread > market_spread * 3:
                    issues.append(
                        f"❌ Our spread (${our_spread:.0f}) is 3x wider than market (${market_spread:.0f})"
                    )

                if issues:
                    print("  ISSUES DETECTED:")
                    for issue in issues:
                        print(f"    {issue}")

                    print("\n  RECOMMENDED FIXES:")
                    print("    1. Reduce ALO_TICK_MARGIN from $50 to $2-5")
                    print("    2. Tighten spread calculation in low volatility")
                    print("    3. Use microprice for better queue position")
                else:
                    print("  ✅ No critical issues detected")
    except Exception as e:
        print(f"  Error in analysis: {e}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    return True


if __name__ == "__main__":
    analyze_trading_performance()
