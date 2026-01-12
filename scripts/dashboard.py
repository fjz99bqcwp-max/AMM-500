#!/usr/bin/env python3
"""
Real-time Dashboard - Quick status check

Shows current bot status, recent performance, and monitoring status
"""

import requests
import subprocess
import time
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
URL = "https://api.hyperliquid.xyz/info"


def check_bot_status():
    """Check if bot is running."""
    result = subprocess.run(["pgrep", "-f", "mmb-1.py"], capture_output=True, text=True)
    if result.stdout.strip():
        return f"✅ RUNNING (PID: {result.stdout.strip()})"
    return "❌ STOPPED"


def check_monitor_status():
    """Check if monitor is running."""
    result = subprocess.run(["pgrep", "-f", "mmb_continuous.py"], capture_output=True, text=True)
    if result.stdout.strip():
        return f"✅ RUNNING (PID: {result.stdout.strip()})"
    return "❌ STOPPED"


def get_account_value():
    """Get current account value."""
    try:
        resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=10)
        data = resp.json()
        margin_summary = data.get("marginSummary", {})
        return float(margin_summary.get("accountValue", 0))
    except:
        return None


def get_position():
    """Get current position."""
    try:
        resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=10)
        data = resp.json()
        positions = data.get("assetPositions", [])
        if positions:
            pos = positions[0].get("position", {})
            szi = float(pos.get("szi", 0))
            entry_px = float(pos.get("entryPx", 0)) if pos.get("entryPx") else 0
            return szi, entry_px
        return 0.0, 0.0
    except:
        return None, None


def get_recent_performance():
    """Get last 20 fills performance - REAL HYPERLIQUID DATA ONLY."""
    try:
        resp = requests.post(URL, json={"type": "userFills", "user": WALLET}, timeout=10)
        fills = resp.json()

        if not fills:
            return None

        # Use ONLY most recent 20 fills
        recent = fills[:20]
        buys = [f for f in recent if f.get("side") == "B"]
        sells = [f for f in recent if f.get("side") == "A"]

        if not buys or not sells:
            return None

        # WEIGHTED AVERAGE (correct calculation)
        buy_notional = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in buys)
        buy_volume = sum(float(f.get("sz", 0)) for f in buys)
        avg_buy = buy_notional / buy_volume if buy_volume > 0 else 0

        sell_notional = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in sells)
        sell_volume = sum(float(f.get("sz", 0)) for f in sells)
        avg_sell = sell_notional / sell_volume if sell_volume > 0 else 0

        spread_bps = ((avg_sell - avg_buy) / avg_buy) * 10000 if avg_buy > 0 else 0

        return {
            "buys": len(buys),
            "sells": len(sells),
            "spread_bps": spread_bps,
            "profitable": spread_bps > 0,
        }
    except Exception as e:
        print(f"Error fetching fills: {e}")
        return None


def main():
    print("=" * 80)
    print("📊 TRADING BOT DASHBOARD")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # System status
    print("🤖 SYSTEM STATUS")
    print("-" * 80)
    print(f"Bot:           {check_bot_status()}")
    print(f"Monitor:       {check_monitor_status()}")
    print()

    # Account status
    print("💰 ACCOUNT STATUS")
    print("-" * 80)
    account_value = get_account_value()
    if account_value:
        pnl = account_value - 1000
        pnl_pct = (pnl / 1000) * 100
        status = "✅" if pnl >= 0 else "❌"
        print(f"Account Value: ${account_value:.2f}")
        print(f"PnL:           ${pnl:+.2f} ({pnl_pct:+.2f}%) {status}")
    else:
        print("Unable to fetch account data")
    print()

    # Position
    print("📈 POSITION")
    print("-" * 80)
    size, entry = get_position()
    if size is not None:
        if abs(size) < 0.0001:
            print("Position:      FLAT (no position)")
        else:
            direction = "LONG" if size > 0 else "SHORT"
            print(f"Position:      {direction} {abs(size):.5f} BTC")
            print(f"Entry:         ${entry:,.2f}")
    else:
        print("Unable to fetch position")
    print()

    # Performance
    print("📊 RECENT PERFORMANCE (Last 20 fills)")
    print("-" * 80)
    perf = get_recent_performance()
    if perf:
        status = "✅ PROFITABLE" if perf["profitable"] else "❌ LOSING"
        print(f"Buys:          {perf['buys']}")
        print(f"Sells:         {perf['sells']}")
        print(f"Spread:        {perf['spread_bps']:+.2f} bps {status}")

        # OPT#14 adaptive status
        spread_bps = perf["spread_bps"]
        if spread_bps < -2.0:
            print(f"\n🔧 OPT#14 Mode: DEFENSIVE ($5 distance, 1 level)")
        elif spread_bps < 2.0:
            print(f"\n⚙️  OPT#14 Mode: MODERATE ($3.5 distance, 2 levels)")
        else:
            print(f"\n✅ OPT#14 Mode: NORMAL ($2 distance, 3 levels)")
    else:
        print("Insufficient data (need buys AND sells)")

    print()
    print("=" * 80)
    print("\n💡 Quick Commands:")
    print("   View monitor log:  tail -f logs/monitor.log")
    print("   View bot log:      tail -f logs/bot_$(date +%Y-%m-%d).log")
    print("   Stop bot:          pkill -f mmb-1.py")
    print("   Stop monitor:      pkill -f continuous_monitor.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
