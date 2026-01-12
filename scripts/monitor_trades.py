#!/usr/bin/env python3
"""Monitor for 10 complete round-trip trades."""

import requests
import time
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
URL = "https://api.hyperliquid.xyz/info"


def get_fills_since(start_time):
    resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
    fills = resp.json()
    return [f for f in fills if f.get("time", 0) > start_time]


def main():
    start_time = int(time.time() * 1000)
    print(f"Monitoring for 10 complete trades starting at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    target_trades = 10
    completed_trades = 0
    total_buys = 0
    total_sells = 0
    check_interval = 30  # seconds

    while completed_trades < target_trades:
        time.sleep(check_interval)

        fills = get_fills_since(start_time)
        buys = [f for f in fills if f.get("side") == "B"]
        sells = [f for f in fills if f.get("side") == "A"]

        # A complete trade is min(buys, sells)
        new_completed = min(len(buys), len(sells))

        if new_completed > completed_trades:
            # Show new fills
            for f in fills:
                ts = datetime.fromtimestamp(f.get("time", 0) / 1000).strftime("%H:%M:%S")
                side = "BUY " if f.get("side") == "B" else "SELL"
                px = float(f.get("px", 0))
                sz = float(f.get("sz", 0))
                print(f"  {ts} | {side} {sz:.5f} BTC @ ${px:,.0f}")

            completed_trades = new_completed
            total_buys = len(buys)
            total_sells = len(sells)

        elapsed = (time.time() * 1000 - start_time) / 1000 / 60
        print(
            f"\n[{elapsed:.1f} min] Complete trades: {completed_trades}/{target_trades} | Buys: {total_buys} | Sells: {total_sells}"
        )

        if len(buys) > 0 and len(sells) > 0:
            avg_buy = sum(float(f.get("px", 0)) for f in buys) / len(buys)
            avg_sell = sum(float(f.get("px", 0)) for f in sells) / len(sells)
            spread = avg_sell - avg_buy
            spread_bps = spread / ((avg_buy + avg_sell) / 2) * 10000
            status = "✅" if spread_bps > 0 else "❌"
            print(f"  Spread: ${spread:.2f} ({spread_bps:.2f} bps) {status}")

    # Final summary
    print("\n" + "=" * 60)
    print("10 COMPLETE TRADES ACHIEVED!")
    print("=" * 60)

    fills = get_fills_since(start_time)
    buys = [f for f in fills if f.get("side") == "B"]
    sells = [f for f in fills if f.get("side") == "A"]

    if len(buys) > 0 and len(sells) > 0:
        avg_buy = sum(float(f.get("px", 0)) for f in buys) / len(buys)
        avg_sell = sum(float(f.get("px", 0)) for f in sells) / len(sells)
        spread = avg_sell - avg_buy
        mid = (avg_buy + avg_sell) / 2
        spread_bps = spread / mid * 10000

        print(f"Total Buys: {len(buys)}")
        print(f"Total Sells: {len(sells)}")
        print(f"Avg Buy: ${avg_buy:.2f}")
        print(f"Avg Sell: ${avg_sell:.2f}")
        print(f"Spread Captured: ${spread:.2f} ({spread_bps:.2f} bps)")

        # Estimated PnL per trade
        min_pairs = min(len(buys), len(sells))
        vol_per_trade = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in fills) / len(
            fills
        )
        pnl_per_trade = spread_bps / 10000 * vol_per_trade
        print(f"Est. PnL per trade: ${pnl_per_trade:.4f}")

        if spread_bps > 0:
            print("\n✅ OPT#12 VALIDATED - POSITIVE SPREAD!")
        else:
            print("\n❌ NEGATIVE SPREAD - NEEDS ATTENTION")


if __name__ == "__main__":
    main()
