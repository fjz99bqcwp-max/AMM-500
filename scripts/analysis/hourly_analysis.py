#!/usr/bin/env python3
"""Analyze fills hour by hour to identify when losses occurred."""

import requests
from datetime import datetime

WALLET = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
URL = "https://api.hyperliquid.xyz/info"


def analyze_hourly():
    """Analyze fills grouped by hour."""

    # Get fills from API
    resp = requests.post(URL, json={"type": "userFills", "user": WALLET})
    fills = resp.json()

    if not fills:
        print("No fills found")
        return

    # Group fills by hour
    hourly_data = {}

    for fill in fills:
        timestamp = datetime.fromtimestamp(fill.get("time", 0) / 1000)
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")

        if hour_key not in hourly_data:
            hourly_data[hour_key] = {"buys": [], "sells": [], "buy_count": 0, "sell_count": 0}

        price = float(fill.get("px", 0))
        if fill.get("side") == "B":
            hourly_data[hour_key]["buys"].append(price)
            hourly_data[hour_key]["buy_count"] += 1
        else:
            hourly_data[hour_key]["sells"].append(price)
            hourly_data[hour_key]["sell_count"] += 1

    # Sort by hour
    sorted_hours = sorted(hourly_data.keys())

    print("=" * 80)
    print("HOURLY PERFORMANCE ANALYSIS")
    print("=" * 80)

    total_spread_bps = 0
    hours_with_data = 0

    for hour in sorted_hours:
        data = hourly_data[hour]

        if not data["buys"] or not data["sells"]:
            continue

        avg_buy = sum(data["buys"]) / len(data["buys"])
        avg_sell = sum(data["sells"]) / len(data["sells"])
        spread = avg_sell - avg_buy
        spread_bps = (spread / avg_buy) * 10000

        total_spread_bps += spread_bps
        hours_with_data += 1

        status = "✅" if spread_bps > 0 else "❌"

        print(f"\n{hour}")
        print(f"  Buys:  {data['buy_count']:3d} @ ${avg_buy:,.2f}")
        print(f"  Sells: {data['sell_count']:3d} @ ${avg_sell:,.2f}")
        print(f"  Spread: ${spread:+.2f} ({spread_bps:+.2f} bps) {status}")

    print("\n" + "=" * 80)
    print("OVERALL 24H SUMMARY")
    print("=" * 80)

    if hours_with_data > 0:
        avg_spread = total_spread_bps / hours_with_data
        status = "✅ PROFITABLE" if avg_spread > 0 else "❌ LOSING"
        print(f"Average Spread: {avg_spread:+.2f} bps {status}")
        print(f"Hours analyzed: {hours_with_data}")


if __name__ == "__main__":
    analyze_hourly()
