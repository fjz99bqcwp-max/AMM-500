#!/usr/bin/env python3
"""Test Hyperliquid API candle fetch"""
import requests
import json

url = "https://api.hyperliquid.xyz/info"

# Try recent data ranges
test_ranges = [
    ("Dec 1-2, 2024", 1733011200000, 1733097600000),
    ("Nov 1-2, 2024", 1730419200000, 1730505600000),
    ("Oct 1-2, 2024", 1727740800000, 1727827200000),
    ("Sep 1-2, 2024", 1725148800000, 1725235200000),
]

print("Testing different date ranges for BTC candles...")
for name, start, end in test_ranges:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "1h",
            "startTime": start,
            "endTime": end
        }
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    count = len(data) if isinstance(data, list) else 0
    print(f"  {name}: {count} candles")
    if count > 0:
        print(f"    Sample: t={data[0].get('t', 'N/A')}, o={data[0].get('o', 'N/A')}")
