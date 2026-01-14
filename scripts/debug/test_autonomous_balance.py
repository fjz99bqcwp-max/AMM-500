#!/usr/bin/env python3
"""Quick test of autonomous monitor balance detection."""

import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "config" / ".env")

WALLET = os.getenv("WALLET_ADDRESS", "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C")
URL = "https://api.hyperliquid.xyz/info"
SYMBOL = "US500"

def log(level, msg):
    print(f"[{level}] {msg}")

# Get perps state
resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=10)
account_state = resp.json()
margin_summary = account_state.get("marginSummary", {})
account_value = float(margin_summary.get("accountValue", 0))

print(f"=== Perps State ===")
print(f"Account Value: ${account_value:.2f}")

# HIP-3 fix: Check Spot USDH  
print(f"\n=== Checking Spot USDH (HIP-3 margin) ===")
try:
    spot_resp = requests.post(URL, json={"type": "spotClearinghouseState", "user": WALLET}, timeout=10)
    spot_state = spot_resp.json()
    print(f"Spot balances: {spot_state.get('balances', [])}")
    for b in spot_state.get("balances", []):
        if b.get("coin") == "USDH":
            spot_usdh = float(b.get("total", 0))
            print(f"Found USDH: ${spot_usdh:.2f}")
            if spot_usdh > 0:
                log("INFO", f"HIP-3: Found ${spot_usdh:.2f} USDH in Spot (usable as margin)")
                account_value = spot_usdh
            break
except Exception as e:
    log("ERROR", f"Could not check Spot USDH: {e}")

print(f"\n=== With HIP-3 Fix ===")
print(f"Total Equity: ${account_value:.2f}")
