#!/usr/bin/env python3
"""
Deep investigation to find the missing $1015+ and 135+ orders.
Check vaults, subaccounts, and recent activity.
"""
import requests
from datetime import datetime

MAINNET = 'https://api.hyperliquid.xyz/info'
wallet = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

def query(payload):
    r = requests.post(MAINNET, json=payload)
    return r.json() if r.status_code == 200 else None

print(f"Wallet: {wallet}")
print("="*70)

# 1. Check if this wallet owns any vaults
print("\n1. Checking vaults (vault leader)...")
data = query({"type": "vaultDetails", "user": wallet})
if data:
    print(f"   Vault details: {data}")
else:
    print("   No vault details found")

# 2. Check referral/rewards
print("\n2. Checking referral state...")
data = query({"type": "referral", "user": wallet})
if data:
    print(f"   Referral data: {data}")

# 3. Count fills to understand volume
print("\n3. Checking fill history...")
data = query({"type": "userFills", "user": wallet})
if data:
    print(f"   Total fills ever: {len(data)}")
    
    # Find fills from today
    today = datetime.now().strftime('%Y-%m-%d')
    today_fills = [f for f in data if today in str(f.get('time', ''))]
    print(f"   Fills today: {len(today_fills)}")
    
    if today_fills:
        print("   Today's fills:")
        for f in today_fills[-20:]:
            ts = f.get('time', 0)
            if isinstance(ts, int):
                dt = datetime.fromtimestamp(ts/1000).strftime('%H:%M:%S')
            else:
                dt = str(ts)[:16]
            print(f"     {dt} {f.get('coin')} {f.get('side')} {f.get('sz')} @ {f.get('px')} (oid={f.get('oid')})")

# 4. Check order status by recent known OIDs
print("\n4. Checking order status for recent OIDs...")
# Get some OIDs from recent fills
recent_oids = []
if data:
    for f in data[-100:]:
        oid = f.get('oid')
        if oid:
            recent_oids.append(oid)
    
    # Check last 5 unique OIDs
    unique_oids = list(dict.fromkeys(recent_oids[-20:]))
    print(f"   Checking {len(unique_oids)} recent order IDs...")
    for oid in unique_oids[-10:]:
        status = query({"type": "orderStatus", "user": wallet, "oid": oid})
        if status and status.get('status') == 'order':
            order = status.get('order', {}).get('order', {})
            print(f"   OID {oid}: {order.get('side')} {order.get('sz')} @ {order.get('limitPx')} - Status: {order.get('orderStatus', 'unknown')}")

# 5. Check userFunding history (might show more USD movements)
print("\n5. Checking funding history...")
data = query({"type": "userFunding", "user": wallet, "startTime": 0})
if data:
    recent = data[-5:] if len(data) > 5 else data
    for f in recent:
        print(f"   {f}")

# 6. Check withdrawals
print("\n6. Checking withdrawals...")
data = query({"type": "userNonFundingLedgerUpdates", "user": wallet, "startTime": 0})
if data:
    print(f"   Total ledger updates: {len(data)}")
    # Show last 5
    for u in data[-5:]:
        print(f"   {u.get('time', '')}: {u.get('type', '')} - {u}")

# 7. Summary
print("\n" + "="*70)
print("SUMMARY:")
print(f"  - Spot USDH: $453.87")
print(f"  - km Perp margin: $0.003") 
print(f"  - Open orders: 0")
print("")
print("If user claims $1469 and 135+ orders:")
print("  a) Check if viewing correct wallet in Hyperliquid UI")
print("  b) Check if funds were transferred/withdrawn")
print("  c) Check if '135 orders' refers to order HISTORY, not OPEN orders")
print("="*70)
