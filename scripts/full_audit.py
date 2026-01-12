#!/usr/bin/env python3
"""Full trading data audit and analysis"""
import requests
from datetime import datetime, timedelta
from collections import defaultdict

wallet = "0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283"
url = "https://api.hyperliquid.xyz/info"

print("=" * 70)
print("HYPERLIQUID TRADING AUDIT")
print("=" * 70)

# Fetch fills
print("\nFetching trade data...")
response = requests.post(url, json={"type": "userFills", "user": wallet})
fills = response.json()

btc_fills = [f for f in fills if f["coin"] == "BTC"]
print(f"Total BTC fills available: {len(btc_fills)}")

# Last 24h analysis
now = datetime.now()
fills_24h = []
for f in btc_fills:
    ts = int(f["time"]) / 1000
    dt = datetime.fromtimestamp(ts)
    if now - dt < timedelta(hours=24):
        fills_24h.append(f)

print(f"Fills in last 24h: {len(fills_24h)}")

if not fills_24h:
    print("No fills in last 24h!")
    exit()

# Separate buys and sells
buys = [f for f in fills_24h if f["side"] == "B"]
sells = [f for f in fills_24h if f["side"] == "A"]

print(f"\n{'='*70}")
print("TRADE BREAKDOWN")
print("=" * 70)

avg_buy = 0.0
avg_sell = 0.0
total_buy_size = 0.0
total_sell_size = 0.0

if buys:
    avg_buy = sum(float(f["px"]) for f in buys) / len(buys)
    total_buy_size = sum(float(f["sz"]) for f in buys)
    print(f"BUY trades: {len(buys)}")
    print(f"  Average price: ${avg_buy:,.2f}")
    print(f"  Total size: {total_buy_size:.6f} BTC")

if sells:
    avg_sell = sum(float(f["px"]) for f in sells) / len(sells)
    total_sell_size = sum(float(f["sz"]) for f in sells)
    print(f"\nSELL trades: {len(sells)}")
    print(f"  Average price: ${avg_sell:,.2f}")
    print(f"  Total size: {total_sell_size:.6f} BTC")

if buys and sells:
    spread = avg_sell - avg_buy
    spread_bps = (spread / avg_buy) * 10000
    print(f"\n*** SPREAD CAPTURED: ${spread:+.2f} ({spread_bps:+.2f} bps) ***")
    if spread < 0:
        print("⚠️  ADVERSE SELECTION: Selling LOWER than buying!")
    else:
        print("✅ POSITIVE SPREAD: Capturing bid-ask spread")

# PnL Analysis
print(f"\n{'='*70}")
print("PNL ANALYSIS")
print("=" * 70)

total_closed_pnl = sum(float(f.get("closedPnl", "0")) for f in fills_24h)
total_fees = sum(float(f["fee"]) for f in fills_24h)
net_pnl = total_closed_pnl - total_fees

print(f"Closed PnL: ${total_closed_pnl:+.4f}")
print(f"Total Fees: ${total_fees:.4f}")
print(f"Net PnL: ${net_pnl:+.4f}")

# Per-trade analysis
wins = 0
losses = 0
neutral = 0
win_amounts = []
loss_amounts = []

for f in fills_24h:
    pnl = float(f.get("closedPnl", "0"))
    fee = float(f["fee"])
    net = pnl - fee
    if net > 0.0001:
        wins += 1
        win_amounts.append(net)
    elif net < -0.0001:
        losses += 1
        loss_amounts.append(abs(net))
    else:
        neutral += 1

print(f"\n{'='*70}")
print("WIN/LOSS ANALYSIS")
print("=" * 70)
print(f"Winning trades: {wins}")
print(f"Losing trades: {losses}")
print(f"Neutral trades: {neutral}")

if wins + losses > 0:
    win_rate = wins / (wins + losses) * 100
    print(f"Win Rate: {win_rate:.1f}%")

if win_amounts:
    avg_win = sum(win_amounts) / len(win_amounts)
    print(f"Average Win: ${avg_win:.4f}")

if loss_amounts:
    avg_loss = sum(loss_amounts) / len(loss_amounts)
    print(f"Average Loss: ${avg_loss:.4f}")

# Profit Factor
gross_profit = sum(
    float(f.get("closedPnl", "0")) for f in fills_24h if float(f.get("closedPnl", "0")) > 0
)
gross_loss = abs(
    sum(float(f.get("closedPnl", "0")) for f in fills_24h if float(f.get("closedPnl", "0")) < 0)
)

print(f"\n{'='*70}")
print("PROFIT FACTOR (Target: >1.1)")
print("=" * 70)
print(f"Gross Profit: ${gross_profit:.4f}")
print(f"Gross Loss: ${gross_loss:.4f}")

if gross_loss > 0:
    profit_factor = gross_profit / gross_loss
    print(f"PROFIT FACTOR: {profit_factor:.3f}")
    if profit_factor >= 1.1:
        print("✅ TARGET MET!")
    else:
        print(f"❌ Need {1.1 - profit_factor:.3f} improvement")
else:
    print("No losses (cannot calculate PF)")

# Per-trade profit target
print(f"\n{'='*70}")
print("PER-TRADE PROFIT (Target: $0.01)")
print("=" * 70)
if len(fills_24h) > 0:
    avg_pnl_per_trade = net_pnl / len(fills_24h)
    print(f"Average Net PnL per trade: ${avg_pnl_per_trade:.4f}")
    if avg_pnl_per_trade >= 0.01:
        print("✅ TARGET MET!")
    else:
        print(f"❌ Need ${0.01 - avg_pnl_per_trade:.4f} improvement per trade")

# Recent trades
print(f"\n{'='*70}")
print("LAST 20 TRADES")
print("=" * 70)
for f in fills_24h[:20]:
    ts = int(f["time"]) / 1000
    dt = datetime.fromtimestamp(ts)
    side = "BUY " if f["side"] == "B" else "SELL"
    px = float(f["px"])
    sz = float(f["sz"])
    fee = float(f["fee"])
    pnl = float(f.get("closedPnl", "0"))
    net = pnl - fee
    status = "✅" if net > 0 else "❌" if net < 0 else "⚪"
    print(f"{dt.strftime('%H:%M:%S')} | {side} {sz:.5f} @ ${px:,.0f} | Net: ${net:+.4f} {status}")

print(f"\n{'='*70}")
print("AUDIT COMPLETE")
print("=" * 70)
