"""Analyze live bot performance and detect issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperliquid.info import Info
from src.utils.config import Config
import json
from datetime import datetime, timedelta

def main():
    """Analyze live trading performance."""
    config = Config.load()
    info = Info('https://api.hyperliquid.xyz', skip_ws=True)
    
    print("="*80)
    print("LIVE BOT PERFORMANCE ANALYSIS")
    print("="*80)
    
    # 1. Account State
    print("\n1. ACCOUNT STATE")
    print("-"*80)
    state = info.user_state(config.wallet_address)
    margin_summary = state['marginSummary']
    
    equity = float(margin_summary['accountValue'])
    total_pos_value = float(margin_summary['totalNtlPos'])
    margin_used = float(margin_summary['totalMarginUsed'])
    
    print(f"Equity: ${equity:.2f}")
    print(f"Position Value: ${total_pos_value:.2f}")
    print(f"Margin Used: ${margin_used:.2f}")
    print(f"Margin Utilization: {(margin_used/equity*100):.1f}%" if equity > 0 else "N/A")
    
    # Check for issues
    issues = []
    if equity < 500:
        issues.append(f"⚠️  LOW EQUITY: ${equity:.2f} (< $500)")
    if margin_used / equity > 0.80:
        issues.append(f"⚠️  HIGH MARGIN: {(margin_used/equity*100):.1f}% (>80%)")
    
    # 2. Open Orders
    print("\n2. OPEN ORDERS")
    print("-"*80)
    open_orders = info.open_orders(config.wallet_address)
    us500_orders = [o for o in open_orders if 'US500' in o['coin']]
    
    print(f"Total US500 Orders: {len(us500_orders)}")
    
    if us500_orders:
        bids = [o for o in us500_orders if o['side'] == 'B']
        asks = [o for o in us500_orders if o['side'] == 'A']
        print(f"  Bids: {len(bids)}")
        print(f"  Asks: {len(asks)}")
        
        # Show sample orders
        if bids:
            print(f"\n  Top 3 Bids:")
            for o in sorted(bids, key=lambda x: float(x['limitPx']), reverse=True)[:3]:
                print(f"    ${o['limitPx']} x {o['sz']}")
        
        if asks:
            print(f"\n  Top 3 Asks:")
            for o in sorted(asks, key=lambda x: float(x['limitPx']))[:3]:
                print(f"    ${o['limitPx']} x {o['sz']}")
    else:
        issues.append("❌ NO ORDERS PLACED - Bot not working!")
    
    # Check order balance
    if us500_orders:
        if abs(len(bids) - len(asks)) > 20:
            issues.append(f"⚠️  UNBALANCED ORDERS: {len(bids)} bids vs {len(asks)} asks")
    
    # 3. Recent Fills (Last Hour)
    print("\n3. RECENT FILLS (Last Hour)")
    print("-"*80)
    fills = info.user_fills(config.wallet_address)
    us500_fills = [f for f in fills if 'US500' in f['coin']]
    
    # Filter last hour
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    recent_fills = []
    
    for f in us500_fills:
        # Parse timestamp (ms)
        fill_time = datetime.fromtimestamp(int(f['time']) / 1000)
        if fill_time > one_hour_ago:
            recent_fills.append(f)
    
    print(f"Fills in last hour: {len(recent_fills)}")
    
    if recent_fills:
        total_pnl = sum(float(f.get('closedPnl', 0)) for f in recent_fills)
        total_fees = sum(abs(float(f['fee'])) for f in recent_fills)
        buys = sum(1 for f in recent_fills if f['side'] == 'B')
        sells = sum(1 for f in recent_fills if f['side'] == 'A')
        
        print(f"  Buys: {buys}, Sells: {sells}")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Total Fees: ${total_fees:.4f}")
        print(f"  Net: ${(total_pnl - total_fees):.2f}")
        
        # Sample fills
        print(f"\n  Last 5 Fills:")
        for i, f in enumerate(recent_fills[:5], 1):
            side = "BUY" if f['side'] == 'B' else "SELL"
            print(f"    {i}. {side} {f['sz']} @ ${f['px']} - Fee: ${abs(float(f['fee'])):.4f}")
    else:
        if len(us500_orders) > 0:
            issues.append("⚠️  NO FILLS in last hour - Orders too far from market?")
    
    # 4. Current Position
    print("\n4. CURRENT POSITION")
    print("-"*80)
    positions = state.get('assetPositions', [])
    us500_pos = None
    for pos in positions:
        if 'US500' in pos['position']['coin']:
            us500_pos = pos
            break
    
    if us500_pos:
        size = float(us500_pos['position']['szi'])
        entry_px = float(us500_pos['position']['entryPx']) if us500_pos['position']['entryPx'] else 0
        unrealized_pnl = float(us500_pos['position']['unrealizedPnl'])
        
        print(f"Position Size: {size:.2f} lots")
        print(f"Entry Price: ${entry_px:.2f}")
        print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
        
        # Check for excessive position
        if abs(size) > 5:
            issues.append(f"⚠️  LARGE POSITION: {size:.2f} lots (>5)")
    else:
        print("No open position")
    
    # 5. Issues Summary
    print("\n5. ISSUES DETECTED")
    print("-"*80)
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No critical issues detected")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-"*80)
    
    if len(us500_orders) == 0:
        print("❌ CRITICAL: Bot not placing orders!")
        print("   → Check logs for errors")
        print("   → Verify strategy._update_quotes() is being called")
        print("   → Check if orderbook data is available")
    elif len(recent_fills) == 0 and len(us500_orders) > 0:
        print("⚠️  No fills in last hour - possible issues:")
        print("   → Orders may be too far from market price")
        print("   → Check if spread is too wide (current: MIN_SPREAD_BPS=3)")
        print("   → Verify orders are within ±2% range")
    
    if equity < 500:
        print("⚠️  Low equity - consider:")
        print("   → Reducing leverage")
        print("   → Adding more capital")
    
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)

if __name__ == "__main__":
    main()
