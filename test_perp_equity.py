#!/usr/bin/env python3
"""
Test corrected balance calculation using PERP ACCOUNT EQUITY
"""
import requests
import os
from dotenv import load_dotenv
load_dotenv('config/.env')

WALLET = os.getenv('WALLET_ADDRESS')
URL = "https://api.hyperliquid.xyz/info"

def test_perp_equity():
    """Test perp account equity calculation"""
    print("🧪 TESTING PERP ACCOUNT EQUITY CALCULATION")
    print("=" * 50)
    
    try:
        # Get perp account state
        perp_resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=10)
        perp_data = perp_resp.json()
        
        margin_summary = perp_data.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        total_ntl_pos = float(margin_summary.get("totalNtlPos", 0))
        
        print(f"📊 PERP ACCOUNT SUMMARY:")
        print(f"  Account Value (EQUITY): ${account_value:.2f}")
        print(f"  Margin Used: ${total_margin_used:.2f}")
        print(f"  Notional Position: ${total_ntl_pos:.2f}")
        
        # Check positions
        positions = perp_data.get("assetPositions", [])
        print(f"\n📊 ACTIVE POSITIONS:")
        
        us500_position = None
        for pos in positions:
            position = pos.get("position", {})
            coin = position.get("coin", 'Unknown')
            size = float(position.get("szi", 0))
            entry_px = float(position.get("entryPx", 0))
            unrealized_pnl = float(position.get("unrealizedPnl", 0))
            
            if coin in ["km:US500", "US500"]:
                us500_position = position
                print(f"  ✅ {coin}: Size {size:.4f} | Entry ${entry_px:.2f} | PnL ${unrealized_pnl:.2f}")
            elif size != 0:
                print(f"  {coin}: Size {size:.4f} | Entry ${entry_px:.2f} | PnL ${unrealized_pnl:.2f}")
                
        if not us500_position:
            print("  No US500 position found")
            
        # Get spot balance for reference
        print(f"\n📊 SPOT BALANCE (REFERENCE ONLY):")
        spot_resp = requests.post(URL, json={"type": "spotClearinghouseState", "user": WALLET}, timeout=10)
        spot_data = spot_resp.json()
        
        for balance in spot_data.get('balances', []):
            coin = balance.get('coin', 'Unknown')
            total = float(balance.get('total', '0'))
            hold = float(balance.get('hold', '0'))
            available = total - hold
            if total > 0:
                print(f"  {coin}: Total ${total:.2f} | Available ${available:.2f} | Hold ${hold:.2f}")
                
        print(f"\n💡 CONCLUSION:")
        print(f"  ✅ MAIN TRADING BALANCE: ${account_value:.2f} (Perp Account Equity)")
        print(f"  📊 Expected Balance: $1,466.48")
        print(f"  🎯 Difference: ${1466.48 - account_value:+.2f}")
        
        return account_value
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

if __name__ == "__main__":
    test_perp_equity()