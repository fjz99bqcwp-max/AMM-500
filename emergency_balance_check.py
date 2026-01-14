#!/usr/bin/env python3
"""
Emergency balance and position check
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from hyperliquid.utils import constants
import requests

# Load config
from dotenv import load_dotenv
load_dotenv('config/.env')

WALLET_ADDRESS = os.getenv('WALLET_ADDRESS')

async def check_emergency_status():
    """Check current account status directly via API"""
    
    print("🚨 EMERGENCY BALANCE CHECK")
    print("=" * 50)
    
    # Check spot balance
    try:
        spot_url = "https://api.hyperliquid.xyz/info"
        spot_payload = {
            "type": "spotClearinghouseState",
            "user": WALLET_ADDRESS
        }
        
        spot_response = requests.post(spot_url, json=spot_payload, timeout=10)
        spot_data = spot_response.json()
        
        print(f"📊 SPOT BALANCES:")
        for balance in spot_data.get('balances', []):
            coin = balance.get('coin', 'Unknown')
            total = float(balance.get('total', '0'))
            hold = float(balance.get('hold', '0'))
            available = total - hold
            print(f"  {coin}: Total ${total:.2f} | Available ${available:.2f} | Hold ${hold:.2f}")
            
    except Exception as e:
        print(f"❌ Error fetching spot balances: {e}")
    
    # Check perp positions and balance
    try:
        perp_url = "https://api.hyperliquid.xyz/info"
        perp_payload = {
            "type": "clearinghouseState",
            "user": WALLET_ADDRESS
        }
        
        perp_response = requests.post(perp_url, json=perp_payload, timeout=10)
        perp_data = perp_response.json()
        
        print(f"\n📊 PERP ACCOUNT:")
        margin_summary = perp_data.get('marginSummary', {})
        account_value = float(margin_summary.get('accountValue', 0))
        total_margin_used = float(margin_summary.get('totalMarginUsed', 0))
        total_ntl_pos = float(margin_summary.get('totalNtlPos', 0))
        
        print(f"  Account Value: ${account_value:.2f}")
        print(f"  Margin Used: ${total_margin_used:.2f}")
        print(f"  Notional Position: ${total_ntl_pos:.2f}")
        
        # Check positions
        positions = perp_data.get('assetPositions', [])
        print(f"\n📊 ACTIVE POSITIONS:")
        
        if positions:
            for pos in positions:
                position = pos.get('position', {})
                coin = position.get('coin', 'Unknown')
                size = float(position.get('szi', 0))
                entry_px = float(position.get('entryPx', 0))
                position_value = abs(size) * entry_px
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                
                print(f"  {coin}: Size {size:.4f} | Entry ${entry_px:.2f} | Value ${position_value:.2f} | PnL ${unrealized_pnl:.2f}")
        else:
            print("  No active positions")
            
        # Check cross/isolated margin
        cross_margin_used = float(margin_summary.get('crossMarginUsed', 0))
        print(f"\n💰 MARGIN BREAKDOWN:")
        print(f"  Cross Margin: ${cross_margin_used:.2f}")
        print(f"  Total Margin: ${total_margin_used:.2f}")
        
    except Exception as e:
        print(f"❌ Error fetching perp data: {e}")

if __name__ == "__main__":
    asyncio.run(check_emergency_status())