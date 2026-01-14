#!/usr/bin/env python3
"""
Deep investigation of balance vs trading ability discrepancy
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    from eth_account import Account
    import requests
except ImportError as e:
    print(f"❌ Error: Required packages not installed: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv('config/.env')

def deep_balance_investigation():
    """Deep investigation of all balance sources and trading capability"""
    
    # Get credentials
    private_key = os.getenv("PRIVATE_KEY")
    wallet_address = os.getenv("WALLET_ADDRESS")
    
    print("🔍 DEEP BALANCE INVESTIGATION")
    print("=" * 60)
    print(f"Wallet: {wallet_address}")
    print("=" * 60)
    
    # 1. Check all possible margin sources
    try:
        # Signed API
        account = Account.from_key(private_key)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(wallet_address)
        
        print(f"\n1️⃣ SIGNED USER STATE FULL DUMP:")
        print(f"Keys: {list(user_state.keys())}")
        
        margin_summary = user_state.get('marginSummary', {})
        print(f"\n📊 MARGIN SUMMARY:")
        for key, value in margin_summary.items():
            print(f"  {key}: {value}")
            
        cross_summary = user_state.get('crossMarginSummary', {})
        print(f"\n🏦 CROSS MARGIN SUMMARY:")
        for key, value in cross_summary.items():
            print(f"  {key}: {value}")
            
        print(f"\n💳 WITHDRAWABLE: ${user_state.get('withdrawable', 0)}")
        print(f"💰 CROSS MAINTENANCE MARGIN USED: ${user_state.get('crossMaintenanceMarginUsed', 0)}")
        
        # Check asset positions in detail
        asset_positions = user_state.get('assetPositions', [])
        print(f"\n📊 ASSET POSITIONS ({len(asset_positions)}):")
        for i, pos_data in enumerate(asset_positions):
            print(f"  Position {i}:")
            position = pos_data.get('position', {})
            for key, value in position.items():
                print(f"    {key}: {value}")
                
    except Exception as e:
        print(f"❌ Signed API error: {e}")
    
    # 2. Check unsigned API for comparison
    try:
        print(f"\n2️⃣ UNSIGNED API COMPARISON:")
        
        # Perp clearinghouse
        perp_resp = requests.post("https://api.hyperliquid.xyz/info", 
                                json={"type": "clearinghouseState", "user": wallet_address}, 
                                timeout=10)
        perp_data = perp_resp.json()
        
        print(f"📊 UNSIGNED PERP DATA:")
        margin_summary_unsigned = perp_data.get('marginSummary', {})
        for key, value in margin_summary_unsigned.items():
            print(f"  {key}: {value}")
            
        print(f"💳 Withdrawable: {perp_data.get('withdrawable', 0)}")
        
        # Check spot
        spot_resp = requests.post("https://api.hyperliquid.xyz/info",
                                json={"type": "spotClearinghouseState", "user": wallet_address},
                                timeout=10)
        spot_data = spot_resp.json()
        
        print(f"\n💎 SPOT BALANCES:")
        for balance in spot_data.get('balances', []):
            coin = balance.get('coin', 'Unknown')
            total = float(balance.get('total', 0))
            if total > 0.000001:  # Show any meaningful balance
                print(f"  {coin}: ${total:.8f} (Hold: ${balance.get('hold', 0)})")
        
    except Exception as e:
        print(f"❌ Unsigned API error: {e}")
    
    # 3. Check recent fills to understand trading source
    try:
        print(f"\n3️⃣ RECENT FILLS ANALYSIS:")
        fills_resp = requests.post("https://api.hyperliquid.xyz/info",
                                 json={"type": "userFills", "user": wallet_address},
                                 timeout=10)
        fills_data = fills_resp.json()
        
        # Look at most recent fills
        recent_fills = fills_data[-10:] if fills_data else []
        print(f"📊 LAST 10 FILLS:")
        
        for fill in recent_fills:
            timestamp = fill.get('time', 0)
            coin = fill.get('coin', 'Unknown')
            side = fill.get('side', 'Unknown')
            size = float(fill.get('sz', 0))
            price = float(fill.get('px', 0))
            fee = float(fill.get('fee', 0))
            
            # Convert timestamp to readable
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            print(f"  [{dt.strftime('%H:%M:%S')}] {side} {size} {coin} @ ${price} | Fee: ${fee}")
            
        print(f"\n📈 TOTAL FILLS: {len(fills_data)}")
        
    except Exception as e:
        print(f"❌ Fills API error: {e}")
        
    # 4. Check if there's cross-margin or other funding sources
    print(f"\n4️⃣ POTENTIAL FUNDING SOURCES:")
    print(f"  • Cross-margin account")
    print(f"  • Isolated margin pools")  
    print(f"  • Pending settlements")
    print(f"  • Funding payments")
    print(f"  • Fee rebates")
    
    print(f"\n💡 CONCLUSION:")
    print(f"  Balance shows $0 but trading continues - investigating margin sources...")

if __name__ == "__main__":
    deep_balance_investigation()