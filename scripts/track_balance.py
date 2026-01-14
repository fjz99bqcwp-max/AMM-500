#!/usr/bin/env python3
"""
Signed Balance Tracking Script for AMM-500 Bot
Uses private key authentication to get accurate perp equity data
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    from eth_account import Account
except ImportError as e:
    print(f"❌ Error: Required packages not installed: {e}")
    print("Run: pip install hyperliquid-python-sdk eth-account")
    sys.exit(1)

# Load environment variables
load_dotenv('config/.env')

def get_signed_balance_data():
    """Get accurate balance data using signed API queries"""
    
    # Get credentials
    private_key = os.getenv("PRIVATE_KEY")  # Use existing PRIVATE_KEY from .env
    wallet_address = os.getenv("WALLET_ADDRESS", "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C")
    
    if not private_key:
        print("❌ Error: PRIVATE_KEY not set in config/.env")
        print("This should be your wallet's private key for signing transactions")
        return None
    
    print("🔐 SIGNED BALANCE TRACKING")
    print("=" * 50)
    print(f"Wallet: {wallet_address}")
    print("=" * 50)
    
    try:
        # Initialize SDK with mainnet and signed queries
        account = Account.from_key(private_key)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        exchange = Exchange(
            wallet=account,
            base_url=constants.MAINNET_API_URL,
            account_address=wallet_address
        )
        
        # Get signed user state - this includes private balance data
        user_state = info.user_state(wallet_address)
        
        print(f"✅ Successfully retrieved signed user state")
        print(f"Keys available: {list(user_state.keys()) if isinstance(user_state, dict) else 'Not a dict'}")
        
        # Parse clearinghouse state (perp account)
        perp_state = user_state.get('clearinghouseState', {})
        if isinstance(perp_state, list) and len(perp_state) > 0:
            perp_state = perp_state[0]  # Take first element if list
        
        margin_summary = perp_state.get('marginSummary', {})
        
        # Extract key balance metrics
        account_value = float(margin_summary.get('accountValue', 0))
        total_margin_used = float(margin_summary.get('totalMarginUsed', 0))
        total_ntl_pos = float(margin_summary.get('totalNtlPos', 0))
        total_raw_usd = float(margin_summary.get('totalRawUsd', 0))
        withdrawable = float(perp_state.get('withdrawable', 0))
        
        print(f"\n📊 PERP ACCOUNT SUMMARY:")
        print(f"  💰 Account Equity: ${account_value:.2f}")
        print(f"  💳 Withdrawable: ${withdrawable:.2f}")
        print(f"  🔒 Margin Used: ${total_margin_used:.2f}")  
        print(f"  📈 Notional Position: ${total_ntl_pos:.2f}")
        print(f"  💵 Total Raw USD: ${total_raw_usd:.2f}")
        
        # Cross margin summary if available
        cross_summary = perp_state.get('crossMarginSummary', {})
        if cross_summary:
            cross_margin_used = float(cross_summary.get('crossMarginUsed', 0))
            maintenance_margin = float(cross_summary.get('maintenanceMargin', 0))
            print(f"  🏦 Cross Margin Used: ${cross_margin_used:.2f}")
            print(f"  ⚠️  Maintenance Margin: ${maintenance_margin:.2f}")
        
        # Check for active positions
        asset_positions = perp_state.get('assetPositions', [])
        active_positions = []
        
        print(f"\n📊 ACTIVE POSITIONS:")
        if asset_positions:
            for pos_data in asset_positions:
                position = pos_data.get('position', {})
                coin = position.get('coin', 'Unknown')
                size = float(position.get('szi', 0))
                entry_px = float(position.get('entryPx', 0))
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                
                if size != 0:  # Only show non-zero positions
                    active_positions.append({
                        'coin': coin,
                        'size': size,
                        'entry_px': entry_px,
                        'unrealized_pnl': unrealized_pnl,
                        'notional_value': abs(size) * entry_px
                    })
                    
                    direction = "LONG" if size > 0 else "SHORT"
                    print(f"  🎯 {coin}: {direction} {abs(size):.4f} @ ${entry_px:.2f}")
                    print(f"     Notional: ${abs(size) * entry_px:.2f} | PnL: ${unrealized_pnl:+.2f}")
        
        if not active_positions:
            print(f"  📭 No active positions")
        
        # Get spot balances for comparison
        spot_state = user_state.get('spotState', {})
        spot_balances = spot_state.get('balances', [])
        
        print(f"\n💎 SPOT BALANCES:")
        total_spot_value = 0
        for balance in spot_balances:
            coin = balance.get('coin', 'Unknown')
            total = float(balance.get('total', 0))
            hold = float(balance.get('hold', 0))
            available = total - hold
            
            if total > 0.001:  # Only show meaningful balances
                print(f"  {coin}: ${total:.6f} (Available: ${available:.6f}, Hold: ${hold:.6f})")
                total_spot_value += total
        
        print(f"  💵 Total Spot Value: ${total_spot_value:.6f}")
        
        # Summary comparison
        print(f"\n💡 BALANCE SUMMARY:")
        print(f"  🏦 Perp Account Equity: ${account_value:.2f} (MAIN TRADING BALANCE)")
        print(f"  💎 Spot Account Value: ${total_spot_value:.6f}")
        print(f"  📊 Total Account Value: ${account_value + total_spot_value:.2f}")
        
        # Margin utilization
        if account_value > 0:
            margin_utilization = (total_margin_used / account_value) * 100
            print(f"  📈 Margin Utilization: {margin_utilization:.1f}%")
            
            if total_ntl_pos > 0:
                effective_leverage = total_ntl_pos / account_value
                print(f"  ⚖️  Effective Leverage: {effective_leverage:.2f}x")
        
        return {
            'perp_equity': account_value,
            'withdrawable': withdrawable,
            'margin_used': total_margin_used,
            'notional_position': total_ntl_pos,
            'spot_total': total_spot_value,
            'total_account_value': account_value + total_spot_value,
            'positions': active_positions,
            'margin_utilization_pct': (total_margin_used / account_value * 100) if account_value > 0 else 0,
            'effective_leverage': (total_ntl_pos / account_value) if account_value > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Error getting signed balance data: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

if __name__ == "__main__":
    balance_data = get_signed_balance_data()
    if balance_data:
        print(f"\n✅ Successfully retrieved signed balance data")
        print(f"Main trading balance (Perp Equity): ${balance_data['perp_equity']:.2f}")
    else:
        print(f"\n❌ Failed to retrieve balance data")
        sys.exit(1)