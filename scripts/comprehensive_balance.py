#!/usr/bin/env python3
"""
Comprehensive balance check - get all balances across all markets
"""
import os
import sys
from dotenv import load_dotenv
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Load environment
load_dotenv('config/.env')
private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

if not private_key or not wallet_address:
    print("ERROR: Missing PRIVATE_KEY or WALLET_ADDRESS in config/.env")
    sys.exit(1)

# Initialize exchange
account = eth_account.Account.from_key(private_key)
exchange = Exchange(account, constants.MAINNET_API_URL, account_address=wallet_address, perp_dexs=['km'])

try:
    print("=== COMPREHENSIVE BALANCE CHECK ===\n")
    
    info = exchange.info
    
    # 1. PERP ACCOUNT BALANCES
    print("1. PERPETUAL ACCOUNT BALANCES:")
    print("=" * 40)
    
    perp_state = info.user_state(wallet_address)
    margin_summary = perp_state.get("marginSummary", {})
    
    account_value = float(margin_summary.get("accountValue", 0))
    total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
    total_ntl_pos = float(margin_summary.get("totalNtlPos", 0))
    
    print(f"Account Value: ${account_value:.2f}")
    print(f"Total Margin Used: ${total_margin_used:.2f}")  
    print(f"Total Notional Position: ${total_ntl_pos:.2f}")
    print(f"Available Margin: ${account_value - total_margin_used:.2f}")
    
    # Check all positions
    asset_positions = perp_state.get("assetPositions", [])
    print(f"\nPositions ({len(asset_positions)} total):")
    total_unrealized = 0.0
    
    for pos in asset_positions:
        pos_info = pos.get("position", {})
        coin = pos_info.get("coin", "")
        size = float(pos_info.get("szi", 0))
        unrealized = float(pos_info.get("unrealizedPnl", 0))
        entry_px = float(pos_info.get("entryPx", 0))
        
        if abs(size) > 0.0001 or abs(unrealized) > 0.01:
            print(f"  {coin}: {size:.4f} @ ${entry_px:.2f} | PnL: ${unrealized:+.2f}")
            total_unrealized += unrealized
    
    if len(asset_positions) == 0:
        print("  No positions")
    
    print(f"Total Unrealized PnL: ${total_unrealized:+.2f}")
    
    # 2. SPOT ACCOUNT BALANCES  
    print(f"\n2. SPOT ACCOUNT BALANCES:")
    print("=" * 40)
    
    spot_state = info.spot_user_state(wallet_address)
    spot_balances = spot_state.get("balances", [])
    
    total_spot_value = 0.0
    spot_breakdown = {}
    
    for balance in spot_balances:
        coin = balance.get("coin", "")
        total = float(balance.get("total", 0))
        hold = float(balance.get("hold", 0))
        entry_notional = float(balance.get("entryNotional", 0))
        
        if abs(total) > 0.01:  # Show significant balances
            available = total - hold
            print(f"  {coin}:")
            print(f"    Total: {total:.8f}")
            print(f"    Hold: {hold:.8f}")
            print(f"    Available: {available:.8f}")
            if entry_notional != 0:
                print(f"    Entry Notional: ${entry_notional:.2f}")
            
            # For USD values, add to total
            if coin in ["USDC", "USDH"]:
                total_spot_value += total
                spot_breakdown[coin] = total
    
    print(f"\nTotal Spot USD Value: ${total_spot_value:.2f}")
    
    # 3. COMBINED TOTAL
    print(f"\n3. COMBINED TOTALS:")
    print("=" * 40)
    
    grand_total = account_value + total_spot_value + total_unrealized
    
    print(f"Perp Account Value: ${account_value:.2f}")
    print(f"Spot Account Value: ${total_spot_value:.2f}")
    print(f"Total Unrealized PnL: ${total_unrealized:+.2f}")
    print(f"GRAND TOTAL: ${grand_total:.2f}")
    
    # 4. COMPARISON WITH EXPECTED
    expected_total = 1469.10
    difference = grand_total - expected_total
    
    print(f"\n4. BALANCE ANALYSIS:")
    print("=" * 40)
    print(f"Expected Total: ${expected_total:.2f}")
    print(f"Actual Total: ${grand_total:.2f}")
    print(f"Difference: ${difference:+.2f}")
    
    if abs(difference) > 10:
        print(f"⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        if difference < 0:
            print("  - Balance is lower than expected")
            print("  - Check for recent losses or withdrawals")
        else:
            print("  - Balance is higher than expected") 
            print("  - Check for recent gains or deposits")
    else:
        print("✅ Balance matches expected range")
    
    # 5. US500 SPECIFIC INFO
    print(f"\n5. US500 SPECIFIC:")
    print("=" * 40)
    
    us500_found = False
    for pos in asset_positions:
        pos_info = pos.get("position", {})
        coin = pos_info.get("coin", "")
        if "US500" in coin:
            us500_found = True
            size = float(pos_info.get("szi", 0))
            unrealized = float(pos_info.get("unrealizedPnl", 0))
            entry_px = float(pos_info.get("entryPx", 0))
            margin_used = float(pos_info.get("marginUsed", 0))
            
            print(f"  Position: {size:.4f} {coin}")
            print(f"  Entry Price: ${entry_px:.2f}")
            print(f"  Unrealized PnL: ${unrealized:+.2f}")
            print(f"  Margin Used: ${margin_used:.2f}")
            
            # Calculate current notional
            # We would need current price to calculate this accurately
            
    if not us500_found:
        print("  No US500 position found")
    
    # Check for USDH specifically (used for HIP-3 margin)
    usdh_total = spot_breakdown.get("USDH", 0)
    if usdh_total > 0:
        print(f"  USDH Available for Margin: ${usdh_total:.2f}")
    else:
        print("  No USDH found in Spot (needed for US500 margin)")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)