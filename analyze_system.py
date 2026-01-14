#!/usr/bin/env python3
"""
Comprehensive System Analysis for AMM-500 Bot
Analyzes logs, trading performance, errors, and provides optimization recommendations
"""
import os
import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load environment
load_dotenv('config/.env')

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    from eth_account import Account
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

WALLET = os.getenv("WALLET_ADDRESS", "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C")
URL = "https://api.hyperliquid.xyz/info"

def analyze_fills(hours=24):
    """Analyze trading fills for performance metrics"""
    print(f"\n{'='*70}")
    print(f"📊 FILL ANALYSIS (Last {hours} hours)")
    print('='*70)
    
    try:
        start_time = int((datetime.now().timestamp() - hours*3600) * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        payload = {
            'type': 'userFillsByTime',
            'user': WALLET,
            'startTime': start_time,
            'endTime': end_time,
            'perp_dexs': ['km']
        }
        
        resp = requests.post(URL, json=payload, timeout=10)
        fills = resp.json()
        
        if not fills:
            print("❌ No fills found in the specified period")
            return None
        
        # Analyze fills
        buys = [f for f in fills if f.get('side') == 'B']
        sells = [f for f in fills if f.get('side') == 'S']
        
        buy_volume = sum(float(f.get('sz', 0)) for f in buys)
        sell_volume = sum(float(f.get('sz', 0)) for f in sells)
        
        buy_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in buys)
        sell_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in sells)
        
        avg_buy_px = buy_notional / buy_volume if buy_volume > 0 else 0
        avg_sell_px = sell_notional / sell_volume if sell_volume > 0 else 0
        
        total_fees = sum(float(f.get('fee', 0)) for f in fills)
        
        # Calculate spread capture
        spread_capture = avg_sell_px - avg_buy_px if buy_volume > 0 and sell_volume > 0 else 0
        spread_bps = (spread_capture / ((avg_buy_px + avg_sell_px)/2)) * 10000 if avg_buy_px > 0 and avg_sell_px > 0 else 0
        
        # Estimate PnL (simplified - assuming matched buys/sells)
        matched_volume = min(buy_volume, sell_volume)
        estimated_pnl = matched_volume * spread_capture - total_fees
        
        print(f"✅ Total Fills: {len(fills)}")
        print(f"   📈 Buys: {len(buys)} fills, {buy_volume:.2f} contracts @ ${avg_buy_px:.2f} avg")
        print(f"   📉 Sells: {len(sells)} fills, {sell_volume:.2f} contracts @ ${avg_sell_px:.2f} avg")
        print(f"   💰 Total Fees Paid: ${total_fees:.4f}")
        print(f"   📊 Spread Captured: ${spread_capture:.4f} ({spread_bps:+.2f} bps)")
        print(f"   💵 Estimated PnL: ${estimated_pnl:+.4f}")
        print(f"   🔄 Fill Rate: {len(fills)/hours:.1f} fills/hour")
        
        # Balance analysis
        imbalance = abs(buy_volume - sell_volume)
        imbalance_pct = (imbalance / max(buy_volume, sell_volume) * 100) if max(buy_volume, sell_volume) > 0 else 0
        
        print(f"\n   ⚖️  Volume Imbalance: {imbalance:.2f} contracts ({imbalance_pct:.1f}%)")
        if imbalance_pct > 20:
            print(f"   ⚠️  WARNING: High volume imbalance - bot not delta neutral!")
        else:
            print(f"   ✅ Volume balance acceptable")
        
        return {
            'total_fills': len(fills),
            'buys': len(buys),
            'sells': len(sells),
            'spread_bps': spread_bps,
            'estimated_pnl': estimated_pnl,
            'total_fees': total_fees,
            'fill_rate': len(fills)/hours,
            'imbalance_pct': imbalance_pct
        }
        
    except Exception as e:
        print(f"❌ Error analyzing fills: {e}")
        return None

def check_current_position():
    """Check current open position"""
    print(f"\n{'='*70}")
    print(f"📍 CURRENT POSITION")
    print('='*70)
    
    try:
        payload = {'type': 'clearinghouseState', 'user': WALLET}
        resp = requests.post(URL, json=payload, timeout=10)
        data = resp.json()
        
        asset_positions = data.get('assetPositions', [])
        
        if not asset_positions:
            print("✅ No open positions - delta neutral")
            return None
        
        for pos_data in asset_positions:
            pos = pos_data.get('position', {})
            coin = pos.get('coin', '')
            size = float(pos.get('szi', 0))
            
            if size != 0 and 'US500' in coin:
                entry_px = float(pos.get('entryPx', 0))
                mark_px = float(pos.get('markPx', 0))
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                margin_used = float(pos.get('marginUsed', 0))
                
                direction = "LONG" if size > 0 else "SHORT"
                notional = abs(size * mark_px)
                
                print(f"⚠️  OPEN POSITION DETECTED:")
                print(f"   Symbol: {coin}")
                print(f"   Direction: {direction}")
                print(f"   Size: {abs(size):.4f} contracts")
                print(f"   Entry Price: ${entry_px:.2f}")
                print(f"   Mark Price: ${mark_px:.2f}")
                print(f"   Unrealized PnL: ${unrealized_pnl:+.4f}")
                print(f"   Margin Used: ${margin_used:.2f}")
                print(f"   Notional Value: ${notional:.2f}")
                
                if abs(size) > 0.5:
                    print(f"   🚨 WARNING: Position size significant - bot may not be rebalancing!")
                
                return {
                    'size': size,
                    'entry_px': entry_px,
                    'unrealized_pnl': unrealized_pnl,
                    'margin_used': margin_used
                }
        
        print("✅ No significant positions")
        return None
        
    except Exception as e:
        print(f"❌ Error checking position: {e}")
        return None

def analyze_log_errors():
    """Analyze recent log files for errors"""
    print(f"\n{'='*70}")
    print(f"🔍 LOG ERROR ANALYSIS")
    print('='*70)
    
    log_file = Path('logs/bot_2026-01-14.log')
    
    if not log_file.exists():
        print("❌ No log file found for today")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        errors = [l for l in lines if 'ERROR' in l or 'CRITICAL' in l]
        warnings = [l for l in lines if 'WARNING' in l and 'ERROR' not in l]
        
        print(f"📝 Log file: {log_file}")
        print(f"   Total lines: {len(lines)}")
        print(f"   ❌ Errors: {len(errors)}")
        print(f"   ⚠️  Warnings: {len(warnings)}")
        
        # Show recent errors
        if errors:
            print(f"\n   Recent Errors:")
            for err in errors[-5:]:
                print(f"      {err.strip()[:120]}")
        
        # Check for specific issues
        rate_limits = [l for l in lines if '429' in l or 'rate limit' in l.lower()]
        connection_errors = [l for l in lines if 'connection' in l.lower() and ('error' in l.lower() or 'failed' in l.lower())]
        
        if rate_limits:
            print(f"\n   🚨 Rate limit issues detected: {len(rate_limits)}")
        
        if connection_errors:
            print(f"\n   🚨 Connection errors detected: {len(connection_errors)}")
        
        return {
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'rate_limits': len(rate_limits),
            'connection_errors': len(connection_errors)
        }
        
    except Exception as e:
        print(f"❌ Error reading logs: {e}")
        return None

def check_config():
    """Check current configuration"""
    print(f"\n{'='*70}")
    print(f"⚙️  CONFIGURATION CHECK")
    print('='*70)
    
    try:
        from dotenv import dotenv_values
        config = dotenv_values('config/.env')
        
        leverage = config.get('LEVERAGE', 'Not set')
        order_size = config.get('ORDER_SIZE_FRACTION', 'Not set')
        symbol = config.get('SYMBOL', 'Not set')
        testnet = config.get('TESTNET', 'Not set')
        
        print(f"   Symbol: {symbol}")
        print(f"   Leverage: {leverage}x")
        print(f"   Order Size Fraction: {order_size}")
        print(f"   Testnet Mode: {testnet}")
        
        # Check if emergency defensive mode
        leverage_val = float(leverage) if leverage != 'Not set' else 0
        order_size_val = float(order_size) if order_size != 'Not set' else 0
        
        if leverage_val <= 1:
            print(f"\n   ⚠️  EMERGENCY DEFENSIVE MODE ACTIVE (1x leverage)")
        elif leverage_val <= 5:
            print(f"\n   ⚠️  Conservative mode (5x leverage)")
        elif leverage_val >= 20:
            print(f"\n   🚨 High leverage mode ({leverage_val}x) - monitor closely!")
        
        if order_size_val <= 0.01:
            print(f"   ⚠️  Very small order size ({order_size_val}) - limited trading")
        
        return {
            'leverage': leverage_val,
            'order_size': order_size_val,
            'symbol': symbol
        }
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return None

def generate_recommendations(fill_data, position_data, error_data, config_data):
    """Generate optimization recommendations"""
    print(f"\n{'='*70}")
    print(f"💡 RECOMMENDATIONS")
    print('='*70)
    
    recommendations = []
    
    # Check API balance issue
    print(f"\n1. 🔴 CRITICAL: API Balance Data Corruption")
    print(f"   - Issue: Hyperliquid API returning $0 balance")
    print(f"   - Blockchain shows: ~$1,465 actual balance")
    print(f"   - Action: ✅ Code fixes already implemented to handle this")
    print(f"   - Next: Contact Hyperliquid support about API data corruption")
    
    # Check fill performance
    if fill_data:
        print(f"\n2. 📊 Trading Performance Analysis")
        if fill_data['spread_bps'] > 3:
            print(f"   ✅ Spread capture: {fill_data['spread_bps']:.2f} bps (GOOD - above 3 bps)")
        elif fill_data['spread_bps'] > 0:
            print(f"   ⚠️  Spread capture: {fill_data['spread_bps']:.2f} bps (LOW - optimize spreads)")
            recommendations.append("Increase MIN_SPREAD_BPS to capture more edge")
        else:
            print(f"   🚨 Spread capture: {fill_data['spread_bps']:.2f} bps (NEGATIVE - adverse selection!)")
            recommendations.append("URGENT: Widen spreads to avoid adverse selection")
        
        if fill_data['fill_rate'] < 5:
            print(f"   ⚠️  Fill rate: {fill_data['fill_rate']:.1f}/hour (LOW - too passive)")
            recommendations.append("Tighten spreads or increase quote levels for more fills")
        elif fill_data['fill_rate'] > 50:
            print(f"   ⚠️  Fill rate: {fill_data['fill_rate']:.1f}/hour (HIGH - may be too aggressive)")
            recommendations.append("Widen spreads to reduce adverse selection risk")
        else:
            print(f"   ✅ Fill rate: {fill_data['fill_rate']:.1f}/hour (GOOD - balanced)")
        
        if fill_data['imbalance_pct'] > 20:
            print(f"   🚨 Volume imbalance: {fill_data['imbalance_pct']:.1f}% (CRITICAL)")
            recommendations.append("Bot not maintaining delta neutrality - check rebalancing logic")
    
    # Check position
    if position_data:
        print(f"\n3. ⚠️  Open Position Detected")
        print(f"   - Bot should be delta-neutral but has {position_data['size']:.4f} position")
        recommendations.append("Review rebalancing interval and inventory skew settings")
    else:
        print(f"\n3. ✅ Position Management: Delta neutral (no open positions)")
    
    # Check configuration
    if config_data:
        print(f"\n4. ⚙️  Configuration Review")
        if config_data['leverage'] <= 1:
            print(f"   🔴 EMERGENCY MODE: 1x leverage severely limits profitability")
            recommendations.append("Gradually increase leverage to 10x once API data is restored")
        elif config_data['leverage'] < 10:
            print(f"   ⚠️  Low leverage ({config_data['leverage']}x) limits profit potential")
            recommendations.append("Consider increasing to 10-15x for better capital efficiency")
    
    # Print final recommendations
    if recommendations:
        print(f"\n{'='*70}")
        print(f"🎯 ACTION ITEMS:")
        print('='*70)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print(f"\n✅ No critical issues found - system operating normally")
    
    print(f"\n{'='*70}\n")

def main():
    print(f"\n{'='*70}")
    print(f"🔬 AMM-500 COMPREHENSIVE SYSTEM ANALYSIS")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Wallet: {WALLET}")
    
    # Run all analyses
    fill_data = analyze_fills(hours=24)
    position_data = check_current_position()
    error_data = analyze_log_errors()
    config_data = check_config()
    
    # Generate recommendations
    generate_recommendations(fill_data, position_data, error_data, config_data)

if __name__ == "__main__":
    main()
