#!/usr/bin/env python3
"""Monitor for 10 complete fills on US500 HIP-3."""

import time
import requests
from hyperliquid.info import Info

WALLET = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"

def get_fills(start_time_ms: int) -> list:
    """Get HIP-3 fills since start time."""
    now_ms = int(time.time() * 1000)
    resp = requests.post('https://api.hyperliquid.xyz/info', json={
        'type': 'userFillsByTime',
        'user': WALLET,
        'startTime': start_time_ms,
        'endTime': now_ms,
        'perp_dexs': ['km']
    })
    return resp.json() if resp.status_code == 200 else []

def get_status():
    """Get account status."""
    info = Info(skip_ws=True)
    
    # Get perp state (regular API)
    state = info.user_state(WALLET)
    margin = state.get('marginSummary', {})
    perp_equity = float(margin.get('accountValue', 0))
    
    # Get Spot USDH balance (used for HIP-3 margin)
    spot = info.spot_user_state(WALLET)
    usdh_total = 0
    for bal in spot.get('balances', []):
        if bal.get('coin') == 'USDH':
            usdh_total = float(bal.get('total', 0))
            break
    
    acct_val = perp_equity + usdh_total
    
    # Position
    positions = state.get('assetPositions', [])
    pos = None
    for p in positions:
        if p.get('position', {}).get('coin') == 'US500':
            pos = p['position']
            break
    
    pos_size = float(pos.get('szi', 0)) if pos else 0
    unrealized = float(pos.get('unrealizedPnl', 0)) if pos else 0
    
    # Orderbook - use direct POST for HIP-3
    try:
        book = info.post("/info", {"type": "l2Book", "coin": "km:US500"})
        bids = book.get('levels', [[],[]])[0]
        asks = book.get('levels', [[],[]])[1]
        best_bid = float(bids[0]['px']) if bids else 0
        best_ask = float(asks[0]['px']) if asks else 0
        spread_bps = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 10000 if best_bid and best_ask else 0
    except:
        best_bid, best_ask, spread_bps = 0, 0, 0
    
    return {
        'account_value': acct_val,
        'position': pos_size,
        'unrealized_pnl': unrealized,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread_bps': spread_bps
    }

def main():
    print("=" * 60)
    print("US500 HIP-3 Fill Monitor - Waiting for 10 fills")
    print("=" * 60)
    
    start_time_ms = int(time.time() * 1000)
    seen_fills = set()
    total_pnl = 0.0
    
    while len(seen_fills) < 10:
        fills = get_fills(start_time_ms)
        
        # Process new fills
        for f in fills:
            fill_id = f.get('tid', f.get('hash', str(f)))
            if fill_id not in seen_fills:
                seen_fills.add(fill_id)
                side = f.get('side', 'N/A')
                px = f.get('px', 'N/A')
                sz = f.get('sz', 'N/A')
                fee = float(f.get('fee', 0))
                ts = f.get('time', 0)
                dt = time.strftime('%H:%M:%S', time.localtime(ts/1000)) if ts else 'N/A'
                
                print(f"\n  FILL #{len(seen_fills)}: {dt} {side.upper()} {sz} @ ${px} (fee: ${fee:.4f})")
        
        # Status update every 30 seconds
        status = get_status()
        ts = time.strftime('%H:%M:%S')
        print(f"\r[{ts}] Fills: {len(seen_fills)}/10 | "
              f"Pos: {status['position']:.1f} | "
              f"PnL: ${status['unrealized_pnl']:.2f} | "
              f"Acct: ${status['account_value']:.2f} | "
              f"Book: ${status['best_bid']:.2f}/{status['best_ask']:.2f} ({status['spread_bps']:.1f}bps)", 
              end='', flush=True)
        
        time.sleep(10)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: Collected {len(seen_fills)} fills")
    
    # Final summary
    fills = get_fills(start_time_ms)
    total_fees = sum(float(f.get('fee', 0)) for f in fills)
    total_volume = sum(float(f.get('px', 0)) * float(f.get('sz', 0)) for f in fills)
    
    print(f"Total Volume: ${total_volume:.2f}")
    print(f"Total Fees: ${total_fees:.4f}")
    print(f"Avg Rebate: ${-total_fees/len(fills) if fills else 0:.4f}/fill")
    print("=" * 60)

if __name__ == "__main__":
    main()
