#!/usr/bin/env python3
"""Check HIP-3 USDH balance status."""
import json
from hyperliquid.info import Info

info = Info(skip_ws=True)
addr = '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'

print('=== Checking HIP-3 State ===')
try:
    hip3 = info.hip3_spot_user_state(addr)
    print('hip3_spot_user_state:', json.dumps(hip3, indent=2))
except Exception as e:
    print(f'hip3_spot_user_state error: {e}')

print('\n=== Checking HIP-3 Perps (USDH) ===')
try:
    hip3_state = info.hip3_clearinghouse_state(addr, 'USDH')
    print('hip3_clearinghouse_state:')
    print(json.dumps(hip3_state, indent=2)[:3000])
except Exception as e:
    print(f'hip3_clearinghouse_state error: {e}')

print('\n=== Checking regular perps state ===')
try:
    state = info.user_state(addr)
    print('user_state margin summary:', json.dumps(state.get('marginSummary', {}), indent=2))
except Exception as e:
    print(f'user_state error: {e}')

# Also check balances from spot
print('\n=== Checking spot balances ===')
try:
    spot = info.spot_user_state(addr)
    print('spot_user_state:', json.dumps(spot, indent=2)[:1000])
except Exception as e:
    print(f'spot_user_state error: {e}')
