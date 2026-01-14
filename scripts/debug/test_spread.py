import requests

# Get recent fills
response = requests.post(
    'https://api.hyperliquid.xyz/info',
    headers={'Content-Type': 'application/json'},
    json={'type': 'userFills', 'user': '0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C'}
)
fills = response.json()

# Filter for km:US500 only
us500_fills = [f for f in fills[:50] if f.get('coin') == 'km:US500']

# Calculate spread as in the strategy
buys = [(float(f['px']), float(f['sz'])) for f in us500_fills if f.get('side') == 'B']
sells = [(float(f['px']), float(f['sz'])) for f in us500_fills if f.get('side') == 'A']

print(f'km:US500 fills in last 50 (filtered): {len(buys)} buys, {len(sells)} sells')

if buys:
    buy_sum = sum(p*s for p,s in buys)
    buy_size = sum(s for p,s in buys)
    avg_buy = buy_sum / buy_size if buy_size > 0 else 0
    print(f'Avg buy price: ${avg_buy:.2f}, total size: {buy_size:.3f}')

if sells:
    sell_sum = sum(p*s for p,s in sells)
    sell_size = sum(s for p,s in sells)
    avg_sell = sell_sum / sell_size if sell_size > 0 else 0
    print(f'Avg sell price: ${avg_sell:.2f}, total size: {sell_size:.3f}')

if buys and sells:
    mid = (avg_buy + avg_sell) / 2
    spread_bps = (avg_sell - avg_buy) / mid * 10000 if mid > 0 else 0
    print(f'Spread: {spread_bps:.2f} bps')
    if spread_bps > 0:
        print('POSITIVE spread = Good (selling higher than buying)')
    else:
        print('NEGATIVE spread = Bad (buying higher than selling)')

