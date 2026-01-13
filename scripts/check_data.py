#!/usr/bin/env python3
"""Check data quality"""
import json
import math

with open('data/us500_synthetic_180d.json') as f:
    candles = json.load(f)

prices = [c['c'] for c in candles]
print(f'Price range: {min(prices):.2f} - {max(prices):.2f}')
print(f'First 10 prices: {[round(p,2) for p in prices[:10]]}')

returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
print(f'First 10 returns: {[round(r*100, 4) for r in returns[:10]]}%')
print(f'Max return: {max(returns)*100:.4f}%')
print(f'Min return: {min(returns)*100:.4f}%')

daily_returns = []
for i in range(0, len(prices)-1440, 1440):
    daily_ret = (prices[i+1440] - prices[i]) / prices[i]
    daily_returns.append(daily_ret)

if daily_returns:
    mean_daily = sum(daily_returns) / len(daily_returns)
    var_daily = sum((r - mean_daily)**2 for r in daily_returns) / len(daily_returns)
    vol_daily = math.sqrt(var_daily)
    print(f'Actual daily volatility: {vol_daily*100:.2f}%')
    print(f'Annualized: {vol_daily * math.sqrt(252) * 100:.1f}%')
