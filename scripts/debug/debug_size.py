price = 694.75
min_size = max(0.00012, 10.0 / price)
print(f'Price: {price}')
print(f'Min size: {min_size:.6f} contracts')
print(f'Notional: ${min_size * price:.2f}')

# Check what the strategy might be calculating
equity = 1469.11
max_exposure = 25000
leverage = 5
collateral = 1000

# Risk manager calc
collateral_pct = (collateral / equity) if equity > 0 else 0
kelly_fraction = 0.15  # 15% Kelly
kelly_size = collateral_pct * kelly_fraction / (0.15 ** 2)  # kelly formula with default vol

print(f'\nStrategy calc:')
print(f'Collateral: ${collateral}')
print(f'Equity: ${equity}')
print(f'Kelly calc: {kelly_size:.6f} (likely too small)')

# Try typical strategy sizing
typical_size_pct = 0.05  # 5% of exposure
typical_size = (max_exposure * typical_size_pct) / price / leverage
print(f'Typical sizing (5% of max): {typical_size:.6f} contracts')
