#!/usr/bin/env python3
"""Test order sizing for US500."""

def round_size(size, lot_size):
    from decimal import Decimal, ROUND_DOWN
    decimal_size = Decimal(str(size))
    decimal_lot = Decimal(str(lot_size))
    return float(decimal_size.quantize(decimal_lot, rounding=ROUND_DOWN))

# US500 parameters
symbol = "US500"
price = 693.0
min_size = 0.1  # szDecimals=1
lot_size = 0.1

# Example: $1000 collateral, 0.1 Kelly fraction -> $100 position
collateral = 1000
kelly_fraction = 0.10
base_size_usd = collateral * kelly_fraction  # $100
base_size = base_size_usd / price  # ~0.144 contracts

print(f"Collateral: ${collateral}")
print(f"Kelly fraction: {kelly_fraction*100}%")
print(f"Base size USD: ${base_size_usd}")
print(f"Base size contracts: {base_size:.6f}")
print(f"Rounded to lot size {lot_size}: {round_size(base_size, lot_size):.2f}")
print(f"Min size: {min_size}")
print(f"Final size: {max(round_size(base_size, lot_size), min_size):.2f}")
print(f"Notional value: ${max(round_size(base_size, lot_size), min_size) * price:.2f}")
