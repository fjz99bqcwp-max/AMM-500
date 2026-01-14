price = 694.75
equity = 1469.11
collateral = 1000

# Min size calculation  
min_size = max(0.00012, 10.0 / price)
print(f"Minimum size for US500 at ${price}: {min_size:.8f} contracts")
print(f"Minimum notional: ${min_size * price:.2f}")

# Kelly sizing - if no historical trades
# Initial kelly_size from _calculate_kelly_dynamic_size is likely very small
vol = 0.15  # 15% annual vol (default)
collateral_pct = (collateral / equity) if equity > 0 else 0
kelly_fraction = 0.15  # 15% Kelly
# kelly = (edge) / vol^2, but edge is 0 initially
kelly_size_initial = 0  # No edge yet

print(f"\nInitial Kelly sizing (no edge yet):")
print(f"  Kelly size: {kelly_size_initial:.8f} (because no historical edge)")
print(f"  After max(min_size, kelly_size): {max(min_size, kelly_size_initial):.8f}")

# So the final size should be min_size = 0.0144 contracts
final_size = max(min_size, kelly_size_initial)
print(f"\nFinal order size: {final_size:.8f} contracts")
print(f"Final notional: ${final_size * price:.2f}")

# But if it's being rounded...
from decimal import Decimal

def round_size(size, lot_size):
    """Round size to lot_size precision."""
    if size <= 0:
        return 0
    decimal_size = Decimal(str(size))
    decimal_lot = Decimal(str(lot_size))
    rounded = (decimal_size / decimal_lot).quantize(0)  * decimal_lot
    return float(rounded)

# LOT_SIZE for exchange
LOT_SIZE = 0.0001

rounded = round_size(final_size, LOT_SIZE)
print(f"\nAfter rounding to LOT_SIZE={LOT_SIZE}:")
print(f"  Rounded size: {rounded:.8f} contracts")
print(f"  Rounded notional: ${rounded * price:.2f}")
