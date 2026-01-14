#!/usr/bin/env python3
"""Check US500 asset metadata from Hyperliquid."""
from hyperliquid.info import Info

info = Info(perp_dexs=['km'], skip_ws=True)
meta = info.meta()

print("Looking for US500 in universe...")
for asset in meta.get('universe', []):
    name = asset.get('name', '').upper()
    if 'US500' in name or 'SP' in name or '500' in name:
        print(f"\nFound: {asset.get('name')}")
        for k, v in asset.items():
            print(f"  {k}: {v}")

# Also check meta_and_asset_ctxs
print("\n\nChecking metaAndAssetCtxs...")
try:
    mac = info.meta_and_asset_ctxs()
    for asset_ctx in mac:
        if isinstance(asset_ctx, list):
            for a in asset_ctx:
                if isinstance(a, dict):
                    name = a.get('name', '').upper()
                    if 'US500' in name or 'SP' in name:
                        print(f"\nFound: {a}")
        elif isinstance(asset_ctx, dict):
            name = asset_ctx.get('name', '').upper()
            if 'US500' in name or 'SP' in name:
                print(f"\nFound: {asset_ctx}")
except Exception as e:
    print(f"Error: {e}")

# Try spot meta too
print("\n\nChecking spot_meta...")
try:
    spot = info.spot_meta()
    for token in spot.get('tokens', []):
        if 'US500' in str(token).upper():
            print(f"\nFound spot: {token}")
except Exception as e:
    print(f"Error: {e}")
