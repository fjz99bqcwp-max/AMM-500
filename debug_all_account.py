#!/usr/bin/env python3
"""
Debug all account information from all possible API endpoints
"""
import requests
import os
import json
from dotenv import load_dotenv
load_dotenv('config/.env')

WALLET = os.getenv('WALLET_ADDRESS')
URL = "https://api.hyperliquid.xyz/info"

def debug_all_account_data():
    """Debug all possible account data"""
    print("🔍 COMPREHENSIVE ACCOUNT DEBUG")
    print("=" * 60)
    print(f"Wallet: {WALLET}")
    print("=" * 60)
    
    # 1. Check user state
    try:
        print("\n1️⃣ USER STATE")
        user_resp = requests.post(URL, json={"type": "userState", "user": WALLET}, timeout=10)
        print(f"Status: {user_resp.status_code}")
        if user_resp.status_code == 200:
            user_data = user_resp.json()
            print(f"Keys: {list(user_data.keys()) if isinstance(user_data, dict) else 'Not a dict'}")
            if isinstance(user_data, dict) and 'balances' in user_data:
                for bal in user_data['balances']:
                    print(f"  {bal}")
        else:
            print(f"Error: {user_resp.text}")
    except Exception as e:
        print(f"❌ User State Error: {e}")
    
    # 2. Check spot clearinghouse
    try:
        print("\n2️⃣ SPOT CLEARINGHOUSE STATE")
        spot_resp = requests.post(URL, json={"type": "spotClearinghouseState", "user": WALLET}, timeout=10)
        print(f"Status: {spot_resp.status_code}")
        if spot_resp.status_code == 200:
            spot_data = spot_resp.json()
            print(f"Keys: {list(spot_data.keys()) if isinstance(spot_data, dict) else 'Not a dict'}")
            if 'balances' in spot_data:
                for bal in spot_data['balances']:
                    print(f"  {bal}")
        else:
            print(f"Error: {spot_resp.text}")
    except Exception as e:
        print(f"❌ Spot Error: {e}")
    
    # 3. Check perp clearinghouse 
    try:
        print("\n3️⃣ PERP CLEARINGHOUSE STATE")
        perp_resp = requests.post(URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=10)
        print(f"Status: {perp_resp.status_code}")
        if perp_resp.status_code == 200:
            perp_data = perp_resp.json()
            print(f"Keys: {list(perp_data.keys()) if isinstance(perp_data, dict) else 'Not a dict'}")
            
            if 'marginSummary' in perp_data:
                margin = perp_data['marginSummary']
                print(f"  Margin Summary: {margin}")
            
            if 'assetPositions' in perp_data:
                print(f"  Asset Positions ({len(perp_data['assetPositions'])})")
                for i, pos in enumerate(perp_data['assetPositions']):
                    print(f"    {i}: {pos}")
                    
        else:
            print(f"Error: {perp_resp.text}")
    except Exception as e:
        print(f"❌ Perp Error: {e}")
    
    # 4. Check all positions
    try:
        print("\n4️⃣ ALL POSITIONS")
        pos_resp = requests.post(URL, json={"type": "allPositions", "user": WALLET}, timeout=10)
        print(f"Status: {pos_resp.status_code}")
        if pos_resp.status_code == 200:
            pos_data = pos_resp.json()
            print(f"Positions: {pos_data}")
        else:
            print(f"Error: {pos_resp.text}")
    except Exception as e:
        print(f"❌ Positions Error: {e}")

if __name__ == "__main__":
    debug_all_account_data()