# USDH Transfer Guide: Spot → Perps

## Current Situation
- **Spot Balance**: $1,469.10 USDH ✅
- **Perps Balance**: $0.00 ❌ (need funds here to trade)

## Why Manual Transfer is Needed
The `usdClassTransfer` API action requires signing with the **main wallet's private key**, but you only have the **API wallet** configured (which is used for trading operations, not transfers).

## Manual Transfer Steps

### Step 1: Go to Hyperliquid
Open: https://app.hyperliquid.xyz

### Step 2: Connect Your Wallet
- Connect wallet: `0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C`
- This should be your MetaMask or hardware wallet

### Step 3: Transfer USDH
1. Click the **"Transfer"** button (usually top right)
2. Select:
   - **From**: Spot
   - **To**: Perps
   - **Asset**: USDH
   - **Amount**: 1469.10 (or click "Max")
3. Click **"Transfer"**
4. Confirm the transaction in your wallet

### Step 4: Verify Transfer
After the transaction confirms (usually a few seconds):

```bash
cd /Users/nheosdisplay/VSC/AMM/AMM-500
python check_balance.py
```

You should see:
- Spot: ~$0
- **Perps: ~$1,469** ✅

### Step 5: Bot Will Auto-Start
Once funds are in Perps, the autonomous monitor (currently running) will automatically detect the balance and start placing orders on km:US500!

## Alternative: Programmatic Transfer (Advanced)

If you want to automate this in the future, you would need to:

1. **Add main wallet private key** to config (NOT recommended for security)
2. Or use **MetaMask/WalletConnect SDK** to sign the transfer
3. Or use the **Hyperliquid SDK with main wallet**

For now, the manual UI transfer is the safest and easiest option.

## After Transfer

Monitor the bot:
```bash
# Check if bot detected funds
tail -f logs/autonomous_state.json

# Check US500 candles being collected
tail -f data/us500_candles_1m.csv

# Watch bot logs
tail -f logs/bot_$(date +%Y-%m-%d).log
```

The bot should start trading within 5 minutes of funds appearing in Perps!
