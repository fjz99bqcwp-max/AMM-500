# AMM-500 Deployment Guide

## Current Status: ✅ Ready for 7-Day Paper Trading

### Backtest Results (30-day, BTC)
All targets met with optimized parameters:

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Sharpe Ratio** | 2.18 | 1.5-3.0 | ✅ |
| **Ann ROI** | 59.1% | >5% | ✅ |
| **Max Drawdown** | 0.47% | <1% | ✅ |
| **Trades/Day** | 3017 | >500 | ✅ |
| **Win Rate** | 90.6% | - | ✅ |
| **Fill Rate** | 11.18% | - | ✅ |

**Performance Details**:
- Net PnL: $48.57 (on $1000 capital)
- Gross PnL: $58.10
- Fees Paid: $7.09
- Slippage Cost: $13.49
- Funding: -$1.67 (received)
- Rebates: $2.31
- Maker Volume: 85.1%

---

## Phase 1: Paper Trading (7 Days) ⏱️ NEXT STEP

**Objective**: Validate strategy on real mainnet BTC data with simulated orders (no funds needed).

### Setup

1. **Verify Connection**:
   ```bash
   python amm-500.py --paper --status
   ```
   
   Expected output:
   ```
   CONNECTION STATUS: OK
   Asset: BTC - Bitcoin Perpetual
   Orderbook (BTC):
     Best Bid: $91,127.00
     Best Ask: $91,128.00
     Spread: 0.11 bps
   ```

2. **Start Paper Trading**:
   ```bash
   # Run in background with logging
   nohup python amm-500.py --paper > logs/paper_$(date +%Y%m%d).log 2>&1 &
   
   # Or run in terminal (Ctrl+C to stop)
   python amm-500.py --paper
   ```

3. **Monitor**:
   ```bash
   # Watch logs
   tail -f logs/paper_*.log
   
   # Check status
   python scripts/quick_status.sh
   
   # View fills
   python scripts/monitor_fills.py
   ```

### Expected Behavior

**First Hour**:
- Places 18 order levels on each side (bid/ask)
- Rebalances every 3 seconds
- Fill rate: 10-20% of orders
- Trades: ~120-150/hour
- PnL: Small positive (+$0.10 to +$0.50)

**First Day**:
- Total trades: 2500-3500
- Net PnL: +$1-3
- Max DD: <1%
- Maker volume: >80%
- No position accumulation (delta-neutral)

**7 Days Target**:
- Total trades: 18,000-25,000
- Net PnL: +$7-15 (on $1000 capital)
- Sharpe: 1.5-2.5
- Max DD: <1%
- Win rate: >85%

### Monitoring Checklist

Daily checks:
- [ ] Bot is running (`ps aux | grep amm-500`)
- [ ] No position accumulation (should be delta-neutral)
- [ ] Max DD < 1%
- [ ] Fill rate 10-20%
- [ ] Maker volume > 80%
- [ ] No errors in logs (check for 429 rate limits)

Red flags (STOP bot if seen):
- ❌ Max DD > 2%
- ❌ Position size > $500 (5% of capital)
- ❌ Taker volume > 30%
- ❌ Consecutive losses > 50
- ❌ Rate limit errors (429)

### Paper Trading Results

After 7 days, analyze:
```bash
python scripts/analyze_live.py
```

**Success Criteria**:
- ✅ Sharpe > 1.5
- ✅ Ann ROI > 5%
- ✅ Max DD < 1%
- ✅ Trades/day > 500
- ✅ No crashes or errors
- ✅ Maker volume > 80%

If all pass → **Proceed to Phase 2**

---

## Phase 2: Live Trading - Small Scale (7 Days)

⚠️ **WARNING**: Real money at risk. Start small!

### Prerequisites
- ✅ Paper trading successful (7 days)
- ✅ Wallet funded on mainnet
- ✅ Risk limits understood
- ✅ Monitoring setup

### Wallet Setup

1. **Fund Wallet**:
   - Address: `0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C`
   - Network: Arbitrum One (Hyperliquid L1)
   - Asset: USDC
   - **Initial Amount**: $100-500 (start small!)

2. **Verify Balance**:
   ```bash
   python amm-500.py --status
   ```
   
   Should show:
   ```
   Account:
     Equity: $500.00
     Available: $500.00
   ```

### Risk Configuration

Update `config/.env` for live:
```bash
# Conservative live settings
LEVERAGE=3              # Lower than backtest (was 5x)
COLLATERAL=500          # Half of wallet balance
MIN_SPREAD_BPS=5        # Wider than backtest (was 4)
MAX_SPREAD_BPS=20       # Wider margins
ORDER_SIZE_FRACTION=0.01 # Smaller orders (was 0.015)
ORDER_LEVELS=12          # Fewer levels (was 18)
```

### Start Live Trading

```bash
# Remove --paper flag to go live
nohup python amm-500.py > logs/live_$(date +%Y%m%d).log 2>&1 &
```

### Daily Monitoring (CRITICAL)

Check **every 6 hours**:
- Current PnL
- Position size (should be near 0)
- Fill rate
- Max drawdown
- No anomalies in logs

**Emergency Stop**:
```bash
# Find PID
ps aux | grep amm-500

# Kill process
kill <PID>

# Close all positions manually via UI
# https://app.hyperliquid.xyz/trade/BTC
```

### Live Trading Results

After 7 days, compare to paper trading:
- PnL vs expected
- Sharpe ratio degradation (expect 20-40% lower than backtest)
- Fill rates (expect 5-10% lower than backtest)
- Slippage costs (expect 1.5-2x backtest)

**Success Criteria**:
- ✅ Positive net PnL
- ✅ Max DD < 2% (slightly higher tolerance)
- ✅ No major bugs/crashes
- ✅ Sharpe > 1.0 (lower than backtest)

If successful → **Scale up gradually**

---

## Phase 3: Scale Up (Gradual)

After successful live run, increase capital in steps:

1. **Week 1-2**: $500 capital, 3x leverage
2. **Week 3-4**: $1,000 capital, 4x leverage
3. **Week 5-8**: $2,000 capital, 5x leverage
4. **Beyond**: Up to $10,000 with proven track record

**Never exceed**:
- 10x leverage (risk of liquidation)
- 50% of total wallet balance
- $10k exposure without VPS deployment

---

## Phase 4: Production Deployment (Optional)

For 24/7 uptime and low latency:

### VPS Setup

Recommended providers:
- **Dwellir** (optimized for Hyperliquid, <50ms latency)
- **Chainstack** (Arbitrum nodes)
- **AWS/GCP** (us-east-1 region)

Specs:
- 4 vCPU
- 8 GB RAM
- 50 GB SSD
- Ubuntu 22.04

### Installation on VPS

```bash
# SSH into VPS
ssh user@your-vps-ip

# Clone repo
git clone https://github.com/fjz99bqcwp-max/AMM-500.git
cd AMM-500

# Setup Python
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy config (use scp from local)
scp config/.env user@your-vps-ip:~/AMM-500/config/

# Test connection
python amm-500.py --status

# Run with systemd (auto-restart on crash)
sudo cp scripts/amm-500.service /etc/systemd/system/
sudo systemctl enable amm-500
sudo systemctl start amm-500
```

### Monitoring

Set up Prometheus + Grafana:
```bash
# Metrics endpoint (port 8000)
curl http://localhost:8000/metrics

# Grafana dashboard
# Import: scripts/grafana_dashboard.json
```

Alerts (configure via webhook):
- Max DD > 2%
- Fill rate < 5%
- Consecutive losses > 100
- API errors

---

## Troubleshooting

### Common Issues

**1. Rate Limit (429 errors)**
```
ERROR: Rate limit exceeded
```
**Fix**: Increase cooldowns in `src/exchange.py`:
```python
_api_cooldowns = {
    "account_state": 120.0,  # Was 60s
    "orderbook": 60.0,       # Was 30s
}
```

**2. Insufficient Margin**
```
ERROR: Insufficient margin for order
```
**Fix**: Reduce leverage or order size in config

**3. High Slippage**
```
Slippage cost > $50/day
```
**Fix**: Widen min_spread or reduce order size

**4. Low Fill Rate (<5%)**
**Fix**: Tighten spreads or increase order levels

**5. Position Accumulation**
```
Position: 2.5 BTC (should be ~0)
```
**Fix**: Check rebalance logic in `src/strategy.py`, increase rebalance frequency

---

## Performance Expectations

### Realistic Live vs Backtest

| Metric | Backtest | Live (Expected) |
|--------|----------|-----------------|
| Sharpe | 2.18 | 1.3-1.8 |
| Ann ROI | 59% | 30-45% |
| Max DD | 0.47% | 0.8-1.5% |
| Trades/Day | 3017 | 2000-2500 |
| Fill Rate | 11% | 6-9% |

**Degradation factors**:
- Real slippage (1.5-2x backtest)
- API latency (100-300ms vs simulated 50-200ms)
- Queue position uncertainty
- Market impact on small cap
- Funding rate volatility

---

## Safety Checklist

Before going live:
- [ ] Paper trading ran successfully for 7 days
- [ ] Wallet funded with **only risk capital**
- [ ] Stop loss configured (<5% max DD)
- [ ] Monitoring alerts set up
- [ ] Emergency stop procedure tested
- [ ] Config backed up to safe location
- [ ] Rate limits understood (10 req/min)
- [ ] Position limits set (max $5k exposure initially)

**Never**:
- ❌ Trade with funds you can't afford to lose
- ❌ Use leverage >10x
- ❌ Deploy without testing paper mode first
- ❌ Ignore max drawdown alerts
- ❌ Run unmonitored for >24 hours initially

---

## Next Steps

1. **Now**: Start 7-day paper trading
   ```bash
   python amm-500.py --paper
   ```

2. **Day 3-4**: Review paper performance, check for issues

3. **Day 7**: Analyze results, decide go/no-go for live

4. **Day 8**: If successful, fund wallet ($100-500) and start live

5. **Day 15**: Review live performance, scale if positive

6. **Month 1**: Consider VPS deployment for 24/7 operation

---

## Contact & Support

- GitHub Issues: https://github.com/fjz99bqcwp-max/AMM-500/issues
- Hyperliquid Discord: https://discord.gg/hyperliquid
- Documentation: README.md

**Remember**: Start small, monitor closely, scale gradually. Good luck! 🚀
