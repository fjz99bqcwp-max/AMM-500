# AMM-500 Deployment Guide - 7-Day Paper Trading
**Date**: January 15, 2026  
**Platform**: Hyperliquid US500-USDH (HIP-3)  
**Wallet**: 0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C  
**Hardware**: Apple M4 Mac mini (10 cores, 24GB RAM)

---

## 📋 Pre-Deployment Checklist

### ✅ Code Implementations Complete
- [x] Smart orderbook placement ([strategy.py#L1377](../src/core/strategy.py#L1377))
- [x] Reduce-only mode ([strategy.py#L1444](../src/core/strategy.py#L1444))
- [x] USDH margin queries ([exchange.py#L1879](../src/core/exchange.py#L1879))
- [x] xyz100 primary integration ([data_fetcher.py#L337](../src/utils/data_fetcher.py#L337))
- [x] PyTorch vol predictor enabled by default
- [x] M4 parallel order placement (10 cores)
- [x] `--reduce-only` CLI flag added to amm-500.py

### ✅ Configuration Verified
- [x] SYMBOL=US500 (HIP-3 USDH perpetuals)
- [x] LEVERAGE=10 (conservative, max 25x)
- [x] MIN_SPREAD_BPS=3 (US500 optimized)
- [x] USDH_MARGIN_CAP=0.90 (90% enforcement)
- [x] 15+ new parameters added (smart placement, reduce-only, xyz100)

### ✅ Syntax Validation
- [x] No errors in strategy.py, exchange.py, data_fetcher.py, amm-500.py
- [x] All Python files compile successfully

### ✅ Documentation
- [x] Comprehensive README.md created
- [x] OPTIMIZATION_SUMMARY.md → docs/
- [x] Root directory organized (removed README_OLD.md)

---

## 🚀 7-Day Paper Trading Setup

### Step 1: Environment Verification
```bash
cd /Users/nheosdisplay/VSC/AMM/AMM-500
source .venv/bin/activate

# Verify Python version
python --version  # Should be 3.10+

# Verify dependencies
pip list | grep -E "(hyperliquid|torch|yfinance|loguru)"
```

### Step 2: Configuration Check
```bash
# Verify critical parameters
cat config/.env | grep -E "^(SYMBOL|LEVERAGE|PRIVATE_KEY|PAPER_TRADING)"

# Expected output:
# SYMBOL=US500
# LEVERAGE=10
# PRIVATE_KEY=0x...
# PAPER_TRADING=true  (or set via --paper flag)
```

### Step 3: Fetch Historical Data
```bash
# Fetch data (automatically uses xyz100 PRIMARY, falls back to US500/BTC)
python amm-500.py --fetch-data --fetch-days 30

# Verify data quality
ls -lh data/*.csv
# Should see: xyz100_proxy.csv, xyz100_scaled.csv, BTC_candles_1m_30d.csv

# The data_fetcher.py automatically prioritizes:
# 1. xyz100 (S&P 100 via yfinance) - PRIMARY
# 2. US500 direct (if >50% bars available)
# 3. BTC scaled (fallback)
```

### Step 4: Start Paper Trading (7 Days)
```bash
# Option 1: Interactive script
./scripts/automation/start_paper_trading.sh
# Select: 1 (Paper), 1 (7 days)

# Option 2: Direct command
python amm-500.py --paper --duration 7d

# Option 3: Background with nohup
nohup python amm-500.py --paper > logs/paper_trading_7d.log 2>&1 &
echo $! > logs/paper_trading.pid
```

### Step 5: Monitor Real-Time
```bash
# Tail logs for key events
tail -f logs/bot_$(date +%Y-%m-%d).log | grep -E "(FILL|L2 Analysis|reduce-only|USDH margin)"

# Watch for:
# - "✅ Smart placement: Found gap at 6905.50" (orderbook analysis)
# - "⚠️ Reduce-only mode ACTIVE - USDH margin 82%" (risk management)
# - "FILL: BUY 0.5 @ 6900.00 (maker)" (successful fills)
# - "USDH margin ratio: 0.35 (35%)" (margin monitoring)
```

### Step 6: Daily Health Checks
```bash
# Check bot status
python amm-500.py --status

# Query open orders
python scripts/check_open_orders.py

# Analyze performance so far
python scripts/analysis/analyze_paper_results.py --days-so-far
```

---

## 📊 Target Metrics (7-Day Validation)

| Metric | Target | Measurement | Gate |
|--------|--------|-------------|------|
| **Sharpe Ratio** | >2.5 | Risk-adjusted returns | Primary |
| **7-Day ROI** | >5% | ~260% annualized | Primary |
| **Max Drawdown** | <0.5% | Capital preservation | Critical |
| **Trades/Day** | >2000 | HFT frequency | Important |
| **Maker Ratio** | >90% | Fee optimization | Important |
| **Reduce-Only %** | 10-20% | Position management | Validation |
| **Smart Fill Rate** | >15% | Orderbook advantage | Validation |
| **USDH Margin Peak** | <85% | Risk control | Critical |

### Expected PnL (Baseline)
```
Capital: $1000
Expected 7-Day Return: $50-75 (5-7.5%)
Total Fills: 14,000-20,000
Maker Fills: 12,600-18,000 (>90%)
Reduce-Only Triggers: 50-150 times
Smart Placements: 2,100-4,000 (15-20% advantage)
```

---

## 📈 Post-Trading Analysis

### After 7 Days Complete
```bash
# Generate comprehensive analysis
python scripts/analysis/analyze_paper_results.py --days 7 --metrics all --export-csv

# Key reports generated:
# - logs/paper_trading_summary_7d.json
# - logs/paper_trading_metrics.csv
# - logs/trade_log.json (all fills)

# Analyze specific features
python scripts/analysis/verify_targets.py --check-smart-placement
python scripts/analysis/verify_targets.py --check-reduce-only
python scripts/analysis/verify_targets.py --check-usdh-margin
```

### Validation Criteria
```
✅ PASS Criteria (proceed to live deployment):
- Sharpe >2.5
- ROI >5%
- Drawdown <0.5%
- Trades >14,000
- Maker ratio >90%
- No USDH margin >90% events
- <5 consecutive losing days

⚠️ REVIEW Criteria (extend paper trading):
- Sharpe 2.0-2.5 (acceptable but not optimal)
- ROI 3-5% (lower than target)
- Drawdown 0.5-1% (review risk parameters)
- Maker ratio 85-90% (tighten spreads)

❌ FAIL Criteria (revise strategy):
- Sharpe <2.0
- ROI <3%
- Drawdown >1%
- Trades <10,000/week
- Maker ratio <85%
- USDH margin >90% occurred
```

---

## 🎯 Phase 3: Live Deployment (After Validation)

### Step 1: Low Capital Test ($100-500)
```bash
# Update config/.env
COLLATERAL=100  # Start small
LEVERAGE=10     # Conservative

# Remove paper mode
PAPER_TRADING=false

# Start live
python amm-500.py
```

### Step 2: Monitor 24 Hours
- Check every 1-2 hours for first 8 hours
- Monitor USDH margin continuously
- Verify fills are as expected (maker >90%)
- Watch for reduce-only triggers

### Step 3: Scale Gradually
```
Day 1-3: $100-500 (validation)
Day 4-7: $500-1000 (confidence building)
Day 8-30: $1000-5000 (proven performance)
```

---

## 🚨 Emergency Procedures

### High USDH Margin (>85%)
```bash
# Force reduce-only mode
python amm-500.py --reduce-only

# Or update config
echo "AUTO_REDUCE_ONLY=true" >> config/.env
echo "REDUCE_ONLY_MARGIN=0.75" >> config/.env  # Lower threshold

# Restart bot
pkill -f "python.*amm-500"
python amm-500.py
```

### High Drawdown (>2%)
```bash
# Stop bot immediately
pkill -f "python.*amm-500"

# Cancel all orders
python scripts/cancel_orders.py --symbol US500

# Review logs
tail -200 logs/bot_$(date +%Y-%m-%d).log

# Analyze what went wrong
python scripts/analysis/analyze_live_trading.py --last-24h
```

### Bot Crash/Restart
```bash
# Check if running
ps aux | grep "amm-500.py"

# View crash logs
tail -100 logs/bot_$(date +%Y-%m-%d).log | grep -E "(ERROR|CRITICAL)"

# Restart with autonomous monitoring
python scripts/automation/amm_autonomous.py &
```

---

## 📝 Monitoring Commands Reference

### Real-Time Monitoring
```bash
# Overall bot activity
tail -f logs/bot_$(date +%Y-%m-%d).log

# Filter for fills only
tail -f logs/bot_$(date +%Y-%m-%d).log | grep "FILL"

# Filter for risk events
tail -f logs/bot_$(date +%Y-%m-%d).log | grep -E "(reduce-only|USDH margin|drawdown)"

# Filter for smart placement
tail -f logs/bot_$(date +%Y-%m-%d).log | grep -E "(Smart placement|liquidity gap|queue join)"
```

### Performance Checks
```bash
# Check Sharpe ratio so far
python scripts/analysis/analyze_paper_results.py --sharpe-only

# Check current PnL
python scripts/analysis/analyze_paper_results.py --pnl-only

# Check maker ratio
python scripts/analysis/analyze_paper_results.py --maker-ratio

# Full metrics dump
python scripts/analysis/analyze_paper_results.py --days-so-far --metrics all
```

---

## 🎉 Success Indicators

### What Good Looks Like (After 7 Days)
- ✅ Bot running continuously without crashes
- ✅ Sharpe ratio >2.5 (risk-adjusted profitability)
- ✅ $50-75 profit on $1000 (5-7.5% return)
- ✅ 14,000-20,000 total fills (consistent HFT activity)
- ✅ >90% maker ratio (fee optimization working)
- ✅ Reduce-only mode triggered 50-150 times (smart risk management)
- ✅ Smart placement used 15-20% of time (orderbook advantage)
- ✅ Max USDH margin <85% (well within safety limits)
- ✅ No losing days >$20
- ✅ Drawdown <0.5% throughout

### Red Flags to Watch
- ⚠️ Taker ratio >10% (spreads too tight, getting crossed)
- ⚠️ Consecutive losses >20 (market conditions changed)
- ⚠️ USDH margin peaks >90% (leverage too high)
- ⚠️ Sharpe <2.0 (poor risk-adjusted returns)
- ⚠️ ROI <3% (underperforming)
- ⚠️ Bot restarts >3 times/day (stability issues)
- ⚠️ Reduce-only mode active >50% of time (too conservative)

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: Bot not placing orders
- Check PRIVATE_KEY in config/.env
- Verify USDH margin <90%
- Check if reduce-only mode active (will only close)
- Ensure orderbook data available

**Issue**: High taker ratio (>10%)
- Increase MIN_SPREAD_BPS (3 → 5)
- Reduce QUOTE_REFRESH_INTERVAL (3s → 5s)
- Check orderbook depth (may be too thin)

**Issue**: Reduce-only mode always active
- Check REDUCE_ONLY_MARGIN threshold (lower to 0.70)
- Verify inventory skew not too high
- Review REDUCE_ONLY_SKEW parameter (increase to 0.02)

**Issue**: xyz100 data insufficient
- Verify yfinance installed: `pip install yfinance`
- Check internet connection
- Fallback to BTC: `python amm-500.py --fetch-data --days 30`
- Review XYZ100_MIN_BARS parameter

---

## ✅ Deployment Timeline

### Day 0 (Today): Final Preparation
- [x] Code implementations complete
- [x] Configuration verified
- [x] Documentation updated
- [x] Syntax validation passed

### Day 1-7: Paper Trading
- [ ] Start 7-day paper trading
- [ ] Monitor daily (health checks)
- [ ] Collect performance metrics
- [ ] Analyze results after completion

### Day 8: Analysis & Decision
- [ ] Generate comprehensive analysis report
- [ ] Validate against target metrics
- [ ] Decision: Proceed to live or extend paper trading

### Day 9+ (If Validated): Live Deployment
- [ ] Start with $100-500 USDH
- [ ] Monitor 24 hours continuously
- [ ] Scale gradually to $1000-5000
- [ ] Deploy autonomous monitoring

---

**Status**: Ready for 7-day paper trading validation

**Next Action**: Run `python amm-500.py --paper` to start paper trading

**Documentation**: See [README.md](../README.md) for full setup guide

---

*Generated: January 15, 2026 | Updated: Ready for deployment*
