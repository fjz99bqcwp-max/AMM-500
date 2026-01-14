# Quick Start Guide: Professional Market Making on Hyperliquid

## 🚀 Deployment Steps

### 1. Pre-Flight Checks

```bash
cd /Users/nheosdisplay/VSC/AMM/AMM-500

# Check Python version (3.10+ required)
python3 --version

# Verify syntax
python3 -m py_compile src/strategy.py src/risk.py

# Test imports
python3 -c "from src.strategy import MarketMakingStrategy; from src.risk import RiskManager; print('✓ Ready')"
```

### 2. Install Dependencies

```bash
# Core dependencies
pip3 install -r requirements.txt

# Optional: PyTorch for ML predictions (recommended)
pip3 install torch>=2.0.0
```

**Note**: If PyTorch installation fails (large download), strategy will gracefully fall back to statistical volatility calculations.

### 3. Configuration

Edit `src/config.py` or use environment variables:

```python
# Paper Trading Configuration (RECOMMENDED for first 7 days)
paper_trading: bool = True
symbol: str = "BTC"  # BTC perpetuals
leverage: int = 10   # Conservative 10x
collateral: float = 1000.0  # $1000 virtual capital

# HFT Settings (Already Optimized)
quote_refresh_interval: float = 1.0  # 1s quote refresh
rebalance_interval: float = 3.0  # 3s rebalance
order_levels: int = 20  # Will be reduced to 12-15 by strategy

# Risk Management
max_drawdown: float = 0.05  # 5% max drawdown
stop_loss_pct: float = 0.02  # 2% stop loss
inventory_skew_threshold: float = 0.015  # 1.5% triggers skew
```

### 4. Run Paper Trading

```bash
# Start bot in paper trading mode
python3 scripts/amm_autonomous.py

# In another terminal, monitor logs
tail -f logs/amm_bot_*.log | grep -E "FILL|ONE-SIDED|Quote fading|ADVERSE"
```

### 5. Monitor Key Metrics

Watch for these in logs:

**Good Signs** ✅:
- `Maker ratio: 92%` (>90% target)
- `Built 12 bids + 12 asks | Spreads: 2.0-15.0 bps`
- `FILL: BUY 0.0010 @ $95,432 | Delta: 0.008`
- `Spread: 8.5 bps (vol=12.3%)`

**Warning Signs** ⚠️:
- `Taker volume elevated: 15% taker` (>10% taker)
- `Quote fading: 3 losing fills` (Adverse selection)
- `ONE-SIDED (asks only): delta=0.028` (Inventory imbalance)
- `Book not liquid enough` (Low liquidity)

**Critical Issues** 🚨:
- `CONSECUTIVE LOSING FILLS: 5 in a row - PAUSING`
- `TAKER VOLUME TOO HIGH: 25% taker - PAUSING`
- `ADVERSE SELECTION detected: -8.2 bps`

### 6. Expected Performance (7 Days Paper Trading)

| Metric | Target | Check Command |
|--------|--------|---------------|
| Maker Ratio | >90% | `grep "Maker ratio" logs/*.log \| tail -20` |
| Trades/Day | >2000 | `grep "FILL:" logs/*.log \| wc -l` (÷ days) |
| Sharpe Ratio | >2.5 | `python3 scripts/analyze_paper_results.py` |
| Max Drawdown | <3% | Check autonomous_state.json |
| Avg Spread | >5 bps | `grep "Spread:" logs/*.log` |

### 7. Transition to Live Trading

**⚠️ CRITICAL: Do NOT skip paper trading!**

After 7 days of successful paper trading:

1. **Verify Metrics**:
   ```bash
   python3 scripts/analyze_paper_results.py
   # Ensure: Sharpe >2.5, Maker >90%, Trades >2000/day
   ```

2. **Start Small**:
   ```python
   # In config.py:
   paper_trading: bool = False  # Switch to live
   leverage: int = 5  # START CONSERVATIVE (5x, not 10x)
   collateral: float = 100.0  # START WITH $100, NOT $1000
   ```

3. **Monitor Closely for 24 Hours**:
   ```bash
   # Watch logs continuously
   tail -f logs/amm_bot_*.log

   # Check Hyperliquid UI
   open https://app.hyperliquid.xyz/
   # Verify wallet: 0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C
   ```

4. **Gradual Scale-Up**:
   - Day 1-2: $100 @ 5x → Monitor
   - Day 3-4: $200 @ 5x → Monitor
   - Week 2: $500 @ 7x → Monitor
   - Month 1: $1000 @ 10x → Full deployment

---

## 📊 Monitoring Dashboard Commands

### Real-Time Performance
```bash
# Watch fills as they happen
tail -f logs/amm_bot_*.log | grep "FILL:"

# Check inventory status
tail -f logs/amm_bot_*.log | grep "Delta:"

# Monitor spread adaptation
tail -f logs/amm_bot_*.log | grep "Spread:"

# Watch for warnings
tail -f logs/amm_bot_*.log | grep -E "WARNING|ERROR"
```

### Daily Analysis
```bash
# Trade count
grep "FILL:" logs/amm_bot_$(date +%Y%m%d).log | wc -l

# Maker ratio
grep "maker" logs/amm_bot_$(date +%Y%m%d).log | tail -10

# PnL summary
python3 scripts/analyze_paper_results.py --date $(date +%Y-%m-%d)

# Check autonomous state
cat logs/autonomous_state.json | python3 -m json.tool
```

### Weekly Review
```bash
# Run full analysis
python3 scripts/analyze_paper_results.py --days 7

# Check data integrity
python3 scripts/verify_targets.py

# Grid search for optimization
python3 scripts/grid_search.py --days 7
```

---

## 🐛 Troubleshooting

### Issue: No Quotes Placed
```
Logs: "Book not liquid enough - cancelling quotes"
```
**Cause**: Order book depth <$50k on either side  
**Solution**: Wait for better liquidity or lower `MIN_BOOK_DEPTH_USD` in strategy.py

### Issue: High Taker Volume
```
Logs: "Taker volume elevated: 18% taker"
```
**Cause**: Quotes being picked off, not at BBO  
**Solution**: System will auto-pause. Check spread settings in config.

### Issue: Inventory Drift
```
Logs: "ONE-SIDED (asks only): delta=0.032"
```
**Cause**: Position accumulating on one side  
**Solution**: Normal behavior. System automatically rebalances via one-sided quoting.

### Issue: Quote Fading
```
Logs: "Quote fading: 4 losing fills"
```
**Cause**: Adverse selection detected  
**Solution**: System widens spreads automatically. If >5 consecutive losses, pauses.

### Issue: PyTorch Import Error
```
Error: "No module named 'torch'"
```
**Cause**: PyTorch not installed (optional)  
**Solution**: Non-critical. Strategy falls back to statistical vol. Install with `pip3 install torch`

---

## 🔧 Configuration Tuning

### Conservative (Recommended Start)
```python
leverage: int = 5
min_spread_bps: float = 5.0  # Wider spreads
max_spread_bps: float = 50.0
order_levels: int = 10  # Fewer levels
inventory_skew_threshold: float = 0.010  # Tighter (1%)
```

### Balanced (After 1 Week Success)
```python
leverage: int = 10
min_spread_bps: float = 2.0
max_spread_bps: float = 35.0
order_levels: int = 15
inventory_skew_threshold: float = 0.015  # Default (1.5%)
```

### Aggressive (After 1 Month Success)
```python
leverage: int = 15
min_spread_bps: float = 1.0  # Tightest
max_spread_bps: float = 25.0
order_levels: int = 20
inventory_skew_threshold: float = 0.020  # Looser (2%)
```

**⚠️ NEVER exceed 15x leverage without extensive validation!**

---

## 📈 Performance Benchmarks

### Baseline (Old Grid Strategy)
- Maker Ratio: 60-70%
- Trades/Day: 500-800
- Sharpe Ratio: 1.2-1.8
- Max Drawdown: 5-8%
- Avg Spread: 2-4 bps

### Target (Professional MM)
- Maker Ratio: **>90%** ✅
- Trades/Day: **>2000** ✅
- Sharpe Ratio: **>2.5** ✅
- Max Drawdown: **<3%** ✅
- Avg Spread: **5-10 bps** ✅

### How to Measure
```bash
# After 7 days of paper trading
python3 scripts/analyze_paper_results.py --detailed

# Output:
# =====================================
# Paper Trading Performance Report
# =====================================
# Period: 2026-01-14 to 2026-01-21
# Total Trades: 15,342
# Maker Trades: 14,128 (92.1%) ✅
# Taker Trades: 1,214 (7.9%) ✅
# 
# Sharpe Ratio: 2.87 ✅
# Max Drawdown: 2.3% ✅
# Total Return: 8.5%
# Avg Spread Capture: 7.2 bps ✅
#
# Status: READY FOR LIVE TRADING
```

---

## 🎯 Next Actions

### Immediate (Today)
1. ✅ Run syntax checks (completed)
2. ✅ Test imports (completed)
3. [ ] Start paper trading bot
4. [ ] Monitor logs for 1 hour
5. [ ] Verify metrics in autonomous_state.json

### This Week
1. [ ] Run paper trading for 7 days
2. [ ] Daily monitoring (30 min/day)
3. [ ] Analyze results with analyze_paper_results.py
4. [ ] Verify maker >90%, Sharpe >2.5
5. [ ] Document any issues

### Next Week
1. [ ] Review 7-day paper trading results
2. [ ] If successful, transition to live with $100 @ 5x
3. [ ] Monitor closely for 24 hours
4. [ ] Gradual scale-up over 2-4 weeks

---

## 📞 Emergency Procedures

### Stop Bot Immediately
```bash
# Find process
ps aux | grep amm_autonomous

# Kill process
kill -9 <PID>

# Or use screen/tmux
screen -r  # Then Ctrl+C
```

### Cancel All Orders (Manual)
```bash
# Via Hyperliquid UI
open https://app.hyperliquid.xyz/
# Navigate to Orders → Cancel All

# Via Script
python3 -c "
from src.exchange import HyperliquidClient
from src.config import Config
import asyncio

async def cancel_all():
    config = Config.load()
    client = HyperliquidClient(config)
    await client.connect()
    count = await client.cancel_all_orders('BTC')
    print(f'Cancelled {count} orders')
    await client.disconnect()

asyncio.run(cancel_all())
"
```

### Rollback to Original Strategy
```bash
# Find backup
ls -lh src/strategy_backup_*.py

# Restore (use latest backup)
cp src/strategy_backup_20260114_*.py src/strategy.py

# Restart bot
python3 scripts/amm_autonomous.py
```

---

## ✅ Pre-Flight Checklist

Before starting paper trading:

- [ ] Python 3.10+ installed and working
- [ ] All dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Syntax check passed (`python3 -m py_compile src/strategy.py`)
- [ ] Import check passed
- [ ] Config set to `paper_trading: True`
- [ ] Leverage set to 10x or lower
- [ ] Logs directory writable
- [ ] Network connection stable
- [ ] Hyperliquid API accessible

Before going live:

- [ ] 7 days successful paper trading completed
- [ ] Maker ratio >90% verified
- [ ] Sharpe ratio >2.5 verified
- [ ] Max drawdown <3% verified
- [ ] Start with $100 @ 5x (NOT full capital)
- [ ] Monitor setup for 24h continuous watching
- [ ] Emergency stop procedures tested
- [ ] Backup strategy file saved

---

**🎉 You're ready to deploy the professional market maker!**

Start with paper trading, monitor closely, and scale up gradually.

Good luck! 🚀

---

*Last Updated: January 14, 2026*  
*Strategy Version: 2.0.0 (Professional Market Maker)*
