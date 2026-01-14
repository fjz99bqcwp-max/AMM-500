# AMM-500: High-Frequency Market Making Bot

A high-frequency trading bot for delta-neutral market making on **BTC** perpetuals on Hyperliquid. Clean, optimized, and ready for autonomous operation.

> ⚠️ **WARNING: HIGH RISK**  
> This is a leveraged trading bot. You can lose your entire investment. Always:
> - Test on paper trading first (7 days minimum)
> - Start with small amounts ($1000 recommended)
> - Enable autonomous monitoring with kill switches
> - Understand the strategy before using real money
> - Never risk more than you can afford to lose

**Performance (30-day backtest, 10x leverage):**  
Sharpe 2.18 | Annual ROI 59.1% | Max DD 0.47% | 3000+ trades/day

**Documentation:**
- 📘 [HFT Optimization Guide](HFT_OPTIMIZATION_GUIDE.md) - Comprehensive HFT recommendations
- 🤖 [Autonomous Setup Guide](AUTONOMOUS_SETUP_GUIDE.md) - 24/7 monitoring setup
- 📊 [Cleanup Summary](CLEANUP_OPTIMIZATION_SUMMARY.md) - Recent optimizations

## Quick Start

```bash
# 1. Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (add your wallet keys)
cp config/.env.example config/.env
nano config/.env

# 3. Fetch BTC historical data
python amm-500.py --fetch-data --fetch-days 180

# 4. Run 30-day backtest
python amm-500.py --backtest --days 30

# 5. Start 7-day paper trading (recommended)
./scripts/start_paper_trading.sh
# Select: 1 (Paper Trading), 1 (7 days)

# 6. Analyze results after 7 days
python scripts/analyze_paper_results.py

# 7. Go live (after successful paper trading)
./scripts/start_paper_trading.sh
# Select: 2 (Live Trading), type "LIVE" to confirm
```

## Essential Commands

```bash
# Activate environment
source .venv/bin/activate

# Paper trading (7-day session with monitoring)
./scripts/start_paper_trading.sh

# Manual paper trading
python amm-500.py --paper

# Live trading (REAL MONEY - use with caution!)
python amm-500.py

# Backtest (30 days)
python amm-500.py --backtest --days 30

# Analyze paper trading results
python scripts/analyze_paper_results.py

# Check connection
python amm-500.py --status

# Monitor logs
tail -f logs/bot_$(date +%Y-%m-%d).log
tail -f logs/autonomous_v3.log

# Stop bot
pkill -f "python.*amm-500"
pkill -f "python.*autonomous"
```

## Autonomous Monitoring

The bot includes comprehensive 24/7 monitoring with `amm_autonomous_v3.py`:

**Features:**
- ✅ Real-time wallet tracking (equity, PnL, margin, positions)
- ✅ Auto-restart on crash (max 5/hour)
- ✅ Email & Slack alerts (DD>2%, Taker>30%, Loss>$50)
- ✅ Kill switches (DD>5%, 10 losses, $100 loss)
- ✅ Trade metrics (maker/taker ratio, fees, PnL)
- ✅ State persistence (survives restarts)

**Start monitoring:**
```bash
python scripts/amm_autonomous_v3.py
# Monitors every 5 minutes
```

**Configure alerts in `config/.env`:**
```env
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ALERT_EMAIL=your-alert-email@gmail.com
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK
```

See [AUTONOMOUS_SETUP_GUIDE.md](AUTONOMOUS_SETUP_GUIDE.md) for complete setup.

## Architecture

```
AMM-500/
├── amm-500.py                    # Main entry point
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── HFT_OPTIMIZATION_GUIDE.md    # HFT recommendations
├── AUTONOMOUS_SETUP_GUIDE.md    # Monitoring setup
├── config/
│   ├── .env                      # Your credentials (not in git)
│   └── .env.example              # Configuration template
├── src/                          # Core trading logic
│   ├── config.py                 # Configuration management
│   ├── exchange.py               # Hyperliquid client
│   ├── strategy.py               # Market making strategy
│   ├── risk.py                   # Risk management
│   ├── backtest.py               # Backtesting framework
│   ├── data_fetcher.py           # Data fetching
│   ├── metrics.py                # Prometheus metrics
│   └── utils.py                  # Utilities
├── scripts/                      # Essential scripts (8 total)
│   ├── amm_autonomous_v3.py      # Enhanced monitoring
│   ├── start_paper_trading.sh    # Interactive launcher
│   ├── analyze_paper_results.py  # Performance analysis
│   ├── fetch_real_btc.py         # BTC data fetcher
│   ├── grid_search.py            # Parameter optimization
│   └── verify_targets.py         # Target validator
├── tests/                        # Unit tests
├── data/                         # Historical data (BTC)
└── logs/                         # Trading logs
```

## Performance Targets

| Metric | Current | Optimized (HFT) | How to Achieve |
|--------|---------|-----------------|----------------|
| **Sharpe Ratio** | 2.18 | 2.8-3.2 | Tighter spreads, 1s rebalance, ML vol prediction |
| **Annual ROI** | 59% | 80-95% | Higher turnover, better spread capture |
| **Max Drawdown** | 0.47% | 0.3-0.4% | Faster delta-neutral, stricter risk |
| **Trades/Day** | 3000+ | 4500-5500 | 1 bps spreads, 0.5s quote refresh |
| **Maker Ratio** | ~88% | >95% | ALO orders only, taker cap <10% |

See [HFT_OPTIMIZATION_GUIDE.md](HFT_OPTIMIZATION_GUIDE.md) for implementation details.

## Configuration

**Required in `config/.env`:**
```env
# Wallet credentials
PRIVATE_KEY=your_wallet_private_key
WALLET_ADDRESS=0xYourWalletAddress

# Trading parameters
TESTNET=False              # Use mainnet (paper mode still simulates)
LEVERAGE=10                # 10x recommended, max 50x for BTC
COLLATERAL=1000            # Starting capital in USD
MIN_SPREAD_BPS=5           # Minimum 5 bps (HFT: 1 bps)
MAX_SPREAD_BPS=50          # Maximum 50 bps

# Risk management
MAX_DRAWDOWN=0.05          # 5% max drawdown
STOP_LOSS_PCT=0.02         # 2% stop loss

# Optional: Email/Slack alerts
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ALERT_EMAIL=your-alert-email@gmail.com
SLACK_WEBHOOK=https://hooks.slack.com/services/...
```

## Parameter Tuning for US500

### Optimized Defaults

| Parameter | Value | Reason |
|-----------|-------|--------|
| `MIN_SPREAD_BPS` | 1 | US500's lower volatility allows tighter spreads |
| `MAX_SPREAD_BPS` | 50 | Still wide during high vol events |
| `REBALANCE_INTERVAL` | 3s | Fast delta-neutral for index |
| `ORDER_LEVELS` | 20 | Aggressive market making |
| `LEVERAGE` | 20x | Conservative for index (max 25x) |
| `HIGH_VOL_THRESHOLD` | 50% | Lower than crypto (100%) |
| `FUNDING_HEDGE_THRESHOLD` | 0.015% | Earlier hedge for index |
| `INVENTORY_SKEW_THRESHOLD` | 1.5% | Aggressive skew trigger |

### Spread Scaling by Volatility

| Volatility | Min Spread | Max Spread | Behavior |
|------------|------------|------------|----------|
| < 5% | 1 bps | 5 bps | Very tight (calm markets) |
| 5-10% | 2 bps | 10 bps | Normal index vol |
| 10-15% | 5 bps | 25 bps | Elevated vol |
| > 15% | 10 bps | 50 bps | High vol (rare for index) |

### Adaptive Mode Triggers

| Recent Spread | Mode | Distance | Levels |
|---------------|------|----------|--------|
| < -3 bps | DEFENSIVE | $2.00 | 1 |
| < +1 bps | MODERATE | $1.00 | 2 |
| +1 to +10 bps | NORMAL | $0.50 | 3 |
| +10 to +15 bps | AGGRESSIVE | $0.50 | 5 |
| > +15 bps | VERY_AGGRESSIVE | $0.25 | 8 |

## Progressive Test Phases

### Phase 1: Backtest x25 (12 Months Data)

Stress-test at x25 leverage to validate strategy robustness.

```bash
# Fetch data (uses BTC proxy if needed)
python amm-500.py --fetch-data --fetch-days 365

# Run stress-test backtest
LEVERAGE=25 python amm-500.py --backtest --months 12
```

**Validation Criteria:**
| Metric | Target | Notes |
|--------|--------|-------|
| Liq Prob (x25) | **< 5%** | Primary gate |
| Sharpe Ratio | **> 2.0** | Risk-adjusted returns |
| Max Drawdown | **< 5%** | Capital preservation |
| Trades/Day | **> 500** | Strategy efficiency |

⚠️ **If liq prob > 5%: STOP. Refine parameters before proceeding.**

### Phase 2: Paper Trading x10 ($1000 Virtual, 7 Days)

```bash
# Configure for paper trading
TESTNET=True
LEVERAGE=10
COLLATERAL=1000

# Run paper trading
## Testing & Validation

**Run 7-day paper trading to validate performance:**

```bash
cd /Users/nheosdisplay/VSC/AMM/AMM-500
source .venv/bin/activate
./scripts/start_paper_trading.sh
```

The script will:
- Prompt for mode (Paper/Live) and duration (7/30/Continuous)
- Launch bot with autonomous monitoring (amm_autonomous_v3.py)
- Track performance metrics (equity, PnL, trades, alerts)
- Auto-stop after duration (or run continuously)

**After 7 days, analyze results:**

```bash
python scripts/analyze_paper_results.py
```

**Target Metrics (7-day paper):**
| Metric | Target | Notes |
|--------|--------|-------|
| Trades/Day | > 1000 | High-frequency target |
| Maker Ratio | > 90% | Minimize taker fees |
| 7-Day ROI | > 5% | ~200% annualized |
| Max Drawdown | < 0.5% | Low risk tolerance |
| Sharpe Ratio | 1.5-3.0 | Risk-adjusted performance |

**Go live only after:**
✅ Passing 7-day paper trading targets  
✅ Understanding alert configurations  
✅ Testing kill switches manually  
✅ Funding wallet with exactly $1000 USDC

For complete autonomous setup: See [AUTONOMOUS_SETUP_GUIDE.md](AUTONOMOUS_SETUP_GUIDE.md)

## Risk Management

**Automatic Protections (Built-in):**
1. **Dynamic Leverage Adjustment**: Reduces leverage when drawdown/volatility increase
2. **Position Limits**: Enforces maximum exposure based on collateral
3. **Stop Loss**: Closes positions on adverse moves > 2%
4. **Circuit Breakers**: Pauses trading when approaching liquidation
5. **Funding Hedge**: Monitors funding rates, hedges when > 0.01%
6. **Kill Switches** (amm_autonomous_v3.py):
   - Max DD > 5% → Auto-stop bot
   - 10 consecutive losses → Auto-stop bot
   - Session loss > $100 → Auto-stop bot

**Risk Levels:**

| Level | Leverage | Spread | Action |
|-------|----------|--------|--------|
| LOW | 15-20x | 5-10 bps | Normal operation |
| MEDIUM | 10-15x | 10-15 bps | Increased monitoring |
| HIGH | 5-10x | 15-25 bps | Wide spreads, reduce size |
| CRITICAL | < 5x | Pause | Close positions, review |

**BTC-Specific Considerations:**
- High volatility → Tighter stop losses, wider spreads
- High funding rates → Reduce inventory bias, activate funding hedge
- Low liquidity periods → Widen spreads, reduce position size

## Monitoring & Logs

**Autonomous Monitoring (Recommended):**

The `amm_autonomous_v3.py` script provides 24/7 monitoring with:
- Real-time wallet tracking (equity, PnL, margin, positions)
- Email + Slack alerts (configurable cooldowns)
- Auto-restart on bot crashes (rate-limited)
- Kill switches (DD > 5%, 10 consecutive losses, $100 session loss)
- Performance metrics (trades/day, maker ratio, Sharpe, ROI)

**Start monitoring:**
```bash
source .venv/bin/activate
python scripts/amm_autonomous_v3.py

pkill -f "amm_autonomous_v3"
```

**Check logs:**
```bash
# Live bot logs
tail -f logs/bot_$(date +%Y-%m-%d).log

# Monitoring state
cat logs/autonomous_state.json | jq '.trade_summary'

# Search for errors
grep -i error logs/bot_*.log
```

For complete autonomous setup (launch daemon, web dashboard, alerts): [AUTONOMOUS_SETUP_GUIDE.md](AUTONOMOUS_SETUP_GUIDE.md)

## Troubleshooting

**Common Issues:**

**"Connection timeout" / "API rate limit exceeded"**
- Check network connection to Hyperliquid
- Reduce rebalance_interval if hitting rate limits
- Verify TESTNET setting in config/.env

**"Order rejected" / "Insufficient margin"**
- Check wallet balance (min $1000 for 10x leverage)
- Verify lot size meets minimum (0.001 BTC for BTC-PERP)
- Check max leverage (50x for BTC)

**"Bot not placing orders"**
- Check spread configuration (min_spread_bps)
- Verify market conditions (may be paused if volatility too high)
- Review logs for risk level (CRITICAL = paused)

**"Autonomous monitoring not sending alerts"**
- Verify SMTP/Slack credentials in config/.env
- Check email spam folder
- Review logs/autonomous_state.json for alert cooldowns

**"WebSocket connection errors"**
- Normal occasional reconnects expected
- Persistent errors → check firewall/network
- Bot will retry automatically (tenacity decorator)

**For more troubleshooting:** See [docs/FIXES_AND_STATUS.md](docs/FIXES_AND_STATUS.md)

## Disclaimer

This software is provided **for educational purposes only**. Trading derivatives with leverage involves **substantial risk of loss**.

**Key Risks:**
- **Market Risk**: Crypto prices can move violently against your position
- **Liquidation Risk**: Leveraged positions (10x+) can be fully liquidated
- **Liquidity Risk**: Low liquidity may cause slippage and adverse fills
- **Technical Risk**: Software bugs, network failures, API outages
- **Platform Risk**: Hyperliquid smart contract and oracle risks

**Always:**
✅ Test thoroughly with paper trading (7+ days)  
✅ Start with minimal capital ($1000 max initially)  
✅ Never trade more than you can afford to lose completely  
✅ Monitor actively with autonomous_v3.py + alerts  
✅ Understand all kill switch triggers before going live  

**The author assumes NO responsibility for trading losses.**

## License & Support

**License:** MIT License - See LICENSE file for details.

**Documentation:**
- [HFT_OPTIMIZATION_GUIDE.md](HFT_OPTIMIZATION_GUIDE.md) - HFT performance tuning
- [AUTONOMOUS_SETUP_GUIDE.md](AUTONOMOUS_SETUP_GUIDE.md) - 24/7 monitoring setup
- [CLEANUP_OPTIMIZATION_SUMMARY.md](CLEANUP_OPTIMIZATION_SUMMARY.md) - Project cleanup summary
- [docs/FIXES_AND_STATUS.md](docs/FIXES_AND_STATUS.md) - Bug fixes and status

**Hyperliquid Resources:**
- Docs: https://hyperliquid.gitbook.io/hyperliquid-docs
- Trade BTC-PERP: https://app.hyperliquid.xyz/trade/BTC
- Python SDK: https://github.com/hyperliquid-dex/hyperliquid-python-sdk

---

**Ready to start?** Run `./scripts/start_paper_trading.sh` for 7-day paper trading.
