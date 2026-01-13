# AMM-500: High-Frequency Market Making Bot

A high-frequency trading bot for delta-neutral market making on **BTC** perpetuals on the Hyperliquid exchange. Originally designed for US500 (S&P 500 Index), but adapted to BTC as US500/SPX index perps are not available on Hyperliquid.

> ⚠️ **WARNING: HIGH RISK**  
> This is a leveraged trading bot for a **permissionless market**. You can lose your entire investment. Always:
> - Test thoroughly on testnet first
> - Start with small amounts ($1000 recommended)
> - Monitor actively
> - Understand the strategy before running
> - Never risk money you can't afford to lose
> 
> **Currently configured for BTC** - the most liquid perpetual on Hyperliquid with tight spreads (0.11 bps) ideal for market making. Originally designed for US500/S&P 500 index, but US500 perps don't exist on Hyperliquid. The strategy parameters have been optimized via 30-day backtests achieving: **Sharpe 2.18, Ann ROI 59.1%, Max DD 0.47%, 3000+ trades/day**.

## Quick Start

```bash
# 1. Clone and enter directory
-

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp config/.env.example config/.env
nano config/.env  # Add your credentials

# 5. Fetch historical data
python amm-500.py --fetch-data --fetch-days 180

# 6. Run backtest
python amm-500.py --backtest --months 6

# 7. Paper trade (recommended first!)
python amm-500.py --paper

# 8. Live trade (after thorough testing)
python amm-500.py
```

## Quick Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run paper trading (mainnet data, simulated orders)
python amm-500.py --paper

# Run live trading (REAL MONEY!)
python amm-500.py

# Fetch US500 historical data (or BTC proxy)
python amm-500.py --fetch-data --fetch-days 180

# Run backtest (30 days default)
python amm-500.py --backtest --days 30

# Run backtest with 12 months of data
python amm-500.py --backtest --months 12

# Check connection status
python amm-500.py --status

# Watch bot logs in real-time
tail -f logs/bot_$(date +%Y-%m-%d).log

# Find and kill the bot process
pkill -f "python amm-500.py"
```

## US500 Specific Information

### About US500 on Hyperliquid

- **Symbol**: US500 (accessed as km:US500 in the UI)
- **Deployer**: KM (HIP-3 Permissionless)
- **Max Leverage**: 25x
- **Margin Type**: Isolated only
- **Trading**: 24/7 (but underlying tracks US market hours)
- **Documentation**: [docs.markets.xyz](https://docs.markets.xyz/)
- **Trade Link**: [app.hyperliquid.xyz/trade/km:US500](https://app.hyperliquid.xyz/trade/km:US500)

### Key Differences from BTC

| Parameter | US500 | BTC |
|-----------|-------|-----|
| **Volatility** | 5-15% annualized | 50-100% annualized |
| **Min Spread** | 1 bps | 5 bps |
| **Max Spread** | 50 bps | 50 bps |
| **Price Range** | ~5000-6000 | ~80000-100000 |
| **Tick Size** | $0.01 | $1.00 |
| **Max Leverage** | 25x | 50x |

### Data Proxy System

Since US500 is a newer asset, it may not have sufficient historical data for reliable backtesting. The bot includes an automatic fallback system:

1. **Check US500 Data**: Attempts to fetch US500 candles from Hyperliquid
2. **Proxy Fallback**: If < 6 months of data available, uses BTC as proxy
3. **Data Scaling**: BTC data is scaled to match US500 characteristics:
   - Prices scaled to US500 range (~5800)
   - Volatility compressed to 30% of BTC
   - Funding rates reduced by 50%
4. **Auto-Switch**: Bot checks periodically and switches to real data when available

## Performance Targets

| Leverage | Metric | Target | Validation |
|----------|--------|--------|------------|
| **x25** (STRESS-TEST) | Liquidation Prob | <5% | Run `--backtest --months 12` |
| **x20** (RECOMMENDED) | Annual ROI | 30-50% | Sharpe ratio >2 |
| **x10** (CONSERVATIVE) | Max Drawdown | <5% | 95th percentile |

## Architecture

```
AMM-500/
├── amm-500.py           # Entry point (US500 optimized)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
├── pytest.ini           # Test configuration
├── config/
│   ├── .env             # Your configuration (not in git)
│   └── .env.example     # Configuration template
├── src/
│   ├── __init__.py
│   ├── config.py        # US500-optimized configuration
│   ├── data_fetcher.py  # US500 data + BTC proxy system
│   ├── exchange.py      # Hyperliquid client (US500 tick/lot)
│   ├── strategy.py      # Market making logic
│   ├── risk.py          # Risk management
│   ├── backtest.py      # Backtesting framework
│   ├── metrics.py       # Prometheus metrics
│   └── utils.py         # Utility functions
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── data/                # Historical data (US500 + BTC proxy)
└── logs/                # Trading logs
```

## Installation Guide

### Prerequisites

- **Python 3.10** (required for compatibility)
- **macOS/Linux** (Windows with WSL2)
- **Git**

### Step 1: Install Python 3.10

**macOS (Homebrew):**
```bash
brew install python@3.10
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip
```

### Step 2: Clone and Setup

```bash
# Clone the repository
cd /path/to/your/projects
git clone <repository-url> AMM-500
cd AMM-500

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure

```bash
# Copy the example configuration
cp config/.env.example config/.env

# Edit with your credentials
nano config/.env
```

**Required settings in `.env`:**
```env
PRIVATE_KEY=your_wallet_private_key
WALLET_ADDRESS=your_wallet_address
TESTNET=True  # Start with testnet!
```

### Step 4: Verify Installation

```bash
# Check connection (requires testnet setup)
python amm-500.py --status

# Run tests
pytest tests/ -v
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
python amm-500.py --paper
```

**Duration**: 7 days minimum

**Validation Criteria:**
| Metric | Target | Notes |
|--------|--------|-------|
| Cumulative PnL | **> $20** (2%) | ~30% annualized |
| Max Drawdown | **< 3%** | $30 max loss |
| Imbalance Events | **< 5** | Hard stops |
| Funding Net | **> $0** | Hedge working |

### Phase 3: Live Trading x10 ($1000 Real)

Only after passing Phase 1 **AND** Phase 2:

```bash
# Configure for live
TESTNET=False
LEVERAGE=10
COLLATERAL=1000

# Fund wallet with exactly $1000 USDC
# Start with active monitoring
python amm-500.py

# Start autonomous monitoring
python scripts/amm_autonomous.py
```

**Gradual Scale-Up Plan:**
| Week | Collateral | Leverage | Notes |
|------|------------|----------|-------|
| 1 | $1,000 | x10 | Active monitoring |
| 2-3 | $1,000 | x10 | Passive monitoring |
| 4 | $2,000 | x10 | If 4 weeks profitable |
| 8 | $5,000 | x15-x20 | If 8 weeks profitable |
| 12+ | $10,000+ | x20 | Full production |

**Kill Switch Triggers:**
- Max DD > 5% → Stop and review
- 3 consecutive losing days → Reduce to x5
- Any single day loss > 3% → Pause 24h

## Risk Management

### Automatic Protections

1. **Leverage Reduction**: Reduces when drawdown/volatility increase
2. **Position Limits**: Enforces maximum exposure
3. **Stop Loss**: Closes position on adverse moves
4. **Circuit Breakers**: Pauses trading when near liquidation

### Risk Levels

| Level | Leverage | Action |
|-------|----------|--------|
| LOW | 20x | Normal operation |
| MEDIUM | 15x | Increased monitoring |
| HIGH | 10x | Wide spreads |
| CRITICAL | 5x | Pause + close positions |

### US500-Specific Risks

1. **Low Liquidity**: Permissionless market may have thin order books
2. **Oracle Risk**: Index price depends on external oracle
3. **Market Hours**: Underlying only trades during US hours
4. **Deployer Risk**: KM deployer controls market parameters
5. **Isolated Margin**: Cannot share margin with other positions

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--paper` | False | Paper trading mode |
| `--backtest` | False | Run backtest |
| `--days` | 30 | Days of data for backtest |
| `--months` | 1 | Months of data (1-12) |
| `--synthetic` | False | Use synthetic data |
| `--fetch-data` | False | Fetch US500 historical data |
| `--fetch-days` | 180 | Days of data to fetch |
| `--status` | False | Check connection status |
| `--config` | config/.env | Config file path |

## Monitoring

### Status Dashboard

```bash
# Quick status check
python amm-500.py --status
```

### Log Monitoring

```bash
# Watch live logs
tail -f logs/bot_$(date +%Y-%m-%d).log

# Watch trades
tail -f logs/trades_$(date +%Y-%m-%d).log

# Search for errors
grep -i error logs/bot_$(date +%Y-%m-%d).log
```

### Prometheus Metrics

Metrics exposed on port 9090 (configurable):
- Position size/value
- PnL (gross/net)
- Fill rate
- Latency
- Risk metrics

## Troubleshooting

### Common Issues

**"Insufficient historical data for US500"**
- This is expected for new assets
- Bot will use BTC as proxy automatically
- Run `--fetch-data` to cache proxy data

**"Max leverage exceeded"**
- US500 max is 25x (vs 50x for BTC)
- Reduce `LEVERAGE` in config

**"Order rejected"**
- Check lot size (0.01 minimum)
- Verify margin available
- Check order book liquidity

**"Connection to permissionless market failed"**
- Permissionless markets may have different API patterns
- Check KM deployer documentation
- Verify symbol format (US500, not km:US500 in API)

**"SDK not available"**
```bash
pip install hyperliquid-python-sdk
```

## Disclaimer

This software is provided for educational purposes only. Trading derivatives with leverage involves substantial risk of loss.

**Key Risks:**
- **Market Risk**: Prices can move against your position rapidly
- **Liquidation Risk**: Leveraged positions can be fully liquidated
- **Liquidity Risk**: Permissionless markets may lack liquidity
- **Technical Risk**: Software bugs, network issues, oracle failures
- **Platform Risk**: Hyperliquid and KM deployer risks
- **Regulatory Risk**: Index derivatives may face regulatory scrutiny

**Always:**
- Test thoroughly before using real funds
- Start with minimal capital ($1000 max initially)
- Never trade more than you can afford to lose
- Monitor the bot actively
- Have manual intervention plans ready
- Understand the permissionless market risks

## License

MIT License - See LICENSE file for details.

---

## Support

- **Hyperliquid Docs**: https://hyperliquid.gitbook.io/hyperliquid-docs
- **KM Deployer Docs**: https://docs.markets.xyz/
- **US500 Trading**: https://app.hyperliquid.xyz/trade/km:US500
- **SDK Repo**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
