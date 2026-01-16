# AMM-500: Professional US500-USDH Market Maker

Autonomous high-frequency market making bot for Hyperliquid US500-USDH HIP-3 perpetuals.

## Features

### Professional Market Making
- **Book-Aware Quoting**: Microprice, order book imbalance, queue position tracking
- **Dynamic Spreads**: Vol-scaled exponential gaps (1-50 bps based on volatility)
- **Smart Order Management**: Minimize cancels by matching existing orders
- **Position Feedback**: Skew quotes based on inventory with USDH margin
- **Depth-Aware Sizing**: Larger sizes at tighter levels, exponential decay

### PyTorch Volatility Predictor
- LSTM model trained on xyz100 (^OEX) or BTC data
- Real-time volatility estimation for spread adjustment
- Automatic fallback to realized volatility

### Risk Management
- **Kill Switches**: DD>2%, taker>30%, margin<10%, 3 losing days
- **Auto-Restart**: Max 5 restarts per hour
- **Reduce-Only Mode**: Gradually close positions when risk triggers hit
- **Funding Rate Awareness**: Adjust for funding costs

### Alerts & Monitoring
- Slack/email notifications for critical events
- Prometheus metrics (port 9090)
- Real-time PnL tracking

## Installation

```bash
# Clone and setup
cd AMM-500
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy and edit the environment file:

```bash
cp config/.env.example config/.env
```

Key settings in `config/.env`:

```env
# Hyperliquid
HYPERLIQUID_PRIVATE_KEY=your_private_key
HYPERLIQUID_TESTNET=true

# Trading
SYMBOL=US500
LEVERAGE=5
COLLATERAL=1000
ORDER_LEVELS=15
MIN_SPREAD_BPS=1
MAX_SPREAD_BPS=5
REBALANCE_INTERVAL=0.5

# Risk
MAX_DRAWDOWN=0.02
MAX_TAKER_RATIO=0.05
MARGIN_PAUSE_THRESHOLD=0.90

# Alerts (optional)
SLACK_WEBHOOK_URL=
ALERT_EMAIL=
```

## Usage

### Fetch Training Data

```bash
python amm-500.py fetch --days 30
```

This downloads xyz100 (^OEX) data via yfinance with BTC fallback.

### Run Backtest

```bash
python amm-500.py backtest --days 30
```

### Paper Trading

```bash
python amm-500.py run --paper
```

### Live Trading (with autonomous mode)

```bash
python amm-500.py run --autonomous
```

The `--autonomous` flag enables:
- Kill switches (DD>2%, taker>30%, margin<10%, 3 losing days)
- Auto-restart (max 5/hour)
- Slack/email alerts
- Prometheus metrics

## Parameter Optimization

Run grid search over 108 parameter combinations:

```bash
python scripts/grid_search.py
```

Results saved to `data/grid_search_results.csv`.

## Target Verification

Verify backtest meets professional targets:

```bash
python scripts/verify_targets.py
```

Targets:
- Sharpe > 2.5
- ROI > 5%
- Max DD < 0.5%
- Trades/day > 2000
- Maker ratio > 90%

## Tests

```bash
pytest tests/ -v
```

## Architecture

```
amm-500.py              # Main CLI entry point
├── src/
│   ├── core/
│   │   ├── strategy.py   # MM logic: microprice, imbalance, quotes
│   │   ├── exchange.py   # Hyperliquid L2 WebSocket client
│   │   ├── risk.py       # Kill switches, drawdown tracking
│   │   ├── backtest.py   # Realistic simulation engine
│   │   └── metrics.py    # Prometheus metrics
│   └── utils/
│       ├── config.py     # Load/validate configuration
│       ├── data_fetcher.py # xyz100/BTC data fetching
│       └── utils.py      # Helper functions
├── scripts/
│   ├── grid_search.py    # Parameter optimization
│   └── verify_targets.py # Target verification
├── tests/
│   └── test_strategy.py  # Strategy unit tests
├── config/
│   └── .env              # Configuration (create from .env.example)
├── data/                 # Cached data
└── logs/                 # Trading logs
```

## Backtest Realism

The backtest engine simulates:
- **Latency**: 50-200ms random delays
- **Partial Fills**: 30% fill rate
- **Adverse Selection**: 35% probability
- **Queue Priority**: Time-based fill priority
- **Funding Rates**: 8-hour funding costs
- **Maker/Taker Fees**: -0.02% maker rebate, 0.05% taker fee

## Kill Switch Conditions

| Condition | Action |
|-----------|--------|
| Drawdown > 2% | Reduce position 20% |
| Taker ratio > 30% | Pause new orders |
| Margin ratio < 10% | Emergency close |
| 3 consecutive losing days | Full stop |
| Volatility > 15% | Widen spreads to 50 bps |

## Monitoring

Prometheus metrics available at `http://localhost:9090/metrics`:
- `amm_pnl_total`: Total PnL
- `amm_position_size`: Current position
- `amm_spread_bps`: Current spread
- `amm_volatility`: Estimated volatility
- `amm_trades_total`: Trade count
- `amm_maker_ratio`: Maker/taker ratio

## License

MIT
