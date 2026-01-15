#!/bin/bash
# AMM-500 Structure Reorganization Script
# Ultra-clean HFT bot for US500-USDH perpetuals

set -e

echo "🚀 AMM-500 Structure Reorganization"
echo "===================================="

# Create new directory structure
echo "📁 Creating new directory structure..."
mkdir -p src/core
mkdir -p src/utils  
mkdir -p scripts/automation
mkdir -p scripts/analysis
mkdir -p tests
mkdir -p config
mkdir -p data
mkdir -p logs

# Move and rename core strategy files
echo "📦 Moving core files..."

# Strategy: strategy_us500_pro.py → src/core/strategy.py
if [ -f "src/core/strategy_us500_pro.py" ]; then
    echo "  ✓ Renaming strategy_us500_pro.py → strategy.py"
    mv src/core/strategy_us500_pro.py src/core/strategy.py
fi

# Core files already in place (exchange.py, risk.py, backtest.py, metrics.py)
echo "  ✓ Core files: exchange.py, risk.py, backtest.py, metrics.py"

# Utils files (config.py, data_fetcher.py, utils.py, xyz100_fallback.py)
echo "  ✓ Utils files in place"

# Move automation scripts
echo "📦 Moving automation scripts..."
if [ -f "scripts/automation/amm_autonomous_v3.py" ]; then
    echo "  ✓ Renaming amm_autonomous_v3.py → amm_autonomous.py"
    cp scripts/automation/amm_autonomous_v3.py scripts/automation/amm_autonomous.py
fi

if [ -f "setup_us500_optimization.sh" ]; then
    echo "  ✓ Renaming setup_us500_optimization.sh → setup_bot.sh"
    mv setup_us500_optimization.sh scripts/automation/setup_bot.sh
fi

# Move analysis scripts
echo "📦 Organizing analysis scripts..."
for script in grid_search.py verify_targets.py analyze_paper_results.py; do
    if [ -f "scripts/$script" ]; then
        mv scripts/$script scripts/analysis/ 2>/dev/null || true
    fi
done

# Move test files
echo "📦 Moving test files..."
if [ -f "tests/test_us500_strategy.py" ]; then
    echo "  ✓ Renaming test_us500_strategy.py → test_strategy.py"
    mv tests/test_us500_strategy.py tests/test_strategy.py
fi

# Clean up obsolete markdown files
echo "🗑️  Removing obsolete markdown files..."
rm -f HFT_OPTIMIZATION_GUIDE.md
rm -f AUTONOMOUS_SETUP_GUIDE.md
rm -f CLEANUP_OPTIMIZATION_SUMMARY.md
rm -f US500_TRANSFORMATION_README.md
rm -f PROFESSIONAL_*.md
rm -f QUICK_*.md
rm -f REAL_TIME_*.md
rm -f TRANSFORMATION_*.md
rm -f SYSTEM_OPTIMIZATION_*.md

# Clean up old log files and data
echo "🗑️  Cleaning logs and data..."
find logs/ -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
find logs/ -type f -name "*.json" -mtime +7 -delete 2>/dev/null || true

# Remove archive folder
echo "🗑️  Removing archive folder..."
rm -rf archive/ 2>/dev/null || true

# Clean up old scripts
echo "🗑️  Cleaning old scripts..."
rm -rf scripts/old_* 2>/dev/null || true

# Remove __pycache__ directories
echo "🗑️  Removing __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "✅ Reorganization Complete!"
echo ""
echo "📂 New Structure:"
echo "  AMM-500/"
echo "  ├── amm-500.py                 # Main entry point"
echo "  ├── requirements.txt           # Dependencies"
echo "  ├── pyproject.toml             # Project metadata"
echo "  ├── pytest.ini                 # Test configuration"
echo "  ├── .gitignore                 # Git ignore rules"
echo "  ├── README.md                  # Comprehensive documentation"
echo "  ├── config/"
echo "  │   ├── .env                   # Your credentials (gitignored)"
echo "  │   └── .env.example           # Template"
echo "  ├── src/core/"
echo "  │   ├── strategy.py            # US500 MM strategy (renamed)"
echo "  │   ├── exchange.py            # Hyperliquid client"
echo "  │   ├── risk.py                # Risk management"
echo "  │   ├── backtest.py            # Backtesting"
echo "  │   └── metrics.py             # Performance metrics"
echo "  ├── src/utils/"
echo "  │   ├── config.py              # Configuration"
echo "  │   ├── data_fetcher.py        # Data fetching"
echo "  │   ├── utils.py               # Utilities"
echo "  │   └── xyz100_fallback.py     # xyz100/BTC data"
echo "  ├── scripts/automation/"
echo "  │   ├── amm_autonomous.py      # 24/7 monitoring (renamed)"
echo "  │   ├── start_paper_trading.sh # Interactive launcher"
echo "  │   └── setup_bot.sh           # Setup script (renamed)"
echo "  ├── scripts/analysis/"
echo "  │   ├── grid_search.py         # Parameter optimization"
echo "  │   ├── verify_targets.py      # Target validation"
echo "  │   └── analyze_paper_results.py # Performance analysis"
echo "  ├── tests/"
echo "  │   └── test_strategy.py       # Strategy tests (renamed)"
echo "  ├── data/                      # Historical data (empty)"
echo "  └── logs/                      # Trading logs (empty)"
echo ""
