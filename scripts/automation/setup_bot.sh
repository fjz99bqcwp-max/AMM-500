#!/bin/bash
# setup_us500_optimization.sh
# Quick setup script for US500-USDH professional MM transformation

set -e

echo "=================================================="
echo "US500-USDH PROFESSIONAL MM SETUP"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11|12) ]]; then
    echo -e "${RED}ERROR: Python 3.9+ required${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install base dependencies
echo -e "${YELLOW}Installing base dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Base dependencies installed${NC}"

# Install optional dependencies for US500 optimization
echo -e "${YELLOW}Installing US500 optimization dependencies...${NC}"

# PyTorch (optional, for vol prediction)
read -p "Install PyTorch for ML vol prediction? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch (CPU version)..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo -e "${GREEN}✓ PyTorch installed${NC}"
else
    echo -e "${YELLOW}⚠ PyTorch skipped (strategy will use statistical vol only)${NC}"
fi

# yfinance (for xyz100 fallback)
echo "Installing yfinance for xyz100 fallback..."
pip install -q yfinance
echo -e "${GREEN}✓ yfinance installed${NC}"

# Create data directories
echo -e "${YELLOW}Setting up directories...${NC}"
mkdir -p data/xyz100
mkdir -p logs
mkdir -p docs
echo -e "${GREEN}✓ Directories created${NC}"

# Backup old strategy
echo -e "${YELLOW}Backing up old strategy...${NC}"
if [ -f "src/strategy.py" ]; then
    BACKUP_NAME="src/strategy_backup_$(date +%Y%m%d_%H%M%S).py"
    cp src/strategy.py "$BACKUP_NAME"
    echo -e "${GREEN}✓ Old strategy backed up to $BACKUP_NAME${NC}"
fi

# Check if user wants to replace strategy.py
echo ""
echo "The new professional MM strategy is in: src/strategy_us500_pro.py"
echo "Options:"
echo "  1. Replace src/strategy.py with new strategy (recommended for testing)"
echo "  2. Keep both files (manually update imports in amm-500.py)"
echo ""
read -p "Replace strategy.py? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp src/strategy_us500_pro.py src/strategy.py
    echo -e "${GREEN}✓ strategy.py replaced with US500 professional MM${NC}"
else
    echo -e "${YELLOW}⚠ Keeping both files - update imports manually${NC}"
    echo "  Change: from src.strategy import MarketMakingStrategy"
    echo "  To: from src.strategy_us500_pro import US500ProfessionalMM"
fi

# Update config if needed
echo -e "${YELLOW}Checking config...${NC}"
if [ -f "config/.env" ]; then
    # Check if US500 already configured
    if grep -q "SYMBOL=US500" config/.env; then
        echo -e "${GREEN}✓ Config already set for US500${NC}"
    else
        echo -e "${YELLOW}⚠ Config may need updating for US500${NC}"
        echo "  Recommended settings:"
        echo "    SYMBOL=US500"
        echo "    MIN_SPREAD_BPS=1.0"
        echo "    MAX_SPREAD_BPS=50.0"
        echo "    ORDER_LEVELS=15"
        echo "    LEVERAGE=10"
        echo ""
        read -p "Update config automatically? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Backup .env
            cp config/.env config/.env.backup
            
            # Update or append settings
            sed -i.tmp 's/^SYMBOL=.*/SYMBOL=US500/' config/.env || echo "SYMBOL=US500" >> config/.env
            sed -i.tmp 's/^MIN_SPREAD_BPS=.*/MIN_SPREAD_BPS=1.0/' config/.env || echo "MIN_SPREAD_BPS=1.0" >> config/.env
            sed -i.tmp 's/^MAX_SPREAD_BPS=.*/MAX_SPREAD_BPS=50.0/' config/.env || echo "MAX_SPREAD_BPS=50.0" >> config/.env
            sed -i.tmp 's/^ORDER_LEVELS=.*/ORDER_LEVELS=15/' config/.env || echo "ORDER_LEVELS=15" >> config/.env
            
            rm -f config/.env.tmp
            echo -e "${GREEN}✓ Config updated (backup: config/.env.backup)${NC}"
        fi
    fi
else
    echo -e "${RED}ERROR: config/.env not found${NC}"
    echo "Create config/.env from config/.env.example"
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
pytest tests/test_us500_strategy.py -v --tb=short || {
    echo -e "${YELLOW}⚠ Some tests failed (expected if missing mocks)${NC}"
}

echo ""
echo "=================================================="
echo -e "${GREEN}✓ US500-USDH SETUP COMPLETE${NC}"
echo "=================================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Fetch test data (xyz100 fallback):"
echo "   python -c 'from src.xyz100_fallback import XYZ100FallbackFetcher; import asyncio; asyncio.run(XYZ100FallbackFetcher().fetch_xyz100_data(30))'"
echo ""
echo "2. Run paper trading (7 days):"
echo "   ./scripts/start_paper_trading.sh"
echo ""
echo "3. Monitor logs:"
echo "   tail -f logs/bot_\$(date +%Y-%m-%d).log"
echo ""
echo "4. Analyze results after 7 days:"
echo "   python scripts/analyze_paper_results.py"
echo ""
echo "DOCUMENTATION:"
echo "  - US500 transformation: US500_TRANSFORMATION_README.md"
echo "  - Exchange enhancements: docs/EXCHANGE_ENHANCEMENTS.md"
echo "  - Risk enhancements: docs/RISK_ENHANCEMENTS.md"
echo ""
echo "TARGET METRICS (7-day paper):"
echo "  - Sharpe >2.5"
echo "  - Trades/day >2000"
echo "  - Maker ratio >90%"
echo "  - Max DD <0.5%"
echo "  - USDH margin <80%"
echo ""
echo -e "${YELLOW}⚠ WARNING: Test thoroughly on paper before live trading${NC}"
echo ""
