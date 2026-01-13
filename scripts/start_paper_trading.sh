#!/bin/bash
#
# Start 7-day paper trading session for AMM-500
#
# Usage: ./scripts/start_paper_trading.sh
#

set -e

echo "=============================================="
echo "AMM-500 Paper Trading Startup"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check Python environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "✅ Virtual environment found"

# Check config
if [ ! -f "config/.env" ]; then
    echo "❌ Config file not found!"
    echo "Copy config/.env.example to config/.env and fill in your API keys"
    exit 1
fi

echo "✅ Config file found"

# Test connection
echo ""
echo "Testing connection to Hyperliquid..."
python amm-500.py --paper --status 2>&1 | grep -A5 "CONNECTION STATUS"

if [ $? -ne 0 ]; then
    echo "❌ Connection test failed!"
    exit 1
fi

echo ""
echo "✅ Connection successful"
echo ""

# Create log directory
mkdir -p logs

# Generate log filename
LOG_FILE="logs/paper_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "Starting Paper Trading Session"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Asset: BTC"
echo "  Leverage: 5x"
echo "  Min Spread: 4 bps"
echo "  Order Levels: 18"
echo "  Mode: Paper Trading (simulated orders)"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  ps aux | grep amm-500"
echo "  kill <PID>"
echo ""
echo "=============================================="
echo ""

# Prompt for confirmation
read -p "Start paper trading? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Start paper trading in background
echo "Starting bot..."
nohup python amm-500.py --paper > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "✅ Paper trading started!"
echo ""
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitoring first 30 seconds..."
echo ""

# Show first few lines
sleep 3
tail -20 "$LOG_FILE"

echo ""
echo "=============================================="
echo "Paper Trading Session Active"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Monitor logs: tail -f $LOG_FILE"
echo "  2. Check status: python amm-500.py --paper --status"
echo "  3. View fills: python scripts/monitor_fills.py"
echo "  4. After 7 days: python scripts/analyze_live.py"
echo ""
echo "Target metrics:"
echo "  • Sharpe: 1.5-2.5"
echo "  • Ann ROI: >30%"
echo "  • Max DD: <1%"
echo "  • Trades/day: >2000"
echo ""
echo "Red flags (stop if seen):"
echo "  ❌ Max DD > 2%"
echo "  ❌ Position > $500"
echo "  ❌ Rate limit errors (429)"
echo "  ❌ Taker volume > 30%"
echo ""
echo "Good luck! 🚀"
echo ""
