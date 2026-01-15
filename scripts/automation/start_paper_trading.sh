#!/bin/bash
#
# Start 7-day paper trading session for AMM-500 with autonomous monitoring
#
# Usage: ./scripts/start_paper_trading.sh
#

set -e

echo "=============================================="
echo "AMM-500 Autonomous Trading Startup"
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

# Mode selection
echo ""
echo "Select trading mode:"
echo "  1) Paper Trading (recommended - simulated orders)"
echo "  2) Live Trading (REAL MONEY - USE WITH CAUTION)"
echo ""
read -p "Choice (1-2): " MODE
echo ""

if [ "$MODE" = "1" ]; then
    BOT_MODE="--paper"
    MODE_NAME="PAPER TRADING"
elif [ "$MODE" = "2" ]; then
    echo "⚠️⚠️⚠️  LIVE TRADING MODE ⚠️⚠️⚠️"
    echo ""
    echo "This will trade with REAL MONEY on Hyperliquid mainnet."
    echo "You can lose your entire investment."
    echo ""
    read -p "Type 'LIVE' to confirm: " CONFIRM
    
    if [ "$CONFIRM" != "LIVE" ]; then
        echo "Cancelled."
        exit 0
    fi
    
    BOT_MODE=""
    MODE_NAME="LIVE TRADING"
else
    echo "Invalid choice."
    exit 1
fi

# Duration selection
echo ""
echo "Select trading duration:"
echo "  1) 7 days (recommended for testing)"
echo "  2) 30 days (full month)"
echo "  3) Continuous (24/7 until stopped)"
echo ""
read -p "Choice (1-3): " DURATION
echo ""

case $DURATION in
    1) DURATION_NAME="7 days" ;;
    2) DURATION_NAME="30 days" ;;
    3) DURATION_NAME="Continuous (24/7)" ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

# Create log directory
mkdir -p logs

# Configuration summary
echo "=============================================="
echo "CONFIGURATION SUMMARY"
echo "=============================================="
echo ""
echo "Mode: $MODE_NAME"
echo "Duration: $DURATION_NAME"
echo "Asset: BTC-PERP"
echo "Leverage: 10x (from config)"
echo "Collateral: \$1000 (from config)"
echo "Min Spread: 5 bps"
echo ""
echo "Features:"
echo "  ✅ Real-time wallet API tracking"
echo "  ✅ Auto-restart on crash (max 5/hour)"
echo "  ✅ Email/Slack alerts (if configured)"
echo "  ✅ Kill switches (DD>5%, Loss>\$100, 10 losses)"
echo ""
echo "Log files:"
echo "  - Bot: logs/bot_\$(date +%Y-%m-%d).log"
echo "  - Monitoring: logs/autonomous_v3.log"
echo "  - Bot stdout: logs/bot.out"
echo "  - Monitor stdout: logs/autonomous_v3.out"
echo ""
echo "=============================================="
echo ""

# Final confirmation
read -p "Start trading? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Activate environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "python.*amm-500.py" 2>/dev/null || true
pkill -f "python.*amm_autonomous" 2>/dev/null || true
sleep 2

# Start bot
echo "Starting bot process..."
if [ "$MODE" = "1" ]; then
    nohup python amm-500.py --paper > logs/bot.out 2>&1 &
else
    nohup python amm-500.py > logs/bot.out 2>&1 &
fi

BOT_PID=$!
echo "✅ Bot started (PID: $BOT_PID)"
sleep 3

# Check if bot is running
if ! ps -p $BOT_PID > /dev/null; then
    echo "❌ Bot failed to start! Check logs/bot.out"
    exit 1
fi

# Start autonomous monitoring
echo "Starting autonomous monitoring..."
nohup python scripts/amm_autonomous_v3.py > logs/autonomous_v3.out 2>&1 &
MONITOR_PID=$!
echo "✅ Monitoring started (PID: $MONITOR_PID)"
sleep 2

# Check if monitor is running
if ! ps -p $MONITOR_PID > /dev/null; then
    echo "❌ Monitoring failed to start! Check logs/autonomous_v3.out"
    kill $BOT_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ SYSTEM RUNNING"
echo "=============================================="
echo ""
echo "Bot PID: $BOT_PID"
echo "Monitor PID: $MONITOR_PID"
echo ""
echo "To monitor:"
echo "  tail -f logs/bot_\$(date +%Y-%m-%d).log"
echo "  tail -f logs/autonomous_v3.log"
echo ""
echo "To check status:"
echo "  ps aux | grep -E 'amm-500|autonomous_v3'"
echo ""
echo "To stop:"
echo "  kill $BOT_PID $MONITOR_PID"
echo "  # OR: pkill -f 'python.*amm-500.py'"
echo ""
echo "⚠️  Monitoring system will auto-restart bot if it crashes"
echo "=============================================="
echo ""

# Optional: schedule stop after duration
if [ "$DURATION" != "3" ]; then
    if [ "$DURATION" = "1" ]; then
        SECONDS=$((7 * 24 * 3600))
        STOP_TIME=$(date -v+7d "+%Y-%m-%d %H:%M:%S")
    else
        SECONDS=$((30 * 24 * 3600))
        STOP_TIME=$(date -v+30d "+%Y-%m-%d %H:%M:%S")
    fi
    
    echo "Scheduling automatic stop at: $STOP_TIME"
    (sleep $SECONDS && pkill -f "python.*(amm-500|autonomous)" && echo "Auto-stopped after $DURATION_NAME") &
    echo "Auto-stop scheduled."
    echo ""
fi

# Save PIDs to file for easy cleanup
echo "$BOT_PID" > logs/bot.pid
echo "$MONITOR_PID" > logs/monitor.pid

echo "PIDs saved to logs/*.pid"
echo ""
echo "Happy trading! 🚀"

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
