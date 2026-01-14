# Complete Autonomous System Setup Guide

**AMM-500 Autonomous Market Making Bot**  
**Date:** January 14, 2026  
**Version:** 3.0 (Enhanced)

---

## Overview

This guide provides step-by-step instructions to make `amm_autonomous_v3.py` completely autonomous for 24/7 unattended operation.

---

## Features Already Implemented ✅

The `amm_autonomous_v3.py` script already includes:

1. **Real-time Wallet API Tracking** - Monitors equity, PnL, margin, positions
2. **Async Logging** - Non-blocking log writes with aiofiles pattern
3. **Auto-Restart on Crash** - Detects dead processes and restarts (max 5/hour)
4. **Alert System** - Email and Slack notifications
5. **Kill Switches** - Auto-stops on critical conditions
6. **Trade Metrics Tracking** - Realized PnL, fees, maker/taker ratio
7. **Process Health Monitoring** - CPU, memory, uptime
8. **State Persistence** - Survives restarts

---

## Additional Enhancements Needed

### 1. Systemd Service (Linux) or Launch Daemon (macOS)

For true 24/7 operation, the monitoring script should auto-start on boot.

#### macOS Launch Daemon

Create `/Library/LaunchDaemons/com.amm500.autonomous.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.amm500.autonomous</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/Users/nheosdisplay/VSC/AMM/AMM-500/.venv/bin/python</string>
        <string>/Users/nheosdisplay/VSC/AMM/AMM-500/scripts/amm_autonomous_v3.py</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>/Users/nheosdisplay/VSC/AMM/AMM-500</string>
    
    <key>StandardOutPath</key>
    <string>/Users/nheosdisplay/VSC/AMM/AMM-500/logs/autonomous_v3.out</string>
    
    <key>StandardErrorPath</key>
    <string>/Users/nheosdisplay/VSC/AMM/AMM-500/logs/autonomous_v3.err</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

**Load the daemon:**

```bash
sudo launchctl load /Library/LaunchDaemons/com.amm500.autonomous.plist
sudo launchctl start com.amm500.autonomous
```

**Check status:**

```bash
sudo launchctl list | grep amm500
```

**Stop:**

```bash
sudo launchctl stop com.amm500.autonomous
sudo launchctl unload /Library/LaunchDaemons/com.amm500.autonomous.plist
```

---

### 2. Environment Variables for Alerts

Set up email and Slack alerts by configuring environment variables in `config/.env`:

```bash
# Email Alert Configuration (Gmail example)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password  # Generate at https://myaccount.google.com/apppasswords
ALERT_EMAIL=your-alert-email@gmail.com

# Slack Webhook (optional)
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Gmail Setup:**
1. Enable 2FA on Gmail
2. Generate App Password at https://myaccount.google.com/apppasswords
3. Use App Password (not your regular password) in `SMTP_PASS`

**Slack Setup:**
1. Go to https://api.slack.com/messaging/webhooks
2. Create an Incoming Webhook for your workspace
3. Copy webhook URL to `SLACK_WEBHOOK`

---

### 3. Paper/Live Toggle in start_paper_trading.sh

Enhance the start script to support both paper and live modes with autonomous monitoring:

```bash
#!/bin/bash
# scripts/start_paper_trading.sh - ENHANCED

set -e

echo "=============================================="
echo "AMM-500 Autonomous Trading Startup"
echo "=============================================="
echo ""

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Check environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    exit 1
fi

if [ ! -f "config/.env" ]; then
    echo "❌ Config file not found!"
    exit 1
fi

# Mode selection
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

# Configuration summary
echo "=============================================="
echo "CONFIGURATION SUMMARY"
echo "=============================================="
echo ""
echo "Mode: $MODE_NAME"
echo "Duration: $DURATION_NAME"
echo "Symbol: BTC-PERP"
echo "Leverage: 10x (from config)"
echo "Collateral: \$1000 (from config)"
echo ""
echo "Features:"
echo "  ✅ Real-time wallet tracking"
echo "  ✅ Auto-restart on crash"
echo "  ✅ Email/Slack alerts"
echo "  ✅ Kill switches (DD>5%, Loss>\$100)"
echo ""
echo "Log files:"
echo "  - Bot: logs/bot_\$(date +%Y-%m-%d).log"
echo "  - Monitoring: logs/autonomous_v3.log"
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

# Start the bot
echo ""
echo "Starting bot process..."
source .venv/bin/activate

# Kill any existing processes
pkill -f "python amm-500.py" 2>/dev/null || true
pkill -f "python scripts/amm_autonomous" 2>/dev/null || true
sleep 2

# Start bot
if [ "$MODE" = "1" ]; then
    nohup python amm-500.py --paper > logs/bot.out 2>&1 &
else
    nohup python amm-500.py > logs/bot.out 2>&1 &
fi

BOT_PID=$!
echo "✅ Bot started (PID: $BOT_PID)"
sleep 3

# Start autonomous monitoring
echo "Starting autonomous monitoring..."
nohup python scripts/amm_autonomous_v3.py > logs/autonomous_v3.out 2>&1 &
MONITOR_PID=$!
echo "✅ Monitoring started (PID: $MONITOR_PID)"

echo ""
echo "=============================================="
echo "SYSTEM RUNNING"
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
echo "  ps aux | grep -E 'amm-500|autonomous'"
echo ""
echo "To stop:"
echo "  kill $BOT_PID $MONITOR_PID"
echo "  # OR send Ctrl+C to monitoring (will stop bot too)"
echo ""
echo "⚠️  Leave terminal open or use 'nohup' if closing"
echo "=============================================="
echo ""

# Optional: schedule stop after duration
if [ "$DURATION" != "3" ]; then
    if [ "$DURATION" = "1" ]; then
        SECONDS=$((7 * 24 * 3600))
    else
        SECONDS=$((30 * 24 * 3600))
    fi
    
    echo "Scheduling automatic stop in $DURATION_NAME..."
    (sleep $SECONDS && kill $BOT_PID $MONITOR_PID) &
    echo "Auto-stop scheduled."
fi

# Keep script running if continuous mode
if [ "$DURATION" = "3" ]; then
    echo "Press Ctrl+C to stop..."
    wait
fi
```

Make it executable:

```bash
chmod +x scripts/start_paper_trading.sh
```

---

### 4. Monitoring Dashboard (Optional)

Create a simple web dashboard for real-time monitoring:

```python
# NEW FILE: scripts/web_dashboard.py
from flask import Flask, jsonify, render_template_string
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

STATE_FILE = Path(__file__).parent.parent / "logs" / "autonomous_state.json"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AMM-500 Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #00ff00; padding: 20px; }
        .metric { margin: 10px 0; }
        .good { color: #00ff00; }
        .warn { color: #ffaa00; }
        .bad { color: #ff0000; }
        h1 { text-align: center; }
    </style>
</head>
<body>
    <h1>🤖 AMM-500 Live Dashboard</h1>
    <div class="metric">Last Update: {{ last_update }}</div>
    <div class="metric">Cycle: {{ cycle }}</div>
    <div class="metric">Equity: ${{ equity }}</div>
    <div class="metric">PnL: <span class="{{ pnl_class }}">${{ pnl }}</span></div>
    <div class="metric">Drawdown: <span class="{{ dd_class }}">{{ dd }}%</span></div>
    <div class="metric">Trades: {{ trades }}</div>
    <div class="metric">Maker: {{ maker }}%</div>
    <div class="metric">Status: <span class="{{ status_class }}">{{ status }}</span></div>
</body>
</html>
"""

@app.route("/")
def dashboard():
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        
        # Extract metrics
        equity = state.get("wallet_equity", 0)
        pnl = state.get("session_pnl", 0)
        dd = state.get("drawdown_pct", 0)
        trades = state.get("total_trades", 0)
        maker_pct = state.get("maker_ratio", 0) * 100
        
        return render_template_string(
            HTML_TEMPLATE,
            last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cycle=state.get("cycle_count", 0),
            equity=f"{equity:.2f}",
            pnl=f"{pnl:+.2f}",
            pnl_class="good" if pnl >= 0 else "bad",
            dd=f"{dd:.2f}",
            dd_class="good" if dd < 2 else "warn" if dd < 5 else "bad",
            trades=trades,
            maker=f"{maker_pct:.1f}",
            status="RUNNING" if equity > 0 else "ERROR",
            status_class="good" if equity > 0 else "bad"
        )
    except Exception as e:
        return f"Error loading state: {e}", 500

@app.route("/api/state")
def api_state():
    try:
        with open(STATE_FILE) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

Run dashboard:

```bash
python scripts/web_dashboard.py &
# Access at http://localhost:8080
```

---

### 5. Comprehensive Test Suite

```python
# tests/test_autonomous_v3.py
import pytest
from datetime import datetime, timedelta
from scripts.amm_autonomous_v3 import (
    WalletTracker,
    WalletState,
    AlertManager,
    AlertState,
    KillSwitchManager,
    ProcessManager,
    TradeMetrics
)

def test_wallet_tracker_drawdown():
    """Test drawdown calculation."""
    tracker = WalletTracker("0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C")
    tracker.peak_equity = 1500
    tracker.session_start_equity = 1400
    
    # 10% drawdown from peak
    dd = tracker.get_drawdown_pct(1350)
    assert dd == 10.0
    
    # Session PnL
    session_pnl = tracker.get_session_pnl(1450)
    assert session_pnl == 50.0

def test_alert_cooldown():
    """Test alert cooldown mechanism."""
    alert_mgr = AlertManager()
    
    # First alert should be allowed
    assert alert_mgr.state.can_send_alert("drawdown")
    
    # Mark as sent
    alert_mgr.state.last_drawdown_alert = datetime.now()
    
    # Second alert within cooldown should be blocked
    assert not alert_mgr.state.can_send_alert("drawdown", cooldown_minutes=30)
    
    # After cooldown expires, should be allowed
    alert_mgr.state.last_drawdown_alert = datetime.now() - timedelta(minutes=31)
    assert alert_mgr.state.can_send_alert("drawdown", cooldown_minutes=30)

def test_kill_switch_triggers():
    """Test kill switch conditions."""
    kill_switch = KillSwitchManager()
    
    # Normal conditions - no trigger
    assert not kill_switch.check_kill_conditions(
        drawdown_pct=2.0,
        session_pnl=-20,
        consecutive_losses=5
    )
    
    # Drawdown exceeded
    assert kill_switch.check_kill_conditions(
        drawdown_pct=6.0,
        session_pnl=-20,
        consecutive_losses=5
    )
    assert "Drawdown exceeded" in kill_switch.trigger_reason
    
    # Consecutive losses
    kill_switch.triggered = False
    assert kill_switch.check_kill_conditions(
        drawdown_pct=2.0,
        session_pnl=-20,
        consecutive_losses=11
    )
    assert "Consecutive losses" in kill_switch.trigger_reason
    
    # Session loss
    kill_switch.triggered = False
    assert kill_switch.check_kill_conditions(
        drawdown_pct=2.0,
        session_pnl=-150,
        consecutive_losses=5
    )
    assert "Session loss" in kill_switch.trigger_reason

def test_process_manager_restart_limit():
    """Test restart rate limiting."""
    pm = ProcessManager()
    
    # Should allow first restart
    assert pm.can_restart()
    
    # Simulate 5 restarts in last hour
    for _ in range(5):
        pm.restart_history.append(datetime.now())
    
    # Should block 6th restart
    assert not pm.can_restart()
    
    # After 1 hour, should allow again
    pm.restart_history.clear()
    pm.restart_history.append(datetime.now() - timedelta(hours=2))
    assert pm.can_restart()

def test_trade_metrics():
    """Test trade metrics tracking."""
    metrics = TradeMetrics()
    
    # Add some trades
    metrics.total_trades = 100
    metrics.maker_trades = 95
    metrics.taker_trades = 5
    metrics.realized_pnl = 25.50
    metrics.total_fees = 5.25
    
    # Calculate session PnL
    metrics.session_pnl = metrics.realized_pnl - metrics.total_fees
    
    assert metrics.session_pnl == 20.25
    assert metrics.maker_trades / metrics.total_trades == 0.95  # 95% maker

def test_wallet_state_healthy():
    """Test wallet health check."""
    # Healthy state
    state = WalletState(
        equity=1500,
        margin_used=300,
        available_margin=1200,
        unrealized_pnl=50,
        margin_ratio=0.2
    )
    assert state.is_healthy
    
    # Unhealthy state (high margin)
    state_bad = WalletState(
        equity=1500,
        margin_used=1300,
        available_margin=200,
        unrealized_pnl=-100,
        margin_ratio=0.85
    )
    assert not state_bad.is_healthy
```

Run tests:

```bash
source .venv/bin/activate
pytest tests/test_autonomous_v3.py -v
```

---

## Complete Workflow for 7-Day Paper Trading

### Step 1: Configuration

Edit `config/.env`:

```bash
# Wallet
WALLET_ADDRESS=0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C
PRIVATE_KEY=your_private_key_here

# Trading
TESTNET=False  # Mainnet data, paper orders
LEVERAGE=10
COLLATERAL=1000
MIN_SPREAD_BPS=5
MAX_SPREAD_BPS=50

# Alerts
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ALERT_EMAIL=your-alert-email@gmail.com
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Step 2: Start Paper Trading

```bash
cd /Users/nheosdisplay/VSC/AMM/AMM-500
source .venv/bin/activate
./scripts/start_paper_trading.sh
# Select: 1 (Paper Trading)
# Select: 1 (7 days)
```

### Step 3: Monitor

```bash
# Watch bot logs
tail -f logs/bot_$(date +%Y-%m-%d).log

# Watch monitoring logs
tail -f logs/autonomous_v3.log

# Check processes
ps aux | grep -E 'amm-500|autonomous'
```

### Step 4: Check Metrics (After 7 Days)

```python
# scripts/analyze_paper_results.py
import json
from pathlib import Path
from datetime import datetime, timedelta

def analyze_7day_results():
    """Analyze 7-day paper trading results."""
    state_file = Path("logs/autonomous_state.json")
    
    with open(state_file) as f:
        state = json.load(f)
    
    # Extract metrics
    total_trades = state.get("total_trades", 0)
    session_pnl = state.get("cumulative_pnl", 0)
    total_fees = state.get("total_fees_paid", 0)
    maker_trades = state.get("maker_trades", 0)
    
    # Calculate
    trades_per_day = total_trades / 7
    maker_ratio = maker_trades / total_trades if total_trades > 0 else 0
    net_pnl = session_pnl - total_fees
    roi = (net_pnl / 1000) * 100  # Assuming $1000 collateral
    sharpe = calculate_sharpe(...)  # From daily PnL
    
    print("=== 7-DAY PAPER TRADING RESULTS ===")
    print(f"Total Trades: {total_trades} ({trades_per_day:.1f}/day)")
    print(f"Maker Ratio: {maker_ratio:.1%}")
    print(f"Gross PnL: ${session_pnl:+.2f}")
    print(f"Fees Paid: ${total_fees:.2f}")
    print(f"Net PnL: ${net_pnl:+.2f}")
    print(f"ROI (7d): {roi:+.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Targets
    print("\n=== TARGET COMPARISON ===")
    print(f"Trades/day: {trades_per_day:.0f} (target: >1000)")
    print(f"Maker ratio: {maker_ratio:.1%} (target: >90%)")
    print(f"ROI: {roi:+.2f}% (target: >5%)")
    print(f"Sharpe: {sharpe:.2f} (target: 1.5-3.0)")

if __name__ == "__main__":
    analyze_7day_results()
```

---

## Troubleshooting

### Bot Not Starting

```bash
# Check Python environment
source .venv/bin/activate
python amm-500.py --status

# Check dependencies
pip install -r requirements.txt

# Check credentials
python -c "from dotenv import load_dotenv; import os; load_dotenv('config/.env'); print('Wallet:', os.getenv('WALLET_ADDRESS'))"
```

### Monitoring Not Sending Alerts

```bash
# Test email
python -c "
import smtplib
from email.mime.text import MIMEText
msg = MIMEText('Test')
msg['Subject'] = 'Test'
msg['From'] = 'your-email@gmail.com'
msg['To'] = 'your-email@gmail.com'
with smtplib.SMTP('smtp.gmail.com', 587) as s:
    s.starttls()
    s.login('your-email@gmail.com', 'your-app-password')
    s.send_message(msg)
print('Email sent!')
"

# Test Slack
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{"text":"Test from AMM-500"}'
```

### High Memory Usage

```bash
# Check process memory
ps aux | grep python | awk '{print $2, $4, $11}'

# If >1GB, restart with limits
ulimit -v 1048576  # 1GB limit
python scripts/amm_autonomous_v3.py
```

---

## Final Checklist

- [ ] `config/.env` configured with wallet and API keys
- [ ] Email/Slack alerts configured and tested
- [ ] `start_paper_trading.sh` executable
- [ ] Virtual environment activated
- [ ] BTC historical data fetched
- [ ] Kill switch thresholds reviewed
- [ ] Tests passing (`pytest tests/`)
- [ ] Launch daemon configured (optional)
- [ ] Monitoring dashboard running (optional)

---

## Support

For issues:
1. Check logs: `logs/bot_YYYY-MM-DD.log` and `logs/autonomous_v3.log`
2. Review state: `cat logs/autonomous_state.json | jq`
3. Test connection: `python amm-500.py --status`
4. Run tests: `pytest tests/ -v`

---

**End of Autonomous System Setup Guide**
