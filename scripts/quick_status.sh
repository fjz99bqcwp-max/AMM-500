#!/bin/bash
# Quick monitoring script to check system status

echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SYSTEM STATUS CHECK - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check processes
echo "🤖 Processes:"
if pgrep -f "amm-500.py" > /dev/null; then
    echo "  ✅ Bot running (PID: $(pgrep -f amm-500.py))"
else
    echo "  ❌ Bot STOPPED"
fi

if pgrep -f "mmb_continuous" > /dev/null; then
    echo "  ✅ Monitor running (PID: $(pgrep -f mmb_continuous))"
else
    echo "  ❌ Monitor STOPPED"
fi

echo ""

# Check current cycle
if [ -f logs/monitor_state.json ]; then
    CYCLE=$(python3 -c "import json; print(json.load(open('logs/monitor_state.json'))['cycle_count'])" 2>/dev/null)
    RESTARTS=$(python3 -c "import json; print(json.load(open('logs/monitor_state.json'))['bot_restarts'])" 2>/dev/null)
    echo "📈 Progress:"
    echo "  Current cycle: $CYCLE/50"
    echo "  Bot restarts: $RESTARTS"
    echo ""
fi

# Recent bot activity
echo "📊 Recent Activity:"
if [ -f logs/bot_run.log ]; then
    LAST_ACTION=$(tail -1 logs/bot_run.log | grep -oE "Actions: [0-9]+" | cut -d' ' -f2)
    FILL_RATE=$(tail -1 logs/bot_run.log | grep -oE "Fill Rate: [0-9.]+%" | cut -d' ' -f3)
    echo "  Actions: ${LAST_ACTION:-N/A}"
    echo "  Fill rate: ${FILL_RATE:-N/A}"
fi

echo ""

# Account status
echo "💰 Account:"
python3 -c "
import requests
resp = requests.post('https://api.hyperliquid.xyz/info', json={'type':'clearinghouseState','user':'0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283'}, timeout=5)
data = resp.json()
value = data['marginSummary']['accountValue']
pnl = float(value) - 1000.0
print(f'  Value: \${value}')
print(f'  PnL: \${pnl:+.2f} ({pnl/10:.2f}%)')
" 2>/dev/null || echo "  Unable to fetch account data"

echo ""

# Recent fills
echo "🔄 Recent Fills (last 2 hours):"
python3 -c "
import requests
from datetime import datetime
resp = requests.post('https://api.hyperliquid.xyz/info', json={'type':'userFills','user':'0x90c6949CD6850c9d3995Ed438b8653Cc6CEE6283'}, timeout=5)
fills = resp.json()
now_ms = int(datetime.now().timestamp() * 1000)
recent = [f for f in fills if int(f['time']) > now_ms - 2*60*60*1000]
print(f'  Total: {len(recent)} fills')
if recent:
    buys = len([f for f in recent if f['side'] == 'B'])
    sells = len([f for f in recent if f['side'] == 'A'])
    print(f'  Buys: {buys} | Sells: {sells}')
" 2>/dev/null || echo "  Unable to fetch fills"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "💡 Commands:"
echo "  View live log:  tail -f logs/monitor_live.log"
echo "  View bot log:   tail -f logs/bot_run.log"
echo "  Check status:   python3 scripts/dashboard.py"
echo "════════════════════════════════════════════════════════════════════════════"
