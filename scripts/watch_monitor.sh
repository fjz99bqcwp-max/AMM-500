#!/bin/bash
# Watch monitor output and track cycles

echo "Waiting for 5 monitoring cycles..."
echo "Started: $(date)"
echo ""

CYCLE_COUNT=0
TARGET_CYCLES=5

while [ $CYCLE_COUNT -lt $TARGET_CYCLES ]; do
    sleep 60
    
    # Check if monitor is still running
    if ! ps aux | grep "amm_autonomous.py" | grep -v grep > /dev/null; then
        echo "Monitor stopped running!"
        break
    fi
    
    echo "$(date) - Waiting for cycles... (check terminal for live output)"
done

echo ""
echo "Monitoring period complete!"
echo "Check terminal for full cycle details"
