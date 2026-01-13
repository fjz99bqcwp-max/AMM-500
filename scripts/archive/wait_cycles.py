#!/usr/bin/env python3
"""
Quick summary of monitoring cycles
"""

import sys
import time
from datetime import datetime


def wait_for_cycles(target_cycles=5):
    """Wait for target number of cycles and provide updates"""
    print(f"\n{'='*60}")
    print(f"MONITORING 5 CYCLES - STARTED {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    cycle_interval = 300  # 5 minutes
    start_time = time.time()

    for i in range(1, target_cycles + 1):
        if i == 1:
            print(f"✅ Cycle #{i} completed")
        else:
            wait_time = cycle_interval
            print(f"\n⏳ Waiting {wait_time//60} minutes for Cycle #{i}...")
            time.sleep(wait_time)
            print(f"✅ Cycle #{i} should be complete - check terminal output")

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"MONITORING COMPLETE - {total_time:.1f} minutes elapsed")
    print(f"Check the monitor terminal for detailed cycle results")
    print(f"{'='*60}\n")

    # Provide summary
    print("📊 ANALYSIS CHECKLIST:")
    print("  □ Did bot detect adverse selection?")
    print("  □ Did OPT#14 switch to DEFENSIVE mode?")
    print("  □ Did spread improve over cycles?")
    print("  □ Were there any errors?")
    print("  □ Did net PnL improve?")


if __name__ == "__main__":
    wait_for_cycles(5)
