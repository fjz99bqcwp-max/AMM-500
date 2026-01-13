#!/usr/bin/env python3
"""Quick test to see debug logs."""
import subprocess
import time
import sys

print("Starting bot...")
proc = subprocess.Popen(
    [sys.executable, "amm-500.py", "--paper"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Waiting 12 seconds...")
time.sleep(12)

print("Terminating bot...")
proc.terminate()
time.sleep(1)

print("\n=== OUTPUT ===")
output = proc.communicate()[0]
for line in output.split('\n'):
    if any(x in line for x in ['_update_orders', 'Early exit', 'EARLY EXIT', 'Orders needed', 'Final check']):
        print(line)
