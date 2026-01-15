#!/usr/bin/env python3
"""
Log Monitor - Display and analyze bot logs with dark-theme colors.

This script monitors the bot's logs in real-time and provides a summary
of trading activity, errors, and performance metrics.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter, defaultdict
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import Colors


def parse_log_line(line: str) -> dict:
    """Parse a log line and extract components."""
    # Pattern: YYYY-MM-DD HH:MM:SS | LEVEL | module:function:line - message
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)\s+\| ([^:]+):([^:]+):(\d+) - (.+)'
    match = re.match(pattern, line)
    
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2),
            'module': match.group(3),
            'function': match.group(4),
            'line': match.group(5),
            'message': match.group(6),
        }
    return None


def colorize_level(level: str) -> str:
    """Colorize log level for console output."""
    colors = {
        'DBUG': Colors.GRAY,
        'INFO': Colors.CYAN,
        'SUCC': Colors.GREEN,
        'WARN': Colors.YELLOW,
        'ERR!': Colors.RED,
        'CRIT': Colors.RED + Colors.BOLD,
    }
    color = colors.get(level, Colors.RESET)
    return f"{color}[{level}]{Colors.RESET}"


def analyze_logs(log_file: Path, duration_minutes: int = 10) -> dict:
    """Analyze logs and return summary statistics."""
    stats = {
        'total_lines': 0,
        'by_level': Counter(),
        'by_module': Counter(),
        'errors': [],
        'warnings': [],
        'trades': [],
        'orders_placed': 0,
        'orders_cancelled': 0,
        'websocket_reconnects': 0,
        'rate_limits': 0,
        'start_time': None,
        'end_time': None,
    }
    
    if not log_file.exists():
        return stats
    
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if not parsed:
                continue
            
            stats['total_lines'] += 1
            stats['by_level'][parsed['level']] += 1
            stats['by_module'][parsed['module']] += 1
            
            # Track timestamps
            if stats['start_time'] is None:
                stats['start_time'] = parsed['timestamp']
            stats['end_time'] = parsed['timestamp']
            
            # Collect errors and warnings
            if parsed['level'] in ['ERR!', 'CRIT']:
                stats['errors'].append(parsed['message'])
            elif parsed['level'] == 'WARN':
                stats['warnings'].append(parsed['message'])
            
            # Track specific events
            msg = parsed['message'].lower()
            if 'order placed' in msg:
                stats['orders_placed'] += 1
            elif 'cancelled' in msg and 'order' in msg:
                stats['orders_cancelled'] += 1
            elif 'websocket' in msg and 'reconnect' in msg:
                stats['websocket_reconnects'] += 1
            elif 'rate limited' in msg:
                stats['rate_limits'] += 1
            elif 'fill' in msg or 'trade executed' in msg:
                stats['trades'].append(parsed['message'])
    
    return stats


def print_summary(stats: dict) -> None:
    """Print a formatted summary of log statistics."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}BOT LOG SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    
    # Time range
    if stats['start_time'] and stats['end_time']:
        print(f"{Colors.GRAY}Time Range:{Colors.RESET} {stats['start_time']} → {stats['end_time']}")
    print(f"{Colors.GRAY}Total Log Lines:{Colors.RESET} {stats['total_lines']:,}\n")
    
    # Log levels
    print(f"{Colors.BOLD}Log Levels:{Colors.RESET}")
    for level, count in stats['by_level'].most_common():
        color = {
            'DBUG': Colors.GRAY,
            'INFO': Colors.CYAN,
            'SUCC': Colors.GREEN,
            'WARN': Colors.YELLOW,
            'ERR!': Colors.RED,
            'CRIT': Colors.RED,
        }.get(level, Colors.RESET)
        print(f"  {color}{level:4s}{Colors.RESET}: {count:5,} ({count/stats['total_lines']*100:5.1f}%)")
    
    # Top modules
    print(f"\n{Colors.BOLD}Top Modules:{Colors.RESET}")
    for module, count in stats['by_module'].most_common(10):
        print(f"  {Colors.CYAN}{module[:20]:20s}{Colors.RESET}: {count:5,}")
    
    # Trading activity
    print(f"\n{Colors.BOLD}Trading Activity:{Colors.RESET}")
    print(f"  {Colors.GREEN}Orders Placed:{Colors.RESET}     {stats['orders_placed']:5,}")
    print(f"  {Colors.YELLOW}Orders Cancelled:{Colors.RESET}  {stats['orders_cancelled']:5,}")
    print(f"  {Colors.BLUE}Fills/Trades:{Colors.RESET}      {len(stats['trades']):5,}")
    
    # System events
    print(f"\n{Colors.BOLD}System Events:{Colors.RESET}")
    print(f"  {Colors.ORANGE}WebSocket Reconnects:{Colors.RESET} {stats['websocket_reconnects']:3,}")
    print(f"  {Colors.YELLOW}Rate Limits Hit:{Colors.RESET}      {stats['rate_limits']:3,}")
    
    # Errors
    if stats['errors']:
        print(f"\n{Colors.BOLD}{Colors.RED}Errors ({len(stats['errors'])}):{Colors.RESET}")
        for i, error in enumerate(stats['errors'][:5], 1):
            print(f"  {Colors.RED}{i}.{Colors.RESET} {error[:100]}")
        if len(stats['errors']) > 5:
            print(f"  {Colors.GRAY}...and {len(stats['errors']) - 5} more{Colors.RESET}")
    
    # Warnings (show unique warnings)
    if stats['warnings']:
        unique_warnings = list(set(stats['warnings']))[:5]
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Warnings ({len(stats['warnings'])} total, {len(set(stats['warnings']))} unique):{Colors.RESET}")
        for i, warning in enumerate(unique_warnings, 1):
            print(f"  {Colors.YELLOW}{i}.{Colors.RESET} {warning[:100]}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def main():
    """Monitor logs and display summary."""
    log_dir = Path(__file__).parent.parent / "logs"
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"bot_{today}.log"
    
    print(f"{Colors.CYAN}Monitoring bot logs for 10 minutes...{Colors.RESET}")
    print(f"{Colors.GRAY}Log file: {log_file}{Colors.RESET}\n")
    
    # Monitor for 10 minutes
    start_time = time.time()
    duration = 10 * 60  # 10 minutes
    
    last_line_count = 0
    
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        remaining = duration - elapsed
        
        # Quick stats
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                new_lines = len(lines) - last_line_count
                last_line_count = len(lines)
                
                if new_lines > 0:
                    # Show last few lines
                    print(f"{Colors.GRAY}[{elapsed//60:02d}:{elapsed%60:02d}]{Colors.RESET} +{new_lines} new lines")
                    for line in lines[-min(3, new_lines):]:
                        parsed = parse_log_line(line)
                        if parsed:
                            time_str = parsed['timestamp'].split()[1]
                            level_colored = colorize_level(parsed['level'])
                            module = parsed['module'][:12].ljust(12)
                            print(f"  {Colors.GRAY}{time_str}{Colors.RESET} {level_colored} {Colors.DIM}{module}{Colors.RESET} {parsed['message'][:80]}")
        
        # Progress bar
        progress = int((elapsed / duration) * 50)
        bar = '█' * progress + '░' * (50 - progress)
        print(f"\r{Colors.CYAN}[{bar}]{Colors.RESET} {remaining//60:02d}:{remaining%60:02d} remaining", end='', flush=True)
        
        time.sleep(10)  # Update every 10 seconds
    
    print(f"\n\n{Colors.GREEN}✓ Monitoring complete!{Colors.RESET}\n")
    
    # Generate summary
    stats = analyze_logs(log_file)
    print_summary(stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Monitoring interrupted by user{Colors.RESET}")
        # Still show summary
        log_dir = Path(__file__).parent.parent / "logs"
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"bot_{today}.log"
        if log_file.exists():
            stats = analyze_logs(log_file)
            print_summary(stats)
