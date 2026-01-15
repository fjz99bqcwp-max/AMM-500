#!/usr/bin/env python3
"""
Real-Time Trading Analytics Dashboard for Hyperliquid US500-USDH
Analyzes live trading logs and data with colored output optimized for dark themes.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# ANSI color codes optimized for dark terminal backgrounds
class Colors:
    ERROR = '\033[91m'       # Bright red
    WARN = '\033[93m'        # Bright yellow
    INFO = '\033[96m'        # Bright cyan
    SUCCESS = '\033[92m'     # Bright green
    DATA = '\033[95m'        # Bright magenta
    BOLD = '\033[1m'         # Bold
    DIM = '\033[2m'          # Dim
    RESET = '\033[0m'        # Reset


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    total_fills: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    total_volume: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    total_bids_placed: int = 0
    total_asks_placed: int = 0
    total_cancellations: int = 0
    errors_detected: int = 0
    avg_spread: float = 0.0
    uptime_minutes: float = 0.0
    
    @property
    def maker_ratio(self) -> float:
        """Calculate maker/taker ratio."""
        if self.total_fills == 0:
            return 0.0
        return (self.maker_fills / self.total_fills) * 100
    
    @property
    def net_pnl(self) -> float:
        """Net PnL after fees."""
        return self.realized_pnl - self.fees_paid
    
    @property
    def effective_spread(self) -> float:
        """Effective spread captured per trade."""
        if self.total_fills == 0:
            return 0.0
        return self.net_pnl / self.total_fills


class LiveTradingAnalyzer:
    """Analyze real-time trading performance from Hyperliquid logs and data."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.log_dir = workspace_root / "logs"
        self.data_dir = workspace_root / "data"
        
    def analyze_bot_log(self, hours: int = 1) -> TradingMetrics:
        """Analyze bot log file for last N hours."""
        metrics = TradingMetrics()
        
        # Find today's log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"bot_{today}.log"
        
        if not log_file.exists():
            print(f"{Colors.ERROR}Error: Log file not found: {log_file}{Colors.RESET}")
            return metrics
        
        print(f"{Colors.INFO}📊 Analyzing log file: {log_file.name}{Colors.RESET}")
        
        # Calculate time threshold
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Patterns to search for
        patterns = {
            'fill': re.compile(r'FILL.*?maker.*?(\d+\.\d+).*?@(\d+\.\d+)'),
            'taker': re.compile(r'FILL.*?taker', re.IGNORECASE),
            'maker': re.compile(r'FILL.*?maker', re.IGNORECASE),
            'bid_placed': re.compile(r'Placed.*?bid', re.IGNORECASE),
            'ask_placed': re.compile(r'Placed.*?ask', re.IGNORECASE),
            'cancelled': re.compile(r'Cancelled.*?(\d+).*?orders', re.IGNORECASE),
            'error': re.compile(r'ERROR|failed|exception', re.IGNORECASE),
            'timestamp': re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
        }
        
        start_time = None
        end_time = None
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Extract timestamp
                    ts_match = patterns['timestamp'].search(line)
                    if not ts_match:
                        continue
                    
                    try:
                        log_time = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    
                    # Skip old logs
                    if log_time < cutoff_time:
                        continue
                    
                    # Track time range
                    if start_time is None:
                        start_time = log_time
                    end_time = log_time
                    
                    # Count fills
                    if patterns['fill'].search(line):
                        metrics.total_fills += 1
                        if patterns['maker'].search(line):
                            metrics.maker_fills += 1
                        elif patterns['taker'].search(line):
                            metrics.taker_fills += 1
                    
                    # Count orders placed
                    if patterns['bid_placed'].search(line):
                        metrics.total_bids_placed += 1
                    if patterns['ask_placed'].search(line):
                        metrics.total_asks_placed += 1
                    
                    # Count cancellations
                    cancel_match = patterns['cancelled'].search(line)
                    if cancel_match:
                        try:
                            count = int(cancel_match.group(1))
                            metrics.total_cancellations += count
                        except (ValueError, IndexError):
                            pass
                    
                    # Count errors
                    if patterns['error'].search(line):
                        metrics.errors_detected += 1
        
        except Exception as e:
            print(f"{Colors.ERROR}Error reading log file: {e}{Colors.RESET}")
            return metrics
        
        # Calculate uptime
        if start_time and end_time:
            metrics.uptime_minutes = (end_time - start_time).total_seconds() / 60.0
        
        return metrics
    
    def load_trade_log(self) -> Dict:
        """Load trade log JSON data."""
        trade_log_file = self.data_dir / "trade_log.json"
        
        if not trade_log_file.exists():
            return {}
        
        try:
            with open(trade_log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"{Colors.WARN}Warning: Could not load trade log: {e}{Colors.RESET}")
            return {}
    
    def detect_errors(self) -> List[Tuple[str, int]]:
        """Detect and categorize errors from recent logs."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"bot_{today}.log"
        
        if not log_file.exists():
            return []
        
        error_counts = defaultdict(int)
        error_patterns = [
            (r"batch_cancel.*?failed", "Batch Cancel Errors"),
            (r"Order sync.*?failed", "Order Sync Errors"),
            (r"get_open_orders.*?no attribute", "SDK Method Errors"),
            (r"Risk Level:.*?critical", "Critical Risk Alerts"),
            (r"Drawdown.*?exceeds", "Drawdown Alerts"),
            (r"Connection.*?error", "Connection Errors"),
            (r"timeout", "Timeout Errors"),
        ]
        
        try:
            # Only check last 10000 lines for performance
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-10000:] if len(lines) > 10000 else lines
                
                for line in recent_lines:
                    for pattern, error_type in error_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            error_counts[error_type] += 1
        
        except Exception as e:
            print(f"{Colors.ERROR}Error analyzing errors: {e}{Colors.RESET}")
        
        # Sort by count descending
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    def print_dashboard(self, hours: int = 1):
        """Print comprehensive trading dashboard."""
        print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.INFO}   🚀 HYPERLIQUID US500-USDH LIVE TRADING ANALYTICS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")
        
        # Analyze metrics
        metrics = self.analyze_bot_log(hours=hours)
        trade_log = self.load_trade_log()
        errors = self.detect_errors()
        
        # === PERFORMANCE METRICS ===
        print(f"{Colors.BOLD}{Colors.DATA}📈 PERFORMANCE METRICS (Last {hours}h){Colors.RESET}")
        print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
        
        print(f"  {Colors.INFO}⏱️  Uptime:{Colors.RESET}            {metrics.uptime_minutes:.1f} minutes")
        print(f"  {Colors.INFO}📊 Total Fills:{Colors.RESET}        {metrics.total_fills}")
        print(f"  {Colors.SUCCESS}✓  Maker Fills:{Colors.RESET}       {metrics.maker_fills} ({metrics.maker_ratio:.1f}%)")
        print(f"  {Colors.WARN}⚡ Taker Fills:{Colors.RESET}       {metrics.taker_fills}")
        
        if trade_log:
            pnl = trade_log.get('total_pnl', 0.0)
            fees = trade_log.get('total_fees', 0.0)
            volume = trade_log.get('total_volume', 0.0)
            
            pnl_color = Colors.SUCCESS if pnl > 0 else Colors.ERROR
            print(f"  {pnl_color}💰 Realized PnL:{Colors.RESET}      ${pnl:.2f}")
            print(f"  {Colors.WARN}💸 Fees Paid:{Colors.RESET}        ${fees:.2f}")
            print(f"  {Colors.INFO}📦 Total Volume:{Colors.RESET}     ${volume:.2f}")
            
            net_pnl = pnl - fees
            net_color = Colors.SUCCESS if net_pnl > 0 else Colors.ERROR
            print(f"  {net_color}{Colors.BOLD}🎯 Net PnL:{Colors.RESET}          ${net_pnl:.2f}")
        
        print()
        
        # === ORDER ACTIVITY ===
        print(f"{Colors.BOLD}{Colors.DATA}📋 ORDER ACTIVITY{Colors.RESET}")
        print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
        
        print(f"  {Colors.SUCCESS}📤 Bids Placed:{Colors.RESET}       {metrics.total_bids_placed}")
        print(f"  {Colors.ERROR}📥 Asks Placed:{Colors.RESET}       {metrics.total_asks_placed}")
        print(f"  {Colors.WARN}🗑️  Cancellations:{Colors.RESET}     {metrics.total_cancellations}")
        
        total_orders = metrics.total_bids_placed + metrics.total_asks_placed
        fill_rate = (metrics.total_fills / total_orders * 100) if total_orders > 0 else 0.0
        print(f"  {Colors.INFO}🎲 Fill Rate:{Colors.RESET}         {fill_rate:.2f}%")
        
        print()
        
        # === ERROR ANALYSIS ===
        print(f"{Colors.BOLD}{Colors.ERROR}🔍 ERROR ANALYSIS{Colors.RESET}")
        print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
        
        if errors:
            print(f"  {Colors.WARN}⚠️  Total Errors:{Colors.RESET}      {sum(count for _, count in errors)}\n")
            for error_type, count in errors[:5]:  # Show top 5
                bar_length = int(count / max(errors[0][1], 1) * 30)
                bar = '█' * bar_length
                print(f"  {Colors.ERROR}{error_type:30}{Colors.RESET} {Colors.DIM}│{Colors.RESET} {bar} {count}")
        else:
            print(f"  {Colors.SUCCESS}✓ No errors detected!{Colors.RESET}")
        
        print()
        
        # === STRATEGY HEALTH ===
        print(f"{Colors.BOLD}{Colors.DATA}🏥 STRATEGY HEALTH{Colors.RESET}")
        print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
        
        # Health indicators
        health_score = 100
        issues = []
        
        if metrics.maker_ratio < 90:
            health_score -= 20
            issues.append(f"{Colors.WARN}⚠️  Maker ratio below 90% (taker fees reducing profit){Colors.RESET}")
        
        if metrics.errors_detected > 100:
            health_score -= 30
            issues.append(f"{Colors.ERROR}❌ High error rate detected ({metrics.errors_detected} errors){Colors.RESET}")
        
        if metrics.total_fills < 5 and metrics.uptime_minutes > 30:
            health_score -= 15
            issues.append(f"{Colors.WARN}⚠️  Low fill activity for uptime{Colors.RESET}")
        
        if trade_log.get('total_pnl', 0) < -50:
            health_score -= 25
            issues.append(f"{Colors.ERROR}❌ Significant losses detected{Colors.RESET}")
        
        # Print health score with color
        if health_score >= 80:
            health_color = Colors.SUCCESS
            health_emoji = "💚"
        elif health_score >= 60:
            health_color = Colors.WARN
            health_emoji = "💛"
        else:
            health_color = Colors.ERROR
            health_emoji = "❤️"
        
        print(f"  {health_color}{Colors.BOLD}{health_emoji} Health Score: {health_score}/100{Colors.RESET}\n")
        
        if issues:
            print(f"  {Colors.BOLD}Issues Detected:{Colors.RESET}")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"  {Colors.SUCCESS}✓ All systems operating normally{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}\n")
        
        # === RECOMMENDATIONS ===
        if issues:
            print(f"{Colors.BOLD}{Colors.INFO}💡 RECOMMENDATIONS{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
            
            if metrics.maker_ratio < 90:
                print(f"  • Widen spreads to improve maker ratio")
            if metrics.errors_detected > 100:
                print(f"  • Check logs for recurring errors and apply fixes")
            if metrics.total_fills < 5 and metrics.uptime_minutes > 30:
                print(f"  • Consider tightening spreads to increase fill activity")
            if trade_log.get('total_pnl', 0) < -50:
                print(f"  • Review risk parameters and position sizing")
            
            print()


def main():
    """Main entry point."""
    import sys
    
    workspace_root = Path(__file__).parent.parent
    analyzer = LiveTradingAnalyzer(workspace_root)
    
    # Parse hours argument
    hours = 1
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print(f"{Colors.ERROR}Invalid hours argument, using default: 1{Colors.RESET}")
    
    try:
        analyzer.print_dashboard(hours=hours)
    except KeyboardInterrupt:
        print(f"\n{Colors.INFO}Dashboard interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.ERROR}Fatal error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
