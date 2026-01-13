#!/usr/bin/env python3
"""
Continuous Monitoring and Auto-Optimization System

This script continuously:
1. Analyzes real-time trading data directly from Hyperliquid API (2000 fills)
2. Detects errors and performance issues
3. Monitors strategy adherence with weighted averages
4. Auto-applies optimizations autonomously
5. Restarts bot if critical errors detected
6. Tracks detailed cycle metrics with state persistence
7. Runs indefinitely with robust error handling
"""

import requests
import time
import subprocess
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# User wallet for real-time data validation
WALLET = "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C"
URL = "https://api.hyperliquid.xyz/info"

# AMM-500 paths (auto-detected)
BASE_DIR = Path(__file__).parent.parent
BOT_SCRIPT = str(BASE_DIR / "amm-500.py")
STATE_FILE = str(BASE_DIR / "logs" / "autonomous_state.json")
VENV_PYTHON = str(BASE_DIR / ".venv" / "bin" / "python")

# Performance thresholds
MIN_SPREAD_BPS = 0.0  # Minimum acceptable spread
MIN_WIN_RATE = 0.50  # 50% minimum win rate
MAX_DRAWDOWN = 0.05  # 5% max drawdown
TARGET_FILLS_PER_HOUR = 10  # Expected fill rate
MAX_CONSECUTIVE_ERRORS = 3  # Auto-restart bot after this many errors


def log(level: str, message: str, file=None):
    """Print log message with timestamp and level, optionally write to file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    msg = f"{timestamp} | {level:5s} | {message}"
    print(msg)
    if file:
        try:
            with open(file, "a") as f:
                f.write(msg + "\n")
                f.flush()
        except Exception:
            pass  # Don't fail on logging errors


class PerformanceMonitor:
    """Monitor and analyze trading performance with autonomous optimization."""

    def __init__(self):
        self.trade_count = 0
        self.cycle_count = 0
        self.last_optimization = None
        self.issues_detected = []  # Reset each cycle
        self.consecutive_errors = 0
        self.last_successful_cycle = datetime.now()
        self.performance_history = []  # Track performance over time
        self.bot_restarts = 0
        self.last_fill_time = 0  # Track last fill timestamp for new trade detection
        self.total_trades_seen = set()  # Track unique fills by hash

        # Balance tracking for optimization
        self.starting_balance: float = 0.0
        self.last_balance: float = 0.0
        self.balance_history: List[Dict] = []  # Track balance over time
        self.session_pnl: float = 0.0  # PnL since monitor started

        self.load_state()

    def load_state(self):
        """Load previous state from disk for continuity."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                    self.cycle_count = state.get("cycle_count", 0)
                    self.bot_restarts = state.get("bot_restarts", 0)
                    self.last_fill_time = state.get("last_fill_time", 0)
                    self.starting_balance = state.get("starting_balance", 0.0)
                    self.last_balance = state.get("last_balance", 0.0)
                    self.balance_history = state.get("balance_history", [])
                    self.session_pnl = state.get("session_pnl", 0.0)
                    log(
                        "INFO",
                        f"Loaded state: {self.cycle_count} cycles, {self.bot_restarts} restarts",
                    )
        except Exception as e:
            log("WARN", f"Could not load state: {e}")

    def save_state(self):
        """Save current state to disk for persistence across restarts."""
        try:
            state = {
                "cycle_count": self.cycle_count,
                "bot_restarts": self.bot_restarts,
                "last_fill_time": self.last_fill_time,
                "last_cycle": datetime.now().isoformat(),
                "performance_history": self.performance_history[-50:],  # Keep last 50
                "starting_balance": self.starting_balance,
                "last_balance": self.last_balance,
                "balance_history": self.balance_history[-100:],  # Keep last 100
                "session_pnl": self.session_pnl,
            }
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log("WARN", f"Could not save state: {e}")

    def get_account_state(self) -> Optional[Dict]:
        """Fetch current account state from Hyperliquid with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    URL, json={"type": "clearinghouseState", "user": WALLET}, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()

                # Log rate limit status
                rate_limit_remaining = resp.headers.get("X-RateLimit-Remaining")
                rate_limit_reset = resp.headers.get("X-RateLimit-Reset")
                rate_limit_limit = resp.headers.get("X-RateLimit-Limit")

                if rate_limit_remaining is not None:
                    remaining_pct = (int(rate_limit_remaining) / int(rate_limit_limit or 100)) * 100
                    log(
                        "INFO",
                        f"📊 Rate Limit: {rate_limit_remaining}/{rate_limit_limit} ({remaining_pct:.1f}%) remaining, reset: {rate_limit_reset}",
                    )
                else:
                    # Log all headers to see what's available for rate limiting
                    rate_headers = {
                        k: v
                        for k, v in resp.headers.items()
                        if any(term in k.lower() for term in ["rate", "limit", "retry", "reset"])
                    }
                    if rate_headers:
                        log("INFO", f"📊 Rate headers found: {rate_headers}")
                    else:
                        log("INFO", f"📊 No rate limit headers found (status: {resp.status_code})")

                # Validate response structure
                if not isinstance(data, dict) or "marginSummary" not in data:
                    log("ERROR", f"Invalid account state response structure")
                    continue

                # Reset error counter on success
                self.consecutive_errors = max(0, self.consecutive_errors - 1)
                return data

            except Exception as e:
                if attempt < max_retries - 1:
                    log("WARN", f"API Error (attempt {attempt+1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    log("ERROR", f"API Error after {max_retries} attempts: {str(e)[:100]}")
                    self.issues_detected.append(f"API Error: {str(e)[:50]}")
                    self.consecutive_errors += 1

        return None

    def get_perp_balance(self, account_state: Optional[Dict]) -> Optional[Dict]:
        """Extract perpetual balance info from account state."""
        if not account_state:
            return None

        try:
            margin_summary = account_state.get("marginSummary", {})

            # Key balance metrics
            account_value = float(margin_summary.get("accountValue", 0))
            total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
            total_ntl_pos = float(margin_summary.get("totalNtlPos", 0))

            # Get position info
            asset_positions = account_state.get("assetPositions", [])
            position_size = 0.0
            unrealized_pnl = 0.0
            entry_px = 0.0
            liquidation_px = 0.0

            for pos in asset_positions:
                pos_info = pos.get("position", {})
                if pos_info.get("coin") == "BTC":
                    position_size = float(pos_info.get("szi", 0))
                    unrealized_pnl = float(pos_info.get("unrealizedPnl", 0))
                    entry_px = float(pos_info.get("entryPx", 0))
                    liquidation_px = float(pos_info.get("liquidationPx", 0) or 0)

            return {
                "equity": account_value,
                "margin_used": total_margin_used,
                "position_notional": total_ntl_pos,
                "position_size": position_size,
                "unrealized_pnl": unrealized_pnl,
                "entry_price": entry_px,
                "liquidation_price": liquidation_px,
            }
        except Exception as e:
            log("ERROR", f"Failed to parse balance: {e}")
            return None

    def update_balance_tracking(self, balance_info: Dict) -> Dict:
        """Update balance tracking and calculate session PnL."""
        if not balance_info:
            return {"error": "No balance info"}

        equity = balance_info["equity"]
        timestamp = datetime.now().isoformat()

        # Initialize starting balance on first run
        if self.starting_balance == 0:
            self.starting_balance = equity
            log("INFO", f"💰 Starting balance set: ${equity:.2f}")

        # Calculate changes
        cycle_change = equity - self.last_balance if self.last_balance > 0 else 0
        session_change = equity - self.starting_balance
        session_pct = (
            (session_change / self.starting_balance * 100) if self.starting_balance > 0 else 0
        )

        # Update session PnL
        self.session_pnl = session_change

        # Track balance history
        self.balance_history.append(
            {
                "timestamp": timestamp,
                "cycle": self.cycle_count,
                "equity": equity,
                "cycle_change": cycle_change,
                "session_pnl": session_change,
                "position_size": balance_info.get("position_size", 0),
                "unrealized_pnl": balance_info.get("unrealized_pnl", 0),
            }
        )

        # Update last balance
        prev_balance = self.last_balance
        self.last_balance = equity

        return {
            "equity": equity,
            "prev_equity": prev_balance,
            "cycle_pnl": cycle_change,
            "session_pnl": session_change,
            "session_pnl_pct": session_pct,
            "position_size": balance_info.get("position_size", 0),
            "unrealized_pnl": balance_info.get("unrealized_pnl", 0),
        }

    def get_fills(self, limit: int = 2000) -> List[Dict]:
        """Fetch recent fills - using max API limit for accuracy."""
        try:
            resp = requests.post(URL, json={"type": "userFills", "user": WALLET}, timeout=10)
            resp.raise_for_status()
            fills = resp.json()

            # Log rate limit status
            rate_limit_remaining = resp.headers.get("X-RateLimit-Remaining")
            rate_limit_reset = resp.headers.get("X-RateLimit-Reset")
            rate_limit_limit = resp.headers.get("X-RateLimit-Limit")

            if rate_limit_remaining is not None:
                remaining_pct = (int(rate_limit_remaining) / int(rate_limit_limit or 100)) * 100
                log(
                    "INFO",
                    f"📊 Fills Rate Limit: {rate_limit_remaining}/{rate_limit_limit} ({remaining_pct:.1f}%) remaining, reset: {rate_limit_reset}",
                )
            else:
                # Log all headers to see what's available for rate limiting
                rate_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if any(term in k.lower() for term in ["rate", "limit", "retry", "reset"])
                }
                if rate_headers:
                    log("INFO", f"📊 Fills Rate headers found: {rate_headers}")
                else:
                    log(
                        "INFO", f"📊 Fills No rate limit headers found (status: {resp.status_code})"
                    )

            if not isinstance(fills, list):
                log("ERROR", f"Invalid fills response type: {type(fills)}")
                return []

            # Validate fill structure
            valid_fills = []
            for f in fills[:limit]:
                if all(key in f for key in ["px", "side", "time", "sz"]):
                    valid_fills.append(f)
                else:
                    log("WARN", f"Invalid fill structure: {f}")

            return valid_fills
        except Exception as e:
            log("ERROR", f"Fills API Error: {e}")
            self.issues_detected.append(f"Fills API Error: {e}")
            return []

    def analyze_recent_performance(self, fills: List[Dict], window_minutes: int = 30) -> Dict:
        """Analyze performance in recent time window with improved accuracy."""
        if not fills:
            return {"error": "No fills available"}

        # Use millisecond timestamp from Hyperliquid API
        now = time.time() * 1000
        cutoff = now - (window_minutes * 60 * 1000)

        recent = [f for f in fills if f.get("time", 0) > cutoff]

        if not recent:
            return {"error": f"No fills in last {window_minutes} minutes"}

        buys = [f for f in recent if f.get("side") == "B"]
        sells = [f for f in recent if f.get("side") == "A"]

        analysis = {
            "total_fills": len(recent),
            "buys": len(buys),
            "sells": len(sells),
            "time_window_min": window_minutes,
        }

        if buys and sells:
            # Calculate weighted average by size for more accuracy
            buy_sum = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in buys)
            buy_size_total = sum(float(f.get("sz", 0)) for f in buys)
            sell_sum = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in sells)
            sell_size_total = sum(float(f.get("sz", 0)) for f in sells)

            avg_buy = buy_sum / buy_size_total if buy_size_total > 0 else 0
            avg_sell = sell_sum / sell_size_total if sell_size_total > 0 else 0
            mid = (avg_buy + avg_sell) / 2
            spread = avg_sell - avg_buy
            spread_bps = (spread / mid) * 10000 if mid > 0 else 0

            # Calculate realized PnL from fees
            total_fees = sum(float(f.get("fee", 0)) for f in recent)
            total_volume = buy_size_total + sell_size_total

            analysis["avg_buy"] = avg_buy
            analysis["avg_sell"] = avg_sell
            analysis["spread_dollars"] = spread
            analysis["spread_bps"] = spread_bps
            analysis["profitable"] = spread_bps > 0
            analysis["total_fees"] = total_fees
            analysis["total_volume_btc"] = total_volume
            analysis["net_pnl_estimate"] = (
                spread * min(buy_size_total, sell_size_total) - total_fees
            )

            # OPT#17: Calculate adverse selection count and fills per hour
            adverse_count = 0
            fills_per_hour = len(recent) * (60 / window_minutes)  # Scale to per hour

            # Count periods with adverse selection (spread < -2 bps in 5-min windows)
            window_size = 5 * 60 * 1000  # 5 minutes in ms
            start_time = min(f.get("time", 0) for f in recent)
            end_time = max(f.get("time", 0) for f in recent)

            for window_start in range(int(start_time), int(end_time), int(window_size)):
                window_end = window_start + window_size
                window_fills = [f for f in recent if window_start <= f.get("time", 0) < window_end]

                if len(window_fills) >= 4:  # Need minimum fills for analysis
                    window_buys = [f for f in window_fills if f.get("side") == "B"]
                    window_sells = [f for f in window_fills if f.get("side") == "A"]

                    if window_buys and window_sells:
                        w_buy_sum = sum(
                            float(f.get("px", 0)) * float(f.get("sz", 0)) for f in window_buys
                        )
                        w_buy_size = sum(float(f.get("sz", 0)) for f in window_buys)
                        w_sell_sum = sum(
                            float(f.get("px", 0)) * float(f.get("sz", 0)) for f in window_sells
                        )
                        w_sell_size = sum(float(f.get("sz", 0)) for f in window_sells)

                        w_avg_buy = w_buy_sum / w_buy_size if w_buy_size > 0 else 0
                        w_avg_sell = w_sell_sum / w_sell_size if w_sell_size > 0 else 0
                        w_mid = (w_avg_buy + w_avg_sell) / 2
                        w_spread_bps = (
                            ((w_avg_sell - w_avg_buy) / w_mid) * 10000 if w_mid > 0 else 0
                        )

                        if w_spread_bps < -2.0:
                            adverse_count += 1

            analysis["adverse_selection_count"] = adverse_count
            analysis["fills_per_hour"] = fills_per_hour
        else:
            analysis["error"] = "Insufficient buy/sell balance"

        return analysis

    def check_strategy_adherence(self, fills: List[Dict]) -> List[str]:
        """Check if strategy is following rules with real Hyperliquid data."""
        issues = []

        # Check for OPT#14 adaptive behavior on last 20 fills
        recent_20 = fills[:20]
        buys_20 = [f for f in recent_20 if f.get("side") == "B"]
        sells_20 = [f for f in recent_20 if f.get("side") == "A"]

        if buys_20 and sells_20:
            # Weighted average by size
            buy_sum = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in buys_20)
            buy_size = sum(float(f.get("sz", 0)) for f in buys_20)
            sell_sum = sum(float(f.get("px", 0)) * float(f.get("sz", 0)) for f in sells_20)
            sell_size = sum(float(f.get("sz", 0)) for f in sells_20)

            avg_buy_20 = buy_sum / buy_size if buy_size > 0 else 0
            avg_sell_20 = sell_sum / sell_size if sell_size > 0 else 0
            mid_20 = (avg_buy_20 + avg_sell_20) / 2
            spread_bps_20 = ((avg_sell_20 - avg_buy_20) / mid_20) * 10000 if mid_20 > 0 else 0

            if spread_bps_20 < -2.0:
                issues.append(
                    f"ADVERSE SELECTION: {spread_bps_20:.2f} bps (should trigger $5 defensive distance)"
                )
            elif spread_bps_20 < 0:
                issues.append(f"NEGATIVE SPREAD: {spread_bps_20:.2f} bps")
            elif spread_bps_20 > 15:
                issues.append(f"EXCELLENT SPREAD: {spread_bps_20:.2f} bps (strategy working well)")

        # Check fill rate in last hour
        now = time.time() * 1000
        one_hour_ago = now - (60 * 60 * 1000)
        fills_last_hour = [f for f in fills if f.get("time", 0) > one_hour_ago]

        fill_rate = len(fills_last_hour)
        if fill_rate < TARGET_FILLS_PER_HOUR:
            issues.append(
                f"LOW FILL RATE: {fill_rate} fills/hour (target: {TARGET_FILLS_PER_HOUR}+)"
            )
        elif fill_rate > 200:
            issues.append(
                f"HIGH FILL RATE: {fill_rate} fills/hour (may indicate adverse selection)"
            )

        return issues

    def detect_errors(self, account_state: Optional[Dict]) -> List[str]:
        """Detect errors in account state or execution."""
        errors = []

        if not account_state:
            errors.append("CRITICAL: Cannot fetch account state")
            return errors

        # Check for position issues
        margin_summary = account_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))

        if account_value < 900:  # Started with $1000
            errors.append(f"DRAWDOWN WARNING: Account value ${account_value:.2f} (< $900)")

        # Check for stuck positions
        asset_positions = account_state.get("assetPositions", [])
        for pos in asset_positions:
            position_value = abs(float(pos.get("position", {}).get("szi", 0)))
            if position_value > 0.01:  # Large position
                errors.append(f"LARGE POSITION: {position_value:.5f} BTC")

        return errors

    def optimize_strategy(
        self, analysis: Dict, issues: List[str], balance_update: Dict = None
    ) -> Optional[Dict]:
        """Determine optimization needs autonomously using real balance data."""
        optimizations = {}

        spread_bps = analysis.get("spread_bps", 0)
        net_pnl = analysis.get("net_pnl_estimate", 0)
        total_fees = analysis.get("total_fees", 0)

        # Use REAL balance data for optimization decisions
        cycle_pnl = 0.0
        session_pnl = 0.0
        session_pnl_pct = 0.0
        equity = 0.0

        if balance_update and "equity" in balance_update:
            cycle_pnl = balance_update.get("cycle_pnl", 0)
            session_pnl = balance_update.get("session_pnl", 0)
            session_pnl_pct = balance_update.get("session_pnl_pct", 0)
            equity = balance_update.get("equity", 0)

        # Check for adverse selection
        adverse_selection = any("ADVERSE SELECTION" in issue for issue in issues)

        # Calculate recent balance trend (last 5 cycles)
        recent_pnl_trend = 0.0
        if len(self.balance_history) >= 5:
            recent_changes = [h.get("cycle_change", 0) for h in self.balance_history[-5:]]
            recent_pnl_trend = sum(recent_changes)

        # OPT#17: Enhanced adaptive modes with aggressive for >+10 bps
        # Priority: Balance-based decisions over spread-based
        if session_pnl_pct < -0.5 or (cycle_pnl < -0.50 and recent_pnl_trend < -1.0):
            # Real money loss detected - emergency defensive
            optimizations["mode"] = "EMERGENCY_DEFENSIVE"
            optimizations["defensive_distance"] = 10.0
            optimizations["order_levels"] = 1
            optimizations["reason"] = (
                f"REAL LOSS: cycle=${cycle_pnl:+.2f}, session=${session_pnl:+.2f} ({session_pnl_pct:+.2f}%)"
            )
            optimizations["urgency"] = "CRITICAL"
        elif adverse_selection or spread_bps < -3.0 or cycle_pnl < -0.20:  # Tighter threshold
            optimizations["mode"] = "DEFENSIVE"
            optimizations["defensive_distance"] = 6.0  # Reduced from 8.0
            optimizations["order_levels"] = 1
            optimizations["reason"] = (
                f"Adverse: spread={spread_bps:.1f}bps, cycle_pnl=${cycle_pnl:+.2f}"
            )
            optimizations["urgency"] = "HIGH"
        elif spread_bps < 1.0 or cycle_pnl < 0:  # Tighter thresholds
            optimizations["mode"] = "MODERATE"
            optimizations["defensive_distance"] = 3.0  # Reduced from 5.5
            optimizations["order_levels"] = 2
            optimizations["reason"] = (
                f"Negative: spread={spread_bps:.1f}bps, cycle_pnl=${cycle_pnl:+.2f}"
            )
            optimizations["urgency"] = "MEDIUM"
        elif spread_bps > 15.0 and cycle_pnl > 0.15:  # More aggressive trigger
            optimizations["mode"] = "VERY_AGGRESSIVE"
            optimizations["defensive_distance"] = 1.0  # Very close to BBO
            optimizations["order_levels"] = 8  # More levels for high profitability
            optimizations["reason"] = (
                f"Very Profitable: spread={spread_bps:.1f}bps, cycle_pnl=${cycle_pnl:+.2f}"
            )
            optimizations["urgency"] = "LOW"
        elif spread_bps > 10.0 and cycle_pnl > 0.10:  # NEW: Aggressive mode
            optimizations["mode"] = "AGGRESSIVE"
            optimizations["defensive_distance"] = 2.0
            optimizations["order_levels"] = 5
            optimizations["reason"] = (
                f"Excellent: spread={spread_bps:.1f}bps, cycle_pnl=${cycle_pnl:+.2f}"
            )
            optimizations["urgency"] = "LOW"
        else:
            optimizations["mode"] = "NORMAL"
            optimizations["defensive_distance"] = 3.0
            optimizations["order_levels"] = 3
            optimizations["reason"] = (
                f"Stable: spread={spread_bps:.1f}bps, cycle_pnl=${cycle_pnl:+.2f}"
            )
            optimizations["urgency"] = "LOW"

        # Add balance metrics to optimization
        optimizations["equity"] = equity
        optimizations["cycle_pnl"] = cycle_pnl
        optimizations["session_pnl"] = session_pnl
        optimizations["session_pnl_pct"] = session_pnl_pct

        # Additional warnings
        warnings = []
        if total_fees > 5 and net_pnl < total_fees * 0.5:
            warnings.append(f"HIGH FEES: ${total_fees:.2f} eating into profits")
        if analysis.get("total_fills", 0) < 5:
            warnings.append("LOW FILLS - consider tightening spreads")
        if "LOW FILL RATE" in str(issues):
            warnings.append("LOW FILL RATE - may need parameter adjustment")

        if warnings:
            optimizations["warnings"] = warnings

        return optimizations

    def check_bot_running(self) -> bool:
        """Check if bot is currently running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "amm-500.py"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def restart_bot(self, reason: str = "Auto-restart"):
        """Restart the trading bot autonomously with proper logging."""
        log("WARN", f"🔄 RESTARTING BOT: {reason}")
        self.bot_restarts += 1

        # Kill existing bot gracefully
        subprocess.run(["pkill", "-f", "amm-500.py"], capture_output=True)
        time.sleep(3)

        # Verify bot stopped
        if self.check_bot_running():
            log("WARN", "Bot still running, forcing kill...")
            subprocess.run(["pkill", "-9", "-f", "amm-500.py"], capture_output=True)
            time.sleep(2)

        # Start new instance in background with proper logging
        log_file = str(BASE_DIR / "logs" / f"bot_{datetime.now().strftime('%Y-%m-%d')}.log")
        try:
            subprocess.Popen(
                [VENV_PYTHON, BOT_SCRIPT],
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )
            log("INFO", "✅ Bot restarted")
        except Exception as e:
            log("ERROR", f"Failed to restart bot: {e}")

        time.sleep(5)

    def monitor_cycle(self):
        """Execute one monitoring cycle with direct Hyperliquid data."""
        self.cycle_count += 1
        self.issues_detected = []  # Reset issues for this cycle

        print("\n" + "=" * 80)
        log("INFO", f"🔍 MONITORING CYCLE #{self.cycle_count}")
        print("=" * 80)

        # 1. Fetch data directly from Hyperliquid
        log("INFO", "📊 Fetching account data from Hyperliquid...")
        account_state = self.get_account_state()
        fills = self.get_fills(2000)  # Fetch max available data

        # 2. Fetch and track perpetual balance
        balance_info = self.get_perp_balance(account_state)
        balance_update = self.update_balance_tracking(balance_info) if balance_info else {}

        # Log balance info
        if balance_info:
            equity = balance_info["equity"]
            position = balance_info["position_size"]
            unrealized = balance_info["unrealized_pnl"]
            cycle_pnl = balance_update.get("cycle_pnl", 0)
            session_pnl = balance_update.get("session_pnl", 0)
            session_pct = balance_update.get("session_pnl_pct", 0)

            log(
                "DATA",
                f"💰 Equity: ${equity:.2f} | Position: {position:+.4f} BTC | Unrealized: ${unrealized:+.2f}",
            )
            if self.cycle_count > 1:
                pnl_emoji = "📈" if cycle_pnl >= 0 else "📉"
                log(
                    "DATA",
                    f"{pnl_emoji} Cycle PnL: ${cycle_pnl:+.4f} | Session PnL: ${session_pnl:+.2f} ({session_pct:+.2f}%)",
                )
        else:
            log("WARN", "⚠️  Could not fetch balance")

        if not fills:
            log("WARN", "⚠️  No fills yet - bot may be starting up")
            return

        # Count new trades using timestamp-based detection
        # Each fill has a unique time+side+px+sz combination
        current_fill_hashes = set()
        new_trades = 0
        latest_fill_time = 0

        for f in fills:
            fill_hash = (
                f"{f.get('time', 0)}_{f.get('side', '')}_{f.get('px', '')}_{f.get('sz', '')}"
            )
            current_fill_hashes.add(fill_hash)
            fill_time = f.get("time", 0)
            if fill_time > latest_fill_time:
                latest_fill_time = fill_time
            # Count fills newer than our last recorded fill time
            if fill_time > self.last_fill_time:
                new_trades += 1

        # Update tracking
        if latest_fill_time > self.last_fill_time:
            self.last_fill_time = latest_fill_time
        self.trade_count = len(fills)

        log("INFO", f"Total fills: {len(fills)} (new since last cycle: {new_trades})")

        # 2. Analyze performance with weighted averages
        log("INFO", "📈 Analyzing performance (weighted by size)...")
        analysis_30m = self.analyze_recent_performance(fills, 30)

        # Initialize variables with defaults
        spread_bps = 0
        net_pnl = 0
        total_fees = 0
        volume_btc = 0

        if "error" not in analysis_30m:
            spread_bps = analysis_30m.get("spread_bps", 0)
            net_pnl = analysis_30m.get("net_pnl_estimate", 0)
            total_fees = analysis_30m.get("total_fees", 0)
            volume_btc = analysis_30m.get("total_volume_btc", 0)
            profitable = analysis_30m.get("profitable", False)
            status = "✅" if profitable else "❌"

            log("DATA", f"30-min window: {analysis_30m['total_fills']} fills")
            log("DATA", f"Spread: {spread_bps:+.2f} bps {status}")

            # OPT#17: Add adverse selection count and fills rate/hour
            adverse_count = analysis_30m.get("adverse_selection_count", 0)
            fills_per_hour = analysis_30m.get("fills_per_hour", 0)
            log(
                "DATA",
                f"Adverse Selection: {adverse_count} instances | Fill Rate: {fills_per_hour:.1f} fills/hour",
            )

            log(
                "DATA",
                f"Net PnL: ${net_pnl:+.4f} | Fees: ${total_fees:.4f} | Volume: {volume_btc:.6f} BTC",
            )
            log("DATA", f"Buys: {analysis_30m['buys']} @ ${analysis_30m.get('avg_buy', 0):,.2f}")
            log("DATA", f"Sells: {analysis_30m['sells']} @ ${analysis_30m.get('avg_sell', 0):,.2f}")
        else:
            log("WARN", f"{analysis_30m.get('error')}")
            analysis_30m = {}

        # 3. Check strategy adherence
        log("INFO", "🎯 Checking strategy adherence...")
        strategy_issues = self.check_strategy_adherence(fills)

        if strategy_issues:
            for issue in strategy_issues:
                log("WARN", f"⚠️  {issue}")
        else:
            log("INFO", "✅ Strategy following rules")

        # 4. Detect errors
        log("INFO", "🔍 Detecting errors...")
        errors = self.detect_errors(account_state)

        if errors:
            for error in errors:
                log("ERROR", f"❌ {error}")
        else:
            log("INFO", "✅ No errors detected")

        # 5. Apply optimizations AUTONOMOUSLY (using real balance data)
        log("INFO", "⚙️  Evaluating optimization needs (using real balance)...")
        optimization = self.optimize_strategy(
            analysis_30m, strategy_issues + errors, balance_update
        )

        if optimization:
            mode = optimization.get("mode", "UNKNOWN")
            reason = optimization.get("reason", "")
            urgency = optimization.get("urgency", "LOW")
            cycle_pnl = optimization.get("cycle_pnl", 0)
            session_pnl = optimization.get("session_pnl", 0)
            session_pnl_pct = optimization.get("session_pnl_pct", 0)

            if urgency == "CRITICAL":
                log("OPTIM", f"🚨 EMERGENCY OPTIMIZATION: {mode} MODE")
            elif urgency == "HIGH":
                log("OPTIM", f"🔧 CRITICAL OPTIMIZATION: {mode} MODE")
            else:
                log("OPTIM", f"🔧 OPTIMIZATION RECOMMENDED: {mode} MODE")

            log("OPTIM", f"   Reason: {reason}")

            # Log warnings if any
            if "warnings" in optimization:
                for warning in optimization["warnings"]:
                    log("WARN", f"   ⚠️  {warning}")

            # Track in performance history with REAL balance data
            self.performance_history.append(
                {
                    "cycle": self.cycle_count,
                    "timestamp": datetime.now().isoformat(),
                    "spread_bps": spread_bps,
                    "net_pnl": net_pnl,
                    "cycle_pnl": cycle_pnl,
                    "session_pnl": session_pnl,
                    "session_pnl_pct": session_pnl_pct,
                    "equity": optimization.get("equity", 0),
                    "mode": mode,
                    "urgency": urgency,
                }
            )

            # Auto-apply optimization decision (logged for strategy to pick up)
            apply_message = f"AUTO-APPLY: {mode} mode - bot will adapt on next update"
            log("OPTIM", f"   ✅ {apply_message}")
            self.last_optimization = datetime.now()

        else:
            log("INFO", "✅ No optimization needed - strategy performing well")

        # 6. Check for critical errors requiring restart
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log("ERROR", f"❌ {self.consecutive_errors} consecutive errors - restarting bot")
            self.restart_bot(reason=f"{self.consecutive_errors} consecutive API errors")
            self.consecutive_errors = 0
        elif not self.check_bot_running():
            log("ERROR", "❌ Bot not running - restarting")
            self.restart_bot(reason="Bot process died")

        # Mark successful cycle
        self.last_successful_cycle = datetime.now()
        self.save_state()

        # 7. Summary
        print("\n" + "=" * 80)
        log("INFO", "📊 CYCLE SUMMARY")
        print("=" * 80)
        log("DATA", f"Cycle: #{self.cycle_count}")
        log("DATA", f"Total trades monitored: {self.trade_count}")
        log("DATA", f"New trades this cycle: {new_trades}")
        log("DATA", f"Issues detected: {len(strategy_issues + errors)}")

        # Balance summary (REAL DATA)
        if balance_update and "equity" in balance_update:
            equity = balance_update.get("equity", 0)
            cycle_pnl = balance_update.get("cycle_pnl", 0)
            session_pnl = balance_update.get("session_pnl", 0)
            session_pct = balance_update.get("session_pnl_pct", 0)
            pnl_emoji = "📈" if session_pnl >= 0 else "📉"
            log("DATA", f"💰 Equity: ${equity:.2f}")
            log(
                "DATA",
                f"{pnl_emoji} Cycle PnL: ${cycle_pnl:+.4f} | Session: ${session_pnl:+.2f} ({session_pct:+.2f}%)",
            )

        # Performance summary with key metrics
        if "error" not in analysis_30m:
            perf_status = "✅ PROFITABLE" if analysis_30m.get("profitable") else "❌ LOSING"
            log("DATA", f"Performance: {perf_status}")
            log("DATA", f"  Spread: {analysis_30m.get('spread_bps', 0):+.2f} bps")
            log("DATA", f"  Net PnL (est): ${analysis_30m.get('net_pnl_estimate', 0):+.4f}")
        else:
            log("DATA", f"Performance: ⚠️  INSUFFICIENT DATA")

        print("=" * 80)


def main():
    """Main monitoring loop with autonomous operation."""
    monitor = PerformanceMonitor()

    print("\n" + "=" * 80)
    log("INFO", "🚀 CONTINUOUS MONITORING & AUTO-OPTIMIZATION SYSTEM v2.0")
    print("=" * 80)
    log("INFO", f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("INFO", "Bot: OPT#14 (Adaptive Anti-Picking-Off) + Weighted Averages")
    log("INFO", "Data: Real-time from Hyperliquid API (2000 fills)")
    log("INFO", "Monitoring: 5-minute cycles, autonomous optimization")
    log("INFO", "State: Persistent across restarts")
    log("INFO", "Press Ctrl+C to stop")
    print("=" * 80)

    cycle_interval = 300  # 5 minutes between cycles
    trades_per_cycle_target = 10
    milestone_cycles = [10, 25, 50]  # Report at cycles 10, 25, and 50

    try:
        while True:
            # Execute monitoring cycle
            try:
                monitor.monitor_cycle()

                # Milestone reporting
                if monitor.cycle_count in milestone_cycles:
                    log("INFO", f"🎯 MILESTONE: Reached cycle {monitor.cycle_count}/50")
                    log("INFO", f"   Total restarts: {monitor.bot_restarts}")
                    log("INFO", f"   Consecutive errors: {monitor.consecutive_errors}")

                    if monitor.cycle_count == 50:
                        log("INFO", "   ✅ 50 CYCLES COMPLETE - Mission accomplished!")
                        log("INFO", "   System will continue running indefinitely...")
                    else:
                        log("INFO", f"   Continuing to next milestone...")

            except Exception as e:
                log("ERROR", f"Cycle error: {str(e)[:200]}")
                monitor.consecutive_errors += 1
                if monitor.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log("ERROR", "Too many cycle errors - attempting recovery")
                    time.sleep(30)  # Wait before retry

            log(
                "INFO",
                f"⏳ Next cycle in {cycle_interval}s... (or after {trades_per_cycle_target} new trades)\n",
            )

            # Wait for either time or trade count
            start_trade_count = monitor.trade_count
            for i in range(cycle_interval):
                time.sleep(1)

                # Check if we hit trade target
                if monitor.trade_count - start_trade_count >= trades_per_cycle_target:
                    log(
                        "INFO",
                        f"✅ {trades_per_cycle_target} new trades detected - starting next cycle early",
                    )
                    break

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        log("INFO", "🛑 MONITORING STOPPED BY USER")
        print("=" * 80)
        log("DATA", f"Total cycles: {monitor.cycle_count}")
        log("DATA", f"Total trades monitored: {monitor.trade_count}")
        log("DATA", f"Bot restarts: {monitor.bot_restarts}")
        log(
            "DATA",
            f"Last successful cycle: {monitor.last_successful_cycle.strftime('%Y-%m-%d %H:%M:%S')}",
        )
        log("INFO", f"Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        monitor.save_state()
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
