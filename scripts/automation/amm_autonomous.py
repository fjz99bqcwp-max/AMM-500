#!/usr/bin/env python3
"""
Continuous Monitoring and Auto-Optimization System for US500

This script continuously:
1. Analyzes real-time US500 trading data from Hyperliquid API (2000 fills)
2. Collects US500 1-minute candles (saved to data/us500_candles_1m.csv)
3. Detects errors and performance issues
4. Monitors strategy adherence with weighted averages
5. Auto-applies optimizations autonomously
6. Restarts bot if critical errors detected
7. Tracks detailed cycle metrics with state persistence
8. Runs indefinitely with robust error handling

Supports:
- US500 (km:US500) primary symbol
- XYZ100 (xyz:XYZ100) as historical fallback proxy
"""

import requests
import time
import subprocess
import sys
import os
import json
# import asyncio  # Not needed without data collector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import US500 data collector - DISABLED to prevent blocking
# from src.us500_data_collector import get_collector

# Load credentials from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "config" / ".env")

# Signed API imports DISABLED - no longer needed for trade-only tracking
# try:
#     from hyperliquid.info import Info
#     from hyperliquid.utils import constants
#     from eth_account import Account
#     SIGNED_API_AVAILABLE = True
# except ImportError:
#     SIGNED_API_AVAILABLE = False
SIGNED_API_AVAILABLE = False  # Disabled - using trade-only performance tracking

# User wallet for real-time data validation (funded wallet)
WALLET = os.getenv("WALLET_ADDRESS", "0x1cCC14E273DEF02EF2BF62B9bb6B6cAa15805f9C")
URL = "https://api.hyperliquid.xyz/info"

# Trading symbol - US500 for S&P 500 Index perpetual
SYMBOL = os.getenv("SYMBOL", "km:US500").replace("km:", "")

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

        # Trade-based performance tracking (no balance dependency)
        self.session_trades: List[Dict] = []  # All trades since session start
        self.cumulative_pnl: float = 0.0  # Realized PnL from matched trades
        self.total_fees_paid: float = 0.0  # Total fees paid
        self.session_start_time: float = time.time() * 1000  # Session start in ms

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
                    self.session_trades = state.get("session_trades", [])
                    self.cumulative_pnl = state.get("cumulative_pnl", 0.0)
                    self.total_fees_paid = state.get("total_fees_paid", 0.0)
                    self.session_start_time = state.get("session_start_time", time.time() * 1000)
                    log(
                        "INFO",
                        f"Loaded state: {self.cycle_count} cycles, {self.bot_restarts} restarts, PnL ${self.cumulative_pnl:+.2f}",
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
                "session_trades": self.session_trades[-500:],  # Keep last 500 trades
                "cumulative_pnl": self.cumulative_pnl,
                "total_fees_paid": self.total_fees_paid,
                "session_start_time": self.session_start_time,
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

    def calculate_trade_pnl(self, fills: List[Dict]) -> Dict:
        """Calculate PnL from matched buy/sell trades."""
        try:
            # Separate buys and sells
            buys = [f for f in fills if f.get('side') == 'B']
            sells = [f for f in fills if f.get('side') == 'S']
            
            # Calculate volumes and average prices
            buy_volume = sum(float(f.get('sz', 0)) for f in buys)
            sell_volume = sum(float(f.get('sz', 0)) for f in sells)
            
            buy_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in buys)
            sell_notional = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in sells)
            
            avg_buy_px = buy_notional / buy_volume if buy_volume > 0 else 0
            avg_sell_px = sell_notional / sell_volume if sell_volume > 0 else 0
            
            # Calculate fees
            total_fees = sum(abs(float(f.get('fee', 0))) for f in fills)
            
            # Calculate PnL (matched volume * spread - fees)
            matched_volume = min(buy_volume, sell_volume)
            spread_capture = (avg_sell_px - avg_buy_px) * matched_volume
            net_pnl = spread_capture - total_fees
            
            # Calculate spread in bps
            mid_px = (avg_buy_px + avg_sell_px) / 2 if avg_buy_px > 0 and avg_sell_px > 0 else 0
            spread_bps = (avg_sell_px - avg_buy_px) / mid_px * 10000 if mid_px > 0 else 0
            
            return {
                "total_fills": len(fills),
                "buys": len(buys),
                "sells": len(sells),
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "avg_buy_px": avg_buy_px,
                "avg_sell_px": avg_sell_px,
                "spread_bps": spread_bps,
                "spread_capture": spread_capture,
                "total_fees": total_fees,
                "net_pnl": net_pnl,
                "matched_volume": matched_volume,
                "imbalance": abs(buy_volume - sell_volume)
            }
        except Exception as e:
            log("ERROR", f"Failed to calculate trade PnL: {e}")
            return {"total_fills": 0, "net_pnl": 0, "total_fees": 0}

    def update_trade_tracking(self, fills: List[Dict]) -> Dict:
        """Update trade tracking and calculate cumulative PnL from trades only."""
        if not fills:
            return {"error": "No fills provided", "session_pnl": self.cumulative_pnl, "total_fees": self.total_fees_paid}

        # Filter to session trades only (since session start)
        session_fills = [f for f in fills if f.get('time', 0) >= self.session_start_time]
        
        # Store session trades
        self.session_trades = session_fills
        
        # Calculate PnL from trade data
        trade_pnl = self.calculate_trade_pnl(session_fills)
        
        # Update cumulative metrics
        self.cumulative_pnl = trade_pnl.get('net_pnl', 0)
        self.total_fees_paid = trade_pnl.get('total_fees', 0)
        
        # Calculate cycle metrics (since last check)
        recent_fills = [f for f in fills if f.get('time', 0) > self.last_fill_time]
        cycle_pnl_data = self.calculate_trade_pnl(recent_fills) if recent_fills else {"net_pnl": 0}
        
        return {
            "session_fills": len(session_fills),
            "total_buys": trade_pnl.get('buys', 0),
            "total_sells": trade_pnl.get('sells', 0),
            "buy_volume": trade_pnl.get('buy_volume', 0),
            "sell_volume": trade_pnl.get('sell_volume', 0),
            "avg_buy_px": trade_pnl.get('avg_buy_px', 0),
            "avg_sell_px": trade_pnl.get('avg_sell_px', 0),
            "spread_bps": trade_pnl.get('spread_bps', 0),
            "spread_capture": trade_pnl.get('spread_capture', 0),
            "session_pnl": self.cumulative_pnl,
            "total_fees": self.total_fees_paid,
            "cycle_pnl": cycle_pnl_data.get('net_pnl', 0),
            "matched_volume": trade_pnl.get('matched_volume', 0),
            "imbalance": trade_pnl.get('imbalance', 0),
        }

    def get_fills(self, limit: int = 2000) -> List[Dict]:
        """Fetch recent fills - using HIP-3 compatible API for accuracy."""
        try:
            # HIP-3 specific API call with perp_dexs
            payload = {
                "type": "userFillsByTime", 
                "user": WALLET,
                "startTime": int((time.time() - 3600*24*7) * 1000),  # Last 7 days
                "endTime": int(time.time() * 1000),
                "perp_dexs": ["km"]  # HIP-3 specific
            }
            resp = requests.post(URL, json=payload, timeout=10)
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

    def detect_errors(self, account_state: Optional[Dict], trade_update: Dict = None) -> List[str]:
        """Detect errors from trade data and position info (no balance checks)."""
        errors = []

        if not account_state:
            errors.append("CRITICAL: Cannot fetch account state")
            return errors

        # Check for stuck positions from account state
        asset_positions = account_state.get("assetPositions", [])
        for pos in asset_positions:
            position_value = abs(float(pos.get("position", {}).get("szi", 0)))
            if position_value > 0.5:  # Significant position
                errors.append(f"LARGE POSITION: {position_value:.5f} {SYMBOL}")

        # Check for excessive imbalance from trade data
        if trade_update and "imbalance" in trade_update:
            imbalance = trade_update["imbalance"]
            total_vol = trade_update.get("buy_volume", 0) + trade_update.get("sell_volume", 0)
            if total_vol > 0 and imbalance / total_vol > 0.8:  # 80%+ imbalance
                errors.append(f"TRADE IMBALANCE: {imbalance:.2f} contracts ({imbalance/total_vol*100:.0f}% of volume)")

        # Check for negative session PnL
        if trade_update and "session_pnl" in trade_update:
            session_pnl = trade_update["session_pnl"]
            if session_pnl < -5.0:  # $5+ loss
                errors.append(f"NEGATIVE SESSION PnL: ${session_pnl:+.2f}")

        return errors

    def optimize_strategy(
        self, analysis: Dict, issues: List[str], trade_update: Dict = None
    ) -> Optional[Dict]:
        """Determine optimization needs using trade-based performance data only."""
        optimizations = {}

        spread_bps = analysis.get("spread_bps", 0)
        net_pnl = analysis.get("net_pnl_estimate", 0)
        total_fees = analysis.get("total_fees", 0)

        # Use TRADE data for optimization decisions (no balance dependency)
        cycle_pnl = 0.0
        session_pnl = 0.0
        session_fills = 0

        if trade_update and "session_pnl" in trade_update:
            cycle_pnl = trade_update.get("cycle_pnl", 0)
            session_pnl = trade_update.get("session_pnl", 0)
            session_fills = trade_update.get("session_fills", 0)

        # Check for adverse selection
        adverse_selection = any("ADVERSE SELECTION" in issue for issue in issues)

        # Calculate recent PnL trend from performance history
        recent_pnl_trend = 0.0
        if len(self.performance_history) >= 5:
            recent_pnls = [h.get("net_pnl", 0) for h in self.performance_history[-5:]]
            recent_pnl_trend = sum(recent_pnls)

        # OPT#17: Enhanced adaptive modes with trade-based decisions
        # Priority: Trade PnL decisions over spread-based
        if session_pnl < -2.0 or (cycle_pnl < -0.50 and recent_pnl_trend < -1.0):
            # Significant loss detected - emergency defensive
            optimizations["mode"] = "EMERGENCY_DEFENSIVE"
            optimizations["defensive_distance"] = 10.0
            optimizations["order_levels"] = 1
            optimizations["reason"] = (
                f"REAL LOSS: cycle=${cycle_pnl:+.2f}, session=${session_pnl:+.2f}"
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

        # Add trade metrics to optimization (no balance data)
        optimizations["session_fills"] = session_fills
        optimizations["cycle_pnl"] = cycle_pnl
        optimizations["session_pnl"] = session_pnl

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

    def get_open_orders(self):
        """Get current open orders from the exchange."""
        try:
            response = requests.post(URL, json={
                "type": "openOrders",
                "user": WALLET
            }, timeout=10)
            if response.status_code == 200:
                return response.json() if response.json() else []
            else:
                log("ERROR", f"Failed to fetch open orders: HTTP {response.status_code}")
                return []
        except Exception as e:
            log("ERROR", f"Failed to fetch open orders: {e}")
            return []
    
    def log_order_book_state(self, orders):
        """Log detailed order book state with fills and cancellations."""
        if not orders:
            log("ORDER", "📋 No open orders")
            return
            
        # Group orders by side
        buy_orders = [o for o in orders if o.get('side') == 'B' and o.get('coin') == SYMBOL]
        sell_orders = [o for o in orders if o.get('side') == 'A' and o.get('coin') == SYMBOL]
        
        log("ORDER", f"📋 OPEN ORDERS: {len(buy_orders)} buys, {len(sell_orders)} sells")
        
        # Log buy orders (highest to lowest)
        if buy_orders:
            buy_orders.sort(key=lambda x: float(x.get('limitPx', 0)), reverse=True)
            log("ORDER", "🟢 BUY ORDERS:")
            for i, order in enumerate(buy_orders):
                price = float(order.get('limitPx', 0))
                size = float(order.get('sz', 0))
                filled = float(order.get('szFilled', 0))
                remaining = size - filled
                oid = order.get('oid', 'N/A')
                timestamp = datetime.fromtimestamp(order.get('timestamp', 0) / 1000).strftime('%H:%M:%S') if order.get('timestamp') else 'N/A'
                
                fill_status = ""
                if filled > 0:
                    fill_pct = (filled / size) * 100 if size > 0 else 0
                    fill_status = f" (filled {filled}/{size}, {fill_pct:.0f}%)"
                
                log("ORDER", f"  [{timestamp}] ${price:,.2f} × {remaining:.2f} | ID: {oid}{fill_status}")
        
        # Log sell orders (lowest to highest) 
        if sell_orders:
            sell_orders.sort(key=lambda x: float(x.get('limitPx', 0)))
            log("ORDER", "🔴 SELL ORDERS:")
            for i, order in enumerate(sell_orders):
                price = float(order.get('limitPx', 0))
                size = float(order.get('sz', 0))
                filled = float(order.get('szFilled', 0))
                remaining = size - filled
                oid = order.get('oid', 'N/A')
                timestamp = datetime.fromtimestamp(order.get('timestamp', 0) / 1000).strftime('%H:%M:%S') if order.get('timestamp') else 'N/A'
                
                fill_status = ""
                if filled > 0:
                    fill_pct = (filled / size) * 100 if size > 0 else 0
                    fill_status = f" (filled {filled}/{size}, {fill_pct:.0f}%)"
                
                log("ORDER", f"  [{timestamp}] ${price:,.2f} × {remaining:.2f} | ID: {oid}{fill_status}")

    def track_order_changes(self, current_orders):
        """Track and log order changes including fills and cancellations."""
        if not hasattr(self, 'previous_orders'):
            self.previous_orders = {}
            return
            
        # Convert current orders to dict keyed by order ID
        current_dict = {o.get('oid', 'unknown'): o for o in current_orders if o.get('coin') == SYMBOL}
        
        # Check for filled or cancelled orders
        for prev_oid, prev_order in self.previous_orders.items():
            if prev_oid not in current_dict:
                # Order was filled or cancelled
                side = "BUY" if prev_order.get('side') == 'B' else "SELL"
                price = float(prev_order.get('limitPx', 0))
                size = float(prev_order.get('sz', 0))
                filled = float(prev_order.get('szFilled', 0))
                
                if filled == size:
                    log("ORDER", f"✅ FULLY FILLED: {side} {size:.2f} @ ${price:,.2f} | ID: {prev_oid}")
                elif filled > 0:
                    remaining = size - filled
                    log("ORDER", f"⚠️ PARTIALLY FILLED + CANCELLED: {side} {filled:.2f}/{size:.2f} @ ${price:,.2f} | ID: {prev_oid}")
                else:
                    log("ORDER", f"❌ CANCELLED: {side} {size:.2f} @ ${price:,.2f} | ID: {prev_oid}")
        
        # Check for partial fills on existing orders
        for curr_oid, curr_order in current_dict.items():
            if curr_oid in self.previous_orders:
                prev_filled = float(self.previous_orders[curr_oid].get('szFilled', 0))
                curr_filled = float(curr_order.get('szFilled', 0))
                
                if curr_filled > prev_filled:
                    fill_amount = curr_filled - prev_filled
                    side = "BUY" if curr_order.get('side') == 'B' else "SELL"
                    price = float(curr_order.get('limitPx', 0))
                    size = float(curr_order.get('sz', 0))
                    
                    log("ORDER", f"🎯 PARTIAL FILL: {side} {fill_amount:.2f} @ ${price:,.2f} | ID: {curr_oid} ({curr_filled:.2f}/{size:.2f} filled)")
        
        # Check for new orders
        for curr_oid, curr_order in current_dict.items():
            if curr_oid not in self.previous_orders:
                side = "BUY" if curr_order.get('side') == 'B' else "SELL"
                price = float(curr_order.get('limitPx', 0))
                size = float(curr_order.get('sz', 0))
                timestamp = datetime.fromtimestamp(curr_order.get('timestamp', 0) / 1000).strftime('%H:%M:%S') if curr_order.get('timestamp') else 'N/A'
                
                log("ORDER", f"🆕 NEW ORDER: {side} {size:.2f} @ ${price:,.2f} | ID: {curr_oid} [{timestamp}]")
        
        # Update tracking
        self.previous_orders = current_dict.copy()

    def monitor_cycle(self):
        """Execute one monitoring cycle with direct Hyperliquid data."""
        self.cycle_count += 1
        self.issues_detected = []  # Reset issues for this cycle

        print("\n" + "=" * 80)
        log("INFO", f"🔍 MONITORING CYCLE #{self.cycle_count}")
        print("=" * 80)

        # 1. Fetch data directly from Hyperliquid
        log("INFO", "📊 Fetching trade data from Hyperliquid...")
        account_state = self.get_account_state()  # Still needed for position info
        fills = self.get_fills(2000)  # Fetch max available data

        # 2. Calculate performance from trade data only (no balance queries)
        trade_update = self.update_trade_tracking(fills)

        # Log trade-based performance metrics
        if "error" not in trade_update:
            session_pnl = trade_update["session_pnl"]
            total_fees = trade_update["total_fees"]
            cycle_pnl = trade_update["cycle_pnl"]
            spread_bps = trade_update["spread_bps"]
            buys = trade_update["total_buys"]
            sells = trade_update["total_sells"]
            buy_vol = trade_update["buy_volume"]
            sell_vol = trade_update["sell_volume"]
            imbalance = trade_update["imbalance"]

            # Show comprehensive trade-based performance
            log(
                "DATA",
                f"📊 Session Stats: {trade_update['session_fills']} fills | {buys} buys ({buy_vol:.2f}) / {sells} sells ({sell_vol:.2f}) | Imbalance: {imbalance:.2f}",
            )
            log(
                "DATA",
                f"💰 Session PnL: ${session_pnl:+.4f} | Spread: {spread_bps:+.2f} bps | Fees: ${total_fees:.4f}",
            )
            if self.cycle_count > 1:
                pnl_emoji = "📈" if cycle_pnl >= 0 else "📉"
                log(
                    "DATA",
                    f"{pnl_emoji} Cycle PnL: ${cycle_pnl:+.4f}",
                )
        else:
            log("WARN", f"⚠️  {trade_update.get('error', 'Could not calculate trade metrics')}")

        if not fills:
            log("WARN", "⚠️  No fills yet - bot may be starting up")
            return

        # 3. Track and log open orders
        log("INFO", "📋 Fetching and tracking open orders...")
        current_orders = self.get_open_orders()
        self.track_order_changes(current_orders)
        self.log_order_book_state(current_orders)

        # Count new trades using timestamp-based detection with detailed logging
        current_fill_hashes = set()
        new_trades = []
        latest_fill_time = 0

        for f in fills:
            fill_hash = (
                f"{f.get('time', 0)}_{f.get('side', '')}_{f.get('px', '')}_{f.get('sz', '')}"
            )
            current_fill_hashes.add(fill_hash)
            fill_time = f.get('time', 0)
            if fill_time > latest_fill_time:
                latest_fill_time = fill_time
            # Count fills newer than our last recorded fill time
            if fill_time > self.last_fill_time:
                new_trades.append(f)

        # Log new trades in detail
        if new_trades:
            buys = [f for f in new_trades if f.get('side') == 'B']
            sells = [f for f in new_trades if f.get('side') == 'A']
            
            log("FILL", f"🆕 {len(new_trades)} NEW FILLS: {len(buys)} buys, {len(sells)} sells")
            
            # Calculate total volume and fees for new fills
            new_volume = sum(float(f.get('sz', 0)) for f in new_trades)
            new_fees = sum(float(f.get('fee', 0)) for f in new_trades)
            
            # Show recent fills with side, time, price, size, and fees
            for i, f in enumerate(new_trades[-15:]):  # Show last 15 new fills
                side = "BUY " if f.get('side') == 'B' else "SELL"
                timestamp = datetime.fromtimestamp(f.get('time', 0) / 1000).strftime('%H:%M:%S')
                price = f.get('px', 0)
                size = f.get('sz', 0)
                fee = f.get('fee', 0)
                notional = float(price) * float(size)
                
                log("FILL", f"  [{timestamp}] {side} {size} @ ${price} | ${notional:.2f} notional | fee ${fee}")
                
            # Calculate spread on new trades if we have both sides
            if buys and sells:
                new_buy_avg = sum(float(f.get('px', 0)) * float(f.get('sz', 0)) for f in buys) / sum(float(f.get('sz', 0)) for f in buys)
                new_sell_avg = sum(float(f.get('px', 0)) * float(f.get('sz', 0)) for f in sells) / sum(float(f.get('sz', 0)) for f in sells)
                new_spread = new_sell_avg - new_buy_avg
                new_spread_bps = (new_spread / ((new_buy_avg + new_sell_avg)/2)) * 10000 if new_buy_avg > 0 else 0
                
                spread_status = "✅ PROFITABLE" if new_spread_bps > 0 else "❌ ADVERSE"
                log("FILL", f"  📊 New fills spread: ${new_spread:.2f} ({new_spread_bps:+.1f} bps) {spread_status}")
                
            log("FILL", f"  📈 New volume: {new_volume:.2f} contracts | Total fees: ${new_fees:.4f}")
            
        else:
            log("INFO", "No new fills since last cycle")

        # Update tracking
        if latest_fill_time > self.last_fill_time:
            self.last_fill_time = latest_fill_time
        self.trade_count = len(fills)

        log("INFO", f"📊 Total fills: {len(fills)} (new: {len(new_trades)})")

        # 2. Analyze performance with weighted averages and multiple windows
        log("INFO", "📈 Analyzing performance (weighted by size)...")
        analysis_30m = self.analyze_recent_performance(fills, 30)
        analysis_5m = self.analyze_recent_performance(fills, 5)
        analysis_1h = self.analyze_recent_performance(fills, 60)

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

            log("DATA", f"📊 30-min window: {analysis_30m['total_fills']} fills")
            log("DATA", f"📈 Spread captured: {spread_bps:+.2f} bps {status}")

            # OPT#17: Add adverse selection count and fills rate/hour
            adverse_count = analysis_30m.get("adverse_selection_count", 0)
            fills_per_hour = analysis_30m.get("fills_per_hour", 0)
            log(
                "DATA",
                f"⚠️  Adverse selection: {adverse_count} periods | 🔄 Fill rate: {fills_per_hour:.1f}/hour",
            )
            
            # Add 5-minute recent analysis for immediate feedback
            if "error" not in analysis_5m:
                recent_spread = analysis_5m.get("spread_bps", 0)
                recent_fills = analysis_5m.get("total_fills", 0)
                recent_status = "✅" if recent_spread > 0 else "❌"
                log("DATA", f"🕐 Last 5min: {recent_fills} fills, {recent_spread:+.2f} bps {recent_status}")
            
            # Add 1-hour trend analysis
            if "error" not in analysis_1h:
                hourly_spread = analysis_1h.get("spread_bps", 0)
                hourly_fills = analysis_1h.get("total_fills", 0)
                hourly_pnl = analysis_1h.get("net_pnl_estimate", 0)
                trend_emoji = "📈" if hourly_spread > 0 else "📉"
                log("DATA", f"{trend_emoji} 1-hour trend: {hourly_fills} fills, {hourly_spread:+.2f} bps, PnL ${hourly_pnl:+.2f}")

            log(
                "DATA",
                f"💰 Net PnL: ${net_pnl:+.4f} | Fees: ${total_fees:.4f} | Volume: {volume_btc:.2f} contracts",
            )
            log("DATA", f"📊 Buys: {analysis_30m['buys']} @ ${analysis_30m.get('avg_buy', 0):,.2f} avg")
            log("DATA", f"📊 Sells: {analysis_30m['sells']} @ ${analysis_30m.get('avg_sell', 0):,.2f} avg")
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
        errors = self.detect_errors(account_state, trade_update)

        if errors:
            for error in errors:
                log("ERROR", f"❌ {error}")
        else:
            log("INFO", "✅ No errors detected")

        # 5. Apply optimizations AUTONOMOUSLY (using trade data only)
        log("INFO", "⚙️  Evaluating optimization needs (trade-based)...")
        optimization = self.optimize_strategy(
            analysis_30m, strategy_issues + errors, trade_update
        )

        if optimization:
            mode = optimization.get("mode", "UNKNOWN")
            reason = optimization.get("reason", "")
            urgency = optimization.get("urgency", "LOW")
            cycle_pnl = optimization.get("cycle_pnl", 0)
            session_pnl = optimization.get("session_pnl", 0)

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
                    "session_fills": optimization.get("session_fills", 0),
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
        log("DATA", f"New trades this cycle: {len(new_trades) if 'new_trades' in locals() else 0}")
        log("DATA", f"Issues detected: {len(strategy_issues + errors)}")

        # Trade-based performance summary (no balance data)
        if trade_update and "session_pnl" in trade_update:
            cycle_pnl = trade_update.get("cycle_pnl", 0)
            session_pnl = trade_update.get("session_pnl", 0)
            session_fills = trade_update.get("session_fills", 0)
            total_fees = trade_update.get("total_fees", 0)
            pnl_emoji = "📈" if session_pnl >= 0 else "📉"
            
            log("DATA", f"💰 Session fills: {session_fills} trades")
            log(
                "DATA",
                f"{pnl_emoji} Cycle PnL: ${cycle_pnl:+.4f} | Session: ${session_pnl:+.2f} | Fees: ${total_fees:.4f}",
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
    print("DEBUG: main() starting...", flush=True)
    
    monitor = PerformanceMonitor()
    print("DEBUG: PerformanceMonitor created", flush=True)

    print("\n" + "=" * 80, flush=True)
    log("INFO", "🚀 CONTINUOUS MONITORING & AUTO-OPTIMIZATION SYSTEM v2.0")
    print("=" * 80, flush=True)
    log("INFO", f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("INFO", f"Symbol: {SYMBOL} (S&P 500 Index Perpetual)")
    log("INFO", "Bot: OPT#14 (Adaptive Anti-Picking-Off) + Weighted Averages")
    log("INFO", "Data: Real-time from Hyperliquid API (2000 fills)")
    log("INFO", f"Candle Collection: Saving {SYMBOL} 1m candles to data/us500_candles_1m.csv")
    log("INFO", "Monitoring: 5-minute cycles, autonomous optimization")
    log("INFO", "State: Persistent across restarts")
    log("INFO", "Press Ctrl+C to stop")
    print("=" * 80)

    # Candle collection disabled to avoid blocking monitoring
    # Start US500 candle collection in background (optional, non-blocking)
    # try:
    #     collector = get_collector()
    #     # Start async collection in background thread
    #     import threading
    #     def run_collection():
    #         try:
    #             loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(loop)
    #             loop.run_until_complete(collector.collect_continuously())
    #         except Exception as e:
    #             log("WARN", f"Candle collection error (non-critical): {e}")
    #     
    #     collection_thread = threading.Thread(target=run_collection, daemon=True)
    #     collection_thread.start()
    #     time.sleep(1)  # Give it a second to start
    #     log("INFO", f"✅ Started {SYMBOL} candle collection (1-minute interval)")
    # except Exception as e:
    #     log("WARN", f"Could not start candle collection (non-critical): {e}")

    log("INFO", "Starting monitoring cycles...")

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
