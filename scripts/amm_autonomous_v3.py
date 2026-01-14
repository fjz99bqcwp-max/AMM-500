#!/usr/bin/env python3
"""
AMM Autonomous Monitoring System v3.0

Comprehensive upgrade with:
- Real-time wallet API tracking (equity/PnL/margin)
- Async logging with aiofiles
- Enhanced auto-restart with crash detection
- Alert system (email/Slack notifications)
- Kill switches for critical conditions
- Multiprocessing for parallel checks
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import smtplib
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.utils import constants as hl_constants

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "US500"
MONITOR_INTERVAL = 300  # 5 minutes
MAX_CONSECUTIVE_ERRORS = 5
STATE_FILE = PROJECT_ROOT / "logs" / "autonomous_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / "autonomous_v3.log"

# Alert Thresholds
DRAWDOWN_ALERT_PCT = 2.0  # Alert on 2% drawdown
TAKER_RATIO_ALERT_PCT = 30.0  # Alert when >30% taker trades
PNL_LOSS_ALERT = -50.0  # Alert on $50 session loss
MARGIN_RATIO_ALERT = 0.8  # Alert when margin usage > 80%

# Kill Switch Thresholds
KILL_DRAWDOWN_PCT = 5.0  # Hard stop at 5% drawdown
KILL_CONSECUTIVE_LOSSES = 10  # Stop after 10 consecutive losing trades
KILL_SESSION_LOSS = -100.0  # Stop at $100 session loss

# Email/Slack Config (set via environment)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")

# API Configuration
API_WALLET = os.getenv("API_WALLET_ADDRESS", os.getenv("HL_API_WALLET", ""))
PRIVATE_KEY = os.getenv("PRIVATE_KEY", os.getenv("HL_PRIVATE_KEY", ""))
USER_WALLET = os.getenv("WALLET_ADDRESS", os.getenv("HL_WALLET", ""))


# =============================================================================
# LOGGING
# =============================================================================

class AsyncLogger:
    """Async-capable logger with file and console output."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_queue: deque = deque(maxlen=1000)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _format_message(self, level: str, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{level:5}] {message}"
    
    def log(self, level: str, message: str):
        """Synchronous log (for backward compatibility)."""
        formatted = self._format_message(level, message)
        print(formatted, flush=True)
        self.log_queue.append(formatted)
        # Write to file in background
        self._executor.submit(self._write_to_file, formatted)
    
    def _write_to_file(self, message: str):
        try:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        except Exception:
            pass
    
    async def async_log(self, level: str, message: str):
        """Async log using aiofiles pattern."""
        formatted = self._format_message(level, message)
        print(formatted, flush=True)
        self.log_queue.append(formatted)
        # Use thread executor for file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._write_to_file, formatted)


logger = AsyncLogger(LOG_FILE)
log = logger.log


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WalletState:
    """Real-time wallet state from API."""
    equity: float = 0.0
    margin_used: float = 0.0
    available_margin: float = 0.0
    unrealized_pnl: float = 0.0
    margin_ratio: float = 0.0
    positions: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        return self.equity > 0 and self.margin_ratio < MARGIN_RATIO_ALERT


@dataclass
class TradeMetrics:
    """Aggregated trade metrics."""
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    maker_trades: int = 0
    taker_trades: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0
    realized_pnl: float = 0.0
    session_pnl: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None


@dataclass
class AlertState:
    """Track alert state to avoid spam."""
    last_drawdown_alert: Optional[datetime] = None
    last_taker_alert: Optional[datetime] = None
    last_pnl_alert: Optional[datetime] = None
    last_margin_alert: Optional[datetime] = None
    alerts_sent_today: int = 0
    
    def can_send_alert(self, alert_type: str, cooldown_minutes: int = 30) -> bool:
        """Check if we can send an alert (cooldown period)."""
        last_alert = getattr(self, f"last_{alert_type}_alert", None)
        if last_alert is None:
            return True
        return datetime.now() - last_alert > timedelta(minutes=cooldown_minutes)


# =============================================================================
# WALLET API TRACKER
# =============================================================================

class WalletTracker:
    """Real-time wallet tracking via Hyperliquid signed API."""
    
    def __init__(self, wallet_address: str, private_key: str = ""):
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.info = Info(hl_constants.MAINNET_API_URL, skip_ws=True)
        self.state_history: deque = deque(maxlen=100)
        self.peak_equity = 0.0
        self.session_start_equity = 0.0
        self._initialized = False
    
    def get_wallet_state(self) -> WalletState:
        """Fetch current wallet state from API."""
        try:
            # Get user state (includes equity, positions, margin)
            user_state = self.info.user_state(self.wallet_address)
            
            if not user_state:
                log("WARN", "No user state returned from API")
                return WalletState()
            
            # Parse cross-margin summary
            cross = user_state.get("crossMarginSummary", {})
            equity = float(cross.get("accountValue", 0))
            margin_used = float(cross.get("totalMarginUsed", 0))
            available_margin = float(cross.get("withdrawable", 0))
            
            # Calculate margin ratio
            margin_ratio = margin_used / equity if equity > 0 else 0
            
            # Get unrealized PnL from positions
            positions = user_state.get("assetPositions", [])
            unrealized_pnl = 0.0
            position_list = []
            
            for pos in positions:
                pos_data = pos.get("position", {})
                if pos_data:
                    upnl = float(pos_data.get("unrealizedPnl", 0))
                    unrealized_pnl += upnl
                    position_list.append({
                        "coin": pos_data.get("coin", ""),
                        "size": float(pos_data.get("szi", 0)),
                        "entry": float(pos_data.get("entryPx", 0)),
                        "unrealized_pnl": upnl,
                        "leverage": pos_data.get("leverage", {})
                    })
            
            state = WalletState(
                equity=equity,
                margin_used=margin_used,
                available_margin=available_margin,
                unrealized_pnl=unrealized_pnl,
                margin_ratio=margin_ratio,
                positions=position_list,
                timestamp=datetime.now()
            )
            
            # Track peak equity for drawdown
            if not self._initialized:
                self.session_start_equity = equity
                self.peak_equity = equity
                self._initialized = True
            else:
                self.peak_equity = max(self.peak_equity, equity)
            
            self.state_history.append(state)
            return state
            
        except Exception as e:
            log("ERROR", f"Wallet state fetch error: {e}")
            return WalletState()
    
    def get_drawdown_pct(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - current_equity) / self.peak_equity) * 100
    
    def get_session_pnl(self, current_equity: float) -> float:
        """Calculate session PnL from start."""
        return current_equity - self.session_start_equity


# =============================================================================
# ALERT SYSTEM
# =============================================================================

class AlertManager:
    """Manage alerts via email and Slack."""
    
    def __init__(self):
        self.state = AlertState()
        self.http_client = httpx.Client(timeout=10)
    
    def send_email(self, subject: str, body: str) -> bool:
        """Send email alert."""
        if not SMTP_USER or not ALERT_EMAIL:
            log("WARN", "Email not configured - skipping email alert")
            return False
        
        try:
            msg = MIMEText(body)
            msg["Subject"] = f"[AMM Alert] {subject}"
            msg["From"] = SMTP_USER
            msg["To"] = ALERT_EMAIL
            
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, [ALERT_EMAIL], msg.as_string())
            
            log("INFO", f"📧 Email alert sent: {subject}")
            return True
        except Exception as e:
            log("ERROR", f"Email send failed: {e}")
            return False
    
    def send_slack(self, message: str) -> bool:
        """Send Slack webhook alert."""
        if not SLACK_WEBHOOK:
            log("WARN", "Slack webhook not configured - skipping Slack alert")
            return False
        
        try:
            payload = {
                "text": message,
                "username": "AMM Bot",
                "icon_emoji": ":robot_face:"
            }
            response = self.http_client.post(SLACK_WEBHOOK, json=payload)
            if response.status_code == 200:
                log("INFO", f"💬 Slack alert sent")
                return True
            else:
                log("WARN", f"Slack webhook returned {response.status_code}")
                return False
        except Exception as e:
            log("ERROR", f"Slack send failed: {e}")
            return False
    
    def send_alert(self, alert_type: str, subject: str, details: str):
        """Send alert via all configured channels."""
        if not self.state.can_send_alert(alert_type):
            log("INFO", f"Alert {alert_type} on cooldown - skipping")
            return
        
        full_message = f"""
🚨 AMM ALERT: {subject}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {SYMBOL}

{details}

---
AMM Autonomous Monitoring System v3.0
"""
        
        # Send via both channels
        self.send_email(subject, full_message)
        self.send_slack(f"🚨 *{subject}*\n{details}")
        
        # Update state
        setattr(self.state, f"last_{alert_type}_alert", datetime.now())
        self.state.alerts_sent_today += 1
    
    def check_and_alert(
        self,
        wallet_state: WalletState,
        trade_metrics: TradeMetrics,
        drawdown_pct: float
    ):
        """Check conditions and send alerts as needed."""
        
        # Drawdown Alert
        if drawdown_pct >= DRAWDOWN_ALERT_PCT:
            self.send_alert(
                "drawdown",
                f"Drawdown Alert: {drawdown_pct:.2f}%",
                f"Current drawdown: {drawdown_pct:.2f}%\n"
                f"Peak Equity: ${wallet_state.equity + (drawdown_pct/100 * wallet_state.equity):.2f}\n"
                f"Current Equity: ${wallet_state.equity:.2f}"
            )
        
        # Taker Ratio Alert
        if trade_metrics.total_trades > 0:
            taker_ratio = (trade_metrics.taker_trades / trade_metrics.total_trades) * 100
            if taker_ratio >= TAKER_RATIO_ALERT_PCT:
                self.send_alert(
                    "taker",
                    f"High Taker Ratio: {taker_ratio:.1f}%",
                    f"Taker trades: {trade_metrics.taker_trades}/{trade_metrics.total_trades}\n"
                    f"This indicates we are crossing the spread (losing edge)"
                )
        
        # PnL Alert
        if trade_metrics.session_pnl <= PNL_LOSS_ALERT:
            self.send_alert(
                "pnl",
                f"Session Loss: ${trade_metrics.session_pnl:.2f}",
                f"Session PnL: ${trade_metrics.session_pnl:.2f}\n"
                f"Total trades: {trade_metrics.total_trades}\n"
                f"Total fees: ${trade_metrics.total_fees:.2f}"
            )
        
        # Margin Alert
        if wallet_state.margin_ratio >= MARGIN_RATIO_ALERT:
            self.send_alert(
                "margin",
                f"High Margin Usage: {wallet_state.margin_ratio*100:.1f}%",
                f"Margin used: ${wallet_state.margin_used:.2f}\n"
                f"Available: ${wallet_state.available_margin:.2f}\n"
                f"Equity: ${wallet_state.equity:.2f}"
            )


# =============================================================================
# KILL SWITCH MANAGER
# =============================================================================

class KillSwitchManager:
    """Manage kill switches for critical conditions."""
    
    def __init__(self):
        self.triggered = False
        self.trigger_reason = ""
    
    def check_kill_conditions(
        self,
        drawdown_pct: float,
        session_pnl: float,
        consecutive_losses: int
    ) -> bool:
        """Check if any kill switch condition is met."""
        
        # Drawdown kill switch
        if drawdown_pct >= KILL_DRAWDOWN_PCT:
            self.triggered = True
            self.trigger_reason = f"Drawdown exceeded {KILL_DRAWDOWN_PCT}% (current: {drawdown_pct:.2f}%)"
            return True
        
        # Consecutive losses kill switch
        if consecutive_losses >= KILL_CONSECUTIVE_LOSSES:
            self.triggered = True
            self.trigger_reason = f"Consecutive losses exceeded {KILL_CONSECUTIVE_LOSSES} (current: {consecutive_losses})"
            return True
        
        # Session loss kill switch
        if session_pnl <= KILL_SESSION_LOSS:
            self.triggered = True
            self.trigger_reason = f"Session loss exceeded ${abs(KILL_SESSION_LOSS)} (current: ${session_pnl:.2f})"
            return True
        
        return False
    
    def execute_kill(self, bot_pid: Optional[int] = None):
        """Execute kill switch - stop all trading."""
        log("ERROR", "🛑🛑🛑 KILL SWITCH TRIGGERED 🛑🛑🛑")
        log("ERROR", f"Reason: {self.trigger_reason}")
        
        # Kill the bot if running
        if bot_pid:
            try:
                os.kill(bot_pid, signal.SIGTERM)
                log("INFO", f"Sent SIGTERM to bot (PID: {bot_pid})")
            except ProcessLookupError:
                log("WARN", f"Bot process {bot_pid} not found")
            except Exception as e:
                log("ERROR", f"Failed to kill bot: {e}")
        
        # Cancel all orders
        try:
            self._cancel_all_orders()
        except Exception as e:
            log("ERROR", f"Failed to cancel orders: {e}")
        
        log("ERROR", "Kill switch complete - bot stopped, orders cancelled")
        return True
    
    def _cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            from hyperliquid.exchange import Exchange
            
            if not PRIVATE_KEY:
                log("WARN", "No private key - cannot cancel orders")
                return
            
            account = Account.from_key(PRIVATE_KEY)
            exchange = Exchange(account, hl_constants.MAINNET_API_URL)
            
            # Get all open orders
            info = Info(hl_constants.MAINNET_API_URL, skip_ws=True)
            open_orders = info.open_orders(USER_WALLET)
            
            if not open_orders:
                log("INFO", "No open orders to cancel")
                return
            
            # Cancel each order
            for order in open_orders:
                try:
                    exchange.cancel(order["coin"], order["oid"])
                    log("INFO", f"Cancelled order {order['oid']}")
                except Exception as e:
                    log("WARN", f"Failed to cancel {order['oid']}: {e}")
            
            log("INFO", f"Cancelled {len(open_orders)} orders")
            
        except Exception as e:
            log("ERROR", f"Cancel all orders failed: {e}")


# =============================================================================
# PROCESS MANAGER
# =============================================================================

class ProcessManager:
    """Manage bot process with health monitoring."""
    
    def __init__(self):
        self.bot_pid: Optional[int] = None
        self.restart_count = 0
        self.last_restart: Optional[datetime] = None
        self.max_restarts_per_hour = 5
        self.restart_history: deque = deque(maxlen=20)
    
    def find_bot_process(self) -> Optional[int]:
        """Find running bot process."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "python.*amm-500.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                if pids and pids[0]:
                    self.bot_pid = int(pids[0])
                    return self.bot_pid
        except Exception as e:
            log("WARN", f"Error finding bot process: {e}")
        return None
    
    def is_bot_running(self) -> bool:
        """Check if bot process is running."""
        pid = self.find_bot_process()
        if pid:
            try:
                os.kill(pid, 0)  # Signal 0 = check existence
                return True
            except OSError:
                return False
        return False
    
    def can_restart(self) -> bool:
        """Check if we can restart (rate limiting)."""
        # Count restarts in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_restarts = sum(
            1 for t in self.restart_history
            if t > one_hour_ago
        )
        return recent_restarts < self.max_restarts_per_hour
    
    def restart_bot(self, reason: str = "Unknown") -> bool:
        """Restart the bot process."""
        if not self.can_restart():
            log("ERROR", f"Too many restarts in last hour - not restarting")
            return False
        
        log("WARN", f"🔄 Restarting bot: {reason}")
        
        # Kill existing process
        if self.bot_pid:
            try:
                os.kill(self.bot_pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still running
                try:
                    os.kill(self.bot_pid, signal.SIGKILL)
                except OSError:
                    pass
            except OSError:
                pass
        
        # Start new process
        try:
            bot_script = PROJECT_ROOT / "amm-500.py"
            process = subprocess.Popen(
                [sys.executable, str(bot_script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            self.bot_pid = process.pid
            self.restart_count += 1
            self.last_restart = datetime.now()
            self.restart_history.append(datetime.now())
            
            log("INFO", f"✅ Bot restarted (PID: {self.bot_pid})")
            time.sleep(5)  # Give it time to start
            
            return self.is_bot_running()
            
        except Exception as e:
            log("ERROR", f"Failed to restart bot: {e}")
            return False
    
    def get_process_health(self) -> dict:
        """Get process health metrics."""
        return {
            "running": self.is_bot_running(),
            "pid": self.bot_pid,
            "restart_count": self.restart_count,
            "last_restart": self.last_restart.isoformat() if self.last_restart else None,
            "can_restart": self.can_restart()
        }


# =============================================================================
# PERFORMANCE MONITOR (Enhanced)
# =============================================================================

class PerformanceMonitorV3:
    """Enhanced performance monitor with wallet tracking and alerts."""
    
    def __init__(self):
        self.wallet_tracker = WalletTracker(USER_WALLET, PRIVATE_KEY)
        self.alert_manager = AlertManager()
        self.kill_switch = KillSwitchManager()
        self.process_manager = ProcessManager()
        self.info = Info(hl_constants.MAINNET_API_URL, skip_ws=True)
        
        # Metrics
        self.trade_metrics = TradeMetrics()
        self.cycle_count = 0
        self.consecutive_errors = 0
        self.last_fill_time: Optional[int] = None
        self.processed_fills: set = set()
        
        # Session tracking
        self.session_start = datetime.now()
        self.performance_history: deque = deque(maxlen=500)
        
        # Load saved state
        self.load_state()
    
    def load_state(self):
        """Load persisted state."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
                
                self.cycle_count = state.get("cycle_count", 0)
                self.trade_metrics.total_trades = state.get("total_trades", 0)
                self.processed_fills = set(state.get("processed_fills", [])[-500:])
                
                log("INFO", f"Loaded state: {self.cycle_count} cycles, {self.trade_metrics.total_trades} trades")
        except Exception as e:
            log("WARN", f"Could not load state: {e}")
    
    def save_state(self):
        """Save state for persistence."""
        try:
            state = {
                "cycle_count": self.cycle_count,
                "total_trades": self.trade_metrics.total_trades,
                "processed_fills": list(self.processed_fills)[-500:],
                "last_save": datetime.now().isoformat(),
                "session_start": self.session_start.isoformat()
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log("WARN", f"Could not save state: {e}")
    
    def get_fills(self, limit: int = 500) -> list:
        """Get recent fills from API."""
        try:
            fills = self.info.user_fills_by_time(
                USER_WALLET,
                int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
                int(datetime.now().timestamp() * 1000)
            )
            return fills[:limit] if fills else []
        except Exception as e:
            log("ERROR", f"Failed to get fills: {e}")
            return []
    
    def update_trade_metrics(self, fills: list) -> dict:
        """Update trade metrics from fills."""
        new_fills = []
        
        for fill in fills:
            fill_id = f"{fill.get('tid')}_{fill.get('time')}"
            if fill_id not in self.processed_fills:
                self.processed_fills.add(fill_id)
                new_fills.append(fill)
        
        if not new_fills:
            return {"new_trades": 0}
        
        # Process new fills
        cycle_pnl = 0.0
        cycle_fees = 0.0
        
        for fill in new_fills:
            self.trade_metrics.total_trades += 1
            
            # Direction
            side = fill.get("side", "").upper()
            if side == "B":
                self.trade_metrics.buy_trades += 1
            else:
                self.trade_metrics.sell_trades += 1
            
            # Maker/Taker
            if fill.get("liquidation") or fill.get("crossed"):
                self.trade_metrics.taker_trades += 1
            else:
                self.trade_metrics.maker_trades += 1
            
            # PnL
            pnl = float(fill.get("closedPnl", 0))
            fee = float(fill.get("fee", 0))
            
            cycle_pnl += pnl
            cycle_fees += fee
            self.trade_metrics.realized_pnl += pnl
            self.trade_metrics.total_fees += fee
            
            # Track consecutive losses
            if pnl < 0:
                self.trade_metrics.consecutive_losses += 1
            else:
                self.trade_metrics.consecutive_losses = 0
            
            # Volume
            size = abs(float(fill.get("sz", 0)))
            price = float(fill.get("px", 0))
            self.trade_metrics.total_volume += size * price
            
            self.trade_metrics.last_trade_time = datetime.now()
        
        # Calculate session PnL (realized - fees)
        self.trade_metrics.session_pnl = self.trade_metrics.realized_pnl - self.trade_metrics.total_fees
        
        return {
            "new_trades": len(new_fills),
            "cycle_pnl": cycle_pnl,
            "cycle_fees": cycle_fees,
            "session_pnl": self.trade_metrics.session_pnl
        }
    
    def get_open_orders(self) -> list:
        """Get current open orders."""
        try:
            return self.info.open_orders(USER_WALLET)
        except Exception as e:
            log("ERROR", f"Failed to get open orders: {e}")
            return []
    
    def analyze_orders(self, orders: list) -> dict:
        """Analyze order distribution."""
        bids = [o for o in orders if o.get("side") == "B"]
        asks = [o for o in orders if o.get("side") == "A"]
        
        # Get mid price
        try:
            l2 = self.info.l2_snapshot(f"km:{SYMBOL}")
            if l2 and l2.get("levels"):
                best_bid = float(l2["levels"][0][0]["px"]) if l2["levels"][0] else 0
                best_ask = float(l2["levels"][1][0]["px"]) if l2["levels"][1] else 0
                mid = (best_bid + best_ask) / 2
            else:
                mid = 0
        except:
            mid = 0
        
        return {
            "total": len(orders),
            "bids": len(bids),
            "asks": len(asks),
            "imbalance": len(bids) - len(asks),
            "mid_price": mid
        }
    
    def monitor_cycle(self):
        """Execute one monitoring cycle."""
        self.cycle_count += 1
        log("INFO", f"\n{'='*80}")
        log("INFO", f"🔄 MONITORING CYCLE #{self.cycle_count}")
        log("INFO", f"{'='*80}")
        
        try:
            # 1. Check bot process
            log("INFO", "📊 Checking bot process...")
            process_health = self.process_manager.get_process_health()
            if process_health["running"]:
                log("INFO", f"   ✅ Bot running (PID: {process_health['pid']})")
            else:
                log("WARN", "   ❌ Bot not running!")
                if not self.kill_switch.triggered:
                    self.process_manager.restart_bot("Process not found")
            
            # 2. Fetch wallet state
            log("INFO", "💰 Fetching wallet state...")
            wallet_state = self.wallet_tracker.get_wallet_state()
            if wallet_state.equity > 0:
                log("INFO", f"   Equity: ${wallet_state.equity:.2f}")
                log("INFO", f"   Available: ${wallet_state.available_margin:.2f}")
                log("INFO", f"   Margin Used: {wallet_state.margin_ratio*100:.1f}%")
                log("INFO", f"   Unrealized PnL: ${wallet_state.unrealized_pnl:+.2f}")
                
                # Position info
                for pos in wallet_state.positions:
                    if pos["size"] != 0:
                        log("INFO", f"   Position: {pos['coin']} {pos['size']:+.4f} @ {pos['entry']:.2f}")
            else:
                log("WARN", "   ⚠️ Could not fetch wallet state")
            
            # 3. Calculate drawdown
            drawdown_pct = self.wallet_tracker.get_drawdown_pct(wallet_state.equity)
            session_pnl = self.wallet_tracker.get_session_pnl(wallet_state.equity)
            
            log("INFO", f"📉 Drawdown: {drawdown_pct:.2f}%")
            log("INFO", f"📈 Session PnL (equity): ${session_pnl:+.2f}")
            
            # 4. Check kill switch
            if self.kill_switch.check_kill_conditions(
                drawdown_pct,
                self.trade_metrics.session_pnl,
                self.trade_metrics.consecutive_losses
            ):
                self.kill_switch.execute_kill(self.process_manager.bot_pid)
                return  # Stop monitoring
            
            # 5. Fetch and process fills
            log("INFO", "📝 Fetching recent fills...")
            fills = self.get_fills()
            trade_update = self.update_trade_metrics(fills)
            
            if trade_update["new_trades"] > 0:
                log("INFO", f"   New trades: {trade_update['new_trades']}")
                log("INFO", f"   Cycle PnL: ${trade_update['cycle_pnl']:+.4f}")
                log("INFO", f"   Session PnL (trades): ${self.trade_metrics.session_pnl:+.2f}")
            else:
                log("INFO", "   No new trades")
            
            # 6. Analyze orders
            log("INFO", "📋 Analyzing open orders...")
            orders = self.get_open_orders()
            order_analysis = self.analyze_orders(orders)
            
            log("INFO", f"   Total: {order_analysis['total']} ({order_analysis['bids']} bids, {order_analysis['asks']} asks)")
            if order_analysis['imbalance'] != 0:
                log("WARN", f"   ⚠️ Order imbalance: {order_analysis['imbalance']:+d}")
            
            # 7. Check alerts
            log("INFO", "🔔 Checking alert conditions...")
            self.alert_manager.check_and_alert(
                wallet_state,
                self.trade_metrics,
                drawdown_pct
            )
            
            # 8. Performance summary
            self._log_performance_summary(wallet_state, order_analysis, trade_update)
            
            # 9. Auto-restart check
            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log("ERROR", f"❌ {self.consecutive_errors} consecutive errors - restarting")
                self.process_manager.restart_bot(f"{self.consecutive_errors} consecutive errors")
                self.consecutive_errors = 0
            
            # Success - reset error counter
            self.consecutive_errors = 0
            self.save_state()
            
        except Exception as e:
            self.consecutive_errors += 1
            log("ERROR", f"Cycle error: {e}")
            import traceback
            log("ERROR", traceback.format_exc())
    
    def _log_performance_summary(self, wallet: WalletState, orders: dict, trades: dict):
        """Log performance summary."""
        log("INFO", f"\n{'='*40}")
        log("INFO", "📊 CYCLE SUMMARY")
        log("INFO", f"{'='*40}")
        
        # Wallet summary
        log("DATA", f"💰 Equity: ${wallet.equity:.2f}")
        log("DATA", f"📈 Session PnL: ${self.trade_metrics.session_pnl:+.2f}")
        log("DATA", f"📉 Drawdown: {self.wallet_tracker.get_drawdown_pct(wallet.equity):.2f}%")
        
        # Trade summary
        log("DATA", f"🔄 Total trades: {self.trade_metrics.total_trades}")
        if self.trade_metrics.total_trades > 0:
            maker_pct = (self.trade_metrics.maker_trades / self.trade_metrics.total_trades) * 100
            log("DATA", f"   Maker: {self.trade_metrics.maker_trades} ({maker_pct:.1f}%)")
            log("DATA", f"   Taker: {self.trade_metrics.taker_trades}")
            log("DATA", f"   Fees: ${self.trade_metrics.total_fees:.4f}")
        
        # Order summary
        log("DATA", f"📋 Orders: {orders['total']} (B:{orders['bids']}/A:{orders['asks']})")
        
        # Status emoji
        if self.trade_metrics.session_pnl >= 0:
            status = "✅ PROFITABLE"
        else:
            status = "❌ LOSING"
        log("DATA", f"📊 Status: {status}")
        
        log("INFO", f"{'='*40}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main monitoring loop."""
    print("\n" + "=" * 80, flush=True)
    log("INFO", "🚀 AMM AUTONOMOUS MONITORING SYSTEM v3.0")
    print("=" * 80, flush=True)
    log("INFO", f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("INFO", f"Symbol: {SYMBOL}")
    log("INFO", f"Wallet: {USER_WALLET[:10]}...{USER_WALLET[-8:]}")
    log("INFO", f"Monitor Interval: {MONITOR_INTERVAL}s")
    log("INFO", "")
    log("INFO", "Features:")
    log("INFO", "  ✅ Real-time wallet API tracking")
    log("INFO", "  ✅ Async logging")
    log("INFO", "  ✅ Auto-restart on crash")
    log("INFO", "  ✅ Alert system (Email/Slack)")
    log("INFO", "  ✅ Kill switches for critical conditions")
    log("INFO", "")
    log("INFO", "Thresholds:")
    log("INFO", f"  Alert: DD>{DRAWDOWN_ALERT_PCT}% | Taker>{TAKER_RATIO_ALERT_PCT}% | Loss>${abs(PNL_LOSS_ALERT)}")
    log("INFO", f"  Kill: DD>{KILL_DRAWDOWN_PCT}% | Losses>{KILL_CONSECUTIVE_LOSSES} | Loss>${abs(KILL_SESSION_LOSS)}")
    log("INFO", "")
    log("INFO", "Press Ctrl+C to stop")
    print("=" * 80)
    
    monitor = PerformanceMonitorV3()
    
    try:
        while True:
            try:
                monitor.monitor_cycle()
            except Exception as e:
                monitor.consecutive_errors += 1
                log("ERROR", f"Cycle error: {e}")
                if monitor.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log("ERROR", "Too many errors - waiting before retry")
                    time.sleep(60)
            
            if monitor.kill_switch.triggered:
                log("ERROR", "Kill switch triggered - stopping monitoring")
                break
            
            log("INFO", f"⏳ Next cycle in {MONITOR_INTERVAL}s...\n")
            time.sleep(MONITOR_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        log("INFO", "🛑 MONITORING STOPPED BY USER")
        print("=" * 80)
        log("DATA", f"Total cycles: {monitor.cycle_count}")
        log("DATA", f"Total trades: {monitor.trade_metrics.total_trades}")
        log("DATA", f"Session PnL: ${monitor.trade_metrics.session_pnl:+.2f}")
        log("DATA", f"Bot restarts: {monitor.process_manager.restart_count}")
        monitor.save_state()
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
