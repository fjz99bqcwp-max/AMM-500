"""
Dark-theme optimized logging configuration for AMM-500 bot.

This module provides custom formatters and utilities for clean, readable logs
optimized for dark terminal themes with proper CET timezone support.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


# =============================================================================
# Timezone Configuration
# =============================================================================

CET = timezone(timedelta(hours=1))


def get_cet_timestamp() -> str:
    """Get current timestamp in CET timezone (time-only for cleaner logs)"""
    return datetime.now(CET).strftime("%H:%M:%S")


# =============================================================================
# ANSI Colors - Optimized for Dark Terminal Themes
# =============================================================================

class Colors:
    """ANSI color codes optimized for dark terminal backgrounds"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Dark-theme optimized - brighter, more visible colors
    RED = "\033[38;5;203m"        # Soft red (not too harsh)
    GREEN = "\033[38;5;114m"      # Soft green
    YELLOW = "\033[38;5;221m"     # Warm yellow
    BLUE = "\033[38;5;111m"       # Bright blue
    MAGENTA = "\033[38;5;176m"    # Soft magenta
    CYAN = "\033[38;5;80m"        # Teal cyan
    ORANGE = "\033[38;5;215m"     # Orange for warnings
    GRAY = "\033[38;5;245m"       # Dim gray for timestamps
    WHITE = "\033[38;5;255m"      # Bright white for important


# =============================================================================
# Console Formatter with Dark-Theme Colors
# =============================================================================

def format_console(record: dict) -> str:
    """
    Custom formatter for console output with dark-theme optimized colors.
    
    Format: [HH:MM:SS] [LEVEL] module  message
    """
    # Get CET timestamp
    timestamp = datetime.fromtimestamp(record["time"].timestamp(), tz=CET).strftime("%H:%M:%S")
    
    # Level coloring
    level_colors = {
        'TRACE': Colors.GRAY,
        'DEBUG': Colors.GRAY,
        'INFO': Colors.CYAN,
        'SUCCESS': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }
    
    level_tags = {
        'TRACE': 'TRCE',
        'DEBUG': 'DBUG',
        'INFO': 'INFO',
        'SUCCESS': 'SUCC',
        'WARNING': 'WARN',
        'ERROR': 'ERR!',
        'CRITICAL': 'CRIT',
    }
    
    level = record["level"].name
    level_color = level_colors.get(level, Colors.RESET)
    level_tag = level_tags.get(level, 'INFO')
    
    # Shorten module name for cleaner output
    module = record["name"]
    if module.startswith('src.'):
        module = module[4:]  # Remove 'src.' prefix
    if module.startswith('core.'):
        module = module[5:]  # Remove 'core.' prefix
    if module.startswith('utils.'):
        module = module[6:]  # Remove 'utils.' prefix
    module = module[:12].ljust(12)  # Fixed width (12 chars)
    
    # Get message
    msg = record["message"]
    
    # Context-aware message coloring for dark theme
    msg_color = Colors.RESET
    if any(word in msg.lower() for word in ['success', 'executed', 'completed', 'connected', 'started', '✅']):
        msg_color = Colors.GREEN
    elif 'force' in msg.lower() or 'emergency' in msg.lower():
        msg_color = Colors.ORANGE
    elif any(word in msg.lower() for word in ['cycle', 'iteration', 'refresh']):
        msg_color = Colors.CYAN
    elif 'buy' in msg.lower() or 'sell' in msg.lower() or 'bid' in msg.lower() or 'ask' in msg.lower():
        msg_color = Colors.BLUE
    elif any(word in msg.lower() for word in ['skip', 'wait', 'pending']):
        msg_color = Colors.YELLOW
    elif 'hold' in msg.lower() or 'flat' in msg.lower():
        msg_color = Colors.GRAY
    elif any(word in msg.lower() for word in ['error', 'failed', 'halted', 'stop', 'critical', '❌']):
        msg_color = Colors.RED
    elif 'pnl' in msg.lower() or 'profit' in msg.lower() or '$' in msg:
        # Color based on profit/loss indicators
        if any(indicator in msg for indicator in ['+', '📈', 'gain']):
            msg_color = Colors.GREEN
        elif any(indicator in msg for indicator in ['-', '📉', 'loss']):
            msg_color = Colors.RED
    elif '⚠️' in msg or 'warning' in msg.lower():
        msg_color = Colors.YELLOW
    
    # Format: [time] [LEVEL] module  message
    formatted = (
        f"{Colors.GRAY}[{timestamp}]{Colors.RESET} "
        f"{level_color}[{level_tag}]{Colors.RESET} "
        f"{Colors.DIM}{module}{Colors.RESET} "
        f"{msg_color}{msg}{Colors.RESET}"
    )
    
    # Add exception info if present
    if record["exception"]:
        formatted += f"\n{Colors.RED}{record['exception']}{Colors.RESET}"
    
    return formatted + "\n"


# =============================================================================
# File Formatter - Clean Text Without Colors
# =============================================================================

# Unicode icons to strip from file logs
ICONS_TO_STRIP = [
    '✅', '❌', '⚠️', '🔄', '📈', '📉', '🚦', '🟡', '🟢', '🔴',
    '💀', '🔌', '⏩', '🎯', '💰', '📊', 'ℹ️', '✓', '✗', '⚠',
    '│', '·', '━', '▶', '◀', '●', '○', '◉', '◎', '★', '☆',
    '🔒', '⏱️', '🔋', '💡', '🚀', '⚡', '🎲', '🔧', '📝', '🔍'
]


def strip_icons(text: str) -> str:
    """Remove Unicode icons from text for clean file logs"""
    for icon in ICONS_TO_STRIP:
        text = text.replace(icon, '')
    return text.strip()


def format_file(record: dict) -> str:
    """
    Plain formatter for file output with CET timezone - no colors, no icons.
    
    Format: YYYY-MM-DD HH:MM:SS | LEVEL | module:function:line - message
    """
    # Get CET timestamp
    timestamp = datetime.fromtimestamp(record["time"].timestamp(), tz=CET).strftime("%Y-%m-%d %H:%M:%S")
    
    level_tags = {
        'TRACE': 'TRCE',
        'DEBUG': 'DBUG',
        'INFO': 'INFO',
        'SUCCESS': 'SUCC',
        'WARNING': 'WARN',
        'ERROR': 'ERR!',
        'CRITICAL': 'CRIT',
    }
    
    level = record["level"].name
    level_tag = level_tags.get(level, 'INFO')
    
    # Module name
    module = record["name"]
    if module.startswith('src.'):
        module = module[4:]
    
    # Function and line
    function = record["function"]
    line = record["line"]
    
    # Strip icons from message
    msg = strip_icons(record["message"])
    
    # Format: timestamp | LEVEL | module:function:line - message
    formatted = f"{timestamp} | {level_tag: <4} | {module}:{function}:{line} - {msg}"
    
    return formatted + "\n"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_dark_logging(
    log_dir: Path,
    log_level: str = "INFO",
    log_retention_days: int = 30,
    enable_trade_log: bool = True
) -> None:
    """
    Configure loguru with dark-theme optimized formatters.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_retention_days: Days to keep old logs
        enable_trade_log: Whether to create separate trade log
    """
    log_dir.mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with dark-theme colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=format_console,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Main bot log file (no colors, clean text)
    logger.add(
        log_dir / "bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format=format_file,
        rotation="00:00",  # Rotate at midnight
        retention=f"{log_retention_days} days",
        compression="gz",
        enqueue=True,  # Thread-safe
    )
    
    # Trade log (if enabled)
    if enable_trade_log:
        logger.add(
            log_dir / "trades_{time:YYYY-MM-DD}.log",
            level="INFO",
            format=format_file,
            rotation="00:00",
            retention=f"{log_retention_days} days",
            compression="gz",
            enqueue=True,
            filter=lambda record: any(
                keyword in record["message"].lower()
                for keyword in ["fill", "trade", "order", "pnl", "profit", "loss"]
            ),
        )
    
    logger.info(f"Logging initialized: {log_level} level, {log_retention_days} day retention")


# =============================================================================
# Convenience Logging Functions
# =============================================================================

def log_phase(phase_num: int, phase_name: str) -> None:
    """Log a phase transition with emphasis"""
    logger.info(f"{'='*60}")
    logger.info(f"Phase {phase_num}: {phase_name}")
    logger.info(f"{'='*60}")


def log_trade(side: str, size: float, price: float, pnl: Optional[float] = None) -> None:
    """Log a trade execution with proper formatting"""
    pnl_str = f" | PnL: ${pnl:+.2f}" if pnl is not None else ""
    logger.info(f"Trade executed: {side} {size} @ ${price:.2f}{pnl_str}")


def log_status(
    position: float,
    equity: float,
    pnl: float,
    bids: int,
    asks: int,
    risk_level: str
) -> None:
    """Log bot status summary"""
    pnl_sign = '+' if pnl >= 0 else ''
    pnl_emoji = '📈' if pnl >= 0 else '📉'
    logger.info(
        f"Status: Pos={position:.4f} | Equity=${equity:.2f} | "
        f"PnL=${pnl_sign}{pnl:.2f} {pnl_emoji} | "
        f"Orders: {bids}B/{asks}A | Risk={risk_level}"
    )
