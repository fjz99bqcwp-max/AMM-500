"""
Utility Functions for AMM-500
==============================
Common helper functions.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
import json

from loguru import logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with loguru."""
    import sys
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>{name}</cyan> | {message}",
        colorize=True
    )
    
    # File handler
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days"
        )


def format_price(price: float, decimals: int = 2) -> str:
    """Format price for display."""
    return f"${price:,.{decimals}f}"


def format_size(size: float, decimals: int = 4) -> str:
    """Format size for display."""
    return f"{size:.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage for display."""
    return f"{value * 100:.{decimals}f}%"


def format_bps(value: float, decimals: int = 1) -> str:
    """Format basis points for display."""
    return f"{value:.{decimals}f} bps"


def round_price(price: float, tick_size: float = 0.01) -> float:
    """Round price to tick size."""
    return round(price / tick_size) * tick_size


def round_size(size: float, lot_size: float = 0.0001) -> float:
    """Round size to lot size."""
    return round(size / lot_size) * lot_size


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return a / b if b != 0 else default


async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """Retry async function with exponential backoff."""
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {current_delay}s: {e}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff
    
    raise last_exception


def load_json(path: str) -> Optional[Dict]:
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON {path}: {e}")
        return None


def save_json(data: Dict, path: str) -> bool:
    """Save data to JSON file."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON {path}: {e}")
        return False


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(datetime.now().timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds to datetime."""
    return datetime.fromtimestamp(ms / 1000)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0
    
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = (time.perf_counter() - self.start) * 1000  # ms
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.1f}ms")
