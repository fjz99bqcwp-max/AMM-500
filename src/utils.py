"""
Utility Functions and Helpers
Common utilities used across the HFT bot.
"""

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import numpy as np
from loguru import logger

# Try to import numba for JIT compilation
try:
    from numba import jit, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, JIT compilation disabled")
    # Create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Type variable for generic functions
T = TypeVar("T")


# =============================================================================
# Time Utilities
# =============================================================================

def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def get_timestamp_us() -> int:
    """Get current timestamp in microseconds for high-precision timing."""
    return int(time.time() * 1_000_000)


def timestamp_to_datetime(timestamp_ms: int) -> datetime:
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp."""
    return int(dt.timestamp() * 1000)


class LatencyTracker:
    """Track and report latency statistics."""
    
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.latencies: List[float] = []
        self._start_time: Optional[float] = None
    
    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and record latency."""
        if self._start_time is None:
            return 0.0
        
        latency = (time.perf_counter() - self._start_time) * 1000  # Convert to ms
        self.latencies.append(latency)
        
        # Keep only last window_size entries
        if len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size:]
        
        self._start_time = None
        return latency
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        arr = np.array(self.latencies)
        return {
            "avg": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    
    def log_stats(self) -> None:
        """Log latency statistics."""
        stats = self.get_stats()
        logger.info(
            f"{self.name} latency (ms): avg={stats['avg']:.2f}, "
            f"p50={stats['p50']:.2f}, p95={stats['p95']:.2f}, p99={stats['p99']:.2f}"
        )


def measure_latency(tracker: LatencyTracker):
    """Decorator to measure function latency."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            tracker.start()
            try:
                return await func(*args, **kwargs)
            finally:
                tracker.stop()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            tracker.start()
            try:
                return func(*args, **kwargs)
            finally:
                tracker.stop()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# =============================================================================
# Price and Size Utilities
# =============================================================================

def round_price(price: float, tick_size: float = 0.1) -> float:
    """
    Round price to the nearest tick size using Decimal for precision.
    
    Args:
        price: Price to round
        tick_size: Minimum price increment (default 0.1 for BTC)
    
    Returns:
        Rounded price with no floating point artifacts
    """
    # Use Decimal for precise rounding to avoid float artifacts
    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))
    # Round to nearest tick
    rounded = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * decimal_tick
    return float(rounded)


def round_size(size: float, lot_size: float = 0.001) -> float:
    """
    Round size down to the nearest lot size.
    
    Args:
        size: Size to round
        lot_size: Minimum size increment (default 0.001 for BTC)
    
    Returns:
        Rounded size (always rounds down to avoid over-ordering)
    """
    decimal_size = Decimal(str(size))
    decimal_lot = Decimal(str(lot_size))
    return float(decimal_size.quantize(decimal_lot, rounding=ROUND_DOWN))


def calculate_mid_price(best_bid: float, best_ask: float) -> float:
    """Calculate mid price from bid/ask."""
    return (best_bid + best_ask) / 2


def calculate_spread_bps(best_bid: float, best_ask: float) -> float:
    """Calculate spread in basis points."""
    mid = calculate_mid_price(best_bid, best_ask)
    if mid == 0:
        return 0.0
    return ((best_ask - best_bid) / mid) * 10000


def price_to_tick(price: float, tick_size: float = 0.1) -> int:
    """Convert price to tick number."""
    return int(round(price / tick_size))


def tick_to_price(tick: int, tick_size: float = 0.1) -> float:
    """Convert tick number to price."""
    return tick * tick_size


# =============================================================================
# Order Book Utilities (JIT-compiled for performance)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def calculate_vwap_numba(prices: np.ndarray, sizes: np.ndarray) -> float:
        """Calculate volume-weighted average price using Numba."""
        total_value = 0.0
        total_size = 0.0
        for i in range(len(prices)):
            total_value += prices[i] * sizes[i]
            total_size += sizes[i]
        if total_size == 0:
            return 0.0
        return total_value / total_size
    
    @jit(nopython=True, cache=True)
    def calculate_imbalance_numba(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
        """Calculate order book imbalance using Numba."""
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total
    
    @jit(nopython=True, cache=True)
    def calculate_microprice_numba(
        best_bid: float, 
        best_ask: float, 
        bid_size: float, 
        ask_size: float
    ) -> float:
        """Calculate microprice (size-weighted mid price) using Numba."""
        total_size = bid_size + ask_size
        if total_size == 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_size + best_ask * bid_size) / total_size
else:
    # Fallback implementations without Numba
    def calculate_vwap_numba(prices: np.ndarray, sizes: np.ndarray) -> float:
        total_value = np.sum(prices * sizes)
        total_size = np.sum(sizes)
        if total_size == 0:
            return 0.0
        return total_value / total_size
    
    def calculate_imbalance_numba(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total
    
    def calculate_microprice_numba(
        best_bid: float, 
        best_ask: float, 
        bid_size: float, 
        ask_size: float
    ) -> float:
        total_size = bid_size + ask_size
        if total_size == 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_size + best_ask * bid_size) / total_size


def calculate_vwap(prices: List[float], sizes: List[float]) -> float:
    """Calculate volume-weighted average price."""
    return calculate_vwap_numba(np.array(prices), np.array(sizes))


def calculate_imbalance(bid_sizes: List[float], ask_sizes: List[float]) -> float:
    """Calculate order book imbalance (-1 to 1, positive = more bids)."""
    return calculate_imbalance_numba(np.array(bid_sizes), np.array(ask_sizes))


def calculate_microprice(
    best_bid: float, 
    best_ask: float, 
    bid_size: float, 
    ask_size: float
) -> float:
    """Calculate microprice (size-weighted mid price)."""
    return calculate_microprice_numba(best_bid, best_ask, bid_size, ask_size)


# =============================================================================
# Volatility Utilities
# =============================================================================

def calculate_realized_volatility(
    prices: np.ndarray, 
    window: int = 20,
    annualize: bool = True
) -> float:
    """
    Calculate realized volatility from price series.
    
    Args:
        prices: Array of prices
        window: Rolling window size
        annualize: If True, annualize the volatility
    
    Returns:
        Volatility (annualized if specified, in percentage)
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Use last 'window' returns
    if len(log_returns) > window:
        log_returns = log_returns[-window:]
    
    # Calculate standard deviation
    vol = np.std(log_returns)
    
    if annualize:
        # Assume 1-minute data, ~525600 minutes per year
        vol = vol * np.sqrt(525600) * 100
    
    return float(vol)


def calculate_ewma_volatility(
    prices: np.ndarray,
    span: int = 20,
    annualize: bool = True,
    samples_per_minute: float = 30.0  # Default: ~2 second sampling
) -> float:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    
    Args:
        prices: Array of prices
        span: EWMA span parameter
        annualize: If True, annualize the volatility
        samples_per_minute: How many price samples per minute (for annualization)
    
    Returns:
        EWMA volatility (as percentage if annualized)
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Calculate EWMA variance
    alpha = 2 / (span + 1)
    variance = 0.0
    for ret in log_returns:
        variance = alpha * (ret ** 2) + (1 - alpha) * variance
    
    vol = np.sqrt(variance)
    
    if annualize:
        # Annualize: samples/min * 60 min/hr * 24 hr/day * 365 days
        samples_per_year = samples_per_minute * 60 * 24 * 365
        vol = vol * np.sqrt(samples_per_year) * 100
    
    return float(vol)


# =============================================================================
# Implied Volatility Calculation (JIT-compiled for HFT performance)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def calculate_iv_fast(prices: np.ndarray, window: int = 60) -> float:
        """
        Fast IV calculation using Numba JIT compilation.
        
        Uses Parkinson volatility estimator (high-low range based) for efficiency.
        Falls back to close-to-close if only close prices available.
        
        Args:
            prices: Array of recent prices (minute-level recommended)
            window: Window size in periods (default 60 = 1 hour of minute data)
        
        Returns:
            Annualized implied volatility as decimal (e.g., 0.15 = 15%)
        """
        n = len(prices)
        if n < 2:
            return 0.15  # Default 15% if insufficient data
        
        # Use last 'window' prices
        start_idx = max(0, n - window)
        recent_prices = prices[start_idx:]
        
        # Calculate log returns
        sum_sq = 0.0
        count = 0
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                log_ret = np.log(recent_prices[i] / recent_prices[i-1])
                sum_sq += log_ret * log_ret
                count += 1
        
        if count == 0:
            return 0.15
        
        # Variance and annualization (assume minute data: 525600 minutes/year)
        variance = sum_sq / count
        vol = np.sqrt(variance) * np.sqrt(525600)
        
        # Clamp to reasonable range [0.05, 0.50] (5% to 50% annual vol)
        return max(0.05, min(0.50, vol))
    
    @jit(nopython=True, cache=True)
    def calculate_spread_from_iv(
        iv: float,
        vol_threshold: float = 0.15,
        min_spread_bps: float = 2.0,
        max_spread_bps: float = 50.0,
        low_vol_spread_bps: float = 3.0,
        high_vol_spread_bps: float = 40.0
    ) -> float:
        """
        Calculate optimal spread based on implied volatility.
        
        - Low vol (<15%): Use tight spreads (2-3 bps)
        - High vol (>15%): Auto-widen to 30-50 bps
        
        Args:
            iv: Implied volatility (decimal, e.g., 0.15 = 15%)
            vol_threshold: Threshold for widening (default 0.15 = 15%)
            min_spread_bps: Minimum spread in basis points
            max_spread_bps: Maximum spread in basis points
            low_vol_spread_bps: Spread to use in low vol
            high_vol_spread_bps: Spread to use in high vol
        
        Returns:
            Optimal spread in basis points
        """
        if iv < vol_threshold:
            # Low volatility: use tight spreads (2-3 bps)
            # Linear interpolation from min to low_vol target
            vol_ratio = iv / vol_threshold
            spread = min_spread_bps + (low_vol_spread_bps - min_spread_bps) * vol_ratio
        else:
            # High volatility: auto-widen (30-50 bps)
            # Linear interpolation from low_vol to high_vol target
            excess_vol = (iv - vol_threshold) / (0.30 - vol_threshold)  # Scale to 30%
            excess_vol = min(1.0, max(0.0, excess_vol))
            spread = low_vol_spread_bps + (high_vol_spread_bps - low_vol_spread_bps) * excess_vol
        
        return max(min_spread_bps, min(max_spread_bps, spread))

else:
    # Fallback without Numba
    def calculate_iv_fast(prices: np.ndarray, window: int = 60) -> float:
        """Fast IV calculation (non-JIT fallback)."""
        n = len(prices)
        if n < 2:
            return 0.15
        
        start_idx = max(0, n - window)
        recent_prices = prices[start_idx:]
        
        log_returns = np.diff(np.log(recent_prices))
        if len(log_returns) == 0:
            return 0.15
        
        variance = np.mean(log_returns ** 2)
        vol = np.sqrt(variance) * np.sqrt(525600)
        
        return max(0.05, min(0.50, vol))
    
    def calculate_spread_from_iv(
        iv: float,
        vol_threshold: float = 0.15,
        min_spread_bps: float = 2.0,
        max_spread_bps: float = 50.0,
        low_vol_spread_bps: float = 3.0,
        high_vol_spread_bps: float = 40.0
    ) -> float:
        """Calculate optimal spread based on IV (non-JIT fallback)."""
        if iv < vol_threshold:
            vol_ratio = iv / vol_threshold
            spread = min_spread_bps + (low_vol_spread_bps - min_spread_bps) * vol_ratio
        else:
            excess_vol = (iv - vol_threshold) / (0.30 - vol_threshold)
            excess_vol = min(1.0, max(0.0, excess_vol))
            spread = low_vol_spread_bps + (high_vol_spread_bps - low_vol_spread_bps) * excess_vol
        
        return max(min_spread_bps, min(max_spread_bps, spread))


# =============================================================================
# Risk Utilities
# =============================================================================

def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion fraction for optimal position sizing.
    
    Args:
        win_rate: Probability of winning (0 to 1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (absolute value)
    
    Returns:
        Kelly fraction (0 to 1, capped at 0.25 for safety)
    """
    if avg_loss == 0:
        return 0.0
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win rate, q = 1-p, b = avg_win/avg_loss
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    
    # Cap at 25% for safety (half-Kelly is common in practice)
    return max(0.0, min(0.25, kelly))


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525600  # minutes
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    return (annual_return - risk_free_rate) / annual_std


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0
    
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = i
        
        dd = (peak - value) / peak if peak > 0 else 0.0
        
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i
    
    return max_dd, max_dd_peak_idx, max_dd_trough_idx


# =============================================================================
# Async Utilities
# =============================================================================

async def retry_async(
    func: Callable,
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Hyperliquid limit: 1200 weight per minute per IP
    """
    
    def __init__(self, rate: float = 1200, per: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum tokens (weight) allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, weight: int = 1) -> None:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            weight: Number of tokens to acquire
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
            self.last_update = now
            
            if self.tokens < weight:
                wait_time = (weight - self.tokens) * (self.per / self.rate)
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = weight
            
            self.tokens -= weight
    
    def get_available_tokens(self) -> float:
        """Get current available tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.rate, self.tokens + elapsed * (self.rate / self.per))


# =============================================================================
# Data Structures
# =============================================================================

class CircularBuffer:
    """Fixed-size circular buffer for efficient data storage."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity)
        self.index = 0
        self.size = 0
    
    def append(self, value: float) -> None:
        """Append a value to the buffer."""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_array(self) -> np.ndarray:
        """Get buffer contents as array (oldest to newest)."""
        if self.size < self.capacity:
            return self.buffer[:self.size].copy()
        return np.concatenate([
            self.buffer[self.index:],
            self.buffer[:self.index]
        ])
    
    def get_last(self, n: int = 1) -> np.ndarray:
        """Get last n values."""
        arr = self.get_array()
        return arr[-n:] if len(arr) >= n else arr
    
    def mean(self) -> float:
        """Get mean of buffer."""
        if self.size == 0:
            return 0.0
        if self.size < self.capacity:
            return float(np.mean(self.buffer[:self.size]))
        return float(np.mean(self.buffer))
    
    def std(self) -> float:
        """Get standard deviation of buffer."""
        if self.size < 2:
            return 0.0
        if self.size < self.capacity:
            return float(np.std(self.buffer[:self.size]))
        return float(np.std(self.buffer))


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_price(price: float) -> str:
    """Format price for display."""
    return f"${price:,.2f}"


def format_size(size: float) -> str:
    """Format size for display."""
    return f"{size:.4f}"


def format_pnl(pnl: float) -> str:
    """Format PnL with color indicator."""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:,.2f}"


def format_percentage(pct: float) -> str:
    """Format percentage for display."""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def format_timestamp(ts_ms: int) -> str:
    """Format timestamp for logging."""
    dt = timestamp_to_datetime(ts_ms)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
