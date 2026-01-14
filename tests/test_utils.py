"""
Tests for utility functions.
"""

import numpy as np
import pytest
from src.utils import (
    round_price,
    round_size,
    calculate_mid_price,
    calculate_spread_bps,
    calculate_vwap,
    calculate_imbalance,
    calculate_microprice,
    calculate_realized_volatility,
    calculate_kelly_fraction,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    CircularBuffer,
    RateLimiter,
    LatencyTracker,
    get_timestamp_ms,
    format_price,
    format_pnl,
)


class TestPriceUtils:
    """Tests for price-related utility functions."""
    
    def test_round_price_default_tick(self):
        """Test price rounding with default tick size."""
        assert round_price(100.05, 0.1) == 100.1  # Rounds to nearest 0.1
        assert round_price(100.15, 0.1) == 100.2
        assert round_price(100.149, 0.1) == 100.1
    
    def test_round_price_custom_tick(self):
        """Test price rounding with custom tick size."""
        assert round_price(100.025, 0.05) == 100.05  # Rounds to nearest 0.05
        assert round_price(100.075, 0.05) == 100.10
    
    def test_round_size_default_lot(self):
        """Test size rounding with default lot size."""
        assert round_size(0.0015, 0.001) == 0.001
        assert round_size(0.0099, 0.001) == 0.009
        assert round_size(1.2345, 0.001) == 1.234
    
    def test_round_size_rounds_down(self):
        """Size should always round down to avoid over-ordering."""
        assert round_size(0.9999, 0.001) == 0.999
        assert round_size(0.0019, 0.001) == 0.001
    
    def test_calculate_mid_price(self):
        """Test mid price calculation."""
        assert calculate_mid_price(100.0, 100.2) == 100.1
        assert calculate_mid_price(99.5, 100.5) == 100.0
    
    def test_calculate_spread_bps(self):
        """Test spread calculation in basis points."""
        # 0.2% spread at $100 = 20 bps
        spread = calculate_spread_bps(99.9, 100.1)
        assert abs(spread - 20.0) < 0.1
        
        # Wider spread
        spread = calculate_spread_bps(99.0, 101.0)
        assert abs(spread - 200.0) < 0.1


class TestOrderBookUtils:
    """Tests for order book analysis functions."""
    
    def test_calculate_vwap(self):
        """Test VWAP calculation."""
        prices = [100.0, 99.5, 99.0]
        sizes = [1.0, 2.0, 1.0]
        vwap = calculate_vwap(prices, sizes)
        expected = (100 * 1 + 99.5 * 2 + 99 * 1) / 4
        assert abs(vwap - expected) < 0.001
    
    def test_calculate_vwap_empty(self):
        """Test VWAP with empty arrays."""
        assert calculate_vwap([], []) == 0.0
    
    def test_calculate_imbalance(self):
        """Test order book imbalance calculation."""
        # Equal sizes = 0 imbalance
        imbalance = calculate_imbalance([1.0, 1.0], [1.0, 1.0])
        assert abs(imbalance) < 0.001
        
        # More bids = positive imbalance
        imbalance = calculate_imbalance([2.0], [1.0])
        assert imbalance > 0
        
        # More asks = negative imbalance
        imbalance = calculate_imbalance([1.0], [2.0])
        assert imbalance < 0
    
    def test_calculate_microprice(self):
        """Test microprice calculation."""
        # Equal sizes = mid price
        microprice = calculate_microprice(99.0, 101.0, 1.0, 1.0)
        assert abs(microprice - 100.0) < 0.001
        
        # Larger ask size = closer to bid
        microprice = calculate_microprice(99.0, 101.0, 1.0, 2.0)
        assert microprice < 100.0
        
        # Larger bid size = closer to ask
        microprice = calculate_microprice(99.0, 101.0, 2.0, 1.0)
        assert microprice > 100.0


class TestVolatilityUtils:
    """Tests for volatility calculations."""
    
    def test_calculate_realized_volatility(self):
        """Test realized volatility calculation."""
        # Constant prices = 0 volatility
        prices = np.array([100.0] * 100)
        vol = calculate_realized_volatility(prices, window=20, annualize=False)
        assert abs(vol) < 0.001
        
        # Increasing prices should have some volatility
        prices = np.linspace(100, 110, 100)
        vol = calculate_realized_volatility(prices, window=20, annualize=False)
        assert vol > 0
    
    def test_calculate_realized_volatility_short_series(self):
        """Test volatility with short price series."""
        prices = np.array([100.0])
        vol = calculate_realized_volatility(prices)
        assert vol == 0.0


class TestRiskUtils:
    """Tests for risk calculation functions."""
    
    def test_calculate_kelly_fraction(self):
        """Test Kelly criterion calculation."""
        # 60% win rate, 1:1 risk/reward
        kelly = calculate_kelly_fraction(0.6, 1.0, 1.0)
        assert 0 < kelly < 0.25  # Should be capped
        
        # 50% win rate, 1:1 = 0 Kelly
        kelly = calculate_kelly_fraction(0.5, 1.0, 1.0)
        assert abs(kelly) < 0.001
        
        # Losing strategy = 0 Kelly
        kelly = calculate_kelly_fraction(0.3, 1.0, 1.0)
        assert kelly == 0.0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Positive returns with low volatility = high Sharpe
        returns = np.array([0.001] * 100)  # 0.1% per period
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0
        
        # Negative returns = negative Sharpe
        returns = np.array([-0.001] * 100)
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # No drawdown
        equity = np.array([100, 101, 102, 103])
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        assert dd == 0.0
        
        # 10% drawdown
        equity = np.array([100, 110, 100, 105])
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        assert abs(dd - 0.0909) < 0.01  # ~9.09%
        assert peak_idx == 1
        assert trough_idx == 2


class TestCircularBuffer:
    """Tests for CircularBuffer class."""
    
    def test_append_and_get(self):
        """Test basic append and retrieval."""
        buf = CircularBuffer(5)
        
        for i in range(3):
            buf.append(float(i))
        
        arr = buf.get_array()
        assert len(arr) == 3
        assert list(arr) == [0.0, 1.0, 2.0]
    
    def test_circular_behavior(self):
        """Test that buffer wraps around correctly."""
        buf = CircularBuffer(3)
        
        for i in range(5):
            buf.append(float(i))
        
        arr = buf.get_array()
        assert len(arr) == 3
        assert list(arr) == [2.0, 3.0, 4.0]
    
    def test_mean_and_std(self):
        """Test statistical functions."""
        buf = CircularBuffer(10)
        
        for i in range(10):
            buf.append(float(i))
        
        assert abs(buf.mean() - 4.5) < 0.001
        assert buf.std() > 0


class TestLatencyTracker:
    """Tests for LatencyTracker class."""
    
    def test_basic_tracking(self):
        """Test basic latency tracking."""
        tracker = LatencyTracker("test", window_size=10)
        
        tracker.start()
        # Simulate some work
        import time
        time.sleep(0.001)
        latency = tracker.stop()
        
        assert latency > 0
        
        stats = tracker.get_stats()
        assert stats['avg'] > 0
        assert stats['min'] <= stats['avg'] <= stats['max']
    
    def test_empty_stats(self):
        """Test stats with no measurements."""
        tracker = LatencyTracker("test")
        stats = tracker.get_stats()
        
        assert stats['avg'] == 0.0
        assert stats['p95'] == 0.0


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(rate=100, per=1.0)
        
        # Should be able to acquire immediately
        await limiter.acquire(1)
        available = limiter.get_available_tokens()
        assert available < 100
    
    @pytest.mark.asyncio
    async def test_token_regeneration(self):
        """Test that tokens regenerate over time."""
        import asyncio
        
        limiter = RateLimiter(rate=100, per=1.0)
        
        # Use some tokens
        await limiter.acquire(50)
        
        # Wait for regeneration
        await asyncio.sleep(0.5)
        
        available = limiter.get_available_tokens()
        assert available > 50


class TestFormatters:
    """Tests for formatting functions."""
    
    def test_format_price(self):
        """Test price formatting."""
        assert format_price(1234.56) == "$1,234.56"
        assert format_price(0.12) == "$0.12"
    
    def test_format_pnl(self):
        """Test PnL formatting."""
        assert format_pnl(100.0) == "+$100.00"
        assert format_pnl(-50.0) == "$-50.00"
        assert format_pnl(0.0) == "+$0.00"


class TestTimestampUtils:
    """Tests for timestamp utilities."""
    
    def test_get_timestamp_ms(self):
        """Test millisecond timestamp generation."""
        ts = get_timestamp_ms()
        
        # Should be a reasonable timestamp (after 2020)
        assert ts > 1577836800000  # Jan 1, 2020
        
        # Should be 13 digits
        assert len(str(ts)) == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
