#!/usr/bin/env python3
"""
Tests for Market Making Strategy
=================================
Coverage target: >90% of core MM logic
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.strategy import MarketMakingStrategy, VolatilityPredictor


class TestVolatilityPredictor:
    """Tests for PyTorch volatility predictor."""
    
    def test_init(self):
        """Test predictor initialization."""
        predictor = VolatilityPredictor()
        assert predictor.model is not None
        assert predictor.lookback == 60
    
    def test_predict_with_short_data(self):
        """Test prediction with insufficient data."""
        predictor = VolatilityPredictor()
        returns = np.random.randn(10) * 0.01  # Too short
        vol = predictor.predict(returns)
        # Should return realized vol fallback
        assert vol > 0
    
    def test_predict_with_sufficient_data(self):
        """Test prediction with sufficient data."""
        predictor = VolatilityPredictor()
        returns = np.random.randn(100) * 0.01
        vol = predictor.predict(returns)
        assert 0 < vol < 1  # Reasonable volatility range
    
    def test_train(self):
        """Test model training."""
        predictor = VolatilityPredictor()
        # Create dummy training data
        returns = pd.Series(np.random.randn(1000) * 0.01)
        
        # Should not raise
        predictor.train(returns, epochs=1)


class TestMicroprice:
    """Tests for microprice calculation."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 5
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        return MarketMakingStrategy(config)
    
    def test_microprice_balanced_book(self, strategy):
        """Test microprice with balanced order book."""
        book = {
            "bids": [[100.0, 10.0], [99.5, 10.0]],
            "asks": [[100.5, 10.0], [101.0, 10.0]],
        }
        microprice = strategy._calculate_microprice(book)
        # With equal sizes, microprice should be near mid
        mid = (100.0 + 100.5) / 2
        assert abs(microprice - mid) < 0.1
    
    def test_microprice_bid_heavy(self, strategy):
        """Test microprice with bid-heavy book."""
        book = {
            "bids": [[100.0, 100.0], [99.5, 50.0]],  # Large bids
            "asks": [[100.5, 10.0], [101.0, 10.0]],   # Small asks
        }
        microprice = strategy._calculate_microprice(book)
        # Should be closer to ask (buying pressure)
        mid = (100.0 + 100.5) / 2
        assert microprice > mid
    
    def test_microprice_ask_heavy(self, strategy):
        """Test microprice with ask-heavy book."""
        book = {
            "bids": [[100.0, 10.0], [99.5, 10.0]],
            "asks": [[100.5, 100.0], [101.0, 50.0]],
        }
        microprice = strategy._calculate_microprice(book)
        # Should be closer to bid (selling pressure)
        mid = (100.0 + 100.5) / 2
        assert microprice < mid


class TestImbalance:
    """Tests for order book imbalance."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 5
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        return MarketMakingStrategy(config)
    
    def test_balanced_imbalance(self, strategy):
        """Test imbalance with balanced book."""
        book = {
            "bids": [[100.0, 10.0]],
            "asks": [[100.5, 10.0]],
        }
        imbalance = strategy._calculate_imbalance(book)
        assert abs(imbalance) < 0.1
    
    def test_bid_heavy_imbalance(self, strategy):
        """Test imbalance with more bids."""
        book = {
            "bids": [[100.0, 100.0]],
            "asks": [[100.5, 10.0]],
        }
        imbalance = strategy._calculate_imbalance(book)
        assert imbalance > 0.5  # Positive = bid heavy
    
    def test_ask_heavy_imbalance(self, strategy):
        """Test imbalance with more asks."""
        book = {
            "bids": [[100.0, 10.0]],
            "asks": [[100.5, 100.0]],
        }
        imbalance = strategy._calculate_imbalance(book)
        assert imbalance < -0.5  # Negative = ask heavy


class TestSpreadCalculation:
    """Tests for dynamic spread calculation."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 50
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        return MarketMakingStrategy(config)
    
    def test_low_vol_tight_spread(self, strategy):
        """Test that low vol produces tight spread."""
        spread = strategy._calculate_spread(volatility=0.03)  # 3% vol
        assert spread < 0.001  # < 10 bps
    
    def test_high_vol_wide_spread(self, strategy):
        """Test that high vol produces wide spread."""
        spread = strategy._calculate_spread(volatility=0.20)  # 20% vol
        assert spread > 0.002  # > 20 bps


class TestQuoteBuilding:
    """Tests for quote generation."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 50
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        return MarketMakingStrategy(config)
    
    def test_exponential_gaps(self, strategy):
        """Test that quotes have exponential spacing."""
        book = {
            "bids": [[100.0, 10.0]],
            "asks": [[100.5, 10.0]],
        }
        bids, asks = strategy._build_quotes(
            book=book,
            mid=100.25,
            spread=0.005,  # 50 bps
            volatility=0.10,
        )
        
        # Check we got quotes
        assert len(bids) > 0
        assert len(asks) > 0
        
        # Check bids are descending
        bid_prices = [b[0] for b in bids]
        assert all(bid_prices[i] > bid_prices[i+1] for i in range(len(bid_prices)-1))
        
        # Check asks are ascending
        ask_prices = [a[0] for a in asks]
        assert all(ask_prices[i] < ask_prices[i+1] for i in range(len(ask_prices)-1))
        
        # Check exponential gap growth
        if len(bid_prices) >= 3:
            gap1 = bid_prices[0] - bid_prices[1]
            gap2 = bid_prices[1] - bid_prices[2]
            assert gap2 > gap1 * 0.9  # Second gap should be larger
    
    def test_depth_aware_sizing(self, strategy):
        """Test that sizes decrease with depth."""
        book = {
            "bids": [[100.0, 10.0]],
            "asks": [[100.5, 10.0]],
        }
        bids, asks = strategy._build_quotes(
            book=book,
            mid=100.25,
            spread=0.005,
            volatility=0.10,
        )
        
        # Inner levels should have larger sizes
        if len(bids) >= 2:
            assert bids[0][1] >= bids[-1][1]


class TestReduceOnly:
    """Tests for reduce-only mode."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 50
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        config.risk.max_drawdown = 0.02
        config.risk.max_taker_ratio = 0.05
        config.risk.margin_pause_threshold = 0.90
        return MarketMakingStrategy(config)
    
    def test_high_drawdown_triggers_reduce(self, strategy):
        """Test reduce-only activates on high drawdown."""
        should_reduce = strategy._should_reduce_only(
            current_drawdown=0.025,  # > 2%
            taker_ratio=0.03,
            margin_ratio=0.50,
        )
        assert should_reduce is True
    
    def test_high_taker_triggers_reduce(self, strategy):
        """Test reduce-only activates on high taker ratio."""
        should_reduce = strategy._should_reduce_only(
            current_drawdown=0.01,
            taker_ratio=0.35,  # > 30%
            margin_ratio=0.50,
        )
        assert should_reduce is True
    
    def test_normal_conditions_no_reduce(self, strategy):
        """Test no reduce-only under normal conditions."""
        should_reduce = strategy._should_reduce_only(
            current_drawdown=0.005,
            taker_ratio=0.03,
            margin_ratio=0.50,
        )
        assert should_reduce is False


class TestOrderUpdate:
    """Tests for smart order matching."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 50
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        return MarketMakingStrategy(config)
    
    def test_matching_orders_not_cancelled(self, strategy):
        """Test that matching orders are kept."""
        existing = [
            {"price": 100.0, "size": 1.0, "oid": "1"},
            {"price": 99.5, "size": 1.0, "oid": "2"},
        ]
        new_quotes = [
            [100.0, 1.0],  # Same as existing
            [99.0, 1.0],   # New level
        ]
        
        to_cancel, to_place = strategy._update_orders(
            existing_orders=existing,
            new_quotes=new_quotes,
            side="buy",
        )
        
        # Order at 100.0 should not be cancelled
        assert "1" not in to_cancel
        # Order at 99.5 should be cancelled (not in new quotes)
        assert "2" in to_cancel
        # New order at 99.0 should be placed
        assert any(p[0] == 99.0 for p in to_place)


class TestIntegration:
    """Integration tests for full strategy cycle."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = Mock()
        config.trading.leverage = 10
        config.trading.min_spread_bps = 1
        config.trading.max_spread_bps = 50
        config.trading.order_levels = 10
        config.trading.collateral = 1000
        config.risk.max_drawdown = 0.02
        config.risk.max_taker_ratio = 0.05
        config.risk.margin_pause_threshold = 0.90
        return MarketMakingStrategy(config)
    
    @pytest.mark.asyncio
    async def test_full_cycle(self, strategy):
        """Test complete strategy cycle."""
        book = {
            "bids": [[100.0, 50.0], [99.5, 30.0], [99.0, 20.0]],
            "asks": [[100.5, 50.0], [101.0, 30.0], [101.5, 20.0]],
        }
        
        # Mock exchange
        exchange = AsyncMock()
        exchange.get_orderbook.return_value = book
        exchange.get_position.return_value = {"size": 0, "unrealized_pnl": 0}
        exchange.get_open_orders.return_value = []
        exchange.get_margin_ratio.return_value = 0.50
        
        # Run one cycle
        with patch.object(strategy, "_estimate_volatility", return_value=0.10):
            await strategy.run_cycle(exchange)
        
        # Should have placed orders
        assert exchange.batch_place_orders.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
