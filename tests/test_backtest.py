"""
Tests for backtesting module.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.core.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    MarketDataLoader,
    MonteCarloSimulator,
    SimulatedOrder,
    SimulatedFill,
    SimulatedPosition,
)


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default backtest configuration."""
        config = BacktestConfig()

        assert config.initial_capital == 1000.0
        assert config.leverage == 25
        assert config.maker_rebate == 0.00003
        assert config.taker_fee == 0.00035
        assert config.order_levels == 20

    def test_custom_values(self):
        """Test custom backtest configuration."""
        config = BacktestConfig(
            initial_capital=5000.0,
            leverage=10,
            order_levels=3,
        )

        assert config.initial_capital == 5000.0
        assert config.leverage == 10
        assert config.order_levels == 3


class TestSimulatedOrder:
    """Tests for SimulatedOrder dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = SimulatedOrder(
            order_id="test_1",
            side="buy",
            price=100000.0,
            size=0.1,
            timestamp=1000000,
        )

        assert order.order_id == "test_1"
        assert order.is_filled == False
        assert order.remaining == 0.1

    def test_order_fill(self):
        """Test order fill tracking."""
        order = SimulatedOrder(
            order_id="test_1",
            side="buy",
            price=100000.0,
            size=0.1,
            timestamp=1000000,
            filled_size=0.1,
        )

        assert order.is_filled == True
        assert order.remaining == 0.0


class TestSimulatedPosition:
    """Tests for SimulatedPosition dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        position = SimulatedPosition()

        assert position.size == 0.0
        assert position.unrealized_pnl == 0.0

    def test_update_mark(self):
        """Test mark price update."""
        position = SimulatedPosition(
            size=0.1,
            entry_price=100000.0,
        )

        position.update_mark(101000.0)

        # 0.1 BTC * $1000 profit = $100
        assert position.unrealized_pnl == 100.0

    def test_update_mark_short(self):
        """Test mark price update for short position."""
        position = SimulatedPosition(
            size=-0.1,
            entry_price=100000.0,
        )

        position.update_mark(99000.0)

        # -0.1 BTC * -$1000 loss = $100 profit for short
        assert position.unrealized_pnl == 100.0


class TestMarketDataLoader:
    """Tests for MarketDataLoader class."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = MarketDataLoader()

        data = loader.generate_synthetic_data(
            days=1,
            interval_minutes=60,
            initial_price=100000.0,
            volatility=0.02,
            seed=42,
        )

        assert len(data) == 24  # 24 hours * 1 per hour
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns

        # Prices should be reasonable
        assert data["close"].min() > 50000
        assert data["close"].max() < 200000

    def test_generate_synthetic_data_longer_period(self):
        """Test longer synthetic data generation."""
        loader = MarketDataLoader()

        data = loader.generate_synthetic_data(
            days=7,
            interval_minutes=1,
            seed=42,
        )

        # 7 days * 24 hours * 60 minutes = 10080
        assert len(data) == 10080

    def test_generate_orderbook_snapshots(self):
        """Test orderbook snapshot generation."""
        loader = MarketDataLoader()

        price_data = loader.generate_synthetic_data(days=1, interval_minutes=60)
        snapshots = loader.generate_orderbook_snapshots(price_data, levels=5)

        assert len(snapshots) == len(price_data)

        # Check first snapshot structure
        snap = snapshots[0]
        assert "bids" in snap
        assert "asks" in snap
        assert "mid_price" in snap
        assert len(snap["bids"]) == 5
        assert len(snap["asks"]) == 5


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    @pytest.fixture
    def engine(self):
        """Create backtest engine."""
        config = BacktestConfig(
            initial_capital=2000.0,
            leverage=25,
            min_spread_bps=5.0,
            order_levels=3,
            fill_probability=1.0,  # Guaranteed fills for testing
        )
        return BacktestEngine(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        loader = MarketDataLoader()
        return loader.generate_synthetic_data(
            days=1,
            interval_minutes=1,
            seed=42,
        )

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.equity == 2000.0
        assert engine.position.size == 0.0
        assert len(engine.open_orders) == 0

    def test_run_basic(self, engine, sample_data):
        """Test basic backtest run."""
        result = engine.run(sample_data)

        assert result is not None
        assert result.total_trades >= 0
        assert result.duration_days > 0

    def test_run_generates_equity_curve(self, engine, sample_data):
        """Test that equity curve is generated."""
        result = engine.run(sample_data)

        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 2000.0  # Starting equity

    def test_fee_calculation(self, engine, sample_data):
        """Test that fees are calculated."""
        result = engine.run(sample_data)

        # Either fees paid or rebates earned (depending on fills)
        assert result.total_fees >= 0 or result.total_rebates >= 0

    def test_place_order(self, engine):
        """Test order placement."""
        order_id = engine._place_order("buy", 100000.0, 0.1, 1000000)

        assert order_id in engine.open_orders
        assert engine.open_orders[order_id].side == "buy"
        assert engine.open_orders[order_id].price == 100000.0
        assert engine.open_orders[order_id].size == 0.1

    def test_calculate_volatility(self, engine):
        """Test volatility calculation."""
        # Add some price history
        for i in range(30):
            engine._price_history.append(100000 + i * 10)

        vol = engine._calculate_recent_volatility(window=20)
        assert vol > 0


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = BacktestResult()

        assert result.total_trades == 0
        assert result.net_pnl == 0.0
        assert result.max_drawdown == 0.0

    def test_summary_generation(self):
        """Test summary string generation."""
        result = BacktestResult(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            duration_days=30,
            total_trades=100,
            net_pnl=500.0,
            roi_pct=25.0,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
        )

        summary = result.summary()

        assert "Net PnL" in summary
        assert "500" in summary
        assert "Sharpe" in summary


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create Monte Carlo simulator."""
        config = BacktestConfig(initial_capital=2000.0, leverage=25)
        return MonteCarloSimulator(config)

    @pytest.fixture
    def base_result(self):
        """Create base backtest result."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.001, 1000)

        return BacktestResult(
            returns=returns,
            equity_curve=np.cumprod(1 + returns) * 2000,
        )

    def test_run_monte_carlo(self, simulator, base_result):
        """Test Monte Carlo simulation."""
        results = simulator.run(
            base_result,
            num_simulations=100,  # Small number for testing
            horizon_days=7,
        )

        assert "liquidation_probability" in results
        assert "expected_return" in results
        assert "probability_of_profit" in results

        # Probabilities should be between 0 and 1
        assert 0 <= results["liquidation_probability"] <= 1
        assert 0 <= results["probability_of_profit"] <= 1

    def test_monte_carlo_percentiles(self, simulator, base_result):
        """Test that percentiles are calculated."""
        results = simulator.run(
            base_result,
            num_simulations=100,
            horizon_days=7,
        )

        assert "baseline" in results
        assert "return_percentiles" in results["baseline"]
        assert "5th" in results["baseline"]["return_percentiles"]
        assert "95th" in results["baseline"]["return_percentiles"]

        # 5th percentile should be less than or equal to 95th
        assert results["baseline"]["return_percentiles"]["5th"] <= results["baseline"]["return_percentiles"]["95th"]


class TestBacktestIntegration:
    """Integration tests for backtesting."""

    def test_full_backtest_flow(self):
        """Test complete backtest flow."""
        # Generate data
        loader = MarketDataLoader()
        data = loader.generate_synthetic_data(
            days=7,
            interval_minutes=1,
            seed=42,
        )

        # Run backtest
        config = BacktestConfig(
            initial_capital=2000.0,
            leverage=10,
            min_spread_bps=10.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(data)

        # Verify result
        assert result.duration_days == pytest.approx(7, rel=0.1)
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0

        # Run Monte Carlo
        mc = MonteCarloSimulator(config)
        mc_results = mc.run(result, num_simulations=100, horizon_days=7)

        assert mc_results is not None
        assert "liquidation_probability" in mc_results

    def test_backtest_preserves_capital_bounds(self):
        """Test that equity doesn't go negative."""
        loader = MarketDataLoader()
        data = loader.generate_synthetic_data(
            days=30,
            volatility=0.05,  # High volatility
            seed=42,
        )

        config = BacktestConfig(
            initial_capital=2000.0,
            leverage=25,
        )
        engine = BacktestEngine(config)
        result = engine.run(data)

        # Equity should never be negative
        assert all(e >= 0 for e in result.equity_curve)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
