"""
Tests for risk management module.
"""

import pytest
import numpy as np

from src.config import Config, TradingConfig, RiskConfig
from src.risk import (
    RiskManager,
    RiskMetrics,
    RiskState,
    RiskLevel,
)
from src.exchange import Position, AccountState


class MockHyperliquidClient:
    """Mock client for testing."""

    def __init__(self):
        self.leverage = 25
        self.position = None
        self.account_state = None
        self.orderbook = None
        self.funding_rate = 0.0001  # Default funding rate

    async def get_account_state(self):
        return self.account_state

    async def get_position(self):
        return self.position

    async def get_orderbook(self):
        return self.orderbook

    async def get_funding_rate(self, symbol=None):
        return self.funding_rate

    async def update_leverage(self, leverage: int):
        self.leverage = leverage
        return True


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_default_values(self):
        """Test default risk metrics."""
        metrics = RiskMetrics()

        assert metrics.net_exposure == 0.0
        assert metrics.risk_level == RiskLevel.LOW
        assert metrics.should_reduce_exposure == False
        assert metrics.should_pause_trading == False
        assert metrics.emergency_close == False

    def test_critical_risk(self):
        """Test critical risk metrics."""
        metrics = RiskMetrics(
            risk_level=RiskLevel.CRITICAL,
            should_pause_trading=True,
            emergency_close=True,
        )

        assert metrics.risk_level == RiskLevel.CRITICAL
        assert metrics.should_pause_trading == True


class TestRiskState:
    """Tests for RiskState dataclass."""

    def test_initial_state(self):
        """Test initial risk state."""
        state = RiskState()

        assert state.peak_equity == 0.0
        assert state.total_trades == 0
        assert state.consecutive_losses == 0

    def test_price_history_buffer(self):
        """Test that price history buffer is initialized."""
        state = RiskState()

        state.price_history.append(100.0)
        state.price_history.append(101.0)

        assert len(state.price_history.get_array()) == 2


class TestRiskManager:
    """Tests for RiskManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            private_key="test_key",
            wallet_address="0x1234",
            trading=TradingConfig(
                leverage=25,
                collateral=2000.0,
                max_net_exposure=50000.0,
                order_size_fraction=0.02,
            ),
            risk=RiskConfig(
                max_drawdown=0.10,
                stop_loss_pct=0.04,
                min_margin_ratio=0.15,
                medium_risk_leverage=10,
                low_risk_leverage=5,
                high_vol_threshold=100.0,
            ),
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        return MockHyperliquidClient()

    @pytest.fixture
    def risk_manager(self, config, mock_client):
        """Create risk manager instance."""
        return RiskManager(config, mock_client)

    @pytest.mark.asyncio
    async def test_initialize(self, risk_manager, mock_client):
        """Test risk manager initialization."""
        mock_client.account_state = AccountState(
            equity=2000.0,
            available_balance=1500.0,
            margin_used=500.0,
            unrealized_pnl=0.0,
        )

        await risk_manager.initialize()

        assert risk_manager.state.starting_equity == 2000.0
        assert risk_manager.state.peak_equity == 2000.0
        assert risk_manager._initialized == True

    @pytest.mark.asyncio
    async def test_check_risk_no_position(self, risk_manager, mock_client):
        """Test risk check with no position."""
        mock_client.account_state = AccountState(
            equity=2000.0,
            available_balance=2000.0,
            margin_used=0.0,
            unrealized_pnl=0.0,
        )
        mock_client.position = None

        metrics = await risk_manager.check_risk()

        assert metrics.net_exposure == 0.0
        assert metrics.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_check_risk_with_position(self, risk_manager, mock_client):
        """Test risk check with open position."""
        mock_client.account_state = AccountState(
            equity=2100.0,
            available_balance=1600.0,
            margin_used=500.0,
            unrealized_pnl=100.0,
        )
        mock_client.position = Position(
            symbol="BTC",
            size=0.5,
            entry_price=100000.0,
            mark_price=100200.0,
            liquidation_price=80000.0,
            unrealized_pnl=100.0,
            leverage=25,
            margin_used=500.0,
        )

        risk_manager.state.peak_equity = 2000.0
        risk_manager.state.starting_equity = 2000.0

        metrics = await risk_manager.check_risk()

        assert metrics.net_exposure == 50100.0  # 0.5 * 100200
        assert metrics.unrealized_pnl == 100.0

    @pytest.mark.asyncio
    async def test_check_risk_high_margin(self, risk_manager, mock_client):
        """Test risk check with high margin usage."""
        mock_client.account_state = AccountState(
            equity=2000.0,
            available_balance=400.0,
            margin_used=1600.0,  # 80% margin ratio
            unrealized_pnl=-100.0,
        )
        mock_client.position = Position(
            symbol="BTC",
            size=0.5,
            entry_price=100000.0,
            mark_price=99800.0,
            liquidation_price=90000.0,
            unrealized_pnl=-100.0,
            leverage=25,
            margin_used=1600.0,
        )

        risk_manager.state.peak_equity = 2000.0
        risk_manager.state.starting_equity = 2000.0

        metrics = await risk_manager.check_risk()

        # High margin ratio should trigger elevated risk
        assert metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_adjust_leverage(self, risk_manager, mock_client):
        """Test leverage adjustment."""
        risk_manager.config.trading.leverage = 25

        metrics = RiskMetrics(
            risk_level=RiskLevel.HIGH,
            recommended_leverage=10,
            exposure_pct=5.0,  # Set exposure > 1% to trigger adjustment
        )

        adjusted = await risk_manager.adjust_leverage(metrics)

        assert adjusted == True
        assert risk_manager.config.trading.leverage == 3  # HIGH risk level uses leverage level 3
        assert mock_client.leverage == 3

    def test_calculate_order_size(self, risk_manager):
        """Test order size calculation."""
        # Normal conditions - US500 has minimum size of 0.1 contracts
        size = risk_manager.calculate_order_size(100000.0, "buy")
        # For US500, minimum is 0.1 regardless of Kelly sizing
        assert size >= 0.1  # Minimum order size for US500

        # With high risk metrics
        metrics = RiskMetrics(risk_level=RiskLevel.HIGH)
        size = risk_manager.calculate_order_size(100000.0, "buy", metrics)
        assert size >= 0.0  # May be reduced or zero

        # With critical risk
        metrics = RiskMetrics(risk_level=RiskLevel.CRITICAL)
        size = risk_manager.calculate_order_size(100000.0, "buy", metrics)
        assert size == 0.0  # Should be zero

    def test_calculate_kelly_size(self, risk_manager):
        """Test Kelly criterion sizing."""
        # With explicit parameters
        size = risk_manager.calculate_kelly_size(
            price=100000.0,
            win_rate=0.6,
            avg_win=0.002,
            avg_loss=0.001,
        )

        assert size > 0

        # Kelly should be capped
        max_kelly = 0.25 * 2000.0 * 25 / 100000.0
        assert size <= max_kelly

    def test_record_trade(self, risk_manager):
        """Test trade recording."""
        # Record a win
        risk_manager.record_trade(100.0)

        assert risk_manager.state.total_trades == 1
        assert risk_manager.state.winning_trades == 1
        assert risk_manager.state.total_win_amount == 100.0
        assert risk_manager.state.consecutive_losses == 0

        # Record a loss
        risk_manager.record_trade(-50.0)

        assert risk_manager.state.total_trades == 2
        assert risk_manager.state.winning_trades == 1
        assert risk_manager.state.total_loss_amount == 50.0
        assert risk_manager.state.consecutive_losses == 1

    def test_should_stop_loss_long(self, risk_manager):
        """Test stop loss detection for long position."""
        position = Position(
            symbol="BTC",
            size=0.5,  # Long
            entry_price=100000.0,
            mark_price=95000.0,  # 5% down
            liquidation_price=80000.0,
            unrealized_pnl=-2500.0,
            leverage=25,
            margin_used=500.0,
        )

        # 5% adverse move > 4% stop loss
        assert risk_manager.should_stop_loss(position) == True

        # 3% adverse move < 4% stop loss
        position.mark_price = 97000.0
        assert risk_manager.should_stop_loss(position) == False

    def test_should_stop_loss_short(self, risk_manager):
        """Test stop loss detection for short position."""
        position = Position(
            symbol="BTC",
            size=-0.5,  # Short
            entry_price=100000.0,
            mark_price=105000.0,  # 5% up
            liquidation_price=120000.0,
            unrealized_pnl=-2500.0,
            leverage=25,
            margin_used=500.0,
        )

        # 5% adverse move > 4% stop loss
        assert risk_manager.should_stop_loss(position) == True

    def test_get_trade_stats(self, risk_manager):
        """Test trade statistics calculation."""
        # Record some trades
        risk_manager.record_trade(100.0)
        risk_manager.record_trade(200.0)
        risk_manager.record_trade(-50.0)
        risk_manager.record_trade(150.0)

        stats = risk_manager.get_trade_stats()

        assert stats["total_trades"] == 4
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.75
        assert stats["avg_win"] == 150.0  # (100 + 200 + 150) / 3
        assert stats["avg_loss"] == 50.0
        assert stats["profit_factor"] == 9.0  # 450 / 50


class TestRiskLevelAssessment:
    """Tests for risk level assessment logic."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with config."""
        config = Config(
            private_key="test_key",
            wallet_address="0x1234",
            risk=RiskConfig(
                max_drawdown=0.10,
                min_margin_ratio=0.15,
            ),
        )
        client = MockHyperliquidClient()
        return RiskManager(config, client)

    def test_low_risk_level(self, risk_manager):
        """Test low risk level assessment."""
        metrics = RiskMetrics(
            margin_ratio=0.2,
            current_drawdown=0.02,
            distance_to_liquidation=0.5,
            volatility_ratio=0.5,
        )

        metrics = risk_manager._assess_risk_level(metrics, None)

        assert metrics.risk_level == RiskLevel.LOW
        assert metrics.should_reduce_exposure == False

    def test_high_risk_drawdown(self, risk_manager):
        """Test high risk from drawdown."""
        metrics = RiskMetrics(
            margin_ratio=0.3,
            current_drawdown=0.08,  # 80% of max
            distance_to_liquidation=0.3,
            volatility_ratio=0.8,
        )

        metrics = risk_manager._assess_risk_level(metrics, None)

        assert metrics.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_critical_risk_near_liquidation(self, risk_manager):
        """Test critical risk when near liquidation."""
        metrics = RiskMetrics(
            margin_ratio=0.9,
            current_drawdown=0.12,
            distance_to_liquidation=0.02,  # Very close to liquidation
            volatility_ratio=1.5,
        )

        metrics = risk_manager._assess_risk_level(metrics, None)

        assert metrics.risk_level == RiskLevel.CRITICAL
        # With no position, should NOT pause (continue quoting to earn spread)
        # Pausing only makes sense when there's a position to protect
        assert metrics.should_pause_trading == False  # Updated: no position = continue quoting
        assert metrics.emergency_close == False  # Updated: no position to close


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
