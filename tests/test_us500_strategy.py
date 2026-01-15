"""
TESTS FOR US500-USDH PROFESSIONAL MM STRATEGY
/Users/nheosdisplay/VSC/AMM/AMM-500/tests/test_us500_strategy.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.strategy_us500_pro import US500ProfessionalMM, BookDepthAnalysis, InventoryState
from src.config import Config
from src.exchange import HyperliquidClient, OrderBook, OrderSide
from src.risk import RiskManager, RiskMetrics, RiskLevel


@pytest.fixture
def config():
    """Mock config for testing."""
    config = Mock(spec=Config)
    config.trading = Mock()
    config.trading.symbol = "US500"
    config.trading.min_spread_bps = 1.0
    config.trading.max_spread_bps = 50.0
    config.trading.order_levels = 15
    config.trading.leverage = 10
    config.trading.collateral = 1000.0
    config.execution = Mock()
    config.execution.quote_refresh_interval = 1.0
    config.execution.rebalance_interval = 1.0
    config.wallet_address = "0xtest"
    return config


@pytest.fixture
def mock_client():
    """Mock Hyperliquid client."""
    client = Mock(spec=HyperliquidClient)
    client.get_orderbook = AsyncMock()
    client.get_position = AsyncMock()
    client.get_account_state = AsyncMock()
    client.get_usdh_margin_state = AsyncMock()
    client.place_order = AsyncMock()
    client.cancel_all_orders = AsyncMock(return_value=0)
    client.batch_cancel_orders = AsyncMock()
    client.get_funding_rate = AsyncMock(return_value=0.0001)
    client.on_orderbook_update = Mock()
    client.on_user_update = Mock()
    return client


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    rm = Mock(spec=RiskManager)
    rm.initialize = AsyncMock()
    rm.assess_risk = AsyncMock()
    rm.calculate_order_size = Mock(return_value=1.0)
    return rm


@pytest.fixture
def strategy(config, mock_client, mock_risk_manager):
    """Create strategy instance."""
    return US500ProfessionalMM(config, mock_client, mock_risk_manager)


@pytest.fixture
def mock_orderbook():
    """Mock order book."""
    return OrderBook(
        symbol="US500",
        bids=[(6900, 10), (6899, 15), (6898, 20)],
        asks=[(6901, 10), (6902, 15), (6903, 20)],
        best_bid=6900,
        best_ask=6901,
        best_bid_size=10,
        best_ask_size=10,
        timestamp=1234567890
    )


# =============================================================================
# L2 BOOK ANALYSIS TESTS
# =============================================================================

def test_analyze_order_book(strategy, mock_orderbook):
    """Test L2 order book analysis."""
    mock_orderbook.mid_price = 6900.5
    
    analysis = strategy._analyze_order_book(mock_orderbook)
    
    assert isinstance(analysis, BookDepthAnalysis)
    assert analysis.total_bid_depth > 0
    assert analysis.total_ask_depth > 0
    assert -1 <= analysis.imbalance <= 1
    assert analysis.weighted_mid > 0


def test_book_liquidity_check(strategy, mock_orderbook):
    """Test book liquidity threshold."""
    # Deep book - should be liquid
    mock_orderbook.bids = [(6900 - i, 100) for i in range(10)]
    mock_orderbook.asks = [(6901 + i, 100) for i in range(10)]
    
    analysis = strategy._analyze_order_book(mock_orderbook)
    assert analysis.is_liquid, "Deep book should be liquid"
    
    # Thin book - should be illiquid
    mock_orderbook.bids = [(6900, 1)]
    mock_orderbook.asks = [(6901, 1)]
    
    analysis = strategy._analyze_order_book(mock_orderbook)
    assert not analysis.is_liquid, "Thin book should be illiquid"


# =============================================================================
# SPREAD CALCULATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_calculate_spread_low_vol(strategy, mock_orderbook):
    """Test spread calculation in low volatility."""
    # Simulate low volatility
    prices = [6900 + i * 0.1 for i in range(100)]
    for p in prices:
        strategy.price_buffer.append(p)
    
    risk_metrics = RiskMetrics(risk_level=RiskLevel.LOW)
    
    min_spread, max_spread = strategy._calculate_spread(mock_orderbook, risk_metrics)
    
    assert min_spread >= 1.0, "Min spread should be at least 1 bps"
    assert max_spread <= 50.0, "Max spread should not exceed 50 bps"
    assert max_spread > min_spread, "Max spread should exceed min spread"


@pytest.mark.asyncio
async def test_calculate_spread_high_vol(strategy, mock_orderbook):
    """Test spread calculation in high volatility."""
    # Simulate high volatility
    prices = []
    for i in range(100):
        prices.append(6900 + np.random.randn() * 50)
    
    for p in prices:
        strategy.price_buffer.append(p)
    
    risk_metrics = RiskMetrics(risk_level=RiskLevel.HIGH)
    
    min_spread, max_spread = strategy._calculate_spread(mock_orderbook, risk_metrics)
    
    assert min_spread > 2.0, "High vol should have wider min spread"
    assert max_spread > 20.0, "High vol should have wider max spread"


@pytest.mark.asyncio
async def test_adverse_selection_widens_spread(strategy, mock_orderbook):
    """Test that adverse selection detection widens spreads."""
    # Simulate losing fills
    for i in range(10):
        strategy.metrics.add_fill(OrderSide.BUY, 6900, 1.0)
        strategy.metrics.add_fill(OrderSide.SELL, 6895, 1.0)  # Selling lower than buying
    
    risk_metrics = RiskMetrics(risk_level=RiskLevel.LOW)
    
    min_spread, max_spread = strategy._calculate_spread(mock_orderbook, risk_metrics)
    
    # Should widen due to adverse selection
    assert min_spread > 3.0, "Adverse selection should widen spread"


# =============================================================================
# INVENTORY SKEWING TESTS
# =============================================================================

def test_inventory_skew_long(strategy):
    """Test inventory skew when long."""
    strategy.inventory.delta = 0.03  # Long 3%
    
    bid_skew, ask_skew = strategy._calculate_inventory_skew()
    
    assert bid_skew > 1.0, "Should widen bids when long"
    assert ask_skew < 1.0, "Should tighten asks when long"


def test_inventory_skew_short(strategy):
    """Test inventory skew when short."""
    strategy.inventory.delta = -0.03  # Short 3%
    
    bid_skew, ask_skew = strategy._calculate_inventory_skew()
    
    assert bid_skew < 1.0, "Should tighten bids when short"
    assert ask_skew > 1.0, "Should widen asks when short"


def test_inventory_skew_usdh_margin(strategy):
    """Test increased skew with high USDH margin."""
    strategy.inventory.delta = 0.02
    strategy.inventory.usdh_margin_ratio = 0.85  # High margin
    
    bid_skew, ask_skew = strategy._calculate_inventory_skew()
    
    # Should have stronger skew due to high margin
    assert bid_skew > 1.5, "High margin should increase skew urgency"


# =============================================================================
# TIERED QUOTES TESTS
# =============================================================================

def test_build_tiered_quotes(strategy, mock_orderbook):
    """Test exponential tiered quote building."""
    mock_orderbook.mid_price = 6900.5
    
    bids, asks = strategy._build_tiered_quotes(
        mock_orderbook,
        min_spread_bps=2.0,
        max_spread_bps=20.0,
        total_size=10.0
    )
    
    assert len(bids) > 0, "Should generate bid levels"
    assert len(asks) > 0, "Should generate ask levels"
    
    # Check exponential spacing
    if len(bids) >= 2:
        gap1 = bids[0].price - bids[1].price
        gap2 = bids[1].price - bids[2].price if len(bids) >= 3 else gap1
        assert gap2 >= gap1 * 0.9, "Gaps should increase exponentially"
    
    # Check size concentration (top levels should be larger)
    if len(bids) >= 5:
        top_5_size = sum(b.size for b in bids[:5])
        total_bid_size = sum(b.size for b in bids)
        assert top_5_size / total_bid_size > 0.6, "Top 5 should have >60% of volume"


def test_tiered_quotes_lot_size(strategy, mock_orderbook):
    """Test that all quote sizes meet minimum lot size (0.1 for US500)."""
    mock_orderbook.mid_price = 6900.5
    
    bids, asks = strategy._build_tiered_quotes(
        mock_orderbook,
        min_spread_bps=2.0,
        max_spread_bps=20.0,
        total_size=1.0
    )
    
    for bid in bids:
        assert bid.size >= 0.1, f"Bid size {bid.size} below minimum 0.1"
    
    for ask in asks:
        assert ask.size >= 0.1, f"Ask size {ask.size} below minimum 0.1"


# =============================================================================
# DELTA REBALANCING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_check_rebalance_triggers(strategy, mock_orderbook, mock_client):
    """Test delta rebalance triggers at 1.5% threshold."""
    # Below threshold - no rebalance
    strategy.inventory.delta = 0.01
    mock_client.get_orderbook.return_value = mock_orderbook
    
    await strategy._check_rebalance()
    
    assert mock_client.place_ioc_order.call_count == 0, "Should not rebalance at 1%"
    
    # Above threshold - should rebalance
    strategy.inventory.delta = 0.02
    strategy.inventory.position_size = 10.0
    
    await strategy._check_rebalance()
    
    assert mock_client.place_ioc_order.call_count == 1, "Should rebalance at 2%"


@pytest.mark.asyncio
async def test_rebalance_cancels_quotes(strategy, mock_orderbook, mock_client):
    """Test that rebalancing cancels active quotes."""
    strategy.inventory.delta = 0.02
    strategy.inventory.position_size = 10.0
    strategy.active_bids = {"oid1": Mock(), "oid2": Mock()}
    
    mock_client.get_orderbook.return_value = mock_orderbook
    
    await strategy._check_rebalance()
    
    assert len(strategy.active_bids) == 0, "Should clear active bids after rebalance"


# =============================================================================
# USDH MARGIN TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_usdh_margin_refresh(strategy, mock_client):
    """Test USDH margin state refresh."""
    mock_client.get_usdh_margin_state.return_value = {
        "margin_used": 500.0,
        "margin_ratio": 0.75,
        "margin_available": 500.0
    }
    
    position_mock = Mock()
    position_mock.size = 10.0
    position_mock.notional_value = 69000.0
    position_mock.entry_price = 6900.0
    position_mock.mark_price = 6900.0
    position_mock.unrealized_pnl = 0.0
    
    mock_client.get_position.return_value = position_mock
    mock_client.get_account_state.return_value = Mock()
    
    await strategy._refresh_inventory()
    
    assert strategy.inventory.usdh_margin_used == 500.0
    assert strategy.inventory.usdh_margin_ratio == 0.75


@pytest.mark.asyncio
async def test_one_sided_quoting_extreme_delta(strategy, mock_orderbook, mock_client):
    """Test one-sided quoting at extreme imbalance (>2.5%)."""
    strategy.inventory.delta = 0.03  # 3% long
    
    mock_client.get_orderbook.return_value = mock_orderbook
    mock_client.cancel_all_orders.return_value = 0
    
    risk_metrics = RiskMetrics(risk_level=RiskLevel.LOW)
    
    # Mock risk manager
    strategy.risk_manager.calculate_order_size.return_value = 1.0
    
    await strategy._update_quotes(risk_metrics)
    
    # Should only have asks (no bids when long)
    assert len(strategy.active_asks) > 0 or mock_client.place_order.call_count > 0
    # Bids should be cancelled
    assert mock_client.cancel_all_orders.called or len(strategy.active_bids) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_full_iteration_cycle(strategy, mock_orderbook, mock_client, mock_risk_manager):
    """Test full strategy iteration cycle."""
    # Setup mocks
    mock_client.get_orderbook.return_value = mock_orderbook
    mock_orderbook.mid_price = 6900.5
    
    position_mock = Mock()
    position_mock.size = 0.0
    position_mock.notional_value = 0.0
    position_mock.entry_price = 0.0
    position_mock.mark_price = 6900.0
    position_mock.unrealized_pnl = 0.0
    mock_client.get_position.return_value = position_mock
    
    mock_client.get_account_state.return_value = Mock()
    mock_client.get_usdh_margin_state.return_value = {
        "margin_used": 100.0,
        "margin_ratio": 0.10,
        "margin_available": 900.0
    }
    
    risk_metrics = RiskMetrics(risk_level=RiskLevel.LOW)
    mock_risk_manager.assess_risk.return_value = risk_metrics
    mock_risk_manager.calculate_order_size.return_value = 1.0
    
    # Start strategy
    await strategy.start()
    
    assert strategy.state.value == "running"
    
    # Run one iteration
    await strategy.run_iteration()
    
    # Should have attempted to update quotes
    assert mock_client.get_orderbook.called


def test_metrics_tracking(strategy):
    """Test performance metrics tracking."""
    # Simulate fills
    strategy.metrics.add_fill(OrderSide.BUY, 6900, 1.0)
    strategy.metrics.add_fill(OrderSide.SELL, 6905, 1.0)
    
    # Check spread calculation
    spread = strategy.metrics.get_weighted_spread_bps()
    
    assert spread is not None
    assert spread > 0, "Spread should be positive when selling higher"


@pytest.mark.asyncio
async def test_emergency_close(strategy, mock_client):
    """Test emergency close procedure."""
    strategy.inventory.position_size = 10.0
    
    mock_orderbook = OrderBook(
        symbol="US500",
        bids=[(6900, 10)],
        asks=[(6901, 10)],
        best_bid=6900,
        best_ask=6901,
        best_bid_size=10,
        best_ask_size=10,
        timestamp=123
    )
    mock_client.get_orderbook.return_value = mock_orderbook
    mock_client.place_market_order = AsyncMock()
    
    await strategy._emergency_close()
    
    assert mock_client.place_market_order.called
    assert strategy.state.value == "paused"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
