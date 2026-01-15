"""
Professional Market Making Strategy for Hyperliquid
Transform from grid-based to professional HFT market making with:
- Real-time L2 order book integration for dynamic quoting
- Adaptive spread/sizing based on book depth and inventory
- Volatility-adaptive exponential tiering (1-50 bps)
- Quote fading on adverse selection detection
- Inventory skew management for delta-neutral operation
- Optimized for 1s quote refresh on Apple M4 hardware

WARNING: High-frequency trading with leverage carries significant financial risk.
Thoroughly test on testnet before using real funds.
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from .config import Config
from .exchange import (
    HyperliquidClient,
    Order,
    OrderBook,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)
from .risk import RiskManager, RiskMetrics, RiskLevel
from .utils import (
    calculate_imbalance,
    calculate_microprice,
    calculate_realized_volatility,
    round_price,
    round_size,
    CircularBuffer,
    get_timestamp_ms,
)

# Optional PyTorch integration for volatility/spread prediction
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - ML predictions disabled")


class StrategyState(Enum):
    """Strategy state machine states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class QuoteLevel:
    """A single quote level (bid or ask)."""

    price: float
    size: float
    side: OrderSide
    order_id: Optional[str] = None
    created_at: float = 0.0  # Timestamp when order was placed


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""

    quotes_sent: int = 0
    quotes_filled: int = 0
    quotes_cancelled: int = 0
    total_volume: float = 0.0
    maker_volume: float = 0.0
    taker_volume: float = 0.0
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    net_pnl: float = 0.0
    inventory_pnl: float = 0.0
    spread_capture: float = 0.0
    actions_today: int = 0
    last_reset: int = 0

    # Professional MM: Adverse selection tracking
    consecutive_losing_fills: int = 0  # Track consecutive losses for quote fading

    # OPT#14: Track recent fill prices AND SIZES for weighted adverse selection detection
    recent_buy_prices: List[float] = field(default_factory=list)
    recent_sell_prices: List[float] = field(default_factory=list)
    recent_buy_sizes: List[float] = field(default_factory=list)  # NEW: track sizes
    recent_sell_sizes: List[float] = field(default_factory=list)  # NEW: track sizes
    recent_fill_times: List[float] = field(default_factory=list)  # NEW: track timestamps
    max_recent_fills: int = 10  # REDUCED from 20 to 10 for faster adaptation
    fill_data_max_age: float = 300.0  # NEW: 5 minutes max age for fill data

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        if self.quotes_sent == 0:
            return 0.0
        return self.quotes_filled / self.quotes_sent

    @property
    def avg_spread_capture(self) -> float:
        """Average spread captured per fill."""
        if self.quotes_filled == 0:
            return 0.0
        return self.spread_capture / self.quotes_filled

    def get_recent_spread_bps(self) -> Optional[float]:
        """Calculate WEIGHTED spread from recent fills. Returns None if insufficient data."""
        # IMPROVED: Need both buys AND sells for valid spread calculation
        if len(self.recent_buy_prices) < 3 or len(self.recent_sell_prices) < 3:
            return None  # Not enough data - use default spread
        
        # Prune old fill data (older than 5 minutes)
        import time
        now = time.time()
        if hasattr(self, 'recent_fill_times') and self.recent_fill_times:
            # Keep only fills from last 5 minutes
            while self.recent_fill_times and (now - self.recent_fill_times[0]) > self.fill_data_max_age:
                self.recent_fill_times.pop(0)
                if self.recent_buy_prices:
                    self.recent_buy_prices.pop(0)
                    self.recent_buy_sizes.pop(0)
                if self.recent_sell_prices:
                    self.recent_sell_prices.pop(0)
                    self.recent_sell_sizes.pop(0)
        
        # Re-check after pruning
        if len(self.recent_buy_prices) < 3 or len(self.recent_sell_prices) < 3:
            return None  # Data too stale - use default spread

        # NEW: Weighted average by size (more accurate for market making)
        buy_sum = sum(p * s for p, s in zip(self.recent_buy_prices, self.recent_buy_sizes))
        buy_size_total = sum(self.recent_buy_sizes)

        sell_sum = sum(p * s for p, s in zip(self.recent_sell_prices, self.recent_sell_sizes))
        sell_size_total = sum(self.recent_sell_sizes)

        if buy_size_total == 0 or sell_size_total == 0:
            return None

        avg_buy = buy_sum / buy_size_total
        avg_sell = sell_sum / sell_size_total
        mid = (avg_buy + avg_sell) / 2

        if mid == 0:
            return None

        spread_bps = (avg_sell - avg_buy) / mid * 10000
        return spread_bps

    def add_fill(self, side: OrderSide, price: float, size: float = 0.0) -> None:
        """Track a fill for adverse selection detection (with size for weighted average)."""
        import time
        now = time.time()
        
        # Initialize fill times list if needed
        if not hasattr(self, 'recent_fill_times'):
            self.recent_fill_times = []
        
        self.recent_fill_times.append(now)
        if len(self.recent_fill_times) > self.max_recent_fills * 2:
            self.recent_fill_times.pop(0)
        
        if side == OrderSide.BUY:
            self.recent_buy_prices.append(price)
            self.recent_buy_sizes.append(
                size if size > 0 else 0.0001
            )  # Default tiny size if not provided
            if len(self.recent_buy_prices) > self.max_recent_fills:
                self.recent_buy_prices.pop(0)
                self.recent_buy_sizes.pop(0)
        else:
            self.recent_sell_prices.append(price)
            self.recent_sell_sizes.append(size if size > 0 else 0.0001)
            if len(self.recent_sell_prices) > self.max_recent_fills:
                self.recent_sell_prices.pop(0)
                self.recent_sell_sizes.pop(0)




@dataclass
class InventoryState:
    """Current inventory state with delta-neutral tracking."""

    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    delta: float = 0.0  # Directional exposure

    @property
    def is_balanced(self) -> bool:
        """Check if inventory is within acceptable range (±1.5% for HFT)."""
        return abs(self.delta) < 0.015  # Tighter tolerance for professional MM
    
    @property
    def skew_urgency(self) -> float:
        """Return urgency factor (0-1) for inventory rebalancing."""
        return min(abs(self.delta) / 0.05, 1.0)  # Max urgency at 5%


@dataclass
class BookDepthAnalysis:
    """L2 order book depth analysis for professional market making."""
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0
    imbalance: float = 0.0  # -1 to +1, positive = more bids
    weighted_mid: float = 0.0  # Microprice
    top_5_bid_depth: float = 0.0
    top_5_ask_depth: float = 0.0
    book_pressure: float = 0.0  # Directional pressure
    
    @property
    def is_liquid(self) -> bool:
        """Check if book has sufficient liquidity for quoting."""
        # US500: Lower threshold to $5000 per side (more realistic for index)
        return self.total_bid_depth > 5000 and self.total_ask_depth > 5000


class MarketMakingStrategy:
    """
    Professional Market Making Strategy with L2 Order Book Integration.
    
    TRANSFORMED from grid-based to professional HFT market making:
    - Real-time L2 order book analysis for dynamic quoting
    - Exponentially tiered spreads (1-50 bps, concentrated near mid)
    - Adaptive sizing based on inventory skew and book depth
    - Volatility-adaptive spreads with ML predictions (optional)
    - Quote fading on adverse selection detection
    - Delta-neutral inventory management (±1.5% tolerance)
    - Optimized for 1s quote refresh on Apple M4 hardware
    
    Quote Placement Philosophy:
    - Top 3-5 levels: 1-5 bps, 70% of volume (tight for high fill rate)
    - Middle 3-5 levels: 5-15 bps, 20% of volume
    - Outer 3-5 levels: 15-50 bps, 10% of volume (tail risk capture)
    
    Features:
    - Post-only orders to earn maker rebates (0.003%)
    - L2-aware spread adaptation based on book imbalance
    - Inventory skew to rebalance positions
    - Funding rate awareness
    - Circuit breakers and quote fading
    
    Usage:
        config = Config.load()
        client = HyperliquidClient(config)
        risk_manager = RiskManager(config, client)
        strategy = MarketMakingStrategy(config, client, risk_manager)

        await strategy.start()
        # ... bot runs
        await strategy.stop()
    """

    # Fee structure (Hyperliquid)
    MAKER_REBATE = 0.00003  # 0.003% rebate
    TAKER_FEE = 0.00035  # 0.035% fee
    
    # Professional MM parameters
    QUOTE_REFRESH_INTERVAL = 1.0  # 1s for HFT
    MIN_ORDER_AGE_SECONDS = 5.0  # Cancel/replace orders older than 5s
    MAX_BID_ORDERS = 15  # Reduced from 100 for concentrated liquidity
    MAX_ASK_ORDERS = 15
    MIN_BOOK_DEPTH_USD = 5000  # Minimum liquidity per side ($5K for US500)
    ADVERSE_SELECTION_THRESHOLD = 3  # Consecutive losing fills trigger fading

    # Legacy parameters (maintained for compatibility)
    PRICE_TOLERANCE_PCT = 0.0025
    SIZE_TOLERANCE_PCT = 0.30
    SKIP_UPDATE_IF_ALL_MATCHED = True
    MIN_SECONDS_BETWEEN_UPDATES = 15.0
    MAX_PENDING_CANCELS_BEFORE_BATCH = 3

    def __init__(self, config: Config, client: HyperliquidClient, risk_manager: RiskManager):
        """Initialize the strategy."""
        self.config = config
        self.client = client
        self.risk_manager = risk_manager

        # State
        self.state = StrategyState.STOPPED
        self.metrics = StrategyMetrics()
        self.inventory = InventoryState()

        # Real account tracking (from REST API, not WebSocket fills)
        self.starting_equity: float = 1000.0  # HARDCODED: $1000 starting capital for performance tracking
        self.current_equity: float = 1000.0  # Current equity ($1000 + realized PnL from fills)
        self.unrealized_pnl: float = 0.0  # Unrealized PnL from open positions

        # API call optimization metrics
        self._api_call_metrics = {
            "orderbook_fetches": 0,
            "orderbook_cache_hits": 0,
            "inventory_refreshes": 0,
            "order_syncs": 0,
        }

        # Active quotes - OrderedDict for FIFO order management
        self.active_bids: OrderedDict[str, QuoteLevel] = OrderedDict()
        self.active_asks: OrderedDict[str, QuoteLevel] = OrderedDict()

        # Price tracking
        self.price_buffer = CircularBuffer(500)
        self.volatility_buffer = CircularBuffer(100)  # Track realized vol
        self.last_trade_price = 0.0
        self.funding_rate = 0.0
        self.last_orderbook: Optional[OrderBook] = None
        self.last_mid_price: float = 0.0
        self.last_book_analysis: Optional[BookDepthAnalysis] = None  # L2 analysis
        self._cached_orderbook: Optional[OrderBook] = None
        self._orderbook_cache_time: float = 0.0
        self._orderbook_cache_ttl: float = 0.5  # Cache orderbook for 0.5 seconds

        # Timing
        self.last_quote_time = 0.0
        self.last_rebalance_time = 0.0
        self.last_funding_check = 0.0
        self._last_inventory_refresh: float = 0.0

        # Configuration shortcuts
        self.symbol = config.trading.symbol
        self.min_spread_bps = config.trading.min_spread_bps
        self.max_spread_bps = config.trading.max_spread_bps
        self.order_levels = config.trading.order_levels
        self.quote_interval = config.execution.quote_refresh_interval
        self.rebalance_interval = config.execution.rebalance_interval

        # Register callbacks
        self.client.on_orderbook_update(self._on_orderbook_update)
        self.client.on_user_update(self._on_user_update)
        
        # Trade tracker for data logging and verification
        from .trade_tracker import get_tracker
        self.trade_tracker = get_tracker(config.wallet_address, self.symbol)

    async def _get_cached_orderbook(self) -> Optional[OrderBook]:
        """
        Get orderbook with caching to reduce API calls.

        Returns cached orderbook if it's fresh (< 0.5s old), otherwise fetches new one.
        """
        now = time.time()
        if (
            self._cached_orderbook
            and (now - self._orderbook_cache_time) < self._orderbook_cache_ttl
        ):
            self._api_call_metrics["orderbook_cache_hits"] += 1
            return self._cached_orderbook

        # Cache miss - fetch fresh orderbook
        orderbook = await self.client.get_orderbook()
        if orderbook:
            self._cached_orderbook = orderbook
            self._orderbook_cache_time = now
            self._api_call_metrics["orderbook_fetches"] += 1

        return orderbook

    async def start(self) -> None:
        """Start the strategy."""
        if self.state != StrategyState.STOPPED:
            logger.warning(f"Cannot start strategy in state: {self.state}")
            return

        logger.info("Starting Market Making Strategy...")
        self.state = StrategyState.STARTING

        try:
            # Initialize risk manager
            await self.risk_manager.initialize()

            # AGGRESSIVE STARTUP CLEANUP: Cancel ALL orders for this symbol
            # This is critical for HIP-3 perps where stale orders accumulate
            logger.info(f"Startup cleanup: Cancelling all {self.symbol} orders...")
            cancelled_count = await self.client.cancel_all_orders(self.symbol)
            logger.info(f"Startup cleanup: Cancelled {cancelled_count} orders")
            
            # Double-check by trying again (in case of race condition)
            cancelled_count_2 = await self.client.cancel_all_orders(self.symbol)
            if cancelled_count_2 > 0:
                logger.warning(f"Startup cleanup (2nd pass): Cancelled {cancelled_count_2} additional orders")
            
            self.active_bids.clear()
            self.active_asks.clear()
            logger.info("Cleared local order tracking on startup")

            # Load initial state
            await self._refresh_inventory()
            
            # Start trade tracking session with current equity
            self.trade_tracker.reset_session()
            self.trade_tracker.start_session(self.current_equity)
            logger.info(f"Trade tracker started with ${self.current_equity:.2f} equity")

            # Sync recent fills for OPT#14 adaptive mode
            await self._sync_recent_fills()

            # Get initial funding rate
            self.funding_rate = await self.client.get_funding_rate() or 0.0

            # Reset daily metrics
            self._reset_daily_metrics()

            self.state = StrategyState.RUNNING
            logger.info("Strategy started successfully")

        except Exception as e:
            logger.error(f"Failed to start strategy: {e}")
            self.state = StrategyState.ERROR
            raise

    async def stop(self) -> None:
        """Stop the strategy gracefully."""
        if self.state == StrategyState.STOPPED:
            return

        logger.info("Stopping strategy...")
        self.state = StrategyState.STOPPING

        try:
            # Cancel all orders
            await self.client.cancel_all_orders(self.symbol)
            self.active_bids.clear()
            self.active_asks.clear()

            self.state = StrategyState.STOPPED
            logger.info("Strategy stopped")

        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            self.state = StrategyState.ERROR

    async def pause(self) -> None:
        """Pause trading (cancel orders but keep running)."""
        if self.state != StrategyState.RUNNING:
            return

        logger.info("Pausing strategy...")
        self.state = StrategyState.PAUSED

        await self.client.cancel_all_orders(self.symbol)
        self.active_bids.clear()
        self.active_asks.clear()

    async def resume(self) -> None:
        """Resume trading after pause."""
        if self.state != StrategyState.PAUSED:
            return

        logger.info("Resuming strategy...")
        self.state = StrategyState.RUNNING

    async def _sync_active_orders(self) -> None:
        """
        Synchronize locally tracked active orders with exchange reality.

        For HIP-3 perps (km:US500), the standard openOrders API returns 0.
        We must use orderStatus by OID to verify individual orders.
        """
        now = time.time()

        # AGGRESSIVE SYNC: Check every 30 seconds if we think we have orders
        # This is critical for HIP-3 where orders get filled quickly
        time_since_last_sync = now - getattr(self, "_last_order_sync", 0)
        have_tracked_orders = len(self.active_bids) > 0 or len(self.active_asks) > 0
        should_sync = have_tracked_orders and time_since_last_sync > 30.0

        if not should_sync:
            return

        self._last_order_sync = now
        logger.debug("Performing order synchronization check")
        self._api_call_metrics["order_syncs"] += 1

        try:
            is_hip3 = self.symbol.upper() == 'US500'
            
            if is_hip3:
                # HIP-3: openOrders API doesn't work for km: perps
                # Use historicalOrders and filter for status='open' instead
                import requests
                
                try:
                    # IMPORTANT: When using API wallet, orders are placed ON BEHALF OF the main wallet
                    # So we must query the MAIN wallet (config.wallet_address), NOT the signing wallet
                    query_address = self.config.wallet_address
                    
                    logger.debug(f"Querying historicalOrders for main wallet: {query_address}")
                    
                    # Use historicalOrders since openOrders doesn't work for HIP-3
                    resp = requests.post("https://api.hyperliquid.xyz/info", json={
                        "type": "historicalOrders",
                        "user": query_address
                    }, timeout=10)
                    historical_orders = resp.json()
                    
                    # Filter for our symbol and open status
                    target_symbols = [f'km:{self.symbol}', self.symbol.upper()]
                    
                    # CRITICAL FIX: historicalOrders returns multiple records per order
                    # We must deduplicate by OID and only keep orders where LATEST status is 'open'
                    from collections import defaultdict
                    by_oid = defaultdict(list)
                    for o in historical_orders:
                        coin = o.get('order', {}).get('coin', '')
                        if coin in target_symbols:
                            oid = str(o.get('order', {}).get('oid', ''))
                            if oid:
                                by_oid[oid].append(o)
                    
                    # Find orders where latest status is 'open'
                    open_orders = []
                    for oid, records in by_oid.items():
                        records.sort(key=lambda x: x.get('statusTimestamp', 0), reverse=True)
                        latest = records[0]
                        if latest.get('status') == 'open':
                            open_orders.append(latest)
                    
                    # DEBUG: Log what historicalOrders returns
                    logger.debug(f"HIP-3 historicalOrders: {len(historical_orders)} total, {len(open_orders)} open for {self.symbol}")
                    
                    # Build set of actual open order IDs from historicalOrders format
                    # historicalOrders returns: {"order": {"oid": 123, "coin": "km:US500", ...}, "status": "open", ...}
                    exchange_order_ids = set()
                    for o in open_orders:
                        order = o.get('order', {})
                        oid = str(order.get('oid', ''))
                        if oid:
                            exchange_order_ids.add(oid)
                    
                    logger.debug(f"Found {len(exchange_order_ids)} open orders on exchange")
                    
                    # Find stale orders (tracked locally but not on exchange)
                    all_tracked_oids = set(self.active_bids.keys()) | set(self.active_asks.keys())
                    stale_orders = all_tracked_oids - exchange_order_ids
                    
                    if stale_orders:
                        logger.warning(f"HIP-3 order sync: Found {len(stale_orders)} phantom orders (tracked locally but not on exchange)")
                        logger.debug(f"Tracked OIDs: {all_tracked_oids}")
                        logger.debug(f"Exchange OIDs: {exchange_order_ids}")
                        for oid in stale_orders:
                            self.active_bids.pop(oid, None)
                            self.active_asks.pop(oid, None)
                        # IMPROVEMENT: If all orders are phantom, clear everything and start fresh
                        if len(stale_orders) == len(all_tracked_oids) and len(stale_orders) > 0:
                            logger.warning("All tracked orders were phantom - resetting fill metrics to avoid stale defensive mode")
                            self.metrics.recent_buy_prices.clear()
                            self.metrics.recent_sell_prices.clear()
                            self.metrics.recent_buy_sizes.clear()
                            self.metrics.recent_sell_sizes.clear()
                            if hasattr(self.metrics, 'recent_fill_times'):
                                self.metrics.recent_fill_times.clear()
                    
                    # Check for orphaned orders (on exchange but not tracked)
                    orphaned = exchange_order_ids - all_tracked_oids
                    if orphaned:
                        logger.warning(f"HIP-3 order sync: Found {len(orphaned)} orphaned orders (on exchange but not tracked locally) - CANCELLING")
                        # CRITICAL FIX: Cancel orphaned orders to prevent accumulation
                        # These are orders from previous bot sessions or tracking failures
                        # NOTE: For HIP-3 perps, use km: prefix in symbol
                        api_symbol = f"km:{self.symbol}" if self.symbol.upper() == "US500" else self.symbol
                        try:
                            orphan_cancels = [(api_symbol, oid) for oid in orphaned]
                            cancelled = await self.client.cancel_orders_batch(orphan_cancels)
                            logger.info(f"HIP-3 order sync: Cancelled {cancelled} orphaned orders")
                        except Exception as cancel_err:
                            logger.error(f"Failed to cancel orphaned orders: {cancel_err}")
                            # Fallback: cancel all orders for this symbol
                            await self.client.cancel_all_orders(self.symbol)
                            logger.info("Fallback: Cancelled all orders via cancel_all_orders")
                    
                    logger.debug(f"HIP-3 order sync complete: {len(self.active_bids)} bids, {len(self.active_asks)} asks tracked | {len(exchange_order_ids)} total on exchange")
                    return
                    
                except Exception as e:
                    logger.error(f"HIP-3 openOrders API failed: {e}, falling back to orderStatus checks")
                    # Fallback to orderStatus checks...
                    stale_orders = []
                    all_tracked_oids = list(self.active_bids.keys()) + list(self.active_asks.keys())
                    
                    for oid in all_tracked_oids:
                        try:
                            resp = requests.post("https://api.hyperliquid.xyz/info", json={
                                "type": "orderStatus",
                                "user": self.config.wallet_address,
                                "oid": int(oid)
                            }, timeout=5)
                            data = resp.json()
                            
                            if data.get("status") != "order":
                                stale_orders.append(oid)
                            else:
                                order = data.get("order", {}).get("order", {})
                                status = order.get("orderStatus", "")
                                if status in ["filled", "canceled", "cancelled"]:
                                    stale_orders.append(oid)
                        except Exception as e2:
                            logger.debug(f"Error checking order {oid}: {e2}")
                            # Assume stale if we can't verify
                            stale_orders.append(oid)
                    
                    if stale_orders:
                        logger.info(f"HIP-3 fallback sync: Removing {len(stale_orders)} unverifiable orders")
                        for oid in stale_orders:
                            self.active_bids.pop(oid, None)
                            self.active_asks.pop(oid, None)
                    
                    logger.debug(f"HIP-3 fallback sync complete: {len(self.active_bids)} bids, {len(self.active_asks)} asks")
                    return
            
            # Standard perps: Use open_orders API
            from hyperliquid.info import Info
            from hyperliquid.utils import constants as C

            info = Info(C.MAINNET_API_URL)
            open_orders = info.open_orders(self.config.wallet_address)

            if not open_orders:
                # No open orders on exchange
                if self.active_bids or self.active_asks:
                    logger.warning(
                        f"Order sync: Exchange has 0 open orders but locally tracking "
                        f"{len(self.active_bids)} bids + {len(self.active_asks)} asks. Clearing local tracking."
                    )
                    self.active_bids.clear()
                    self.active_asks.clear()
                return

            # Build set of actual open order IDs from exchange
            exchange_order_ids = set()
            for order in open_orders:
                coin = order.get("coin", "")
                symbol_match = coin == self.symbol.replace("/", "")
                if symbol_match:
                    exchange_order_ids.add(str(order.get("oid", "")))

            # Check for orders we're tracking locally but not on exchange
            local_order_ids = set(self.active_bids.keys()) | set(self.active_asks.keys())
            stale_orders = local_order_ids - exchange_order_ids

            if stale_orders:
                logger.warning(
                    f"Order sync: Found {len(stale_orders)} stale orders locally tracked "
                    f"but not on exchange: {list(stale_orders)[:5]}..."
                )

                # Remove stale orders from local tracking
                for oid in stale_orders:
                    if oid in self.active_bids:
                        del self.active_bids[oid]
                    if oid in self.active_asks:
                        del self.active_asks[oid]

                logger.info(
                    f"Order sync: Cleared {len(stale_orders)} stale orders. "
                    f"Active: {len(self.active_bids)} bids, {len(self.active_asks)} asks"
                )

            # Optional: Log summary periodically
            total_local = len(self.active_bids) + len(self.active_asks)
            total_exchange = len(exchange_order_ids)
            if total_local != total_exchange:
                logger.warning(
                    f"Order sync: Discrepancy detected - Local: {total_local}, Exchange: {total_exchange}"
                )

        except Exception as e:
            logger.error(f"Error during order synchronization: {e}")
            # Don't clear orders on sync failure - better to keep potentially stale orders
            # than to lose all tracking due to a temporary API issue

    async def _enforce_order_limits(self) -> None:
        """
        Enforce FIFO order limits - cancel oldest orders if limits exceeded.
        
        Limits: 100 bids max, 100 asks max (200 total)
        When limit reached, cancels oldest orders first (FIFO).
        """
        orders_to_cancel = []
        
        # Check bid orders
        if len(self.active_bids) > self.MAX_BID_ORDERS:
            excess_bids = len(self.active_bids) - self.MAX_BID_ORDERS
            logger.warning(f"Bid limit exceeded: {len(self.active_bids)}/{self.MAX_BID_ORDERS}, canceling {excess_bids} oldest bids")
            
            # Get oldest orders (OrderedDict maintains insertion order)
            oldest_bid_oids = list(self.active_bids.keys())[:excess_bids]
            for oid in oldest_bid_oids:
                orders_to_cancel.append((self.symbol, oid))
                del self.active_bids[oid]
        
        # Check ask orders
        if len(self.active_asks) > self.MAX_ASK_ORDERS:
            excess_asks = len(self.active_asks) - self.MAX_ASK_ORDERS
            logger.warning(f"Ask limit exceeded: {len(self.active_asks)}/{self.MAX_ASK_ORDERS}, canceling {excess_asks} oldest asks")
            
            # Get oldest orders (OrderedDict maintains insertion order)
            oldest_ask_oids = list(self.active_asks.keys())[:excess_asks]
            for oid in oldest_ask_oids:
                orders_to_cancel.append((self.symbol, oid))
                del self.active_asks[oid]
        
        # Cancel excess orders
        if orders_to_cancel:
            logger.info(f"FIFO enforcement: Canceling {len(orders_to_cancel)} oldest orders (limit: 100 bids + 100 asks)")
            await self.client.cancel_orders_batch(orders_to_cancel)
            self.metrics.quotes_cancelled += len(orders_to_cancel)

    async def _sync_recent_fills(self) -> None:
        """
        Sync recent fills from exchange to populate internal metrics.

        This ensures OPT#14 adaptive mode has data immediately on startup,
        rather than waiting for new fills to accumulate.
        """
        try:
            logger.debug("Syncing recent fills for OPT#14 metrics")

            # Get recent fills from exchange
            fills = await self.client.get_user_fills(limit=50)

            if not fills:
                logger.debug("No recent fills found")
                return

            # Target symbol for filtering (handle km: prefix for HIP-3 perps)
            symbol = self.config.trading.symbol.upper()
            # For US500, API returns "km:US500"
            target_coin = f"km:{symbol}" if symbol == "US500" else symbol

            # Add fills to metrics (this populates recent_buy_prices, recent_sell_prices)
            synced_count = 0
            for fill in fills:
                # Filter to only include fills for our trading symbol
                coin = fill.get("coin", "")
                if coin != target_coin:
                    continue

                side = fill.get("side")  # "B" for buy, "A" for sell
                price = float(fill.get("px", 0))
                size = float(fill.get("sz", 0))

                if side == "B":
                    order_side = OrderSide.BUY
                elif side == "A":
                    order_side = OrderSide.SELL
                else:
                    continue

                self.metrics.add_fill(order_side, price, size)
                synced_count += 1

            logger.info(f"Synced {synced_count} recent {target_coin} fills for OPT#14 adaptive mode")

        except Exception as e:
            logger.error(f"Error syncing recent fills: {e}")

    async def run_iteration(self) -> None:
        """
        Run a single iteration of the strategy.

        This should be called in the main bot loop.
        """
        # Handle paused state - check if we can resume
        if self.state == StrategyState.PAUSED:
            # Check if conditions allow resuming
            risk_metrics = await self.risk_manager.check_risk()
            if not risk_metrics.should_pause_trading and not risk_metrics.emergency_close:
                logger.info("Risk conditions improved, resuming trading...")
                await self.resume()
            else:
                # Still in risk situation - wait
                return
        
        if self.state != StrategyState.RUNNING:
            return

        try:
            # Check if we're in rate limit backoff
            if hasattr(self.client, "_rate_limit_backoff") and self.client._rate_limit_backoff > 0:
                backoff_remaining = self.client._rate_limit_backoff - (
                    time.time() - self.client._last_429_time
                )
                if backoff_remaining > 0:
                    logger.debug(f"Rate limit backoff: waiting {backoff_remaining:.1f}s")
                    return  # Skip this iteration

            # Refresh inventory less frequently (every 2 seconds instead of every iteration)
            now = time.time()
            if now - getattr(self, "_last_inventory_refresh", 0) > 2.0:
                await self._refresh_inventory()
                self._last_inventory_refresh = now
                self._api_call_metrics["inventory_refreshes"] += 1
            
            # CRITICAL: Check minimum equity before placing orders
            # Minimum $10 required to place minimum size orders ($0.10 * 100 = $10 margin)
            MIN_EQUITY_REQUIRED = 10.0
            if self.current_equity < MIN_EQUITY_REQUIRED:
                # Log every 60 seconds to avoid spam
                if now - getattr(self, "_last_equity_warning", 0) > 60.0:
                    logger.warning(
                        f"INSUFFICIENT EQUITY: ${self.current_equity:.2f} < ${MIN_EQUITY_REQUIRED:.2f} required. "
                        f"Please fund account to trade. Pausing order placement..."
                    )
                    self._last_equity_warning = now
                # Cancel any existing orders and pause
                if self.active_bids or self.active_asks:
                    await self.client.cancel_all_orders(self.symbol)
                    self.active_bids.clear()
                    self.active_asks.clear()
                return

            # Synchronize active orders with exchange every 30 seconds
            # Critical for HIP-3 to catch orphaned orders quickly
            if now - getattr(self, "_last_order_sync", 0) > 30.0:
                await self._sync_active_orders()
                self._last_order_sync = now
            
            # AGGRESSIVE CLEANUP: Every 2 minutes, do a full cancel-all if we detect drift
            # This catches any orders that slipped through normal sync
            if now - getattr(self, "_last_aggressive_cleanup", 0) > 120.0:
                self._last_aggressive_cleanup = now
                try:
                    import requests
                    resp = requests.post("https://api.hyperliquid.xyz/info", json={
                        "type": "openOrders",
                        "user": self.config.wallet_address
                    }, timeout=10)
                    all_orders = resp.json()
                    symbol_orders = [o for o in all_orders if o.get('coin') in [f'km:{self.symbol}', self.symbol.upper()]]
                    tracked_count = len(self.active_bids) + len(self.active_asks)
                    
                    # If exchange has significantly more orders than we're tracking, cancel all
                    if len(symbol_orders) > tracked_count + 5:
                        logger.warning(f"AGGRESSIVE CLEANUP: Exchange has {len(symbol_orders)} orders but only tracking {tracked_count}. Cancelling all!")
                        await self.client.cancel_all_orders(self.symbol)
                        self.active_bids.clear()
                        self.active_asks.clear()
                except Exception as e:
                    logger.debug(f"Aggressive cleanup check failed: {e}")

            # Check risk first
            risk_metrics = await self.risk_manager.check_risk()

            # Handle critical risk
            if risk_metrics.emergency_close:
                logger.critical("Emergency close triggered by risk manager!")
                await self.risk_manager.emergency_close_all()
                self.state = StrategyState.PAUSED
                return

            if risk_metrics.should_pause_trading:
                await self.pause()
                return

            # Adjust leverage if needed
            await self.risk_manager.adjust_leverage(risk_metrics)

            # Update quotes
            await self._update_quotes(risk_metrics)

            # Check rebalance
            if self._should_rebalance():
                await self._rebalance_inventory(risk_metrics)

            # Update funding rate periodically (cache previous value)
            # OPTIMIZATION: Reduced from 5 minutes to 15 minutes since funding rates change slowly
            if time.time() - self.last_funding_check > 900:  # Every 15 minutes
                new_rate = await self.client.get_funding_rate()
                if new_rate is not None:
                    self.funding_rate = new_rate
                self.last_funding_check = time.time()

            # Check daily action limit
            if self.metrics.actions_today >= self.config.execution.target_actions_per_day:
                logger.warning("Daily action limit reached, reducing activity")
                await asyncio.sleep(5)  # Slow down

        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")

    async def _update_quotes(self, risk_metrics: RiskMetrics) -> None:
        """
        Update bid and ask quotes with L2 order book awareness.
        
        PROFESSIONAL MM APPROACH:
        1. Analyze L2 book depth and imbalance
        2. Calculate volatility-adaptive spread (min/max for tiering)
        3. Check book liquidity before quoting
        4. Build exponentially tiered quotes (concentrated near mid)
        5. Apply inventory skew for delta-neutral operation
        6. Handle one-sided quoting for extreme imbalance
        """
        # Check quote interval (1s for HFT)
        now = time.time()
        if now - self.last_quote_time < self.QUOTE_REFRESH_INTERVAL:
            return

        self.last_quote_time = now

        # Get current orderbook
        orderbook = await self._get_cached_orderbook()
        if not orderbook or not orderbook.mid_price:
            logger.warning("No orderbook available")
            return
        
        # Update price buffer
        self.price_buffer.append(orderbook.mid_price)
        self.last_mid_price = orderbook.mid_price

        # Analyze L2 book depth
        self.last_book_analysis = self._analyze_order_book(orderbook)
        
        # Check if book is liquid enough
        if not self.last_book_analysis.is_liquid:
            logger.debug(f"Book depth: bids=${self.last_book_analysis.total_bid_depth:.0f}, asks=${self.last_book_analysis.total_ask_depth:.0f} (threshold: $5000)")
            logger.warning("Book not liquid enough - cancelling quotes")
            await self._cancel_all_quotes()
            return

        # Calculate adaptive spread (returns min/max for tiering)
        min_spread_bps, max_spread_bps = self._calculate_spread(orderbook, risk_metrics)

        # Calculate quote sizes
        base_size = self.risk_manager.calculate_order_size(
            orderbook.mid_price, "both", risk_metrics
        )

        if base_size <= 0:
            # Risk too high, cancel existing quotes
            await self._cancel_all_quotes()
            return

        # Build tiered quotes with exponential spread distribution
        new_bids, new_asks = self._build_tiered_quotes(
            orderbook, min_spread_bps, max_spread_bps, base_size
        )
        
        # Handle one-sided quoting for extreme inventory imbalance (>2.5%)
        if abs(self.inventory.delta) > 0.025:
            if self.inventory.delta > 0:  # Long - only quote asks
                new_bids = []
                await self._cancel_all_side("buy")
                logger.info(f"ONE-SIDED (asks only): delta={self.inventory.delta:.3f}")
            else:  # Short - only quote bids
                new_asks = []
                await self._cancel_all_side("sell")
                logger.info(f"ONE-SIDED (bids only): delta={self.inventory.delta:.3f}")

        # Update orders
        await self._update_orders(new_bids, new_asks)

    async def _update_quotes_OLD(self, risk_metrics: RiskMetrics) -> None:
        """OLD: Original grid-based quote update logic."""
        # Check quote interval
        now = time.time()
        if now - self.last_quote_time < self.quote_interval:
            return

        self.last_quote_time = now

        # Get current orderbook (with caching)
        orderbook = await self._get_cached_orderbook()
        if not orderbook or not orderbook.mid_price:
            logger.warning("No orderbook available")
            return

        # Calculate spread
        spread_bps = self._calculate_spread_OLD(orderbook, risk_metrics)

        # Calculate quote prices with inventory skew
        bid_price, ask_price = self._calculate_quote_prices(orderbook, spread_bps, risk_metrics)

        # Calculate quote sizes
        quote_size = self.risk_manager.calculate_order_size(
            orderbook.mid_price, "both", risk_metrics
        )

        if quote_size <= 0:
            # Risk too high, cancel existing quotes
            await self._cancel_all_quotes()
            return

        # DEBUG: Log what we're trying to do
        notional_value = quote_size * orderbook.mid_price
        logger.debug(
            f"Quote size calculated: {quote_size:.8f} contracts at ${orderbook.mid_price:.2f} = "
            f"${notional_value:.2f} notional (min $10.00)"
        )

        # Build new quote levels
        logger.debug(f"Building quote levels with base_size={quote_size:.8f}")
        new_bids, new_asks = self._build_quote_levels(bid_price, ask_price, quote_size, spread_bps)
        logger.debug(f"Built {len(new_bids)} bid levels, {len(new_asks)} ask levels")

        # ONE-SIDED QUOTING for inventory rebalancing - SURVIVAL MODE
        # When delta exceeds 15%, only quote on the side that reduces position
        # OPT#15: Lowered from 25% to 15% to prevent deep position accumulation
        if abs(self.inventory.delta) > 0.15:
            # Calculate minimum size for this symbol
            # US500: szDecimals=1 means minimum 0.1 contracts (~$69 at $693)
            mid_price = orderbook.mid_price
            if self.symbol.upper() == "US500":
                min_level_size = 0.1  # szDecimals=1 -> minimum 0.1 contracts
            else:
                min_level_size = 0.00012  # BTC minimum
            
            if self.inventory.delta > 0:  # Long position, only quote asks (to sell)
                new_bids = []
                # Cancel ALL bids on exchange (not just tracked ones)
                await self._cancel_all_side("buy")
                # AGGRESSIVE ASK PLACEMENT: Place asks just $0.02 above market ask (tight for ALO)
                # For rebalancing LONG, we want to SELL so place asks close to BBO
                aggressive_ask = round_price(
                    orderbook.best_ask + 0.02, 0.01
                )  # $0.02 above BBO (ALO safe, using 0.01 tick)
                new_asks = [
                    QuoteLevel(
                        price=aggressive_ask + i * 0.01,  # $0.01 spacing for tight quotes
                        size=round_size(max(quote_size * (1 - i * 0.1), min_level_size), 0.1),
                        side=OrderSide.SELL,
                    )
                    for i in range(min(self.order_levels, 6))
                    if round_size(max(quote_size * (1 - i * 0.1), min_level_size), 0.1) >= min_level_size
                ]
                logger.info(
                    f"REBALANCE MODE: ASKS only at ${aggressive_ask:.2f}+ (delta={self.inventory.delta:.2f}, pos={self.inventory.position_size:+.4f})"
                )
            else:  # Short position, only quote bids (to buy)
                new_asks = []
                # Cancel ALL asks on exchange (not just tracked ones)
                await self._cancel_all_side("sell")
                # AGGRESSIVE BID PLACEMENT: Place bids just $0.02 below best bid (tight for ALO)
                # For rebalancing SHORT, we want to BUY so place bids close to BBO
                aggressive_bid = round_price(
                    orderbook.best_bid - 0.02, 0.01
                )  # $0.02 below BBO (ALO safe, using 0.01 tick)
                new_bids = [
                    QuoteLevel(
                        price=aggressive_bid - i * 0.01,  # $0.01 spacing for tight quotes
                        size=round_size(max(quote_size * (1 - i * 0.1), min_level_size), 0.1),
                        side=OrderSide.BUY,
                    )
                    for i in range(min(self.order_levels, 6))
                    if round_size(max(quote_size * (1 - i * 0.1), min_level_size), 0.1) >= min_level_size
                ]
                logger.info(
                    f"REBALANCE MODE: BIDS only at ${aggressive_bid:.2f}- (delta={self.inventory.delta:.2f}, pos={self.inventory.position_size:+.4f})"
                )

        # Update orders
        await self._update_orders(new_bids, new_asks)

    def _analyze_order_book(self, orderbook: OrderBook) -> BookDepthAnalysis:
        """
        Analyze L2 order book depth and liquidity for professional market making.
        
        Returns comprehensive book analysis including:
        - Total depth (top 10 levels)
        - Imbalance and directional pressure
        - Microprice (size-weighted mid)
        - Liquidity concentration
        """
        analysis = BookDepthAnalysis()
        
        if not orderbook.bids or not orderbook.asks:
            return analysis
        
        # Calculate total depth (top 10 levels for professional MM)
        top_10_bids = orderbook.bids[:10]
        top_10_asks = orderbook.asks[:10]
        
        analysis.total_bid_depth = sum(price * size for price, size in top_10_bids)
        analysis.total_ask_depth = sum(price * size for price, size in top_10_asks)
        
        # Top 5 depth (concentration analysis)
        analysis.top_5_bid_depth = sum(price * size for price, size in orderbook.bids[:5])
        analysis.top_5_ask_depth = sum(price * size for price, size in orderbook.asks[:5])
        
        # Book imbalance (-1 to +1, positive = more bid pressure)
        total_depth = analysis.total_bid_depth + analysis.total_ask_depth
        if total_depth > 0:
            analysis.imbalance = (analysis.total_bid_depth - analysis.total_ask_depth) / total_depth
        
        # Microprice (size-weighted mid for better fair value estimation)
        if orderbook.best_bid and orderbook.best_ask:
            best_bid_size = orderbook.best_bid_size or 0
            best_ask_size = orderbook.best_ask_size or 0
            total_size = best_bid_size + best_ask_size
            
            if total_size > 0:
                analysis.weighted_mid = (
                    orderbook.best_bid * best_ask_size + orderbook.best_ask * best_bid_size
                ) / total_size
            else:
                analysis.weighted_mid = orderbook.mid_price or 0
        
        # Book pressure (directional force, scaled by depth)
        analysis.book_pressure = analysis.imbalance * min(total_depth / 100000, 1.0)
        
        return analysis

    def _calculate_inventory_skew(self) -> Tuple[float, float]:
        """
        Calculate quote skew factors based on inventory position.
        
        Returns (bid_skew_factor, ask_skew_factor) where:
        - >1.0 = widen that side (discourage fills)
        - <1.0 = tighten that side (encourage fills)
        """
        if abs(self.inventory.delta) < 0.005:  # Within 0.5%
            return 1.0, 1.0
        
        # Linear skew up to 5% delta
        urgency = self.inventory.skew_urgency
        
        if self.inventory.delta > 0:  # Long position - discourage more buys
            bid_skew = 1.0 + urgency * 2.0  # Widen bids up to 3x
            ask_skew = 1.0 - urgency * 0.3  # Tighten asks slightly
        else:  # Short position - discourage more sells
            bid_skew = 1.0 - urgency * 0.3  # Tighten bids
            ask_skew = 1.0 + urgency * 2.0  # Widen asks up to 3x
        
        return max(bid_skew, 0.5), max(ask_skew, 0.5)

    def _build_tiered_quotes(
        self,
        orderbook: OrderBook,
        min_spread_bps: float,
        max_spread_bps: float,
        total_size: float
    ) -> Tuple[List[QuoteLevel], List[QuoteLevel]]:
        """
        Build exponentially tiered quotes concentrated near mid price.
        
        PROFESSIONAL MM QUOTE DISTRIBUTION:
        - Levels 1-5: 1-5 bps, 70% of volume (tight for high fill rate)
        - Levels 6-10: 5-15 bps, 20% of volume (medium spread)
        - Levels 11-15: 15-50 bps, 10% of volume (wide spread, tail risk)
        
        Uses exponential decay for sizing and exponential expansion for spreads.
        """
        logger.debug(f"_build_tiered_quotes called: min={min_spread_bps:.1f}, max={max_spread_bps:.1f}, size={total_size:.4f}")
        logger.debug(f"Orderbook: mid={orderbook.mid_price}, best_bid={orderbook.best_bid}, best_ask={orderbook.best_ask}")
        if not orderbook.mid_price or not orderbook.best_bid or not orderbook.best_ask:
            logger.warning(f"_build_tiered_quotes: Missing orderbook data (mid={orderbook.mid_price}, bid={orderbook.best_bid}, ask={orderbook.best_ask}), returning empty")
            return [], []
        
        mid = orderbook.mid_price
        bids = []
        asks = []
        
        # Get inventory skew factors
        bid_skew, ask_skew = self._calculate_inventory_skew()
        
        # Number of levels (reduced to 12-15 for concentrated liquidity)
        total_levels = min(self.MAX_BID_ORDERS, 12)
        
        # Size distribution: exponentially decreasing (70% in top 5 levels)
        # CRITICAL FIX: Use total_size as BASE for first level, not total across all levels
        # This ensures at least the first few levels meet minimum lot size
        sizes = []
        decay_factor = 0.9  # Exponential decay (0.9 for smoother distribution across 12 levels)
        for i in range(total_levels):
            level_size = total_size * (decay_factor ** i)
            sizes.append(level_size)
        
        # NO normalization - let sizes decay naturally from base
        # This way if total_size = 0.1, level 0 = 0.1, level 1 = 0.07, etc.
        
        # Spread distribution: exponential expansion from min to max
        spreads = []
        for i in range(total_levels):
            t = i / (total_levels - 1) if total_levels > 1 else 0
            spread_bps = min_spread_bps * (max_spread_bps / min_spread_bps) ** t
            spreads.append(spread_bps)
        
        # Apply inventory skew to spreads
        bid_spreads = [s * bid_skew for s in spreads]
        ask_spreads = [s * ask_skew for s in spreads]
        
        # Build quote levels
        tick_size = 1.0 if self.symbol == "BTC" else 0.01
        lot_size = 0.0001 if self.symbol == "BTC" else 0.1
        
        logger.debug(f"Building {total_levels} levels: tick={tick_size}, lot={lot_size}, mid={mid:.2f}")
        
        for i in range(total_levels):
            # Bid
            bid_offset = mid * (bid_spreads[i] / 10000)
            bid_price = round_price(mid - bid_offset, tick_size)
            
            # Ensure we don't cross the book
            if orderbook.best_bid and bid_price >= orderbook.best_bid:
                old_bid = bid_price
                bid_price = round_price(orderbook.best_bid - tick_size, tick_size)
                logger.debug(f"Level {i}: Bid crossed book! {old_bid:.2f} >= {orderbook.best_bid:.2f}, adjusted to {bid_price:.2f}")
            
            bid_size = round_size(sizes[i], lot_size)
            if bid_size >= lot_size:
                bids.append(QuoteLevel(
                    price=bid_price,
                    size=bid_size,
                    side=OrderSide.BUY
                ))
            else:
                logger.debug(f"Level {i}: Bid size too small: {bid_size:.4f} < {lot_size:.4f}")
            
            # Ask
            ask_offset = mid * (ask_spreads[i] / 10000)
            ask_price = round_price(mid + ask_offset, tick_size)
            
            # Ensure we don't cross the book
            if orderbook.best_ask and ask_price <= orderbook.best_ask:
                old_ask = ask_price
                ask_price = round_price(orderbook.best_ask + tick_size, tick_size)
                logger.debug(f"Level {i}: Ask crossed book! {old_ask:.2f} <= {orderbook.best_ask:.2f}, adjusted to {ask_price:.2f}")
            
            ask_size = round_size(sizes[i], lot_size)
            if ask_size >= lot_size:
                asks.append(QuoteLevel(
                    price=ask_price,
                    size=ask_size,
                    side=OrderSide.SELL
                ))
            else:
                logger.debug(f"Level {i}: Ask size too small: {ask_size:.4f} < {lot_size:.4f}")
        
        # Remove duplicates (same price levels)
        bids = self._deduplicate_levels(bids)
        asks = self._deduplicate_levels(asks)
        
        # Log summary (ALWAYS log, even if empty)
        logger.debug(f"After building: {len(bids)} bids, {len(asks)} asks")
        if bids and asks:
            total_bid_notional = sum(l.price * l.size for l in bids)
            total_ask_notional = sum(l.price * l.size for l in asks)
            logger.debug(
                f"Built {len(bids)} bids (${total_bid_notional:.0f}) + "
                f"{len(asks)} asks (${total_ask_notional:.0f}) | "
                f"Spreads: {min_spread_bps:.1f}-{max_spread_bps:.1f} bps | "
                f"Skew: {bid_skew:.2f}/{ask_skew:.2f}"
            )
        else:
            logger.warning(f"Empty quotes! bids={len(bids)}, asks={len(asks)}")
        
        return bids, asks

    def _deduplicate_levels(self, levels: List[QuoteLevel]) -> List[QuoteLevel]:
        """Remove duplicate price levels, keeping the largest size."""
        if not levels:
            return []
        
        by_price = {}
        for level in levels:
            if level.price not in by_price:
                by_price[level.price] = level
            else:
                # Keep the larger size
                if level.size > by_price[level.price].size:
                    by_price[level.price] = level
        
        # Sort by price
        result = sorted(by_price.values(), key=lambda x: x.price,
                       reverse=(levels[0].side == OrderSide.BUY))
        return result

    def _calculate_spread(self, orderbook: OrderBook, risk_metrics: RiskMetrics) -> Tuple[float, float]:
        """
        Calculate volatility-adaptive spread for professional market making.
        
        PROFESSIONAL MM APPROACH (L2-aware):
        - Base spread on realized volatility (1-50 bps range)
        - Adjust for book imbalance (adverse selection risk)
        - Widen on consecutive losing fills (quote fading)
        - Tighten when book is deep and liquid
        - Return (min_spread_bps, max_spread_bps) for exponential tiering
        
        Evolution from grid-based:
        - Old: Fixed 1-50 bps grid, no book awareness
        - New: Dynamic 1-50 bps based on vol, book depth, adverse selection
        """
        # Calculate realized volatility
        prices = self.price_buffer.get_array()
        if len(prices) >= 20:
            returns = np.diff(np.log(prices[-60:]))  # Last 60 ticks
            realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
            self.volatility_buffer.append(realized_vol)
        else:
            realized_vol = 0.10  # Default 10%
        
        # Base spread on volatility
        if realized_vol < 0.08:  # Low vol (<8%)
            min_spread = 1.0  # 1 bp
            max_spread = 10.0
        elif realized_vol < 0.15:  # Medium vol (8-15%)
            min_spread = 2.0
            max_spread = 20.0
        elif realized_vol < 0.25:  # High vol (15-25%)
            min_spread = 5.0
            max_spread = 35.0
        else:  # Very high vol (>25%)
            min_spread = 10.0
            max_spread = 50.0
        
        # Adjust for book conditions
        if self.last_book_analysis:
            # Widen if book is imbalanced (adverse selection risk)
            if abs(self.last_book_analysis.imbalance) > 0.3:
                min_spread *= 1.5
                max_spread *= 1.3
                logger.debug(f"Book imbalance {self.last_book_analysis.imbalance:.2f} - widening spreads")
            
            # Tighten if book is deep and liquid
            if self.last_book_analysis.is_liquid:
                min_spread *= 0.8
                max_spread *= 0.9
        
        # Check for adverse selection (quote fading)
        recent_spread = self.metrics.get_recent_spread_bps()
        if recent_spread is not None and recent_spread < -2.0:
            # Losing money - widen spreads
            min_spread *= 2.0
            max_spread *= 1.5
            logger.warning(f"Adverse selection detected: {recent_spread:.2f} bps - widening spreads")
        
        # Quote fading on consecutive losing fills
        if self.metrics.consecutive_losing_fills >= self.ADVERSE_SELECTION_THRESHOLD:
            min_spread *= 2.5
            max_spread *= 2.0
            logger.warning(f"Quote fading: {self.metrics.consecutive_losing_fills} losing fills")
        
        # Ensure reasonable bounds
        min_spread = max(min_spread, 1.0)  # At least 1 bp
        max_spread = min(max_spread, 50.0)  # At most 50 bps
        
        logger.debug(f"Spread: {min_spread:.1f}-{max_spread:.1f} bps (vol={realized_vol:.2%})")
        
        return min_spread, max_spread

    def _calculate_spread_OLD(self, orderbook: OrderBook, risk_metrics: RiskMetrics) -> float:
        """
        Calculate adaptive spread using IV-based calculation with numba.

        OPTIMIZATION #14: ADAPTIVE ANTI-PICKING-OFF

        Evolution:
        - OPT#10: 3 bps = too tight, adverse selection (-4.17 bps)
        - OPT#11: 25+ bps = too wide, 0% fill rate
        - OPT#12: 5-6 bps AT BBO = adverse selection again (-5.04 bps)
        - OPT#13: 8-10 bps + $2-3 BEHIND BBO = better but still losses during high-churn

        Analysis from 24h data (1793 fills):
        - Overall: +7.85 bps (profitable!)
        - But 10 hours with losses (05:00-12:00 had -1.5 to -7.3 bps)
        - Problem: High fill rate during ranging/volatile periods = adverse selection

        OPT#14 Solution: ADAPTIVE defensive distance
        - Monitor last 20 fills to detect adverse selection in real-time
        - If spread < -2 bps: Increase distance to $5 (from $2)
        - If spread < +2 bps: Moderate distance $3.5
        - If spread > +2 bps: Standard distance $2
        - This dynamically adapts to market conditions
        """
        from .utils import calculate_iv_fast, calculate_spread_from_iv

        # Calculate IV from recent prices using numba-accelerated function
        prices = self.price_buffer.get_array()
        if len(prices) >= 10:
            iv = calculate_iv_fast(prices, window=60)  # 1-hour window
        else:
            iv = 0.10  # Default 10% if insufficient data

        # Get IV-based spread - WIDER to avoid adverse selection
        vol_threshold = getattr(self.config.risk, "vol_threshold_for_wide_spread", 0.15)
        base_spread = calculate_spread_from_iv(
            iv=iv,
            vol_threshold=vol_threshold,
            min_spread_bps=max(
                self.min_spread_bps, 1.0  # OPT#17: Tighten to 1 bps minimum in low vol
            ),  # OPT#16: 12 bps minimum (raised from 10)
            max_spread_bps=min(self.max_spread_bps, 35.0),  # OPT#16: Cap at 35 bps (was 30)
            low_vol_spread_bps=2.0,  # OPT#17: 2 bps in calm markets (tightened from 14)
            high_vol_spread_bps=35.0,  # OPT#17: 35 bps in volatile markets (widened from 28)
        )

        # ORDER FLOW TOXICITY DETECTION - Only for EXTREME imbalance
        # Raised threshold: 85% imbalance (was 70%)
        if orderbook.best_bid_size and orderbook.best_ask_size:
            total_size = orderbook.best_bid_size + orderbook.best_ask_size
            if total_size > 0:
                imbalance = abs(orderbook.best_bid_size - orderbook.best_ask_size) / total_size
                if imbalance > 0.85:  # >85% imbalance = toxic (raised from 70%)
                    toxicity_factor = (imbalance - 0.7) * 15  # Max +4.5 bps
                    base_spread += toxicity_factor
                    logger.debug(  # Changed to DEBUG to reduce log noise
                        f"TOXIC FLOW detected: imbalance={imbalance:.2f}, +{toxicity_factor:.1f} bps"
                    )

        # Widen for inventory imbalance - REDUCED from OPT#11
        if abs(self.inventory.delta) > 0.02:  # Higher threshold (was 0.01)
            inventory_factor = abs(self.inventory.delta) * 10  # +10 bps per unit delta (was +20)
            base_spread += inventory_factor

        # Widen for elevated risk - LESS AGGRESSIVE multipliers
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 1.2,  # Reduced from 1.4
            RiskLevel.HIGH: 1.5,  # Reduced from 2.0
            RiskLevel.CRITICAL: 2.0,  # Reduced from 3.0
        }
        base_spread *= risk_multipliers.get(risk_metrics.risk_level, 1.0)

        # FUNDING RATE SKEW for hedging (>0.02% threshold)
        if risk_metrics.should_hedge_funding:
            base_spread += 3  # +3 bps for funding hedge (was +5)
        elif abs(self.funding_rate) > 0.0002:  # >0.02% (raised threshold)
            base_spread += abs(self.funding_rate) * 2500  # Reduced from 5000

        # Cap spread at max - REDUCED CAP
        return min(base_spread, 20.0)  # Max 20 bps (was max_spread_bps)

    def _calculate_quote_prices(
        self, orderbook: OrderBook, spread_bps: float, risk_metrics: RiskMetrics
    ) -> Tuple[float, float]:
        """
        Calculate bid and ask prices with inventory skew.

        OPTIMIZATION #11: ANTI-ADVERSE-SELECTION PRICING

        Analysis shows we're buying high and selling low (-4.17 bps).
        The issue is we're too close to BBO and getting picked off.

        New approach:
        1. Quote BEHIND BBO by at least 5 ticks ($5)
        2. When we have inventory, be VERY aggressive on the reducing side
        3. Don't try to compete with HFT - focus on capturing wide spreads
        4. FADE THE MOVE: After getting filled, quote opposite side aggressively

        Target: 50%+ win rate, $0.01+ per trade profit
        """
        mid = orderbook.mid_price
        half_spread_pct = spread_bps / 2 / 10000  # Convert bps to decimal

        # Calculate spread in dollars
        half_spread_dollars = mid * half_spread_pct

        # Base quotes - start from mid with our calculated spread
        base_bid = mid - half_spread_dollars
        base_ask = mid + half_spread_dollars

        # OPTIMIZATION #17: AGGRESSIVE INVENTORY SKEW
        # Enhanced skew for imbalance >1.5% - more aggressive one-sided quoting
        inventory_skew_pct = 0.8  # Increased from 50% to 80% skew at 20% delta
        delta_factor = min(abs(self.inventory.delta) / 0.20, 1.0)  # Normalize to 20%
        skew_amount = half_spread_dollars * inventory_skew_pct * delta_factor

        # AGGRESSIVE ONE-SIDED: If imbalance >0.5%, quote only on reducing side (EMERGENCY: reduced from 1.5%)
        imbalance_threshold = 0.005  # EMERGENCY: 0.5% threshold for aggressive skew (reduced from 1.5%)
        if abs(self.inventory.delta) > imbalance_threshold:
            if self.inventory.delta > imbalance_threshold:  # Long - need to SELL
                # Aggressive sell-only: Lower ASK significantly, keep bid normal
                aggressive_skew = half_spread_dollars * 1.5  # 150% of half-spread
                ask_price = base_ask - aggressive_skew
                bid_price = base_bid  # Keep bid normal
                logger.debug(
                    f"AGGRESSIVE SELL: delta={self.inventory.delta:.3f}, ask-${aggressive_skew:.1f}"
                )
            elif self.inventory.delta < -imbalance_threshold:  # Short - need to BUY
                # Aggressive buy-only: Raise BID significantly, keep ask normal
                aggressive_skew = half_spread_dollars * 1.5  # 150% of half-spread
                bid_price = base_bid + aggressive_skew
                ask_price = base_ask  # Keep ask normal
                logger.debug(
                    f"AGGRESSIVE BUY: delta={self.inventory.delta:.3f}, bid+${aggressive_skew:.1f}"
                )
        else:
            # Standard balanced skew for small imbalances
            if self.inventory.delta > 0.01:  # Long - need to SELL (threshold raised)
                ask_price = base_ask - skew_amount
                bid_price = base_bid
                logger.debug(f"STANDARD SELL SKEW: ask-${skew_amount:.1f}")
            elif self.inventory.delta < -0.01:  # Short - need to BUY
                bid_price = base_bid + skew_amount
                ask_price = base_ask
                logger.debug(f"STANDARD BUY SKEW: bid+${skew_amount:.1f}")
            else:
                # Neutral - no skew
                bid_price = base_bid
                ask_price = base_ask
        # OPTIMIZATION #17: ENHANCED ADAPTIVE ANTI-PICKING-OFF
        # Use bps-based distances that scale with price (not fixed dollar amounts)
        # For US500 at $693: 10 bps = $0.69, 20 bps = $1.39
        # For BTC at $90000: 10 bps = $9.00, 20 bps = $18.00
        # NOTE: We scale levels as percentage of configured order_levels, not hardcoded values
        configured_levels = self.config.trading.order_levels  # User's configured level count
        recent_spread = self.metrics.get_recent_spread_bps()
        if recent_spread is not None:
            if recent_spread < -3.0:  # Severe adverse selection
                defensive_bps = 3.0  # 3 bps behind BBO (increased from 1bps to avoid adverse selection)
                self.order_levels = max(10, configured_levels // 4)  # 25% of configured levels
                logger.debug(
                    f"OPT#17 DEFENSIVE: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, {self.order_levels} levels"
                )
            elif recent_spread < 1.0:  # Low profit threshold
                defensive_bps = 0.0  # AT BBO (aggressive for HIP-3)
                self.order_levels = max(20, configured_levels // 2)  # 50% of configured levels
                logger.debug(
                    f"OPT#17 CAUTIOUS: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, {self.order_levels} levels"
                )
            elif recent_spread > 10.0:  # NEW: Aggressive mode for very profitable spreads
                defensive_bps = 0.0  # AT BBO
                self.order_levels = configured_levels  # Full configured levels
                logger.debug(
                    f"OPT#17 AGGRESSIVE: spread {recent_spread:.2f} bps → {defensive_bps:.1f} bps distance, {self.order_levels} levels"
                )
            else:  # Normal profitable range
                defensive_bps = 0.0  # AT BBO
                self.order_levels = max(50, int(configured_levels * 0.75))  # 75% of configured levels
                logger.debug(
                    f"OPT#17 NORMAL: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, {self.order_levels} levels"
                )
        else:
            # Default when insufficient data - use configured levels
            defensive_bps = 0.0  # AT BBO to ensure fills
            self.order_levels = configured_levels
            logger.debug(f"OPT#17 DEFAULT: insufficient/stale data → using {configured_levels} levels at BBO")
        
        # Convert bps to dollar distance
        defensive_distance = mid * (defensive_bps / 10000.0)

        # Apply defensive distance - quote BEHIND BBO
        bid_price = max(base_bid, orderbook.best_bid - defensive_distance)
        ask_price = min(base_ask, orderbook.best_ask + defensive_distance)

        # Round to tick size: $0.01 for US500, $1 for BTC
        tick_size = 0.01 if self.symbol.upper() == "US500" else 1.0
        bid_price = round_price(bid_price, tick_size)
        ask_price = round_price(ask_price, tick_size)

        # ENSURE MINIMUM SPREAD - EMERGENCY ADVERSE SELECTION PROTECTION
        # US500 HIP-3: Experiencing severe adverse selection, need much wider spreads
        # BTC: Wider spreads, use 8 bps minimum
        if self.symbol.upper() == "US500":
            # For US500, check recent fill history for adverse selection
            recent_spread = self.metrics.get_recent_spread_bps()
            if recent_spread is not None and recent_spread < -5.0:  # Severe adverse selection
                min_spread_bps = 15.0  # EMERGENCY: 15 bps when severe adverse selection
                logger.warning(f"US500 EMERGENCY protection: {recent_spread:.1f} bps -> {min_spread_bps} bps min")
            elif recent_spread is not None and recent_spread < -1.0:  # Adverse selection detected
                min_spread_bps = 8.0  # Increase to 8 bps when adverse selection detected  
                logger.debug(f"US500 adverse selection protection: {recent_spread:.1f} bps -> {min_spread_bps} bps min")
            elif recent_spread is not None and recent_spread < 1.0:  # Poor performance
                min_spread_bps = 6.0  # Moderate increase to 6 bps
                logger.debug(f"US500 moderate protection: {recent_spread:.1f} bps -> {min_spread_bps} bps min") 
            else:
                min_spread_bps = 5.0  # 5 bps minimum when performing well (increased from 2 bps for better profitability)
        else:
            min_spread_bps = max(self.min_spread_bps, 8.0)  # 8 bps for main perps
        
        min_spread_dollars = mid * (min_spread_bps / 10000)
        if ask_price - bid_price < min_spread_dollars:
            gap = min_spread_dollars - (ask_price - bid_price)
            bid_price -= gap / 2
            ask_price += gap / 2
            bid_price = round_price(bid_price, tick_size)
            ask_price = round_price(ask_price, tick_size)

        # NOTE: Defensive distance already applied above - no need to re-apply

        # Final validation - never cross the spread
        if bid_price >= ask_price:
            mid_point = (bid_price + ask_price) / 2
            # US500: 1 bps minimum, others: 5 bps
            min_half_spread = mid * (0.0001 if self.symbol.upper() == "US500" else 0.0005)
            bid_price = round_price(mid_point - min_half_spread, tick_size)
            ask_price = round_price(mid_point + min_half_spread, tick_size)

        # Calculate actual spread in bps for logging
        actual_spread_bps = (ask_price - bid_price) / mid * 10000

        logger.debug(
            f"Quotes: bid=${bid_price:.0f} ask=${ask_price:.0f} | "
            f"BBO: ${orderbook.best_bid:.0f}@${orderbook.best_ask:.0f} | "
            f"Spread: {actual_spread_bps:.1f}bps | Delta: {self.inventory.delta:.3f}"
        )

        return bid_price, ask_price

    def _build_quote_levels(
        self, bid_price: float, ask_price: float, base_size: float, spread_bps: float
    ) -> Tuple[List[QuoteLevel], List[QuoteLevel]]:
        """
        Build multiple quote levels on each side.

        OPT#14: Reduce levels during adverse selection to limit exposure.
        """
        bids = []
        asks = []

        # OPT#14: Adapt number of levels based on recent performance
        # For 100-level operation, we scale back proportionally rather than drastically
        recent_spread = self.metrics.get_recent_spread_bps()
        if recent_spread is not None and recent_spread < -2.0:
            # Adverse selection detected - reduce to 50% of configured levels
            effective_levels = max(10, self.order_levels // 2)
            logger.warning(
                f"Reducing to {effective_levels} levels due to adverse selection "
                f"(recent spread: {recent_spread:.2f} bps)"
            )
        elif recent_spread is not None and recent_spread < 2.0:
            # Low profitability - reduce to 75% of configured levels
            effective_levels = max(20, int(self.order_levels * 0.75))
        else:
            # Normal operation - use configured levels
            effective_levels = self.order_levels

        # Calculate level spacing
        mid = (bid_price + ask_price) / 2
        level_spacing = mid * (spread_bps / 10000) * 0.5  # Half spread per level

        # US500: szDecimals=1 means minimum size is 0.1 contracts (~$69 at $693)
        # Other symbols use the standard $10 notional minimum
        if self.symbol.upper() == "US500":
            min_size = 0.1  # szDecimals=1 -> minimum 0.1 contracts
        else:
            min_size = max(0.00012, 10.5 / mid)  # $10.50 notional for safety

        # OPTIMIZATION #4: Ensure base_size is scaled to allow all levels
        # With pyramiding reduction, scale up base_size so smallest level still meets minimum
        # Use adaptive pyramid factor based on level count
        if effective_levels > 50:
            pyramid_factor = 0.005  # 0.5% per level for 50+ levels
        elif effective_levels > 10:
            pyramid_factor = 0.03  # 3% per level for 10-50 levels
        else:
            pyramid_factor = 0.08  # 8% per level for <10 levels
        levels_pyramid_factor = 1 - (effective_levels - 1) * pyramid_factor
        required_base = min_size / max(
            levels_pyramid_factor, 0.1
        )  # Ensure smallest level >= min_size
        effective_base = max(base_size, required_base)

        # Log if base_size was scaled up
        if base_size < required_base:
            logger.debug(
                f"Scaled base_size {base_size:.6f} → {effective_base:.6f} to meet min_size {min_size:.6f}"
            )

        # Get lot size for this symbol (US500: 0.1, BTC: 0.00001)
        if self.symbol.upper() == "US500":
            lot_size = 0.1  # szDecimals=1
            tick_size = 0.01  # US500 uses $0.01 tick size
        else:
            lot_size = 0.00001  # BTC
            tick_size = 1.0  # BTC uses $1 tick size

        for i in range(effective_levels):  # Use effective_levels instead of self.order_levels
            # Decrease size for outer levels (pyramiding)
            # Use adaptive pyramid factor based on level count
            if effective_levels > 50:
                pyramid_factor = 0.005  # 0.5% per level for 50+ levels
            elif effective_levels > 10:
                pyramid_factor = 0.03  # 3% per level for 10-50 levels
            else:
                pyramid_factor = 0.08  # 8% per level for <10 levels
            level_size = effective_base * (1 - i * pyramid_factor)
            # Skip if below minimum size requirement (safety check)
            if level_size < min_size:
                logger.warning(f"Level {i} size {level_size:.6f} < min {min_size:.6f} - skipping")
                continue

            # Bid levels (decreasing prices)
            bid_level_price = round_price(bid_price - i * level_spacing, tick_size)
            bids.append(
                QuoteLevel(
                    price=bid_level_price,
                    size=round_size(level_size, lot_size),
                    side=OrderSide.BUY,
                )
            )

            # Ask levels (increasing prices)
            ask_level_price = round_price(ask_price + i * level_spacing, tick_size)
            asks.append(
                QuoteLevel(
                    price=ask_level_price,
                    size=round_size(level_size, lot_size),
                    side=OrderSide.SELL,
                )
            )

        # Debug logging
        if bids:
            logger.debug(f"Bid levels: {[(b.price, b.size, b.size*b.price) for b in bids]}")
        if asks:
            logger.debug(f"Ask levels: {[(a.price, a.size, a.size*a.price) for a in asks]}")

        return bids, asks

    async def _update_orders(self, new_bids: List[QuoteLevel], new_asks: List[QuoteLevel]) -> None:
        """
        Update orders efficiently with intelligent recycling.

        Instead of cancelling orders on every small price change, we:
        1. Check if existing orders are "close enough" to desired prices
        2. Reuse orders if within tolerance and not too old
        3. Only cancel/replace when truly necessary
        4. SKIP entire update if nothing meaningful changed

        This dramatically reduces API calls and rate limit usage.
        """
        logger.debug(f"_update_orders called: {len(new_bids)} new bids, {len(new_asks)} new asks | Active: {len(self.active_bids)} bids, {len(self.active_asks)} asks")
        orders_to_cancel = []
        orders_to_place = []
        now = time.time()
        
        # CRITICAL FIX: Force sync on first update cycle OR if we haven't synced in 30s
        if not hasattr(self, "_last_order_sync"):
            self._last_order_sync = 0
        
        time_since_sync = now - self._last_order_sync
        have_tracked_orders = len(self.active_bids) + len(self.active_asks)
        # Force sync if:
        # 1. First run (never synced)
        # 2. Have tracked orders but 30+ seconds since last sync
        # 3. Have many tracked orders (possible phantom accumulation)
        if (self._last_order_sync == 0 and have_tracked_orders > 0) or \
           (have_tracked_orders > 0 and time_since_sync > 30) or \
           (have_tracked_orders > 10):
            logger.info(f"Forcing order sync: tracked={have_tracked_orders}, time_since_sync={time_since_sync:.0f}s")
            await self._sync_active_orders()
            self._last_order_sync = now
        
        # SAFETY LIMIT: Maximum orders per side to prevent accumulation
        # Set to 2x configured order_levels to allow for normal operation
        MAX_ORDERS_PER_SIDE = self.config.trading.order_levels * 2
        
        # Check if we have too many orders tracked - if so, cancel all and start fresh
        total_tracked = len(self.active_bids) + len(self.active_asks)
        if total_tracked > MAX_ORDERS_PER_SIDE * 2:
            logger.warning(f"Order accumulation detected: {total_tracked} orders tracked (max {MAX_ORDERS_PER_SIDE*2}). Cancelling all.")
            await self._cancel_all_quotes()
            # Clear and start fresh
            self.active_bids.clear()
            self.active_asks.clear()

        # Track last actual update time
        if not hasattr(self, "_last_actual_update"):
            self._last_actual_update = 0

        # Track which new quotes are matched to existing orders
        matched_bid_indices = set()
        matched_ask_indices = set()

        # INTELLIGENT BID RECYCLING
        for oid, existing in list(self.active_bids.items()):
            # Find closest matching new bid
            best_match_idx = None
            best_match_distance = float("inf")

            for i, new_quote in enumerate(new_bids):
                if i in matched_bid_indices:
                    continue  # Already matched

                price_diff_pct = abs(new_quote.price - existing.price) / existing.price
                size_diff_pct = abs(new_quote.size - existing.size) / existing.size

                # Check if this quote is close enough to reuse
                if (
                    price_diff_pct <= self.PRICE_TOLERANCE_PCT
                    and size_diff_pct <= self.SIZE_TOLERANCE_PCT
                ):
                    distance = price_diff_pct + size_diff_pct
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_idx = i

            if best_match_idx is not None:
                # Order can be recycled - don't cancel it
                matched_bid_indices.add(best_match_idx)
                logger.debug(
                    f"Recycling bid order at ${existing.price:.0f} (wanted ${new_bids[best_match_idx].price:.0f})"
                )
            else:
                # Order needs to be cancelled
                age = now - existing.created_at
                if age < self.MIN_ORDER_AGE_SECONDS:
                    # Order too young - keep it to avoid churn
                    logger.debug(f"Keeping young bid at ${existing.price:.0f} (age={age:.1f}s)")
                else:
                    orders_to_cancel.append((self.symbol, oid))
                    del self.active_bids[oid]

        # INTELLIGENT ASK RECYCLING
        for oid, existing in list(self.active_asks.items()):
            # Find closest matching new ask
            best_match_idx = None
            best_match_distance = float("inf")

            for i, new_quote in enumerate(new_asks):
                if i in matched_ask_indices:
                    continue  # Already matched

                price_diff_pct = abs(new_quote.price - existing.price) / existing.price
                size_diff_pct = abs(new_quote.size - existing.size) / existing.size

                # Check if this quote is close enough to reuse
                if (
                    price_diff_pct <= self.PRICE_TOLERANCE_PCT
                    and size_diff_pct <= self.SIZE_TOLERANCE_PCT
                ):
                    distance = price_diff_pct + size_diff_pct
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_idx = i

            if best_match_idx is not None:
                # Order can be recycled - don't cancel it
                matched_ask_indices.add(best_match_idx)
                logger.debug(
                    f"Recycling ask order at ${existing.price:.0f} (wanted ${new_asks[best_match_idx].price:.0f})"
                )
            else:
                # Order needs to be cancelled
                age = now - existing.created_at
                if age < self.MIN_ORDER_AGE_SECONDS:
                    # Order too young - keep it to avoid churn
                    logger.debug(f"Keeping young ask at ${existing.price:.0f} (age={age:.1f}s)")
                else:
                    orders_to_cancel.append((self.symbol, oid))
                    del self.active_asks[oid]

        # EARLY EXIT #1: If we have enough orders and none to cancel, skip entirely
        have_enough_bids = len(self.active_bids) >= len(new_bids)
        have_enough_asks = len(self.active_asks) >= len(new_asks)
        no_cancels_needed = len(orders_to_cancel) == 0

        if have_enough_bids and have_enough_asks and no_cancels_needed:
            # Already have sufficient orders on both sides, skip API calls
            return

        # EARLY EXIT #2: If all new quotes matched existing orders, skip
        all_matched = (len(matched_bid_indices) == len(new_bids)) and (
            len(matched_ask_indices) == len(new_asks)
        )
        if all_matched and no_cancels_needed:
            return

        # THROTTLE: Don't update too frequently unless significant changes needed
        time_since_last_update = now - self._last_actual_update
        cancels_urgent = len(orders_to_cancel) >= self.MAX_PENDING_CANCELS_BEFORE_BATCH
        time_ok = time_since_last_update >= self.MIN_SECONDS_BETWEEN_UPDATES

        if not time_ok and not cancels_urgent:
            # Not enough time passed and cancels aren't urgent - skip
            return

        # Calculate how many orders we need to place (accounting for matched orders)
        bids_needed = len(new_bids) - len(matched_bid_indices)
        asks_needed = len(new_asks) - len(matched_ask_indices)

        # If nothing to do, skip
        if bids_needed == 0 and asks_needed == 0 and len(orders_to_cancel) == 0:
            return

        # Place only unmatched new orders (limit to what we actually need)
        bids_placed = 0
        for i, quote in enumerate(new_bids):
            if i not in matched_bid_indices and bids_placed < bids_needed:
                orders_to_place.append(
                    OrderRequest(
                        symbol=self.symbol,
                        side=OrderSide.BUY,
                        size=quote.size,
                        price=quote.price,
                        time_in_force=TimeInForce.ALO,  # Post-only
                    )
                )
                bids_placed += 1

        asks_placed = 0
        for i, quote in enumerate(new_asks):
            if i not in matched_ask_indices and asks_placed < asks_needed:
                orders_to_place.append(
                    OrderRequest(
                        symbol=self.symbol,
                        side=OrderSide.SELL,
                        size=quote.size,
                        price=quote.price,
                        time_in_force=TimeInForce.ALO,  # Post-only
                    )
                )
                asks_placed += 1

        # FINAL CHECK: If nothing to place and nothing to cancel, skip
        if not orders_to_place and not orders_to_cancel:
            return
        
        # FIFO Order Management: Enforce order limits before placing new orders
        await self._enforce_order_limits()

        # OPTIMIZED: Use batch cancel for lower latency instead of individual cancels
        if orders_to_cancel:
            cancelled = await self.client.cancel_orders_batch(orders_to_cancel)
            self.metrics.quotes_cancelled += cancelled
            self.metrics.actions_today += cancelled

        # Place new orders in batch with fresh BBO validation
        if orders_to_place:
            # OPTIMIZATION: Use cached orderbook for BBO validation to avoid extra API call
            # Only fetch fresh BBO if cached orderbook is stale (> 0.5s old)
            now = time.time()
            use_cached_bbo = self._cached_orderbook and (now - self._orderbook_cache_time) < 0.5

            if (
                use_cached_bbo
                and self._cached_orderbook
                and self._cached_orderbook.best_bid
                and self._cached_orderbook.best_ask
            ):
                fresh_bid = self._cached_orderbook.best_bid
                fresh_ask = self._cached_orderbook.best_ask
                logger.debug("Using cached BBO for order validation")
            else:
                # Fallback: fetch fresh orderbook if cache is stale
                orderbook = await self.client.get_orderbook()
                if orderbook and orderbook.best_bid and orderbook.best_ask:
                    fresh_bid = orderbook.best_bid
                    fresh_ask = orderbook.best_ask
                    # Update cache
                    self._cached_orderbook = orderbook
                    self._orderbook_cache_time = now
                else:
                    logger.warning("Could not get BBO for order validation, skipping placement")
                    return

            # OPTIMIZATION #9: Minimal ALO margin - just enough to ensure post-only
            # The $50 margin was DESTROYING our strategy by pushing us $50 away from BBO!
            # ALO only requires we don't CROSS the spread, not that we're far from it.
            # $1 margin (1 tick) is sufficient to ensure ALO acceptance.
            ALO_MARGIN = 1.0  # Reduced from $50 to $1 - just 1 tick buffer

            validated_orders = []
            for req in orders_to_place:
                if req.side == OrderSide.BUY:
                    # Bid must be < fresh_ask to avoid crossing (not < fresh_bid!)
                    # We CAN quote AT the best bid, just not ABOVE the best ask
                    if req.price >= fresh_ask:
                        # Would cross, adjust to 1 tick below best ask
                        safe_price = fresh_ask - ALO_MARGIN
                        req = OrderRequest(
                            symbol=req.symbol,
                            side=req.side,
                            size=req.size,
                            price=round(safe_price),
                            time_in_force=req.time_in_force,
                        )
                    validated_orders.append(req)
                else:
                    # Ask must be > fresh_bid to avoid crossing (not > fresh_ask!)
                    # We CAN quote AT the best ask, just not BELOW the best bid
                    if req.price <= fresh_bid:
                        # Would cross, adjust to 1 tick above best bid
                        safe_price = fresh_bid + ALO_MARGIN
                        req = OrderRequest(
                            symbol=req.symbol,
                            side=req.side,
                            size=req.size,
                            price=round(safe_price),
                            time_in_force=req.time_in_force,
                        )
                    validated_orders.append(req)

            orders_to_place = validated_orders

            results = await self.client.place_orders_batch(orders_to_place)

            placed_bids = 0
            placed_asks = 0
            for order, request in zip(results, orders_to_place):
                if order:
                    quote = QuoteLevel(
                        price=request.price,
                        size=request.size,
                        side=request.side,
                        order_id=order.order_id,
                        created_at=time.time(),  # Track creation time
                    )
                    if request.side == OrderSide.BUY:
                        self.active_bids[order.order_id] = quote
                        placed_bids += 1
                    else:
                        self.active_asks[order.order_id] = quote
                        placed_asks += 1

                    self.metrics.quotes_sent += 1
                    self.metrics.actions_today += 1

            if placed_bids > 0 or placed_asks > 0:
                logger.info(
                    f"Order update: +{placed_bids} bids, +{placed_asks} asks, cancelled {len(orders_to_cancel)}"
                )
                self._last_actual_update = time.time()
        elif orders_to_cancel:
            # Only cancellations happened, still update timestamp
            self._last_actual_update = time.time()

    async def _cancel_all_quotes(self) -> None:
        """Cancel all active quotes."""
        await self.client.cancel_all_orders(self.symbol)
        cancelled_count = len(self.active_bids) + len(self.active_asks)
        self.active_bids.clear()
        self.active_asks.clear()
        self.metrics.quotes_cancelled += cancelled_count
        self.metrics.actions_today += cancelled_count

    async def _cancel_all_side(self, side: str) -> None:
        """
        Cancel all orders on one side from the exchange.

        OPTIMIZATION: Use cached order sync data when available to avoid extra API calls.

        Args:
            side: "buy" or "sell"
        """
        try:
            is_hip3 = self.symbol.upper() == 'US500'
            
            # For HIP-3, always use local tracking since API doesn't return orders
            # For standard perps, check if we have recent sync data
            now = time.time()
            recent_sync = hasattr(self, "_last_order_sync") and (now - self._last_order_sync) < 10.0

            if recent_sync or is_hip3:
                # Use local tracking
                if side == "buy":
                    orders_to_cancel = [(self.symbol, oid) for oid in self.active_bids.keys()]
                    order_count = len(self.active_bids)
                    self.active_bids.clear()
                else:
                    orders_to_cancel = [(self.symbol, oid) for oid in self.active_asks.keys()]
                    order_count = len(self.active_asks)
                    self.active_asks.clear()

                if orders_to_cancel:
                    cancelled = await self.client.cancel_orders_batch(orders_to_cancel)
                    self.metrics.quotes_cancelled += cancelled
                    self.metrics.actions_today += cancelled
                    logger.info(
                        f"Cancelled {cancelled}/{order_count} {side} orders (local tracking)"
                    )
                return

            # Fallback for standard perps: Get all open orders from exchange via info API
            from hyperliquid.info import Info
            from hyperliquid.utils import constants as C

            info = Info(C.MAINNET_API_URL)
            open_orders = info.open_orders(self.config.wallet_address)

            if not open_orders:
                return

            # Filter by side: 'B' for buy, 'A' for sell (Hyperliquid format)
            target_side = "B" if side == "buy" else "A"
            target_coin = self.symbol.replace('/', '')
            orders_to_cancel = [
                (self.symbol, o["oid"])
                for o in open_orders
                if o.get("side") == target_side and o.get("coin") == target_coin
            ]

            if side == "buy":
                self.active_bids.clear()
            else:
                self.active_asks.clear()

            if orders_to_cancel:
                cancelled = await self.client.cancel_orders_batch(orders_to_cancel)
                self.metrics.quotes_cancelled += cancelled
                self.metrics.actions_today += cancelled
                logger.info(f"Cancelled {cancelled} {side} orders for rebalancing")
        except Exception as e:
            logger.error(f"Error cancelling {side} orders: {e}")

    def _should_rebalance(self) -> bool:
        """
        Check if inventory rebalancing is needed.

        Rebalance triggers: SURVIVAL MODE
        - Time-based: every rebalance_interval (120s for SURVIVAL mode)
        - Critical: delta > 70% (immediate rebalance needed)

        Minimum cooldown of 60s between rebalances for SURVIVAL mode.
        """
        now = time.time()

        # Minimum 60s cooldown between rebalances for SURVIVAL mode
        time_since_last = now - self.last_rebalance_time
        if time_since_last < 60.0:
            return False

        # Time-based rebalance
        if time_since_last >= self.rebalance_interval:
            return True

        # IMPROVED: Critical inventory imbalance - more aggressive threshold
        # If delta > 30% (reduced from 70%), force rebalance
        if abs(self.inventory.delta) > 0.30:
            logger.warning(f"Force rebalance triggered: delta={self.inventory.delta:.3f} > 30%")
            return True

        return False

    async def _rebalance_inventory(self, risk_metrics: RiskMetrics) -> None:
        """
        Rebalance inventory to reduce delta exposure.

        Uses one-sided quoting with priority cancellation.
        When critically imbalanced (>100%), uses IOC taker orders with fresh BBO.
        """
        self.last_rebalance_time = time.time()

        await self._refresh_inventory()

        if self.inventory.is_balanced:
            return

        logger.info(f"Rebalancing inventory, delta: {self.inventory.delta:.3f}")

        # IMPROVED: Critical imbalance - lower threshold to 50% (from 100%)
        # This bypasses WebSocket lag issues and prevents position accumulation
        abs_delta = abs(self.inventory.delta)
        if abs_delta > 0.50:  # > 50% imbalance (reduced from 100% to act faster)
            logger.warning(f"Critical imbalance detected: {abs_delta:.1%} - using aggressive rebalance")
            await self._aggressive_rebalance()
            return

        # Normal rebalance: Cancel quotes on the side we don't want using batch cancel
        if self.inventory.delta > 0:  # Long, want to sell
            # Cancel all bids, keep asks - use batch cancel for lower latency
            orders_to_cancel = [(self.symbol, oid) for oid in self.active_bids.keys()]
            if orders_to_cancel:
                await self.client.cancel_orders_batch(orders_to_cancel)
            self.active_bids.clear()
        else:  # Short, want to buy
            # Cancel all asks, keep bids - use batch cancel for lower latency
            orders_to_cancel = [(self.symbol, oid) for oid in self.active_asks.keys()]
            if orders_to_cancel:
                await self.client.cancel_orders_batch(orders_to_cancel)
            self.active_asks.clear()

    async def _aggressive_rebalance(self) -> None:
        """
        Aggressive rebalance using IOC taker orders.

        Fetches fresh BBO via REST API and places crossing orders.
        Used when inventory imbalance exceeds 100%.
        """
        # Get fresh BBO from REST (not stale WebSocket)
        fresh_bbo = await self.client.get_fresh_bbo(self.symbol)
        if not fresh_bbo:
            logger.warning("Could not fetch fresh BBO for aggressive rebalance")
            return

        best_bid, best_ask = fresh_bbo

        # Calculate size to reduce position by 50%
        position_size = abs(self.inventory.position_size)
        rebalance_size = position_size * 0.5

        # Minimum size check ($10 notional)
        min_size = max(0.00012, 10.5 / best_bid)
        if rebalance_size < min_size:
            rebalance_size = min_size

        # Round to lot size
        rebalance_size = round(rebalance_size, 5)

        if self.inventory.delta > 0:  # Long, need to sell
            # Place sell IOC at best_bid - 1 (cross the spread aggressively)
            ioc_price = best_bid - 1
            logger.info(
                f"IOC SELL rebalance: {rebalance_size} @ {ioc_price} (fresh BBO: {best_bid}@{best_ask})"
            )
            await self.client.place_ioc_order(
                self.symbol, OrderSide.SELL, rebalance_size, ioc_price
            )
        else:  # Short, need to buy
            # Place buy IOC at best_ask + 1 (cross the spread aggressively)
            ioc_price = best_ask + 1
            logger.info(
                f"IOC BUY rebalance: {rebalance_size} @ {ioc_price} (fresh BBO: {best_bid}@{best_ask})"
            )
            await self.client.place_ioc_order(self.symbol, OrderSide.BUY, rebalance_size, ioc_price)

    async def _refresh_inventory(self) -> None:
        """Refresh inventory state from exchange including real account value."""
        position = await self.client.get_position()

        if position:
            self.inventory.position_size = position.size
            self.inventory.position_value = position.notional_value
            self.inventory.entry_price = position.entry_price
            self.inventory.mark_price = position.mark_price
            self.unrealized_pnl = position.unrealized_pnl

            # Calculate delta as signed_notional / equity (directional exposure)
            # This is the standard equity-based delta: how much of our equity is at risk
            # Use position.size * mark_price to preserve sign (short = negative)
            equity = self.risk_manager.state.starting_equity or 1000.0  # Fallback to $1000
            signed_notional = position.size * position.mark_price
            self.inventory.delta = signed_notional / equity if equity > 0 else 0
        else:
            self.inventory = InventoryState()
            self.unrealized_pnl = 0.0

        # Fetch real account value from REST API for position tracking (NOT equity tracking)
        account_state = await self.client.get_account_state()
        if account_state:
            # HARDCODED EQUITY TRACKING: Use $1000 + realized PnL from trade_tracker
            # This ensures performance metrics are based on $1000 starting capital
            realized_pnl = self.trade_tracker.data.get("realized_pnl", 0.0)
            self.current_equity = self.starting_equity + realized_pnl
            
            # Update metrics with PnL from $1000 base
            self.metrics.net_pnl = self.current_equity - self.starting_equity

    def _on_orderbook_update(self, orderbook: OrderBook) -> None:
        """Handle orderbook update (callback)."""
        self.last_orderbook = orderbook
        if orderbook.mid_price:
            self.last_mid_price = orderbook.mid_price
            self.price_buffer.append(orderbook.mid_price)

    def _on_user_update(self, data: Dict) -> None:
        """Handle user update (fills, order changes)."""
        fills = data.get("fills", [])

        for fill in fills:
            self._process_fill(fill)

    def _process_fill(self, fill: Dict) -> None:
        """
        Process a fill event with enhanced tracking.

        OPTIMIZATION #9: Added detailed fill logging to track:
        - Fill price vs mid price (how far from fair value)
        - Per-fill P&L estimation
        - Buy vs Sell balance for spread tracking
        """
        try:
            oid = str(fill.get("oid", ""))
            side = fill.get("side", "")  # 'B' for buy, 'A' for sell
            size = float(fill.get("sz", 0))
            price = float(fill.get("px", 0))
            fee = float(fill.get("fee", 0))
            closed_pnl = float(fill.get("closedPnl", 0))

            is_maker = fill.get("liquidation") != True and fill.get("crossed") != True

            # Update metrics
            self.metrics.quotes_filled += 1
            self.metrics.total_volume += size * price

            if is_maker:
                self.metrics.maker_volume += size * price
                rebate = size * price * self.MAKER_REBATE
                self.metrics.rebates_earned += rebate
            else:
                self.metrics.taker_volume += size * price

            self.metrics.fees_paid += abs(fee)
            self.metrics.gross_pnl += closed_pnl

            # Remove from active quotes
            if oid in self.active_bids:
                del self.active_bids[oid]
            if oid in self.active_asks:
                del self.active_asks[oid]

            # Update last trade price
            self.last_trade_price = price

            # OPT#14: Track fill prices AND SIZES for weighted adverse selection detection
            fill_side = OrderSide.BUY if side == "B" else OrderSide.SELL
            self.metrics.add_fill(fill_side, price, size)  # NOW INCLUDES SIZE

            # Calculate slippage from mid price
            current_mid = self.price_buffer.get_last() if self.price_buffer.get_last() else price
            slippage = price - current_mid if side == "B" else current_mid - price

            # Net P&L for this fill
            net_pnl = closed_pnl - abs(fee) + (size * price * self.MAKER_REBATE if is_maker else 0)

            # Log fill with details
            side_str = "BUY" if side == "B" else "SELL"
            status = "✅" if net_pnl > 0 else "❌" if net_pnl < 0 else "⚪"
            logger.info(
                f"FILL {status}: {side_str} {size:.5f} @ ${price:,.0f} | "
                f"Slip: ${slippage:+.0f} | Fee: ${fee:.4f} | "
                f"PnL: ${closed_pnl:+.4f} | Net: ${net_pnl:+.4f} | "
                f"{'maker' if is_maker else 'TAKER'}"
            )
            
            # Log fill to trade tracker and verify order count
            self.trade_tracker.log_fill(fill)
            self.trade_tracker.update_equity(self.current_equity)
            
            # VERIFY: Check order count matches exchange after each fill
            expected_bids = len(self.active_bids)
            expected_asks = len(self.active_asks)
            if not self.trade_tracker.verify_order_count(expected_bids, expected_asks):
                logger.warning("Order count mismatch detected after fill - will sync on next iteration")
                # Reset last sync time to force sync on next run_iteration
                self._last_order_sync = 0

        except Exception as e:
            logger.error(f"Error processing fill: {e}")

    def _reset_daily_metrics(self) -> None:
        """Reset daily action counter."""
        self.metrics.actions_today = 0
        self.metrics.last_reset = get_timestamp_ms()

    def get_metrics(self) -> StrategyMetrics:
        """Get current strategy metrics."""
        # Net PnL is now updated from real account value in _refresh_inventory()
        # The formula below is a fallback if account refresh hasn't happened yet
        if self.metrics.net_pnl == 0:
            self.metrics.net_pnl = (
                self.metrics.gross_pnl + self.metrics.rebates_earned - self.metrics.fees_paid
            )
        return self.metrics

    def get_status(self) -> Dict:
        """Get current strategy status including real account metrics."""
        # Calculate API efficiency
        total_orderbook_calls = (
            self._api_call_metrics["orderbook_fetches"]
            + self._api_call_metrics["orderbook_cache_hits"]
        )
        cache_hit_rate = (
            (self._api_call_metrics["orderbook_cache_hits"] / total_orderbook_calls * 100)
            if total_orderbook_calls > 0
            else 0
        )

        return {
            "state": self.state.value,
            "inventory": {
                "position_size": self.inventory.position_size,
                "delta": self.inventory.delta,
                "is_balanced": self.inventory.is_balanced,
            },
            "quotes": {
                "active_bids": len(self.active_bids),
                "active_asks": len(self.active_asks),
            },
            "metrics": {
                "fill_rate": self.metrics.fill_rate,
                "actions_today": self.metrics.actions_today,
                "net_pnl": self.metrics.net_pnl,
            },
            "account": {
                "starting_equity": self.starting_equity,
                "current_equity": self.current_equity,
                "unrealized_pnl": self.unrealized_pnl,
                "pnl_pct": (
                    ((self.current_equity / self.starting_equity) - 1) * 100
                    if self.starting_equity > 0
                    else 0
                ),
            },
            "funding_rate": self.funding_rate,
            "api_efficiency": {
                "orderbook_cache_hit_rate": round(cache_hit_rate, 1),
                "inventory_refreshes": self._api_call_metrics["inventory_refreshes"],
                "order_syncs": self._api_call_metrics["order_syncs"],
                "orderbook_fetches": self._api_call_metrics["orderbook_fetches"],
            },
        }
