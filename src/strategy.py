"""
Delta-Neutral Market Making Strategy
Core trading logic for the HFT bot.

WARNING: This is a high-frequency trading strategy using leverage.
It carries significant financial risk. Thoroughly test on testnet
before using real funds. Past performance does not guarantee future results.
"""

import asyncio
import time
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

    # OPT#14: Track recent fill prices AND SIZES for weighted adverse selection detection
    recent_buy_prices: List[float] = field(default_factory=list)
    recent_sell_prices: List[float] = field(default_factory=list)
    recent_buy_sizes: List[float] = field(default_factory=list)  # NEW: track sizes
    recent_sell_sizes: List[float] = field(default_factory=list)  # NEW: track sizes
    max_recent_fills: int = 20  # Track last 20 fills

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
        if not self.recent_buy_prices or not self.recent_sell_prices:
            return None

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
    """Current inventory state."""

    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    delta: float = 0.0  # Directional exposure

    @property
    def is_balanced(self) -> bool:
        """Check if inventory is roughly balanced."""
        # RELAXED to 10%: Allows two-sided quoting more often for better fill rates
        # 5% was too tight given minimum order sizes, causing constant rebalance mode
        return abs(self.delta) < 0.10  # Within 10% of neutral


class MarketMakingStrategy:
    """
    Delta-Neutral Market Making Strategy.

    Core Logic:
    1. Post bid and ask orders around the mid price
    2. Capture the spread when both sides fill
    3. Manage inventory to stay delta-neutral
    4. Adjust spreads based on volatility and inventory
    5. Dynamically manage leverage based on risk

    Features:
    - Post-only orders to earn maker rebates (0.003%)
    - Adaptive spread based on volatility
    - Inventory-based quote skewing
    - Funding rate awareness
    - Circuit breakers for risk management

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
    MAKER_REBATE = 0.00003  # 0.003% rebate (3 bp credit, varies by tier)
    TAKER_FEE = 0.00035  # 0.035% fee (can be lower with referral)

    # Order recycling thresholds (AGGRESSIVE - prevent unnecessary cancellations)
    # At $90k BTC: 0.25% = $225 tolerance - wide enough to handle normal volatility
    PRICE_TOLERANCE_PCT = 0.0025  # 0.25% - reuse order if within this range (~$225 at $90k)
    SIZE_TOLERANCE_PCT = 0.30  # 30% - reuse if size within this range
    MIN_ORDER_AGE_SECONDS = 30.0  # Don't replace orders younger than 30s

    # Skip update thresholds (when to skip API calls entirely)
    SKIP_UPDATE_IF_ALL_MATCHED = True  # If all orders recycled, skip update cycle
    MIN_SECONDS_BETWEEN_UPDATES = 15.0  # Minimum 15 seconds between actual order updates
    MAX_PENDING_CANCELS_BEFORE_BATCH = 3  # Only cancel if 3+ orders need it

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
        self.starting_equity: float = config.trading.collateral  # Starting capital
        self.current_equity: float = config.trading.collateral  # Current account value
        self.unrealized_pnl: float = 0.0  # Unrealized PnL from open positions

        # API call optimization metrics
        self._api_call_metrics = {
            "orderbook_fetches": 0,
            "orderbook_cache_hits": 0,
            "inventory_refreshes": 0,
            "order_syncs": 0,
        }

        # Active quotes
        self.active_bids: Dict[str, QuoteLevel] = {}
        self.active_asks: Dict[str, QuoteLevel] = {}

        # Price tracking
        self.price_buffer = CircularBuffer(500)
        self.last_trade_price = 0.0
        self.funding_rate = 0.0
        self.last_orderbook: Optional[OrderBook] = None
        self.last_mid_price: float = 0.0
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

            # Cancel any existing orders
            await self.client.cancel_all_orders(self.symbol)

            # Load initial state
            await self._refresh_inventory()

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

        OPTIMIZATION: Only sync when we detect potential discrepancies through:
        - Fill events that should have removed orders
        - Long periods without order updates
        - Explicit discrepancies found

        This reduces API calls from every 60s to only when needed.
        """
        now = time.time()

        # Only sync if we haven't had a successful order update in 5+ minutes
        # or if we suspect stale tracking (no updates in 10+ minutes)
        time_since_last_update = now - getattr(self, "_last_actual_update", 0)
        should_sync = time_since_last_update > 300.0  # 5 minutes

        if not should_sync:
            return

        logger.debug("Performing order synchronization check")
        self._api_call_metrics["order_syncs"] += 1

        try:
            # Get actual open orders from exchange
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
                if order.get("coin") == self.symbol.replace("/", ""):  # BTC for BTC/USD
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

            # Synchronize active orders with exchange every 60 seconds (reduced from 30)
            if now - getattr(self, "_last_order_sync", 0) > 60.0:
                await self._sync_active_orders()
                self._last_order_sync = now

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
        """Update bid and ask quotes."""
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
        spread_bps = self._calculate_spread(orderbook, risk_metrics)

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

    def _calculate_spread(self, orderbook: OrderBook, risk_metrics: RiskMetrics) -> float:
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

        # AGGRESSIVE ONE-SIDED: If imbalance >1.5%, quote only on reducing side (tightened from 2%)
        imbalance_threshold = 0.015  # 1.5% threshold for aggressive skew (reduced from 2%)
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
        recent_spread = self.metrics.get_recent_spread_bps()
        if recent_spread is not None:
            if recent_spread < -3.0:  # Severe adverse selection
                defensive_bps = 30.0  # 30 bps behind BBO (was $8 = 115 bps for US500!)
                self.order_levels = 1  # Only 1 level
                logger.debug(
                    f"OPT#17 DEFENSIVE: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, 1 level"
                )
            elif recent_spread < 1.0:  # Low profit threshold
                defensive_bps = 15.0  # 15 bps behind BBO
                self.order_levels = 2  # 2 levels
                logger.debug(
                    f"OPT#17 CAUTIOUS: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, 2 levels"
                )
            elif recent_spread > 10.0:  # NEW: Aggressive mode for very profitable spreads
                defensive_bps = 3.0  # 3 bps behind BBO (tight)
                self.order_levels = 5  # 5 levels for more liquidity
                logger.debug(
                    f"OPT#17 AGGRESSIVE: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, 5 levels"
                )
            else:  # Normal profitable range
                defensive_bps = 8.0  # 8 bps behind BBO
                self.order_levels = 3  # 3 levels
                logger.debug(
                    f"OPT#17 NORMAL: spread {recent_spread:.2f} bps → {defensive_bps:.0f} bps distance, 3 levels"
                )
        else:
            # Default when insufficient data
            defensive_bps = 8.0  # 8 bps
            self.order_levels = min(self.config.trading.order_levels, 3)
            logger.debug(f"OPT#17 DEFAULT: insufficient data → {defensive_bps:.0f} bps distance")
        
        # Convert bps to dollar distance
        defensive_distance = mid * (defensive_bps / 10000.0)

        # Apply defensive distance - quote BEHIND BBO
        bid_price = max(base_bid, orderbook.best_bid - defensive_distance)
        ask_price = min(base_ask, orderbook.best_ask + defensive_distance)

        # Round to tick size: $0.01 for US500, $1 for BTC
        tick_size = 0.01 if self.symbol.upper() == "US500" else 1.0
        bid_price = round_price(bid_price, tick_size)
        ask_price = round_price(ask_price, tick_size)

        # ENSURE MINIMUM SPREAD - OPT#13: 8 bps minimum
        min_spread_dollars = mid * (max(self.min_spread_bps, 8.0) / 10000)
        if ask_price - bid_price < min_spread_dollars:
            gap = min_spread_dollars - (ask_price - bid_price)
            bid_price -= gap / 2
            ask_price += gap / 2
            bid_price = round_price(bid_price, tick_size)
            ask_price = round_price(ask_price, tick_size)

        # OPTIMIZATION #14: ADAPTIVE DEFENSIVE DISTANCE
        # Distance is already set above based on recent spread (OPT#17)
        # This section just applies the distance - no need to recalculate
        # Note: defensive_distance was set in lines 820-850 based on spread profitability

        if orderbook.best_bid > 0:
            # Our bid should be BELOW best bid by defensive_distance
            max_bid = orderbook.best_bid - defensive_distance
            if bid_price > max_bid:
                bid_price = max_bid

        if orderbook.best_ask > 0:
            # Our ask should be ABOVE best ask by defensive_distance
            min_ask = orderbook.best_ask + defensive_distance
            if ask_price < min_ask:
                ask_price = min_ask

        # Final validation - never cross the spread
        if bid_price >= ask_price:
            mid_point = (bid_price + ask_price) / 2
            min_half_spread = mid * 0.0005  # 5 bps minimum spread
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
        recent_spread = self.metrics.get_recent_spread_bps()
        if recent_spread is not None and recent_spread < -2.0:
            # Adverse selection detected - reduce to 1 level
            effective_levels = 1
            logger.warning(
                f"Reducing to {effective_levels} level due to adverse selection "
                f"(recent spread: {recent_spread:.2f} bps)"
            )
        elif recent_spread is not None and recent_spread < 2.0:
            # Low profitability - reduce to 2 levels
            effective_levels = min(2, self.order_levels)
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
        # With 15% pyramiding reduction, level 2 is 70% of base
        # Scale up base_size so smallest level still meets minimum
        levels_pyramid_factor = 1 - (effective_levels - 1) * 0.15  # Adjusted for effective_levels
        required_base = min_size / max(
            levels_pyramid_factor, 0.5
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
            level_size = effective_base * (1 - i * 0.15)
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
        orders_to_cancel = []
        orders_to_place = []
        now = time.time()

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
            # OPTIMIZATION: Check if we have recent sync data (< 10 seconds old)
            now = time.time()
            recent_sync = hasattr(self, "_last_order_sync") and (now - self._last_order_sync) < 10.0

            if recent_sync:
                # Use local tracking instead of API call
                if side == "buy":
                    orders_to_cancel = [(self.symbol, oid) for oid in self.active_bids.keys()]
                    self.active_bids.clear()
                else:
                    orders_to_cancel = [(self.symbol, oid) for oid in self.active_asks.keys()]
                    self.active_asks.clear()

                if orders_to_cancel:
                    cancelled = await self.client.cancel_orders_batch(orders_to_cancel)
                    self.metrics.quotes_cancelled += cancelled
                    self.metrics.actions_today += cancelled
                    logger.info(
                        f"Cancelled {cancelled} {side} orders for rebalancing (using cached data)"
                    )
                return

            # Fallback: Get all open orders from exchange via info API
            from hyperliquid.info import Info
            from hyperliquid.utils import constants as C

            info = Info(C.MAINNET_API_URL)
            open_orders = info.open_orders(self.config.wallet_address)

            if not open_orders:
                return

            # Filter by side: 'B' for buy, 'A' for sell (Hyperliquid format)
            target_side = "B" if side == "buy" else "A"
            orders_to_cancel = [
                (self.symbol, o["oid"])
                for o in open_orders
                if o.get("side") == target_side and o.get("coin") == "BTC"
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

        # Critical inventory imbalance: delta > 70% needs faster action
        if abs(self.inventory.delta) > 0.70:
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

        # Critical imbalance: use IOC taker orders with fresh REST BBO
        # This bypasses WebSocket lag issues
        abs_delta = abs(self.inventory.delta)
        if abs_delta > 1.0:  # > 100% imbalance
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

        # Fetch real account value from REST API for accurate PnL tracking
        account_state = await self.client.get_account_state()
        if account_state:
            self.current_equity = account_state.equity
            # Update metrics with real PnL from account value
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
