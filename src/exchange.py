"""
Hyperliquid Exchange Client for AMM-500
Handles all API interactions with the Hyperliquid exchange for US500 trading.

This module is optimized for US500 (S&P 500 Index) perpetual trading
via the KM deployer's permissionless market (km:US500).

US500 Specific Notes:
- Symbol format: "US500" for API calls (not km:US500)
- Max leverage: 25x (KM deployer limit)
- Isolated margin only
- Tick size and lot size may differ from BTC

WARNING: This module handles real trading operations.
Always test thoroughly on testnet before using with real funds.
Financial losses can occur with algorithmic trading.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import websockets
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import Config
from .utils import (
    LatencyTracker,
    RateLimiter,
    get_timestamp_ms,
    retry_async,
    round_price,
    round_size,
)

# Try to import the Hyperliquid SDK
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning(
        "hyperliquid-python-sdk not installed. " "Install with: pip install hyperliquid-python-sdk"
    )


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""

    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(Enum):
    """Time in force options."""

    GTC = "Gtc"  # Good til cancelled
    IOC = "Ioc"  # Immediate or cancel
    ALO = "Alo"  # Add liquidity only (post-only)


@dataclass
class OrderRequest:
    """Order request structure."""

    symbol: str
    side: OrderSide
    size: float
    price: Optional[float] = None  # None for market orders
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.ALO  # Default to post-only
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass
class Order:
    """Order structure."""

    order_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
    filled_size: float = 0.0
    status: str = "open"
    timestamp: int = 0
    client_order_id: Optional[str] = None

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def is_filled(self) -> bool:
        return self.filled_size >= self.size

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class Position:
    """Position structure."""

    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    leverage: int
    margin_used: float

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def notional_value(self) -> float:
        return abs(self.size) * self.mark_price


@dataclass
class OrderBook:
    """Local order book snapshot."""

    symbol: str
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bids[0][1] if self.bids else None

    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0][1] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None


@dataclass
class AccountState:
    """Account state structure."""

    equity: float  # Total balance (including held funds)
    available_balance: float  # Available for new orders (equity - hold)
    margin_used: float
    unrealized_pnl: float
    hold_balance: float = 0.0  # Funds held by open orders
    positions: List[Position] = field(default_factory=list)
    open_orders: List[Order] = field(default_factory=list)

    @property
    def margin_ratio(self) -> float:
        if self.equity == 0:
            return 0.0
        return self.margin_used / self.equity

    @property
    def net_exposure(self) -> float:
        return sum(p.size * p.mark_price for p in self.positions)
    
    @property
    def total_balance(self) -> float:
        """Total balance including held funds - use for equity tracking."""
        return self.equity


class HyperliquidClient:
    """
    Async client for Hyperliquid exchange.

    Handles:
    - REST API calls for orders and account info
    - WebSocket connections for real-time data
    - Rate limiting and retries
    - Order batching for efficiency

    Usage:
        config = Config.load()
        client = HyperliquidClient(config)
        await client.connect()

        # Place order
        order = await client.place_order(OrderRequest(...))

        # Get order book
        book = await client.get_orderbook("US500")

        await client.disconnect()
    """

    # US500 tick size and lot size on Hyperliquid (KM deployer)
    # US500 prices use $0.01 tick size (index points with 2 decimals)
    # Order sizing parameters
    # For BTC perpetual: min size is 0.0001 BTC
    # For US500 futures: szDecimals=1 means min size is 0.1 contracts
    TICK_SIZE = 0.01
    LOT_SIZE = 0.0001  # Default: 0.0001 BTC minimum for BTC perpetual
    
    # Symbol-specific lot sizes based on szDecimals from Hyperliquid API
    # szDecimals defines the precision: 10^(-szDecimals) is the minimum increment
    SYMBOL_LOT_SIZES = {
        "US500": 0.1,    # szDecimals=1 -> 10^(-1) = 0.1
        "SPX": 0.1,      # Same asset, different name
        "BTC": 0.0001,   # szDecimals=4 -> 10^(-4) = 0.0001
        "ETH": 0.001,    # szDecimals=3 -> 10^(-3) = 0.001
    }
    
    def _get_lot_size(self, symbol: str) -> float:
        """Get the minimum lot size for a symbol based on szDecimals."""
        sym = symbol.upper().replace("KM:", "")
        return self.SYMBOL_LOT_SIZES.get(sym, self.LOT_SIZE)

    def _get_api_symbol(self, symbol: str) -> str:
        """Convert symbol to API format for HIP-3 perps."""
        # HIP-3 perps like US500 need km: prefix for API calls
        if symbol.upper() == "US500":
            return f"km:{symbol}"
        return symbol

    def __init__(self, config: Config):
        """Initialize the client."""
        self.config = config
        self._exchange: Optional[Exchange] = None
        self._info: Optional[Info] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._connected = False

        # Rate limiter: SURVIVAL MODE - recovering from Hyperliquid rate limit block
        # Only 10 requests per minute max (1 per 6 seconds)
        self._rate_limiter = RateLimiter(rate=10, per=60.0)  # 1 per 6 seconds max

        # Global rate limit backoff state
        self._rate_limit_backoff = 0.0  # Seconds to wait before next request
        self._last_429_time = 0.0  # When we last hit a 429
        self._consecutive_429s = 0  # Track consecutive 429 errors

        # API call cooldowns - EXTREME SURVIVAL MODE
        self._last_api_calls: Dict[str, float] = {}
        self._api_cooldowns = {
            "account_state": 60.0,  # 60 seconds between account state refreshes
            "funding_rate": 900.0,  # 15 minutes between funding rate checks
            "leverage": 600.0,  # 10 minutes between leverage changes
            "orderbook": 30.0,  # 30s between orderbook fetches in REST mode
        }

        # Latency tracking
        self._order_latency = LatencyTracker("Order")
        self._ws_latency = LatencyTracker("WebSocket")

        # Local state
        self._orderbook: Dict[str, OrderBook] = {}
        self._positions: Dict[str, Position] = {}
        self._open_orders: Dict[str, Order] = {}
        self._account_state: Optional[AccountState] = None

        # Exchange BBO tracking for staleness detection
        self._last_exchange_bbo: Optional[tuple] = None  # (bid, ask)
        self._last_exchange_bbo_time: int = 0

        # Callbacks for WebSocket updates
        self._orderbook_callbacks: List[Callable[[OrderBook], None]] = []
        self._trade_callbacks: List[Callable[[Dict], None]] = []
        self._user_callbacks: List[Callable[[Dict], None]] = []

        # Session for HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None

        # Paper trading simulation state
        self._paper_position_size: float = 0.0
        self._paper_entry_price: float = 0.0
        self._paper_realized_pnl: float = 0.0
        self._paper_equity: float = config.trading.collateral
        self._paper_fills: int = 0
        self._paper_last_mid: float = 0.0

    def _check_cooldown(self, api_name: str) -> bool:
        """Check if an API call is allowed based on cooldown and backoff."""
        import time

        now = time.time()

        # Check global rate limit backoff first
        if self._rate_limit_backoff > 0:
            time_since_429 = now - self._last_429_time
            if time_since_429 < self._rate_limit_backoff:
                return False  # Still in backoff period
            else:
                # Backoff period over, reset
                self._rate_limit_backoff = 0.0
                self._consecutive_429s = 0

        cooldown = self._api_cooldowns.get(api_name, 2.0)  # Default 2s
        last_call = self._last_api_calls.get(api_name, 0)

        if now - last_call < cooldown:
            return False

        self._last_api_calls[api_name] = now
        return True

    def _handle_rate_limit(self) -> None:
        """Handle a 429 rate limit error with exponential backoff."""
        import time

        self._last_429_time = time.time()
        self._consecutive_429s += 1

        # Exponential backoff: 2, 4, 8, 16, 32 seconds max
        self._rate_limit_backoff = min(2**self._consecutive_429s, 32)

        logger.warning(f"Rate limited (429). Backing off for {self._rate_limit_backoff}s")

    @retry(
        stop=stop_after_attempt(5),  # Increased from 3 to 5 retries for better reliability
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
        reraise=True,
    )
    async def connect(self) -> None:
        """
        Connect to the exchange.

        Initializes SDK clients and WebSocket connection.
        """
        if self._connected:
            logger.warning("Already connected")
            return

        if not SDK_AVAILABLE:
            raise RuntimeError(
                "Hyperliquid SDK not available. " "Install with: pip install hyperliquid-python-sdk"
            )

        try:
            # Initialize HTTP session with connection pooling for lower latency
            connector = aiohttp.TCPConnector(
                limit=20,  # Connection pool size
                limit_per_host=10,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                keepalive_timeout=30,  # Keep connections alive
            )
            timeout = aiohttp.ClientTimeout(
                total=10,
                connect=3,  # Fast connection timeout
                sock_read=5,
            )
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            # Initialize SDK clients
            # Create wallet from private key using eth-account
            from eth_account import Account

            wallet = Account.from_key(self.config.private_key)

            base_url = self.config.network.api_url
            
            # Include HIP-3 perp DEXs (km for km:US500)
            perp_dexs = ['km'] if self.config.trading.symbol.upper() == 'US500' else None

            self._info = Info(base_url=base_url, skip_ws=True, perp_dexs=perp_dexs)

            # If the wallet address differs from the private key's address,
            # we're using an API wallet and need to specify account_address
            derived_address = wallet.address.lower()
            configured_address = self.config.wallet_address.lower()

            if derived_address != configured_address:
                # Using API wallet - specify account_address
                logger.info(
                    f"Using API wallet {derived_address[:10]}... for account {configured_address[:10]}..."
                )
                self._exchange = Exchange(
                    wallet=wallet,
                    base_url=base_url,
                    account_address=self.config.wallet_address,
                    perp_dexs=perp_dexs,
                )
            else:
                # Using main wallet directly
                self._exchange = Exchange(
                    wallet=wallet,
                    base_url=base_url,
                    perp_dexs=perp_dexs,
                )

            # Set leverage (may fail if wallet not yet funded on testnet)
            leverage_ok = await self._set_leverage(
                self.config.trading.symbol, self.config.trading.leverage
            )
            if not leverage_ok:
                logger.warning(
                    f"Could not set leverage to {self.config.trading.leverage}x - "
                    "wallet may need funding on testnet"
                )

            # Try WebSocket connection, but continue in REST-only mode if it fails
            try:
                await self._connect_websocket()
            except Exception as ws_err:
                logger.warning(f"WebSocket connection failed: {ws_err}")
                logger.info("Continuing in REST-only mode - will poll orderbook via REST API")
                self._ws = None
                self._ws_task = None

            # Load initial state
            await self._refresh_account_state()

            self._connected = True
            logger.info(
                f"Connected to Hyperliquid "
                f"({'testnet' if self.config.network.testnet else 'mainnet'})"
            )

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        self._connected = False

        # Cancel WebSocket task
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Disconnected from Hyperliquid")

    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection with optimized settings."""
        ws_url = self.config.network.ws_url

        try:
            self._ws = await websockets.connect(
                ws_url,
                ping_interval=20,  # Match Hyperliquid's expected ping interval
                ping_timeout=10,
                close_timeout=5,
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression for lower latency
            )

            # Small delay to let connection stabilize before subscribing
            await asyncio.sleep(0.1)

            # Subscribe to channels with small delays between each
            await self._subscribe_orderbook(self.config.trading.symbol)
            await asyncio.sleep(0.05)
            await self._subscribe_trades(self.config.trading.symbol)
            await asyncio.sleep(0.05)
            await self._subscribe_user_events()

            # Start message handler
            self._ws_task = asyncio.create_task(self._ws_message_handler())

            logger.info(f"WebSocket connected to {ws_url}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def _subscribe_orderbook(self, symbol: str) -> None:
        """Subscribe to order book updates."""
        if not self._ws:
            return

        msg = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": symbol,
            },
        }
        await self._ws.send(json.dumps(msg))
        logger.debug(f"Subscribed to {symbol} order book")

    async def _subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade updates."""
        if not self._ws:
            return

        msg = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": symbol,
            },
        }
        await self._ws.send(json.dumps(msg))
        logger.debug(f"Subscribed to {symbol} trades")

    async def _subscribe_user_events(self) -> None:
        """Subscribe to user-specific events (fills, orders)."""
        if not self._ws:
            return

        msg = {
            "method": "subscribe",
            "subscription": {
                "type": "userEvents",
                "user": self.config.wallet_address,
            },
        }
        await self._ws.send(json.dumps(msg))
        logger.debug("Subscribed to user events")

    async def _ws_message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                self._ws_latency.start()

                try:
                    data = json.loads(message)
                    await self._process_ws_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from WebSocket: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

                self._ws_latency.stop()

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self._connected:
                # Attempt to reconnect
                await self._reconnect_websocket()
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")

    async def _reconnect_websocket(self) -> None:
        """Attempt to reconnect WebSocket with exponential backoff.

        SURVIVAL MODE: Disabled reconnection to avoid rate limit issues.
        The bot will run in REST-only mode.
        """
        # SURVIVAL MODE: Don't attempt reconnection - just log and continue
        # WebSocket is blocked with "100 connections" error
        logger.warning("WebSocket reconnection DISABLED (SURVIVAL MODE) - running REST-only")

        # Clean up old connections
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        # Don't attempt to reconnect - just return

    async def _process_ws_message(self, data: Dict) -> None:
        """Process a WebSocket message."""
        channel = data.get("channel")

        if channel == "l2Book":
            await self._handle_orderbook_update(data.get("data", {}))
        elif channel == "trades":
            await self._handle_trade_update(data.get("data", []))
        elif channel == "userEvents":
            await self._handle_user_update(data.get("data", {}))

    async def _handle_orderbook_update(self, data: Dict) -> None:
        """Handle order book update."""
        coin = data.get("coin", self.config.trading.symbol)
        levels = data.get("levels", [[], []])

        # Parse bids and asks
        bids = [(float(l["px"]), float(l["sz"])) for l in levels[0]]
        asks = [(float(l["px"]), float(l["sz"])) for l in levels[1]]

        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        # Check if this WebSocket update is stale compared to a recent exchange correction
        # This happens when failed ALO orders tell us the real BBO but WebSocket is behind
        if coin in self._orderbook and bids and asks:
            old_book = self._orderbook[coin]
            ws_bid = bids[0][0] if bids else 0
            ws_ask = asks[0][0] if asks else 0

            # If we recently got a correction from the exchange (within last 10 seconds)
            # and this WebSocket update is significantly different (>$25), reject it
            now = get_timestamp_ms()
            if hasattr(self, "_last_exchange_bbo_time") and self._last_exchange_bbo_time > 0:
                time_since_correction = now - self._last_exchange_bbo_time
                if time_since_correction < 10000:  # 10 seconds window
                    if hasattr(self, "_last_exchange_bbo") and self._last_exchange_bbo:
                        exc_bid, exc_ask = self._last_exchange_bbo
                        diff = abs(ws_bid - exc_bid)
                        if diff > 25:
                            logger.debug(
                                f"Rejecting stale WS BBO {ws_bid}@{ws_ask}, exchange was {exc_bid}@{exc_ask} ({time_since_correction:.0f}ms ago)"
                            )
                            return  # Skip this stale update

        # Limit depth
        depth = self.config.performance.orderbook_depth
        bids = bids[:depth]
        asks = asks[:depth]

        # Update local order book
        self._orderbook[coin] = OrderBook(
            symbol=coin,
            bids=bids,
            asks=asks,
            timestamp=get_timestamp_ms(),
        )

        # Notify callbacks
        for callback in self._orderbook_callbacks:
            try:
                callback(self._orderbook[coin])
            except Exception as e:
                logger.error(f"Order book callback error: {e}")

    async def _handle_trade_update(self, trades: List[Dict]) -> None:
        """Handle trade updates."""
        for trade in trades:
            for callback in self._trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")

    async def _handle_user_update(self, data: Dict) -> None:
        """Handle user-specific updates (fills, order updates)."""
        for callback in self._user_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"User callback error: {e}")

        # Refresh account state on fills
        if data.get("fills"):
            await self._refresh_account_state()

    def on_orderbook_update(self, callback: Callable[[OrderBook], None]) -> None:
        """Register callback for order book updates."""
        self._orderbook_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_user_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for user updates."""
        self._user_callbacks.append(callback)

    # =========================================================================
    # Order Management
    # =========================================================================

    async def place_order(self, request: OrderRequest) -> Optional[Order]:
        """
        Place a single order.

        Args:
            request: Order request details

        Returns:
            Order object if successful, None otherwise
        """
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")

        await self._rate_limiter.acquire(weight=1)
        self._order_latency.start()

        try:
            # Get symbol-specific lot size
            lot_size = self._get_lot_size(request.symbol)
            
            # Round price and size
            price = round_price(request.price, self.TICK_SIZE) if request.price else None
            size = round_size(request.size, lot_size)

            # For HIP-3 perps (US500), szDecimals=1 means min size is 0.1 contracts
            if request.symbol.upper() == "US500":
                min_size = 0.1  # szDecimals=1 -> 10^(-1) = 0.1
                if size < min_size:
                    logger.warning(
                        f"Order size {size:.4f} < min {min_size} for US500 - rejecting"
                    )
                    return None
            else:
                if size < lot_size:
                    logger.warning(f"Order size {request.size} too small, minimum is {lot_size}")
                    return None

            # Build order with API symbol (km: prefix for HIP-3 perps)
            api_symbol = self._get_api_symbol(request.symbol)
            is_buy = request.side == OrderSide.BUY

            order_result = self._exchange.order(
                coin=api_symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": request.time_in_force.value}},
                reduce_only=request.reduce_only,
            )

            latency = self._order_latency.stop()

            # Handle both dict and string responses from SDK
            if isinstance(order_result, str):
                logger.warning(f"Order returned string response: {order_result}")
                return None

            if order_result.get("status") == "ok":
                response = order_result.get("response", {})
                order_data = response.get("data", {})

                # Extract order details
                statuses = order_data.get("statuses", [{}])
                if statuses and statuses[0].get("resting"):
                    resting = statuses[0]["resting"]
                    order = Order(
                        order_id=str(resting.get("oid", "")),
                        symbol=request.symbol,
                        side=request.side,
                        size=size,
                        price=price or 0.0,
                        status="open",
                        timestamp=get_timestamp_ms(),
                        client_order_id=request.client_order_id,
                    )
                    self._open_orders[order.order_id] = order

                    logger.debug(
                        f"Order placed: {request.side.value} {size} {request.symbol} "
                        f"@ {price} (latency: {latency:.1f}ms)"
                    )
                    return order
                elif statuses and statuses[0].get("filled"):
                    filled = statuses[0]["filled"]
                    order = Order(
                        order_id=str(filled.get("oid", "")),
                        symbol=request.symbol,
                        side=request.side,
                        size=size,
                        price=price or 0.0,
                        filled_size=size,
                        status="filled",
                        timestamp=get_timestamp_ms(),
                        client_order_id=request.client_order_id,
                    )
                    logger.debug(f"Order filled immediately: {order}")
                    return order
            else:
                error = (
                    order_result.get("response", {}).get("data", {}).get("error", "Unknown error")
                    if isinstance(order_result, dict)
                    else str(order_result)
                )
                logger.error(f"Order failed: {error}")
                return None

        except Exception as e:
            self._order_latency.stop()
            logger.error(f"Error placing order: {e}")
            return None

    async def place_orders_batch(self, requests: List[OrderRequest]) -> List[Optional[Order]]:
        """
        Place multiple orders in a single batch.

        Hyperliquid supports up to 40 orders per batch for efficiency.
        In paper trading mode, simulates order placement without real API calls.

        Args:
            requests: List of order requests

        Returns:
            List of Order objects (None for failed orders)
        """
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")

        if not requests:
            return []

        # Paper trading mode - simulate orders
        if self.config.execution.paper_trading:
            return await self._simulate_orders_batch(requests)

        # Limit batch size
        max_batch = self.config.execution.max_orders_per_batch
        if len(requests) > max_batch:
            logger.warning(f"Batch size {len(requests)} exceeds max {max_batch}, splitting")
            results = []
            for i in range(0, len(requests), max_batch):
                batch = requests[i : i + max_batch]
                results.extend(await self.place_orders_batch(batch))
            return results

        await self._rate_limiter.acquire(weight=len(requests))
        self._order_latency.start()

        try:
            # Build batch orders
            orders_data = []
            for req in requests:
                # Get symbol-specific lot size
                lot_size = self._get_lot_size(req.symbol)
                
                price = round_price(req.price, self.TICK_SIZE) if req.price else 0.0
                size = round_size(req.size, lot_size)

                # For HIP-3 perps (US500), szDecimals=1 means min size is 0.1 contracts
                # At $693 price, that's $69.30 minimum notional
                if req.symbol.upper() == "US500":
                    min_size = 0.1  # szDecimals=1 -> 10^(-1) = 0.1
                    if size < min_size:
                        logger.warning(
                            f"Order REJECTED: size={size:.4f} < min={min_size} for US500"
                        )
                        continue
                else:
                    if size < lot_size:
                        continue

                # Use API symbol (km: prefix for HIP-3 perps)
                api_symbol = self._get_api_symbol(req.symbol)
                
                orders_data.append(
                    {
                        "coin": api_symbol,
                        "is_buy": req.side == OrderSide.BUY,
                        "sz": size,
                        "limit_px": price,
                        "order_type": {"limit": {"tif": req.time_in_force.value}},
                        "reduce_only": req.reduce_only,
                    }
                )

            if not orders_data:
                return [None] * len(requests)

            # Place batch
            result = self._exchange.bulk_orders(orders_data)

            latency = self._order_latency.stop()
            logger.debug(f"Batch of {len(orders_data)} orders placed (latency: {latency:.1f}ms)")
            
            # Debug: Log full result for HIP-3 troubleshooting
            if requests and requests[0].symbol.upper() == "US500":
                logger.debug(f"HIP-3 order result: {result}")

            # Parse results
            orders = []
            if result.get("status") == "ok":
                response = result.get("response", {})
                statuses = response.get("data", {}).get("statuses", [])

                # Log order statuses for debugging
                resting_count = sum(1 for s in statuses if s.get("resting"))
                filled_count = sum(1 for s in statuses if s.get("filled"))
                failed_count = len(statuses) - resting_count - filled_count
                if resting_count or filled_count or failed_count:
                    logger.info(
                        f"Order batch result: {resting_count} resting, {filled_count} filled, {failed_count} failed"
                    )
                    # Debug: Log actual order IDs
                    for i, s in enumerate(statuses):
                        if s.get("resting"):
                            oid = s["resting"].get("oid", "?")
                            logger.debug(f"  Order {i}: OID={oid} RESTING")
                        elif s.get("filled"):
                            oid = s["filled"].get("oid", "?")
                            logger.debug(f"  Order {i}: OID={oid} FILLED")
                        else:
                            logger.debug(f"  Order {i}: FAILED - {s}")

                # Extract BBO from failed ALO orders and update internal orderbook
                # Error format: "Post only order would have immediately matched, bbo was 90901@90902. asset=0"
                for i, status in enumerate(statuses):
                    if not status.get("resting") and not status.get("filled"):
                        error_msg = status.get("error", "")
                        if "bbo was" in error_msg:
                            # Log the ALO failure with bbo info
                            logger.warning(f"Order {i} failed: {status}")
                            try:
                                import re

                                bbo_match = re.search(r"bbo was (\d+)@(\d+)", error_msg)
                                if bbo_match:
                                    exchange_bid = float(bbo_match.group(1))
                                    exchange_ask = float(bbo_match.group(2))

                                    # Store exchange BBO for staleness detection
                                    self._last_exchange_bbo = (exchange_bid, exchange_ask)
                                    self._last_exchange_bbo_time = get_timestamp_ms()

                                    # Update internal orderbook with exchange's actual BBO
                                    symbol = requests[0].symbol if requests else "BTC"
                                    if symbol in self._orderbook:
                                        old_book = self._orderbook[symbol]
                                        # Only update if significantly different (>$50 stale)
                                        if (
                                            old_book.best_bid
                                            and abs(old_book.best_bid - exchange_bid) > 50
                                        ):
                                            logger.debug(
                                                f"BBO stale: local {old_book.best_bid}@{old_book.best_ask} vs exchange {exchange_bid}@{exchange_ask}"
                                            )
                                            # Create updated orderbook with exchange BBO
                                            self._orderbook[symbol] = OrderBook(
                                                symbol=symbol,
                                                bids=[(exchange_bid, 1.0)],
                                                asks=[(exchange_ask, 1.0)],
                                                timestamp=get_timestamp_ms(),
                                            )
                                            # Notify callbacks about corrected BBO
                                            for callback in self._orderbook_callbacks:
                                                try:
                                                    callback(self._orderbook[symbol])
                                                except Exception as e:
                                                    pass
                                    break  # Only need to update once
                            except Exception:
                                pass  # Ignore parse errors
                        else:
                            logger.warning(f"Order {i} failed: {status}")

                for i, (req, status) in enumerate(zip(requests, statuses)):
                    if status.get("resting"):
                        order = Order(
                            order_id=str(status["resting"]["oid"]),
                            symbol=req.symbol,
                            side=req.side,
                            size=round_size(req.size, self.LOT_SIZE),
                            price=round_price(req.price, self.TICK_SIZE) if req.price else 0.0,
                            status="open",
                            timestamp=get_timestamp_ms(),
                        )
                        self._open_orders[order.order_id] = order
                        orders.append(order)
                    elif status.get("filled"):
                        order = Order(
                            order_id=str(status["filled"]["oid"]),
                            symbol=req.symbol,
                            side=req.side,
                            size=round_size(req.size, self.LOT_SIZE),
                            price=round_price(req.price, self.TICK_SIZE) if req.price else 0.0,
                            filled_size=round_size(req.size, self.LOT_SIZE),
                            status="filled",
                            timestamp=get_timestamp_ms(),
                        )
                        orders.append(order)
                    else:
                        orders.append(None)
            else:
                orders = [None] * len(requests)

            return orders

        except Exception as e:
            self._order_latency.stop()
            logger.error(f"Error placing batch orders: {e}")
            return [None] * len(requests)

    async def _simulate_orders_batch(self, requests: List[OrderRequest]) -> List[Optional[Order]]:
        """
        Simulate order placement for paper trading.

        Creates simulated orders that track with real market data.
        Orders are assumed to fill if price crosses.
        """
        import uuid

        orders = []
        for req in requests:
            # Get symbol-specific lot size
            lot_size = self._get_lot_size(req.symbol)
            
            price = round_price(req.price, self.TICK_SIZE) if req.price else 0.0
            size = round_size(req.size, lot_size)

            # For HIP-3 perps (US500), szDecimals=1 means min size is 0.1 contracts
            if req.symbol.upper() == "US500":
                min_size = 0.1  # szDecimals=1 -> 10^(-1) = 0.1
                if size < min_size:
                    logger.warning(f"[PAPER] Order size {size:.4f} < min {min_size} for US500")
                    orders.append(None)
                    continue
            else:
                if size < lot_size:
                    logger.warning(f"[PAPER] Order size {size:.8f} too small (min={lot_size})")
                    orders.append(None)
                    continue

            # Generate simulated order ID
            order_id = f"sim_{uuid.uuid4().hex[:12]}"

            order = Order(
                order_id=order_id,
                symbol=req.symbol,
                side=req.side,
                size=size,
                price=price,
                status="open",
                timestamp=get_timestamp_ms(),
                client_order_id=req.client_order_id,
            )

            self._open_orders[order_id] = order
            orders.append(order)

            # Debug: log order price
            logger.debug(f"[PAPER] Placed {order.side.value} @ {order.price:.1f}")

        if orders:
            logger.debug(f"[PAPER] Simulated {len([o for o in orders if o])} orders")

        # Check for fills on existing orders after placing new ones
        await self._simulate_fills()

        return orders

    async def _simulate_fills(self) -> int:
        """
        Simulate order fills for paper trading based on current market price.

        Returns number of fills.
        """
        if not self.config.execution.paper_trading:
            return 0

        # Get current orderbook
        symbol = self.config.trading.symbol
        book = await self.get_orderbook(symbol)
        if not book or not book.mid_price:
            return 0

        mid_price = book.mid_price
        self._paper_last_mid = mid_price
        fills = 0
        orders_to_remove = []

        # Debug: log order count and prices periodically
        if self._open_orders and len(self._open_orders) > 0:
            order_prices = [(o.side.value, o.price) for o in self._open_orders.values()]
            logger.debug(
                f"[PAPER] Checking {len(self._open_orders)} orders | bid={book.best_bid:.1f} ask={book.best_ask:.1f} | orders={order_prices[:4]}"
            )

        for order_id, order in list(self._open_orders.items()):
            filled = False

            # Check if order would fill
            # For paper trading, use probabilistic fill model based on distance from market
            if order.side == OrderSide.BUY:
                # Buy order fills if ask drops to or below our bid
                if book.best_ask and book.best_ask <= order.price:
                    filled = True
                    logger.debug(
                        f"[PAPER] BUY fill check: bid={order.price:.1f} >= ask={book.best_ask:.1f} -> FILL"
                    )
                elif book.best_ask:
                    # Probabilistic fill: orders within 15 bps have 5% fill chance per check
                    gap_bps = (book.best_ask - order.price) / book.best_ask * 10000
                    if gap_bps <= 15:
                        import random

                        if random.random() < 0.05:  # 5% chance
                            filled = True
                            logger.debug(
                                f"[PAPER] BUY probabilistic fill: gap={gap_bps:.1f}bps -> FILL"
                            )
            else:  # SELL
                # Sell order fills if bid rises to or above our ask
                if book.best_bid and book.best_bid >= order.price:
                    filled = True
                    logger.debug(
                        f"[PAPER] SELL fill check: ask={order.price:.1f} <= bid={book.best_bid:.1f} -> FILL"
                    )
                elif book.best_bid:
                    # Probabilistic fill: orders within 15 bps have 5% fill chance per check
                    gap_bps = (order.price - book.best_bid) / book.best_bid * 10000
                    if gap_bps <= 15:
                        import random

                        if random.random() < 0.05:  # 5% chance
                            filled = True
                            logger.debug(
                                f"[PAPER] SELL probabilistic fill: gap={gap_bps:.1f}bps -> FILL"
                            )
                    elif len(self._open_orders) < 5:  # Only log occasionally
                        logger.debug(
                            f"[PAPER] SELL order at {order.price:.1f} waiting (bid={book.best_bid:.1f}, gap={order.price - book.best_bid:.1f})"
                        )

            if filled:
                fill_price = order.price
                fill_size = order.size

                # Update paper position
                if order.side == OrderSide.BUY:
                    # Buying - increase position
                    if self._paper_position_size >= 0:
                        # Adding to long or opening long
                        old_value = self._paper_position_size * self._paper_entry_price
                        new_value = fill_size * fill_price
                        self._paper_position_size += fill_size
                        if self._paper_position_size > 0:
                            self._paper_entry_price = (
                                old_value + new_value
                            ) / self._paper_position_size
                    else:
                        # Closing short
                        close_size = min(fill_size, abs(self._paper_position_size))
                        pnl = (self._paper_entry_price - fill_price) * close_size
                        self._paper_realized_pnl += pnl
                        self._paper_equity += pnl
                        self._paper_position_size += fill_size
                        if self._paper_position_size > 0:
                            self._paper_entry_price = fill_price
                else:
                    # Selling - decrease position
                    if self._paper_position_size <= 0:
                        # Adding to short or opening short
                        old_value = abs(self._paper_position_size) * self._paper_entry_price
                        new_value = fill_size * fill_price
                        self._paper_position_size -= fill_size
                        if self._paper_position_size < 0:
                            self._paper_entry_price = (old_value + new_value) / abs(
                                self._paper_position_size
                            )
                    else:
                        # Closing long
                        close_size = min(fill_size, self._paper_position_size)
                        pnl = (fill_price - self._paper_entry_price) * close_size
                        self._paper_realized_pnl += pnl
                        self._paper_equity += pnl
                        self._paper_position_size -= fill_size
                        if self._paper_position_size < 0:
                            self._paper_entry_price = fill_price

                orders_to_remove.append(order_id)
                fills += 1
                self._paper_fills += 1

                logger.debug(
                    f"[PAPER FILL] {order.side.value} {fill_size:.4f} @ {fill_price:.1f} | "
                    f"Position: {self._paper_position_size:.4f} | PnL: ${self._paper_realized_pnl:.2f}"
                )

        # Remove filled orders
        for order_id in orders_to_remove:
            del self._open_orders[order_id]

        return fills

    async def cancel_orders_batch(self, orders: List[Tuple[str, str]]) -> int:
        """
        Cancel multiple orders in a single batch request for lower latency.

        Args:
            orders: List of (symbol, order_id) tuples

        Returns:
            Number of orders successfully cancelled
        """
        if not orders:
            return 0

        if not self._exchange:
            raise RuntimeError("Not connected to exchange")

        # Paper trading - just remove from cache
        if self.config.execution.paper_trading:
            for symbol, order_id in orders:
                if order_id in self._open_orders:
                    del self._open_orders[order_id]
            return len(orders)

        await self._rate_limiter.acquire(weight=1)

        try:
            from hyperliquid.utils.signing import CancelRequest

            cancel_requests = [
                CancelRequest(coin=symbol, oid=int(order_id)) for symbol, order_id in orders
            ]

            result = self._exchange.bulk_cancel(cancel_requests)

            if isinstance(result, dict) and result.get("status") == "ok":
                for symbol, order_id in orders:
                    if order_id in self._open_orders:
                        del self._open_orders[order_id]
                logger.debug(f"Batch cancelled {len(orders)} orders")
                return len(orders)
            elif isinstance(result, str):
                for symbol, order_id in orders:
                    if order_id in self._open_orders:
                        del self._open_orders[order_id]
                return len(orders)
            else:
                logger.warning(f"Batch cancel failed: {result}")
                return 0

        except Exception as e:
            logger.error(f"Error in batch cancel: {e}")
            return 0

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a single order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")

        # Paper trading - check for fill first, then remove from cache
        if self.config.execution.paper_trading:
            # Debug: log all order IDs we have
            if self._open_orders:
                logger.debug(
                    f"[PAPER] cancel_order({order_id}) | open_orders: {list(self._open_orders.keys())[:5]}"
                )

            # First check if this specific order filled
            if order_id in self._open_orders:
                order = self._open_orders[order_id]
                book = await self.get_orderbook(symbol)

                if book and book.mid_price:
                    self._paper_last_mid = book.mid_price
                    filled = False

                    if order.side == OrderSide.BUY:
                        if book.best_ask and book.best_ask <= order.price:
                            filled = True
                        else:
                            logger.debug(
                                f"[PAPER] Cancel BUY: order={order.price:.1f} ask={book.best_ask:.1f} gap={book.best_ask - order.price:.1f}"
                            )
                    else:  # SELL
                        if book.best_bid and book.best_bid >= order.price:
                            filled = True
                        else:
                            logger.debug(
                                f"[PAPER] Cancel SELL: order={order.price:.1f} bid={book.best_bid:.1f} gap={order.price - book.best_bid:.1f}"
                            )

                    if filled:
                        # Process fill before "cancelling"
                        fill_price = order.price
                        fill_size = order.size

                        if order.side == OrderSide.BUY:
                            if self._paper_position_size >= 0:
                                old_value = self._paper_position_size * self._paper_entry_price
                                new_value = fill_size * fill_price
                                self._paper_position_size += fill_size
                                if self._paper_position_size > 0:
                                    self._paper_entry_price = (
                                        old_value + new_value
                                    ) / self._paper_position_size
                            else:
                                close_size = min(fill_size, abs(self._paper_position_size))
                                pnl = (self._paper_entry_price - fill_price) * close_size
                                self._paper_realized_pnl += pnl
                                self._paper_equity += pnl
                                self._paper_position_size += fill_size
                                if self._paper_position_size > 0:
                                    self._paper_entry_price = fill_price
                        else:
                            if self._paper_position_size <= 0:
                                old_value = abs(self._paper_position_size) * self._paper_entry_price
                                new_value = fill_size * fill_price
                                self._paper_position_size -= fill_size
                                if self._paper_position_size < 0:
                                    self._paper_entry_price = (old_value + new_value) / abs(
                                        self._paper_position_size
                                    )
                            else:
                                close_size = min(fill_size, self._paper_position_size)
                                pnl = (fill_price - self._paper_entry_price) * close_size
                                self._paper_realized_pnl += pnl
                                self._paper_equity += pnl
                                self._paper_position_size -= fill_size
                                if self._paper_position_size < 0:
                                    self._paper_entry_price = fill_price

                        self._paper_fills += 1
                        logger.info(
                            f"[PAPER FILL] {order.side.value} {fill_size:.4f} @ {fill_price:.1f} | "
                            f"Position: {self._paper_position_size:.4f} | PnL: ${self._paper_realized_pnl:.2f}"
                        )

                del self._open_orders[order_id]
                return True
            return False

        await self._rate_limiter.acquire(weight=1)

        try:
            result = self._exchange.cancel(name=symbol, oid=int(order_id))

            # Handle both dict and string responses
            if isinstance(result, str):
                # String response usually indicates success or simple error
                if order_id in self._open_orders:
                    del self._open_orders[order_id]
                logger.debug(f"Cancelled order {order_id} (response: {result})")
                return True
            elif isinstance(result, dict) and result.get("status") == "ok":
                if order_id in self._open_orders:
                    del self._open_orders[order_id]
                logger.debug(f"Cancelled order {order_id}")
                return True
            elif isinstance(result, dict):
                # Dict response but not "ok" status
                error = result.get("response", {}).get("data", {}).get("error", "Unknown")
                logger.warning(f"Failed to cancel order {order_id}: {error}")
                return False
            else:
                # Unexpected response type
                logger.warning(f"Failed to cancel order {order_id}: {result}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders by fetching from exchange first.
        
        This fetches orders directly from the exchange (not local cache)
        to ensure ALL orders are cancelled, including stale ones from
        previous bot sessions.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")

        coin = symbol or self.config.trading.symbol

        # Paper trading - simulate fills first, then clear
        if self.config.execution.paper_trading:
            await self._simulate_fills()  # Check for fills before cancelling
            orders_to_cancel = [
                order for order in self._open_orders.values() if order.symbol == coin
            ]
            if not orders_to_cancel:
                logger.debug(f"No open orders to cancel for {coin}")
                return 0

            # Clear orders
            self._open_orders = {
                oid: order for oid, order in self._open_orders.items() if order.symbol != coin
            }
            logger.debug(f"[PAPER] Cancelled {len(orders_to_cancel)} orders for {coin}")
            return len(orders_to_cancel)

        await self._rate_limiter.acquire(weight=1)

        try:
            from hyperliquid.utils.signing import CancelRequest
            import requests

            coin = symbol or self.config.trading.symbol
            api_symbol = self._get_api_symbol(coin)
            is_hip3 = coin.upper() == "US500"
            
            # CRITICAL FIX: For HIP-3 perps, openOrders returns empty
            # Use historicalOrders and deduplicate by OID, checking latest status
            if is_hip3:
                resp = requests.post("https://api.hyperliquid.xyz/info", json={
                    "type": "historicalOrders",
                    "user": self.config.wallet_address
                }, timeout=10)
                historical = resp.json()
                
                # IMPORTANT: historicalOrders returns multiple records per order
                # We must group by OID and only keep orders where LATEST status is 'open'
                from collections import defaultdict
                by_oid = defaultdict(list)
                for o in historical:
                    if o.get("order", {}).get("coin") == api_symbol:
                        oid = o.get("order", {}).get("oid")
                        by_oid[oid].append(o)
                
                # Find orders where latest status is 'open'
                orders_to_cancel = []
                for oid, records in by_oid.items():
                    records.sort(key=lambda x: x.get("statusTimestamp", 0), reverse=True)
                    latest = records[0]
                    if latest.get("status") == "open":
                        orders_to_cancel.append({
                            "oid": oid,
                            "coin": api_symbol
                        })
            else:
                # Standard perps: openOrders works fine
                exchange_orders = self._info.open_orders(self.config.wallet_address)
                orders_to_cancel = [
                    o for o in exchange_orders if o.get("coin") == api_symbol
                ]
            
            if not orders_to_cancel:
                logger.debug(f"No open orders to cancel for {coin} ({api_symbol})")
                # Clear local cache too
                self._open_orders = {
                    oid: order for oid, order in self._open_orders.items() if order.symbol != coin
                }
                return 0

            logger.info(f"Found {len(orders_to_cancel)} orders on exchange for {api_symbol}")
            
            # Build cancel requests for bulk_cancel
            cancel_requests = [
                CancelRequest(coin=api_symbol, oid=int(o.get("oid")))
                for o in orders_to_cancel
                if o.get("oid")
            ]

            result = self._exchange.bulk_cancel(cancel_requests)

            if isinstance(result, dict) and result.get("status") == "ok":
                # Clear local order cache for this symbol
                self._open_orders = {
                    oid: order for oid, order in self._open_orders.items() if order.symbol != coin
                }
                cancelled = len(orders_to_cancel)
                logger.info(f"Cancelled {cancelled} orders for {coin}")
                return cancelled
            elif isinstance(result, str):
                # String response - assume success for bulk cancel
                self._open_orders = {
                    oid: order for oid, order in self._open_orders.items() if order.symbol != coin
                }
                cancelled = len(orders_to_cancel)
                logger.info(f"Cancelled {cancelled} orders for {coin} (str response)")
                return cancelled
            else:
                logger.warning(f"Failed to cancel all orders: {result}")
                return 0

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0

    # =========================================================================
    # Account and Position Management
    # =========================================================================

    async def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self._exchange:
            return False

        try:
            # For HIP-3 perps (US500), use km: prefix for API calls
            api_symbol = f"km:{symbol}" if symbol.upper() == "US500" else symbol
            
            # HIP-3 perps use isolated margin, core perps can use cross
            is_cross = symbol.upper() != "US500"
            
            result = self._exchange.update_leverage(
                leverage=leverage,
                name=api_symbol,  # SDK uses 'name' parameter
                is_cross=is_cross,  # HIP-3 uses isolated margin
            )

            if isinstance(result, dict) and result.get("status") == "ok":
                logger.info(f"Set leverage to {leverage}x for {symbol} (isolated={not is_cross})")
                return True
            elif isinstance(result, str):
                # String response - SDK sometimes returns string on success
                logger.info(f"Set leverage to {leverage}x for {symbol}")
                return True
            else:
                logger.warning(f"Failed to set leverage: {result}")
                return False

        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    async def update_leverage(self, leverage: int) -> bool:
        """Update leverage for the trading symbol."""
        # Check cooldown to prevent API spam
        if not self._check_cooldown("leverage"):
            return False
        return await self._set_leverage(self.config.trading.symbol, leverage)

    async def _refresh_account_state(self) -> None:
        """Refresh account state from the exchange with rate limiting."""
        # Paper trading mode - use simulated account state
        if self.config.execution.paper_trading:
            # Simulate account with initial collateral
            mark_price = self._paper_last_mid or 91000.0  # Default BTC price
            
            # Calculate unrealized PnL from open position
            unrealized_pnl = 0.0
            if self._paper_position_size != 0:
                if self._paper_position_size > 0:
                    unrealized_pnl = (mark_price - self._paper_entry_price) * self._paper_position_size
                else:
                    unrealized_pnl = (self._paper_entry_price - mark_price) * abs(self._paper_position_size)
            
            # Update simulated equity with realized and unrealized PnL
            self._paper_equity = self.config.trading.collateral + self._paper_realized_pnl + unrealized_pnl
            
            # Create simulated position
            positions = []
            if self._paper_position_size != 0:
                positions.append(
                    Position(
                        symbol=self.config.trading.symbol,
                        size=self._paper_position_size,
                        entry_price=self._paper_entry_price,
                        mark_price=mark_price,
                        liquidation_price=0.0,
                        unrealized_pnl=unrealized_pnl,
                        leverage=self.config.trading.leverage,
                        margin_used=abs(self._paper_position_size) * mark_price / self.config.trading.leverage,
                    )
                )
            
            # Create simulated account state
            margin_used = sum(abs(p.size) * p.mark_price / p.leverage for p in positions)
            
            self._account_state = AccountState(
                equity=self._paper_equity,
                available_balance=self._paper_equity - margin_used,
                margin_used=margin_used,
                unrealized_pnl=unrealized_pnl,
                positions=positions,
                open_orders=list(self._open_orders.values()),
            )
            
            # Update position cache
            self._positions = {p.symbol: p for p in positions}
            
            logger.debug(
                f"[PAPER] Account: Equity=${self._paper_equity:.2f}, "
                f"Position={self._paper_position_size:.4f}, "
                f"Realized=${self._paper_realized_pnl:.2f}, "
                f"Unrealized=${unrealized_pnl:.2f}"
            )
            return
        
        # Real trading mode - fetch from exchange
        if not self._info:
            return

        # Check cooldown to prevent API spam
        if not self._check_cooldown("account_state"):
            return  # Use cached state

        try:
            # Rate limit
            await self._rate_limiter.acquire(weight=1)

            # Get user state
            user_state = self._info.user_state(self.config.wallet_address)

            if not user_state:
                return

            # Parse margin summary
            margin = user_state.get("marginSummary", {})

            # Parse positions
            positions = []
            for pos_data in user_state.get("assetPositions", []):
                pos = pos_data.get("position", {})
                size = float(pos.get("szi", 0))
                if size != 0:
                    # Calculate mark price from positionValue / abs(size) if markPx not available
                    mark_px = float(pos.get("markPx", 0))
                    if mark_px == 0:
                        position_value = float(pos.get("positionValue", 0))
                        if position_value > 0 and abs(size) > 0:
                            mark_px = position_value / abs(size)

                    positions.append(
                        Position(
                            symbol=pos.get("coin", ""),
                            size=size,
                            entry_price=float(pos.get("entryPx", 0)),
                            mark_price=mark_px,
                            liquidation_price=(
                                float(pos.get("liquidationPx", 0))
                                if pos.get("liquidationPx")
                                else 0.0
                            ),
                            unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                            leverage=int(pos.get("leverage", {}).get("value", 1)),
                            margin_used=float(pos.get("marginUsed", 0)),
                        )
                    )

            # Parse open orders from user_state (works for main perps)
            open_orders = []
            for order_data in user_state.get("openOrders", []):
                open_orders.append(
                    Order(
                        order_id=str(order_data.get("oid", "")),
                        symbol=order_data.get("coin", ""),
                        side=OrderSide.BUY if order_data.get("side") == "B" else OrderSide.SELL,
                        size=float(order_data.get("sz", 0)),
                        price=float(order_data.get("limitPx", 0)),
                        status="open",
                        timestamp=order_data.get("timestamp", 0),
                    )
                )
            
            # For HIP-3 perps (km:US500), user_state.openOrders doesn't include HIP-3 orders
            # We need to fetch them separately via info.open_orders() which uses the perp_dexs
            if self.config.trading.symbol.upper() == "US500":
                try:
                    hip3_orders = self._info.open_orders(self.config.wallet_address)
                    target_coin = f"km:{self.config.trading.symbol}"
                    for order_data in hip3_orders:
                        if order_data.get("coin") == target_coin:
                            open_orders.append(
                                Order(
                                    order_id=str(order_data.get("oid", "")),
                                    symbol=self.config.trading.symbol,  # Store as "US500" not "km:US500"
                                    side=OrderSide.BUY if order_data.get("side") == "B" else OrderSide.SELL,
                                    size=float(order_data.get("sz", 0)),
                                    price=float(order_data.get("limitPx", 0)),
                                    status="open",
                                    timestamp=order_data.get("timestamp", 0),
                                )
                            )
                    if hip3_orders:
                        logger.debug(f"HIP-3: Fetched {len([o for o in hip3_orders if o.get('coin') == target_coin])} orders from km:US500")
                except Exception as e:
                    logger.warning(f"Could not fetch HIP-3 open orders: {e}")

            # Update local cache
            perps_equity = float(margin.get("accountValue", 0))
            perps_available = float(margin.get("withdrawable", 0))
            
            # For HIP-3 perps (km:US500), check USDH balance (Spot + Perp)
            # km:US500 trades on ISOLATED margin using USDH as collateral
            spot_usdh_total = 0.0
            spot_usdh_hold = 0.0
            perp_usdh_total = 0.0  # USDH deposited in perp for isolated margin
            if self.config.trading.symbol.upper() == "US500":
                try:
                    # Get Spot USDH balance
                    spot_state = self._info.spot_user_state(self.config.wallet_address)
                    for b in spot_state.get("balances", []):
                        if b.get("coin") == "USDH":
                            spot_usdh_total = float(b.get("total", 0))  # Total including held
                            spot_usdh_hold = float(b.get("hold", 0))   # Held by open orders
                            break
                    
                    # For isolated margin positions, USDH used as margin is tracked separately
                    # The position margin is part of the total USDH available for trading
                    # Total USDH available = spot USDH + any USDH already in isolated positions
                    logger.debug(f"HIP-3 USDH: spot=${spot_usdh_total:.2f}, hold=${spot_usdh_hold:.2f}")
                except Exception as e:
                    logger.debug(f"Could not check Spot USDH: {e}")
            
            # For US500 (km:US500): Use USDH as the trading balance
            # km:US500 uses ISOLATED margin with USDH collateral, NOT cross-margin USDC
            if self.config.trading.symbol.upper() == "US500":
                # Total equity = Spot USDH (available for new positions)
                # The margin already in positions is separate and managed by exchange
                total_equity = spot_usdh_total  # USDH is the collateral for km:US500
                total_available = spot_usdh_total - spot_usdh_hold  # Available for new orders
                logger.debug(f"US500 ISOLATED: USDH equity=${total_equity:.2f}, available=${total_available:.2f}")
            else:
                # For other symbols that use cross-margin USDC
                total_equity = perps_equity + spot_usdh_total
                total_available = perps_available + (spot_usdh_total - spot_usdh_hold)
            
            self._account_state = AccountState(
                equity=total_equity,  # USDH balance for US500 isolated margin
                available_balance=total_available,  # Available for new orders
                margin_used=float(margin.get("totalMarginUsed", 0)),
                hold_balance=spot_usdh_hold,  # Track held amount
                unrealized_pnl=float(margin.get("totalUnrealizedPnl", 0)),
                positions=positions,
                open_orders=open_orders,
            )

            # Update position cache
            self._positions = {p.symbol: p for p in positions}

            # Update order cache
            self._open_orders = {o.order_id: o for o in open_orders}

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                self._handle_rate_limit()
            logger.error(f"Error refreshing account state: {e}")

    async def get_account_state(self) -> Optional[AccountState]:
        """Get current account state."""
        # Simulate fills first in paper trading mode
        if self.config.execution.paper_trading:
            await self._simulate_fills()

        await self._refresh_account_state()
        return self._account_state

    async def get_position(self, symbol: Optional[str] = None) -> Optional[Position]:
        """Get position for a symbol."""
        symbol = symbol or self.config.trading.symbol

        # Paper trading - return simulated position
        if self.config.execution.paper_trading:
            await self._simulate_fills()

            if self._paper_position_size == 0:
                return None

            mark_price = self._paper_last_mid or self._paper_entry_price
            unrealized_pnl = 0.0
            if self._paper_position_size > 0:
                unrealized_pnl = (mark_price - self._paper_entry_price) * self._paper_position_size
            else:
                unrealized_pnl = (self._paper_entry_price - mark_price) * abs(
                    self._paper_position_size
                )

            return Position(
                symbol=symbol,
                size=self._paper_position_size,
                entry_price=self._paper_entry_price,
                mark_price=mark_price,
                liquidation_price=0.0,
                unrealized_pnl=unrealized_pnl,
                leverage=self.config.trading.leverage,
                margin_used=abs(self._paper_position_size)
                * mark_price
                / self.config.trading.leverage,
            )

        await self._refresh_account_state()
        return self._positions.get(symbol)

    @retry(
        stop=stop_after_attempt(5),  # Increased from 3 to 5 retries for better reliability
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
        reraise=True,
    )
    async def get_orderbook(self, symbol: Optional[str] = None) -> Optional[OrderBook]:
        """Get current order book with freshness check."""
        symbol = symbol or self.config.trading.symbol

        # Check if cached orderbook is fresh enough (< 500ms old)
        # If WebSocket is working, this should always be fresh
        # If WebSocket is stale, we force a REST API refresh
        if symbol in self._orderbook:
            cached = self._orderbook[symbol]
            now = get_timestamp_ms()
            age_ms = now - cached.timestamp if cached.timestamp else float("inf")
            if age_ms < 500:  # Less than 500ms old - use cache
                return cached
            # Cache is stale - fall through to REST API

        # Fetch fresh data from REST API
        if not self._info:
            return None

        try:
            # For HIP-3 perps (US500), use km: prefix for API calls
            api_symbol = f"km:{symbol}" if symbol.upper() == "US500" else symbol
            
            # Use direct POST for HIP-3 perps (l2_snapshot doesn't work with km: prefix)
            if symbol.upper() == "US500":
                book = self._info.post("/info", {"type": "l2Book", "coin": api_symbol})
            else:
                book = self._info.l2_snapshot(name=symbol)

            if book:
                levels = book.get("levels", [[], []])
                bids = [(float(l["px"]), float(l["sz"])) for l in levels[0]]
                asks = [(float(l["px"]), float(l["sz"])) for l in levels[1]]

                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])

                self._orderbook[symbol] = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=get_timestamp_ms(),
                )
                return self._orderbook[symbol]

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                self._handle_rate_limit()
            logger.error(f"Error fetching order book: {e}")

        return None

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Optional[float]:
        """Get current funding rate with rate limiting."""
        if not self._info:
            return None

        # Check cooldown to prevent API spam (funding rate doesn't change often)
        if not self._check_cooldown("funding_rate"):
            return None  # Return None to use cached value in caller

        symbol = symbol or self.config.trading.symbol

        try:
            await self._rate_limiter.acquire(weight=1)
            meta = self._info.meta_and_asset_ctxs()

            if meta and len(meta) > 1:
                for asset in meta[1]:
                    if asset.get("coin") == symbol:
                        return float(asset.get("funding", 0))
            return None

        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            return None

    async def get_user_fills(self, limit: int = 50) -> List[Dict]:
        """Get recent user fills from the exchange."""
        try:
            await self._rate_limiter.acquire(weight=1)

            import aiohttp
            import json

            url = "https://api.hyperliquid.xyz/info"
            payload = {"type": "userFills", "user": self.config.wallet_address}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        fills = await resp.json()
                        return fills[:limit] if fills else []
                    else:
                        logger.error(f"Failed to fetch user fills: HTTP {resp.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching user fills: {e}")
            return []

    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics."""
        return {
            "order": self._order_latency.get_stats(),
            "websocket": self._ws_latency.get_stats(),
        }

    def get_paper_trading_stats(self) -> Dict[str, float]:
        """Get paper trading simulation statistics."""
        return {
            "fills": self._paper_fills,
            "position_size": self._paper_position_size,
            "entry_price": self._paper_entry_price,
            "realized_pnl": self._paper_realized_pnl,
            "equity": self._paper_equity,
            "open_orders": len(self._open_orders),
        }

    @property
    def paper_fills(self) -> int:
        """Get number of paper trading fills."""
        return self._paper_fills

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    async def get_fresh_bbo(self, symbol: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Get fresh BBO (best bid/ask) directly from REST API.

        Bypasses WebSocket cache to get real-time exchange prices.
        Use this for critical rebalancing when WebSocket may be stale.

        Returns:
            Tuple of (best_bid, best_ask) or None if unavailable
        """
        symbol = symbol or self.config.trading.symbol

        if not self._info:
            return None

        try:
            await self._rate_limiter.acquire(weight=1)
            book = self._info.l2_snapshot(name=symbol)

            if book and book.get("levels"):
                levels = book["levels"]
                if levels[0] and levels[1]:
                    best_bid = float(levels[0][0]["px"])
                    best_ask = float(levels[1][0]["px"])
                    return (best_bid, best_ask)
        except Exception as e:
            logger.warning(f"Error fetching fresh BBO: {e}")

        return None

    async def place_ioc_order(
        self, symbol: str, side: OrderSide, size: float, price: float
    ) -> Optional[str]:
        """
        Place an IOC (Immediate Or Cancel) order for aggressive rebalancing.

        This order will match immediately or be cancelled - no resting orders.
        Use for critical rebalancing when we need guaranteed execution.

        Args:
            symbol: Trading symbol (e.g., "BTC")
            side: OrderSide.BUY or OrderSide.SELL
            size: Order size
            price: Limit price (should cross the spread)

        Returns:
            Order ID if filled, None otherwise
        """
        if not self._exchange:
            return None

        try:
            await self._rate_limiter.acquire(weight=1)

            is_buy = side == OrderSide.BUY

            # IOC order - executes immediately or cancels
            order_result = self._exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,  # Only reduce position
            )

            if order_result and order_result.get("status") == "ok":
                response = order_result.get("response", {})
                data = response.get("data", {})
                statuses = data.get("statuses", [])

                if statuses:
                    status = statuses[0]
                    if status.get("filled"):
                        filled = status["filled"]
                        logger.info(
                            f"IOC rebalance filled: {side.value} {filled.get('totalSz')} @ {filled.get('avgPx')}"
                        )
                        return str(filled.get("oid", ""))
                    else:
                        logger.debug("IOC order not filled (price didn't cross)")

            return None

        except Exception as e:
            logger.warning(f"IOC order error: {e}")
            return None
