"""
Hyperliquid Exchange Client for US500-USDH (HIP-3)
===================================================
Low-latency L2 WebSocket client with USDH margin support.

Features:
- L2 orderbook streaming via WebSocket
- USDH margin queries and management
- Batch order placement/cancellation
- Rate limiting with exponential backoff
- Simulation mode for paper trading
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum

import aiohttp
from loguru import logger

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from hyperliquid.utils.signing import CancelRequest
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    logger.warning("hyperliquid-python-sdk not installed")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    ALO = "Alo"
    GTC = "Gtc"
    IOC = "Ioc"


@dataclass
class OrderRequest:
    """Order placement request."""
    symbol: str
    side: OrderSide
    price: float
    size: float
    time_in_force: TimeInForce = TimeInForce.ALO
    reduce_only: bool = False


@dataclass
class Order:
    """Placed order."""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    size: float
    filled: float = 0.0
    status: str = "open"


@dataclass
class Orderbook:
    """L2 Orderbook snapshot."""
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: float = 0.0
    
    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    @property
    def spread_bps(self) -> float:
        if not self.bids or not self.asks or self.mid_price == 0:
            return 0.0
        return (self.asks[0][0] - self.bids[0][0]) / self.mid_price * 10000


class HyperliquidClient:
    """
    Hyperliquid exchange client with L2 WebSocket support.
    
    Supports:
    - US500-USDH (HIP-3 perpetuals)
    - USDH margin management
    - Batch order operations
    - Paper trading simulation
    """
    
    def __init__(self, config):
        self.config = config
        self._connected = False
        self._paper_trading = config.execution.paper_trading
        
        # API clients
        self._info: Optional[Info] = None
        self._exchange: Optional[Exchange] = None
        
        # WebSocket
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        
        # State
        self._orderbook: Dict[str, Orderbook] = {}
        self._open_orders: Dict[str, Order] = {}
        self._position: Dict[str, Any] = {}
        self._callbacks: List[Callable] = []
        
        # Rate limiting
        self._last_request = 0.0
        self._min_interval = 0.1  # 100ms between requests
        
        # Paper trading simulation
        self._sim_orders: Dict[str, Order] = {}
        self._sim_position: float = 0.0
        self._sim_fills: List[dict] = []
    
    async def connect(self) -> None:
        """Connect to Hyperliquid."""
        if not HYPERLIQUID_AVAILABLE:
            raise RuntimeError("hyperliquid-python-sdk not installed")
        
        logger.info("Connecting to Hyperliquid...")
        
        # Determine API URL
        base_url = constants.MAINNET_API_URL
        
        # Initialize Info client
        self._info = Info(base_url=base_url, skip_ws=True)
        
        # Initialize Exchange client if not paper trading
        if not self._paper_trading:
            private_key = self.config.credentials.private_key
            wallet = self.config.credentials.wallet_address
            api_wallet = self.config.credentials.api_wallet_address
            
            self._exchange = Exchange(
                wallet=api_wallet or wallet,
                base_url=base_url,
                account_address=wallet if api_wallet else None
            )
            self._exchange.private_key = private_key
            
            # Set leverage
            await self._set_leverage()
        
        # Connect WebSocket
        await self._connect_websocket()
        
        self._connected = True
        logger.info(f"Connected to Hyperliquid ({'paper' if self._paper_trading else 'live'})")
    
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._ws_task:
            self._ws_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._ws_session:
            await self._ws_session.close()
        self._connected = False
        logger.info("Disconnected from Hyperliquid")
    
    async def _connect_websocket(self) -> None:
        """Connect to L2 WebSocket."""
        ws_url = self.config.network.mainnet_ws_url
        
        self._ws_session = aiohttp.ClientSession()
        self._ws = await self._ws_session.ws_connect(ws_url)
        
        # Subscribe to L2 orderbook
        symbol = self.config.trading.symbol
        api_symbol = f"km:{symbol}" if symbol == "US500" else symbol
        
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": api_symbol}
        }
        await self._ws.send_json(subscribe_msg)
        
        # Start message handler
        self._ws_task = asyncio.create_task(self._ws_handler())
        
        logger.info(f"WebSocket connected to {ws_url}")
    
    async def _ws_handler(self) -> None:
        """Handle WebSocket messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_message(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
    
    async def _handle_ws_message(self, data: dict) -> None:
        """Process WebSocket message."""
        if data.get("channel") == "l2Book":
            book_data = data.get("data", {})
            coin = book_data.get("coin", "")
            
            # Parse levels
            levels = book_data.get("levels", [[], []])
            bids = [(float(p["px"]), float(p["sz"])) for p in levels[0]]
            asks = [(float(p["px"]), float(p["sz"])) for p in levels[1]]
            
            # Normalize symbol
            symbol = coin.replace("km:", "") if coin.startswith("km:") else coin
            
            self._orderbook[symbol] = Orderbook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=time.time()
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self._orderbook[symbol])
                except Exception:
                    pass
    
    async def _set_leverage(self) -> None:
        """Set leverage for the trading symbol."""
        if self._paper_trading or not self._exchange:
            return
        
        symbol = self.config.trading.symbol
        api_symbol = f"km:{symbol}" if symbol == "US500" else symbol
        leverage = self.config.trading.leverage
        
        try:
            self._exchange.update_leverage(leverage, api_symbol, is_cross=False)
            logger.info(f"Set leverage to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
    
    # =========================================================================
    # Orderbook
    # =========================================================================
    
    async def get_orderbook(self, symbol: str) -> Optional[Orderbook]:
        """Get current orderbook for symbol."""
        return self._orderbook.get(symbol)
    
    def subscribe_orderbook(self, callback: Callable[[Orderbook], None]) -> None:
        """Subscribe to orderbook updates."""
        self._callbacks.append(callback)
    
    # =========================================================================
    # Order Management
    # =========================================================================
    
    async def place_orders_batch(self, requests: List[OrderRequest]) -> List[Optional[Order]]:
        """Place multiple orders in a batch."""
        if not requests:
            return []
        
        await self._rate_limit()
        
        if self._paper_trading:
            return await self._sim_place_orders(requests)
        
        if not self._exchange:
            raise RuntimeError("Exchange not connected")
        
        # Convert to API format
        orders_data = []
        for req in requests:
            api_symbol = f"km:{req.symbol}" if req.symbol == "US500" else req.symbol
            
            # Round price and size
            price = round(req.price, 2)
            size = round(req.size, 4)
            
            orders_data.append({
                "coin": api_symbol,
                "is_buy": req.side == OrderSide.BUY,
                "sz": size,
                "limit_px": price,
                "order_type": {"limit": {"tif": req.time_in_force.value}},
                "reduce_only": req.reduce_only
            })
        
        try:
            # Rate limit retry
            for attempt in range(3):
                try:
                    result = self._exchange.bulk_orders(orders_data)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            
            # Parse results
            orders = []
            if result.get("status") == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                for i, status in enumerate(statuses):
                    if status.get("resting"):
                        oid = status["resting"]["oid"]
                        order = Order(
                            order_id=str(oid),
                            symbol=requests[i].symbol,
                            side=requests[i].side,
                            price=requests[i].price,
                            size=requests[i].size
                        )
                        orders.append(order)
                        self._open_orders[str(oid)] = order
                    else:
                        orders.append(None)
            
            return orders
        
        except Exception as e:
            logger.error(f"Error placing orders: " + str(e))
            return [None] * len(requests)
    
    async def cancel_orders_batch(self, orders: List[Tuple[str, str]]) -> int:
        """Cancel multiple orders. orders = [(symbol, order_id), ...]"""
        if not orders:
            return 0
        
        await self._rate_limit()
        
        if self._paper_trading:
            for symbol, oid in orders:
                self._sim_orders.pop(oid, None)
            return len(orders)
        
        if not self._exchange:
            raise RuntimeError("Exchange not connected")
        
        try:
            cancel_requests = []
            for symbol, order_id in orders:
                api_symbol = f"km:{symbol}" if symbol == "US500" else symbol
                cancel_requests.append(CancelRequest(coin=api_symbol, oid=int(order_id)))
            
            # Rate limit retry
            for attempt in range(3):
                try:
                    result = self._exchange.bulk_cancel(cancel_requests)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            
            # Remove from cache
            for symbol, order_id in orders:
                self._open_orders.pop(order_id, None)
            
            return len(orders)
        
        except Exception as e:
            logger.error(f"Error cancelling orders: " + str(e))
            return 0
    
    async def get_open_orders(self, symbol: str) -> List[Order]:
        """Get open orders for symbol."""
        if self._paper_trading:
            return [o for o in self._sim_orders.values() if o.symbol == symbol]
        
        if not self._info:
            return []
        
        try:
            wallet = self.config.credentials.wallet_address
            orders = self._info.open_orders(wallet)
            
            result = []
            for o in orders:
                coin = o.get("coin", "")
                sym = coin.replace("km:", "") if coin.startswith("km:") else coin
                if sym == symbol:
                    result.append(Order(
                        order_id=str(o["oid"]),
                        symbol=sym,
                        side=OrderSide.BUY if o["side"] == "B" else OrderSide.SELL,
                        price=float(o["limitPx"]),
                        size=float(o["sz"])
                    ))
            return result
        
        except Exception as e:
            logger.error(f"Error getting open orders: " + str(e))
            return []
    
    # =========================================================================
    # Position & Margin
    # =========================================================================
    
    async def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for symbol."""
        if self._paper_trading:
            return {"size": self._sim_position, "entry_price": 0.0}
        
        if not self._info:
            return None
        
        try:
            wallet = self.config.credentials.wallet_address
            state = self._info.user_state(wallet)
            
            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                coin = p.get("coin", "")
                sym = coin.replace("km:", "") if coin.startswith("km:") else coin
                if sym == symbol:
                    return {
                        "size": float(p.get("szi", 0)),
                        "entry_price": float(p.get("entryPx", 0)),
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                        "margin_used": float(p.get("marginUsed", 0))
                    }
            return {"size": 0.0, "entry_price": 0.0}
        
        except Exception as e:
            logger.error(f"Error getting position: " + str(e))
            return None
    
    async def get_margin_ratio(self) -> float:
        """Get current margin utilization ratio."""
        if self._paper_trading:
            return 0.1
        
        if not self._info:
            return 0.0
        
        try:
            wallet = self.config.credentials.wallet_address
            state = self._info.user_state(wallet)
            summary = state.get("marginSummary", {})
            
            account_value = float(summary.get("accountValue", 1))
            margin_used = float(summary.get("totalMarginUsed", 0))
            
            if account_value > 0:
                return margin_used / account_value
            return 0.0
        
        except Exception:
            return 0.0
    
    async def get_balance(self) -> float:
        """Get USDH balance."""
        if self._paper_trading:
            return self.config.trading.collateral
        
        if not self._info:
            return 0.0
        
        try:
            wallet = self.config.credentials.wallet_address
            state = self._info.user_state(wallet)
            return float(state.get("marginSummary", {}).get("accountValue", 0))
        except Exception:
            return 0.0
    
    # =========================================================================
    # Paper Trading Simulation
    # =========================================================================
    
    async def _sim_place_orders(self, requests: List[OrderRequest]) -> List[Optional[Order]]:
        """Simulate order placement."""
        orders = []
        for req in requests:
            oid = f"sim_{int(time.time() * 1000)}_{len(self._sim_orders)}"
            order = Order(
                order_id=oid,
                symbol=req.symbol,
                side=req.side,
                price=req.price,
                size=req.size
            )
            self._sim_orders[oid] = order
            orders.append(order)
        return orders
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request = time.time()
    
    @property
    def is_connected(self) -> bool:
        return self._connected
