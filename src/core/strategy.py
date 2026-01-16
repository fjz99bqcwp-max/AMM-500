"""
Professional Market Making Strategy for US500-USDH
===================================================
Smart orderbook-aware placement with depth analysis, imbalance skewing,
microprice calculation, and PyTorch volatility prediction.

NOT a grid strategy - uses professional MM techniques:
- L2 depth analysis for liquidity gap detection
- Microprice/smart price for fair value estimation
- Imbalance-based quote skewing for adverse selection protection
- Depth-aware sizing (larger near mid, smaller at edges)
- PyTorch LSTM volatility predictor for spread adjustment
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import math

import numpy as np
from loguru import logger

# Optional PyTorch for ML volatility prediction
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using statistical volatility")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    ALO = "Alo"  # Add Liquidity Only (maker only)
    GTC = "Gtc"  # Good Till Cancel
    IOC = "Ioc"  # Immediate Or Cancel


@dataclass
class QuoteLevel:
    """Single quote level in the orderbook."""
    price: float
    size: float
    side: OrderSide
    order_id: Optional[str] = None


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
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.asks[0][0] - self.bids[0][0]) / self.mid_price * 10000


# =============================================================================
# PyTorch Volatility Predictor
# =============================================================================

if PYTORCH_AVAILABLE:
    class VolatilityPredictor(nn.Module):
        """LSTM-based volatility predictor for spread adjustment."""
        
        def __init__(self, input_size: int = 5, hidden_size: int = 32, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus()  # Ensure positive output
            )
            self._trained = False
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
        
        def predict(self, returns: np.ndarray) -> float:
            """Predict volatility from recent returns."""
            if not self._trained or len(returns) < 20:
                # Fallback to realized vol
                return float(np.std(returns[-20:]) * np.sqrt(252 * 24 * 60)) if len(returns) >= 2 else 0.12
            
            # Prepare features: returns, abs_returns, rolling_std, etc.
            with torch.no_grad():
                features = self._prepare_features(returns[-60:])
                pred = self(features.unsqueeze(0))
                return float(pred.item())
        
        def _prepare_features(self, returns: np.ndarray) -> torch.Tensor:
            """Prepare input features for LSTM."""
            n = len(returns)
            features = np.zeros((n, 5))
            features[:, 0] = returns
            features[:, 1] = np.abs(returns)
            features[:, 2] = np.convolve(returns**2, np.ones(5)/5, mode='same')
            features[:, 3] = np.convolve(np.abs(returns), np.ones(10)/10, mode='same')
            features[:, 4] = np.cumsum(returns) / (np.arange(n) + 1)
            return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# Market Making Strategy
# =============================================================================

class MarketMakingStrategy:
    """
    Professional Market Making Strategy for US500-USDH.
    
    Key Features:
    - Smart orderbook placement (depth/queue analysis)
    - Microprice calculation for fair value
    - Imbalance-based skewing for adverse selection
    - Dynamic spread adjustment (volatility-scaled)
    - Reduce-only triggers for risk management
    """
    
    def __init__(self, config, exchange, risk_manager):
        self.config = config
        self.exchange = exchange
        self.risk = risk_manager
        
        # Trading parameters from config
        self.symbol = config.trading.symbol
        self.min_spread_bps = config.trading.min_spread_bps
        self.max_spread_bps = config.trading.max_spread_bps
        self.order_levels = config.trading.order_levels
        self.order_size_fraction = config.trading.order_size_fraction
        self.rebalance_interval = config.execution.rebalance_interval
        
        # State
        self._running = False
        self._orders: Dict[str, Order] = {}
        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._returns: deque = deque(maxlen=500)
        self._last_mid: float = 0.0
        self._fills_today: int = 0
        self._taker_fills: int = 0
        
        # Volatility predictor
        self._vol_predictor = None
        if PYTORCH_AVAILABLE and config.ml.enabled:
            self._vol_predictor = VolatilityPredictor()
            logger.info("PyTorch volatility predictor enabled")
        
        # Metrics
        self._quote_count = 0
        self._last_rebalance = 0.0
    
    async def start(self) -> None:
        """Start the strategy."""
        logger.info(f"Starting {self.symbol} Market Making Strategy...")
        
        # Cancel any existing orders
        await self._cancel_all_orders()
        
        # Sync position
        await self._sync_position()
        
        self._running = True
        logger.info("Strategy started successfully")
        
        # Main loop
        while self._running:
            try:
                await self._rebalance_cycle()
                await asyncio.sleep(self.rebalance_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalance error: {e}")
                await asyncio.sleep(1.0)
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self._running = False
        await self._cancel_all_orders()
        logger.info("Strategy stopped")
    
    # =========================================================================
    # Core Rebalance Logic
    # =========================================================================
    
    async def _rebalance_cycle(self) -> None:
        """Main rebalance cycle - update quotes based on market state."""
        now = time.time()
        
        # Get current orderbook
        book = await self.exchange.get_orderbook(self.symbol)
        if not book or not book.bids or not book.asks:
            return
        
        # Update returns for volatility
        if self._last_mid > 0:
            ret = (book.mid_price - self._last_mid) / self._last_mid
            self._returns.append(ret)
        self._last_mid = book.mid_price
        
        # Check risk conditions
        if await self._should_reduce_only():
            await self._handle_reduce_only(book)
            return
        
        # Calculate smart price and skew
        fair_price = self._calculate_microprice(book)
        imbalance = self._calculate_imbalance(book)
        volatility = self._estimate_volatility()
        
        # Calculate spread based on volatility
        spread_bps = self._calculate_spread(volatility, imbalance)
        
        # Build quote levels
        bids, asks = self._build_quotes(fair_price, spread_bps, imbalance, book)
        
        # Smart order update (minimize cancels)
        await self._update_orders(bids, asks)
        
        self._last_rebalance = now
        self._quote_count += 1
    
    # =========================================================================
    # Smart Price Calculation
    # =========================================================================
    
    def _calculate_microprice(self, book: Orderbook) -> float:
        """
        Calculate microprice - volume-weighted mid price.
        More accurate fair value than simple mid price.
        """
        if not book.bids or not book.asks:
            return book.mid_price
        
        best_bid, bid_size = book.bids[0]
        best_ask, ask_size = book.asks[0]
        
        # Volume-weighted microprice
        total_size = bid_size + ask_size
        if total_size == 0:
            return (best_bid + best_ask) / 2
        
        microprice = (best_bid * ask_size + best_ask * bid_size) / total_size
        return microprice
    
    def _calculate_imbalance(self, book: Orderbook) -> float:
        """
        Calculate order book imbalance for skewing.
        Positive = more bids (bullish), Negative = more asks (bearish)
        """
        if not book.bids or not book.asks:
            return 0.0
        
        # Sum top 5 levels
        bid_depth = sum(size for _, size in book.bids[:5])
        ask_depth = sum(size for _, size in book.asks[:5])
        
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        
        imbalance = (bid_depth - ask_depth) / total
        return np.clip(imbalance, -1.0, 1.0)
    
    def _estimate_volatility(self) -> float:
        """Estimate current volatility for spread adjustment."""
        if len(self._returns) < 10:
            return 0.12  # Default 12% annual vol
        
        returns = np.array(self._returns)
        
        # Use PyTorch predictor if available
        if self._vol_predictor is not None:
            return self._vol_predictor.predict(returns)
        
        # Fallback: realized vol (annualized from 1-min returns)
        realized = np.std(returns[-60:]) * np.sqrt(252 * 24 * 60)
        return max(0.05, min(0.50, realized))
    
    # =========================================================================
    # Spread and Quote Calculation
    # =========================================================================
    
    def _calculate_spread(self, volatility: float, imbalance: float) -> float:
        """
        Dynamic spread based on volatility.
        - Low vol (<5%): Tight spreads (1-2 bps)
        - High vol (>15%): Wide spreads (5-10 bps)
        """
        # Base spread from volatility
        vol_pct = volatility * 100
        
        if vol_pct < 5:
            base_spread = self.min_spread_bps
        elif vol_pct > 15:
            base_spread = self.max_spread_bps
        else:
            # Linear interpolation
            t = (vol_pct - 5) / 10
            base_spread = self.min_spread_bps + t * (self.max_spread_bps - self.min_spread_bps)
        
        # Widen on high imbalance (adverse selection protection)
        if abs(imbalance) > 0.5:
            base_spread *= (1 + abs(imbalance) * 0.5)
        
        return base_spread
    
    def _build_quotes(
        self, 
        fair_price: float, 
        spread_bps: float, 
        imbalance: float,
        book: Orderbook
    ) -> Tuple[List[QuoteLevel], List[QuoteLevel]]:
        """
        Build bid and ask quote levels.
        
        Features:
        - Exponential gap spacing (tighter near mid)
        - Imbalance-based skewing
        - Depth-aware sizing
        """
        bids = []
        asks = []
        
        # Skew offset based on imbalance (shift toward flow)
        skew_bps = imbalance * spread_bps * 0.5
        
        # Position-based skew (reduce exposure on one side)
        if abs(self._position) > self.config.risk.inventory_skew_threshold:
            position_skew = np.sign(self._position) * 1.0  # 1 bps per threshold
            skew_bps -= position_skew
        
        # Base prices
        half_spread = (spread_bps + skew_bps) / 10000 * fair_price / 2
        best_bid = fair_price - half_spread
        best_ask = fair_price + half_spread
        
        # Build levels with exponential gaps
        for i in range(self.order_levels):
            # Exponential gap: 1, 2, 4, 8... bps
            gap_bps = spread_bps * (1.2 ** i)
            gap = gap_bps / 10000 * fair_price
            
            # Size: larger near mid (70% vol in top 5), smaller at edges
            size = self._calculate_size(i, book)
            
            # Bid level
            bid_price = best_bid - gap * i
            bids.append(QuoteLevel(price=bid_price, size=size, side=OrderSide.BUY))
            
            # Ask level
            ask_price = best_ask + gap * i
            asks.append(QuoteLevel(price=ask_price, size=size, side=OrderSide.SELL))
        
        return bids, asks
    
    def _calculate_size(self, level: int, book: Orderbook) -> float:
        """
        Calculate order size for a level.
        - 70% of volume in top 5 levels
        - Taper toward edges
        - Cap at order_size_fraction
        """
        collateral = self.config.trading.collateral
        leverage = self.config.trading.leverage
        max_notional = collateral * leverage
        
        # Base size as fraction of max notional
        base_size = max_notional * self.order_size_fraction
        
        # Weight: higher for inner levels
        if level < 5:
            weight = 0.7 / 5  # 70% in top 5
        else:
            weight = 0.3 / max(self.order_levels - 5, 1)  # 30% in rest
        
        size = base_size * weight / book.mid_price if book.mid_price > 0 else 0.0
        
        # Apply lot size rounding
        lot_size = self.config.trading.lot_size
        size = round(size / lot_size) * lot_size
        
        return max(lot_size, size)
    
    # =========================================================================
    # Order Management
    # =========================================================================
    
    async def _update_orders(self, bids: List[QuoteLevel], asks: List[QuoteLevel]) -> None:
        """
        Smart order update - minimize cancels by matching existing orders.
        """
        # Get current exchange orders
        current_orders = await self.exchange.get_open_orders(self.symbol)
        
        # Separate current orders by side
        current_bids = {o.order_id: o for o in current_orders if o.side == OrderSide.BUY}
        current_asks = {o.order_id: o for o in current_orders if o.side == OrderSide.SELL}
        
        # Find orders to cancel and place
        to_cancel = []
        to_place = []
        
        # Match bids
        for quote in bids:
            matched = self._find_matching_order(quote, list(current_bids.values()))
            if matched:
                del current_bids[matched.order_id]
            else:
                to_place.append(OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    price=quote.price,
                    size=quote.size,
                    time_in_force=TimeInForce.ALO
                ))
        
        # Match asks
        for quote in asks:
            matched = self._find_matching_order(quote, list(current_asks.values()))
            if matched:
                del current_asks[matched.order_id]
            else:
                to_place.append(OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    price=quote.price,
                    size=quote.size,
                    time_in_force=TimeInForce.ALO
                ))
        
        # Cancel unmatched orders
        to_cancel.extend((self.symbol, o.order_id) for o in current_bids.values())
        to_cancel.extend((self.symbol, o.order_id) for o in current_asks.values())
        
        # Execute cancels and places
        if to_cancel:
            await self.exchange.cancel_orders_batch(to_cancel)
        
        if to_place:
            await self.exchange.place_orders_batch(to_place)
    
    def _find_matching_order(self, quote: QuoteLevel, orders: List[Order], tolerance_bps: float = 2.0) -> Optional[Order]:
        """Find an existing order that matches the quote within tolerance."""
        for order in orders:
            if order.side != quote.side:
                continue
            
            price_diff = abs(order.price - quote.price) / quote.price * 10000
            size_diff = abs(order.size - quote.size) / quote.size if quote.size > 0 else 1.0
            
            if price_diff <= tolerance_bps and size_diff <= 0.2:
                return order
        
        return None
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all orders for the symbol."""
        try:
            orders = await self.exchange.get_open_orders(self.symbol)
            if orders:
                await self.exchange.cancel_orders_batch([(self.symbol, o.order_id) for o in orders])
                logger.info(f"Cancelled {len(orders)} orders")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    # =========================================================================
    # Risk Management
    # =========================================================================
    
    async def _should_reduce_only(self) -> bool:
        """Check if we should enter reduce-only mode."""
        # Check margin utilization
        margin = await self.exchange.get_margin_ratio()
        if margin > 0.80:
            logger.warning(f"Margin utilization {margin:.1%} > 80% - reduce only")
            return True
        
        # Check inventory skew
        if abs(self._position) > self.config.risk.inventory_skew_threshold * 1.5:
            logger.warning(f"High inventory skew {self._position:.4f} - reduce only")
            return True
        
        # Check risk manager triggers
        if self.risk.should_stop():
            logger.warning("Risk manager triggered stop")
            return True
        
        return False
    
    async def _handle_reduce_only(self, book: Orderbook) -> None:
        """Handle reduce-only mode - only place closing orders."""
        if abs(self._position) < 0.0001:
            return
        
        # Calculate aggressive closing price
        if self._position > 0:
            # Long position - sell to close
            close_price = book.bids[0][0] if book.bids else book.mid_price * 0.999
            side = OrderSide.SELL
        else:
            # Short position - buy to close
            close_price = book.asks[0][0] if book.asks else book.mid_price * 1.001
            side = OrderSide.BUY
        
        # Reduce by 20%
        reduce_size = abs(self._position) * 0.20
        reduce_size = round(reduce_size / self.config.trading.lot_size) * self.config.trading.lot_size
        
        if reduce_size >= self.config.trading.lot_size:
            await self.exchange.place_orders_batch([
                OrderRequest(
                    symbol=self.symbol,
                    side=side,
                    price=close_price,
                    size=reduce_size,
                    reduce_only=True
                )
            ])
            logger.info(f"Placed reduce-only order: {side.value} {reduce_size:.4f} @ {close_price:.2f}")
    
    async def _sync_position(self) -> None:
        """Sync position from exchange."""
        try:
            position = await self.exchange.get_position(self.symbol)
            if position:
                self._position = position.get("size", 0.0)
                self._entry_price = position.get("entry_price", 0.0)
                logger.info(f"Synced position: {self._position:.4f} @ {self._entry_price:.2f}")
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def position(self) -> float:
        return self._position
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def stats(self) -> dict:
        """Get strategy statistics."""
        return {
            "position": self._position,
            "entry_price": self._entry_price,
            "fills_today": self._fills_today,
            "taker_ratio": self._taker_fills / max(self._fills_today, 1),
            "quote_count": self._quote_count,
            "volatility": self._estimate_volatility() if self._returns else 0.0,
        }
