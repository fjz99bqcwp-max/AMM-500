#!/usr/bin/env python3
"""
Script to generate professional market making strategy file.
This creates src/strategy_professional.py with L2-aware quoting.
"""

PROFESSIONAL_STRATEGY = '''"""
Professional Market Making Strategy for Hyperliquid BTC Perpetuals
High-frequency trading with L2-aware dynamic quoting, adaptive sizing, and inventory management.

Transform from grid-based to professional market making:
- Real-time L2 order book integration for dynamic quoting
- Adaptive spread and sizing based on book depth and inventory
- Inventory skew management for delta-neutral operation  
- Volatility-adaptive spreads with exponential tiering
- Quote fading on adverse selection detection
- Optimized for Apple M4 hardware (10-core, 24GB RAM)

WARNING: High-frequency trading with leverage carries significant financial risk.
Thoroughly test on testnet before using real funds.
"""

import asyncio
import time
import numpy as np
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
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

# Optional PyTorch integration for vol/spread prediction
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
    """A single quote level with L2 book awareness."""
    price: float
    size: float
    side: OrderSide
    order_id: Optional[str] = None
    created_at: float = 0.0
    book_depth_at_level: float = 0.0


@dataclass
class StrategyMetrics:
    """Strategy performance metrics with adverse selection tracking."""
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
    
    # Adverse selection tracking
    recent_buy_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    recent_sell_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    recent_buy_sizes: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    recent_sell_sizes: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    recent_fill_times: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    consecutive_losing_fills: int = 0
    fill_data_max_age: float = 300.0
    
    @property
    def fill_rate(self) -> float:
        if self.quotes_sent == 0:
            return 0.0
        return self.quotes_filled / self.quotes_sent
    
    @property
    def maker_ratio(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.maker_volume / self.total_volume
    
    @property
    def avg_spread_capture(self) -> float:
        if self.quotes_filled == 0:
            return 0.0
        return self.spread_capture / self.quotes_filled
    
    def get_recent_spread_bps(self) -> Optional[float]:
        """Calculate weighted spread from recent fills."""
        if len(self.recent_buy_prices) < 3 or len(self.recent_sell_prices) < 3:
            return None
        
        now = time.time()
        while self.recent_fill_times and (now - self.recent_fill_times[0]) > self.fill_data_max_age:
            self.recent_fill_times.popleft()
            if self.recent_buy_prices:
                self.recent_buy_prices.popleft()
                self.recent_buy_sizes.popleft()
            if self.recent_sell_prices:
                self.recent_sell_prices.popleft()
                self.recent_sell_sizes.popleft()
        
        if len(self.recent_buy_prices) < 3 or len(self.recent_sell_prices) < 3:
            return None
        
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
    
    def add_fill(self, side: OrderSide, price: float, size: float, mark_price: float) -> None:
        """Track fills for adverse selection detection."""
        now = time.time()
        self.recent_fill_times.append(now)
        
        if side == OrderSide.BUY:
            self.recent_buy_prices.append(price)
            self.recent_buy_sizes.append(size)
        else:
            self.recent_sell_prices.append(price)
            self.recent_sell_sizes.append(size)
        
        # Track consecutive losing fills
        if side == OrderSide.BUY and price > mark_price * 1.0005:
            self.consecutive_losing_fills += 1
        elif side == OrderSide.SELL and price < mark_price * 0.9995:
            self.consecutive_losing_fills += 1
        else:
            self.consecutive_losing_fills = 0


@dataclass
class InventoryState:
    """Current inventory state."""
    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    delta: float = 0.0
    
    @property
    def is_balanced(self) -> bool:
        return abs(self.delta) < 0.015  # Within 1.5%
    
    @property
    def skew_urgency(self) -> float:
        return min(abs(self.delta) / 0.05, 1.0)


# TODO: Continue implementation with full L2-aware market making logic
# This is a template - full implementation to be added

'''

if __name__ == "__main__":
    output_file = "src/strategy_professional.py"
    with open(output_file, "w") as f:
        f.write(PROFESSIONAL_STRATEGY)
    print(f"Generated {output_file}")
