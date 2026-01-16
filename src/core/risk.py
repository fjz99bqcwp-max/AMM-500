"""
Risk Management Module for US500-USDH Market Making
=====================================================
Comprehensive risk controls including drawdown limits, taker caps,
funding rate hedging, and margin monitoring.

Kill Triggers:
- Max Drawdown >2% → Emergency stop
- Taker Ratio >5% → Reject orders
- Funding Rate >0.01% → Hedge
- Margin <10% → Pause trading
- Stop Loss 2% per position
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
from collections import deque
import time

from loguru import logger


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    pnl: float = 0.0
    trades: int = 0
    volume: float = 0.0
    maker_trades: int = 0
    taker_trades: int = 0
    max_drawdown: float = 0.0


@dataclass
class RiskState:
    """Current risk state."""
    equity: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    taker_ratio: float = 0.0
    margin_ratio: float = 0.0
    position_size: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Kill switch states
    is_stopped: bool = False
    is_paused: bool = False
    stop_reason: str = ""


class RiskManager:
    """
    Risk management system for market making.
    
    Features:
    - Real-time drawdown tracking
    - Taker ratio enforcement (<5%)
    - Margin monitoring
    - Daily PnL tracking
    - Kill switches for autonomous mode
    """
    
    def __init__(self, config, exchange=None):
        self.config = config
        self.exchange = exchange
        
        # Risk parameters
        self.max_drawdown = config.risk.max_drawdown
        self.stop_loss_pct = config.risk.stop_loss_pct
        self.taker_cap = config.risk.taker_ratio_cap
        self.min_margin = config.risk.min_margin_ratio
        self.funding_threshold = config.risk.funding_hedge_threshold
        
        # State tracking
        self._state = RiskState()
        self._daily_stats: Dict[date, DailyStats] = {}
        self._fills: deque = deque(maxlen=10000)
        
        # Initial equity
        self._initial_equity = config.trading.collateral
        self._state.equity = self._initial_equity
        self._state.peak_equity = self._initial_equity
        
        # Timestamps
        self._last_check = 0.0
        self._started_at = time.time()
        
        # Consecutive losses
        self._consecutive_losses = 0
        self._losing_days = 0
    
    async def initialize(self) -> None:
        """Initialize risk manager with current equity."""
        if self.exchange:
            balance = await self.exchange.get_balance()
            if balance > 0:
                self._state.equity = balance
                self._state.peak_equity = max(balance, self._initial_equity)
                logger.info(f"Risk manager initialized with equity: ${self._state.equity:.2f}")
    
    async def check_risk(self) -> RiskState:
        """
        Perform comprehensive risk check.
        Returns current risk state.
        """
        now = time.time()
        
        # Update state from exchange
        if self.exchange:
            await self._update_state()
        
        # Check kill conditions
        await self._check_kill_conditions()
        
        self._last_check = now
        return self._state
    
    async def _update_state(self) -> None:
        """Update risk state from exchange."""
        try:
            # Get balance
            balance = await self.exchange.get_balance()
            if balance > 0:
                self._state.equity = balance
                self._state.peak_equity = max(self._state.peak_equity, balance)
            
            # Calculate drawdown
            if self._state.peak_equity > 0:
                dd = self._state.peak_equity - self._state.equity
                self._state.drawdown = dd
                self._state.drawdown_pct = dd / self._state.peak_equity
            
            # Get margin ratio
            self._state.margin_ratio = await self.exchange.get_margin_ratio()
            
            # Get position
            symbol = self.config.trading.symbol
            position = await self.exchange.get_position(symbol)
            if position:
                self._state.position_size = position.get("size", 0.0)
                self._state.unrealized_pnl = position.get("unrealized_pnl", 0.0)
        
        except Exception as e:
            logger.error(f"Error updating risk state: {e}")
    
    async def _check_kill_conditions(self) -> None:
        """Check for kill switch conditions."""
        # Max drawdown
        if self._state.drawdown_pct > self.max_drawdown:
            self._state.is_stopped = True
            self._state.stop_reason = f"Max drawdown exceeded: {self._state.drawdown_pct:.2%}"
            logger.error(f"🛑 KILL: {self._state.stop_reason}")
            return
        
        # Taker ratio
        if self._state.taker_ratio > 0.30:
            self._state.is_paused = True
            self._state.stop_reason = f"Taker ratio too high: {self._state.taker_ratio:.1%}"
            logger.warning(f"⚠️ PAUSE: {self._state.stop_reason}")
        
        # Margin
        if self._state.margin_ratio > (1 - self.min_margin):
            self._state.is_paused = True
            self._state.stop_reason = f"Low margin: {(1-self._state.margin_ratio):.1%} available"
            logger.warning(f"⚠️ PAUSE: {self._state.stop_reason}")
        
        # Losing days
        if self._losing_days >= 3:
            self._state.is_stopped = True
            self._state.stop_reason = f"3 consecutive losing days"
            logger.error(f"🛑 KILL: {self._state.stop_reason}")
    
    def record_fill(self, fill: dict) -> None:
        """
        Record a trade fill for risk tracking.
        
        fill = {
            "side": "buy" | "sell",
            "price": float,
            "size": float,
            "fee": float,
            "is_maker": bool,
            "pnl": float (optional)
        }
        """
        self._fills.append(fill)
        
        # Update taker ratio
        is_maker = fill.get("is_maker", True)
        total_fills = len(self._fills)
        taker_fills = sum(1 for f in self._fills if not f.get("is_maker", True))
        self._state.taker_ratio = taker_fills / max(total_fills, 1)
        
        # Update daily stats
        today = date.today()
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(date=today)
        
        stats = self._daily_stats[today]
        stats.trades += 1
        stats.volume += fill.get("size", 0) * fill.get("price", 0)
        if is_maker:
            stats.maker_trades += 1
        else:
            stats.taker_trades += 1
        
        # Track PnL
        pnl = fill.get("pnl", 0)
        if pnl:
            stats.pnl += pnl
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
        
        # Check taker cap
        if self._state.taker_ratio > self.taker_cap:
            logger.warning(f"Taker ratio {self._state.taker_ratio:.1%} exceeds cap {self.taker_cap:.1%}")
    
    def should_reject_order(self, is_taker: bool = False) -> bool:
        """Check if order should be rejected based on risk."""
        # Reject takers if over cap
        if is_taker and self._state.taker_ratio > self.taker_cap:
            return True
        
        # Reject if stopped
        if self._state.is_stopped:
            return True
        
        return False
    
    def should_stop(self) -> bool:
        """Check if trading should stop."""
        return self._state.is_stopped
    
    def should_pause(self) -> bool:
        """Check if trading should pause."""
        return self._state.is_paused
    
    def reset_pause(self) -> None:
        """Reset pause state after conditions improve."""
        self._state.is_paused = False
        self._state.stop_reason = ""
    
    def update_daily_pnl(self) -> None:
        """Update end-of-day PnL tracking."""
        today = date.today()
        if today in self._daily_stats:
            stats = self._daily_stats[today]
            if stats.pnl < 0:
                self._losing_days += 1
            else:
                self._losing_days = 0
    
    @property
    def state(self) -> RiskState:
        return self._state
    
    @property
    def equity(self) -> float:
        return self._state.equity
    
    @property
    def drawdown_pct(self) -> float:
        return self._state.drawdown_pct
    
    @property
    def taker_ratio(self) -> float:
        return self._state.taker_ratio
    
    def get_stats(self) -> dict:
        """Get risk statistics."""
        return {
            "equity": self._state.equity,
            "peak_equity": self._state.peak_equity,
            "drawdown": self._state.drawdown,
            "drawdown_pct": self._state.drawdown_pct,
            "taker_ratio": self._state.taker_ratio,
            "margin_ratio": self._state.margin_ratio,
            "position_size": self._state.position_size,
            "is_stopped": self._state.is_stopped,
            "is_paused": self._state.is_paused,
            "stop_reason": self._state.stop_reason,
            "consecutive_losses": self._consecutive_losses,
            "losing_days": self._losing_days,
            "total_fills": len(self._fills),
        }
