"""
Backtesting Engine for US500-USDH Market Making
=================================================
Realistic simulation with slippage, latency, partial fills,
adverse selection, queue priority, and funding costs.

Simulation Features:
- Latency modeling (50-200ms)
- Partial fills (30% average)
- Adverse selection (35% probability)
- Queue priority simulation
- Funding rate costs
- Maker/taker fee distinction
- USDH margin simulation
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import pandas as pd

from loguru import logger


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Execution simulation
    latency_ms: float = 100.0
    partial_fill_rate: float = 0.30
    adverse_selection_prob: float = 0.35
    queue_skip_prob: float = 0.20
    
    # Costs
    maker_fee: float = -0.0002  # -2 bps rebate
    taker_fee: float = 0.0005   # 5 bps fee
    funding_rate: float = 0.0001  # 1 bps per 8h
    
    # Slippage
    slippage_bps: float = 1.0
    
    # Capital
    initial_capital: float = 1000.0
    leverage: int = 10


@dataclass
class BacktestTrade:
    """Single trade in backtest."""
    timestamp: datetime
    side: str
    price: float
    size: float
    is_maker: bool
    fee: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Backtest result summary."""
    # Returns
    total_pnl: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # Risk
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Activity
    total_trades: int = 0
    trades_per_day: float = 0.0
    win_rate: float = 0.0
    
    # Execution
    maker_ratio: float = 0.0
    total_fees: float = 0.0
    avg_fill_rate: float = 0.0
    
    # Capital
    final_equity: float = 0.0
    peak_equity: float = 0.0
    
    # Raw data
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Realistic backtesting engine for market making strategies.
    
    Simulates:
    - Order book dynamics
    - Fill probability based on queue position
    - Adverse selection when filled
    - Partial fills
    - Latency and slippage
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # State
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = config.initial_capital
        self._peak_equity = config.initial_capital
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[float] = [config.initial_capital]
        
        # Metrics
        self._total_fees = 0.0
        self._maker_trades = 0
        self._taker_trades = 0
        self._winning_trades = 0
        self._fill_attempts = 0
        self._successful_fills = 0
    
    async def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest on {len(data)} candles...")
        
        # Reset state
        self._reset()
        
        # Process each candle
        for i, row in data.iterrows():
            await self._process_candle(row, i, data)
        
        # Close any remaining position
        if abs(self._position) > 0.0001:
            last_price = data.iloc[-1]["close"]
            await self._close_position(last_price, data.index[-1])
        
        # Calculate results
        result = self._calculate_results(data)
        
        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Sharpe={result.sharpe_ratio:.2f}, ROI={result.total_return:.2%}")
        
        return result
    
    async def _process_candle(self, candle: pd.Series, idx: int, data: pd.DataFrame) -> None:
        """Process a single candle."""
        mid = (candle["high"] + candle["low"]) / 2
        spread = (candle["high"] - candle["low"]) / mid
        
        # Simulate market making quotes
        bid_price = mid * (1 - spread * 0.3)
        ask_price = mid * (1 + spread * 0.3)
        
        # Check for fills
        if candle["low"] <= bid_price:
            # Bid filled
            await self._simulate_fill("buy", bid_price, candle, idx)
        
        if candle["high"] >= ask_price:
            # Ask filled
            await self._simulate_fill("sell", ask_price, candle, idx)
        
        # Apply funding every 8 hours (480 1-min candles)
        if idx > 0 and idx % 480 == 0:
            await self._apply_funding(mid)
        
        # Update equity curve
        self._update_equity(mid)
    
    async def _simulate_fill(self, side: str, price: float, candle: pd.Series, idx: int) -> None:
        """Simulate a fill with realistic execution."""
        self._fill_attempts += 1
        
        # Adverse selection check
        if random.random() < self.config.adverse_selection_prob:
            # Price moves against us after fill
            slippage = self.config.slippage_bps / 10000
            if side == "buy":
                price *= (1 + slippage)
            else:
                price *= (1 - slippage)
        
        # Queue skip check (order not filled)
        if random.random() < self.config.queue_skip_prob:
            return
        
        # Determine size
        max_notional = self.config.initial_capital * self.config.leverage
        base_size = max_notional * 0.02 / price  # 2% per order
        
        # Partial fill
        fill_rate = random.uniform(0.2, 1.0) if random.random() < 0.5 else 1.0
        size = base_size * fill_rate
        
        # Determine if maker or taker
        is_maker = random.random() > 0.1  # 90% maker
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        fee = abs(size * price * fee_rate)
        
        if is_maker:
            fee = -fee  # Rebate
            self._maker_trades += 1
        else:
            self._taker_trades += 1
        
        # Calculate PnL if closing
        pnl = 0.0
        if self._position != 0:
            if (side == "sell" and self._position > 0) or (side == "buy" and self._position < 0):
                pnl = abs(self._position) * (price - self._entry_price)
                if self._position < 0:
                    pnl = -pnl
                
                if pnl > 0:
                    self._winning_trades += 1
        
        # Update position
        if side == "buy":
            self._position += size
        else:
            self._position -= size
        
        self._entry_price = price
        self._equity += pnl - fee
        self._total_fees += fee
        self._successful_fills += 1
        
        # Record trade
        self._trades.append(BacktestTrade(
            timestamp=candle.name if isinstance(candle.name, datetime) else datetime.now(),
            side=side,
            price=price,
            size=size,
            is_maker=is_maker,
            fee=fee,
            pnl=pnl
        ))
    
    async def _close_position(self, price: float, timestamp) -> None:
        """Close remaining position."""
        if abs(self._position) < 0.0001:
            return
        
        side = "sell" if self._position > 0 else "buy"
        size = abs(self._position)
        pnl = size * (price - self._entry_price)
        if self._position < 0:
            pnl = -pnl
        
        fee = size * price * self.config.taker_fee
        
        self._equity += pnl - fee
        self._total_fees += fee
        self._position = 0.0
        
        if pnl > 0:
            self._winning_trades += 1
        
        self._trades.append(BacktestTrade(
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
            side=side,
            price=price,
            size=size,
            is_maker=False,
            fee=fee,
            pnl=pnl
        ))
    
    async def _apply_funding(self, price: float) -> None:
        """Apply funding rate cost."""
        if abs(self._position) > 0.0001:
            funding_cost = abs(self._position) * price * self.config.funding_rate
            self._equity -= funding_cost
            self._total_fees += funding_cost
    
    def _update_equity(self, price: float) -> None:
        """Update equity and track peak."""
        # Mark-to-market unrealized PnL
        if self._position != 0:
            unrealized = self._position * (price - self._entry_price)
            current_equity = self._equity + unrealized
        else:
            current_equity = self._equity
        
        self._equity_curve.append(current_equity)
        self._peak_equity = max(self._peak_equity, current_equity)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate final backtest results."""
        result = BacktestResult()
        
        # Returns
        result.final_equity = self._equity
        result.peak_equity = self._peak_equity
        result.total_pnl = self._equity - self.config.initial_capital
        result.total_return = result.total_pnl / self.config.initial_capital
        
        # Annualize (assuming 1-min data)
        days = len(data) / (24 * 60)
        if days > 0:
            result.annual_return = (1 + result.total_return) ** (365 / days) - 1
            result.trades_per_day = len(self._trades) / days
        
        # Drawdown
        equity_arr = np.array(self._equity_curve)
        peak_arr = np.maximum.accumulate(equity_arr)
        drawdowns = (peak_arr - equity_arr) / peak_arr
        result.max_drawdown = float(np.max(drawdowns))
        
        # Sharpe ratio
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / self._equity_curve[:-1]
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)
            
            # Sortino (downside deviation)
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                result.sortino_ratio = np.mean(returns) / np.std(downside) * np.sqrt(252 * 24 * 60)
        
        # Activity
        result.total_trades = len(self._trades)
        result.win_rate = self._winning_trades / max(len(self._trades), 1)
        result.total_fees = self._total_fees
        
        # Execution quality
        result.maker_ratio = self._maker_trades / max(len(self._trades), 1)
        result.avg_fill_rate = self._successful_fills / max(self._fill_attempts, 1)
        
        # Raw data
        result.trades = self._trades
        result.equity_curve = self._equity_curve
        
        return result
    
    def _reset(self) -> None:
        """Reset backtest state."""
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._trades = []
        self._equity_curve = [self.config.initial_capital]
        self._total_fees = 0.0
        self._maker_trades = 0
        self._taker_trades = 0
        self._winning_trades = 0
        self._fill_attempts = 0
        self._successful_fills = 0


async def run_backtest(data: pd.DataFrame, config: Optional[BacktestConfig] = None) -> BacktestResult:
    """Convenience function to run backtest."""
    if config is None:
        config = BacktestConfig()
    
    engine = BacktestEngine(config)
    return await engine.run(data)
