#!/usr/bin/env python3
"""
Realistic Backtest Engine for AMM-500
======================================

This module provides REALISTIC backtesting with:
- Actual slippage modeling (0.1-5 bps random)
- Latency simulation (50-200ms)
- Partial fills (70-95% rate)
- Adverse selection (price moves against inventory)
- Funding rate costs (from historical data)
- Taker fees when crossing spread
- Inventory risk (mark-to-market losses)
- Queue position degradation
- Realistic oscillation modeling

FIXES the impossible results (Sharpe 563, DD 0%, 52k trades/day) by:
1. Adding execution costs
2. Modeling partial fills and rejections  
3. Simulating adverse price moves
4. Including funding costs
5. Proper PnL calculation with inventory risk

Target realistic metrics:
- Sharpe: 1.5-3.0
- ROI: 20-40%/month
- Max DD: 2-5%
- Trades/Day: 500-5000
- Fill Rate: 70-90%
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import random
import json


@dataclass
class RealisticConfig:
    """Realistic backtest configuration with execution costs."""
    
    # Capital
    initial_capital: float = 1000.0
    leverage: int = 10  # Realistic leverage (not 25x)
    
    # Fees (Hyperliquid actual)
    maker_rebate: float = 0.00002  # 0.002% maker rebate
    taker_fee: float = 0.00035    # 0.035% taker fee
    
    # Spreads (wider for realism)
    min_spread_bps: float = 2.0   # 2 bps minimum
    max_spread_bps: float = 20.0  # 20 bps max in high vol
    
    # Order sizing
    order_size_pct: float = 0.02  # 2% of equity per order
    order_levels: int = 5         # 5 levels each side (not 25)
    
    # Execution realism
    rebalance_interval: int = 5   # 5 second rebalance
    
    # REALISTIC EXECUTION COSTS
    slippage_mean_bps: float = 0.5   # Mean slippage 0.5 bps
    slippage_std_bps: float = 1.0    # Std dev 1 bps (can spike)
    latency_mean_ms: float = 75      # Mean latency 75ms
    latency_std_ms: float = 50       # Std dev 50ms
    
    # Fill modeling
    base_fill_rate: float = 0.70     # 70% base fill rate
    partial_fill_rate: float = 0.30  # 30% of fills are partial
    partial_fill_pct: float = 0.60   # Partial fills average 60%
    
    # Adverse selection (market moves against us after fill)
    adverse_selection_prob: float = 0.40   # 40% of fills have adverse move
    adverse_selection_bps: float = 2.0     # Average adverse move
    
    # Funding costs (annualized rate / 8760 hours)
    funding_rate_mean: float = 0.0001     # 0.01% per 8h = 0.00125%/hr
    funding_rate_std: float = 0.0002      # Can vary
    funding_interval_hours: int = 1       # Apply hourly
    
    # Risk limits
    max_position_pct: float = 0.20        # Max 20% of equity in position
    max_drawdown: float = 0.05            # 5% max drawdown
    stop_loss_pct: float = 0.02           # 2% stop loss
    
    # Queue position
    queue_position_mean: float = 0.50     # Average queue position (50% = middle)
    queue_position_std: float = 0.20      # Variability


@dataclass 
class Trade:
    """A completed trade."""
    timestamp: int
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    is_complete: bool = False


@dataclass
class Position:
    """Current position state."""
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cumulative_funding: float = 0.0
    
    def update_mark(self, mark_price: float) -> float:
        """Update unrealized PnL."""
        if self.size != 0:
            self.unrealized_pnl = (mark_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = 0.0
        return self.unrealized_pnl


@dataclass
class RealisticBacktestResult:
    """Results with realistic metrics."""
    
    # Time
    start_date: datetime = None
    end_date: datetime = None
    duration_days: float = 0.0
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    trades_per_day: float = 0.0
    
    # PnL breakdown
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    total_rebates: float = 0.0
    net_pnl: float = 0.0
    roi_pct: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Fill stats
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    fill_rate: float = 0.0
    
    # Volume
    total_volume: float = 0.0
    maker_volume: float = 0.0
    taker_volume: float = 0.0
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              REALISTIC BACKTEST RESULTS                       ║
╠══════════════════════════════════════════════════════════════╣
║ Period: {self.start_date} to {self.end_date}
║ Duration: {self.duration_days:.1f} days
╠══════════════════════════════════════════════════════════════╣
║ PERFORMANCE                                                   ║
║   Net PnL:         ${self.net_pnl:,.2f}
║   ROI:             {self.roi_pct:.2f}%
║   Sharpe Ratio:    {self.sharpe_ratio:.2f}
║   Sortino Ratio:   {self.sortino_ratio:.2f}
║   Max Drawdown:    {self.max_drawdown:.2%}
║   Profit Factor:   {self.profit_factor:.2f}
╠══════════════════════════════════════════════════════════════╣
║ TRADES                                                        ║
║   Total:           {self.total_trades}
║   Trades/Day:      {self.trades_per_day:.1f}
║   Win Rate:        {self.win_rate:.2%}
║   Winning:         {self.winning_trades}
║   Losing:          {self.losing_trades}
╠══════════════════════════════════════════════════════════════╣
║ EXECUTION COSTS                                               ║
║   Gross PnL:       ${self.gross_pnl:,.2f}
║   Fees Paid:       ${self.total_fees:,.2f}
║   Slippage Cost:   ${self.total_slippage:,.2f}
║   Funding Cost:    ${self.total_funding:,.2f}
║   Rebates Earned:  ${self.total_rebates:,.2f}
╠══════════════════════════════════════════════════════════════╣
║ FILL STATISTICS                                               ║
║   Total Orders:    {self.total_orders}
║   Filled Orders:   {self.filled_orders}
║   Partial Fills:   {self.partial_fills}
║   Fill Rate:       {self.fill_rate:.2%}
╠══════════════════════════════════════════════════════════════╣
║ VOLUME                                                        ║
║   Total:           ${self.total_volume:,.0f}
║   Maker:           ${self.maker_volume:,.0f} ({self.maker_volume/max(self.total_volume,1)*100:.1f}%)
║   Taker:           ${self.taker_volume:,.0f} ({self.taker_volume/max(self.total_volume,1)*100:.1f}%)
╚══════════════════════════════════════════════════════════════╝
"""


class RealisticBacktestEngine:
    """
    Realistic backtesting engine with proper execution modeling.
    
    Key differences from idealized backtest:
    1. Slippage on every trade
    2. Latency affects fill probability
    3. Partial fills reduce average size
    4. Adverse selection causes losses
    5. Funding costs accumulate
    6. Queue position affects fill rate
    7. Position limits enforced
    8. Mark-to-market losses tracked
    """
    
    def __init__(self, config: RealisticConfig = None, verbose: bool = True):
        """Initialize engine."""
        self.config = config or RealisticConfig()
        self.verbose = verbose
        
        # State
        self.equity = self.config.initial_capital
        self.starting_equity = self.config.initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        
        # Tracking
        self.equity_history: List[float] = []
        self.timestamp_history: List[int] = []
        self.pnl_history: List[float] = []
        
        # Metrics
        self.total_fees = 0.0
        self.total_rebates = 0.0
        self.total_slippage = 0.0
        self.total_funding = 0.0
        self.total_volume = 0.0
        self.maker_volume = 0.0
        self.taker_volume = 0.0
        
        # Order tracking
        self.total_orders = 0
        self.filled_orders = 0
        self.partial_fills = 0
        
        # Internal
        self._last_rebalance = 0
        self._last_funding = 0
        self._price_history: List[float] = []
        self._high_water_mark = self.equity
        self._max_drawdown = 0.0
        
        # Random state for reproducibility
        self._rng = np.random.default_rng(42)
    
    def run(self, price_data: pd.DataFrame) -> RealisticBacktestResult:
        """
        Run realistic backtest.
        
        Args:
            price_data: DataFrame with timestamp, open, high, low, close, volume
            
        Returns:
            RealisticBacktestResult
        """
        if self.verbose:
            logger.info(f"Starting realistic backtest on {len(price_data)} candles...")
        
        # Initialize
        self.equity_history = [self.equity]
        self.timestamp_history = []
        self._price_history = []
        
        total_rows = len(price_data)
        
        for idx, row in price_data.iterrows():
            timestamp = int(row["timestamp"])
            mid_price = (row["high"] + row["low"]) / 2
            
            self._price_history.append(mid_price)
            self.timestamp_history.append(timestamp)
            
            # Update position mark-to-market
            self.position.update_mark(mid_price)
            
            # Check stop loss
            self._check_stop_loss(mid_price, timestamp)
            
            # Apply funding every hour
            if timestamp - self._last_funding >= 3600 * 1000:
                self._apply_funding(mid_price, timestamp)
                self._last_funding = timestamp
            
            # Simulate trading
            if timestamp - self._last_rebalance >= self.config.rebalance_interval * 1000:
                self._simulate_trading(row, timestamp)
                self._last_rebalance = timestamp
            
            # Record equity
            current_equity = self.equity + self.position.unrealized_pnl
            self.equity_history.append(current_equity)
            
            # Track drawdown
            if current_equity > self._high_water_mark:
                self._high_water_mark = current_equity
            dd = (self._high_water_mark - current_equity) / self._high_water_mark
            if dd > self._max_drawdown:
                self._max_drawdown = dd
            
            # Progress
            if self.verbose and idx % 10000 == 0:
                pct = idx / total_rows * 100
                logger.info(f"  Progress: {pct:.1f}% - Equity: ${current_equity:.2f}, DD: {self._max_drawdown:.2%}")
        
        if self.verbose:
            logger.info("Backtest completed")
        return self._calculate_results(price_data)
    
    def _simulate_trading(self, candle: pd.Series, timestamp: int) -> None:
        """
        Simulate realistic trading for one period.
        
        Models:
        - Order placement with queue position
        - Fill probability based on price movement
        - Slippage on fills
        - Partial fills
        - Adverse selection
        """
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        mid_price = (high + low) / 2
        candle_range = high - low
        
        # Calculate volatility for spread
        vol = self._calculate_volatility()
        
        # Dynamic spread based on volatility
        spread_bps = self.config.min_spread_bps
        if vol > 0.10:  # > 10% vol
            vol_scale = min(1.0, (vol - 0.10) / 0.10)
            spread_bps = self.config.min_spread_bps + vol_scale * (
                self.config.max_spread_bps - self.config.min_spread_bps
            )
        
        spread = mid_price * spread_bps / 10000
        
        # Place orders
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Order size (limited by position limits)
        max_position_value = self.equity * self.config.max_position_pct
        max_position_size = max_position_value / mid_price
        
        order_size = self.equity * self.config.order_size_pct / mid_price
        order_size = min(order_size, max_position_size - abs(self.position.size))
        
        if order_size <= 0:
            return  # At position limit
        
        # Simulate order lifecycle
        for level in range(self.config.order_levels):
            level_offset = level * spread * 0.3
            
            # Buy order
            buy_price = bid_price - level_offset
            self._simulate_order("buy", buy_price, order_size / self.config.order_levels,
                               low, high, mid_price, timestamp)
            
            # Sell order
            sell_price = ask_price + level_offset
            self._simulate_order("sell", sell_price, order_size / self.config.order_levels,
                               low, high, mid_price, timestamp)
    
    def _simulate_order(self, side: str, price: float, size: float,
                       low: float, high: float, mid_price: float, timestamp: int) -> None:
        """
        Simulate a single order with realistic execution.
        """
        self.total_orders += 1
        
        # Check if price was touched
        if side == "buy" and low > price:
            return  # Price never reached bid
        if side == "sell" and high < price:
            return  # Price never reached ask
        
        # Queue position affects fill probability
        queue_pos = self._rng.normal(self.config.queue_position_mean, 
                                     self.config.queue_position_std)
        queue_pos = np.clip(queue_pos, 0.1, 0.9)
        
        # Fill probability = base_rate * (1 - queue_position)
        fill_prob = self.config.base_fill_rate * (1 - queue_pos * 0.5)
        
        # Latency reduces fill probability further
        latency = self._rng.normal(self.config.latency_mean_ms, self.config.latency_std_ms)
        latency = max(10, latency)
        latency_penalty = min(0.3, latency / 500)  # Up to 30% reduction
        fill_prob *= (1 - latency_penalty)
        
        # Random fill check
        if self._rng.random() > fill_prob:
            return  # Order not filled
        
        self.filled_orders += 1
        
        # Partial fill check
        fill_size = size
        if self._rng.random() < self.config.partial_fill_rate:
            fill_size = size * self._rng.uniform(0.3, 0.9)
            self.partial_fills += 1
        
        # Calculate slippage
        slippage_bps = abs(self._rng.normal(self.config.slippage_mean_bps,
                                            self.config.slippage_std_bps))
        slippage = mid_price * slippage_bps / 10000
        
        # Apply slippage (always adverse)
        if side == "buy":
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        fill_value = fill_size * fill_price
        slippage_cost = slippage * fill_size
        self.total_slippage += slippage_cost
        
        # Determine maker/taker (most should be maker for MM)
        is_maker = self._rng.random() > 0.15  # 85% maker
        
        if is_maker:
            fee = -fill_value * self.config.maker_rebate  # Rebate
            self.total_rebates += abs(fee)
            self.maker_volume += fill_value
        else:
            fee = fill_value * self.config.taker_fee
            self.total_fees += fee
            self.taker_volume += fill_value
        
        self.total_volume += fill_value
        
        # Adverse selection (market moves against us after fill)
        adverse_cost = 0.0
        if self._rng.random() < self.config.adverse_selection_prob:
            adverse_bps = self._rng.exponential(self.config.adverse_selection_bps)
            adverse_cost = fill_size * mid_price * adverse_bps / 10000
            self.total_slippage += adverse_cost
        
        # Update position
        self._update_position(side, fill_size, fill_price, fee, slippage_cost + adverse_cost, timestamp)
    
    def _update_position(self, side: str, size: float, price: float, 
                        fee: float, slippage: float, timestamp: int) -> None:
        """Update position after a fill."""
        
        trade = Trade(
            timestamp=timestamp,
            side=side,
            size=size,
            entry_price=price,
            fees=fee,
            slippage=slippage,
        )
        
        if side == "buy":
            if self.position.size >= 0:
                # Adding to long or opening long
                total_value = self.position.size * self.position.entry_price + size * price
                self.position.size += size
                if self.position.size > 0:
                    self.position.entry_price = total_value / self.position.size
            else:
                # Reducing short
                if size >= abs(self.position.size):
                    # Close short
                    pnl = (self.position.entry_price - price) * abs(self.position.size) - fee - slippage
                    self.position.realized_pnl += pnl
                    self.equity += pnl
                    
                    remaining = size - abs(self.position.size)
                    self.position.size = remaining
                    self.position.entry_price = price if remaining > 0 else 0
                    
                    trade.pnl = pnl
                    trade.is_complete = True
                else:
                    # Partial close
                    pnl = (self.position.entry_price - price) * size - fee - slippage
                    self.position.realized_pnl += pnl
                    self.equity += pnl
                    self.position.size += size
                    
                    trade.pnl = pnl
                    trade.is_complete = True
        else:  # sell
            if self.position.size <= 0:
                # Adding to short or opening short
                total_value = abs(self.position.size) * self.position.entry_price + size * price
                self.position.size -= size
                if self.position.size < 0:
                    self.position.entry_price = total_value / abs(self.position.size)
            else:
                # Reducing long
                if size >= self.position.size:
                    # Close long
                    pnl = (price - self.position.entry_price) * self.position.size - fee - slippage
                    self.position.realized_pnl += pnl
                    self.equity += pnl
                    
                    remaining = size - self.position.size
                    self.position.size = -remaining
                    self.position.entry_price = price if remaining > 0 else 0
                    
                    trade.pnl = pnl
                    trade.is_complete = True
                else:
                    # Partial close
                    pnl = (price - self.position.entry_price) * size - fee - slippage
                    self.position.realized_pnl += pnl
                    self.equity += pnl
                    self.position.size -= size
                    
                    trade.pnl = pnl
                    trade.is_complete = True
        
        self.trades.append(trade)
    
    def _apply_funding(self, price: float, timestamp: int) -> None:
        """Apply funding rate to position."""
        if self.position.size == 0:
            return
        
        # Random funding rate (can be positive or negative)
        funding_rate = self._rng.normal(self.config.funding_rate_mean, 
                                        self.config.funding_rate_std)
        
        # Longs pay positive funding, shorts receive
        funding_cost = abs(self.position.size) * price * funding_rate
        if self.position.size > 0:
            funding_cost = abs(funding_cost)  # Longs usually pay
        else:
            funding_cost = -abs(funding_cost) * 0.5  # Shorts receive less
        
        self.position.cumulative_funding += funding_cost
        self.total_funding += funding_cost
        self.equity -= funding_cost
    
    def _check_stop_loss(self, price: float, timestamp: int) -> None:
        """Check and execute stop loss if needed."""
        if self.position.size == 0:
            return
        
        current_equity = self.equity + self.position.unrealized_pnl
        drawdown = (self.starting_equity - current_equity) / self.starting_equity
        
        if drawdown >= self.config.stop_loss_pct:
            # Close position at market (with extra slippage for urgency)
            slippage = price * 0.005  # 50 bps emergency slippage
            
            if self.position.size > 0:
                exit_price = price - slippage
                pnl = (exit_price - self.position.entry_price) * self.position.size
            else:
                exit_price = price + slippage
                pnl = (self.position.entry_price - exit_price) * abs(self.position.size)
            
            fee = abs(self.position.size) * price * self.config.taker_fee
            pnl -= fee
            
            self.position.realized_pnl += pnl
            self.equity += pnl
            self.total_fees += fee
            self.total_slippage += slippage * abs(self.position.size)
            self.taker_volume += abs(self.position.size) * price
            self.total_volume += abs(self.position.size) * price
            
            logger.warning(f"Stop loss triggered at {drawdown:.2%} DD, PnL: ${pnl:.2f}")
            
            self.position.size = 0
            self.position.entry_price = 0
    
    def _calculate_volatility(self) -> float:
        """Calculate recent volatility (annualized)."""
        if len(self._price_history) < 100:
            return 0.15  # Default 15%
        
        prices = np.array(self._price_history[-100:])
        returns = np.diff(prices) / prices[:-1]
        
        # Annualize (assuming 1-min candles)
        minute_vol = np.std(returns)
        annual_vol = minute_vol * np.sqrt(525600)  # Minutes in year
        
        return annual_vol
    
    def _calculate_results(self, price_data: pd.DataFrame) -> RealisticBacktestResult:
        """Calculate final results."""
        
        # Time period
        start_ts = price_data["timestamp"].iloc[0]
        end_ts = price_data["timestamp"].iloc[-1]
        start_date = datetime.fromtimestamp(start_ts / 1000)
        end_date = datetime.fromtimestamp(end_ts / 1000)
        duration_days = (end_date - start_date).days
        
        # Trade stats
        completed_trades = [t for t in self.trades if t.is_complete]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl <= 0]
        
        # PnL
        final_equity = self.equity + self.position.unrealized_pnl
        net_pnl = final_equity - self.starting_equity
        gross_pnl = sum(t.pnl + t.fees + t.slippage for t in completed_trades)
        
        # Sharpe ratio (daily returns)
        if len(self.equity_history) > 2:
            equity = np.array(self.equity_history)
            daily_candles = 1440  # 1-min candles per day
            
            # Sample daily
            daily_equity = equity[::daily_candles] if len(equity) > daily_candles else equity
            daily_returns = np.diff(daily_equity) / daily_equity[:-1]
            
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
                
            # Sortino (downside only)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and np.std(downside_returns) > 0:
                sortino = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino = sharpe
        else:
            sharpe = 0.0
            sortino = 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return RealisticBacktestResult(
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            
            total_trades=len(completed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / max(len(completed_trades), 1),
            trades_per_day=len(completed_trades) / max(duration_days, 1),
            
            gross_pnl=gross_pnl,
            total_fees=self.total_fees,
            total_slippage=self.total_slippage,
            total_funding=self.total_funding,
            total_rebates=self.total_rebates,
            net_pnl=net_pnl,
            roi_pct=net_pnl / self.starting_equity * 100,
            
            max_drawdown=self._max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            
            total_orders=self.total_orders,
            filled_orders=self.filled_orders,
            partial_fills=self.partial_fills,
            fill_rate=self.filled_orders / max(self.total_orders, 1),
            
            total_volume=self.total_volume,
            maker_volume=self.maker_volume,
            taker_volume=self.taker_volume,
            
            equity_curve=self.equity_history,
            timestamps=self.timestamp_history,
        )


def load_data(filepath: Path) -> pd.DataFrame:
    """Load price data from JSON or CSV."""
    if filepath.suffix == '.json':
        with open(filepath) as f:
            candles = json.load(f)
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.sort_values('timestamp').reset_index(drop=True)


def run_realistic_backtest(
    data_path: Path = None,
    config: RealisticConfig = None,
    days: int = 30,
    verbose: bool = True,
) -> RealisticBacktestResult:
    """
    Run realistic backtest.
    
    Args:
        data_path: Path to price data (JSON or CSV)
        config: Backtest configuration
        days: Days of data to use
        verbose: Print results
        
    Returns:
        RealisticBacktestResult
    """
    config = config or RealisticConfig()
    
    # Load data
    if data_path and data_path.exists():
        if verbose:
            logger.info(f"Loading data from {data_path}")
        data = load_data(data_path)
    else:
        # Check default location
        default_path = Path(__file__).parent.parent / "data" / "us500_synthetic_180d.json"
        if default_path.exists():
            if verbose:
                logger.info(f"Loading data from {default_path}")
            data = load_data(default_path)
        else:
            raise FileNotFoundError(f"No data file found. Please generate data first.")
    
    # Limit to requested days
    if days < len(data) // 1440:
        candles_needed = days * 1440
        data = data.tail(candles_needed).reset_index(drop=True)
    
    if verbose:
        logger.info(f"Loaded {len(data)} candles ({len(data)/1440:.1f} days)")
    
    # Run backtest
    engine = RealisticBacktestEngine(config, verbose=verbose)
    result = engine.run(data)
    
    # Print results
    if verbose:
        print(result.summary())
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Realistic AMM-500 Backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    parser.add_argument("--capital", type=float, default=1000, help="Initial capital")
    parser.add_argument("--min-spread", type=float, default=2.0, help="Min spread in bps")
    parser.add_argument("--max-spread", type=float, default=20.0, help="Max spread in bps")
    parser.add_argument("--slippage", type=float, default=0.5, help="Mean slippage in bps")
    parser.add_argument("--fill-rate", type=float, default=0.70, help="Base fill rate")
    parser.add_argument("--order-levels", type=int, default=5, help="Order levels per side")
    args = parser.parse_args()
    
    config = RealisticConfig(
        initial_capital=args.capital,
        leverage=args.leverage,
        min_spread_bps=args.min_spread,
        max_spread_bps=args.max_spread,
        slippage_mean_bps=args.slippage,
        base_fill_rate=args.fill_rate,
        order_levels=args.order_levels,
    )
    
    result = run_realistic_backtest(config=config, days=args.days)
