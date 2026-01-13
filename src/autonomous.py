"""
Autonomous Trading System - Dynamic Risk Adjustment & Performance-Based Optimization

This module implements self-managing trading logic that dynamically adjusts:
- Order sizes based on drawdown and performance
- Leverage based on Sharpe ratio and volatility
- Spread parameters based on fill rates and market conditions
- Position limits based on risk metrics

Key Features:
1. Performance-based dynamic sizing: Reduce exposure during drawdowns
2. Sharpe-ratio triggered aggression: Increase levels when Sharpe > 3
3. Automatic leverage adjustment: Scale down on losses, up on consistent profits
4. ML-ready architecture for future predictive spread/vol forecasting

Hardware: Optimized for Apple M4 with multiprocessing support
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from loguru import logger


class AutonomousMode(Enum):
    """Operating modes for autonomous system."""
    CONSERVATIVE = "conservative"   # After losses, wide spreads, low leverage
    NORMAL = "normal"               # Default operation
    AGGRESSIVE = "aggressive"       # High Sharpe, tight spreads, more levels
    VERY_AGGRESSIVE = "very_aggressive"  # Exceptional performance
    DEFENSIVE = "defensive"         # High volatility, reduce exposure
    RECOVERY = "recovery"           # Post-drawdown recovery mode


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking for autonomous decisions."""
    
    # PnL tracking
    session_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Rolling performance (last N trades)
    rolling_pnl: List[float] = field(default_factory=list)
    rolling_window: int = 100  # Last 100 trades
    
    # Sharpe calculation
    returns_buffer: List[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    
    # Fill metrics
    total_fills: int = 0
    fill_rate: float = 0.0
    avg_spread_captured: float = 0.0
    
    # Time tracking
    last_update: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    
    def update_pnl(self, pnl: float, equity: float) -> None:
        """Update performance metrics with new PnL."""
        self.session_pnl += pnl
        self.current_equity = equity
        
        # Track peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Rolling PnL
        self.rolling_pnl.append(pnl)
        if len(self.rolling_pnl) > self.rolling_window:
            self.rolling_pnl.pop(0)
        
        # Track consecutive wins/losses
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Calculate return for Sharpe
        if self.peak_equity > 0:
            ret = pnl / self.peak_equity
            self.returns_buffer.append(ret)
            if len(self.returns_buffer) > 1000:
                self.returns_buffer.pop(0)
        
        self._calculate_sharpe()
        self.last_update = time.time()
    
    def _calculate_sharpe(self) -> None:
        """Calculate rolling Sharpe ratio."""
        if len(self.returns_buffer) < 20:
            self.sharpe_ratio = 0.0
            return
        
        returns = np.array(self.returns_buffer[-100:])  # Last 100 returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            # Annualize assuming ~8640 trades/day (HFT)
            self.sharpe_ratio = mean_return / std_return * np.sqrt(8640 * 365)
        else:
            self.sharpe_ratio = 0.0
    
    @property
    def rolling_win_rate(self) -> float:
        """Calculate win rate from rolling PnL."""
        if not self.rolling_pnl:
            return 0.5
        wins = sum(1 for p in self.rolling_pnl if p > 0)
        return wins / len(self.rolling_pnl)


@dataclass
class AutonomousConfig:
    """Configuration for autonomous adjustments."""
    
    # Drawdown thresholds for sizing reduction
    drawdown_reduce_threshold: float = 0.02  # Start reducing at 2% DD
    drawdown_halt_threshold: float = 0.05    # Pause at 5% DD
    
    # Leverage adjustment
    base_leverage: int = 20
    min_leverage: int = 5
    max_leverage: int = 25
    
    # Size adjustment multipliers
    min_size_multiplier: float = 0.3   # Reduce to 30% at worst
    max_size_multiplier: float = 1.5   # Increase to 150% at best
    
    # Sharpe thresholds for aggression
    sharpe_aggressive_threshold: float = 3.0
    sharpe_very_aggressive_threshold: float = 5.0
    sharpe_defensive_threshold: float = 0.5
    
    # Order levels
    min_order_levels: int = 1
    max_order_levels: int = 30
    base_order_levels: int = 20


class AutonomousRiskManager:
    """
    Autonomous risk management that dynamically adjusts trading parameters.
    
    Implements performance-based optimization:
    - Reduces order size by 20% when drawdown > 2%
    - Reduces leverage to 15x when drawdown > 2%
    - Increases order levels to 30 when Sharpe > 3
    - Switches to defensive mode during high volatility
    
    Usage:
        autonomous = AutonomousRiskManager(config)
        adjustments = autonomous.get_adjustments(metrics)
        # Apply adjustments to strategy
    """
    
    def __init__(self, config: Optional[AutonomousConfig] = None):
        self.config = config or AutonomousConfig()
        self.metrics = PerformanceMetrics()
        self.current_mode = AutonomousMode.NORMAL
        self._last_mode_change = time.time()
        self._mode_cooldown = 60  # Minimum 60s between mode changes
        
        # Adjustment history for smoothing
        self._leverage_history: List[int] = []
        self._size_multiplier_history: List[float] = []
        
        logger.info("Autonomous risk manager initialized")
    
    def update_metrics(self, pnl: float, equity: float, fills: int = 0, 
                       fill_rate: float = 0.0, avg_spread: float = 0.0) -> None:
        """Update performance metrics from trading activity."""
        self.metrics.update_pnl(pnl, equity)
        self.metrics.total_fills += fills
        self.metrics.fill_rate = fill_rate
        self.metrics.avg_spread_captured = avg_spread
    
    def get_adjustments(self, current_volatility: float = 0.0) -> Dict:
        """
        Get dynamic parameter adjustments based on current performance.
        
        Returns:
            Dict with keys:
            - leverage: Recommended leverage
            - size_multiplier: Order size multiplier (1.0 = normal)
            - order_levels: Number of order levels
            - mode: Current operating mode
            - spread_adjustment: BPS to add/subtract from base spread
            - should_pause: Whether to pause trading
        """
        adjustments = {
            "leverage": self.config.base_leverage,
            "size_multiplier": 1.0,
            "order_levels": self.config.base_order_levels,
            "mode": self.current_mode.value,
            "spread_adjustment": 0.0,
            "should_pause": False,
            "reason": "",
        }
        
        # 1. Check drawdown - highest priority
        if self.metrics.current_drawdown >= self.config.drawdown_halt_threshold:
            adjustments["should_pause"] = True
            adjustments["mode"] = AutonomousMode.DEFENSIVE.value
            adjustments["reason"] = f"Drawdown {self.metrics.current_drawdown*100:.1f}% >= halt threshold"
            self._update_mode(AutonomousMode.DEFENSIVE)
            return adjustments
        
        # 2. Reduce exposure on moderate drawdown
        if self.metrics.current_drawdown >= self.config.drawdown_reduce_threshold:
            # Linear reduction from 1.0 to min_multiplier as DD goes from 2% to 5%
            dd_progress = (self.metrics.current_drawdown - self.config.drawdown_reduce_threshold) / \
                         (self.config.drawdown_halt_threshold - self.config.drawdown_reduce_threshold)
            dd_progress = min(1.0, max(0.0, dd_progress))
            
            size_mult = 1.0 - (1.0 - self.config.min_size_multiplier) * dd_progress
            adjustments["size_multiplier"] = size_mult * 0.8  # Additional 20% reduction
            adjustments["leverage"] = max(self.config.min_leverage, 
                                          int(self.config.base_leverage * (1 - dd_progress * 0.5)))
            adjustments["spread_adjustment"] = 5.0 * dd_progress  # Widen spreads
            adjustments["mode"] = AutonomousMode.RECOVERY.value
            adjustments["reason"] = f"Drawdown {self.metrics.current_drawdown*100:.1f}%, reducing exposure"
            self._update_mode(AutonomousMode.RECOVERY)
            
        # 3. Increase aggression on high Sharpe
        elif self.metrics.sharpe_ratio >= self.config.sharpe_very_aggressive_threshold:
            adjustments["size_multiplier"] = self.config.max_size_multiplier
            adjustments["order_levels"] = self.config.max_order_levels
            adjustments["spread_adjustment"] = -2.0  # Tighten spreads
            adjustments["mode"] = AutonomousMode.VERY_AGGRESSIVE.value
            adjustments["reason"] = f"Sharpe {self.metrics.sharpe_ratio:.1f} >= {self.config.sharpe_very_aggressive_threshold}"
            self._update_mode(AutonomousMode.VERY_AGGRESSIVE)
            
        elif self.metrics.sharpe_ratio >= self.config.sharpe_aggressive_threshold:
            adjustments["size_multiplier"] = 1.2
            adjustments["order_levels"] = 25
            adjustments["spread_adjustment"] = -1.0
            adjustments["mode"] = AutonomousMode.AGGRESSIVE.value
            adjustments["reason"] = f"Sharpe {self.metrics.sharpe_ratio:.1f} >= {self.config.sharpe_aggressive_threshold}"
            self._update_mode(AutonomousMode.AGGRESSIVE)
            
        # 4. Go conservative on low Sharpe or consecutive losses
        elif self.metrics.sharpe_ratio < self.config.sharpe_defensive_threshold or \
             self.metrics.consecutive_losses >= 5:
            adjustments["size_multiplier"] = 0.7
            adjustments["order_levels"] = 10
            adjustments["spread_adjustment"] = 3.0
            adjustments["mode"] = AutonomousMode.CONSERVATIVE.value
            adjustments["reason"] = f"Low Sharpe ({self.metrics.sharpe_ratio:.1f}) or {self.metrics.consecutive_losses} consecutive losses"
            self._update_mode(AutonomousMode.CONSERVATIVE)
            
        # 5. High volatility check
        if current_volatility > 0.15:  # 15% vol threshold for US500
            adjustments["leverage"] = min(adjustments["leverage"], 10)
            adjustments["spread_adjustment"] += 5.0
            adjustments["mode"] = AutonomousMode.DEFENSIVE.value
            adjustments["reason"] = f"High volatility: {current_volatility*100:.1f}%"
            self._update_mode(AutonomousMode.DEFENSIVE)
        
        # Smooth leverage changes
        self._leverage_history.append(adjustments["leverage"])
        if len(self._leverage_history) > 10:
            self._leverage_history.pop(0)
        adjustments["leverage"] = int(np.mean(self._leverage_history))
        
        return adjustments
    
    def _update_mode(self, new_mode: AutonomousMode) -> None:
        """Update mode with cooldown check."""
        if time.time() - self._last_mode_change >= self._mode_cooldown:
            if new_mode != self.current_mode:
                logger.info(f"Mode change: {self.current_mode.value} -> {new_mode.value}")
                self.current_mode = new_mode
                self._last_mode_change = time.time()
    
    def get_status(self) -> Dict:
        """Get current autonomous system status."""
        return {
            "mode": self.current_mode.value,
            "sharpe_ratio": round(self.metrics.sharpe_ratio, 2),
            "current_drawdown": f"{self.metrics.current_drawdown*100:.2f}%",
            "max_drawdown": f"{self.metrics.max_drawdown*100:.2f}%",
            "session_pnl": f"${self.metrics.session_pnl:.2f}",
            "rolling_win_rate": f"{self.metrics.rolling_win_rate*100:.1f}%",
            "consecutive_losses": self.metrics.consecutive_losses,
            "consecutive_wins": self.metrics.consecutive_wins,
            "total_fills": self.metrics.total_fills,
        }


# Multiprocessing utilities for M4 optimization
def _run_backtest_chunk(args: Tuple) -> Dict:
    """Run backtest on a chunk of data (for parallel processing)."""
    from .backtest import BacktestEngine, BacktestConfig
    
    data_chunk, config_dict = args
    config = BacktestConfig(**config_dict)
    engine = BacktestEngine(config)
    
    # Run on chunk
    result = engine.run(data_chunk)
    return {
        "total_pnl": result.total_pnl,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "trades": result.total_trades,
    }


class ParallelBacktester:
    """
    Multiprocessing backtest runner optimized for Apple M4.
    
    Uses all 10 cores for parallel parameter sweeps and data chunk processing.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"Parallel backtester initialized with {self.n_workers} workers")
    
    def run_parameter_sweep(self, data, param_grid: List[Dict]) -> List[Dict]:
        """
        Run backtests with different parameter combinations in parallel.
        
        Args:
            data: Historical data for backtesting
            param_grid: List of parameter dictionaries to test
            
        Returns:
            List of results for each parameter combination
        """
        logger.info(f"Starting parallel sweep of {len(param_grid)} configurations")
        
        # Prepare arguments
        args = [(data, params) for params in param_grid]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(_run_backtest_chunk, args))
        
        logger.info(f"Parallel sweep complete: {len(results)} results")
        return results
    
    def find_optimal_params(self, data, base_config: Dict, 
                           param_ranges: Dict[str, List]) -> Dict:
        """
        Find optimal parameters using parallel grid search.
        
        Args:
            data: Historical data
            base_config: Base configuration dictionary
            param_ranges: Dict of param_name -> list of values to try
            
        Returns:
            Best parameter configuration
        """
        import itertools
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        param_grid = []
        for combo in combinations:
            config = base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value
            param_grid.append(config)
        
        results = self.run_parameter_sweep(data, param_grid)
        
        # Find best by Sharpe ratio
        best_idx = max(range(len(results)), 
                      key=lambda i: results[i].get("sharpe_ratio", 0))
        
        best_result = results[best_idx]
        best_params = param_grid[best_idx]
        
        logger.info(f"Best params found: Sharpe={best_result['sharpe_ratio']:.2f}, "
                   f"PnL=${best_result['total_pnl']:.2f}")
        
        return {
            "params": best_params,
            "result": best_result,
        }
