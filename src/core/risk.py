"""
Risk Management Module
Handles position sizing, leverage adjustment, and risk monitoring.

WARNING: High-frequency trading with leverage carries significant risk.
This module implements safeguards, but losses can still occur.
Always monitor the bot and be prepared to intervene manually.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from src.utils.config import Config
from src.core.exchange import AccountState, HyperliquidClient, Order, OrderBook, Position
from src.utils.utils import (
    CircularBuffer,
    calculate_ewma_volatility,
    calculate_kelly_fraction,
    calculate_max_drawdown,
)


class RiskLevel(Enum):
    """Risk level classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""

    # Position metrics
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    exposure_pct: float = 0.0  # As percentage of max allowed

    # Margin metrics
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    available_margin: float = 0.0
    distance_to_liquidation: float = 1.0  # 1.0 = no position

    # PnL metrics
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0

    # Volatility
    current_volatility: float = 0.0
    volatility_ratio: float = 0.0  # Current vol / threshold

    # Funding rate - UPDATED: hedge threshold per recommendations
    current_funding_rate: float = 0.0
    should_hedge_funding: bool = False  # True if funding > 0.04%
    funding_hedge_direction: str = "none"  # "short" if positive, "long" if negative

    # Risk classification
    risk_level: RiskLevel = RiskLevel.LOW
    recommended_leverage: int = 10  # Default to x10, will be set by RiskManager

    # STRESS-TEST safeguards
    current_imbalance: float = 0.0  # Current delta/position imbalance
    max_imbalance_duration: float = 0.0  # Time in seconds at high imbalance
    funding_net_cost: float = 0.0  # Cumulative funding payments
    hard_stop_triggered: bool = False  # Imbalance >0.5% trigger
    auto_reduce_triggered: bool = False  # PnL <-1% trigger

    # Flags
    should_reduce_exposure: bool = False
    should_pause_trading: bool = False
    emergency_close: bool = False


@dataclass
class RiskState:
    """Internal risk management state."""

    peak_equity: float = 0.0
    starting_equity: float = 0.0
    session_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    total_win_amount: float = 0.0
    total_loss_amount: float = 0.0
    last_check_time: float = 0.0
    consecutive_losses: int = 0
    consecutive_losing_fills: int = 0  # NEW: Track adverse selection
    price_history: CircularBuffer = field(default_factory=lambda: CircularBuffer(1000))

    # STRESS-TEST tracking for x25
    imbalance_start_time: float = 0.0  # When imbalance exceeded threshold
    cumulative_funding_cost: float = 0.0  # Total funding paid
    last_funding_time: float = 0.0
    current_delta: float = 0.0  # Current position delta
    
    # ENHANCEMENT: Taker volume tracking
    taker_volume_24h: float = 0.0  # Taker volume in last 24h
    maker_volume_24h: float = 0.0  # Maker volume in last 24h
    taker_ratio_threshold: float = 0.10  # Max 10% taker volume


class RiskManager:
    """
    Risk management system for the HFT bot.

    Responsibilities:
    - Monitor position exposure and margin usage
    - Calculate optimal position sizes
    - Dynamically adjust leverage based on conditions
    - Detect and respond to dangerous situations
    - Track PnL and drawdown
    - Implement stop-loss and circuit breakers

    Usage:
        config = Config.load()
        client = HyperliquidClient(config)
        risk_manager = RiskManager(config, client)

        # In main loop
        metrics = await risk_manager.check_risk()
        if metrics.should_pause_trading:
            await cancel_all_orders()
    """

    def __init__(self, config: Config, client: HyperliquidClient):
        """Initialize risk manager."""
        self.config = config
        self.client = client
        self.state = RiskState()
        self._initialized = False
        
        # FILL-BASED TRACKING: Bypass broken balance API for HIP-3
        # Track equity from configured collateral + accumulated PnL from fills
        self._fill_based_equity = config.trading.collateral
        self._fill_based_pnl = 0.0
        self._fill_based_fees = 0.0
        self._use_fill_tracking = True  # Enable fill-based tracking

        # Configure thresholds - TIGHTENED for safety per recommendations
        self.max_drawdown = config.risk.max_drawdown  # 5% max drawdown
        self.stop_loss_pct = config.risk.stop_loss_pct  # 1-2% stop loss
        self.min_margin_ratio = config.risk.min_margin_ratio
        self.high_vol_threshold = config.risk.high_vol_threshold
        self.funding_rate_threshold = config.risk.funding_rate_hedge_threshold  # 0.04%

        # Leverage levels for x25 STRESS-TEST per recommendations:
        # Leverage levels based on configured leverage
        # LOW: configured leverage (e.g., x10 production, x25 stress-test)
        # MEDIUM: 50% of configured
        # HIGH: minimal
        # CRITICAL: close
        configured_leverage = config.trading.leverage
        self.leverage_levels = {
            RiskLevel.LOW: configured_leverage,  # Use configured leverage
            RiskLevel.MEDIUM: max(5, configured_leverage // 2),  # 50% of configured
            RiskLevel.HIGH: 3,  # Minimal exposure
            RiskLevel.CRITICAL: 1,  # Close positions
        }

        # Track last leverage adjustment to avoid spam
        self._last_leverage_adjust_time = 0
        self._leverage_adjust_cooldown = 60  # 60 seconds between adjustments

        # Track last risk log to avoid spam
        self._last_risk_log_time = 0
        self._risk_log_cooldown = 30  # Log at most every 30 seconds

        # STRESS-TEST specific thresholds
        # max_imbalance_pct: position_value / equity threshold
        # With 10x leverage, a 0.001 BTC position @ $91k = $91 / $1000 = 9.1%
        # Set to 150% to allow more headroom for position management
        self.max_imbalance_pct = getattr(config.risk, "max_imbalance_pct", 1.5)  # 150%
        self.auto_reduce_pnl_pct = getattr(config.risk, "auto_reduce_pnl_pct", -0.01)  # -1%

    def _is_low_liq_prob(self) -> bool:
        """Check if conditions allow for x25 leverage (liq prob <5%)."""
        # This would require Monte Carlo validation - default to False
        # User should validate with backtest before enabling x25
        return False

    async def initialize(self) -> None:
        """Initialize risk manager with current account state."""
        try:
            account = await self.client.get_account_state()
            if account and account.equity > 1.0:
                self.state.starting_equity = account.equity
                self.state.peak_equity = account.equity
                self._fill_based_equity = account.equity  # Sync fill-based with actual
                logger.info(f"Risk manager initialized with equity: ${account.equity:,.2f}")
            else:
                # Balance API returned $0, use configured collateral
                self.state.starting_equity = self.config.trading.collateral
                self.state.peak_equity = self.config.trading.collateral
                self._fill_based_equity = self.config.trading.collateral
                logger.info(f"Risk manager initialized with configured collateral: ${self.config.trading.collateral:,.2f} (balance API returned $0)")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            raise

    def update_from_fill(self, closed_pnl: float, fee: float) -> None:
        """
        Update fill-based equity tracking from a trade fill.
        
        This bypasses the broken balance API for HIP-3 by tracking
        equity directly from fill data.
        
        Args:
            closed_pnl: The closedPnl field from the fill
            fee: The fee field from the fill
        """
        self._fill_based_pnl += closed_pnl
        self._fill_based_fees += fee
        # Update equity: starting + realized PnL - fees
        self._fill_based_equity = self.config.trading.collateral + self._fill_based_pnl - self._fill_based_fees
        
        # Also update state for drawdown tracking
        if self._fill_based_equity > self.state.peak_equity:
            self.state.peak_equity = self._fill_based_equity
        
        logger.debug(f"Fill-based equity: ${self._fill_based_equity:.2f} (PnL: ${self._fill_based_pnl:.2f}, fees: ${self._fill_based_fees:.2f})")

    def get_fill_based_equity(self) -> float:
        """Get the fill-based equity value."""
        return self._fill_based_equity

    def sync_fills_from_exchange(self, fills: list) -> None:
        """
        Sync fill-based tracking from exchange fills history.
        
        Call this on startup to sync with actual trading history.
        """
        total_pnl = 0.0
        total_fees = 0.0
        
        for fill in fills:
            coin = fill.get('coin', '')
            if 'US500' in coin or coin == 'km:US500':
                total_pnl += float(fill.get('closedPnl', 0))
                total_fees += float(fill.get('fee', 0))
        
        self._fill_based_pnl = total_pnl
        self._fill_based_fees = total_fees
        self._fill_based_equity = self.config.trading.collateral + total_pnl - total_fees
        
        logger.info(f"Synced fill-based tracking: equity=${self._fill_based_equity:.2f} (PnL: ${total_pnl:.2f}, fees: ${total_fees:.2f})")

    async def check_risk(self) -> RiskMetrics:
        """
        Perform comprehensive risk check.

        STRESS-TEST at x25:
        - Hard stop if imbalance >0.5%
        - Auto-reduce to x10 if PnL -1%
        - Track imbalance duration and funding net cost

        Returns:
            RiskMetrics with current risk assessment
        """
        metrics = RiskMetrics()

        try:
            # Get current state
            account = await self.client.get_account_state()
            position = await self.client.get_position()
            orderbook = await self.client.get_orderbook()
            funding_rate = await self.client.get_funding_rate()

            if not account:
                logger.warning("Could not fetch account state for risk check")
                metrics.risk_level = RiskLevel.CRITICAL
                metrics.should_reduce_exposure = True
                return metrics

            # CRITICAL: Check for zero or near-zero equity (no margin)
            # BYPASS: For HIP-3, balance API is unreliable (USDH split across spot/perp/margin)
            # Use fill-based tracking instead when balance shows $0 but we have configured collateral
            if account.equity < 1.0:
                if self._use_fill_tracking and self._fill_based_equity > 1.0:
                    # Balance API returned $0 but we have fill-based equity
                    # Override account equity with our tracked value
                    logger.debug(
                        f"Balance API returned ${account.equity:.2f}, using fill-based equity: ${self._fill_based_equity:.2f}"
                    )
                    account.equity = self._fill_based_equity
                    account.available_balance = self._fill_based_equity * 0.8  # Conservative 80% available
                else:
                    logger.error(
                        f"❌ CRITICAL: No margin available! equity=${account.equity:.2f} | "
                        f"Cannot trade without funds - deposit USDH to continue"
                    )
                    metrics.risk_level = RiskLevel.CRITICAL
                    metrics.should_reduce_exposure = True
                    metrics.current_drawdown = 1.0  # 100% loss
                    return metrics

            # Update equity tracking
            self._update_equity_tracking(account.equity)

            # Calculate position metrics
            metrics = self._calculate_position_metrics(account, position, metrics)

            # Calculate margin metrics
            metrics = self._calculate_margin_metrics(account, position, metrics)

            # Calculate PnL metrics
            metrics = self._calculate_pnl_metrics(account, metrics)

            # Calculate volatility metrics
            if orderbook and orderbook.mid_price:
                self.state.price_history.append(orderbook.mid_price)
            metrics = self._calculate_volatility_metrics(metrics)

            # Calculate funding rate metrics and hedging signal
            metrics = self._calculate_funding_metrics(funding_rate, metrics)

            # STRESS-TEST: Calculate imbalance and check hard stops
            metrics = self._check_stress_test_safeguards(account, position, metrics)

            # Determine risk level and recommendations
            metrics = self._assess_risk_level(metrics, position)

            # Log if risk level is elevated (throttled to avoid spam)
            import time

            current_time = time.time()
            if metrics.risk_level != RiskLevel.LOW:
                if current_time - self._last_risk_log_time > self._risk_log_cooldown:
                    logger.warning(
                        f"Risk Level: {metrics.risk_level.value} | "
                        f"Exposure: {metrics.exposure_pct:.1f}% | "
                        f"Drawdown: {metrics.current_drawdown:.2%} | "
                        f"Vol: {metrics.current_volatility:.1f}%"
                    )
                    self._last_risk_log_time = current_time

            # Log funding rate hedging signal
            if metrics.should_hedge_funding:
                logger.info(
                    f"Funding rate {metrics.current_funding_rate:.4%} > threshold, "
                    "biasing towards short positions"
                )

            return metrics

        except Exception as e:
            import traceback

            logger.error(f"Error in risk check: {e}\n{traceback.format_exc()}")
            # Return conservative metrics on error
            metrics.risk_level = RiskLevel.HIGH
            metrics.should_reduce_exposure = True
            return metrics

    def _update_equity_tracking(self, current_equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        self.state.session_pnl = current_equity - self.state.starting_equity

    def _calculate_position_metrics(
        self, account: AccountState, position: Optional[Position], metrics: RiskMetrics
    ) -> RiskMetrics:
        """Calculate position-related metrics."""
        if position:
            metrics.net_exposure = position.size * position.mark_price
            metrics.gross_exposure = abs(metrics.net_exposure)
        else:
            metrics.net_exposure = 0.0
            metrics.gross_exposure = 0.0

        max_exposure = self.config.trading.max_net_exposure
        metrics.exposure_pct = (
            (metrics.gross_exposure / max_exposure * 100) if max_exposure > 0 else 0
        )

        return metrics

    def _calculate_margin_metrics(
        self, account: AccountState, position: Optional[Position], metrics: RiskMetrics
    ) -> RiskMetrics:
        """Calculate margin-related metrics."""
        metrics.margin_used = account.margin_used
        metrics.margin_ratio = account.margin_ratio
        metrics.available_margin = account.available_balance

        # Calculate distance to liquidation
        if position and position.liquidation_price > 0 and position.mark_price > 0:
            if position.is_long:
                metrics.distance_to_liquidation = (
                    position.mark_price - position.liquidation_price
                ) / position.mark_price
            else:
                metrics.distance_to_liquidation = (
                    position.liquidation_price - position.mark_price
                ) / position.mark_price
        else:
            metrics.distance_to_liquidation = 1.0

        return metrics

    def _calculate_pnl_metrics(self, account: AccountState, metrics: RiskMetrics) -> RiskMetrics:
        """Calculate PnL-related metrics."""
        metrics.unrealized_pnl = account.unrealized_pnl
        metrics.realized_pnl = self.state.session_pnl - account.unrealized_pnl

        # Calculate current drawdown from peak
        if self.state.peak_equity > 0:
            metrics.current_drawdown = (
                self.state.peak_equity - account.equity
            ) / self.state.peak_equity

        metrics.max_drawdown = max(metrics.current_drawdown, metrics.max_drawdown)

        return metrics

    def _check_stress_test_safeguards(
        self, account: AccountState, position: Optional[Position], metrics: RiskMetrics
    ) -> RiskMetrics:
        """
        STRESS-TEST safeguards for x25 validation.

        Hard stops:
        - Pause trading if imbalance (delta) >0.5%
        - Auto-reduce leverage to x10 if unrealized PnL -1%

        Metrics tracked:
        - Max imbalance duration (time at high delta)
        - Funding net cost (cumulative payments)
        """
        import time

        now = time.time()

        # Calculate current imbalance (delta as % of collateral)
        if position and position.size != 0 and account.equity > 0:
            position_value = abs(position.size * position.mark_price)
            metrics.current_imbalance = position_value / account.equity
            self.state.current_delta = position.size
        else:
            metrics.current_imbalance = 0.0
            self.state.current_delta = 0.0

        # Hard stop check: Imbalance >0.5%
        if metrics.current_imbalance > self.max_imbalance_pct:
            if self.state.imbalance_start_time == 0:
                self.state.imbalance_start_time = now

            metrics.max_imbalance_duration = now - self.state.imbalance_start_time

            # HARD STOP: Pause trading if imbalance persists
            metrics.hard_stop_triggered = True
            metrics.should_pause_trading = True
            logger.critical(
                f"HARD STOP: Imbalance {metrics.current_imbalance:.2%} > "
                f"{self.max_imbalance_pct:.2%} threshold! Duration: {metrics.max_imbalance_duration:.1f}s"
            )
        else:
            self.state.imbalance_start_time = 0
            metrics.max_imbalance_duration = 0

        # Auto-reduce check: PnL -1%
        pnl_pct = (
            metrics.unrealized_pnl / self.state.starting_equity
            if self.state.starting_equity > 0
            else 0
        )
        if pnl_pct < self.auto_reduce_pnl_pct:
            metrics.auto_reduce_triggered = True
            metrics.recommended_leverage = self.leverage_levels[RiskLevel.MEDIUM]  # x10
            logger.warning(
                f"AUTO-REDUCE: Unrealized PnL {pnl_pct:.2%} < "
                f"{self.auto_reduce_pnl_pct:.2%}. Reducing to {metrics.recommended_leverage}x"
            )

        # Track cumulative funding cost
        if metrics.current_funding_rate != 0 and position and position.size != 0:
            # Funding is paid every 8 hours, but we track per check
            time_since_last = (
                now - self.state.last_funding_time if self.state.last_funding_time > 0 else 0
            )
            if time_since_last >= 8 * 3600:  # Every 8 hours
                # Funding cost = position_value * funding_rate
                position_value = position.size * position.mark_price
                funding_payment = position_value * metrics.current_funding_rate
                self.state.cumulative_funding_cost += funding_payment
                self.state.last_funding_time = now

        metrics.funding_net_cost = self.state.cumulative_funding_cost

        return metrics

    def _calculate_volatility_metrics(self, metrics: RiskMetrics) -> RiskMetrics:
        """Calculate volatility metrics with proper smoothing."""
        prices = self.state.price_history.get_array()

        if len(prices) >= 60:  # Need at least 60 samples for stable estimate
            # Use 60-sample EWMA (~2 minutes of data at 2s intervals)
            # Assume samples represent ~2 second intervals for annualization
            refresh_interval = max(self.config.execution.quote_refresh_interval, 1.0)
            samples_per_minute = 60.0 / refresh_interval
            metrics.current_volatility = calculate_ewma_volatility(
                prices, span=60, samples_per_minute=samples_per_minute
            )
        elif len(prices) >= 20:
            # Fallback for startup
            refresh_interval = max(self.config.execution.quote_refresh_interval, 1.0)
            samples_per_minute = 60.0 / refresh_interval
            metrics.current_volatility = calculate_ewma_volatility(
                prices, span=20, samples_per_minute=samples_per_minute
            )
        else:
            # Warmup period: use conservative fallback estimate
            # US500 typically 10-15% annualized volatility = 5-7% for shorter periods
            metrics.current_volatility = 5.0

        if self.high_vol_threshold > 0:
            metrics.volatility_ratio = metrics.current_volatility / self.high_vol_threshold

        return metrics

    def _calculate_funding_metrics(
        self, funding_rate: Optional[float], metrics: RiskMetrics
    ) -> RiskMetrics:
        """
        Calculate funding rate metrics and hedging signal.

        Per recommendations: Monitor and hedge funding rates (>0.04%).
        If funding rate is positive >0.04%, short skew to earn funding.
        If funding rate is negative <-0.04%, long skew to earn funding.
        """
        if funding_rate is not None:
            metrics.current_funding_rate = funding_rate

            # Hedge if funding rate is significantly positive (longs pay shorts)
            # Per recommendations: short if positive >0.04%
            if funding_rate > self.funding_rate_threshold:
                metrics.should_hedge_funding = True
                metrics.funding_hedge_direction = "short"  # Bias short to earn funding
            # Also consider very negative funding (shorts pay longs)
            elif funding_rate < -self.funding_rate_threshold:
                metrics.should_hedge_funding = True
                metrics.funding_hedge_direction = "long"  # Bias long to earn funding
            else:
                metrics.should_hedge_funding = False
                metrics.funding_hedge_direction = "none"
        else:
            metrics.current_funding_rate = 0.0
            metrics.should_hedge_funding = False
            metrics.funding_hedge_direction = "none"

        return metrics

    def check_taker_volume_ratio(self, metrics: RiskMetrics) -> bool:
        """
        Check if taker volume ratio is within acceptable limits.
        
        PROFESSIONAL MM SAFEGUARD:
        - Target: >90% maker volume (earn rebates)
        - Warning: <90% maker (some taker fills)
        - Pause: <80% maker (excessive taker fills)
        
        Returns True if trading should be paused.
        """
        total_volume = self.state.taker_volume_24h + self.state.maker_volume_24h
        if total_volume == 0:
            return False
        
        taker_ratio = self.state.taker_volume_24h / total_volume
        maker_ratio = 1.0 - taker_ratio
        
        if maker_ratio < 0.80:  # Less than 80% maker
            logger.warning(
                f"TAKER VOLUME TOO HIGH: {taker_ratio:.1%} taker, {maker_ratio:.1%} maker - "
                "PAUSING to improve maker ratio"
            )
            metrics.should_pause_trading = True
            return True
        elif maker_ratio < 0.90:  # Less than 90% maker
            logger.info(
                f"Taker volume elevated: {taker_ratio:.1%} taker, {maker_ratio:.1%} maker"
            )
        
        return False

    def check_consecutive_losing_fills(self, consecutive_losses: int, metrics: RiskMetrics) -> None:
        """
        Check for consecutive losing fills (adverse selection).
        
        ENHANCEMENT: Auto-pause on 3 consecutive losing fills.
        This indicates severe adverse selection or picking-off.
        """
        if consecutive_losses >= 3:
            logger.warning(
                f"CONSECUTIVE LOSING FILLS: {consecutive_losses} in a row - "
                "Widening spreads and reducing activity"
            )
            # Don't pause immediately, but flag for spread widening
            # The strategy layer will handle quote fading
            metrics.should_reduce_exposure = True
        
        if consecutive_losses >= 5:
            logger.error(
                f"SEVERE ADVERSE SELECTION: {consecutive_losses} consecutive losses - "
                "PAUSING trading temporarily"
            )
            metrics.should_pause_trading = True

    def _assess_risk_level(self, metrics: RiskMetrics, position: Optional[Position]) -> RiskMetrics:
        """Assess overall risk level and set recommendations."""
        risk_scores = []
        
        # Check if we have an active position
        has_position = position is not None and abs(position.size) > 0.0001

        # Score based on margin ratio (higher = more risk)
        if metrics.margin_ratio > 0.8:
            risk_scores.append(4)  # Critical
        elif metrics.margin_ratio > 0.6:
            risk_scores.append(3)  # High
        elif metrics.margin_ratio > 0.4:
            risk_scores.append(2)  # Medium
        else:
            risk_scores.append(1)  # Low

        # Score based on drawdown
        # CRITICAL FIX: Only treat drawdown as critical risk if we have an active position
        # Historical drawdown without position shouldn't prevent market making
        if metrics.current_drawdown > self.max_drawdown:
            risk_scores.append(4 if has_position else 2)  # Critical if holding, Medium if flat
        elif metrics.current_drawdown > self.max_drawdown * 0.7:
            risk_scores.append(3 if has_position else 2)  # High if holding, Medium if flat
        elif metrics.current_drawdown > self.max_drawdown * 0.5:
            risk_scores.append(2)
        else:
            risk_scores.append(1)

        # Score based on distance to liquidation
        if metrics.distance_to_liquidation < 0.05:
            risk_scores.append(4)
        elif metrics.distance_to_liquidation < 0.10:
            risk_scores.append(3)
        elif metrics.distance_to_liquidation < 0.15:
            risk_scores.append(2)
        else:
            risk_scores.append(1)

        # Score based on volatility
        if metrics.volatility_ratio > 1.5:
            risk_scores.append(3)
        elif metrics.volatility_ratio > 1.0:
            risk_scores.append(2)
        else:
            risk_scores.append(1)

        # Aggregate risk level
        max_score = max(risk_scores)
        avg_score = sum(risk_scores) / len(risk_scores)

        if max_score >= 4 or avg_score >= 3.5:
            metrics.risk_level = RiskLevel.CRITICAL
        elif max_score >= 3 or avg_score >= 2.5:
            metrics.risk_level = RiskLevel.HIGH
        elif max_score >= 2 or avg_score >= 1.5:
            metrics.risk_level = RiskLevel.MEDIUM
        else:
            metrics.risk_level = RiskLevel.LOW

        # Set recommended leverage
        metrics.recommended_leverage = self.leverage_levels[metrics.risk_level]

        # Set action flags
        metrics.should_reduce_exposure = metrics.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        
        # Only pause trading if CRITICAL AND we have a position to protect
        # For market-making with no position, we should keep quoting
        has_position = position is not None and abs(position.size) > 0.0001
        metrics.should_pause_trading = metrics.risk_level == RiskLevel.CRITICAL and has_position

        # Emergency close only if we have a position AND:
        # - margin is critically low, OR
        # - drawdown severely exceeded, OR  
        # - margin ratio dangerously high
        if has_position and (
            metrics.distance_to_liquidation < 0.03
            or metrics.current_drawdown > self.max_drawdown * 1.5  # Increased from 1.2 to 1.5
            or metrics.margin_ratio > 0.9
        ):
            metrics.emergency_close = True
            metrics.should_pause_trading = True
        
        # NEVER emergency close when we have no position - just log a warning
        if not has_position and metrics.current_drawdown > self.max_drawdown:
            logger.warning(
                f"Drawdown {metrics.current_drawdown:.2%} exceeds threshold {self.max_drawdown:.2%} "
                f"but no position - continuing to quote"
            )

        return metrics

    async def adjust_leverage(self, metrics: RiskMetrics) -> bool:
        """
        Adjust leverage based on risk assessment.

        DISABLED for production: Leverage stays at configured value.
        Only adjusts DOWN (never up) in case of elevated risk.

        Args:
            metrics: Current risk metrics

        Returns:
            True if leverage was adjusted
        """
        # In production mode, don't auto-adjust leverage
        # Leverage is set at startup and only changes if risk is elevated AND we have exposure
        current_leverage = self.config.trading.leverage

        # Only reduce leverage if risk is HIGH or CRITICAL
        if metrics.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return False

        # Don't reduce leverage if we have no exposure (nothing at risk)
        if metrics.exposure_pct < 1.0:  # Less than 1% exposure
            return False

        recommended = self.leverage_levels[metrics.risk_level]

        # Only adjust down, never up
        if recommended >= current_leverage:
            return False

        import time

        current_time = time.time()

        # Check cooldown (60 seconds between adjustments)
        if current_time - self._last_leverage_adjust_time < self._leverage_adjust_cooldown:
            return False

        logger.info(
            f"Reducing leverage from {current_leverage}x to {recommended}x "
            f"(risk level: {metrics.risk_level.value})"
        )

        self._last_leverage_adjust_time = current_time

        try:
            success = await self.client.update_leverage(recommended)
            if success:
                self.config.trading.leverage = recommended
                return True
            else:
                logger.warning("Failed to reduce leverage - will retry after cooldown")
        except Exception as e:
            logger.warning(f"Leverage adjustment failed: {e}")

        return False

    def calculate_order_size(
        self,
        price: float,
        side: str,
        metrics: Optional[RiskMetrics] = None,
        funding_rate: float = 0.0,
    ) -> float:
        """
        Calculate safe order size using KELLY SIZING: trade_size = (edge - funding_cost) / vol².

        Dynamic sizing based on:
        - Expected edge (win_rate * avg_win - loss_rate * avg_loss)
        - Funding cost deduction (reduces size in high funding periods)
        - Current volatility squared (higher vol = smaller size)
        - Risk level adjustments

        Args:
            price: Current price
            side: 'buy' or 'sell'
            metrics: Optional pre-calculated risk metrics
            funding_rate: Current 8h funding rate (e.g., 0.0001 = 0.01%)

        Returns:
            Safe order size in base currency (BTC)
        """
        # CRITICAL: Check for zero margin - but use fill-based tracking as fallback
        if metrics and metrics.available_margin < 1.0:
            # BYPASS: Use fill-based equity when balance API returns $0
            if self._use_fill_tracking and self._fill_based_equity > 1.0:
                logger.debug(f"Margin API returned ${metrics.available_margin:.2f}, using fill-based: ${self._fill_based_equity:.2f}")
                # Continue with fill-based equity
            else:
                logger.warning(f"No margin available (${metrics.available_margin:.2f}) - returning zero size")
                return 0.0

        # Calculate Kelly-optimal size: (edge - funding_cost) / vol²
        kelly_size = self._calculate_kelly_dynamic_size(price, metrics, funding_rate)

        # Apply risk level multipliers
        if metrics:
            # CRITICAL FIX: Allow orders even at CRITICAL risk if we have no exposure
            # This prevents historical drawdown from blocking new market making activity
            if metrics.risk_level == RiskLevel.CRITICAL:
                if metrics.exposure_pct > 0.01:  # Have active position
                    return 0.0  # No new orders when position at risk
                else:
                    # No position - allow conservative market making
                    kelly_size *= 0.1  # Very conservative 10% of normal size
            elif metrics.risk_level == RiskLevel.HIGH:
                kelly_size *= 0.25
            elif metrics.risk_level == RiskLevel.MEDIUM:
                kelly_size *= 0.5

        # Cap based on max exposure
        max_position_usd = self.config.trading.max_net_exposure
        max_size = max_position_usd / price if price > 0 else 0

        # Minimum size based on ORDER_SIZE_FRACTION for professional MM multi-level quoting
        # Use ORDER_SIZE_FRACTION * collateral as base size per level
        # For US500 with 0.3 fraction and $1000 collateral: $300 / $689 = 0.435 contracts
        base_size_from_fraction = (self.config.trading.order_size_fraction * self.config.trading.collateral) / price if price > 0 else 0
        
        # Also respect symbol's minimum lot size
        symbol = self.config.trading.symbol.upper()
        if symbol == "US500":
            symbol_min_size = 0.1  # szDecimals=1 -> 10^(-1) = 0.1 contracts
        else:
            symbol_min_size = max(0.00012, 10.0 / price) if price > 0 else 0.00012  # Ensures $10+ notional
        
        # Use the larger of fraction-based or symbol minimum
        min_size = max(base_size_from_fraction, symbol_min_size)

        final_size = max(min_size, min(kelly_size, max_size))
        
        # Log sizing details (only when size is near minimum to avoid spam)
        if final_size == min_size:
            logger.debug(
                f"Order size capped at minimum: {final_size:.6f} (kelly={kelly_size:.6f}, "
                f"min={min_size:.6f}, max={max_size:.6f}, price=${price:.2f})"
            )
        
        return final_size

    def _calculate_kelly_dynamic_size(
        self, price: float, metrics: Optional[RiskMetrics] = None, funding_rate: float = 0.0
    ) -> float:
        """
        Calculate dynamic Kelly size: trade_size = (edge - funding_cost) / vol².

        REFINED for <5% liq prob at x25:
        - Deducts funding cost from expected edge
        - At 0.01% funding/8h, annual cost = 0.01% * 3 * 365 = 10.95%
        - This reduces size during high funding periods

        Formula: size = ((expected_edge - funding_cost) / volatility²) * base_fraction

        This ensures:
        - Larger sizes when edge is high and vol is low
        - Smaller sizes when vol is high (protective)
        - Smaller sizes when funding costs eat into edge
        - Adapts to current market conditions
        """
        # Get volatility (annualized, convert to decimal)
        if metrics and metrics.current_volatility > 0:
            vol = metrics.current_volatility / 100  # e.g., 50% -> 0.5
        else:
            vol = 0.15  # Default 15% annual vol

        # Calculate expected edge from historical performance
        if self.state.total_trades >= 20:
            win_rate = self.state.winning_trades / self.state.total_trades
            avg_win = self.state.total_win_amount / max(self.state.winning_trades, 1)
            avg_loss = abs(
                self.state.total_loss_amount
                / max(self.state.total_trades - self.state.winning_trades, 1)
            )

            # Expected edge per trade
            edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            # Default assumption: 0.1% edge (market making typical)
            edge = 0.001

        # FUNDING COST DEDUCTION: convert 8h funding rate to per-trade cost
        # Assume ~5 second rebalance = 17280 trades/day = 2160 trades/8h funding period
        # funding_cost_per_trade = funding_rate / 2160
        funding_cost_per_trade = abs(funding_rate) / 2160 if funding_rate != 0 else 0

        # Subtract funding cost from edge
        adjusted_edge = max(edge - funding_cost_per_trade, 0.0001)  # Floor at 0.01%

        # Kelly formula: size = (edge - funding_cost) / vol²
        # Apply fractional Kelly (half-Kelly for safety)
        vol_squared = vol * vol
        if vol_squared > 0:
            kelly_fraction = (adjusted_edge / vol_squared) * 0.5  # Half-Kelly
        else:
            kelly_fraction = 0.02  # Default 2%

        # Cap Kelly fraction at 10% of collateral for safety
        kelly_fraction = max(0.005, min(kelly_fraction, 0.10))

        # Convert to position size
        base_size_usd = self.config.trading.collateral * kelly_fraction
        return base_size_usd / price

    def calculate_kelly_size(
        self,
        price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            price: Current price
            win_rate: Historical win rate (default from state)
            avg_win: Average win amount (default from state)
            avg_loss: Average loss amount (default from state)

        Returns:
            Kelly-optimal position size
        """
        # Use historical stats if not provided
        if win_rate is None:
            if self.state.total_trades > 20:
                win_rate = self.state.winning_trades / self.state.total_trades
            else:
                win_rate = 0.5  # Default assumption

        if avg_win is None:
            if self.state.winning_trades > 0:
                avg_win = self.state.total_win_amount / self.state.winning_trades
            else:
                avg_win = 0.001  # Default 0.1%

        if avg_loss is None:
            losing_trades = self.state.total_trades - self.state.winning_trades
            if losing_trades > 0:
                avg_loss = abs(self.state.total_loss_amount / losing_trades)
            else:
                avg_loss = 0.001  # Default 0.1%

        # Calculate Kelly fraction (capped at 25%)
        kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        # Apply to collateral
        position_usd = self.config.trading.collateral * kelly * self.config.trading.leverage
        return position_usd / price

    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade for statistics.

        Args:
            pnl: Trade profit/loss
        """
        self.state.total_trades += 1

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.total_win_amount += pnl
            self.state.consecutive_losses = 0
        else:
            self.state.total_loss_amount += abs(pnl)
            self.state.consecutive_losses += 1

        # Log if consecutive losses exceed threshold
        if self.state.consecutive_losses >= 5:
            logger.warning(f"Consecutive losses: {self.state.consecutive_losses}")

    def should_stop_loss(self, position: Position) -> bool:
        """
        Check if position should be stopped out.

        Args:
            position: Current position

        Returns:
            True if stop loss should trigger
        """
        if position.entry_price <= 0:
            return False

        # Calculate adverse move percentage
        if position.is_long:
            adverse_move = (position.entry_price - position.mark_price) / position.entry_price
        else:
            adverse_move = (position.mark_price - position.entry_price) / position.entry_price

        return adverse_move > self.stop_loss_pct

    def get_trade_stats(self) -> Dict:
        """Get trading statistics."""
        total = self.state.total_trades
        wins = self.state.winning_trades
        losses = total - wins

        win_rate = wins / total if total > 0 else 0
        avg_win = self.state.total_win_amount / wins if wins > 0 else 0
        avg_loss = abs(self.state.total_loss_amount / losses) if losses > 0 else 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": (
                (self.state.total_win_amount / abs(self.state.total_loss_amount))
                if self.state.total_loss_amount != 0
                else float("inf")
            ),
            "session_pnl": self.state.session_pnl,
            "peak_equity": self.state.peak_equity,
            "current_drawdown": (
                (self.state.peak_equity - self.state.starting_equity - self.state.session_pnl)
                / self.state.peak_equity
                if self.state.peak_equity > 0
                else 0
            ),
        }

    async def emergency_close_all(self) -> bool:
        """
        Emergency close all positions.

        Returns:
            True if successfully closed
        """
        logger.critical("EMERGENCY CLOSE triggered!")

        try:
            # Cancel all orders first
            await self.client.cancel_all_orders()

            # Get current position
            position = await self.client.get_position()

            if position and position.size != 0:
                # Close with market order
                from src.core.exchange import OrderRequest, OrderSide, OrderType, TimeInForce

                close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

                # Get current price for limit order
                orderbook = await self.client.get_orderbook()
                if orderbook:
                    # Use very aggressive price to ensure fill in emergency
                    # 3% slippage is acceptable for emergency exits
                    if close_side == OrderSide.SELL:
                        price = orderbook.best_bid * 0.97 if orderbook.best_bid else 0
                    else:
                        price = orderbook.best_ask * 1.03 if orderbook.best_ask else 0

                    close_order = OrderRequest(
                        symbol=position.symbol,
                        side=close_side,
                        size=abs(position.size),
                        price=price,
                        order_type=OrderType.LIMIT,
                        time_in_force=TimeInForce.IOC,  # Immediate or cancel
                        reduce_only=True,
                    )

                    result = await self.client.place_order(close_order)
                    if result:
                        logger.info(f"Emergency close order placed: {result}")
                        return True

            return True  # No position to close

        except Exception as e:
            logger.error(f"Emergency close failed: {e}")
            return False
