"""
RISK.PY ENHANCEMENTS FOR US500-USDH MARGIN CAPS
Add to /Users/nheosdisplay/VSC/AMM/AMM-500/src/risk.py

1. USDH margin ratio monitoring (<90% cap)
2. Auto-hedge on funding >0.01%
3. US500-specific risk parameters
"""

# =============================================================================
# Add to RiskMetrics dataclass
# =============================================================================

@dataclass
class RiskMetrics:
    """Risk metrics with USDH margin tracking."""
    # ... existing fields ...
    
    # USDH-specific
    usdh_margin_used: float = 0.0
    usdh_margin_ratio: float = 0.0
    usdh_margin_available: float = 0.0
    usdh_margin_warning: bool = False  # True if >80%
    usdh_margin_critical: bool = False  # True if >90%
    
    # US500-specific
    us500_vol: float = 0.0  # Current realized vol
    us500_target_vol: float = 0.12  # Target 12% annualized
    us500_vol_ratio: float = 0.0  # Current / target
    
    # Funding hedge (US500: lower threshold 0.01% vs 0.04% crypto)
    should_hedge_funding: bool = False
    funding_hedge_threshold: float = 0.0001  # 0.01% for US500
    funding_cost_daily: float = 0.0


# =============================================================================
# Add to RiskManager class
# =============================================================================

async def assess_risk_us500_usdh(self) -> RiskMetrics:
    """
    Enhanced risk assessment for US500-USDH with margin monitoring.
    
    Checks:
    1. USDH margin ratio (<90% required)
    2. US500 volatility vs target
    3. Funding rate hedge trigger (>0.01%)
    4. Position delta imbalance
    5. Standard risk metrics
    """
    # Get base risk assessment
    metrics = await self.assess_risk()
    
    # Get USDH margin state
    try:
        usdh_state = await self.client.get_usdh_margin_state()
        
        if usdh_state:
            metrics.usdh_margin_used = usdh_state.get("margin_used", 0)
            metrics.usdh_margin_ratio = usdh_state.get("margin_ratio", 0)
            metrics.usdh_margin_available = usdh_state.get("margin_available", 0)
            
            # Check thresholds
            if metrics.usdh_margin_ratio > 0.90:
                metrics.usdh_margin_critical = True
                metrics.should_pause_trading = True
                logger.error(f"CRITICAL: USDH margin {metrics.usdh_margin_ratio:.1%} >90%")
            
            elif metrics.usdh_margin_ratio > 0.80:
                metrics.usdh_margin_warning = True
                logger.warning(f"WARNING: USDH margin {metrics.usdh_margin_ratio:.1%} >80%")
    
    except Exception as e:
        logger.error(f"Failed to get USDH margin state: {e}")
    
    # US500 volatility check
    try:
        # Get realized vol from recent price history
        account = await self.client.get_account_state()
        if account and hasattr(self, 'state') and self.state.price_history:
            prices = self.state.price_history.get_array()
            if len(prices) >= 60:
                returns = np.diff(np.log(prices[-60:]))
                realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)
                
                metrics.us500_vol = realized_vol
                metrics.us500_vol_ratio = realized_vol / metrics.us500_target_vol
                
                # Adjust risk level if vol elevated
                if metrics.us500_vol_ratio > 1.5:  # 50% above target
                    if metrics.risk_level == RiskLevel.LOW:
                        metrics.risk_level = RiskLevel.MEDIUM
                        logger.warning(f"US500 vol elevated: {realized_vol:.2%} (target {metrics.us500_target_vol:.2%})")
    
    except Exception as e:
        logger.error(f"Failed to assess US500 volatility: {e}")
    
    # Funding hedge check (US500: 0.01% threshold vs 0.04% crypto)
    funding_rate = metrics.current_funding_rate
    if abs(funding_rate) > metrics.funding_hedge_threshold:
        metrics.should_hedge_funding = True
        metrics.funding_hedge_direction = "short" if funding_rate > 0 else "long"
        
        # Estimate daily cost
        # Funding paid 3x per day (every 8h)
        metrics.funding_cost_daily = abs(funding_rate) * 3
        
        logger.warning(
            f"Funding hedge triggered: {funding_rate:.4%} (threshold {metrics.funding_hedge_threshold:.4%})"
        )
    
    return metrics


def check_usdh_margin_safety(self, metrics: RiskMetrics) -> bool:
    """
    Check if USDH margin is safe for continued trading.
    
    Returns:
        False if margin ratio >90% (pause trading)
    """
    if metrics.usdh_margin_critical:
        logger.error("USDH margin CRITICAL - trading should be paused")
        return False
    
    if metrics.usdh_margin_warning:
        logger.warning("USDH margin elevated - reduce position size")
    
    return True


def calculate_max_position_size_usdh(
    self, 
    price: float, 
    metrics: RiskMetrics
) -> float:
    """
    Calculate max position size considering USDH margin constraints.
    
    Enforces:
    - Max 90% USDH margin utilization
    - Max 50x leverage (US500 limit 25x, use 20x conservative)
    - Standard position limits
    
    Args:
        price: Current mid price
        metrics: Current risk metrics
    
    Returns:
        Max position size in contracts
    """
    # Get available USDH margin
    available_margin = metrics.usdh_margin_available
    
    # Conservative: Only use 70% of available margin
    usable_margin = available_margin * 0.70
    
    # US500 max leverage: 20x (conservative, limit is 25x)
    max_leverage = 20
    
    # Max notional = usable_margin * leverage
    max_notional = usable_margin * max_leverage
    
    # Convert to contracts
    max_size = max_notional / price
    
    # Apply standard position limits
    config_max = self.config.trading.max_net_exposure / price
    
    final_max = min(max_size, config_max)
    
    logger.debug(
        f"Max position size (USDH): {final_max:.2f} contracts "
        f"(margin: ${usable_margin:.2f}, leverage: {max_leverage}x)"
    )
    
    return final_max


async def auto_hedge_funding(self, metrics: RiskMetrics) -> None:
    """
    Automatically hedge funding rate if >0.01% for US500.
    
    Strategy:
    - If funding positive (>0.01%): Reduce long bias, increase short bias
    - If funding negative (<-0.01%): Reduce short bias, increase long bias
    
    This is less aggressive than closing positions, but reduces funding cost.
    """
    if not metrics.should_hedge_funding:
        return
    
    logger.info(
        f"Auto-hedging funding: {metrics.current_funding_rate:.4%} "
        f"(cost: {metrics.funding_cost_daily:.4%}/day)"
    )
    
    # Get current position
    position = await self.client.get_position()
    
    if not position or abs(position.size) < 0.1:
        logger.info("No position to hedge")
        return
    
    # Calculate hedge amount (10% of position)
    hedge_size = abs(position.size) * 0.10
    
    # Round to lot size (0.1 for US500)
    hedge_size = round(hedge_size / 0.1) * 0.1
    
    if hedge_size < 0.1:
        logger.debug("Hedge size too small (<0.1)")
        return
    
    # Place hedge order
    try:
        if metrics.funding_hedge_direction == "short":
            # Positive funding - reduce long / increase short
            if position.size > 0:
                # Reduce long position
                logger.info(f"Hedging positive funding: selling {hedge_size} contracts")
                await self.client.place_market_order(
                    "US500", 
                    OrderSide.SELL, 
                    hedge_size
                )
        
        else:  # "long"
            # Negative funding - reduce short / increase long
            if position.size < 0:
                # Reduce short position
                logger.info(f"Hedging negative funding: buying {hedge_size} contracts")
                await self.client.place_market_order(
                    "US500", 
                    OrderSide.BUY, 
                    hedge_size
                )
    
    except Exception as e:
        logger.error(f"Failed to place funding hedge: {e}")


# =============================================================================
# USAGE IN STRATEGY
# =============================================================================

# In US500ProfessionalMM.run_iteration():

# Get enhanced risk metrics
risk_metrics = await self.risk_manager.assess_risk_us500_usdh()

# Check USDH margin safety
if not self.risk_manager.check_usdh_margin_safety(risk_metrics):
    logger.error("USDH margin unsafe - pausing trading")
    await self.pause()
    return

# Auto-hedge funding if needed
if risk_metrics.should_hedge_funding:
    await self.risk_manager.auto_hedge_funding(risk_metrics)

# Check if should pause
if risk_metrics.should_pause_trading:
    await self.pause()
    return

# Calculate position size with USDH constraints
max_size = self.risk_manager.calculate_max_position_size_usdh(
    mid_price, 
    risk_metrics
)
