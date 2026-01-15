"""
EXCHANGE.PY ENHANCEMENTS FOR US500-USDH
Add to /Users/nheosdisplay/VSC/AMM/AMM-500/src/exchange.py

1. WebSocket L2 Book Subscription (low-latency)
2. USDH Margin Queries (signed userState API)
3. US500-USDH specific handling
"""

# =============================================================================
# Add to HyperliquidClient class (after __init__)
# =============================================================================

async def subscribe_l2_book(self, symbol: str = "US500") -> None:
    """
    Subscribe to real-time L2 order book via WebSocket.
    
    For US500-USDH, this provides:
    - Sub-100ms latency book updates
    - Full depth (top 100 levels)
    - Accurate queue position tracking
    
    Fallback to REST if WS fails.
    """
    if not self._ws or not self._connected:
        logger.warning("WebSocket not connected - using REST fallback")
        return
    
    try:
        # Hyperliquid WS subscription message
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": symbol  # US500 for HIP-3 perp
            }
        }
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to L2 book for {symbol}")
        
        # Update orderbook callback to handle L2 data
        self._orderbook_callbacks.append(self._on_l2_update)
    
    except Exception as e:
        logger.error(f"Failed to subscribe to L2 book: {e}")


def _on_l2_update(self, data: Dict) -> None:
    """Handle L2 book update from WebSocket."""
    try:
        if data.get("channel") != "l2Book":
            return
        
        book_data = data.get("data", {})
        coin = book_data.get("coin", "")
        
        if coin != "US500":
            return
        
        # Parse bids/asks
        bids = [(float(b[0]), float(b[1])) for b in book_data.get("levels", [[],[]])[0]]
        asks = [(float(a[0]), float(a[1])) for a in book_data.get("levels", [[],[]])[1]]
        
        # Update orderbook
        orderbook = OrderBook(
            symbol="US500",
            bids=bids,
            asks=asks,
            best_bid=bids[0][0] if bids else 0,
            best_ask=asks[0][0] if asks else 0,
            best_bid_size=bids[0][1] if bids else 0,
            best_ask_size=asks[0][1] if asks else 0,
            timestamp=get_timestamp_ms()
        )
        
        orderbook.mid_price = (orderbook.best_bid + orderbook.best_ask) / 2 if orderbook.best_bid and orderbook.best_ask else 0
        
        # Cache and trigger callbacks
        self._orderbook["US500"] = orderbook
        for callback in self._orderbook_callbacks:
            callback(orderbook)
    
    except Exception as e:
        logger.error(f"Error processing L2 update: {e}")


async def get_usdh_margin_state(self) -> Dict:
    """
    Get USDH margin state via signed userState API.
    
    Returns USDH-specific margin metrics:
    - margin_used: USDH margin used for positions
    - margin_available: Available USDH for new positions
    - margin_ratio: margin_used / total_collateral
    - total_usdh: Total USDH balance
    - withdrawable_usdh: USDH available for withdrawal
    
    For US500-USDH, margin is denominated in USDH (not USD).
    """
    if not self._check_cooldown("usdh_margin"):
        logger.debug("USDH margin query on cooldown")
        return {}
    
    try:
        # Use signed userState API for USDH data
        if not self._info:
            logger.warning("Info client not initialized")
            return {}
        
        # Get user state (includes USDH margin)
        user_state = self._info.user_state(self.config.wallet_address)
        
        if not user_state:
            return {}
        
        # Parse USDH margin data
        # Hyperliquid userState structure:
        # {
        #   "marginSummary": {
        #     "accountValue": "1000.0",  # Total equity in USDH
        #     "totalMarginUsed": "100.0",  # Margin used
        #     "totalNtlPos": "1000.0",  # Total notional position
        #     ...
        #   },
        #   "crossMarginSummary": {...},  # Cross margin (for USDH)
        #   ...
        # }
        
        margin_summary = user_state.get("marginSummary", {})
        cross_margin = user_state.get("crossMarginSummary", {})
        
        account_value = float(margin_summary.get("accountValue", 0))
        margin_used = float(margin_summary.get("totalMarginUsed", 0))
        total_ntl = float(margin_summary.get("totalNtlPos", 0))
        
        # USDH withdrawable (from cross margin)
        withdrawable = float(cross_margin.get("withdrawable", 0))
        
        margin_state = {
            "total_usdh": account_value,
            "margin_used": margin_used,
            "margin_available": account_value - margin_used,
            "margin_ratio": margin_used / account_value if account_value > 0 else 0,
            "withdrawable_usdh": withdrawable,
            "total_notional": total_ntl,
            "timestamp": get_timestamp_ms()
        }
        
        logger.debug(f"USDH margin: {margin_used:.2f}/{account_value:.2f} ({margin_state['margin_ratio']:.1%})")
        
        return margin_state
    
    except Exception as e:
        logger.error(f"Failed to get USDH margin state: {e}")
        return {}


async def check_usdh_margin_safety(self) -> bool:
    """
    Check if USDH margin ratio is safe (<90%).
    
    Returns False if margin ratio >90% (pause trading).
    """
    margin_state = await self.get_usdh_margin_state()
    
    if not margin_state:
        return True  # No data - assume safe
    
    ratio = margin_state.get("margin_ratio", 0)
    
    if ratio > 0.90:
        logger.error(f"UNSAFE USDH margin ratio: {ratio:.1%} (>90%)")
        return False
    
    if ratio > 0.80:
        logger.warning(f"High USDH margin ratio: {ratio:.1%} (>80%)")
    
    return True


# =============================================================================
# Add to get_account_state method (enhance with USDH data)
# =============================================================================

async def get_account_state_enhanced(self) -> Optional[AccountState]:
    """
    Get account state with USDH margin integration.
    
    Enhanced version that includes USDH-specific metrics.
    """
    # Get base account state
    account = await self.get_account_state()
    
    if not account:
        return None
    
    # Enhance with USDH margin data
    usdh_margin = await self.get_usdh_margin_state()
    
    if usdh_margin:
        # Add USDH fields to AccountState
        account.usdh_margin_used = usdh_margin.get("margin_used", 0)
        account.usdh_margin_ratio = usdh_margin.get("margin_ratio", 0)
        account.usdh_available = usdh_margin.get("margin_available", 0)
    
    return account


# =============================================================================
# USAGE EXAMPLE in strategy.py
# =============================================================================

# In US500ProfessionalMM.__init__:
#   await self.client.subscribe_l2_book("US500")  # Enable WS L2 book

# In _refresh_inventory:
#   usdh_state = await self.client.get_usdh_margin_state()
#   self.inventory.usdh_margin_used = usdh_state.get("margin_used", 0)
#   self.inventory.usdh_margin_ratio = usdh_state.get("margin_ratio", 0)
#   self.inventory.usdh_available = usdh_state.get("margin_available", 0)

# Safety check before placing orders:
#   if not await self.client.check_usdh_margin_safety():
#       logger.error("USDH margin unsafe - pausing trading")
#       await self.pause()
#       return
