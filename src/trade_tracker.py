"""
Trade Data Tracker - Logs all trades and verifies data integrity.

This module:
1. Saves all trades to data/trade_log.json
2. Verifies order count matches exchange after each trade
3. Tracks PnL, volume, and order history
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import requests
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent / "data" / "trade_log.json"


class TradeTracker:
    """Tracks and verifies all trading activity."""
    
    def __init__(self, wallet_address: str, symbol: str = "US500"):
        self.wallet_address = wallet_address
        self.symbol = symbol
        self.api_symbol = f"km:{symbol}" if symbol.upper() == "US500" else symbol
        self.data = self._load_data()
        self._last_order_count = 0
        
    def _load_data(self) -> Dict:
        """Load existing data or create new."""
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupt trade_log.json, creating new")
        
        return {
            "session_start": None,
            "last_updated": None,
            "initial_equity": None,
            "current_equity": None,
            "total_trades": 0,
            "total_volume": 0.0,
            "realized_pnl": 0.0,
            "trades": [],
            "order_history": [],
            "verification_errors": []
        }
    
    def _save_data(self):
        """Save data to file."""
        self.data["last_updated"] = datetime.utcnow().isoformat()
        with open(DATA_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def start_session(self, initial_equity: float):
        """Start a new trading session."""
        self.data["session_start"] = datetime.utcnow().isoformat()
        self.data["initial_equity"] = initial_equity
        self.data["current_equity"] = initial_equity
        logger.info(f"Trade tracking session started with ${initial_equity:.2f} equity")
        self._save_data()
    
    def log_order_placed(self, order_id: str, side: str, price: float, size: float):
        """Log an order placement."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "placed",
            "order_id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "notional": price * size
        }
        self.data["order_history"].append(entry)
        self._save_data()
        logger.debug(f"Logged order: {side} {size} @ ${price:.2f}")
    
    def log_order_cancelled(self, order_id: str):
        """Log an order cancellation."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "cancelled",
            "order_id": order_id
        }
        self.data["order_history"].append(entry)
        self._save_data()
    
    def log_fill(self, fill_data: Dict[str, Any]):
        """Log a trade fill."""
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "fill_time": fill_data.get("time"),
            "order_id": fill_data.get("oid"),
            "side": fill_data.get("side"),  # "B" or "A"
            "price": float(fill_data.get("px", 0)),
            "size": float(fill_data.get("sz", 0)),
            "fee": float(fill_data.get("fee", 0)),
            "closed_pnl": float(fill_data.get("closedPnl", 0))
        }
        
        # Calculate notional
        trade["notional"] = trade["price"] * trade["size"]
        
        self.data["trades"].append(trade)
        self.data["total_trades"] += 1
        self.data["total_volume"] += trade["notional"]
        self.data["realized_pnl"] += trade["closed_pnl"]
        
        self._save_data()
        
        side_str = "BUY" if trade["side"] == "B" else "SELL"
        logger.info(f"FILL LOGGED: {side_str} {trade['size']} @ ${trade['price']:.2f} | PnL: ${trade['closed_pnl']:.4f}")
        
        return trade
    
    def update_equity(self, equity: float):
        """Update current equity."""
        self.data["current_equity"] = equity
        self._save_data()
    
    def get_exchange_order_count(self) -> int:
        """Get order count from exchange using historicalOrders (openOrders doesn't work for HIP-3)."""
        try:
            # Use historicalOrders and deduplicate by OID
            resp = requests.post("https://api.hyperliquid.xyz/info", json={
                "type": "historicalOrders",
                "user": self.wallet_address
            }, timeout=10)
            orders = resp.json()
            
            # CRITICAL FIX: historicalOrders returns multiple records per order
            # Deduplicate by OID and only count orders where LATEST status is 'open'
            from collections import defaultdict
            by_oid = defaultdict(list)
            for o in orders:
                if o.get('order', {}).get('coin') == self.api_symbol:
                    oid = o.get('order', {}).get('oid')
                    if oid:
                        by_oid[oid].append(o)
            
            # Count orders where latest status is 'open'
            open_count = 0
            for oid, records in by_oid.items():
                records.sort(key=lambda x: x.get('statusTimestamp', 0), reverse=True)
                if records[0].get('status') == 'open':
                    open_count += 1
            
            return open_count
        except Exception as e:
            logger.error(f"Failed to get exchange orders: {e}")
            return -1
    
    def verify_order_count(self, expected_bids: int, expected_asks: int) -> bool:
        """
        Verify local tracking matches exchange.
        
        Returns True if counts match, False if discrepancy found.
        """
        exchange_count = self.get_exchange_order_count()
        expected_total = expected_bids + expected_asks
        
        if exchange_count < 0:
            # API error, skip verification
            return True
        
        if exchange_count != expected_total:
            error = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "order_count_mismatch",
                "expected": expected_total,
                "expected_bids": expected_bids,
                "expected_asks": expected_asks,
                "actual": exchange_count,
                "difference": exchange_count - expected_total
            }
            self.data["verification_errors"].append(error)
            self._save_data()
            
            logger.error(
                f"ORDER COUNT MISMATCH! Expected {expected_total} "
                f"(bids={expected_bids}, asks={expected_asks}), "
                f"Exchange has {exchange_count}"
            )
            return False
        
        logger.debug(f"Order count verified: {exchange_count} orders on exchange")
        self._last_order_count = exchange_count
        return True
    
    def get_summary(self) -> str:
        """Get a summary of trading activity."""
        if not self.data["session_start"]:
            return "No active session"
        
        pnl = self.data["realized_pnl"]
        if self.data["initial_equity"] and self.data["initial_equity"] > 0:
            pnl_pct = (pnl / self.data["initial_equity"]) * 100
        else:
            pnl_pct = 0
        
        return (
            f"Session: {self.data['session_start']}\n"
            f"Trades: {self.data['total_trades']}\n"
            f"Volume: ${self.data['total_volume']:.2f}\n"
            f"Realized PnL: ${pnl:.4f} ({pnl_pct:.2f}%)\n"
            f"Equity: ${self.data['current_equity']:.2f}\n"
            f"Verification Errors: {len(self.data['verification_errors'])}"
        )
    
    def get_recent_trades(self, n: int = 10) -> List[Dict]:
        """Get last N trades."""
        return self.data["trades"][-n:]
    
    def reset_session(self):
        """Reset all data for a new session."""
        self.data = {
            "session_start": None,
            "last_updated": None,
            "initial_equity": None,
            "current_equity": None,
            "total_trades": 0,
            "total_volume": 0.0,
            "realized_pnl": 0.0,
            "trades": [],
            "order_history": [],
            "verification_errors": []
        }
        self._save_data()
        logger.info("Trade tracking data reset")


# Singleton instance
_tracker: Optional[TradeTracker] = None

def get_tracker(wallet_address: str = None, symbol: str = "US500") -> TradeTracker:
    """Get or create the trade tracker singleton."""
    global _tracker
    if _tracker is None:
        if wallet_address is None:
            raise ValueError("wallet_address required for first initialization")
        _tracker = TradeTracker(wallet_address, symbol)
    return _tracker
