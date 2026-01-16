"""
Prometheus Metrics Server for AMM-500
======================================
Exposes trading metrics on port 9090 for monitoring.

Metrics:
- amm500_position: Current position size
- amm500_equity: Current equity
- amm500_drawdown: Current drawdown percentage
- amm500_taker_ratio: Taker vs maker ratio
- amm500_fills_total: Total fills count
- amm500_orders_placed: Orders placed counter
- amm500_pnl_total: Total realized PnL
"""

import asyncio
from typing import Optional
import time

from loguru import logger

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - metrics disabled")


class MetricsServer:
    """Prometheus metrics server for trading bot monitoring."""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self._running = False
        
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Gauges (current values)
        self.position = Gauge("amm500_position", "Current position size")
        self.equity = Gauge("amm500_equity", "Current equity in USD")
        self.drawdown = Gauge("amm500_drawdown_pct", "Current drawdown percentage")
        self.taker_ratio = Gauge("amm500_taker_ratio", "Taker vs maker ratio")
        self.margin_ratio = Gauge("amm500_margin_ratio", "Margin utilization ratio")
        self.volatility = Gauge("amm500_volatility", "Current estimated volatility")
        self.spread_bps = Gauge("amm500_spread_bps", "Current spread in basis points")
        self.bid_count = Gauge("amm500_bid_count", "Number of active bid orders")
        self.ask_count = Gauge("amm500_ask_count", "Number of active ask orders")
        
        # Counters (cumulative)
        self.fills_total = Counter("amm500_fills_total", "Total fills", ["side", "maker"])
        self.orders_placed = Counter("amm500_orders_placed", "Orders placed", ["side"])
        self.orders_cancelled = Counter("amm500_orders_cancelled", "Orders cancelled")
        self.errors_total = Counter("amm500_errors_total", "Total errors", ["type"])
        
        # Histograms
        self.fill_latency = Histogram(
            "amm500_fill_latency_ms", 
            "Fill latency in milliseconds",
            buckets=[10, 25, 50, 100, 250, 500, 1000]
        )
        self.order_latency = Histogram(
            "amm500_order_latency_ms",
            "Order placement latency in milliseconds",
            buckets=[50, 100, 200, 500, 1000, 2000]
        )
        self.pnl_per_trade = Histogram(
            "amm500_pnl_per_trade",
            "PnL per trade in USD",
            buckets=[-10, -5, -1, 0, 1, 5, 10, 50]
        )
    
    def start(self) -> bool:
        """Start the metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics server not started")
            return False
        
        try:
            start_http_server(self.port)
            self._running = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
            return True
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {self.port} in use - metrics server disabled")
            else:
                logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the metrics server."""
        self._running = False
        logger.info("Metrics server stopped")
    
    def update_position(self, size: float) -> None:
        """Update position metric."""
        if PROMETHEUS_AVAILABLE:
            self.position.set(size)
    
    def update_equity(self, value: float) -> None:
        """Update equity metric."""
        if PROMETHEUS_AVAILABLE:
            self.equity.set(value)
    
    def update_drawdown(self, pct: float) -> None:
        """Update drawdown metric."""
        if PROMETHEUS_AVAILABLE:
            self.drawdown.set(pct * 100)
    
    def update_taker_ratio(self, ratio: float) -> None:
        """Update taker ratio metric."""
        if PROMETHEUS_AVAILABLE:
            self.taker_ratio.set(ratio)
    
    def update_margin(self, ratio: float) -> None:
        """Update margin ratio metric."""
        if PROMETHEUS_AVAILABLE:
            self.margin_ratio.set(ratio)
    
    def update_volatility(self, vol: float) -> None:
        """Update volatility metric."""
        if PROMETHEUS_AVAILABLE:
            self.volatility.set(vol * 100)
    
    def update_spread(self, bps: float) -> None:
        """Update spread metric."""
        if PROMETHEUS_AVAILABLE:
            self.spread_bps.set(bps)
    
    def update_order_counts(self, bids: int, asks: int) -> None:
        """Update bid/ask count metrics."""
        if PROMETHEUS_AVAILABLE:
            self.bid_count.set(bids)
            self.ask_count.set(asks)
    
    def record_fill(self, side: str, is_maker: bool, pnl: float = 0.0) -> None:
        """Record a fill."""
        if PROMETHEUS_AVAILABLE:
            self.fills_total.labels(side=side, maker=str(is_maker)).inc()
            self.pnl_per_trade.observe(pnl)
    
    def record_order(self, side: str) -> None:
        """Record order placement."""
        if PROMETHEUS_AVAILABLE:
            self.orders_placed.labels(side=side).inc()
    
    def record_cancel(self) -> None:
        """Record order cancellation."""
        if PROMETHEUS_AVAILABLE:
            self.orders_cancelled.inc()
    
    def record_error(self, error_type: str) -> None:
        """Record an error."""
        if PROMETHEUS_AVAILABLE:
            self.errors_total.labels(type=error_type).inc()
    
    def record_latency(self, latency_ms: float, metric_type: str = "fill") -> None:
        """Record latency measurement."""
        if PROMETHEUS_AVAILABLE:
            if metric_type == "fill":
                self.fill_latency.observe(latency_ms)
            else:
                self.order_latency.observe(latency_ms)
    
    @property
    def is_running(self) -> bool:
        return self._running
