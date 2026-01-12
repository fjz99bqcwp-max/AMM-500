"""
Prometheus Metrics Exporter for HFT Bot.

Exposes trading metrics via HTTP endpoint for Prometheus scraping.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from aiohttp import web
from loguru import logger


@dataclass
class BotMetrics:
    """Container for all bot metrics."""

    # Trading metrics
    position_size: float = 0.0
    position_value: float = 0.0
    delta: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0

    # Order metrics
    active_bids: int = 0
    active_asks: int = 0
    quotes_sent: int = 0
    quotes_filled: int = 0
    quotes_cancelled: int = 0

    # Performance metrics
    fill_rate: float = 0.0
    actions_today: int = 0
    total_volume: float = 0.0
    maker_volume: float = 0.0
    taker_volume: float = 0.0

    # PnL metrics
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    net_pnl: float = 0.0
    paper_pnl: float = 0.0
    paper_fills: int = 0

    # Risk metrics
    equity: float = 0.0
    collateral: float = 0.0
    margin_ratio: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    leverage: float = 0.0
    risk_level: str = "LOW"

    # Market metrics
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    volatility: float = 0.0
    funding_rate: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Latency metrics
    order_latency_avg: float = 0.0
    order_latency_p95: float = 0.0
    ws_latency_avg: float = 0.0

    # System metrics
    uptime_seconds: float = 0.0
    last_update: float = 0.0
    bot_state: str = "stopped"
    paper_trading: bool = False


class MetricsExporter:
    """
    Prometheus metrics exporter.

    Runs an HTTP server that exposes metrics in Prometheus format.
    """

    def __init__(self, port: int = 9090):
        """Initialize the metrics exporter."""
        self.port = port
        self.metrics = BotMetrics()
        self._start_time = time.time()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._running = False

    async def start(self) -> None:
        """Start the metrics HTTP server."""
        self._app = web.Application()
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/", self._handle_index)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        # Try to bind with reuse_address to avoid port conflicts
        try:
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.port, reuse_address=True)
            await self._site.start()
            self._running = True
            logger.info(f"Prometheus metrics server started on http://0.0.0.0:{self.port}/metrics")
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.warning(f"Port {self.port} in use, metrics server disabled")
                self._running = False
            else:
                raise

    async def stop(self) -> None:
        """Stop the metrics HTTP server."""
        if self._runner:
            await self._runner.cleanup()
        self._running = False
        logger.info("Prometheus metrics server stopped")

    def update(self, **kwargs) -> None:
        """Update metrics from bot state."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        self.metrics.last_update = time.time()
        self.metrics.uptime_seconds = time.time() - self._start_time

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle /metrics endpoint."""
        output = self._generate_prometheus_output()
        return web.Response(text=output, content_type="text/plain")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle /health endpoint."""
        is_healthy = self.metrics.bot_state in ["running", "starting"]
        status = 200 if is_healthy else 503
        return web.json_response({"status": "ok" if is_healthy else "unhealthy"}, status=status)

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Handle / endpoint with links."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>HFT Bot Metrics</title></head>
        <body>
            <h1>HFT Bot Prometheus Metrics</h1>
            <ul>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    def _generate_prometheus_output(self) -> str:
        """Generate Prometheus-format metrics output."""
        m = self.metrics
        lines = []

        # Helper to add metric
        def add_metric(
            name: str, value: float, help_text: str, metric_type: str = "gauge", labels: Dict = None
        ):
            lines.append(f"# HELP hft_{name} {help_text}")
            lines.append(f"# TYPE hft_{name} {metric_type}")
            if labels:
                label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                lines.append(f"hft_{name}{{{label_str}}} {value}")
            else:
                lines.append(f"hft_{name} {value}")

        # Position metrics
        add_metric("position_size", m.position_size, "Current position size in BTC")
        add_metric("position_value", m.position_value, "Current position value in USD")
        add_metric("delta", m.delta, "Current delta exposure")
        add_metric("entry_price", m.entry_price, "Position entry price")
        add_metric("mark_price", m.mark_price, "Current mark price")

        # Order metrics
        add_metric("active_bids", m.active_bids, "Number of active bid orders")
        add_metric("active_asks", m.active_asks, "Number of active ask orders")
        add_metric("quotes_sent_total", m.quotes_sent, "Total quotes sent", "counter")
        add_metric("quotes_filled_total", m.quotes_filled, "Total quotes filled", "counter")
        add_metric(
            "quotes_cancelled_total", m.quotes_cancelled, "Total quotes cancelled", "counter"
        )

        # Performance metrics
        add_metric("fill_rate", m.fill_rate, "Current fill rate (0-1)")
        add_metric("actions_today", m.actions_today, "Actions taken today")
        add_metric("total_volume", m.total_volume, "Total trading volume in USD")
        add_metric("maker_volume", m.maker_volume, "Maker volume in USD")
        add_metric("taker_volume", m.taker_volume, "Taker volume in USD")

        # PnL metrics
        add_metric("gross_pnl", m.gross_pnl, "Gross PnL in USD")
        add_metric("fees_paid", m.fees_paid, "Total fees paid in USD")
        add_metric("rebates_earned", m.rebates_earned, "Total rebates earned in USD")
        add_metric("net_pnl", m.net_pnl, "Net PnL in USD")
        add_metric("paper_pnl", m.paper_pnl, "Paper trading PnL in USD")
        add_metric("paper_fills", m.paper_fills, "Paper trading fill count")

        # Risk metrics
        add_metric("equity", m.equity, "Current account equity in USD")
        add_metric("collateral", m.collateral, "Collateral in USD")
        add_metric("margin_ratio", m.margin_ratio, "Current margin ratio")
        add_metric("current_drawdown", m.current_drawdown, "Current drawdown from peak (0-1)")
        add_metric("peak_equity", m.peak_equity, "Peak equity in USD")
        add_metric("leverage", m.leverage, "Current effective leverage")

        # Risk level as labeled metric
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            val = 1 if m.risk_level == level else 0
            lines.append(f'hft_risk_level{{level="{level}"}} {val}')

        # Market metrics
        add_metric("mid_price", m.mid_price, "Current mid price")
        add_metric("best_bid", m.best_bid, "Best bid price")
        add_metric("best_ask", m.best_ask, "Best ask price")
        add_metric("spread_bps", m.spread_bps, "Current spread in basis points")
        add_metric("volatility", m.volatility, "Current volatility (annualized)")
        add_metric("funding_rate", m.funding_rate, "Current funding rate")

        # Trade stats
        add_metric("total_trades", m.total_trades, "Total number of trades", "counter")
        add_metric("winning_trades", m.winning_trades, "Number of winning trades", "counter")
        add_metric("losing_trades", m.losing_trades, "Number of losing trades", "counter")
        add_metric("win_rate", m.win_rate, "Win rate (0-1)")
        add_metric("avg_win", m.avg_win, "Average win amount in USD")
        add_metric("avg_loss", m.avg_loss, "Average loss amount in USD")
        add_metric("profit_factor", m.profit_factor, "Profit factor (wins/losses)")

        # Latency metrics
        add_metric("order_latency_avg_ms", m.order_latency_avg, "Average order latency in ms")
        add_metric("order_latency_p95_ms", m.order_latency_p95, "P95 order latency in ms")
        add_metric("ws_latency_avg_ms", m.ws_latency_avg, "Average websocket latency in ms")

        # System metrics
        add_metric("uptime_seconds", m.uptime_seconds, "Bot uptime in seconds")
        add_metric("last_update_timestamp", m.last_update, "Last metrics update timestamp")

        # Bot state as labeled metric
        for state in ["running", "paused", "stopped", "starting", "error"]:
            val = 1 if m.bot_state == state else 0
            lines.append(f'hft_bot_state{{state="{state}"}} {val}')

        # Paper trading flag
        add_metric("paper_trading", 1 if m.paper_trading else 0, "Paper trading mode enabled")

        return "\n".join(lines) + "\n"


# Global metrics instance for easy access
_metrics_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter(port: int = 9090) -> MetricsExporter:
    """Get or create the global metrics exporter."""
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = MetricsExporter(port=port)
    return _metrics_exporter
