#!/usr/bin/env python3
"""
Hyperliquid HFT Bot - AMM-500 Entry Point
Delta-Neutral Market Making for BTC Perpetuals

This bot implements high-frequency market making on BTC perpetuals on Hyperliquid.
Originally designed for US500 (S&P 500), adapted to BTC as the most liquid market.

WARNING: This is a high-frequency trading bot using leverage.
It carries SIGNIFICANT FINANCIAL RISK. You can lose your entire investment.

- Always test on testnet first (TESTNET=True in config)
- Start with small amounts
- Monitor the bot actively
- Have stop-losses in place
- Understand the strategy before running

BTC Market Notes:
- Trading hours: 24/7 on Hyperliquid
- Highest liquidity perpetual (~$91k price, 0.11 bps spread)
- Strategy optimized for 5x leverage (conservative risk management)
- Achieves 59% annual ROI with <0.5% max drawdown in backtests
- Ideal for HFT market making with tight spreads and deep orderbook

Usage:
    python amm-500.py                  # Run the bot (LIVE - use with caution!)
    python amm-500.py --backtest       # Run backtests
    python amm-500.py --paper          # Paper trading mode (RECOMMENDED - 7 days)
    python amm-500.py --status         # Check connection status
    python amm-500.py --fetch-data     # Fetch BTC historical data

For production, consider:
- Deploying on a VPS (Dwellir, Chainstack) for <100ms latency
- Using dedicated Hyperliquid RPC nodes
- Setting up monitoring and alerts
"""

import argparse
import asyncio
import multiprocessing
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from src.config import Config
from src.exchange import HyperliquidClient
from src.risk import RiskManager
from src.strategy import US500ProfessionalMM, StrategyState
from src.backtest import run_backtest, BacktestConfig
from src.metrics import get_metrics_exporter, MetricsExporter
from src.data_fetcher import US500DataManager


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(config: Config) -> None:
    """Configure logging."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=config.logging.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler - main log
    logger.add(
        log_dir / "bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention=f"{config.logging.log_retention_days} days",
        compression="gz",
    )

    # Trade log (if enabled)
    if config.logging.log_trades:
        logger.add(
            log_dir / "trades_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "trade" in record["extra"],
            rotation="00:00",
            retention=f"{config.logging.log_retention_days} days",
        )


# =============================================================================
# Bot Runner
# =============================================================================


class HFTBot:
    """
    Main bot class that orchestrates all components.
    Optimized for US500 index perpetual trading.
    """

    def __init__(self, config: Config, metrics_port: int = 9090):
        """Initialize the bot."""
        self.config = config
        self.client: Optional[HyperliquidClient] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategy: Optional[US500ProfessionalMM] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        self.metrics_exporter = get_metrics_exporter(port=metrics_port)

    async def start(self) -> None:
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("AMM-500 - US500 Index Market Making Bot Starting")
        logger.info("=" * 60)

        # Determine mode display
        if self.config.execution.paper_trading:
            mode = "MAINNET (Paper Trading - Simulated Orders)"
        elif self.config.network.testnet:
            mode = "TESTNET"
        else:
            mode = "MAINNET (LIVE)"

        logger.info(f"Mode: {mode}")
        logger.info(f"Symbol: {self.config.trading.symbol} (km:US500)")
        logger.info(f"Leverage: {self.config.trading.leverage}x")
        logger.info(f"Collateral: ${self.config.trading.collateral:,.2f}")
        logger.info(f"Max Exposure: ${self.config.trading.max_net_exposure:,.2f}")
        logger.info(f"Min Spread: {self.config.trading.min_spread_bps} bps (optimized for US500)")
        logger.info(f"Max Spread: {self.config.trading.max_spread_bps} bps")

        if not self.config.network.testnet and not self.config.execution.paper_trading:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE - REAL FUNDS AT RISK")
            logger.warning("US500 is a permissionless market - ELEVATED RISK")
            logger.warning("=" * 60)

            # Safety pause for live mode
            logger.info("Starting in 5 seconds... Press Ctrl+C to abort")
            await asyncio.sleep(5)

        try:
            # Start metrics server
            await self.metrics_exporter.start()
            self.metrics_exporter.update(
                bot_state="starting", paper_trading=self.config.execution.paper_trading
            )

            # Initialize components
            logger.info("Initializing exchange client...")
            self.client = HyperliquidClient(self.config)
            await self.client.connect()

            logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config, self.client)

            logger.info("Initializing strategy...")
            self.strategy = US500ProfessionalMM(self.config, self.client, self.risk_manager)
            await self.strategy.start()

            self.running = True
            self.metrics_exporter.update(bot_state="running")
            logger.info("Bot started successfully!")

            # Run main loop
            await self._main_loop()

        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            raise
        finally:
            await self.stop()

    async def _main_loop(self) -> None:
        """Main trading loop."""
        iteration = 0
        last_status_log = 0
        status_interval = 60  # Log status every 60 seconds

        while self.running and not self._shutdown_event.is_set():
            try:
                start_time = asyncio.get_event_loop().time()

                # Run strategy iteration
                if self.strategy is not None:
                    await self.strategy.run_iteration()

                # Periodic status logging
                if start_time - last_status_log >= status_interval:
                    await self._log_status()
                    last_status_log = start_time

                iteration += 1

                # Calculate sleep time to maintain target frequency
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, self.config.execution.quote_refresh_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop iteration {iteration}: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    async def _log_status(self) -> None:
        """Log current bot status and update Prometheus metrics."""
        if not self.strategy:
            return

        status = self.strategy.get_status()
        metrics = self.strategy.get_metrics()
        risk_stats = self.risk_manager.get_trade_stats() if self.risk_manager else {}
        latency = (
            self.client.get_latency_stats()
            if self.client
            else {"order": {"avg": 0, "p95": 0}, "websocket": {"avg": 0}}
        )

        # Get paper trading stats if in paper mode
        paper_stats = ""
        paper_pnl = 0.0
        paper_fills = 0
        if self.config.execution.paper_trading and self.client is not None:
            ps = self.client.get_paper_trading_stats()
            paper_pnl = ps["realized_pnl"]
            paper_fills = ps["fills"]
            paper_stats = f" | Fills: {paper_fills} | Paper PnL: ${paper_pnl:.2f}"

        # Update Prometheus metrics
        try:
            orderbook = None
            mid_price: float = 0.0
            if self.strategy is not None:
                orderbook = self.strategy.last_orderbook
                mid_price = self.strategy.last_mid_price if self.strategy.last_mid_price else 0.0
            best_bid: float = orderbook.best_bid if orderbook and orderbook.best_bid else 0.0
            best_ask: float = orderbook.best_ask if orderbook and orderbook.best_ask else 0.0
            spread_bps = ((best_ask - best_bid) / mid_price * 10000) if mid_price > 0 else 0.0

            self.metrics_exporter.update(
                # Position metrics
                position_size=status["inventory"]["position_size"],
                delta=status["inventory"]["delta"],
                # Order metrics
                active_bids=status["quotes"]["active_bids"],
                active_asks=status["quotes"]["active_asks"],
                quotes_sent=metrics.quotes_sent,
                quotes_filled=metrics.quotes_filled,
                quotes_cancelled=metrics.quotes_cancelled,
                # Performance metrics
                fill_rate=metrics.fill_rate,
                actions_today=metrics.actions_today,
                total_volume=metrics.total_volume,
                maker_volume=metrics.maker_volume,
                taker_volume=metrics.taker_volume,
                # PnL metrics
                gross_pnl=metrics.gross_pnl,
                fees_paid=metrics.fees_paid,
                rebates_earned=metrics.rebates_earned,
                net_pnl=metrics.net_pnl,
                paper_pnl=paper_pnl,
                paper_fills=paper_fills,
                # Risk metrics
                equity=risk_stats.get("peak_equity", 0),
                collateral=self.config.trading.collateral,
                current_drawdown=risk_stats.get("current_drawdown", 0),
                peak_equity=risk_stats.get("peak_equity", 0),
                leverage=self.config.trading.leverage,
                # Market metrics
                mid_price=mid_price,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
                funding_rate=status.get("funding_rate", 0),
                # Trade stats
                total_trades=risk_stats.get("total_trades", 0),
                winning_trades=risk_stats.get("winning_trades", 0),
                losing_trades=risk_stats.get("losing_trades", 0),
                win_rate=risk_stats.get("win_rate", 0),
                avg_win=risk_stats.get("avg_win", 0),
                avg_loss=risk_stats.get("avg_loss", 0),
                profit_factor=risk_stats.get("profit_factor", 0),
                # Latency metrics
                order_latency_avg=latency["order"]["avg"],
                order_latency_p95=latency["order"]["p95"],
                ws_latency_avg=latency["websocket"]["avg"],
                # Bot state
                bot_state=status["state"],
            )
        except Exception as e:
            logger.debug(f"Error updating metrics: {e}")

        # Get real account info if available
        account = status.get("account", {})
        pnl_pct = account.get("pnl_pct", 0)
        current_equity = account.get("current_equity", 0)
        pnl_emoji = "📈" if pnl_pct >= 0 else "📉"

        logger.info(
            f"Status: {status['state']} | "
            f"Position: {status['inventory']['position_size']:.4f} | "
            f"Delta: {status['inventory']['delta']:.3f} | "
            f"Bids: {status['quotes']['active_bids']} | "
            f"Asks: {status['quotes']['active_asks']} | "
            f"Actions: {metrics.actions_today} | "
            f"Fill Rate: {metrics.fill_rate:.2%} | "
            f"Equity: ${current_equity:.2f} | "
            f"PnL: ${metrics.net_pnl:+.2f} ({pnl_pct:+.2f}%) {pnl_emoji}"
            f"{paper_stats}"
        )

        # Log additional debug metrics for better monitoring
        if self.risk_manager:
            risk_stats = self.risk_manager.get_trade_stats()
            funding_net_cost = risk_stats.get("funding_net_cost", 0.0)
            max_imbalance_duration = risk_stats.get("max_imbalance_duration", 0.0)
            risk_multipliers = risk_stats.get("risk_multipliers", {})

            logger.debug(
                f"Funding Net Cost: ${funding_net_cost:+.2f} | "
                f"Max Imbalance Duration: {max_imbalance_duration:.1f}s | "
                f"Risk Multipliers: leverage={risk_multipliers.get('leverage', 1.0):.1f}x, "
                f"spread={risk_multipliers.get('spread', 1.0):.2f}, "
                f"position={risk_multipliers.get('position', 1.0):.2f}"
            )

        if latency["order"]["avg"] > 0:
            logger.debug(
                f"Latency - Order: {latency['order']['avg']:.1f}ms (p95: {latency['order']['p95']:.1f}ms) | "
                f"WS: {latency['websocket']['avg']:.1f}ms"
            )

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
        self.metrics_exporter.update(bot_state="stopped")

        if self.strategy:
            await self.strategy.stop()

        if self.client:
            await self.client.disconnect()

        await self.metrics_exporter.stop()
        logger.info("Bot stopped")

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


async def run_bot(config: Config) -> None:
    """Run the bot with signal handling."""
    bot = HFTBot(config)

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        bot.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


# =============================================================================
# Data Fetching
# =============================================================================


async def fetch_us500_data(days: int = 180) -> None:
    """Fetch US500 historical data or use BTC proxy if insufficient."""
    logger.info(f"Fetching US500 historical data ({days} days)...")
    
    data_manager = US500DataManager()
    
    try:
        candles_df, funding_df, is_proxy = await data_manager.get_trading_data(
            days=days,
            min_required_days=180  # 6 months minimum for reliable backtesting
        )
        
        if is_proxy:
            logger.warning("=" * 60)
            logger.warning("USING BTC DATA AS PROXY FOR US500")
            logger.warning("US500 has insufficient historical data")
            logger.warning("Bot will auto-switch to US500 data when available")
            logger.warning("=" * 60)
        else:
            logger.info(f"Fetched {len(candles_df)} US500 candles")
            
        logger.info(f"Data saved to data/ directory")
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise


# =============================================================================
# Status Check
# =============================================================================


async def check_status(config: Config) -> None:
    """Check connection status and account info."""
    logger.info("Checking connection status...")

    try:
        client = HyperliquidClient(config)
        await client.connect()

        account = await client.get_account_state()
        position = await client.get_position()
        orderbook = await client.get_orderbook()
        funding = await client.get_funding_rate()

        print("\n" + "=" * 50)
        print("CONNECTION STATUS: OK")
        print("=" * 50)
        print(f"\nAsset: {config.trading.symbol} - Bitcoin Perpetual")
        print(f"Exchange: Hyperliquid (Most Liquid Perp)")

        if account:
            print(f"\nAccount:")
            print(f"  Equity: ${account.equity:,.2f}")
            print(f"  Available: ${account.available_balance:,.2f}")
            print(f"  Margin Used: ${account.margin_used:,.2f}")
            print(f"  Unrealized PnL: ${account.unrealized_pnl:,.2f}")

        if position:
            print(f"\nPosition ({config.trading.symbol}):")
            print(f"  Size: {position.size:.4f}")
            print(f"  Entry Price: ${position.entry_price:,.2f}")
            print(f"  Mark Price: ${position.mark_price:,.2f}")
            print(f"  Unrealized PnL: ${position.unrealized_pnl:,.2f}")
            print(f"  Liquidation Price: ${position.liquidation_price:,.2f}")
        else:
            print(f"\nNo open position in {config.trading.symbol}")

        if orderbook:
            print(f"\nOrderbook ({config.trading.symbol}):")
            print(f"  Best Bid: ${orderbook.best_bid:,.2f}")
            print(f"  Best Ask: ${orderbook.best_ask:,.2f}")
            print(f"  Spread: {orderbook.spread_bps:.2f} bps")

        if funding is not None:
            print(f"\nFunding Rate: {funding:.4%}")

        print("=" * 50 + "\n")

        await client.disconnect()

    except Exception as e:
        print(f"\nCONNECTION FAILED: {e}")
        raise


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AMM-500 - US500 Index Market Making Bot (HIP-3/KM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python amm-500.py                  Run the bot (uses config/.env)
  python amm-500.py --paper          Force paper trading mode
  python amm-500.py --fetch-data     Fetch US500 historical data (or BTC proxy)
  python amm-500.py --backtest       Run backtests with historical data
  python amm-500.py --backtest --days 60   Backtest with 60 days of data
  python amm-500.py --backtest --months 12  Backtest with 12 months data
  python amm-500.py --status         Check connection and account status

US500 Specific Notes:
  - Uses km:US500 permissionless market (HIP-3)
  - Lower volatility than crypto (5-15% vs 50-100%)
  - Tighter spreads possible (1 bps min vs 5 bps)
  - Max leverage: 25x
  - Isolated margin only

For more information, see README.md
        """,
    )

    parser.add_argument(
        "--paper", action="store_true", help="Paper trading mode (mainnet data, simulated orders)"
    )

    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest instead of live trading"
    )

    parser.add_argument(
        "--days", type=int, default=30, help="Days of data for backtest (default: 30)"
    )

    parser.add_argument(
        "--months",
        type=int,
        default=1,
        choices=range(1, 13),
        help="Months of data for backtest (default: 1, max: 12)",
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real API data for backtest",
    )

    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch US500 historical data (or BTC proxy if insufficient)",
    )

    parser.add_argument(
        "--fetch-days",
        type=int,
        default=180,
        help="Days of data to fetch (default: 180 = 6 months)",
    )

    parser.add_argument("--status", action="store_true", help="Check connection status and exit")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (default: config/.env)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = Config.load(config_path)

    # Force paper mode if requested (uses mainnet data, simulates orders)
    if args.paper:
        config.execution.paper_trading = True
        config.network.testnet = False  # Use mainnet for real US500 data
        logger.info("Paper trading mode enabled (mainnet data, simulated orders)")
        logger.info(f"Using mainnet for real {config.trading.symbol} market data")

    # Setup logging
    setup_logging(config)

    # Validate config for live mode
    if not config.network.testnet and not args.backtest and not args.status and not args.fetch_data:
        try:
            config.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)

    # Handle data fetching mode
    if args.fetch_data:
        logger.info(f"Fetching US500 data for {args.fetch_days} days...")
        asyncio.run(fetch_us500_data(args.fetch_days))
        return

    # Run appropriate mode
    if args.backtest:
        # Determine data duration from months (convert to days)
        data_days = args.months * 30 if args.months > 1 else args.days
        data_type = "real API" if not args.synthetic else "synthetic"
        logger.info(
            f"Running US500 backtest with {data_days} days ({args.months} months) of {data_type} data..."
        )
        logger.info(f"Using leverage: {config.trading.leverage}x (recommended: 10x)")
        logger.info("Note: May use BTC proxy data if US500 history is insufficient")

        backtest_config = BacktestConfig(
            initial_capital=config.trading.collateral,
            leverage=config.trading.leverage,
            min_spread_bps=config.trading.min_spread_bps,
            max_spread_bps=config.trading.max_spread_bps,
            order_size_pct=config.trading.order_size_fraction,
            order_levels=config.trading.order_levels,
            rebalance_interval=int(config.execution.rebalance_interval),
            max_drawdown=config.risk.max_drawdown,
            stop_loss_pct=config.risk.stop_loss_pct,
        )

        # Use multiprocessing for Monte Carlo simulations on Mac Mini M4
        num_cores = min(multiprocessing.cpu_count(), 10)  # Cap at 10 cores for M4
        logger.info(f"Using {num_cores} CPU cores for parallel Monte Carlo simulations")

        result = run_backtest(
            config=backtest_config,
            synthetic_days=data_days,
            use_real_data=not args.synthetic,
            plot=True,
            monte_carlo=True,
            num_processes=num_cores,
        )

    elif args.status:
        # Check status
        asyncio.run(check_status(config))

    else:
        # Run bot
        asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
