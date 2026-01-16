#!/usr/bin/env python3
"""
AMM-500: Professional Market Making Bot for Hyperliquid US500-USDH
===================================================================

Autonomous HFT market maker with:
- Smart orderbook-aware placement
- L2 depth analysis and microprice
- Imbalance-based skewing
- PyTorch volatility prediction
- USDH margin management
- Kill switches and auto-restart

Usage:
    python amm-500.py                    # Standard mode
    python amm-500.py --autonomous       # With kill switches and auto-restart
    python amm-500.py --paper            # Paper trading mode
    python amm-500.py --backtest --days 30
    python amm-500.py --fetch-data --days 30

Exchange: Hyperliquid (https://app.hyperliquid.xyz)
Symbol: US500-USDH (HIP-3 perpetual)
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

# Local imports
from src.utils.config import load_config, Config
from src.utils.utils import setup_logging
from src.core.exchange import HyperliquidClient
from src.core.strategy import MarketMakingStrategy
from src.core.risk import RiskManager
from src.core.metrics import MetricsServer


# =============================================================================
# Constants
# =============================================================================

VERSION = "2.0.0"
RESTART_COOLDOWN = 60  # Seconds between restarts
MAX_RESTARTS_PER_HOUR = 5


# =============================================================================
# Alerts
# =============================================================================

async def send_alert(config: Config, message: str, level: str = "warning") -> None:
    """Send alert via Slack/email."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] AMM-500 {level.upper()}: {message}"
    
    # Log locally
    if level == "error":
        logger.error(full_msg)
    else:
        logger.warning(full_msg)
    
    # Slack webhook
    if config.alerts.slack_webhook_url:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(
                    config.alerts.slack_webhook_url,
                    json={"text": full_msg}
                )
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


# =============================================================================
# Bot Class
# =============================================================================

class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: Config, autonomous: bool = False):
        self.config = config
        self.autonomous = autonomous
        
        # Components
        self.exchange: Optional[HyperliquidClient] = None
        self.strategy: Optional[MarketMakingStrategy] = None
        self.risk: Optional[RiskManager] = None
        self.metrics: Optional[MetricsServer] = None
        
        # State
        self._running = False
        self._restart_count = 0
        self._restart_times: list = []
        self._start_time = time.time()
    
    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("=" * 60)
        logger.info(f"AMM-500 v{VERSION} - US500-USDH Market Making Bot")
        logger.info("=" * 60)
        
        # Mode info
        mode = "PAPER" if self.config.execution.paper_trading else "LIVE"
        logger.info(f"Mode: {mode}")
        logger.info(f"Symbol: {self.config.trading.symbol}")
        logger.info(f"Leverage: {self.config.trading.leverage}x")
        logger.info(f"Collateral: ${self.config.trading.collateral:,.2f}")
        
        if self.autonomous:
            logger.info("🤖 AUTONOMOUS MODE - Kill switches enabled")
        
        logger.info("=" * 60)
        
        try:
            # Initialize components
            await self._initialize()
            
            self._running = True
            
            # Main loop
            while self._running:
                try:
                    # Run strategy
                    await self.strategy.start()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Strategy error: {e}")
                    
                    if self.autonomous:
                        await self._handle_error(e)
                    else:
                        raise
                
                await asyncio.sleep(1)
        
        finally:
            await self.stop()
    
    async def _initialize(self) -> None:
        """Initialize bot components."""
        # Exchange client
        logger.info("Initializing exchange client...")
        self.exchange = HyperliquidClient(self.config)
        await self.exchange.connect()
        
        # Risk manager
        logger.info("Initializing risk manager...")
        self.risk = RiskManager(self.config, self.exchange)
        await self.risk.initialize()
        
        # Strategy
        logger.info("Initializing strategy...")
        self.strategy = MarketMakingStrategy(self.config, self.exchange, self.risk)
        
        # Metrics server
        logger.info("Starting metrics server...")
        self.metrics = MetricsServer(port=9090)
        self.metrics.start()
        
        logger.info("✅ Bot initialized successfully")
    
    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping bot...")
        self._running = False
        
        if self.strategy:
            await self.strategy.stop()
        
        if self.exchange:
            await self.exchange.disconnect()
        
        if self.metrics:
            self.metrics.stop()
        
        logger.info("Bot stopped")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle error in autonomous mode."""
        error_msg = str(error)
        await send_alert(self.config, f"Error: {error_msg}", "error")
        
        # Check restart limit
        now = time.time()
        self._restart_times = [t for t in self._restart_times if now - t < 3600]
        
        if len(self._restart_times) >= MAX_RESTARTS_PER_HOUR:
            await send_alert(
                self.config, 
                f"Max restarts ({MAX_RESTARTS_PER_HOUR}/hour) exceeded - stopping",
                "error"
            )
            self._running = False
            return
        
        # Wait and restart
        logger.info(f"Restarting in {RESTART_COOLDOWN}s...")
        await asyncio.sleep(RESTART_COOLDOWN)
        
        self._restart_times.append(now)
        self._restart_count += 1
        
        # Reinitialize
        try:
            await self._initialize()
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            await send_alert(self.config, f"Restart failed: {e}", "error")
    
    async def status_loop(self) -> None:
        """Print status updates periodically."""
        while self._running:
            try:
                if self.strategy and self.risk:
                    stats = self.strategy.stats
                    risk = self.risk.get_stats()
                    
                    logger.info(
                        f"Position: {stats['position']:.4f} | "
                        f"Equity: ${risk['equity']:.2f} | "
                        f"DD: {risk['drawdown_pct']:.2%} | "
                        f"Taker: {risk['taker_ratio']:.1%} | "
                        f"Fills: {risk['total_fills']}"
                    )
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.update_position(stats['position'])
                        self.metrics.update_equity(risk['equity'])
                        self.metrics.update_drawdown(risk['drawdown_pct'])
                        self.metrics.update_taker_ratio(risk['taker_ratio'])
                    
                    # Autonomous checks
                    if self.autonomous:
                        await self._autonomous_checks(risk)
            
            except Exception as e:
                logger.error(f"Status loop error: {e}")
            
            await asyncio.sleep(10)
    
    async def _autonomous_checks(self, risk: dict) -> None:
        """Perform autonomous mode checks."""
        # Drawdown >2%
        if risk['drawdown_pct'] > 0.02:
            await send_alert(
                self.config,
                f"Drawdown {risk['drawdown_pct']:.2%} > 2% threshold",
                "error"
            )
        
        # Taker ratio >30%
        if risk['taker_ratio'] > 0.30:
            await send_alert(
                self.config,
                f"Taker ratio {risk['taker_ratio']:.1%} > 30% threshold",
                "warning"
            )
        
        # Margin <10%
        if risk['margin_ratio'] > 0.90:
            await send_alert(
                self.config,
                f"Low margin: {(1-risk['margin_ratio']):.1%} available",
                "warning"
            )
        
        # 3 losing days
        if risk.get('losing_days', 0) >= 3:
            await send_alert(
                self.config,
                f"3 consecutive losing days - stopping",
                "error"
            )
            self._running = False


# =============================================================================
# CLI Commands
# =============================================================================

async def run_bot(config: Config, autonomous: bool = False) -> None:
    """Run the trading bot."""
    bot = TradingBot(config, autonomous)
    
    # Signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))
    
    # Run bot with status loop
    await asyncio.gather(
        bot.start(),
        bot.status_loop()
    )


async def run_backtest(config: Config, days: int) -> None:
    """Run backtest."""
    from src.core.backtest import run_backtest as backtest, BacktestConfig
    from src.utils.data_fetcher import DataFetcher
    
    logger.info(f"Running backtest for {days} days...")
    
    # Fetch or load data
    fetcher = DataFetcher(config)
    data = fetcher.load_cached()
    
    if data is None or len(data) < days * 24 * 60:
        logger.info("Fetching fresh data...")
        data = await fetcher.fetch(days)
    
    # Run backtest
    bt_config = BacktestConfig(
        initial_capital=config.trading.collateral,
        leverage=config.trading.leverage
    )
    
    result = await backtest(data, bt_config)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Trades/Day:       {result.trades_per_day:.1f}")
    print(f"Win Rate:         {result.win_rate:.1%}")
    print(f"Maker Ratio:      {result.maker_ratio:.1%}")
    print(f"")
    print(f"Total PnL:        ${result.total_pnl:,.2f}")
    print(f"Total Return:     {result.total_return:.2%}")
    print(f"Annual Return:    {result.annual_return:.2%}")
    print(f"")
    print(f"Max Drawdown:     {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"")
    print(f"Final Equity:     ${result.final_equity:,.2f}")
    print(f"Peak Equity:      ${result.peak_equity:,.2f}")
    print("=" * 60)
    
    # Verify targets
    print("\n📊 TARGET VERIFICATION:")
    targets = [
        ("Sharpe Ratio > 2.5", result.sharpe_ratio > 2.5, result.sharpe_ratio),
        ("ROI > 5% (7-day)", result.total_return > 0.05, result.total_return),
        ("Max DD < 0.5%", result.max_drawdown < 0.005, result.max_drawdown),
        ("Trades/Day > 2000", result.trades_per_day > 2000, result.trades_per_day),
        ("Maker Ratio > 90%", result.maker_ratio > 0.90, result.maker_ratio),
    ]
    
    for name, passed, value in targets:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {value:.4f}")


async def fetch_data(config: Config, days: int) -> None:
    """Fetch historical data."""
    from src.utils.data_fetcher import DataFetcher
    
    logger.info(f"Fetching {days} days of data...")
    
    fetcher = DataFetcher(config)
    data = await fetcher.fetch(days)
    
    logger.info(f"Fetched {len(data)} rows")
    print(f"\nData saved to data/combined_{days}d.csv")
    print(f"Date range: {data.index.min()} to {data.index.max()}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AMM-500: US500-USDH Market Making Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python amm-500.py                    # Run in standard mode
  python amm-500.py --autonomous       # Run with kill switches
  python amm-500.py --paper            # Paper trading mode
  python amm-500.py --backtest --days 30
  python amm-500.py --fetch-data --days 30
        """
    )
    
    # Mode flags
    parser.add_argument("--autonomous", "-a", action="store_true",
                       help="Enable autonomous mode with kill switches")
    parser.add_argument("--paper", "-p", action="store_true",
                       help="Enable paper trading mode")
    
    # Backtest
    parser.add_argument("--backtest", "-b", action="store_true",
                       help="Run backtest")
    parser.add_argument("--days", "-d", type=int, default=30,
                       help="Days of data for backtest/fetch")
    
    # Data
    parser.add_argument("--fetch-data", "-f", action="store_true",
                       help="Fetch historical data")
    
    # Config
    parser.add_argument("--config", "-c", type=str,
                       help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"logs/amm500_{datetime.now().strftime('%Y%m%d')}.log"
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level, log_file)
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override paper trading if specified
    if args.paper:
        config.execution.paper_trading = True
    
    # Run appropriate command
    try:
        if args.fetch_data:
            asyncio.run(fetch_data(config, args.days))
        elif args.backtest:
            asyncio.run(run_backtest(config, args.days))
        else:
            asyncio.run(run_bot(config, args.autonomous))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
