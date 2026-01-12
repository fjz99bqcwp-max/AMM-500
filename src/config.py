"""
Configuration Management Module for AMM-500
Optimized for US500 (S&P 500 Index) perpetual trading on Hyperliquid.

US500 Specific Optimizations:
- Tighter spreads (1-2 bps min) due to lower volatility
- Faster rebalancing (3s) for delta-neutral maintenance
- Lower funding hedge threshold (0.015%) for index perps
- Aggressive inventory skew at 1.5% imbalance
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from loguru import logger  # type: ignore


@dataclass
class TradingConfig:
    """
    Trading parameters optimized for US500.
    
    US500 Characteristics:
    - Lower volatility than crypto (5-15% annualized vs 50-100%)
    - More predictable during US market hours
    - Tighter bid-ask spreads possible
    - Max 25x leverage (KM deployer limit)
    """

    symbol: str = "US500"  # KM deployer symbol (km:US500)
    leverage: int = 20  # Conservative for index perps (max 25x)
    max_net_exposure: float = 25000.0  # Lower exposure for index
    collateral: float = 1000.0  # Starting collateral for paper trading
    min_spread_bps: float = 1.0  # TIGHTER: 1 bps min for US500's lower vol
    max_spread_bps: float = 50.0  # Max 50 bps at >15% volatility
    order_size_fraction: float = 0.02
    order_levels: int = 20  # 20 levels for aggressive market making


@dataclass
class RiskConfig:
    """
    Risk management configuration - optimized for US500 index trading.
    
    US500 has lower volatility, allowing tighter risk parameters.
    """

    max_drawdown: float = 0.05  # 5% max drawdown
    stop_loss_pct: float = 0.02  # 2% stop loss
    min_margin_ratio: float = 0.15
    medium_risk_leverage: int = 15  # Reduce to 15x on medium risk
    low_risk_leverage: int = 10  # Reduce to 10x on high risk
    high_vol_threshold: float = 50.0  # Lower threshold for index (50% vs 100%)
    funding_rate_hedge_threshold: float = 0.00015  # 0.015% hedge threshold
    
    # US500-specific: Aggressive inventory management
    inventory_skew_threshold: float = 0.015  # 1.5% imbalance triggers aggressive skew


@dataclass
class ExecutionConfig:
    """Execution settings - optimized for fast delta-neutral on US500."""

    rebalance_interval: float = 3.0  # 3s ultra-fast rebalancing
    quote_refresh_interval: float = 1.0  # 1s quote refresh
    max_orders_per_batch: int = 40
    target_actions_per_day: int = 15000
    paper_trading: bool = False


@dataclass
class NetworkConfig:
    """Network settings configuration."""

    testnet: bool = True
    mainnet_url: str = "https://api.hyperliquid.xyz"
    testnet_url: str = "https://api.hyperliquid-testnet.xyz"
    mainnet_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    testnet_ws_url: str = "wss://api.hyperliquid-testnet.xyz/ws"
    request_timeout: int = 10
    max_retries: int = 3

    @property
    def api_url(self) -> str:
        """Get the appropriate API URL based on testnet setting."""
        return self.testnet_url if self.testnet else self.mainnet_url

    @property
    def ws_url(self) -> str:
        """Get the appropriate WebSocket URL based on testnet setting."""
        return self.testnet_ws_url if self.testnet else self.mainnet_ws_url


@dataclass
class LoggingConfig:
    """Logging settings configuration."""

    log_level: str = "INFO"
    log_trades: bool = True
    log_retention_days: int = 30


@dataclass
class PerformanceConfig:
    """Performance settings configuration."""

    enable_jit: bool = True
    ws_ping_interval: int = 15
    orderbook_depth: int = 10


@dataclass
class US500Config:
    """
    US500-specific configuration.
    
    This section contains parameters specific to trading the US500 index
    perpetual on Hyperliquid via the KM deployer.
    """
    
    # KM deployer market identifier
    km_symbol: str = "km:US500"
    
    # Data management
    use_btc_proxy: bool = True  # Use BTC data as proxy if US500 history insufficient
    min_required_candles: int = 259200  # 6 months of 1m data (180 days * 24 * 60)
    auto_switch_to_real: bool = True  # Auto-switch to US500 data when available
    
    # Volatility scaling (US500 has lower vol than crypto)
    vol_multiplier: float = 0.3  # US500 vol is ~30% of BTC vol typically
    
    # Trading hour awareness (optional enhancement)
    respect_market_hours: bool = False  # Trade 24/7 by default
    us_market_open_hour: int = 14  # 9:30 AM ET in UTC (approximate)
    us_market_close_hour: int = 21  # 4:00 PM ET in UTC
    
    # Spread optimization for US500
    low_vol_spread_bps: float = 1.0  # 1 bp when vol < 5%
    med_vol_spread_bps: float = 3.0  # 3 bps when vol 5-10%
    high_vol_spread_bps: float = 10.0  # 10 bps when vol 10-15%
    extreme_vol_spread_bps: float = 50.0  # 50 bps when vol > 15%


@dataclass
class Config:
    """
    Master configuration class for AMM-500.
    Optimized for US500 index perpetual trading.
    """

    # Credentials
    private_key: str = ""
    wallet_address: str = ""
    session_private_key: Optional[str] = None
    session_public_key: Optional[str] = None

    # Sub-configurations
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    us500: US500Config = field(default_factory=US500Config)

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from environment variables.

        Args:
            env_path: Optional path to .env file. Defaults to config/.env

        Returns:
            Config: Loaded configuration object
        """
        # Determine .env path
        if env_path is None:
            project_root = Path(__file__).parent.parent
            env_path = project_root / "config" / ".env"

        # Load .env file if it exists
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"No .env file found at {env_path}, using environment variables")

        # Helper function to get env vars with type conversion
        def get_env(key: str, default: str = "") -> str:
            return os.getenv(key, default)

        def get_env_float(key: str, default: float) -> float:
            return float(os.getenv(key, str(default)))

        def get_env_int(key: str, default: int) -> int:
            return int(os.getenv(key, str(default)))

        def get_env_bool(key: str, default: bool) -> bool:
            return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

        # Build configuration with US500 defaults
        config = cls(
            private_key=get_env("PRIVATE_KEY"),
            wallet_address=get_env("WALLET_ADDRESS"),
            session_private_key=get_env("SESSION_PRIVATE_KEY") or None,
            session_public_key=get_env("SESSION_PUBLIC_KEY") or None,
            trading=TradingConfig(
                symbol=get_env("SYMBOL", "US500"),  # Default to US500
                leverage=get_env_int("LEVERAGE", 20),  # 20x default for index
                max_net_exposure=get_env_float("MAX_NET_EXPOSURE", 25000.0),
                collateral=get_env_float("COLLATERAL", 1000.0),
                min_spread_bps=get_env_float("MIN_SPREAD_BPS", 1.0),  # 1 bp min for US500
                max_spread_bps=get_env_float("MAX_SPREAD_BPS", 50.0),
                order_size_fraction=get_env_float("ORDER_SIZE_FRACTION", 0.02),
                order_levels=get_env_int("ORDER_LEVELS", 20),
            ),
            risk=RiskConfig(
                max_drawdown=get_env_float("MAX_DRAWDOWN", 0.05),
                stop_loss_pct=get_env_float("STOP_LOSS_PCT", 0.02),
                min_margin_ratio=get_env_float("MIN_MARGIN_RATIO", 0.15),
                medium_risk_leverage=get_env_int("MEDIUM_RISK_LEVERAGE", 15),
                low_risk_leverage=get_env_int("LOW_RISK_LEVERAGE", 10),
                high_vol_threshold=get_env_float("HIGH_VOL_THRESHOLD", 50.0),
                funding_rate_hedge_threshold=get_env_float(
                    "FUNDING_RATE_HEDGE_THRESHOLD", 0.00015
                ),
                inventory_skew_threshold=get_env_float("INVENTORY_SKEW_THRESHOLD", 0.015),
            ),
            execution=ExecutionConfig(
                rebalance_interval=get_env_float("REBALANCE_INTERVAL", 3.0),
                quote_refresh_interval=get_env_float("QUOTE_REFRESH_INTERVAL", 1.0),
                max_orders_per_batch=get_env_int("MAX_ORDERS_PER_BATCH", 40),
                target_actions_per_day=get_env_int("TARGET_ACTIONS_PER_DAY", 15000),
                paper_trading=get_env_bool("PAPER_TRADING", False),
            ),
            network=NetworkConfig(
                testnet=get_env_bool("TESTNET", True),
                mainnet_url=get_env("MAINNET_URL", "https://api.hyperliquid.xyz"),
                testnet_url=get_env("TESTNET_URL", "https://api.hyperliquid-testnet.xyz"),
                mainnet_ws_url=get_env("MAINNET_WS_URL", "wss://api.hyperliquid.xyz/ws"),
                testnet_ws_url=get_env("TESTNET_WS_URL", "wss://api.hyperliquid-testnet.xyz/ws"),
                request_timeout=get_env_int("REQUEST_TIMEOUT", 10),
                max_retries=get_env_int("MAX_RETRIES", 3),
            ),
            logging=LoggingConfig(
                log_level=get_env("LOG_LEVEL", "INFO"),
                log_trades=get_env_bool("LOG_TRADES", True),
                log_retention_days=get_env_int("LOG_RETENTION_DAYS", 30),
            ),
            performance=PerformanceConfig(
                enable_jit=get_env_bool("ENABLE_JIT", True),
                ws_ping_interval=get_env_int("WS_PING_INTERVAL", 15),
                orderbook_depth=get_env_int("ORDERBOOK_DEPTH", 10),
            ),
            us500=US500Config(
                km_symbol=get_env("KM_SYMBOL", "km:US500"),
                use_btc_proxy=get_env_bool("USE_BTC_PROXY", True),
                min_required_candles=get_env_int("MIN_REQUIRED_CANDLES", 259200),
                auto_switch_to_real=get_env_bool("AUTO_SWITCH_TO_REAL", True),
                vol_multiplier=get_env_float("VOL_MULTIPLIER", 0.3),
                respect_market_hours=get_env_bool("RESPECT_MARKET_HOURS", False),
            ),
        )

        return config

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Check credentials
        if not self.private_key or self.private_key == "your_private_key_here":
            errors.append("PRIVATE_KEY is not set")

        if not self.wallet_address or self.wallet_address == "your_wallet_address_here":
            errors.append("WALLET_ADDRESS is not set")

        # Check trading params - US500 max leverage is 25x
        if self.trading.leverage < 1 or self.trading.leverage > 25:
            errors.append(f"LEVERAGE must be between 1 and 25 for US500, got {self.trading.leverage}")

        if self.trading.collateral <= 0:
            errors.append(f"COLLATERAL must be positive, got {self.trading.collateral}")

        if self.trading.max_net_exposure <= 0:
            errors.append(f"MAX_NET_EXPOSURE must be positive")

        # Check risk params
        if self.risk.max_drawdown <= 0 or self.risk.max_drawdown >= 1:
            errors.append(f"MAX_DRAWDOWN must be between 0 and 1")

        if self.risk.stop_loss_pct <= 0 or self.risk.stop_loss_pct >= 1:
            errors.append(f"STOP_LOSS_PCT must be between 0 and 1")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {errors}")

        logger.info("Configuration validated successfully (US500 mode)")
        return True

    def is_paper_trading(self) -> bool:
        """Check if we're in paper trading mode."""
        return self.network.testnet

    def get_max_position_size(self) -> float:
        """Calculate maximum position size based on leverage and collateral."""
        return self.trading.collateral * self.trading.leverage

    def get_km_symbol(self) -> str:
        """Get the KM deployer symbol for API calls."""
        return self.us500.km_symbol

    def __repr__(self) -> str:
        return (
            f"Config(symbol={self.trading.symbol}, "
            f"km_symbol={self.us500.km_symbol}, "
            f"leverage={self.trading.leverage}x, "
            f"testnet={self.network.testnet}, "
            f"collateral=${self.trading.collateral})"
        )
