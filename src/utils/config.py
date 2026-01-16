"""
Configuration Module for AMM-500
=================================
Loads and validates configuration from .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger


@dataclass
class CredentialsConfig:
    """API credentials."""
    private_key: str = ""
    wallet_address: str = ""
    api_wallet_address: str = ""


@dataclass
class TradingConfig:
    """Trading parameters."""
    symbol: str = "US500"
    leverage: int = 5
    collateral: float = 1000.0
    max_exposure: float = 250000.0
    min_spread_bps: float = 1.0
    max_spread_bps: float = 5.0
    order_levels: int = 15
    order_size_fraction: float = 0.02
    lot_size: float = 0.0001


@dataclass
class ExecutionConfig:
    """Execution settings."""
    rebalance_interval: float = 0.5
    quote_refresh_interval: float = 2.0
    max_orders_per_batch: int = 40
    paper_trading: bool = False


@dataclass
class RiskConfig:
    """Risk management settings."""
    max_drawdown: float = 0.02
    stop_loss_pct: float = 0.02
    taker_ratio_cap: float = 0.05
    inventory_skew_threshold: float = 0.015
    funding_hedge_threshold: float = 0.0001
    min_margin_ratio: float = 0.10


@dataclass
class NetworkConfig:
    """Network settings."""
    testnet: bool = False
    mainnet_url: str = "https://api.hyperliquid.xyz"
    mainnet_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    testnet_url: str = "https://api.hyperliquid-testnet.xyz"
    testnet_ws_url: str = "wss://api.hyperliquid-testnet.xyz/ws"


@dataclass
class DataConfig:
    """Data source settings."""
    use_xyz100_primary: bool = True
    btc_fallback_enabled: bool = True
    target_volatility: float = 0.12


@dataclass
class MLConfig:
    """ML settings."""
    enabled: bool = True
    model_path: str = "data/vol_model.pt"


@dataclass
class AlertConfig:
    """Alert settings."""
    slack_webhook_url: str = ""
    alert_email: str = ""


@dataclass
class Config:
    """Complete configuration."""
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable."""
    return os.getenv(key, default)


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    val = os.getenv(key)
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get int environment variable."""
    val = os.getenv(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get bool environment variable."""
    val = os.getenv(key)
    if val:
        return val.lower() in ("true", "1", "yes", "on")
    return default


def load_config(env_path: Optional[str] = None) -> Config:
    """
    Load configuration from .env file.
    
    Args:
        env_path: Path to .env file. If None, searches in config/ and project root.
    
    Returns:
        Loaded Config object
    """
    # Find .env file
    if env_path:
        load_dotenv(env_path)
    else:
        # Try config/.env first, then root
        config_env = Path("config/.env")
        root_env = Path(".env")
        
        if config_env.exists():
            load_dotenv(config_env)
            logger.info(f"Loaded config from {config_env}")
        elif root_env.exists():
            load_dotenv(root_env)
            logger.info(f"Loaded config from {root_env}")
        else:
            logger.warning("No .env file found - using defaults")
    
    config = Config(
        credentials=CredentialsConfig(
            private_key=_get_env("PRIVATE_KEY"),
            wallet_address=_get_env("WALLET_ADDRESS"),
            api_wallet_address=_get_env("API_WALLET_ADDRESS"),
        ),
        trading=TradingConfig(
            symbol=_get_env("SYMBOL", "US500"),
            leverage=_get_env_int("LEVERAGE", 5),
            collateral=_get_env_float("COLLATERAL", 1000.0),
            max_exposure=_get_env_float("MAX_EXPOSURE", 250000.0),
            min_spread_bps=_get_env_float("MIN_SPREAD_BPS", 1.0),
            max_spread_bps=_get_env_float("MAX_SPREAD_BPS", 5.0),
            order_levels=_get_env_int("ORDER_LEVELS", 15),
            order_size_fraction=_get_env_float("ORDER_SIZE_FRACTION", 0.02),
            lot_size=_get_env_float("LOT_SIZE", 0.0001),
        ),
        execution=ExecutionConfig(
            rebalance_interval=_get_env_float("REBALANCE_INTERVAL", 0.5),
            quote_refresh_interval=_get_env_float("QUOTE_REFRESH_INTERVAL", 2.0),
            max_orders_per_batch=_get_env_int("MAX_ORDERS_PER_BATCH", 40),
            paper_trading=_get_env_bool("PAPER_TRADING", False),
        ),
        risk=RiskConfig(
            max_drawdown=_get_env_float("MAX_DRAWDOWN", 0.02),
            stop_loss_pct=_get_env_float("STOP_LOSS_PCT", 0.02),
            taker_ratio_cap=_get_env_float("TAKER_RATIO_CAP", 0.05),
            inventory_skew_threshold=_get_env_float("INVENTORY_SKEW_THRESHOLD", 0.015),
            funding_hedge_threshold=_get_env_float("FUNDING_HEDGE_THRESHOLD", 0.0001),
            min_margin_ratio=_get_env_float("MIN_MARGIN_RATIO", 0.10),
        ),
        network=NetworkConfig(
            testnet=_get_env_bool("TESTNET", False),
            mainnet_url=_get_env("MAINNET_URL", "https://api.hyperliquid.xyz"),
            mainnet_ws_url=_get_env("MAINNET_WS_URL", "wss://api.hyperliquid.xyz/ws"),
        ),
        data=DataConfig(
            use_xyz100_primary=_get_env_bool("USE_XYZ100_PRIMARY", True),
            btc_fallback_enabled=_get_env_bool("BTC_FALLBACK_ENABLED", True),
            target_volatility=_get_env_float("TARGET_VOLATILITY", 0.12),
        ),
        ml=MLConfig(
            enabled=_get_env_bool("ML_VOLATILITY_PREDICT", True),
            model_path=_get_env("ML_MODEL_PATH", "data/vol_model.pt"),
        ),
        alerts=AlertConfig(
            slack_webhook_url=_get_env("SLACK_WEBHOOK_URL"),
            alert_email=_get_env("ALERT_EMAIL"),
        ),
    )
    
    # Validate critical settings
    _validate_config(config)
    
    return config


def _validate_config(config: Config) -> None:
    """Validate configuration."""
    errors = []
    
    # Check credentials
    if not config.credentials.private_key and not config.execution.paper_trading:
        errors.append("PRIVATE_KEY required for live trading")
    
    if not config.credentials.wallet_address and not config.execution.paper_trading:
        errors.append("WALLET_ADDRESS required for live trading")
    
    # Check trading params
    if config.trading.leverage < 1 or config.trading.leverage > 25:
        errors.append(f"LEVERAGE must be 1-25, got {config.trading.leverage}")
    
    if config.trading.min_spread_bps >= config.trading.max_spread_bps:
        errors.append("MIN_SPREAD_BPS must be < MAX_SPREAD_BPS")
    
    # Check risk params
    if config.risk.max_drawdown <= 0 or config.risk.max_drawdown > 1:
        errors.append("MAX_DRAWDOWN must be 0-1")
    
    if config.risk.taker_ratio_cap < 0 or config.risk.taker_ratio_cap > 1:
        errors.append("TAKER_RATIO_CAP must be 0-1")
    
    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        raise ValueError(f"Configuration validation failed: {errors}")
    
    logger.info("Configuration validated successfully")
