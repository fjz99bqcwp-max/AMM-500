"""
Tests for configuration module.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.utils.config import (
    Config,
    TradingConfig,
    RiskConfig,
    ExecutionConfig,
    NetworkConfig,
    LoggingConfig,
    PerformanceConfig,
)


class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_default_values(self):
        """Test default trading configuration."""
        config = TradingConfig()

        assert config.symbol == "US500"  # Updated for US500 trading
        assert config.leverage == 10  # Default for US500 (max 25x)
        assert config.max_net_exposure == 25000.0  # Updated for US500
        assert config.collateral == 1000.0
        assert config.order_levels == 20  # Updated for US500 market making

    def test_custom_values(self):
        """Test custom trading configuration."""
        config = TradingConfig(
            symbol="ETH",
            leverage=10,
            collateral=5000.0,
        )

        assert config.symbol == "ETH"
        assert config.leverage == 10
        assert config.collateral == 5000.0


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self):
        """Test default risk configuration."""
        config = RiskConfig()

        assert config.max_drawdown == 0.05  # Updated - tightened to 5%
        assert config.stop_loss_pct == 0.02  # Updated - tightened to 2%
        assert config.min_margin_ratio == 0.15
        assert config.medium_risk_leverage == 15  # Conservative medium risk
        assert config.low_risk_leverage == 10  # Conservative high risk


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""

    def test_default_values(self):
        """Test default network configuration."""
        config = NetworkConfig()

        assert config.testnet == True
        assert "hyperliquid" in config.mainnet_url

    def test_api_url_property(self):
        """Test API URL property based on testnet setting."""
        config = NetworkConfig(testnet=True)
        assert "testnet" in config.api_url

        config = NetworkConfig(testnet=False)
        assert "testnet" not in config.api_url

    def test_ws_url_property(self):
        """Test WebSocket URL property based on testnet setting."""
        config = NetworkConfig(testnet=True)
        assert "testnet" in config.ws_url

        config = NetworkConfig(testnet=False)
        assert "testnet" not in config.ws_url


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert config.private_key == ""
        assert config.wallet_address == ""
        assert isinstance(config.trading, TradingConfig)
        assert isinstance(config.risk, RiskConfig)
        assert isinstance(config.network, NetworkConfig)

    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["PRIVATE_KEY"] = "test_private_key"
        os.environ["WALLET_ADDRESS"] = "0x1234"
        os.environ["LEVERAGE"] = "15"
        os.environ["TESTNET"] = "true"

        try:
            # Load without .env file
            config = Config.load(env_path=Path("/nonexistent/.env"))

            assert config.private_key == "test_private_key"
            assert config.wallet_address == "0x1234"
            assert config.trading.leverage == 15
            assert config.network.testnet == True
        finally:
            # Clean up
            del os.environ["PRIVATE_KEY"]
            del os.environ["WALLET_ADDRESS"]
            del os.environ["LEVERAGE"]
            del os.environ["TESTNET"]

    def test_load_from_env_file(self):
        """Test loading configuration from .env file."""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PRIVATE_KEY=file_private_key\n")
            f.write("WALLET_ADDRESS=0x5678\n")
            f.write("LEVERAGE=20\n")
            f.write("COLLATERAL=3000\n")
            env_path = Path(f.name)

        try:
            config = Config.load(env_path=env_path)

            assert config.private_key == "file_private_key"
            assert config.wallet_address == "0x5678"
            assert config.trading.leverage == 20
            assert config.trading.collateral == 3000.0
        finally:
            env_path.unlink()
            # Clean up environment variables set by load_dotenv
            if "PRIVATE_KEY" in os.environ:
                del os.environ["PRIVATE_KEY"]
            if "WALLET_ADDRESS" in os.environ:
                del os.environ["WALLET_ADDRESS"]
            if "LEVERAGE" in os.environ:
                del os.environ["LEVERAGE"]
            if "COLLATERAL" in os.environ:
                del os.environ["COLLATERAL"]

    def test_validate_missing_credentials(self):
        """Test validation fails with missing credentials."""
        config = Config()

        with pytest.raises(ValueError) as excinfo:
            config.validate()

        assert "PRIVATE_KEY" in str(excinfo.value)

    def test_validate_invalid_leverage(self):
        """Test validation fails with invalid leverage."""
        config = Config(
            private_key="valid_key",
            wallet_address="0x1234",
            trading=TradingConfig(leverage=100),  # Too high
        )

        with pytest.raises(ValueError) as excinfo:
            config.validate()

        assert "LEVERAGE" in str(excinfo.value)

    def test_validate_success(self):
        """Test validation succeeds with valid config."""
        config = Config(
            private_key="valid_key",
            wallet_address="0x1234",
            trading=TradingConfig(leverage=25),
            risk=RiskConfig(max_drawdown=0.10, stop_loss_pct=0.05),
        )

        assert config.validate() == True

    def test_is_paper_trading(self):
        """Test paper trading detection."""
        config = Config(network=NetworkConfig(testnet=True))
        assert config.is_paper_trading() == True

        config = Config(network=NetworkConfig(testnet=False))
        assert config.is_paper_trading() == False

    def test_get_max_position_size(self):
        """Test maximum position size calculation."""
        config = Config(trading=TradingConfig(collateral=2000.0, leverage=25))

        assert config.get_max_position_size() == 50000.0

    def test_repr(self):
        """Test string representation."""
        config = Config(
            trading=TradingConfig(symbol="BTC", leverage=25),
            network=NetworkConfig(testnet=True),
        )

        repr_str = repr(config)
        assert "BTC" in repr_str
        assert "25x" in repr_str
        assert "testnet=True" in repr_str


class TestConfigIntegration:
    """Integration tests for configuration module."""

    def test_full_config_flow(self):
        """Test complete configuration loading and validation."""
        # Create a complete .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PRIVATE_KEY=0xabc123\n")
            f.write("WALLET_ADDRESS=0xdef456\n")
            f.write("SYMBOL=BTC\n")
            f.write("LEVERAGE=25\n")
            f.write("MAX_NET_EXPOSURE=50000\n")
            f.write("COLLATERAL=2000\n")
            f.write("MAX_DRAWDOWN=0.10\n")
            f.write("STOP_LOSS_PCT=0.04\n")
            f.write("TESTNET=true\n")
            f.write("LOG_LEVEL=INFO\n")
            env_path = Path(f.name)

        try:
            config = Config.load(env_path=env_path)
            config.validate()

            # Verify all values loaded correctly
            assert config.private_key == "0xabc123"
            assert config.wallet_address == "0xdef456"
            assert config.trading.symbol == "BTC"
            assert config.trading.leverage == 25
            assert config.trading.max_net_exposure == 50000.0
            assert config.trading.collateral == 2000.0
            assert config.risk.max_drawdown == 0.10
            assert config.risk.stop_loss_pct == 0.04
            assert config.network.testnet == True
            assert config.logging.log_level == "INFO"

            # Verify computed properties
            assert config.is_paper_trading() == True
            assert config.get_max_position_size() == 50000.0

        finally:
            env_path.unlink()
            # Clean up environment variables set by load_dotenv
            if "PRIVATE_KEY" in os.environ:
                del os.environ["PRIVATE_KEY"]
            if "WALLET_ADDRESS" in os.environ:
                del os.environ["WALLET_ADDRESS"]
            if "SYMBOL" in os.environ:
                del os.environ["SYMBOL"]
            if "LEVERAGE" in os.environ:
                del os.environ["LEVERAGE"]
            if "MAX_NET_EXPOSURE" in os.environ:
                del os.environ["MAX_NET_EXPOSURE"]
            if "COLLATERAL" in os.environ:
                del os.environ["COLLATERAL"]
            if "MAX_DRAWDOWN" in os.environ:
                del os.environ["MAX_DRAWDOWN"]
            if "STOP_LOSS_PCT" in os.environ:
                del os.environ["STOP_LOSS_PCT"]
            if "TESTNET" in os.environ:
                del os.environ["TESTNET"]
            if "LOG_LEVEL" in os.environ:
                del os.environ["LOG_LEVEL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
