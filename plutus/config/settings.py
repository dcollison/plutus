import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class NotificationConfig:
    pushover_token: Optional[str] = None
    pushover_user: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = None


@dataclass
class TradingConfig:
    pairs: list[str]
    max_position_size: float
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float


class Settings:
    def __init__(self, config_file: str = "config/trading_config.yaml"):
        self.config_file: Path = Path(config_file)
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        if self.config_file.exists():
            logger.info(f"Loading configuration from {self.config_file}")
            with self.config_file.open() as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(
                f"No configuration file found at {self.config_file}, using default"
            )
            config = self.default_config()
            self.save_config(config)

        # API Configuration
        self.kraken_api_key = os.getenv(
            "KRAKEN_API_KEY", config.get("kraken_api_key", "")
        )
        self.kraken_api_secret = os.getenv(
            "KRAKEN_PRIVATE_KEY", config.get("kraken_api_secret", "")
        )
        self.use_sandbox = config.get("use_sandbox", True)

        # Trading Configuration
        trading_config = config.get("trading", {})
        self.trading_config = TradingConfig(
            pairs=trading_config.get("pairs", ["XBTUSD", "ETHUSD"]),
            max_position_size=trading_config.get("max_position_size", 0.1),
            risk_per_trade=trading_config.get("risk_per_trade", 0.02),
            stop_loss_pct=trading_config.get("stop_loss_pct", 0.02),
            take_profit_pct=trading_config.get("take_profit_pct", 0.04),
        )

        # Application Settings
        self.update_interval = config.get("update_interval", 30)  # seconds
        self.enable_dashboard = config.get("enable_dashboard", True)
        self.enable_notifications = config.get("enable_notifications", True)

        # Notification Configuration
        notif_config = config.get("notifications", {})
        self.notification_config = NotificationConfig(
            pushover_token=notif_config.get("pushover_token"),
            pushover_user=notif_config.get("pushover_user"),
            telegram_token=notif_config.get("telegram_token"),
            telegram_chat_id=notif_config.get("telegram_chat_id"),
            discord_webhook=notif_config.get("discord_webhook"),
        )

        # Data Settings
        self.historical_data_days = config.get("historical_data_days", 30)

    def default_config(self) -> dict:
        """Return default configuration"""
        return {
            "kraken_api_key": "",
            "kraken_api_secret": "",
            "use_sandbox": True,
            "trading": {
                "pairs": ["XBTUSD", "ETHUSD"],
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
            },
            "update_interval": 30,
            "enable_dashboard": True,
            "enable_notifications": True,
            "notifications": {
                "pushover_token": None,
                "pushover_user": None,
                "telegram_token": None,
                "telegram_chat_id": None,
                "discord_webhook": None,
            },
            "historical_data_days": 30,
        }

    def save_config(self, config: dict):
        """Save configuration to YAML file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
