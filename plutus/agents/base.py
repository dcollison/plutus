from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class Signal:
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""


@dataclass
class Trade:
    timestamp: datetime
    pair: str
    action: str
    quantity: float
    price: float
    order_id: Optional[str] = None
    status: str = "pending"


class BaseAgent(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.pairs = config.get("pairs", [])
        self.enabled = config.get("enabled", True)
        self.min_confidence = config.get("min_confidence", 0.7)
        self.max_position_size = config.get("max_position_size", 0.1)
        self.trades = []
        self.positions = {}

    @abstractmethod
    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """
        Analyze market data and return trading signals

        Args:
            data: Dictionary with pair names as keys and OHLCV DataFrames as values

        Returns:
            Dictionary with pair names as keys and Signal objects as values
        """
        pass

    @abstractmethod
    def get_indicators(self) -> list[str]:
        """Return list of required technical indicators"""
        pass

    def validate_signal(self, signal: Signal, pair: str) -> bool:
        """Validate a trading signal"""
        if signal.confidence < self.min_confidence:
            return False

        if signal.action not in ["buy", "sell", "hold"]:
            return False

        # Check if we already have a position
        if pair in self.positions and signal.action != "sell":
            return False

        return True

    def calculate_position_size(
        self, signal: Signal, balance: float, price: float
    ) -> float:
        """Calculate position size based on risk management"""
        max_risk_amount = balance * self.max_position_size
        position_size = max_risk_amount / price
        return position_size

    def log_trade(self, trade: Trade):
        """Log a trade"""
        self.trades.append(trade)
        logger.info(
            f"{self.name}: {trade.action.upper()} {trade.quantity} {trade.pair} at {trade.price}"
        )
