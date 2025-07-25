from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from plutus.trading_clients.trading_client import TradingClient


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
    def __init__(self, name: str, config: dict, trading_client: TradingClient):
        self.name = name
        self.config = config
        self.trading_client = trading_client
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
        """
        ...

    @abstractmethod
    def get_indicators(self) -> list[str]:
        """Return list of required technical indicators"""
        ...

    async def run(self, data: dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """
        The main method that is called on each tick.
        It analyzes the data, validates signals, and places trades.
        """
        signals = await self.analyse(data)
        for pair, signal in signals.items():
            if self.validate_signal(signal, pair):
                logger.info(
                    f"{self.name}: Valid signal | Action: {signal.action.upper()} | Pair: {pair} | Confidence: {signal.confidence:.2f}"
                )
                balance_info = await self.trading_client.get_account_balance()
                usd_balance = balance_info.get("USD", 0)

                if signal.action == "buy":
                    position_size = self.calculate_position_size(
                        signal, usd_balance, signal.price
                    )
                    order_result = await self.trading_client.place_order(
                        pair=pair,
                        type_="buy",
                        ordertype="market",
                        volume=position_size,
                        price=signal.price,
                        timestamp=timestamp,
                    )
                    if order_result:
                        self.positions[pair] = (
                            self.positions.get(pair, 0) + position_size
                        )

                elif signal.action == "sell":
                    current_position = self.positions.get(pair, 0)
                    if current_position > 0:
                        order_result = await self.trading_client.place_order(
                            pair=pair,
                            type_="sell",
                            ordertype="market",
                            volume=current_position,
                            price=signal.price,
                            timestamp=timestamp,
                        )
                        if order_result:
                            self.positions.pop(pair, None)

    def validate_signal(self, signal: Signal, pair: str) -> bool:
        """Validate a trading signal"""
        if signal.confidence < self.min_confidence:
            return False

        if signal.action not in ["buy", "sell"]:
            return False

        # Only buy if we have no position
        if signal.action == "buy" and pair in self.positions:
            logger.debug(f"Ignoring BUY signal for {pair}; position already open.")
            return False

        # Only sell if we have a position
        if signal.action == "sell" and pair not in self.positions:
            logger.debug(f"Ignoring SELL signal for {pair}; no position open.")
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
