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
class Position:
    pair: str
    volume: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class BaseAgent(ABC):
    def __init__(self, name: str, config: dict, trading_client: TradingClient):
        self.name = name
        self.config = config
        self.trading_client = trading_client
        self.pairs = config.get("pairs", [])
        self.enabled = config.get("enabled", True)
        self.min_confidence = config.get("min_confidence", 0.7)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)  # 2% stop loss
        self.take_profit_pct = config.get("take_profit_pct", 0.04)  # 4% take profit
        self.positions: dict[str, Position] = {}

    @abstractmethod
    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]: ...

    @abstractmethod
    def get_indicators(self) -> list[str]: ...

    async def run(self, data: dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        # 1. Check if any open positions should be closed (SL/TP)
        await self.check_positions_for_exit(data, timestamp)

        # 2. Analyse market for new entry signals
        signals = await self.analyse(data)
        for pair, signal in signals.items():
            if self.validate_entry_signal(signal, pair):
                logger.info(
                    f"[{self.name}] Valid Entry Signal | {signal.action.upper()} {pair} | Confidence: {signal.confidence:.2f} | Reason: {signal.reasoning}"
                )
                await self.execute_trade(signal, pair, timestamp)

    async def check_positions_for_exit(
        self, data: dict[str, pd.DataFrame], timestamp: pd.Timestamp
    ):
        """Checks open positions to see if stop-loss or take-profit levels have been hit."""
        for pair, position in list(self.positions.items()):
            if pair not in data or data[pair].empty:
                continue

            current_price = data[pair]["close"].iloc[-1]
            exit_reason = None

            if position.stop_loss and current_price <= position.stop_loss:
                exit_reason = f"Stop-Loss triggered at ${current_price:,.2f}"
            elif position.take_profit and current_price >= position.take_profit:
                exit_reason = f"Take-Profit triggered at ${current_price:,.2f}"

            if exit_reason:
                logger.info(
                    f"[{self.name}] Exit Signal | SELL {pair} | Reason: {exit_reason}"
                )
                exit_signal = Signal(
                    action="sell",
                    confidence=1.0,
                    price=current_price,
                    reasoning=exit_reason,
                )
                await self.execute_trade(exit_signal, pair, timestamp)

    def validate_entry_signal(self, signal: Signal, pair: str) -> bool:
        """Validate a new signal for entering a position."""
        if signal.action != "buy":
            return False
        if signal.confidence < self.min_confidence:
            return False
        # Prevent opening a new position if one already exists for the pair
        if pair in self.positions:
            logger.debug(
                f"[{self.name}] Ignoring BUY signal for {pair}; position already open."
            )
            return False
        return True

    async def execute_trade(self, signal: Signal, pair: str, timestamp: pd.Timestamp):
        """Executes a trade based on the signal."""
        if signal.action == "buy":
            balance_info = await self.trading_client.get_account_balance()
            usd_balance = balance_info.get("USD", 0)
            position_size = self.calculate_position_size(usd_balance, signal.price)

            # Set SL and TP prices for the new position
            stop_loss_price = signal.price * (1 - self.stop_loss_pct)
            take_profit_price = signal.price * (1 + self.take_profit_pct)

            order_result = await self.trading_client.place_order(
                pair=pair,
                type_="buy",
                ordertype="market",
                volume=position_size,
                price=signal.price,
                timestamp=timestamp,
                agent_name=self.name,
            )
            if order_result:
                self.positions[pair] = Position(
                    pair=pair,
                    volume=position_size,
                    entry_price=signal.price,
                    entry_timestamp=timestamp,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                )
                logger.success(
                    f"[{self.name}] Opened position: BUY {position_size:.4f} {pair} at ${signal.price:,.2f}"
                )

        elif signal.action == "sell":
            if pair in self.positions:
                position_to_close = self.positions[pair]
                order_result = await self.trading_client.place_order(
                    pair=pair,
                    type_="sell",
                    ordertype="market",
                    volume=position_to_close.volume,
                    price=signal.price,
                    timestamp=timestamp,
                    agent_name=self.name,
                )
                if order_result:
                    pnl = (
                        signal.price - position_to_close.entry_price
                    ) * position_to_close.volume
                    logger.success(
                        f"[{self.name}] Closed position: SELL {position_to_close.volume:.4f} {pair} at ${signal.price:,.2f} | PnL: ${pnl:,.2f}"
                    )
                    del self.positions[pair]

    def calculate_position_size(self, balance: float, price: float) -> float:
        """Calculate position size based on risk management."""
        # This is a simplified calculation. A more robust one would use account equity.
        if price <= 0:
            return 0.0
        # Risk 1% of the total balance per trade
        risk_amount = balance * 0.01
        position_size = risk_amount / (price * self.stop_loss_pct)
        return position_size
