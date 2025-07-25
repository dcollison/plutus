import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent, Signal
from plutus.trading_clients.trading_client import TradingClient


class MomentumBot(BaseAgent):
    def __init__(self, name: str, config: dict, trading_client: TradingClient):
        super().__init__(name, config, trading_client)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        # Strategy parameters are now time-based (in hours)
        self.ma_fast_hours = config.get("ma_fast_hours", 10)
        self.ma_slow_hours = config.get("ma_slow_hours", 20)

    def get_indicators(self) -> list[str]:
        return ["rsi", "sma", "volume"]

    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}

        for pair, df in data.items():
            # Add a guard clause to ensure there are enough data points to infer frequency.
            if len(df.index) < 3:
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Not enough data to infer interval"
                )
                continue

            # Infer the data interval in minutes from the DataFrame's index
            freq = pd.infer_freq(df.index)
            if freq is None:
                # If frequency can't be inferred, fall back to calculating from the last two timestamps
                interval_minutes = (df.index[-1] - df.index[-2]).total_seconds() / 60
            else:
                interval_minutes = pd.to_timedelta(freq).total_seconds() / 60

            # Prevent division by zero if interval is 0 (e.g. duplicate timestamps)
            if interval_minutes == 0:
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Invalid data interval (0 minutes)"
                )
                continue

            # Convert time-based strategy (e.g., 10 hours) to period-based strategy
            ma_fast_periods = max(
                1, round((self.ma_fast_hours * 60) / interval_minutes)
            )
            ma_slow_periods = max(
                1, round((self.ma_slow_hours * 60) / interval_minutes)
            )

            if df.empty or len(df) < max(self.rsi_period, ma_slow_periods):
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Insufficient data for indicators"
                )
                continue

            # Calculate indicators with the adjusted periods
            rsi = self.calculate_rsi(df["close"], self.rsi_period)
            ma_fast = df["close"].rolling(ma_fast_periods).mean()
            ma_slow = df["close"].rolling(ma_slow_periods).mean()

            current_price = df["close"].iloc[-1]
            current_rsi = rsi.iloc[-1]

            signal = self.generate_signal(
                current_price, current_rsi, ma_fast.iloc[-1], ma_slow.iloc[-1]
            )
            if signal.action != "hold":
                logger.debug(
                    f"{self.name}: Signal | Action: {signal.action.upper()} | Pair: {pair} | Confidence: {signal.confidence:.2f} | Reasoning: {signal.reasoning}"
                )

            signals[pair] = signal

        return signals

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(
        self, price: float, rsi: float, ma_fast: float, ma_slow: float
    ) -> Signal:
        """
        Generate a trading signal based on a combination of RSI and Moving Average Crossover.
        The logic is structured to prioritize sell signals in overbought conditions,
        then checks for buy/sell signals based on MA crossovers.
        """
        action = "hold"
        confidence = 0.0
        reasoning = ""
        scaling_factor = 18  # Used to scale MA difference into a confidence score

        # Sell Signal 1: RSI is overbought (highest priority sell signal)
        if rsi > self.rsi_overbought:
            action = "sell"
            # Confidence scales from 0.0 (at RSI=70) to 1.0 (at RSI=100)
            confidence = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            confidence = min(confidence, 0.95)  # Cap confidence at 0.95
            reasoning = f"RSI overbought ({rsi:.1f})"

        # Sell Signal 2: Bearish Moving Average Crossover
        elif ma_fast < ma_slow:
            action = "sell"
            if ma_slow > 0:
                percentage_diff = (ma_slow - ma_fast) / ma_slow
                confidence = min(percentage_diff * scaling_factor, 0.9)
            reasoning = (
                f"Bearish MA crossover (fast: {ma_fast:.2f} < slow: {ma_slow:.2f})"
            )

        # Buy Signal: Bullish MA Crossover and RSI is not yet overbought
        elif ma_fast > ma_slow and rsi < self.rsi_overbought:
            action = "buy"
            if ma_slow > 0:
                percentage_diff = (ma_fast - ma_slow) / ma_slow
                confidence = min(percentage_diff * scaling_factor, 0.9)
            reasoning = f"Bullish MA crossover (fast: {ma_fast:.2f} > slow: {ma_slow:.2f}) and RSI not overbought ({rsi:.1f})"

        return Signal(
            action=action, confidence=confidence, price=price, reasoning=reasoning
        )
