import pandas as pd

from plutus.agents.base import BaseAgent, Signal


class MomentumBot(BaseAgent):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.ma_fast = config.get("ma_fast", 10)
        self.ma_slow = config.get("ma_slow", 20)

    def get_indicators(self) -> list[str]:
        return ["rsi", "sma", "volume"]

    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}

        for pair, df in data.items():
            if df.empty or len(df) < max(self.rsi_period, self.ma_slow):
                signals[pair] = Signal("hold", 0.0, reasoning="Insufficient data")
                continue

            # Calculate indicators
            rsi = self.calculate_rsi(df["close"], self.rsi_period)
            ma_fast = df["close"].rolling(self.ma_fast).mean()
            ma_slow = df["close"].rolling(self.ma_slow).mean()

            current_price = df["close"].iloc[-1]
            current_rsi = rsi.iloc[-1]

            # Generate signal
            signal = self.generate_signal(
                current_price, current_rsi, ma_fast.iloc[-1], ma_slow.iloc[-1]
            )
            signals[pair] = signal

        return signals

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(
        self, price: float, rsi: float, ma_fast: float, ma_slow: float
    ) -> Signal:
        """Generate trading signal based on momentum indicators"""
        confidence = 0.0
        action = "hold"
        reasoning = ""

        # Buy signal: RSI oversold and fast MA above slow MA
        if rsi < self.rsi_oversold and ma_fast > ma_slow:
            action = "buy"
            confidence = min(0.9, (self.rsi_oversold - rsi) / self.rsi_oversold + 0.3)
            reasoning = f"RSI oversold ({rsi:.1f}) with bullish MA crossover"

        # Sell signal: RSI overbought or fast MA below slow MA
        elif rsi > self.rsi_overbought or ma_fast < ma_slow:
            action = "sell"
            confidence = min(
                0.9, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + 0.3
            )
            reasoning = f"RSI overbought ({rsi:.1f}) or bearish MA crossover"

        return Signal(
            action=action, confidence=confidence, price=price, reasoning=reasoning
        )
