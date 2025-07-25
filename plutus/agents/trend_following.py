import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent, Signal
from plutus.trading_clients.trading_client import TradingClient


class TrendFollowingBot(BaseAgent):
    def __init__(self, name: str, config: dict, trading_client: TradingClient):
        super().__init__(name, config, trading_client)
        self.fast_period = config.get("fast_period", 12)
        self.slow_period = config.get("slow_period", 26)
        self.signal_period = config.get("signal_period", 9)
        self.adx_period = config.get("adx_period", 14)
        self.adx_threshold = config.get("adx_threshold", 25)

    def get_indicators(self) -> list[str]:
        return ["macd", "adx"]

    def _calculate_adx(self, df: pd.DataFrame, period: int):
        """Calculates the Average Directional Index (ADX)."""
        # FIX: Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        df["H-L"] = df["high"] - df["low"]
        df["H-PC"] = abs(df["high"] - df["close"].shift(1))
        df["L-PC"] = abs(df["low"] - df["close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

        df["+DM"] = (df["high"] - df["high"].shift(1)) > (
            df["low"].shift(1) - df["low"]
        )
        df["+DM"] = df["+DM"].where(df["+DM"], 0) * (df["high"] - df["high"].shift(1))

        df["-DM"] = (df["low"].shift(1) - df["low"]) > (
            df["high"] - df["high"].shift(1)
        )
        df["-DM"] = df["-DM"].where(df["-DM"], 0) * (df["low"].shift(1) - df["low"])

        tr_sum = df["TR"].rolling(window=period).sum()
        plus_di = 100 * (df["+DM"].rolling(window=period).sum() / tr_sum)
        minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr_sum)

        # To avoid division by zero
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
        adx = dx.rolling(window=period).mean()
        return adx

    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}
        for pair, df in data.items():
            if len(df) < max(self.slow_period, self.adx_period * 2):
                signals[pair] = Signal("hold", 0.0, reasoning="Insufficient data")
                continue

            # --- Indicator Calculation ---
            exp1 = df["close"].ewm(span=self.fast_period, adjust=False).mean()
            exp2 = df["close"].ewm(span=self.slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()
            adx = self._calculate_adx(df, self.adx_period)

            if adx.isna().all():
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="ADX could not be calculated"
                )
                continue

            current_price = df["close"].iloc[-1]
            current_adx = adx.iloc[-1]

            # --- Signal Generation ---
            action = "hold"
            confidence = 0.0
            reasoning = ""

            is_strong_trend = current_adx > self.adx_threshold
            is_bullish_crossover = (
                macd.iloc[-1] > signal_line.iloc[-1]
                and macd.iloc[-2] < signal_line.iloc[-2]
            )

            if is_strong_trend and is_bullish_crossover:
                action = "buy"
                confidence = 0.8
                reasoning = (
                    f"Strong trend (ADX {current_adx:.1f}) with bullish MACD crossover."
                )

            signals[pair] = Signal(
                action=action,
                confidence=confidence,
                price=current_price,
                reasoning=reasoning,
            )

        return signals
