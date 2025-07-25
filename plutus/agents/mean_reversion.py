import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent, Signal
from plutus.trading_clients.trading_client import TradingClient


class MeanReversionBot(BaseAgent):
    def __init__(self, name: str, config: dict, trading_client: TradingClient):
        super().__init__(name, config, trading_client)
        self.window = config.get("window", 20)
        self.std_dev = config.get("std_dev", 2)
        self.trend_filter_ema_hours = config.get("trend_filter_ema_hours", 200)

    def get_indicators(self) -> list[str]:
        return ["bollinger_bands", "ema"]

    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}
        for pair, df in data.items():
            # --- Data Interval Calculation ---
            # FIX: Changed the check from < 2 to < 3 to satisfy pandas.infer_freq
            if len(df.index) < 3:
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Not enough data to infer interval"
                )
                continue

            freq = pd.infer_freq(df.index)
            interval_minutes = (
                pd.to_timedelta(freq).total_seconds() / 60
                if freq
                else (df.index[-1] - df.index[-2]).total_seconds() / 60
            )

            if interval_minutes == 0:
                signals[pair] = Signal("hold", 0.0, reasoning="Invalid data interval")
                continue

            # --- Convert Strategy Params to Periods ---
            trend_ema_periods = max(
                1, round((self.trend_filter_ema_hours * 60) / interval_minutes)
            )

            if len(df) < max(self.window, trend_ema_periods):
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Insufficient data for indicators"
                )
                continue

            # --- Indicator Calculation ---
            rolling_mean = df["close"].rolling(window=self.window).mean()
            rolling_std = df["close"].rolling(window=self.window).std()
            upper_band = rolling_mean + (rolling_std * self.std_dev)
            lower_band = rolling_mean - (rolling_std * self.std_dev)
            trend_ema = df["close"].ewm(span=trend_ema_periods, adjust=False).mean()

            current_price = df["close"].iloc[-1]

            # --- Signal Generation ---
            action = "hold"
            confidence = 0.0
            reasoning = ""

            is_uptrend = current_price > trend_ema.iloc[-1]

            # Only look for buy signals (dips) in an established uptrend
            if is_uptrend and current_price < lower_band.iloc[-1]:
                action = "buy"
                confidence = 0.85
                reasoning = f"Uptrend confirmed. Price hit lower Bollinger Band."

            signals[pair] = Signal(
                action=action,
                confidence=confidence,
                price=current_price,
                reasoning=reasoning,
            )

        return signals
