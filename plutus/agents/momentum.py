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
        self.ma_fast_hours = config.get("ma_fast_hours", 10)
        self.ma_slow_hours = config.get("ma_slow_hours", 20)
        # Add a long-term EMA for trend filtering
        self.trend_filter_ema_hours = config.get("trend_filter_ema_hours", 200)

    def get_indicators(self) -> list[str]:
        return ["rsi", "sma", "ema", "volume"]

    async def analyse(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}

        for pair, df in data.items():
            if len(df.index) < 3:
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Not enough data to infer interval"
                )
                continue

            freq = pd.infer_freq(df.index)
            if freq is None:
                interval_minutes = (df.index[-1] - df.index[-2]).total_seconds() / 60
            else:
                interval_minutes = pd.to_timedelta(freq).total_seconds() / 60

            if interval_minutes == 0:
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Invalid data interval (0 minutes)"
                )
                continue

            # Convert time-based strategy parameters to period-based
            ma_fast_periods = max(
                1, round((self.ma_fast_hours * 60) / interval_minutes)
            )
            ma_slow_periods = max(
                1, round((self.ma_slow_hours * 60) / interval_minutes)
            )
            trend_ema_periods = max(
                1, round((self.trend_filter_ema_hours * 60) / interval_minutes)
            )

            if df.empty or len(df) < max(
                self.rsi_period, ma_slow_periods, trend_ema_periods
            ):
                signals[pair] = Signal(
                    "hold", 0.0, reasoning="Insufficient data for indicators"
                )
                continue

            # Calculate indicators
            rsi = self.calculate_rsi(df["close"], self.rsi_period)
            ma_fast = df["close"].rolling(ma_fast_periods).mean()
            ma_slow = df["close"].rolling(ma_slow_periods).mean()
            trend_ema = df["close"].ewm(span=trend_ema_periods, adjust=False).mean()

            current_price = df["close"].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_trend_ema = trend_ema.iloc[-1]

            signal = self.generate_signal(
                current_price,
                current_rsi,
                ma_fast.iloc[-1],
                ma_slow.iloc[-1],
                current_trend_ema,
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
        self, price: float, rsi: float, ma_fast: float, ma_slow: float, trend_ema: float
    ) -> Signal:
        """
        Generate a trading signal using a trend filter.
        Only allows long positions when the price is above the long-term trend EMA,
        and short positions when the price is below.
        """
        action = "hold"
        confidence = 0.0
        reasoning = ""
        scaling_factor = 18

        # Determine long-term trend
        is_uptrend = price > trend_ema
        is_downtrend = price < trend_ema

        # --- Buy Logic (only in an uptrend) ---
        if is_uptrend and ma_fast > ma_slow and rsi < self.rsi_overbought:
            action = "buy"
            if ma_slow > 0:
                percentage_diff = (ma_fast - ma_slow) / ma_slow
                confidence = min(percentage_diff * scaling_factor, 0.9)
            reasoning = f"Uptrend confirmed. Bullish MA crossover and RSI not overbought ({rsi:.1f})"

        # --- Sell Logic (only in a downtrend) ---
        elif is_downtrend:
            if rsi > self.rsi_overbought:
                action = "sell"
                confidence = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                confidence = min(confidence, 0.95)
                reasoning = f"Downtrend confirmed. RSI overbought ({rsi:.1f})"
            elif ma_fast < ma_slow:
                action = "sell"
                if ma_slow > 0:
                    percentage_diff = (ma_slow - ma_fast) / ma_slow
                    confidence = min(percentage_diff * scaling_factor, 0.9)
                reasoning = f"Downtrend confirmed. Bearish MA crossover."

        return Signal(
            action=action, confidence=confidence, price=price, reasoning=reasoning
        )
