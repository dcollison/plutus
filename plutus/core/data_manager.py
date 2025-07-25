import asyncio
import pandas as pd
from loguru import logger
from pathlib import Path

from plutus.trading_clients.trading_client import TradingClient


class DataManager:
    def __init__(self, trading_client: TradingClient, backtesting: bool = False):
        self.trading_client = trading_client
        self.ohlc_data: dict[str, pd.DataFrame] = {}
        self.live_data = {}
        self.pairs: list[str] = []
        self.backtesting = backtesting

    async def initialise(
        self, pairs: list[str], start_date: str = None, end_date: str = None
    ):
        """
        Initialise the data manager.
        If backtesting, it will load or fetch historical data for the given date range.
        Otherwise, it fetches the latest data.
        """
        logger.info("Initialising DataManager...")
        self.pairs = pairs
        if self.backtesting:
            if not start_date or not end_date:
                raise ValueError(
                    "start_date and end_date must be provided for backtesting"
                )
            await self.fetch_and_save_historical_data(start_date, end_date)
        else:
            await self.update_live_data()

    def _determine_interval(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
        """
        Dynamically determine a suitable interval in minutes based on the time range.
        This helps to fetch data efficiently without making an excessive number of API calls.
        """
        duration_days = (end_ts - start_ts).days
        # Kraken supported intervals in minutes: 1, 5, 15, 30, 60 (1H), 240 (4H), 1440 (1D)
        if duration_days > 365:  # For ranges over a year, use daily data.
            logger.info(
                f"Date range > 1 year ({duration_days} days). Using 1-day interval (1440)."
            )
            return 1440
        elif duration_days > 90:  # For ranges over 3 months, use 4-hour data.
            logger.info(
                f"Date range > 90 days ({duration_days} days). Using 4-hour interval (240)."
            )
            return 240
        elif duration_days > 30:  # For ranges over 1 month, use 1-hour data.
            logger.info(
                f"Date range > 30 days ({duration_days} days). Using 1-hour interval (60)."
            )
            return 60
        elif duration_days > 7:  # For ranges over 1 week, use 15-minute data.
            logger.info(
                f"Date range > 7 days ({duration_days} days). Using 15-minute interval."
            )
            return 15
        else:  # For ranges of a week or less, use 5-minute data.
            logger.info(
                f"Date range <= 7 days ({duration_days} days). Using 5-minute interval."
            )
            return 5

    async def fetch_and_save_historical_data(self, start_date: str, end_date: str):
        """
        Fetch historical OHLC data from the exchange for all pairs and save it to CSV.
        """
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)

        interval = self._determine_interval(start_ts, end_ts)

        logger.info(
            f"Fetching historical data from {start_date} to {end_date} with a {interval}-minute interval..."
        )

        for pair in self.pairs:
            all_data = []
            since = int(start_ts.timestamp())

            while True:
                try:
                    ohlc_result = await self.trading_client.get_ohlc(
                        pair, interval=interval, since=since
                    )
                    if not ohlc_result or "last" not in ohlc_result:
                        break

                    actual_pair = next(key for key in ohlc_result if key != "last")
                    data = ohlc_result[actual_pair]
                    last_ts = ohlc_result["last"]

                    if not data:
                        break

                    all_data.extend(data)
                    # Kraken's `since` is inclusive, so we need to add 1 to the last timestamp to avoid duplicates.
                    since = int(last_ts) + 1
                    last_dt = pd.to_datetime(last_ts, unit="s", utc=True)
                    logger.debug(f"Fetched data for {pair} up to {last_dt}")

                    if last_dt > end_ts:
                        break

                    await asyncio.sleep(1)  # Respect API rate limits

                except Exception as e:
                    logger.error(f"Error fetching historical data for {pair}: {e}")
                    break

            if all_data:
                df = pd.DataFrame(
                    all_data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vwap",
                        "volume",
                        "count",
                    ],
                )
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                numeric_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                    "volume",
                    "count",
                ]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                df = df[(df.index >= start_ts) & (df.index <= end_ts)]
                self.ohlc_data[pair] = df

                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                file_path = data_dir / f"{pair.replace('/', '_')}_{interval}m.csv"
                df.to_csv(file_path)
                logger.info(f"Saved historical data for {pair} to {file_path}")

    async def update_live_data(self):
        """
        Update live market data for all configured pairs for live trading.
        """
        if self.backtesting:
            return

        logger.info("Updating live data for pairs: {}", self.pairs)
        for pair in self.pairs:
            try:
                ohlc_result = await self.trading_client.get_ohlc(pair)
                if not ohlc_result or "last" not in ohlc_result:
                    logger.warning(f"No OHLC data returned for pair {pair}.")
                    continue

                actual_pair = next(key for key in ohlc_result if key != "last")
                data = ohlc_result[actual_pair]
                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vwap",
                        "volume",
                        "count",
                    ],
                )
                numeric_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                    "volume",
                    "count",
                ]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df.set_index("timestamp", inplace=True)
                self.ohlc_data[pair] = df
                logger.info(f"Successfully updated OHLC data for {pair}.")

            except Exception as e:
                logger.exception(f"Failed to update live data for {pair}: {e}")

    def get_data_for_agent(self, agent_pairs: list[str]) -> dict[str, pd.DataFrame]:
        """Get the relevant data for a specific agent."""
        agent_data = {}
        for pair in agent_pairs:
            if pair in self.ohlc_data:
                agent_data[pair] = self.ohlc_data[pair]
        return agent_data
