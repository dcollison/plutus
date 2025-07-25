import pandas as pd
from loguru import logger

from plutus.trading_clients.trading_client import TradingClient


class DataManager:
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.ohlc_data: dict[str, pd.DataFrame] = {}
        self.live_data = {}
        self.pairs: list[str] = []

    async def initialise(self, pairs: list[str]):
        """
        Initialise the data manager with historical data.
        This should be called with the list of pairs to track.
        """
        logger.info("Initialising DataManager...")
        self.pairs = pairs
        await self.update_live_data()

    async def update_live_data(self):
        """
        Update live market data for all configured pairs.
        It fetches OHLC data from the trading client, converts the data into a pandas DataFrame,
        and ensures that price and volume columns are numeric.
        """
        logger.info("Updating live data for pairs: {}", self.pairs)
        if not self.pairs:
            logger.warning("No pairs to update in DataManager.")
            return

        for pair in self.pairs:
            try:
                # Fetch OHLC data from the trading client
                ohlc_result = await self.trading_client.get_ohlc(pair)

                key = tuple(ohlc_result.keys())[0]
                # The result from Kraken is a dictionary where the key is the pair name.
                if key in ohlc_result:
                    data = ohlc_result[key]
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

                    # Convert relevant columns to numeric types, as the API returns them as strings.
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

                    # Convert Unix timestamp to datetime objects and set as index
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)

                    self.ohlc_data[pair] = df
                    logger.info(f"Successfully updated OHLC data for {pair}.")
                else:
                    logger.warning(f"No OHLC data returned for pair {pair}.")

            except Exception as e:
                logger.error(f"Failed to update live data for {pair}: {e}")

    def get_data_for_agent(self, agent_pairs: list[str]) -> dict[str, pd.DataFrame]:
        """Get the relevant data for a specific agent."""
        agent_data = {}
        for pair in agent_pairs:
            if pair in self.ohlc_data:
                agent_data[pair] = self.ohlc_data[pair]
        return agent_data
