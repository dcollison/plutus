import pandas as pd
from loguru import logger

from plutus.trading_clients.trading_client import TradingClient


class DataManager:
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.ohlc_data = {}
        self.live_data = {}

    async def initialise(self):
        """Initialise the data manager with historical data."""
        logger.info("Initialising DataManager...")
        # In a real application, you would load historical data here
        pass

    async def update_live_data(self):
        """Update live market data for all pairs."""
        logger.info("Updating live data...")
        # This is where you would call the trading client to get the latest ticker or OHLC data
        # For now, we'll just log a message
        logger.info("Live data updated.")

    def get_data_for_agent(self, agent_pairs: list[str]) -> dict[str, pd.DataFrame]:
        """Get the relevant data for a specific agent."""
        agent_data = {}
        for pair in agent_pairs:
            if pair in self.ohlc_data:
                agent_data[pair] = self.ohlc_data[pair]
        return agent_data
