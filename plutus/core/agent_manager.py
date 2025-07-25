import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent
from plutus.agents.momentum import MomentumBot  # Import the MomentumBot
from plutus.core.data_manager import DataManager
from plutus.trading_clients.trading_client import TradingClient


class AgentManager:
    def __init__(
        self,
        trading_config: dict,
        trading_client: TradingClient,
        data_manager: DataManager,
    ):
        self.trading_config: dict = trading_config
        self.trading_client: TradingClient = trading_client
        self.data_manager = data_manager
        self.agents: dict[int, BaseAgent] = {}

    def load_agents(self):
        """
        Load all the trading agents that are defined in the configuration.
        """
        # For now, we'll just load the MomentumBot.
        # In the future, you could load this from a config file.
        momentum_bot_config = {
            "pairs": ["XBTUSD", "ETHUSD"],
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "ma_fast": 10,
            "ma_slow": 20,
        }
        self.agents[1] = MomentumBot(
            name="MomentumBot",
            config=momentum_bot_config,
            trading_client=self.trading_client,
        )

    async def run_agents(
        self, data: dict[str, pd.DataFrame] = None, timestamp: pd.Timestamp = None
    ):
        """
        Run all loaded agents. This method will fetch the required data
        for each agent and then execute the agent's strategy.
        If data is provided (e.g., in a backtest), it will be used directly.
        """
        logger.debug(f"Running {len(self.agents)} agent(s)")

        for agent_id, agent in self.agents.items():
            # If data is provided for the backtest, use it. Otherwise, fetch from DataManager.
            if data:
                # Filter the provided data for the pairs this agent is interested in.
                agent_data = {
                    pair: df for pair, df in data.items() if pair in agent.pairs
                }
            else:
                # Get the data required for this agent for live trading
                agent_data = self.data_manager.get_data_for_agent(agent.pairs)
                timestamp = pd.Timestamp.now(tz="UTC")

            # If we have data, run the agent
            if agent_data:
                await agent.run(agent_data, timestamp)
