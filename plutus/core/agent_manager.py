import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent
from plutus.agents.momentum import MomentumBot
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
        # Define parameter ranges for momentum bot variations
        rsi_periods = [10, 14, 20]
        ma_fast_hours_options = [5, 10, 15]
        ma_slow_hours_options = [20, 30, 40]
        agent_id = 1

        for rsi in rsi_periods:
            for fast_ma in ma_fast_hours_options:
                for slow_ma in ma_slow_hours_options:
                    # Ensure fast_ma is less than slow_ma
                    if fast_ma >= slow_ma:
                        continue

                    momentum_bot_config = {
                        "pairs": ["XBTUSD", "ETHUSD"],
                        "rsi_period": rsi,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "ma_fast_hours": fast_ma,
                        "ma_slow_hours": slow_ma,
                        "trend_filter_ema_hours": 200,  # Long-term trend filter
                        "min_confidence": 0.5,
                    }
                    self.agents[agent_id] = MomentumBot(
                        name=f"MomentumBot-RSI{rsi}-MA{fast_ma}/{slow_ma}",
                        config=momentum_bot_config,
                        trading_client=self.trading_client,
                    )
                    agent_id += 1

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
            if data:
                agent_data = {
                    pair: df for pair, df in data.items() if pair in agent.pairs
                }
            else:
                agent_data = self.data_manager.get_data_for_agent(agent.pairs)
                timestamp = pd.Timestamp.now(tz="UTC")

            if agent_data:
                await agent.run(agent_data, timestamp)
