import pandas as pd
from loguru import logger

from plutus.agents.base_agent import BaseAgent
from plutus.agents.momentum import MomentumBot
from plutus.agents.trend_following import TrendFollowingBot
from plutus.agents.mean_reversion import MeanReversionBot
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
        agent_id_counter = 1

        # --- Load Momentum Bots ---
        rsi_periods = [14, 21]
        ma_fast_hours_options = [10, 20]
        ma_slow_hours_options = [30, 50]

        for rsi in rsi_periods:
            for fast_ma in ma_fast_hours_options:
                for slow_ma in ma_slow_hours_options:
                    if fast_ma >= slow_ma:
                        continue
                    config = {
                        "pairs": ["XBTUSD", "ETHUSD"],
                        "rsi_period": rsi,
                        "ma_fast_hours": fast_ma,
                        "ma_slow_hours": slow_ma,
                        "min_confidence": 0.6,
                    }
                    self.agents[agent_id_counter] = MomentumBot(
                        name=f"Momentum-RSI{rsi}-MA{fast_ma}/{slow_ma}",
                        config=config,
                        trading_client=self.trading_client,
                    )
                    agent_id_counter += 1

        # --- Load Trend Following Bots ---
        macd_settings = [(12, 26, 9), (20, 50, 10)]
        for fast, slow, sig in macd_settings:
            config = {
                "pairs": ["XBTUSD"],
                "fast_period": fast,
                "slow_period": slow,
                "signal_period": sig,
                "min_confidence": 0.7,
            }
            self.agents[agent_id_counter] = TrendFollowingBot(
                name=f"TrendFollow-MACD{fast}/{slow}/{sig}",
                config=config,
                trading_client=self.trading_client,
            )
            agent_id_counter += 1

        # --- Load Mean Reversion Bots ---
        bollinger_settings = [(20, 2), (30, 2.5)]
        for window, std in bollinger_settings:
            config = {
                "pairs": ["ETHUSD"],
                "window": window,
                "std_dev": std,
                "min_confidence": 0.75,
            }
            self.agents[agent_id_counter] = MeanReversionBot(
                name=f"MeanRevert-BB{window}/{std}",
                config=config,
                trading_client=self.trading_client,
            )
            agent_id_counter += 1

        logger.info(f"Loaded {len(self.agents)} agents.")

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
