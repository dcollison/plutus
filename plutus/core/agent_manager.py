from plutus.agents.base_agent import BaseAgent
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
        # In a real application, you would load agent configurations
        # from a database or a configuration file.
        # For now, we'll just leave this as a placeholder.
        pass

    async def run_agents(self):
        """
        Run all loaded agents. This method will fetch the required data
        for each agent and then execute the agent's strategy.
        """
        for agent_id, agent in self.agents.items():
            # Get the data required for this agent
            agent_data = self.data_manager.get_data_for_agent(agent.pairs)

            # If we have data, run the agent
            if agent_data:
                await agent.run(agent_data)
