from plutus.agents.base import BaseAgent
from plutus.trading_clients.trading_client import TradingClient


class AgentManager:
    def __init__(self, trading_config: dict, trading_client):
        self.trading_config: dict = trading_config
        self.trading_client: TradingClient = trading_client
        self.agents: dict[int, BaseAgent] = {}

    def load_agents(self):
        pass
