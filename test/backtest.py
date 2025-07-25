import pandas as pd
from loguru import logger
from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from plutus.trading_clients.trading_client import TradingClient


class SimulatedTradingClient(TradingClient):
    """
    A simulated trading client for backtesting.
    It mimics the behavior of a real trading client but operates on historical data.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = {"USD": initial_balance}
        self.positions = {}
        self.trade_history = []

    async def get_account_balance(self) -> dict:
        return self.balance

    async def place_order(
        self,
        pair: str,
        type_: str,
        ordertype: str,
        volume: float,
        price: float = None,
        timestamp: pd.Timestamp = None,
        agent_name: str = "Unknown",
        **kwargs,
    ) -> dict:
        """Simulates placing an order and records which agent placed it."""
        cost = volume * price
        if type_ == "buy":
            if self.balance["USD"] < cost:
                logger.warning(f"[{agent_name}] Insufficient funds to place buy order.")
                return {}
            self.balance["USD"] -= cost
            self.positions[pair] = self.positions.get(pair, 0) + volume
        elif type_ == "sell":
            if self.positions.get(pair, 0) < volume:
                logger.warning(f"[{agent_name}] Not enough {pair} to sell.")
                return {}
            self.balance["USD"] += cost
            self.positions[pair] -= volume
            if self.positions[pair] < 1e-9:
                del self.positions[pair]

        trade = {
            "agent_name": agent_name,
            "pair": pair,
            "type": type_,
            "ordertype": ordertype,
            "volume": volume,
            "price": price,
            "cost": cost,
            "timestamp": timestamp,
        }
        self.trade_history.append(trade)
        logger.info(f"[{agent_name}] Simulated order placed: {trade}")
        return {"txid": [f"sim_{len(self.trade_history)}"]}

    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> dict:
        return {}


class BacktestEngine:
    """
    The main engine for running backtests.
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        data_manager: DataManager,
        start_date: str,
        end_date: str,
    ):
        self.agent_manager = agent_manager
        self.data_manager = data_manager
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)

    async def run(self):
        """
        Runs the backtest by iterating through historical data timestep by timestep.
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}...")

        if not self.agent_manager.agents:
            logger.error("No agents loaded. Aborting backtest.")
            return

        first_agent = next(iter(self.agent_manager.agents.values()))
        if not first_agent.pairs:
            logger.error(f"Agent {first_agent.name} has no pairs assigned.")
            return

        first_pair = first_agent.pairs[0]
        if first_pair not in self.data_manager.ohlc_data:
            logger.error(f"No historical data for {first_pair}. Aborting backtest.")
            return

        historical_data = self.data_manager.ohlc_data[first_pair]

        backtest_range = historical_data[
            (historical_data.index >= self.start_date)
            & (historical_data.index <= self.end_date)
        ]

        for timestamp in backtest_range.index:
            current_data_slice = {}
            for pair, full_hist_df in self.data_manager.ohlc_data.items():
                current_data_slice[pair] = full_hist_df.loc[
                    full_hist_df.index <= timestamp
                ]

            if current_data_slice:
                await self.agent_manager.run_agents(
                    data=current_data_slice, timestamp=timestamp
                )

        logger.info("Backtest finished.")
        self.summarize_results()

    def summarize_results(self):
        """
        Prints a detailed summary of the backtest performance for each agent.
        """
        sim_client = self.agent_manager.trading_client
        initial_balance = sim_client.initial_balance

        logger.info("--- Backtest Summary ---")
        logger.info(f"Initial Portfolio Value: ${initial_balance:,.2f}")

        for agent_id, agent in self.agent_manager.agents.items():
            agent_trades = [
                t for t in sim_client.trade_history if t["agent_name"] == agent.name
            ]

            if not agent_trades:
                logger.info(f"\n--- Results for {agent.name} ---")
                logger.info("No trades executed.")
                continue

            profit = 0
            for trade in agent_trades:
                if trade["type"] == "buy":
                    profit -= trade["cost"]
                else:  # sell
                    profit += trade["cost"]

            final_positions_value = 0
            if agent.positions:
                for pair, volume in agent.positions.items():
                    if volume > 1e-9:
                        last_price = self.data_manager.ohlc_data[pair]["close"].iloc[-1]
                        position_value = volume * last_price
                        final_positions_value += position_value

            net_profit = profit + final_positions_value
            # Note: This is a simplified P&L calculation. A real backtest would track
            # the balance changes per agent, but this gives a good per-agent performance metric.

            logger.info(f"--- Results for {agent.name} ---")
            logger.info(f"Total Trades Executed: {len(agent_trades)}")
            logger.info(f"Net Profit/Loss: ${net_profit:,.2f}")

            if agent.positions:
                logger.info("Ending positions held by this agent:")
                for pair, volume in agent.positions.items():
                    if volume > 1e-9:
                        last_price = self.data_manager.ohlc_data[pair]["close"].iloc[-1]
                        position_value = volume * last_price
                        logger.info(
                            f"  - {pair}: {volume:.4f} units @ ${last_price:,.2f} = ${position_value:,.2f}"
                        )

        logger.info("------------------------")
