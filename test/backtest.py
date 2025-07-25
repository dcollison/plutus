import pandas as pd
from loguru import logger
from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from plutus.trading_clients.trading_client import TradingClient
from plutus.agents.base_agent import Signal


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
        **kwargs,
    ) -> dict:
        """Simulates placing an order."""
        if ordertype == "market":
            # In a real backtest, you'd use the current price from the data feed.
            # For simplicity, we'll assume the provided price is the execution price.
            pass

        cost = volume * price
        if type_ == "buy":
            if self.balance["USD"] < cost:
                logger.warning("Insufficient funds to place buy order.")
                return {}
            self.balance["USD"] -= cost
            self.positions[pair] = self.positions.get(pair, 0) + volume
        elif type_ == "sell":
            if self.positions.get(pair, 0) < volume:
                logger.warning(f"Not enough {pair} to sell.")
                return {}
            self.balance["USD"] += cost
            self.positions[pair] -= volume
            # Clean up positions that are effectively zero due to floating point inaccuracies
            if self.positions[pair] < 1e-9:
                del self.positions[pair]

        trade = {
            "pair": pair,
            "type": type_,
            "ordertype": ordertype,
            "volume": volume,
            "price": price,
            "cost": cost,
            "timestamp": timestamp,  # Use the timestamp from the simulation
        }
        self.trade_history.append(trade)
        logger.info(f"Simulated order placed: {trade}")
        return {"txid": [f"sim_{len(self.trade_history)}"]}

    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> dict:
        # This will be handled by the DataManager loading historical data.
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

        # Use the first pair of the first agent to get the main timeline for the backtest.
        if not self.agent_manager.agents:
            logger.error("No agents loaded. Aborting backtest.")
            return

        first_agent = next(iter(self.agent_manager.agents.values()))
        if not first_agent.pairs:
            logger.error(
                f"Agent {first_agent.name} has no pairs assigned. Aborting backtest."
            )
            return

        first_pair = first_agent.pairs[0]
        if first_pair not in self.data_manager.ohlc_data:
            logger.error(
                f"No historical data found for the first pair {first_pair}. Aborting backtest."
            )
            return

        historical_data = self.data_manager.ohlc_data[first_pair]

        # Filter data for the specified date range
        backtest_range = historical_data[
            (historical_data.index >= self.start_date)
            & (historical_data.index <= self.end_date)
        ]

        for timestamp in backtest_range.index:
            # Create a slice of data up to the current timestamp for each pair
            current_data_slice = {}
            for pair, full_hist_df in self.data_manager.ohlc_data.items():
                # The agent needs a window of data to calculate indicators
                current_data_slice[pair] = full_hist_df.loc[
                    full_hist_df.index <= timestamp
                ]

            if current_data_slice:
                # Pass the sliced data and the current timestamp to the agent manager
                await self.agent_manager.run_agents(
                    data=current_data_slice, timestamp=timestamp
                )

        logger.info("Backtest finished.")
        self.summarize_results()

    def summarize_results(self):
        """
        Prints a summary of the backtest performance.
        """
        logger.info("--- Backtest Summary ---")
        sim_client = self.agent_manager.trading_client
        initial_balance = sim_client.initial_balance
        final_balance = sim_client.balance.get("USD", 0)

        # Calculate final portfolio value including held assets
        final_portfolio_value = final_balance
        if sim_client.positions:
            logger.info("Ending positions held:")
            for pair, volume in sim_client.positions.items():
                if volume > 1e-9:  # Only show positions with a meaningful amount
                    # Get the very last price from the historical data to value the position
                    last_price = self.data_manager.ohlc_data[pair]["close"].iloc[-1]
                    position_value = volume * last_price
                    final_portfolio_value += position_value
                    logger.info(
                        f"  - {pair}: {volume:.4f} units @ ${last_price:,.2f} = ${position_value:,.2f}"
                    )

        total_trades = len(sim_client.trade_history)
        profit = final_portfolio_value - initial_balance
        profit_pct = (profit / initial_balance) * 100

        logger.info(f"Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        logger.info(f"Total Trades Executed: {total_trades}")
        logger.info(f"Net Profit: ${profit:,.2f} ({profit_pct:.2f}%)")
        logger.info("----------------------")
