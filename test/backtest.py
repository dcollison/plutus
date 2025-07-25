import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from plutus.trading_clients.trading_client import TradingClient


class SimulatedTradingClient(TradingClient):
    """
    A simulated trading client for backtesting.
    It mimics the behavior of a real trading client but operates on historical data.
    Each instance of this class represents a single, isolated trading account.
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
            # Use a small tolerance for float comparison
            if self.positions.get(pair, 0) < volume - 1e-9:
                logger.warning(
                    f"[{agent_name}] Not enough {pair} to sell. Has: {self.positions.get(pair, 0)}, needs: {volume}"
                )
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
        # Use debug level for simulated orders to reduce noise
        logger.debug(f"[{agent_name}] Simulated order placed: {trade}")
        return {"txid": [f"sim_{len(self.trade_history)}"]}

    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> dict:
        # This is not used by the backtester's simulated client
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

        # Determine the primary dataframe for the backtest time range
        # This assumes all pairs have data for the same range.
        first_pair = next(iter(self.data_manager.ohlc_data.keys()))
        historical_data = self.data_manager.ohlc_data[first_pair]
        backtest_range = historical_data[
            (historical_data.index >= self.start_date)
            & (historical_data.index <= self.end_date)
        ]

        for timestamp in backtest_range.index:
            current_data_slice = {}
            for pair, full_hist_df in self.data_manager.ohlc_data.items():
                # Provide each agent with all data up to the current timestamp
                current_data_slice[pair] = full_hist_df.loc[
                    full_hist_df.index <= timestamp
                ]

            if current_data_slice:
                await self.agent_manager.run_agents(
                    data=current_data_slice, timestamp=timestamp
                )

        await self.liquidate_all_positions(backtest_range.index[-1])
        logger.info("Backtest finished.")
        self.summarize_results()

    async def liquidate_all_positions(self, final_timestamp: pd.Timestamp):
        """
        Sells all open positions at the final closing price of the simulation for each agent.
        """
        logger.info("Liquidating all open positions at end of simulation...")
        for agent in self.agent_manager.agents.values():
            # Use list(agent.positions.items()) to avoid issues with modifying dict during iteration
            for pair, position in list(agent.positions.items()):
                if position.volume > 1e-9:
                    last_price = self.data_manager.ohlc_data[pair]["close"].iloc[-1]
                    logger.info(
                        f"Liquidating {position.volume:.4f} of {pair} for {agent.name} at ${last_price:,.2f}"
                    )
                    # The agent uses its own isolated trading client to liquidate
                    await agent.trading_client.place_order(
                        pair=pair,
                        type_="sell",
                        ordertype="market",
                        volume=position.volume,
                        price=last_price,
                        timestamp=final_timestamp,
                        agent_name=agent.name,
                    )

    def _calculate_benchmark_performance(self, initial_balance: float):
        """Calculates performance for Buy & Hold and DCA strategies."""
        benchmark_results = []
        for pair in self.data_manager.pairs:
            pair_data = self.data_manager.ohlc_data[pair]
            backtest_data = pair_data[
                (pair_data.index >= self.start_date)
                & (pair_data.index <= self.end_date)
            ]
            if backtest_data.empty:
                continue

            start_price = backtest_data["close"].iloc[0]
            end_price = backtest_data["close"].iloc[-1]

            # Buy and Hold
            bnh_coins = initial_balance / start_price
            bnh_final_value = bnh_coins * end_price
            bnh_profit = bnh_final_value - initial_balance
            benchmark_results.append(
                {
                    "strategy": f"Buy & Hold ({pair})",
                    "net_profit": bnh_profit,
                    "profit_pct": (bnh_profit / initial_balance) * 100,
                    "trades": 1,
                }
            )
        return benchmark_results

    def summarize_results(self):
        """Prints a summary table of the backtest performance for each agent and benchmarks."""
        results = []
        total_initial_balance = 0

        # Agent Performance
        for agent in self.agent_manager.agents.values():
            # Each agent has its own client with its own history and balance
            client = agent.trading_client
            initial_balance = client.initial_balance
            total_initial_balance += initial_balance

            final_balance = client.balance["USD"]
            net_profit = final_balance - initial_balance

            results.append(
                {
                    "strategy": agent.name,
                    "net_profit": net_profit,
                    "profit_pct": (
                        (net_profit / initial_balance) * 100 if initial_balance else 0
                    ),
                    "trades": len(client.trade_history),
                }
            )

        # Benchmark Performance (calculated on a single initial balance for comparison)
        if self.agent_manager.agents:
            first_agent_client = next(
                iter(self.agent_manager.agents.values())
            ).trading_client
            benchmark_initial_balance = first_agent_client.initial_balance
            results.extend(
                self._calculate_benchmark_performance(benchmark_initial_balance)
            )

        console = Console()
        console.print()
        table = Table(
            title="Backtest Performance Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Strategy", style="dim", width=35)
        table.add_column("Net P/L ($)", justify="right")
        table.add_column("Net P/L (%)", justify="right")
        table.add_column("Trades", justify="center")

        sorted_results = sorted(results, key=lambda x: x["net_profit"], reverse=True)

        for result in sorted_results:
            profit_str = f"${result['net_profit']:,.2f}"
            profit_pct_str = f"{result['profit_pct']:.2f}%"
            style = "green" if result["net_profit"] > 0 else "red"
            table.add_row(
                result["strategy"],
                f"[{style}]{profit_str}[/{style}]",
                f"[{style}]{profit_pct_str}[/{style}]",
                str(result["trades"]),
            )

        console.print(table)
        if self.agent_manager.agents:
            console.print(
                f"[bold]Initial Portfolio Value (per agent):[/] ${benchmark_initial_balance:,.2f}"
            )
