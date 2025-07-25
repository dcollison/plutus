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

        await self.liquidate_all_positions(backtest_range.index[-1])
        logger.info("Backtest finished.")
        self.summarize_results()

    async def liquidate_all_positions(self, final_timestamp: pd.Timestamp):
        """
        Sells all open positions at the final closing price of the simulation.
        """
        logger.info("Liquidating all open positions at end of simulation...")
        for agent_id, agent in self.agent_manager.agents.items():
            for pair, volume in list(agent.positions.items()):
                if volume > 1e-9:
                    last_price = self.data_manager.ohlc_data[pair]["close"].iloc[-1]
                    logger.info(
                        f"Liquidating {volume:.4f} of {pair} for {agent.name} at ${last_price:,.2f}"
                    )
                    await agent.trading_client.place_order(
                        pair=pair,
                        type_="sell",
                        ordertype="market",
                        volume=volume,
                        price=last_price,
                        timestamp=final_timestamp,
                        agent_name=agent.name,
                    )
                    agent.positions.pop(pair, None)

    def _calculate_benchmark_performance(self):
        """Calculates performance for Buy & Hold and DCA strategies and returns the results."""
        initial_balance = self.agent_manager.trading_client.initial_balance
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

            # Dollar Cost Averaging (weekly)
            dca_schedule = pd.date_range(
                start=self.start_date, end=self.end_date, freq="W"
            )
            if not dca_schedule.empty:
                investment_per_period = initial_balance / len(dca_schedule)
                total_coins_dca = 0
                for investment_date in dca_schedule:
                    closest_price_point = backtest_data.index.get_indexer(
                        [investment_date], method="nearest"
                    )[0]
                    dca_price = backtest_data["close"].iloc[closest_price_point]
                    total_coins_dca += investment_per_period / dca_price

                dca_final_value = total_coins_dca * end_price
                dca_profit = dca_final_value - initial_balance
                benchmark_results.append(
                    {
                        "strategy": f"DCA (Weekly, {pair})",
                        "net_profit": dca_profit,
                        "profit_pct": (dca_profit / initial_balance) * 100,
                        "trades": len(dca_schedule),
                    }
                )
        return benchmark_results

    def summarize_results(self):
        """Prints a detailed summary table of the backtest performance for each agent and benchmarks."""
        sim_client = self.agent_manager.trading_client
        initial_balance = sim_client.initial_balance
        results = []

        # Agent Performance
        for agent_id, agent in self.agent_manager.agents.items():
            agent_trades = [
                t for t in sim_client.trade_history if t["agent_name"] == agent.name
            ]
            profit = sum(
                -t["cost"] if t["type"] == "buy" else t["cost"] for t in agent_trades
            )
            results.append(
                {
                    "strategy": agent.name,
                    "net_profit": profit,
                    "profit_pct": (
                        (profit / initial_balance) * 100 if initial_balance else 0
                    ),
                    "trades": len(agent_trades),
                }
            )

        # Benchmark Performance
        results.extend(self._calculate_benchmark_performance())

        # --- Display Results using Rich Table ---
        console = Console()
        console.print()

        table = Table(
            title="Backtest Performance Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Strategy", style="dim", width=25)
        table.add_column("Net P/L ($)", justify="right")
        table.add_column("Net P/L (%)", justify="right")
        table.add_column("Trades", justify="center")

        # Sort results by net profit in descending order
        sorted_results = sorted(results, key=lambda x: x["net_profit"], reverse=True)

        for result in sorted_results:
            profit_str = f"${result['net_profit']:,.2f}"
            profit_pct_str = f"{result['profit_pct']:.2f}%"

            # Color code the profit/loss
            style = "green" if result["net_profit"] > 0 else "red"

            table.add_row(
                result["strategy"],
                f"[{style}]{profit_str}[/{style}]",
                f"[{style}]{profit_pct_str}[/{style}]",
                str(result["trades"]),
            )

        console.print(table)
        console.print(f"[bold]Initial Portfolio Value:[/] ${initial_balance:,.2f}")
