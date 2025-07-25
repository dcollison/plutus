import asyncio
import sys
from loguru import logger

from plutus.config.settings import Settings
from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from backtest import BacktestEngine, SimulatedTradingClient
from plutus.trading_clients.kraken_client import KrakenClient


async def main():
    # Setup logger
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level="DEBUG",
        enqueue=True,
    )
    logger.add("logs/plutus_backtest_{time}.log", enqueue=True)

    settings = Settings()

    # Use the real KrakenClient to fetch historical data
    real_client = KrakenClient(
        api_key=settings.kraken_api_key,
        api_secret=settings.kraken_api_secret,
    )

    # Use the SimulatedTradingClient for paper trading during the backtest
    simulated_client = SimulatedTradingClient(initial_balance=10000.0)

    # DataManager gets the REAL client to fetch data if needed.
    data_manager = DataManager(real_client, backtesting=True)

    # AgentManager gets the SIMULATED client to execute paper trades.
    agent_manager = AgentManager(
        settings.trading_config,
        simulated_client,
        data_manager,
    )

    # Load agents
    agent_manager.load_agents()

    # Define your backtest period here
    start_date = "2025-01-01"
    end_date = "2025-07-24"

    # Initialise the data manager. It will use the real_client to fetch if necessary.
    await data_manager.initialise(
        settings.trading_config.pairs, start_date=start_date, end_date=end_date
    )

    backtest_engine = BacktestEngine(
        agent_manager,
        data_manager,
        start_date=start_date,
        end_date=end_date,
    )

    await backtest_engine.run()


if __name__ == "__main__":
    asyncio.run(main())
