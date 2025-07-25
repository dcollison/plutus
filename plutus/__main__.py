import asyncio
import sys
import time
from itertools import count

from loguru import logger
from rich.console import Console

from plutus.config.settings import Settings
from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from plutus.core.portfolio_manager import PortfolioManager
from plutus.trading_clients.kraken_client import KrakenClient


class Plutus:
    def __init__(self):
        self.settings = Settings()
        self.trading_client = KrakenClient(
            api_key=self.settings.kraken_api_key,
            api_secret=self.settings.kraken_api_secret,
            sandbox=self.settings.use_sandbox,
        )
        self.data_manager = DataManager(self.trading_client)
        self.portfolio_manager = PortfolioManager(self.trading_client)
        self.agent_manager = AgentManager(
            self.settings.trading_config,
            self.trading_client,
            self.data_manager,
        )
        self.dash_app = None
        self.running = False

    async def start(self):
        """Start the trading application"""
        logger.info("Starting Plutus...")
        await self.data_manager.initialise(self.settings.trading_config.pairs)
        self.agent_manager.load_agents()
        self.running = True
        await self.trading_loop()

    async def trading_loop(self):
        """Main trading loop"""
        cycles = count(1)
        while self.running:
            i_cycle = next(cycles)
            t_start = time.perf_counter()
            try:
                logger.info(f"Cycle {i_cycle:} Started")
                await self.data_manager.update_live_data()
                await self.agent_manager.run_agents()
                await self.portfolio_manager.update_portfolio()
                t_end = time.perf_counter()
                logger.info(f"Cycle {i_cycle:} Complete | {t_end - t_start:0.2f}s")
                await asyncio.sleep(self.settings.update_interval)
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                t_end = time.perf_counter()
                logger.info(f"Cycle {i_cycle:} Failed | {t_end - t_start:0.2f}s")
                await asyncio.sleep(60)

    async def stop(self):
        """Stop the application and clean up resources."""
        logger.info("Stopping application...")
        self.running = False
        await self.trading_client.close()


async def main():
    """The main entry point for the application."""
    # Setup logger
    logger.remove()
    console = Console()
    logger.add(lambda message: console.print(message, end=""))
    logger.add("logs/plutus_{time}.log", enqueue=True)

    app = Plutus()
    try:
        await app.start()
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        await app.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
