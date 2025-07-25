import asyncio
import signal
import sys
import time
from itertools import count

from loguru import logger

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
        # self.notifier = PushNotifier(self.settings.notification_config)
        self.dash_app = None
        self.running = False

    async def start(self):
        """Start the trading application"""
        logger.info("Starting Plutus...")

        # Initialize data manager
        await self.data_manager.initialise(self.settings.trading_config.pairs)

        # Load and start bots
        self.agent_manager.load_agents()

        # Start trading loop
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

                # Update market data
                await self.data_manager.update_live_data()

                # Run active agents
                await self.agent_manager.run_agents()

                # Update portfolio
                await self.portfolio_manager.update_portfolio()

                t_end = time.perf_counter()
                logger.info(f"Cycle {i_cycle:} Complete | {t_end - t_start:0.2f}s")
                # Sleep until next iteration
                await asyncio.sleep(self.settings.update_interval)

            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                t_end = time.perf_counter()
                logger.info(f"Cycle {i_cycle:} Failed | {t_end - t_start:0.2f}s")
                await asyncio.sleep(60)  # Wait before retrying

    def stop(self):
        """Stop the application"""
        logger.info("Stopping application...")
        self.running = False


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received shutdown signal {signum}...")
    sys.exit(0)


if __name__ == "__main__":
    # Setup logger
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level="DEBUG",
        enqueue=True,
    )
    logger.add("logs/plutus_{time}.log", enqueue=True)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run application
    app = Plutus()
    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        app.stop()
