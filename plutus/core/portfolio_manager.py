from loguru import logger

from plutus.trading_clients.trading_client import TradingClient


class PortfolioManager:
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.balance = {}
        self.open_positions = {}
        self.trade_history = []

    async def update_portfolio(self):
        """Update the portfolio with the latest account information."""
        logger.info("Updating portfolio...")
        try:
            self.balance = await self.trading_client.get_account_balance()
            # In a real application, you would also update open positions and trade history
            logger.info(f"Portfolio updated. Current balance: {self.balance}")
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
