import aiohttp
from loguru import logger

from plutus.config.settings import NotificationConfig


class PushNotifier:
    def __init__(self, config: NotificationConfig):
        self.config = config

    async def send_notification(self, message: str):
        """Send a notification to all configured services."""
        if self.config.pushover_token and self.config.pushover_user:
            await self.send_pushover(message)
        if self.config.telegram_token and self.config.telegram_chat_id:
            await self.send_telegram(message)
        if self.config.discord_webhook:
            await self.send_discord(message)

    async def send_pushover(self, message: str):
        """Send a notification via Pushover."""
        url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": self.config.pushover_token,
            "user": self.config.pushover_user,
            "message": message,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    logger.info("Pushover notification sent.")
                else:
                    logger.error("Failed to send Pushover notification.")

    async def send_telegram(self, message: str):
        """Send a notification via Telegram."""
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        data = {"chat_id": self.config.telegram_chat_id, "text": message}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    logger.info("Telegram notification sent.")
                else:
                    logger.error("Failed to send Telegram notification.")

    async def send_discord(self, message: str):
        """Send a notification via Discord."""
        data = {"content": message}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.config.discord_webhook, json=data) as response:
                if response.status == 204:
                    logger.info("Discord notification sent.")
                else:
                    logger.error("Failed to send Discord notification.")
