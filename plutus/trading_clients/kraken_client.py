import base64
import hashlib
import hmac
import time
from urllib.parse import urlencode

import aiohttp
from async_lru import alru_cache
from loguru import logger

from plutus.config.settings import Settings
from plutus.trading_clients.trading_client import TradingClient

SETTINGS = Settings()
TTL: float = SETTINGS.update_interval


class KrakenClient(TradingClient):
    def __init__(self, api_key: str = "", api_secret: str = "", sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.kraken.com"
        self.sandbox = sandbox
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, endpoint: str, data: dict) -> str:
        """Generate API signature for authenticated requests"""
        postdata = urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        try:
            signature = hmac.new(
                base64.b64decode(self.api_secret), message, hashlib.sha512
            )
            return base64.b64encode(signature.digest()).decode()
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            raise Exception("Invalid API secret format")

    async def _request(
        self, endpoint: str, data: dict = None, private: bool = False
    ) -> dict:
        """Make API request"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}{endpoint}"
        headers = {"User-Agent": "Plutus Trading Bot"}

        if private:
            if not self.api_key or not self.api_secret:
                raise Exception("API key and secret required for private endpoints")
            if data is None:
                data = {}
            data["nonce"] = str(int(time.time() * 1000000))
            headers["API-Key"] = self.api_key
            headers["API-Sign"] = self._generate_signature(endpoint, data)

        try:
            # Private endpoints and public endpoints with data use POST.
            # Public endpoints without data use GET.
            if private or data:
                async with self.session.post(
                    url, data=data, headers=headers
                ) as response:
                    result = await response.json()
            else:
                async with self.session.get(url, headers=headers) as response:
                    result = await response.json()

            if result.get("error") and result["error"]:
                error_msg = (
                    ", ".join(result["error"])
                    if isinstance(result["error"], list)
                    else result["error"]
                )
                raise Exception(f"Kraken API Error: {error_msg}")

            return result.get("result", {})
        except aiohttp.ContentTypeError:
            raise Exception("Invalid response from Kraken API - check endpoint URL")
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            raise

    # Public API methods
    @alru_cache(ttl=TTL)
    async def get_server_time(self) -> dict:
        return await self._request("/0/public/Time")

    @alru_cache(ttl=TTL)
    async def get_asset_info(self, assets: tuple[str] = None) -> dict:
        data = {}
        if assets:
            data["asset"] = ",".join(assets)
        return await self._request("/0/public/Assets", data)

    @alru_cache(ttl=TTL)
    async def get_tradable_pairs(self, pairs: tuple[str] = None) -> dict:
        data = {}
        if pairs:
            data["pair"] = ",".join(pairs)
        return await self._request("/0/public/AssetPairs", data)

    @alru_cache(ttl=TTL)
    async def get_ticker(self, pairs: tuple[str]) -> dict:
        data = {"pair": ",".join(pairs)}
        return await self._request("/0/public/Ticker", data)

    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> dict:
        """Get OHLC data. Note: The cache is removed to allow for pagination."""
        data = {"pair": pair, "interval": interval}
        if since:
            data["since"] = since
        return await self._request("/0/public/OHLC", data)

    @alru_cache(ttl=TTL)
    async def get_order_book(self, pair: str, count: int = 100) -> dict:
        data = {"pair": pair, "count": count}
        return await self._request("/0/public/Depth", data)

    @alru_cache(ttl=TTL)
    async def get_recent_trades(self, pair: str, since: int = None) -> dict:
        data = {"pair": pair}
        if since:
            data["since"] = since
        return await self._request("/0/public/Trades", data)

    # Private API methods
    async def get_account_balance(self) -> dict:
        return await self._request("/0/private/Balance", {}, private=True)

    async def get_trade_balance(self, asset: str = "ZUSD") -> dict:
        data = {"asset": asset}
        return await self._request("/0/private/TradeBalance", data, private=True)

    async def get_open_orders(self) -> dict:
        return await self._request("/0/private/OpenOrders", {}, private=True)

    async def get_closed_orders(self, start: int = None, end: int = None) -> dict:
        data = {}
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        return await self._request("/0/private/ClosedOrders", data, private=True)

    async def place_order(
        self,
        pair: str,
        type_: str,
        ordertype: str,
        volume: float,
        price: float = None,
        **kwargs,
    ) -> dict:
        if self.sandbox:
            logger.warning("SANDBOX MODE: Order would be placed but not executed")
            return {
                "txid": ["SANDBOX_ORDER_" + str(int(time.time()))],
                "descr": {"order": f"{type_} {volume} {pair} @ {ordertype}"},
            }
        data = {
            "pair": pair,
            "type": type_,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        if price:
            data["price"] = str(price)
        data.update(kwargs)
        return await self._request("/0/private/AddOrder", data, private=True)

    async def cancel_order(self, txid: str) -> dict:
        if self.sandbox:
            logger.warning("SANDBOX MODE: Order cancellation simulated")
            return {"count": 1}
        data = {"txid": txid}
        return await self._request("/0/private/CancelOrder", data, private=True)
