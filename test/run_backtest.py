import asyncio
from rich.console import Console
from loguru import logger

from plutus.config.settings import Settings
from plutus.core.agent_manager import AgentManager
from plutus.core.data_manager import DataManager
from backtest import BacktestEngine, SimulatedTradingClient
from plutus.trading_clients.kraken_client import KrakenClient

# Import the agent classes we want to test
from plutus.agents.momentum import MomentumBot
from plutus.agents.trend_following import TrendFollowingBot
from plutus.agents.mean_reversion import MeanReversionBot


def load_agents_for_backtest(settings: Settings) -> list:
    """
    Manually creates agent instances for backtesting, giving each its own
    SimulatedTradingClient to ensure isolated state.
    """
    agents = []
    agent_id_counter = 1

    # --- Load Momentum Bots ---
    rsi_periods = [14, 21]
    ma_fast_hours_options = [10, 20]
    ma_slow_hours_options = [30, 50]

    for rsi in rsi_periods:
        for fast_ma in ma_fast_hours_options:
            for slow_ma in ma_slow_hours_options:
                if fast_ma >= slow_ma:
                    continue
                config = {
                    "pairs": ["XBTUSD", "ETHUSD"],
                    "rsi_period": rsi,
                    "ma_fast_hours": fast_ma,
                    "ma_slow_hours": slow_ma,
                    "min_confidence": 0.6,
                }
                client = SimulatedTradingClient(initial_balance=10000.0)
                agents.append(
                    MomentumBot(
                        name=f"Momentum-RSI{rsi}-MA{fast_ma}/{slow_ma}",
                        config=config,
                        trading_client=client,
                    )
                )
                agent_id_counter += 1

    # --- Load Trend Following Bots ---
    macd_settings = [(12, 26, 9), (20, 50, 10)]
    for fast, slow, sig in macd_settings:
        config = {
            "pairs": ["XBTUSD"],
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": sig,
            "min_confidence": 0.7,
        }
        client = SimulatedTradingClient(initial_balance=10000.0)
        agents.append(
            TrendFollowingBot(
                name=f"TrendFollow-MACD{fast}/{slow}/{sig}",
                config=config,
                trading_client=client,
            )
        )
        agent_id_counter += 1

    # --- Load Mean Reversion Bots ---
    bollinger_settings = [(20, 2), (30, 2.5)]
    for window, std in bollinger_settings:
        config = {
            "pairs": ["ETHUSD"],
            "window": window,
            "std_dev": std,
            "min_confidence": 0.75,
        }
        client = SimulatedTradingClient(initial_balance=10000.0)
        agents.append(
            MeanReversionBot(
                name=f"MeanRevert-BB{window}/{std}",
                config=config,
                trading_client=client,
            )
        )
        agent_id_counter += 1

    logger.info(f"Loaded {len(agents)} agents for backtest.")
    return agents


async def main():
    # Setup logger
    logger.remove()
    console = Console()
    logger.add(lambda message: console.print(message, end=""), level="INFO")
    logger.add("logs/plutus_backtest_{time}.log", enqueue=True)

    settings = Settings()

    async with KrakenClient(
        api_key=settings.kraken_api_key,
        api_secret=settings.kraken_api_secret,
    ) as real_client:
        data_manager = DataManager(real_client, backtesting=True)

        # We create a dummy AgentManager. Its role in the backtest is just to hold the agents.
        # We pass a dummy client because the agents will get their own dedicated clients.
        agent_manager = AgentManager(
            settings.trading_config,
            SimulatedTradingClient(),  # This client is just a placeholder and won't be used.
            data_manager,
        )

        # Manually create and load agents, each with its own isolated client instance.
        agent_manager.agents = {
            i: agent for i, agent in enumerate(load_agents_for_backtest(settings))
        }

        start_date = "2025-01-01"
        end_date = "2025-07-24"

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
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Backtest failed with an error: {e}")
