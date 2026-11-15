"""Core contracts: Signal, Agent, Context. No storage or network deps."""

from hedge_fund.core.agent import Agent, BaseAgent, Context, MarketView
from hedge_fund.core.signal import Signal

__all__ = ["Agent", "BaseAgent", "Context", "MarketView", "Signal"]
