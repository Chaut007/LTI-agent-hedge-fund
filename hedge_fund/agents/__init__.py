"""Phase 0 agents: deterministic, zero-cost, no LLM.

All five implement `hedge_fund.core.agent.Agent`. Phase 1's LLM agents will
implement the same protocol and plug into the same aggregator, which is the
reason this roster can be replaced one agent at a time instead of rewritten.
"""

from hedge_fund.agents.fundamental import QualityAgent, ValueAgent
from hedge_fund.agents.price import LowVolAgent, MomentumAgent, ReversalAgent

#: Every Phase 0 agent, in roster order.
PHASE_0 = (MomentumAgent, ValueAgent, QualityAgent, LowVolAgent, ReversalAgent)


def phase_0_agents() -> list:
    """A fresh instance of each Phase 0 agent, with default parameters."""
    return [agent() for agent in PHASE_0]


__all__ = [
    "LowVolAgent",
    "MomentumAgent",
    "PHASE_0",
    "QualityAgent",
    "ReversalAgent",
    "ValueAgent",
    "phase_0_agents",
]
