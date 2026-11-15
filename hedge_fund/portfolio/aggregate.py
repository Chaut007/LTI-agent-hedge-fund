"""Combining many agents' Signals into one score per ticker.

This is where the multi-agent design either earns its keep or quietly cancels
itself out, so the arithmetic is deliberately plain and the known flaw is
stated rather than buried.

**Each agent contributes `score x confidence x weight`**, and the result is
divided by the weight actually present. An agent that could not evaluate a
ticker does not dilute the agents that could — a company with no filings is
ranked on its price factors alone, not penalised to zero for the fundamentals
it never reported.

**The known flaw: horizons are averaged.** MomentumAgent says "cheap over the
next six months" and ReversalAgent says "bouncing over the next week", and
equal weights average them into a number that means neither. They are
genuinely opposed — that is what the factors say — and the honest v1 answer is
to run the blend you can defend and measure it, not to invent a term that
makes the conflict disappear. Two ways out are already open:

1. Pass explicit `weights` and drop the agent whose horizon you do not want.
   `MEDIUM_TERM` below does exactly that.
2. Let the track-record store choose the weights from out-of-sample
   performance, which is the whole point of recording them. That is step 6,
   and it needs live results that do not exist yet.

Until then, `EQUAL` is a baseline to beat, not a recommendation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from hedge_fund.core.signal import Signal

# Equal weight on all five Phase 0 agents. Averages a 5-day and a 252-day view;
# see the module docstring.
EQUAL: dict[str, float] = {
    "momentum": 1.0,
    "value": 1.0,
    "quality": 1.0,
    "low_vol": 1.0,
    "reversal": 1.0,
}

# Horizon-consistent: everything here speaks to months, not days. Reversal is
# excluded rather than down-weighted, because the problem with it is its
# horizon, not its size.
MEDIUM_TERM: dict[str, float] = {
    "momentum": 1.0,
    "value": 1.0,
    "quality": 1.0,
    "low_vol": 1.0,
}


@dataclass(frozen=True, slots=True)
class AggregatedScore:
    ticker: str
    score: float
    contributions: dict[str, float]
    """agent -> its weighted contribution to `score`. The audit trail."""
    agents: int
    cost_usd: float

    def explain(self) -> str:
        parts = ", ".join(
            f"{agent} {value:+.3f}"
            for agent, value in sorted(
                self.contributions.items(), key=lambda kv: -abs(kv[1])
            )
        )
        return f"{self.ticker} {self.score:+.3f} ({parts})"


@dataclass(frozen=True)
class WeightedAggregator:
    """Confidence-weighted average of agent scores.

    `weights` keys are agent names. An agent emitting Signals while absent
    from `weights` is an error, not a silent zero: it means the roster and the
    weighting disagree, and the most likely cause is a typo in an agent name
    that would otherwise remove a factor from the strategy without saying so.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(EQUAL))
    min_agents: int = 1

    def __post_init__(self) -> None:
        for agent, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"weight for {agent!r} is negative: {weight}")
        if not any(w > 0 for w in self.weights.values()):
            raise ValueError("at least one agent weight must be positive")

    def combine(self, signals: Iterable[Signal]) -> dict[str, AggregatedScore]:
        numerator: dict[str, float] = defaultdict(float)
        denominator: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        counts: dict[str, int] = defaultdict(int)
        costs: dict[str, float] = defaultdict(float)

        for signal in signals:
            if signal.agent not in self.weights:
                raise KeyError(
                    f"agent {signal.agent!r} emitted a Signal but has no weight. "
                    f"Known: {sorted(self.weights)}. Add it to `weights`, or stop "
                    "running it — a missing weight would drop the factor silently."
                )
            weight = self.weights[signal.agent]
            costs[signal.ticker] += signal.cost_usd
            if weight == 0.0:
                continue
            effective = weight * signal.confidence
            numerator[signal.ticker] += effective * signal.score
            denominator[signal.ticker] += effective
            contributions[signal.ticker][signal.agent] = effective * signal.score
            counts[signal.ticker] += 1

        out: dict[str, AggregatedScore] = {}
        for ticker, total in denominator.items():
            if total <= 0.0 or counts[ticker] < self.min_agents:
                continue
            out[ticker] = AggregatedScore(
                ticker=ticker,
                score=numerator[ticker] / total,
                contributions=dict(contributions[ticker]),
                agents=counts[ticker],
                cost_usd=costs[ticker],
            )
        return out
