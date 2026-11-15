"""Agents -> aggregator -> optimizer -> risk -> target weights.

The decision pipeline, and the boundary the whole design rests on: agents
return Signals, and everything from there to a weight is deterministic
arithmetic you can check by hand. That is the reason a paid LLM agent can be
added later without re-earning trust — it cannot reach the money.

Three properties this layer owes the design:

- **It can hold nothing.** If no name clears `min_score`, or the risk layer
  vetoes, the result is `{}` and the book sits in cash. A strategy structurally
  incapable of saying "no" cannot be risk-managed.
- **It records what each agent cost.** `cost_by_date` accumulates
  `Signal.cost_usd`, which is the number that decides whether Phase 1 is worth
  switching on.
- **It records every risk intervention.** `risk_actions` is the audit trail. A
  limit that never fires is not protecting anything, and a limit that fires
  constantly means the layer above is wrong — both are worth knowing.

`optimizer` and `risk` both default to off, reproducing exactly what step 4
measured. That baseline is what step 5 has to beat, so it is not quietly
replaced: turning the new machinery on is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

from hedge_fund.core.agent import Agent, Context
from hedge_fund.core.signal import Signal
from hedge_fund.portfolio.aggregate import AggregatedScore, WeightedAggregator
from hedge_fund.portfolio.optimizer import Optimizer, TopNOptimizer
from hedge_fund.portfolio.risk import RiskAction, RiskDecision, RiskLimits, RiskManager
from hedge_fund.portfolio.state import PortfolioState

# Defined here rather than imported from hedge_fund.backtest: the engine imports
# this package for PortfolioState, so a dependency back onto backtest would be a
# cycle. It is the same structural type either way.
Weights = Mapping[str, float]

#: Bars used to estimate average daily dollar volume for the liquidity limit.
ADV_BARS = 20


@dataclass
class SignalStrategy:
    """Run a roster of agents each rebalance and size what they rank highest.

    top_n / min_score: used to build the default `TopNOptimizer` when no
                optimizer is supplied. Ignored if one is.
    optimizer:  turns scores into a proposal. None means `TopNOptimizer`, which
                is what step 4 measured.
    risk:       holds the proposal to hard limits, with a veto. None means no
                risk layer at all — the step 4 baseline.
    """

    agents: Sequence[Agent]
    aggregator: WeightedAggregator = field(default_factory=WeightedAggregator)
    top_n: int = 10
    min_score: float = 0.0
    optimizer: Optimizer | None = None
    risk: RiskManager | None = None
    name: str = "signal_strategy"

    # Observability, not state the decision depends on.
    last_scores: dict[str, AggregatedScore] = field(default_factory=dict, repr=False)
    last_signals: list[Signal] = field(default_factory=list, repr=False)
    last_decision: RiskDecision | None = field(default=None, repr=False)
    cost_by_date: dict[date, float] = field(default_factory=dict, repr=False)
    risk_actions: list[tuple[date, RiskAction]] = field(default_factory=list, repr=False)
    _state: PortfolioState | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("SignalStrategy needs at least one agent")
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        names = [a.name for a in self.agents]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            # Two instances of one agent would double its weight without
            # appearing to, since the aggregator keys contributions by name.
            raise ValueError(f"duplicate agent names in roster: {sorted(duplicates)}")
        if self.optimizer is None:
            self.optimizer = TopNOptimizer(top_n=self.top_n, min_score=self.min_score)

    # -- portfolio state -------------------------------------------------------

    def observe(self, state: PortfolioState) -> None:
        """Called by the engine before each decision. Implements PortfolioAware."""
        self._state = state

    @property
    def state(self) -> PortfolioState:
        """The last observed book, or an empty one before the first observation."""
        if self._state is not None:
            return self._state
        return PortfolioState(asof=date.min, nav=1.0, peak_nav=1.0, weights={})

    @property
    def limits(self) -> RiskLimits:
        """The limits in force. Permissive when no risk layer is attached."""
        if self.risk is not None:
            return self.risk.limits
        # No risk layer: bounds wide enough not to shape the proposal, so the
        # optimizer's own top_n decides the book. max_position stays consistent
        # with max_positions or RiskLimits refuses to construct.
        return RiskLimits(
            max_position=1.0,
            min_positions=1,
            max_positions=10_000,
            max_turnover=float("inf"),
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(self.cost_by_date.values())

    # -- the decision ----------------------------------------------------------

    def _adv(self, ctx: Context, tickers: Sequence[str]) -> dict[str, float]:
        """Average daily dollar volume per ticker, from the asof-bounded view.

        Only used when a risk layer is attached and NAV is known, since without
        both the liquidity limit cannot be evaluated, and a guessed capacity is
        worse than no check.
        """
        bars = ctx.view.prices(list(tickers), bars=ADV_BARS)
        if bars.empty:
            return {}
        dollar = bars["close"] * bars["volume"]
        means = dollar.groupby(bars["ticker"]).mean()
        return {
            str(ticker): float(value)
            for ticker, value in means.items()
            if value == value and value > 0.0
        }

    def target_weights(self, ctx: Context) -> Weights:
        signals: list[Signal] = []
        for agent in self.agents:
            signals.extend(agent.evaluate(ctx))

        self.last_signals = signals
        scores = self.aggregator.combine(signals)
        self.last_scores = scores
        self.cost_by_date[ctx.asof] = sum(s.cost_usd for s in signals)

        state = self.state
        limits = self.limits
        plain = {ticker: score.score for ticker, score in scores.items()}
        assert self.optimizer is not None  # set in __post_init__
        proposal = self.optimizer.weights(plain, state, limits)

        if self.risk is None:
            self.last_decision = RiskDecision(weights=dict(proposal))
            return proposal

        adv = self._adv(ctx, ctx.universe) if state.nav > 0.0 else None
        decision = self.risk.review(proposal, state, universe=ctx.universe, adv=adv)
        self.last_decision = decision
        self.risk_actions.extend((ctx.asof, action) for action in decision.actions)
        return decision.weights

    # -- reporting -------------------------------------------------------------

    def explain_last(self, limit: int = 10) -> str:
        """Why the current book looks the way it does. For `hf backtest --explain`."""
        if not self.last_scores:
            return "no decision recorded yet"
        ranked = sorted(self.last_scores.values(), key=lambda s: (-s.score, s.ticker))
        lines = [f"{len(self.last_signals)} signals -> {len(ranked)} scored tickers"]
        lines += [f"  {s.explain()}" for s in ranked[:limit]]
        if len(ranked) > limit:
            lines.append(f"  ... {len(ranked) - limit} more")
        if self.last_decision is not None and self.last_decision.actions:
            lines.append(self.last_decision.describe())
        return "\n".join(lines)

    def risk_summary(self) -> str:
        """How often each limit actually fired over the whole run."""
        if not self.risk_actions:
            return "risk: no limit fired" if self.risk else "risk: no risk layer attached"
        counts: dict[str, int] = {}
        for _, action in self.risk_actions:
            counts[action.breach.value] = counts.get(action.breach.value, 0) + 1
        rows = sorted(counts.items(), key=lambda kv: -kv[1])
        return "\n".join(
            [f"risk interventions ({len(self.risk_actions)} total):"]
            + [f"  {name:22} {count}" for name, count in rows]
        )
