"""Aggregation and sizing: the deterministic half of the decision.

The invariant worth defending here is that no agent can size a position. An
agent's reach ends at a Signal; everything that turns Signals into weights is
arithmetic in this layer, checkable by hand, and tested as such.
"""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.agents import phase_0_agents
from hedge_fund.agents.price import LowVolAgent, MomentumAgent
from hedge_fund.backtest import Backtester, BacktestConfig
from hedge_fund.backtest.costs import ZERO_COST
from hedge_fund.core.signal import Signal
from hedge_fund.portfolio import (
    EQUAL,
    MEDIUM_TERM,
    SignalStrategy,
    WeightedAggregator,
)

ASOF = date(2024, 6, 3)


def sig(agent: str, ticker: str, score: float, confidence: float = 1.0, cost: float = 0.0) -> Signal:
    return Signal(
        agent=agent,
        ticker=ticker,
        asof=ASOF,
        score=score,
        confidence=confidence,
        horizon_days=21,
        cost_usd=cost,
    )


# -- aggregation ------------------------------------------------------------------


def test_equal_weights_average_scores() -> None:
    out = WeightedAggregator(dict(EQUAL)).combine(
        [sig("momentum", "A", 1.0), sig("value", "A", 0.0)]
    )
    assert out["A"].score == pytest.approx(0.5)
    assert out["A"].agents == 2


def test_confidence_scales_a_contribution() -> None:
    """A half-confident +1 counts half as much, both ways."""
    out = WeightedAggregator(dict(EQUAL)).combine(
        [sig("momentum", "A", 1.0, confidence=0.5), sig("value", "A", -1.0)]
    )
    # (0.5*1 + 1*-1) / (0.5 + 1) = -1/3
    assert out["A"].score == pytest.approx(-1 / 3)


def test_a_missing_agent_does_not_dilute_the_present_ones() -> None:
    """A company with no filings is judged on price factors, not penalised."""
    out = WeightedAggregator(dict(EQUAL)).combine([sig("momentum", "A", 1.0)])
    assert out["A"].score == pytest.approx(1.0)
    assert out["A"].agents == 1


def test_explicit_weights_are_honoured() -> None:
    out = WeightedAggregator({"momentum": 3.0, "value": 1.0}).combine(
        [sig("momentum", "A", 1.0), sig("value", "A", -1.0)]
    )
    assert out["A"].score == pytest.approx(0.5)  # (3 - 1) / 4


def test_zero_weight_excludes_an_agent_but_still_counts_its_cost() -> None:
    out = WeightedAggregator({"momentum": 1.0, "news": 0.0}).combine(
        [sig("momentum", "A", 1.0), sig("news", "A", -1.0, cost=0.25)]
    )
    assert out["A"].score == pytest.approx(1.0)
    assert out["A"].cost_usd == pytest.approx(0.25), "a disabled agent still billed"


def test_unknown_agent_raises_rather_than_being_dropped() -> None:
    """A typo'd agent name would otherwise remove a factor in silence."""
    with pytest.raises(KeyError, match="no weight"):
        WeightedAggregator({"momentum": 1.0}).combine([sig("momentim", "A", 1.0)])


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        WeightedAggregator({"momentum": -1.0})


def test_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        WeightedAggregator({"momentum": 0.0})


def test_min_agents_filters_thinly_covered_tickers() -> None:
    out = WeightedAggregator(dict(EQUAL), min_agents=2).combine(
        [sig("momentum", "A", 1.0), sig("momentum", "B", 1.0), sig("value", "B", 1.0)]
    )
    assert set(out) == {"B"}


def test_contributions_sum_to_the_numerator() -> None:
    """The audit trail must reconcile with the score it explains."""
    signals = [sig("momentum", "A", 0.8, 0.5), sig("value", "A", -0.4), sig("quality", "A", 0.2)]
    out = WeightedAggregator(dict(EQUAL)).combine(signals)["A"]
    denominator = 0.5 + 1.0 + 1.0
    assert sum(out.contributions.values()) / denominator == pytest.approx(out.score)
    assert "momentum" in out.explain()


def test_costs_accumulate_across_agents() -> None:
    out = WeightedAggregator(dict(EQUAL)).combine(
        [sig("momentum", "A", 1.0, cost=0.1), sig("value", "A", 1.0, cost=0.2)]
    )
    assert out["A"].cost_usd == pytest.approx(0.3)


def test_medium_term_preset_excludes_reversal() -> None:
    assert "reversal" not in MEDIUM_TERM
    assert set(MEDIUM_TERM) < set(EQUAL)


# -- sizing -------------------------------------------------------------------------


class _Fixed:
    """An agent that emits preset scores, so sizing can be tested alone."""

    def __init__(self, name: str, scores: dict[str, float], cost: float = 0.0) -> None:
        self.name = name
        self._scores = scores
        self._cost = cost

    def evaluate(self, ctx) -> list[Signal]:
        return [
            Signal(
                agent=self.name, ticker=t, asof=ctx.asof, score=s,
                confidence=1.0, horizon_days=21, cost_usd=self._cost,
            )
            for t, s in self._scores.items()
        ]


def _strategy(scores: dict[str, float], **kwargs) -> SignalStrategy:
    return SignalStrategy(
        agents=[_Fixed("momentum", scores)],
        aggregator=WeightedAggregator({"momentum": 1.0}),
        **kwargs,
    )


def test_top_n_are_held_equally(synthetic) -> None:
    strategy = _strategy({"T00": 0.9, "T01": 0.8, "T02": 0.7, "T03": 0.6}, top_n=2)
    weights = strategy.target_weights(synthetic.context())
    assert weights == {"T00": 0.5, "T01": 0.5}


def test_gross_exposure_is_exactly_one(synthetic) -> None:
    strategy = _strategy({f"T{i:02d}": 1.0 - i / 20 for i in range(12)}, top_n=7)
    weights = strategy.target_weights(synthetic.context())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(weights) == 7


def test_min_score_can_empty_the_book(synthetic) -> None:
    """A strategy that cannot say "nothing" cannot be risk-managed."""
    strategy = _strategy({"T00": -0.5, "T01": -0.9}, min_score=0.0)
    assert strategy.target_weights(synthetic.context()) == {}


def test_min_score_filters_below_threshold(synthetic) -> None:
    strategy = _strategy({"T00": 0.5, "T01": 0.1, "T02": -0.2}, top_n=10, min_score=0.3)
    assert set(strategy.target_weights(synthetic.context())) == {"T00"}


def test_fewer_eligible_than_top_n_still_sums_to_one(synthetic) -> None:
    strategy = _strategy({"T00": 0.5, "T01": 0.4}, top_n=10)
    weights = strategy.target_weights(synthetic.context())
    assert len(weights) == 2
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ties_break_on_ticker_for_reproducibility(synthetic) -> None:
    strategy = _strategy({"T05": 0.5, "T01": 0.5, "T09": 0.5}, top_n=2)
    assert list(strategy.target_weights(synthetic.context())) == ["T01", "T05"]


def test_agent_costs_are_recorded_per_date(synthetic) -> None:
    strategy = SignalStrategy(
        agents=[_Fixed("news", {"T00": 1.0, "T01": 0.5}, cost=0.02)],
        aggregator=WeightedAggregator({"news": 1.0}),
    )
    strategy.target_weights(synthetic.context())
    assert strategy.cost_by_date[synthetic.asof] == pytest.approx(0.04)
    assert strategy.total_cost_usd == pytest.approx(0.04)


def test_duplicate_agent_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SignalStrategy(agents=[MomentumAgent(), MomentumAgent()])


def test_empty_roster_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        SignalStrategy(agents=[])


def test_top_n_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_n"):
        SignalStrategy(agents=[MomentumAgent()], top_n=0)


def test_explain_last_names_contributions(synthetic) -> None:
    strategy = SignalStrategy(
        agents=[MomentumAgent(), LowVolAgent()],
        aggregator=WeightedAggregator({"momentum": 1.0, "low_vol": 1.0}),
        top_n=3,
    )
    strategy.target_weights(synthetic.context())
    text = strategy.explain_last(limit=3)
    assert "momentum" in text and "low_vol" in text
    assert "signals ->" in text


def test_explain_before_any_decision() -> None:
    assert "no decision" in SignalStrategy(agents=[MomentumAgent()]).explain_last()


# -- end to end through the backtester ----------------------------------------------


def test_phase0_runs_through_the_backtester(synthetic) -> None:
    strategy = SignalStrategy(
        agents=phase_0_agents(),
        aggregator=WeightedAggregator(dict(EQUAL)),
        top_n=4,
    )
    config = BacktestConfig(
        start=synthetic.trading_days[300].date(),
        end=synthetic.asof,
        rebalance="weekly",
        costs=ZERO_COST,
    )
    result = Backtester(synthetic.lake, synthetic.tickers).run(strategy, config)

    assert result.metrics.days > 0
    assert result.decisions, "the strategy was consulted"
    held = result.weights.iloc[-1]
    assert held[held > 0].count() <= 4
    assert result.weights.sum(axis=1).max() == pytest.approx(1.0, abs=1e-9)
    assert strategy.total_cost_usd == 0.0, "Phase 0 must be free"


def test_phase0_survives_a_universe_with_no_filings(prices_only) -> None:
    """Value and Quality contribute nothing; the price agents carry the book."""
    strategy = SignalStrategy(
        agents=phase_0_agents(),
        aggregator=WeightedAggregator(dict(EQUAL)),
        top_n=3,
    )
    config = BacktestConfig(
        start=prices_only.trading_days[300].date(),
        end=prices_only.asof,
        rebalance="weekly",
        costs=ZERO_COST,
    )
    result = Backtester(prices_only.lake, prices_only.tickers).run(strategy, config)
    assert result.metrics.days > 0
    agents_used = {s.agent for s in strategy.last_signals}
    assert agents_used == {"momentum", "low_vol", "reversal"}
