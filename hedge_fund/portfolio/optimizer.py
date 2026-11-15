"""Scores to target weights. Deterministic arithmetic, checkable by hand.

Two optimizers, and the difference between them is the measured finding of
step 4.

`TopNOptimizer` is what step 4 ran: take the best N names, hold them equally,
re-decide from scratch every rebalance. On the 474-name universe that traded
3.28% of the book per day and paid 14.9% of NAV in costs, against equal
weight's 0.16% and 0.7%. Gross of costs the strategy matched equal weight;
net of them it lost. The signal was not the problem — the trading was.

`BandedOptimizer` is the response. It changes nothing about which names are
chosen, only about when a change is worth making:

- **A no-trade band.** A name whose target has moved less than `band` from what
  is already held is left alone. Cross-sectional ranks jitter every month
  without saying anything new, and acting on that jitter is pure cost.
- **A turnover budget** applied before the risk layer's, so the optimizer
  chooses what to give up rather than having it scaled off the end.

Neither is a prediction. Both are statements about cost, which is the one part
of a backtest that is not guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from hedge_fund.portfolio.risk import RiskLimits
from hedge_fund.portfolio.state import PortfolioState


@runtime_checkable
class Optimizer(Protocol):
    name: str

    def weights(
        self,
        scores: Mapping[str, float],
        state: PortfolioState,
        limits: RiskLimits,
    ) -> dict[str, float]:
        """Target weights from aggregated scores. May return `{}` for all cash."""
        ...


def _pick(scores: Mapping[str, float], count: int, min_score: float) -> list[str]:
    """The `count` best names above `min_score`, ties broken by ticker.

    Deterministic ordering matters: without the ticker tiebreak the same inputs
    give different books across runs, and a permutation test on a schedule that
    is not reproducible measures nothing.
    """
    eligible = [(t, s) for t, s in scores.items() if s > min_score]
    eligible.sort(key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in eligible[:count]]


@dataclass(frozen=True)
class TopNOptimizer:
    """Equal weight across the best N names. What step 4 measured.

    Kept as the baseline the banded version has to beat, not as a
    recommendation — on the measured runs it is the version that loses.
    """

    top_n: int = 10
    min_score: float = 0.0
    name: str = "top_n"

    def weights(
        self,
        scores: Mapping[str, float],
        state: PortfolioState,
        limits: RiskLimits,
    ) -> dict[str, float]:
        count = min(self.top_n, limits.max_positions)
        held = _pick(scores, count, self.min_score)
        if not held:
            return {}
        weight = min(limits.max_gross / len(held), limits.max_position)
        return {ticker: weight for ticker in held}


@dataclass(frozen=True)
class BandedOptimizer:
    """Top N, but only trade what is worth trading.

    band:          a target within this distance of the current weight is left
                   at the current weight. 0.0 reproduces TopNOptimizer.
    turnover_cap:  one-way turnover the optimizer will propose. None defers
                   entirely to the risk layer's `max_turnover`.
    keep_ranked:   a held name still inside the top `top_n * keep_ranked` is not
                   sold just because it slipped out of the top N. Rank jitter at
                   the boundary is the single largest source of turnover in a
                   cross-sectional strategy, and this is what blunts it.
    """

    top_n: int = 10
    min_score: float = 0.0
    band: float = 0.01
    turnover_cap: float | None = None
    keep_ranked: float = 1.5
    name: str = "banded"

    def __post_init__(self) -> None:
        if self.band < 0.0:
            raise ValueError(f"band must be >= 0, got {self.band}")
        if self.keep_ranked < 1.0:
            raise ValueError(
                f"keep_ranked must be >= 1 (1.0 means no hysteresis), got {self.keep_ranked}"
            )
        if self.turnover_cap is not None and self.turnover_cap <= 0.0:
            raise ValueError("turnover_cap must be positive, or None to defer to risk")

    def weights(
        self,
        scores: Mapping[str, float],
        state: PortfolioState,
        limits: RiskLimits,
    ) -> dict[str, float]:
        count = min(self.top_n, limits.max_positions)
        wanted = _pick(scores, count, self.min_score)

        # Hysteresis: a name already held keeps its place while it stays inside
        # the wider ring, so a one-rank slip does not force a round trip.
        keep_count = min(int(round(count * self.keep_ranked)), limits.max_positions)
        tolerated = set(_pick(scores, keep_count, self.min_score))
        chosen = list(wanted)
        for ticker in state.weights:
            if state.weight(ticker) > 0.0 and ticker in tolerated and ticker not in chosen:
                chosen.append(ticker)
        if not chosen:
            return {}
        if len(chosen) > limits.max_positions:
            ranked = sorted(chosen, key=lambda t: (-scores.get(t, float("-inf")), t))
            chosen = ranked[: limits.max_positions]

        ideal = min(limits.max_gross / len(chosen), limits.max_position)
        target = {ticker: ideal for ticker in chosen}

        # No-trade band: keep the current weight where the move is too small to
        # be worth its spread. Applied before normalising, so the saved trades
        # are real rather than undone by the rescale below.
        banded: dict[str, float] = {}
        for ticker in set(target) | set(state.weights):
            want = target.get(ticker, 0.0)
            have = state.weight(ticker)
            if have <= 0.0 and want <= 0.0:
                continue
            # Exits are never banded: a name that left the selection is sold.
            if want == 0.0:
                continue
            banded[ticker] = have if abs(want - have) < self.band and have > 0.0 else want

        if not banded:
            return {}

        gross = sum(banded.values())
        if gross > limits.max_gross:
            scale = limits.max_gross / gross
            banded = {t: w * scale for t, w in banded.items()}

        cap = self.turnover_cap if self.turnover_cap is not None else limits.max_turnover
        turnover = state.turnover_to(banded)
        if turnover > cap:
            lam = cap / turnover
            partial = {
                t: state.weight(t) + lam * (banded.get(t, 0.0) - state.weight(t))
                for t in set(banded) | set(state.weights)
            }
            banded = {t: w for t, w in partial.items() if w > 1e-9}
            # The blend reintroduces everything currently held, so it can carry
            # the book past max_positions. Keep the largest, which are the names
            # the blend moved furthest toward — the risk layer would trim these
            # anyway, and doing it here means the proposal is already legal.
            if len(banded) > limits.max_positions:
                ranked = sorted(banded.items(), key=lambda kv: (-kv[1], kv[0]))
                banded = dict(ranked[: limits.max_positions])

        return banded
