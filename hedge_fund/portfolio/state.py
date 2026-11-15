"""What the book looks like right now, for the layer that is allowed to know.

Agents must not see portfolio state. A momentum agent that knows it already
holds a name has been handed a reason to say something different about it, and
the whole point of the Signal contract is that an agent's opinion does not
depend on the book. So `Context` stays as it is: asof, view, universe.

But turnover control and a drawdown stop are impossible without the book, and
both belong in step 5. Hence a separate channel: a strategy that wants
portfolio state declares `observe`, and the engine calls it before asking for
targets. Opt-in, so nothing that does not need state is given it, and nothing
in `core` changes.

In live trading this is what the broker reports. In a backtest the engine
supplies the same shape, which is the point — the optimizer cannot tell the
difference, so it cannot be right in one and wrong in the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The book as of a close, after any fill that day.

    weights:  fraction of NAV per ticker, drifted with prices since the last
              rebalance. Omitted tickers are zero.
    nav:      net asset value in the engine's units (1.0 at inception).
    peak_nav: the highest NAV seen so far, which is what a drawdown is measured
              against.
    halted:   the kill switch has fired and the book is flat pending a manual
              restart. A strategy that is told this and still returns weights
              is overruled by the engine, not obeyed.
    """

    asof: date
    nav: float
    peak_nav: float
    weights: Mapping[str, float] = field(default_factory=dict)
    halted: bool = False

    @property
    def drawdown(self) -> float:
        """Fall from peak as a positive fraction. 0.0 at a new high."""
        if self.peak_nav <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.nav / self.peak_nav)

    @property
    def gross(self) -> float:
        return float(sum(abs(w) for w in self.weights.values()))

    def weight(self, ticker: str) -> float:
        return float(self.weights.get(ticker, 0.0))

    def turnover_to(self, targets: Mapping[str, float]) -> float:
        """One-way turnover to reach `targets`: (purchases + sales) / 2.

        The same convention the engine charges costs on, so a turnover budget
        here and a cost there mean the same thing.
        """
        names = set(self.weights) | set(targets)
        return sum(abs(float(targets.get(t, 0.0)) - self.weight(t)) for t in names) / 2.0


@runtime_checkable
class PortfolioAware(Protocol):
    """A strategy that needs to know the book. Implemented by SignalStrategy."""

    def observe(self, state: PortfolioState) -> None: ...
