"""What the backtester drives: anything that turns a Context into target weights.

The engine never sees Signals. Agents, aggregation, risk and sizing all live
behind `Strategy.target_weights`; the engine's only contract with a strategy
is "here is a Context pinned to today, give me fractions of NAV". That keeps
the simulator agnostic to how the portfolio was decided, and it is why the
same engine can replay recorded decisions for a permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Protocol, runtime_checkable

from hedge_fund.core.agent import Context

Weights = Mapping[str, float]


@runtime_checkable
class Strategy(Protocol):
    name: str

    def target_weights(self, ctx: Context) -> Weights:
        """Fractions of NAV per ticker, decided with what `ctx` lets you see.

        Omitted tickers are zero. `{}` means all cash. Weights are validated by
        the engine against its long-only / gross-exposure limits.
        """
        ...


@dataclass(frozen=True)
class EqualWeight:
    """1/N across every universe ticker that has a bar on or before `asof`.

    Reads the view rather than trusting the universe list, so a ticker that has
    not started trading yet is left out instead of being assigned weight the
    engine would have to park in cash. This is the baseline every real
    strategy must beat, and the simplest thing that exercises the asof path.
    """

    name: str = "equal_weight"

    def target_weights(self, ctx: Context) -> Weights:
        bars = ctx.view.prices(list(ctx.universe), bars=1)
        live = sorted(bars["ticker"].unique()) if not bars.empty else []
        if not live:
            return {}
        weight = 1.0 / len(live)
        return {ticker: weight for ticker in live}


@dataclass(frozen=True)
class ReplayStrategy:
    """Replays a recorded decision schedule, ignoring the Context.

    Used by the permutation test to re-run the engine over shuffled decisions
    without re-invoking the strategy that produced them.

    A date with no recorded decision is an error, not "all cash": the schedule
    was produced under one rebalance/lag configuration and replaying it under
    another would silently liquidate the book between the recorded dates.
    """

    decisions: Mapping[date, Weights]
    name: str = "replay"

    def target_weights(self, ctx: Context) -> Weights:
        try:
            return dict(self.decisions[ctx.asof])
        except KeyError:
            raise ValueError(
                f"replay has no decision for {ctx.asof}. Replay with the same "
                "rebalance and execution_lag that produced the schedule."
            ) from None


@dataclass
class RecordingStrategy:
    """Wraps a strategy and keeps every (asof, max date it could see) pair.

    Test helper: proves the engine never handed a strategy a view that could
    reach past its own asof.
    """

    inner: Strategy
    seen: list[tuple[date, date | None]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"recording({self.inner.name})"

    def target_weights(self, ctx: Context) -> Weights:
        bars = ctx.view.prices(list(ctx.universe), bars=10_000)
        newest = bars["date"].max() if not bars.empty else None
        self.seen.append((ctx.asof, newest))
        return self.inner.target_weights(ctx)
