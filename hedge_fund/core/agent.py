"""The Agent contract, and the only data handle an agent is allowed to hold.

Design note — why agents never receive a raw path or a live DuckDB connection:

The single most expensive mistake in a research pipeline is lookahead bias, and
discipline alone does not prevent it. So an agent is handed a `MarketView` that
is already bound to one `asof` date. Every method on it clamps its own query to
that date. There is no argument an agent can pass to see past it, which turns a
"remember not to peek" rule into something the type actually enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from hedge_fund.core.signal import Signal


@runtime_checkable
class MarketView(Protocol):
    """Read-only, asof-bounded access to the lake.

    Implemented by `hedge_fund.data.lake.AsofView`. Declared here so `core`
    stays free of any dependency on the storage layer.
    """

    @property
    def asof(self) -> date: ...

    def prices(
        self,
        tickers: Sequence[str] | None = None,
        bars: int | None = None,
        calendar_days: int | None = None,
    ) -> pd.DataFrame:
        """Daily bars with `date <= asof`, windowed by exactly one of:

        bars:          last N trading bars per ticker — what a factor wants.
                       `bars=252` is twelve months of trading, holidays included.
        calendar_days: a civil-calendar lookback. 252 calendar days is only
                       about 173 bars; say so explicitly if that is what you mean.

        Defaults to `bars=400`. Columns: ticker, date, open, high, low, close,
        adj_close, volume.
        """
        ...

    def fundamentals(
        self,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """XBRL facts whose SEC filing date is `<= asof`, restatements resolved.

        Columns: ticker, cik, taxonomy, tag, unit, period_start, period_end,
        value, form, filed, accn, fy, fp, duration_days, period_kind.
        """
        ...

    def latest_fundamentals(
        self,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        period_kind: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Freshest visible fact per (ticker, taxonomy, tag, unit, period_kind).

        One row per period length, because a 10-Q reports both a 3-month and a
        year-to-date figure for the same tag ending the same day. Check
        `period_kind` / `duration_days` before using a value in a ratio, or
        pass `period_kind="quarterly"` to narrow.
        """
        ...

    def macro(
        self,
        series_ids: Sequence[str] | None = None,
        calendar_days: int = 400,
    ) -> pd.DataFrame:
        """Macro observations with `date <= asof` over a calendar-day lookback.

        Columns: series_id, date, value.
        """
        ...


@dataclass(frozen=True, slots=True)
class Context:
    """Everything an agent is given, and nothing more."""

    asof: date
    view: MarketView
    universe: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.view.asof != self.asof:
            raise ValueError(
                f"Context.asof ({self.asof}) disagrees with its MarketView "
                f"({self.view.asof}). Refusing to run: one of them is lying "
                "about what day it is."
            )


@runtime_checkable
class Agent(Protocol):
    """A source of Signals.

    `evaluate` receives the whole universe at once rather than one ticker at a
    time, because most useful factors are cross-sectional (a rank is meaningless
    for a single name) and because one vectorised pass beats 500 round trips.

    Return a Signal only for tickers you actually have an opinion on. Omitting a
    ticker means "insufficient data"; returning score 0.0 means "evaluated, no
    edge". The aggregator treats those differently.
    """

    name: str

    def evaluate(self, ctx: Context) -> list[Signal]: ...


class BaseAgent:
    """Optional convenience base: name from the class, and a cost accumulator."""

    name: str = "unnamed"
    horizon_days: int = 21

    def evaluate(self, ctx: Context) -> list[Signal]:  # pragma: no cover
        raise NotImplementedError

    def signal(
        self,
        ticker: str,
        ctx: Context,
        score: float,
        confidence: float = 1.0,
        rationale: str = "",
        cost_usd: float = 0.0,
        horizon_days: int | None = None,
    ) -> Signal:
        return Signal(
            agent=self.name,
            ticker=ticker,
            asof=ctx.asof,
            score=score,
            confidence=confidence,
            horizon_days=horizon_days or self.horizon_days,
            rationale=rationale,
            cost_usd=cost_usd,
        )
