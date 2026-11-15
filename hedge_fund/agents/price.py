"""Agents that need nothing but price history. Zero cost, no LLM, no filings.

Three documented factors, each in its standard construction rather than a
variant tuned to look good on this data:

- Momentum, 12-1: the twelve-month return excluding the most recent month.
  The skip is not decoration — including the last month mixes in short-term
  reversal, which points the other way, and the two partly cancel.
- Low volatility: 60-day realised volatility, ranked ascending. Low-vol
  stocks have historically beaten high-vol ones on a risk-adjusted basis.
- Short-term reversal: the 5-day return, ranked ascending. Recent losers
  bounce over days, which is the opposite sign to momentum over months.

Momentum and Reversal are deliberately opposed. That is not a bug in the
roster: they disagree at different horizons, and `Signal.horizon_days` is how
each says which horizon it means. A naive average of the two is wrong, and
the aggregator says so where it does it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedge_fund.agents.scoring import coverage_confidence, cross_sectional_scores
from hedge_fund.core.agent import BaseAgent, Context
from hedge_fund.core.signal import Signal

TRADING_DAYS = 252


def price_panel(ctx: Context, bars: int) -> pd.DataFrame:
    """Dates x tickers of adj_close, as far back as `bars` per ticker.

    adj_close is correct here: these are all return calculations, and a 2:1
    split must not read as a -50% day.
    """
    frame = ctx.view.prices(list(ctx.universe), bars=bars)
    if frame.empty:
        return pd.DataFrame(dtype="float64")
    panel = frame.pivot(index="date", columns="ticker", values="adj_close")
    return panel.sort_index().astype("float64")


@dataclass(frozen=True)
class MomentumAgent(BaseAgent):
    """12-1 month momentum. Needs the full window or it says nothing.

    A ticker with eight months of history is not scored. Measuring "twelve
    month momentum" over whatever happens to be available would put a
    six-month return and a twelve-month return in the same ranking, which
    ranks listing dates rather than momentum.
    """

    name: str = "momentum"
    horizon_days: int = 126
    lookback: int = TRADING_DAYS
    skip: int = 21

    @property
    def required_bars(self) -> int:
        return self.lookback + 1

    def evaluate(self, ctx: Context) -> list[Signal]:
        panel = price_panel(ctx, self.required_bars + 5)
        if panel.empty:
            return []

        raw: dict[str, float] = {}
        for ticker in panel.columns:
            series = panel[ticker].dropna()
            if len(series) < self.required_bars:
                continue
            recent = series.iloc[-(self.skip + 1)]
            base = series.iloc[-self.required_bars]
            if base <= 0 or not np.isfinite(recent) or not np.isfinite(base):
                continue
            raw[str(ticker)] = float(recent / base - 1.0)

        scores = cross_sectional_scores(raw, higher_is_better=True)
        return [
            self.signal(
                ticker,
                ctx,
                score,
                rationale=f"12-1 momentum {raw[ticker]:+.1%}",
            )
            for ticker, score in scores.items()
        ]


@dataclass(frozen=True)
class LowVolAgent(BaseAgent):
    """60-day realised volatility, ranked ascending (calm is good).

    Unlike momentum, a short window still estimates the same quantity — just
    noisily — so a ticker with 40 of 60 days is scored with reduced
    confidence rather than dropped.
    """

    name: str = "low_vol"
    horizon_days: int = 63
    window: int = 60
    min_window: int = 30

    def evaluate(self, ctx: Context) -> list[Signal]:
        panel = price_panel(ctx, self.window + 5)
        if panel.empty:
            return []

        raw: dict[str, float] = {}
        observations: dict[str, int] = {}
        for ticker in panel.columns:
            series = panel[ticker].dropna()
            returns = series.pct_change(fill_method=None).dropna().iloc[-self.window :]
            if len(returns) < self.min_window:
                continue
            sd = float(returns.std(ddof=1))
            if not np.isfinite(sd) or sd <= 0.0:
                continue
            raw[str(ticker)] = sd * np.sqrt(TRADING_DAYS)
            observations[str(ticker)] = len(returns)

        scores = cross_sectional_scores(raw, higher_is_better=False)
        return [
            self.signal(
                ticker,
                ctx,
                score,
                confidence=coverage_confidence(observations[ticker], self.window),
                rationale=f"annualised vol {raw[ticker]:.1%} over {observations[ticker]}d",
            )
            for ticker, score in scores.items()
        ]


@dataclass(frozen=True)
class ReversalAgent(BaseAgent):
    """5-day return, ranked ascending (recent losers bounce).

    Opposite sign to momentum by design, over a horizon twenty-five times
    shorter. Its `horizon_days=5` is the field that keeps the aggregator from
    treating the disagreement as noise to be averaged away.
    """

    name: str = "reversal"
    horizon_days: int = 5
    window: int = 5

    def evaluate(self, ctx: Context) -> list[Signal]:
        panel = price_panel(ctx, self.window + 3)
        if panel.empty:
            return []

        raw: dict[str, float] = {}
        for ticker in panel.columns:
            series = panel[ticker].dropna()
            if len(series) < self.window + 1:
                continue
            base = series.iloc[-(self.window + 1)]
            last = series.iloc[-1]
            if base <= 0 or not np.isfinite(base) or not np.isfinite(last):
                continue
            raw[str(ticker)] = float(last / base - 1.0)

        scores = cross_sectional_scores(raw, higher_is_better=False)
        return [
            self.signal(
                ticker,
                ctx,
                score,
                rationale=f"{self.window}-day return {raw[ticker]:+.1%}",
            )
            for ticker, score in scores.items()
        ]
