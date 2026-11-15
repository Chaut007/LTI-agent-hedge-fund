"""Turning a raw factor value into a Signal score.

Every agent faces the same problem: it computes some number per ticker — a
12-month return, a debt ratio, a volatility — and has to express it as a score
in [-1, 1]. Two decisions are made here once, for all of them.

**Rank, not z-score.** Factor values have fat tails. One ticker with a 400%
return or a debt/equity of 90 would dominate a z-score and quietly become the
whole portfolio. Cross-sectional rank is bounded by construction, so an
outlier can be the top-ranked name but cannot be worth ten of them.

**A score is relative to today's universe, not to history.** Rank 1 of 30
means "best available today". In a month where everything fell, the top-ranked
momentum name still scores +1 — it is the least-bad. That is the correct
behaviour for a long-only cross-sectional strategy, which must always hold
something, and the wrong behaviour for a timing strategy, which must be able to
say "nothing here". Step 5's risk layer is where "all cash" gets decided; an
agent's job is only to rank.
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


def cross_sectional_scores(
    values: Mapping[str, float], higher_is_better: bool = True
) -> dict[str, float]:
    """Rank `values` across tickers and map them onto [-1, 1].

    Worst-ranked gets -1.0, best +1.0, median 0.0. Ties share the average of
    the ranks they span. Tickers whose value is None or non-finite are dropped,
    not scored zero: "could not evaluate" and "evaluated, no edge" are
    different statements and the Agent contract keeps them apart.

    With `higher_is_better=False` the ordering inverts, for factors where small
    is good (volatility, leverage, recent return).
    """
    series = pd.Series(dict(values), dtype="float64")
    series = series[series.apply(lambda v: isinstance(v, float) and math.isfinite(v))]
    if series.empty:
        return {}
    if len(series) == 1:
        # A rank is meaningless with one observation. Say "no edge", which is
        # an honest statement, rather than inventing a +1.
        return {str(series.index[0]): 0.0}

    ranks = series.rank(method="average", ascending=higher_is_better)
    scaled = 2.0 * (ranks - 1.0) / (len(series) - 1.0) - 1.0
    return {str(ticker): float(score) for ticker, score in scaled.items()}


def combine_scores(
    parts: Mapping[str, Mapping[str, float]], min_parts: int = 1
) -> dict[str, float]:
    """Average several sub-scores per ticker, e.g. Quality's three ratios.

    A ticker is scored only if at least `min_parts` of the sub-scores exist for
    it, so a company missing two of three quality ratios is dropped rather than
    judged on the one that happened to be filed. The average is over the parts
    actually present, not over all of them, so a present part is never diluted
    by a missing one.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sub in parts.values():
        for ticker, score in sub.items():
            totals[ticker] = totals.get(ticker, 0.0) + score
            counts[ticker] = counts.get(ticker, 0) + 1
    return {
        ticker: totals[ticker] / counts[ticker]
        for ticker in totals
        if counts[ticker] >= min_parts
    }


def coverage_confidence(available: int, required: int, floor: float = 0.3) -> float:
    """Confidence from how much history a window actually had.

    A momentum score computed from 180 of the 252 bars it wanted is a real
    signal but a weaker one, and the aggregator should lean on it less. Scales
    linearly from `floor` at zero coverage to 1.0 at full coverage.
    """
    if required <= 0:
        return 1.0
    ratio = max(0.0, min(1.0, available / required))
    return float(floor + (1.0 - floor) * ratio)


def staleness_confidence(age_days: int, fresh_days: int = 120, stale_days: int = 450) -> float:
    """Confidence from how old the filing a fundamental rests on is.

    Full confidence up to `fresh_days` (a quarter plus filing lag), decaying
    linearly to 0.3 at `stale_days` — about the point where a company has
    missed an annual report and the number says more about the filing calendar
    than the business.
    """
    if age_days <= fresh_days:
        return 1.0
    if age_days >= stale_days:
        return 0.3
    span = stale_days - fresh_days
    return float(1.0 - 0.7 * (age_days - fresh_days) / span)
