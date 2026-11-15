"""Point-in-time fundamental quantities, assembled from XBRL facts.

Nothing in EDGAR is shaped like a ratio. Getting "earnings yield" means
summing four quarterly net income figures that were each filed on a different
day, finding a share count filed with the most recent of them, multiplying by
a price from the same asof date, and knowing which filing the whole thing
rests on so staleness can be scored. That assembly is here, once, rather than
in each agent.

Every value returned carries the `filed` date of the newest fact it used.
Agents use it for confidence; it is also the audit trail — the answer to "what
did we know, and when did we know it".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

import pandas as pd

from hedge_fund.core.agent import MarketView

# EDGAR has no single revenue tag. Filers use whichever their accountants
# chose, and the choice changes across a company's own history. Tried in
# order; the first with data wins.
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
SHARE_TAGS = (
    "EntityCommonStockSharesOutstanding",  # dei, filed on the cover page
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
)
EQUITY_TAGS = ("StockholdersEquity",)
DEBT_TAGS = ("Liabilities",)


@dataclass(frozen=True, slots=True)
class Fact:
    """One assembled quantity and the newest filing it depends on."""

    value: float
    filed: date
    tag: str

    def age_days(self, asof: date) -> int:
        return (asof - self.filed).days


def _rows_by_ticker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    return {str(t): g for t, g in frame.groupby("ticker", sort=False)}


def latest_instant(
    view: MarketView, tickers: Sequence[str], tags: Sequence[str]
) -> dict[str, Fact]:
    """Newest visible balance-sheet value per ticker, trying `tags` in order.

    Instant facts (assets, equity, share counts) have no period, so the
    freshest one visible is simply the right answer.
    """
    frame = view.latest_fundamentals(tickers, list(tags), period_kind="instant")
    out: dict[str, Fact] = {}
    for ticker, rows in _rows_by_ticker(frame).items():
        for tag in tags:  # preference order, not whatever the frame ordered by
            match = rows[rows["tag"] == tag]
            if not match.empty:
                row = match.sort_values("period_end").iloc[-1]
                out[ticker] = Fact(float(row["value"]), row["filed"], tag)
                break
    return out


def trailing_twelve_months(
    view: MarketView, tickers: Sequence[str], tags: Sequence[str]
) -> dict[str, Fact]:
    """TTM total of a flow item (income, revenue), per ticker.

    Preference order, and why:

    1. Four most recent distinct quarterly periods. Freshest possible view —
       up to three months old — and it is what the market is actually pricing.
    2. The latest annual figure, if four quarters are not on file. Up to
       fifteen months stale after filing lag, so `Fact.filed` matters: the
       agent discounts it rather than treating it as current.

    Quarters are summed only when four *distinct* period ends are visible.
    Three quarters are not annualised to four: scaling 9 months by 4/3 invents
    a quarter, and for a seasonal business it invents the wrong one.
    """
    quarterly = view.fundamentals(tickers, list(tags))
    out: dict[str, Fact] = {}

    for ticker, rows in _rows_by_ticker(quarterly).items():
        for tag in tags:
            match = rows[(rows["tag"] == tag) & (rows["period_kind"] == "quarterly")]
            if match.empty:
                continue
            # One row per period end: the view already resolved restatements,
            # but a period can appear under several filings.
            latest_per_period = (
                match.sort_values(["period_end", "filed"])
                .groupby("period_end", as_index=False)
                .last()
                .sort_values("period_end")
            )
            if len(latest_per_period) < 4:
                continue
            last_four = latest_per_period.iloc[-4:]
            out[ticker] = Fact(
                float(last_four["value"].sum()),
                max(last_four["filed"]),
                f"{tag}:ttm",
            )
            break

    missing = [t for t in tickers if t not in out]
    if missing:
        annual = view.latest_fundamentals(missing, list(tags), period_kind="annual")
        for ticker, rows in _rows_by_ticker(annual).items():
            for tag in tags:
                match = rows[rows["tag"] == tag]
                if not match.empty:
                    row = match.sort_values("period_end").iloc[-1]
                    out[ticker] = Fact(float(row["value"]), row["filed"], f"{tag}:annual")
                    break
    return out


def last_close(view: MarketView, tickers: Sequence[str]) -> dict[str, float]:
    """Most recent close on or before asof. `close`, not `adj_close`.

    Market cap needs the price actually printed that day times the shares
    actually outstanding. `adj_close` is retroactively adjusted for splits, so
    pairing it with a historical share count double-counts every split since.
    """
    bars = view.prices(tickers, bars=1)
    if bars.empty:
        return {}
    return {
        str(row.ticker): float(row.close)
        for row in bars.itertuples(index=False)
        if pd.notna(row.close)
    }


def market_cap(view: MarketView, tickers: Sequence[str]) -> dict[str, Fact]:
    """Shares outstanding x last close, in dollars.

    The share count comes from a filing and the price from the tape, so the
    two are as of different moments: shares as of the last report, price as of
    today. That is what any real-time market cap is, and `Fact.filed` records
    which filing the share count came from.
    """
    shares = latest_instant(view, tickers, SHARE_TAGS)
    prices = last_close(view, tickers)
    return {
        ticker: Fact(fact.value * prices[ticker], fact.filed, fact.tag)
        for ticker, fact in shares.items()
        if ticker in prices and fact.value > 0
    }


def ratio(
    numerator: Mapping[str, Fact],
    denominator: Mapping[str, Fact],
    *,
    require_positive_denominator: bool = True,
) -> dict[str, Fact]:
    """Divide two assembled quantities, keeping the newer filing date.

    A non-positive denominator yields no value rather than a signed one.
    Negative book equity is real (airlines, leveraged buyouts) but a
    book/price of -0.4 is not "cheap", and ranking it as such would buy
    exactly the companies the factor is meant to avoid.
    """
    out: dict[str, Fact] = {}
    for ticker, num in numerator.items():
        den = denominator.get(ticker)
        if den is None or den.value == 0.0:
            continue
        if require_positive_denominator and den.value <= 0.0:
            continue
        out[ticker] = Fact(
            num.value / den.value,
            max(num.filed, den.filed),
            f"{num.tag}/{den.tag}",
        )
    return out


def values(facts: Mapping[str, Fact]) -> dict[str, float]:
    return {ticker: fact.value for ticker, fact in facts.items()}


def newest_filing(*groups: Mapping[str, Fact]) -> dict[str, date]:
    """Per ticker, the newest filing date across several assembled quantities."""
    out: dict[str, date] = {}
    for group in groups:
        for ticker, fact in group.items():
            current = out.get(ticker)
            out[ticker] = fact.filed if current is None else max(current, fact.filed)
    return out
