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
from datetime import date, timedelta
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
# Diluted first, deliberately. The dei cover-page count is filed per share
# class and `companyfacts` does not expose the class dimension, so a
# multi-class filer silently yields one class only: Visa came back as 469M
# shares against ~2,000M actual, a 77% understatement that made its market cap
# $123B instead of ~$520B and its earnings yield 14.9% instead of ~3.5%.
# The weighted diluted count is consolidated by construction — it is the
# denominator of reported EPS — so it cannot lose a class.
SHARE_TAGS = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",  # dei, cover page, per class
    "CommonStockSharesOutstanding",
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
        # Freshest tag wins, for the same reason as in `trailing_twelve_months`:
        # a filer that switched tags leaves the old one populated but frozen.
        candidates = []
        for tag in tags:
            match = rows[rows["tag"] == tag]
            if not match.empty:
                row = match.sort_values(["filed", "period_end"]).iloc[-1]
                candidates.append(Fact(float(row["value"]), row["filed"], tag))
        if candidates:
            out[ticker] = max(candidates, key=lambda fact: fact.filed)
    return out


#: A valid TTM window: four quarters must cover about a year, and must tile it
#: without gaps or overlaps. Fiscal quarters are 90-92 days, so four of them
#: cover 360-368; the tolerance allows for 52/53-week retail calendars.
TTM_MIN_DAYS = 330
TTM_MAX_DAYS = 400
TTM_GAP_TOLERANCE = 20


def _tiles_a_year(quarters: pd.DataFrame) -> bool:
    """Do these four quarters actually form a trailing twelve months?

    This check exists because without it the function was silently wrong for
    nearly every filer. A 10-K filer does not report its fiscal Q4 as a
    quarterly fact — the annual report carries the full year instead — so the
    visible quarterly series has a hole once a year. Taking "the four most
    recent quarterly facts" therefore reaches back across that hole and sums a
    15-month window.

    Measured on real filings: AAPL's four most recent quarterly GrossProfit
    facts spanned 455 days and skipped Jul-Sep entirely, and NVDA's produced a
    gross margin of 438.9%. Apple's looked plausible at 45.4% only because its
    margin is stable; the dollar total was wrong either way.
    """
    starts = quarters["period_start"]
    if starts.isna().any():
        return False
    covered = float(quarters["duration_days"].sum())
    span = (max(quarters["period_end"]) - min(starts)).days
    return (
        TTM_MIN_DAYS <= covered <= TTM_MAX_DAYS
        and TTM_MIN_DAYS <= span <= TTM_MAX_DAYS
        and abs(span - covered) <= TTM_GAP_TOLERANCE
    )


#: A quarter is "the same quarter a year earlier" within this many days.
YEAR_MATCH_TOLERANCE = 15


def _dedupe_by_period(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per period end, newest filing winning, sorted by period end."""
    return (
        frame.sort_values(["period_end", "filed"])
        .groupby("period_end", as_index=False)
        .last()
        .sort_values("period_end")
        .reset_index(drop=True)
    )


def _contiguous_tail(quarters: pd.DataFrame, after: date) -> pd.DataFrame:
    """Quarters that tile forward from `after` without a gap, newest last."""
    following = quarters[quarters["period_start"].notna()]
    following = following[following["period_start"] > after].reset_index(drop=True)
    kept: list[int] = []
    boundary = after
    for i, row in enumerate(following.itertuples(index=False)):
        if abs((row.period_start - boundary).days - 1) > 5:
            break  # a hole: everything after it is not contiguous with `after`
        kept.append(i)
        boundary = row.period_end
    return following.iloc[kept]


def _match_prior_year(quarters: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame | None:
    """The same quarters one year earlier, or None if any is missing."""
    rows = []
    for target in window["period_end"]:
        wanted = target - timedelta(days=365)
        candidates = quarters[
            quarters["period_end"].apply(
                lambda e: abs((e - wanted).days) <= YEAR_MATCH_TOLERANCE
            )
        ]
        if len(candidates) != 1:
            return None
        rows.append(candidates.iloc[0])
    return pd.DataFrame(rows)


def trailing_twelve_months(
    view: MarketView, tickers: Sequence[str], tags: Sequence[str]
) -> dict[str, Fact]:
    """TTM total of a flow item (income, revenue), per ticker.

    **Why this is not just "sum the last four quarters".** A 10-K filer does
    not report its fiscal Q4 as a quarterly fact — the annual report carries
    the whole year instead — so the visible quarterly series has a hole once a
    year and a run of four consecutive quarters usually does not exist at all.
    Measured on real EDGAR data for 20 large caps: summing the four most recent
    quarterly facts gave NVDA a gross margin of 438.9% and spanned 455 days for
    AAPL; searching backwards for a window that does tile a year reached a 2013
    filing for V and a 2015 one for UNH.

    So the primary construction is the standard one:

        TTM = latest annual + quarters since it - the same quarters a year before

    which is exact, uses the freshest filings available, and needs no run of
    four. Order of preference:

    1. Annual plus delta, as above.
    2. Four consecutive quarters that tile a year, for filers that publish only
       10-Qs in the visible window.
    3. The latest annual alone, up to fifteen months stale — `Fact.filed`
       carries that so `staleness_confidence` can discount it.

    Three quarters are never annualised to four: scaling nine months by 4/3
    invents a quarter, and for a seasonal business it invents the wrong one.
    """
    quarterly_all = view.fundamentals(tickers, list(tags))
    annual_all = view.fundamentals(tickers, list(tags))
    by_ticker_q = _rows_by_ticker(quarterly_all)
    by_ticker_a = _rows_by_ticker(annual_all)
    out: dict[str, Fact] = {}

    for ticker in tickers:
        rows_q = by_ticker_q.get(ticker)
        rows_a = by_ticker_a.get(ticker)
        if rows_q is None and rows_a is None:
            continue
        # Every tag is tried and the FRESHEST answer wins — not the first one
        # that happens to work. Filers migrate between revenue tags, and a
        # static preference order silently locks onto one they abandoned:
        # NVDA's RevenueFromContractWithCustomer... stops in 2020 while its
        # Revenues runs to 2024, so "first tag that returns something" gave a
        # 2020 figure (-86% against published) and never reached the live tag.
        # Mastercard had the same shape two years apart. Order is now only a
        # tiebreaker between tags of equal freshness.
        candidates = [
            fact
            for fact in (_ttm_for_tag(rows_q, rows_a, tag) for tag in tags)
            if fact is not None
        ]
        if candidates:
            out[ticker] = max(candidates, key=lambda fact: fact.filed)
    return out


def _ttm_for_tag(
    rows_q: pd.DataFrame | None, rows_a: pd.DataFrame | None, tag: str
) -> Fact | None:
    quarters = (
        _dedupe_by_period(rows_q[(rows_q["tag"] == tag) & (rows_q["period_kind"] == "quarterly")])
        if rows_q is not None
        else pd.DataFrame()
    )
    annuals = (
        _dedupe_by_period(rows_a[(rows_a["tag"] == tag) & (rows_a["period_kind"] == "annual")])
        if rows_a is not None
        else pd.DataFrame()
    )

    # -- 1. annual + delta ----------------------------------------------------
    if not annuals.empty:
        annual = annuals.iloc[-1]
        if not quarters.empty and pd.notna(annual["period_end"]):
            since = _contiguous_tail(quarters, annual["period_end"])
            if not since.empty:
                prior = _match_prior_year(quarters, since)
                if prior is not None and len(prior) == len(since):
                    value = (
                        float(annual["value"])
                        + float(since["value"].sum())
                        - float(prior["value"].sum())
                    )
                    filed = max(
                        [annual["filed"], *since["filed"], *prior["filed"]]
                    )
                    return Fact(value, filed, f"{tag}:ttm")

    # -- 2. four consecutive quarters that tile a year ------------------------
    if len(quarters) >= 4:
        for end in range(len(quarters), 3, -1):
            candidate = quarters.iloc[end - 4 : end]
            if _tiles_a_year(candidate):
                return Fact(
                    float(candidate["value"].sum()),
                    max(candidate["filed"]),
                    f"{tag}:4q",
                )

    # -- 3. the latest annual, stale and flagged ------------------------------
    if not annuals.empty:
        annual = annuals.iloc[-1]
        return Fact(float(annual["value"]), annual["filed"], f"{tag}:annual")
    return None


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


#: A share count below this fraction of the consolidated diluted count is
#: missing share classes and must not be used.
MIN_SHARE_COVERAGE = 0.8


def diluted_shares(view: MarketView, tickers: Sequence[str]) -> dict[str, Fact]:
    """Weighted diluted share count — consolidated across every class.

    Not reachable through `latest_instant`: this is a *period* fact (a weighted
    average over a quarter or year), so it has a duration and never appears
    under `period_kind="instant"`. It matters because the dei cover-page count
    is filed per share class and `companyfacts` does not expose the class
    dimension, so a multi-class filer silently yields one class: Visa returned
    469M against roughly 2,000M actual.
    """
    frame = view.fundamentals(
        tickers, ["WeightedAverageNumberOfDilutedSharesOutstanding"]
    )
    out: dict[str, Fact] = {}
    for ticker, rows in _rows_by_ticker(frame).items():
        usable = rows[rows["period_kind"].isin(("quarterly", "annual"))]
        if usable.empty:
            continue
        row = usable.sort_values(["filed", "period_end"]).iloc[-1]
        if row["value"] > 0:
            out[ticker] = Fact(
                float(row["value"]), row["filed"], "WeightedAverageDilutedShares"
            )
    return out


def market_cap(view: MarketView, tickers: Sequence[str]) -> dict[str, Fact]:
    """Shares outstanding x last close, in dollars.

    The share count comes from a filing and the price from the tape, so the two
    are as of different moments: shares as of the last report, price as of
    today. That is what any real-time market cap is. Measured against published
    caps for 20 large caps, this lands within 1% for 15 of them. The three it
    does not are both known and unfixable from this lake:

    **A split between the filing and the asof.** NVDA filed 2.47bn shares on
    2024-05-29 and split 10-for-1 on 2024-06-10; AVGO split in July 2024. The
    error is exactly tenfold and silent. It cannot be corrected here, because
    yfinance's `close` is already split-adjusted retroactively — so
    `close / adj_close` reflects dividends only and carries no trace of the
    split. (That retroactive adjustment is itself a point-in-time leak: the
    price series knows about a split the filings had not yet reported.) The fix
    belongs in the data layer: ingest yfinance's split actions, which this
    module would then apply. Until then these names are wrong by a whole-number
    factor, which is at least the easiest kind of error to spot.

    **A multi-class filer whose diluted count is also unavailable.** Visa's
    cover-page fact reports Class A only — 469m against roughly 2,000m — and
    `companyfacts` does not expose the share-class dimension. Where the
    consolidated diluted count exists it is preferred and the problem goes
    away (Mastercard and Meta both land within 3% that way); where it does not,
    the figure is understated.
    """
    cover = latest_instant(view, tickers, SHARE_TAGS)
    diluted = diluted_shares(view, tickers)
    prices = last_close(view, tickers)

    out: dict[str, Fact] = {}
    for ticker in tickers:
        price = prices.get(ticker)
        if not price:
            continue
        chosen = cover.get(ticker)
        alternative = diluted.get(ticker)
        if alternative is not None and alternative.value > 0:
            if chosen is None or chosen.value < MIN_SHARE_COVERAGE * alternative.value:
                # The cover page is missing share classes; the diluted count is
                # consolidated by construction, so it wins.
                chosen = alternative
        if chosen is None or chosen.value <= 0:
            continue
        out[ticker] = Fact(chosen.value * price, chosen.filed, chosen.tag)
    return out


#: A ratio whose stalest input is older than this is not a fact about the
#: company any more. Amazon stopped filing GrossProfit in 2010, so pairing it
#: with 2024 revenue produced a "gross margin" of 0.9% that no staleness score
#: flagged, because the ratio reported the newer of its two dates.
MAX_RATIO_AGE_DAYS = 550


def ratio(
    numerator: Mapping[str, Fact],
    denominator: Mapping[str, Fact],
    *,
    require_positive_denominator: bool = True,
    asof: date | None = None,
    max_age_days: int = MAX_RATIO_AGE_DAYS,
) -> dict[str, Fact]:
    """Divide two assembled quantities, keeping the STALER filing date.

    The filing date is `min`, not `max`. A ratio is only as current as its
    oldest input: a 2024 revenue over a 2009 gross profit is a 2009 number
    wearing a 2024 date, and reporting the newer date told the agent to trust
    it completely. With `asof` given, such a pair is dropped outright rather
    than merely discounted.

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
        filed = min(num.filed, den.filed)
        if asof is not None and (asof - filed).days > max_age_days:
            continue
        out[ticker] = Fact(num.value / den.value, filed, f"{num.tag}/{den.tag}")
    return out


def oldest_filing(*groups: Mapping[str, Fact]) -> dict[str, date]:
    """Per ticker, the OLDEST filing date across several quantities.

    What a confidence score must be based on: a composite is as trustworthy as
    its least trustworthy part. `newest_filing` is kept for reporting what is
    freshest, but it is the wrong input to `staleness_confidence`.
    """
    out: dict[str, date] = {}
    for group in groups:
        for ticker, fact in group.items():
            current = out.get(ticker)
            out[ticker] = fact.filed if current is None else min(current, fact.filed)
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
