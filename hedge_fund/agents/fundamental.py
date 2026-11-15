"""Agents that read SEC filings. Still zero cost, still no LLM.

Both are composites: Value ranks two ratios and averages the ranks, Quality
ranks three. Averaging *ranks* rather than the ratios themselves is the point
— earnings yield and book/price live on different scales, and a company with
a book/price of 8 would otherwise swamp the earnings-yield half of its own
score.

Both discount by filing age through `staleness_confidence`. A ratio resting
on a filing from fourteen months ago is a real number about a company that
may no longer exist in that shape, and the aggregator should lean on it less
than on one filed last month.
"""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.agents import fundamentals as f
from hedge_fund.agents.scoring import (
    combine_scores,
    cross_sectional_scores,
    staleness_confidence,
)
from hedge_fund.core.agent import BaseAgent, Context
from hedge_fund.core.signal import Signal

NET_INCOME = ("NetIncomeLoss",)
GROSS_PROFIT = ("GrossProfit",)


@dataclass(frozen=True)
class ValueAgent(BaseAgent):
    """Cheapness: earnings yield and book-to-price, equally weighted.

    Earnings yield is TTM net income over market cap; book-to-price is
    shareholders' equity over market cap. Both are inverted price multiples,
    so high is cheap, and both are undefined or meaningless when the
    denominator is non-positive — `ratio` drops those rather than ranking a
    negative multiple as a bargain.
    """

    name: str = "value"
    horizon_days: int = 252

    def evaluate(self, ctx: Context) -> list[Signal]:
        tickers = list(ctx.universe)
        caps = f.market_cap(ctx.view, tickers)
        if not caps:
            return []

        earnings = f.trailing_twelve_months(ctx.view, tickers, NET_INCOME)
        equity = f.latest_instant(ctx.view, tickers, f.EQUITY_TAGS)

        earnings_yield = f.ratio(earnings, caps, asof=ctx.asof)
        book_to_price = f.ratio(equity, caps, asof=ctx.asof)

        scores = combine_scores(
            {
                "earnings_yield": cross_sectional_scores(f.values(earnings_yield)),
                "book_to_price": cross_sectional_scores(f.values(book_to_price)),
            }
        )
        # Oldest, not newest: a composite is as stale as its stalest part.
        filings = f.oldest_filing(earnings_yield, book_to_price)

        out: list[Signal] = []
        for ticker, score in scores.items():
            filed = filings[ticker]
            parts = []
            if ticker in earnings_yield:
                parts.append(f"E/P {earnings_yield[ticker].value:+.1%}")
            if ticker in book_to_price:
                parts.append(f"B/P {book_to_price[ticker].value:.2f}")
            out.append(
                self.signal(
                    ticker,
                    ctx,
                    score,
                    confidence=staleness_confidence((ctx.asof - filed).days),
                    rationale=f"{', '.join(parts)} (filed {filed})",
                )
            )
        return out


@dataclass(frozen=True)
class QualityAgent(BaseAgent):
    """Profitability, leverage and margin, equally weighted.

    - ROE: TTM net income over shareholders' equity. High is good.
    - Debt/equity: total liabilities over equity. Low is good.
    - Gross margin: TTM gross profit over TTM revenue. High is good.

    A ticker needs at least two of the three to be scored. One ratio is not a
    quality assessment, and judging a company on whichever number its filer
    happened to tag would rank disclosure habits instead of businesses.
    """

    name: str = "quality"
    horizon_days: int = 252
    min_ratios: int = 2

    def evaluate(self, ctx: Context) -> list[Signal]:
        tickers = list(ctx.universe)

        earnings = f.trailing_twelve_months(ctx.view, tickers, NET_INCOME)
        revenue = f.trailing_twelve_months(ctx.view, tickers, f.REVENUE_TAGS)
        gross = f.trailing_twelve_months(ctx.view, tickers, GROSS_PROFIT)
        equity = f.latest_instant(ctx.view, tickers, f.EQUITY_TAGS)
        debt = f.latest_instant(ctx.view, tickers, f.DEBT_TAGS)

        roe = f.ratio(earnings, equity, asof=ctx.asof)
        leverage = f.ratio(debt, equity, asof=ctx.asof)
        margin = f.ratio(gross, revenue, asof=ctx.asof)

        scores = combine_scores(
            {
                "roe": cross_sectional_scores(f.values(roe)),
                "leverage": cross_sectional_scores(
                    f.values(leverage), higher_is_better=False
                ),
                "margin": cross_sectional_scores(f.values(margin)),
            },
            min_parts=self.min_ratios,
        )
        filings = f.oldest_filing(roe, leverage, margin)

        out: list[Signal] = []
        for ticker, score in scores.items():
            filed = filings[ticker]
            parts = []
            if ticker in roe:
                parts.append(f"ROE {roe[ticker].value:+.1%}")
            if ticker in leverage:
                parts.append(f"D/E {leverage[ticker].value:.2f}")
            if ticker in margin:
                parts.append(f"margin {margin[ticker].value:.1%}")
            out.append(
                self.signal(
                    ticker,
                    ctx,
                    score,
                    confidence=staleness_confidence((ctx.asof - filed).days),
                    rationale=f"{', '.join(parts)} (filed {filed})",
                )
            )
        return out
