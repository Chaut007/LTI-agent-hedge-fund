"""Synthetic lake builder shared by the agent and portfolio tests.

Filing dates are derived from the asof the test will use, rather than written
as literals. Two ad-hoc smoke scripts during development "failed" only because
their hand-picked filing dates landed after the asof — the point-in-time view
was correct and the fixture was wrong. Deriving the dates removes that whole
class of self-inflicted confusion from the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hedge_fund.core.agent import Context
from hedge_fund.data.lake import (
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTALS,
    PRICE_COLUMNS,
    PRICES,
    Lake,
)

FILING_LAG = timedelta(days=35)


@dataclass(frozen=True)
class SyntheticLake:
    lake: Lake
    tickers: tuple[str, ...]
    asof: date
    trading_days: pd.DatetimeIndex

    def context(self, asof: date | None = None) -> Context:
        at = asof or self.asof
        return Context(asof=at, view=self.lake.view(at), universe=self.tickers)


def _price_rows(tickers, days, prices) -> list[dict]:
    return [
        {
            "ticker": ticker,
            "date": day.date(),
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "adj_close": px,
            "volume": 1e6,
        }
        for i, day in enumerate(days)
        for ticker, px in zip(tickers, prices[i])
    ]


def _fact(ticker, index, tag, taxonomy, unit, value, period_end, filed, period_start, suffix):
    return {
        "ticker": ticker,
        "cik": 100 + index,
        "taxonomy": taxonomy,
        "tag": tag,
        "unit": unit,
        "period_start": period_start,
        "period_end": period_end,
        "value": value,
        "form": "10-Q",
        "filed": filed,
        "accn": f"{ticker}-{suffix}",
        "fy": period_end.year,
        "fp": "Q1",
    }


def build_synthetic_lake(
    directory,
    n_tickers: int = 12,
    n_days: int = 400,
    seed: int = 7,
    with_fundamentals: bool = True,
) -> SyntheticLake:
    """Prices with a spread of drifts and vols, plus visible quarterly filings.

    Ticker i has drift increasing and volatility increasing in i, so momentum
    and low-vol disagree by construction — which is what makes the aggregator
    worth testing. Fundamentals are scaled by i too, so Value and Quality
    produce a real ranking rather than ties.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = tuple(f"T{i:02d}" for i in range(n_tickers))
    returns = rng.normal(
        np.linspace(-0.0004, 0.0008, n_tickers),
        np.linspace(0.008, 0.030, n_tickers),
        size=(n_days, n_tickers),
    )
    prices = 100 * np.cumprod(1 + returns, axis=0)

    lake = Lake(directory)
    lake.ensure_dir()
    lake.write(PRICES, pd.DataFrame(_price_rows(tickers, days, prices), columns=PRICE_COLUMNS))

    asof = days[-1].date()

    if with_fundamentals:
        facts: list[dict] = []
        # Eight quarters ending well before asof, so four are always visible
        # even after the 35-day filing lag.
        quarter_ends = [asof - timedelta(days=90 * k + 60) for k in range(8)]
        for i, ticker in enumerate(tickers):
            for q, period_end in enumerate(quarter_ends):
                filed = period_end + FILING_LAG
                for tag, value in (
                    ("NetIncomeLoss", (1.0 + 0.4 * i) * 1e8),
                    ("Revenues", 1e9),
                    ("GrossProfit", (0.2 + 0.04 * i) * 1e9),
                ):
                    facts.append(
                        _fact(
                            ticker, i, tag, "us-gaap", "USD", value,
                            period_end, filed,
                            period_end - timedelta(days=90), f"{q}-{tag}",
                        )
                    )
            # Balance-sheet items: one instant fact, filed before asof.
            instant_end = quarter_ends[0]
            instant_filed = instant_end + FILING_LAG
            for tag, taxonomy, unit, value in (
                ("StockholdersEquity", "us-gaap", "USD", (5.0 - 0.3 * i) * 1e9),
                ("Liabilities", "us-gaap", "USD", (2.0 + 0.5 * i) * 1e9),
                ("EntityCommonStockSharesOutstanding", "dei", "shares", 1e8),
            ):
                facts.append(
                    _fact(
                        ticker, i, tag, taxonomy, unit, value,
                        instant_end, instant_filed, None, f"inst-{tag}",
                    )
                )
        lake.write(FUNDAMENTALS, pd.DataFrame(facts, columns=FUNDAMENTAL_COLUMNS))

    return SyntheticLake(lake=lake, tickers=tickers, asof=asof, trading_days=days)


@pytest.fixture()
def synthetic(tmp_path) -> SyntheticLake:
    return build_synthetic_lake(tmp_path / "synthetic")


@pytest.fixture()
def prices_only(tmp_path) -> SyntheticLake:
    """No filings at all: Value and Quality must decline, not crash."""
    return build_synthetic_lake(tmp_path / "prices_only", with_fundamentals=False)
