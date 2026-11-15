"""The test that matters most: prove the lake cannot leak the future.

Every other test in this project is about correctness of a calculation. This
one is about whether any result is meaningful at all. If an agent can see a
filing before it was filed, every backtest number downstream is fiction, and
the failure is silent and flattering — it makes the strategy look better, so
nothing alerts you.

No network: the lake is built from hand-written facts whose timing is known
exactly, which is the only way to assert on the boundary.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.core.agent import Context
from hedge_fund.data.lake import (
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTALS,
    MACRO_COLUMNS,
    MACRO,
    PRICE_COLUMNS,
    PRICES,
    Lake,
)

Q1_END = date(2023, 3, 31)
Q1_FILED = date(2023, 5, 5)
Q1_RESTATED_FILED = date(2023, 8, 1)


def _fact(**overrides) -> dict:
    base = {
        "ticker": "AAPL",
        "cik": 320193,
        "taxonomy": "us-gaap",
        "tag": "NetIncomeLoss",
        "unit": "USD",
        "period_start": date(2023, 1, 1),
        "period_end": Q1_END,
        "value": 100.0,
        "form": "10-Q",
        "filed": Q1_FILED,
        "accn": "0000320193-23-000064",
        "fy": 2023,
        "fp": "Q2",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def lake(tmp_path) -> Lake:
    lake = Lake(tmp_path / "lake")

    facts = pd.DataFrame(
        [
            # Q4 2022, filed well before any asof we test.
            _fact(
                period_start=date(2022, 10, 1),
                period_end=date(2022, 12, 31),
                value=90.0,
                filed=date(2023, 2, 2),
                accn="0000320193-23-000006",
            ),
            # Q1 2023: quarter ENDS in March but is only FILED in May.
            _fact(value=100.0),
            # Same Q1 period, restated in August with a different number.
            _fact(
                value=111.0,
                filed=Q1_RESTATED_FILED,
                accn="0000320193-23-000077",
                form="10-K/A",
            ),
        ],
        columns=FUNDAMENTAL_COLUMNS,
    )
    lake.write(FUNDAMENTALS, facts)

    bars = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": d,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": price,
                "adj_close": price,
                "volume": 1000.0,
            }
            for d, price in [
                (date(2023, 4, 27), 10.0),
                (date(2023, 4, 28), 11.0),
                (date(2023, 5, 1), 12.0),
                (date(2023, 5, 2), 13.0),
            ]
        ],
        columns=PRICE_COLUMNS,
    )
    lake.write(PRICES, bars)

    lake.write(
        MACRO,
        pd.DataFrame(
            [
                {"series_id": "DGS10", "date": date(2023, 4, 28), "value": 3.45},
                {"series_id": "DGS10", "date": date(2023, 5, 2), "value": 3.57},
            ],
            columns=MACRO_COLUMNS,
        ),
    )
    return lake


# -- fundamentals: the filing date is the boundary, not the period end --------


def test_quarter_is_invisible_before_it_is_filed(lake: Lake) -> None:
    """Q1 ended 2023-03-31 but nobody could read it on 2023-04-30."""
    view = lake.view(date(2023, 4, 30))
    facts = view.fundamentals(["AAPL"], ["NetIncomeLoss"])

    assert Q1_END not in set(facts["period_end"]), (
        "Q1 2023 leaked into a 2023-04-30 query. The view is filtering on "
        "period_end instead of filed."
    )
    assert set(facts["period_end"]) == {date(2022, 12, 31)}


def test_quarter_becomes_visible_on_its_filing_date(lake: Lake) -> None:
    view = lake.view(Q1_FILED)
    facts = view.fundamentals(["AAPL"], ["NetIncomeLoss"])
    assert Q1_END in set(facts["period_end"])


def test_latest_fundamentals_uses_newest_visible_period(lake: Lake) -> None:
    before = lake.view(date(2023, 4, 30)).latest_fundamentals(
        ["AAPL"], ["NetIncomeLoss"]
    )
    after = lake.view(date(2023, 5, 10)).latest_fundamentals(
        ["AAPL"], ["NetIncomeLoss"]
    )

    assert before.iloc[0]["value"] == pytest.approx(90.0)
    assert before.iloc[0]["period_end"] == date(2022, 12, 31)
    assert after.iloc[0]["value"] == pytest.approx(100.0)
    assert after.iloc[0]["period_end"] == Q1_END


# -- restatements: you get the number you had, not the corrected one ---------


def test_restatement_not_applied_before_it_was_filed(lake: Lake) -> None:
    """In July, Q1 still reads 100. The August correction does not time travel."""
    facts = lake.view(date(2023, 7, 1)).fundamentals(["AAPL"], ["NetIncomeLoss"])
    q1 = facts[facts["period_end"] == Q1_END]

    assert len(q1) == 1, "restatement and original both returned; dedupe failed"
    assert q1.iloc[0]["value"] == pytest.approx(100.0)


def test_restatement_applied_once_filed(lake: Lake) -> None:
    facts = lake.view(date(2023, 8, 15)).fundamentals(["AAPL"], ["NetIncomeLoss"])
    q1 = facts[facts["period_end"] == Q1_END]

    assert len(q1) == 1
    assert q1.iloc[0]["value"] == pytest.approx(111.0)
    assert q1.iloc[0]["form"] == "10-K/A"


# -- prices and macro --------------------------------------------------------


def test_prices_stop_at_asof(lake: Lake) -> None:
    bars = lake.view(date(2023, 4, 28)).prices(["AAPL"], calendar_days=30)
    assert bars["date"].max() == date(2023, 4, 28)
    assert date(2023, 5, 1) not in set(bars["date"])


def test_prices_respect_lookback_window(lake: Lake) -> None:
    bars = lake.view(date(2023, 5, 2)).prices(["AAPL"], calendar_days=2)
    assert bars["date"].min() >= date(2023, 4, 30)


def test_macro_stops_at_asof(lake: Lake) -> None:
    rows = lake.view(date(2023, 4, 30)).macro(["DGS10"], calendar_days=60)
    assert rows["date"].max() == date(2023, 4, 28)


# -- the view and the context cannot disagree about the date -----------------


def test_context_rejects_mismatched_asof(lake: Lake) -> None:
    with pytest.raises(ValueError, match="disagrees"):
        Context(
            asof=date(2023, 5, 5),
            view=lake.view(date(2023, 6, 1)),
            universe=("AAPL",),
        )


def test_cold_lake_returns_empty_frames_not_errors(tmp_path) -> None:
    """An agent running before any ingest should get nothing, not a crash."""
    empty = Lake(tmp_path / "nothing").view(date(2023, 5, 5))
    assert empty.prices(["AAPL"]).empty
    assert empty.fundamentals(["AAPL"]).empty
    assert empty.latest_fundamentals(["AAPL"]).empty
    assert empty.macro(["DGS10"]).empty


# -- period length: a 10-Q reports the quarter AND the year to date ----------


@pytest.fixture()
def dual_period_lake(tmp_path) -> Lake:
    """One filing carrying both a 3-month and a 6-month NetIncomeLoss.

    Both end 2023-06-30 and are filed the same day, so ranking on period_end
    alone cannot separate them and picks by parquet row order.
    """
    lake = Lake(tmp_path / "dual")
    lake.write(
        FUNDAMENTALS,
        pd.DataFrame(
            [
                _fact(
                    period_start=date(2023, 4, 1),
                    period_end=date(2023, 6, 30),
                    value=19_881_000_000,
                    filed=date(2023, 8, 4),
                    accn="0000320193-23-000077",
                ),
                _fact(
                    period_start=date(2023, 1, 1),
                    period_end=date(2023, 6, 30),
                    value=44_000_000_000,
                    filed=date(2023, 8, 4),
                    accn="0000320193-23-000077",
                ),
                _fact(
                    tag="Assets",
                    period_start=None,
                    period_end=date(2023, 7, 1),
                    value=335_000_000_000,
                    filed=date(2023, 8, 4),
                    accn="0000320193-23-000077",
                ),
            ],
            columns=FUNDAMENTAL_COLUMNS,
        ),
    )
    return lake


def test_quarterly_and_ytd_are_both_returned(dual_period_lake: Lake) -> None:
    """Neither may swallow the other: they are different facts."""
    got = dual_period_lake.view(date(2023, 9, 1)).latest_fundamentals(
        ["AAPL"], ["NetIncomeLoss"]
    )
    assert len(got) == 2, "a period length was dropped; the ranking lost a fact"
    assert set(got["period_kind"]) == {"quarterly", "other"}


def test_period_kind_filter_selects_the_quarter(dual_period_lake: Lake) -> None:
    """An earnings-yield agent asks for the quarter and gets exactly that."""
    got = dual_period_lake.view(date(2023, 9, 1)).latest_fundamentals(
        ["AAPL"], ["NetIncomeLoss"], period_kind="quarterly"
    )
    assert len(got) == 1
    assert got.iloc[0]["value"] == pytest.approx(19_881_000_000)
    assert got.iloc[0]["duration_days"] == 90


def test_balance_sheet_facts_are_instant(dual_period_lake: Lake) -> None:
    got = dual_period_lake.view(date(2023, 9, 1)).latest_fundamentals(
        ["AAPL"], ["Assets"]
    )
    assert got.iloc[0]["period_kind"] == "instant"
    assert got.iloc[0]["period_start"] is None


def test_unknown_period_kind_is_rejected(dual_period_lake: Lake) -> None:
    with pytest.raises(ValueError, match="unknown period_kind"):
        dual_period_lake.view(date(2023, 9, 1)).latest_fundamentals(
            ["AAPL"], period_kind="monthly"
        )


def test_view_exposes_duration_columns(lake: Lake) -> None:
    """Agents need these to tell a ratio's numerator from its denominator."""
    facts = lake.view(date(2023, 8, 15)).fundamentals(["AAPL"])
    assert {"duration_days", "period_kind"} <= set(facts.columns)


def test_same_tag_in_two_taxonomies_is_not_deduped(tmp_path) -> None:
    """us-gaap and dei can publish one tag name; they are not restatements."""
    lake = Lake(tmp_path / "taxonomies")
    lake.write(
        FUNDAMENTALS,
        pd.DataFrame(
            [
                _fact(tag="SharesOutstanding", taxonomy="us-gaap", value=1.0),
                _fact(tag="SharesOutstanding", taxonomy="dei", value=2.0),
            ],
            columns=FUNDAMENTAL_COLUMNS,
        ),
    )
    got = lake.view(date(2023, 6, 1)).fundamentals(["AAPL"], ["SharesOutstanding"])
    assert sorted(got["taxonomy"]) == ["dei", "us-gaap"]


def test_missing_dates_are_None_regardless_of_the_other_rows(tmp_path) -> None:
    """The NA sentinel must not depend on how many rows happen to be null.

    `.dt.date` keeps datetime64 (and NaT) when every row is null, but yields
    object (and None) when any row holds a real date. Agents comparing with
    `is None` would then pass on one query and fail on the next.
    """
    lake = Lake(tmp_path / "sentinel")
    lake.write(
        FUNDAMENTALS,
        pd.DataFrame(
            [
                _fact(tag="Assets", period_start=None, period_end=date(2023, 3, 31)),
                _fact(tag="NetIncomeLoss", period_start=date(2023, 1, 1)),
            ],
            columns=FUNDAMENTAL_COLUMNS,
        ),
    )
    view = lake.view(date(2023, 6, 1))

    only_nulls = view.latest_fundamentals(["AAPL"], ["Assets"])
    assert only_nulls.iloc[0]["period_start"] is None, "all-null column kept NaT"

    mixed = view.latest_fundamentals(["AAPL"], ["Assets", "NetIncomeLoss"])
    missing = mixed.loc[mixed["tag"] == "Assets", "period_start"].iloc[0]
    assert missing is None
