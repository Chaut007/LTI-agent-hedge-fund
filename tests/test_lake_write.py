"""The write path and the shape guarantees of the read path.

`Lake.upsert` is the only way incremental ingest touches disk, and it was
untested. The dtype-parity test exists because three separate bugs this
session were the same bug: a column whose type depended on what else happened
to be in the result. Pinning cold == warm closes that class.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.data.lake import (
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTAL_VIEW_COLUMNS,
    FUNDAMENTALS,
    MACRO_COLUMNS,
    MACRO,
    PRICE_COLUMNS,
    PRICES,
    AsofView,
    Lake,
)

ASOF = date(2024, 3, 1)


def _bar(ticker: str, d: date, close: float) -> dict:
    return {
        "ticker": ticker,
        "date": d,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 1000.0,
    }


def _bars(ticker: str, start: date, n: int) -> list[dict]:
    """n consecutive *weekday* bars starting at `start`."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(_bar(ticker, d, 100.0 + len(out)))
        d = date.fromordinal(d.toordinal() + 1)
    return out


# -- upsert ------------------------------------------------------------------


def test_upsert_creates_the_table_when_absent(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    frame = pd.DataFrame(_bars("AAPL", date(2024, 1, 1), 3), columns=PRICE_COLUMNS)
    _, total = lake.upsert(PRICES, frame, keys=["ticker", "date"])
    assert total == 3
    assert lake.exists(PRICES)


def test_upsert_merges_and_newcomers_win_on_keys(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    first = pd.DataFrame(_bars("AAPL", date(2024, 1, 1), 3), columns=PRICE_COLUMNS)
    lake.upsert(PRICES, first, keys=["ticker", "date"])

    overlap_day = first["date"].iloc[-1]
    second = pd.DataFrame(
        [_bar("AAPL", overlap_day, 999.0), *_bars("AAPL", date(2024, 1, 8), 2)],
        columns=PRICE_COLUMNS,
    )
    _, total = lake.upsert(PRICES, second, keys=["ticker", "date"])

    stored = pd.read_parquet(lake.path(PRICES))
    assert total == 5, "overlapping key should replace, not duplicate"
    assert stored.loc[stored["date"] == overlap_day, "close"].item() == 999.0


def test_upsert_output_is_sorted_by_keys(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    frame = pd.DataFrame(
        [
            _bar("MSFT", date(2024, 1, 2), 1.0),
            _bar("AAPL", date(2024, 1, 3), 1.0),
            _bar("AAPL", date(2024, 1, 2), 1.0),
        ],
        columns=PRICE_COLUMNS,
    )
    lake.upsert(PRICES, frame, keys=["ticker", "date"])
    stored = pd.read_parquet(lake.path(PRICES))
    assert stored[["ticker", "date"]].values.tolist() == [
        ["AAPL", date(2024, 1, 2)],
        ["AAPL", date(2024, 1, 3)],
        ["MSFT", date(2024, 1, 2)],
    ]


def test_upsert_with_empty_frame_leaves_existing_table_untouched(tmp_path) -> None:
    """A run where every fetch failed must not rewrite the lake."""
    lake = Lake(tmp_path / "lake")
    frame = pd.DataFrame(_bars("AAPL", date(2024, 1, 1), 3), columns=PRICE_COLUMNS)
    lake.upsert(PRICES, frame, keys=["ticker", "date"])
    mtime_before = lake.path(PRICES).stat().st_mtime_ns

    _, total = lake.upsert(
        PRICES, pd.DataFrame(columns=PRICE_COLUMNS), keys=["ticker", "date"]
    )

    assert total == 3
    assert lake.path(PRICES).stat().st_mtime_ns == mtime_before, "file was rewritten"


def test_upsert_with_empty_frame_on_cold_lake_creates_empty_table(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    _, total = lake.upsert(
        PRICES, pd.DataFrame(columns=PRICE_COLUMNS), keys=["ticker", "date"]
    )
    assert total == 0
    assert lake.exists(PRICES)


def test_upsert_preserves_value_dtype_across_round_trips(tmp_path) -> None:
    """An all-integer batch must not flip the stored column to int64."""
    lake = Lake(tmp_path / "lake")

    def fact(end: date, value: float) -> dict:
        return {
            "ticker": "AAPL",
            "cik": 320193,
            "taxonomy": "us-gaap",
            "tag": "Assets",
            "unit": "USD",
            "period_start": None,
            "period_end": end,
            "value": value,
            "form": "10-Q",
            "filed": end,
            "accn": f"a-{end.isoformat()}",
            "fy": 2024,
            "fp": "Q1",
        }

    first = pd.DataFrame([fact(date(2024, 1, 1), 1.5)], columns=FUNDAMENTAL_COLUMNS)
    first["value"] = first["value"].astype("float64")
    lake.upsert(FUNDAMENTALS, first, keys=["accn"])

    second = pd.DataFrame([fact(date(2024, 2, 1), 2)], columns=FUNDAMENTAL_COLUMNS)
    second["value"] = second["value"].astype("float64")
    lake.upsert(FUNDAMENTALS, second, keys=["accn"])

    assert pd.read_parquet(lake.path(FUNDAMENTALS))["value"].dtype == "float64"


# -- cold == warm: the shape of a read must not depend on whether data exists --


@pytest.fixture()
def warm(tmp_path) -> Lake:
    lake = Lake(tmp_path / "warm")
    lake.write(
        PRICES, pd.DataFrame(_bars("AAPL", date(2024, 1, 1), 5), columns=PRICE_COLUMNS)
    )
    lake.write(
        FUNDAMENTALS,
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "cik": 320193,
                    "taxonomy": "us-gaap",
                    "tag": "NetIncomeLoss",
                    "unit": "USD",
                    "period_start": date(2023, 10, 1),
                    "period_end": date(2023, 12, 31),
                    "value": 1.0,
                    "form": "10-K",
                    "filed": date(2024, 2, 1),
                    "accn": "a1",
                    "fy": 2023,
                    "fp": "FY",
                },
                {
                    # An instant fact: no period_start, so duration_days is
                    # NULL. Its presence used to flip the whole column from
                    # int64 to float64 — a dtype that depended on which rows
                    # the query happened to select.
                    "ticker": "AAPL",
                    "cik": 320193,
                    "taxonomy": "us-gaap",
                    "tag": "Assets",
                    "unit": "USD",
                    "period_start": None,
                    "period_end": date(2023, 12, 31),
                    "value": 2.0,
                    "form": "10-K",
                    "filed": date(2024, 2, 1),
                    "accn": "a1",
                    "fy": None,
                    "fp": "FY",
                },
            ],
            columns=FUNDAMENTAL_COLUMNS,
        ),
    )
    lake.write(
        MACRO,
        pd.DataFrame(
            [{"series_id": "DGS10", "date": date(2024, 1, 2), "value": 4.0}],
            columns=MACRO_COLUMNS,
        ),
    )
    return lake


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {c: str(frame[c].dtype) for c in frame.columns}


def test_cold_and_warm_reads_have_identical_dtypes(tmp_path, warm: Lake) -> None:
    """Previously a cold read returned object for every column.

    `close.mean()` on an empty object column returns nan instead of raising,
    so an agent run before any ingest produced nan scores that the aggregator
    accepted as real numbers.
    """
    cold = Lake(tmp_path / "cold").view(ASOF)
    hot = warm.view(ASOF)

    for name, cold_frame, warm_frame in [
        ("prices", cold.prices(["AAPL"]), hot.prices(["AAPL"])),
        ("fundamentals", cold.fundamentals(["AAPL"]), hot.fundamentals(["AAPL"])),
        (
            "latest_fundamentals",
            cold.latest_fundamentals(["AAPL"]),
            hot.latest_fundamentals(["AAPL"]),
        ),
        ("macro", cold.macro(["DGS10"]), hot.macro(["DGS10"])),
    ]:
        assert list(cold_frame.columns) == list(warm_frame.columns), name
        assert _dtypes(cold_frame) == _dtypes(warm_frame), (
            f"{name}: cold {_dtypes(cold_frame)} != warm {_dtypes(warm_frame)}"
        )


def test_cold_numeric_columns_are_numeric(tmp_path) -> None:
    cold = Lake(tmp_path / "cold").view(ASOF)
    bars = cold.prices(["AAPL"])
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        assert bars[column].dtype.kind == "f", column
    assert cold.fundamentals(["AAPL"])["value"].dtype.kind == "f"


# -- prices windowing: bars are exact, calendar days are calendar days ---------


def test_bars_returns_exactly_n_trading_bars_per_ticker(warm: Lake) -> None:
    """252 bars must be 252 bars, not whatever 252 calendar days happen to hold."""
    lake = warm
    extra = pd.DataFrame(_bars("MSFT", date(2024, 1, 1), 5), columns=PRICE_COLUMNS)
    lake.upsert(PRICES, extra, keys=["ticker", "date"])

    out = lake.view(ASOF).prices(["AAPL", "MSFT"], bars=3)
    counts = out.groupby("ticker").size().to_dict()
    assert counts == {"AAPL": 3, "MSFT": 3}
    # and they are the *most recent* three
    aapl = out[out["ticker"] == "AAPL"]["date"].tolist()
    assert aapl == sorted(aapl)
    assert aapl[-1] == date(2024, 1, 5)


def test_bars_window_still_respects_asof(warm: Lake) -> None:
    out = warm.view(date(2024, 1, 3)).prices(["AAPL"], bars=100)
    assert out["date"].max() == date(2024, 1, 3)
    assert len(out) == 3  # Jan 1, 2, 3 are the only bars <= asof


def test_calendar_days_is_a_date_window_not_a_bar_count(warm: Lake) -> None:
    out = warm.view(date(2024, 1, 5)).prices(["AAPL"], calendar_days=1)
    assert out["date"].tolist() == [date(2024, 1, 4), date(2024, 1, 5)]


def test_bars_and_calendar_days_together_are_rejected(warm: Lake) -> None:
    with pytest.raises(ValueError, match="not both"):
        warm.view(ASOF).prices(["AAPL"], bars=5, calendar_days=5)


def test_default_window_is_bars(warm: Lake) -> None:
    assert len(warm.view(ASOF).prices(["AAPL"])) == 5


# -- the view must not carry a route to unbounded data -----------------------


def test_view_does_not_reference_its_lake(warm: Lake) -> None:
    """`ctx.view._lake.query(...)` used to read the entire future."""
    view = warm.view(date(2000, 1, 1))  # pinned long before any data
    assert not hasattr(view, "_lake")
    assert not hasattr(view, "query")
    assert not hasattr(view, "unbounded_query")
    # the only state it carries is a directory and a date
    assert set(AsofView.__slots__) == {"_dir", "_asof"}


def test_unbounded_query_lives_only_on_the_lake(warm: Lake) -> None:
    rows = warm.unbounded_query(
        "SELECT COUNT(*) AS n FROM read_parquet(?)", [str(warm.path(PRICES))]
    )
    assert int(rows["n"].iloc[0]) == 5
    assert warm.view(date(2000, 1, 1)).prices(["AAPL"]).empty


def test_status_omits_the_fred_caveat_when_the_lake_holds_no_macro(tmp_path) -> None:
    """A caveat about data that is absent reads as a caveat about data present.

    `hf status` printed "revised FRED series are not point-in-time:
    CPIAUCSL, UNRATE" on a lake with an empty macro table, and that line is
    reprinted verbatim in the project report — telling a reader the series
    were there and merely imperfect.
    """
    from hedge_fund.cli import _revisable_series_held

    assert _revisable_series_held(Lake(tmp_path / "empty")) == []


def test_status_names_the_revisable_series_the_lake_does_hold(tmp_path) -> None:
    from hedge_fund.cli import _revisable_series_held
    from hedge_fund.data.lake import MACRO, MACRO_COLUMNS

    lake = Lake(tmp_path / "lake")
    lake.write(
        MACRO,
        pd.DataFrame(
            [
                {"series_id": "UNRATE", "date": date(2024, 1, 1), "value": 3.7},
                {"series_id": "DGS10", "date": date(2024, 1, 1), "value": 4.0},
            ],
            columns=MACRO_COLUMNS,
        ),
    )
    # DGS10 is never revised, so it is not a caveat.
    assert _revisable_series_held(lake) == ["UNRATE"]
