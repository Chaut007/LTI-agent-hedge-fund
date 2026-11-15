"""Coverage for `_normalise`, the function that absorbs yfinance's shape churn.

This function exists only because yfinance changes its column layout between
versions and between single- and multi-ticker requests. That makes it the most
defect-prone code in the data layer and the most likely to break on a dependency
bump, so the shapes it has to survive are pinned here rather than discovered in
production.

Two of these tests cover bugs found in review: a ticker whose symbol is also an
OHLCV field name, and a flat frame returned for a multi-ticker request.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.data.prices import AmbiguousFrameError, _field_level, _normalise

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DATES = pd.to_datetime(["2024-01-02", "2024-01-03"])


def _frame(columns: pd.MultiIndex | list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(1.0, index=DATES, columns=columns)
    frame.index.name = "Date"
    return frame


def ticker_first(tickers: list[str]) -> pd.DataFrame:
    """What `group_by="ticker"` returns: (ticker, field)."""
    return _frame(pd.MultiIndex.from_product([tickers, FIELDS]))


def field_first(tickers: list[str]) -> pd.DataFrame:
    """What some yfinance versions return instead: (field, ticker)."""
    return _frame(pd.MultiIndex.from_product([FIELDS, tickers]))


# -- the field/ticker level must be identified by field names, not tickers ----


@pytest.mark.parametrize("tickers", [["AAPL", "MSFT"], ["LOW", "AAPL"], ["OPEN", "LOW"]])
def test_level_detection_survives_ticker_field_name_collisions(tickers) -> None:
    """LOW (Lowe's) and OPEN (Opendoor) are real symbols AND field names.

    Detecting the layout by asking "is a requested ticker in level 0" reports
    the wrong level for these, which silently dropped the entire batch.
    """
    assert _field_level(ticker_first(tickers).columns) == 1
    assert _field_level(field_first(tickers).columns) == 0


@pytest.mark.parametrize("layout", [ticker_first, field_first])
@pytest.mark.parametrize("tickers", [["AAPL", "MSFT"], ["LOW", "AAPL"], ["OPEN", "LOW"]])
def test_every_requested_ticker_survives_both_layouts(layout, tickers) -> None:
    out = _normalise(layout(tickers), tickers)
    assert sorted(out["ticker"].unique()) == sorted(tickers)
    assert len(out) == 2 * len(tickers)


def test_low_batch_is_not_dropped() -> None:
    """Direct regression: the reported failure returned an EMPTY frame."""
    out = _normalise(field_first(["LOW", "AAPL"]), ["LOW", "AAPL"])
    assert not out.empty
    assert "LOW" in set(out["ticker"])


# -- a frame that cannot be attributed must not be guessed at ----------------


def test_flat_frame_for_one_ticker_is_attributed() -> None:
    out = _normalise(_frame(FIELDS), ["AAPL"])
    assert set(out["ticker"]) == {"AAPL"}


def test_flat_frame_for_several_tickers_refuses_to_guess() -> None:
    """Previously stamped tickers[0], filing AAPL's prices under ZZZZ."""
    with pytest.raises(AmbiguousFrameError, match="cannot attribute"):
        _normalise(_frame(FIELDS), ["ZZZZ", "AAPL"])


# -- field mapping and cleaning -----------------------------------------------


def test_columns_are_renamed_and_ordered() -> None:
    out = _normalise(ticker_first(["AAPL"]), ["AAPL"])
    assert list(out.columns) == [
        "ticker", "date", "open", "high", "low", "close", "adj_close", "volume",
    ]
    assert isinstance(out["date"].iloc[0], date)


def test_missing_adj_close_falls_back_to_close() -> None:
    """auto_adjust=True collapses the two columns; close is already adjusted."""
    columns = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    frame = _frame(columns)
    frame[("AAPL", "Close")] = [10.0, 11.0]
    out = _normalise(frame, ["AAPL"])
    assert out["adj_close"].tolist() == out["close"].tolist() == [10.0, 11.0]


def test_nonpositive_and_unparseable_closes_are_dropped() -> None:
    frame = ticker_first(["AAPL"])
    frame[("AAPL", "Close")] = [0.0, 12.0]
    out = _normalise(frame, ["AAPL"])
    assert out["close"].tolist() == [12.0]


def test_duplicate_dates_collapse_to_the_last_row() -> None:
    frame = _frame(FIELDS)
    frame = pd.concat([frame, frame.tail(1)])
    out = _normalise(frame, ["AAPL"])
    assert out["date"].is_unique


def test_empty_and_none_inputs_return_the_empty_shape() -> None:
    for raw in (None, pd.DataFrame()):
        out = _normalise(raw, ["AAPL"])
        assert out.empty
        assert list(out.columns) == [
            "ticker", "date", "open", "high", "low", "close", "adj_close", "volume",
        ]
