"""`ingest_prices` with yfinance mocked out.

The network function was untested because it needs the network. It does not:
`yfinance.download` is a module attribute, and the only thing the ingest
needs from it is a frame of the right shape. Mocking it pins the two recovery
paths — the ambiguous-frame refetch and the omitted-ticker retry — that a
real yfinance only exercises when it is misbehaving, which is exactly when
nobody is watching.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import yfinance

from hedge_fund.data.lake import PRICES, Lake
from hedge_fund.data.prices import IncompleteFetchError, ingest_prices

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DAYS = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])


def _yf_frame(tickers: list[str]) -> pd.DataFrame:
    """What `group_by="ticker"` returns: (ticker, field) columns, or empty."""
    if not tickers:
        return pd.DataFrame()
    columns = pd.MultiIndex.from_product([tickers, FIELDS])
    frame = pd.DataFrame(10.0, index=DAYS, columns=columns)
    frame.index.name = "Date"
    return frame


def _flat_frame() -> pd.DataFrame:
    frame = pd.DataFrame(10.0, index=DAYS, columns=FIELDS)
    frame.index.name = "Date"
    return frame


def _install(monkeypatch, behaviour):
    """Replace yfinance.download with `behaviour(tickers) -> frame`, recording calls."""
    calls: list[list[str]] = []

    def fake_download(tickers, **kwargs):
        calls.append(list(tickers))
        return behaviour(list(tickers))

    monkeypatch.setattr(yfinance, "download", fake_download)
    return calls


def _run(lake: Lake, tickers: list[str], **kwargs):
    return ingest_prices(lake, tickers, start=date(2024, 1, 1), end=date(2024, 1, 5), **kwargs)


# -- the happy path ---------------------------------------------------------------


def test_full_batch_lands_every_ticker(tmp_path, monkeypatch) -> None:
    calls = _install(monkeypatch, _yf_frame)
    report = _run(Lake(tmp_path / "lake"), ["AME", "DLR", "HAL"])
    assert report.returned == 3
    assert report.missing == ()
    assert report.rows_fetched == 9
    assert calls == [["AME", "DLR", "HAL"]], "one batch, no retries needed"


# -- silent omission by a rate-limited batch ----------------------------------------


def test_ticker_a_batch_omitted_is_refetched_alone(tmp_path, monkeypatch) -> None:
    """yfinance returns the batch frame minus DLR, with no error. Retry singly."""

    def behaviour(tickers):
        if len(tickers) > 1:
            return _yf_frame([t for t in tickers if t != "DLR"])
        return _yf_frame(tickers)

    calls = _install(monkeypatch, behaviour)
    report = _run(Lake(tmp_path / "lake"), ["AME", "DLR", "HAL"])

    assert report.missing == (), "DLR must be recovered, not reported as NO DATA"
    assert report.returned == 3
    assert ["DLR"] in calls, "the retry must be a single-ticker fetch"
    assert calls.count(["DLR"]) == 1, "exactly one retry, not a loop"


def test_genuinely_missing_ticker_stays_missing_after_one_retry(tmp_path, monkeypatch) -> None:
    def behaviour(tickers):
        return _yf_frame([t for t in tickers if t != "ZZZZ"])

    calls = _install(monkeypatch, behaviour)
    report = _run(Lake(tmp_path / "lake"), ["AME", "ZZZZ"])

    assert report.missing == ("ZZZZ",)
    assert report.returned == 1
    assert calls.count(["ZZZZ"]) == 1
    assert report.ok


def test_entirely_empty_fetch_is_reported_not_ok(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, lambda tickers: pd.DataFrame())
    report = _run(Lake(tmp_path / "lake"), ["AME", "DLR"])
    assert report.returned == 0
    assert report.missing == ("AME", "DLR")
    assert not report.ok


# -- ambiguous flat frame -----------------------------------------------------------


def test_flat_frame_for_a_batch_triggers_per_ticker_refetch(tmp_path, monkeypatch) -> None:
    """Only one of the batch resolved and yfinance dropped the ticker level."""

    def behaviour(tickers):
        if len(tickers) > 1:
            return _flat_frame()
        return _yf_frame(tickers) if tickers == ["AME"] else pd.DataFrame()

    calls = _install(monkeypatch, behaviour)
    report = _run(Lake(tmp_path / "lake"), ["AME", "ZZZZ"])

    assert report.returned == 1 and report.missing == ("ZZZZ",)
    assert ["AME"] in calls and ["ZZZZ"] in calls, "each name refetched on its own"
    stored = pd.read_parquet(Lake(tmp_path / "lake").path(PRICES))
    assert set(stored["ticker"]) == {"AME"}, "nothing was filed under the wrong ticker"


# -- full refresh guard, through the real entry point --------------------------------


def test_full_refresh_refuses_a_partial_fetch_and_keeps_the_table(tmp_path, monkeypatch) -> None:
    lake = Lake(tmp_path / "lake")
    _install(monkeypatch, _yf_frame)
    _run(lake, ["AME", "DLR", "HAL"])
    before = lake.path(PRICES).stat().st_mtime_ns

    _install(monkeypatch, lambda tickers: _yf_frame([t for t in tickers if t == "AME"]))
    with pytest.raises(IncompleteFetchError, match="left untouched"):
        _run(lake, ["AME", "DLR", "HAL"], full_refresh=True)

    assert lake.path(PRICES).stat().st_mtime_ns == before
    assert len(pd.read_parquet(lake.path(PRICES))) == 9


def test_incremental_run_upserts_rather_than_replacing(tmp_path, monkeypatch) -> None:
    lake = Lake(tmp_path / "lake")
    _install(monkeypatch, _yf_frame)
    first = _run(lake, ["AME", "DLR"])
    second = _run(lake, ["HAL"])
    assert first.rows_in_lake == 6
    assert second.rows_fetched == 3
    assert second.rows_in_lake == 9, "the lake total grows; the fetch count does not"
