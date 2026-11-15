"""Daily OHLCV ingest via yfinance.

Two honesty notes that belong next to the code, not in a changelog.

1. `adj_close` is adjusted retroactively — as of today, not as of the bar's own
   date. That is what you want for computing returns (it is a clean total-return
   series) and wrong if you ever need the literal price printed on the tape that
   day. Use `close` for anything price-level, `adj_close` for returns.

2. yfinance is an unofficial scraper of a free endpoint. It breaks, it rate
   limits, and it silently returns short frames. Every ingest therefore reports
   what it actually got per ticker so a partial fetch is visible rather than
   quietly becoming a gap in the lake.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import pandas as pd

from hedge_fund.data.lake import PRICE_COLUMNS, PRICES, Lake

_BATCH_SIZE = 40
_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
}
_FIELD_KEYS = frozenset(_FIELD_MAP)


class IncompleteFetchError(RuntimeError):
    """A full refresh fetched too little to justify destroying the existing table."""


@dataclass(frozen=True)
class IngestReport:
    requested: int
    returned: int
    rows_fetched: int
    rows_in_lake: int
    missing: tuple[str, ...]
    first: date | None
    last: date | None

    @property
    def ok(self) -> bool:
        """False when nothing at all came back for a non-empty request."""
        return self.requested == 0 or self.returned > 0

    def describe(self) -> str:
        # Two different populations, named so they cannot be read as one: what
        # this run fetched, and what the lake holds afterwards. An earlier
        # version printed the lake total next to the fetched span, so a daily
        # top-up of 2 bars looked like a 2.5M-row fetch.
        lines = [
            f"prices: fetched {self.rows_fetched:,} rows for "
            f"{self.returned}/{self.requested} tickers; lake now {self.rows_in_lake:,} rows",
        ]
        if self.first and self.last:
            lines.append(f"  fetched span: {self.first} .. {self.last}")
        if self.missing:
            shown = ", ".join(self.missing[:10])
            more = f" (+{len(self.missing) - 10} more)" if len(self.missing) > 10 else ""
            lines.append(f"  NO DATA for {len(self.missing)}: {shown}{more}")
        return "\n".join(lines)


class AmbiguousFrameError(ValueError):
    """yfinance returned a frame whose rows cannot be attributed to a ticker.

    Raised instead of guessing. `ingest_prices` recovers by refetching the
    batch one ticker at a time, which makes attribution unambiguous.
    """


def _field_level(columns: pd.MultiIndex) -> int:
    """Which level of a yfinance MultiIndex holds the OHLCV field names.

    Do NOT infer this by testing whether level 0 contains a requested ticker:
    `LOW` (Lowe's, an S&P 500 member) and `OPEN` (Opendoor) are real tickers
    that are also field names, so that test reports the wrong level and the
    whole batch is silently dropped.

    Counting field-name matches per level is robust instead, because the field
    level matches the full OHLCV set (6) while a ticker level can only match
    through those rare collisions (1-2).
    """
    scores = [
        len({str(v).strip().lower() for v in columns.get_level_values(level)} & _FIELD_KEYS)
        for level in range(columns.nlevels)
    ]
    return int(max(range(len(scores)), key=lambda level: scores[level]))


def _normalise(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Flatten whatever shape yfinance returned into tidy long rows.

    Raises AmbiguousFrameError when a multi-ticker request comes back without a
    ticker level, since there is then no way to tell which company the bars
    belong to.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    frames: list[pd.DataFrame] = []

    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker" gives (ticker, field); some versions give (field, ticker).
        ticker_level = 1 - _field_level(raw.columns)
        for label in raw.columns.get_level_values(ticker_level).unique():
            block = raw.xs(label, axis=1, level=ticker_level).copy()
            block["ticker"] = str(label).upper()
            frames.append(block)
    elif len(tickers) == 1:
        block = raw.copy()
        block["ticker"] = str(tickers[0]).upper()
        frames.append(block)
    else:
        # A flat frame for several tickers means only one of them resolved, and
        # nothing in the frame says which. Stamping tickers[0] here would file
        # one company's prices under another's symbol — silent, permanent
        # cross-contamination of the lake.
        raise AmbiguousFrameError(
            f"flat frame returned for {len(tickers)} tickers "
            f"({', '.join(tickers[:5])}...); cannot attribute rows"
        )

    out = []
    for block in frames:
        block = block.reset_index()
        rename = {}
        for column in block.columns:
            key = str(column).strip().lower()
            if key in _FIELD_MAP:
                rename[column] = _FIELD_MAP[key]
            elif key in {"date", "datetime", "index"}:
                rename[column] = "date"
        block = block.rename(columns=rename)
        if "date" not in block.columns or "close" not in block.columns:
            continue
        if "adj_close" not in block.columns:
            # auto_adjust=True collapses them; close already is the adjusted series.
            block["adj_close"] = block["close"]
        for column in PRICE_COLUMNS:
            if column not in block.columns:
                block[column] = pd.NA
        block = block[PRICE_COLUMNS]
        out.append(block)

    if not out:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    frame = pd.concat(out, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.date
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    return frame.drop_duplicates(subset=["ticker", "date"], keep="last")


def ingest_prices(
    lake: Lake,
    tickers: Sequence[str],
    start: date,
    end: date | None = None,
    full_refresh: bool = False,
    min_coverage: float = 0.95,
) -> IngestReport:
    """Fetch bars and merge them into the lake.

    `full_refresh` replaces the whole table instead of upserting, and is
    refused (raising IncompleteFetchError, table untouched) when fewer than
    `min_coverage` of the requested tickers came back.
    """
    import yfinance as yf

    tickers = [t.upper() for t in tickers]
    end = end or date.today()
    # yfinance treats `end` as exclusive.
    fetch_end = end + timedelta(days=1)

    def fetch(batch: Sequence[str]) -> pd.DataFrame:
        return yf.download(
            tickers=list(batch),
            start=start.isoformat(),
            end=fetch_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

    collected: list[pd.DataFrame] = []
    for index in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[index : index + _BATCH_SIZE]
        try:
            normalised = _normalise(fetch(batch), batch)
        except AmbiguousFrameError:
            # Only one ticker in the batch resolved and the frame does not say
            # which. Refetch singly so each frame has exactly one owner.
            for ticker in batch:
                single = _normalise(fetch([ticker]), [ticker])
                if not single.empty:
                    collected.append(single)
            continue
        if not normalised.empty:
            collected.append(normalised)

    # One retry, one ticker at a time, for anything a batch came back without.
    # yfinance rate-limits silently: a 40-ticker batch can return a frame that
    # simply omits a few of them, with no error raised. The report would list
    # them as NO DATA — true, but useless when a single refetch fixes it. On
    # the first 501-ticker run, 27 established S&P names (AME, DLR, HAL, ETR
    # among them) went missing this way, and every one fetched cleanly alone.
    fetched = {t for f in collected for t in f["ticker"].unique()}
    for ticker in [t for t in tickers if t not in fetched]:
        single = _normalise(fetch([ticker]), [ticker])
        if not single.empty:
            collected.append(single)

    frame = (
        pd.concat(collected, ignore_index=True)
        if collected
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )

    returned = sorted(frame["ticker"].unique()) if not frame.empty else []
    missing = tuple(sorted(set(tickers) - set(returned)))
    coverage = len(returned) / len(tickers) if tickers else 1.0

    if full_refresh and lake.exists(PRICES):
        # A full refresh replaces the table, so a batch that yfinance silently
        # dropped would take 20 years of those tickers' history with it. Refuse
        # unless the fetch is near-complete; the existing table is untouched.
        if coverage < min_coverage:
            raise IncompleteFetchError(
                f"full_refresh fetched only {len(returned)}/{len(tickers)} tickers "
                f"({coverage:.0%}, threshold {min_coverage:.0%}); existing "
                f"{PRICES} left untouched. Missing: {', '.join(missing[:10])}"
                f"{' ...' if len(missing) > 10 else ''}. Re-run, or pass a lower "
                "min_coverage if the gap is expected."
            )

    if full_refresh or not lake.exists(PRICES):
        lake.write(PRICES, frame.sort_values(["ticker", "date"], ignore_index=True))
        in_lake = len(frame)
    else:
        _, in_lake = lake.upsert(PRICES, frame, keys=["ticker", "date"])

    return IngestReport(
        requested=len(tickers),
        returned=len(returned),
        rows_fetched=len(frame),
        rows_in_lake=in_lake,
        missing=missing,
        first=frame["date"].min() if not frame.empty else None,
        last=frame["date"].max() if not frame.empty else None,
    )
