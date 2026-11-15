"""The lake: parquet files on disk, queried through DuckDB.

Three things matter here.

1. `Lake.view(asof)` is the only sanctioned read path. It returns an `AsofView`
   whose every query carries its own `<= asof` clause, so a caller cannot see
   data that did not exist yet. Prices clamp on the bar date; fundamentals clamp
   on the SEC *filing* date, not the period end — a Q1 report is not knowable in
   April just because the quarter ended in March.

   The view deliberately does NOT hold a reference to its `Lake`. An earlier
   version did, and `ctx.view._lake.query(...)` reached the whole future in one
   attribute access. Now the view knows only a directory and a date, and the
   unbounded query lives on `Lake` under a name that says what it is. This is
   Python, so a caller who imports duckdb and opens the parquet file themselves
   can still read anything; what the design prevents is reaching unbounded data
   *through the object an agent is handed*.

2. Restatements are resolved, not stacked. A company can refile the same period
   with different numbers; the view keeps the latest filing that was visible as
   of the query date, which is what you would actually have had in hand.

3. Period length is part of the key. A 10-Q reports both a 3-month and a
   year-to-date figure for the same tag ending the same day; the view labels
   each with `period_kind` and `duration_days` so they can never be confused.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

PRICES = "prices.parquet"
FUNDAMENTALS = "fundamentals.parquet"
MACRO = "macro.parquet"
UNIVERSE = "universe.parquet"

PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
FUNDAMENTAL_COLUMNS = [
    "ticker",
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "period_start",
    "period_end",
    "value",
    "form",
    "filed",
    "accn",
    "fy",
    "fp",
]
MACRO_COLUMNS = ["series_id", "date", "value"]
UNIVERSE_COLUMNS = ["ticker", "cik", "name"]

# Derived on read, not stored: how long a period the fact covers. Without this
# a 3-month and a year-to-date fact ending the same day are indistinguishable,
# and an earnings-yield agent silently mixes them.
FUNDAMENTAL_VIEW_COLUMNS = FUNDAMENTAL_COLUMNS + ["duration_days", "period_kind"]
PERIOD_KINDS = ("instant", "quarterly", "annual", "other")

DEFAULT_PRICE_BARS = 400

# Dtypes a cold (file-less) read must return so that agent code behaves the
# same whether or not an ingest has run. Previously every column came back as
# object, so `close.mean()` on an empty lake returned nan instead of raising,
# and the nan flowed into the aggregator as a real score. `_assert_matches_warm`
# in the tests pins these against what DuckDB actually produces.
_DTYPES: dict[str, str] = {
    # strings: pandas 3 returns the dedicated `str` dtype from DuckDB
    "ticker": "str",
    "taxonomy": "str",
    "tag": "str",
    "unit": "str",
    "form": "str",
    "accn": "str",
    "fp": "str",
    "period_kind": "str",
    "series_id": "str",
    "name": "str",
    # dates: object, because they hold datetime.date (see _shape)
    "date": "object",
    "period_start": "object",
    "period_end": "object",
    "filed": "object",
    # floats
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "float64",
    "value": "float64",
    # nullable integers: these can be NULL (instant facts have no duration,
    # fy is not always reported). Left to DuckDB they come back int64 when the
    # result happens to have no nulls and float64 when it does — a dtype that
    # depends on which rows were selected. Pinned to Int64 on every read.
    "cik": "Int64",
    "fy": "Int64",
    "duration_days": "Int64",
}
_INT_COLUMNS = ("cik", "fy", "duration_days")

_PERIOD_KIND_SQL = """
    CASE
        WHEN period_start IS NULL THEN 'instant'
        WHEN date_diff('day', period_start, period_end) BETWEEN 60 AND 120
            THEN 'quarterly'
        WHEN date_diff('day', period_start, period_end) BETWEEN 300 AND 400
            THEN 'annual'
        ELSE 'other'
    END AS period_kind
"""
_DURATION_SQL = "date_diff('day', period_start, period_end) AS duration_days"

_TICKER_FILTER = "AND ticker IN (SELECT UNNEST(CAST(? AS VARCHAR[])))"
_TAG_FILTER = "AND tag IN (SELECT UNNEST(CAST(? AS VARCHAR[])))"
_SERIES_FILTER = "AND series_id IN (SELECT UNNEST(CAST(? AS VARCHAR[])))"
_KIND_FILTER = "AND period_kind IN (SELECT UNNEST(CAST(? AS VARCHAR[])))"

_DATE_COLUMNS = ("date", "period_start", "period_end", "filed")


# -- connection ----------------------------------------------------------------

_CONNECTION_LOCK = threading.Lock()
_CONNECTION: duckdb.DuckDBPyConnection | None = None


def _run(sql: str, params: Sequence[object]) -> pd.DataFrame:
    """Execute on one process-wide DuckDB connection.

    Opening a connection per query was the dominant cost of a backtest: tens of
    thousands of connect/teardown cycles over a loop that does nothing else.
    `read_parquet` still opens the file on each call, so a table rewritten by an
    ingest is visible to the very next query — there is no staleness to manage.
    The lock serialises calls; DuckDB connections are not safe to share
    concurrently.
    """
    global _CONNECTION
    with _CONNECTION_LOCK:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        return _CONNECTION.execute(sql, list(params)).fetch_df()


# -- frame helpers -------------------------------------------------------------


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype=_DTYPES.get(c, "object")) for c in columns}
    )


def _shape(frame: pd.DataFrame) -> pd.DataFrame:
    """Pin every view result to one dtype per column, regardless of its rows.

    Dates come back as `datetime.date`, never `Timestamp`; nullable integers
    come back as `Int64`. Applied to every read so the cold path (`_empty`)
    and the warm path agree exactly — `test_cold_and_warm_reads_have_identical_dtypes`
    holds them to it.

    DuckDB hands back pandas Timestamps, parquet round-trips can give either,
    and `Timestamp("2023-03-31") != date(2023, 3, 31)`. Agents compare these
    against calendar dates constantly, so normalise once here rather than
    letting both types circulate.

    Missing dates are always `None`. The `.astype(object)` is load-bearing:
    without it, `.dt.date` on a column where EVERY row is null keeps
    datetime64 dtype and the gaps stay `NaT`, while a column with at least one
    real date yields object dtype and `None`. That makes the sentinel depend on
    the other rows in the result set, so `row["period_start"] is None` would
    work for one query and silently fail for the next.
    """
    for column in _DATE_COLUMNS:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            frame[column] = parsed.dt.date.astype(object).where(parsed.notna(), None)
    # Nullable ints: see the _DTYPES note. Without this, duration_days is
    # int64 for a result with no instant facts and float64 for one with any.
    for column in _INT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


# -- the lake ------------------------------------------------------------------


class Lake:
    """Handle on a lake directory. Cheap to construct; owns no cursor."""

    def __init__(self, lake_dir: Path) -> None:
        self.dir = Path(lake_dir)

    # paths

    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    # writing

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        """Replace a table wholesale."""
        self.ensure_dir()
        target = self.path(name)
        frame.to_parquet(target, index=False)
        return target

    def upsert(
        self, name: str, frame: pd.DataFrame, keys: Sequence[str]
    ) -> tuple[Path, int]:
        """Merge `frame` into an existing table, newcomers winning on `keys`.

        Returns the path and the total row count afterwards. Used by the
        incremental ingests so a re-run tops up rather than refetching history.
        An empty `frame` leaves an existing table untouched rather than
        rewriting it for nothing.
        """
        self.ensure_dir()
        if frame.empty:
            if not self.exists(name):
                return self.write(name, frame), 0
            return self.path(name), self.row_count(name)
        if self.exists(name):
            existing = pd.read_parquet(self.path(name))
            combined = pd.concat([existing, frame], ignore_index=True)
        else:
            combined = frame.copy()
        combined = combined.drop_duplicates(subset=list(keys), keep="last")
        sort_cols = [c for c in keys if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, ignore_index=True)
        return self.write(name, combined), len(combined)

    # reading

    def view(self, asof: date) -> "AsofView":
        """The only sanctioned read path. Everything downstream goes through this."""
        return AsofView(self.dir, asof)

    def unbounded_query(self, sql: str, params: Sequence[object] = ()) -> pd.DataFrame:
        """Run SQL with NO asof bound. Ingest bookkeeping and tests only.

        Named so that its appearance in an agent is a review failure on sight.
        Not reachable from an `AsofView`.
        """
        return _run(sql, params)

    def row_count(self, name: str) -> int:
        if not self.exists(name):
            return 0
        out = _run("SELECT COUNT(*) AS n FROM read_parquet(?)", [str(self.path(name))])
        return int(out["n"].iloc[0])

    def universe(self) -> pd.DataFrame:
        if not self.exists(UNIVERSE):
            return _empty(UNIVERSE_COLUMNS)
        return pd.read_parquet(self.path(UNIVERSE))

    def coverage(self) -> pd.DataFrame:
        """One row per table: row count and date span. For `hf status`."""
        summary = []
        specs = [
            (PRICES, "date"),
            (FUNDAMENTALS, "filed"),
            (MACRO, "date"),
            (UNIVERSE, None),
        ]
        for name, date_col in specs:
            path = self.path(name)
            if not path.exists():
                summary.append({"table": name, "rows": 0, "first": None, "last": None})
                continue
            if date_col:
                sql = (
                    f"SELECT COUNT(*) AS rows, MIN({date_col}) AS first, "
                    f"MAX({date_col}) AS last FROM read_parquet(?)"
                )
            else:
                sql = (
                    "SELECT COUNT(*) AS rows, NULL AS first, NULL AS last "
                    "FROM read_parquet(?)"
                )
            out = _run(sql, [str(path)]).iloc[0].to_dict()
            summary.append({"table": name, **out})
        return pd.DataFrame(summary)


# -- the asof view -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsofView:
    """Read-only lake access pinned to a single date.

    Implements `hedge_fund.core.agent.MarketView`. Holds a directory and a
    date — never a `Lake` — so no unbounded query is reachable through it.
    """

    _dir: Path
    _asof: date

    @property
    def asof(self) -> date:
        return self._asof

    def _path(self, name: str) -> Path:
        return self._dir / name

    # prices

    def prices(
        self,
        tickers: Sequence[str] | None = None,
        bars: int | None = None,
        calendar_days: int | None = None,
    ) -> pd.DataFrame:
        """Daily bars with `date <= asof`, windowed by exactly one of:

        bars:           the last N trading bars per ticker. This is what a
                        factor wants — `bars=252` is twelve months of trading
                        regardless of weekends and holidays.
        calendar_days:  a calendar-day lookback. 252 calendar days is only
                        ~173 trading days; use this when the window really is
                        a date, not a bar count.

        Defaults to `bars=DEFAULT_PRICE_BARS`. The two are mutually exclusive
        because an agent passing 252 and silently getting 173 bars was the
        original bug; forcing the unit into the argument name removes it.
        """
        if bars is not None and calendar_days is not None:
            raise ValueError("prices(): pass bars or calendar_days, not both")
        if bars is None and calendar_days is None:
            bars = DEFAULT_PRICE_BARS
        if not self._path(PRICES).exists():
            return _empty(PRICE_COLUMNS)

        columns = ", ".join(PRICE_COLUMNS)
        params: list[object] = [str(self._path(PRICES)), self._asof]
        if bars is not None:
            if bars < 1:
                raise ValueError(f"prices(): bars must be >= 1, got {bars}")
            sql = f"""
                WITH visible AS (
                    SELECT {columns},
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker ORDER BY date DESC
                           ) AS _recency
                    FROM read_parquet(?)
                    WHERE date <= ?
                      {_TICKER_FILTER if tickers else ""}
                )
                SELECT {columns} FROM visible
                WHERE _recency <= ?
                ORDER BY ticker, date
            """
            if tickers:
                params.append(list(tickers))
            params.append(int(bars))
        else:
            if calendar_days < 0:
                raise ValueError(
                    f"prices(): calendar_days must be >= 0, got {calendar_days}"
                )
            start = self._asof - timedelta(days=int(calendar_days))
            sql = f"""
                SELECT {columns}
                FROM read_parquet(?)
                WHERE date <= ?
                  AND date >= ?
                  {_TICKER_FILTER if tickers else ""}
                ORDER BY ticker, date
            """
            params.append(start)
            if tickers:
                params.append(list(tickers))
        return _shape(_run(sql, params))

    # fundamentals

    def _visible_facts_sql(self, tickers, tags) -> tuple[str, list[object]]:
        """Facts filed on or before asof, latest restatement per period.

        `taxonomy` is part of the partition: us-gaap and dei can publish the
        same tag name, and they are different facts, not restatements of one
        another. `accn` ends the ORDER BY so a period reported twice on the
        same day resolves deterministically rather than by parquet row order.
        """
        sql = f"""
            WITH visible AS (
                SELECT * FROM read_parquet(?)
                WHERE filed <= ?
                  {_TICKER_FILTER if tickers else ""}
                  {_TAG_FILTER if tags else ""}
            ),
            deduped AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY ticker, taxonomy, tag, unit,
                                 period_start, period_end
                    ORDER BY filed DESC, accn DESC
                ) AS _rn
                FROM visible
            )
            SELECT {", ".join(FUNDAMENTAL_COLUMNS)},
                   {_DURATION_SQL},
                   {_PERIOD_KIND_SQL}
            FROM deduped WHERE _rn = 1
        """
        params: list[object] = [str(self._path(FUNDAMENTALS)), self._asof]
        if tickers:
            params.append(list(tickers))
        if tags:
            params.append(list(tags))
        return sql, params

    def fundamentals(
        self,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not self._path(FUNDAMENTALS).exists():
            return _empty(FUNDAMENTAL_VIEW_COLUMNS)
        sql, params = self._visible_facts_sql(tickers, tags)
        return _shape(
            _run(f"SELECT * FROM ({sql}) ORDER BY ticker, tag, period_end", params)
        )

    def latest_fundamentals(
        self,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        period_kind: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Freshest visible fact per (ticker, taxonomy, tag, unit, period_kind).

        `period_kind` is part of the key on purpose. A 10-Q reports both a
        3-month and a year-to-date figure for the same tag ending the same day,
        so ranking on period_end alone picks one arbitrarily and an agent
        computing earnings yield gets a quarterly numerator for some tickers
        and a 6- or 9-month one for others, differing by a factor of 2-3 with
        nothing to indicate which. Returning one row per kind makes the caller
        choose, and `duration_days` says exactly what each row covers.

        Pass `period_kind="quarterly"` (or a list) to narrow. Balance-sheet
        items carry no period and come back as "instant".
        """
        if not self._path(FUNDAMENTALS).exists():
            return _empty(FUNDAMENTAL_VIEW_COLUMNS)

        kinds: list[str] | None = None
        if period_kind is not None:
            kinds = [period_kind] if isinstance(period_kind, str) else list(period_kind)
            unknown = set(kinds) - set(PERIOD_KINDS)
            if unknown:
                raise ValueError(
                    f"unknown period_kind {sorted(unknown)}; "
                    f"expected any of {list(PERIOD_KINDS)}"
                )

        inner, params = self._visible_facts_sql(tickers, tags)
        sql = f"""
            WITH facts AS ({inner}),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY ticker, taxonomy, tag, unit, period_kind
                    ORDER BY period_end DESC, filed DESC, accn DESC
                ) AS _rn
                FROM facts
            )
            SELECT {", ".join(FUNDAMENTAL_VIEW_COLUMNS)}
            FROM ranked
            WHERE _rn = 1
              {_KIND_FILTER if kinds else ""}
            ORDER BY ticker, tag, period_kind
        """
        if kinds:
            params = params + [kinds]
        return _shape(_run(sql, params))

    # macro

    def macro(
        self,
        series_ids: Sequence[str] | None = None,
        calendar_days: int = 400,
    ) -> pd.DataFrame:
        """Macro observations with `date <= asof` over a calendar-day lookback.

        Calendar days are the natural unit here: macro series are monthly or
        daily on the civil calendar, not on trading sessions.
        """
        if calendar_days < 0:
            raise ValueError(f"macro(): calendar_days must be >= 0, got {calendar_days}")
        if not self._path(MACRO).exists():
            return _empty(MACRO_COLUMNS)
        start = self._asof - timedelta(days=int(calendar_days))
        sql = f"""
            SELECT {", ".join(MACRO_COLUMNS)}
            FROM read_parquet(?)
            WHERE date <= ?
              AND date >= ?
              {_SERIES_FILTER if series_ids else ""}
            ORDER BY series_id, date
        """
        params: list[object] = [str(self._path(MACRO)), self._asof, start]
        if series_ids:
            params.append(list(series_ids))
        return _shape(_run(sql, params))
