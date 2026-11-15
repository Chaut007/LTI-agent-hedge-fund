"""Load the wide price panel the simulator computes returns from.

This is the one place the backtester reads *unbounded* data, and it is
deliberate: the engine is the market, not a participant. It needs every price
to mark positions to market. Strategies never touch this — they get an
`AsofView` through the Context the engine hands them, one date at a time.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from hedge_fund.data.lake import PRICES, Lake


def load_price_panel(
    lake: Lake,
    tickers: Sequence[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Dates x tickers of `column` (default adj_close), float64, DatetimeIndex.

    adj_close is the right series for returns: it folds splits and dividends
    in, so a 2:1 split is not a -50% day. Use `close` only for price levels.
    Missing cells are NaN; the engine decides how to treat them.
    """
    if column not in {"open", "high", "low", "close", "adj_close", "volume"}:
        raise ValueError(f"unknown price column {column!r}")
    if not lake.exists(PRICES):
        return pd.DataFrame(dtype="float64")

    clauses, params = ["1=1"], [str(lake.path(PRICES))]
    if tickers:
        clauses.append("ticker IN (SELECT UNNEST(CAST(? AS VARCHAR[])))")
        params.append([t.upper() for t in tickers])
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)

    rows = lake.unbounded_query(
        f"SELECT ticker, date, {column} AS px FROM read_parquet(?) "
        f"WHERE {' AND '.join(clauses)} ORDER BY date, ticker",
        params,
    )
    if rows.empty:
        return pd.DataFrame(dtype="float64")

    panel = rows.pivot(index="date", columns="ticker", values="px")
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    panel.columns.name = None
    return panel.sort_index().astype("float64")
