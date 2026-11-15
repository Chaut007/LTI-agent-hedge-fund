"""Macro series from FRED.

Point-in-time caveat, stated plainly because it is easy to miss:

FRED serves the *current* value of each series. Market-priced series (Treasury
yields, VIX, fed funds) are never revised, so for those the current value is
also the historical one and the lake is point-in-time correct. Statistical
releases (CPI, unemployment, GDP) ARE revised, sometimes months later and
sometimes substantially. For those, what this ingest stores is today's revised
figure, not the number that was published at the time.

So a backtest using CPIAUCSL has a mild lookahead: it sees a cleaner inflation
series than anyone had live. True vintages exist in ALFRED (the
`realtime_start` / `realtime_end` parameters), which is the fix when the macro
agent becomes load-bearing. Until then `REVISABLE` marks which series to
distrust, and `hf status` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from hedge_fund.config import Settings
from hedge_fund.data._http import FredClient
from hedge_fund.data.lake import MACRO, MACRO_COLUMNS, Lake

DEFAULT_SERIES: tuple[str, ...] = (
    "DGS10",      # 10-year Treasury yield
    "DGS2",       # 2-year Treasury yield
    "T10Y2Y",     # 10y-2y spread
    "DFF",        # effective fed funds rate
    "VIXCLS",     # VIX close
    "CPIAUCSL",   # CPI, all urban consumers
    "UNRATE",     # unemployment rate
    "BAMLH0A0HYM2",  # high-yield credit spread
)

# Series subject to later revision: usable, but not truly point-in-time here.
REVISABLE: frozenset[str] = frozenset({"CPIAUCSL", "UNRATE"})


@dataclass(frozen=True)
class MacroReport:
    series_requested: int
    series_fetched: int
    rows_total: int
    empty: tuple[str, ...]
    revisable: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """False when nothing at all was fetched for a non-empty request."""
        return self.series_requested == 0 or self.series_fetched > 0

    def describe(self) -> str:
        lines = [
            f"macro: {self.rows_total:,} observations across "
            f"{self.series_fetched}/{self.series_requested} series",
        ]
        if self.empty:
            lines.append(f"  returned nothing: {', '.join(self.empty)}")
        if self.revisable:
            lines.append(
                f"  REVISED SERIES (not point-in-time): {', '.join(self.revisable)}"
            )
        return "\n".join(lines)


def ingest_macro(
    settings: Settings,
    lake: Lake,
    series_ids: Sequence[str] = DEFAULT_SERIES,
) -> MacroReport:
    client = FredClient(settings)
    frames: list[pd.DataFrame] = []
    empty: list[str] = []

    try:
        for series_id in series_ids:
            observations = client.observations(series_id)
            if not observations:
                empty.append(series_id)
                continue
            frame = pd.DataFrame(observations)
            frame["series_id"] = series_id
            # FRED encodes missing observations as "."
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
            frame = frame.dropna(subset=["date", "value"])[MACRO_COLUMNS]
            if frame.empty:
                empty.append(series_id)
                continue
            frames.append(frame)
    finally:
        client.close()

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=MACRO_COLUMNS)
    )
    _, total = lake.upsert(MACRO, combined, keys=["series_id", "date"])

    return MacroReport(
        series_requested=len(series_ids),
        series_fetched=len(frames),
        rows_total=total,
        empty=tuple(empty),
        revisable=tuple(sorted(set(series_ids) & REVISABLE)),
    )
