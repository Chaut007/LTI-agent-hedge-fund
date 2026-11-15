"""SEC EDGAR companyfacts ingest — the point-in-time backbone of the lake.

Why this source rather than a convenient fundamentals API: every XBRL fact
carries the `filed` date of the filing that reported it. That one field is what
lets the lake answer "what was knowable on 2023-04-30" instead of "what do we
now know about Q1 2023". Free fundamentals APIs almost never expose it, and
without it a backtest reads next quarter's earnings a month early and looks
brilliant.

The same company can report the same period more than once (restatements, 10-K
superseding 10-Q). Every version is stored, keyed by accession number; the
`AsofView` picks the latest one that was visible on the query date. Throwing
away superseded versions here would make a historical query answer with numbers
nobody had at the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from hedge_fund.config import Settings
from hedge_fund.data._http import SecClient
from hedge_fund.data.lake import FUNDAMENTAL_COLUMNS, FUNDAMENTALS, Lake

# Enough to build the Value and Quality agents, and no more. Pulling every tag
# a filer reports turns a 30-ticker lake into gigabytes of XBRL minutiae.
DEFAULT_TAGS: tuple[str, ...] = (
    # balance sheet
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "LiabilitiesAndStockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    # income statement
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    # share counts
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
)

_UPSERT_KEYS = (
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "period_start",
    "period_end",
    "accn",
)


@dataclass(frozen=True)
class EdgarReport:
    filers_requested: int
    filers_fetched: int
    rows_added: int
    rows_total: int
    skipped_no_cik: tuple[str, ...]
    no_facts: tuple[str, ...]
    """Filers the SEC answered 404 for: a legitimate 'no XBRL on file'."""
    errors: dict[str, str]
    """ticker -> short reason, for fetches that actually failed."""

    @property
    def ok(self) -> bool:
        """False when nothing at all was fetched for a non-empty request."""
        return self.filers_requested == 0 or self.filers_fetched > 0

    def describe(self) -> str:
        lines = [
            f"fundamentals: {self.rows_total:,} facts in lake "
            f"({self.rows_added:,} new) from {self.filers_fetched}/"
            f"{self.filers_requested} filers",
        ]
        if self.skipped_no_cik:
            lines.append(
                f"  no SEC CIK, skipped: {', '.join(self.skipped_no_cik[:10])}"
            )
        if self.no_facts:
            lines.append(f"  no XBRL facts on file (404): {', '.join(self.no_facts[:10])}")
        if self.errors:
            # Group by reason so thirty identical 403s read as one line naming
            # the cause, not thirty tickers with no explanation. A bad
            # SEC_USER_AGENT is the single most likely failure and must be
            # distinguishable from a transient timeout.
            by_reason: dict[str, list[str]] = {}
            for ticker, reason in self.errors.items():
                by_reason.setdefault(reason, []).append(ticker)
            lines.append(f"  fetch errors ({len(self.errors)}):")
            for reason, tickers in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
                shown = ", ".join(tickers[:6])
                more = f" (+{len(tickers) - 6})" if len(tickers) > 6 else ""
                lines.append(f"    {reason}: {shown}{more}")
            if any("403" in r for r in by_reason):
                lines.append(
                    "    403 from the SEC almost always means SEC_USER_AGENT is "
                    "missing or not a real contact string."
                )
        return "\n".join(lines)


def _describe_error(exc: BaseException) -> str:
    """One short, groupable line per failure class."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    return type(exc).__name__


def flatten_company_facts(
    payload: dict[str, Any],
    ticker: str,
    tags: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Turn one companyfacts document into tidy rows.

    Pure function, no network: this is what the point-in-time test exercises.
    """
    wanted = set(tags) if tags else None
    cik = payload.get("cik")
    rows: list[dict[str, Any]] = []

    for taxonomy, tag_map in (payload.get("facts") or {}).items():
        for tag, tag_body in (tag_map or {}).items():
            if wanted is not None and tag not in wanted:
                continue
            for unit, facts in ((tag_body or {}).get("units") or {}).items():
                for fact in facts or ():
                    filed = fact.get("filed")
                    period_end = fact.get("end")
                    # A fact with no filing date cannot be placed in time, so it
                    # cannot be used without risking lookahead. Drop it.
                    if not filed or not period_end:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "period_start": fact.get("start"),
                            "period_end": period_end,
                            "value": fact.get("val"),
                            "form": fact.get("form"),
                            "filed": filed,
                            "accn": fact.get("accn"),
                            "fy": fact.get("fy"),
                            "fp": fact.get("fp"),
                        }
                    )
    return rows


def facts_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=FUNDAMENTAL_COLUMNS)
    if frame.empty:
        return frame
    for column in ("period_start", "period_end", "filed"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    # Pin to float64 rather than letting pandas infer. A batch of all-integer
    # dollar facts would infer int64 while a batch containing EPS infers
    # float64, and the lake would end up with a different parquet schema per
    # ingest. float64 holds every integer up to 2^53, far beyond any figure on
    # a balance sheet.
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")
    frame["fy"] = pd.to_numeric(frame["fy"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["period_end", "filed", "value"])
    return frame.reset_index(drop=True)


def ingest_fundamentals(
    settings: Settings,
    lake: Lake,
    universe: pd.DataFrame,
    tags: Sequence[str] | None = DEFAULT_TAGS,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> EdgarReport:
    """Fetch companyfacts for every filer in `universe` and merge into the lake."""
    if universe.empty:
        raise ValueError("universe is empty — run `hf universe` first")

    with_cik = universe.dropna(subset=["cik"])
    skipped = tuple(
        sorted(universe.loc[universe["cik"].isna(), "ticker"].astype(str))
    )

    client = SecClient(settings)
    collected: list[dict[str, Any]] = []
    no_facts: list[str] = []
    errors: dict[str, str] = {}
    fetched = 0
    total = len(with_cik)

    try:
        for position, row in enumerate(with_cik.itertuples(index=False), start=1):
            ticker = str(row.ticker)
            if on_progress:
                on_progress(ticker, position, total)
            try:
                payload = client.company_facts(int(row.cik))
            except Exception as exc:  # one bad filer must not sink the run,
                errors[ticker] = _describe_error(exc)  # but the reason is kept
                continue
            if payload is None:
                # 404: the SEC has no companyfacts document for this CIK. That
                # is a fact about the filer, not a failure of the fetch.
                no_facts.append(ticker)
                continue
            fetched += 1
            collected.extend(flatten_company_facts(payload, ticker, tags))
    finally:
        client.close()

    frame = facts_to_frame(collected)
    # COUNT(*) through DuckDB rather than loading the whole table just to
    # measure it; upsert reads it once more anyway.
    before = lake.row_count(FUNDAMENTALS)
    _, after = lake.upsert(FUNDAMENTALS, frame, keys=list(_UPSERT_KEYS))

    return EdgarReport(
        filers_requested=total,
        filers_fetched=fetched,
        rows_added=max(after - before, 0),
        rows_total=after,
        skipped_no_cik=skipped,
        no_facts=tuple(no_facts),
        errors=errors,
    )
