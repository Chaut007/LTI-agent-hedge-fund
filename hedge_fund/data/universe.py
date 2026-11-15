"""Universe definition and the ticker to CIK mapping.

READ THIS BEFORE TRUSTING ANY BACKTEST RUN AGAINST THIS UNIVERSE
----------------------------------------------------------------
A free universe is a *current* universe. Whatever list you put in
`data/universe/members.csv` is today's membership, which means every company
that went bankrupt, got acquired or was dropped from the index is missing.
Backtesting on it only ever asks "how did the survivors do", and survivors did
well by construction. The effect is not small — it has been measured at
roughly 1-4% of annual return depending on period and universe.

Point-in-time index membership is a paid dataset (CRSP, Compustat). Until this
project has one, the honest reading of any long backtest is "optimistic by an
unknown amount, probably 1-4%/yr". Treat relative comparisons between agents as
meaningful (they share the same bias) and absolute return levels as inflated.

`hf status` reprints this warning so it cannot be quietly forgotten.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from hedge_fund.config import Settings
from hedge_fund.data._http import SecClient
from hedge_fund.data.lake import UNIVERSE, Lake

SURVIVORSHIP_WARNING = (
    "universe is CURRENT membership only: delisted and acquired names are "
    "absent, so absolute backtest returns are optimistic (historically ~1-4%/yr). "
    "Agent-vs-agent comparisons remain valid."
)

# A deliberately small, liquid starter set so the pipeline can be exercised
# end to end on day one. Replace it with your real universe via
# `hf universe --from-csv <file>`; this is a bootstrap sample, not a strategy.
SEED_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "JPM", "V", "MA", "UNH", "XOM", "CVX", "JNJ", "PG",
    "HD", "COST", "ABBV", "MRK", "KO", "PEP", "WMT", "BAC",
    "CSCO", "ADBE", "CRM", "TMO", "LIN", "ACN",
)


def read_members_file(path: Path) -> list[str]:
    """Read tickers from a CSV. Accepts a bare one-column list or a `ticker` header."""
    tickers: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[0].strip().upper()
            if not value or value in {"TICKER", "SYMBOL"}:
                continue
            tickers.append(value)
    if not tickers:
        raise ValueError(f"no tickers found in {path}")
    return tickers


def resolve_tickers(
    settings: Settings, from_csv: Path | None = None
) -> tuple[list[str], str]:
    """Return the tickers and a human-readable statement of where they came from.

    The seed list is a fallback only when NO file was named. An explicit
    `--from-csv` that does not exist is an error: silently trading a 30-name
    demo list because of a typo in a path is exactly the failure this must
    prevent, and `cmd_universe` prints `source` so the choice is always visible.
    """
    if from_csv is not None:
        if not from_csv.exists():
            raise FileNotFoundError(
                f"--from-csv {from_csv} does not exist. Refusing to fall back to the "
                "built-in seed list when a file was named explicitly."
            )
        return read_members_file(from_csv), f"csv:{from_csv}"
    if settings.universe_file.exists():
        return read_members_file(settings.universe_file), f"csv:{settings.universe_file}"
    return list(SEED_TICKERS), "SEED_TICKERS (built-in bootstrap sample, not a strategy)"


def build_universe(
    settings: Settings,
    lake: Lake,
    from_csv: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Map the configured tickers to SEC CIKs and store them.

    Returns the universe frame and the `source` string from `resolve_tickers`
    so the caller can say where the list came from. Tickers with no CIK match
    (ADRs, some share classes, anything the SEC file spells differently) are
    kept with a null cik so prices still ingest; only the EDGAR fundamentals
    step skips them. Duplicates in the input collapse to one row so a repeated
    symbol does not cost a repeated SEC request.
    """
    raw_tickers, source = resolve_tickers(settings, from_csv)
    tickers = list(dict.fromkeys(raw_tickers))  # dedupe, order preserved

    client = SecClient(settings)
    try:
        payload = client.company_tickers()
    finally:
        client.close()

    lookup: dict[str, tuple[int, str]] = {}
    for entry in (payload or {}).values():
        symbol = str(entry.get("ticker", "")).strip().upper()
        if symbol:
            lookup[symbol] = (int(entry["cik_str"]), str(entry.get("title", "")))

    rows = []
    for ticker in tickers:
        # SEC writes class shares as BRK-B; many price feeds use BRK.B.
        cik, name = lookup.get(ticker, lookup.get(ticker.replace(".", "-"), (None, "")))
        rows.append({"ticker": ticker, "cik": cik, "name": name})

    frame = pd.DataFrame(rows, columns=["ticker", "cik", "name"])
    frame["cik"] = frame["cik"].astype("Int64")
    lake.write(UNIVERSE, frame)
    return frame, source
