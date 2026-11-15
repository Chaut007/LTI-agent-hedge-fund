"""Flattening companyfacts is pure, so it is testable without touching the SEC.

The shape of that JSON is the thing most likely to surprise us: facts nest four
levels deep, the same tag appears under several units, instant facts have no
`start`, and one period can be reported by several filings.
"""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.data.edgar import facts_to_frame, flatten_company_facts

PAYLOAD = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 24160000000,
                            "accn": "0000320193-23-000064",
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-05-05",
                        },
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 24161000000,
                            "accn": "0000320193-23-000106",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-11-03",
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    # Instant fact: balance sheet item, no `start`.
                    "USD": [
                        {
                            "end": "2023-04-01",
                            "val": 332160000000,
                            "accn": "0000320193-23-000064",
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-05-05",
                        }
                    ]
                }
            },
            "SomethingWeDoNotWant": {
                "units": {"USD": [{"end": "2023-03-31", "val": 1, "filed": "2023-05-05"}]}
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2023-04-21",
                            "val": 15728700000,
                            "accn": "0000320193-23-000064",
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-05-05",
                        }
                    ]
                }
            }
        },
    },
}


def test_tag_filter_keeps_only_requested_tags() -> None:
    rows = flatten_company_facts(PAYLOAD, "AAPL", tags=["NetIncomeLoss", "Assets"])
    assert {r["tag"] for r in rows} == {"NetIncomeLoss", "Assets"}


def test_both_filings_of_one_period_are_kept() -> None:
    """Dedupe is the view's job at query time, not the ingest's.

    Discarding the superseded 10-Q here would make a June query answer with a
    number that was not published until November.
    """
    rows = flatten_company_facts(PAYLOAD, "AAPL", tags=["NetIncomeLoss"])
    assert len(rows) == 2
    assert {r["filed"] for r in rows} == {"2023-05-05", "2023-11-03"}


def test_instant_facts_have_no_period_start() -> None:
    rows = flatten_company_facts(PAYLOAD, "AAPL", tags=["Assets"])
    assert rows[0]["period_start"] is None
    assert rows[0]["period_end"] == "2023-04-01"


def test_crosses_taxonomies() -> None:
    rows = flatten_company_facts(
        PAYLOAD, "AAPL", tags=["EntityCommonStockSharesOutstanding"]
    )
    assert rows[0]["taxonomy"] == "dei"
    assert rows[0]["unit"] == "shares"


def test_facts_without_a_filing_date_are_dropped() -> None:
    """Unplaceable in time means unusable without risking lookahead."""
    payload = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {"USD": [{"end": "2023-03-31", "val": 10}]}  # no `filed`
                }
            }
        },
    }
    assert flatten_company_facts(payload, "X", tags=["Assets"]) == []


def test_frame_conversion_types_the_dates_and_numbers() -> None:
    rows = flatten_company_facts(PAYLOAD, "AAPL", tags=["NetIncomeLoss", "Assets"])
    frame = facts_to_frame(rows)

    assert frame["filed"].iloc[0] == date(2023, 5, 5)
    assert frame["value"].dtype.kind == "f"
    assert frame["cik"].iloc[0] == 320193
    # The instant fact keeps a null start rather than being coerced to a date.
    assets = frame[frame["tag"] == "Assets"].iloc[0]
    assert assets["period_start"] is None or assets["period_start"] != assets["period_end"]


def test_empty_payload_is_not_an_error() -> None:
    assert flatten_company_facts({"cik": 1, "facts": {}}, "X") == []
    assert facts_to_frame([]).empty


def test_value_coercion_drops_unusable_rows() -> None:
    payload = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "val": None, "filed": "2023-05-05"},
                            {"end": "2023-06-30", "val": 5, "filed": "2023-08-05"},
                        ]
                    }
                }
            }
        },
    }
    frame = facts_to_frame(flatten_company_facts(payload, "X", tags=["Assets"]))
    assert len(frame) == 1
    assert frame["value"].iloc[0] == pytest.approx(5.0)
