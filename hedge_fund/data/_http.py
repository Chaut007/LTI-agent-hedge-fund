"""Shared HTTP helpers: a rate limiter and a polite SEC client.

SEC publishes a fair-access limit of 10 requests/second and requires a
User-Agent naming a real contact. Exceeding either gets an IP blocked, so the
limiter is process-wide and the User-Agent is mandatory rather than optional.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from hedge_fund.config import Settings

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
FRED_BASE = "https://api.stlouisfed.org/fred"

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Simple blocking limiter: at most `rps` calls per second, thread-safe."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / max(rps, 0.1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


def _get_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    attempts: int = 4,
    timeout: float = 30.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        if limiter:
            limiter.wait()
        try:
            response = session.get(
                url, headers=headers, params=params, timeout=timeout
            )
        except requests.RequestException as exc:  # network hiccup
            last_error = exc
            time.sleep(2**attempt)
            continue
        if response.status_code == 404:
            return None
        if response.status_code in _RETRY_STATUSES:
            last_error = requests.HTTPError(
                f"{response.status_code} from {url}", response=response
            )
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"giving up on {url} after {attempts} attempts") from last_error


class SecClient:
    """Rate-limited JSON client for sec.gov and data.sec.gov."""

    def __init__(self, settings: Settings) -> None:
        self._user_agent = settings.require_sec_user_agent()
        self._limiter = RateLimiter(settings.sec_requests_per_second)
        self._session = requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }

    def get_json(self, url: str) -> Any:
        return _get_json(
            self._session, url, headers=self._headers, limiter=self._limiter
        )

    def company_tickers(self) -> Any:
        return self.get_json(f"{SEC_BASE}/files/company_tickers.json")

    def company_facts(self, cik: int) -> Any:
        """All XBRL facts for one filer. Returns None when the CIK has none."""
        return self.get_json(
            f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        )

    def close(self) -> None:
        self._session.close()


class FredClient:
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.require_fred_api_key()
        self._limiter = RateLimiter(10.0)
        self._session = requests.Session()

    def observations(self, series_id: str) -> list[dict[str, Any]]:
        payload = _get_json(
            self._session,
            f"{FRED_BASE}/series/observations",
            params={
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
            },
            limiter=self._limiter,
        )
        if not payload:
            return []
        return payload.get("observations", [])

    def close(self) -> None:
        self._session.close()
