"""Project paths and runtime settings.

Nothing here reaches the network. Secrets and contact details come from the
environment so they never land in the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LAKE_DIR = DATA_DIR / "lake"
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_FILE = DATA_DIR / "universe" / "members.csv"


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or unusable."""


@dataclass(frozen=True, slots=True)
class Settings:
    lake_dir: Path
    raw_dir: Path
    universe_file: Path
    sec_user_agent: str | None
    fred_api_key: str | None
    sec_requests_per_second: float

    def require_sec_user_agent(self) -> str:
        """SEC fair-access requires a real contact string on every request.

        See https://www.sec.gov/os/webmaster-faq#developers — requests without
        one get throttled or blocked, so this fails loudly rather than silently
        hammering their servers.
        """
        if not self.sec_user_agent:
            raise ConfigError(
                "SEC_USER_AGENT is not set. SEC requires a contact string on every\n"
                "request to data.sec.gov. Set it to something like:\n"
                '  SEC_USER_AGENT="hedge-fund-research you@example.com"\n'
                "and re-run. Nothing else is sent to the SEC."
            )
        return self.sec_user_agent

    def require_fred_api_key(self) -> str:
        if not self.fred_api_key:
            raise ConfigError(
                "FRED_API_KEY is not set. Get a free key at\n"
                "  https://fredaccount.stlouisfed.org/apikeys\n"
                "then set FRED_API_KEY and re-run."
            )
        return self.fred_api_key


def load_settings() -> Settings:
    return Settings(
        lake_dir=Path(os.environ.get("HF_LAKE_DIR", LAKE_DIR)),
        raw_dir=Path(os.environ.get("HF_RAW_DIR", RAW_DIR)),
        universe_file=Path(os.environ.get("HF_UNIVERSE_FILE", UNIVERSE_FILE)),
        sec_user_agent=os.environ.get("SEC_USER_AGENT", "").strip() or None,
        fred_api_key=os.environ.get("FRED_API_KEY", "").strip() or None,
        # SEC documents 10 req/s; stay under it.
        sec_requests_per_second=float(os.environ.get("SEC_RPS", "8")),
    )
