"""Append-only record of every backtest attempted. Feeds the Deflated Sharpe.

The DSR needs to know how many strategy configurations were tried. The
researcher's memory is not an acceptable source for that number — it is biased
in exactly one direction — so the CLI writes a line here on every run and the
validation step reads them all back.

A "trial" is a distinct (strategy, params) pair. Re-running the same
configuration to look at its output again is not a new look at the data and
does not increase N; trying a new parameter does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def _stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Trial:
    strategy: str
    params: dict[str, Any]
    start: str
    end: str
    n_obs: int
    sharpe_period: float
    sharpe_annual: float
    recorded_at: str
    trial_id: str

    @classmethod
    def build(
        cls,
        strategy: str,
        params: Mapping[str, Any],
        start: date,
        end: date,
        n_obs: int,
        sharpe_period: float,
    ) -> "Trial":
        params = dict(params)
        return cls(
            strategy=strategy,
            params=params,
            start=start.isoformat(),
            end=end.isoformat(),
            n_obs=int(n_obs),
            sharpe_period=float(sharpe_period),
            sharpe_annual=float(sharpe_period) * (252.0**0.5),
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            trial_id=_stable_hash(
                {"strategy": strategy, "params": params, "start": start, "end": end}
            ),
        )


class TrialLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, trial: Trial) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trial), sort_keys=True) + "\n")

    def __iter__(self) -> Iterator[Trial]:
        if not self.path.exists():
            return iter(())
        trials: list[Trial] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    trials.append(Trial(**json.loads(line)))
        return iter(trials)

    def distinct(self) -> list[Trial]:
        """One Trial per trial_id, the most recent record winning."""
        latest: dict[str, Trial] = {}
        for trial in self:
            latest[trial.trial_id] = trial
        return list(latest.values())

    def sharpes(self) -> list[float]:
        """Per-period Sharpe of every distinct trial, for `deflated_sharpe`."""
        return [t.sharpe_period for t in self.distinct()]

    def __len__(self) -> int:
        return len(self.distinct())
