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

    def distinct(self, start: date | None = None, end: date | None = None) -> list[Trial]:
        """One Trial per trial_id, the most recent record winning.

        With `start`/`end`, only trials evaluated over exactly that window.
        """
        latest: dict[str, Trial] = {}
        for trial in self:
            if start is not None and trial.start != start.isoformat():
                continue
            if end is not None and trial.end != end.isoformat():
                continue
            latest[trial.trial_id] = trial
        return list(latest.values())

    def sharpes(
        self, start: date | None = None, end: date | None = None
    ) -> list[float]:
        """Per-period Sharpe of each distinct trial, for `deflated_sharpe`.

        **Pass the current run's window.** The Deflated Sharpe charges for
        how many configurations were tried *on this data*, and it does so
        through the variance of their Sharpes. Trials measured over a
        different window are not alternative configurations of the same
        experiment, and pooling them makes the charge meaningless.

        Measured here: one 61-day smoke test, per-period Sharpe 0.3426
        against a cluster at 0.065, raised the variance of 19 trials from
        0.000011 to 0.004068 — a factor of 370. E[max SR] scales with its
        square root, so the deflated Sharpe of a genuine nine-year run read
        0.6% instead of passing. A short exploratory run silently
        disqualified every real result in the log.
        """
        return [t.sharpe_period for t in self.distinct(start, end)]

    def __len__(self) -> int:
        return len(self.distinct())
