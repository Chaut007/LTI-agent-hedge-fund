"""The one type every agent returns.

An agent's whole output surface is a Signal. It never names a share count, a
dollar amount or a weight — position sizing lives in the optimizer, which is
deterministic code. That separation is the reason an LLM agent can be added
later without touching anything downstream.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import date


def _as_float(name: str, value: object) -> float:
    """Coerce any real number to a plain float, rejecting bool and non-numbers.

    `isinstance(value, (int, float))` is the wrong test here twice over. It
    rejects np.float32 and np.int64 — which every vectorised agent produces,
    since numpy scalars are not Python float/int subclasses except np.float64 —
    and it accepts `bool`, so a comparison accidentally passed as a score
    becomes 1.0, i.e. maximum conviction. `numbers.Real` covers every real
    numeric type including numpy's, and bool is excluded explicitly.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"Signal.{name} must be a number, not a bool — got {value!r}. "
            "A bool would silently become 1.0 (maximum conviction); if that is "
            "what you mean, pass 1.0 explicitly."
        )
    if not isinstance(value, numbers.Real):
        raise ValueError(
            f"Signal.{name} must be a real number, got {type(value).__name__} "
            f"{value!r}"
        )
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"Signal.{name} must be finite, got {value!r}")
    return as_float


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Signal.{name} must be an integer, not a bool")
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"Signal.{name} must be an integer, got {type(value).__name__} "
            f"{value!r}"
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class Signal:
    """One agent's view on one ticker, as of one date.

    score:        -1.0 (max short conviction) .. +1.0 (max long conviction).
                  0.0 means "no opinion", which is different from not emitting
                  a Signal at all (that means "I could not evaluate this").
    confidence:   0.0 .. 1.0. How much the aggregator should lean on `score`.
    horizon_days: the holding period this score is meant for. A 5-day reversal
                  signal and a 12-month momentum signal must not be averaged as
                  if they said the same thing.
    cost_usd:     what producing this Signal cost. 0.0 for deterministic agents.
                  Carried here so a paid agent can be held to account later.
    """

    agent: str
    ticker: str
    asof: date
    score: float
    confidence: float
    horizon_days: int
    rationale: str = ""
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.agent:
            raise ValueError("Signal.agent must be a non-empty agent name")
        if not self.ticker:
            raise ValueError("Signal.ticker must be a non-empty ticker")
        # Normalise to plain Python numbers so a Signal built from np.float32
        # compares and serialises identically to one built from float. Frozen
        # dataclass, so assignment goes through object.__setattr__.
        for field_name in ("score", "confidence", "cost_usd"):
            object.__setattr__(
                self, field_name, _as_float(field_name, getattr(self, field_name))
            )
        object.__setattr__(
            self, "horizon_days", _as_int("horizon_days", self.horizon_days)
        )
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Signal.score must be in [-1, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Signal.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.horizon_days < 1:
            raise ValueError(
                f"Signal.horizon_days must be >= 1, got {self.horizon_days}"
            )
        if self.cost_usd < 0.0:
            raise ValueError(f"Signal.cost_usd must be >= 0, got {self.cost_usd}")

    @property
    def weighted_score(self) -> float:
        """Convenience for aggregators that want conviction-scaled scores."""
        return self.score * self.confidence
