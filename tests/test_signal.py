"""Signal is the contract every agent shares, so its guards are worth testing.

A score of 1.5 or a NaN confidence does not fail loudly on its own — it flows
into the aggregator, skews a rank, and shows up as an unexplained position.
Rejecting it at construction keeps that class of bug out of the pipeline.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from hedge_fund.core.signal import Signal

ASOF = date(2024, 6, 3)


def make(**overrides) -> Signal:
    base = dict(
        agent="momentum",
        ticker="AAPL",
        asof=ASOF,
        score=0.5,
        confidence=0.8,
        horizon_days=252,
    )
    base.update(overrides)
    return Signal(**base)


def test_valid_signal_round_trips() -> None:
    signal = make(rationale="12-1 momentum in top decile", cost_usd=0.0)
    assert signal.score == pytest.approx(0.5)
    assert signal.weighted_score == pytest.approx(0.4)


def test_signal_is_immutable() -> None:
    signal = make()
    with pytest.raises(AttributeError):
        signal.score = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("score", [1.01, -1.01, 5.0])
def test_score_outside_unit_interval_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        make(score=score)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_outside_zero_one_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        make(confidence=confidence)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    """NaN is the dangerous one: it compares false to everything and spreads."""
    with pytest.raises(ValueError, match="finite"):
        make(score=value)
    with pytest.raises(ValueError, match="finite"):
        make(confidence=value)


def test_horizon_must_be_at_least_one_day() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        make(horizon_days=0)


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        make(cost_usd=-0.01)


def test_identity_fields_must_be_populated() -> None:
    with pytest.raises(ValueError, match="agent"):
        make(agent="")
    with pytest.raises(ValueError, match="ticker"):
        make(ticker="")


def test_zero_score_is_a_real_opinion() -> None:
    """Score 0.0 means "evaluated, no edge" and must not be confused with error."""
    assert make(score=0.0).score == 0.0


# -- numeric types: every agent is vectorised, so numpy scalars must work -----


@pytest.mark.parametrize(
    "value",
    [np.float32(0.5), np.float64(0.5), np.int64(0), np.int32(1), np.float16(0.25)],
)
def test_numpy_scalars_are_accepted_and_normalised(value) -> None:
    """np.float32 is not a Python float subclass; it was being rejected."""
    signal = make(score=value)
    assert isinstance(signal.score, float), type(signal.score)
    assert signal.score == pytest.approx(float(value))


def test_numpy_integer_horizon_is_accepted() -> None:
    signal = make(horizon_days=np.int64(252))
    assert signal.horizon_days == 252
    assert isinstance(signal.horizon_days, int)


@pytest.mark.parametrize("field", ["score", "confidence", "cost_usd"])
def test_bool_is_rejected_not_promoted_to_one(field: str) -> None:
    """`True` is an int subclass; it was silently becoming score 1.0."""
    with pytest.raises(ValueError, match="not a bool"):
        make(**{field: True})


def test_bool_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a bool"):
        make(horizon_days=True)


@pytest.mark.parametrize("value", ["0.5", None, [0.5], {"score": 0.5}])
def test_non_numeric_score_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="real number"):
        make(score=value)


def test_float_horizon_is_rejected() -> None:
    """A horizon of 21.0 is almost certainly a units mistake, not a count."""
    with pytest.raises(ValueError, match="integer"):
        make(horizon_days=21.0)
