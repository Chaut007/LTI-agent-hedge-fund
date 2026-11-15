"""The optimizers, and the measured problem the banded one exists to solve.

Step 4 measured `TopNOptimizer` trading 3.28% of the book per day and paying
14.9% of NAV in costs, against equal weight's 0.16% and 0.7%. Gross of costs
it matched equal weight; net of them it lost. So the tests that matter here
are the ones that show `BandedOptimizer` cutting turnover **without changing
which names are chosen** — if it changed the selection it would be a different
strategy, not a cheaper one.
"""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.portfolio.optimizer import BandedOptimizer, Optimizer, TopNOptimizer
from hedge_fund.portfolio.risk import RiskLimits
from hedge_fund.portfolio.state import PortfolioState

ASOF = date(2024, 6, 3)

# Twenty names, descending score, so rank order is obvious by ticker.
SCORES = {f"T{i:02d}": 1.0 - i * 0.05 for i in range(20)}


def state(weights=None) -> PortfolioState:
    return PortfolioState(asof=ASOF, nav=1.0, peak_nav=1.0, weights=weights or {})


def limits(**overrides) -> RiskLimits:
    base = dict(max_position=0.25, min_positions=1, max_positions=20, max_turnover=1.0)
    base.update(overrides)
    return RiskLimits(**base)


def test_both_satisfy_the_optimizer_protocol() -> None:
    assert isinstance(TopNOptimizer(), Optimizer)
    assert isinstance(BandedOptimizer(), Optimizer)


# -- TopNOptimizer ------------------------------------------------------------------


def test_top_n_holds_the_best_names_equally() -> None:
    out = TopNOptimizer(top_n=4).weights(SCORES, state(), limits())
    assert set(out) == {"T00", "T01", "T02", "T03"}
    assert all(w == pytest.approx(0.25) for w in out.values())
    assert sum(out.values()) == pytest.approx(1.0)


def test_top_n_respects_the_position_cap() -> None:
    """Two names cannot be 50% each under a 25% cap; the book is under-invested."""
    out = TopNOptimizer(top_n=2).weights(SCORES, state(), limits(max_position=0.25))
    assert all(w == pytest.approx(0.25) for w in out.values())
    assert sum(out.values()) == pytest.approx(0.5)


def test_top_n_is_bounded_by_max_positions() -> None:
    out = TopNOptimizer(top_n=50).weights(SCORES, state(), limits(max_positions=6))
    assert len(out) == 6


def test_min_score_can_empty_the_book() -> None:
    """The threshold is strict, so the best score exactly at it does not qualify."""
    assert max(SCORES.values()) == pytest.approx(1.0)
    assert TopNOptimizer(top_n=10, min_score=1.0).weights(SCORES, state(), limits()) == {}
    kept = TopNOptimizer(top_n=10, min_score=0.99).weights(SCORES, state(), limits())
    assert set(kept) == {"T00"}


def test_ties_break_on_ticker_for_reproducibility() -> None:
    tied = {"TB": 0.5, "TA": 0.5, "TC": 0.5}
    out = TopNOptimizer(top_n=2).weights(tied, state(), limits())
    assert sorted(out) == ["TA", "TB"]


# -- the no-trade band ---------------------------------------------------------------


def test_band_zero_reproduces_top_n() -> None:
    """The banded optimizer must be a strict generalisation, not a different one."""
    held = state({"T00": 0.3, "T05": 0.2})
    plain = TopNOptimizer(top_n=4).weights(SCORES, held, limits())
    banded = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.0).weights(
        SCORES, held, limits()
    )
    assert banded == pytest.approx(plain)


def test_a_small_drift_is_left_alone() -> None:
    """0.24 against a 0.25 target is inside the band, so it is not traded."""
    held = state({"T00": 0.24, "T01": 0.25, "T02": 0.25, "T03": 0.25})
    out = BandedOptimizer(top_n=4, band=0.02, keep_ranked=1.0).weights(
        SCORES, held, limits()
    )
    assert out["T00"] == pytest.approx(0.24)
    assert held.turnover_to(out) == pytest.approx(0.0)


def test_a_large_drift_is_traded() -> None:
    held = state({"T00": 0.10, "T01": 0.25, "T02": 0.25, "T03": 0.25})
    out = BandedOptimizer(top_n=4, band=0.02, keep_ranked=1.0).weights(
        SCORES, held, limits()
    )
    assert out["T00"] == pytest.approx(0.25)


def test_an_exit_is_never_banded() -> None:
    """A name that left the selection is sold, however small the position."""
    held = state({"T00": 0.25, "T01": 0.25, "T02": 0.25, "GONE": 0.001})
    out = BandedOptimizer(top_n=3, band=0.10, keep_ranked=1.0).weights(
        SCORES, held, limits()
    )
    assert "GONE" not in out


# -- hysteresis ---------------------------------------------------------------------


def test_a_name_that_slipped_one_rank_is_kept() -> None:
    """Rank jitter at the boundary is the biggest single source of turnover."""
    held = state({f"T{i:02d}": 0.10 for i in range(5)})
    # T04 is now rank 5, outside top-4, but inside the kept ring of 6.
    out = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.5).weights(
        SCORES, held, limits()
    )
    assert "T04" in out


def test_a_name_that_fell_far_is_sold() -> None:
    held = state({"T00": 0.2, "T19": 0.2})
    out = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.5).weights(
        SCORES, held, limits()
    )
    assert "T19" not in out, "rank 20 is well outside the kept ring"


def test_hysteresis_never_exceeds_max_positions() -> None:
    held = state({f"T{i:02d}": 0.05 for i in range(20)})
    out = BandedOptimizer(top_n=8, band=0.0, keep_ranked=2.0).weights(
        SCORES, held, limits(max_positions=10)
    )
    assert len(out) <= 10


def test_keep_ranked_one_means_no_hysteresis() -> None:
    held = state({f"T{i:02d}": 0.10 for i in range(5)})
    out = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.0).weights(
        SCORES, held, limits()
    )
    assert set(out) == {"T00", "T01", "T02", "T03"}


# -- turnover cap --------------------------------------------------------------------


def test_turnover_cap_blends_toward_the_target() -> None:
    held = state({f"T{i:02d}": 0.25 for i in range(4)})
    far = {f"U{i}": 1.0 - i * 0.01 for i in range(4)}  # an entirely new book
    out = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.0, turnover_cap=0.10).weights(
        far, held, limits()
    )
    assert held.turnover_to(out) == pytest.approx(0.10, abs=1e-9)
    assert any(t.startswith("U") for t in out), "it moves toward the new names"
    assert any(t.startswith("T") for t in out), "but does not get all the way there"


def test_turnover_cap_defers_to_the_risk_limit_when_unset() -> None:
    held = state({f"T{i:02d}": 0.25 for i in range(4)})
    far = {f"U{i}": 1.0 - i * 0.01 for i in range(4)}
    out = BandedOptimizer(top_n=4, band=0.0, keep_ranked=1.0).weights(
        far, held, limits(max_turnover=0.05)
    )
    assert held.turnover_to(out) == pytest.approx(0.05, abs=1e-9)


def test_gross_never_exceeds_the_limit_after_banding() -> None:
    held = state({f"T{i:02d}": 0.25 for i in range(4)})
    out = BandedOptimizer(top_n=4, band=0.05, keep_ranked=1.5).weights(
        SCORES, held, limits(max_gross=1.0, max_position=0.25)
    )
    assert sum(out.values()) <= 1.0 + 1e-9


def test_empty_scores_give_an_empty_book() -> None:
    assert BandedOptimizer().weights({}, state(), limits()) == {}


def test_banded_from_cash_is_a_normal_entry() -> None:
    out = BandedOptimizer(top_n=4, band=0.05).weights(SCORES, state(), limits())
    assert set(out) == {"T00", "T01", "T02", "T03"}
    assert sum(out.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs", [{"band": -0.01}, {"keep_ranked": 0.5}, {"turnover_cap": 0.0}]
)
def test_nonsense_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        BandedOptimizer(**kwargs)


# -- the point of the exercise ---------------------------------------------------------


def test_banding_cuts_turnover_without_changing_the_selection() -> None:
    """The measured claim: cheaper, not different.

    Over a sequence of small score perturbations the banded optimizer must
    trade materially less than TopN while holding substantially the same names.
    """
    import random

    rng = random.Random(0)
    plain_opt = TopNOptimizer(top_n=10)
    banded_opt = BandedOptimizer(top_n=10, band=0.01, keep_ranked=1.4)
    lim = limits(max_position=0.20, max_positions=20, max_turnover=1.0)

    plain_state = state()
    banded_state = state()
    plain_turnover = 0.0
    banded_turnover = 0.0
    overlap = []

    for _ in range(24):
        jittered = {t: s + rng.gauss(0, 0.03) for t, s in SCORES.items()}
        plain_target = plain_opt.weights(jittered, plain_state, lim)
        banded_target = banded_opt.weights(jittered, banded_state, lim)
        plain_turnover += plain_state.turnover_to(plain_target)
        banded_turnover += banded_state.turnover_to(banded_target)
        if plain_target and banded_target:
            shared = set(plain_target) & set(banded_target)
            overlap.append(len(shared) / len(set(plain_target)))
        plain_state = state(plain_target)
        banded_state = state(banded_target)

    assert banded_turnover < plain_turnover * 0.7, (
        f"banded traded {banded_turnover:.3f} against plain {plain_turnover:.3f}; "
        "the band is not earning its place"
    )
    assert sum(overlap) / len(overlap) > 0.75, (
        "banding changed which names are held, so it is a different strategy "
        "rather than a cheaper one"
    )


def test_banded_blend_respects_max_positions() -> None:
    """The blend unions with the held book, so it must be trimmed."""
    held = state({f"H{i:03d}": 0.004 for i in range(200)})
    targets = {f"N{i:02d}": 1.0 - i * 0.01 for i in range(50)}
    out = BandedOptimizer(top_n=50, band=0.0, keep_ranked=1.0, turnover_cap=0.20).weights(
        targets, held, limits(max_position=0.04, max_positions=50)
    )
    assert len(out) <= 50
