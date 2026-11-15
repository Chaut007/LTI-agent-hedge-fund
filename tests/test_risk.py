"""The risk layer: what it clamps, what it refuses, and what it records.

The distinction worth defending is clamp-versus-veto. A position over its cap
is a size mistake and gets cut. A short in a long-only book is not a size
mistake — it means the layer above misunderstands its mandate, and correcting
it silently would ship that misunderstanding into a plausible-looking book.
"""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.portfolio.risk import (
    AS_DESIGNED,
    BOOTSTRAP_INFORMED,
    DIVERSIFIED,
    Breach,
    RiskLimits,
    RiskManager,
    with_kill_switch,
)
from hedge_fund.portfolio.state import PortfolioState

ASOF = date(2024, 6, 3)


def state(weights=None, nav=1_000_000.0, peak=1_000_000.0, halted=False) -> PortfolioState:
    return PortfolioState(
        asof=ASOF, nav=nav, peak_nav=peak, weights=weights or {}, halted=halted
    )


def limits(**overrides) -> RiskLimits:
    base = dict(max_position=0.10, min_positions=1, max_positions=15, max_turnover=1.0)
    base.update(overrides)
    return RiskLimits(**base)


# -- PortfolioState ---------------------------------------------------------------


def test_drawdown_is_measured_from_peak() -> None:
    assert state(nav=850.0, peak=1000.0).drawdown == pytest.approx(0.15)
    assert state(nav=1200.0, peak=1200.0).drawdown == 0.0


def test_drawdown_never_goes_negative_above_a_stale_peak() -> None:
    assert state(nav=1500.0, peak=1000.0).drawdown == 0.0


def test_gross_counts_absolute_weights() -> None:
    assert state({"A": 0.6, "B": -0.3}).gross == pytest.approx(0.9)


def test_turnover_uses_the_one_way_convention() -> None:
    """Deploying a whole book from cash is 0.5, matching what the engine charges."""
    assert state().turnover_to({"A": 0.5, "B": 0.5}) == pytest.approx(0.5)
    # a full switch from A to B moves 1.0 of notional, which is turnover 1.0
    assert state({"A": 1.0}).turnover_to({"B": 1.0}) == pytest.approx(1.0)
    assert state({"A": 0.5, "B": 0.5}).turnover_to({"A": 0.5, "B": 0.5}) == 0.0


# -- RiskLimits construction -------------------------------------------------------


def test_cap_times_count_must_reach_max_gross() -> None:
    """8 names at 10% cannot be fully invested; refusing to construct says so."""
    with pytest.raises(ValueError, match="never be fully invested"):
        RiskLimits(max_position=0.10, min_positions=8, max_positions=8, max_gross=1.0)


def test_the_designed_limits_are_self_consistent() -> None:
    assert AS_DESIGNED.max_position * AS_DESIGNED.max_positions >= AS_DESIGNED.max_gross
    assert DIVERSIFIED.max_position * DIVERSIFIED.max_positions >= DIVERSIFIED.max_gross


def test_presets_differ_only_where_intended() -> None:
    assert AS_DESIGNED.kill_switch_dd == 0.15, "the value measured to fire 91-93% of paths"
    assert BOOTSTRAP_INFORMED.kill_switch_dd == 0.40
    assert with_kill_switch(AS_DESIGNED, 0.5).kill_switch_dd == 0.5
    assert with_kill_switch(AS_DESIGNED, 0.5).max_position == AS_DESIGNED.max_position


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_position": 0.0},
        {"max_position": 1.5},
        {"min_positions": 0},
        {"max_gross": 0.0},
        {"max_turnover": 0.0},
        {"kill_switch_dd": 0.0},
        {"kill_switch_dd": 1.0},
        {"min_positions": 10, "max_positions": 5},
    ],
)
def test_nonsense_limits_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        limits(**kwargs)


# -- clamping ----------------------------------------------------------------------


def test_position_over_cap_is_cut_to_the_cap_and_recorded() -> None:
    decision = RiskManager(limits()).review({"A": 0.4, "B": 0.05}, state())
    assert decision.weights["A"] == pytest.approx(0.10)
    assert decision.weights["B"] == pytest.approx(0.05)
    assert Breach.POSITION_CAP in decision.breaches
    assert not decision.vetoed


def test_too_many_positions_drops_the_smallest() -> None:
    targets = {f"T{i:02d}": 0.05 - i * 0.001 for i in range(20)}
    decision = RiskManager(limits(max_positions=15)).review(targets, state())
    assert len(decision.weights) == 15
    assert "T00" in decision.weights and "T19" not in decision.weights
    assert Breach.TOO_MANY_POSITIONS in decision.breaches


def test_too_few_positions_goes_flat_rather_than_inventing_names() -> None:
    """Padding a book to satisfy a diversification rule is the risk layer trading."""
    decision = RiskManager(limits(min_positions=8)).review({"A": 0.05, "B": 0.05}, state())
    assert decision.weights == {}
    assert Breach.TOO_FEW_POSITIONS in decision.breaches
    assert not decision.vetoed, "holding cash is a decision, not a refusal"


def test_gross_over_one_is_scaled_down() -> None:
    targets = {f"T{i}": 0.10 for i in range(12)}  # gross 1.20
    decision = RiskManager(limits()).review(targets, state())
    assert sum(decision.weights.values()) == pytest.approx(1.0)
    assert Breach.GROSS_EXPOSURE in decision.breaches


def test_clean_proposal_passes_untouched() -> None:
    targets = {f"T{i}": 0.10 for i in range(10)}
    decision = RiskManager(limits()).review(targets, state())
    assert decision.weights == pytest.approx(targets)
    assert decision.actions == ()
    assert "no intervention" in decision.describe()


# -- turnover budget ----------------------------------------------------------------


def test_turnover_over_budget_moves_part_of_the_way() -> None:
    held = {f"T{i}": 0.10 for i in range(10)}
    targets = {f"U{i}": 0.10 for i in range(10)}  # a complete switch: turnover 1.0
    # Room for all 20 names, so the blend is not trimmed and the budget is the
    # only thing binding.
    decision = RiskManager(limits(max_turnover=0.20, max_positions=20)).review(
        targets, state(held)
    )

    assert Breach.TURNOVER_BUDGET in decision.breaches
    assert state(held).turnover_to(decision.weights) == pytest.approx(0.20, abs=1e-9)
    # it ends between where it was and where it wanted to be
    assert 0.0 < decision.weights["U0"] < 0.10
    assert 0.0 < decision.weights["T0"] < 0.10


def test_turnover_budget_is_a_ceiling_not_a_target() -> None:
    """Trimming to max_positions can leave the move under budget. That is fine."""
    held = {f"T{i}": 0.10 for i in range(10)}
    targets = {f"U{i}": 0.10 for i in range(10)}
    decision = RiskManager(limits(max_turnover=0.20, max_positions=15)).review(
        targets, state(held)
    )
    assert state(held).turnover_to(decision.weights) <= 0.20 + 1e-9
    assert len(decision.weights) <= 15


def test_turnover_within_budget_is_left_alone() -> None:
    held = {f"T{i}": 0.10 for i in range(10)}
    targets = dict(held)
    targets["T0"] = 0.08
    targets["T1"] = 0.12
    decision = RiskManager(limits(max_turnover=0.20)).review(targets, state(held))
    assert Breach.TURNOVER_BUDGET not in decision.breaches


# -- liquidity ----------------------------------------------------------------------


def test_order_beyond_a_share_of_adv_is_limited() -> None:
    # NAV 1e6, target 10% = 100k of notional. ADV 200k, 5% cap = 10k allowed.
    decision = RiskManager(limits(min_positions=1)).review(
        {"THIN": 0.10}, state(nav=1_000_000.0), adv={"THIN": 200_000.0}
    )
    assert Breach.ILLIQUID_ORDER in decision.breaches
    assert decision.weights["THIN"] == pytest.approx(0.01)


def test_liquid_name_is_not_limited() -> None:
    decision = RiskManager(limits(min_positions=1)).review(
        {"BIG": 0.10}, state(nav=1_000_000.0), adv={"BIG": 500_000_000.0}
    )
    assert Breach.ILLIQUID_ORDER not in decision.breaches


def test_liquidity_check_is_skipped_without_adv() -> None:
    decision = RiskManager(limits(min_positions=1)).review({"X": 0.10}, state())
    assert Breach.ILLIQUID_ORDER not in decision.breaches


def test_liquidity_measures_the_trade_not_the_position() -> None:
    """A name already held at target needs no order, so no capacity is used."""
    decision = RiskManager(limits(min_positions=1)).review(
        {"THIN": 0.10}, state({"THIN": 0.10}, nav=1_000_000.0), adv={"THIN": 200_000.0}
    )
    assert Breach.ILLIQUID_ORDER not in decision.breaches


# -- veto --------------------------------------------------------------------------


def test_short_in_a_long_only_book_is_vetoed_not_clamped() -> None:
    decision = RiskManager(limits()).review({"A": 0.5, "B": -0.2}, state())
    assert decision.vetoed and decision.weights == {}
    assert Breach.SHORT_NOT_ALLOWED in decision.breaches
    assert "VETOED" in decision.describe()


def test_short_is_allowed_when_the_mandate_says_so() -> None:
    decision = RiskManager(limits(allow_short=True, min_positions=1)).review(
        {"A": 0.5, "B": -0.05}, state()
    )
    assert not decision.vetoed
    assert decision.weights["B"] == pytest.approx(-0.05)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "oops", None])
def test_unusable_weight_is_vetoed(bad) -> None:
    decision = RiskManager(limits()).review({"A": bad}, state())
    assert decision.vetoed
    assert Breach.NON_FINITE_WEIGHT in decision.breaches


def test_weighting_outside_the_universe_is_vetoed() -> None:
    decision = RiskManager(limits()).review({"GHOST": 0.1}, state(), universe=["A", "B"])
    assert decision.vetoed
    assert Breach.UNKNOWN_TICKER in decision.breaches


def test_a_halted_book_refuses_everything() -> None:
    decision = RiskManager(limits()).review({"A": 0.1}, state(halted=True))
    assert decision.vetoed and decision.weights == {}
    assert Breach.HALTED in decision.breaches


def test_empty_proposal_is_not_an_error() -> None:
    decision = RiskManager(limits()).review({}, state())
    assert decision.weights == {} and not decision.vetoed


def test_zero_weights_are_dropped_not_held() -> None:
    decision = RiskManager(limits(min_positions=1)).review({"A": 0.1, "B": 0.0}, state())
    assert set(decision.weights) == {"A"}


# -- the audit trail ---------------------------------------------------------------


def test_actions_name_the_ticker_and_the_change() -> None:
    decision = RiskManager(limits()).review({"A": 0.4, "B": 0.05}, state())
    action = next(a for a in decision.actions if a.breach is Breach.POSITION_CAP)
    assert action.ticker == "A"
    assert "0.4000" in action.detail and "0.1000" in action.detail
    assert "position_cap" in str(action) and "[A]" in str(action)


def test_the_manager_holds_no_state_between_calls() -> None:
    manager = RiskManager(limits())
    first = manager.review({"A": 0.4}, state())
    second = manager.review({f"T{i}": 0.10 for i in range(10)}, state())
    assert Breach.POSITION_CAP in first.breaches
    assert second.actions == (), "the earlier breach must not carry over"


# -- the turnover blend must not break the hard limits it runs after ------------


def test_turnover_blend_cannot_exceed_max_positions() -> None:
    """The blend reintroduces every held name; the count limit still binds.

    Measured consequence of getting this wrong: a 0.20 budget grew the book to
    207 names against a limit of 50 and left it permanently 43% invested,
    while the report showed no position-limit breach.
    """
    held = {f"H{i:03d}": 0.004 for i in range(200)}
    targets = {f"N{i:02d}": 0.02 for i in range(50)}
    decision = RiskManager(
        RiskLimits(max_position=0.04, min_positions=1, max_positions=50, max_turnover=0.20)
    ).review(targets, state(held))

    assert len(decision.weights) <= 50
    assert Breach.TOO_MANY_POSITIONS in decision.breaches


def test_turnover_blend_cannot_exceed_the_position_cap() -> None:
    held = {"A": 0.9}
    decision = RiskManager(
        RiskLimits(max_position=0.10, min_positions=1, max_positions=15, max_turnover=0.01)
    ).review({"A": 1.0}, state(held))
    assert all(w <= 0.10 + 1e-9 for w in decision.weights.values())


def test_budget_overrun_is_recorded_when_a_hard_limit_forces_it() -> None:
    """When the count limit and the budget conflict, the limit wins and says so."""
    held = {f"H{i:03d}": 0.004 for i in range(200)}
    targets = {f"N{i:02d}": 0.02 for i in range(50)}
    decision = RiskManager(
        RiskLimits(max_position=0.04, min_positions=1, max_positions=50, max_turnover=0.20)
    ).review(targets, state(held))

    overruns = [a for a in decision.actions if "overrun" in a.detail]
    assert overruns, "an unmet budget must be reported, not silently exceeded"
    assert str(overruns[0]).startswith("turnover_budget")


def test_enforcement_is_idempotent() -> None:
    """Re-running the clamps on an already-legal book changes nothing."""
    legal = {f"T{i}": 0.04 for i in range(25)}
    lim = RiskLimits(max_position=0.04, min_positions=1, max_positions=50, max_turnover=1.0)
    once = RiskManager(lim).review(legal, state())
    twice = RiskManager(lim).review(once.weights, state(once.weights))
    assert twice.weights == pytest.approx(once.weights)
    assert twice.actions == ()
