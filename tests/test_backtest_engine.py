"""The simulator's arithmetic, pinned by hand-computable cases.

Prices here are chosen so every expected number can be worked out on paper:
one ticker compounds +10%/day, the other is flat. If 50/50 does not earn
exactly 5% a day, the engine is wrong, and no amount of realistic data would
have shown it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hedge_fund.backtest import (
    Backtester,
    BacktestConfig,
    EqualWeight,
    RecordingStrategy,
    ReplayStrategy,
)
from hedge_fund.backtest.costs import ZERO_COST, CostModel
from hedge_fund.backtest.engine import WeightError, decision_dates, replay
from hedge_fund.data.lake import PRICE_COLUMNS, PRICES, Lake
from hedge_fund.portfolio.state import PortfolioState

DAYS = pd.bdate_range("2024-01-01", periods=6)  # Mon 1 Jan .. Mon 8 Jan
GROWTH = [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]  # +10%/day
FLAT = [50.0] * 6


def _row(ticker: str, d: pd.Timestamp, px: float) -> dict:
    return {
        "ticker": ticker,
        "date": d.date(),
        "open": px,
        "high": px,
        "low": px,
        "close": px,
        "adj_close": px,
        "volume": 1000.0,
    }


@pytest.fixture()
def lake(tmp_path) -> Lake:
    lake = Lake(tmp_path / "lake")
    rows = [
        _row(t, d, px)
        for d, a, b in zip(DAYS, GROWTH, FLAT)
        for t, px in (("AAA", a), ("BBB", b))
    ]
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))
    return lake


def _config(**overrides) -> BacktestConfig:
    base = dict(
        start=date(2024, 1, 1), end=date(2024, 1, 8), execution_lag=1, costs=ZERO_COST
    )
    base.update(overrides)
    return BacktestConfig(**base)


# -- return attribution and execution lag --------------------------------------


def test_fifty_fifty_earns_half_the_growth_each_day(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config())
    assert res.net_returns.iloc[0] == 0.0, "nothing is held before the first fill"
    assert np.allclose(res.net_returns.iloc[1:], 0.05)
    assert res.nav.iloc[-1] == pytest.approx(1.05**5)


def test_lag_two_delays_first_return_by_one_more_day(lake: Lake) -> None:
    """Decide T, fill T+1, earn from T+2: two all-cash days, not one."""
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config(execution_lag=2))
    assert list(res.net_returns.iloc[:2]) == [0.0, 0.0]
    assert np.allclose(res.net_returns.iloc[2:], 0.05)


def test_default_lag_is_two() -> None:
    assert BacktestConfig(start=date(2024, 1, 1), end=date(2024, 1, 2)).execution_lag == 2


def test_lag_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="execution_lag"):
        _config(execution_lag=0)


def test_decisions_that_never_fill_are_dropped(lake: Lake) -> None:
    """With lag 2 the last day's decision has no fill day and is discarded."""
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config(execution_lag=2))
    assert len(res.decisions) == 6, "the strategy is still consulted every day"
    # ...but the final fill happened on day 6 for day 5's decision; day 6's is pending forever.
    assert res.turnover.iloc[-1] > 0  # fill of day-5 decision
    assert res.weights.iloc[-1].sum() == pytest.approx(1.0)


# -- costs are charged on turnover, and drift is not turnover -------------------


def test_first_fill_pays_for_half_the_book_and_later_days_only_for_drift(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(
        EqualWeight(), _config(costs=CostModel(spread_bps=50, impact_bps=0))
    )
    # all-cash -> 50/50 is turnover 0.5; cost = 2 * 0.005 * 0.5
    assert res.turnover.iloc[0] == pytest.approx(0.5)
    assert res.costs.iloc[0] == pytest.approx(0.005)
    # after a +10%/0% day, 50/50 drifts to (0.55, 0.5)/1.05; restoring it is tiny
    drift = 0.55 / 1.05 - 0.5
    assert np.allclose(res.turnover.iloc[1:], drift)
    assert np.allclose(res.net_returns.iloc[1:], 0.05 - 2 * 0.005 * drift)


def test_zero_cost_model_charges_nothing(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config())
    assert res.costs.sum() == 0.0
    assert res.metrics.total_cost_drag == 0.0


def test_repeating_the_same_targets_pays_only_for_drift(lake: Lake) -> None:
    """A strategy that re-decides an unchanged book must not re-buy it."""

    class Sticky:
        name = "sticky"

        def target_weights(self, ctx):
            return {"BBB": 1.0}  # flat asset: no drift at all

    res = Backtester(lake, ["AAA", "BBB"]).run(
        Sticky(), _config(costs=CostModel(spread_bps=50, impact_bps=0))
    )
    # Turnover is (purchases + sales) / 2 — the standard one-way convention —
    # so deploying the whole book from cash is 0.5, not 1.0. The cost is still
    # charged on the full 100% of NAV bought: 2 * per_side * 0.5 = per_side.
    assert res.turnover.iloc[0] == pytest.approx(0.5)
    assert res.costs.iloc[0] == pytest.approx(0.005)
    assert (res.turnover.iloc[1:] == 0.0).all()


# -- the strategy can never see past its own asof ------------------------------


def test_strategy_never_sees_a_bar_after_asof(lake: Lake) -> None:
    rec = RecordingStrategy(EqualWeight())
    Backtester(lake, ["AAA", "BBB"]).run(rec, _config())
    assert len(rec.seen) == 6
    for asof, newest in rec.seen:
        assert newest is not None
        assert newest <= asof, f"view for {asof} exposed a bar dated {newest}"


def test_strategy_sees_exactly_up_to_asof(lake: Lake) -> None:
    rec = RecordingStrategy(EqualWeight())
    Backtester(lake, ["AAA", "BBB"]).run(rec, _config())
    assert [newest for _, newest in rec.seen] == [d.date() for d in DAYS]


# -- weight validation ----------------------------------------------------------


class _Fixed:
    name = "fixed"

    def __init__(self, weights):
        self._w = weights

    def target_weights(self, ctx):
        return dict(self._w)


def test_gross_exposure_over_limit_raises(lake: Lake) -> None:
    with pytest.raises(WeightError, match="gross exposure"):
        Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 0.8, "BBB": 0.8}), _config())


def test_short_weight_raises_unless_allowed(lake: Lake) -> None:
    bt = Backtester(lake, ["AAA", "BBB"])
    with pytest.raises(WeightError, match="negative"):
        bt.run(_Fixed({"AAA": 0.5, "BBB": -0.3}), _config())
    res = bt.run(_Fixed({"AAA": 0.5, "BBB": -0.3}), _config(allow_short=True))
    assert res.weights.iloc[-1]["BBB"] < 0


def test_unknown_ticker_raises(lake: Lake) -> None:
    with pytest.raises(WeightError, match="no prices"):
        Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"ZZZ": 1.0}), _config())


def test_non_finite_weight_raises(lake: Lake) -> None:
    with pytest.raises(WeightError):
        Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": float("nan")}), _config())


def test_all_cash_is_allowed(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({}), _config())
    assert (res.net_returns == 0.0).all()
    assert res.nav.iloc[-1] == 1.0


# -- rebalance schedule ---------------------------------------------------------


def test_decision_dates_weekly_is_last_trading_day_of_each_week() -> None:
    out = decision_dates(DAYS, "weekly")
    assert [d.date() for d in out] == [date(2024, 1, 5), date(2024, 1, 8)]


def test_decision_dates_monthly() -> None:
    days = pd.bdate_range("2024-01-01", "2024-03-15")
    out = decision_dates(days, "monthly")
    assert [d.date() for d in out] == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 15)]


def test_weekly_rebalance_consults_strategy_only_on_fridays(lake: Lake) -> None:
    rec = RecordingStrategy(EqualWeight())
    Backtester(lake, ["AAA", "BBB"]).run(rec, _config(rebalance="weekly"))
    assert [a for a, _ in rec.seen] == [date(2024, 1, 5), date(2024, 1, 8)]


def test_unknown_rebalance_raises() -> None:
    with pytest.raises(ValueError, match="rebalance"):
        decision_dates(DAYS, "fortnightly")  # type: ignore[arg-type]


# -- panel edge cases -----------------------------------------------------------


def test_stale_days_counted_when_a_held_name_stops_printing(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    rows = [_row("AAA", d, px) for d, px in zip(DAYS, GROWTH)]
    rows += [_row("BBB", d, 50.0) for d in DAYS[:3]]  # BBB vanishes after day 3
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))

    res = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 0.5, "BBB": 0.5}), _config())
    assert res.stale_days == 3, "BBB held on days 4,5,6 off a forward-filled price"
    assert "stale" in res.describe()


def test_window_outside_lake_dates_raises(lake: Lake) -> None:
    with pytest.raises(ValueError, match="no trading days"):
        Backtester(lake, ["AAA", "BBB"]).run(
            EqualWeight(), _config(start=date(2030, 1, 1), end=date(2030, 2, 1))
        )


def test_empty_universe_raises(lake: Lake) -> None:
    with pytest.raises(ValueError, match="universe"):
        Backtester(lake, [])


def test_benchmark_returns_are_reported_when_present(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config(benchmark="AAA"))
    assert res.benchmark_returns is not None
    assert np.allclose(res.benchmark_returns.iloc[1:], 0.10)
    assert "vs AAA" in res.describe()


def test_missing_benchmark_is_none_not_error(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config(benchmark="SPY"))
    assert res.benchmark_returns is None


# -- replay -----------------------------------------------------------------------


def test_replay_reproduces_the_original_run(lake: Lake) -> None:
    bt = Backtester(lake, ["AAA", "BBB"])
    original = bt.run(EqualWeight(), _config())
    again = replay(lake, ["AAA", "BBB"], original.decisions, _config())
    assert np.allclose(again.net_returns, original.net_returns)
    assert again.strategy == "replay"


def test_replay_uses_recorded_decisions_not_the_view(lake: Lake) -> None:
    """Change one recorded day; the replay must follow the record, not the data."""
    bt = Backtester(lake, ["AAA", "BBB"])
    original = bt.run(EqualWeight(), _config())
    edited = dict(original.decisions)
    edited[date(2024, 1, 3)] = {"BBB": 1.0}  # flat asset on that day only
    res = bt.run(ReplayStrategy(decisions=edited), _config())
    assert res.weights.loc["2024-01-03"]["BBB"] == 1.0
    assert res.net_returns.loc["2024-01-04"] == 0.0  # earned BBB's 0% that day
    assert res.net_returns.loc["2024-01-05"] == pytest.approx(0.05)  # back to 50/50


def test_replay_refuses_a_date_it_has_no_decision_for(lake: Lake) -> None:
    """A schedule replayed under a different rebalance must fail, not go to cash."""
    s = ReplayStrategy(decisions={date(2024, 1, 1): {"AAA": 1.0}})
    with pytest.raises(ValueError, match="no decision for 2024-01-02"):
        Backtester(lake, ["AAA", "BBB"]).run(s, _config())


# -- the kill switch: enforced by the engine, because only it knows NAV ----------


class _AlwaysLong:
    """Holds AAA at full weight every day, so the book tracks AAA exactly."""

    name = "always_long"

    def target_weights(self, ctx):
        return {"AAA": 1.0}


@pytest.fixture()
def falling_lake(tmp_path) -> Lake:
    """AAA rises 3 days then falls hard, so a drawdown stop must fire."""
    lake = Lake(tmp_path / "falling")
    days = pd.bdate_range("2024-01-01", periods=10)
    prices = [100.0, 110.0, 120.0, 108.0, 96.0, 84.0, 72.0, 78.0, 84.0, 90.0]
    rows = [_row("AAA", d, p) for d, p in zip(days, prices)]
    rows += [_row("BBB", d, 50.0) for d in days]
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))
    return lake


def _kill_config(**overrides) -> BacktestConfig:
    base = dict(
        start=date(2024, 1, 1), end=date(2024, 1, 12), execution_lag=1, costs=ZERO_COST
    )
    base.update(overrides)
    return BacktestConfig(**base)


def test_no_kill_switch_by_default(falling_lake: Lake) -> None:
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(_AlwaysLong(), _kill_config())
    assert res.config.kill_switch_dd is None
    assert res.weights.iloc[-1]["AAA"] == pytest.approx(1.0), "still invested"


def test_kill_switch_flattens_the_book_and_keeps_it_flat(falling_lake: Lake) -> None:
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(
        _AlwaysLong(), _kill_config(kill_switch_dd=0.20)
    )
    weights = res.weights["AAA"]
    fired_on = weights[weights == 0.0].index.min()
    assert pd.notna(fired_on), "the stop never fired on a 40% fall"
    # Once flat it stays flat for the rest of the run, recovery included. The
    # firing day itself still carries the loss that triggered the stop.
    assert (weights.loc[fired_on:] == 0.0).all()
    assert res.net_returns.loc[fired_on] < 0.0
    assert (res.net_returns.loc[fired_on:].iloc[1:] == 0.0).all()


def test_kill_switch_fires_only_past_the_threshold(falling_lake: Lake) -> None:
    """A stop above the realised drawdown must never fire."""
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(
        _AlwaysLong(), _kill_config(kill_switch_dd=0.95)
    )
    assert (res.weights["AAA"] > 0).all()


def test_cooldown_re_enters_after_a_fixed_wait(tmp_path) -> None:
    """"Resume when the drawdown recovers" cannot work: a flat book's NAV
    never moves, so it can never reach a new high. A cooldown terminates."""
    lake = Lake(tmp_path / "vshape")
    days = pd.bdate_range("2024-01-01", periods=12)
    # down 40%, then back above the old high
    prices = [100.0, 100.0, 60.0, 60.0, 80.0, 100.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
    rows = [_row("AAA", d, p) for d, p in zip(days, prices)]
    rows += [_row("BBB", d, 50.0) for d in days]
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))

    halting = Backtester(lake, ["AAA", "BBB"]).run(
        _AlwaysLong(),
        BacktestConfig(start=date(2024, 1, 1), end=date(2024, 1, 17),
                       execution_lag=1, costs=ZERO_COST, kill_switch_dd=0.20),
    )
    resuming = Backtester(lake, ["AAA", "BBB"]).run(
        _AlwaysLong(),
        BacktestConfig(start=date(2024, 1, 1), end=date(2024, 1, 17),
                       execution_lag=1, costs=ZERO_COST, kill_switch_dd=0.20,
                       kill_switch_mode="cooldown", kill_switch_cooldown_days=2),
    )
    assert halting.weights["AAA"].iloc[-1] == 0.0, "halt sits out the recovery"
    assert resuming.weights["AAA"].iloc[-1] > 0.0, "cooldown re-enters"
    assert resuming.nav.iloc[-1] > halting.nav.iloc[-1]


def test_liquidation_cost_is_charged_once(falling_lake: Lake) -> None:
    """The stop sells the book, and that sale is paid for exactly one time."""
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(
        _AlwaysLong(),
        _kill_config(kill_switch_dd=0.20, costs=CostModel(spread_bps=50, impact_bps=0)),
    )
    fired = res.turnover[res.weights["AAA"] == 0.0].index.min()
    # Going flat from a full position is turnover 0.5 on the stop day...
    assert res.turnover.loc[fired] == pytest.approx(0.5, abs=1e-9)
    assert res.costs.loc[fired] == pytest.approx(0.005, abs=1e-9)
    # ...and nothing trades afterwards.
    assert (res.turnover.loc[fired:].iloc[1:] == 0.0).all()


def test_nav_compounds_the_liquidation_cost_exactly_once(falling_lake: Lake) -> None:
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(
        _AlwaysLong(),
        _kill_config(kill_switch_dd=0.20, costs=CostModel(spread_bps=50, impact_bps=0)),
    )
    rebuilt = float(np.cumprod(1.0 + res.net_returns.to_numpy())[-1])
    assert res.nav.iloc[-1] == pytest.approx(rebuilt, rel=1e-12), (
        "NAV and the net return series disagree, so a cost was applied twice "
        "or compounded in the wrong order"
    )


def test_invalid_kill_switch_is_rejected() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="kill_switch_dd"):
            _kill_config(kill_switch_dd=bad)


# -- observe(): the strategy is told the book, and cannot refuse the stop --------


class _Watcher:
    """Records every PortfolioState it is shown."""

    name = "watcher"

    def __init__(self) -> None:
        self.seen: list[PortfolioState] = []

    def observe(self, state: PortfolioState) -> None:
        self.seen.append(state)

    def target_weights(self, ctx):
        return {"AAA": 1.0}


def test_observe_is_called_with_the_drifted_book(falling_lake: Lake) -> None:
    watcher = _Watcher()
    Backtester(falling_lake, ["AAA", "BBB"]).run(watcher, _kill_config())

    assert len(watcher.seen) == 10, "once per decision date"
    first, second = watcher.seen[0], watcher.seen[1]
    assert first.weights == {}, "nothing is held before the first fill"
    assert first.nav == pytest.approx(1.0)
    assert second.weight("AAA") == pytest.approx(1.0), "fully invested by the second look"
    assert second.nav > 1.0, "AAA rose 10% on day two"


def test_observed_drawdown_tracks_the_fall(falling_lake: Lake) -> None:
    watcher = _Watcher()
    Backtester(falling_lake, ["AAA", "BBB"]).run(watcher, _kill_config())
    assert max(s.drawdown for s in watcher.seen) > 0.3
    assert watcher.seen[0].drawdown == 0.0


def test_a_strategy_cannot_trade_through_a_halt(falling_lake: Lake) -> None:
    """_AlwaysLong asks for full exposure every day; the engine overrules it."""
    watcher = _Watcher()
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(
        watcher, _kill_config(kill_switch_dd=0.20)
    )
    assert any(s.halted for s in watcher.seen), "the strategy is told it is halted"
    halted_dates = [s.asof for s in watcher.seen if s.halted]
    for asof in halted_dates:
        assert res.decisions[asof] == {}, "no decision is recorded while halted"
    assert res.weights["AAA"].iloc[-1] == 0.0


def test_a_strategy_without_observe_still_runs(falling_lake: Lake) -> None:
    """EqualWeight knows nothing of portfolio state and must be unaffected."""
    res = Backtester(falling_lake, ["AAA", "BBB"]).run(EqualWeight(), _kill_config())
    assert res.metrics.days == 10


# -- gross exposure: the number that stops a de-levered book reading as a win ----


def test_mean_gross_exposure_is_reported(lake: Lake) -> None:
    """Weights are recorded after each day's fill, so lag decides the average.

    With lag 1 the fill lands on the decision's own close and the book is
    invested on all six days. With lag 2 day one has no fill yet.
    """
    same_close = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config())
    assert same_close.metrics.mean_gross_exposure == pytest.approx(1.0, abs=1e-9)
    assert "mean gross" in same_close.describe()

    delayed = Backtester(lake, ["AAA", "BBB"]).run(EqualWeight(), _config(execution_lag=2))
    assert delayed.metrics.mean_gross_exposure == pytest.approx(5 / 6, abs=1e-9)


def test_a_half_invested_book_is_flagged(lake: Lake) -> None:
    """Sharpe cannot distinguish this from a fully invested book, so the
    report must. A measured step 5 run scored Sharpe 1.06 at 37% invested."""
    res = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 0.3}), _config())
    assert res.metrics.underinvested
    assert "WARNING" in res.describe()
    assert "scale-invariant" in res.describe()


def test_a_fully_invested_book_is_not_flagged(lake: Lake) -> None:
    res = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 1.0}), _config())
    assert not res.metrics.underinvested
    assert "WARNING" not in res.describe()


def test_sharpe_is_blind_to_cash_but_cagr_is_not(lake: Lake) -> None:
    """The reason the warning exists, stated as a test."""
    full = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 1.0}), _config())
    half = Backtester(lake, ["AAA", "BBB"]).run(_Fixed({"AAA": 0.5}), _config())
    assert full.metrics.sharpe == pytest.approx(half.metrics.sharpe, rel=1e-9)
    assert half.metrics.cagr < full.metrics.cagr
    assert half.metrics.mean_gross_exposure < full.metrics.mean_gross_exposure
