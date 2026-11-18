"""The three tests that decide whether a backtest number means anything.

The permutation test is checked the way it will be used: a strategy with
planted selection skill must pass, and equal weight on the same universe —
which earns a fine Sharpe purely by being long a drifting market — must fail.
That second assertion is the point of the whole module.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from hedge_fund.backtest import Backtester, BacktestConfig, EqualWeight
from hedge_fund.backtest.costs import ZERO_COST
from hedge_fund.data.lake import PRICE_COLUMNS, PRICES, Lake
from hedge_fund.validation import (
    Trial,
    TrialLog,
    block_bootstrap,
    deflated_sharpe,
    permutation_test,
    validate,
)
from hedge_fund.validation.bootstrap import circular_block_indices
from hedge_fund.validation.deflated import (
    expected_max_sharpe,
    period_sharpe,
    probabilistic_sharpe,
)
from hedge_fund.validation.permutation import live_tickers_by_date, shuffle_decisions

# -- bootstrap -------------------------------------------------------------------


def test_block_indices_have_right_shape_and_range() -> None:
    idx = circular_block_indices(n_obs=100, block_len=21, n_paths=7, rng=np.random.default_rng(0))
    assert idx.shape == (7, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_blocks_are_contiguous_modulo_wrap() -> None:
    idx = circular_block_indices(n_obs=50, block_len=10, n_paths=1, rng=np.random.default_rng(0))[0]
    steps = np.diff(idx[:10])  # first block
    assert set(steps.tolist()) <= {1, 1 - 50}, "a block must step by one, wrapping at the end"


def test_block_len_larger_than_series_is_clamped() -> None:
    idx = circular_block_indices(n_obs=5, block_len=99, n_paths=3, rng=np.random.default_rng(0))
    assert idx.shape == (3, 5)


def test_bootstrap_of_a_constant_series_has_zero_spread() -> None:
    r = np.full(300, 0.001)
    out = block_bootstrap(r, n_paths=200, seed=0)
    assert np.allclose(out.max_dd_paths, 0.0)
    assert out.sharpe_ci() == (0.0, 0.0)  # std is zero -> sharpe defined as 0


def test_bootstrap_is_reproducible_by_seed() -> None:
    r = np.random.default_rng(3).normal(0.0005, 0.01, 400)
    a = block_bootstrap(r, n_paths=100, seed=11)
    b = block_bootstrap(r, n_paths=100, seed=11)
    assert np.array_equal(a.max_dd_paths, b.max_dd_paths)


def test_prob_dd_exceeds_is_monotone_in_threshold() -> None:
    r = np.random.default_rng(5).normal(0.0, 0.02, 500)
    out = block_bootstrap(r, n_paths=300, seed=0)
    probs = [out.prob_dd_exceeds(t) for t in (0.05, 0.10, 0.20, 0.40)]
    assert probs == sorted(probs, reverse=True)
    assert out.dd_percentile(99) >= out.dd_percentile(50)


def test_bootstrap_describe_flags_ci_through_zero() -> None:
    r = np.random.default_rng(9).normal(0.0, 0.01, 300)
    text = block_bootstrap(r, n_paths=200, seed=0).describe()
    assert "includes zero" in text


# -- deflated Sharpe ------------------------------------------------------------


def test_expected_max_sharpe_is_zero_for_a_single_trial() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_expected_max_sharpe_grows_with_trials_and_variance() -> None:
    a = expected_max_sharpe(10, 0.01)
    b = expected_max_sharpe(100, 0.01)
    c = expected_max_sharpe(100, 0.04)
    assert 0 < a < b < c
    # variance scales E[max] by sqrt: quadrupling variance doubles it
    assert c == pytest.approx(2 * b)


def test_psr_of_zero_sharpe_against_zero_is_one_half() -> None:
    assert probabilistic_sharpe(0.0, 0.0, 500, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_rises_with_sample_size() -> None:
    sr = 0.05  # per day, ~0.8 annual
    assert probabilistic_sharpe(sr, 0.0, 50, 0.0, 3.0) < probabilistic_sharpe(sr, 0.0, 1000, 0.0, 3.0)


def test_fat_tails_and_negative_skew_lower_psr() -> None:
    base = probabilistic_sharpe(0.05, 0.0, 500, 0.0, 3.0)
    assert probabilistic_sharpe(0.05, 0.0, 500, -1.0, 3.0) < base
    assert probabilistic_sharpe(0.05, 0.0, 500, 0.0, 8.0) < base


def test_dsr_equals_psr_with_no_other_trials() -> None:
    r = np.random.default_rng(1).normal(0.001, 0.01, 400)
    out = deflated_sharpe(r)
    assert out.n_trials == 1
    assert out.expected_max_sharpe == 0.0
    assert out.dsr == pytest.approx(out.psr)


def test_dsr_is_deflated_by_a_history_of_trials() -> None:
    """The same returns, after 40 other looks, are less convincing."""
    r = np.random.default_rng(2).normal(0.0008, 0.01, 400)
    alone = deflated_sharpe(r)
    others = list(np.random.default_rng(3).normal(0.0, 0.03, 40)) + [period_sharpe(r)]
    crowded = deflated_sharpe(r, others)
    assert crowded.n_trials == 41
    assert crowded.expected_max_sharpe > 0
    assert crowded.dsr < alone.dsr


def test_period_sharpe_matches_annual_metric() -> None:
    from hedge_fund.backtest.metrics import sharpe

    r = np.random.default_rng(4).normal(0.001, 0.01, 300)
    assert period_sharpe(r) * math.sqrt(252) == pytest.approx(sharpe(r))


# -- permutation ----------------------------------------------------------------


@pytest.fixture()
def drift_lake(tmp_path) -> tuple[Lake, list[str]]:
    """Ten noisy tickers, one of which drifts +0.4%/day. 300 trading days."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", periods=300)
    tickers = [f"T{i:02d}" for i in range(10)]
    rets = rng.normal(0.0, 0.01, size=(300, 10))
    rets[:, 0] += 0.004
    prices = 100 * np.cumprod(1 + rets, axis=0)
    rows = [
        {
            "ticker": t,
            "date": d.date(),
            "open": p,
            "high": p,
            "low": p,
            "close": p,
            "adj_close": p,
            "volume": 1e6,
        }
        for i, d in enumerate(days)
        for t, p in zip(tickers, prices[i])
    ]
    lake = Lake(tmp_path / "lake")
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))
    return lake, tickers


class _PickT00:
    name = "pick_t00"

    def target_weights(self, ctx):
        return {"T00": 0.5, "T01": 0.25, "T02": 0.25}


_CFG = BacktestConfig(
    start=date(2023, 2, 1), end=date(2024, 2, 1), rebalance="weekly", costs=ZERO_COST
)


def test_shuffle_keeps_weights_and_position_count_per_day() -> None:
    decisions = {date(2024, 1, 1): {"A": 0.5, "B": 0.3}, date(2024, 1, 2): {}}
    live = {date(2024, 1, 1): ["A", "B", "C", "D"], date(2024, 1, 2): ["A"]}
    out = shuffle_decisions(decisions, live, np.random.default_rng(0))
    assert sorted(out[date(2024, 1, 1)].values()) == [0.3, 0.5]
    assert len(out[date(2024, 1, 1)]) == 2
    assert set(out[date(2024, 1, 1)]) <= {"A", "B", "C", "D"}
    assert out[date(2024, 1, 2)] == {}


def test_shuffle_refuses_when_too_few_live_names() -> None:
    with pytest.raises(ValueError, match="tradeable"):
        shuffle_decisions(
            {date(2024, 1, 1): {"A": 0.5, "B": 0.5}},
            {date(2024, 1, 1): ["A"]},
            np.random.default_rng(0),
        )


def test_live_tickers_excludes_nan_and_unknown_dates() -> None:
    panel = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [1.0, 1.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    out = live_tickers_by_date(panel, [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    assert out == {date(2024, 1, 1): ["A", "B"], date(2024, 1, 2): ["B"], date(2024, 1, 3): []}


def test_planted_skill_passes_and_equal_weight_fails(drift_lake) -> None:
    """Equal weight earns a good Sharpe here with zero selection skill.

    A test that cannot fail a profitable strategy is not a test. This one
    must call equal weight out, and must let the strategy that knows the
    drifting name through.
    """
    lake, tickers = drift_lake
    bt = Backtester(lake, tickers)

    skilled = bt.run(_PickT00(), _CFG)
    naive = bt.run(EqualWeight(), _CFG)
    assert naive.metrics.sharpe > 1.0, "the trap only works if naive looks good"

    p_skilled = permutation_test(bt, skilled, n_permutations=60, seed=0).p_value
    p_naive = permutation_test(bt, naive, n_permutations=60, seed=0).p_value
    assert p_skilled <= 0.05
    assert p_naive > 0.5


def test_permutation_is_reproducible_by_seed(drift_lake) -> None:
    lake, tickers = drift_lake
    bt = Backtester(lake, tickers)
    res = bt.run(_PickT00(), _CFG)
    a = permutation_test(bt, res, n_permutations=20, seed=7)
    b = permutation_test(bt, res, n_permutations=20, seed=7)
    assert np.array_equal(a.null_sharpes, b.null_sharpes)


def test_permutation_requires_decisions(drift_lake) -> None:
    lake, tickers = drift_lake
    bt = Backtester(lake, tickers)
    res = bt.run(_PickT00(), _CFG)
    hollow = res.__class__(**{**res.__dict__, "decisions": {}})
    with pytest.raises(ValueError, match="decisions"):
        permutation_test(bt, hollow, n_permutations=5)


# -- the combined report ----------------------------------------------------------


def test_validate_verdicts(drift_lake) -> None:
    lake, tickers = drift_lake
    bt = Backtester(lake, tickers)
    good = validate(bt, bt.run(_PickT00(), _CFG), n_permutations=40, n_bootstrap=200, seed=0)
    bad = validate(bt, bt.run(EqualWeight(), _CFG), n_permutations=40, n_bootstrap=200, seed=0)
    assert good.verdict.startswith("PASS")
    assert bad.verdict.startswith("FAIL")
    assert "selection not better than random" in bad.verdict
    assert "verdict:" in good.describe()


def test_validate_can_skip_permutation(drift_lake) -> None:
    lake, tickers = drift_lake
    bt = Backtester(lake, tickers)
    rep = validate(bt, bt.run(EqualWeight(), _CFG), n_permutations=0, n_bootstrap=100)
    assert rep.permutation is None
    assert "permutation" not in rep.describe()


# -- trial log --------------------------------------------------------------------


def test_trial_log_dedupes_by_configuration(tmp_path) -> None:
    log = TrialLog(tmp_path / "trials.jsonl")
    for k in range(3):
        log.append(Trial.build("x", {"k": k}, date(2023, 1, 1), date(2024, 1, 1), 250, 0.01 * k))
    log.append(Trial.build("x", {"k": 0}, date(2023, 1, 1), date(2024, 1, 1), 250, 0.5))
    assert len(log) == 3, "re-running a configuration is not a new trial"
    assert sorted(log.sharpes()) == [0.01, 0.02, 0.5], "rerun replaces its earlier record"


def test_trial_id_is_stable_across_param_order(tmp_path) -> None:
    a = Trial.build("x", {"a": 1, "b": 2}, date(2023, 1, 1), date(2024, 1, 1), 1, 0.0)
    b = Trial.build("x", {"b": 2, "a": 1}, date(2023, 1, 1), date(2024, 1, 1), 1, 0.0)
    c = Trial.build("x", {"a": 1, "b": 3}, date(2023, 1, 1), date(2024, 1, 1), 1, 0.0)
    assert a.trial_id == b.trial_id != c.trial_id


def test_missing_log_is_empty_not_error(tmp_path) -> None:
    log = TrialLog(tmp_path / "nope.jsonl")
    assert list(log) == []
    assert log.sharpes() == []
    assert len(log) == 0


# --- the Deflated Sharpe must compare like with like ----------------------


def _trial(log, strategy, start, end, n_obs, sharpe_period, **params):
    from hedge_fund.validation import Trial

    log.append(
        Trial.build(
            strategy=strategy, params=params, start=start, end=end,
            n_obs=n_obs, sharpe_period=sharpe_period,
        )
    )


def test_a_short_run_is_not_pooled_with_a_long_one(tmp_path) -> None:
    """A 61-day exploratory run is not a configuration of a nine-year test.

    Measured on the real log: one such trial at 0.3426/day against a cluster
    at 0.065 raised the variance of 19 trials by a factor of 370. E[max SR]
    scales with its square root, so a genuine nine-year result's deflated
    Sharpe read 0.6% instead of passing. Pooling windows lets any throwaway
    run disqualify every real one.
    """
    from hedge_fund.validation import TrialLog

    log = TrialLog(tmp_path / "trials.jsonl")
    long_start, long_end = date(2016, 1, 1), date(2024, 12, 31)
    for i, sr in enumerate((0.060, 0.065, 0.067)):
        _trial(log, "s", long_start, long_end, 2264, sr, top_n=i)
    _trial(log, "s", date(2024, 1, 1), date(2024, 3, 31), 61, 0.3426, top_n=50)

    assert len(log.sharpes()) == 4
    scoped = log.sharpes(long_start, long_end)
    assert len(scoped) == 3
    assert max(scoped) < 0.1


def test_scoping_keeps_every_trial_on_the_window_asked_for(tmp_path) -> None:
    from hedge_fund.validation import TrialLog

    log = TrialLog(tmp_path / "trials.jsonl")
    start, end = date(2016, 1, 1), date(2024, 12, 31)
    for i, sr in enumerate((0.060, 0.065, 0.067, 0.071)):
        _trial(log, "s", start, end, 2264, sr, top_n=i)

    # Same window: every configuration tried on this data still counts.
    assert sorted(log.sharpes(start, end)) == [0.060, 0.065, 0.067, 0.071]
