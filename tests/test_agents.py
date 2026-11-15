"""The five Phase 0 agents, and the scoring they share.

Two things are worth more than the rest here: that an agent which cannot
evaluate a ticker omits it rather than scoring it zero, and that every factor
is the standard construction rather than something tuned to this fixture.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hedge_fund.agents import fundamentals as f
from hedge_fund.agents import phase_0_agents
from hedge_fund.agents.fundamental import QualityAgent, ValueAgent
from hedge_fund.agents.price import LowVolAgent, MomentumAgent, ReversalAgent
from hedge_fund.agents.scoring import (
    combine_scores,
    coverage_confidence,
    cross_sectional_scores,
    staleness_confidence,
)
from hedge_fund.core.agent import Agent, Context
from hedge_fund.data.lake import (
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTALS,
    PRICE_COLUMNS,
    PRICES,
    Lake,
)

# -- scoring --------------------------------------------------------------------


def test_scores_span_minus_one_to_one() -> None:
    out = cross_sectional_scores({"A": 1.0, "B": 2.0, "C": 3.0})
    assert out == {"A": -1.0, "B": 0.0, "C": 1.0}


def test_higher_is_better_false_inverts() -> None:
    out = cross_sectional_scores({"A": 1.0, "B": 3.0}, higher_is_better=False)
    assert out == {"A": 1.0, "B": -1.0}


def test_ties_share_the_average_rank() -> None:
    out = cross_sectional_scores({"A": 5.0, "B": 5.0, "C": 9.0})
    assert out["A"] == out["B"] == pytest.approx(-0.5)
    assert out["C"] == 1.0


def test_unevaluable_tickers_are_dropped_not_zeroed() -> None:
    """Omission means "no data"; 0.0 means "no edge". They must not merge."""
    out = cross_sectional_scores(
        {"A": 1.0, "B": float("nan"), "C": float("inf"), "D": None, "E": 2.0}  # type: ignore[dict-item]
    )
    assert set(out) == {"A", "E"}


def test_single_observation_scores_zero_not_one() -> None:
    assert cross_sectional_scores({"A": 42.0}) == {"A": 0.0}


def test_empty_input_is_empty_output() -> None:
    assert cross_sectional_scores({}) == {}


def test_combine_averages_only_the_parts_present() -> None:
    out = combine_scores(
        {"x": {"A": 1.0, "B": 0.0}, "y": {"A": 0.0}}, min_parts=1
    )
    assert out["A"] == pytest.approx(0.5)
    assert out["B"] == pytest.approx(0.0), "a missing part must not dilute a present one"


def test_combine_enforces_min_parts() -> None:
    out = combine_scores({"x": {"A": 1.0, "B": 1.0}, "y": {"A": 1.0}}, min_parts=2)
    assert set(out) == {"A"}


@pytest.mark.parametrize(
    "available,required,expected", [(0, 60, 0.3), (60, 60, 1.0), (30, 60, 0.65)]
)
def test_coverage_confidence_scales_linearly(available, required, expected) -> None:
    assert coverage_confidence(available, required) == pytest.approx(expected)


def test_staleness_confidence_decays_with_filing_age() -> None:
    assert staleness_confidence(30) == 1.0
    assert staleness_confidence(120) == 1.0
    assert staleness_confidence(500) == pytest.approx(0.3)
    assert 0.3 < staleness_confidence(300) < 1.0


# -- price agents ----------------------------------------------------------------


def test_momentum_measures_twelve_months_skipping_the_last(tmp_path) -> None:
    """Hand-built series: the skipped month must not count toward the score."""
    lake = Lake(tmp_path / "lake")
    days = pd.bdate_range("2022-01-03", periods=300)
    # AAA: flat for the first 279 bars then +50% in the final 21 (skipped window)
    # BBB: +20% over the measured window then flat
    aaa = np.concatenate([np.full(279, 100.0), np.linspace(100.0, 150.0, 21)])
    bbb = np.concatenate([np.linspace(100.0, 120.0, 279), np.full(21, 120.0)])
    rows = [
        {
            "ticker": t, "date": d.date(), "open": p, "high": p, "low": p,
            "close": p, "adj_close": p, "volume": 1e6,
        }
        for d, a, b in zip(days, aaa, bbb)
        for t, p in (("AAA", a), ("BBB", b))
    ]
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))

    asof = days[-1].date()
    ctx = Context(asof=asof, view=lake.view(asof), universe=("AAA", "BBB"))
    signals = {s.ticker: s for s in MomentumAgent().evaluate(ctx)}

    assert signals["BBB"].score > signals["AAA"].score, (
        "AAA's entire gain is inside the skipped final month, so BBB must rank "
        "higher on 12-1 momentum"
    )
    assert "+0.0%" in signals["AAA"].rationale


def test_momentum_omits_tickers_without_the_full_window(synthetic) -> None:
    """Short history is not scored on a shorter window — it is not scored."""
    early = synthetic.trading_days[120].date()  # only ~120 bars exist
    signals = MomentumAgent().evaluate(synthetic.context(early))
    assert signals == []


def test_momentum_requires_lookback_plus_one_bars(synthetic) -> None:
    agent = MomentumAgent()
    assert agent.required_bars == agent.lookback + 1
    ctx = synthetic.context()
    assert len(agent.evaluate(ctx)) == len(synthetic.tickers)


def test_low_vol_ranks_calm_above_wild(synthetic) -> None:
    """The fixture's volatility rises with ticker index, so the ranking is known."""
    signals = {s.ticker: s.score for s in LowVolAgent().evaluate(synthetic.context())}
    assert signals["T00"] > signals["T11"]


def test_low_vol_discounts_confidence_on_short_history(synthetic) -> None:
    agent = LowVolAgent()
    short = synthetic.trading_days[40].date()  # ~40 bars: above min_window, below window
    signals = agent.evaluate(synthetic.context(short))
    assert signals, "a shorter window still estimates volatility"
    assert all(s.confidence < 1.0 for s in signals)
    full = agent.evaluate(synthetic.context())
    assert all(s.confidence == 1.0 for s in full)


def test_low_vol_omits_below_min_window(synthetic) -> None:
    too_short = synthetic.trading_days[10].date()
    assert LowVolAgent().evaluate(synthetic.context(too_short)) == []


def test_reversal_ranks_recent_losers_first(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    days = pd.bdate_range("2024-01-01", periods=10)
    up = np.linspace(100.0, 130.0, 10)
    down = np.linspace(100.0, 80.0, 10)
    rows = [
        {
            "ticker": t, "date": d.date(), "open": p, "high": p, "low": p,
            "close": p, "adj_close": p, "volume": 1e6,
        }
        for d, u, w in zip(days, up, down)
        for t, p in (("UP", u), ("DOWN", w))
    ]
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))
    asof = days[-1].date()
    ctx = Context(asof=asof, view=lake.view(asof), universe=("UP", "DOWN"))
    scores = {s.ticker: s.score for s in ReversalAgent().evaluate(ctx)}
    assert scores["DOWN"] == 1.0 and scores["UP"] == -1.0


def test_momentum_and_reversal_disagree_by_design(synthetic) -> None:
    """They are opposed factors at opposed horizons. The horizons say so."""
    ctx = synthetic.context()
    momentum = MomentumAgent().evaluate(ctx)
    reversal = ReversalAgent().evaluate(ctx)
    assert momentum[0].horizon_days == 126
    assert reversal[0].horizon_days == 5

    m = pd.Series({s.ticker: s.score for s in momentum})
    r = pd.Series({s.ticker: s.score for s in reversal})
    assert m.corr(r) < 0.9, "the two must not be the same signal under two names"


# -- fundamental agents -----------------------------------------------------------


def test_value_scores_every_ticker_with_filings(synthetic) -> None:
    signals = ValueAgent().evaluate(synthetic.context())
    assert len(signals) == len(synthetic.tickers)
    assert all(-1.0 <= s.score <= 1.0 for s in signals)
    assert "E/P" in signals[0].rationale and "B/P" in signals[0].rationale


def _two_ticker_lake(tmp_path, overrides: dict[str, dict[str, float]]) -> Context:
    """Two tickers identical except for the tags named in `overrides`.

    The 12-ticker fixture varies income, equity and leverage together, so a
    direction assertion on it is confounded — rising ROE and margin outvote
    rising leverage, and the "expected" ordering is whichever factor happens
    to dominate. Isolating one tag at a time is the only way to assert that a
    specific ratio points the way it should.
    """
    base = {
        "NetIncomeLoss": 1e8,
        "Revenues": 1e9,
        "GrossProfit": 3e8,
        "StockholdersEquity": 2e9,
        "Liabilities": 1e9,
        "EntityCommonStockSharesOutstanding": 1e8,
    }
    flows = ("NetIncomeLoss", "Revenues", "GrossProfit")
    days = pd.bdate_range("2023-01-02", periods=30)
    asof = days[-1].date()
    quarter_ends = [asof - timedelta(days=90 * k + 60) for k in range(6)]

    rows, facts = [], []
    for ticker, override in overrides.items():
        values = {**base, **override}
        for day in days:
            rows.append(
                {
                    "ticker": ticker, "date": day.date(), "open": 10.0, "high": 10.0,
                    "low": 10.0, "close": 10.0, "adj_close": 10.0, "volume": 1e6,
                }
            )
        for tag, value in values.items():
            taxonomy = "dei" if tag.startswith("Entity") else "us-gaap"
            unit = "shares" if taxonomy == "dei" else "USD"
            if tag in flows:
                for q, period_end in enumerate(quarter_ends):
                    facts.append(
                        f_row(ticker, tag, taxonomy, unit, value, period_end,
                              period_end + timedelta(days=35),
                              period_end - timedelta(days=90), f"{q}-{tag}")
                    )
            else:
                facts.append(
                    f_row(ticker, tag, taxonomy, unit, value, quarter_ends[0],
                          quarter_ends[0] + timedelta(days=35), None, f"inst-{tag}")
                )

    lake = Lake(tmp_path / "pair")
    lake.write(PRICES, pd.DataFrame(rows, columns=PRICE_COLUMNS))
    lake.write(FUNDAMENTALS, pd.DataFrame(facts, columns=FUNDAMENTAL_COLUMNS))
    return Context(asof=asof, view=lake.view(asof), universe=tuple(overrides))


def f_row(ticker, tag, taxonomy, unit, value, period_end, filed, period_start, suffix):
    return {
        "ticker": ticker, "cik": 1, "taxonomy": taxonomy, "tag": tag, "unit": unit,
        "period_start": period_start, "period_end": period_end, "value": value,
        "form": "10-Q", "filed": filed, "accn": f"{ticker}-{suffix}",
        "fy": period_end.year, "fp": "Q1",
    }


def test_value_prefers_higher_earnings_yield(tmp_path) -> None:
    """Same equity, shares and price; only earnings differ."""
    ctx = _two_ticker_lake(
        tmp_path, {"RICH": {"NetIncomeLoss": 5e8}, "POOR": {"NetIncomeLoss": 1e7}}
    )
    scores = {s.ticker: s.score for s in ValueAgent().evaluate(ctx)}
    assert scores["RICH"] > scores["POOR"]


def test_value_prefers_higher_book_to_price(tmp_path) -> None:
    """Same earnings, shares and price; only book equity differs."""
    ctx = _two_ticker_lake(
        tmp_path, {"ASSET": {"StockholdersEquity": 8e9}, "THIN": {"StockholdersEquity": 2e8}}
    )
    scores = {s.ticker: s.score for s in ValueAgent().evaluate(ctx)}
    assert scores["ASSET"] > scores["THIN"]


def test_quality_needs_two_of_three_ratios(synthetic) -> None:
    signals = QualityAgent().evaluate(synthetic.context())
    assert len(signals) == len(synthetic.tickers)
    assert all("ROE" in s.rationale for s in signals)


def test_quality_penalises_leverage(tmp_path) -> None:
    """Same earnings, equity and margin; only liabilities differ."""
    ctx = _two_ticker_lake(
        tmp_path, {"SAFE": {"Liabilities": 2e8}, "LEVERED": {"Liabilities": 9e9}}
    )
    scores = {s.ticker: s.score for s in QualityAgent().evaluate(ctx)}
    assert scores["SAFE"] > scores["LEVERED"]


def test_quality_rewards_roe(tmp_path) -> None:
    ctx = _two_ticker_lake(
        tmp_path, {"PROFITABLE": {"NetIncomeLoss": 6e8}, "MARGINAL": {"NetIncomeLoss": 1e6}}
    )
    scores = {s.ticker: s.score for s in QualityAgent().evaluate(ctx)}
    assert scores["PROFITABLE"] > scores["MARGINAL"]


def test_quality_rewards_gross_margin(tmp_path) -> None:
    ctx = _two_ticker_lake(
        tmp_path, {"FAT": {"GrossProfit": 8e8}, "SLIM": {"GrossProfit": 5e7}}
    )
    scores = {s.ticker: s.score for s in QualityAgent().evaluate(ctx)}
    assert scores["FAT"] > scores["SLIM"]


def test_fundamental_agents_decline_without_filings(prices_only) -> None:
    """No crash, no zero scores — an empty list, meaning "I cannot judge"."""
    ctx = prices_only.context()
    assert ValueAgent().evaluate(ctx) == []
    assert QualityAgent().evaluate(ctx) == []
    assert len(MomentumAgent().evaluate(ctx)) == len(prices_only.tickers)


def test_confidence_falls_as_filings_age(synthetic) -> None:
    """Same filings, later asof: the same ratio, trusted less."""
    fresh = ValueAgent().evaluate(synthetic.context())
    assert all(s.confidence == 1.0 for s in fresh)
    # The view clamps to asof, so a later asof keeps the same newest filing.
    later = synthetic.asof + timedelta(days=300)
    aged = ValueAgent().evaluate(synthetic.context(later))
    assert aged and all(s.confidence < 1.0 for s in aged)


# -- the assembly helpers ----------------------------------------------------------


def test_ttm_sums_four_quarters(synthetic) -> None:
    """The fixture files consecutive quarters and no annual, so this is the 4q path.

    The tag records which construction produced the number: `:ttm` is annual
    plus delta, `:4q` is four consecutive quarters, `:annual` is a stale
    fallback. That distinction is the audit trail for a figure whose
    provenance decides whether it can be trusted.
    """
    view = synthetic.lake.view(synthetic.asof)
    out = f.trailing_twelve_months(view, list(synthetic.tickers), ("NetIncomeLoss",))
    # T00's quarterly net income is 1e8; four of them is 4e8.
    assert out["T00"].value == pytest.approx(4e8)
    assert out["T00"].tag.endswith(":4q")


def test_ttm_declines_with_fewer_than_four_quarters(synthetic) -> None:
    """Three quarters are not annualised into four."""
    early = synthetic.trading_days[-1].date()
    view = synthetic.lake.view(early)
    # Narrow the visible window by asking at a date only two quarters in.
    two_quarters = synthetic.asof - timedelta(days=400)
    out = f.trailing_twelve_months(
        synthetic.lake.view(two_quarters), list(synthetic.tickers), ("NetIncomeLoss",)
    )
    assert out == {} or all(fact.tag.endswith(":annual") for fact in out.values())


def test_market_cap_uses_close_not_adj_close(synthetic) -> None:
    view = synthetic.lake.view(synthetic.asof)
    caps = f.market_cap(view, list(synthetic.tickers))
    shares = f.latest_instant(view, list(synthetic.tickers), f.SHARE_TAGS)
    closes = f.last_close(view, list(synthetic.tickers))
    assert caps["T00"].value == pytest.approx(shares["T00"].value * closes["T00"])


def test_ratio_drops_non_positive_denominators() -> None:
    num = {"A": f.Fact(10.0, date(2024, 1, 1), "n"), "B": f.Fact(10.0, date(2024, 1, 1), "n")}
    den = {"A": f.Fact(-5.0, date(2024, 1, 2), "d"), "B": f.Fact(5.0, date(2024, 1, 2), "d")}
    out = f.ratio(num, den)
    assert set(out) == {"B"}, "negative book equity is not a bargain"
    assert out["B"].filed == date(2024, 1, 1), (
        "keeps the STALER filing date: a ratio is only as current as its oldest "
        "input, and reporting the newer one hid a 2009 numerator behind a 2024 date"
    )


def test_ratio_can_allow_negative_denominators() -> None:
    num = {"A": f.Fact(10.0, date(2024, 1, 1), "n")}
    den = {"A": f.Fact(-5.0, date(2024, 1, 1), "d")}
    assert f.ratio(num, den, require_positive_denominator=False)["A"].value == -2.0


def test_newest_filing_takes_the_max_across_groups() -> None:
    a = {"X": f.Fact(1.0, date(2024, 1, 1), "a")}
    b = {"X": f.Fact(1.0, date(2024, 3, 1), "b")}
    assert f.newest_filing(a, b) == {"X": date(2024, 3, 1)}


# -- roster ----------------------------------------------------------------------


def test_every_phase_0_agent_satisfies_the_protocol() -> None:
    agents = phase_0_agents()
    assert len(agents) == 5
    assert all(isinstance(a, Agent) for a in agents)
    assert len({a.name for a in agents}) == 5


def test_no_phase_0_agent_costs_anything(synthetic) -> None:
    """The whole premise of Phase 0: it must be free to run."""
    ctx = synthetic.context()
    for agent in phase_0_agents():
        assert all(s.cost_usd == 0.0 for s in agent.evaluate(ctx)), agent.name


def test_agents_cannot_see_past_asof(synthetic) -> None:
    """Belt and braces: the Context is pinned, so an agent's own reads are too."""
    mid = synthetic.trading_days[300].date()
    ctx = synthetic.context(mid)
    for agent in phase_0_agents():
        for signal in agent.evaluate(ctx):
            assert signal.asof == mid


# -- TTM on EDGAR's real shape: the annual hole and the abandoned tag ----------
#
# Both bugs below were found on first contact with real filings and were silent:
# they produced plausible-looking dollar totals, not errors.


def _ttm_lake(tmp_path, rows: list[dict], name: str = "ttm") -> Context:
    lake = Lake(tmp_path / name)
    days = pd.bdate_range("2024-05-01", periods=5)
    lake.write(
        PRICES,
        pd.DataFrame(
            [
                {"ticker": "X", "date": d.date(), "open": 10.0, "high": 10.0,
                 "low": 10.0, "close": 10.0, "adj_close": 10.0, "volume": 1e6}
                for d in days
            ],
            columns=PRICE_COLUMNS,
        ),
    )
    lake.write(FUNDAMENTALS, pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS))
    asof = date(2024, 6, 28)
    return Context(asof=asof, view=lake.view(asof), universe=("X",))


def _flow(tag, start, end, value, filed, taxonomy="us-gaap"):
    return f_row("X", tag, taxonomy, "USD", value, end, filed, start, f"{tag}-{end}")


def test_ttm_uses_annual_plus_delta_across_the_q4_hole(tmp_path) -> None:
    """A 10-K filer reports no Q4 quarterly, so four consecutive quarters do
    not exist. TTM = annual + quarters since - same quarters a year before."""
    rows = [
        # FY2023 annual: 1000, year ending 2023-12-31
        _flow("Revenues", date(2023, 1, 1), date(2023, 12, 31), 1000.0, date(2024, 2, 15)),
        # Q1 2024 = 300, and Q1 2023 = 200 -> TTM should be 1000 + 300 - 200 = 1100
        _flow("Revenues", date(2024, 1, 1), date(2024, 3, 31), 300.0, date(2024, 5, 1)),
        _flow("Revenues", date(2023, 1, 1), date(2023, 3, 31), 200.0, date(2023, 5, 1)),
        _flow("Revenues", date(2023, 4, 1), date(2023, 6, 30), 210.0, date(2023, 8, 1)),
        _flow("Revenues", date(2023, 7, 1), date(2023, 9, 30), 220.0, date(2023, 11, 1)),
        # note: no 2023 Q4 quarterly fact, exactly as a 10-K filer behaves
    ]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], ("Revenues",))
    assert out["X"].value == pytest.approx(1100.0)
    assert out["X"].tag.endswith(":ttm")
    assert out["X"].filed == date(2024, 5, 1), "the newest filing the sum rests on"


def test_ttm_rejects_a_window_that_does_not_tile_a_year(tmp_path) -> None:
    """Four quarters spanning 15 months with a hole are not a TTM.

    Measured consequence of accepting one: NVDA scored a gross margin of
    438.9% and AAPL's window spanned 455 days skipping a whole quarter.
    """
    rows = [
        _flow("Revenues", date(2023, 1, 1), date(2023, 3, 31), 100.0, date(2023, 5, 1)),
        _flow("Revenues", date(2023, 4, 1), date(2023, 6, 30), 100.0, date(2023, 8, 1)),
        # a hole where Jul-Dec 2023 should be
        _flow("Revenues", date(2024, 1, 1), date(2024, 3, 31), 100.0, date(2024, 5, 1)),
        _flow("Revenues", date(2024, 4, 1), date(2024, 6, 30), 100.0, date(2024, 6, 20)),
    ]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], ("Revenues",))
    assert out == {}, "a gapped window must be declined, not summed to 400"


def test_ttm_accepts_four_genuinely_consecutive_quarters(tmp_path) -> None:
    rows = [
        _flow("Revenues", date(2023, 7, 1), date(2023, 9, 30), 100.0, date(2023, 11, 1)),
        _flow("Revenues", date(2023, 10, 1), date(2023, 12, 31), 110.0, date(2024, 2, 1)),
        _flow("Revenues", date(2024, 1, 1), date(2024, 3, 31), 120.0, date(2024, 5, 1)),
        _flow("Revenues", date(2024, 4, 1), date(2024, 6, 30), 130.0, date(2024, 6, 20)),
    ]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], ("Revenues",))
    assert out["X"].value == pytest.approx(460.0)
    assert out["X"].tag.endswith(":4q")


def test_ttm_prefers_the_freshest_tag_not_the_first(tmp_path) -> None:
    """Filers migrate between revenue tags and leave the old one frozen.

    NVDA's RevenueFromContractWithCustomer... stops in 2020 while its Revenues
    runs to 2024. Taking the first tag that returned anything gave a 2020
    figure, 86% below the published number.
    """
    stale, live = f.REVENUE_TAGS[0], f.REVENUE_TAGS[1]
    rows = [
        # the abandoned tag: a complete, valid, but four-year-old year
        _flow(stale, date(2019, 1, 1), date(2019, 3, 31), 10.0, date(2019, 5, 1)),
        _flow(stale, date(2019, 4, 1), date(2019, 6, 30), 10.0, date(2019, 8, 1)),
        _flow(stale, date(2019, 7, 1), date(2019, 9, 30), 10.0, date(2019, 11, 1)),
        _flow(stale, date(2019, 10, 1), date(2019, 12, 31), 10.0, date(2020, 2, 1)),
        # the live tag: current and much larger
        _flow(live, date(2023, 7, 1), date(2023, 9, 30), 500.0, date(2023, 11, 1)),
        _flow(live, date(2023, 10, 1), date(2023, 12, 31), 500.0, date(2024, 2, 1)),
        _flow(live, date(2024, 1, 1), date(2024, 3, 31), 500.0, date(2024, 5, 1)),
        _flow(live, date(2024, 4, 1), date(2024, 6, 30), 500.0, date(2024, 6, 20)),
    ]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], f.REVENUE_TAGS)
    assert out["X"].value == pytest.approx(2000.0), "the stale tag must not win"
    assert out["X"].tag.startswith(live)


def test_ttm_falls_back_to_annual_when_quarters_are_unusable(tmp_path) -> None:
    """JPM's quarterly Revenues stop in 2014; the annual is the honest answer."""
    rows = [
        _flow("Revenues", date(2023, 1, 1), date(2023, 12, 31), 900.0, date(2024, 2, 16)),
        _flow("Revenues", date(2014, 7, 1), date(2014, 9, 30), 50.0, date(2014, 11, 3)),
    ]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], ("Revenues",))
    assert out["X"].value == pytest.approx(900.0)
    assert out["X"].tag.endswith(":annual")
    assert out["X"].age_days(date(2024, 6, 28)) == 133, "staleness is visible to the agent"


def test_ttm_declines_when_nothing_is_usable(tmp_path) -> None:
    """XOM exposes no usable revenue tag; an empty result beats a wrong one."""
    rows = [_flow("Revenues", date(2024, 1, 1), date(2024, 3, 31), 100.0, date(2024, 5, 1))]
    out = f.trailing_twelve_months(_ttm_lake(tmp_path, rows).view, ["X"], ("Revenues",))
    assert out == {}


def test_latest_instant_prefers_the_freshest_tag(tmp_path) -> None:
    rows = [
        f_row("X", f.SHARE_TAGS[1], "us-gaap", "shares", 1e6, date(2020, 3, 31),
              date(2020, 5, 1), None, "old"),
        f_row("X", f.SHARE_TAGS[0], "dei", "shares", 5e6, date(2024, 3, 31),
              date(2024, 5, 1), None, "new"),
    ]
    out = f.latest_instant(_ttm_lake(tmp_path, rows).view, ["X"], f.SHARE_TAGS)
    assert out["X"].value == pytest.approx(5e6)
    assert out["X"].filed == date(2024, 5, 1)


def test_ratio_keeps_the_stalest_filing_not_the_freshest(tmp_path) -> None:
    """A ratio is only as current as its oldest input.

    Amazon stopped filing GrossProfit in 2010, so a 2024 revenue over a 2009
    gross profit produced a 0.9% "gross margin" that no staleness score
    flagged, because the ratio reported the newer of its two dates.
    """
    fresh = {"X": f.Fact(100.0, date(2024, 5, 1), "new")}
    stale = {"X": f.Fact(50.0, date(2010, 1, 29), "old")}
    out = f.ratio(fresh, stale)
    assert out["X"].filed == date(2010, 1, 29), "the ratio must inherit the old date"


def test_ratio_drops_a_pairing_that_is_too_old_to_mean_anything(tmp_path) -> None:
    fresh = {"X": f.Fact(100.0, date(2024, 5, 1), "new")}
    ancient = {"X": f.Fact(50.0, date(2010, 1, 29), "old")}
    assert f.ratio(fresh, ancient, asof=date(2024, 6, 28)) == {}
    recent = {"X": f.Fact(50.0, date(2024, 2, 1), "ok")}
    assert f.ratio(fresh, recent, asof=date(2024, 6, 28))["X"].value == pytest.approx(2.0)


def test_oldest_filing_is_what_confidence_must_use() -> None:
    a = {"X": f.Fact(1.0, date(2024, 5, 1), "a")}
    b = {"X": f.Fact(1.0, date(2019, 3, 1), "b")}
    assert f.oldest_filing(a, b) == {"X": date(2019, 3, 1)}
    assert f.newest_filing(a, b) == {"X": date(2024, 5, 1)}


def test_market_cap_prefers_diluted_when_the_cover_page_loses_a_class(tmp_path) -> None:
    """Visa's cover page reports Class A only: 469m against ~2,000m actual."""
    rows = [
        # cover page: one class only
        f_row("X", f.SHARE_TAGS[1], "dei", "shares", 469e6, date(2024, 3, 31),
              date(2024, 4, 24), None, "cover"),
        # consolidated diluted count, a period fact
        f_row("X", "WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap",
              "shares", 2000e6, date(2024, 3, 31), date(2024, 4, 24),
              date(2024, 1, 1), "diluted"),
    ]
    ctx = _ttm_lake(tmp_path, rows, name="cap")
    cap = f.market_cap(ctx.view, ["X"])
    assert cap["X"].value == pytest.approx(2000e6 * 10.0), "price is 10.0 in the fixture"
    assert "Diluted" in cap["X"].tag


def test_market_cap_keeps_the_cover_page_when_it_is_consistent(tmp_path) -> None:
    rows = [
        f_row("X", f.SHARE_TAGS[1], "dei", "shares", 1000e6, date(2024, 3, 31),
              date(2024, 4, 24), None, "cover"),
        f_row("X", "WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap",
              "shares", 1010e6, date(2024, 3, 31), date(2024, 4, 24),
              date(2024, 1, 1), "diluted"),
    ]
    ctx = _ttm_lake(tmp_path, rows, name="cap2")
    cap = f.market_cap(ctx.view, ["X"])
    assert cap["X"].value == pytest.approx(1000e6 * 10.0)
