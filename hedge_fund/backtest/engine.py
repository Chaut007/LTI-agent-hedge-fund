"""Walk-forward simulator.

The engine is the market, the strategy is a participant, and the boundary
between them is the Context. On every decision date T the engine builds
`Context(asof=T, view=lake.view(T))` and asks the strategy for target weights.
The strategy can only see through that view, so it cannot see past T no matter
how it is written. The engine itself reads the full price panel — it has to,
to mark positions to market — but it never shows that panel to anyone.

Timeline, with the default `execution_lag=2`:

    close T      strategy decides w_T from view(T)
    close T+1    orders fill (we have no intraday prices; the close is the
                 conservative stand-in for a 10:00 order). Turnover and costs
                 are charged here.
    close T+2    first return earned by w_T

`execution_lag=1` fills at the same close the decision used. That is the usual
optimistic textbook assumption and is allowed, but it is not the default,
because it is not what the live pipeline does.

Between fills, weights drift with prices exactly as a real book would. A
strategy that returns the same targets every day therefore pays turnover only
for the drift it undoes, not for the whole book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from hedge_fund.backtest.costs import CostModel
from hedge_fund.backtest.metrics import Metrics, compute_metrics
from hedge_fund.backtest.panel import load_price_panel
from hedge_fund.backtest.strategy import Strategy, Weights
from hedge_fund.core.agent import Context
from hedge_fund.data.lake import Lake
from hedge_fund.portfolio.state import PortfolioAware, PortfolioState

Rebalance = Literal["daily", "weekly", "monthly"]

_GROSS_TOLERANCE = 1e-9


class WeightError(ValueError):
    """A strategy returned weights the engine refuses to simulate.

    Raised, not corrected: a strategy that breaches its own exposure limit has a
    bug, and silently rescaling it would hide that bug behind a plausible
    equity curve.
    """


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    rebalance: Rebalance = "daily"
    execution_lag: int = 2
    costs: CostModel = field(default_factory=CostModel)
    max_gross: float = 1.0
    allow_short: bool = False
    initial_nav: float = 1.0
    benchmark: str | None = None
    """A ticker in the lake to report buy-and-hold returns for, e.g. SPY."""
    kill_switch_dd: float | None = None
    """Drawdown from peak that flattens the book. None disables the stop.

    Enforced here rather than in the strategy because the engine is the only
    thing that knows NAV — and in live trading that is the broker, not the
    model. A strategy can be told the drawdown through `observe`, but it cannot
    decline to be stopped.
    """
    kill_switch_mode: Literal["halt", "cooldown"] = "halt"
    """What happens after the stop fires.

    "halt" stays flat for the rest of the run, which is what "requires a manual
    restart" actually costs you — including the recovery you sat out.

    "cooldown" re-enters after `kill_switch_cooldown_days` trading days. Note
    what it cannot be: re-entering "once the drawdown recovers" is unreachable
    by construction, because a flat book's NAV never moves and so can never
    reach a new high. A cooldown is the only form of automatic restart that
    terminates.
    """
    kill_switch_cooldown_days: int = 21
    """Trading days flat before a "cooldown" restart. Ignored when mode is "halt"."""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")
        if self.execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1: a fill cannot happen before the "
                "close whose prices the decision used"
            )
        if self.max_gross <= 0:
            raise ValueError("max_gross must be positive")
        if self.kill_switch_dd is not None and not 0.0 < self.kill_switch_dd < 1.0:
            raise ValueError(
                f"kill_switch_dd must be in (0, 1) or None, got {self.kill_switch_dd}"
            )


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    config: BacktestConfig
    nav: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    """Weights held at each close, after any fill that day."""
    decisions: dict[date, Weights]
    """What the strategy asked for, keyed by decision date."""
    stale_days: int
    """Held-position days priced off a forward-filled (stale) bar. See notes."""
    benchmark_returns: pd.Series | None
    metrics: Metrics

    def describe(self) -> str:
        lines = [
            f"{self.strategy}: {self.config.start} .. {self.config.end}, "
            f"{self.config.rebalance}, lag {self.config.execution_lag}, "
            f"costs {self.config.costs.per_side_bps:.0f} bps/side",
            self.metrics.describe(),
        ]
        if self.benchmark_returns is not None:
            bench = compute_metrics(self.benchmark_returns)
            excess = self.metrics.cagr - bench.cagr
            lines.append(
                f"  vs {self.config.benchmark}: CAGR {bench.cagr:+.2%}, "
                f"Sharpe {bench.sharpe:.2f}, max DD {bench.max_drawdown:.2%} "
                f"-> excess CAGR {excess:+.2%}"
            )
        if self.stale_days:
            lines.append(
                f"  NOTE {self.stale_days} position-days used a stale (forward-"
                "filled) price — a name stopped printing while held"
            )
        return "\n".join(lines)


def decision_dates(
    trading_days: pd.DatetimeIndex, rebalance: Rebalance
) -> pd.DatetimeIndex:
    """Which trading days the strategy is consulted on."""
    if rebalance == "daily":
        return trading_days
    if rebalance == "weekly":
        key = trading_days.to_period("W")
    elif rebalance == "monthly":
        key = trading_days.to_period("M")
    else:
        raise ValueError(f"unknown rebalance {rebalance!r}")
    # last trading day of each period
    last = pd.Series(trading_days, index=trading_days).groupby(key).max()
    return pd.DatetimeIndex(last.values)


def _validate_weights(
    weights: Weights, tickers: Sequence[str], config: BacktestConfig, asof: date
) -> np.ndarray:
    index = {t: i for i, t in enumerate(tickers)}
    vector = np.zeros(len(tickers), dtype="float64")
    for ticker, weight in weights.items():
        if ticker not in index:
            raise WeightError(
                f"{asof}: strategy weighted {ticker!r}, which has no prices in the "
                "backtest panel — it cannot be marked to market"
            )
        if not np.isfinite(weight):
            raise WeightError(f"{asof}: weight for {ticker} is {weight!r}")
        if weight < 0 and not config.allow_short:
            raise WeightError(
                f"{asof}: negative weight {weight:.4f} for {ticker} with "
                "allow_short=False"
            )
        vector[index[ticker]] = float(weight)
    gross = float(np.abs(vector).sum())
    if gross > config.max_gross + _GROSS_TOLERANCE:
        raise WeightError(
            f"{asof}: gross exposure {gross:.4f} exceeds max_gross "
            f"{config.max_gross}"
        )
    return vector


class Backtester:
    """One instance per (lake, universe). Loads the price panel once.

    The panel is cached on first `run` so a permutation test can re-simulate
    hundreds of shuffled decision schedules without re-reading parquet each
    time. The cache is per instance: an ingest that changes the lake needs a
    new Backtester, which is the right granularity for a research session.
    """

    def __init__(self, lake: Lake, universe: Sequence[str]) -> None:
        self.lake = lake
        self.universe = tuple(dict.fromkeys(t.upper() for t in universe))
        if not self.universe:
            raise ValueError("universe is empty")
        self._panel: pd.DataFrame | None = None

    @property
    def panel(self) -> pd.DataFrame:
        """Full-history adj_close panel for the universe. Engine-only."""
        if self._panel is None:
            # Full history, not just [start, end]: the first simulated day
            # needs the bar before it for its return.
            panel = load_price_panel(self.lake, self.universe)
            if panel.empty:
                raise ValueError("no prices in the lake for this universe")
            self._panel = panel
        return self._panel

    def run(self, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
        panel = self.panel

        # Forward-fill so a gap day earns 0 and the full move lands when the
        # name prints again. Leading NaNs (pre-listing) stay NaN → 0 return; a
        # strategy that reads the view will not have weighted them anyway.
        stale_mask = panel.isna() & panel.ffill().notna()
        filled = panel.ffill()
        asset_returns = filled.pct_change(fill_method=None).fillna(0.0)

        in_window = (panel.index >= pd.Timestamp(config.start)) & (
            panel.index <= pd.Timestamp(config.end)
        )
        days = panel.index[in_window]
        if len(days) == 0:
            raise ValueError(
                f"no trading days in the lake between {config.start} and {config.end}"
            )

        tickers = list(panel.columns)
        n = len(tickers)
        day_pos = {d: i for i, d in enumerate(days)}

        # Decision date -> fill date, dropping decisions that never get filled.
        decide_on = decision_dates(days, config.rebalance)
        decide_set = set(decide_on)
        fills: dict[pd.Timestamp, pd.Timestamp] = {}
        for d in decide_on:
            fill_i = day_pos[d] + config.execution_lag - 1
            if fill_i < len(days):
                fills[days[fill_i]] = d

        held = np.zeros(n)
        nav = config.initial_nav
        peak_nav = config.initial_nav
        halted = False
        halted_on: date | None = None
        halted_at: int | None = None
        decisions: dict[date, Weights] = {}
        pending: dict[pd.Timestamp, np.ndarray] = {}
        stale_days = 0
        observer = strategy if isinstance(strategy, PortfolioAware) else None

        out_nav = np.empty(len(days))
        out_gross = np.empty(len(days))
        out_net = np.empty(len(days))
        out_cost = np.zeros(len(days))
        out_turn = np.zeros(len(days))
        out_weights = np.empty((len(days), n))

        returns_arr = asset_returns.loc[days].to_numpy()
        stale_arr = stale_mask.loc[days].to_numpy()

        for i, d in enumerate(days):
            r = returns_arr[i]
            gross = float(held @ r)
            stale_days += int(np.count_nonzero(stale_arr[i] & (held != 0)))

            denom = 1.0 + gross
            drifted = held * (1.0 + r) / denom if denom != 0.0 else np.zeros(n)

            # Decide at today's close, with a view pinned to today. This comes
            # BEFORE the fill step because with execution_lag=1 the decision
            # and its fill share a close; the fill must find it in `pending`.
            if d in decide_set:
                asof = d.date()
                if observer is not None:
                    # Mark to market at today's close, before the cost of the
                    # trade about to be decided. `drifted` is already today's
                    # book, so NAV must be today's too — reporting yesterday's
                    # alongside today's weights would make every turnover
                    # budget and drawdown the strategy reads a day stale.
                    mark_nav = nav * (1.0 + gross)
                    observer.observe(
                        PortfolioState(
                            asof=asof,
                            nav=mark_nav,
                            peak_nav=max(peak_nav, mark_nav),
                            weights={
                                tickers[k]: float(drifted[k])
                                for k in np.nonzero(drifted)[0]
                            },
                            halted=halted,
                        )
                    )
                ctx = Context(asof=asof, view=self.lake.view(asof), universe=self.universe)
                weights = {} if halted else strategy.target_weights(ctx)
                decisions[asof] = dict(weights)
                pending[d] = _validate_weights(weights, tickers, config, asof)

            cost = 0.0
            turnover = 0.0
            if d in fills:
                target = pending.pop(fills[d])
                turnover = float(np.abs(target - drifted).sum()) / 2.0
                cost = float(config.costs.charge(turnover))
                held = target
            else:
                held = drifted

            net = gross - cost

            # The stop is judged on the book as it stands after today's move,
            # so the drawdown it reacts to has actually happened — but it is
            # applied before NAV compounds, so the liquidation cost lands in
            # today's return exactly once.
            if config.kill_switch_dd is not None:
                provisional = nav * (1.0 + net)
                reference = max(peak_nav, provisional)
                drawdown = (
                    0.0 if reference <= 0.0 else max(0.0, 1.0 - provisional / reference)
                )
                if not halted and drawdown > config.kill_switch_dd:
                    halted = True
                    halted_on = d.date()
                    halted_at = i
                    # Flatten now and drop anything queued. A stop that waited
                    # for the next scheduled fill is not a stop.
                    liquidation = float(np.abs(held).sum()) / 2.0
                    liquidation_cost = float(config.costs.charge(liquidation))
                    held = np.zeros(n)
                    pending.clear()
                    turnover += liquidation
                    cost += liquidation_cost
                    net -= liquidation_cost
                elif (
                    halted
                    and config.kill_switch_mode == "cooldown"
                    and halted_at is not None
                    and i - halted_at >= config.kill_switch_cooldown_days
                ):
                    halted = False
                    halted_at = None
                    # Measure the next drawdown from here, not from the old
                    # peak. Keeping the pre-stop peak would re-fire the switch
                    # on the first small loss after restarting.
                    peak_nav = nav * (1.0 + net)

            nav *= 1.0 + net
            peak_nav = max(peak_nav, nav)

            out_nav[i] = nav
            out_gross[i] = gross
            out_net[i] = net
            out_cost[i] = cost
            out_turn[i] = turnover
            out_weights[i] = held

        index = pd.DatetimeIndex(days, name="date")
        net_series = pd.Series(out_net, index=index, name="net")
        turnover_series = pd.Series(out_turn, index=index, name="turnover")
        cost_series = pd.Series(out_cost, index=index, name="cost")

        benchmark = None
        if config.benchmark:
            benchmark = self._benchmark_returns(config.benchmark, index)

        return BacktestResult(
            strategy=strategy.name,
            config=config,
            nav=pd.Series(out_nav, index=index, name="nav"),
            gross_returns=pd.Series(out_gross, index=index, name="gross"),
            net_returns=net_series,
            costs=cost_series,
            turnover=turnover_series,
            weights=pd.DataFrame(out_weights, index=index, columns=tickers),
            decisions=decisions,
            stale_days=stale_days,
            benchmark_returns=benchmark,
            metrics=compute_metrics(
                net_series,
                turnover_series,
                cost_series,
                pd.DataFrame(out_weights, index=index, columns=tickers),
            ),
        )

    def _benchmark_returns(
        self, ticker: str, index: pd.DatetimeIndex
    ) -> pd.Series | None:
        panel = load_price_panel(self.lake, [ticker])
        if panel.empty or ticker.upper() not in panel.columns:
            return None
        series = panel[ticker.upper()].ffill().pct_change(fill_method=None)
        return series.reindex(index).fillna(0.0).rename(ticker.upper())


def replay(
    lake: Lake,
    universe: Sequence[str],
    decisions: Mapping[date, Weights],
    config: BacktestConfig,
    name: str = "replay",
) -> BacktestResult:
    """Re-simulate a recorded decision schedule. Used by the permutation test."""
    from hedge_fund.backtest.strategy import ReplayStrategy

    return Backtester(lake, universe).run(
        ReplayStrategy(decisions=decisions, name=name), config
    )
