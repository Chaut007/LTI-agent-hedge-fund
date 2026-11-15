"""Permutation test: is the stock selection better than random?

The null hypothesis for a cross-sectional strategy is "which names you picked
did not matter". So: keep every decision exactly as recorded — same dates,
same number of positions, same weights — but hand each day's weights to a
random draw of the names that were tradeable that day. Re-simulate through the
same engine with the same costs. Do that a few hundred times and you have the
distribution of Sharpe a strategy with this *shape* and no *skill* produces.

The observed Sharpe is then placed in that distribution. If 30% of random
portfolios of the same size and turnover did as well, the backtest has shown
nothing about selection — whatever return it earned came from being long the
market with K names, not from the K it chose.

This is the cheapest honest test in the repo. It costs nothing but CPU and it
catches the most common way a backtest flatters: by being long equities over a
period when equities went up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from hedge_fund.backtest.engine import Backtester, BacktestConfig, BacktestResult
from hedge_fund.backtest.strategy import ReplayStrategy, Weights


def shuffle_decisions(
    decisions: Mapping[date, Weights],
    live_by_date: Mapping[date, list[str]],
    rng: np.random.Generator,
) -> dict[date, Weights]:
    """Same weights per day, assigned to random tradeable tickers."""
    shuffled: dict[date, Weights] = {}
    for asof, weights in decisions.items():
        values = list(weights.values())
        if not values:
            shuffled[asof] = {}
            continue
        candidates = live_by_date.get(asof, [])
        if len(candidates) < len(values):
            raise ValueError(
                f"{asof}: {len(values)} positions but only {len(candidates)} "
                "tradeable names — the recorded decision could not have happened"
            )
        picks = rng.choice(len(candidates), size=len(values), replace=False)
        shuffled[asof] = {candidates[i]: w for i, w in zip(picks, values)}
    return shuffled


def live_tickers_by_date(panel: pd.DataFrame, dates: list[date]) -> dict[date, list[str]]:
    """Tickers with a real (non-NaN) price on each date."""
    out: dict[date, list[str]] = {}
    for asof in dates:
        stamp = pd.Timestamp(asof)
        if stamp in panel.index:
            row = panel.loc[stamp]
            out[asof] = sorted(row.index[row.notna()].tolist())
        else:
            out[asof] = []
    return out


@dataclass(frozen=True)
class PermutationResult:
    observed_sharpe: float
    null_sharpes: np.ndarray
    n_permutations: int
    seed: int | None

    @property
    def p_value(self) -> float:
        """P(random selection does at least this well). Add-one estimator."""
        return float((1 + np.count_nonzero(self.null_sharpes >= self.observed_sharpe))
                     / (self.n_permutations + 1))

    @property
    def null_mean(self) -> float:
        return float(self.null_sharpes.mean())

    @property
    def null_95th(self) -> float:
        return float(np.percentile(self.null_sharpes, 95))

    @property
    def passes(self) -> bool:
        return self.p_value <= 0.05

    @property
    def degenerate(self) -> bool:
        """Every shuffle produced the same portfolio: nothing was selected.

        Happens when a strategy holds every tradeable name — equal weight over
        the whole universe, say. Reassigning N identical weights to the same N
        names is the identity, so the null has zero spread and p is 1 by
        construction. That is the right answer (there is no selection to test)
        but it should be said, not left to look like a strange coincidence.
        """
        return bool(self.null_sharpes.std() < 1e-12)

    def describe(self) -> str:
        lines = [
            f"permutation test ({self.n_permutations} random portfolios of the same shape)",
            f"  observed Sharpe {self.observed_sharpe:.2f}",
            f"  null Sharpe     mean {self.null_mean:.2f}, 95th pct {self.null_95th:.2f}",
            f"  p-value         {self.p_value:.3f}"
            + (
                "  PASS: selection beats random at 5%"
                if self.passes
                else "  random portfolios of this shape do as well — no selection skill shown"
            ),
        ]
        if self.degenerate:
            lines.append(
                "  (null has zero spread: the strategy holds every tradeable name, so "
                "there is no selection to permute — the test is not applicable, not failed)"
            )
        return "\n".join(lines)


def permutation_test(
    backtester: Backtester,
    result: BacktestResult,
    n_permutations: int = 200,
    seed: int | None = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> PermutationResult:
    if not result.decisions:
        raise ValueError("result has no recorded decisions to permute")

    rng = np.random.default_rng(seed)
    live = live_tickers_by_date(backtester.panel, list(result.decisions))
    # Same engine, same costs, same lag; skip the benchmark reload each run.
    config = replace(result.config, benchmark=None)

    nulls = np.empty(n_permutations)
    for k in range(n_permutations):
        if on_progress:
            on_progress(k + 1, n_permutations)
        shuffled = shuffle_decisions(result.decisions, live, rng)
        run = backtester.run(ReplayStrategy(decisions=shuffled, name="null"), config)
        nulls[k] = run.metrics.sharpe

    return PermutationResult(
        observed_sharpe=result.metrics.sharpe,
        null_sharpes=nulls,
        n_permutations=n_permutations,
        seed=seed,
    )
