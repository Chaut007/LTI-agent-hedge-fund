"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

The best Sharpe out of N tries is not an estimate of anything: it is the
maximum of N noisy draws, and the maximum of N draws from a zero-mean
distribution is positive and grows with log N. Pick the best of fifty
parameter settings and you have manufactured a Sharpe of roughly 2 standard
errors out of nothing.

DSR asks the only honest question: given how many things were tried, and how
non-normal the returns are, what is the probability that the true Sharpe is
above zero? It needs the trial count, which is why `TrialLog` exists — every
backtest the CLI runs is recorded, because the researcher is the one party who
cannot be trusted to remember how many times they looked.

All Sharpe ratios in this module are PER PERIOD (daily), not annualised, as in
the paper. Multiply by sqrt(252) for the familiar number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm

from hedge_fund.backtest.metrics import ZERO_VOL, kurtosis, skewness

EULER_MASCHERONI = 0.5772156649015329


def period_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > ZERO_VOL else 0.0


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max SR] over `n_trials` draws from a zero-mean SR distribution.

    Eq. (6) of the paper. With one trial there was no selection, so the
    expected maximum is zero and DSR collapses to the plain PSR.
    """
    if n_trials <= 1 or sharpe_variance <= 0.0:
        return 0.0
    n = float(n_trials)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(
        math.sqrt(sharpe_variance)
        * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    )


def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float
) -> float:
    """P(true SR > sr_benchmark), adjusted for sample size and non-normality.

    Eq. (3). `kurt` is RAW kurtosis (normal = 3). Fat tails and negative skew
    both widen the standard error of the Sharpe estimate and lower the PSR.
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0.0:
        # Degenerate moments (possible with tiny samples); the estimate's
        # standard error is undefined, so claim nothing.
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(denom)
    return float(norm.cdf(z))


@dataclass(frozen=True)
class DeflatedResult:
    sharpe_period: float
    sharpe_annual: float
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    trial_sharpe_variance: float
    expected_max_sharpe: float
    """The Sharpe you would expect from the best of n_trials with no skill."""
    psr: float
    """P(true SR > 0) ignoring multiple testing."""
    dsr: float
    """P(true SR > expected_max_sharpe): the deflated probability."""

    @property
    def passes(self) -> bool:
        return self.dsr >= 0.95

    def describe(self) -> str:
        lines = [
            f"deflated Sharpe ({self.n_trials} trial{'s' if self.n_trials != 1 else ''} "
            f"on record, {self.n_obs} obs)",
            f"  Sharpe        {self.sharpe_annual:.2f} annual "
            f"({self.sharpe_period:.4f}/day), skew {self.skew:+.2f}, kurt {self.kurtosis:.2f}",
            f"  PSR           {self.psr:.1%}  P(true Sharpe > 0), single trial",
        ]
        if self.n_trials > 1:
            lines.append(
                f"  E[max SR]     {self.expected_max_sharpe * math.sqrt(252):.2f} annual — "
                f"what luck alone yields from the best of {self.n_trials}"
            )
        lines.append(
            f"  DSR           {self.dsr:.1%}  P(true Sharpe > E[max SR])"
            + ("  PASS" if self.passes else "  below 95%: not distinguishable from selection luck")
        )
        return "\n".join(lines)


def deflated_sharpe(
    returns: np.ndarray, trial_sharpes: Sequence[float] | None = None
) -> DeflatedResult:
    """DSR of `returns` given the per-period Sharpes of every trial attempted.

    `trial_sharpes` should include this run. With None or a single entry no
    deflation is applied and DSR == PSR.
    """
    r = np.asarray(returns, dtype="float64")
    sr = period_sharpe(r)
    trials = [float(s) for s in (trial_sharpes or [sr])]
    n_trials = len(trials)
    variance = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, variance)
    sk, ku = skewness(r), kurtosis(r)
    return DeflatedResult(
        sharpe_period=sr,
        sharpe_annual=sr * math.sqrt(252.0),
        n_obs=int(r.size),
        skew=sk,
        kurtosis=ku,
        n_trials=n_trials,
        trial_sharpe_variance=variance,
        expected_max_sharpe=sr_star,
        psr=probabilistic_sharpe(sr, 0.0, r.size, sk, ku),
        dsr=probabilistic_sharpe(sr, sr_star, r.size, sk, ku),
    )
