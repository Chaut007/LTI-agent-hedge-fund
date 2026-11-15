"""Performance statistics from a daily return series.

Pure functions over numpy arrays so the validation module can call them tens
of thousands of times on resampled series without touching pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Below this, a standard deviation is treated as zero. A constant series does
# not have std exactly 0.0 in floating point — summation rounding leaves ~1e-19
# — and dividing a mean by that yields a Sharpe of 1e16. Nothing real has daily
# volatility of 1e-12 (one ten-billionth of a percent).
ZERO_VOL = 1e-12


def sharpe(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe with a zero risk-free rate. 0.0 for a flat series."""
    r = np.asarray(returns, dtype="float64")
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= ZERO_VOL or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough fall of the compounded curve, as a positive fraction."""
    r = np.asarray(returns, dtype="float64")
    if r.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float(np.max(1.0 - curve / peak))


def cagr(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    if total <= 0.0:
        return -1.0
    return total ** (periods / r.size) - 1.0


def skewness(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    if r.size < 3:
        return 0.0
    sd = r.std(ddof=0)
    if sd <= ZERO_VOL:
        return 0.0
    return float(np.mean(((r - r.mean()) / sd) ** 3))


def kurtosis(returns: np.ndarray) -> float:
    """Raw (not excess) kurtosis; a normal distribution scores 3.0."""
    r = np.asarray(returns, dtype="float64")
    if r.size < 4:
        return 3.0
    sd = r.std(ddof=0)
    if sd <= ZERO_VOL:
        return 3.0
    return float(np.mean(((r - r.mean()) / sd) ** 4))


@dataclass(frozen=True)
class Metrics:
    days: int
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_daily_turnover: float
    total_cost_drag: float
    """Cumulative fraction of NAV paid in costs over the whole run."""
    skew: float
    kurtosis: float
    mean_gross_exposure: float = 1.0
    """Average gross exposure. Reported because Sharpe hides it.

    Sharpe is scale-invariant to cash: a book half in cash earns half the
    return at half the volatility and scores the same ratio. So a strategy
    that cannot deploy its capital looks identical to one that can, by Sharpe
    alone. A measured case from step 5: a turnover budget that left the book
    37.5% invested reported Sharpe 1.06 — matching a fully-invested equal
    weight — while CAGR fell from 19.7% to 6.0%. Without this line that run
    reads as a success.
    """

    @property
    def underinvested(self) -> bool:
        return self.mean_gross_exposure < 0.9

    def describe(self) -> str:
        lines = [
            f"  days            {self.days}",
            f"  CAGR            {self.cagr:+.2%}",
            f"  volatility      {self.volatility:.2%}",
            f"  Sharpe          {self.sharpe:.2f}",
            f"  max drawdown    {self.max_drawdown:.2%}",
            f"  hit rate        {self.hit_rate:.1%}",
            f"  turnover/day    {self.avg_daily_turnover:.2%}",
            f"  cost drag       {self.total_cost_drag:.2%} of NAV, cumulative",
            f"  mean gross      {self.mean_gross_exposure:.1%} invested",
            f"  skew / kurtosis {self.skew:+.2f} / {self.kurtosis:.2f}",
        ]
        if self.underinvested:
            lines.append(
                f"  WARNING only {self.mean_gross_exposure:.0%} invested on average. "
                "Sharpe is scale-invariant to cash, so it is NOT comparable with a "
                "fully-invested strategy — read CAGR instead."
            )
        return "\n".join(lines)


def compute_metrics(
    net_returns: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
) -> Metrics:
    r = net_returns.to_numpy(dtype="float64")
    gross = 1.0
    if weights is not None and not weights.empty:
        gross = float(weights.abs().sum(axis=1).mean())
    return Metrics(
        days=int(r.size),
        cagr=cagr(r),
        volatility=float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.size > 1 else 0.0,
        sharpe=sharpe(r),
        max_drawdown=max_drawdown(r),
        hit_rate=float((r > 0).mean()) if r.size else 0.0,
        avg_daily_turnover=float(turnover.mean()) if turnover is not None and len(turnover) else 0.0,
        total_cost_drag=float(costs.sum()) if costs is not None and len(costs) else 0.0,
        skew=skewness(r),
        kurtosis=kurtosis(r),
        mean_gross_exposure=gross,
    )
