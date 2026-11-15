"""Block bootstrap: turn one equity curve into a distribution of outcomes.

A backtest produces a single realised path. Its max drawdown is one draw from
a distribution we never see — the same strategy, the same returns, in a
different order, would have drawn down somewhere else. Resampling the return
series in contiguous blocks (so volatility clustering survives) gives that
distribution, and the question that matters for a kill switch becomes
answerable: how often would a 15% drawdown have fired?

Circular block bootstrap (Politis & Romano 1992): sample start indices uniformly,
take `block_len` consecutive returns wrapping at the end, concatenate to the
original length. Default block of 21 trading days keeps a month of dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedge_fund.backtest.metrics import ZERO_VOL, cagr, max_drawdown, sharpe


def circular_block_indices(
    n_obs: int, block_len: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_paths, n_obs) integer index array into the original series."""
    if n_obs < 2:
        raise ValueError("need at least 2 observations to bootstrap")
    block_len = max(1, min(block_len, n_obs))
    n_blocks = int(np.ceil(n_obs / block_len))
    starts = rng.integers(0, n_obs, size=(n_paths, n_blocks))
    offsets = np.arange(block_len)
    # (n_paths, n_blocks, block_len) -> wrap -> flatten -> trim
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return idx.reshape(n_paths, -1)[:, :n_obs]


@dataclass(frozen=True)
class BootstrapResult:
    n_paths: int
    block_len: int
    observed_sharpe: float
    observed_max_dd: float
    observed_cagr: float
    sharpe_paths: np.ndarray
    max_dd_paths: np.ndarray
    cagr_paths: np.ndarray

    def dd_percentile(self, q: float) -> float:
        return float(np.percentile(self.max_dd_paths, q))

    def prob_dd_exceeds(self, threshold: float) -> float:
        """Fraction of resampled paths whose max drawdown exceeded `threshold`."""
        return float((self.max_dd_paths > threshold).mean())

    def sharpe_ci(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0 * 100.0
        return (
            float(np.percentile(self.sharpe_paths, lo)),
            float(np.percentile(self.sharpe_paths, 100.0 - lo)),
        )

    def describe(self, kill_switch_dd: float | None = 0.15) -> str:
        lo, hi = self.sharpe_ci()
        lines = [
            f"bootstrap ({self.n_paths} paths, {self.block_len}-day blocks)",
            f"  Sharpe       {self.observed_sharpe:.2f}  95% CI [{lo:.2f}, {hi:.2f}]",
            f"  max drawdown {self.observed_max_dd:.1%} observed; "
            f"median {self.dd_percentile(50):.1%}, "
            f"95th pct {self.dd_percentile(95):.1%}, "
            f"99th pct {self.dd_percentile(99):.1%}",
        ]
        if kill_switch_dd is not None:
            p = self.prob_dd_exceeds(kill_switch_dd)
            lines.append(
                f"  P(max DD > {kill_switch_dd:.0%}) = {p:.1%}"
                + ("  <-- kill switch would fire on most paths" if p > 0.5 else "")
            )
        if lo <= 0.0 <= hi:
            lines.append("  Sharpe CI includes zero: not distinguishable from no edge")
        return "\n".join(lines)


def block_bootstrap(
    returns: np.ndarray,
    n_paths: int = 2_000,
    block_len: int = 21,
    seed: int | None = 0,
) -> BootstrapResult:
    r = np.asarray(returns, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = circular_block_indices(r.size, block_len, n_paths, rng)
    paths = r[idx]  # (n_paths, n_obs)

    # Vectorised drawdown over all paths at once.
    curves = np.cumprod(1.0 + paths, axis=1)
    peaks = np.maximum.accumulate(curves, axis=1)
    dd = (1.0 - curves / peaks).max(axis=1)

    means = paths.mean(axis=1)
    stds = paths.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > ZERO_VOL, means / stds * np.sqrt(252.0), 0.0)
    totals = curves[:, -1]
    cagrs = np.where(totals > 0, totals ** (252.0 / r.size) - 1.0, -1.0)

    return BootstrapResult(
        n_paths=n_paths,
        block_len=block_len,
        observed_sharpe=sharpe(r),
        observed_max_dd=max_drawdown(r),
        observed_cagr=cagr(r),
        sharpe_paths=sharpes,
        max_dd_paths=dd,
        cagr_paths=cagrs,
    )
