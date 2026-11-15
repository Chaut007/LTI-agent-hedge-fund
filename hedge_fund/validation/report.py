"""Run every validation on a backtest result and render one verdict.

Three questions, three tests, in the order a sceptic would ask them:

1. Is the stock selection better than random?       -> permutation test
2. Given how many things were tried, is the Sharpe
   distinguishable from selection luck?             -> deflated Sharpe
3. If it is real, how bad can the drawdown get?     -> block bootstrap

A strategy has to pass 1 and 2 before 3 is worth reading. A beautiful
drawdown distribution for a strategy with no selection skill is a beautiful
drawdown distribution for a market index with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from hedge_fund.backtest.engine import Backtester, BacktestResult
from hedge_fund.validation.bootstrap import BootstrapResult, block_bootstrap
from hedge_fund.validation.deflated import DeflatedResult, deflated_sharpe
from hedge_fund.validation.permutation import PermutationResult, permutation_test


@dataclass(frozen=True)
class ValidationReport:
    strategy: str
    permutation: PermutationResult | None
    deflated: DeflatedResult
    bootstrap: BootstrapResult
    kill_switch_dd: float

    @property
    def verdict(self) -> str:
        failed = []
        if self.permutation is not None and not self.permutation.passes:
            failed.append("selection not better than random")
        if not self.deflated.passes:
            failed.append("Sharpe not distinguishable from selection luck")
        if failed:
            return "FAIL: " + "; ".join(failed)
        p_kill = self.bootstrap.prob_dd_exceeds(self.kill_switch_dd)
        if p_kill > 0.5:
            return (
                f"PASS with warning: kill switch at {self.kill_switch_dd:.0%} fires on "
                f"{p_kill:.0%} of resampled paths"
            )
        return "PASS"

    def describe(self) -> str:
        parts = [f"validation: {self.strategy}"]
        if self.permutation is not None:
            parts.append(self.permutation.describe())
        parts.append(self.deflated.describe())
        parts.append(self.bootstrap.describe(self.kill_switch_dd))
        parts.append(f"verdict: {self.verdict}")
        return "\n\n".join(parts)


def validate(
    backtester: Backtester,
    result: BacktestResult,
    trial_sharpes: Sequence[float] | None = None,
    n_permutations: int = 200,
    n_bootstrap: int = 2_000,
    block_len: int = 21,
    kill_switch_dd: float = 0.15,
    seed: int | None = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> ValidationReport:
    """Run all three tests. `trial_sharpes` comes from `TrialLog.sharpes()`.

    `n_permutations=0` skips the permutation test (useful when a strategy has
    no selection component to test, e.g. a single-asset timing rule).
    """
    returns = result.net_returns.to_numpy()
    permutation = (
        permutation_test(backtester, result, n_permutations, seed, on_progress)
        if n_permutations > 0
        else None
    )
    return ValidationReport(
        strategy=result.strategy,
        permutation=permutation,
        deflated=deflated_sharpe(returns, trial_sharpes),
        bootstrap=block_bootstrap(returns, n_bootstrap, block_len, seed),
        kill_switch_dd=kill_switch_dd,
    )
