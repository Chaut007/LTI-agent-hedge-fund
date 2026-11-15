"""Is the backtest real? Permutation test, Deflated Sharpe, block bootstrap."""

from hedge_fund.validation.bootstrap import BootstrapResult, block_bootstrap
from hedge_fund.validation.deflated import DeflatedResult, deflated_sharpe
from hedge_fund.validation.permutation import PermutationResult, permutation_test
from hedge_fund.validation.report import ValidationReport, validate
from hedge_fund.validation.trials import Trial, TrialLog

__all__ = [
    "BootstrapResult",
    "DeflatedResult",
    "PermutationResult",
    "Trial",
    "TrialLog",
    "ValidationReport",
    "block_bootstrap",
    "deflated_sharpe",
    "permutation_test",
    "validate",
]
