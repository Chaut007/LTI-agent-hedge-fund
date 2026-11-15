"""Transaction costs. The part of a backtest that decides whether it is fiction.

Commission at a zero-commission broker is zero; the cost is the spread you
cross and the price you move. Both are charged on *turnover* — the fraction of
NAV that changes hands — so a strategy that re-decides the same portfolio
daily pays nothing, and one that churns pays every day.

Defaults are deliberately pessimistic for liquid US large caps. If a strategy
only works below them, it does not work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Per-side costs in basis points of traded notional.

    spread_bps:   half the quoted bid-ask spread — what you pay crossing it.
                  5 bps is wide for a mega-cap at 10:00 ET and about right for
                  the bottom of the S&P 500.
    impact_bps:   price impact of your own order. Flat here; a sqrt-of-ADV
                  model belongs in step 5 once positions are sized against ADV.
    commission_bps: 0 at Alpaca/IBKR Lite. Kept so the model can price a
                  broker that charges.
    """

    spread_bps: float = 5.0
    impact_bps: float = 5.0
    commission_bps: float = 0.0

    @property
    def per_side_bps(self) -> float:
        return self.spread_bps + self.impact_bps + self.commission_bps

    @property
    def per_side(self) -> float:
        """As a fraction: 10 bps -> 0.001."""
        return self.per_side_bps / 10_000.0

    def charge(self, turnover: float | np.ndarray) -> float | np.ndarray:
        """Cost as a fraction of NAV for a given one-way turnover.

        `turnover` is sum(|w_new - w_old|) / 2 — the fraction of the book that
        actually traded. A full flip from 100% A to 100% B is turnover 1.0 and
        pays the per-side cost on the sells and again on the buys, which the
        factor of two below accounts for.
        """
        return 2.0 * self.per_side * np.asarray(turnover, dtype="float64")


ZERO_COST = CostModel(spread_bps=0.0, impact_bps=0.0, commission_bps=0.0)
