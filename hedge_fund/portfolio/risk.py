"""The risk layer: hard limits with a veto, and a record of every intervention.

Two rules shape this file.

**It clamps, and when it cannot clamp it vetoes.** A position over its cap is
cut to the cap, because the rest of the portfolio is still fine. A short in a
long-only book is not a size problem, it is a sign the layer above is wrong
about what it is allowed to do — so the whole proposal is refused and the book
goes flat rather than being quietly corrected into something nobody asked for.

**Every intervention is recorded, not just applied.** `RiskDecision.actions`
says what was changed and why. A risk layer that silently fixes things teaches
you nothing about the strategy feeding it, and the strategy is the thing that
needs fixing. The counts are what tell you a limit is doing real work instead
of sitting unreached.

These limits were agreed during design. One of them has since been measured
and found wrong: `kill_switch_dd` at 0.15 fires on 91-93% of bootstrapped
paths for a fully-invested long-only book, which makes it a stop on ordinary
market behaviour rather than on evidence the model broke. The default here is
therefore taken from the bootstrap, and the old value is kept in a named
preset so the comparison can still be run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, Sequence

from hedge_fund.portfolio.state import PortfolioState


class Breach(str, Enum):
    """Why the risk layer intervened. The audit trail's vocabulary."""

    POSITION_CAP = "position_cap"
    GROSS_EXPOSURE = "gross_exposure"
    TOO_MANY_POSITIONS = "too_many_positions"
    TOO_FEW_POSITIONS = "too_few_positions"
    TURNOVER_BUDGET = "turnover_budget"
    ILLIQUID_ORDER = "illiquid_order"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    NON_FINITE_WEIGHT = "non_finite_weight"
    UNKNOWN_TICKER = "unknown_ticker"
    HALTED = "halted"


# Breaches that mean the layer above misunderstands its mandate. Clamping these
# would hide a bug behind a plausible book.
VETO_BREACHES = frozenset(
    {
        Breach.SHORT_NOT_ALLOWED,
        Breach.NON_FINITE_WEIGHT,
        Breach.UNKNOWN_TICKER,
        Breach.HALTED,
    }
)


@dataclass(frozen=True, slots=True)
class RiskAction:
    breach: Breach
    detail: str
    ticker: str | None = None

    def __str__(self) -> str:
        where = f" [{self.ticker}]" if self.ticker else ""
        return f"{self.breach.value}{where}: {self.detail}"


@dataclass(frozen=True)
class RiskLimits:
    """Hard bounds the optimizer's proposal is held to.

    max_position:   cap per name as a fraction of NAV.
    min_positions:  below this the book is too concentrated to be held at all,
                    so it goes flat. Not padded out with names the strategy did
                    not choose — inventing positions to satisfy a diversification
                    rule is worse than holding cash.
    max_positions:  above this the lowest-ranked names are dropped.
    max_turnover:   one-way turnover allowed per rebalance. The measured reason
                    Phase 0 lost: it traded 3.28% of the book per day.
    max_order_vs_adv: an order may not exceed this share of the name's average
                    daily dollar volume. Needs NAV and volume, so it is only
                    checked when both are supplied.
    kill_switch_dd: drawdown from peak that flattens the book. Enforced by the
                    engine, which is the only thing that knows NAV; carried here
                    so one object holds every limit.
    """

    max_position: float = 0.10
    min_positions: int = 8
    max_positions: int = 15
    max_gross: float = 1.0
    allow_short: bool = False
    max_turnover: float = 0.20
    max_order_vs_adv: float = 0.05
    kill_switch_dd: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.max_position <= 1.0:
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")
        if self.min_positions < 1:
            raise ValueError("min_positions must be at least 1")
        if self.max_positions < self.min_positions:
            raise ValueError(
                f"max_positions ({self.max_positions}) is below min_positions "
                f"({self.min_positions})"
            )
        if self.max_gross <= 0.0:
            raise ValueError("max_gross must be positive")
        if self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be positive")
        if not 0.0 < self.kill_switch_dd < 1.0:
            raise ValueError("kill_switch_dd must be in (0, 1)")
        # A cap below 1/max_positions cannot be satisfied: the book could not
        # reach full investment even holding the maximum number of names.
        if self.max_position * self.max_positions < self.max_gross - 1e-9:
            raise ValueError(
                f"max_position {self.max_position} x max_positions "
                f"{self.max_positions} = {self.max_position * self.max_positions:.3f}, "
                f"below max_gross {self.max_gross}: the book could never be fully "
                "invested. Raise the cap, hold more names, or lower max_gross."
            )


#: The limits as agreed in design, including the 15% stop later measured to fire
#: on 91-93% of bootstrapped paths. Kept so that comparison can be reproduced.
AS_DESIGNED = RiskLimits(kill_switch_dd=0.15)

#: Bootstrap-informed: the 95th-percentile drawdown measured for a fully
#: invested long-only book on this universe was 44-51%, so 40% is near the edge
#: of ordinary rather than inside it.
BOOTSTRAP_INFORMED = RiskLimits(kill_switch_dd=0.40)

#: Diversified: the measured sweeps said concentration cost Sharpe at every step,
#: so this holds far more names than the design's 8-15 and caps each smaller.
DIVERSIFIED = RiskLimits(max_position=0.04, min_positions=20, max_positions=50)


@dataclass(frozen=True)
class RiskDecision:
    weights: dict[str, float]
    actions: tuple[RiskAction, ...] = ()
    vetoed: bool = False

    @property
    def breaches(self) -> frozenset[Breach]:
        return frozenset(a.breach for a in self.actions)

    def describe(self) -> str:
        if not self.actions:
            return "risk: no intervention"
        head = "risk: VETOED, book flat" if self.vetoed else "risk: clamped"
        return "\n".join([head, *(f"  {a}" for a in self.actions)])


class RiskManager:
    """Enforces `RiskLimits` on a proposal. Deterministic, no model, no LLM.

    Holds no state: everything it needs arrives as arguments, so the same call
    in a backtest and in live trading cannot diverge.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def review(
        self,
        targets: Mapping[str, float],
        state: PortfolioState,
        universe: Sequence[str] | None = None,
        adv: Mapping[str, float] | None = None,
    ) -> RiskDecision:
        """Hold `targets` to the limits, reporting every change.

        `adv` is average daily dollar volume per ticker. Supplied together with
        a NAV above zero, it enables the liquidity check; omitted, that check is
        skipped rather than guessed at.
        """
        limits = self.limits
        actions: list[RiskAction] = []

        if state.halted:
            return RiskDecision(
                weights={},
                actions=(RiskAction(Breach.HALTED, "kill switch has fired; book stays flat"),),
                vetoed=True,
            )

        known = set(universe) if universe else None
        weights: dict[str, float] = {}

        for ticker, raw in targets.items():
            try:
                weight = float(raw)
            except (TypeError, ValueError):
                weight = float("nan")
            if weight != weight or weight in (float("inf"), float("-inf")):
                return RiskDecision(
                    weights={},
                    actions=(
                        RiskAction(Breach.NON_FINITE_WEIGHT, f"weight is {raw!r}", ticker),
                    ),
                    vetoed=True,
                )
            if known is not None and ticker not in known:
                return RiskDecision(
                    weights={},
                    actions=(
                        RiskAction(
                            Breach.UNKNOWN_TICKER,
                            "weighted a name outside the universe",
                            ticker,
                        ),
                    ),
                    vetoed=True,
                )
            if weight < 0.0 and not limits.allow_short:
                return RiskDecision(
                    weights={},
                    actions=(
                        RiskAction(
                            Breach.SHORT_NOT_ALLOWED,
                            f"weight {weight:+.4f} with allow_short=False",
                            ticker,
                        ),
                    ),
                    vetoed=True,
                )
            if weight != 0.0:
                weights[ticker] = weight

        if not weights:
            return RiskDecision(weights={}, actions=tuple(actions))

        weights = self._enforce_caps(weights, limits, actions)

        if len(weights) < limits.min_positions:
            # Flat, not padded. Adding names the strategy did not pick to satisfy
            # a diversification rule would be the risk layer taking positions.
            actions.append(
                RiskAction(
                    Breach.TOO_FEW_POSITIONS,
                    f"{len(weights)} names is below min_positions "
                    f"{limits.min_positions}; holding cash instead of inventing names",
                )
            )
            return RiskDecision(weights={}, actions=tuple(actions), vetoed=False)

        # -- liquidity ---------------------------------------------------------
        if adv and state.nav > 0.0:
            for ticker, weight in list(weights.items()):
                capacity = adv.get(ticker)
                if not capacity or capacity <= 0.0:
                    continue
                traded = abs(weight - state.weight(ticker)) * state.nav
                allowed = limits.max_order_vs_adv * capacity
                if traded > allowed + 1e-9:
                    room = allowed / state.nav
                    toward = state.weight(ticker)
                    limited = (
                        min(weight, toward + room) if weight > toward else max(weight, toward - room)
                    )
                    actions.append(
                        RiskAction(
                            Breach.ILLIQUID_ORDER,
                            f"order {traded:,.0f} exceeds {limits.max_order_vs_adv:.0%} "
                            f"of ADV {capacity:,.0f}; weight {weight:+.4f} -> {limited:+.4f}",
                            ticker,
                        )
                    )
                    weights[ticker] = limited

        weights = self._enforce_gross(weights, limits, actions)

        # -- turnover budget ---------------------------------------------------
        turnover = state.turnover_to(weights)
        if turnover > limits.max_turnover + 1e-9:
            # Move part of the way instead of all of it. The book ends between
            # where it was and where the strategy wanted, which is what a real
            # desk does when a rebalance is too large to execute at once.
            lam = limits.max_turnover / turnover
            blended = {
                t: state.weight(t) + lam * (weights.get(t, 0.0) - state.weight(t))
                for t in set(weights) | set(state.weights)
            }
            blended = {t: w for t, w in blended.items() if abs(w) > 1e-9}
            actions.append(
                RiskAction(
                    Breach.TURNOVER_BUDGET,
                    f"turnover {turnover:.4f} exceeds {limits.max_turnover:.4f}; "
                    f"moved {lam:.1%} of the way to target",
                )
            )
            # The blend reintroduces every name currently held, so it can carry
            # the book past max_positions and the per-name cap. Those are hard
            # limits and the budget is a preference, so they are re-applied and
            # the resulting overrun is recorded rather than hidden. Without this
            # a tight budget grew the book to 207 names against a limit of 50,
            # permanently half-invested, while reporting no breach.
            blended = self._enforce_caps(blended, limits, actions)
            blended = self._enforce_gross(blended, limits, actions)
            after = state.turnover_to(blended)
            if after > limits.max_turnover + 1e-9:
                actions.append(
                    RiskAction(
                        Breach.TURNOVER_BUDGET,
                        f"budget overrun to {after:.4f}: the position limit "
                        f"({limits.max_positions} names) cannot be met within it",
                    )
                )
            weights = blended

        return RiskDecision(weights=weights, actions=tuple(actions))

    # -- enforcement applied wherever the book changes shape -------------------

    @staticmethod
    def _enforce_caps(
        weights: dict[str, float], limits: RiskLimits, actions: list[RiskAction]
    ) -> dict[str, float]:
        """Per-name cap, then the position count. Idempotent, so safe to re-run."""
        for ticker, weight in list(weights.items()):
            if abs(weight) > limits.max_position + 1e-12:
                capped = limits.max_position * (1.0 if weight > 0 else -1.0)
                actions.append(
                    RiskAction(Breach.POSITION_CAP, f"{weight:+.4f} -> {capped:+.4f}", ticker)
                )
                weights[ticker] = capped

        if len(weights) > limits.max_positions:
            ranked = sorted(weights.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
            dropped = ranked[limits.max_positions :]
            shown = ", ".join(t for t, _ in dropped[:8])
            more = f" (+{len(dropped) - 8} more)" if len(dropped) > 8 else ""
            actions.append(
                RiskAction(
                    Breach.TOO_MANY_POSITIONS,
                    f"{len(weights)} names -> {limits.max_positions}, dropped {shown}{more}",
                )
            )
            weights = dict(ranked[: limits.max_positions])
        return weights

    @staticmethod
    def _enforce_gross(
        weights: dict[str, float], limits: RiskLimits, actions: list[RiskAction]
    ) -> dict[str, float]:
        gross = sum(abs(w) for w in weights.values())
        if gross > limits.max_gross + 1e-9:
            scale = limits.max_gross / gross
            actions.append(
                RiskAction(
                    Breach.GROSS_EXPOSURE,
                    f"gross {gross:.4f} -> {limits.max_gross:.4f}, scaled by {scale:.4f}",
                )
            )
            return {t: w * scale for t, w in weights.items()}
        return weights


def with_kill_switch(limits: RiskLimits, drawdown: float) -> RiskLimits:
    """Convenience for sweeping the stop without rebuilding every other limit."""
    return replace(limits, kill_switch_dd=drawdown)
