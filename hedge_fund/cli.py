"""Command line entry point.

    hf universe            resolve tickers to SEC CIKs
    hf prices              ingest daily bars
    hf fundamentals        ingest SEC XBRL facts
    hf macro               ingest FRED series
    hf ingest              all of the above, in order
    hf status              what the lake holds, and what to distrust
    hf peek --asof DATE    prove the asof view hides the future
    hf backtest            walk-forward simulation, optionally validated

Exit codes: 0 ok, 1 a step fetched nothing or an input was unusable,
2 configuration problem. `hf ingest` returns the first non-zero code of its
steps, so a cron job cannot go green on a run that ingested nothing.
`hf backtest --validate` exits 1 on a FAIL verdict for the same reason.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from typing import Callable

from hedge_fund.agents import phase_0_agents
from hedge_fund.backtest import Backtester, BacktestConfig, EqualWeight, Strategy
from hedge_fund.backtest.costs import CostModel
from hedge_fund.backtest.engine import WeightError
from hedge_fund.config import ConfigError, load_settings
from hedge_fund.data import edgar, macro, prices, universe
from hedge_fund.data.lake import Lake
from hedge_fund.data.prices import IncompleteFetchError
from hedge_fund.portfolio import (
    DIVERSIFIED,
    EQUAL,
    FUNDAMENTAL_ONLY,
    MEDIUM_TERM,
    BandedOptimizer,
    RiskManager,
    SignalStrategy,
    TopNOptimizer,
    WeightedAggregator,
)
from hedge_fund.validation import Trial, TrialLog, validate
from hedge_fund.validation.deflated import period_sharpe

DEFAULT_START = date(2005, 1, 1)


def _phase0(
    weights: dict[str, float],
    args: argparse.Namespace,
    banded: bool = False,
    risk: bool = False,
) -> Strategy:
    roster = [a for a in phase_0_agents() if weights.get(a.name, 0.0) > 0.0]
    optimizer = (
        BandedOptimizer(
            top_n=args.top_n,
            min_score=args.min_score,
            band=args.band,
            keep_ranked=args.keep_ranked,
        )
        if banded
        else TopNOptimizer(top_n=args.top_n, min_score=args.min_score)
    )
    return SignalStrategy(
        agents=roster,
        aggregator=WeightedAggregator(weights=weights),
        top_n=args.top_n,
        min_score=args.min_score,
        optimizer=optimizer,
        risk=RiskManager(DIVERSIFIED) if risk else None,
    )


# Strategies the CLI can run by name. Each builds from the --top-n / --min-score
# arguments, which is why these are factories rather than classes.
STRATEGIES: dict[str, Callable[[argparse.Namespace], Strategy]] = {
    # The baseline that failed all three validation gates with Sharpe 1.12.
    "equal_weight": lambda args: EqualWeight(),
    # All five Phase 0 agents, equally weighted. Averages a 5-day and a
    # 252-day horizon; see portfolio/aggregate.py.
    "phase0": lambda args: _phase0(dict(EQUAL), args),
    # Horizon-consistent: reversal dropped, everything else months-scale.
    "phase0_medium": lambda args: _phase0(dict(MEDIUM_TERM), args),
    # Step 5: the same selection, traded less. Isolates the no-trade band and
    # hysteresis from the risk layer, so the turnover fix can be measured alone.
    "phase0_banded": lambda args: _phase0(dict(EQUAL), args, banded=True),
    # Step 5 complete: banded trading plus enforced hard limits.
    "phase0_risk": lambda args: _phase0(dict(EQUAL), args, banded=True, risk=True),
    # Value and Quality alone, on filings for the full universe. The one
    # Phase 0 hypothesis never actually tested.
    "fundamental": lambda args: _phase0(dict(FUNDAMENTAL_ONLY), args),
}

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _lake() -> Lake:
    lake = Lake(load_settings().lake_dir)
    lake.ensure_dir()
    return lake


def _tickers(lake: Lake) -> list[str]:
    frame = lake.universe()
    if frame.empty:
        raise SystemExit("lake has no universe yet — run `hf universe` first")
    return frame["ticker"].astype(str).tolist()


# -- commands ------------------------------------------------------------------


def cmd_universe(args: argparse.Namespace) -> int:
    settings = load_settings()
    lake = _lake()
    try:
        frame, source = universe.build_universe(
            settings, lake, Path(args.from_csv) if args.from_csv else None
        )
    except FileNotFoundError as exc:
        print(f"universe: {exc}", file=sys.stderr)
        return EXIT_FAILED
    matched = int(frame["cik"].notna().sum())
    print(f"universe: {len(frame)} tickers from {source}")
    print(f"  {matched} matched to a SEC CIK")
    if matched < len(frame):
        unmatched = frame.loc[frame["cik"].isna(), "ticker"].tolist()
        print(f"  unmatched (prices only, no fundamentals): {', '.join(unmatched)}")
    print(f"\nNOTE: {universe.SURVIVORSHIP_WARNING}")
    return EXIT_OK if len(frame) else EXIT_FAILED


def cmd_prices(args: argparse.Namespace) -> int:
    lake = _lake()
    try:
        report = prices.ingest_prices(
            lake,
            _tickers(lake),
            start=args.start,
            end=args.end,
            full_refresh=args.full_refresh,
            min_coverage=args.min_coverage,
        )
    except IncompleteFetchError as exc:
        print(f"prices: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(report.describe())
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_fundamentals(args: argparse.Namespace) -> int:
    settings = load_settings()
    lake = _lake()

    def progress(ticker: str, position: int, total: int) -> None:
        print(f"  [{position:>4}/{total}] {ticker}", end="\r", flush=True)

    report = edgar.ingest_fundamentals(
        settings,
        lake,
        lake.universe(),
        on_progress=None if args.quiet else progress,
    )
    print(" " * 40, end="\r")
    print(report.describe())
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_macro(args: argparse.Namespace) -> int:
    settings = load_settings()
    lake = _lake()
    report = macro.ingest_macro(settings, lake)
    print(report.describe())
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run every ingest step; report the first failure rather than hiding it.

    Each step still runs even after an earlier one fails, so one bad source
    does not block the others, but the exit code reflects the worst outcome.
    A nightly cron that ingested nothing must not go green.
    """
    worst = EXIT_OK
    for step in (cmd_universe, cmd_prices, cmd_fundamentals):
        code = step(args)
        worst = max(worst, code)
        print()
    try:
        worst = max(worst, cmd_macro(args))
    except ConfigError as exc:
        # Macro is optional: a missing FRED key is a skip, not a failure.
        print(f"macro: skipped — {exc}")
    if worst != EXIT_OK:
        print(f"\ningest finished with failures (exit {worst})", file=sys.stderr)
    return worst


def cmd_status(args: argparse.Namespace) -> int:
    lake = _lake()
    print(f"lake: {lake.dir}")
    print(lake.coverage().to_string(index=False))
    print(f"\nKNOWN BIASES\n  - {universe.SURVIVORSHIP_WARNING}")
    print(
        f"  - revised FRED series are not point-in-time: "
        f"{', '.join(sorted(macro.REVISABLE))}"
    )
    print(
        "  - adj_close is adjusted as of today; use close for price levels, "
        "adj_close for returns"
    )
    return EXIT_OK


def cmd_peek(args: argparse.Namespace) -> int:
    """Show that the asof view refuses to leak the future — and assert it."""
    lake = _lake()
    view = lake.view(args.asof)
    tickers = _tickers(lake)[: args.limit]

    bars = view.prices(tickers, bars=10)
    print(f"asof {args.asof} — last 10 bars visible for {args.limit} tickers")
    if bars.empty:
        print("  (none)")
    else:
        newest = bars["date"].max()
        print(f"  {len(bars)} rows, newest bar = {newest}")
        assert newest <= args.asof, f"view leaked a bar dated {newest} > {args.asof}"

    facts = view.latest_fundamentals(
        tickers, tags=["Assets", "NetIncomeLoss"], period_kind=["instant", "quarterly"]
    )
    print(f"asof {args.asof} — latest fundamentals knowable")
    if facts.empty:
        print("  (none)")
    else:
        shown = facts[["ticker", "tag", "period_kind", "period_end", "filed", "value"]]
        print(shown.to_string(index=False))
        newest_filing = shown["filed"].max()
        assert newest_filing <= args.asof, (
            f"view leaked a filing dated {newest_filing} > {args.asof}"
        )
        print(f"\n  newest filing used: {newest_filing} (asserted <= {args.asof})")
    return EXIT_OK


def cmd_backtest(args: argparse.Namespace) -> int:
    lake = _lake()
    tickers = _tickers(lake)
    try:
        strategy = STRATEGIES[args.strategy](args)
    except KeyError:
        print(
            f"unknown strategy {args.strategy!r}; known: {', '.join(sorted(STRATEGIES))}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    config = BacktestConfig(
        start=args.start,
        end=args.end or date.today(),
        rebalance=args.rebalance,
        execution_lag=args.lag,
        costs=CostModel(spread_bps=args.spread_bps, impact_bps=args.impact_bps),
        benchmark=args.benchmark,
        kill_switch_dd=args.stop_dd,
        kill_switch_mode=args.stop_mode,
    )
    backtester = Backtester(lake, tickers)
    try:
        result = backtester.run(strategy, config)
    except (ValueError, WeightError) as exc:
        print(f"backtest: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(result.describe())

    if isinstance(strategy, SignalStrategy):
        if strategy.total_cost_usd > 0:
            print(f"  agent cost      ${strategy.total_cost_usd:,.2f} over the run")
        if strategy.risk is not None:
            print()
            print(strategy.risk_summary())
        if args.explain:
            print(f"\nlast rebalance ({result.config.end}):")
            print(strategy.explain_last())

    # Every run is a trial, recorded before the researcher can forget it.
    log = TrialLog(lake.dir.parent / "trials.jsonl")
    log.append(
        Trial.build(
            strategy=strategy.name,
            params={
                "strategy": args.strategy,
                "rebalance": config.rebalance,
                "lag": config.execution_lag,
                "spread_bps": config.costs.spread_bps,
                "impact_bps": config.costs.impact_bps,
                "universe_size": len(tickers),
                # top_n and min_score change the strategy, so they change the
                # trial identity: sweeping them is exactly the multiple testing
                # the Deflated Sharpe exists to charge for.
                "top_n": args.top_n,
                "min_score": args.min_score,
                "band": args.band,
                "keep_ranked": args.keep_ranked,
                "stop_dd": args.stop_dd,
            },
            start=config.start,
            end=config.end,
            n_obs=result.metrics.days,
            sharpe_period=period_sharpe(result.net_returns.to_numpy()),
        )
    )
    print(f"\n  trial recorded ({len(log)} distinct on file -> feeds the Deflated Sharpe)")

    if not args.validate:
        return EXIT_OK

    def progress(k: int, n: int) -> None:
        print(f"  permutation {k:>4}/{n}", end="\r", flush=True)

    print()
    report = validate(
        backtester,
        result,
        trial_sharpes=log.sharpes(),
        n_permutations=args.permutations,
        n_bootstrap=args.bootstrap,
        kill_switch_dd=args.kill_switch_dd,
        seed=args.seed,
        on_progress=None if args.quiet else progress,
    )
    print(" " * 30, end="\r")
    print(report.describe())
    return EXIT_OK if report.verdict.startswith("PASS") else EXIT_FAILED


# -- parser --------------------------------------------------------------------


def _add_ingest_args(parser: argparse.ArgumentParser, *, prices_args: bool) -> None:
    if prices_args:
        parser.add_argument("--start", type=_parse_date, default=DEFAULT_START)
        parser.add_argument("--end", type=_parse_date, default=None)
        parser.add_argument("--full-refresh", action="store_true")
        parser.add_argument(
            "--min-coverage",
            type=float,
            default=0.95,
            help="with --full-refresh, refuse to overwrite if fewer than this "
            "fraction of tickers returned data (default 0.95)",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_universe = sub.add_parser("universe", help="resolve tickers to SEC CIKs")
    p_universe.add_argument("--from-csv", help="CSV of tickers, one per row")
    p_universe.set_defaults(func=cmd_universe)

    p_prices = sub.add_parser("prices", help="ingest daily bars")
    _add_ingest_args(p_prices, prices_args=True)
    p_prices.set_defaults(func=cmd_prices)

    p_fund = sub.add_parser("fundamentals", help="ingest SEC XBRL facts")
    p_fund.add_argument("--quiet", action="store_true")
    p_fund.set_defaults(func=cmd_fundamentals)

    p_macro = sub.add_parser("macro", help="ingest FRED series")
    p_macro.set_defaults(func=cmd_macro)

    p_all = sub.add_parser("ingest", help="universe, prices, fundamentals, macro")
    p_all.add_argument("--from-csv", help="CSV of tickers, one per row")
    _add_ingest_args(p_all, prices_args=True)
    p_all.add_argument("--quiet", action="store_true")
    p_all.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="lake contents and known biases")
    p_status.set_defaults(func=cmd_status)

    p_peek = sub.add_parser("peek", help="show what a given date could know")
    p_peek.add_argument("--asof", type=_parse_date, required=True)
    p_peek.add_argument("--limit", type=int, default=3)
    p_peek.set_defaults(func=cmd_peek)

    p_bt = sub.add_parser("backtest", help="walk-forward simulation")
    p_bt.add_argument("--strategy", default="equal_weight", choices=sorted(STRATEGIES))
    p_bt.add_argument("--start", type=_parse_date, required=True)
    p_bt.add_argument("--end", type=_parse_date, default=None)
    p_bt.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    p_bt.add_argument(
        "--lag", type=int, default=2,
        help="bars from decision close to first earned return (default 2: "
        "decide T, fill T+1, earn from T+2)",
    )
    p_bt.add_argument("--spread-bps", type=float, default=5.0)
    p_bt.add_argument("--impact-bps", type=float, default=5.0)
    p_bt.add_argument("--benchmark", default=None, help="ticker in the lake, e.g. SPY")
    p_bt.add_argument("--top-n", type=int, default=10,
                      help="names held by a signal strategy (default 10)")
    p_bt.add_argument("--min-score", type=float, default=0.0,
                      help="aggregated score a name must beat to be held; "
                      "0.0 means above the universe median (default 0.0)")
    p_bt.add_argument("--band", type=float, default=0.01,
                      help="no-trade band for a banded strategy: a target this "
                      "close to the held weight is left alone (default 0.01)")
    p_bt.add_argument("--keep-ranked", type=float, default=1.4,
                      help="hysteresis: a held name inside top_n x this is not "
                      "sold for slipping out of top_n (default 1.4)")
    p_bt.add_argument("--stop-dd", type=float, default=None,
                      help="engine drawdown stop; flattens the book past this "
                      "fraction. Distinct from --kill-switch-dd, which only sets "
                      "the threshold the bootstrap reports on")
    p_bt.add_argument("--stop-mode", default="halt", choices=["halt", "cooldown"],
                      help="after the stop fires: stay flat, or re-enter after a wait")
    p_bt.add_argument("--explain", action="store_true",
                      help="print the final rebalance's per-agent contributions")
    p_bt.add_argument("--validate", action="store_true",
                      help="permutation test + deflated Sharpe + bootstrap")
    p_bt.add_argument("--permutations", type=int, default=200)
    p_bt.add_argument("--bootstrap", type=int, default=2000)
    p_bt.add_argument("--kill-switch-dd", type=float, default=0.15)
    p_bt.add_argument("--seed", type=int, default=0)
    p_bt.add_argument("--quiet", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def _utf8_console() -> None:
    """Never let the terminal's codepage decide whether a report can print.

    Windows consoles default to a legacy codepage (cp874 on a Thai locale,
    cp1252 on a US one), and `print` raises UnicodeEncodeError on the first
    character outside it — an em-dash in a report was enough to crash
    `hf backtest` after the simulation had already succeeded. Reconfiguring the
    streams to UTF-8 with replacement fixes every string at once; chasing
    individual characters would not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # closed or non-text stream
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"\nconfiguration problem:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
