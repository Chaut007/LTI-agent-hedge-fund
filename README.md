# hedge_fund

[![tests](https://github.com/Chaut007/LTI-agent-hedge-fund/actions/workflows/tests.yml/badge.svg)](https://github.com/Chaut007/LTI-agent-hedge-fund/actions/workflows/tests.yml)

Multi-agent systematic equity strategy. US large caps, daily rebalance, paper
trading first.

**Status: Phase 0, steps 1-5 of 6 complete.** Contracts, data layer, a
validated backtester, five factor agents and an enforced risk layer exist.
**No configuration beats equal weight on risk-adjusted terms.** The three
price factors have no cross-sectional edge at all; Value and Quality, tested
on real filings for the full universe, have a small one that does not survive
contact with the cheapest possible alternative. Step 5 proved execution
cannot rescue either. Read all three results below before building on this.

## What is built

| | |
|---|---|
| ✅ 1. Core contracts | `Signal`, `Agent`, `Context`, asof-bounded `MarketView` |
| ✅ 2. Data layer | yfinance prices, SEC EDGAR fundamentals, FRED macro → DuckDB/parquet |
| ✅ 3. Backtester | walk-forward engine, cost model, permutation test, Deflated Sharpe, bootstrap |
| ✅ 4. Phase 0 agents | Momentum, Value, Quality, LowVol, Reversal + confidence-weighted aggregator |
| ✅ 5. Risk + optimizer | hard limits with a veto, turnover control, NAV-aware kill switch |
| ⬜ 6. Scheduling | GitHub Actions cron → Alpaca paper |

## The step 4 result: the factors have no edge on this universe

Measured on 30 large caps, 2016-2024, monthly rebalance, 10 bps/side. Read
this before adding anything on top of the agent layer.

| top-N held | 5 | 10 | 15 | 25 | all 30 (equal weight) |
|---|---|---|---|---|---|
| Sharpe | 0.83 | 1.00 | 1.08 | 1.14 | **1.25** |

Sharpe rises monotonically as the strategy holds *more* of the universe. Every
step of selection destroys value, and the best thing the agents can do is stop
selecting.

Three cross-checks confirm it is the signal, not the implementation:

- **Not costs.** At zero costs Phase 0 still lost: 1.19 against equal weight's
  1.26. Costs make it worse (a weekly top-10 burned 42% of NAV over 9 years)
  but they are not the cause.
- **Not better than chance.** Permutation test on the best config: observed
  Sharpe 1.14, null mean **1.16**, p = 0.579. Random portfolios of the same
  shape did slightly better than the actual picks.
- **The return is real but it is beta.** The Deflated Sharpe *passed* at 99.6%
  — the 1.14 is not selection luck. It is simply what being long 25 mega-caps
  paid over that window. This is precisely why both tests exist: DSR says the
  number is real, the permutation test says it did not come from the strategy.

### Then the universe was expanded to 474 names, and the shape changed

The obvious objection was universe size: cross-sectional factors are normally
run over 500-3000 names, and 30 mega-caps are too few for a rank to mean
anything. So the universe was rebuilt as the top 500 US listings by market
cap (474 ingested) and everything re-run. Zero cost, monthly:

| top-N held | 20 | 50 | 100 | 200 | 400 | all 474 (equal weight) |
|---|---|---|---|---|---|---|
| Sharpe | 1.04 | 1.06 | 1.02 | 1.07 | 1.08 | **1.07** |

The monotonic destruction is gone — the objection was real and the bigger
universe removed it. But nothing replaced it: the sweep is flat, and Phase 0
at zero cost (1.06) is equal weight (1.07) within noise. Net of 10 bps/side
it loses (0.96 vs 1.06) for the dullest possible reason: turnover is nonzero
and the gross edge is zero.

So the honest reading is that these three price factors, as constructed and
equally weighted, had **no measurable cross-sectional edge** over 2016-2024
on US large and mid caps. That is consistent with the literature for the
window — momentum crashed in 2020 and 2022, low-vol lagged a strong bull
market, and short-term reversal in large caps has been largely arbitraged
since the 2000s. It is an unsurprising result, not a bug.

**Three price factors, not five.** Value and Quality contributed almost
nothing to the runs above: the lake held filings for 20 names, so both were
silent for most of the universe on most dates. The verdict above is a verdict
on momentum, low-vol and reversal. Value and Quality were measured separately
once the filings existed — see the next section.

**What this forbids:** adding Phase 1 LLM agents now, and tuning the three
price factors. The trial log already holds 25+ entries and the Deflated
Sharpe charges for every further look. (Step 5 has since answered the other
possibility that was open here — that cheaper execution might close the gap.
It does not; see below.)

## Value and Quality, measured on real filings

886k XBRL facts for 499 of 500 filers, so Value scores 326-383 names per date
and Quality 234-272 — against 20 before. Same window, same costs, monthly,
`tools/measure_fundamentals.sh`:

| top-N held | 20 | 50 | 100 | 400 | equal weight (489) |
|---|---|---|---|---|---|
| Sharpe | **1.07** | 1.04 | 1.02 | 0.97 | **1.06** |
| permutation p | 0.050 | 0.055 | 0.090 | 0.697 | — |
| turnover/day | 0.69% | 0.58% | 0.49% | 0.39% | 0.16% |

**The ranking carries information.** Unlike the price factors, this sweep is
not flat: Sharpe falls and the p-value rises monotonically as the book widens,
with no point out of order. Four points ordering themselves by chance is a
1-in-24 event. The top of the ranking really is better than the bottom, which
is a different finding from step 4's.

**It is still not enough to beat equal weight.** The top-20 book passes the
permutation test at p = 0.050 and the Deflated Sharpe at 99.8%, and it is the
first PASS in this project. Four reasons it does not license trading it:

- **The whole gain is risk, not skill.** top-20 runs at 1.079x equal weight's
  volatility. Lever equal weight to that same volatility and it returns
  +21.18%; top-20 returned +21.44%. Selection is worth **0.26pp of CAGR**.
  Sharpe 1.07 against 1.06 is a 0.01 gap inside a bootstrap CI of [0.36, 1.81].
- **p = 0.050 is the boundary, and four configurations were tried.** The
  chance that at least one of four tests clears 5% by luck alone is 18.5%.
- **The permutation test asks an easier question than it appears to.** Its
  null is random portfolios *of the same shape* — random 20-name books, mean
  Sharpe 0.89. Beating those is not beating a 489-name equal-weight book at
  1.06, which it merely ties while paying 4.3x the turnover (3.12% cumulative
  cost drag against 0.71%).
- **Survivorship bias favours this factor specifically.** The universe is
  today's top 500 backtested from 2016. Value buys what is cheap, and the
  cheap names still in today's top 500 are the ones that recovered; the ones
  that did not are absent. This bias hits a value factor harder than it hits
  equal weight, and it is not correctable from free data.

So: a real but small signal, smaller than the risk premium it rides on and
smaller than the costs of harvesting it.

**One known universe bias, stated plainly:** "top 500 by *today's* market
cap, backtested from 2016" selects the companies that grew into the top 500.
That inflates absolute returns for every strategy on the universe, including
equal weight. The agent-versus-equal-weight comparison survives it because
both sides carry the same bias; the CAGR figures do not.

**One concrete input to step 5, now acted on:** `KILL_SWITCH_DD = 0.15` fires
on **92.9%** of bootstrapped paths for a 100%-long book (95th-percentile
drawdown 44%, SPY's own was 33.7%). Step 5 therefore defaults the stop to 40%
in `BOOTSTRAP_INFORMED` and keeps the original value in `AS_DESIGNED` so the
comparison stays reproducible.

Diagrams live in `.archify/` — open the `.html` files in a browser.

- `workflow-decision-logic-*` — the current decision path and every point at
  which it can refuse: an agent omitting a ticker, the optimizer finding
  nothing worth holding, the risk manager vetoing, the drawdown stop firing.
- `workflow-daily-pipeline-*` and `dataflow-agent-architecture-*` — drawn at
  step 2 and **now out of date**: neither shows the risk layer, the optimizer,
  or the stop. Kept because the data-layer half is still accurate.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env     # then edit it
```

`SEC_USER_AGENT` is required before any EDGAR call. The SEC fair-access policy
requires a real contact string on every request to `data.sec.gov` and blocks
requests without one, so the code fails loudly rather than hammering their
servers anonymously. `FRED_API_KEY` is optional and only gates `hf macro`.

```bash
export SEC_USER_AGENT="hedge-fund-research you@example.com"
```

## Commands

```bash
hf universe                      # resolve tickers → SEC CIKs
hf prices --start 2005-01-01     # daily OHLCV
hf fundamentals                  # SEC XBRL facts, with filing dates
hf macro                         # FRED series
hf ingest                        # all of the above, in order
hf status                        # what the lake holds + known biases
hf peek --asof 2023-04-30        # prove the view hides the future
hf backtest --strategy phase0 --start 2016-01-01 --rebalance monthly --top-n 25 --validate
pytest                           # 293 tests, no network required
```

Universe defaults to a 30-name liquid starter set so the pipeline can be
exercised on day one. Point it at your real list with
`hf universe --from-csv my_tickers.csv`, or create
`data/universe/members.csv`.

## The one idea the whole design rests on

An agent never gets a file path or a live database connection. It gets a
`MarketView` already pinned to a single `asof` date, and every query on it
carries its own `<= asof` clause:

```python
view = lake.view(date(2023, 4, 30))
view.prices(["AAPL"], bars=252)                 # last 252 trading bars <= 2023-04-30
view.prices(["AAPL"], calendar_days=30)         # or a civil-calendar window; never both
view.latest_fundamentals(
    ["AAPL"], ["NetIncomeLoss"], period_kind="quarterly"
)                                               # facts FILED by 2023-04-30, 3-month only
```

No argument an agent passes can reach past that date. Lookahead bias is the one
bug in this domain that is both silent and flattering — it makes a strategy look
better, so nothing alerts you — and discipline alone does not prevent it.

The view carries only a directory and a date — never its `Lake`. An earlier
version held the lake, and `ctx.view._lake.query(...)` reached the entire
future in one attribute access. The unbounded query now lives on `Lake` as
`unbounded_query`, named so its appearance inside an agent fails review on
sight, and `test_view_does_not_reference_its_lake` pins the view's slots to
exactly `{_dir, _asof}`. This is Python: a caller who imports duckdb and opens
the parquet file directly can still read anything. What the design prevents is
reaching unbounded data *through the object an agent is handed*.

Fundamentals clamp on the SEC **filing** date, not the period end. A quarter
ending 31 March is not knowable on 30 April; it becomes visible the day the
10-Q is filed. Restatements work the same way: a correction filed in August does
not change what a July query returns.

Period length is part of the key, not an afterthought: a 10-Q reports both a
3-month and a year-to-date figure for the same tag ending the same day, so
`latest_fundamentals` returns one row per `period_kind` and every row carries
`duration_days`. Ranking on `period_end` alone picks one arbitrarily, which
silently mixes quarterly and semi-annual numerators inside one ratio.
`tests/test_lake_pit.py` asserts all of this on both sides of each boundary.

## The backtester is the market, not a participant

`Backtester.run` walks forward one trading day at a time. On each decision
date T it builds `Context(asof=T, view=lake.view(T))` and asks the strategy
for target weights. The strategy can only see through that view. The engine
itself reads the full price panel — it has to, to mark positions to market —
but never shows it to anyone. `RecordingStrategy` in the tests asserts that no
Context ever exposed a bar past its own date.

**Timing is the live pipeline's, not a textbook's.** Default `execution_lag=2`:
decide at close T, fill at close T+1 (the conservative stand-in for a 10:00
order), first return earned at T+2. `lag=1` — fill at the same close you
decided on — is allowed but not default, because it is not what will happen.

**Costs are charged on turnover, and drift is not turnover.** Weights float
with prices between fills exactly as a real book does, so re-deciding an
unchanged portfolio pays only for the drift it undoes. Turnover is the
standard `(purchases + sales) / 2`, which makes deploying the whole book from
cash 0.5, not 1.0; the cost charged is still on the full 100% of NAV traded.
Default 10 bps per side is deliberately pessimistic for large caps.

### Three questions before a Sharpe means anything

`hf backtest --validate` asks them in the order a sceptic would:

1. **Is the stock selection better than random?** The permutation test keeps
   every recorded decision — same dates, same position count, same weights —
   and hands each day's weights to random tradeable names instead. Re-simulate
   a few hundred times through the same engine and costs. If 30% of random
   portfolios of the same shape did as well, the strategy showed nothing about
   selection; it was long equities while equities went up.
2. **Given how many things were tried, is the Sharpe distinguishable from
   selection luck?** The Deflated Sharpe Ratio (Bailey & López de Prado) needs
   the trial count, so every `hf backtest` run appends to `data/trials.jsonl`.
   The researcher is the one party who cannot be trusted to remember how many
   times they looked. Re-running an identical configuration is not a new trial;
   a new parameter is.
3. **If it is real, how bad can the drawdown get?** A block bootstrap of the
   return series answers the question the kill switch actually needs: how often
   would a 15% drawdown have fired.

The first result this produced is worth keeping in mind. Equal weight on a
synthetic universe with one drifting name scored **Sharpe 2.12 and failed the
permutation test at p = 1.0**. A profitable backtest with no selection skill is
the default outcome, not the exception — which is why the test exists.


## The step 5 result: trading less cannot rescue the signal

Step 4's diagnosis was that Phase 0 matched equal weight gross of costs and
lost net of them. So step 5 built the fix — a no-trade band, rank hysteresis,
a turnover budget, hard position limits with a veto, and a NAV-aware drawdown
stop — and measured it. The fix does not work, for a reason worth keeping.

**The band and hysteresis barely bite.** Turnover fell 3.28% to 2.83% per day,
cost drag 14.9% to 12.8%, Sharpe 0.96 to 0.98. The dominant turnover is
*membership churn*, not drift inside held names, and an exit is deliberately
never banded — a name that left the selection is sold. A monthly top-50 drawn
from 474 replaces most of its members.

**The turnover budget works only by de-levering.** Sweeping it:

| monthly budget | mean gross | Sharpe | CAGR |
|---|---|---|---|
| 0.20 | 0.375 | 1.06 | +6.0% |
| 0.35 | 0.558 | 1.08 | +9.4% |
| 0.50 | 0.750 | 1.07 | +12.7% |
| 0.75 | 0.964 | 0.96 | +15.1% |
| 1.00 | 0.985 | 0.96 | +15.7% |

Sharpe is flat while CAGR doubles, and gross exposure tracks CAGR exactly. The
book needs roughly a 0.75 budget to stay invested, and at that budget Sharpe is
back to the unconstrained 0.96. There is no budget that keeps the signal and
loses the cost: this signal's information decays about as fast as its
membership churns, so a stale top-50 is not a cheaper version of the strategy
but a weaker one.

**Sharpe hid all of this, and now it cannot.** The 0.20-budget run reported
Sharpe 1.06 — matching fully-invested equal weight — on a book 37.5% in cash,
with a 14% max drawdown against 38%. Sharpe is scale-invariant to cash, so it
cannot tell a strategy that will not deploy capital from one that does.
`Metrics.mean_gross_exposure` is now reported beside it and prints an explicit
warning below 90% that Sharpe is not comparable. That line is the difference
between this run reading as the first success and reading as what it is.

## Known biases in the free data

`hf status` reprints these. They are limitations of a $0 data budget, not bugs,
and they must be stated with any backtest number this repo produces.

**Survivorship bias (the big one).** A free universe is a *current* universe.
Companies that went bankrupt, got acquired or were dropped from the index are
absent, so a long backtest only asks "how did the survivors do" — and survivors
did well by construction. Historically worth roughly 1-4% of annual return.
Point-in-time index membership is a paid dataset (CRSP, Compustat). Until then:
treat agent-vs-agent comparisons as meaningful (they share the bias) and
absolute return levels as inflated by an unknown amount.

The "they share the bias" defence is weaker for Value than for the others, and
that matters for the one result in this repo that passes its tests. Value buys
what is cheap; the cheap names still in today's top 500 are the ones that
recovered, and the ones that did not are not in the universe at all. The bias
is selecting on exactly the variable the factor trades. Read the top-20 result
with that in mind.

**Retroactive price adjustment.** `adj_close` is adjusted as of today, not as of
each bar's own date. That is correct for computing returns and wrong for price
levels. Use `close` for anything price-level, `adj_close` for returns.

**Revised macro series.** FRED serves current values. Treasury yields and VIX
are never revised, so those are point-in-time correct. CPI and unemployment
*are* revised, sometimes months later, so a lake holding them holds a cleaner
series than anyone had live. `macro.REVISABLE` marks them, and `hf status`
names only the ones actually ingested — a caveat printed about absent data
reads as a caveat about present data. The fix is ALFRED vintages, worth doing
when the macro agent becomes load-bearing.

**Foreign filers report in two currencies.** 43 names in the 500 file the same
fact in their home currency and in USD, and `companyfacts` presents both the
same way. Every series is narrowed to one currency before anything is built
from it, and `ratio` refuses to divide two known-different units outright —
there is no FX series here to convert with, so dropping the ticker is the only
honest answer. Before that guard existed, HDFC Bank's book-to-price read 18.05
against a true 0.29, which would have ranked it the cheapest name in the
universe at every rebalance, carrying a correct filing date and confidence 1.00.

**yfinance is an unofficial scraper.** It breaks, rate limits, and silently
returns short frames. Every ingest reports per-ticker coverage so a partial
fetch is visible instead of becoming a quiet gap.

## Layout

```
hedge_fund/
  core/
    signal.py     Signal — the only thing an agent returns
    agent.py      Agent, Context, MarketView protocol
  data/
    lake.py       Lake + AsofView: the sole read path
    prices.py     yfinance → OHLCV
    edgar.py      SEC companyfacts → point-in-time XBRL facts
    macro.py      FRED → macro series
    universe.py   tickers → CIKs, survivorship warning
    _http.py      rate limiters, SEC and FRED clients
  backtest/
    engine.py     walk-forward simulator: Context per day, lag, drift, costs
    strategy.py   Strategy protocol, EqualWeight baseline, Replay/Recording helpers
    costs.py      spread + impact per side, charged on turnover
    metrics.py    Sharpe, drawdown, CAGR, skew, kurtosis
    panel.py      the one unbounded read: dates x tickers of adj_close
  validation/
    permutation.py  same shape, random names: is selection better than chance?
    deflated.py     Deflated Sharpe: is it distinguishable from selection luck?
    bootstrap.py    block bootstrap: how bad can the drawdown get?
    trials.py       append-only log of every backtest, feeds the DSR
    report.py       runs all three, one verdict
  portfolio/
    state.py      PortfolioState: the book, for the layer allowed to know it
    risk.py       RiskLimits + RiskManager: clamp, veto, and record every one
    optimizer.py  TopN (what step 4 measured) and Banded (band + hysteresis)
    aggregate.py  confidence-weighted score per ticker
    strategy.py   agents -> aggregator -> optimizer -> risk -> weights
  config.py       paths, env settings
  cli.py          the `hf` command
tests/
  test_lake_pit.py          no-lookahead + period-length  ← the one that matters
  test_lake_write.py        upsert, cold==warm dtypes, bar windows, view isolation
  test_prices_normalise.py  yfinance column-shape churn
  test_prices_ingest.py     ingest recovery paths, yfinance mocked
  test_risk.py              clamp vs veto, turnover budget, liquidity, audit trail
  test_optimizer.py         band, hysteresis, and that banding only cuts cost
  test_signal.py            contract validation, numpy scalars, bool rejection
  test_edgar_flatten.py     companyfacts parsing
  test_backtest_engine.py   hand-computable arithmetic, lag, costs, no-lookahead
  test_validation.py        skill passes / equal weight fails; DSR; bootstrap; trials
```

## Ground rules carried into later steps

- LLM agents may only emit a `Signal`. All sizing is deterministic code.
- Paper mode is the default; live trading needs an explicit flag and a typed
  confirmation.
- Every decision gets logged with the inputs that produced it, so it can be
  replayed and so each agent can be scored out-of-sample.
- `Signal.cost_usd` travels with every signal: a paid agent has to earn its keep
  against the free ones.

## What remains open

**The top-20 result deserves one honest follow-up, not a strategy.** It is a
small signal with four stated reasons to distrust it, and the one that can
actually be removed is survivorship bias — the others are about effect size.
Point-in-time universe membership is the test that would settle it, and it is
a paid dataset. Running more configurations against the same biased universe
would only raise what the Deflated Sharpe charges.

**Step 6 stays unbuilt.** Scheduling a book that ties equal weight while paying
4.3x the turnover is not a thing to automate.

**Market cap is wrong for names that split between a filing and the asof.**
The share count comes from the filing, the price from the tape, and nothing
reconciles a split in between — a whole-number error. Fixing it needs split
events ingested into the data layer.

**The macro table is empty.** `FRED_API_KEY` was never set, so no agent reads
macro and the CPI/UNRATE revision caveat is currently theoretical.
