#!/usr/bin/env bash
# Measure whether Value and Quality have a cross-sectional edge.
#
#   bash tools/measure_fundamentals.sh [out-dir]   (default: build/runs)
#
# Needs filings for the whole universe in the lake:
#   hf universe --from-csv data/universe/top500.csv
#   SEC_USER_AGENT="... you@example.com" hf fundamentals
#   hf prices --start 2014-01-01 --end 2024-12-31
#
# Each run takes roughly forty minutes: every rebalance reassembles trailing
# twelve months for ~500 filers across 108 monthly decision dates.
#
# One log per run, not one shared log. Two concurrent runs redirected to the
# same file write at independent offsets and overwrite each other's bytes —
# that produced a log with NUL holes, a missing run, and two different Sharpe
# figures for the same configuration.
#
# A lock file makes a second concurrent launch refuse instead of interleave.
set -u

OUT="${1:-build/runs}"
mkdir -p "$OUT"

LOCK="$OUT/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "refusing: $LOCK exists — another sweep is running (or died; rmdir it)" >&2
  exit 2
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

COMMON=(--start 2016-01-01 --end 2024-12-31 --rebalance monthly --quiet)

run () {
  local name="$1"; shift
  local log="$OUT/$name.log"
  echo "[$(date +%H:%M:%S)] start $name"
  python -m hedge_fund.cli backtest "${COMMON[@]}" "$@" > "$log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] done  $name (exit $code)"
  return 0
}

run equal_weight --strategy equal_weight
for n in 50 100 400 20; do
  run "fundamental_top$n" --strategy fundamental --top-n "$n" --validate
done
echo "ALL DONE"
