#!/usr/bin/env bash
# Show what pedsum #13 changed since the oracle pin (tools/pedsum_am.pin) in
# the files the assortative-mating port reads.  Read-only: never checks out,
# fetches or edits the pedsum checkout.  Run at the start of every port unit.
#
#   tools/pedsum_am_drift.sh [pedsum-checkout] [branch]
#
# Defaults: ../pedsum-issue13 and issue-13-assortative-mating.  Exit 0 with
# "no drift" when the branch tip is the pin; else the stat, then the diff.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PEDSUM="$(realpath "${1:-$REPO/../pedsum-issue13}")"
BRANCH="${2:-issue-13-assortative-mating}"
PIN="$(tr -d '[:space:]' < "$REPO/tools/pedsum_am.pin")"

PATHS=(
  'pedsum/assortative_*.py'
  pedsum/base.py
  pedsum/pedigree_ops.py
  pedsum/validate.py
  pedsum/parse.py
  pedsum/cli.py
  'tests/*assortative*'
  pyproject.toml
  pixi.toml
  pixi.lock
)

TIP="$(git -C "$PEDSUM" rev-parse "$BRANCH")"
if ! git -C "$PEDSUM" merge-base --is-ancestor "$PIN" "$TIP"; then
  echo "warning: pin $PIN is not an ancestor of $BRANCH ($TIP); the branch was rewritten" >&2
fi
if [ "$TIP" = "$PIN" ]; then
  echo "no drift: $BRANCH is at the pin $PIN"
  exit 0
fi
echo "pin $PIN -> $BRANCH $TIP"
git -C "$PEDSUM" --no-pager diff --stat "$PIN" "$TIP" -- "${PATHS[@]}"
git -C "$PEDSUM" --no-pager diff "$PIN" "$TIP" -- "${PATHS[@]}"
