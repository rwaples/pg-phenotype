#!/usr/bin/env bash
# Rebuild pg-phenotype against a candidate pedigree-graph checkout and run the
# Rust, Python and R suites (ADR 0001, "Consumer-gate routing").
#
#   pixi run test-against-pg <pg-checkout>      # Rust + Python
#   PG_PHENOTYPE_WITH_R=1 pixi run -e r tools/test_against_pg.sh <pg-checkout>
#
# The pinned git dependency is redirected with a cargo [patch] in a private
# CARGO_HOME (sharing the user's registry and git caches), so cargo, maturin
# and R CMD INSTALL all see the same override.  The patch rewrites the lock
# files, so both are restored on exit, and the editable extension (and, after
# an R run, the installed R package) is rebuilt against the pin so the checkout
# is left as it was found.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <pedigree-graph-checkout>" >&2
  exit 2
fi
PG="$(realpath "$1")"
if [ ! -f "$PG/crates/core/Cargo.toml" ]; then
  echo "error: $PG/crates/core/Cargo.toml not found" >&2
  exit 2
fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SCRATCH="$(mktemp -d)"
USER_CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
cp Cargo.lock "$SCRATCH/Cargo.lock"
[ -f r/src/rust/Cargo.lock ] && cp r/src/rust/Cargo.lock "$SCRATCH/r.Cargo.lock"

restore() {
  status=$?
  cp "$SCRATCH/Cargo.lock" Cargo.lock
  [ -f "$SCRATCH/r.Cargo.lock" ] && cp "$SCRATCH/r.Cargo.lock" r/src/rust/Cargo.lock
  unset CARGO_HOME
  echo "== restoring the editable build against the pinned pedigree-graph-core"
  maturin develop --release --features test-hooks >/dev/null 2>&1 || echo "warning: rerun 'pixi run build-dev'" >&2
  if [ -n "${R_INSTALLED:-}" ]; then
    echo "== restoring the R package against the pinned pedigree-graph-core"
    PG_PHENOTYPE_CARGO_FEATURES=test-hooks R CMD INSTALL --no-multiarch --preclean r >/dev/null 2>&1 \
      || echo "warning: rerun 'pixi run -e r r-install'" >&2
  fi
  rm -rf "$SCRATCH"
  exit "$status"
}
trap restore EXIT

export CARGO_HOME="$SCRATCH/cargo-home"
mkdir -p "$CARGO_HOME" "$USER_CARGO_HOME/registry" "$USER_CARGO_HOME/git"
ln -s "$USER_CARGO_HOME/registry" "$CARGO_HOME/registry"
ln -s "$USER_CARGO_HOME/git" "$CARGO_HOME/git"
cat > "$CARGO_HOME/config.toml" <<TOML
[patch."https://github.com/rwaples/pedigree-graph"]
pedigree-graph-core = { path = "$PG/crates/core" }
TOML
# A candidate whose version differs from the locked pin is not applied until
# each lock's entry is resolved again (cargo warns that the patch "was not
# used"); check_patch catches a lock that still names the pin.
cargo update -p pedigree-graph-core
cargo update --manifest-path r/src/rust/Cargo.toml -p pedigree-graph-core
# Fail unless the lock file the last build wrote resolves pedigree-graph-core
# to the candidate.  The output is captured whole: `grep -q` would close the
# pipe early and, under pipefail, fail the check on cargo's SIGPIPE.
check_patch() {
  local what="$1" tree
  shift
  tree="$(cargo tree "$@" --locked -i pedigree-graph-core --depth 0)" && [[ "$tree" == *"$PG/crates/core"* ]] \
    || { echo "error: $what did not take the patch: pedigree-graph-core is not $PG" >&2; exit 1; }
}

echo "== pedigree-graph-core <- $PG ($(git -C "$PG" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"

echo "== cargo test"
cargo test --release
check_patch "the cargo build" -p pg-phenotype-core

# The pixi tasks (build-dev, test-all, r-test) are spelled out rather than
# called: a nested `pixi run` re-runs activation, and the patched CARGO_HOME
# above must reach the builds.  Keep these lines equal to those tasks.
echo "== Python suite"
maturin develop --release --features test-hooks
PG_PHENOTYPE_REQUIRE_TEST_HOOKS=1 pytest -n 6 --dist worksteal

if [ "${PG_PHENOTYPE_WITH_R:-0}" = "1" ]; then
  echo "== R suite"
  R_INSTALLED=1
  PG_PHENOTYPE_CARGO_FEATURES=test-hooks R CMD INSTALL --no-multiarch --preclean r
  check_patch "the R build" --manifest-path r/src/rust/Cargo.toml
  PG_PHENOTYPE_REQUIRE_TEST_HOOKS=1 Rscript -e 'testthat::test_local("r", stop_on_failure = TRUE)'
fi
echo "== all suites pass against $PG"
