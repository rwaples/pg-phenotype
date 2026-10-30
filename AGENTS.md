# AGENTS.md

pg-phenotype: phenotype statistics (PA-FGRS, Aalen-Johansen CIP, assortative mating, parent-offspring regression, relative-pair correlation) on pedigree-graph, with a Rust core, a pyo3 Python package, and an extendr R package (`r/`). [README.md](README.md) covers usage, the correctness gates, and development commands; [docs/releasing.md](docs/releasing.md) the release.

- Everything runs through pixi; bare `cargo` and `R` are not on PATH.
- Before pushing, run `pixi run ci-local`: the CI push jobs' Rust fmt, clippy and tests, ruff, ty and pytest, and the R binding lint and testthat suite. CI also runs Python 3.13 and every suite against pedigree-graph main (`pixi run -e r test-against-pg ../pedigree-graph`), and the publish workflow builds macOS, Windows and ARM wheels, so a float assertion in a test needs a tolerance, never `==`.
- The R binding crate `r/src/rust` builds only in the `r` env: `pixi run -e r lint-r-rust`. A cargo run outside it caches extendr's failed build script; `cargo clean -p extendr-api -p extendr-ffi` clears that.
- The `tools/*_gate.py`, `make_simace_*` and `consumer_simace.py` scripts import simACE, some of it private (`make_simace_po_regression_golden.py:74`); a simACE refactor can break them without failing any simACE test. The `make_simace_*` and `consumer_simace.py` tools import simACE from a checkout at `SIMACE_REV` (`tools/_simace.py`): later simACE computes these statistics with pg-phenotype, so do not port them to current simACE.
