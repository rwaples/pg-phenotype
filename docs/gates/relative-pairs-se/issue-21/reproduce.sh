#!/usr/bin/env bash
# Issue #21: regenerate every output in this directory from its scripts.
#
# Usage (from the pg-phenotype root):
#   bash reproduce.sh GATE_RESULTS FRESH_RESULTS OLD_RESULTS SEEDCHECK_CURRENT SEEDCHECK_OLD [--rust]
#
# GATE_RESULTS       the gate's 600 reps: <simACE>/results/se_calibration
# FRESH_RESULTS      400 reps on simACE 8b46ab4 (v0.2.0-142, the run's run.yaml `source`).  Copy
#                    se_calibration_fo21.yaml into <simACE>/config/, then from <simACE>:
#                      pixi run simace run se_calibration_fo21 --until cohort --jobs 8
#                    -> <simACE>/results/se_calibration_fo21.  A simACE whose random stream has
#                    changed since 8b46ab4 gives other pedigrees (simACE #78 changed it once).
# OLD_RESULTS        400 pedigrees from the gate reps' source b2181f9 on seeds 920000-920399:
#                      run_old_simace.sh 920000 400 <dir>/se_cal_old_lam3_K10   -> <dir>
# SEEDCHECK_CURRENT  seeds 55030-55032 on simACE 8b46ab4: copy se_calibration_fo21_seedcheck.yaml into
#                    <simACE>/config/ and run `simace run se_calibration_fo21_seedcheck --until simulate`
#                    -> <simACE>/results/se_calibration_fo21_seedcheck/se_cal_fo21_seedcheck
# SEEDCHECK_OLD      the same seeds on b2181f9: run_old_simace.sh 55030 3 <dir>
# --rust             also rerun the shipped SE on FRESH_RESULTS (about 50 min on 7 workers) into
#                    fo21_rust_calibration.tsv and fo21_rust_summary.tsv.
set -euo pipefail
here=docs/gates/relative-pairs-se/issue-21
gate=$1 fresh=$2 old=$3 check_current=$4 check_old=$5 rust=${6:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1
pixi run python -B $here/fo_probe.py "$gate" --out $here/fo_probe.tsv --jobs 6
pixi run python -B $here/fo_probe.py "$fresh" --out $here/fo21_probe.tsv --jobs 6
pixi run python -B $here/fo_probe.py "$old" --pedigree-only --out $here/fo21_old_probe.tsv --jobs 6
pixi run python -B $here/seed_overlap.py "$gate" $here/fo_probe.tsv > $here/seed_overlap.txt
pixi run python -B $here/fo21_summary.py $here/fo21_probe.tsv > $here/fo21_summary.txt
pixi run python -B $here/fo21_summary.py $here/fo21_old_probe.tsv > $here/fo21_old_summary.txt
pixi run python -B $here/seed_check.py "$gate/se_cal_lam3_K10" "$check_current" "$check_old" > $here/seed_check.txt
if [ "$rust" = --rust ]; then
  tmp=$(mktemp -d)
  PG_PHENOTYPE_THREADS=1 RAYON_NUM_THREADS=1 pixi run python -B tools/relative_pairs_se_calibration.py \
    "$fresh" --rust --jobs 7 --out "$tmp"
  mv "$tmp/rust_calibration.tsv" $here/fo21_rust_calibration.tsv
  mv "$tmp/rust_summary.tsv" $here/fo21_rust_summary.tsv
fi
