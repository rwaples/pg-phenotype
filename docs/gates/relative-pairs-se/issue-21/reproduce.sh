#!/usr/bin/env bash
# Issue #21: regenerate every output in this directory from its scripts.
#
# Usage: bash reproduce.sh GATE_RESULTS FRESH_RESULTS OLD_RESULTS
#   GATE_RESULTS   the gate's 600 reps: <simACE>/results/se_calibration
#   FRESH_RESULTS  400 reps of se_calibration_fo21.yaml on the current simACE:
#                  `pixi run simace run se_calibration_fo21 --until cohort --jobs 8` -> results/se_calibration_fo21
#   OLD_RESULTS    400 pedigrees from the gate's simACE source on fresh seeds:
#                  `run_old_simace.sh 920000 400 <dir>/se_cal_lam3_K10` -> <dir>
# Run from the pg-phenotype root.
set -euo pipefail
here=docs/gates/relative-pairs-se/issue-21
gate=$1 fresh=$2 old=$3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1
pixi run python -B $here/fo_probe.py "$gate" --out $here/fo_probe.tsv --jobs 6
pixi run python -B $here/fo_probe.py "$fresh" --out $here/fo21_probe.tsv --jobs 6
pixi run python -B $here/fo_probe.py "$old" --out $here/fo21_old_probe.tsv --jobs 6
pixi run python -B $here/seed_overlap.py "$gate" $here/fo_probe.tsv > $here/seed_overlap.txt
pixi run python -B $here/fo21_summary.py $here/fo21_probe.tsv > $here/fo21_summary.txt
pixi run python -B $here/fo21_summary.py $here/fo21_old_probe.tsv > $here/fo21_old_summary.txt
