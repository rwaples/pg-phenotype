#!/usr/bin/env bash
# Issue #21: simulate se_cal_lam3_K10's pedigree with the simACE source the gate's reps were
# built from (b2181f9, run.yaml `source`), on seeds disjoint from the se_calibration grid.
#
# Setup (once): git -C <simACE> archive -o <dir>/src.tar b2181f9 simace; tar -xf it in <dir>.
# Usage: OLD=<dir> PY=<simACE>/.pixi/envs/default/bin/python \
#            bash run_old_simace.sh FIRST_SEED N_REPS OUT_DIR [JOBS]
# Rep k gets seed FIRST_SEED + k - 1 and writes OUT_DIR/rep{k}/pedigree.parquet.  Seeds 55030-55032
# reproduce the stored se_cal_lam3_K10 reps 1-3 (seed_check.py); the issue run: 920000 400 <dir>/se_cal_old_lam3_K10.
set -euo pipefail
OLD=${OLD:?set OLD to the extracted b2181f9 tree}
PY=${PY:?set PY to a python with simACE's dependencies}
first=$1 n=$2 out=$3 jobs=${4:-8}
export PYTHONPATH=$OLD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 NUMBA_NUM_THREADS=1
one() {
  k=$1
  d=$out/rep$k
  [ -f "$d/pedigree.parquet" ] && return 0
  mkdir -p "$d"
  "$PY" -P -m simace simulate --seed $((first + k - 1)) --N 20000 --G-ped 6 --G-sim 8 \
    --mating-model standard --mating-lambda 3.0 --p-mztwin 0.02 \
    --A1 0.5 --C1 0.0 --E1 0.5 --A2 0.4 --C2 0.2 --E2 0.4 --rA 0.0 --rC 0.0 --rE 0.0 \
    --assort1 0 --assort2 0 --output-pedigree "$d/pedigree.parquet" > "$d/simulate.log" 2>&1
}
export -f one
export first out PY
seq 1 "$n" | xargs -P "$jobs" -I{} bash -c 'one {}'
echo "done: $(ls "$out" | wc -l) reps in $out"
