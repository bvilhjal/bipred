#!/usr/bin/env bash
# Regenerate every self-contained benchmark artifact, in order, single-core.
#
# Concurrent benchmark processes compete for memory bandwidth, so these run
# strictly sequentially -- see benchmarks/README.md. Each script writes its own
# CSV (and PNG where it has one); this only sequences them and records what ran.
#
# Usage:  bash benchmarks/run_all.sh [output-log]
# The interpreter comes from $BIPRED_PYTHON, defaulting to the one on PATH; it
# must be an environment where `import msprime` succeeds, or the coalescent
# falls back to the bundled simulator and none of the cached population-LD
# segments (which are tagged by backend) will hit.

set -euo pipefail

PY="${BIPRED_PYTHON:-python}"
LOG="${1:-benchmarks/run_all.log}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# An artifact cannot identify its source if tracked edits, staged edits or
# untracked source files are mixed into HEAD. Check before opening the log --
# the generated artifacts themselves are expected to dirty the tree afterward.
SOURCE_STATUS="$(git status --porcelain --untracked-files=normal)"
if [ -n "$SOURCE_STATUS" ]; then
  echo "refusing benchmark regeneration from a dirty source tree" >&2
  echo "commit or stash all changes, then rerun; use individual scripts for exploratory work" >&2
  exit 2
fi
SOURCE_REVISION="$(git rev-parse HEAD)"
LDPRED3_SOURCE="$("$PY" -c 'import json; from benchmarks.real_data_inputs import require_ldpred3_source; print(json.dumps(require_ldpred3_source(), sort_keys=True))')"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bipred-mpl}"

# sweep_cost and fit_memory take flags, so they run through -m below rather
# than as bare scripts; everything else is a plain module run.
MODULE_SCRIPTS=(
  "sweep_cost --csv benchmarks/sweep_cost.csv"
  "fit_memory --csv benchmarks/fit_memory.csv"
)

SCRIPTS=(
  rg_architectures
  rg_polygenicity
  rg_methods
  rg_scaling
  mixer_overlap
  overlap_estimation
  sample_overlap
  rg_env_overlap
)

{
  echo "== bipred benchmark regeneration =="
  echo "python:   $($PY -c 'import sys; print(sys.executable)')"
  echo "versions: $($PY -c 'import numpy,numba,ldpred3,bipred; print(f"numpy {numpy.__version__} numba {numba.__version__} ldpred3 {ldpred3.__version__} bipred {bipred.__version__}")' 2>/dev/null)"
  echo "backend:  $($PY -c 'import benchmarks.simulate as s; print(s._backend(), s.SIMULATOR_CACHE_TAG)' 2>/dev/null)"
  echo "revision: $SOURCE_REVISION"
  echo "source:   clean"
  echo "ldpred3 source: $LDPRED3_SOURCE"
  echo
} | tee "$LOG"

failed=()
for name in "${SCRIPTS[@]}"; do
  echo "--- $name : started $(date +%H:%M:%S) ---" | tee -a "$LOG"
  start=$SECONDS
  if "$PY" "benchmarks/$name.py" >>"$LOG" 2>&1; then
    echo "--- $name : ok in $((SECONDS - start))s ---" | tee -a "$LOG"
  else
    echo "--- $name : FAILED after $((SECONDS - start))s ---" | tee -a "$LOG"
    failed+=("$name")
  fi
done

for spec in "${MODULE_SCRIPTS[@]}"; do
  name="${spec%% *}"
  echo "--- $name : started $(date +%H:%M:%S) ---" | tee -a "$LOG"
  start=$SECONDS
  # shellcheck disable=SC2086
  if "$PY" -m "benchmarks.${spec%% *}" ${spec#* } >>"$LOG" 2>&1; then
    echo "--- $name : ok in $((SECONDS - start))s ---" | tee -a "$LOG"
  else
    echo "--- $name : FAILED after $((SECONDS - start))s ---" | tee -a "$LOG"
    failed+=("$name")
  fi
done

echo | tee -a "$LOG"
if [ ${#failed[@]} -eq 0 ]; then
  echo "all $(( ${#SCRIPTS[@]} + ${#MODULE_SCRIPTS[@]} )) scripts completed" | tee -a "$LOG"
else
  echo "FAILED: ${failed[*]}" | tee -a "$LOG"
  exit 1
fi
