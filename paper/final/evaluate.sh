#!/bin/bash
# Paper evaluation of the final run matrix (paper/final/matrix.json), CPU only.
# Rollouts reuse the fixed anchor sets (anchors-test: 610 anchors, anchors-val), so every
# family is scored on identical anchors. History ablations are scored by next-event NLL
# only (rollout sampling does not apply the history mask).
#
#   RUNS=runs/final ANCHORS=eval DATA=data/prepared OUT=eval/final WORKERS=28 bash paper/final/evaluate.sh
set -euo pipefail
: "${RUNS:?}" "${ANCHORS:?}" "${DATA:?}" "${OUT:?}"
WORKERS=${WORKERS:-28};PY=${PY:-python};SEEDS="260930 260931 260932"
MATRIX=$(cd "$(dirname "$0")" && pwd)/matrix.json
mkdir -p "$OUT/results"
step(){ echo "$(date -u +%FT%TZ) $*"; }

step "next-event NLL tables"
$PY -m scripts.summarize_runs --matrix "$MATRIX" --runs "$RUNS" --base depth --out "$OUT/results/nll.json"

models(){ for s in $SEEDS; do printf -- "--model %s=%s " "$1$s" "$RUNS/$1-seed$s/training/best.pt"; done; }
rollouts(){  # family split
  local dest="$OUT/rollouts-$2-$1"
  [ -f "$dest/COMPLETE.json" ] && return 0
  step "rollouts $1 $2"
  $PY -m scripts.rollout_eval --anchors "$ANCHORS/anchors-$2" --out "$dest" --data "$DATA" --paths 32 --workers "$WORKERS" $(models "$1")
}
# Most important first: the paper's model, then the reference, then the small model.
for family in depth reference depth-small; do rollouts $family test; done
for family in depth reference depth-small; do rollouts $family val; done

R="$OUT/results"
step "head-to-head comparisons"
$PY -m scripts.compare_rollouts --a "$OUT/rollouts-test-reference" --b "$OUT/rollouts-test-depth" --label-a reference --label-b depth --out "$R/compare-test-depth-vs-reference.json"
$PY -m scripts.compare_rollouts --a "$OUT/rollouts-val-reference" --b "$OUT/rollouts-val-depth" --label-a reference --label-b depth --out "$R/compare-val-depth-vs-reference.json"
$PY -m scripts.compare_rollouts --a "$OUT/rollouts-test-depth" --b "$OUT/rollouts-test-depth-small" --label-a depth --label-b depth-small --out "$R/compare-test-small-vs-depth.json"

for family in depth reference; do
  step "analyses $family"
  $PY -m scripts.calibration --rollouts "$OUT/rollouts-test-$family" --out "$R/calibration-$family.json"
  $PY -m scripts.stylized_facts --rollouts "$OUT/rollouts-test-$family" --out "$R/stylized-facts-$family.json"
  $PY -m scripts.skew_analysis --rollouts "$OUT/rollouts-test-$family" --anchors "$ANCHORS/anchors-test" --out "$R/skew-$family.json"
  $PY -m scripts.path_reweight --fit "$OUT/rollouts-val-$family" --fit-anchors "$ANCHORS/anchors-val" \
      --test "$OUT/rollouts-test-$family" --test-anchors "$ANCHORS/anchors-test" --out "$R/reweight-$family.json"
  $PY -m scripts.swap_diagnostics --anchors "$ANCHORS/anchors-test" --data "$DATA" --model "$RUNS/$family-seed260931/training/best.pt" \
      --out "$R/swap-diagnostics-$family.json" --per-pool 12 --paths 32 --workers "$WORKERS"
done
step DONE
