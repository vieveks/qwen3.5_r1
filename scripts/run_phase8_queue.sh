#!/usr/bin/env bash
# Phase 8 GPU queue: large-heldout evals, unpacked transfer evals, random-reward control.
# Sequential on purpose: one 5070 Ti. Every eval config sets resume: true, so the
# queue can be rerun after an interruption and it will skip completed rows.
set -e
cd "$(dirname "$0")/.."

PACKED_EVALS=(
  configs/eval_phase8_base_all_traces.yaml
  configs/eval_phase8_independent_seed23.yaml
  configs/eval_phase8_iso_lam_0_25_seed23.yaml
  configs/eval_phase8_iso_lam_0_50_seed23.yaml
  configs/eval_phase8_iso_lam_1_00_seed23.yaml
  configs/eval_phase8_independent_seed37.yaml
  configs/eval_phase8_iso_lam_0_50_seed37.yaml
)

UNPACKED_EVALS=(
  configs/eval_phase8_unpacked_base.yaml
  configs/eval_phase8_unpacked_independent_seed23.yaml
  configs/eval_phase8_unpacked_iso_lam_1_00_seed23.yaml
)

run_step() {
  echo "[phase8-queue] $(date '+%Y-%m-%d %H:%M:%S') START $*"
  conda run -n pytorch_5070ti "$@"
  echo "[phase8-queue] $(date '+%Y-%m-%d %H:%M:%S') DONE  $*"
}

for cfg in "${PACKED_EVALS[@]}"; do
  run_step python -m iso_rlvr.eval.run_packed_eval --config "$cfg"
done

for cfg in "${UNPACKED_EVALS[@]}"; do
  run_step python -m iso_rlvr.eval.run_packed_eval --config "$cfg"
done

run_step python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_random_30step.yaml
run_step python -m iso_rlvr.eval.run_packed_eval --config configs/eval_phase8_random_seed23.yaml

echo "[phase8-queue] $(date '+%Y-%m-%d %H:%M:%S') QUEUE COMPLETE"
