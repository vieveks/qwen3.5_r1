# Phase 2: Controlled Training Start

Date: 2026-05-24

## Objective

Phase 2 starts the first controlled training comparison on the calibrated procedural dataset from Phase 1.

The question is:

```text
Does Iso-RLVR improve held-out family consistency compared with independent correctness reward, without destroying single-instance accuracy?
```

## Starting Point

Phase 1 selected the calibrated dataset:

```text
training dataset: data/iso_math_calibrated.jsonl
profile: calibrated
seed: 29
baseline accuracy: 0.5625
baseline family_accuracy: 0.25
format_failure_rate: 0.0
```

This is the first useful training target because the model solves many individual problems but fails whole families much more often.

## Guardrails

Keep these fixed across the first comparison:

- Model: `Qwen/Qwen2.5-Math-1.5B`
- Training dataset: `data/iso_math_calibrated.jsonl`
- Prompt template: same as baseline eval
- LoRA config: `r=16`, `alpha=32`, `dropout=0.05`
- Rollout shape: same `families_per_step`, `variants_per_family`, and `samples_per_variant`
- Evaluation set: held-out calibrated dataset with a different seed

Do not compare training rewards directly as the main result. The main result is held-out evaluation.

## Held-Out Dataset

Generate held-out calibrated data:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated_heldout.jsonl --families 200 --variants 4 --seed 31 --profile calibrated
```

Baseline held-out eval config:

```text
configs/baseline_eval_calibrated_heldout.yaml
```

Run baseline held-out eval:

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated_heldout.yaml
```

## Training Configs

Smoke configs:

```text
configs/train_independent_calibrated_smoke.yaml
configs/train_iso_calibrated_smoke.yaml
```

The smoke configs intentionally use only three steps. The goal is to verify:

- rollout generation works on calibrated data
- rewards are non-constant at least some of the time
- loss is finite
- adapter saves
- logs include reward mode and metrics

## Reward Modes

The trainer now supports explicit reward modes:

```yaml
reward_mode: independent
```

uses:

```text
reward_i = 1 if answer_i is correct else 0
```

and:

```yaml
reward_mode: iso
lambda_iso: 0.5
```

uses:

```text
reward_i = correctness_i + lambda_iso * family_consistency
```

where strict `family_consistency` is `1` only when every sampled variant in the family is correct.

## Initial Run Order

1. Generate held-out calibrated dataset. Done.
2. Run tests. Done.
3. Start independent reward smoke training. Done.
4. Inspect independent smoke log. Done.
5. Start Iso-RLVR smoke training. Done.
6. Inspect Iso smoke log. Done.
7. If both are healthy, scale to matched longer runs. Pending.

## Health Criteria

A smoke run is healthy if:

```text
adapter saved: yes
train_log.jsonl written: yes
loss: finite
format_failure_rate: below 0.10
reward_mode: correct in log
```

Reward may be sparse in a three-step smoke run, especially on difficult families. If rewards are all zero, rerun with a different rollout seed or slightly longer smoke before drawing conclusions.

## Next Report

The next report should include:

- held-out base metrics
- independent smoke metrics
- Iso smoke metrics
- whether rewards were non-constant
- whether both adapters saved
- recommendation for longer matched runs

## Run Log: 2026-05-24

Held-out calibrated dataset was generated:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated_heldout.jsonl --families 200 --variants 4 --seed 31 --profile calibrated
```

Test suite:

```text
12 passed in 5.20s
```

### Independent Reward Smoke

Command:

```bash
python -m iso_rlvr.train.grpo_lite --config configs/train_independent_calibrated_smoke.yaml
```

Output:

```text
outputs/runs/independent_calibrated_smoke/
outputs/runs/independent_calibrated_smoke/adapter_or_model/adapter_model.safetensors
outputs/runs/independent_calibrated_smoke/train_log.jsonl
```

Metrics:

```json
{"step": 0, "reward_mode": "independent", "accuracy": 0.0, "family_accuracy": 0.0, "mean_reward": 0.0, "min_reward": 0.0, "max_reward": 0.0, "loss": 0.0, "format_failure_rate": 0.0, "avg_tokens": 128.0}
{"step": 1, "reward_mode": "independent", "accuracy": 0.125, "family_accuracy": 0.0, "mean_reward": 0.125, "min_reward": 0.0, "max_reward": 1.0, "loss": -0.03206607699394226, "format_failure_rate": 0.0, "avg_tokens": 128.0}
{"step": 2, "reward_mode": "independent", "accuracy": 0.0, "family_accuracy": 0.0, "mean_reward": 0.0, "min_reward": 0.0, "max_reward": 0.0, "loss": 0.0, "format_failure_rate": 0.0, "avg_tokens": 128.0}
```

Status:

```text
adapter saved: yes
train_log.jsonl written: yes
reward_mode logged correctly: yes
nonzero rewards observed: yes, step 1
finite loss observed: yes
```

### Iso-RLVR Smoke

Command:

```bash
python -m iso_rlvr.train.grpo_lite --config configs/train_iso_calibrated_smoke.yaml
```

Output:

```text
outputs/runs/iso_calibrated_smoke_lam_0_50/
outputs/runs/iso_calibrated_smoke_lam_0_50/adapter_or_model/adapter_model.safetensors
outputs/runs/iso_calibrated_smoke_lam_0_50/train_log.jsonl
```

Metrics:

```json
{"step": 0, "reward_mode": "iso", "accuracy": 0.125, "family_accuracy": 0.0, "mean_reward": 0.125, "min_reward": 0.0, "max_reward": 1.0, "loss": -0.07363663613796234, "format_failure_rate": 0.0, "avg_tokens": 128.0}
{"step": 1, "reward_mode": "iso", "accuracy": 0.25, "family_accuracy": 0.0, "mean_reward": 0.25, "min_reward": 0.0, "max_reward": 1.0, "loss": -0.04359022527933121, "format_failure_rate": 0.0, "avg_tokens": 127.25}
{"step": 2, "reward_mode": "iso", "accuracy": 0.0, "family_accuracy": 0.0, "mean_reward": 0.0, "min_reward": 0.0, "max_reward": 0.0, "loss": 0.0, "format_failure_rate": 0.0, "avg_tokens": 128.0}
```

Status:

```text
adapter saved: yes
train_log.jsonl written: yes
reward_mode logged correctly: yes
nonzero rewards observed: yes, steps 0 and 1
finite loss observed: yes
```

## Smoke Conclusion

The training path is now live for both reward modes.

Important caveat:

The average token count is at or near the `128` token cap in both smoke runs. This means the model is often still reasoning when generation stops. The smoke runs are valid as mechanics tests, but not as evidence about policy quality.

Before a longer matched run, increase `max_new_tokens` to at least `256` and keep that value fixed across:

- independent correctness reward
- Iso-RLVR `lambda_iso = 0.25`
- Iso-RLVR `lambda_iso = 0.50`
- Iso-RLVR `lambda_iso = 1.00`

Recommended immediate next step:

Run held-out base eval on `data/iso_math_calibrated_heldout.jsonl`, then create longer matched training configs with `max_new_tokens: 256`.

## Run Log: 2026-05-25

Phase 2 scaffold was committed and pushed:

```text
commit: 2192f27 Add phase 2 training scaffold
remote: origin/main
```

The package was installed editable in the `pytorch_5070ti` conda environment:

```bash
conda run -n pytorch_5070ti python -m pip install -e .
```

`pytest` was also installed into the same conda environment so the README test command can run directly.

Test suite:

```text
12 passed in 5.15s
```

Pytest emitted a cache warning because it could not write one `.pytest_cache` path, but the tests completed successfully.

### Held-Out Base Eval

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated_heldout.yaml
```

Result:

```json
{
  "accuracy": 0.6,
  "avg_tokens": 209.325,
  "avg_wrong_tokens": 249.90625,
  "family_accuracy": 0.25,
  "format_failure_rate": 0.0
}
```

Output files:

```text
outputs/eval/baseline_qwen25_math_1_5b_calibrated_heldout.jsonl
outputs/eval/baseline_qwen25_math_1_5b_calibrated_heldout.summary.json
```

Conclusion:

The held-out split is still in the target range. Single-instance accuracy is moderate at `0.60`, while strict family accuracy remains low at `0.25`. This keeps the central Iso-RLVR question meaningful: can training improve whole-family consistency without simply overfitting individual variants?

The token-cap issue is also confirmed. Average completion length is `209.325`, and wrong completions average `249.90625` tokens under a `256` token cap. Longer training runs should use `max_new_tokens: 256` at minimum and should track whether wrong answers are still saturating that cap.

### Matched 20-Step Configs

Training configs:

```text
configs/train_independent_calibrated_20step.yaml
configs/train_iso_calibrated_lam_0_25_20step.yaml
configs/train_iso_calibrated_lam_0_50_20step.yaml
configs/train_iso_calibrated_lam_1_00_20step.yaml
```

Adapter eval configs:

```text
configs/eval_independent_calibrated_20step.yaml
configs/eval_iso_calibrated_lam_0_25_20step.yaml
configs/eval_iso_calibrated_lam_0_50_20step.yaml
configs/eval_iso_calibrated_lam_1_00_20step.yaml
```

Eval now accepts an optional `adapter_path`, so saved LoRA adapters can be evaluated against the same held-out set.

### Independent 20-Step Control

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_independent_calibrated_20step.yaml
```

Runtime:

```text
about 1h35m
```

Output:

```text
outputs/runs/independent_calibrated_20step/
outputs/runs/independent_calibrated_20step/adapter_or_model/adapter_model.safetensors
outputs/runs/independent_calibrated_20step/train_log.jsonl
```

Training-log aggregate:

```json
{
  "steps": 20,
  "accuracy": {"first": 0.375, "last": 0.125, "mean": 0.38125, "min": 0.125, "max": 0.75},
  "family_accuracy": {"first": 0.0, "last": 0.0, "mean": 0.05, "min": 0.0, "max": 0.5},
  "mean_reward": {"first": 0.375, "last": 0.125, "mean": 0.38125, "min": 0.125, "max": 0.75},
  "loss": {"first": -0.044440969824790955, "last": 0.13566374778747559, "mean": -0.037719862163066865, "min": -0.14888335764408112, "max": 0.13566374778747559},
  "avg_tokens": {"first": 215.875, "last": 246.75, "mean": 214.225, "min": 151.5, "max": 246.75},
  "avg_wrong_tokens": {"first": 235.8, "last": 256.0, "mean": 236.4080952380952, "min": 167.0, "max": 256.0},
  "format_failure_rate": {"first": 0.0, "last": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
}
```

Held-out adapter eval:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_independent_calibrated_20step.yaml
```

```json
{
  "accuracy": 0.6125,
  "avg_tokens": 204.85,
  "avg_wrong_tokens": 241.25806451612902,
  "family_accuracy": 0.25,
  "format_failure_rate": 0.0
}
```

Conclusion:

The independent control gives a tiny single-instance improvement over the base held-out score (`0.60` to `0.6125`) but no strict family-accuracy improvement (`0.25` to `0.25`).

### Iso 20-Step Run, Lambda 0.50

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_iso_calibrated_lam_0_50_20step.yaml
```

Runtime:

```text
about 35m34s
```

Output:

```text
outputs/runs/iso_calibrated_lam_0_50_20step/
outputs/runs/iso_calibrated_lam_0_50_20step/adapter_or_model/adapter_model.safetensors
outputs/runs/iso_calibrated_lam_0_50_20step/train_log.jsonl
```

Training-log aggregate:

```json
{
  "steps": 20,
  "accuracy": {"first": 0.125, "last": 0.625, "mean": 0.36875, "min": 0.0, "max": 0.75},
  "family_accuracy": {"first": 0.0, "last": 0.0, "mean": 0.075, "min": 0.0, "max": 0.5},
  "mean_reward": {"first": 0.125, "last": 0.625, "mean": 0.40625, "min": 0.0, "max": 1.0},
  "loss": {"first": -0.066669762134552, "last": 0.10743840038776398, "mean": -0.04517454393208027, "min": -0.3075815737247467, "max": 0.10743840038776398},
  "avg_tokens": {"first": 184.25, "last": 187.625, "mean": 202.73125, "min": 160.25, "max": 252.625},
  "avg_wrong_tokens": {"first": 189.28571428571428, "last": 250.0, "mean": 225.81845238095238, "min": 189.28571428571428, "max": 256.0},
  "format_failure_rate": {"first": 0.0, "last": 0.0, "mean": 0.00625, "min": 0.0, "max": 0.125}
}
```

Held-out adapter eval:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_iso_calibrated_lam_0_50_20step.yaml
```

```json
{
  "accuracy": 0.625,
  "avg_tokens": 208.7875,
  "avg_wrong_tokens": 250.96666666666667,
  "family_accuracy": 0.3,
  "format_failure_rate": 0.0
}
```

### First Controlled Pair Conclusion

Held-out comparison:

| Run | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens | Format failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.6000 | 0.2500 | 209.3250 | 249.9063 | 0.0000 |
| Independent 20-step | 0.6125 | 0.2500 | 204.8500 | 241.2581 | 0.0000 |
| Iso 20-step, lambda 0.50 | 0.6250 | 0.3000 | 208.7875 | 250.9667 | 0.0000 |

This is not enough evidence for a strong claim, but it is the first positive signal:

- Independent reward slightly improves single-instance accuracy but does not move strict family accuracy.
- Iso-RLVR with `lambda_iso = 0.50` improves both single-instance accuracy and strict family accuracy on the held-out calibrated split.
- The family-accuracy gain is small (`+0.05` absolute over base and independent), so it needs replication before we trust it.
- Wrong answers still often saturate the `256` token cap, especially in the Iso held-out eval. Generation length remains a confound.

Decision:

Do not run the full lambda sweep blindly yet. The next best step is to replicate the matched pair with a second training seed or add explicit training/eval seed controls, then run `lambda_iso = 0.25` and `1.00` only if the replicated `0.50` result remains directionally positive.

## Replication Plan: Seed 23

Date: 2026-05-25

The trainer now accepts an explicit `seed` field. The seed controls:

- Python random sampling for training families
- Torch generation sampling
- CUDA RNG state when CUDA is available

The seed is also written into every `train_log.jsonl` metrics record.

Second-seed matched pair:

```text
configs/train_independent_calibrated_20step_seed_23.yaml
configs/train_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

Held-out eval configs:

```text
configs/eval_independent_calibrated_20step_seed_23.yaml
configs/eval_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

Protocol:

1. Run tests after seed-control change.
2. Train independent seed 23.
3. Evaluate independent seed 23 adapter on the same held-out calibrated split.
4. Train Iso `lambda_iso = 0.50` seed 23.
5. Evaluate Iso seed 23 adapter on the same held-out calibrated split.
6. Compare against the first controlled pair.

Success criterion for continuing to the lambda sweep:

Iso should again improve strict family accuracy relative to the matched independent run, without losing single-instance accuracy by more than a small amount.

### Seed 23 Independent Run

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_independent_calibrated_20step_seed_23.yaml
```

Runtime:

```text
about 1h43m
```

Output:

```text
outputs/runs/independent_calibrated_20step_seed_23/
outputs/runs/independent_calibrated_20step_seed_23/adapter_or_model/adapter_model.safetensors
outputs/runs/independent_calibrated_20step_seed_23/train_log.jsonl
```

Training-log aggregate:

```json
{
  "steps": 20,
  "accuracy": {"first": 0.0, "last": 0.625, "mean": 0.36875, "min": 0.0, "max": 0.875},
  "family_accuracy": {"first": 0.0, "last": 0.5, "mean": 0.1, "min": 0.0, "max": 0.5},
  "mean_reward": {"first": 0.0, "last": 0.625, "mean": 0.36875, "min": 0.0, "max": 0.875},
  "loss": {"first": 0.0, "last": 0.081861212849617, "mean": -0.027759448532015084, "min": -0.1388397216796875, "max": 0.081861212849617},
  "avg_tokens": {"first": 219.5, "last": 216.25, "mean": 212.89375, "min": 182.5, "max": 253.625},
  "avg_wrong_tokens": {"first": 219.5, "last": 256.0, "mean": 230.4213095238095, "min": 190.0, "max": 256.0},
  "format_failure_rate": {"first": 0.0, "last": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
}
```

Held-out adapter eval:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_independent_calibrated_20step_seed_23.yaml
```

```json
{
  "accuracy": 0.6375,
  "avg_tokens": 206.4,
  "avg_wrong_tokens": 249.3448275862069,
  "family_accuracy": 0.3,
  "format_failure_rate": 0.0
}
```

### Seed 23 Iso Run, Lambda 0.50

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

Runtime:

```text
about 35m17s
```

Output:

```text
outputs/runs/iso_calibrated_lam_0_50_20step_seed_23/
outputs/runs/iso_calibrated_lam_0_50_20step_seed_23/adapter_or_model/adapter_model.safetensors
outputs/runs/iso_calibrated_lam_0_50_20step_seed_23/train_log.jsonl
```

Training-log aggregate:

```json
{
  "steps": 20,
  "accuracy": {"first": 0.0, "last": 0.25, "mean": 0.3375, "min": 0.0, "max": 0.875},
  "family_accuracy": {"first": 0.0, "last": 0.0, "mean": 0.075, "min": 0.0, "max": 0.5},
  "mean_reward": {"first": 0.0, "last": 0.25, "mean": 0.375, "min": 0.0, "max": 1.125},
  "loss": {"first": 0.0, "last": 0.005100801587104797, "mean": -0.024954184237867594, "min": -0.1301363706588745, "max": 0.03617249056696892},
  "avg_tokens": {"first": 219.5, "last": 234.625, "mean": 213.225, "min": 163.75, "max": 256.0},
  "avg_wrong_tokens": {"first": 219.5, "last": 256.0, "mean": 232.69125, "min": 187.4, "max": 256.0},
  "format_failure_rate": {"first": 0.0, "last": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
}
```

Held-out adapter eval:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

```json
{
  "accuracy": 0.6375,
  "avg_tokens": 206.2125,
  "avg_wrong_tokens": 248.51724137931035,
  "family_accuracy": 0.35,
  "format_failure_rate": 0.0
}
```

### Replication Conclusion

Held-out comparison after adding explicit seed control:

| Run | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens | Format failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.6000 | 0.2500 | 209.3250 | 249.9063 | 0.0000 |
| Independent 20-step, seed 13 | 0.6125 | 0.2500 | 204.8500 | 241.2581 | 0.0000 |
| Iso 20-step, lambda 0.50, seed 13 | 0.6250 | 0.3000 | 208.7875 | 250.9667 | 0.0000 |
| Independent 20-step, seed 23 | 0.6375 | 0.3000 | 206.4000 | 249.3448 | 0.0000 |
| Iso 20-step, lambda 0.50, seed 23 | 0.6375 | 0.3500 | 206.2125 | 248.5172 | 0.0000 |

The seed 23 replicate confirms the direction of the first controlled pair:

- Iso `lambda_iso = 0.50` again improves strict family accuracy over the matched independent control by `+0.05` absolute.
- In seed 23, Iso matches independent single-instance accuracy exactly (`0.6375` vs `0.6375`).
- Across both matched pairs, Iso improves family accuracy by `+0.05` without reducing held-out accuracy.
- The result is still small and needs more seeds or a larger evaluation set before making a strong claim.

Updated decision:

The next useful compute is now a lambda sweep at the same 20-step scale for `lambda_iso = 0.25` and `lambda_iso = 1.00`, preferably with explicit seeds and the same held-out eval. Because training cost is high, run one lambda at a time and stop if held-out family accuracy regresses.
