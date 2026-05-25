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
