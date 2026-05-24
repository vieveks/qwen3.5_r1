# Phase 1: Baseline Harness And Dataset Calibration

Date: 2026-05-24

## Executive Summary

Phase 1 established the local baseline harness, downloaded and evaluated `Qwen/Qwen2.5-Math-1.5B`, and calibrated the procedural dataset until the project had a credible first RL target.

The core result is:

```text
Use Dataset V4: data/iso_math_calibrated.jsonl
Baseline accuracy: 0.5625
Baseline family_accuracy: 0.25
Instance-family gap: 0.3125
Format failure rate: 0.0
```

This is the first dataset version that fits the intended experimental regime. The base model can solve many individual variants, but it fails much more often when all variants in the same latent family must be correct. That gap is the behavior Iso-RLVR is designed to improve.

Do not use the earlier datasets for real conclusions:

- V1 was too easy.
- V2 had a better family signal but was still too easy at instance level.
- V3 was too hard and slow.

## What We Are Testing

The project is not just asking whether reward training improves math accuracy. It is asking whether a family-aware reward changes what kind of behavior the model learns.

Standard RLVR asks:

```text
Did this one answer match the verifier?
```

Iso-RLVR asks:

```text
Did the model solve the latent rule well enough to stay correct across equivalent variants?
```

The procedural dataset is therefore central to the experiment. Every row belongs to a family, and each family contains several variants that share the same hidden solution structure. A model that learns a brittle shortcut may get some individual rows right but fail the family.

## Goal

Phase 1 asks whether the current procedural dataset is suitable for testing Iso-RLVR before any real training.

The useful first-training range from the README is:

```text
accuracy: 0.30 to 0.75
family_accuracy: meaningfully lower than accuracy
```

The key signal is a gap between instance-level correctness and family-level correctness. Iso-RLVR only has room to help if the model can solve some individual variants but fails to stay correct across equivalent variants.

## Artifact Ledger

Tracked files added or updated in this phase:

```text
phase1.md
README.md
PLAN.md
configs/baseline_eval_harder.yaml
configs/baseline_eval_challenge.yaml
configs/baseline_eval_challenge_quick.yaml
configs/baseline_eval_calibrated.yaml
src/iso_rlvr/data/build_dataset.py
src/iso_rlvr/data/families.py
tests/test_dataset_profiles.py
```

Ignored/generated artifacts from local runs:

```text
data/iso_math_small.jsonl
data/iso_math_harder.jsonl
data/iso_math_challenge.jsonl
data/iso_math_calibrated.jsonl
outputs/eval/*.jsonl
outputs/eval/*.summary.json
outputs/runs/iso_grpo_lite_smoke/
```

These outputs are intentionally ignored by git, but the commands and summaries needed to reproduce them are recorded below.

## Environment

- Conda env: `pytorch_5070ti`
- Python: `C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- CUDA visible to PyTorch: yes
- Model: `Qwen/Qwen2.5-Math-1.5B`

Note: `pip install -e .` could not write to the Python user-site path from this session. The repo was run with `src` on `sys.path`. `pytest` was installed only into `C:\tmp\qwen_pytest_deps` for test execution.

## Tests

Initial reward tests:

```text
4 passed in 0.02s
```

After adding dataset profiles and hard families:

```text
7 passed in 0.02s
```

After adding `challenge` and `calibrated` profiles:

```text
9 passed in 0.02s
```

## Baseline Dataset V1

Command:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_small.jsonl --families 200 --variants 4 --seed 7
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval.yaml
```

Dataset:

- Rows: 800
- Families: 200
- Variants per family: 4
- Profile: original easy/mixed generator at the time
- Evaluation cap: first 100 rows

Summary:

```json
{
  "accuracy": 0.86,
  "avg_tokens": 204.19,
  "avg_wrong_tokens": 512.0,
  "family_accuracy": 0.8,
  "format_failure_rate": 0.0
}
```

Conclusion:

The dataset is too easy by the README threshold because accuracy is above `0.85`. Do not use this dataset for a real RL comparison.

## Smoke Test

A one-step mechanics smoke test was run with a temporary config:

- Config: `outputs/train_iso_grpo_smoke.yaml`
- Output: `outputs/runs/iso_grpo_lite_smoke/`
- LoRA adapter saved: yes
- Train log written: yes

Smoke metrics:

```json
{
  "accuracy": 0.0,
  "avg_tokens": 64.0,
  "avg_wrong_tokens": 64.0,
  "family_accuracy": 0.0,
  "format_failure_rate": 0.0,
  "loss": 0.0,
  "max_reward": 0.0,
  "mean_reward": 0.0,
  "min_reward": 0.0,
  "step": 0
}
```

Conclusion:

This only verified that the trainer path can run and save an adapter. It was not a meaningful learning test because the dataset was already judged too easy and the smoke batch had all-zero rewards.

## Dataset V2: Harder Procedural Families

Implemented explicit dataset profiles:

- `easy`: original simple families
- `mixed`: simple plus hard families
- `harder`: hard families only

New hard families:

- `rational_linear_equation`
- `nested_linear_equation`
- `two_variable_system`
- `quadratic_root`

Generation command:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_harder.jsonl --families 200 --variants 4 --seed 11 --profile harder
```

Dataset:

- Rows: 800
- Families: 200
- Variants per family: 4

Family-type mix:

```text
rational_linear_equation: 220 rows
nested_linear_equation: 216 rows
two_variable_system: 188 rows
quadratic_root: 176 rows
```

Baseline command:

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_harder.yaml
```

Summary:

```json
{
  "accuracy": 0.87,
  "avg_tokens": 230.29,
  "avg_wrong_tokens": 438.9230769230769,
  "family_accuracy": 0.6,
  "format_failure_rate": 0.0
}
```

Per-family-type accuracy over the first 100 evaluated rows:

```text
nested_linear_equation: 36 rows, accuracy 0.972, avg_tokens 139.4
quadratic_root: 12 rows, accuracy 1.000, avg_tokens 351.5
rational_linear_equation: 40 rows, accuracy 0.725, avg_tokens 247.7
two_variable_system: 12 rows, accuracy 0.917, avg_tokens 323.7
families evaluated: 25
whole-family correct: 15
```

Conclusion:

Dataset V2 improved the family-level signal but did not solve the difficulty problem. Single-instance accuracy is still too high at `0.87`. The useful result is that `family_accuracy` dropped to `0.60`, which means the Iso-RLVR target failure mode is present: many individual variants are solved, but whole families are not consistently solved.

## Field Scan

Relevant dataset lessons:

- MATH introduced competition-style math problems where most problems are not simple direct applications. It is useful as a hard evaluation reference, but it does not provide isomorphic variant families out of the box. Source: https://arxiv.org/abs/2103.03874
- DeepMath-103K is explicitly positioned as a large-scale, challenging, decontaminated, verifiable math dataset for reasoning/RL. It is a better external candidate than GSM8K for hard verifiable math, but it still needs family construction or transformation metadata for Iso-RLVR. Source: https://arxiv.org/abs/2504.11456
- Reasoning Gym argues for procedural environments with configurable difficulty and automatic verification for RLVR. That matches this project more directly than a static dataset because Iso-RLVR needs controlled variants and exact rewards. Source: https://github.com/open-thought/reasoning-gym
- DeepScaleR reports strong 1.5B math RL results from roughly 40K unique problem-answer pairs assembled from contest-style math sources. That supports using harder contest-like tasks once the reward harness is stable. Source: https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview

Interpretation:

The field trend for RLVR is toward verifiable math problems, automatic checking, and controlled difficulty. For this project, the important extra requirement is family structure. A raw benchmark can tell us whether a model is good at math, but it cannot directly tell us whether Iso-RLVR improved rule-preserving behavior unless the data carries explicit variant relations.

## Dataset Recommendation

For this project, the primary training dataset should remain procedural and isomorphic, not a raw external dataset.

Reason:

Iso-RLVR is testing a family-level reward. We need each sample to carry:

- `family_id`
- `variant_id`
- latent rule or invariant
- exact answer
- transformation metadata
- controlled difficulty

Static datasets like MATH or DeepMath-103K are valuable, but they mostly provide independent problems. They should be used as:

- calibration references for difficulty
- held-out evaluation sets
- sources for designing harder family templates
- later seed problems that we transform into variant families

Immediate next dataset target:

Create Dataset V3 with fewer easy algebra templates and more tasks where the answer requires 2-4 dependent operations:

- rational systems asking for `x + y`, `x - y`, or `2x + y`
- multi-step affine word problems with distractor quantities
- percentage/ratio mixture problems
- modular arithmetic with two constraints, not single remainders
- quadratic/factorization tasks where the requested invariant changes by family
- symbolic simplification/evaluation with a hidden substitution

Acceptance target:

```text
accuracy: 0.45 to 0.75
family_accuracy: at least 0.15 lower than accuracy
format_failure_rate: under 0.05
```

If Dataset V3 still scores above `0.85`, use external hard-source templates from MATH or DeepMath-style contest problems and procedurally transform them into families rather than training on them directly.

## Dataset V3: Challenge Profile

Implemented `challenge` profile with:

- `rational_system_target`
- `chinese_remainder`
- `missing_average`
- `affine_composition`

The full 100-example challenge eval timed out at 15 minutes with `max_new_tokens: 512`. A shorter calibration was run instead:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_challenge.jsonl --families 200 --variants 4 --seed 19 --profile challenge
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_challenge_quick.yaml
```

Summary:

```json
{
  "accuracy": 0.05,
  "avg_tokens": 248.65,
  "avg_wrong_tokens": 252.28947368421052,
  "family_accuracy": 0.0,
  "format_failure_rate": 0.0
}
```

Per-family-type accuracy over 40 evaluated rows:

```text
affine_composition: 12 rows, accuracy 0.000, avg_tokens 256.0
chinese_remainder: 12 rows, accuracy 0.000, avg_tokens 251.4
missing_average: 4 rows, accuracy 0.500, avg_tokens 196.2
rational_system_target: 12 rows, accuracy 0.000, avg_tokens 256.0
families evaluated: 10
whole-family correct: 0
```

Conclusion:

Dataset V3 is too hard for first RL. It is useful as a later held-out stress set, not as the initial training set. The two hardest families also tend to hit the generation cap, so they should be used sparingly until prompting or max-token handling improves.

## Dataset V4: Calibrated Profile

Implemented `calibrated` profile with weighted family sampling:

- 50% `rational_linear_equation`
- 30% `missing_average`
- 10% `rational_system_target`
- 10% `chinese_remainder`

Generation and eval:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated.yaml
```

Dataset:

- Rows: 800
- Families: 200
- Variants per family: 4

Family-type mix:

```text
rational_linear_equation: 408 rows
missing_average: 232 rows
rational_system_target: 96 rows
chinese_remainder: 64 rows
```

Summary over 80 evaluated rows:

```json
{
  "accuracy": 0.5625,
  "avg_tokens": 208.5625,
  "avg_wrong_tokens": 238.4857142857143,
  "family_accuracy": 0.25,
  "format_failure_rate": 0.0
}
```

Per-family-type accuracy:

```text
chinese_remainder: 8 rows, accuracy 0.000, avg_tokens 227.4
missing_average: 32 rows, accuracy 0.719, avg_tokens 205.4
rational_linear_equation: 32 rows, accuracy 0.688, avg_tokens 195.1
rational_system_target: 8 rows, accuracy 0.000, avg_tokens 256.0
families evaluated: 20
whole-family correct: 5
```

Conclusion:

Dataset V4 is the first credible candidate for real Iso-RLVR testing. It lands inside the target accuracy range and has a large instance-vs-family gap:

```text
accuracy: 0.5625
family_accuracy: 0.25
gap: 0.3125
```

Use `data/iso_math_calibrated.jsonl` for the next small control experiments.

Recommended next matrix:

```text
Base calibrated eval
Independent correctness reward, tiny run
Iso reward lambda_iso = 0.25, tiny run
Iso reward lambda_iso = 0.50, tiny run
Held-out calibrated eval with a different seed
Challenge profile eval as stress test only
```

## Current Dataset Decision

Use the calibrated procedural dataset for the first meaningful training experiments:

```text
dataset_path: data/iso_math_calibrated.jsonl
generation profile: calibrated
seed: 29
baseline config: configs/baseline_eval_calibrated.yaml
```

Do not use:

- `iso_math_small.jsonl` for training conclusions, because it is too easy.
- `iso_math_harder.jsonl` for training conclusions, because single-instance accuracy is still too high.
- `iso_math_challenge.jsonl` as the first RL dataset, because it is too hard and slow.

## Reproducible Command Set

Environment check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Run tests:

```bash
python -m pytest tests
```

Generate calibrated dataset:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
```

Run calibrated baseline:

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated.yaml
```

Read calibrated summary:

```bash
python -c "import json; print(json.dumps(json.load(open('outputs/eval/baseline_qwen25_math_1_5b_calibrated.summary.json')), indent=2))"
```

## Next Work

The next phase should move from dataset calibration to controlled training comparisons.

Recommended sequence:

1. Add or confirm an independent-correctness training config that uses `data/iso_math_calibrated.jsonl`.
2. Add Iso-RLVR configs for `lambda_iso = 0.25`, `0.50`, and `1.00`.
3. Keep prompt, model, max tokens, training steps, and dataset fixed across the first comparison.
4. Generate a held-out calibrated eval set with a different seed.
5. Evaluate base, independent reward, and Iso-RLVR adapters on the same held-out set.
6. Use the challenge profile only as a later stress test.

The next report should include:

- baseline calibrated metrics
- independent reward metrics
- Iso-RLVR metrics for each lambda
- family-type breakdown
- wrong-answer response length
- format failure rate
- examples where instance accuracy is correct but family consistency fails
- examples where Iso-RLVR improves held-out family consistency
