# Critic Brief: Iso-RLVR Phase 1 And Phase 2

Date: 2026-05-25

This document is the compact review packet for the current experiment state. It summarizes what was built, what was run, the model and code architecture, dataset details, current evidence, and the main reasons a critic should treat the result as early rather than conclusive.

## Executive Summary

The project tests whether isomorphic reward shaping can improve reasoning consistency across transformed variants of the same math problem. The control condition rewards each answer independently. The Iso-RLVR condition adds a family-level bonus only when all sampled variants from the same latent family are correct.

Current evidence:

| Comparison | Accuracy | Family accuracy | Conclusion |
| --- | ---: | ---: | --- |
| Base held-out | 0.6000 | 0.2500 | The held-out set is in a useful difficulty range. |
| Independent, seed 13 | 0.6125 | 0.2500 | Slight accuracy gain, no family-consistency gain. |
| Iso `lambda_iso=0.50`, seed 13 | 0.6250 | 0.3000 | Improves family accuracy by `+0.05` over matched independent. |
| Independent, seed 23 | 0.6375 | 0.3000 | Stronger independent run, family accuracy also improves. |
| Iso `lambda_iso=0.50`, seed 23 | 0.6375 | 0.3500 | Matches accuracy and again improves family accuracy by `+0.05`. |

Current conclusion:

```text
Iso-RLVR lambda 0.50 has a small replicated positive signal on held-out family accuracy,
without reducing held-out single-instance accuracy, across two matched 20-step runs.
```

This is not yet a strong result. The evaluation is small, the trainer is intentionally minimal, and the dataset is procedural. The result is strong enough to justify the next experiment, not strong enough to make a general claim.

## Research Question

The baseline RLVR question is:

```text
Did the model produce the correct final answer for this problem?
```

The Iso-RLVR question is:

```text
Did the model learn a rule that survives equivalent transformations of the same latent problem?
```

The working hypothesis is that outcome-only reward can reinforce brittle answer patterns, while an isomorphic reward should prefer policies that remain correct across variants generated from the same hidden structure.

## Model And Runtime

Primary model:

```text
Qwen/Qwen2.5-Math-1.5B
```

Loading path:

- `src/iso_rlvr/modeling.py`
- `transformers.AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `torch.bfloat16` on CUDA, `torch.float32` on CPU
- `device_map="auto"` when CUDA is available
- tokenizer `pad_token` is set to `eos_token` if missing

Training uses LoRA adapters rather than full model fine-tuning:

```text
r: 16
alpha: 32
dropout: 0.05
bias: none
task_type: CAUSAL_LM
target_modules:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Local runtime used for the current runs:

```text
machine: 5070 Ti PC
conda env: pytorch_5070ti
install mode: pip install -e .
```

The exact package dependencies are in `pyproject.toml`. The core stack is `torch`, `transformers`, `accelerate`, `peft`, `safetensors`, `pyyaml`, `numpy`, `datasets`, and `tqdm`.

Model artifacts produced so far:

| Run | Adapter or output path |
| --- | --- |
| Independent smoke | `outputs/runs/independent_calibrated_smoke/adapter_or_model` |
| Iso smoke, `lambda_iso=0.50` | `outputs/runs/iso_calibrated_smoke_lam_0_50/adapter_or_model` |
| Independent 20-step, seed 13 | `outputs/runs/independent_calibrated_20step/adapter_or_model` |
| Iso 20-step, `lambda_iso=0.50`, seed 13 | `outputs/runs/iso_calibrated_lam_0_50_20step/adapter_or_model` |
| Independent 20-step, seed 23 | `outputs/runs/independent_calibrated_20step_seed_23/adapter_or_model` |
| Iso 20-step, `lambda_iso=0.50`, seed 23 | `outputs/runs/iso_calibrated_lam_0_50_20step_seed_23/adapter_or_model` |

## Code Architecture

The project is intentionally small so the experimental contract is easy to audit.

| Area | File | Responsibility |
| --- | --- | --- |
| Dataset family generation | `src/iso_rlvr/data/families.py` | Builds procedural math families and variants from hidden parameters. |
| Dataset writing | `src/iso_rlvr/data/build_dataset.py` | Writes JSONL rows from generated `ProblemVariant` objects. |
| Model loading | `src/iso_rlvr/modeling.py` | Loads tokenizer/model, resolves CUDA/CPU, counts completion tokens. |
| Answer verifier | `src/iso_rlvr/rewards/answer.py` | Extracts final answer and compares normalized numeric values with `Fraction`. |
| Iso reward | `src/iso_rlvr/rewards/iso.py` | Computes strict family consistency, Iso rewards, and aggregate metrics. |
| Trainer | `src/iso_rlvr/train/grpo_lite.py` | Runs rollout sampling, reward computation, normalized-advantage loss, LoRA updates, and logs. |
| Evaluation | `src/iso_rlvr/eval/run_eval.py` | Runs deterministic held-out eval with optional LoRA adapter loading. |
| Experiment configs | `configs/*.yaml` | Pins model, dataset, reward mode, generation settings, adapter paths, and output paths. |

Important implementation details:

- `grpo_lite.py` is a GRPO-like local scaffold, not a production TRL/verl GRPO implementation.
- It samples families, generates completions, scores final answers, standardizes rewards into advantages, and optimizes mean completion log probability weighted by those advantages.
- There is no reference model KL term in the current trainer.
- The evaluator can load saved LoRA adapters through `peft.PeftModel.from_pretrained`.
- The answer verifier accepts integers, fractions, and decimals by normalizing with Python `Fraction`.
- If no explicit `Answer:` or `\boxed{...}` answer is found, the verifier falls back to the last numeric-looking string in the response.

## Dataset Design

The dataset is generated procedurally. Each row is one variant of a hidden family. A family contains several surface variants that share an underlying structure or answer relationship.

JSONL row fields:

```text
family_id
variant_id
family_type
problem
answer
metadata
```

Evaluation output rows additionally include:

```text
prompt
model_response
extracted_answer
correct
response_tokens
```

Training dataset:

```text
path: data/iso_math_calibrated.jsonl
profile: calibrated
seed: 29
families: 200
variants_per_family: 4
rows: 800
```

Held-out dataset:

```text
path: data/iso_math_calibrated_heldout.jsonl
profile: calibrated
seed: 31
families: 200
variants_per_family: 4
rows: 800
eval slice used so far: first 80 rows
```

The held-out split uses a different generator seed, so the parameters and exact rows differ from the training set. It does not yet test held-out family types.

## Dataset Profiles

The generator supports these profiles:

| Profile | Family types | Current role |
| --- | --- | --- |
| `easy` | `proportional`, `affine`, `linear_equation`, `modular`, `unit_conversion` | Sanity checks only; too easy for meaningful RL. |
| `harder` | `rational_linear_equation`, `nested_linear_equation`, `two_variable_system`, `quadratic_root` | Calibration pass; still not ideal alone. |
| `mixed` | Easy plus harder families | Debugging/general development. |
| `challenge` | `rational_system_target`, `chinese_remainder`, `missing_average`, `affine_composition` | Stress test; too hard/slow for first controlled RL. |
| `calibrated` | Weighted mid-difficulty mix | Current first-RL dataset. |

The calibrated profile is weighted in code as:

```text
rational_linear_equation: 5 entries
missing_average: 3 entries
rational_system_target: 1 entry
chinese_remainder: 1 entry
```

This weighting was chosen because earlier profiles were either too easy or too hard for a first local RL comparison. The target zone was moderate instance accuracy with much lower family accuracy.

## Reward Definitions

Independent reward:

```text
reward_i = 1 if answer_i is correct else 0
```

Iso reward:

```text
reward_i = correctness_i + lambda_iso * family_consistency_family
```

Current strict family consistency:

```text
family_consistency_family = 1 if every sampled variant in that family is correct else 0
```

Otherwise:

```text
family_consistency_family = 0
```

For the current rollout shape, each step samples:

```text
families_per_step: 2
variants_per_family: 4
samples_per_variant: 1
total completions per step: 8
```

This means the Iso bonus is sparse: a sampled family must get all four variants correct in that rollout to receive the extra reward.

## Training Configuration

Matched 20-step configuration:

```text
max_steps: 20
families_per_step: 2
variants_per_family: 4
samples_per_variant: 1
max_new_tokens: 256
temperature: 1.0
top_p: 1.0
learning_rate: 0.000001
LoRA: enabled
```

Prompt template:

```text
Solve the problem. Show your reasoning briefly, then put the final answer after "Answer:".

Problem:
{problem}
```

Reward modes run so far:

```text
independent correctness reward
Iso reward with lambda_iso = 0.50
```

Configs already present but not yet run to completion:

```text
configs/train_iso_calibrated_lam_0_25_20step.yaml
configs/train_iso_calibrated_lam_1_00_20step.yaml
```

Seed control:

- Default seed for first controlled pair: `13`
- Explicit replicate seed: `23`
- The trainer now seeds Python `random`, Torch, and CUDA RNG state when CUDA is available.
- The seed is written into every training log metrics record.

## Evaluation Configuration

Held-out evaluation uses:

```text
dataset_path: data/iso_math_calibrated_heldout.jsonl
max_examples: 80
max_new_tokens: 256
temperature: 0.0
top_p: 1.0
device: auto
```

Metrics:

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of evaluated rows with correct extracted final answer. |
| `family_accuracy` | Fraction of evaluated families where all included variants are correct. |
| `avg_tokens` | Mean completion token count. |
| `avg_wrong_tokens` | Mean completion token count on wrong answers. |
| `format_failure_rate` | Fraction of rows where no answer could be extracted. |

Important caveat: the held-out eval currently uses the first 80 rows of an 800-row file. Because each family has 4 variants, this is effectively 20 families if row order remains family-contiguous.

## Completed Runs

Validation before training:

```text
test command: conda run -n pytorch_5070ti python -m pytest tests
result: 12 passed
```

The test suite covers dataset profile selection, answer extraction/normalization, reward computation, and trainer reward-mode dispatch. It does not validate model quality, statistical significance, or equivalence with a production GRPO implementation.

### Base Held-Out Evaluation

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

Interpretation:

- The held-out calibrated split is in the intended range.
- Single-instance accuracy is moderate.
- Family accuracy is much lower, leaving room for Iso-RLVR to help.
- Wrong answers nearly saturate the 256-token cap, so response length remains a confound.

### First Matched Pair, Seed 13

Independent 20-step:

```json
{
  "accuracy": 0.6125,
  "avg_tokens": 204.85,
  "avg_wrong_tokens": 241.25806451612902,
  "family_accuracy": 0.25,
  "format_failure_rate": 0.0
}
```

Iso `lambda_iso=0.50` 20-step:

```json
{
  "accuracy": 0.625,
  "avg_tokens": 208.7875,
  "avg_wrong_tokens": 250.96666666666667,
  "family_accuracy": 0.3,
  "format_failure_rate": 0.0
}
```

Interpretation:

- Independent reward slightly improves row-level accuracy over base but does not improve strict family accuracy.
- Iso improves both row-level accuracy and family accuracy in this seed.
- The absolute family-accuracy gain over independent is `+0.05`.

### Replicate Matched Pair, Seed 23

Independent 20-step:

```json
{
  "accuracy": 0.6375,
  "avg_tokens": 206.4,
  "avg_wrong_tokens": 249.3448275862069,
  "family_accuracy": 0.3,
  "format_failure_rate": 0.0
}
```

Iso `lambda_iso=0.50` 20-step:

```json
{
  "accuracy": 0.6375,
  "avg_tokens": 206.2125,
  "avg_wrong_tokens": 248.51724137931035,
  "family_accuracy": 0.35,
  "format_failure_rate": 0.0
}
```

Interpretation:

- Independent improved family accuracy relative to base in this seed.
- Iso still improves strict family accuracy by `+0.05` over the matched independent run.
- Iso matches independent row-level accuracy exactly.
- Wrong-answer token length remains very high in both runs.

## Main Finding So Far

Across two matched seeds, Iso `lambda_iso=0.50` improves held-out family accuracy by `+0.05` absolute over independent reward while preserving row-level accuracy.

| Seed | Independent accuracy | Iso accuracy | Independent family accuracy | Iso family accuracy | Family delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.6125 | 0.6250 | 0.2500 | 0.3000 | +0.0500 |
| 23 | 0.6375 | 0.6375 | 0.3000 | 0.3500 | +0.0500 |

This is a replicated directional signal, not proof. The evidence supports continuing the experiment with a lambda sweep, a larger eval slice, and more seeds.

## What A Critic Should Challenge

The most important limitations are:

1. The held-out eval is small: only 80 rows, about 20 families.
2. Train and held-out use different seeds but the same procedural family type distribution.
3. The current held-out split is not a held-out family-type benchmark.
4. The current trainer is a minimal GRPO-like scaffold and has no reference-model KL term.
5. The reward is sparse because strict family consistency requires all sampled variants to be correct.
6. The verifier checks only final numeric answers, not reasoning validity.
7. The answer extractor falls back to the last number, which can hide formatting issues.
8. Wrong answers often hit or approach the 256-token cap, so generation truncation may affect correctness.
9. Only `lambda_iso=0.50` has completed matched seed replication.
10. No statistical interval or bootstrap test has been run yet.
11. The same held-out set is being reused for iteration, so a final blind set is needed.
12. Exact train/held-out row overlap has not yet been formally fingerprinted in the docs, even though generator seeds differ.

These caveats should remain in any external write-up. The project is currently at the "promising local signal" stage.

## Conclusions

What is supported:

- The calibrated dataset exposes the intended gap: row-level accuracy is much higher than family-level accuracy.
- The local training path works for both independent reward and Iso reward.
- LoRA adapters save and can be evaluated on held-out data.
- Iso `lambda_iso=0.50` has twice improved strict held-out family accuracy over a matched independent run.
- The result did not come from a row-level accuracy collapse or format failure spike.

What is not yet supported:

- A claim that Iso-RLVR generally improves mathematical reasoning.
- A claim that the model learned a human-like transformation-invariant solution process.
- A claim that `lambda_iso=0.50` is optimal.
- A claim that this will hold on non-procedural benchmarks.
- A claim that the current GRPO-like trainer is equivalent to a validated production GRPO implementation.

## Recommended Next Experiments

Run these in order:

1. Run an exact train/held-out overlap audit by hashing `(family_type, problem, answer)` and checking intersections.
2. Evaluate the existing base, independent, and Iso adapters on all 800 held-out rows, or at least a larger family-balanced slice.
3. Add family-type breakdown to evaluation summaries.
4. Run `lambda_iso=0.25` at the same 20-step scale and evaluate before launching the next run.
5. Run `lambda_iso=1.00` only if `0.25` does not expose a clear regression pattern.
6. Add one blind calibrated eval seed that is not used for model selection.
7. Add a challenge-profile eval as a stress test, not as the main score.
8. Run at least 3 to 5 total seeds for independent and best Iso lambda.
9. Add bootstrap confidence intervals over families.
10. Validate the trainer against TRL/verl or replace it before making stronger claims.

## Reproduction Commands

Install and test:

```bash
conda run -n pytorch_5070ti python -m pip install -e .
conda run -n pytorch_5070ti python -m pytest tests
```

Generate calibrated train and held-out data:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated_heldout.jsonl --families 200 --variants 4 --seed 31 --profile calibrated
```

Run base held-out eval:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated_heldout.yaml
```

Run the seed 23 matched pair:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_independent_calibrated_20step_seed_23.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_independent_calibrated_20step_seed_23.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.grpo_lite --config configs/train_iso_calibrated_lam_0_50_20step_seed_23.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

## Current Decision

The next compute should not be a long training run yet. The most useful next step is to strengthen measurement:

```text
overlap audit -> larger held-out eval -> family-type breakdown -> lambda 0.25 -> lambda 1.00
```

That sequence will tell us whether the current signal is robust, type-specific, or an artifact of a small eval slice.
