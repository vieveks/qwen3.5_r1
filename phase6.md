# Phase 6: Proper GRPO On Stabilized Packed XML Interface

Status: active planning and implementation spec

Date: 2026-05-28

## Executive Summary

Phase 5 is complete.

The project is no longer blocked by the reward interface. The packed XML answer contract survived:

```text
sampled rollouts
small RL-style gradient updates
heldout evaluation after updates
```

The Phase 5 exit evidence:

```text
sampled parse_complete_rate: 1.0000
post-update parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
malformed outputs: 0
reward_std: non-degenerate
```

Phase 6 therefore resumes the original RLVR goal, but on the stabilized packed XML interface.

The first priority is not reward tuning. The first priority is replacing `grpo_lite` with proper TRL `GRPOTrainer` integration and proving that the real trainer preserves the same interface stability.

## Starting Point

Use the all-traces Phase 5 adapter as the main policy initialization:

```text
Base model: Qwen/Qwen2.5-Math-1.5B
Adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
Heldout: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
```

Do not use the tiny GRPO-lite output as the default initialization. That run was an interface-stability smoke, not a capability-improving checkpoint.

Baseline heldout from the all-traces adapter:

```text
accuracy: 0.7500
family_accuracy: 0.6250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete |
| --- | ---: | ---: | ---: |
| `missing_average` | 1.0000 | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.6000 | 0.4000 | 1.0000 |

The `rational_linear_equation` ceiling around `0.6000` is now a known Phase 6 baseline. It should be improved against, not treated as a surprise.

## Phase 6 Objectives

Primary objective:

```text
Run proper TRL GRPOTrainer on packed XML rows with a stateless reward function.
```

Secondary objective:

```text
Re-test the Iso-RLVR hypothesis on the clean packed XML interface.
```

Do not draw conclusions from Phase 2 or Phase 3 trainer comparisons until they are rerun on this interface. Those earlier results used a reward surface that Phase 4 and Phase 5 showed was too noisy.

## Workstream Order

### 1. TRL GRPOTrainer Integration

Build the real trainer path first.

Expected trainer:

```text
src/iso_rlvr/train/packed_grpo_trl.py
```

Expected first config:

```text
configs/packed_grpo_trl_all_traces_smoke.yaml
```

Initial smoke should use:

```text
base_model: Qwen/Qwen2.5-Math-1.5B
adapter_path: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
train_dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
eval_dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
reward: score_packed_completion
family_bonus_enabled: false
num_generations: 4
temperature: 0.7
max_completion_length: 256
max_steps: 10 to 30
```

The first TRL smoke should be a trainer validation, not a capability experiment.

Gate:

```text
parse_complete_rate >= 0.95
answer_count_mismatch_rate <= 0.05
suspicious_rate <= 0.10
reward_std > 0.05 on sampled rollouts
at least one prompt group with correct and incorrect completions
no malformed-output mode appears repeatedly
```

### 2. Independent Versus Iso Reward Matrix

Only after the TRL smoke passes, rerun the core comparison:

| Run | Reward | Family bonus | Purpose |
| --- | --- | --- | --- |
| Independent GRPO | correctness + format | disabled | Clean independent baseline |
| Iso GRPO | correctness + format + family component | enabled | Test Iso-RLVR signal |

Use the same model, adapter initialization, dataset split, generation parameters, and evaluation code for both runs.

Primary metrics:

- variant accuracy
- family accuracy
- parse complete rate
- answer-count mismatch rate
- suspicious rate
- reward mean/std
- contrast prompt count
- by-family-type metrics

Decision rule:

```text
Iso reward must improve family_accuracy without degrading parse_complete_rate.
```

Accuracy tradeoffs are acceptable only if family consistency improves materially and the interface remains stable.

### 3. Rational-Linear Recovery

The all-traces bridge fixed format stability but reduced `rational_linear_equation` accuracy from the mixed-v1 reference.

Known reference points:

```text
mixed-v1 rational_linear_equation accuracy: 0.8500
all-traces rational_linear_equation accuracy: 0.6000
```

Likely cause:

```text
The deterministic rational-linear trace pattern terminates cleanly but sometimes induces arithmetic mistakes.
```

Do not optimize this before the first TRL smoke. After trainer stability is proven, test targeted fixes:

- improved rational-linear trace format
- answer-only rational-linear plus loop-control decoding
- mixed target distribution with rational traces only for hard rows
- post-SFT rational-linear correction set

Success criterion:

```text
rational_linear_equation accuracy > 0.6000
parse_complete_rate remains >= 0.95
```

## Reward Function Contract

The first Phase 6 reward should match the Phase 5 smoke:

```text
score_packed_completion
family_bonus_enabled: false
format reward enabled
extra answer penalty enabled
length penalty enabled
no correctness credit unless parse-complete
```

Family bonus remains disabled for the first TRL smoke so trainer failures cannot be confused with reward-shaping failures.

After the trainer smoke passes, enable family bonus only in the Iso comparison arm.

Provisional parse-incomplete rule:

```text
Parse-incomplete completions receive zero correctness reward and no family bonus.
They may receive the small configured format penalty.
They stay in the GRPO group.
```

This preserves group statistics while preventing malformed completions from getting task credit.

## Dataset Requirements

Every packed row used for TRL must carry all reward context as columns:

```text
prompt
family_id
family_type
variant_ids
problems
gold_answers
num_variants
metadata
prompt_format
```

The reward function must be stateless:

```text
reward_func(completions, gold_answers, family_id, family_type, variant_ids, num_variants, ...)
```

No side-table lookups.

## Logging Requirements

The TRL trainer path must log enough information to diagnose both trainer behavior and reward-interface behavior.

Per training/eval window:

- parse complete rate
- answer-count mismatch rate
- suspicious rate
- reward mean/std/min/max
- contrast prompt count
- accuracy
- family accuracy
- by-family-type metrics
- malformed completions in full, not truncated

For malformed outputs, full completions matter because these cases need different fixes:

- never started XML
- started XML but looped
- partial closing tag
- invalid value inside a tag
- extra answer tags

## First Implementation Plan

1. Check installed TRL version and whether `GRPOTrainer` is available.
2. Add TRL to project dependencies only if needed.
3. Implement a packed TRL reward adapter around `score_packed_completion`.
4. Implement `packed_grpo_trl.py` with LoRA adapter loading.
5. Add a tiny TRL smoke config.
6. Run a no-training reward-function unit test.
7. Run a tiny TRL smoke.
8. Compare TRL smoke interface metrics against Phase 5 GRPO-lite smoke.
9. Update this document with results.

## Phase 6 Experiments

### Experiment 6.1: TRL Availability And Reward Adapter Preflight

Goal:

```text
Verify the local TRL situation and test the packed reward adapter before trainer wiring.
```

TRL check:

```text
Command: conda run -n pytorch_5070ti python -c "import trl; print(trl.__version__)"
Result: ModuleNotFoundError: No module named 'trl'
```

Conclusion:

```text
TRL is not installed in the current pytorch_5070ti environment.
```

This meant we could not safely target a concrete `GRPOTrainer` API until installing and inspecting the exact local TRL version.

Reward adapter implemented:

```text
Module: src/iso_rlvr/train/packed_grpo_trl.py
Tests: tests/test_packed_grpo_trl.py
```

The first reward-adapter preflight verified:

```text
packed_trl_reward_config
packed_trl_rewards
make_packed_trl_reward_func
```

The unit tests verify that the reward adapter:

- accepts plain string completions
- accepts chat-style completion dictionaries
- accepts dataset columns as keyword arguments
- uses `gold_answers` as the stateless reward source
- accepts optional `completion_ids` for length penalty
- rejects mismatched input lengths
- keeps family bonus disabled by default

Targeted test result:

```text
tests/test_packed_grpo_trl.py: 5 passed
```

Conclusion:

The stateless packed reward path was ready before trainer wiring. The next step was installing and inspecting TRL.

### Experiment 6.2: TRL Installation And API Inspection

Goal:

```text
Install TRL, record the exact local API, and implement the real packed GRPOTrainer path.
```

Installed environment:

```text
Environment: pytorch_5070ti
trl: 0.17.0
transformers: 5.0.0.dev0
accelerate: 1.10.0
datasets: 4.0.0
peft: 0.17.0
```

Local API check:

```text
GRPOTrainer.__init__(model, reward_funcs, args, train_dataset, eval_dataset, processing_class, reward_processing_classes, callbacks, optimizers, peft_config)
GRPOConfig includes num_generations: true
GRPOConfig includes beta: true
```

Important local behavior:

```text
TRL 0.17.0 passes reward functions:
reward_func(prompts=prompts, completions=completions, **dataset_columns)
```

The dataset column names therefore matter directly. A row column named `gold_answers` arrives at the reward function as `gold_answers=...`.

Implementation:

```text
Trainer: src/iso_rlvr/train/packed_grpo_trl.py
Config: configs/packed_grpo_trl_all_traces_smoke.yaml
Dependency: trl==0.17.0
```

The trainer now:

- loads the base model
- loads the Phase 5 all-traces LoRA adapter as trainable
- wraps `score_packed_completion` as a stateless TRL reward function
- logs per-call sampled reward records to `reward_calls.jsonl`
- logs per-prompt reward distributions for the sampled group
- saves the trained adapter
- runs a final packed heldout eval after training

Targeted test result:

```text
tests/test_packed_grpo_trl.py: 5 passed
```

Conclusion:

The real TRL trainer path is implemented against the installed local `trl==0.17.0` API.

### Experiment 6.3: Tiny TRL GRPOTrainer Smoke

Goal:

```text
Verify that proper TRL GRPOTrainer preserves the stabilized packed XML reward interface.
```

Config:

```text
Config: configs/packed_grpo_trl_all_traces_smoke.yaml
Base model: Qwen/Qwen2.5-Math-1.5B
Initial adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Train dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
Eval dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
Output: outputs/phase6/packed_grpo_trl_all_traces_smoke
TRL: 0.17.0
Steps: 10
Per-device train batch size: 4
Num generations: 4
Temperature: 0.7
Max completion length: 256
Learning rate: 1e-6
Beta: 0.04
Reward: packed correctness plus format signal only; family bonus disabled
```

Implementation note:

The first run completed TRL training but failed after saving because final heldout eval expected `max_new_tokens` while the TRL config used `max_completion_length`. The trainer now maps `max_completion_length` to `max_new_tokens` for final eval, and the rerun completed.

Sampled training-batch result:

```text
reward calls: 10
sampled completions: 40
parse_complete_rate: 1.0000 on every sampled batch
answer_count_mismatch_rate: 0.0000 on every sampled batch
suspicious_rate: 0.0000 on every sampled batch
malformed samples: 0
steps with within-prompt contrast: 5 / 10
```

Training reward-call summary:

| Step | Reward mean | Reward std | Contrast prompts | Parse complete | Mismatch | Suspicious |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 |
| 2 | 0.9218 | 0.2220 | 1 | 1.0000 | 0.0000 | 0.0000 |
| 3 | 0.6659 | 0.2218 | 1 | 1.0000 | 0.0000 | 0.0000 |
| 4 | 0.7940 | 0.2560 | 1 | 1.0000 | 0.0000 | 0.0000 |
| 5 | 0.5371 | 0.0003 | 0 | 1.0000 | 0.0000 | 0.0000 |
| 6 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 |
| 7 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 |
| 8 | 0.6620 | 0.2240 | 1 | 1.0000 | 0.0000 | 0.0000 |
| 9 | 0.7939 | 0.2561 | 1 | 1.0000 | 0.0000 | 0.0000 |
| 10 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 |

Trainer summary:

```text
train_runtime: 81.22 seconds
train_loss: 0.02255
```

Final heldout eval after saving:

```text
accuracy: 0.7188
family_accuracy: 0.5625
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
reward_mean: 0.7612
reward_std: 0.3616
```

By family type on final heldout:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.9167 | 0.8333 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.6000 | 0.4000 | 1.0000 | 0.0000 | 0.0000 |

Interpretation:

The first proper TRL smoke passed the trainer-stability gate. The XML interface stayed stable under real `GRPOTrainer` updates:

```text
sampled parse_complete_rate: 1.0000
post-update heldout parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
malformed samples: 0
```

As expected, this 10-step smoke should not be treated as capability improvement. Final heldout accuracy was `0.7188`, slightly below the all-traces starting baseline of `0.7500`, and `rational_linear_equation` stayed at `0.6000`.

Conclusion:

Phase 6 now has a working proper TRL GRPOTrainer path. The next experiment should be the clean independent-versus-iso reward comparison matrix on this packed XML interface.

### Experiment 6.4: Independent Versus Iso TRL Matrix

Goal:

```text
Run the first clean Phase 6 comparison between independent packed GRPO and iso-reward packed GRPO.
```

Design:

```text
Trainer: TRL GRPOTrainer
Base model: Qwen/Qwen2.5-Math-1.5B
Initialization for both arms: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Train dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
Eval dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
Seed: 23 for both arms
Steps: 30
Num generations: 4
Temperature: 0.7
Max completion length: 256
Learning rate: 1e-6
Beta: 0.04
```

Important:

Both arms start from the untouched Phase 5 all-traces adapter. They do not start from the 10-step TRL smoke output. The TRL-smoke checkpoint is a trainer-validation artifact, not the matrix initialization.

Configs:

```text
Independent: configs/packed_grpo_trl_all_traces_independent_30step.yaml
Iso: configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step.yaml
```

Reward settings:

| Arm | Family bonus | Family weights |
| --- | --- | --- |
| Independent | disabled | `0.00 + 0.00` |
| Iso | enabled | `family_mean_weight=0.25`, `all_family_correct_weight=0.25` |

The iso arm corresponds to the planned first `lambda_iso=0.50` scale: a fully correct packed family receives a total family bonus of `0.50` on correct variants.

Run order:

```text
Independent first, fully evaluated.
Iso second, with the independent result locked.
```

Independent result:

```text
Output: outputs/phase6/packed_grpo_trl_all_traces_independent_30step
accuracy: 0.7188
family_accuracy: 0.5625
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
reward_mean: 0.7612
reward_std: 0.3616
sampled contrast steps: 10 / 30
sampled malformed completions: 0
```

Independent by family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.9167 | 0.8333 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.6000 | 0.4000 | 1.0000 | 0.0000 | 0.0000 |

Iso result:

```text
Output: outputs/phase6/packed_grpo_trl_all_traces_iso_lam_0_50_30step
accuracy: 0.7500
family_accuracy: 0.6250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
reward_mean: 1.1217
reward_std: 0.5775
sampled contrast steps: 12 / 30
sampled malformed completions: 0
```

Iso by family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.6000 | 0.4000 | 1.0000 | 0.0000 | 0.0000 |

Matrix comparison:

| Metric | Independent | Iso `lambda_iso=0.50` | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7188 | 0.7500 | +0.0312 |
| Family accuracy | 0.5625 | 0.6250 | +0.0625 |
| Parse complete | 1.0000 | 1.0000 | 0.0000 |
| Mismatch rate | 0.0000 | 0.0000 | 0.0000 |
| Suspicious rate | 0.0000 | 0.0000 | 0.0000 |
| Sampled contrast steps | 10 / 30 | 12 / 30 | +2 |
| Sampled malformed completions | 0 | 0 | 0 |

Interpretation:

This is the first clean support for the Iso-RLVR hypothesis on the stabilized packed XML interface. The iso arm improved family accuracy by `+0.0625` without any parse regression:

```text
parse_complete_rate: 1.0000 in both arms
answer_count_mismatch_rate: 0.0000 in both arms
suspicious_rate: 0.0000 in both arms
sampled malformed completions: 0 in both arms
```

The improvement is small and the heldout set is still tiny, so it should not be overclaimed. But the direction is correct, and it is now measured on the clean reward interface that Phase 5 built.

The improvement came from `missing_average`, where iso reached perfect heldout performance. `rational_linear_equation` remained flat at `0.6000` accuracy and `0.4000` family accuracy in both arms. That confirms the rational-linear stall is not fixed by the current iso reward and should remain a separate recovery workstream.

Conclusion:

The first proper TRL comparison supports continuing Phase 6. Next step should be a confirmation run with either a second seed or a modestly larger heldout/eval set before sweeping `lambda_iso`.

### Experiment 6.5: Second-Seed Confirmation Matrix

Goal:

```text
Check whether the iso family-accuracy advantage replicates with a different seed.
```

Design:

```text
Same model initialization
Same train dataset
Same heldout eval dataset
Same steps
Same generation settings
Same rewards
Different seed: 37
```

Configs:

```text
Independent: configs/packed_grpo_trl_all_traces_independent_30step_seed_37.yaml
Iso: configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step_seed_37.yaml
```

Independent seed 37 result:

```text
Output: outputs/phase6/packed_grpo_trl_all_traces_independent_30step_seed_37
accuracy: 0.7188
family_accuracy: 0.5625
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
reward_mean: 0.7612
reward_std: 0.3617
sampled contrast steps: 14 / 30
sampled malformed completions: 0
```

Independent seed 37 by family type:

| Family type | Accuracy | Family accuracy |
| --- | ---: | ---: |
| `missing_average` | 0.9167 | 0.8333 |
| `rational_linear_equation` | 0.6000 | 0.4000 |

Iso seed 37 result:

```text
Output: outputs/phase6/packed_grpo_trl_all_traces_iso_lam_0_50_30step_seed_37
accuracy: 0.7500
family_accuracy: 0.6250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
reward_mean: 1.1217
reward_std: 0.5775
sampled contrast steps: 14 / 30
sampled malformed completions: 0
```

Iso seed 37 by family type:

| Family type | Accuracy | Family accuracy |
| --- | ---: | ---: |
| `missing_average` | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.6000 | 0.4000 |

Seed 37 matrix comparison:

| Metric | Independent | Iso `lambda_iso=0.50` | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7188 | 0.7500 | +0.0312 |
| Family accuracy | 0.5625 | 0.6250 | +0.0625 |
| Parse complete | 1.0000 | 1.0000 | 0.0000 |
| Mismatch rate | 0.0000 | 0.0000 | 0.0000 |
| Suspicious rate | 0.0000 | 0.0000 | 0.0000 |
| Sampled contrast steps | 14 / 30 | 14 / 30 | 0 |
| Sampled malformed completions | 0 | 0 | 0 |

Contrast prompt groups:

```text
Independent seed 37:
fam_000027, fam_000032, fam_000037, fam_000045, fam_000050,
fam_000086, fam_000098, fam_000106, fam_000114, fam_000119,
fam_000134, fam_000136, fam_000161, fam_000184

Iso seed 37:
fam_000027, fam_000032, fam_000037, fam_000045, fam_000050,
fam_000086, fam_000098, fam_000106, fam_000114, fam_000119,
fam_000134, fam_000136, fam_000161, fam_000184
```

The seed 37 contrast-driver groups were identical across arms. That means the replicated heldout improvement is not explained by iso and independent receiving contrast from different prompt groups in this run.

Two-seed summary:

| Seed | Independent family accuracy | Iso family accuracy | Delta | Parse stable |
| ---: | ---: | ---: | ---: | --- |
| 23 | 0.5625 | 0.6250 | +0.0625 | yes |
| 37 | 0.5625 | 0.6250 | +0.0625 | yes |

Interpretation:

The iso advantage replicated in direction and magnitude on a second seed while preserving the reward interface:

```text
parse_complete_rate: 1.0000 in all four runs
answer_count_mismatch_rate: 0.0000 in all four runs
suspicious_rate: 0.0000 in all four runs
sampled malformed completions: 0 in all four runs
```

This is still a small heldout set, but the result is now stronger than a one-off. The first clean Phase 6 evidence supports the Iso-RLVR hypothesis on the stabilized packed XML interface.

The mechanism remains narrow:

```text
missing_average improves under iso
rational_linear_equation stays flat
```

Conclusion:

The confirmation run supports continuing toward a broader evaluation. The next best step is not a `lambda_iso` sweep yet; it is a larger or more reliable evaluation surface so the family-accuracy gap is measured with less variance.

## First Smoke Pass Condition

The first TRL smoke passes if:

```text
heldout parse_complete_rate >= 0.95
heldout answer_count_mismatch_rate <= 0.05
heldout suspicious_rate <= 0.10
sampled reward_std > 0.05
no repeated malformed-output mode
```

It does not need to improve accuracy.

## Current Recommendation

The proper TRL GRPO smoke has passed, and the independent-versus-iso matrix replicated on a second seed:

```text
seed 23 family_accuracy delta: +0.0625
seed 37 family_accuracy delta: +0.0625
parse_complete_rate delta: 0.0000
sampled malformed completions: 0 in all arms
```

Do not sweep `lambda_iso` yet. The next highest-value check is a larger or more reliable heldout/eval surface. The `rational_linear_equation` baseline is still known and should be improved later as a targeted recovery workstream.

The current Phase 6 thesis:

```text
Same stable reward interface.
Real GRPO trainer confirmed.
Iso-RLVR comparison replicated on two seeds.
Broaden evaluation before sweeping.
```
