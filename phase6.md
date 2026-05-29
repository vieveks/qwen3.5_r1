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

### Experiment 6.6: Broad Heldout Eval Across Calibrated Family Types

Goal:

```text
Evaluate the already-trained seed 23 independent and iso adapters on a broader heldout surface before sweeping lambda_iso.
```

This is an evaluation-only experiment. No new training was run.

Dataset:

```text
Source: data/iso_math_calibrated_heldout_clean.jsonl
Packed output: outputs/phase6/packed_calibrated_heldout_clean_xml_pair_64_seed0.jsonl
Prompt format: xml
Families: 64
Variants per family: 2
Seed: 0
```

Family-type counts:

| Family type | Examples |
| --- | ---: |
| `chinese_remainder` | 11 |
| `missing_average` | 19 |
| `rational_linear_equation` | 28 |
| `rational_system_target` | 6 |

Configs:

```text
Independent: configs/packed_eval_phase6_broad_independent_seed23.yaml
Iso: configs/packed_eval_phase6_broad_iso_seed23.yaml
```

Independent seed 23 broad eval:

```text
Output: outputs/phase6/packed_eval_broad_independent_seed23.jsonl
accuracy: 0.5625
family_accuracy: 0.4531
parse_complete_rate: 0.9063
answer_count_mismatch_rate: 0.0938
suspicious_rate: 0.1250
avg_reward: 0.8248
```

Independent by family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0909 |
| `missing_average` | 0.8684 | 0.7895 | 1.0000 | 0.0000 | 0.0526 |
| `rational_linear_equation` | 0.6964 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Iso seed 23 broad eval:

```text
Output: outputs/phase6/packed_eval_broad_iso_seed23.jsonl
accuracy: 0.5625
family_accuracy: 0.4375
parse_complete_rate: 0.8906
answer_count_mismatch_rate: 0.1094
suspicious_rate: 0.1250
avg_reward: 0.8175
```

Iso by family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 0.9091 | 0.0909 | 0.1818 |
| `missing_average` | 0.8684 | 0.7368 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.6964 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Broad comparison:

| Metric | Independent | Iso `lambda_iso=0.50` | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5625 | 0.5625 | 0.0000 |
| Family accuracy | 0.4531 | 0.4375 | -0.0156 |
| Parse complete | 0.9063 | 0.8906 | -0.0157 |
| Mismatch rate | 0.0938 | 0.1094 | +0.0156 |
| Suspicious rate | 0.1250 | 0.1250 | 0.0000 |

Representative malformed `rational_system_target` behavior:

```text
Gold: ["-16/5", "-16/5"]
Failure: visible algebra trace loops until the token cap and never emits XML.
Parsed: [null, null]
```

Representative malformed `chinese_remainder` behavior in the iso arm:

```xml
<answers>
<answer_1>1</answer_1>
<answer_2>no solution</answer_2>
</answers>
```

The parser correctly rejects `no solution` because the current answer contract only accepts numeric values.

Interpretation:

The broad eval does not support a lambda sweep yet. It surfaced a stronger blocker:

```text
the packed XML interface does not yet generalize to all calibrated family types
```

The two-seed Iso-RLVR result remains valid for the stabilized two-family surface. But the broad surface is not clean because both arms fail the parse gate:

```text
independent parse_complete_rate: 0.9063
iso parse_complete_rate: 0.8906
target: >= 0.9500
```

The failure is concentrated in `rational_system_target`, which was not part of the Phase 5 all-traces SFT bridge. The model falls into long system-solving traces and fails to terminate with XML. `chinese_remainder` is mostly parse-complete but has zero accuracy, and the iso arm has one non-numeric answer-tag failure.

Conclusion:

Do not sweep `lambda_iso` yet. The next step should be broad-interface stabilization: add deterministic trace coverage or another bridge for the heldout calibrated family types, then rerun this broad eval. The clean two-family result is real, but it is not yet a generalization result across unseen family types.

### Experiment 6.7: Broad All-Traces SFT Bridge

Goal:

```text
Repair the broad packed XML interface for all calibrated family types before any more GRPO.
```

Implementation:

```text
Trace builder: src/iso_rlvr/data/build_format_sft_dataset.py
Trainer update: src/iso_rlvr/train/format_sft.py
Tests: tests/test_format_sft_dataset.py
```

New trace coverage:

| Family type | Trace policy |
| --- | --- |
| `missing_average` | existing deterministic sum trace |
| `rational_linear_equation` | existing deterministic isolate/divide trace |
| `rational_system_target` | fixed elimination method for every row |
| `chinese_remainder` | numeric CRT trace with hard assertions for coprime moduli and least nonnegative answer |

Important design choices:

- `rational_system_target` always uses the same elimination procedure. It does not switch between substitution and elimination based on row shape.
- `chinese_remainder` rows assert coprime moduli and numeric least nonnegative answers. No `no solution` target is allowed.
- The SFT trainer now supports `adapter_path`, so this bridge continues from the Phase 5 all-traces adapter rather than starting a fresh LoRA.

Targeted test result:

```text
tests/test_format_sft_dataset.py: 19 passed
tests/test_format_sft.py: 3 passed
```

Generated broad SFT dataset:

```text
Packed source: outputs/phase6/packed_calibrated_train_xml_pair_all_families.jsonl
SFT train: outputs/phase6/format_sft_pair_xml_broad_all_traces_train.jsonl
SFT heldout: outputs/phase6/format_sft_pair_xml_broad_all_traces_heldout.jsonl
Packed train split: outputs/phase6/packed_calibrated_train_xml_pair_broad_all_traces_train.jsonl
Packed heldout split: outputs/phase6/packed_calibrated_train_xml_pair_broad_all_traces_heldout.jsonl
Train rows: 180
Heldout rows: 20
```

Train target styles:

| Target style | Rows |
| --- | ---: |
| `chinese_remainder_trace_xml` | 14 |
| `missing_average_trace_xml` | 54 |
| `rational_linear_trace_xml` | 92 |
| `rational_system_trace_xml` | 20 |

Training config:

```text
Config: configs/format_sft_qwen25_math_1_5b_xml_broad_all_traces_from_phase5_1epoch.yaml
Base model: Qwen/Qwen2.5-Math-1.5B
Initial adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Rows: 180
Steps: 90
Batch size: 2
Learning rate: 1e-4
Max sequence length: 1536
Output: outputs/phase6/format_sft_qwen25_math_1_5b_xml_broad_all_traces_from_phase5_1epoch/adapter_or_model
```

Final logged training losses:

| Step | Loss |
| ---: | ---: |
| 82 | 0.0826 |
| 83 | 0.0028 |
| 84 | 0.0215 |
| 85 | 0.0157 |
| 86 | 0.0515 |
| 87 | 0.0014 |
| 88 | 0.0063 |
| 89 | 0.0137 |

Broad deterministic heldout eval:

```text
Config: configs/packed_eval_phase6_broad_bridge_512.yaml
Dataset: outputs/phase6/packed_calibrated_heldout_clean_xml_pair_64_seed0.jsonl
Max new tokens: 512
accuracy: 0.5938
family_accuracy: 0.4844
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0938
avg_reward: 0.8851
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.4545 |
| `missing_average` | 0.8947 | 0.7895 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.7321 | 0.5714 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0833 | 0.0000 | 1.0000 | 0.0000 | 0.1667 |

Interpretation:

The broad bridge fixed the deterministic parser interface:

```text
parse_complete_rate improved from 0.9063 / 0.8906 to 1.0000
answer_count_mismatch_rate improved to 0.0000
suspicious_rate is back under the 0.1000 gate
```

It did not solve broad capability. `chinese_remainder` is now parse-complete but still has `0.0000` accuracy, and `rational_system_target` is mostly wrong despite complete XML. This is acceptable for the immediate interface gate but not enough to claim broad Iso-RLVR generalization.

### Experiment 6.8: Broad Bridge Sampled Rollout Audit

Goal:

```text
Check whether the broad bridge keeps the XML interface stable under sampled rollouts before any broad GRPO run.
```

Config:

```text
Config: configs/packed_rollout_audit_broad_bridge_sampled.yaml
Adapter: outputs/phase6/format_sft_qwen25_math_1_5b_xml_broad_all_traces_from_phase5_1epoch/adapter_or_model
Dataset: outputs/phase6/packed_calibrated_heldout_clean_xml_pair_64_seed0.jsonl
Output: outputs/phase6/packed_rollout_audit_broad_bridge_sampled.jsonl
Samples per prompt: 4
Temperature: 0.7
Max new tokens: 512
```

Result:

```text
prompts: 64
samples: 256
variant_examples: 512
accuracy: 0.5469
family_accuracy: 0.4219
parse_complete_rate: 0.9922
answer_count_mismatch_rate: 0.0078
suspicious_rate: 0.1523
reward_mean: 0.5867
reward_std: 0.4418
contrast_prompt_count: 9
passes_audit_gate: true
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 0.9773 | 0.0227 | 0.7500 |
| `missing_average` | 0.8684 | 0.7500 | 1.0000 | 0.0000 | 0.0132 |
| `rational_linear_equation` | 0.6607 | 0.4554 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0000 | 0.0000 | 0.9583 | 0.0417 | 0.2083 |

Contrast prompt groups:

```text
missing_average:
fam_000022, fam_000144, fam_000110

rational_linear_equation:
fam_000112, fam_000039, fam_000013, fam_000001, fam_000038, fam_000171
```

The new family types produced no prompt-level correct/incorrect contrast. The sampled learning signal still comes entirely from the original two family types.

Representative sampled malformed modes:

```text
chinese_remainder: sampled trace can enter an increment/search loop and miss XML before the token cap.
rational_system_target: sampled trace can enter a degenerate zero-coefficient elimination loop and miss XML before the token cap.
```

Interpretation:

The sampled audit passes the formal gate because parse completeness is high, reward variance is non-degenerate, and there are contrast prompts:

```text
parse_complete_rate: 0.9922
reward_std: 0.4418
contrast_prompt_count: 9
```

But this should not be treated as readiness for broad Iso-RLVR. The contrast prompts are not broad. They are all `missing_average` and `rational_linear_equation`. `chinese_remainder` and `rational_system_target` are now mostly verifier-compatible, but they are not yet useful RL targets under this bridge.

Conclusion:

The broad parser interface is repaired enough for deterministic eval and mostly stable under sampling. The next blocker is capability/contrast for the new family types, especially `chinese_remainder`. Do not run the broad independent-versus-iso GRPO matrix until at least one new family type has non-trivial sampled correctness and prompt-level contrast.

### Experiment 6.9: New-Family Capability Bridge Plan

Goal:

```text
Turn the now-parse-stable new family types into useful RL targets.
```

Current diagnosis:

```text
The XML interface is no longer the blocker across all four families.
The blocker is that the new family types do not produce useful correctness or contrast.
```

Observed state after the broad bridge:

| Family type | Deterministic accuracy | Sampled accuracy | Sampled contrast |
| --- | ---: | ---: | --- |
| `chinese_remainder` | 0.0000 | 0.0000 | none |
| `rational_system_target` | 0.0833 | 0.0000 | none |

Interpretation:

`chinese_remainder` failed because the first broad trace mostly stated/checks the known answer. That repairs formatting but does not teach a reusable constructive method. The model needs a short algorithm it can imitate:

```text
Given x = r1 mod m1 and x = r2 mod m2:
start at r1
step by m1
check mod m2
stop when the second congruence matches
```

Target trace shape:

```text
Problem 1 start: x = 3
Problem 1 step by first modulus: +5
Problem 1 candidates: 3, 8, 13, 18, 23
Problem 1 check: 23 mod 7 = 2
Problem 1 least value: x = 23
```

This should replace the current CRT trace that begins with the gold answer and verifies it. The constructive walk is short, deterministic, and generated from the row metadata. It also avoids any `no solution` grammar expansion. If a row is not a valid numeric CRT instance, the builder should fail hard.

`rational_system_target` likely failed because the elimination trace is too dense for the 1.5B model. The trace is mathematically valid, but system solving has more arithmetic steps than the single-variable families. The next trace should be shorter and more regular.

Target system trace policy:

```text
Use one fixed method.
Keep at most four arithmetic lines before XML.
Prefer two elimination lines plus target computation.
Avoid explanatory fallback prose.
Avoid row-dependent method switching.
```

Candidate trace shape:

```text
Problem 1 eliminate y: 18x = -144/5, so x = -8/5
Problem 1 eliminate x: 18y = 144/5, so y = 8/5
Problem 1 target: x - y = -8/5 - 8/5 = -16/5
```

Next implementation order:

1. Replace the CRT trace with the constructive candidate walk.
2. Simplify the rational-system trace to the minimum fixed elimination trace.
3. Retrain the broad bridge from the Phase 5 all-traces adapter.
4. Re-run deterministic broad eval.
5. Re-run sampled broad audit.

Success condition before broad GRPO:

```text
parse_complete_rate >= 0.95
at least one new family has non-zero sampled accuracy
at least one new family contributes prompt-level contrast
```

Do not run a lambda sweep or broad independent-versus-iso GRPO matrix until this condition is met.

### Experiment 6.10: Broad Bridge v2 Candidate-Walk CRT Trace

Goal:

```text
Fix chinese_remainder by replacing answer verification with a constructive walk.
```

Pre-implementation token-budget check:

```text
Tokenizer: Qwen/Qwen2.5-Math-1.5B
CRT rows checked: 16
prompt tokens: min 125, median 127, max 130
full constructive CRT completion tokens: min 169, median 214, max 272
capped constructive CRT completion tokens: min 169, median 204, max 221
```

Conclusion from the token check:

```text
An uncapped CRT candidate walk can exceed a 256-token completion budget.
A capped candidate walk fits comfortably under 256 completion tokens.
```

Implementation:

```text
Builder: src/iso_rlvr/data/build_format_sft_dataset.py
Tests: tests/test_format_sft_dataset.py
Train config: configs/format_sft_qwen25_math_1_5b_xml_broad_all_traces_v2_from_phase5_1epoch.yaml
Eval config: configs/packed_eval_phase6_broad_bridge_v2_512.yaml
```

The v2 CRT target used a bounded candidate list:

```text
Problem 1 start: x = 7
Problem 1 step by first modulus: +17
Problem 1 candidates: 7, 24, 41
Problem 1 check: 41 mod 21 = 20
Problem 1 least value: x = 41
```

For longer searches, the target used an ellipsis:

```text
Problem 2 candidates: 1, 6, 11, 16, 21, 26, ..., 41
```

The rational-system trace was also shortened to a fixed three-line elimination form:

```text
Problem 1 eliminate y: -18x = -144/5, so x = -8/5
Problem 1 eliminate x: -18y = 144/5, so y = 8/5
Problem 1 target: x - y = -8/5 - 8/5 = -16/5
```

Targeted test result:

```text
tests/test_format_sft_dataset.py tests/test_format_sft.py: 22 passed
```

Training:

```text
Base adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Rows: 180
Steps: 90
Learning rate: 1e-4
Adapter output: outputs/phase6/format_sft_qwen25_math_1_5b_xml_broad_all_traces_v2_from_phase5_1epoch/adapter_or_model
```

Deterministic broad eval result:

```text
Config: configs/packed_eval_phase6_broad_bridge_v2_512.yaml
accuracy: 0.5703
family_accuracy: 0.4531
parse_complete_rate: 0.8750
answer_count_mismatch_rate: 0.1250
suspicious_rate: 0.1875
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 0.2727 | 0.7273 | 0.7273 |
| `missing_average` | 0.8684 | 0.7368 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.7143 | 0.5357 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6667 |

Observed failure mode:

```text
The model ignored the capped candidate-list target and expanded the numeric sequence until max_new_tokens.
Several CRT completions never reached XML.
```

Representative failure:

```text
Problem 1 candidates: 0, 15, 30, 45, 60, 75, 90, 105, ... 1380, 1
```

Conclusion:

The candidate-walk idea was directionally useful but the comma-separated numeric list created a strong continuation attractor. The v2 bridge is not usable because it reintroduced the exact Phase 4 failure mode: long trace loops that prevent XML emission.

Do not use candidate-list CRT traces again unless constrained decoding or explicit stop control is added.

### Experiment 6.11: Broad Bridge v3 Compact CRT k-Trace

Goal:

```text
Preserve the constructive CRT signal without teaching an open-ended numeric-list continuation.
```

Implementation change:

```text
Replace CRT candidate lists with a compact x = r + mk form and one chosen k.
```

New CRT target shape:

```text
Problem 1 form: x = 7 + 17k
Problem 1 choose k = 2: x = 41
Problem 1 check: 41 mod 21 = 20
Problem 1 least value: x = 41
```

The builder now asserts:

```text
moduli are coprime
answer is the least nonnegative solution modulo mod_a * mod_b
gold answer matches metadata answer
chosen k reaches the gold answer from the first residue
no smaller k satisfies the second congruence
```

Implementation:

```text
Builder: src/iso_rlvr/data/build_format_sft_dataset.py
Tests: tests/test_format_sft_dataset.py
Train config: configs/format_sft_qwen25_math_1_5b_xml_broad_all_traces_v3_from_phase5_1epoch.yaml
Eval config: configs/packed_eval_phase6_broad_bridge_v3_512.yaml
Audit config: configs/packed_rollout_audit_broad_bridge_v3_sampled.yaml
```

Targeted test result:

```text
tests/test_format_sft_dataset.py tests/test_format_sft.py: 22 passed
```

Dataset:

```text
Train rows: 180
missing_average_trace_xml: 54
rational_linear_trace_xml: 92
rational_system_trace_xml: 20
chinese_remainder_trace_xml: 14
```

Training:

```text
Base adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Rows: 180
Steps: 90
Learning rate: 1e-4
Adapter output: outputs/phase6/format_sft_qwen25_math_1_5b_xml_broad_all_traces_v3_from_phase5_1epoch/adapter_or_model
```

Deterministic broad eval result:

```text
Config: configs/packed_eval_phase6_broad_bridge_v3_512.yaml
accuracy: 0.5703
family_accuracy: 0.4375
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0625
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0455 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `missing_average` | 0.8684 | 0.7368 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.6964 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6667 |

Sampled rollout audit result:

```text
Config: configs/packed_rollout_audit_broad_bridge_v3_sampled.yaml
samples: 256
accuracy: 0.5469
family_accuracy: 0.4141
parse_complete_rate: 0.9883
answer_count_mismatch_rate: 0.0117
suspicious_rate: 0.0352
reward_mean: 0.5876
reward_std: 0.4361
contrast_prompt_count: 13
passes_audit_gate: true
```

By family type on sampled audit:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 0.9318 | 0.0682 | 0.0682 |
| `missing_average` | 0.8355 | 0.6842 | 1.0000 | 0.0000 | 0.0132 |
| `rational_linear_equation` | 0.6786 | 0.4821 | 1.0000 | 0.0000 | 0.0000 |
| `rational_system_target` | 0.0208 | 0.0000 | 1.0000 | 0.0000 | 0.2083 |

Correctness-level prompt contrast:

| Family type | Prompts with correct/incorrect contrast |
| --- | ---: |
| `missing_average` | 5 |
| `rational_linear_equation` | 13 |
| `rational_system_target` | 1 |
| `chinese_remainder` | 0 |

Interpretation:

V3 fixes the v2 interface regression. Deterministic parse is perfect, sampled parse is above gate, and the remaining malformed samples are limited to three CRT sampled completions.

However, v3 is still not a broad GRPO-ready bridge. CRT has no sampled correctness contrast and zero sampled accuracy. Rational-system has only one prompt with correctness contrast and very low sampled accuracy. Most usable gradient signal still comes from the original two family types.

Remaining sampled CRT malformed mode:

```text
The model can still turn the compact k-trace into a multi-step search loop under sampling.
It may emit several choose/check lines, then miss XML before the token cap.
```

Conclusion:

The compact CRT k-trace is the right interface direction and should replace the v2 candidate-list trace. It repairs parse stability, but it does not yet teach enough CRT capability for broad GRPO. The next bridge should either add more CRT SFT rows, simplify CRT further to answer-only plus one check, or use a smaller-modulus CRT curriculum before returning to full calibrated CRT rows.

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

The proper TRL GRPO smoke has passed, and the independent-versus-iso matrix replicated on a second seed on the stabilized two-family surface:

```text
seed 23 family_accuracy delta: +0.0625
seed 37 family_accuracy delta: +0.0625
parse_complete_rate delta: 0.0000
sampled malformed completions: 0 in all arms
```

The broad all-family eval changed the next action. It showed that the current packed XML interface does not yet generalize to unseen calibrated family types:

```text
broad independent parse_complete_rate: 0.9063
broad iso parse_complete_rate: 0.8906
rational_system_target parse_complete_rate: 0.0000 in both arms
```

The first broad all-traces bridge repaired that interface failure for the original broad trace set:

```text
broad bridge deterministic parse_complete_rate: 1.0000
broad bridge sampled parse_complete_rate: 0.9922
```

The v2 CRT candidate-walk bridge failed because comma-separated candidate lists caused sequence loops and dropped deterministic parse to `0.8750`. The v3 compact CRT k-trace fixed that regression:

```text
v3 deterministic parse_complete_rate: 1.0000
v3 sampled parse_complete_rate: 0.9883
v3 answer_count_mismatch_rate: 0.0117
```

Do not sweep `lambda_iso` yet. The next blocker is not global parser compliance; it is broad-family capability and correctness contrast. `chinese_remainder` is still `0.0000` sampled accuracy with no correctness-level prompt contrast. `rational_system_target` has only `0.0208` sampled accuracy and one correctness-contrast prompt.

Before broad GRPO, improve the new-family bridge so the new family types produce enough sampled correctness for reward learning. The next highest-value fix is likely a CRT curriculum or a simpler CRT target that does not invite sampled search loops, plus a separate rational-system arithmetic simplification pass.

The `rational_linear_equation` baseline is still known and should be improved later as a targeted recovery workstream, but the immediate blocker is new-family capability and sampled contrast.

The current Phase 6 thesis:

```text
Same stable reward interface.
Real GRPO trainer confirmed.
Iso-RLVR comparison replicated on two seeds.
Broad parser interface repaired with all-family traces.
Compact CRT trace repaired v2 parser regression.
New-family capability and correctness contrast must improve before broad GRPO.
```
