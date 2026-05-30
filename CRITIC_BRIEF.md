# Critic Brief: Iso-RLVR

Date: 2026-05-29

## Executive Summary

The current result is the Phase 6 narrow Iso-RLVR result on a stabilized packed XML verifier interface.

The main comparison uses:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Trainer: TRL GRPOTrainer
Initialization: Phase 5 all-traces LoRA adapter
Families: missing_average, rational_linear_equation
Prompt shape: two-variant packed XML
Reward surface: strict XML parse plus correctness plus optional family component
```

Phase 6 result table:

| Run | Accuracy | Family accuracy | Parse complete | Sampled malformed |
| --- | ---: | ---: | ---: | ---: |
| Base all-traces adapter | 0.7500 | 0.6250 | 1.0000 | 0 |
| Independent 30-step, seed 23 | 0.7188 | 0.5625 | 1.0000 | 0 |
| Iso `lambda_iso=0.25`, seed 23 | 0.7500 | 0.6250 | 1.0000 | 0 |
| Iso `lambda_iso=0.50`, seed 23 | 0.7500 | 0.6250 | 1.0000 | 0 |
| Iso `lambda_iso=1.00`, seed 23 | 0.7813 | 0.6875 | 1.0000 | 0 |
| Independent 30-step, seed 37 | 0.7188 | 0.5625 | 1.0000 | 0 |
| Iso `lambda_iso=0.50`, seed 37 | 0.7500 | 0.6250 | 1.0000 | 0 |

The important result is the monotonic family-accuracy trend in the seed-23 lambda sweep:

```text
Independent: 0.5625
lambda_iso=0.25: 0.6250
lambda_iso=0.50: 0.6250
lambda_iso=1.00: 0.6875
```

Best observed narrow result:

```text
lambda_iso=1.00
accuracy delta over independent: +0.0625
family_accuracy delta over independent: +0.1250
parse_complete_rate delta: 0.0000
```

This is a clean, narrow result. It should not be presented as broad mathematical-reasoning generalization.

## Research Question

The baseline RLVR question is:

```text
Did the model produce the correct final answer for this packed prompt?
```

The Iso-RLVR question is:

```text
Did the model produce correct answers consistently across isomorphic variants from the same latent family?
```

The working hypothesis is that an isomorphic family reward should prefer policies that solve the shared latent structure rather than policies that get isolated variants right by brittle local patterns.

## Current Result

The clean Phase 6 result has three parts.

First, `lambda_iso=0.50` replicated across two seeds:

| Seed | Independent accuracy | Iso accuracy | Independent family accuracy | Iso family accuracy | Family delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 |
| 37 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 |

For seed 37, independent and iso had identical contrast-driver prompt groups. That matters because the result is not explained by the iso arm receiving better contrast prompts.

Second, the seed-23 lambda sweep was monotonic in family accuracy:

| Arm | Accuracy | Family accuracy | Family delta over independent |
| --- | ---: | ---: | ---: |
| Independent | 0.7188 | 0.5625 | 0.0000 |
| Iso `lambda_iso=0.25` | 0.7500 | 0.6250 | +0.0625 |
| Iso `lambda_iso=0.50` | 0.7500 | 0.6250 | +0.0625 |
| Iso `lambda_iso=1.00` | 0.7813 | 0.6875 | +0.1250 |

Third, all narrow Phase 6 runs preserved the verifier interface:

```text
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
sampled malformed completions: 0
```

By-family breakdown for the seed-23 sweep:

| Arm | `missing_average` acc. | `missing_average` family acc. | `rational_linear_equation` acc. | `rational_linear_equation` family acc. |
| --- | ---: | ---: | ---: | ---: |
| Independent | 0.9167 | 0.8333 | 0.6000 | 0.4000 |
| Iso `lambda_iso=0.25` | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso `lambda_iso=0.50` | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso `lambda_iso=1.00` | 1.0000 | 1.0000 | 0.6500 | 0.5000 |

Interpretation:

```text
missing_average benefits strongly from the family reward.
rational_linear_equation improves only at lambda_iso=1.00.
```

## What Changed From Early Runs

The early Phase 2/3 result used a local `grpo_lite` trainer, single-problem prompts, and a loose final-answer extractor. Those runs were useful historically, but Phase 4 and Phase 5 showed that the reward interface was not reliable enough for a strong claim.

The current result is different:

```text
grpo_lite -> TRL GRPOTrainer
single-problem prompt -> packed two-variant XML prompt
loose answer extraction -> strict XML answer parser
unstable reward interface -> Phase 5 SFT-stabilized XML contract
```

The old Phase 2/3 brief is preserved in `CRITIC_BRIEF_phase2.md`. It should be treated as historical context only.

## Model And Runtime

Primary model:

```text
Qwen/Qwen2.5-Math-1.5B
```

Policy initialization:

```text
outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
```

This adapter was trained in Phase 5 to emit deterministic traces followed by strict XML answers. It is not the GRPO-lite output. It is the clean RL initialization.

Runtime stack:

```text
Environment: pytorch_5070ti
trl: 0.17.0
transformers: 5.0.0.dev0
accelerate: 1.10.0
datasets: 4.0.0
peft: 0.17.0
```

Phase 6 training configuration:

```text
Trainer: TRL GRPOTrainer
Steps: 30
Num generations: 4
Temperature: 0.7
Max completion length: 256
Learning rate: 1e-6
Beta: 0.04
LoRA adapter training
```

## Code Architecture

Current files that matter:

| Area | File | Responsibility |
| --- | --- | --- |
| Packed XML parser | `src/iso_rlvr/rewards/packed_answer.py` | Extracts strict XML answer blocks and rejects malformed answer values. |
| Packed reward | `src/iso_rlvr/rewards/packed_iso.py` | Implements `score_packed_completion`, format reward, penalties, correctness, and optional family component. |
| Packed eval | `src/iso_rlvr/eval/run_packed_eval.py` | Runs deterministic packed heldout eval and by-family summaries. |
| Rollout audit | `src/iso_rlvr/eval/run_packed_rollout_audit.py` | Samples completions before training and measures parse stability, reward variance, and contrast. |
| Packed diagnostics | `src/iso_rlvr/eval/packed_diagnostics.py` | Tracks parse completeness, answer-count mismatch, suspicious cases, repeats, and malformed samples. |
| SFT dataset builder | `src/iso_rlvr/data/build_format_sft_dataset.py` | Builds XML answer-only, trace-to-XML, all-traces, and think-bridge datasets. |
| TRL GRPO trainer | `src/iso_rlvr/train/packed_grpo_trl.py` | Runs proper TRL `GRPOTrainer`, wraps the stateless packed reward function, logs reward records. |
| GRPO-lite trainer | `src/iso_rlvr/train/packed_grpo_lite.py` | Historical smoke path only; not the current result path. |

Critical implementation point:

```text
The TRL reward function is stateless. Dataset columns carry gold_answers, family_id,
family_type, variant_ids, and num_variants directly into the reward function.
```

## Dataset Design

The current Phase 6 result uses packed rows. Each prompt contains two related variants from the same family and expects two verifier-compatible XML answers.

Output contract:

```xml
<answers>
<answer_1>33</answer_1>
<answer_2>16/5</answer_2>
</answers>
```

Main Phase 6 train/eval data:

```text
Train dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
Eval dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
Families used in the narrow result: missing_average, rational_linear_equation
```

Each row carries:

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

The broader calibrated family types:

```text
chinese_remainder
rational_system_target
```

are not part of the main Phase 6 claim. They are parse-stable under later bridge work but not yet RL-ready because sampled correctness and correctness-level prompt contrast remain too weak.

## Reward Design

The current reward is based on `score_packed_completion`.

Independent reward:

```text
correctness + format reward - penalties
family bonus disabled
```

Iso reward:

```text
correctness + format reward + family component - penalties
```

Family component weights:

| Lambda | `family_mean_weight` | `all_family_correct_weight` |
| ---: | ---: | ---: |
| 0.25 | 0.125 | 0.125 |
| 0.50 | 0.250 | 0.250 |
| 1.00 | 0.500 | 0.500 |

Critical rule:

```text
No correctness credit unless the XML answer interface is parse-complete.
```

Interface metrics tracked:

```text
parse_complete_rate
answer_count_mismatch_rate
suspicious_rate
sampled malformed completions
reward mean/std
contrast prompt count
by-family-type accuracy and family accuracy
```

## What A Critic Should Challenge

The main limitations are:

1. The main heldout set is small.
2. The claim covers only two procedural family types.
3. The best `lambda_iso=1.00` result has not yet been replicated across multiple seeds.
4. The two-seed replication is for `lambda_iso=0.50`, not for every lambda.
5. Training runs are short: 30 GRPO steps.
6. The result is local to `Qwen/Qwen2.5-Math-1.5B` plus LoRA.
7. The reward checks final answers only, not reasoning validity.
8. The XML interface is engineered and may not transfer unchanged to other answer contracts.
9. Broad calibrated families are not RL-ready under the current bridge.
10. No bootstrap confidence interval over families has been reported.
11. No final blind benchmark untouched by iteration has been run.

The correct claim is narrow:

```text
Iso-RLVR improves family accuracy over independent RLVR on the stabilized two-family packed XML interface.
```

## Phase 7 Status

Phase 7 tested a future-work architecture:

```xml
<think>
free reasoning
</think>
<answers>
<answer_1>...</answer_1>
<answer_2>...</answer_2>
</answers>
```

The verifier scores only `<answers>` and ignores `<think>`.

Phase 7 established:

```text
Parser/reward isolation for <think> works.
Minimal-think SFT teaches the tag surface but destroys too much math behavior.
Hybrid-think SFT preserves working-family capability better.
Hard-family correctness contrast remains too sparse for meaningful GRPO.
Working-family deterministic eval remains stable after a tiny GRPO smoke.
Sampled working-family rollouts still fail the parse gate.
```

Known blocker:

```text
Response prefixing fixes the opening <think> tag, but sampled completions often stop after </think>
and fail to emit the required final <answers> block.
```

Phase 7 is closed as a deferred extension. It should be treated as future work, not as part of the main claim.

## Reproduction Commands

Install and test:

```bash
conda run -n pytorch_5070ti python -m pip install -e .
conda run -n pytorch_5070ti pytest tests
```

Run the TRL smoke:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_smoke.yaml
```

Run the seed-23 Phase 6 matrix:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_independent_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_25_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_1_00_30step.yaml
```

Run the seed-37 confirmation pair:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_independent_30step_seed_37.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step_seed_37.yaml
```

The trainer configs save adapters and run final heldout evaluation after training. Detailed run logs and interpretation are in `phase6.md`.

## Current Decision

Stop Phase 7 and move to the report.

Report the Phase 6 result:

```text
Phase 5 made the verifier interface reliable.
Phase 6 showed a clean, narrow Iso-RLVR family-accuracy gain under proper TRL GRPO.
Phase 7 produced useful diagnostics but remains future work.
```
