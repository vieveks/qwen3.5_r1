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

This means we cannot safely target a concrete `GRPOTrainer` API yet. The next trainer implementation step is to install TRL, record the exact installed version, and then check the local `GRPOTrainer` and `GRPOConfig` signatures before writing trainer code.

Reward adapter implemented:

```text
Module: src/iso_rlvr/train/packed_grpo_trl.py
Tests: tests/test_packed_grpo_trl.py
```

The adapter intentionally does not import TRL. It provides:

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

The stateless packed reward path is ready for TRL integration, but the trainer itself is blocked on installing and inspecting the exact TRL version.

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

Phase 6 should start with proper TRL GRPO integration from the Phase 5 all-traces adapter.

Do not tune `lambda_iso`, rational-linear traces, or the reward family bonus until a tiny TRL smoke proves that the production trainer preserves the stabilized XML interface.

The first Phase 6 thesis:

```text
Same stable reward interface.
Real GRPO trainer.
Then rerun Iso-RLVR comparisons.
```
