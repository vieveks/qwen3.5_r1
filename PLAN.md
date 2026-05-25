# Research Plan

## Hypothesis

Standard RLVR can reward instance-level answer hacks. Isomorphic RLVR should prefer policies that solve the underlying rule across transformed variants.

## Phase 1: Baseline Harness

Status: baseline harness works and the calibrated dataset is the first usable RL candidate. See `phase1.md` for environment details, run outputs, dataset calibration, and conclusions.

Generate procedurally verified math families:

- Linear proportionality
- Two-step affine word problems
- Linear equations
- Modular arithmetic
- Unit conversion

Each family has a hidden parameterization and several surface variants. Every variant has an exact answer.

Evaluate `Qwen/Qwen2.5-Math-1.5B` on the held-out variants.

Record:

- `family_id`
- `variant_id`
- `family_type`
- `prompt`
- `answer`
- `model_response`
- `extracted_answer`
- `correct`
- `response_tokens`

Phase 1 result:

```text
recommended dataset: data/iso_math_calibrated.jsonl
profile: calibrated
baseline accuracy: 0.5625
baseline family_accuracy: 0.25
format_failure_rate: 0.0
```

The calibrated dataset is in the target first-RL range and shows the intended failure mode: instance-level accuracy is much higher than family-level consistency.

## Phase 1.5: Calibration And Guardrails

Before running real RL, keep the dataset and evaluation harness pinned:

- Dataset generation seed: `29`
- Dataset profile: `calibrated`
- Baseline config: `configs/baseline_eval_calibrated.yaml`
- Model: `Qwen/Qwen2.5-Math-1.5B`
- Prompt template: unchanged across baseline, independent reward, and Iso-RLVR runs
- Max tokens: unchanged within a comparison matrix

Do not draw conclusions from:

- `data/iso_math_small.jsonl`: too easy
- `data/iso_math_harder.jsonl`: still too easy at instance level
- `data/iso_math_challenge.jsonl`: too hard and slow for first RL

Use the challenge profile later as a stress test after the control and Iso-RLVR runs are stable.

## Phase 2: Independent RLVR

Train with per-problem correctness reward:

```text
reward_i = 1 if extracted_answer_i == gold_i else 0
```

This is the control condition.

Status: three-step independent calibrated smoke completed. The adapter saved and the log captured one nonzero-reward step. See `phase2.md`.

Minimum Phase 2 run:

```text
dataset_path: data/iso_math_calibrated.jsonl
reward_mode: independent correctness
held-out eval: calibrated profile with a different seed
report: accuracy, family_accuracy, wrong-answer tokens, format failures
```

## Phase 3: Iso-RLVR

Train with family-level reward:

```text
reward_i = correctness_i + lambda_iso * family_consistency
```

For a family, `family_consistency = 1` only when all sampled answers match their gold answers or match an explicitly known transformation relation.

Start with strict family correctness. Then add partial consistency variants.

Status: three-step Iso calibrated smoke with `lambda_iso = 0.50` completed. The adapter saved and the log captured nonzero rewards. See `phase2.md`.

Initial Iso-RLVR runs:

```text
lambda_iso = 0.25
lambda_iso = 0.50
lambda_iso = 1.00
```

The first success criterion is not higher reward. The first success criterion is better held-out family consistency at similar single-instance accuracy compared with independent RLVR.

## Phase 4: Robustness Tests

Evaluate on:

- Seen family types, unseen parameters
- Held-out family types
- Distractor wording
- Symbol renaming
- Larger numbers
- Adversarial surface forms

Add:

- Calibrated held-out seed
- Challenge profile stress set
- Family-type breakdown by reward condition
- Cases where single-instance correctness stays flat but family correctness improves

## Phase 5: Report

The report should answer:

- Does Iso-RLVR improve family-level consistency?
- Does it trade off single-instance accuracy?
- Does it reduce shortcut behavior?
- Does it improve pass@k or only pass@1?
- Are outputs shorter, longer, or unchanged?

## Success Criteria

Minimum interesting result:

```text
Iso-RLVR improves family consistency over independent RLVR at matched single-instance accuracy.
```

Strong result:

```text
Iso-RLVR improves held-out transformation robustness without increasing wrong-answer length.
```

