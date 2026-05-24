# Research Plan

## Hypothesis

Standard RLVR can reward instance-level answer hacks. Isomorphic RLVR should prefer policies that solve the underlying rule across transformed variants.

## Phase 1: Baseline Harness

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

## Phase 2: Independent RLVR

Train with per-problem correctness reward:

```text
reward_i = 1 if extracted_answer_i == gold_i else 0
```

This is the control condition.

## Phase 3: Iso-RLVR

Train with family-level reward:

```text
reward_i = correctness_i + lambda_iso * family_consistency
```

For a family, `family_consistency = 1` only when all sampled answers match their gold answers or match an explicitly known transformation relation.

Start with strict family correctness. Then add partial consistency variants.

## Phase 4: Robustness Tests

Evaluate on:

- Seen family types, unseen parameters
- Held-out family types
- Distractor wording
- Symbol renaming
- Larger numbers
- Adversarial surface forms

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

