# Phase 4 Plan: Objective Correction Before More Scaling

Date: 2026-05-27

## Purpose

Phase 3 showed that the early Iso-RLVR signal did not survive full clean held-out evaluation. The correct next move is not a larger lambda sweep. The correct next move is to fix the training objective and reward sparsity before spending more GPU time.

Phase 4 asks:

```text
Does a family-aware verifiable reward improve transformation consistency when trained with a proper GRPO-style objective?
```

## Current Diagnosis

The project idea remains viable:

```text
Reward models for rule-level consistency across isomorphic problem families.
```

The current implementation is not enough to test it:

- `grpo_lite.py` is a useful scaffold, not a validated GRPO trainer.
- The Iso family bonus is too sparse when it only fires on all variants correct.
- The calibrated dataset contains family types that are either too easy or too hard.
- The early 80-row eval was too small; the full 800-row clean eval is now the benchmark.

Phase 4 should therefore be framed as objective correction, not result chasing.

## What `grpo_lite.py` Actually Did

The current trainer:

1. Samples completions.
2. Scores them with independent or Iso reward.
3. Normalizes rewards across the rollout batch.
4. Recomputes logprobs under the current policy.
5. Applies a normalized-reward weighted logprob loss through LoRA.

This is closer to batch-normalized REINFORCE / reward-weighted LoRA fine-tuning than production GRPO.

Missing pieces relative to proper GRPO:

- old-policy logprobs
- policy ratio
- PPO-style clipping
- reference-model KL or equivalent regularization
- prompt-level group-relative advantage over multiple completions
- careful separation between rollout policy and update policy

Do not make stronger claims from `grpo_lite.py`.

## Phase 4 Core Decision

Choose one training implementation path before writing code.

### Option A: Use TRL GRPO

Use TRL's GRPO trainer and plug in custom reward functions.

Pros:

- Less RL implementation risk.
- Easier to defend to critics.
- Faster path to a credible experiment.
- Lets the project focus on family rewards instead of trainer correctness.

Cons:

- May be awkward to express family-level rewards because rewards depend on related prompts.
- May require custom dataset packing or batch construction.
- Family-grouped advantage may not be directly supported.

### Option B: Use verl

Use verl for a more scalable RLVR-style setup.

Pros:

- Closer to serious RLVR infrastructure.
- Better path if the project later needs vLLM rollout acceleration.
- More natural for larger experiments.

Cons:

- More setup complexity.
- More moving parts before we know the reward works.
- Slower to iterate locally.

### Option C: Build Minimal True Grouped GRPO

Write a new custom trainer, separate from `grpo_lite.py`, that implements:

- grouped rollouts
- old logprobs
- policy ratio
- clipping
- optional KL to frozen reference model
- prompt-level and family-level grouping modes

Pros:

- Maximum control over family grouping.
- The code directly expresses the research idea.
- Easier to instrument every quantity.

Cons:

- Highest bug risk.
- Harder to defend unless carefully tested.
- Slower to reach a reliable result.

Recommended starting point:

```text
Option A first: TRL GRPO with custom rewards.
Keep Option C only if TRL cannot support the family reward structure cleanly.
```

## TRL Integration Design

The first TRL implementation should use family-packed prompts.

One training row should contain one latent family:

```text
Problem 1: ...
Problem 2: ...
Problem 3: ...
Problem 4: ...
```

The model should be instructed to answer in exactly this canonical format:

```text
Answer 1: <number>
Answer 2: <number>
Answer 3: <number>
Answer 4: <number>
```

This makes the reward function compatible with TRL's per-completion reward interface: each generated completion receives one scalar reward computed from all answers inside that packed completion.

This is intentionally a Phase 4A approximation. It tests:

```text
within-generation family consistency
```

It does not fully test:

```text
cross-generation robustness across independently sampled variants
```

That distinction matters. In a packed prompt, the model's answer to later problems is conditioned on earlier problems and earlier answers. This can encourage useful internal consistency, but it can also create copy/repetition and position-bias artifacts. A later true Family-GRPO implementation should test independent variant robustness with separate prompts.

### Packed Answer Parser

Before any trainer work, implement and test a parser for packed completions.

Primary accepted format:

```text
Answer 1: 3
Answer 2: 7
Answer 3: 21
Answer 4: 4
```

Fallbacks may support:

- `1) 3`
- `Problem 1: 3`
- `Answer 1 is 3`
- one boxed answer per answer line

Parser requirements:

- return exactly one extracted answer per expected variant when possible
- mark missing answers explicitly
- mark extra answers explicitly
- preserve answer order
- expose parser failure rate in summaries

Do not run RL if packed answer parsing is noisy or untested.

### Packed Prompt Degenerate Modes

The Phase 4A smoke test must inspect degenerate outputs:

- same answer repeated for every variant
- answer copied from an earlier variant
- only the first answer produced
- later answers missing because of token cap
- answer count differs from variant count
- later variant accuracy is much lower than earlier variant accuracy
- wrong answers are suspiciously correlated

Suspiciously correlated wrong answers include:

```text
all wrong answers differ from gold by the same constant
all wrong answers are a fixed multiple of gold
all wrong answers equal one earlier predicted answer
```

If family consistency improves while row accuracy does not, inspect these cases before treating the result as evidence of rule learning.

## Reward Design

Phase 4 should replace the sparse all-or-nothing family bonus with a staged reward.

### Tier 1: Exact Answer Correctness

Keep the current numeric verifier.

```text
correctness = 1.0 if extracted_answer == gold_answer else 0.0
```

This remains the anchor. Do not train without it.

### Tier 2: Format Reward

Add a small reward for extractable final answers.

```text
format_reward = +0.05 if answer is extractable else -0.10
```

Purpose:

- Prevent unparseable completions.
- Improve reward reliability.
- Avoid making format more important than correctness.

### Tier 3: Partial Family Consistency

Replace the sparse all-or-nothing bonus with a partial family score.

```text
family_mean = correct_variants_in_family / total_variants_in_family
all_family_correct = 1.0 if family_mean == 1.0 else 0.0
```

Default reward:

```text
reward = correctness
       + correctness * 0.25 * family_mean
       + correctness * 0.25 * all_family_correct
       + format_reward
```

Reasoning:

- `family_mean` gives denser signal, but only reinforces correct trajectories.
- `all_family_correct` preserves the original strict Iso objective.
- Correctness remains the dominant term.
- Wrong answers should not receive positive family bonuses just because sibling variants were correct.

Alternative reward variants can be ablated later, but the first serious Phase 4 run should use correctness-gated family bonuses.

### Tier 4: Wrong-Length Penalty

Add a small penalty only for incorrect long answers.

```text
if incorrect:
    reward -= length_penalty_weight * min(response_tokens, token_cap) / token_cap
```

Initial value:

```text
length_penalty_weight = 0.05
```

Do not penalize correct long answers in the first serious run.

### Tier 5: Residual Verifiers

Prefer symbolic residual checks over prose/process rewards.

Examples:

- Linear equation: plug predicted `x` into `a*x + b = c`.
- Missing average: plug predicted value into the average formula.
- Chinese remainder: check every congruence.
- Rational system target: check the target relation implied by metadata.

Candidate reward:

```text
residual_reward = 0.25 if predicted answer satisfies the equation constraints else 0.0
```

This may equal correctness for some tasks, but for richer metadata it can become a more direct rule verifier.

### Tier 6: Transformation-Invariant Reward

This is the most scientifically interesting reward, but it should not be added first.

Question:

```text
Do predicted answers transform according to the known latent family relation?
```

Warning:

Rewarding consistency when all answers are wrong can teach systematic wrongness.

Safer first version:

```text
transformation_reward only activates if at least one or two variants are correct.
```

This should be a later Phase 4.2 experiment, not the first Phase 4 run.

## Grouping Design

There are two plausible GRPO group definitions.

### Design A: Prompt-GRPO + Family Reward

For each prompt, sample multiple completions and compute group-relative advantage within that prompt.

The reward includes family information, but advantage normalization is standard prompt-level GRPO.

Pros:

- Closest to standard GRPO.
- Lower implementation risk.
- Easier to compare against independent GRPO.

Cons:

- Family signal enters only through the scalar reward.
- Does not fully test family-level group-relative optimization.

### Design B: Family-GRPO

Group all variants from the same latent family, possibly with multiple completions per variant.

Compute advantages across the family group.

Pros:

- Directly tests the Iso-RLVR idea.
- Encourages comparison across isomorphic variants.

Cons:

- May compare easy and hard variants unfairly.
- Could introduce difficulty bias.
- Less standard; needs careful ablations.

Recommended first implementation:

```text
Run Design A first.
Then test Design B as an experimental variant.
```

This gives a clean baseline before introducing a novel grouping rule.

### Phase 4C Open Design Question

Before implementing true Family-GRPO, choose the group construction explicitly.

Candidate 1:

```text
group = all variants from one family
one or more completions per variant
advantages computed across the family group
```

Candidate 2:

```text
group = all completions for one variant
standard prompt-GRPO grouping
reward includes family-level signal from sibling variants
```

These produce different gradients and answer different questions. Candidate 1 is the stronger Iso-RLVR intervention, but it may compare easy and hard variants unfairly. Candidate 2 is closer to standard GRPO and is easier to defend as a baseline.

Do not let Phase 4C inherit an implicit grouping assumption from Phase 4A.

## Dataset Plan

Phase 3 showed the calibrated dataset has a sharp difficulty split.

Full clean held-out results:

| Family type | Base accuracy | Base family accuracy | Interpretation |
| --- | ---: | ---: | --- |
| `rational_linear_equation` | 0.7679 | 0.3673 | learnable |
| `missing_average` | 0.6627 | 0.1905 | useful medium difficulty |
| `chinese_remainder` | 0.0268 | 0.0000 | too hard currently |
| `rational_system_target` | 0.0000 | 0.0000 | too hard currently |

Training should not be dominated by unsolved families with all-zero rewards.

### Phase 4 Training Curriculum

Stage 1:

```text
rational_linear_equation
missing_average
```

Stage 2:

```text
add easier chinese_remainder variants
```

Stage 3:

```text
add rational_system_target only after signal exists
```

Evaluation should still report all family types on the full clean held-out set.

## Generation Settings

The full eval showed that hard wrong answers often hit the `256` token cap.

Phase 4 should test:

```text
max_new_tokens: 256 for continuity
max_new_tokens: 512 for hard-family diagnostics
```

Do not switch all headline comparisons to 512 until baseline and independent reward are rerun at 512. Token cap changes can alter both accuracy and length metrics.

## Experiment Matrix

### Phase 4A: Family-Packed TRL Smoke Tests

Run first:

```text
Packed prompt parser tests
Packed reward tests
Packed prompt deterministic base eval
Proper GRPO independent packed reward, 10-20 steps
Proper GRPO partial Iso packed reward, 10-20 steps
```

Purpose:

- Verify rewards are non-constant.
- Verify training loss is finite.
- Verify adapters save and load.
- Verify no format collapse.
- Verify no answer-copying or repeat-answer collapse.
- Verify parser failures are rare.

### Phase 4B: Proper Packed Comparison

Run only after Phase 4A is stable.

Pilot:

```text
200 steps
4 rollouts per packed prompt
256 max tokens unless parser diagnostics show truncation
```

First meaningful comparison:

```text
500 steps
4 or 8 rollouts per packed prompt, depending on timing
256 or 512 max tokens, matched across compared runs
```

The 200-step run is a pilot, not a result to overclaim.

### Phase 4C: True Family-GRPO

Run only after packed TRL experiments have established a clean baseline.

Goal:

```text
test cross-generation robustness across independent family variants
```

### Phase 4D: Transformation-Invariant Reward

Run only after a metadata and difficulty audit.

Audit requirements:

- gold answers are not accidentally identical within many families
- answer transformations are nontrivial
- metadata cleanly encodes transformation relations
- variant difficulty is roughly balanced
- base accuracy by variant position is measured

## First Real Phase 4 Run

Minimum serious packed run:

```text
Base
Proper GRPO + independent reward
Proper GRPO + partial family reward
Proper GRPO + partial family reward + wrong-length penalty
```

Training scale:

```text
500 steps minimum
same seed first
then 3 seeds for the best candidate
```

The 200-step run is only a pilot for timing, reward variance, parser reliability, and collapse detection. Do not report it as the main result unless the effect is unusually large and then immediately replicate at 500 steps.

Primary eval:

```text
full clean held-out, 800 rows
```

Secondary eval:

```text
by-family-type breakdown
wrong-answer length
family pass@k if available
challenge profile as stress test only
```

## Main Metrics

Primary:

- row accuracy
- family accuracy
- family-type breakdown

Secondary:

- wrong-answer token length
- format failure rate
- average reward during training
- reward variance
- per-family reward sparsity
- number of families receiving nonzero family bonus

Do not judge by training reward alone.

## Decision Rules

Continue past the first serious run only if:

```text
partial family reward >= independent reward on family accuracy
with no meaningful row-accuracy regression
```

Stop and inspect if:

```text
family accuracy does not improve
row accuracy drops by more than 1 point
wrong-answer length increases
format failures increase
```

Escalate to Family-GRPO only after Prompt-GRPO + family reward has a clean baseline.

## What Would Count As A Good Result

Weak positive:

```text
family accuracy improves by 1-2 points over independent reward
row accuracy is flat
```

Strong positive:

```text
family accuracy improves by 3+ points
row accuracy is flat or improved
wrong-answer length does not increase
gain appears in at least two family types
```

Negative but useful:

```text
proper GRPO improves row accuracy but not family accuracy
```

This would suggest Iso-RLVR needs a better transformation reward, not just family correctness aggregation.

## Open Questions For Critique

1. Should the first proper GRPO implementation use TRL or verl?
2. Is Prompt-GRPO + family reward the right first baseline, or should Family-GRPO be implemented immediately?
3. Should transformation-consistency rewards ever activate when all predictions are wrong?
4. Should training exclude the currently unsolved family types, or include them with lower sampling probability?
5. Is `512` max tokens necessary for training, or should it be used only in diagnostic eval?
6. How many steps and seeds are enough before updating the public claim?

## Recommended Phase 4 Implementation Order

The canonical order is the subphase sequence above:

```text
Phase 4A: family-packed TRL smoke tests
Phase 4B: packed comparison with 200-step pilot and 500-step first real run
Phase 4C: true Family-GRPO design and implementation
Phase 4D: transformation-invariant reward after metadata/difficulty audit
```

Do not implement all ideas at once. The first files to add should be the packed dataset builder, packed answer parser, packed reward tests, and degenerate-output diagnostics. Trainer work comes after these pass.

## Phase 4 Claim Discipline

Until Phase 4 succeeds, the honest project claim is:

```text
We built a measurement harness for testing family-level transformation consistency in RLVR.
The initial custom trainer did not produce a robust improvement under full clean held-out evaluation.
The next phase tests whether the idea survives under proper GRPO and denser verifiable family rewards.
```
