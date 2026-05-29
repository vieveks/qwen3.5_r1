# Phase 7: Free Reasoning With Structured Final Answers

Status: active implementation log and working spec

Date: 2026-05-29

## Executive Summary

Phase 6 completes the narrow Iso-RLVR result on the two working family types:

```text
missing_average
rational_linear_equation
```

That result should not be blocked by harder broad-family learnability. `chinese_remainder` and `rational_system_target` are now mostly parse-stable under the broad bridge, but they remain weak RL targets because sampled correctness and correctness-level prompt contrast are too low.

Phase 7 is the next architectural experiment for those harder families. The core idea is to stop forcing a deterministic trace format and instead let the model reason freely before emitting strict verifier-compatible answers.

Proposed output contract:

```xml
<think>
free reasoning
</think>
<answers>
<answer_1>23</answer_1>
<answer_2>11</answer_2>
</answers>
```

The verifier must score only the `<answers>` block.

## Motivation

Phase 6 showed two useful failure modes:

1. Deterministic traces can repair the final answer interface.
2. Deterministic traces can also become the task.

For CRT, the candidate-list trace repaired the algorithmic shape but taught open-ended continuation:

```text
Problem 1 candidates: 0, 15, 30, 45, ...
```

The model continued the numeric list until the token cap and often never reached XML. The compact `x = r + mk` trace fixed parse stability, but it still did not produce useful sampled correctness:

```text
chinese_remainder sampled accuracy: 0.0000
chinese_remainder correctness contrast: 0 prompts
```

For rational systems, the simplified elimination trace stayed parse-complete but did not transfer enough arithmetic capability:

```text
rational_system_target sampled accuracy: 0.0208
rational_system_target correctness contrast: 1 prompt
```

The likely issue is no longer just trace format. These families need flexible reasoning space.

## Core Design

Use a two-channel completion:

```text
reasoning channel: <think>...</think>
answer channel: <answers>...</answers>
```

The parser extracts only from `<answers>`.

The reward function ignores `<think>` completely:

```text
no direct reward for reasoning text
no parser fallback into reasoning text
no correctness credit from reasoning text
no format reward for reasoning style beyond tag presence
```

Reasoning improves only indirectly:

```text
better reasoning -> better final answers -> higher reward
```

## SFT Bridge Policy

The first SFT bridge should teach the tag contract, not a reasoning algorithm.

Initial target style:

```xml
<think>
Let me solve this.
</think>
<answers>
<answer_1>{gold_answers[0]}</answer_1>
<answer_2>{gold_answers[1]}</answer_2>
</answers>
```

Decision:

Use a minimal non-empty think anchor in SFT, not a literally empty block.

```text
Preferred anchor: Let me solve this.
```

Reason:

A completely empty think block may teach the model that the reasoning channel is disposable and increase the chance that sampled completions skip `<think>` entirely. A short generic anchor teaches the two-tag contract without imposing a deterministic reasoning algorithm.

Important:

- Do not pre-fill `<think>` with deterministic traces in the first bridge.
- Do not use family-specific or problem-specific reasoning in the initial SFT think block.
- Do not treat the placeholder text as semantically important.
- Do not generate programmatic CRT or rational-system reasoning traces for this bridge.
- Do not reward or parse the reasoning content.
- Keep the existing strict XML numeric grammar for final answers.

Rationale:

The Phase 6 broad bridge failed because deterministic trace content created brittle continuation patterns. Phase 7 should avoid recreating that failure mode.

## Parser Contract

Parser behavior:

```text
1. Locate the final <answers>...</answers> block.
2. Extract numbered answer tags from that block only.
3. Ignore all text inside <think>...</think>.
4. Reject malformed answer values exactly as in Phase 5/6.
5. Do not fall back to extracting answers from reasoning prose.
```

Safe examples:

```xml
<think>
I will try a few values.
x = 3 does not work. x = 8 works.
</think>
<answers>
<answer_1>8</answer_1>
<answer_2>8</answer_2>
</answers>
```

Unsafe behavior to reject:

```text
answer appears only in <think>
missing <answers> block
malformed numeric value inside <answer_1>
extra answer tags
```

## Reward Contract

Start with the Phase 6 reward shape:

```text
score_packed_completion
family_bonus_enabled: false for first smoke
format reward enabled
extra answer penalty enabled
no correctness credit unless answers are parse-complete
```

The reward should not inspect reasoning content.

Length penalty should have a floor:

```text
No length penalty below a minimum useful completion length.
Apply length penalty only above a ceiling.
```

Reason:

A penalty from token 1 would push GRPO toward empty `<think>` blocks. The desired behavior is concise but sufficient reasoning, not shortest possible completions.

Proposed first length policy:

```text
min_free_tokens_before_penalty: 128
soft_cap_tokens: 512
hard_cap_tokens: 768
```

This should be tuned after rollout data, not guessed permanently.

## Token Budget

The Phase 6 packed XML bridge used `max_completion_length: 256`.

That is likely too small for free reasoning. Phase 7 should start with:

```text
max_completion_length: 512
```

If CRT or rational-system traces still truncate before final XML:

```text
max_completion_length: 768
```

Before training, run a no-training rollout audit to measure:

- completion length distribution
- parse completeness
- fraction of completions reaching `<answers>`
- malformed modes
- GPU memory behavior at target batch size

## Initial Experiments

### Experiment 7.1: Parser And Reward Unit Tests

Goal:

```text
Ensure <think> content cannot leak into reward scoring.
```

Tests:

- Parses valid `<think>` plus `<answers>`.
- Ignores numeric-looking text inside `<think>`.
- Rejects completion with correct answer in `<think>` but missing `<answers>`.
- Rejects malformed values inside `<answers>`.
- Uses the final answer block if multiple answer-like strings appear earlier.

Pass condition:

```text
all parser and reward adapter tests pass
```

Result:

```text
Implemented.
Parser: src/iso_rlvr/rewards/packed_answer.py
Tests: tests/test_packed_answer.py, tests/test_packed_iso_reward.py
```

Parser behavior now strips well-formed `<think>...</think>` blocks before any answer extraction. XML parsing also uses only the final `<answers>...</answers>` block, so earlier answer-like strings or draft answer blocks cannot override the verifier-facing final answer block.

The tests verify that:

- valid `<think>` plus `<answers>` parses correctly
- `Answer 1: ...` and `\boxed{...}` inside `<think>` are ignored
- completions with correct answers only inside `<think>` receive no correctness credit
- malformed values inside `<answers>` are still rejected
- the final `<answers>` block is the only XML answer block used for scoring

Targeted test result:

```text
tests/test_packed_answer.py: 20 passed
tests/test_packed_iso_reward.py: 11 passed
tests/test_packed_grpo_trl.py: 5 passed
```

Broader reward/eval regression result:

```text
tests/test_packed_diagnostics.py: 9 passed
tests/test_packed_eval_summary.py: 5 passed
tests/test_packed_rollout_audit.py: 4 passed
tests/test_packed_grpo_lite.py: 4 passed
```

Conclusion:

The Phase 7 parser/reward isolation gate is passed. The verifier now scores only final answers and does not leak reward through the reasoning channel.

### Experiment 7.2: Minimal-Think SFT Bridge

Goal:

```text
Teach the model to emit both tag blocks without teaching a deterministic reasoning trace.
```

Target:

```xml
<think>
Let me solve this.
</think>
<answers>
...
</answers>
```

Start from:

```text
outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
```

or, if the tag contract disrupts too much behavior:

```text
Qwen/Qwen2.5-Math-1.5B base with LoRA
```

Pass condition:

```text
parse_complete_rate >= 0.95
answer_count_mismatch_rate <= 0.05
```

This experiment does not need to improve accuracy.

### Experiment 7.3: No-Training Sampled Rollout Audit

Goal:

```text
Check whether free reasoning produces sampled correctness contrast on hard families before GRPO.
```

Eval surface:

```text
chinese_remainder
rational_system_target
```

Metrics:

- parse complete rate
- answer-count mismatch
- sampled accuracy
- correctness-level contrast prompts
- malformed full completions
- completion length distribution

Pass condition before GRPO:

```text
parse_complete_rate >= 0.90
at least one hard family has non-zero sampled accuracy
at least one hard family has correctness-level prompt contrast
malformed modes are not dominated by missing final <answers>
```

### Experiment 7.4: Tiny Think-GRPO Smoke

Run only after Experiment 7.3 passes.

Goal:

```text
Check whether GRPO updates improve final answers without collapsing the answer interface.
```

First reward:

```text
correctness + format only
family bonus disabled
```

Pass condition:

```text
post-update parse_complete_rate >= 0.95
no repeated malformed mode
hard-family sampled contrast remains nonzero
```

## Scope Boundary

Phase 7 is not needed to claim the narrow Phase 6 result.

Phase 6 claim:

```text
Iso-RLVR improves family accuracy over independent GRPO on the stable two-family packed XML interface.
```

Phase 7 question:

```text
Can a free reasoning channel make harder calibrated family types useful RL targets?
```

Do not mix these claims.

## Current Recommendation

Finish and preserve the Phase 6 narrow result as the main result.

Use Phase 7 to explore `<think> + <answers>` only after Phase 6 is documented and committed.

The Phase 7 thesis:

```text
Reward final answers only.
Let reasoning be free.
Do not turn the reasoning trace into the supervised task.
```
