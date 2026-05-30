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

Result:

```text
Implemented.
Builder mode: --minimal-think
Train dataset: outputs/phase7/format_sft_pair_xml_minimal_think_train.jsonl
Heldout dataset: outputs/phase7/format_sft_pair_xml_minimal_think_heldout.jsonl
Packed train: outputs/phase7/packed_calibrated_train_xml_pair_minimal_think_train.jsonl
Packed heldout: outputs/phase7/packed_calibrated_train_xml_pair_minimal_think_heldout.jsonl
Train rows: 180
Heldout rows: 20
Family types: missing_average, rational_linear_equation, rational_system_target, chinese_remainder
```

Training config:

```text
Config: configs/format_sft_qwen25_math_1_5b_minimal_think_all_families_from_phase5_1epoch.yaml
Base model: Qwen/Qwen2.5-Math-1.5B
Initial adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Rows: 180
Epochs: 1
Max steps: 90
Batch size: 2
Learning rate: 1e-4
Max sequence length: 1536
Output: outputs/phase7/format_sft_qwen25_math_1_5b_minimal_think_all_families_from_phase5_1epoch/adapter_or_model
```

Final logged training losses:

| Step | Loss |
| ---: | ---: |
| 80 | 0.0493 |
| 81 | 0.1178 |
| 82 | 0.0636 |
| 83 | 0.1040 |
| 84 | 0.0680 |
| 85 | 0.1033 |
| 86 | 0.0521 |
| 87 | 0.1011 |
| 88 | 0.0640 |
| 89 | 0.0825 |

Deterministic broad eval:

```text
Config: configs/packed_eval_phase7_minimal_think_bridge_512.yaml
Dataset: outputs/phase6/packed_calibrated_heldout_clean_xml_pair_64_seed0.jsonl
Max new tokens: 512
examples: 64
variant_examples: 128
accuracy: 0.1094
family_accuracy: 0.1094
parse_complete_rate: 0.9844
answer_count_mismatch_rate: 0.0156
suspicious_rate: 0.8906
think_block_rate: 0.8438
nontrivial_think_block_rate: 0.3281
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Think block | Non-trivial think |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `missing_average` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9474 | 0.0526 |
| `rational_linear_equation` | 0.2500 | 0.2500 | 1.0000 | 0.0000 | 0.8929 | 0.7143 |
| `rational_system_target` | 0.0000 | 0.0000 | 0.8333 | 0.1667 | 0.0000 | 0.0000 |

Representative behavior:

```xml
<think>
Let me solve this.
</think>
<answers>
<answer_1>60</answer_1>
<answer_2>60</answer_2>
</answers>
```

Gold:

```text
["63", "63"]
```

Another rational-linear completion showed the model extending the anchor but not doing real reasoning:

```xml
<think>
Let me solve this.
Let me think...
</think>
<answers>
<answer_1>7/2</answer_1>
<answer_2>7/2</answer_2>
</answers>
```

Gold:

```text
["11/2", "11/2"]
```

Conclusion:

The minimal-think bridge passed the narrow interface gate:

```text
parse_complete_rate: 0.9844
answer_count_mismatch_rate: 0.0156
```

But it failed as a capability-preserving bridge. Accuracy collapsed from the Phase 6 broad v3 bridge baseline (`0.5703`) to `0.1094`, and the two previously working families regressed badly. The generic think anchor taught the tag surface, but it also removed the deterministic reasoning paths that were carrying arithmetic.

This result should not be treated as a successful Phase 7 initialization for GRPO. The sampled audit below is still useful as a confirmation that there is no hard-family learning signal.

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

Result:

```text
Config: configs/packed_rollout_audit_phase7_minimal_think_hard_sampled.yaml
Dataset: outputs/phase7/packed_hard_heldout_clean_xml_pair_17_seed0.jsonl
Adapter: outputs/phase7/format_sft_qwen25_math_1_5b_minimal_think_all_families_from_phase5_1epoch/adapter_or_model
Hard-family prompts: 17
Samples per prompt: 4
Samples: 68
Max new tokens: 512
Temperature: 0.7
```

Result:

```text
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.9559
answer_count_mismatch_rate: 0.0441
suspicious_rate: 0.9559
reward_mean: 0.0369
reward_std: 0.0366
contrast_prompt_count: 0
think_block_rate: 0.6029
nontrivial_think_block_rate: 0.1471
passes_audit_gate: false
```

By family type:

| Family type | Accuracy | Parse complete | Mismatch | Think block | Non-trivial think |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 1.0000 | 0.0000 | 0.7045 | 0.0455 |
| `rational_system_target` | 0.0000 | 0.8750 | 0.1250 | 0.4167 | 0.3333 |

Observed malformed rational-system pattern:

```text
The model wrote algebraic expressions containing x or y inside <answer_*> tags.
The strict numeric grammar correctly rejected those values.
```

Example rejected answer values:

```xml
<answer_1>-39/40 - y / 2</answer_1>
<answer_2>23/20 - 9y</answer_2>
```

Conclusion:

The minimal-think bridge does not create a usable hard-family RL signal:

```text
hard-family sampled accuracy: 0.0000
contrast_prompt_count: 0
reward_std: 0.0366
```

Do not run think-GRPO from this adapter. GRPO would have no correctness contrast to optimize and would mostly reinforce shallow formatting behavior.

### Experiment 7.4: Hybrid-Think SFT Bridge

Goal:

```text
Preserve the working-family reasoning behavior while adding the two-tag Phase 7 contract.
```

Rationale:

The minimal-think bridge repeated the Phase 5 answer-only failure mode. It taught the output contract, but the generic anchor:

```text
Let me solve this.
```

was semantically empty. The model learned to emit `<think> + <answers>` without preserving the arithmetic paths that made the Phase 5 all-traces adapter useful.

The hybrid bridge therefore uses different think-block content by family role:

| Family type | Think-block policy |
| --- | --- |
| `missing_average` | Existing deterministic Phase 5 trace inside `<think>` |
| `rational_linear_equation` | Existing deterministic Phase 5 trace inside `<think>` |
| `chinese_remainder` | Generic anchor plus problem restatement |
| `rational_system_target` | Generic anchor plus problem restatement |

Important:

```text
The hard-family think blocks do not contain generated CRT or system-solving algorithms.
They only contain enough context to keep the reasoning channel non-empty.
```

Implementation:

```text
Builder flags: --hybrid-think --think-prompt
Train dataset: outputs/phase7/format_sft_pair_xml_hybrid_think_train.jsonl
Heldout dataset: outputs/phase7/format_sft_pair_xml_hybrid_think_heldout.jsonl
Packed train: outputs/phase7/packed_calibrated_train_xml_pair_hybrid_think_train.jsonl
Packed heldout: outputs/phase7/packed_calibrated_train_xml_pair_hybrid_think_heldout.jsonl
Family types: missing_average, rational_linear_equation, rational_system_target, chinese_remainder
```

Training config:

```text
Config: configs/format_sft_qwen25_math_1_5b_hybrid_think_all_families_from_phase5_1epoch.yaml
Base model: Qwen/Qwen2.5-Math-1.5B
Initial adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Epochs: 1
Batch size: 2
Learning rate: 1e-4
Max sequence length: 1536
Output: outputs/phase7/format_sft_qwen25_math_1_5b_hybrid_think_all_families_from_phase5_1epoch/adapter_or_model
```

Deterministic broad eval:

```text
Config: configs/packed_eval_phase7_hybrid_think_bridge_512.yaml
Dataset: outputs/phase7/packed_calibrated_heldout_clean_think_pair_64_seed0.jsonl
Max new tokens: 512
examples: 64
variant_examples: 128
accuracy: 0.5312
family_accuracy: 0.4219
parse_complete_rate: 0.9688
answer_count_mismatch_rate: 0.0312
suspicious_rate: 0.3125
think_block_rate: 0.9688
nontrivial_think_block_rate: 0.9688
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Think block | Non-trivial think |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `missing_average` | 0.8158 | 0.7368 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.6607 | 0.4643 | 0.9286 | 0.0714 | 0.9286 | 0.9286 |
| `rational_system_target` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Sampled hard-family audit:

```text
Config: configs/packed_rollout_audit_phase7_hybrid_think_hard_sampled.yaml
Dataset: outputs/phase7/packed_hard_heldout_clean_think_pair_17_seed0.jsonl
Samples per prompt: 4
Samples: 68
Max new tokens: 512
Temperature: 0.7
accuracy: 0.0147
family_accuracy: 0.0147
parse_complete_rate: 0.9265
answer_count_mismatch_rate: 0.0735
suspicious_rate: 0.9559
reward_mean: 0.0408
reward_std: 0.1299
contrast_prompt_count: 1
contrast_prompt_ids: ["fam_000009"]
think_block_rate: 0.9559
nontrivial_think_block_rate: 0.9118
passes_audit_gate: true
```

By hard family type:

| Family type | Accuracy | Parse complete | Mismatch | Think block | Non-trivial think |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0227 | 0.9318 | 0.0682 | 0.9545 | 0.9545 |
| `rational_system_target` | 0.0000 | 0.9167 | 0.0833 | 0.9583 | 0.8333 |

Observed failure modes:

- The rational-system family still emits algebraic expressions or prose inside `<answer_*>` tags.
- Some sampled completions are empty or fail to reach the final answer block.
- CRT can still fall back into repeated guessing until the token cap.
- Some malformed answers contain non-numeric markup such as `<sup>` inside answer tags.

Conclusion:

The hybrid bridge is a clear improvement over minimal-think SFT:

```text
minimal-think deterministic accuracy: 0.1094
hybrid-think deterministic accuracy: 0.5312
```

It preserves much more of the Phase 5 reasoning behavior while adding the two-tag contract. It also creates the first nonzero sampled hard-family signal:

```text
hard-family sampled accuracy: 0.0147
contrast_prompt_count: 1
reward_std: 0.1299
```

However, this is still a weak GRPO starting point. The audit passes technically, but the signal is concentrated in one CRT prompt, rational-system accuracy remains `0.0000`, mismatch is above the preferred gate, and suspicious rate remains very high. Treat this as a partial recovery, not as a ready mainline training setup.

### Experiment 7.5: Working-Family Think-GRPO Smoke

Goal:

```text
Check whether GRPO can update the hybrid-think adapter on the two working families without collapsing the two-tag interface.
```

Rationale:

Hard-family contrast is still too sparse for meaningful optimization. The cleaner Phase 7 question is therefore:

```text
Does the Phase 6 Iso-RLVR setup still behave sensibly when the policy has a <think> channel?
```

Before the independent-versus-iso comparison, run a small trainer smoke on the working families only.

Config:

```text
Config: configs/packed_grpo_trl_hybrid_think_working_smoke.yaml
Trainer: TRL GRPOTrainer
Base model: Qwen/Qwen2.5-Math-1.5B
Initial adapter: outputs/phase7/format_sft_qwen25_math_1_5b_hybrid_think_all_families_from_phase5_1epoch/adapter_or_model
Train dataset: outputs/phase7/packed_calibrated_train_xml_pair_hybrid_think_train.jsonl
Train family filter: missing_average, rational_linear_equation
Eval dataset: outputs/phase7/packed_calibrated_heldout_clean_think_pair_64_seed0.jsonl
Eval family filter: none
Steps: 10
Num generations: 4
Temperature: 0.7
Max completion length: 512
Learning rate: 1e-6
Beta: 0.04
Max grad norm: 1.0
Family bonus: disabled
```

First reward:

```text
correctness + format only
family bonus disabled
```

Implementation notes:

```text
Trainer now supports include_train_family_types and include_eval_family_types.
The smoke trains only on the two working families.
CRT and rational_system_target remain eval spectators.
TRL reward records now include think_block_rate and nontrivial_think_block_rate.
```

Sampled training-batch result:

```text
reward calls: 10
sampled completions: 40
variant_examples: 80
sampled accuracy: 0.7500
sampled family_accuracy: 0.6750
sampled parse_complete_rate: 0.8250
sampled answer_count_mismatch_rate: 0.1750
sampled suspicious_rate: 0.1750
sampled think_block_rate: 0.8250
sampled nontrivial_think_block_rate: 0.8250
steps with within-prompt contrast: 5 / 10
```

Training reward-call summary:

| Step | Reward mean | Reward std | Contrast prompts | Parse complete | Mismatch | Suspicious | Think block |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 3 | 0.6352 | 0.4726 | 1 | 0.7500 | 0.2500 | 0.2500 | 0.7500 |
| 4 | 0.5431 | 0.0002 | 0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5 | 0.4750 | 0.5750 | 1 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 6 | 0.3474 | 0.4824 | 1 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 7 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 8 | 0.7625 | 0.4980 | 1 | 0.7500 | 0.2500 | 0.2500 | 0.7500 |
| 9 | 1.0500 | 0.0000 | 0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 10 | 0.7625 | 0.4980 | 1 | 0.7500 | 0.2500 | 0.2500 | 0.7500 |

Observed sampled failure mode:

```text
Some sampled completions were empty.
The empty completions explain most of the parse and think-block failures.
```

Empty sampled completions appeared on:

| Step | Family | Empty samples |
| ---: | --- | ---: |
| 5 | `rational_linear_equation` | 2 |
| 6 | `rational_linear_equation` | 2 |

Representative think blocks stayed close to the SFT trace format:

```text
Step 1 missing_average:
Problem 1 sum needed...
Problem 1 known sum...
Problem 1 missing value...
```

```text
Step 10 rational_linear_equation:
Problem 1 isolate...
Problem 1 divide...
Problem 2 isolate...
Problem 2 divide...
```

There is no strong evidence yet that 10 steps of GRPO changed the reasoning style. The successful sampled completions mostly retained the supervised deterministic traces.

Standalone deterministic broad eval after saving:

```text
Config: configs/packed_eval_phase7_hybrid_think_working_smoke_final_512.yaml
Adapter: outputs/phase7/packed_grpo_trl_hybrid_think_working_smoke/adapter_or_model
Dataset: outputs/phase7/packed_calibrated_heldout_clean_think_pair_64_seed0.jsonl
accuracy: 0.5391
family_accuracy: 0.4219
parse_complete_rate: 0.9844
answer_count_mismatch_rate: 0.0156
suspicious_rate: 0.3125
think_block_rate: 0.9844
nontrivial_think_block_rate: 0.9844
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Think block | Non-trivial think |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `missing_average` | 0.8158 | 0.7368 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.6786 | 0.4643 | 0.9643 | 0.0357 | 0.9643 | 0.9643 |
| `rational_system_target` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Comparison to hybrid bridge before GRPO:

| Metric | Hybrid bridge | After 10-step working smoke | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5312 | 0.5391 | +0.0079 |
| Family accuracy | 0.4219 | 0.4219 | 0.0000 |
| Parse complete | 0.9688 | 0.9844 | +0.0156 |
| Think block | 0.9688 | 0.9844 | +0.0156 |

Conclusion:

```text
The deterministic post-update interface is stable.
The sampled training interface is not clean enough yet for the main comparison.
```

This smoke answers the immediate safety question: a few GRPO updates on the working families did not collapse the saved adapter under greedy broad eval. But sampled rollouts at `temperature: 0.7` still produce an empty-completion tail, with sampled parse completeness only `0.8250`. That is too noisy for the independent-versus-iso think-GRPO comparison.

Next fix:

```text
Run a no-training sampled audit on the working families with lower temperature before the comparison.
Try temperature 0.5 first, then 0.3 if empty completions remain.
```

### Experiment 7.6: Lower-Temperature Working-Family Rollout Audits

Goal:

```text
Check whether the empty-completion tail is caused by high sampling temperature.
```

The comparison matrix should not start while sampled parse completeness is below the Phase 5/6 interface standard. Since the empty completions appeared under `temperature: 0.7`, two lower-temperature audits were run from the hybrid-think adapter before trying another GRPO run.

#### Temperature 0.5 Audit

Config:

```text
Config: configs/packed_rollout_audit_phase7_hybrid_think_working_temp_0_5.yaml
Adapter: outputs/phase7/format_sft_qwen25_math_1_5b_hybrid_think_all_families_from_phase5_1epoch/adapter_or_model
Dataset: outputs/phase7/packed_calibrated_heldout_clean_think_pair_64_seed0.jsonl
Family filter: missing_average, rational_linear_equation
Max examples: 16
Samples per prompt: 4
Temperature: 0.5
Max new tokens: 512
```

Result:

```text
samples: 64
accuracy: 0.6016
family_accuracy: 0.4531
parse_complete_rate: 0.8906
answer_count_mismatch_rate: 0.1094
suspicious_rate: 0.1719
think_block_rate: 0.8906
nontrivial_think_block_rate: 0.8906
reward_std: 0.4367
contrast_prompt_count: 5
passes_audit_gate: false
```

By family type:

| Family type | Accuracy | Parse complete | Mismatch | Think block |
| --- | ---: | ---: | ---: | ---: |
| `missing_average` | 0.6667 | 1.0000 | 0.0000 | 1.0000 |
| `rational_linear_equation` | 0.5179 | 0.7500 | 0.2500 | 0.7500 |

#### Temperature 0.3 Smoke Audit

Config:

```text
Config: configs/packed_rollout_audit_phase7_hybrid_think_working_temp_0_3_smoke.yaml
Family filter: missing_average, rational_linear_equation
Max examples: 8
Samples per prompt: 4
Temperature: 0.3
Max new tokens: 512
```

Result:

```text
samples: 32
accuracy: 0.5781
family_accuracy: 0.5000
parse_complete_rate: 0.8438
answer_count_mismatch_rate: 0.1562
suspicious_rate: 0.2812
think_block_rate: 0.8438
nontrivial_think_block_rate: 0.8438
reward_std: 0.4884
contrast_prompt_count: 2
passes_audit_gate: false
```

By family type:

| Family type | Accuracy | Parse complete | Mismatch | Think block |
| --- | ---: | ---: | ---: | ---: |
| `missing_average` | 0.6562 | 1.0000 | 0.0000 | 1.0000 |
| `rational_linear_equation` | 0.5000 | 0.6875 | 0.3125 | 0.6875 |

Conclusion:

Lowering temperature does not fix the sampled empty-completion tail. The problem is concentrated in `rational_linear_equation`; `missing_average` remains parse-complete and keeps nontrivial think blocks under sampling.

Current diagnosis:

```text
The hybrid-think adapter is deterministic-eval stable, but sampled rational-linear rollouts can terminate immediately.
This is a generation-control problem, not a verifier/parser problem.
```

Candidate fixes before the think-GRPO comparison:

- Add explicit generation prefix support for TRL runs, likely `response_prefix: "<think>\n"`.
- Ensure the reward scorer prepends the same prefix before parsing completions.
- Run a no-training working-family audit with the prefix before any more GRPO.
- If prefixing preserves accuracy and parse completeness, rerun the guarded working-family smoke.

### Experiment 7.7: Response-Prefix Working-Family Audit

Goal:

```text
Test whether forcing sampled completions to begin with <think> fixes the empty-completion tail.
```

Rationale:

The lower-temperature audits showed that the sampled interface problem was not primarily caused by high temperature. The next narrow generation-control fix was to prefix assistant generation with:

```xml
<think>
```

The reward parser must see the same prefix that generation used. Otherwise TRL receives a completion fragment that starts after the prefix, while the verifier expects the full two-tag completion.

Implementation:

```text
Trainer: src/iso_rlvr/train/packed_grpo_trl.py
Config: configs/packed_rollout_audit_phase7_hybrid_think_working_prefix_temp_0_7.yaml
Response prefix: "<think>\n"
Family filter: missing_average, rational_linear_equation
Samples per prompt: 4
Temperature: 0.7
Max new tokens: 512
```

Code behavior:

- TRL train rows append `response_prefix` to the prompt before generation.
- The reward adapter prepends the same `response_prefix` before parsing and scoring completions.
- Reward records now preserve both the raw `model_response` and verifier-facing `parsed_response`.
- Reward records include `response_prefix`, `has_think_block`, and `nontrivial_think_block`.

Targeted test result:

```text
tests/test_packed_grpo_trl.py: passed
tests/test_packed_rollout_audit.py: passed
combined targeted suite: 17 passed
```

Audit result:

```text
samples: 64
accuracy: 0.5547
family_accuracy: 0.4219
parse_complete_rate: 0.8438
answer_count_mismatch_rate: 0.1562
suspicious_rate: 0.1875
think_block_rate: 1.0000
nontrivial_think_block_rate: 1.0000
reward_std: 0.4636
contrast_prompt_count: 5
passes_audit_gate: false
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch | Think block |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.5417 | 0.3889 | 0.8611 | 0.1389 | 1.0000 |
| `rational_linear_equation` | 0.5714 | 0.4643 | 0.8214 | 0.1786 | 1.0000 |

Observed failure-mode change:

```text
The prefix fixed missing think blocks.
It did not fix missing final answer blocks.
```

The malformed samples were no longer mostly empty completions. Instead, many completions produced a complete-looking `<think>...</think>` trace and then stopped before emitting `<answers>...</answers>`.

Representative malformed pattern:

```xml
<think>
Problem 1 isolate...
Problem 1 divide...
Problem 2 isolate...
Problem 2 divide...
</think>
```

Other malformed samples emitted answer tags outside an enclosing `<answers>` block. The strict parser correctly rejected those outputs because the verifier contract requires the final answer block.

Conclusion:

Response prefixing is useful but insufficient. It converts the sampled interface problem from:

```text
empty completion or missing <think>
```

to:

```text
missing transition from </think> to <answers>
```

Do not run the independent-versus-iso think-GRPO comparison from this setup. The sampled parse-complete rate remains below the Phase 5/6 standard:

```text
parse_complete_rate: 0.8438
answer_count_mismatch_rate: 0.1562
```

The next targeted fix should address the reasoning-to-answer transition, not the opening tag. Candidate options:

- add a small transition-focused SFT replay set that preserves working-family traces and always closes with `<answers>`
- use a stronger decode scaffold or staged decode that generates reasoning first, then forces the answer-block opening
- add a format-only repair pass for sampled rollouts before attempting GRPO, if the repair pass can be kept outside the reward target

This is still an extension problem. The Phase 6 core result remains complete without it.

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

The first minimal-think bridge was useful but failed as a GRPO starting point. It proved that the parser/reward contract works with `<think> + <answers>`, but it also showed that one epoch of answer-only minimal-think SFT overwrites too much of the Phase 5 reasoning bridge.

The hybrid-think bridge is the current best Phase 7 adapter. It recovers most of the broad deterministic accuracy that minimal-think destroyed and creates one sampled hard-family contrast prompt, but the hard-family signal remains too sparse for a meaningful hard-family GRPO run.

The working-family GRPO smoke shows that the saved adapter stays stable under deterministic eval after updates, but sampled rollouts have too many malformed completions. Lowering temperature to `0.5` and `0.3` did not fix the issue. Response prefixing with `<think>\n` fixed the missing-think/empty-start problem, but it exposed a second failure mode: sampled completions often stop after `</think>` and never emit the required final `<answers>` block.

Do not proceed to the independent-versus-iso think-GRPO comparison until sampled parse completeness is restored.

Reasonable next checks are:

- add a transition-focused bridge or decode scaffold so completions reliably move from `</think>` to `<answers>`
- rerun the working-family sampled audit only after the transition fix
- rerun the working-family smoke only if sampled parse completeness returns to the Phase 5/6 gate
- improve answer-tag hygiene for rational-system outputs
- increase hard-family SFT data before another GRPO attempt

The key lesson:

```text
Minimal think SFT teaches the tag contract, but by itself it is too answer-only.
Hybrid think SFT preserves more reasoning behavior, but hard-family correctness contrast is still sparse.
Response prefixing fixes the opening tag but not the reasoning-to-answer transition.
Think-GRPO is close, but sampled completions must reliably reach <answers> before comparison runs.
```

The Phase 7 thesis:

```text
Reward final answers only.
Let reasoning be free.
Do not turn the reasoning trace into the supervised task.
```
