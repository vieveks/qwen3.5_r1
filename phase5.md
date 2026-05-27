# Phase 5: Reward Interface Stabilization Before GRPO

Status: active implementation log and working spec

Date: 2026-05-27

## Executive Summary

Phase 4 showed that the project is not currently blocked by GRPO mechanics. It is blocked by the interface between model output and the verifier.

Current best packed result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Prompt: two-variant packed prompt, answer format at tail
Parser: strict packed parser
examples: 8
variant_examples: 16
accuracy: 0.2500
family_accuracy: 0.1250
parse_complete_rate: 0.3750
answer_count_mismatch_rate: 0.6250
suspicious_rate: 0.7500
```

The best result still has only 37.5% parse-complete outputs. That is not high enough for RL. If we run GRPO now, the policy will mostly receive noisy rewards caused by formatting failures, truncation, parser misses, and accidental answer extraction. This would train against the interface, not the intended reasoning behavior.

Phase 5 therefore pivots temporarily from "run GRPO" to "make the reward interface reliable enough that GRPO means something."

## Core Diagnosis

The current RLVR pipeline is:

```text
packed prompt -> model completion -> parser -> verifier -> reward
```

The weak link is:

```text
model completion -> parser
```

Observed failure modes from Phase 4:

- The base model often emits reasoning prose instead of answer lines.
- The instruct model also emits reasoning prose even under chat-template prompting.
- Response-prefix prompting made outputs shorter but caused the model to continue after `Answer 1:` with prose.
- Packed generation is long-horizon structured generation: the model must solve multiple related tasks, remember indices, avoid truncation, and emit a strict final format.
- Parser false positives are dangerous: we already found and fixed a case where `Problem 1: Solve for x: -6x...` could be misread as answer `-6`.

This is not a minor formatting inconvenience. In RL, a bad reward interface changes the optimization target.

## Industry Context

Modern RLHF/RLVR systems frequently use multiple layers of reward-interface control. The relevant techniques are:

### 1. Rule-Based Accuracy Plus Format Rewards

DeepSeek-R1 reports rule-based rewards for reasoning tasks, mainly accuracy rewards and format rewards. The accuracy reward checks deterministic outcomes, while the format reward enforces a required structure around reasoning or answers.

This matters for us because our current reward has the same broad shape: answer correctness plus format diagnostics. The difference is that our model does not yet reliably produce verifier-compatible outputs.

Reference:

- DeepSeek-R1 Nature paper: https://www.nature.com/articles/s41586-025-09422-z

### 2. Constrained Decoding

Production systems often constrain decoding so invalid output structures are impossible or less likely. Examples include JSON schema decoding, regex decoding, grammar decoding, and structured tool/function calling.

This directly addresses our parser problem. Instead of asking the model to "please follow this format," decoding only permits valid structures.

References:

- OpenAI Structured Outputs: https://openai.com/index/introducing-structured-outputs-in-the-api/
- vLLM Structured Outputs: https://docs.vllm.ai/en/v0.10.1/features/structured_outputs.html

### 3. SFT Before RL

Many practical pipelines use SFT before RL to teach the model the interaction contract. For us, this means a small supervised warmup where the model learns:

```text
input packed prompt -> output only machine-verifiable answers
```

This is not intended to improve math ability. It is intended to improve format obedience and reduce reward noise.

### 4. Verifier-Friendly Output Contracts

Reliable systems do not reward arbitrary prose. They define a narrow answer surface:

- JSON object
- XML-like tags
- boxed final answers
- function/tool calls
- executable actions

For our local training setup, XML-like tags are the best immediate option because they are easy to parse, easy to test, and do not require a full constrained decoding stack on day one.

### 5. Rejection, Masking, And Diagnostics

Invalid parse samples should not silently contribute misleading correctness rewards. We should explicitly track:

- parse completeness
- answer-count mismatch
- copied prompt answers
- repeated answers, but only when repetition is not expected from the gold answer pattern
- only-first-answer behavior
- suspicious parse patterns
- truncation

For GRPO, we need to decide whether parse failures receive a negative reward, zero reward, or are masked from the correctness component. This decision should be made before training.

## Model Landscape

Confirmed model options:

| Model | Role | Fit on 5070 Ti 16GB | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-Math-1.5B` | Main training candidate | Good | Math-specialized, trainable locally, poor format compliance so far. |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | Diagnostic | Good | Chat-template smoke still produced verbose reasoning; parse complete was 0.0. |
| `Qwen/Qwen2.5-3B-Instruct` | Format-compliance diagnostic | Likely good for inference and LoRA SFT | General instruct model, not math-specialized, but may obey structure better. |
| `Qwen/Qwen2.5-Math-7B` | Diagnostic only | Quantized inference likely; training poor fit | Math-specialized but too heavy for comfortable GRPO on 16GB. |
| `Qwen/Qwen2.5-Math-7B-Instruct` | Diagnostic only | Quantized inference likely; training poor fit | Useful only to test whether scale/instruction tuning solves format obedience. |

Qwen2.5-Math model sizes are 1.5B, 7B, and 72B. There is no Qwen2.5-Math-3B. The 3B option is from the general Qwen2.5-Instruct family.

References:

- Qwen2.5-Math announcement: https://qwenlm.github.io/blog/qwen2.5-math/
- Qwen2.5-3B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

## Phase 5 Objective

Make packed answer generation reliable enough for RLVR.

Primary gate:

```text
parse_complete_rate >= 0.90 on deterministic tiny packed smoke
```

Target gate:

```text
parse_complete_rate >= 0.95
answer_count_mismatch_rate <= 0.05
suspicious_rate <= 0.10
no known parser false positives
```

Only after this gate should GRPO implementation or GRPO training resume.

## Proposed Output Contract

Move from:

```text
Answer 1: <number>
Answer 2: <number>
```

to XML-like tags:

```xml
<answers>
<answer_1>33</answer_1>
<answer_2>16/5</answer_2>
</answers>
```

Reasons:

- Explicit opening and closing tags reduce ambiguity.
- Each answer has a unique key.
- Prose inside tags can be rejected.
- Missing answers are easy to detect.
- Extra answers are easy to detect.
- This format is compatible with future regex or grammar decoding.
- It is simpler than JSON for fractions such as `16/5` because values can be plain text.

Accepted value grammar should initially remain narrow:

```text
integer: -3, 0, 42
fraction: -16/5
decimal: 3.5
latex fraction: \frac{16}{5}
```

The parser should normalize LaTeX fractions to `a/b`.

## Implementation Plan

### Step 1: XML Packed Parser

Add a parser path for XML-style packed answers.

Expected API:

```python
parse_packed_answers(text: str, expected_count: int) -> PackedAnswerParse
```

New parser behavior:

- Prefer XML answer tags first.
- Fall back to existing indexed answer lines.
- Reject non-numeric content inside answer tags.
- Mark duplicate answer tags as extras.
- Mark missing answer tags as missing.
- Preserve the existing strict parser tests.

Tests:

- Parses clean XML answer block.
- Parses fractions and LaTeX fractions.
- Marks missing tag.
- Marks duplicate tag as extra.
- Rejects prose inside answer tag.
- Does not parse problem statements as answers.

Implementation note:

Packed diagnostics must be gold-aware. The current two-variant isomorphic families often intentionally have the same gold answer for both variants. A repeated predicted answer is therefore not suspicious by itself. It is suspicious when the gold answers differ, or when a repeated wrong answer creates a same-wrong additive or multiplicative offset.

### Step 2: XML Packed Dataset Builder Mode

Extend the packed dataset builder so the prompt can request XML output.

Current prompt family:

```text
Problem 1: ...
Problem 2: ...

Solve each problem silently. Return only the final answers, with no reasoning or extra text. Use exactly this format:

Answer 1: <number>
Answer 2: <number>
```

New prompt family:

```text
Problem 1: ...
Problem 2: ...

Solve each problem silently. Return only the final answers, with no reasoning or extra text. Use exactly this XML format:

<answers>
<answer_1>number</answer_1>
<answer_2>number</answer_2>
</answers>
```

The dataset rows should continue to carry:

- `family_id`
- `family_type`
- `variant_ids`
- `problems`
- `gold_answers`
- `num_variants`
- metadata

These columns are required later for a stateless GRPO reward function.

### Step 3: Format SFT Dataset Builder

Build a supervised dataset from the packed rows.

Input:

```text
row["prompt"]
```

Target:

```xml
<answers>
<answer_1>{gold_answers[0]}</answer_1>
<answer_2>{gold_answers[1]}</answer_2>
</answers>
```

Important:

- Target contains no reasoning.
- Target contains no explanations.
- Target is generated from gold answers.
- Use train/heldout split by family id, not by row index.
- Include both rational equation and missing average families first.

Initial size:

```text
200 to 500 packed examples for first smoke
then 1,000 to 5,000 if needed
```

### Step 4: Tiny Format SFT

Train a small LoRA adapter on `Qwen/Qwen2.5-Math-1.5B`.

Purpose:

```text
teach output contract, not math
```

Suggested starting config:

```text
base_model: Qwen/Qwen2.5-Math-1.5B
method: LoRA SFT
rank: 8 or 16
epochs: 1 to 2
learning_rate: 1e-4 to 2e-4
max_seq_length: enough for packed prompt + short XML answer
target: answer-only XML block
```

Evaluation after every SFT smoke:

```text
run packed eval on heldout families
require parse_complete_rate >= 0.90
track accuracy separately
```

Success means the model reliably emits answer tags. It does not require high accuracy yet.

### Step 5: 3B-Instruct Diagnostic Smoke

Before or alongside SFT, run one diagnostic smoke with:

```text
Qwen/Qwen2.5-3B-Instruct
two-variant XML packed prompt
strict XML parser
8 examples
greedy decoding
```

Purpose:

Determine whether better instruction following alone solves the interface problem.

Expected outcomes:

| Result | Interpretation | Next action |
| --- | --- | --- |
| Parse complete >= 0.90 | 1.5B-specific format issue | Consider 3B as training candidate or use result to justify SFT. |
| Parse complete remains poor | Prompt/model alone insufficient | Proceed with format SFT on 1.5B. |
| Parse complete high but accuracy low | Interface solved, math weaker | Use 3B as diagnostic only; train 1.5B format adapter. |

### Step 6: Optional Constrained Decoding

Do not implement this first unless SFT fails.

Potential options:

- vLLM guided regex
- vLLM guided grammar
- Outlines-style regex or JSON constrained generation
- custom token-level constrained decoder for XML tags

Why defer:

- More moving parts.
- May complicate local 5070 Ti training and eval setup.
- SFT is simpler and directly addresses the observed failure.

Why keep it on the roadmap:

- Industry-grade verifier interfaces often use constrained decoding.
- If we later run large-scale RL, constrained decoding can eliminate whole classes of malformed completions.

## Reward Design After Phase 5

Once parse compliance is high, update reward logic to make the interface explicit.

Candidate reward decomposition:

```text
reward = correctness_component
       + family_bonus
       + format_component
       - extra_answer_penalty
       - length_penalty
```

But the critical rule should be:

```text
No correctness credit unless the answer interface is parse-complete.
```

Provisional decision:

```text
Parse-incomplete completions receive zero correctness reward and no family bonus.
They may receive a small format penalty, but not a large negative reward at first.
They should remain in the GRPO group unless experiments show group statistics are unstable.
```

Rationale:

- A strong negative reward can over-focus early training on avoiding interface failures in brittle ways.
- Zero correctness reward is enough to prevent malformed outputs from getting task credit.
- Dropping parse failures can distort GRPO group statistics, especially when parse completeness is still improving.
- Partial correctness credit for malformed outputs is dangerous because it teaches the model that parser-adjacent prose is acceptable.

This decision can be revisited after the Phase 5 format gate passes and we have reward-distribution data from post-SFT packed smokes.

## GRPO Compatibility Notes

TRL `GRPOTrainer` reward functions receive:

```text
prompts
completions
completion_ids
dataset columns as keyword args
```

Therefore the packed dataset must carry all reward context as columns:

- `family_id`
- `family_type`
- `gold_answers`
- `variant_ids`
- `num_variants`
- prompt format version

The reward function should not depend on side-table lookups. This keeps it stateless and testable.

Reference:

- TRL GRPO docs: https://huggingface.co/docs/trl/v0.17.0/en/grpo_trainer

## Phase 5 Experiments

### Experiment 5.1: XML Parser Unit Tests

Goal:

```text
Ensure XML answer extraction is strict and safe.
```

Pass condition:

```text
all parser tests pass
no problem-statement false positives
```

Result:

```text
Implemented.
Parser prefers XML answer tags before indexed/boxed fallback paths.
Malformed XML attempts do not fall through to old parser heuristics.
Tests pass.
```

### Experiment 5.2: XML Prompt Base Smoke

Model:

```text
Qwen/Qwen2.5-Math-1.5B
```

Goal:

```text
Check whether XML format alone improves parse completeness.
```

Pass condition:

```text
parse_complete_rate materially better than 0.375
```

Expected:

Probably still poor, but worth measuring.

Decision rule:

```text
If parse_complete_rate >= 0.90, run a larger confirmation smoke before SFT.
If the confirmation smoke holds, skip format SFT and resume GRPO planning with this model/prompt.
```

Result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Config: configs/packed_base_eval_stage1_pair_xml_256_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_calibrated_train.jsonl
examples: 8
variant_examples: 16
accuracy: 0.1875
family_accuracy: 0.1250
parse_complete_rate: 0.1250
answer_count_mismatch_rate: 0.8750
suspicious_rate: 1.0000
avg_reward: 0.1453
```

Conclusion:

XML prompting alone did not improve the base math model. It usually still emitted reasoning first, then sometimes began the XML block too late in the 256-token budget. XML is still useful as a stricter answer contract and SFT target, but it is not sufficient as a prompt-only fix for `Qwen/Qwen2.5-Math-1.5B`.

### Experiment 5.3: XML Prompt 3B-Instruct Smoke

Model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Goal:

```text
Check whether better instruction following fixes format compliance.
```

Pass condition:

```text
parse_complete_rate >= 0.90
```

Decision rule:

```text
If parse_complete_rate >= 0.90, run a larger confirmation smoke before SFT.
If parse completeness holds but math accuracy is weak, keep 3B as a diagnostic result and continue SFT on the math-specialized 1.5B model.
If parse completeness and accuracy both look good, consider 3B as a possible training candidate.
```

Tiny smoke result:

```text
Model: Qwen/Qwen2.5-3B-Instruct
Config: configs/packed_3b_instruct_eval_stage1_pair_xml_256_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_calibrated_train.jsonl
examples: 8
variant_examples: 16
accuracy: 0.0625
family_accuracy: 0.0000
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
avg_reward: 0.1146
```

Confirmation result:

```text
Model: Qwen/Qwen2.5-3B-Instruct
Config: configs/packed_3b_instruct_eval_stage1_pair_xml_256_confirm.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_calibrated_train.jsonl
examples: 32
variant_examples: 64
accuracy: 0.0313
family_accuracy: 0.0000
parse_complete_rate: 0.9375
answer_count_mismatch_rate: 0.0625
suspicious_rate: 0.0625
avg_reward: 0.0697
```

By family type on the 32-example confirmation:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `rational_linear_equation` | 0.0500 | 0.0000 | 0.9000 | 0.1000 | 0.1000 |

Observed behavior:

The model emitted short XML answer blocks immediately, with little or no reasoning. The two malformed cases were malformed answer values inside XML tags, such as invalid LaTeX or mixed-fraction strings, not a collapse back into prose.

Conclusion:

`Qwen/Qwen2.5-3B-Instruct` solves the reward-interface problem much better than the 1.5B math model, but its math accuracy is too weak for the current task. Treat it as a diagnostic and format-following reference, not as the main training model. Continue with format SFT on `Qwen/Qwen2.5-Math-1.5B`.

### Experiment 5.4: Tiny Format SFT Dataset

Goal:

```text
Build answer-only XML SFT dataset from packed rows.
```

Pass condition:

```text
dataset examples manually inspect cleanly
train/heldout split is by family_id
targets contain no reasoning
```

Result:

```text
Implemented.
Builder: src/iso_rlvr/data/build_format_sft_dataset.py
Input: outputs/phase5/packed_stage1_pair_xml_calibrated_train.jsonl
Train output: outputs/phase5/format_sft_pair_xml_train.jsonl
Heldout output: outputs/phase5/format_sft_pair_xml_heldout.jsonl
Packed train output: outputs/phase5/packed_stage1_pair_xml_sft_train.jsonl
Packed heldout output: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
Train rows: 144
Heldout rows: 16
Split: by family_id, seed 0, heldout_fraction 0.1
```

Each row contains:

- `prompt`
- `completion`
- `text`
- `family_id`
- `family_type`
- `variant_ids`
- `num_variants`
- `gold_answers`
- `prompt_format`

The `completion` is answer-only XML generated directly from `gold_answers`, with no reasoning.

Conclusion:

The dataset builder is now doing the right split for both SFT and post-SFT evaluation. This matters because the adapter should be evaluated on heldout packed rows from the same heldout family ids used for the SFT heldout file. The previous first-8-row train smoke was useful as a mechanical check, but it was not enough to claim generalization.

### Experiment 5.5: Tiny LoRA Format SFT

Model:

```text
Qwen/Qwen2.5-Math-1.5B
```

Goal:

```text
Raise parse_complete_rate above 0.90.
```

Pass condition:

```text
heldout parse_complete_rate >= 0.90
```

Preferred condition:

```text
heldout parse_complete_rate >= 0.95
suspicious_rate <= 0.10
```

Result:

```text
Implemented.
Trainer: src/iso_rlvr/train/format_sft.py
Config: configs/format_sft_qwen25_math_1_5b_xml_smoke.yaml
Base model: Qwen/Qwen2.5-Math-1.5B
Method: LoRA SFT
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
Batch size: 2
Max steps: 20
Learning rate: 2e-4
Max sequence length: 768
Trainable parameters: 9,232,384
Total parameters: 1,552,946,688
Trainable percent: 0.5945%
Adapter output: outputs/phase5/format_sft_qwen25_math_1_5b_xml_smoke/adapter_or_model
```

Training loss fell quickly in the first five logged steps:

| Step | Loss |
| ---: | ---: |
| 0 | 0.7888 |
| 1 | 0.7339 |
| 2 | 0.5598 |
| 3 | 0.4423 |
| 4 | 0.3282 |

Conclusion:

The tiny LoRA run successfully taught the model the answer-only XML surface at least mechanically. This is not yet a reasoning improvement. It is a reward-interface warmup.

### Experiment 5.6: Post-SFT Packed Reward Smoke

Goal:

```text
Verify reward is now measuring correctness/family consistency rather than format failure.
```

Metrics:

- variant accuracy
- family accuracy
- parse complete rate
- suspicious rate
- answer-count mismatch
- reward distribution
- output length

Pass condition:

```text
parse gate satisfied
no major parser pathologies
```

Only after this should GRPO resume.

Train-row smoke result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_smoke/adapter_or_model
Config: configs/packed_base_eval_stage1_pair_xml_256_after_sft_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_calibrated_train.jsonl
examples: 8
variant_examples: 16
accuracy: 0.1250
family_accuracy: 0.1250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.8750
avg_reward: 0.2320
```

Heldout smoke result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_smoke/adapter_or_model
Config: configs/packed_base_eval_stage1_pair_xml_256_after_sft_heldout_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
examples: 8
variant_examples: 16
accuracy: 0.1250
family_accuracy: 0.1250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.8750
avg_reward: 0.2320
```

Full heldout result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_smoke/adapter_or_model
Config: configs/packed_base_eval_stage1_pair_xml_256_after_sft_heldout_full.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
examples: 16
variant_examples: 32
accuracy: 0.1250
family_accuracy: 0.1250
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.8750
avg_reward: 0.2321
```

By family type on the full heldout run:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| `rational_linear_equation` | 0.2000 | 0.2000 | 1.0000 | 0.0000 | 0.8000 |

Representative heldout completions:

```xml
<answers>
<answer_1>100</answer_1>
<answer_2>100</answer_2>
</answers>
```

Gold for that row:

```text
["32", "32"]
```

Another heldout row:

```xml
<answers>
<answer_1>11/5</answer_1>
<answer_2>11/5</answer_2>
</answers>
```

Gold for that row:

```text
["11/5", "11/5"]
```

Conclusion:

The tiny format SFT solved the immediate reward-interface failure. The heldout parse gate is passed:

```text
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
```

But it did not solve answer quality. Accuracy remains low, and many wrong outputs are collapsed repeated answers. Because these isomorphic families often intentionally have identical gold answers, repetition alone is no longer counted as suspicious after the diagnostics fix. The remaining suspicious signal mostly comes from same-wrong additive or multiplicative offsets, which is a real failure mode.

This means Phase 5 succeeded at stabilizing the parser/verifier interface but has not yet produced a model ready for GRPO. We need a stronger format SFT run, a small reasoning SFT component, or dataset/prompt changes before policy optimization.

Critic interpretation:

The high `suspicious_rate` should not be treated as a minor diagnostic artifact. The representative wrong completion:

```xml
<answers>
<answer_1>100</answer_1>
<answer_2>100</answer_2>
</answers>
```

when the gold answer is:

```text
["32", "32"]
```

shows a real failure mode: the adapter learned to emit valid repeated XML answers without reliably doing the math. This is format-compliant but mathematically inert behavior. It is also a regression relative to the original Phase 1 base-model baseline accuracy of 0.5625 on the calibrated single-problem eval. Therefore the next fix should not be "more format SFT" by itself. It should preserve or teach reasoning while keeping the final XML answer interface.

### Experiment 5.7: Single-Problem XML Fallback Smoke

Run this only if the two-variant XML prompt fails on both 1.5B and 3B.

Goal:

```text
Separate format-compliance failure from packing-horizon failure.
```

Interpretation:

| Result | Interpretation | Next action |
| --- | --- | --- |
| Single-problem XML passes, two-variant XML fails | Packing horizon is the main issue | Format SFT should start with one problem, then curriculum to two. |
| Single-problem XML also fails | General format obedience is the issue | Format SFT is required before any packed RL work. |

### Experiment 5.8: Few-Shot XML Prompt Diagnostic

Goal:

```text
Check whether the base 1.5B model can follow the XML answer contract from an in-context example, without any adapter.
```

Prompt mode:

```text
xml_fewshot
```

The prompt shows a simple two-problem XML example:

```xml
<answers>
<answer_1>3</answer_1>
<answer_2>4</answer_2>
</answers>
```

Then asks the model to solve the real packed pair and return one XML answer block.

No-prefix result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Config: configs/packed_base_eval_stage1_pair_xml_fewshot_256_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_fewshot_calibrated_train.jsonl
examples: 8
variant_examples: 16
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: -0.1000
```

Observed behavior:

The model generated empty completions. The likely cause is that the prompt demonstration plus XML-oriented instruction made the base model terminate immediately instead of beginning a new answer block.

Response-prefix result:

```text
Model: Qwen/Qwen2.5-Math-1.5B
Config: configs/packed_base_eval_stage1_pair_xml_fewshot_prefix_256_smoke.yaml
Dataset: outputs/phase5/packed_stage1_pair_xml_fewshot_calibrated_train.jsonl
response_prefix: "<answers>\n"
examples: 8
variant_examples: 16
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.2500
answer_count_mismatch_rate: 0.7500
suspicious_rate: 0.8750
avg_reward: -0.1000
```

By family type on the response-prefix run:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.0000 | 0.0000 | 0.6667 | 0.3333 | 0.6667 |
| `rational_linear_equation` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Representative failure modes:

- Copied the demonstration answers `3` and `4`.
- Emitted malformed UI-like placeholder text inside answer tags for rational-equation rows.
- Started with XML tags, then drifted into verbose step-by-step prose.
- Sometimes produced complete XML for missing-average rows, but the numbers were wrong.

Conclusion:

Few-shot prompting does not solve the base 1.5B model's interface problem. A response prefix helps the model begin the XML block, but it does not produce reliable, correct, parse-complete answers. This supports the current SFT-first direction, with one important refinement: the next SFT bridge must include a math-preserving reasoning signal, not only answer-only XML targets.

## Recommended Immediate Order

1. Implement XML parser and tests. Done.
2. Add XML prompt mode to packed dataset builder. Done.
3. Build XML packed smoke dataset. Done.
4. Run base 1.5B XML smoke. Done; failed parse gate.
5. Run `Qwen/Qwen2.5-3B-Instruct` XML diagnostic smoke. Done; passed parse gate but failed accuracy.
6. If either XML smoke passes the parse gate, run a larger confirmation smoke and skip SFT if the result holds. Done for 3B; parse holds enough to confirm interface compliance, but accuracy is too weak to skip SFT on the math model.
7. If both XML smokes fail, optionally run the single-problem XML fallback smoke. Not needed right now because 3B shows the XML interface itself is viable.
8. Build answer-only XML SFT dataset. Done.
9. Train tiny LoRA format adapter on `Qwen/Qwen2.5-Math-1.5B`. Done; 20-step smoke completed.
10. Re-evaluate parse gate. Done; heldout parse_complete_rate is 1.0000 and answer_count_mismatch_rate is 0.0000.
11. If parse gate passes, resume GRPO implementation. Blocked for now by low answer accuracy and high wrong-collapse diagnostics; do not resume GRPO yet.
12. Run no-training few-shot XML diagnostic before scaling SFT. Done; no-prefix failed with empty completions, response-prefix improved parse to 0.2500 but accuracy stayed 0.0000.

## What We Should Ask The Critic

1. Is XML the right immediate answer surface, or should we use JSON despite fraction handling?
2. Is the provisional parse-incomplete rule right: zero correctness reward, no family bonus, small format penalty at most, and keep samples in the GRPO group?
3. Is `parse_complete_rate >= 0.90` strict enough, or should the gate be `>= 0.95` before RL?
4. Should we run the 3B-Instruct diagnostic before implementing SFT, or is SFT obviously required?
5. Is two-variant packing still the right horizon, or should Phase 5 temporarily drop to one problem per prompt to isolate format behavior?

Deferred question:

```text
Should final answer tags eventually be paired with separate hidden/visible reasoning tags such as <think> and <answers>?
```

This is real but premature. Phase 5 should stay answer-only unless the critic sees a strong reason to introduce reasoning channels now.

## Current Recommendation

Continue with XML answer contract plus SFT-first stabilization on `Qwen/Qwen2.5-Math-1.5B`, but do not scale answer-only format SFT blindly.

The 20-step LoRA smoke proves the interface can be fixed locally: heldout parse completeness is now 1.0000. However, answer accuracy is still only 0.1250 on full heldout and the model often emits plausible but wrong repeated XML answers. The few-shot diagnostic also failed: no-prefix produced empty completions, and response-prefix only reached parse_complete_rate 0.2500 with accuracy 0.0000. Do not start GRPO yet.

The next step should be a reasoning-preserving supervised bridge:

1. Build a synthetic SFT dataset with short deterministic solution traces plus final XML answers for the current two families.
2. Keep the final answer surface exactly XML, so the reward parser remains unchanged.
3. Train a small LoRA on mixed targets: some answer-only XML rows to preserve the interface, plus worked rows to preserve or restore math behavior.
4. Evaluate every candidate adapter on the packed heldout split.
5. Gate on both sides: `parse_complete_rate >= 0.95`, `answer_count_mismatch_rate <= 0.05`, and accuracy materially above the current 0.1250 post-SFT baseline.
6. Keep GRPO blocked until the post-SFT reward smoke has both high parse compliance and a non-degenerate reward distribution.

The 3B-Instruct smoke confirms the XML contract is viable, but the model's low math accuracy makes it a poor main RL target for this project.

The core Phase 5 thesis:

```text
Reliable verifier interface first.
GRPO second.
```
