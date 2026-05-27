# Phase 5: Reward Interface Stabilization Before GRPO

Status: proposed working spec for critic review

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
- repeated answers
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

Open question for critic:

Should parse-incomplete completions receive:

1. A fixed negative reward.
2. Zero reward.
3. Format-only penalty while masking correctness.
4. Be dropped from the GRPO group.

My current recommendation:

Use a fixed negative or low reward for parse-incomplete completions during early GRPO, but keep detailed diagnostics. Dropping too many samples can distort group statistics. Giving partial correctness credit to malformed outputs is dangerous.

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

## Recommended Immediate Order

1. Implement XML parser and tests.
2. Add XML prompt mode to packed dataset builder.
3. Build XML packed smoke dataset.
4. Run base 1.5B XML smoke.
5. Run `Qwen/Qwen2.5-3B-Instruct` XML diagnostic smoke.
6. Build answer-only XML SFT dataset.
7. Train tiny LoRA format adapter on `Qwen/Qwen2.5-Math-1.5B`.
8. Re-evaluate parse gate.
9. If parse gate passes, resume GRPO implementation.

## What We Should Ask The Critic

1. Is XML the right immediate answer surface, or should we use JSON despite fraction handling?
2. Should parse-incomplete samples get negative reward, zero reward, or be masked/dropped during GRPO?
3. Is `parse_complete_rate >= 0.90` strict enough, or should the gate be `>= 0.95` before RL?
4. Should we run the 3B-Instruct diagnostic before implementing SFT, or is SFT obviously required?
5. Is two-variant packing still the right horizon, or should Phase 5 temporarily drop to one problem per prompt to isolate format behavior?
6. Should final answer tags be answer-only, or should we eventually separate hidden/visible reasoning with `<think>` and `<answers>`?

## Current Recommendation

Proceed with XML answer contract plus tiny format SFT.

Do not spend time on GRPO implementation until the format gate passes. Do not spend time on constrained decoding first unless SFT fails. Run the 3B-Instruct smoke as a cheap diagnostic, but keep `Qwen/Qwen2.5-Math-1.5B` as the main training target because it fits the 5070 Ti and is math-specialized.

The core Phase 5 thesis:

```text
Reliable verifier interface first.
GRPO second.
```
