# Phase 4: Objective Correction Progress

Date: 2026-05-27

## Working Spec

Phase 4 follows `phase4_plan.md`.

The current goal is not to scale the old Iso trainer. Phase 3 showed that the 80-row Iso signal did not survive the full clean held-out eval. Phase 4 starts by correcting the objective path:

```text
packed dataset builder -> packed answer parser -> packed reward tests -> diagnostics -> TRL GRPO smoke test
```

## Step 1: Packed Dataset Builder

Implemented:

```text
src/iso_rlvr/data/build_packed_dataset.py
tests/test_packed_dataset.py
```

The builder converts flat family-variant JSONL into one packed training row per latent family.

Each packed row carries the columns needed by a stateless TRL reward function:

```text
family_id
family_type
num_variants
variant_ids
problems
gold_answers
metadata
prompt
```

This is intentional. The reward function should not need a side table lookup to recover family membership or gold answers.

Prompt format:

```text
Solve each problem. Show brief reasoning if needed, then give the final answers in exactly this format:

Answer 1: <number>
Answer 2: <number>
Answer 3: <number>
Answer 4: <number>

Problem 1: ...
Problem 2: ...
Problem 3: ...
Problem 4: ...
```

CLI:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_packed_dataset --input data/iso_math_calibrated.jsonl --out outputs/phase4/packed_stage1_calibrated_train.jsonl --include-family-type rational_linear_equation,missing_average --expected-variants 4
```

Result:

```text
packed rows: 160
missing_average: 58
rational_linear_equation: 102
```

Conclusion:

The builder supports the Phase 4 Stage 1 curriculum:

```text
rational_linear_equation
missing_average
```

It also supports family-type filtering, expected variant-count filtering, deterministic family ordering, optional shuffling, and `max_families` for smoke tests.

One sample `missing_average` family has identical gold answers across all variants. This is not a builder bug, but it confirms the Phase 4D warning that answer-transform metadata and duplicate-answer frequency need a later audit before making strong transformation-consistency claims.

## Validation

Command:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
18 passed
```

Pytest still emits the existing Windows `.pytest_cache` permission warning, but all tests pass.

## Next Step

Implement packed reward tests before any trainer work.

## Step 2: Packed Answer Parser

Implemented:

```text
src/iso_rlvr/rewards/packed_answer.py
tests/test_packed_answer.py
```

The parser is separate from the older single-answer parser in `src/iso_rlvr/rewards/answer.py`. This is intentional because packed completions need stricter accounting than "take the last number".

Primary supported format:

```text
Answer 1: 3
Answer 2: 7
Answer 3: 21
Answer 4: 4
```

Supported fallback formats:

```text
1) 3
2) 7

Problem 1: 3
Problem 2: 7

\boxed{3}
\boxed{7}
```

Parser return object:

```text
answers: list[str | None]
missing_indices: list[int]
extra_answers: list[str]
mode: indexed | boxed | missing
complete: bool
```

Behavior:

```text
return one extracted answer per expected variant when possible
mark missing answers
mark extra answers
preserve answer order
support the canonical Answer N: value format first
```

The parser treats duplicate indexed answers as extras. For example, if `Answer 1` appears twice, the first value fills answer slot 1 and the second value is recorded in `extra_answers`. This is useful for detecting repeat/copy degeneracy later.

Validation:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
26 passed
```

Conclusion:

The parser is ready for reward implementation. The next file should compute scalar rewards from:

```text
completion
gold_answers
expected variant count
optional token length
```

## Step 3: Packed Reward

Implemented:

```text
src/iso_rlvr/rewards/packed_iso.py
tests/test_packed_iso_reward.py
```

The packed reward computes one scalar per packed completion. It also returns a diagnostic breakdown for tests and later logging.

Default config:

```text
format_reward = +0.05
missing_format_penalty = -0.10
family_mean_weight = 0.25
all_family_correct_weight = 0.25
extra_answer_penalty = 0.10
length_penalty_weight = 0.05
token_cap = 256
```

For each packed completion:

```text
parsed_answers = parse_packed_answers(completion, expected_count=len(gold_answers))
correctness_i = is_correct(parsed_answer_i, gold_answer_i)
family_mean = correct_count / total_variants
all_family_correct = 1.0 if family_mean == 1.0 else 0.0
```

Per-variant reward:

```text
variant_reward_i =
    correctness_i
  + correctness_i * family_mean_weight * family_mean
  + correctness_i * all_family_correct_weight * all_family_correct
  + format_component_i
```

Packed scalar reward:

```text
reward =
    mean(variant_reward_i)
  - extra_answer_penalty * extra_answer_count / total_variants
  - wrong_length_penalty
```

Wrong-length penalty:

```text
wrong_length_penalty =
    length_penalty_weight
  * wrong_fraction
  * min(response_tokens, token_cap) / token_cap
```

It applies only when at least one answer is wrong. Correct long completions are not penalized in this first reward implementation.

Important scoring anchors:

| Case | Reward |
| --- | ---: |
| 4/4 correct, clean format | 1.5500 |
| 2/4 correct, clean format | 0.6125 |
| 1/2 correct, clean format | 0.6125 |
| 1/2 correct, one missing answer | 0.5375 |
| 2/2 correct plus one extra answer | 1.5000 |

Conclusion:

The reward now matches the Phase 4 decision to use correctness-gated family bonuses. Wrong answers cannot receive positive family credit just because sibling variants are correct.

Validation:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
36 passed
```

## Step 4: Packed Output Diagnostics

Implemented:

```text
src/iso_rlvr/eval/packed_diagnostics.py
tests/test_packed_diagnostics.py
```

The diagnostics flag packed-output failure modes that can make a family-consistency result look better than it is.

Current flags:

```text
repeated_answer
copied_answer_indices
only_first_answer
missing_indices
extra_answer_count
answer_count_mismatch
same_wrong_additive_offset
same_wrong_multiplicative_offset
suspicious
```

Covered cases:

```text
same answer repeated for every variant
answer copied from an earlier variant
only the first answer produced
later answers missing
answer count differs from variant count
wrong answers suspiciously correlated by additive or multiplicative offsets
```

This matters because packed prompts create within-generation conditioning. A model can appear more family-consistent by copying or repeating answers, especially when some family types produce identical or near-identical gold answers.

Validation:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
43 passed
```

## Current Phase 4A Status

Completed:

```text
packed dataset builder
packed answer parser
packed reward tests
packed degenerate-output diagnostics
```

Next:

```text
Add a small deterministic packed base-eval path.
Run it on a tiny Stage 1 slice before any GRPO trainer work.
```

Reason:

The parser, reward, and diagnostics now work on synthetic test strings. The next risk is real model output shape: whether Qwen follows the packed answer format often enough for rewards to be reliable.

## Step 5: Packed Base-Eval Smoke

Implemented:

```text
src/iso_rlvr/eval/run_packed_eval.py
tests/test_packed_eval_summary.py
configs/packed_base_eval_stage1_smoke.yaml
configs/packed_base_eval_stage1_512_smoke.yaml
configs/packed_base_eval_stage1_pair_512_smoke.yaml
configs/packed_base_eval_stage1_pair_tail_format_256_smoke.yaml
```

The packed evaluator writes one row per packed family with:

```text
model_response
parsed_answers
missing_indices
extra_answers
correctness
reward
reward components
response_tokens
diagnostics
```

It also writes a summary with:

```text
accuracy
family_accuracy
parse_complete_rate
answer_count_mismatch_rate
suspicious_rate
avg_reward
by_family_type
```

### Smoke 1: Four Variants, Format Before Problems, 256 Tokens

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 32
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: -0.1500
```

Conclusion:

The model ignored the answer-only instruction and spent the entire 256-token budget reasoning through early problems. No packed completion produced parseable `Answer N:` lines.

### Smoke 2: Four Variants, Format Before Problems, 512 Tokens

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_512_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 32
accuracy: 0.2813
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: 0.2136
```

Conclusion:

The parser recovered some answers from sectioned reasoning and boxed values, but the model still did not finish all four variants. Four-variant packed prompts are too long for reliable Phase 4A smoke testing with this base model and current prompting.

### Smoke 3: Two Variants, Format Before Problems, 512 Tokens

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_packed_dataset --input data/iso_math_calibrated.jsonl --out outputs/phase4/packed_stage1_pair_calibrated_train.jsonl --include-family-type rational_linear_equation,missing_average --expected-variants 4 --max-variants-per-family 2
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_pair_512_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 16
accuracy: 0.5000
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: 0.5125
```

Conclusion:

Reducing to two variants helped accuracy, but not parse completeness. The model still reasoned sequentially and usually reached only the first answer before the output became unusable for complete packed reward.

### Smoke 4: Two Variants, Format After Problems, 256 Tokens

The packed prompt was changed so problems come first and the required answer format appears at the end:

```text
Problem 1: ...
Problem 2: ...

Solve each problem silently. Return only the final answers, with no reasoning or extra text. Use exactly this format:

Answer 1: <number>
Answer 2: <number>
```

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_pair_tail_format_256_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 16
accuracy: 0.2500
family_accuracy: 0.1250
parse_complete_rate: 0.3750
answer_count_mismatch_rate: 0.6250
suspicious_rate: 0.7500
avg_reward: 0.2498
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.5000 | 0.3333 | 0.6667 | 0.3333 | 0.6667 |
| `rational_linear_equation` | 0.1000 | 0.0000 | 0.2000 | 0.8000 | 0.8000 |

Conclusion:

Putting the answer format at the end is the first usable packed prompt direction, but it is still not ready for RL. The base model often emits reasoning despite the instruction, then eventually emits answer lines. Parse completeness is nonzero but too low for stable reward training, especially on `rational_linear_equation`.

### Smoke 5: Two Variants, Tail Format Plus Response Prefix, 128 Tokens

This smoke tested whether forcing the generation to begin immediately after `Answer 1:` would reduce reasoning and make the parser complete more often. The evaluator now supports an optional `response_prefix` field in the config. The prefix is appended to the prompt before generation, then prepended back to the generated suffix before parsing and reward scoring.

Config:

```yaml
model_name: Qwen/Qwen2.5-Math-1.5B
dataset_path: outputs/phase4/packed_stage1_pair_calibrated_train.jsonl
output_path: outputs/phase4/packed_base_stage1_pair_tail_prefix_128_smoke.jsonl
max_examples: 8
max_new_tokens: 128
temperature: 0.0
top_p: 1.0
device: auto
resume: true
response_prefix: "\nAnswer 1:"
```

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_pair_tail_prefix_128_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 16
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: -0.1250
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Observed behavior:

The prefix made generations shorter, but the model treated `Answer 1:` as a heading and continued with reasoning prose instead of a numeric answer. A typical parsed response began with:

```text
Answer 1: To find the unknown number, we first calculate...
```

The parser correctly rejected these outputs because the content after `Answer 1:` was not a clean numeric answer. This makes the simple response-prefix strategy unusable for this base model.

Conclusion:

The current best prompt remains the two-variant tail-format prompt without a response prefix. That setup is imperfect but at least produces nonzero parse completeness and family-level correctness. The prefix experiment is useful as a negative result: it shows that we should not rely on a naive forced answer prefix to make Qwen2.5-Math-1.5B base obey the packed answer format.

### Parser Correction: Problem Statement False Positives

While inspecting the instruct-model outputs, we found a parser bug in the fallback indexed-answer path. The parser could treat a problem statement such as:

```text
Problem 1: Solve for x: -6x + 22/3 = -2/3.
```

as if the answer to `Problem 1` were `-6`. This is not acceptable for RL reward use because it can turn ordinary reasoning text into a parse-complete answer set.

The parser was tightened so `Problem N` forms are accepted only when they explicitly look like answer declarations, for example:

```text
Problem 1 final answer: 3
Problem 2 answer: 7
```

The accepted packed answer surfaces are now:

- Canonical `Answer N: <value>` lines.
- Explicit `Problem N answer: <value>` or `Problem N final answer: <value>` lines.
- Numbered answer-only lines such as `1) <value>`.
- Boxed values, including boxed values inside explicit problem sections.

A regression test was added to ensure ordinary problem statements are not parsed as answers.

### Smoke 6: Base Tail Format Rerun With Strict Parser

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_base_eval_stage1_pair_tail_format_256_strict_parser_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 16
accuracy: 0.2500
family_accuracy: 0.1250
parse_complete_rate: 0.3750
answer_count_mismatch_rate: 0.6250
suspicious_rate: 0.7500
avg_reward: 0.2498
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.5000 | 0.3333 | 0.6667 | 0.3333 | 0.6667 |
| `rational_linear_equation` | 0.1000 | 0.0000 | 0.2000 | 0.8000 | 0.8000 |

Conclusion:

The stricter parser did not change the best base-model tail-format result. That means Smoke 4's nonzero parse completeness was not caused by the `Problem N: Solve...` false-positive path. The base model is still the strongest observed setup so far, but it is far below the parse-completeness threshold needed for GRPO.

### Smoke 7: Instruct Chat Template Tail Format, Strict Parser

The evaluator now supports `apply_chat_template: true`. When enabled, the packed prompt is wrapped through the tokenizer chat template with an optional `system_prompt`, and generation begins from the model's assistant turn. The result row records both the original dataset prompt and the rendered `generation_prompt`.

Config:

```yaml
model_name: Qwen/Qwen2.5-Math-1.5B-Instruct
dataset_path: outputs/phase4/packed_stage1_pair_calibrated_train.jsonl
output_path: outputs/phase4/packed_instruct_stage1_pair_tail_format_256_strict_parser_smoke.jsonl
max_examples: 8
max_new_tokens: 256
temperature: 0.0
top_p: 1.0
device: auto
resume: false
apply_chat_template: true
system_prompt: "Solve the math problems. Return only the requested final answer lines with no reasoning or extra text."
```

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_packed_eval --config configs/packed_instruct_eval_stage1_pair_tail_format_256_smoke.yaml
```

Result:

```text
examples: 8
variant_examples: 16
accuracy: 0.0000
family_accuracy: 0.0000
parse_complete_rate: 0.0000
answer_count_mismatch_rate: 1.0000
suspicious_rate: 1.0000
avg_reward: -0.1500
```

By family type:

| Family type | Accuracy | Family accuracy | Parse complete | Mismatch rate | Suspicious rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_average` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `rational_linear_equation` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Observed behavior:

The instruct model ignored the no-reasoning instruction and began responses with long step-by-step solutions such as `Let's solve each problem step-by-step using Python and SymPy.` It often computed at least the first answer correctly in the reasoning trace, but did not emit the requested `Answer 1:` / `Answer 2:` final lines within the 256-token budget. Under the strict parser, these reasoning traces are correctly marked incomplete instead of being partially credited as formatted answers.

Conclusion:

The simple chat-template conversion is not enough. This model family appears strongly biased toward verbose worked solutions on these prompts, even in deterministic decoding. For Phase 4, the practical bottleneck is format control, not mathematical ability.

Current decision:

```text
Do not start GRPO trainer work yet.
First solve packed format compliance. Reward training on a parser-complete rate near zero would mostly optimize formatting accidents and length artifacts.
```

Recommended next experiment:

```text
Keep two-variant tail-format packed prompts as the working prompt family.
Run a tiny supervised format warmup/SFT stage, or test a model with stronger instruction-following format control.
Require parse_complete_rate >= 0.90 on a tiny smoke before any RL run.
```

Validation after code changes:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
52 passed
```
