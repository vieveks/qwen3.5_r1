# Critic Brief: Iso-RLVR Stabilized XML Result

Date: 2026-05-30

This is the compact review packet for the current project state. It is written as a report skeleton: result first, then the interface work that makes the result trustworthy, then limitations and next work.

## Headline Result

On a stable packed XML verifier interface, proper TRL `GRPOTrainer` training with an isomorphic family reward improves heldout family accuracy over independent RLVR on the two working family types:

```text
missing_average
rational_linear_equation
```

Best narrow result:

```text
Independent family_accuracy: 0.5625
Iso lambda_iso=1.00 family_accuracy: 0.6875
Family_accuracy delta: +0.1250
Accuracy delta: +0.0625
Parse complete rate: 1.0000 in all narrow arms
Sampled malformed completions: 0 in all narrow arms
```

The most important evidence is not a single run. It is the combination of:

- reward-interface stabilization before RL
- proper TRL GRPO integration
- matched independent-vs-iso comparison
- seed replication at `lambda_iso=0.50`
- monotonic lambda trend at seed 23
- zero parse regression across the narrow runs

## Current Results Table

This is the table to use for the current report. It replaces the older Phase 2/3 `grpo_lite` table.

| Run | Accuracy | Family accuracy |
| --- | ---: | ---: |
| Base all-traces adapter | 0.7500 | 0.6250 |
| Independent 30-step, seed 23 | 0.7188 | 0.5625 |
| Iso `lambda_iso=0.25`, seed 23 | 0.7500 | 0.6250 |
| Iso `lambda_iso=0.50`, seed 23 | 0.7500 | 0.6250 |
| Iso `lambda_iso=1.00`, seed 23 | 0.7813 | 0.6875 |
| Independent 30-step, seed 37 | 0.7188 | 0.5625 |
| Iso `lambda_iso=0.50`, seed 37 | 0.7500 | 0.6250 |

Note:

```text
phase6.md records Iso lambda_iso=1.00 accuracy as 0.7813.
If an older note says 0.7500 for that row, treat it as stale.
```

## Main Phase 6 Table

All runs below use the same base model, Phase 5 all-traces adapter initialization, train/eval split, trainer settings, and evaluation code. The only intentional variable in the sweep is the family reward scale.

```text
Base model: Qwen/Qwen2.5-Math-1.5B
Trainer: TRL GRPOTrainer
Initialization: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
Train dataset: outputs/phase5/packed_stage1_pair_xml_all_traces_train.jsonl
Eval dataset: outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
Steps: 30
Num generations: 4
Temperature: 0.7
Max completion length: 256
Learning rate: 1e-6
Beta: 0.04
```

| Arm | Accuracy | Family accuracy | Parse complete | Mismatch | Suspicious | Sampled malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 0.7188 | 0.5625 | 1.0000 | 0.0000 | 0.0000 | 0 |
| Iso 0.25 | 0.7500 | 0.6250 | 1.0000 | 0.0000 | 0.0000 | 0 |
| Iso 0.50 | 0.7500 | 0.6250 | 1.0000 | 0.0000 | 0.0000 | 0 |
| Iso 1.00 | 0.7813 | 0.6875 | 1.0000 | 0.0000 | 0.0000 | 0 |

Comparison against independent:

| Arm | Accuracy delta | Family accuracy delta | Parse delta |
| --- | ---: | ---: | ---: |
| Iso 0.25 | +0.0312 | +0.0625 | 0.0000 |
| Iso 0.50 | +0.0312 | +0.0625 | 0.0000 |
| Iso 1.00 | +0.0625 | +0.1250 | 0.0000 |

The family-accuracy trend is monotonic:

```text
independent: 0.5625
lambda_iso=0.25: 0.6250
lambda_iso=0.50: 0.6250
lambda_iso=1.00: 0.6875
```

This is stronger evidence than any single iso-vs-independent delta. It is the expected shape if the family reward is doing incremental work rather than injecting arbitrary reward noise. It is still a narrow result, not a broad generalization claim.

## Seed Replication

The first clean iso comparison at `lambda_iso=0.50` replicated across two seeds.

| Seed | Independent accuracy | Iso accuracy | Independent family accuracy | Iso family accuracy | Family delta | Parse stable |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 23 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 | yes |
| 37 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 | yes |

For seed 37, independent and iso had identical contrast-driver prompt groups. This matters because it rules out the simple explanation that the iso arm improved only because it happened to receive easier or more useful contrast prompts.

## By-Family Interpretation

The narrow result is driven mostly by `missing_average`, with `rational_linear_equation` improving only at the highest lambda.

| Arm | `missing_average` accuracy | `missing_average` family acc. | `rational_linear_equation` accuracy | `rational_linear_equation` family acc. |
| --- | ---: | ---: | ---: | ---: |
| Independent | 0.9167 | 0.8333 | 0.6000 | 0.4000 |
| Iso 0.25 | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso 0.50 | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso 1.00 | 1.0000 | 1.0000 | 0.6500 | 0.5000 |

Interpretation:

```text
missing_average benefits clearly from family-level reward.
rational_linear_equation has a lower capability ceiling but improves at lambda_iso=1.00.
```

Do not claim that the result generalizes to all calibrated family types.

## Why The Result Is Trustworthy

Earlier phases showed that the project was not initially blocked by GRPO mechanics. It was blocked by the interface between model output and verifier.

The original weak link was:

```text
model completion -> parser -> verifier -> reward
```

Phase 5 fixed this by moving to a strict packed XML answer contract:

```xml
<answers>
<answer_1>33</answer_1>
<answer_2>16/5</answer_2>
</answers>
```

The final Phase 5 all-traces adapter passed the interface gate:

```text
greedy parse_complete_rate: 1.0000
sampled parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
malformed samples: 0
reward_std: non-degenerate
```

Then Phase 6 replaced the local `grpo_lite` scaffold with proper TRL:

```text
trl: 0.17.0
transformers: 5.0.0.dev0
accelerate: 1.10.0
datasets: 4.0.0
peft: 0.17.0
```

The TRL smoke preserved the XML interface under real GRPO updates:

```text
sampled parse_complete_rate: 1.0000
post-update heldout parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
sampled malformed completions: 0
```

This is why the Phase 6 result is cleaner than the earlier Phase 2/3 evidence. The reward surface is now verifier-stable.

## Current Code Architecture

The current result is built on the packed XML RLVR path, not the old single-problem `grpo_lite` path.

| Area | File | Responsibility |
| --- | --- | --- |
| Packed answer parser | `src/iso_rlvr/rewards/packed_answer.py` | Extracts strict XML answer blocks, normalizes numeric values, rejects malformed answer values. |
| Packed reward | `src/iso_rlvr/rewards/packed_iso.py` | Scores packed completions with correctness, format, penalties, and optional family reward. |
| Packed diagnostics | `src/iso_rlvr/eval/packed_diagnostics.py` | Tracks parse completeness, mismatch, suspicious patterns, repeated-answer diagnostics, and malformed cases. |
| Packed eval | `src/iso_rlvr/eval/run_packed_eval.py` | Runs deterministic packed heldout evaluation and by-family summaries. |
| Rollout audit | `src/iso_rlvr/eval/run_packed_rollout_audit.py` | Samples grouped completions before training and checks reward variance, contrast, and malformed modes. |
| Format SFT builder | `src/iso_rlvr/data/build_format_sft_dataset.py` | Builds XML and trace-to-XML supervised bridge datasets. |
| TRL trainer | `src/iso_rlvr/train/packed_grpo_trl.py` | Runs proper TRL `GRPOTrainer`, wraps the stateless packed reward function, logs per-prompt reward distributions. |
| Historical trainer | `src/iso_rlvr/train/grpo_lite.py` | Early local scaffold; useful historically but not the current result path. |

Current training stack:

```text
trl: 0.17.0
transformers: 5.0.0.dev0
accelerate: 1.10.0
datasets: 4.0.0
peft: 0.17.0
```

Current model path:

```text
Base model: Qwen/Qwen2.5-Math-1.5B
Policy initialization: Phase 5 all-traces LoRA adapter
Adapter: outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
```

## Reward Contract

Independent reward:

```text
correctness + format reward
family bonus disabled
```

Iso reward:

```text
correctness + format reward + family component
```

The Phase 6 family component uses two weights:

```text
family_mean_weight
all_family_correct_weight
```

The lambda settings correspond to:

| Lambda | `family_mean_weight` | `all_family_correct_weight` |
| ---: | ---: | ---: |
| 0.25 | 0.125 | 0.125 |
| 0.50 | 0.250 | 0.250 |
| 1.00 | 0.500 | 0.500 |

Critical rule:

```text
No correctness credit unless the packed XML answer interface is parse-complete.
```

This prevents malformed completions from receiving accidental task credit.

## Broad-Family Status

The broader calibrated family types are not part of the main Phase 6 claim:

```text
chinese_remainder
rational_system_target
```

They became mostly parse-stable under later bridge experiments, but they are still weak RL targets:

```text
chinese_remainder sampled accuracy: 0.0000
chinese_remainder correctness contrast: 0 prompts
rational_system_target sampled accuracy: 0.0208
rational_system_target correctness contrast: 1 prompt
```

These are capability and rollout-contrast blockers, not evidence against the narrow Iso-RLVR result.

## Phase 7 Status

Phase 7 tested a future-work direction:

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

What Phase 7 established:

```text
Parser/reward isolation for <think> is working.
Minimal-think SFT teaches the tag surface but destroys too much math behavior.
Hybrid-think SFT preserves working-family capability better.
Hard-family correctness contrast remains too sparse for meaningful GRPO.
Working-family deterministic eval remains stable after a tiny GRPO smoke.
Sampled working-family rollouts still fail the parse gate.
```

Known blockers:

| Blocker | Status | Evidence |
| --- | --- | --- |
| Empty or missing-think completions | Mostly fixed by response prefix | `think_block_rate: 1.0000` under `<think>\n` prefix |
| Missing final `<answers>` block after `</think>` | Still open | prefixed `parse_complete_rate: 0.8438` |
| Hard-family correctness contrast | Still too sparse | CRT has one contrast prompt; rational-system has zero sampled accuracy |

Phase 7 is closed as a deferred extension. It should not delay the report.

## What The Report Can Claim

Supported claim:

```text
On a stable two-family packed XML verifier interface, Iso-RLVR improves family accuracy over independent RLVR under proper TRL GRPO, with no parse regression.
```

More specific supported wording:

```text
At lambda_iso=0.50, the family-accuracy improvement replicated across two seeds.
At seed 23, increasing lambda_iso from 0.25 to 1.00 produced a monotonic family-accuracy trend.
The best observed narrow run, lambda_iso=1.00, improved family_accuracy by +0.1250 over independent.
```

Do not claim:

- that Iso-RLVR generally improves mathematical reasoning across benchmarks
- that the effect has been proven statistically
- that the result generalizes to `chinese_remainder` or `rational_system_target`
- that `<think> + <answers>` GRPO has been solved
- that larger models will necessarily solve the deferred families

## Limitations A Critic Should Challenge

The main limitations are:

1. The main heldout set is small.
2. The main claim covers only two procedural family types.
3. The `lambda_iso=1.00` best result has not yet been replicated across multiple seeds.
4. The two-seed replication is for `lambda_iso=0.50`, not for every lambda.
5. The broad calibrated family types are not RL-ready under the current bridge.
6. The result is local to `Qwen/Qwen2.5-Math-1.5B` plus LoRA.
7. The reward checks final answers only, not reasoning validity.
8. The XML interface is engineered; other answer contracts may behave differently.
9. No bootstrap confidence interval over families has been reported yet.
10. There is no final blind benchmark untouched by iteration.
11. Phase 7 shows that adding a reasoning channel introduces new format-control problems.

These caveats should stay in the final write-up.

## Suggested Report Structure

1. Lead with the Phase 6 result table and monotonic lambda trend.
2. State the narrow claim and scope boundary immediately.
3. Explain the reward-interface failure that made earlier results unreliable.
4. Describe the Phase 5 XML stabilization and SFT bridge.
5. Describe the proper TRL GRPO integration.
6. Present the replicated `lambda_iso=0.50` comparison.
7. Present the `lambda_iso` sweep and by-family breakdown.
8. Discuss broad-family failures as scope boundaries, not contradictions.
9. Summarize Phase 7 as future work on free reasoning channels.
10. End with limitations and next experiments.

## Reproduction Commands

Install and test:

```bash
conda run -n pytorch_5070ti python -m pip install -e .
conda run -n pytorch_5070ti pytest tests
```

Run the proper TRL trainer smoke:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_smoke.yaml
```

Run the seed 23 independent-vs-iso matrix:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_independent_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_25_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_1_00_30step.yaml
```

Run the seed 37 confirmation pair:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_independent_30step_seed_37.yaml
conda run -n pytorch_5070ti python -m iso_rlvr.train.packed_grpo_trl --config configs/packed_grpo_trl_all_traces_iso_lam_0_50_30step_seed_37.yaml
```

The configs run final packed heldout eval after saving adapters. For standalone eval, use the matching `packed_eval_*` configs documented in `phase6.md`.

## Next Work

For the main report:

- write the report around the Phase 6 narrow result
- include Phase 5 as methods and reliability evidence
- include Phase 7 only as future work

For future experiments:

- replicate `lambda_iso=1.00` across more seeds
- add family-level bootstrap intervals
- build a larger heldout set for the two working families
- test a final blind procedural split
- revisit `<think> + <answers>` only after enforcing the `</think>\n<answers>` transition
- test harder family types on larger models after the two-tag interface is stable

## Current Decision

Stop running Phase 7. Move to the report.

The current project result is:

```text
Phase 5 made the verifier interface reliable.
Phase 6 showed a clean, narrow Iso-RLVR family-accuracy gain under proper TRL GRPO.
Phase 7 produced useful diagnostics but remains future work.
```
