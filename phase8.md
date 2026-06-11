# Phase 8: Statistical Power And Confound Controls

Status: complete; Phase 6 claim did not survive

Date: 2026-06-11 (runs), 2026-06-12 (analysis)

## Result Summary

The Phase 6 result did not survive the large heldout set.

All seven Phase 6 checkpoints were re-evaluated on 452 fresh families (28x the original heldout). Every arm landed inside a band of roughly one point:

| Arm | Accuracy | Family accuracy | Parse complete |
| --- | ---: | ---: | ---: |
| Base all-traces adapter | 0.7312 | 0.5487 | 1.0000 |
| Independent 30-step, seed 23 | 0.7367 | 0.5531 | 1.0000 |
| Iso `lambda_iso=0.25`, seed 23 | 0.7334 | 0.5487 | 1.0000 |
| Iso `lambda_iso=0.50`, seed 23 | 0.7301 | 0.5465 | 1.0000 |
| Iso `lambda_iso=1.00`, seed 23 | 0.7301 | 0.5465 | 1.0000 |
| Independent 30-step, seed 37 | 0.7312 | 0.5465 | 1.0000 |
| Iso `lambda_iso=0.50`, seed 37 | 0.7279 | 0.5398 | 1.0000 |

Paired comparisons (10000-resample bootstrap over families, sign-flip permutation test):

| Comparison | Family-accuracy delta | 95% CI | p |
| --- | ---: | --- | ---: |
| Iso 0.25 s23 vs independent s23 | -0.0044 | [-0.0199, +0.0111] | 0.78 |
| Iso 0.50 s23 vs independent s23 | -0.0066 | [-0.0221, +0.0066] | 0.55 |
| Iso 1.00 s23 vs independent s23 | -0.0066 | [-0.0221, +0.0066] | 0.55 |
| Iso 0.50 s37 vs independent s37 | -0.0066 | [-0.0221, +0.0088] | 0.57 |
| Independent s23 vs base adapter | +0.0044 | [-0.0066, +0.0177] | 0.73 |
| Iso 1.00 s23 vs base adapter | -0.0022 | [-0.0177, +0.0111] | 1.00 |

Cross-context (unpacked) family consistency, 904 single-variant prompts over the same 452 families:

| Arm | Accuracy | Cross-context family accuracy |
| --- | ---: | ---: |
| Base adapter | 0.7080 | 0.5133 |
| Independent s23 | 0.7113 | 0.5199 |
| Iso `lambda_iso=1.00` s23 | 0.7102 | 0.5155 |

Findings, stated per the pre-registered interpretation rules:

```text
1. No iso arm beats its matched independent arm. Every iso delta is slightly negative
   and every CI straddles zero. The Phase 6 monotonic lambda trend and the +0.1250
   family-accuracy headline were small-sample noise on 16 families.
2. No RL arm differs from the base adapter. Thirty GRPO steps at this scale moved
   nothing measurable, in any reward condition.
3. The Phase 6 observation that independent RLVR degraded the base adapter
   (0.5625 vs 0.6250 on n=16) also did not replicate: +0.0044, p=0.73.
4. The packed-vs-unpacked question is moot at this effect size: cross-context
   consistency is flat across arms (0.51-0.52).
5. The verifier interface remained perfect everywhere: parse_complete 1.0000 across
   all 3,164 packed and 2,712 unpacked generations. The Phase 5 interface result stands.
```

Workstream B (random-reward control) was not completed and is now moot: there is no effect left for the control to explain. The training step also exposed environment drift: the env now has `trl 1.6.0.dev0`, while the Phase 6 checkpoints were trained under `trl 0.17.0`, and `GRPOConfig` no longer accepts the Phase 6 config surface (`max_prompt_length` removed). A newly trained random arm would not be version-matched to the Phase 6 checkpoints anyway. If any future phase trains again, pin `trl` first.

Workstream D (lambda 1.00 seed replication) is cancelled per its own gate: Workstream A killed the arm it would have replicated.

The correct claim after Phase 8:

```text
On a 452-family heldout set with paired bootstrap CIs, 30-step Iso-RLVR is
indistinguishable from independent RLVR and from the untouched SFT adapter.
The Phase 6 positive result was an artifact of a 16-family evaluation set.
```

This is the outcome the success criteria called the minimum useful one: an over-claimed table converted into a calibrated null. `CRITIC_BRIEF.md` and `BLOG_POST.md` must be rewritten around the interface lesson plus this measurement correction, not around an Iso-RLVR gain.

## Purpose

Phase 6 produced a clean-looking narrow result, but it rests on a 16-family heldout set. Every delta in the Phase 6 table is one or two families flipping. The monotonic lambda trend is:

```text
9/16 -> 10/16 -> 10/16 -> 11/16
```

That is not yet evidence. Phase 8 does not chase a bigger effect. Phase 8 asks whether the Phase 6 effect is real at all.

Phase 8 asks three questions:

```text
1. Does the Phase 6 family-accuracy gain survive a large heldout set with paired uncertainty estimates?
2. Does the gain survive a random-reward control arm, given the known Qwen2.5-Math spurious-reward pathology?
3. Does family consistency transfer when variants appear in separate contexts instead of one packed prompt?
```

No new reward designs. No new family types. No new models. Measurement first.

## Current Diagnosis

Three weaknesses in the Phase 6 claim, in priority order.

### Weakness 1: Statistical power

The main heldout set is:

```text
outputs/phase5/packed_stage1_pair_xml_sft_heldout.jsonl
16 packed rows, 16 families, 32 variants
```

The headline `+0.1250` family-accuracy delta is two families. No confidence interval has been computed. This is limitation 1 and limitation 10 from `CRITIC_BRIEF.md`, and it currently nullifies the claim rather than qualifying it.

### Weakness 2: The Qwen2.5-Math confound

Published work (Spurious Rewards, arXiv 2506.10947) showed Qwen2.5-Math models gain substantially from random and even incorrect rewards, an effect that does not reproduce on Llama-3 or OLMo-2. Any RLVR delta on this model family needs a random-reward arm before the reward design gets credit. The procedural dataset reduces contamination risk but does not remove the pathology.

### Weakness 3: Packed prompts changed the hypothesis

The original hypothesis is about rule learning that holds across independently encountered variants. The Phase 6 setup puts both variants in one prompt. The model can read variant 1 while answering variant 2. That measures in-context consistency, which is weaker than the original claim. The fix is an eval-side test: evaluate the packed-trained checkpoints on single-variant prompts and group by family post hoc.

## Phase 8 Design Rules

- Reuse the seven existing Phase 6 checkpoints. No retraining for Workstream A or C.
- The new eval set must come from a fresh generation seed and pass the variant-level overlap audit against the seed-29 calibrated generation with `overlap_count: 0`.
- Generation settings must match the Phase 6 eval exactly: greedy, `max_new_tokens: 256`, no chat template, no response prefix.
- Every comparison is paired by `family_id` across arms. Report bootstrap confidence intervals over families and a paired sign-flip permutation p-value, not just point deltas.
- The random-reward arm uses the identical pipeline, data, and hyperparameters as the Phase 6 independent arm. The only change is that training rewards are replaced by seeded Bernoulli noise. Verifier diagnostics are still logged.

## Workstream A: Large Heldout And Paired Uncertainty

### A1: Generate the Phase 8 eval families

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_dataset --out data/iso_math_phase8_eval.jsonl --families 600 --variants 4 --seed 101 --profile calibrated
```

The calibrated profile draws roughly 5/10 `rational_linear_equation`, 3/10 `missing_average`. 600 families should yield roughly 480 working-family rows after filtering.

### A2: Overlap audit and family-level exclusion

First finding: no clean seed exists. A scan of generation seeds 101 through 140 produced between 6 and 19 variant collisions per 2400 rows against the seed-29 calibrated generation. Collisions are inherent to the generator parameter space at this draw size, so "regenerate until zero" is not a viable rule.

The fixed exclusion rule, decided before any model was evaluated:

```text
Drop every family that contains any variant whose fingerprint appears in the
seed-29 calibrated generation. Filtering is by family, under an automated rule.
No per-row hand selection.
```

Tooling:

```text
src/iso_rlvr/data/filter_overlap.py
```

Commands:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.filter_overlap --input data/iso_math_phase8_eval.jsonl --train data/iso_math_calibrated.jsonl --out data/iso_math_phase8_eval_clean.jsonl --report outputs/phase8/filter_overlap_report.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.audit_dataset_overlap --train data/iso_math_calibrated.jsonl --heldout data/iso_math_phase8_eval_clean.jsonl --out outputs/phase8/overlap_audit_phase8_eval_clean.json
```

Result:

```text
seed 101 generation: 600 families, 2400 rows, 14 colliding rows
dropped families: 14
clean file: data/iso_math_phase8_eval_clean.jsonl (586 families, 2344 rows)
audit on clean file: overlap_count 0
```

Pass condition met:

```text
overlap_count: 0
```

### A3: Pack to the Phase 6 prompt shape

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_packed_dataset --input data/iso_math_phase8_eval_clean.jsonl --out outputs/phase8/packed_phase8_eval_pair_xml.jsonl --include-family-type missing_average,rational_linear_equation --expected-variants 4 --max-variants-per-family 2 --prompt-format xml
```

This mirrors the Phase 5 packing: first two variants per family, two-variant XML prompt.

Result:

```text
outputs/phase8/packed_phase8_eval_pair_xml.jsonl: 452 packed rows, 452 families
28x larger than the Phase 6 heldout set.
```

### A4: Paired comparison tooling

New module:

```text
src/iso_rlvr/eval/bootstrap_compare.py
```

Responsibilities:

- Join two packed eval output files by `family_id`.
- Compute per-family variant accuracy and all-correct family accuracy.
- Paired bootstrap over families (default 10000 resamples, seeded) for the accuracy delta and family-accuracy delta, reporting 95 percent percentile intervals.
- Paired sign-flip permutation test (two-sided) on per-family deltas.
- Deterministic under a fixed seed. Unit tested.

### A5: Rerun all seven Phase 6 checkpoints

Arms, all on `outputs/phase8/packed_phase8_eval_pair_xml.jsonl`, greedy, 256 tokens:

```text
base adapter:      outputs/phase5/format_sft_qwen25_math_1_5b_xml_all_traces_1epoch/adapter_or_model
independent s23:   outputs/phase6/packed_grpo_trl_all_traces_independent_30step/adapter_or_model
iso 0.25 s23:      outputs/phase6/packed_grpo_trl_all_traces_iso_lam_0_25_30step/adapter_or_model
iso 0.50 s23:      outputs/phase6/packed_grpo_trl_all_traces_iso_lam_0_50_30step/adapter_or_model
iso 1.00 s23:      outputs/phase6/packed_grpo_trl_all_traces_iso_lam_1_00_30step/adapter_or_model
independent s37:   outputs/phase6/packed_grpo_trl_all_traces_independent_30step_seed_37/adapter_or_model
iso 0.50 s37:      outputs/phase6/packed_grpo_trl_all_traces_iso_lam_0_50_30step_seed_37/adapter_or_model
```

Configs:

```text
configs/eval_phase8_base_all_traces.yaml
configs/eval_phase8_independent_seed23.yaml
configs/eval_phase8_iso_lam_0_25_seed23.yaml
configs/eval_phase8_iso_lam_0_50_seed23.yaml
configs/eval_phase8_iso_lam_1_00_seed23.yaml
configs/eval_phase8_independent_seed37.yaml
configs/eval_phase8_iso_lam_0_50_seed37.yaml
```

All configs set `resume: true` so the queue is restartable.

### A6: Paired comparisons to report

```text
iso 0.25 s23 vs independent s23
iso 0.50 s23 vs independent s23
iso 1.00 s23 vs independent s23
iso 0.50 s37 vs independent s37
independent s23 vs base adapter
iso 1.00 s23 vs base adapter
```

Pass condition for the Phase 6 claim:

```text
The family-accuracy delta for at least one iso arm vs its matched independent arm
has a 95 percent CI that excludes zero on the large heldout set.
```

If no CI excludes zero, the Phase 6 claim is downgraded to "not distinguishable from noise at this scale" and the blog post must say so.

Also report `independent vs base`. Phase 6 showed independent RLVR scoring below the starting adapter. If that replicates, the honest framing may be "iso reward prevents degradation" rather than "iso reward improves learning."

## Workstream B: Random-Reward Control Arm

### B1: Trainer change

Add `reward_mode` to `packed_grpo_trl.py`:

```text
reward_mode: verifier   (default, current behavior)
reward_mode: random     (training rewards replaced by seeded Bernoulli 0.5)
```

Rules for `random` mode:

- The verifier still runs and full reward records are still logged, so interface drift remains observable.
- The returned training rewards are pure noise, independent of the completion.
- Seeded by `reward_mode_seed` (default: run seed) for reproducibility.
- Unit tested: same pipeline, noise rewards returned, real diagnostics logged.

### B2: Training run

```text
configs/packed_grpo_trl_all_traces_random_30step.yaml
```

Identical to `packed_grpo_trl_all_traces_independent_30step.yaml` except `reward_mode: random` and the output directory. Seed 23, 30 steps, same Phase 5 adapter initialization.

### B3: Evaluation and interpretation

Evaluate on the Phase 8 packed eval set:

```text
configs/eval_phase8_random_seed23.yaml
```

Interpretation rules, fixed before looking at results:

```text
random ~= base, independent ~= base, iso > both: cleanest possible support for the family reward.
random ~= independent, both != base: GRPO is moving the policy regardless of signal; iso deltas
  must be read against the random arm, not against independent.
random ~= iso: the Phase 6 effect is not attributable to the reward design. Claim withdrawn.
```

## Workstream C: Unpacked Consistency Transfer

### C1: Exploded single-variant eval set

Add `--explode-variants` to `build_packed_dataset.py`: one output row per variant, `num_variants: 1`, single-problem XML prompt, `family_id` preserved, plus a unique `row_id` for resume bookkeeping.

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_packed_dataset --input data/iso_math_phase8_eval_clean.jsonl --out outputs/phase8/unpacked_phase8_eval_xml.jsonl --include-family-type missing_average,rational_linear_equation --expected-variants 4 --max-variants-per-family 2 --explode-variants --prompt-format xml
```

Result:

```text
outputs/phase8/unpacked_phase8_eval_xml.jsonl: 904 single-variant rows, 452 families
```

`run_packed_eval.py` keys its resume set on `row_id` when present, since exploded rows share `family_id`.

### C2: Post-hoc family consistency summarizer

New module:

```text
src/iso_rlvr/eval/summarize_family_consistency.py
```

Reads an exploded eval output file, groups rows by `family_id`, and reports:

```text
variant accuracy
cross-context family accuracy (all variants of a family correct in separate prompts)
by-family-type breakdown
parse_complete_rate
```

### C3: Arms to evaluate

Minimum three arms:

```text
configs/eval_phase8_unpacked_base.yaml
configs/eval_phase8_unpacked_independent_seed23.yaml
configs/eval_phase8_unpacked_iso_lam_1_00_seed23.yaml
```

### C4: Interpretation

```text
If iso > independent on cross-context family accuracy with a CI excluding zero,
the rule-learning framing survives.
If the iso advantage disappears out of the packed context, the Phase 6 result is
in-context consistency only, and the claim must be narrowed accordingly.
```

Either outcome is publishable. The second outcome is arguably the more interesting blog post.

## Workstream D: Optional Replication

Only after A, B, and C are read:

```text
configs/packed_grpo_trl_all_traces_iso_lam_1_00_30step_seed_37.yaml
configs/packed_grpo_trl_all_traces_iso_lam_1_00_30step_seed_41.yaml
```

The best Phase 6 arm (`lambda_iso=1.00`) has a single seed. If Workstream A shows its CI excludes zero, replicate it before reporting it as the headline number. If Workstream A kills it, skip Workstream D.

## Execution Order

```text
1. Code: bootstrap_compare, reward_mode random, explode-variants, row_id resume, consistency summarizer. Tests green.
2. Data: A1 generate, A2 filter plus audit (gate: overlap_count 0), A3 pack, C1 explode.
3. GPU queue, sequential: A5 seven packed evals, then C3 three unpacked evals, then B2 random training plus its eval.
4. Analysis: A6 paired comparisons, C4 transfer comparison, B3 interpretation.
5. Update CRITIC_BRIEF.md and BLOG_POST.md with whatever the CIs actually say.
```

The GPU queue is one script:

```bash
bash scripts/run_phase8_queue.sh
```

It logs to `outputs/phase8/queue.log`. All eval configs set `resume: true`, so rerunning the script after an interruption skips completed rows.

## Analysis Commands

Paired comparisons after the packed evals finish:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_iso_lam_0_25_seed23.jsonl --b outputs/phase8/packed_eval_phase8_independent_seed23.jsonl --label-a iso_lam_0_25_s23 --label-b independent_s23 --out outputs/phase8/compare_iso_0_25_vs_independent_seed23.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_iso_lam_0_50_seed23.jsonl --b outputs/phase8/packed_eval_phase8_independent_seed23.jsonl --label-a iso_lam_0_50_s23 --label-b independent_s23 --out outputs/phase8/compare_iso_0_50_vs_independent_seed23.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_iso_lam_1_00_seed23.jsonl --b outputs/phase8/packed_eval_phase8_independent_seed23.jsonl --label-a iso_lam_1_00_s23 --label-b independent_s23 --out outputs/phase8/compare_iso_1_00_vs_independent_seed23.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_iso_lam_0_50_seed37.jsonl --b outputs/phase8/packed_eval_phase8_independent_seed37.jsonl --label-a iso_lam_0_50_s37 --label-b independent_s37 --out outputs/phase8/compare_iso_0_50_vs_independent_seed37.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_independent_seed23.jsonl --b outputs/phase8/packed_eval_phase8_base_all_traces.jsonl --label-a independent_s23 --label-b base_adapter --out outputs/phase8/compare_independent_seed23_vs_base.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_iso_lam_1_00_seed23.jsonl --b outputs/phase8/packed_eval_phase8_base_all_traces.jsonl --label-a iso_lam_1_00_s23 --label-b base_adapter --out outputs/phase8/compare_iso_1_00_seed23_vs_base.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.bootstrap_compare --a outputs/phase8/packed_eval_phase8_random_seed23.jsonl --b outputs/phase8/packed_eval_phase8_independent_seed23.jsonl --label-a random_s23 --label-b independent_s23 --out outputs/phase8/compare_random_vs_independent_seed23.json
```

Cross-context consistency after the unpacked evals finish:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.summarize_family_consistency --input outputs/phase8/unpacked_eval_phase8_base.jsonl --out outputs/phase8/unpacked_consistency_base.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.summarize_family_consistency --input outputs/phase8/unpacked_eval_phase8_independent_seed23.jsonl --out outputs/phase8/unpacked_consistency_independent_seed23.json
conda run -n pytorch_5070ti python -m iso_rlvr.eval.summarize_family_consistency --input outputs/phase8/unpacked_eval_phase8_iso_lam_1_00_seed23.jsonl --out outputs/phase8/unpacked_consistency_iso_lam_1_00_seed23.json
```

## What Phase 8 Does Not Do

- No new family types. `chinese_remainder` and `rational_system_target` stay parked.
- No `<think>` channel work. Phase 7 remains closed.
- No model swap. Qwen3 and non-Qwen replication are Phase 9 candidates, contingent on Phase 8.
- No lambda retuning. The sweep only earns extension if the current points survive CIs.

## Success Criteria

Minimum useful outcome:

```text
The Phase 6 deltas are re-measured on a heldout set 25x larger, with paired CIs,
against both an independent arm and a random-reward arm.
```

That outcome is useful even if every CI includes zero, because it converts an over-claimed table into a calibrated one.

Strong outcome:

```text
At least one iso arm beats its matched independent arm on family accuracy with a CI
excluding zero, the random arm shows no comparable gain, and the iso advantage
survives in the unpacked cross-context eval.
```
