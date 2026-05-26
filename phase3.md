# Phase 3: Measurement Hardening

Date: 2026-05-26

## Objective

Phase 3 strengthens the measurement layer before changing the trainer. The current Iso `lambda_iso=0.50` result is directionally positive across two seeds, but the held-out eval was only 80 rows and did not include family-type breakdowns or a documented dataset-overlap audit.

The Phase 3 question is:

```text
Does the Iso-RLVR family-accuracy signal survive stricter measurement?
```

## Starting State

The latest Phase 2 result:

| Run | Accuracy | Family accuracy |
| --- | ---: | ---: |
| Base held-out, 80 rows | 0.6000 | 0.2500 |
| Independent 20-step, seed 13, 80 rows | 0.6125 | 0.2500 |
| Iso 20-step, `lambda_iso=0.50`, seed 13, 80 rows | 0.6250 | 0.3000 |
| Independent 20-step, seed 23, 80 rows | 0.6375 | 0.3000 |
| Iso 20-step, `lambda_iso=0.50`, seed 23, 80 rows | 0.6375 | 0.3500 |

Conclusion carried into Phase 3:

```text
Promising local signal, not yet a strong result.
```

## Phase 3 Plan

Run the work in this order:

1. Add exact train/held-out overlap audit.
2. Add family-type breakdowns to evaluation summaries.
3. Add full held-out eval configs that use all 800 held-out rows.
4. Run cheap validation checks.
5. Run full held-out evals one at a time.
6. Update this document after each result.
7. Only after measurement is stronger, continue the lambda sweep.

## Added Tooling

Dataset overlap audit:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.audit_dataset_overlap --train data/iso_math_calibrated.jsonl --heldout data/iso_math_calibrated_heldout.jsonl --out outputs/audit/calibrated_train_vs_heldout_overlap.json
```

The audit hashes:

```text
family_type
problem
answer
```

Family-type breakdowns are now written into every new eval summary under:

```text
by_family_type
```

This lets us distinguish a real broad improvement from a gain concentrated in one procedural family type.

Clean held-out generation:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_clean_heldout --train data/iso_math_calibrated.jsonl --out data/iso_math_calibrated_heldout_clean.jsonl --families 200 --variants 4 --profile calibrated --start-seed 32 --max-seed 200
```

This searches for a held-out generator seed with:

```text
train/held-out exact overlap: 0
```

Internal held-out duplicates are still reported. Requiring zero internal duplicates at 800 rows is too strict for the current discrete calibrated generator, especially the heavily weighted rational-linear family type.

## Full Held-Out Eval Configs

These configs evaluate all rows in `data/iso_math_calibrated_heldout_clean.jsonl` because they do not set `max_examples`:

```text
configs/baseline_eval_calibrated_heldout_full.yaml
configs/eval_independent_calibrated_20step_full.yaml
configs/eval_iso_calibrated_lam_0_50_20step_full.yaml
configs/eval_independent_calibrated_20step_seed_23_full.yaml
configs/eval_iso_calibrated_lam_0_50_20step_seed_23_full.yaml
```

They also set:

```text
resume: true
```

Long evals now write one JSONL row at a time and skip completed `(family_id, variant_id)` rows when restarted. This is necessary because a clean 800-row eval can take more than an hour at `max_new_tokens: 256`.

## Run Log

### Tooling Validation

Command:

```bash
conda run -n pytorch_5070ti python -m pytest tests
```

Result:

```text
15 passed
```

Pytest still emits a cache warning because this Windows workspace denies writing one `.pytest_cache` path, but tests pass.

### Existing Held-Out Overlap Audit

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.audit_dataset_overlap --train data/iso_math_calibrated.jsonl --heldout data/iso_math_calibrated_heldout.jsonl --out outputs/audit/calibrated_train_vs_heldout_overlap.json
```

Result:

```json
{
  "train_rows": 800,
  "heldout_rows": 800,
  "train_unique_fingerprints": 798,
  "heldout_unique_fingerprints": 797,
  "train_duplicate_fingerprints": 2,
  "heldout_duplicate_fingerprints": 3,
  "overlap_count": 3
}
```

The 3 exact train/held-out overlaps are all `rational_linear_equation` rows. This does not affect the earlier 80-row eval slice, because those duplicate held-out families occur after the first 80 rows. It does mean the existing 800-row held-out file should not be used as the final full held-out benchmark.

Decision:

```text
Generate a clean held-out file before running full 800-row model evals.
```

### Family-Type Breakdown For Existing 80-Row Evals

The existing eval outputs were re-summarized with the new family-type breakdown command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.summarize_eval_output --input outputs/eval/<run>.jsonl --out outputs/eval/<run>.summary.json
```

Breakdown:

| Run | Type | Examples | Families | Accuracy | Family accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | `chinese_remainder` | 8 | 2 | 0.0000 | 0.0000 |
| Base | `missing_average` | 12 | 3 | 0.5833 | 0.0000 |
| Base | `rational_linear_equation` | 52 | 13 | 0.7885 | 0.3846 |
| Base | `rational_system_target` | 8 | 2 | 0.0000 | 0.0000 |
| Independent seed 13 | `chinese_remainder` | 8 | 2 | 0.0000 | 0.0000 |
| Independent seed 13 | `missing_average` | 12 | 3 | 0.6667 | 0.0000 |
| Independent seed 13 | `rational_linear_equation` | 52 | 13 | 0.7885 | 0.3846 |
| Independent seed 13 | `rational_system_target` | 8 | 2 | 0.0000 | 0.0000 |
| Iso seed 13 | `chinese_remainder` | 8 | 2 | 0.0000 | 0.0000 |
| Iso seed 13 | `missing_average` | 12 | 3 | 0.6667 | 0.0000 |
| Iso seed 13 | `rational_linear_equation` | 52 | 13 | 0.8077 | 0.4615 |
| Iso seed 13 | `rational_system_target` | 8 | 2 | 0.0000 | 0.0000 |
| Independent seed 23 | `chinese_remainder` | 8 | 2 | 0.0000 | 0.0000 |
| Independent seed 23 | `missing_average` | 12 | 3 | 0.7500 | 0.3333 |
| Independent seed 23 | `rational_linear_equation` | 52 | 13 | 0.8077 | 0.3846 |
| Independent seed 23 | `rational_system_target` | 8 | 2 | 0.0000 | 0.0000 |
| Iso seed 23 | `chinese_remainder` | 8 | 2 | 0.0000 | 0.0000 |
| Iso seed 23 | `missing_average` | 12 | 3 | 0.7500 | 0.3333 |
| Iso seed 23 | `rational_linear_equation` | 52 | 13 | 0.8077 | 0.4615 |
| Iso seed 23 | `rational_system_target` | 8 | 2 | 0.0000 | 0.0000 |

Conclusion:

The observed 80-row Iso family-accuracy gain is concentrated in `rational_linear_equation`. `chinese_remainder` and `rational_system_target` remain unsolved in this slice across all runs. `missing_average` improves under training, but the Iso-vs-independent difference is not visible there in this slice.

This narrows the current claim:

```text
The current positive signal is not broad transformation robustness yet.
It is a small family-consistency gain concentrated in one procedural family type.
```

### Clean Held-Out Generation

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.data.build_clean_heldout --train data/iso_math_calibrated.jsonl --out data/iso_math_calibrated_heldout_clean.jsonl --families 200 --variants 4 --profile calibrated --start-seed 32 --max-seed 200
```

Result:

```json
{
  "selected_seed": 81,
  "output_path": "data\\iso_math_calibrated_heldout_clean.jsonl",
  "train_rows": 800,
  "heldout_rows": 800,
  "train_unique_fingerprints": 798,
  "heldout_unique_fingerprints": 794,
  "train_duplicate_fingerprints": 2,
  "heldout_duplicate_fingerprints": 6,
  "overlap_count": 0
}
```

Clean held-out family-type counts:

| Family type | Rows |
| --- | ---: |
| `chinese_remainder` | 112 |
| `missing_average` | 252 |
| `rational_linear_equation` | 392 |
| `rational_system_target` | 44 |

Conclusion:

The clean held-out dataset removes exact train/held-out overlap. It still has 6 internal duplicate fingerprints, which is acceptable for now as an artifact of the finite procedural parameter space. The family-type distribution differs from seed 31, so clean full-heldout numbers should be treated as a new benchmark, not a direct replacement for the earlier 80-row slice.

### First Full Eval Attempt

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated_heldout_full.yaml
```

Status:

```text
stopped at 319/800 after about 32 minutes
```

Conclusion:

The full clean eval is too expensive to run with all-or-nothing output. The evaluator was changed to write incremental JSONL rows and support resume before restarting the full benchmark.

### Base Full Clean Held-Out Eval

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated_heldout_full.yaml
```

Runtime:

```text
about 1h15m
```

Output:

```text
outputs/eval/baseline_qwen25_math_1_5b_calibrated_heldout_clean_full.jsonl
outputs/eval/baseline_qwen25_math_1_5b_calibrated_heldout_clean_full.summary.json
rows: 800
```

Overall result:

```json
{
  "accuracy": 0.58875,
  "avg_tokens": 208.28625,
  "avg_wrong_tokens": 239.7629179331307,
  "family_accuracy": 0.24,
  "format_failure_rate": 0.0
}
```

Family-type breakdown:

| Family type | Examples | Families | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 112 | 28 | 0.0268 | 0.0000 | 239.2411 | 240.6972 |
| `missing_average` | 252 | 63 | 0.6627 | 0.1905 | 210.4881 | 228.0118 |
| `rational_linear_equation` | 392 | 98 | 0.7679 | 0.3673 | 192.6709 | 241.7692 |
| `rational_system_target` | 44 | 11 | 0.0000 | 0.0000 | 256.0000 | 256.0000 |

Conclusion:

The clean full base score is close to the earlier 80-row base score (`0.58875` vs `0.6000` accuracy, `0.24` vs `0.25` family accuracy), so the earlier small slice was not wildly optimistic at the aggregate level. The family-type breakdown confirms the central difficulty split:

- `rational_linear_equation` is the strongest solved type.
- `missing_average` is partially solved, but family consistency is still low.
- `chinese_remainder` is almost entirely unsolved.
- `rational_system_target` is completely unsolved and saturates the 256-token cap.

This means the next comparison should check whether independent and Iso adapters improve only the already-solvable types or produce any movement on the currently unsolved types.

### Independent 20-Step Full Clean Held-Out Eval, Seed 13

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_independent_calibrated_20step_full.yaml
```

Runtime:

```text
about 2h22m
```

Output:

```text
outputs/eval/independent_calibrated_20step_heldout_clean_full.jsonl
outputs/eval/independent_calibrated_20step_heldout_clean_full.summary.json
rows: 800
```

Overall result:

```json
{
  "accuracy": 0.5875,
  "avg_tokens": 208.65625,
  "avg_wrong_tokens": 239.2969696969697,
  "family_accuracy": 0.24,
  "format_failure_rate": 0.0
}
```

Family-type breakdown:

| Family type | Examples | Families | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 112 | 28 | 0.0268 | 0.0000 | 243.2500 | 243.4495 |
| `missing_average` | 252 | 63 | 0.6667 | 0.1905 | 209.1389 | 225.6905 |
| `rational_linear_equation` | 392 | 98 | 0.7602 | 0.3673 | 193.2755 | 239.0000 |
| `rational_system_target` | 44 | 11 | 0.0227 | 0.0000 | 254.8636 | 256.0000 |

Comparison to clean base:

| Run | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens |
| --- | ---: | ---: | ---: | ---: |
| Base full clean | 0.5888 | 0.2400 | 208.2863 | 239.7629 |
| Independent seed 13 full clean | 0.5875 | 0.2400 | 208.6563 | 239.2970 |

Conclusion:

On the full clean held-out set, the seed 13 independent adapter does not reproduce the small 80-row gain. It is effectively flat against base: row accuracy is slightly lower by `-0.00125`, and strict family accuracy is unchanged at `0.24`.

The family-type pattern is also mostly flat. The only visible new movement is one correct `rational_system_target` row, but family accuracy remains `0.0` for that type and wrong answers still saturate the token cap. This makes the Iso seed 13 full-clean eval the next critical comparison: the earlier 80-row result needs to show a real family-accuracy advantage on the clean 800-row benchmark, not just on the small slice.

### Iso `lambda_iso=0.50` 20-Step Full Clean Held-Out Eval, Seed 13

Command:

```bash
conda run -n pytorch_5070ti python -m iso_rlvr.eval.run_eval --config configs/eval_iso_calibrated_lam_0_50_20step_full.yaml
```

Runtime:

```text
about 2h21m
```

Output:

```text
outputs/eval/iso_calibrated_lam_0_50_20step_heldout_clean_full.jsonl
outputs/eval/iso_calibrated_lam_0_50_20step_heldout_clean_full.summary.json
rows: 800
```

Overall result:

```json
{
  "accuracy": 0.58,
  "avg_tokens": 209.27125,
  "avg_wrong_tokens": 241.41071428571428,
  "family_accuracy": 0.23,
  "format_failure_rate": 0.0
}
```

Family-type breakdown:

| Family type | Examples | Families | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chinese_remainder` | 112 | 28 | 0.0268 | 0.0000 | 244.7411 | 245.9083 |
| `missing_average` | 252 | 63 | 0.6429 | 0.1746 | 212.4802 | 229.4444 |
| `rational_linear_equation` | 392 | 98 | 0.7628 | 0.3571 | 191.8291 | 240.8172 |
| `rational_system_target` | 44 | 11 | 0.0000 | 0.0000 | 256.0000 | 256.0000 |

Comparison to clean base and matched independent:

| Run | Accuracy | Family accuracy | Avg tokens | Wrong avg tokens |
| --- | ---: | ---: | ---: | ---: |
| Base full clean | 0.5888 | 0.2400 | 208.2863 | 239.7629 |
| Independent seed 13 full clean | 0.5875 | 0.2400 | 208.6563 | 239.2970 |
| Iso `lambda_iso=0.50` seed 13 full clean | 0.5800 | 0.2300 | 209.2713 | 241.4107 |

Delta from matched independent:

| Metric | Iso - Independent |
| --- | ---: |
| Accuracy | -0.0075 |
| Family accuracy | -0.0100 |
| Avg tokens | +0.6150 |
| Wrong avg tokens | +2.1137 |

Conclusion:

The seed 13 Iso result does not survive the full clean held-out measurement. The earlier 80-row Iso family-accuracy gain was `+0.05` over matched independent, but the full clean result is `-0.01` family accuracy and `-0.0075` row accuracy. The result is also below the base model on both metrics.

The per-family breakdown shows the regression is not offset by movement on the previously unsolved families:

- `chinese_remainder` stays essentially unsolved with `0.0` family accuracy.
- `rational_system_target` remains completely unsolved and still saturates the 256-token cap.
- `missing_average` is worse than independent on row accuracy and family accuracy.
- `rational_linear_equation` is slightly better than independent on row accuracy, but worse on strict family accuracy and still below the earlier 80-row family-consistency signal.

This changes the Phase 3 interpretation:

```text
The current Iso objective, as implemented in the 20-step trainer, is not yet a robust improvement.
The positive Phase 2 signal was a small-slice artifact or at least too fragile to justify a lambda sweep.
```

Decision:

```text
Pause the lambda sweep.
Do not spend more GPU time on the current Iso implementation until the training objective is aligned with GRPO-style group-relative optimization and the measurement target is tightened.
```

Recommended next engineering step:

```text
Implement a true grouped GRPO-style trainer over same-family variants, then re-run a small controlled smoke test before returning to full clean held-out evals.
```

## Decision Rule

Continue to the lambda sweep only if the clean full held-out measurement does not erase the current signal.

Minimum useful continuation:

```text
Iso lambda 0.50 should keep equal or better family accuracy than matched independent reward
without a meaningful accuracy loss.
```

If the full held-out result is flat or negative, pause training and inspect family-type breakdowns before spending more GPU time.

Current status after the seed 13 Iso full-clean eval:

```text
The decision rule failed.
The next phase should be objective correction, not more lambda sweeps on the current trainer.
```
