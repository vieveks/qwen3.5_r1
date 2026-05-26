# Iso-RLVR on Qwen2.5-Math-1.5B

This project tests a sharper RLVR hypothesis:

> Can isomorphic rewards make a small model learn rule-preserving reasoning instead of verifier shortcuts?

The baseline RLVR setup rewards each math problem independently. Iso-RLVR groups equivalent variants of the same underlying problem and rewards the model only when its answers remain correct and transformation-consistent across the family.

## Why This Is Interesting

Outcome-only RLVR can improve final-answer accuracy while still teaching brittle shortcuts. Recent work on verifier gaming and causal reasoning argues that correct answers alone do not guarantee robust reasoning. This project makes that failure testable on a small model.

Model target:

- `Qwen/Qwen2.5-Math-1.5B`

Core comparisons:

- Base model evaluation
- Standard independent reward
- Isomorphic family reward
- Optional Dr. GRPO-style objective later

## Current Status

Phase 1 is complete enough to move into small controlled RL runs.

The first easy dataset was too easy for a meaningful RL comparison. A later challenge profile was too hard and too slow. The current recommended dataset is the calibrated procedural profile:

```text
dataset: data/iso_math_calibrated.jsonl
profile: calibrated
seed: 29
baseline config: configs/baseline_eval_calibrated.yaml
baseline accuracy: 0.5625
baseline family_accuracy: 0.25
format_failure_rate: 0.0
```

That result is useful because single-question accuracy is moderate while whole-family accuracy is much lower. This is the exact gap Iso-RLVR is meant to attack.

Progress log:

- `phase1.md`: full Phase 1 run log, dataset calibration, field scan, and conclusions
- `phase2.md`: first controlled training start, reward-mode configs, and smoke results
- `phase3.md`: measurement-hardening log for overlap audit, family-type breakdowns, and full held-out eval
- `CRITIC_BRIEF.md`: critic-facing summary of the current evidence, code architecture, datasets, limitations, and next experiments
- `PLAN.md`: experiment roadmap

Current Phase 2 controlled configs:

```text
configs/train_independent_calibrated_20step.yaml
configs/train_iso_calibrated_lam_0_25_20step.yaml
configs/train_iso_calibrated_lam_0_50_20step.yaml
configs/train_iso_calibrated_lam_1_00_20step.yaml
```

First held-out controlled pair:

| Run | Accuracy | Family accuracy |
| --- | ---: | ---: |
| Base | 0.6000 | 0.2500 |
| Independent 20-step | 0.6125 | 0.2500 |
| Iso 20-step, lambda 0.50 | 0.6250 | 0.3000 |
| Independent 20-step, seed 23 | 0.6375 | 0.3000 |
| Iso 20-step, lambda 0.50, seed 23 | 0.6375 | 0.3500 |

This is an early replicated positive signal for Iso-RLVR, not a final result. See `phase2.md` for run logs, caveats, and next decisions.

Replication configs with explicit `seed: 23`:

```text
configs/train_independent_calibrated_20step_seed_23.yaml
configs/train_iso_calibrated_lam_0_50_20step_seed_23.yaml
```

## Quick Start

Create a Python environment, then install:

```bash
pip install -e .
```

Generate the currently recommended calibrated dataset:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
```

Run the calibrated baseline evaluation:

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated.yaml
```

Read the summary:

```bash
python -c "import json; print(json.dumps(json.load(open('outputs/eval/baseline_qwen25_math_1_5b_calibrated.summary.json')), indent=2))"
```

Run the minimal local trainer only after the baseline has been inspected:

```bash
python -m iso_rlvr.train.grpo_lite --config configs/train_iso_grpo.yaml
```

## Dataset Profiles

The dataset builder supports multiple profiles for calibration:

| Profile | Purpose | Current Use |
| --- | --- | --- |
| `easy` | Original simple arithmetic/algebra families | Sanity checks only |
| `mixed` | Easy plus harder algebra families | General debugging |
| `harder` | Algebraic hard families only | Too easy in Phase 1 baseline |
| `challenge` | Multi-step stress families | Too hard for first RL, useful later |
| `calibrated` | Weighted mid-difficulty profile | Recommended first RL dataset |

Examples:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_small.jsonl --families 200 --variants 4 --seed 7 --profile easy
python -m iso_rlvr.data.build_dataset --out data/iso_math_harder.jsonl --families 200 --variants 4 --seed 11 --profile harder
python -m iso_rlvr.data.build_dataset --out data/iso_math_challenge.jsonl --families 200 --variants 4 --seed 19 --profile challenge
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
```

## Next Steps

Current progress and conclusions are tracked in `phase1.md`.

### 1. Set Up The Environment

Use a fresh environment with a recent CUDA-compatible PyTorch build.

```bash
cd iso-rlvr-qwen25
pip install -e .
python -m pytest tests
```

Expected result:

```text
4 passed
```

If `torch.cuda.is_available()` is false, fix CUDA/PyTorch before running model evaluation or training.

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 2. Generate The First Dataset

Use the calibrated dataset for the first meaningful experiment:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_calibrated.jsonl --families 200 --variants 4 --seed 29 --profile calibrated
```

Check a few rows:

```bash
python -c "import itertools; print(''.join(itertools.islice(open('data/iso_math_calibrated.jsonl'), 3)))"
```

### 3. Run Baseline Evaluation

Run the base model before training anything.

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval_calibrated.yaml
```

This writes:

```text
outputs/eval/baseline_qwen25_math_1_5b_calibrated.jsonl
outputs/eval/baseline_qwen25_math_1_5b_calibrated.summary.json
```

Read the summary:

```bash
python -c "import json; print(json.dumps(json.load(open('outputs/eval/baseline_qwen25_math_1_5b_calibrated.summary.json')), indent=2))"
```

### 4. Decide Whether The Dataset Is Too Easy

Before RL, inspect baseline metrics.

If accuracy is above `0.85`, the dataset is too easy. Add harder family types or larger numbers before training.

If accuracy is below `0.20`, the dataset is too hard. Reduce max difficulty or improve prompting before training.

The useful range for the first RL run is roughly:

```text
accuracy: 0.30 to 0.75
family_accuracy: meaningfully lower than accuracy
```

That gap is where Iso-RLVR has room to help.

The current calibrated baseline is in range:

```text
accuracy: 0.5625
family_accuracy: 0.25
gap: 0.3125
```

### 5. Run A Tiny Iso-RLVR Training Smoke Test

Only after baseline evaluation works:

```bash
python -m iso_rlvr.train.grpo_lite --config configs/train_iso_grpo.yaml
```

This is a scaffold trainer, not the final production implementation. The first goal is to verify:

- Rollouts generate correctly
- Rewards are non-constant
- Loss is finite
- LoRA adapter saves
- Training log is written

Training outputs go to:

```text
outputs/runs/iso_grpo_lite/
```

### 6. First Real Experiment Matrix

Use the calibrated dataset and run matched small experiments:

```text
Base model evaluation
Independent correctness reward
Iso family reward, lambda_iso = 0.25
Iso family reward, lambda_iso = 0.50
Iso family reward, lambda_iso = 1.00
Held-out calibrated eval with a different seed
Challenge profile eval as stress test only
```

Keep the dataset, model, prompt template, max tokens, and training steps fixed across runs.

### 7. What To Report

The first report should include:

- Baseline accuracy
- Independent reward accuracy
- Iso-RLVR accuracy
- Family accuracy / IsoConsistency
- Wrong-answer token length
- Format failure rate
- Examples where independent reward passes but IsoConsistency fails
- Examples where Iso-RLVR improves held-out variants

The key claim to test is:

> Iso-RLVR improves family-level consistency at similar or better single-instance accuracy.

### 8. Known Caveats

- `grpo_lite.py` is intentionally minimal and should be replaced or validated against TRL/verl for serious runs.
- The first dataset families are simple. If the base model solves them too easily, add harder symbolic and multi-step families.
- Do not judge the idea from reward curves alone. Always evaluate held-out families.
- Do not increase `lambda_iso` until the independent correctness baseline is working.

## Main Metrics

- Accuracy
- Family accuracy
- IsoConsistency: all answers in a family obey the same latent rule
- Average response tokens
- Wrong-answer response tokens
- Format failure rate

## Research Question

Standard question:

> Did the model get the answer right?

Iso-RLVR question:

> Did the model learn a solution rule that survives equivalent transformations?
