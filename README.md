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

## Quick Start

Create a Python environment, then install:

```bash
pip install -e .
```

Generate a small synthetic dataset:

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_small.jsonl --families 200 --variants 4 --seed 7
```

Run baseline evaluation:

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval.yaml
```

Run the minimal local trainer:

```bash
python -m iso_rlvr.train.grpo_lite --config configs/train_iso_grpo.yaml
```

## Next Steps

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

Start with a small dataset so failures are cheap.

```bash
python -m iso_rlvr.data.build_dataset --out data/iso_math_small.jsonl --families 200 --variants 4 --seed 7
```

Check a few rows:

```bash
python -c "import itertools; print(''.join(itertools.islice(open('data/iso_math_small.jsonl'), 3)))"
```

### 3. Run Baseline Evaluation

Run the base model before training anything.

```bash
python -m iso_rlvr.eval.run_eval --config configs/baseline_eval.yaml
```

This writes:

```text
outputs/eval/baseline_qwen25_math_1_5b.jsonl
outputs/eval/baseline_qwen25_math_1_5b.summary.json
```

Read the summary:

```bash
python -c "import json; print(json.dumps(json.load(open('outputs/eval/baseline_qwen25_math_1_5b.summary.json')), indent=2))"
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

Once the smoke test works, run matched small experiments:

```text
Base model evaluation
Independent correctness reward
Iso family reward, lambda_iso = 0.25
Iso family reward, lambda_iso = 0.50
Iso family reward, lambda_iso = 1.00
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
