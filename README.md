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

