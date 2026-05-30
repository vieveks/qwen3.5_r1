# Critic Brief: Phase 2/3 Historical Snapshot

Date: 2026-05-25

Status: historical reference only

This file preserves the old critic-brief state from the early `grpo_lite` phase. It is not the current project claim.

## Historical Result

The early project tested whether isomorphic reward shaping could improve reasoning consistency across transformed variants of the same math problem. The control condition rewarded each answer independently. The Iso-RLVR condition added a family-level bonus when sampled variants from the same latent family were correct.

Old evidence summary:

| Comparison | Accuracy | Family accuracy | Conclusion |
| --- | ---: | ---: | --- |
| Base held-out | 0.6000 | 0.2500 | The held-out set was in a useful difficulty range. |
| Independent, seed 13 | 0.6125 | 0.2500 | Slight accuracy gain, no family-consistency gain. |
| Iso `lambda_iso=0.50`, seed 13 | 0.6250 | 0.3000 | Improved family accuracy by `+0.05` over matched independent. |
| Independent, seed 23 | 0.6375 | 0.3000 | Stronger independent run, family accuracy also improved. |
| Iso `lambda_iso=0.50`, seed 23 | 0.6375 | 0.3500 | Matched accuracy and again improved family accuracy by `+0.05`. |

Old conclusion:

```text
Iso-RLVR lambda 0.50 had a small replicated positive signal on held-out family accuracy,
without reducing held-out single-instance accuracy, across two matched 20-step grpo_lite runs.
```

## Why This Is Historical

This result should no longer be used as the main claim because later phases showed the reward surface was not reliable enough:

- The answer interface depended on loose answer extraction.
- The trainer was the local `grpo_lite` scaffold, not proper TRL `GRPOTrainer`.
- The model-output-to-parser interface was noisy.
- Parser false positives and formatting behavior could affect reward.

Phase 5 replaced this with a strict packed XML answer contract, and Phase 6 reran the core Iso-RLVR comparison under proper TRL GRPO. The current claim is documented in `CRITIC_BRIEF.md`.

## Historical Value

This snapshot is still useful because it shows the project evolution:

```text
early Iso signal -> reward-interface diagnosis -> XML stabilization -> proper TRL result
```

The old result motivated the later work. It is not the result to report.
