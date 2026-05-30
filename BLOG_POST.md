# RLVR Does Not Just Fail at Optimization. It Fails at the Interface.

I started this project with a simple question: can reinforcement learning with verifiable rewards teach a model to solve the same underlying math rule across multiple surface variants, instead of just getting isolated answers right?

That question became a lot more interesting than I expected. The first thing I learned was not about GRPO, reward weights, or clever math datasets. It was that the reward interface itself was the main bottleneck. If the model's answer cannot be parsed reliably, the verifier is not measuring reasoning. It is measuring accidents of formatting, truncation, and parser behavior.

The final result is narrow but clean: on a stabilized packed XML interface, an isomorphic family reward improved family accuracy over independent RLVR, with no parse regression. The best run improved family accuracy by `+0.1250` over the independent baseline.

But the real story is how much work it took to make that number mean anything.

## The Hypothesis

Standard RLVR asks a simple question:

```text
Did the model get this answer right?
```

That is useful, but it is incomplete. A model can get one instance right while still relying on brittle local patterns. For example, it might solve one linear equation correctly but fail when the same latent structure appears with different coefficients or wording.

Iso-RLVR asks a slightly different question:

```text
Did the model get the related variants right together?
```

Instead of rewarding only independent correctness, the reward includes a family-level component. A family contains multiple isomorphic variants: different surface forms that share an underlying rule or answer relationship.

The intuition is simple. If the model learns the underlying rule, it should be more consistent across variants. If it learns shallow answer hacks, family accuracy should stay low even when single-instance accuracy looks decent.

This project was an attempt to test that idea locally on a small model:

```text
Qwen/Qwen2.5-Math-1.5B
LoRA adapters
TRL GRPOTrainer
procedural packed math families
strict XML answer verification
```

## The Reward Interface Problem

The first serious blocker was not optimization. It was the path from model output to reward.

The pipeline looked like this:

```text
packed prompt -> model completion -> parser -> verifier -> reward
```

The weak link was:

```text
model completion -> parser
```

At one point, the best packed result looked like this:

```text
accuracy: 0.2500
family_accuracy: 0.1250
parse_complete_rate: 0.3750
answer_count_mismatch_rate: 0.6250
suspicious_rate: 0.7500
```

That is not a training signal. That is a warning light.

If `parse_complete_rate` is `0.3750`, most completions are not making it cleanly through the verifier interface. If `answer_count_mismatch_rate` is `0.6250`, the model is often producing the wrong number of answers for a packed prompt. Running GRPO on that would mostly teach the model to interact with a brittle parser, not to reason better.

This is easy to miss if you only track accuracy. The verifier can produce a number even when the interface is unstable, but that number may not mean what you think it means.

The fix was to make the answer surface narrow and machine-verifiable.

The output contract changed from loose answer lines:

```text
Answer 1: 33
Answer 2: 16/5
```

to strict XML:

```xml
<answers>
<answer_1>33</answer_1>
<answer_2>16/5</answer_2>
</answers>
```

Then I built a small SFT bridge. The goal of this supervised step was not to improve math ability. It was to teach the model the interaction contract:

```text
packed prompt -> verifier-compatible XML answers
```

That bridge changed the project. The final Phase 5 adapter reached:

```text
greedy parse_complete_rate: 1.0000
sampled parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
malformed samples: 0
reward_std: non-degenerate
```

Only after that did GRPO become a meaningful experiment.

The lesson is blunt: before tuning reward weights, stabilize the reward interface. In RLVR systems, parser failures are not cosmetic. They change the optimization target.

## The Clean Result

After the XML interface was stable, I replaced the local `grpo_lite` scaffold with proper TRL `GRPOTrainer`.

The main Phase 6 experiment used two working family types:

```text
missing_average
rational_linear_equation
```

Both arms started from the same Phase 5 all-traces adapter. The only variable in the sweep was the isomorphic family reward scale.

| Run | Accuracy | Family accuracy | Parse complete | Sampled malformed |
| --- | ---: | ---: | ---: | ---: |
| Base all-traces adapter | 0.7500 | 0.6250 | 1.0000 | 0 |
| Independent 30-step, seed 23 | 0.7188 | 0.5625 | 1.0000 | 0 |
| Iso `lambda_iso=0.25`, seed 23 | 0.7500 | 0.6250 | 1.0000 | 0 |
| Iso `lambda_iso=0.50`, seed 23 | 0.7500 | 0.6250 | 1.0000 | 0 |
| Iso `lambda_iso=1.00`, seed 23 | 0.7813 | 0.6875 | 1.0000 | 0 |
| Independent 30-step, seed 37 | 0.7188 | 0.5625 | 1.0000 | 0 |
| Iso `lambda_iso=0.50`, seed 37 | 0.7500 | 0.6250 | 1.0000 | 0 |

Family accuracy means that all included variants from a family were correct. It is stricter than ordinary accuracy and closer to the thing I wanted to measure: consistency across related variants.

The headline result is the monotonic family-accuracy trend:

```text
Independent: 0.5625
lambda_iso=0.25: 0.6250
lambda_iso=0.50: 0.6250
lambda_iso=1.00: 0.6875
```

The best run was:

```text
lambda_iso=1.00
accuracy delta over independent: +0.0625
family_accuracy delta over independent: +0.1250
parse_complete_rate delta: 0.0000
```

The `lambda_iso=0.50` result also replicated across two seeds:

| Seed | Independent accuracy | Iso accuracy | Independent family accuracy | Iso family accuracy | Family delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 |
| 37 | 0.7188 | 0.7500 | 0.5625 | 0.6250 | +0.0625 |

In the seed-37 confirmation, the independent and iso runs had identical contrast-driver prompt groups. That matters because the improvement was not explained by the iso run happening to receive a luckier set of contrast prompts.

Across the narrow runs, the interface stayed stable:

```text
parse_complete_rate: 1.0000
answer_count_mismatch_rate: 0.0000
suspicious_rate: 0.0000
sampled malformed completions: 0
```

That last part is what makes the result trustworthy. The reward surface was not drifting under the model.

## What the Result Means

The result supports a narrow claim:

```text
On a stable two-family packed XML verifier interface, Iso-RLVR improves family accuracy over independent RLVR under proper TRL GRPO, with no parse regression.
```

It does not prove that Iso-RLVR generally improves mathematical reasoning. It does not prove that the model learned human-like rules. It does not show broad-family generalization.

But it does show that once the verifier interface is stable, a family-level reward can move the metric it is supposed to move.

The by-family breakdown is also useful:

| Arm | `missing_average` acc. | `missing_average` family acc. | `rational_linear_equation` acc. | `rational_linear_equation` family acc. |
| --- | ---: | ---: | ---: | ---: |
| Independent | 0.9167 | 0.8333 | 0.6000 | 0.4000 |
| Iso `lambda_iso=0.25` | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso `lambda_iso=0.50` | 1.0000 | 1.0000 | 0.6000 | 0.4000 |
| Iso `lambda_iso=1.00` | 1.0000 | 1.0000 | 0.6500 | 0.5000 |

Most of the improvement comes from `missing_average`. `rational_linear_equation` only improves at the highest lambda. That is a useful boundary: the family reward helps, but it does not magically erase capability limits.

## What Did Not Work

Several things failed before this result became clean.

The first failure mode was loose formatting. The base model often emitted reasoning prose, copied problem text, or produced answers in the wrong shape. Packed generation made this worse because the model had to solve multiple related tasks and remember answer indices.

The second failure mode was trace engineering. Deterministic traces helped stabilize the interface, but they could also become the task. For harder families like Chinese remainder problems, one trace format taught the model to continue enumerating candidates until the token cap instead of terminating cleanly.

The third failure mode came from Phase 7. I tried adding a free reasoning channel:

```xml
<think>
free reasoning
</think>
<answers>
<answer_1>...</answer_1>
<answer_2>...</answer_2>
</answers>
```

The parser isolation worked: the verifier only scored `<answers>` and ignored `<think>`. Hybrid-think SFT preserved more capability than a minimal empty-think bridge. But sampled rollouts still failed the interface gate.

The clearest blocker was:

```text
The model starts <think>, reasons, reaches </think>, and then often stops before <answers>.
```

Response prefixing fixed the opening `<think>` tag, but it did not fix the transition from `</think>` to `<answers>`.

That is a real future-work direction, but I stopped there because the Phase 6 result already stands on its own.

## Limitations

The limitations are important:

1. The main heldout set is small.
2. The claim covers only two procedural family types.
3. The best `lambda_iso=1.00` result has not yet been replicated across multiple seeds.
4. The two-seed replication is for `lambda_iso=0.50`, not for every lambda.
5. Training runs are short: 30 GRPO steps.
6. The result is local to `Qwen/Qwen2.5-Math-1.5B` plus LoRA.
7. The reward checks final answers only, not reasoning validity.
8. The XML interface is engineered and may not transfer unchanged to other answer contracts.
9. Broader calibrated families are not RL-ready under the current bridge.
10. No bootstrap confidence interval over families has been reported.
11. No final blind benchmark untouched by iteration has been run.

These limitations are not footnotes. They define the claim.

## What I Would Do Next

The immediate next step is not more Phase 7 tinkering. It is writing up the Phase 6 result cleanly.

After that, the next experiments are straightforward:

- replicate `lambda_iso=1.00` across more seeds
- add bootstrap confidence intervals over families
- build a larger heldout set for the two working families
- run a final blind procedural split
- revisit `<think> + <answers>` only after enforcing the `</think>\n<answers>` transition
- test harder family types on larger models once the two-tag interface is stable

The broader lesson is the one I wish I had started with:

```text
Reliable verifier interface first.
GRPO second.
Reward shaping third.
```

If the parser is unreliable, the reward is not what you think it is. And if the reward is not what you think it is, RL will optimize the wrong thing very efficiently.

That was the most useful thing this project taught me.
