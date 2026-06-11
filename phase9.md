# Phase 9: Make RLVR Move At All, Then Re-Test Iso

Status: planned

Date: 2026-06-12

## Purpose

Phase 8 falsified the Phase 6 result. The deeper diagnosis matters more than the null itself:

```text
Not even the independent RLVR arm beat the untouched SFT adapter.
Comparing iso vs independent was premature, because neither arm was learning anything.
```

Phase 9 therefore reorders the project. The iso hypothesis is not tested again until a
prerequisite is established:

```text
Gate question: can any RLVR signal, at local scale, produce a family-accuracy gain
over the starting policy with a 95 percent CI that excludes zero?
```

Only if yes does the iso-vs-independent comparison become meaningful. If no, the honest
writeup is "RLVR at single-GPU scale does not move this substrate," which is also a result.

## Why Phase 6/8 Produced Nothing

Four causes, in order of importance.

### Cause 1: Training scale was a smoke test

```text
30 steps x 4 prompts x 4 generations = 480 completions per run
train set: 144 packed rows
```

Published small-model GRPO results that show real movement use thousands of samples and
hundreds of steps. The reference study on a 1.5B model (arXiv 2503.16219) reached its
gains with ~7000 training samples over many epochs. Our runs were two orders of
magnitude short of that.

### Cause 2: No reward contrast at the difficulty frontier

GRPO learns from within-group reward variance. Our own rollout audits showed the
problem repeatedly: working families were mostly saturated (all generations correct,
zero advantage) and hard families were hopeless (all generations wrong, zero
advantage). Phase 7 measured `chinese_remainder` sampled accuracy at 0.0000 and
correctness contrast at 0 prompts. Training on prompts with zero group variance is
training on nothing, no matter how many steps run.

### Cause 3: The base-model handicap

`Qwen/Qwen2.5-Math-1.5B` is a base model with no instruction following. Phases 4 through 7
were mostly spent buying back format compliance (XML bridge SFT, trace engineering, the
abandoned think-channel). An instruct-tuned model gets the interface contract nearly for
free, and the entire SFT-bridge layer (and its capability side effects) disappears from
the experiment. Separately, the Qwen2.5-Math family has a documented spurious-reward
pathology (arXiv 2506.10947), which any positive result on it must control away.

### Cause 4: Statistical power

Fixed by Phase 8 tooling. Retained here: every comparison is paired by family with
bootstrap CIs, and heldout sets are 1000 families, not 16.

## Stack Decision: The TRL Fix

### What broke

The env drifted to dev builds (`trl 1.6.0.dev0`, `transformers 5.10.0.dev0`,
`peft 0.19.2.dev0`). The Phase 6 trainer config surface no longer exists:

```text
GRPOConfig(max_prompt_length=...)  -> removed in trl 1.x
scale_rewards: bool                -> now a string enum ("group", ...)
beta default                       -> 0.0 (KL regularization off by default)
loss_type default                  -> "dapo"
num_generations default            -> 8
```

Rolling back to `trl 0.17.0` is not viable: it requires `transformers 4.x`, and the env
is on a `transformers 5.x` dev build. The fix is a forward-port.

### The fix

1. Pin the stack to current stable releases, not dev builds:

```text
trl==1.5.1
transformers (stable release compatible with trl 1.5.1)
peft (stable)
```

Record the exact resolved versions in `pyproject.toml` and `CRITIC_BRIEF.md`. No more
floating dev installs; version drift cost us the random-reward arm in Phase 8.

2. Port `packed_grpo_trl.py` to the 1.x API:

```text
drop max_prompt_length (packed prompts are ~300 tokens; no truncation needed)
scale_rewards: "group"
beta: 0.0 (no KL; matches current small-model practice)
loss_type: "dr_grpo" if available in the pinned release, else document the choice
num_generations: 8
mask_truncated_completions: true (overlong filtering)
```

The reward interface ports unchanged: verified in the installed 1.x source that
`GRPOTrainer._calculate_rewards` still forwards all extra dataset columns to reward
functions as kwargs, which is exactly what the stateless packed reward depends on.
`shuffle_dataset=False` also exists in 1.x, which makes family-grouped batches possible
for the separate-context iso variant (Stage 4 option).

Rationale for `dr_grpo`-style settings over the DAPO defaults: the comparative evidence
on small models is that Dr. GRPO loss, no KL, and overlong filtering help, while
clip-higher, token-level loss, and soft overlong punishment hurt at small scale. The one
DAPO idea we keep is dynamic-sampling-style difficulty filtering, implemented offline
(see Dataset Decision).

3. Comparability rule:

```text
No Phase 9 run is compared against any Phase 6 checkpoint.
All Phase 9 arms are trained under the same pinned stack, same seeds policy.
```

### Throughput And Platform

Generation dominates wall time (Phase 8 measured ~55 minutes per 452 greedy 256-token
packed generations on the 5070 Ti). A 300-step run at 4 prompts x 8 generations is
roughly 9600 sampled completions plus optimization: budget one overnight run per arm.

Platform decision: the project moves to native Ubuntu before Stage 0. vLLM has no
native Windows support, and Ubuntu also brings FlashAttention/Triton fast paths and
removes the Windows tooling friction hit in Phase 8. Setup, including the required
NVIDIA driver install (R570+ for the Blackwell 5070 Ti; pip cu128 torch wheels), is
documented in the README under "Ubuntu Migration". With the platform native, enable
`use_vllm` (colocate mode) from Stage 2 onward rather than deferring it to Stage 3;
expect a 3-10x rollout speedup, with `vllm_gpu_memory_utilization` tuned down to fit
training plus the colocated engine in 16 GB.

## Model Decision

```text
Primary:  Qwen3-1.7B (instruct, thinking disabled)
Control:  Llama-3.2-3B-Instruct (non-Qwen replication and spurious-reward control)
Fallback: Qwen3-4B (only if the 1.7B cannot reach the Stage 1 accuracy band)
```

Reasons:

- Instruct models follow the packed XML contract without an SFT bridge. This deletes the
  Phase 5 bridge, its capability side effects, and the Phase 7 think-channel blocker in
  one move. Stage 1 verifies this assumption before anything trains.
- Qwen3-1.7B has published GRPO gains on math benchmarks at exactly this scale, so the
  substrate is known to be trainable with RLVR.
- Llama-3.2-3B-Instruct is the control family where spurious rewards are documented to
  not help. Any iso effect that appears on Qwen3 and survives on Llama is real; an
  effect that appears only on Qwen stays under suspicion.
- Both fit 16 GB with LoRA in bf16.
- Qwen2.5-Math-1.5B and all its adapters are parked. Nothing more is trained on it.

Thinking mode stays disabled for the primary arms: the verifier scores final answers
only, long CoT multiplies generation cost several-fold on local hardware, and Phase 7
showed reasoning-channel work is its own project. Re-enabling it is a future phase, not
a Phase 9 variable.

## Dataset Decision

### Difficulty targeting is the main training fix

New tool, built on the existing rollout audit:

```text
src/iso_rlvr/data/filter_by_pass_rate.py (new)
input: rollout audit records (8 samples per prompt at training temperature)
keep: prompts with 0 < pass_rate < 1
report: pass-rate histogram by family type
```

Training only on prompts with nonzero reward variance is the offline version of DAPO
dynamic sampling, and it directly attacks Cause 2. The filter is refreshed once at the
midpoint of each long run (the frontier moves as the policy improves).

### Scale and composition

```text
Train: 2000+ families, full generator mix including the harder types
       (rational_linear_equation, missing_average, two_variable_system,
        quadratic_root, nested_linear_equation, chinese_remainder,
        rational_system_target), packed two-variant XML
Heldout: 1000 fresh-seed families, family-level overlap filter, audit gate
         overlap_count 0 (Phase 8 tooling, unchanged)
```

The packed two-variant XML interface is retained as the primary contract: it is proven
(parse_complete 1.0000 across ~5900 Phase 8 generations) and keeps Phase 9 comparable in
kind to Phase 8. With 1000 paired families, the bootstrap CI half-width at Phase 8
discordance rates is roughly plus-minus 0.01, so effects of 3 points or more are
detectable. Anything smaller than that is below this project's resolution by design.

## Experiment Ladder

Each stage has a gate. A failed gate stops the phase and gets written up; it does not
get patched around.

### Stage 0: Infrastructure port

- Ubuntu environment bring-up: NVIDIA driver, cu128 torch, fresh env (README
  "Ubuntu Migration"), `torch.cuda.is_available()` true.
- Pin the stack, port the trainer, keep the random `reward_mode` working.
- Regenerate datasets from documented seeds (nothing in `data/` or `outputs/` ships
  with the clone).
- Full pytest green, 5-step smoke run completes, reward records logged.

Gate:

```text
Smoke run trains, saves an adapter, and logs nonzero reward variance.
```

### Stage 1: Model bring-up without SFT

Zero-shot packed XML eval of Qwen3-1.7B (and Llama-3.2-3B) via chat template, sampled
rollout audit at training temperature.

Gate, per model:

```text
sampled parse_complete_rate >= 0.95 with no SFT
baseline family accuracy in [0.20, 0.60] on the chosen difficulty profile
```

The difficulty profile (mix of family types) is tuned to land in that band; that is what
the band is for. If parse fails on an instruct model, a one-epoch answer-only SFT is
permitted as fallback, but that is a finding worth recording in itself.

### Stage 2: Independent RLVR at scale (the prerequisite gate)

One arm, no iso reward:

```text
model: Qwen3-1.7B
steps: 300 (vs Phase 6's 30)
num_generations: 8
learning_rate: 1e-6, beta: 0.0
train: difficulty-filtered packed set, refreshed at step 150
eval: 1000-family heldout, paired bootstrap vs the untrained starting policy
```

Gate:

```text
independent minus base family-accuracy delta, 95 percent CI excludes zero
```

If this fails after one honest retune (learning rate and filter band only), Phase 9
stops and the writeup is the null: local-scale RLVR does not move this substrate. No iso
arm is run, because Phase 8 already demonstrated where that road goes.

### Stage 3: The iso re-test

Only entered if Stage 2 passes. Matched-compute matrix under identical settings:

```text
independent      seeds 23, 37
iso lambda 0.50  seeds 23, 37
iso lambda 1.00  seeds 23, 37
random reward    seed 23   (now version-matched to every other arm)
```

Pre-registered reading rules, same structure as Phase 8:

```text
iso > independent with CI excluding zero on both seeds, random arm flat: supported.
iso ~= independent, both > base: RLVR works here but family reward adds nothing.
random ~= independent gains: the gain is policy-drift, not signal; nothing is claimed.
```

### Stage 4: External validity (pick per Stage 3 outcome)

- Replicate the winning comparison on Llama-3.2-3B-Instruct.
- Unpacked cross-context transfer eval (Phase 8 C tooling, unchanged).
- Optional: separate-context iso training arm using `shuffle_dataset=False` and
  family-grouped batches, which tests the original hypothesis in its strongest form.

## Budget

```text
Stage 0: half a day of coding, no GPU
Stage 1: ~2 hours GPU (evals and audits only)
Stage 2: 1-2 overnight runs
Stage 3: 7 arms, ~1 overnight each without vLLM; move to WSL2 + vLLM first
Stage 4: 1-2 overnight runs
```

The gates exist because the full ladder is roughly two weeks of GPU nights, and Stages
3-4 are only worth their nights if Stage 2 earns them.

## What Phase 9 Does Not Do

- No think-channel work. Thinking stays disabled.
- No GSM-Symbolic or external benchmarks yet. Controlled generator first; external
  validity datasets are the Phase 10 candidate once any effect exists.
- No comparisons against Phase 6 checkpoints or the Qwen2.5-Math adapter line.
- No reward-shape variations beyond lambda in {0.50, 1.00}.

## Success Criteria

Minimum useful outcome:

```text
A version-pinned, difficulty-filtered RLVR pipeline on an instruct model, where the
independent arm's effect is measured with CIs, even if that measurement is a null.
```

Strong outcome:

```text
Stage 2 gate passes; at least one iso arm beats independent with a CI excluding zero
on both seeds; the random arm is flat; the effect survives on Llama-3.2-3B.
```

## References

- Small-model RL study (1.5B, GRPO, what works/what doesn't): arXiv 2503.16219
- Spurious rewards on Qwen2.5-Math, non-Qwen controls: arXiv 2506.10947
- Memorization/contamination critique of Qwen RLVR results: arXiv 2507.10532
- Mechanistic spurious-reward analysis (Qwen vs Llama/OLMo): arXiv 2601.11061
- Qwen3-1.7B-Base GRPO math gains: arXiv 2602.08499
- Dr. GRPO (length/std bias in GRPO): arXiv 2503.20783
- DAPO (dynamic sampling, applied here as offline pass-rate filtering): arXiv 2503.14476
- vLLM on Windows status (WSL2 is the supported path): no native support as of 2026-05
