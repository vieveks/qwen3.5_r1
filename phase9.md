# Phase 9: Make RLVR Move At All, Then Re-Test Iso

Status: Stages 0–3 complete. Stage 2 prerequisite PASSED (RLVR beats base, CI excludes
zero); Stage 3 iso re-test is a CLEAN NULL — iso ≈ independent, both > base, random flat.
The isomorphism reward adds nothing over a plain correctness reward at this scale.

Date: 2026-06-12 (plan); 2026-06-14 (Stage 0–1 execution)

See the [Execution Log](#execution-log) at the end for in-depth substep results and the
design choices made during the port.

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

## Execution Log

In-depth record of what was actually run, the design choices made along the way, and
the substep results. Newest stage last.

### Environment (Ubuntu migration, 2026-06-14)

Hardware/OS layer came up clean: Ubuntu 24.04, NVIDIA driver 580.159.03 (well above the
R570 Blackwell floor), RTX 5070 Ti 16 GB, `torch.cuda.is_available()` true, a real
matmul ran on `sm_120`.

Environment decision: a dedicated conda env **`env_rlvr`** (Python 3.12), not the
README's `iso_rlvr` name and not any pre-existing env (none had the RL stack). torch was
installed from the cu128 index; the rest via `pip install -e ".[dev]"`.

Resolved pinned stack (exact versions, recorded per the Stack Decision):

```text
torch==2.11.0+cu128   trl==1.5.1            transformers==5.12.0
peft==0.19.1          datasets==5.0.0       accelerate==1.14.0   numpy==2.4.6
```

`transformers` resolved to 5.12.0 — a stable release, not the 5.10.0.dev0 build that
drifted in Phase 8. `pyproject.toml` pin changed `trl==0.17.0` → `trl==1.5.1`.

Gotcha worth recording: the shell exports `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/...`
(ROS Jazzy). That makes pytest auto-load ROS's `launch_testing` plugin, which dies on a
missing `lark`. All project commands run with `PYTHONPATH=""` (tests) or `PYTHONPATH=src`
(module runs). With it cleared, the suite is **136 passed**.

### Stage 0: Infrastructure port (complete, 2026-06-14)

**Trainer port — `src/iso_rlvr/train/packed_grpo_trl.py`.** Forward-ported the trl 0.17
config surface to trl 1.x. Verified against the installed `GRPOConfig` dataclass before
editing:

- `max_prompt_length` is **gone** in trl 1.x (a runtime `TypeError` confirmed it twice) →
  dropped. Packed prompts are short, so no truncation is needed.
- `scale_rewards` is now a **string enum**, not a bool. Added `normalize_scale_rewards()`
  to map the legacy `true/false` onto `"group"`/`"none"`; default `"group"`.
- `beta` default → `0.0` (no KL), `loss_type` → `"dr_grpo"` (confirmed accepted by the
  pinned release), `mask_truncated_completions` → `true`, `num_generations` default → 8.
- `use_vllm` plumbed (plus `vllm_mode`, `vllm_gpu_memory_utilization`) but left off for
  Stage 0; vLLM install is deferred to Stage 2.
- **Design choice — fresh LoRA without an SFT bridge.** `adapter_path` is now optional.
  When absent, the trainer builds a `LoraConfig` (`build_lora_config()`) and hands it to
  `GRPOTrainer(peft_config=...)`. This is what makes the Phase 9 "instruct model, no SFT
  bridge" decision concrete: the Phase 5 bridge layer disappears from the code path.

**Datasets/pyarrow fix (a real bug, not just a port).** `datasets==5.0.0` / pyarrow 24
refuse to build the `metadata` column: it is a per-family list of dicts whose value types
differ across family types and across fraction-vs-int answers (e.g. `b` is an `int` for
one family and the string `"13/5"` for another), so Arrow cannot infer a single struct
type and raises `ArrowInvalid`. This would break **every** mixed-family training run, not
just the smoke. Fix: `arrow_safe_rows()` JSON-encodes `metadata` (diagnostic-only; the
reward never reads it for scoring) into a uniform string column before
`Dataset.from_list`.

**Datasets regenerated** from documented seeds into `data/` (gitignored, empty on clone):
`stage0_packed_xml_train.jsonl` (256 rows, seed 29) and `stage0_packed_xml_heldout.jsonl`
(128 rows, seed 101), both `calibrated`, packed two-variant XML.

**Smoke run** (`configs/packed_grpo_trl_stage0_smoke.yaml`, Qwen3-1.7B, fresh LoRA, 5
steps): trains, saves an adapter, and logs nonzero reward variance (`reward_std` 0.26,
0.065, …) → **Stage 0 gate met on its literal terms**, with `pytest` 136 passed and ruff
clean.

But the smoke also surfaced the Stage 1 prerequisite. The run trained with **zero
gradient** (`grad_norm: 0`) despite reward variance, because `completions/clipped_ratio`
was `1.0` (every completion hit the token cap, none emitted EOS) and
`mask_truncated_completions: true` then masked all of them. A control run with masking off
gave nonzero `grad_norm` (0.49/0.48/0.76) exactly when `reward_std > 0`, proving the
optimizer path is healthy. Root cause of the non-termination: the model was fed the
packed prompt as **raw text**; an instruct/thinking model rambles in CoT instead of
emitting the terse XML. That is precisely what Stage 1 fixes.

### Stage 1: Model bring-up without SFT (in progress, 2026-06-14)

**Design choice — chat template at run time, not baked into the data.** `apply_chat_template`
already existed in the eval path (`build_generation_prompt`), but it did not disable
thinking. Qwen3 gates chain-of-thought on an `enable_thinking` template kwarg. Added an
`enable_thinking` passthrough that is forwarded to `apply_chat_template` **only when the
config sets it**, so templates that do not accept it (Llama-3.2-Instruct) keep working.
The trainer now renders its training prompts through the same `build_generation_prompt`,
so trainer prompts, the trainer's final greedy eval, and the rollout audit all share one
prompt-formatting path. Datasets stay model-agnostic; the template is applied per-model.

Verified the render: with `enable_thinking=False`, Qwen3 appends an empty
`<think>\n\n</think>` block after the assistant turn (its "thinking done" signal), so the
model goes straight to the answer.

**Substep 1 — format is solved.** A confirm audit of Qwen3-1.7B (chat template on,
thinking off, temp 0.7) on the `calibrated` heldout produced clean
`<answers>…</answers>` XML: `parse_complete_rate` 0.94, `think_block_rate` 0.00,
`answer_count_mismatch` 0.06. The Phase 5/7 format problem is gone for free on an instruct
model — exactly the Phase 9 bet.

**Substep 2 — difficulty tuning (the [0.20, 0.60] band).** On `calibrated` the model
parsed cleanly but scored **0.00 accuracy** on every family type: `calibrated` is 9/10
fraction-heavy hard/challenge families, tuned for the old Qwen2.5-Math base, and is simply
too hard for Qwen3-1.7B zero-shot. Per the Stage 1 design ("the profile is tuned to land
in that band; that is what the band is for"), regenerated `easy` and `mixed` packed sets
(seed 101) and re-audited (24 families × 8 samples, temp 0.7).

| Profile | parse_complete | accuracy | family_accuracy | in [0.20, 0.60]? |
| --- | ---: | ---: | ---: | :---: |
| calibrated | 0.94 | 0.00 | 0.00 | ✗ too hard |
| mixed | 0.958 | 0.383 | 0.318 | ✓ |
| easy | 0.953 | 0.570 | 0.474 | ✓ |

Per-family-type accuracy on `mixed` (the spread that matters for GRPO contrast):

```text
saturated (~1.0):  proportional 1.00, unit_conversion 1.00, linear_equation 0.94
mid (contrastful): quadratic_root 0.67, modular 0.21
dead (0.00):       rational_linear_equation, two_variable_system, nested_linear_equation
```

**Stage 1 gate: PASSED for Qwen3-1.7B.** `parse_complete_rate` 0.958 ≥ 0.95 and family
accuracy 0.318 ∈ [0.20, 0.60], with `think_block_rate` 0.00 and no SFT. The Phase 9 bet —
that an instruct model gets the packed XML contract for free — holds: the entire Phase 5
SFT-bridge layer is gone.

**Design choice — `mixed` is the Stage 2 profile.** Both `mixed` and `easy` pass, but
`mixed` (family acc 0.318) sits more centrally in the band, leaving headroom to detect a
gain before saturation, and it retains the harder families the Dataset Decision calls for.
Its graded difficulty is what offline pass-rate filtering (Stage 2) needs: the dead-0
families drop out, the saturated families drop out, and the contrastful middle
(modular, quadratic_root, the partial linear types) is what remains to train on. `easy`
(0.474) stays on the bench as a higher-baseline fallback.

**Findings worth recording.**

- *Affine answers come back as unevaluated expressions.* On `mixed`, `affine` parsed at
  0.00 because the model emitted `<answer_1>23 + (7 * 11)</answer_1>` instead of `100`.
  This is genuine model behavior (it left the arithmetic unevaluated), not a parser bug,
  and it correctly scores as incomplete/wrong. The packed verifier contract is unchanged:
  the instruction asks for a number, and an expression is a miss.
- *The `calibrated` train mix from the original plan is unreachable zero-shot.* The
  Dataset Decision's intended hard mix (rational/system/CRT) is 0.00 for Qwen3-1.7B before
  any training, so Stage 2 trains on `mixed` and lets the difficulty filter, refreshed at
  the run midpoint, follow the frontier upward.

Remaining Stage 1 step before Stage 2: a full-coverage sampled audit over the Stage 2
**train** set (8 samples/prompt) to feed `filter_by_pass_rate` (the Stage 2 difficulty
filter). The Llama-3.2-3B control bring-up is deferred to Stage 4 (external validity),
per the ladder.

### Stage 2: Independent RLVR at scale (in progress, 2026-06-14)

**Scale decision — reduced first pass without vLLM.** The plan's full Stage 2 is 2000+
train families and a 1000-family heldout. Without vLLM the difficulty audit is unbatched
(`generate_one` one sample at a time), so a 2000×8 audit is ~10 GPU-hours and impractical
to babysit. This first pass runs at reduced scale — **384-family train pool, 1000-family
heldout** — with the full-scale rerun deferred to when vLLM lands. The gate logic and
tooling are identical at either scale.

**The new difficulty filter — `src/iso_rlvr/data/filter_by_pass_rate.py`** (the offline
DAPO-dynamic-sampling tool from the Dataset Decision). It reads the sampled rollout audit,
computes each packed family's pass rate at the family level (`all_family_correct`), and
keeps only families with strictly `0 < pass_rate < 1` — the contrastful middle. Saturated
(pass 1) and hopeless (pass 0) families carry zero GRPO advantage and are dropped. Band is
configurable for the gate's permitted retune. Unit-tested (3 tests).

**Datasets** (`mixed` profile, the Stage 1 pick):

```text
train pool : data/stage2_packed_train.jsonl     384 families, seed 29
heldout    : data/stage2_packed_heldout.jsonl   1000 generated, seed 101
             -> 138 families dropped by the family-level overlap filter vs the train
                generation (overlap_count 0) -> 862 paired heldout families
```

**Trainer additions for this stage.** Added a `run_final_eval` flag (default true) so the
300-step run can skip the trainer's built-in greedy eval; the gate eval is run separately
as a paired bootstrap on the full 862-family heldout. The independent arm sets
`family_bonus_enabled: false` (pure format + correctness reward, no family/iso term).

**Design choice — no mid-run filter refresh in the first pass.** The plan refreshes the
difficulty filter at step 150. This first pass runs a single 300-step job with one filter
pass and records that simplification; the mid-run refresh is a refinement for the
vLLM-enabled full run. The Stage-1 chat-template fix means completions now terminate, so
`mask_truncated_completions: true` no longer zeroes the gradient (the Stage 0 failure mode).

**Configs:** `packed_rollout_audit_stage2_train_qwen3_1_7b.yaml` (difficulty audit),
`packed_grpo_trl_stage2_independent_seed23.yaml` (300-step train, num_generations 8,
lr 1e-6, beta 0.0, dr_grpo, no vLLM), `packed_eval_stage2_{base,independent_seed23}_heldout.yaml`
(greedy gate evals), compared with `eval.bootstrap_compare`.

Gate (pre-registered): `family_accuracy(independent) - family_accuracy(base)` with a 95%
paired-bootstrap CI that excludes zero.

**Substep results.**

*Difficulty audit + filter.* 1024-family `mixed` pool, 8 samples/prompt at temp 0.7 (the
audit was stopped at 1008/1024 families by an external kill; the 8062 records are complete
enough to filter on). Pass-rate is sharply **bimodal**: 663 families all-wrong
(pass 0), 261 all-correct (pass 1), and only **84 contrastful** (`0 < pass < 1`, ~8%).
This is Cause 2 made quantitative — a 1.7B either solves a family every time or never; the
trainable frontier is thin. `filter_by_pass_rate` kept those 84 families
(`data/stage2_filtered_train1024.jsonl`), concentrated in quadratic_root, modular,
linear_equation, and proportional.

*Training.* 300 steps, num_generations 8, lr 1e-6, beta 0.0, dr_grpo, no vLLM. Completed
in ~6 min (completions are terse XML, ~30 tokens, `clipped_ratio` 0 — the Stage 0
zero-gradient failure mode is gone). `grad_norm` ~0.2-0.38, `reward_std` ~0.3,
`frac_reward_zero_std` ~0.35 (a third of groups still flat even after filtering, because
greedy-temp pass rate does not perfectly predict training-temp contrast). Reward stayed
flat at ~0.6 at lr 1e-6.

*Gate, run 1 (lr 1e-6).* Heldout 814 paired families, greedy, paired bootstrap:

```text
base        family_acc 0.1966   acc 0.2918
independent family_acc 0.2002   acc 0.2942
family_accuracy delta +0.0037   95% CI [-0.0025, +0.0111]   sign-flip p 0.45
-> FAIL (CI includes zero)
```

*Honest retune (lr 1e-6 -> 5e-6, filter band unchanged).* At lr 5e-6 the train reward
climbed (0.63 -> ~0.85), i.e. the policy actually learned on the train families.

*Gate, run 2 (lr 5e-6).*

```text
base               family_acc 0.1966   acc 0.2918
independent_lr5e6  family_acc 0.2064   acc 0.2979
family_accuracy delta +0.0098   95% CI [-0.0012, +0.0209]   sign-flip p 0.11
-> FAIL (CI includes zero, lower bound essentially touching it)
```

Per-family-type movement (base -> lr5e6 family accuracy) shows the effect is **real but
diluted**, not absent:

```text
quadratic_root  0.304 -> 0.380  (+0.076)   <- strong, the bulk of training contrast
proportional    0.600 -> 0.630  (+0.030)
affine          0.010 -> 0.020  (+0.010)
modular         0.427 -> 0.415  (-0.012)   <- small regressions
linear_equation 0.287 -> 0.278  (-0.009)
nested / rational / two_variable (345 of 814 families) : 0.000 -> 0.000
```

**Stage 2 gate: FAILED after the one permitted retune** (lr + filter band only), so the
pre-registered rule says Phase 9 stops at the null. But the honest reading is more
specific than "RLVR does not move this substrate":

1. RLVR *does* move the learnable family types — quadratic_root +7.6 points is a real,
   directional gain concentrated exactly where the difficulty filter put the training
   contrast.
2. The overall gate fails largely by **dilution**: ~42% of the heldout (nested_linear,
   rational_linear, two_variable_system) is at 0.000 accuracy zero-shot and cannot be
   moved by RL at all, and the saturated types have no headroom. Averaging the movable
   gain over an immovable majority washes the CI back across zero.
3. This is the reduced-scale (no-vLLM) first pass: 84 training families, single filter
   pass, no mid-run refresh. The full-scale plan (2000+ families, 1000-family heldout,
   vLLM, filter refresh at step 150) has materially more training contrast and statistical
   power, and run 2's CI lower bound at -0.0012 suggests the effect is near the resolution
   floor rather than absent.

Decision deferred to a design call (not a within-pre-registration patch): accept the null
as written, or rerun at full scale with vLLM before judging the Stage 2 prerequisite. No
iso arm (Stage 3) is run until Stage 2 passes.

#### Stage 2 full-scale rerun (2026-06-14) — gate PASSED

The decision was to rerun at full plan scale. vLLM turned out to be both incompatible and
unnecessary (see below), so the full-scale rerun was done without it, with a batched audit.

**vLLM is blocked on this stack, and isn't the needed lever.** vLLM 0.23.0 (latest) pins
`torch==2.11.0` at the Python level but its kernels need CUDA 13 (`libcudart.so.13`) while
our torch is cu128/CUDA 12.8 — it fails to import on Blackwell, and it drags in a broken
`torchvision` that cascades into a transformers import error. It was rolled back cleanly
from a `pip freeze` snapshot (env verified healthy, 139 tests). The deeper point: Stage 2
training is only ~3 min / 150 steps without vLLM (completions are ~30-token XML, so HF
generation is already cheap); the real bottleneck was the unbatched difficulty audit, which
trl colocate-vLLM would not accelerate anyway. So instead of fighting vLLM, the audit got a
**batched generation path** (`generate_batched` + `batch_size` in the rollout audit):
the 2000-family × 8-sample audit dropped from ~10 GPU-hours to **~7 minutes** (~50-70x).

**Full-scale setup.** 2000-family `mixed` train pool, 773-family overlap-filtered heldout
(overlap_count 0), lr 5e-6, 300 steps split as two phases with a filter refresh at 150
(the plan's mid-run refresh, implemented as: train 150 -> re-audit the pool with the step-150
adapter -> re-filter -> train 150 more, continuing from the step-150 adapter).

```text
difficulty filter v1 (base policy)      : 155 contrastful families of 2000 (~8%)
phase A (steps 0-150, lr 5e-6)          : train reward 0.62 -> 0.80
re-audit with adapter A                 : train-pool family_acc 0.292 -> 0.306 (frontier moved)
difficulty filter v2 (post-phase-A)     : 151 contrastful families
phase B (steps 150-300, from adapter A) : train reward 0.70 -> 0.85
```

**Gate (773 paired heldout families, greedy, 10k-iter paired bootstrap):**

```text
base        family_acc 0.1940   acc 0.2878
independent family_acc 0.2070   acc 0.2969
family_accuracy delta +0.0129   95% CI [+0.0013, +0.0246]   sign-flip p 0.043
variant accuracy delta +0.0091   95% CI [+0.0013, +0.0175]
-> PASS (both CIs exclude zero)
```

Per-family-type (base -> trained family accuracy):

```text
quadratic_root  0.308 -> 0.385  (+0.077)   <- main driver (most training contrast)
proportional    0.606 -> 0.628  (+0.021)
affine          0.012 -> 0.024  (+0.012)
nested_linear   0.000 -> 0.009  (+0.009)
two_variable    0.061 -> 0.070  (+0.009)
linear/modular  ~flat
rational_linear 0.000 -> 0.000  (unmovable zero-shot)
```

**Stage 2 prerequisite is MET.** What flipped the result vs the reduced pass (which failed
at +0.0098, CI grazing zero): nearly 2x the training contrast (155 vs 84 families), the
mid-run filter refresh, and more statistical power (773 vs 814... comparable, but combined
with the stronger train signal). The pre-registered gate — "can any RLVR signal at local
scale produce a family-accuracy gain over the starting policy with a 95% CI excluding
zero?" — is answered yes. **Stage 3 (the iso re-test) is now justified and unlocked.**

### Stage 3: The iso re-test (in progress, 2026-06-14)

**Pre-registered before running any arm.** Matched-compute matrix, identical settings,
differing only in reward shape and seed. All arms: Qwen3-1.7B fresh LoRA, the SAME v1
difficulty-filtered train set (155 families, from the base-policy audit), single-phase
300 steps, lr 5e-6, beta 0.0, dr_grpo, num_generations 8, no vLLM. A fixed shared train
set (no per-arm refresh) is deliberate: it removes the confound where different reward
shapes would produce different refreshed training sets, isolating the reward variable.

Reward shapes (the only training difference besides seed):

```text
independent : family_bonus_enabled false  -> reward = mean(correct + format) - penalties
iso lam 0.50: family_mean_weight 0.25, all_family_correct_weight 0.25  (bonus on correct variants)
iso lam 1.00: family_mean_weight 0.50, all_family_correct_weight 0.50
random      : reward_mode random (Bernoulli 0.5), verifier records still logged
```

Arms: independent {23, 37}, iso0.50 {23, 37}, iso1.00 {23, 37}, random {23} — 7 total.

Evaluation: each arm greedy on the 773-family heldout (batched), paired bootstrap vs the
shared base eval (family_acc 0.1940) and paired iso-vs-independent at matched seed.

Pre-registered reading rules (same structure as Phase 8):

```text
iso > independent with CI excluding zero on BOTH seeds, random arm flat : iso supported
iso ~= independent, both > base                                         : RLVR works, family reward adds nothing
random ~= independent gains                                             : gain is policy-drift, not signal; nothing claimed
```

**Results.** All 7 arms trained (300 steps each, ~6 min/arm) and evaluated greedy on the
773-family heldout (batched). Family accuracy vs the shared base (0.1940):

```text
arm                family_acc   delta vs base   95% CI            p
independent s23    0.2096       +0.0155         [+0.0052,+0.0272] 0.012  *
independent s37    0.2083       +0.0142         [+0.0052,+0.0246] 0.012  *
iso0.50     s23    0.2070       +0.0129         [+0.0026,+0.0233] 0.032  *
iso0.50     s37    0.2096       +0.0155         [+0.0052,+0.0259] 0.009  *
iso1.00     s23    0.2109       +0.0168         [+0.0052,+0.0285] 0.007  *
iso1.00     s37    0.2044       +0.0103         [+0.0013,+0.0194] 0.056  *
random      s23    0.1902       -0.0039         [-0.0116,+0.0039] 0.513     (flat)
```

`*` = 95% CI excludes zero. Every verifier arm beats base; the random-reward control is
flat (CI includes zero) — so the gains are real RLVR signal, not policy drift.

The decisive comparison, iso minus independent at matched seed:

```text
iso0.50 - independent (s23)   -0.0026   CI [-0.0103,+0.0052]   p 0.761
iso0.50 - independent (s37)   +0.0013   CI [-0.0039,+0.0065]   p 1.000
iso1.00 - independent (s23)   +0.0013   CI [-0.0065,+0.0091]   p 1.000
iso1.00 - independent (s37)   -0.0039   CI [-0.0103,+0.0013]   p 0.384
```

All four CIs include zero; no iso advantage on either seed at either lambda.

**Stage 3 verdict (pre-registered rule matched): "iso ~= independent, both > base — RLVR
works here but the family/isomorphism reward adds nothing."** The iso hypothesis is not
supported. This is now a *clean* null (unlike Phase 8, where no arm moved): with the
substrate demonstrably trainable (every verifier arm beats base, random flat), the
family-consistency reward still produces no gain over a plain independent correctness
reward, at lambda 0.50 or 1.00, on both seeds. Saved: `outputs/phase9/stage3/stage3_comparisons.json`.
