# Blog Writing Guide

Use `BLOG_POST.md` as the canonical draft. This guide is for editing it into a publishable post.

## Audience

Write for ML practitioners who know RLHF/RLVR/GRPO at a high level and may be running small local RL experiments. They do not need a transformer primer. They do need practical detail on why verifier interfaces fail.

## Core Message

The post should make three points:

1. The original hypothesis was that isomorphic family rewards can improve consistency across related math variants.
2. The hidden bottleneck was the reward interface, not GRPO optimization.
3. After stabilizing the interface, Iso-RLVR produced a narrow but clean family-accuracy gain.

## Recommended Title

Use a title that names the lesson, not the project:

```text
RLVR Does Not Just Fail at Optimization. It Fails at the Interface.
```

Alternative:

```text
The Reward Interface Problem: What I Learned Building Isomorphic RLVR From Scratch
```

## Structure

Suggested blog structure:

1. **Opening story**
   - Start with why the project began.
   - State the surprise: most time went into making the reward measurable.

2. **The hypothesis**
   - Explain standard RLVR versus Iso-RLVR.
   - Use plain language: correct single answers do not prove robust reasoning.

3. **The reward interface problem**
   - Show the pipeline: `prompt -> completion -> parser -> verifier -> reward`.
   - Show the bad numbers: `parse_complete_rate: 0.3750`, `answer_count_mismatch_rate: 0.6250`.
   - Explain why GRPO on that surface would optimize parser behavior.

4. **The stabilization**
   - Show XML answer contract.
   - Explain SFT bridge as contract teaching, not math teaching.
   - Show `parse_complete_rate: 1.0000`.

5. **The result**
   - Put the lambda sweep table early.
   - Explain family accuracy.
   - Mention two-seed replication for `lambda_iso=0.50`.
   - State zero parse regression.

6. **What did not work**
   - Briefly cover deterministic trace failure modes.
   - Briefly cover Phase 7 `<think> + <answers>` blocker.

7. **Limitations**
   - Keep all 11 limitations.
   - Do not soften them.

8. **Future work**
   - Replicate best lambda.
   - Add intervals and blind eval.
   - Revisit `<think> + <answers>` after fixing the answer transition.

## Tone

Use first person. This is a practitioner write-up, not a paper.

Good tone:

```text
I set out to test reward shaping and ended up debugging the verifier interface.
```

Avoid:

```text
In this paper, we demonstrate...
```

## Length

Target:

```text
1500 to 2500 words
```

`BLOG_POST.md` is intentionally near the upper end. For a shorter version, cut the detailed by-family table and some Phase 7 detail.

## Essential Figure

The essential figure is the lambda sweep table:

| Run | Accuracy | Family accuracy |
| --- | ---: | ---: |
| Independent | 0.7188 | 0.5625 |
| Iso `lambda_iso=0.25` | 0.7500 | 0.6250 |
| Iso `lambda_iso=0.50` | 0.7500 | 0.6250 |
| Iso `lambda_iso=1.00` | 0.7813 | 0.6875 |

If creating a visual chart, plot `lambda_iso` on the x-axis and family accuracy on the y-axis, with independent as the leftmost baseline.

## What Not To Include

Do not include:

- full phase logs
- every config path
- long Python snippets
- TRL API debugging details
- all failed variants of trace engineering

Link to the repo and the phase docs for readers who want the full audit trail.

## Publication Checklist

Before publishing:

- Verify `lambda_iso=1.00` accuracy is `0.7813`.
- Verify the post says the result is narrow.
- Verify Phase 7 is framed as future work.
- Include a repo link.
- Include links to `phase5.md`, `phase6.md`, `phase7.md`, and `CRITIC_BRIEF.md`.
- Keep the limitations section intact.

## Suggested Publishing Plan

Publish first somewhere you control:

```text
personal site, GitHub Pages, or Substack
```

Then cross-post or share:

```text
Hugging Face community blog
Towards Data Science
Reddit r/MachineLearning
X/LinkedIn thread with the lambda sweep table
```

The repo and phase logs are the proof of work. Link them prominently.
