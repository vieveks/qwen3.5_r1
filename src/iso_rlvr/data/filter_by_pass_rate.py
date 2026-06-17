"""Offline difficulty filter (Phase 9 Dataset Decision).

GRPO only learns from within-group reward variance. Prompts the policy always solves
(pass_rate == 1) and prompts it never solves (pass_rate == 0) carry zero advantage and
are dead weight. This tool reads a sampled rollout audit (N samples per packed family at
the training temperature) and keeps only the families whose pass rate is strictly inside
``(pass_low, pass_high)`` -- the offline equivalent of DAPO dynamic sampling.

Pass is measured at the packed-family level (``all_family_correct``): a sample "passes"
when every variant in the family is correct, which is the unit the family reward rewards.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from iso_rlvr.io import read_jsonl, write_jsonl


def compute_pass_rates(audit_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-family pass rate from per-sample audit records."""
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit_records:
        grouped[str(record["family_id"])].append(record)
    for family_id, samples in grouped.items():
        passes = sum(1 for s in samples if bool(s.get("all_family_correct")))
        by_family[family_id] = {
            "family_id": family_id,
            "family_type": str(samples[0].get("family_type", "")),
            "samples": len(samples),
            "pass_count": passes,
            "pass_rate": passes / len(samples) if samples else 0.0,
        }
    return by_family


def _histogram(pass_rates: list[float], buckets: int = 5) -> dict[str, int]:
    """Bucket pass rates into ``buckets`` equal bins over [0, 1]."""
    counts: dict[str, int] = {}
    for i in range(buckets):
        lo, hi = i / buckets, (i + 1) / buckets
        counts[f"[{lo:.2f},{hi:.2f}{']' if i == buckets - 1 else ')'}"] = 0
    keys = list(counts)
    for rate in pass_rates:
        idx = min(int(rate * buckets), buckets - 1)
        counts[keys[idx]] += 1
    return counts


def filter_by_pass_rate(
    audit_records: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    pass_low: float = 0.0,
    pass_high: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep dataset rows whose family pass rate is strictly in (pass_low, pass_high)."""
    stats = compute_pass_rates(audit_records)
    kept_ids = {
        fid for fid, s in stats.items() if pass_low < s["pass_rate"] < pass_high
    }
    kept = [row for row in dataset_rows if str(row["family_id"]) in kept_ids]

    all_rates = [s["pass_rate"] for s in stats.values()]
    dead_zero = sum(1 for r in all_rates if r <= pass_low)
    dead_one = sum(1 for r in all_rates if r >= pass_high)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"audited": 0, "kept": 0})
    for s in stats.values():
        ft = s["family_type"]
        by_type[ft]["audited"] += 1
        if pass_low < s["pass_rate"] < pass_high:
            by_type[ft]["kept"] += 1

    report = {
        "pass_low": pass_low,
        "pass_high": pass_high,
        "audited_families": len(stats),
        "kept_families": len(kept_ids),
        "kept_rows": len(kept),
        "dropped_all_wrong": dead_zero,
        "dropped_all_correct": dead_one,
        "pass_rate_histogram": _histogram(all_rates),
        "by_family_type": {ft: dict(v) for ft, v in sorted(by_type.items())},
    }
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True, help="rollout audit records jsonl")
    parser.add_argument("--dataset", type=Path, required=True, help="packed dataset to filter")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pass-low", type=float, default=0.0)
    parser.add_argument("--pass-high", type=float, default=1.0)
    args = parser.parse_args()

    kept, report = filter_by_pass_rate(
        read_jsonl(args.audit),
        read_jsonl(args.dataset),
        pass_low=args.pass_low,
        pass_high=args.pass_high,
    )
    write_jsonl(args.out, kept)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
