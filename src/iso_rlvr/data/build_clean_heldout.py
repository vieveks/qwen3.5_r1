from __future__ import annotations

import argparse
import json
from pathlib import Path

from iso_rlvr.data.build_dataset import generate_dataset
from iso_rlvr.eval.audit_dataset_overlap import audit_overlap
from iso_rlvr.io import read_jsonl, write_jsonl


def find_clean_heldout(
    train_rows: list[dict],
    families: int,
    variants: int,
    profile: str,
    start_seed: int,
    max_seed: int,
    require_unique_heldout: bool,
) -> tuple[int, list[dict], dict]:
    for seed in range(start_seed, max_seed + 1):
        rows = generate_dataset(families=families, variants=variants, seed=seed, profile=profile)
        report = audit_overlap(train_rows, rows, max_examples=3)
        heldout_unique_ok = (
            report["heldout_duplicate_fingerprints"] == 0 if require_unique_heldout else True
        )
        if report["overlap_count"] == 0 and heldout_unique_ok:
            return seed, rows, report
    raise RuntimeError(
        f"No clean held-out seed found in [{start_seed}, {max_seed}] "
        f"for profile={profile!r}, families={families}, variants={variants}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=200)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--profile", default="calibrated")
    parser.add_argument("--start-seed", type=int, default=32)
    parser.add_argument("--max-seed", type=int, default=200)
    parser.add_argument("--require-unique-heldout", action="store_true")
    args = parser.parse_args()

    seed, rows, report = find_clean_heldout(
        train_rows=read_jsonl(args.train),
        families=args.families,
        variants=args.variants,
        profile=args.profile,
        start_seed=args.start_seed,
        max_seed=args.max_seed,
        require_unique_heldout=args.require_unique_heldout,
    )
    write_jsonl(args.out, rows)
    report = {"selected_seed": seed, "output_path": str(args.out), **report}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
