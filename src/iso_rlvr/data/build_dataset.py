from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from iso_rlvr.data.families import make_family


def build_dataset(out: Path, families: int, variants: int, seed: int, profile: str) -> None:
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for family_idx in range(families):
            family_id = f"fam_{family_idx:06d}"
            for variant in make_family(rng, family_id, variants, profile):
                handle.write(json.dumps(asdict(variant), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=200)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--profile",
        choices=["easy", "mixed", "harder", "challenge", "calibrated"],
        default="mixed",
    )
    args = parser.parse_args()
    build_dataset(args.out, args.families, args.variants, args.seed, args.profile)


if __name__ == "__main__":
    main()

