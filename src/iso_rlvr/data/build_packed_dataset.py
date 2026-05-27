from __future__ import annotations

import argparse
import re
from pathlib import Path
import random
from typing import Any

from iso_rlvr.io import group_by_family, read_jsonl, write_jsonl


VARIANT_SUFFIX_RE = re.compile(r"_v(\d+)$")


def variant_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    variant_id = str(row["variant_id"])
    match = VARIANT_SUFFIX_RE.search(variant_id)
    if match:
        return (int(match.group(1)), variant_id)
    return (10**9, variant_id)


def parse_family_types(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    family_types: set[str] = set()
    for value in values:
        family_types.update(item.strip() for item in value.split(",") if item.strip())
    return family_types or None


def build_packed_prompt(rows: list[dict[str, Any]]) -> str:
    answer_lines = "\n".join(
        f"Answer {idx}: <number>" for idx in range(1, len(rows) + 1)
    )
    problem_lines = "\n\n".join(
        f"Problem {idx}: {row['problem']}" for idx, row in enumerate(rows, start=1)
    )
    return (
        f"{problem_lines}\n\n"
        "Solve each problem silently. Return only the final answers, with no "
        "reasoning or extra text. Use exactly this format:\n\n"
        f"{answer_lines}"
    )


def pack_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot pack an empty family.")

    ordered = sorted(rows, key=variant_sort_key)
    family_ids = {str(row["family_id"]) for row in ordered}
    if len(family_ids) != 1:
        raise ValueError(f"Expected one family_id, found {sorted(family_ids)}.")

    family_types = {str(row["family_type"]) for row in ordered}
    if len(family_types) != 1:
        raise ValueError(
            f"Expected one family_type in {next(iter(family_ids))}, found {sorted(family_types)}."
        )

    return {
        "family_id": str(ordered[0]["family_id"]),
        "family_type": str(ordered[0]["family_type"]),
        "num_variants": len(ordered),
        "variant_ids": [str(row["variant_id"]) for row in ordered],
        "problems": [str(row["problem"]) for row in ordered],
        "gold_answers": [str(row["answer"]) for row in ordered],
        "metadata": [row.get("metadata", {}) for row in ordered],
        "prompt": build_packed_prompt(ordered),
    }


def pack_dataset(
    rows: list[dict[str, Any]],
    include_family_types: set[str] | None = None,
    expected_variants: int | None = None,
    max_variants_per_family: int | None = None,
    max_families: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    families = []
    for family_id, family_rows in group_by_family(rows).items():
        family_type = str(family_rows[0]["family_type"])
        if include_family_types is not None and family_type not in include_family_types:
            continue
        if expected_variants is not None and len(family_rows) != expected_variants:
            continue
        if max_variants_per_family is not None:
            family_rows = sorted(family_rows, key=variant_sort_key)[:max_variants_per_family]
        families.append((family_id, family_rows))

    families.sort(key=lambda item: item[0])
    if shuffle:
        random.Random(seed).shuffle(families)
    if max_families is not None:
        families = families[:max_families]

    return [pack_family(family_rows) for _, family_rows in families]


def build_packed_dataset(
    input_path: Path,
    out: Path,
    include_family_types: set[str] | None = None,
    expected_variants: int | None = None,
    max_variants_per_family: int | None = None,
    max_families: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> None:
    packed = pack_dataset(
        read_jsonl(input_path),
        include_family_types=include_family_types,
        expected_variants=expected_variants,
        max_variants_per_family=max_variants_per_family,
        max_families=max_families,
        shuffle=shuffle,
        seed=seed,
    )
    write_jsonl(out, packed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--include-family-type",
        action="append",
        help="Family type to include. May be repeated or comma-separated.",
    )
    parser.add_argument("--expected-variants", type=int, default=None)
    parser.add_argument("--max-variants-per-family", type=int, default=None)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_packed_dataset(
        args.input,
        args.out,
        include_family_types=parse_family_types(args.include_family_type),
        expected_variants=args.expected_variants,
        max_variants_per_family=args.max_variants_per_family,
        max_families=args.max_families,
        shuffle=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
