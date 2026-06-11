from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iso_rlvr.io import read_jsonl


def summarize_family_consistency(
    rows: list[dict[str, Any]],
    include_by_family_type: bool = True,
) -> dict[str, Any]:
    """Summarize exploded single-variant eval rows by latent family.

    Unlike the packed summary, family consistency here means every variant of a
    family was answered correctly in a separate prompt context.
    """
    if not rows:
        summary: dict[str, Any] = {
            "families": 0,
            "variant_examples": 0,
            "accuracy": 0.0,
            "cross_context_family_accuracy": 0.0,
            "parse_complete_rate": 0.0,
        }
        if include_by_family_type:
            summary["by_family_type"] = {}
        return summary

    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["num_variants"]) != 1:
            raise ValueError(
                f"Expected exploded single-variant rows; family {row['family_id']} "
                f"has num_variants={row['num_variants']}."
            )
        families.setdefault(str(row["family_id"]), []).append(row)

    variant_total = len(rows)
    variant_correct = sum(sum(bool(value) for value in row["correctness"]) for row in rows)
    consistent_families = sum(
        1
        for family_rows in families.values()
        if all(all(bool(value) for value in row["correctness"]) for row in family_rows)
    )
    summary = {
        "families": len(families),
        "variant_examples": variant_total,
        "accuracy": variant_correct / variant_total,
        "cross_context_family_accuracy": consistent_families / len(families),
        "parse_complete_rate": sum(bool(row["parse_complete"]) for row in rows) / variant_total,
    }
    if include_by_family_type:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(str(row["family_type"]), []).append(row)
        summary["by_family_type"] = {
            family_type: summarize_family_consistency(type_rows, include_by_family_type=False)
            for family_type, type_rows in sorted(by_type.items())
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Exploded eval output jsonl.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = summarize_family_consistency(read_jsonl(args.input))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
