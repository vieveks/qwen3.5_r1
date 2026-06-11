from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iso_rlvr.eval.audit_dataset_overlap import row_fingerprint
from iso_rlvr.io import group_by_family, read_jsonl, write_jsonl


def drop_overlapping_families(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop every family that has any variant colliding with the train generation.

    Filtering happens at family granularity under a fixed rule, so no per-row
    hand selection is involved.
    """
    train_fingerprints = {row_fingerprint(row) for row in train_rows}
    kept: list[dict[str, Any]] = []
    dropped_families: list[str] = []
    for family_id, family_rows in sorted(group_by_family(rows).items()):
        if any(row_fingerprint(row) in train_fingerprints for row in family_rows):
            dropped_families.append(family_id)
        else:
            kept.extend(family_rows)
    report = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "input_families": len(group_by_family(rows)),
        "dropped_families": len(dropped_families),
        "dropped_family_ids": dropped_families,
    }
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    kept, report = drop_overlapping_families(read_jsonl(args.input), read_jsonl(args.train))
    write_jsonl(args.out, kept)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
