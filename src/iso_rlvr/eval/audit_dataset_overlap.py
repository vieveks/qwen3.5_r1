from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from iso_rlvr.eval.summary import family_type_counts
from iso_rlvr.io import read_jsonl


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "family_type": row["family_type"],
        "problem": row["problem"],
        "answer": row["answer"],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_overlap(
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    max_examples: int = 10,
) -> dict[str, Any]:
    train_hashes = [row_fingerprint(row) for row in train_rows]
    heldout_hashes = [row_fingerprint(row) for row in heldout_rows]
    train_counts = Counter(train_hashes)
    heldout_counts = Counter(heldout_hashes)
    train_by_hash = {row_fingerprint(row): row for row in train_rows}
    heldout_by_hash = {row_fingerprint(row): row for row in heldout_rows}
    overlap_hashes = sorted(set(train_by_hash) & set(heldout_by_hash))

    examples = []
    for fingerprint in overlap_hashes[:max_examples]:
        examples.append(
            {
                "fingerprint": fingerprint,
                "train": train_by_hash[fingerprint],
                "heldout": heldout_by_hash[fingerprint],
            }
        )

    return {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_unique_fingerprints": len(train_by_hash),
        "heldout_unique_fingerprints": len(heldout_by_hash),
        "train_duplicate_fingerprints": sum(1 for count in train_counts.values() if count > 1),
        "heldout_duplicate_fingerprints": sum(1 for count in heldout_counts.values() if count > 1),
        "overlap_count": len(overlap_hashes),
        "overlap_examples": examples,
        "train_family_type_counts": family_type_counts(train_rows),
        "heldout_family_type_counts": family_type_counts(heldout_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    report = audit_overlap(
        read_jsonl(args.train),
        read_jsonl(args.heldout),
        max_examples=args.max_examples,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
