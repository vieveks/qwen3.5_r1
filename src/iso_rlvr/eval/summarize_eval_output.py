from __future__ import annotations

import argparse
import json
from pathlib import Path

from iso_rlvr.eval.summary import scored_from_eval_rows, summarize_by_family_type
from iso_rlvr.io import read_jsonl
from iso_rlvr.rewards.iso import summarize


def summarize_eval_rows(rows: list[dict]) -> dict:
    summary = summarize(scored_from_eval_rows(rows))
    summary["by_family_type"] = summarize_by_family_type(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = summarize_eval_rows(read_jsonl(args.input))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
