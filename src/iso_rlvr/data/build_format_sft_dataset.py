from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import random
from typing import Any

from iso_rlvr.io import read_jsonl, write_jsonl


def build_xml_completion(gold_answers: list[str]) -> str:
    answer_lines = "\n".join(
        f"<answer_{idx}>{answer}</answer_{idx}>"
        for idx, answer in enumerate(gold_answers, start=1)
    )
    return f"<answers>\n{answer_lines}\n</answers>"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def build_missing_average_trace(row: dict[str, Any]) -> str:
    traces = []
    metadata_rows = row.get("metadata", [])
    gold_answers = [str(answer) for answer in row["gold_answers"]]
    if len(metadata_rows) != len(gold_answers):
        raise ValueError("missing_average metadata must match gold_answers length.")

    for idx, (metadata, gold_answer) in enumerate(
        zip(metadata_rows, gold_answers),
        start=1,
    ):
        known_values = _parse_int_list(str(metadata["known"]))
        mean_text = str(metadata["final_average"])
        mean = Fraction(mean_text)
        count = len(known_values) + 1
        total_needed = mean * count
        known_total = sum(known_values)
        missing_value = total_needed - known_total
        if _format_fraction(missing_value) != gold_answer:
            raise ValueError(
                "missing_average trace does not match gold answer: "
                f"{_format_fraction(missing_value)} != {gold_answer}"
            )

        known_expression = " + ".join(str(value) for value in known_values)
        traces.append(
            "\n".join(
                [
                    f"Problem {idx} sum needed: {mean_text} x {count} = {_format_fraction(total_needed)}",
                    f"Problem {idx} known sum: {known_expression} = {known_total}",
                    f"Problem {idx} missing value: {_format_fraction(total_needed)} - {known_total} = {gold_answer}",
                ]
            )
        )

    return "\n\n".join(traces)


def build_sft_row(
    row: dict[str, Any],
    missing_average_traces: bool = False,
    force_answer_only: bool = False,
) -> dict[str, Any]:
    gold_answers = [str(answer) for answer in row["gold_answers"]]
    xml_completion = build_xml_completion(gold_answers)
    if (
        missing_average_traces
        and row["family_type"] == "missing_average"
        and not force_answer_only
    ):
        completion = f"{build_missing_average_trace(row)}\n\n{xml_completion}"
        target_style = "missing_average_trace_xml"
    else:
        completion = xml_completion
        target_style = "answer_only_xml"

    return {
        "family_id": str(row["family_id"]),
        "family_type": str(row["family_type"]),
        "variant_ids": [str(variant_id) for variant_id in row["variant_ids"]],
        "num_variants": int(row["num_variants"]),
        "gold_answers": gold_answers,
        "metadata": row.get("metadata", []),
        "prompt_format": str(row.get("prompt_format", "")),
        "target_style": target_style,
        "prompt": str(row["prompt"]),
        "completion": completion,
        "text": f"{row['prompt']}\n{completion}",
    }


def build_sft_rows(
    row: dict[str, Any],
    missing_average_traces: bool = False,
    include_answer_only_copy: bool = False,
) -> list[dict[str, Any]]:
    if (
        include_answer_only_copy
        and missing_average_traces
        and row["family_type"] == "missing_average"
    ):
        return [
            build_sft_row(row, missing_average_traces=False),
            build_sft_row(row, missing_average_traces=True),
        ]
    return [build_sft_row(row, missing_average_traces=missing_average_traces)]


def split_by_family_id(
    rows: list[dict[str, Any]],
    heldout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in [0.0, 1.0).")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    heldout_count = int(round(len(shuffled) * heldout_fraction))
    if heldout_fraction > 0 and heldout_count == 0 and shuffled:
        heldout_count = 1

    heldout_family_ids = {str(row["family_id"]) for row in shuffled[:heldout_count]}
    train_rows = [row for row in rows if str(row["family_id"]) not in heldout_family_ids]
    heldout_rows = [row for row in rows if str(row["family_id"]) in heldout_family_ids]
    return train_rows, heldout_rows


def build_format_sft_dataset(
    input_path: Path,
    train_out: Path,
    heldout_out: Path | None = None,
    packed_train_out: Path | None = None,
    packed_heldout_out: Path | None = None,
    heldout_fraction: float = 0.1,
    seed: int = 0,
    max_examples: int | None = None,
    missing_average_traces: bool = False,
    include_answer_only_copy: bool = False,
) -> None:
    rows = read_jsonl(input_path)
    if max_examples is not None:
        rows = rows[:max_examples]

    packed_train_rows, packed_heldout_rows = split_by_family_id(
        rows,
        heldout_fraction=heldout_fraction,
        seed=seed,
    )
    train_rows = [
        sft_row
        for row in packed_train_rows
        for sft_row in build_sft_rows(
            row,
            missing_average_traces=missing_average_traces,
            include_answer_only_copy=include_answer_only_copy,
        )
    ]
    heldout_rows = [
        sft_row
        for row in packed_heldout_rows
        for sft_row in build_sft_rows(
            row,
            missing_average_traces=missing_average_traces,
            include_answer_only_copy=include_answer_only_copy,
        )
    ]

    write_jsonl(train_out, train_rows)
    if heldout_out is not None:
        write_jsonl(heldout_out, heldout_rows)
    if packed_train_out is not None:
        write_jsonl(packed_train_out, packed_train_rows)
    if packed_heldout_out is not None:
        write_jsonl(packed_heldout_out, packed_heldout_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--heldout-out", type=Path, default=None)
    parser.add_argument("--packed-train-out", type=Path, default=None)
    parser.add_argument("--packed-heldout-out", type=Path, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--missing-average-traces", action="store_true")
    parser.add_argument("--include-answer-only-copy", action="store_true")
    args = parser.parse_args()

    build_format_sft_dataset(
        args.input,
        args.train_out,
        heldout_out=args.heldout_out,
        packed_train_out=args.packed_train_out,
        packed_heldout_out=args.packed_heldout_out,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
        max_examples=args.max_examples,
        missing_average_traces=args.missing_average_traces,
        include_answer_only_copy=args.include_answer_only_copy,
    )


if __name__ == "__main__":
    main()
