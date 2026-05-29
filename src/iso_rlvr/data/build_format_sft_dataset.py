from __future__ import annotations

import argparse
from fractions import Fraction
import math
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


def build_rational_linear_trace(row: dict[str, Any]) -> str:
    traces = []
    metadata_rows = row.get("metadata", [])
    gold_answers = [str(answer) for answer in row["gold_answers"]]
    if len(metadata_rows) != len(gold_answers):
        raise ValueError("rational_linear_equation metadata must match gold_answers length.")

    for idx, (metadata, gold_answer) in enumerate(
        zip(metadata_rows, gold_answers),
        start=1,
    ):
        a_text = str(metadata["a"])
        b_text = str(metadata["b"])
        c_text = str(metadata["c"])
        a = Fraction(a_text)
        b = Fraction(b_text)
        c = Fraction(c_text)
        isolated = c - b
        solution = isolated / a
        if _format_fraction(solution) != gold_answer:
            raise ValueError(
                "rational_linear_equation trace does not match gold answer: "
                f"{_format_fraction(solution)} != {gold_answer}"
            )

        traces.append(
            "\n".join(
                [
                    f"Problem {idx} isolate: {a_text}x = {c_text} - ({b_text}) = {_format_fraction(isolated)}",
                    f"Problem {idx} divide: x = {_format_fraction(isolated)} / {a_text} = {gold_answer}",
                ]
            )
        )

    return "\n\n".join(traces)


def build_rational_system_trace(row: dict[str, Any]) -> str:
    traces = []
    metadata_rows = row.get("metadata", [])
    gold_answers = [str(answer) for answer in row["gold_answers"]]
    if len(metadata_rows) != len(gold_answers):
        raise ValueError("rational_system_target metadata must match gold_answers length.")

    for idx, (metadata, gold_answer) in enumerate(
        zip(metadata_rows, gold_answers),
        start=1,
    ):
        a = int(metadata["a"])
        b = int(metadata["b"])
        c = int(metadata["c"])
        d = int(metadata["d"])
        x = Fraction(str(metadata["x"]))
        y = Fraction(str(metadata["y"]))
        target_name = str(metadata["target_name"])
        e = a * x + b * y
        f = c * x + d * y
        det = a * d - b * c
        if det == 0:
            raise ValueError("rational_system_target trace requires nonzero determinant.")

        x_rhs = d * e - b * f
        y_rhs = a * f - c * e
        solved_x = x_rhs / det
        solved_y = y_rhs / det
        if solved_x != x or solved_y != y:
            raise ValueError("rational_system_target metadata is inconsistent.")

        if target_name == "x + y":
            target = x + y
            target_expression = f"({_format_fraction(x)}) + ({_format_fraction(y)})"
        elif target_name == "x - y":
            target = x - y
            target_expression = f"({_format_fraction(x)}) - ({_format_fraction(y)})"
        elif target_name == "2x + y":
            target = 2 * x + y
            target_expression = f"2({_format_fraction(x)}) + ({_format_fraction(y)})"
        elif target_name == "3y - x":
            target = 3 * y - x
            target_expression = f"3({_format_fraction(y)}) - ({_format_fraction(x)})"
        else:
            raise ValueError(f"Unsupported rational_system_target target: {target_name}")
        if _format_fraction(target) != gold_answer:
            raise ValueError(
                "rational_system_target trace does not match gold answer: "
                f"{_format_fraction(target)} != {gold_answer}"
            )

        traces.append(
            "\n".join(
                [
                    f"Problem {idx} eliminate y: {det}x = {_format_fraction(x_rhs)}, so x = {_format_fraction(x)}",
                    f"Problem {idx} eliminate x: {det}y = {_format_fraction(y_rhs)}, so y = {_format_fraction(y)}",
                    f"Problem {idx} target: {target_name} = {target_expression} = {gold_answer}",
                ]
            )
        )

    return "\n\n".join(traces)


def build_chinese_remainder_trace(row: dict[str, Any]) -> str:
    traces = []
    metadata_rows = row.get("metadata", [])
    gold_answers = [str(answer) for answer in row["gold_answers"]]
    if len(metadata_rows) != len(gold_answers):
        raise ValueError("chinese_remainder metadata must match gold_answers length.")

    for idx, (metadata, gold_answer) in enumerate(
        zip(metadata_rows, gold_answers),
        start=1,
    ):
        answer = int(metadata["answer"])
        mod_a = int(metadata["mod_a"])
        mod_b = int(metadata["mod_b"])
        if math.gcd(mod_a, mod_b) != 1:
            raise ValueError("chinese_remainder trace requires coprime moduli.")
        product = mod_a * mod_b
        if not 0 <= answer < product:
            raise ValueError("chinese_remainder answer must be least nonnegative modulo product.")
        rem_a = answer % mod_a
        rem_b = answer % mod_b
        if str(answer) != gold_answer:
            raise ValueError(
                "chinese_remainder trace does not match gold answer: "
                f"{answer} != {gold_answer}"
            )

        if (answer - rem_a) % mod_a != 0:
            raise ValueError("chinese_remainder answer is not reachable from first residue.")
        chosen_k = (answer - rem_a) // mod_a
        for prior_k in range(chosen_k):
            if (rem_a + mod_a * prior_k) % mod_b == rem_b:
                raise ValueError("chinese_remainder answer is not the least nonnegative solution.")

        traces.append(
            "\n".join(
                [
                    f"Problem {idx} form: x = {rem_a} + {mod_a}k",
                    f"Problem {idx} choose k = {chosen_k}: x = {gold_answer}",
                    f"Problem {idx} check: {answer} mod {mod_b} = {rem_b}",
                    f"Problem {idx} least value: x = {gold_answer}",
                ]
            )
        )

    return "\n\n".join(traces)


def build_sft_row(
    row: dict[str, Any],
    missing_average_traces: bool = False,
    rational_linear_traces: bool = False,
    rational_system_traces: bool = False,
    chinese_remainder_traces: bool = False,
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
    elif (
        rational_linear_traces
        and row["family_type"] == "rational_linear_equation"
        and not force_answer_only
    ):
        completion = f"{build_rational_linear_trace(row)}\n\n{xml_completion}"
        target_style = "rational_linear_trace_xml"
    elif (
        rational_system_traces
        and row["family_type"] == "rational_system_target"
        and not force_answer_only
    ):
        completion = f"{build_rational_system_trace(row)}\n\n{xml_completion}"
        target_style = "rational_system_trace_xml"
    elif (
        chinese_remainder_traces
        and row["family_type"] == "chinese_remainder"
        and not force_answer_only
    ):
        completion = f"{build_chinese_remainder_trace(row)}\n\n{xml_completion}"
        target_style = "chinese_remainder_trace_xml"
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
    rational_linear_traces: bool = False,
    rational_system_traces: bool = False,
    chinese_remainder_traces: bool = False,
    include_answer_only_copy: bool = False,
) -> list[dict[str, Any]]:
    if (
        include_answer_only_copy
        and missing_average_traces
        and row["family_type"] == "missing_average"
    ):
        return [
            build_sft_row(row, missing_average_traces=False),
            build_sft_row(
                row,
                missing_average_traces=True,
                rational_linear_traces=rational_linear_traces,
                rational_system_traces=rational_system_traces,
                chinese_remainder_traces=chinese_remainder_traces,
            ),
        ]
    return [
        build_sft_row(
            row,
            missing_average_traces=missing_average_traces,
            rational_linear_traces=rational_linear_traces,
            rational_system_traces=rational_system_traces,
            chinese_remainder_traces=chinese_remainder_traces,
        )
    ]


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
    rational_linear_traces: bool = False,
    rational_system_traces: bool = False,
    chinese_remainder_traces: bool = False,
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
            rational_linear_traces=rational_linear_traces,
            rational_system_traces=rational_system_traces,
            chinese_remainder_traces=chinese_remainder_traces,
            include_answer_only_copy=include_answer_only_copy,
        )
    ]
    heldout_rows = [
        sft_row
        for row in packed_heldout_rows
        for sft_row in build_sft_rows(
            row,
            missing_average_traces=missing_average_traces,
            rational_linear_traces=rational_linear_traces,
            rational_system_traces=rational_system_traces,
            chinese_remainder_traces=chinese_remainder_traces,
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
    parser.add_argument("--rational-linear-traces", action="store_true")
    parser.add_argument("--rational-system-traces", action="store_true")
    parser.add_argument("--chinese-remainder-traces", action="store_true")
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
        rational_linear_traces=args.rational_linear_traces,
        rational_system_traces=args.rational_system_traces,
        chinese_remainder_traces=args.chinese_remainder_traces,
        include_answer_only_copy=args.include_answer_only_copy,
    )


if __name__ == "__main__":
    main()
