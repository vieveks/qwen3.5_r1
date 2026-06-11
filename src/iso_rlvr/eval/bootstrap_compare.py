from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from iso_rlvr.io import read_jsonl


def family_metrics(row: dict[str, Any]) -> dict[str, float]:
    correctness = [bool(value) for value in row["correctness"]]
    if not correctness:
        raise ValueError(f"Family {row.get('family_id')} has empty correctness.")
    return {
        "variant_correct": float(sum(correctness)),
        "variant_total": float(len(correctness)),
        "all_family_correct": 1.0 if all(correctness) else 0.0,
    }


def join_by_family(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family_a = {str(row["family_id"]): row for row in rows_a}
    by_family_b = {str(row["family_id"]): row for row in rows_b}
    if len(by_family_a) != len(rows_a) or len(by_family_b) != len(rows_b):
        raise ValueError("Duplicate family_id rows are not supported in paired comparison.")

    shared = sorted(set(by_family_a) & set(by_family_b))
    if not shared:
        raise ValueError("No shared family_id values between the two eval files.")

    pairs = []
    for family_id in shared:
        row_a = by_family_a[family_id]
        row_b = by_family_b[family_id]
        if list(row_a["gold_answers"]) != list(row_b["gold_answers"]):
            raise ValueError(f"Gold answers differ for family {family_id}; files are not paired.")
        pairs.append(
            {
                "family_id": family_id,
                "family_type": str(row_a["family_type"]),
                "a": family_metrics(row_a),
                "b": family_metrics(row_b),
            }
        )
    return pairs


def point_estimates(pairs: list[dict[str, Any]], arm: str) -> dict[str, float]:
    variant_correct = sum(pair[arm]["variant_correct"] for pair in pairs)
    variant_total = sum(pair[arm]["variant_total"] for pair in pairs)
    return {
        "accuracy": variant_correct / variant_total if variant_total else 0.0,
        "family_accuracy": sum(pair[arm]["all_family_correct"] for pair in pairs) / len(pairs),
    }


def _deltas_for_sample(pairs: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
    sampled = [pairs[idx] for idx in indices]
    estimates_a = point_estimates(sampled, "a")
    estimates_b = point_estimates(sampled, "b")
    return {
        "accuracy_delta": estimates_a["accuracy"] - estimates_b["accuracy"],
        "family_accuracy_delta": estimates_a["family_accuracy"] - estimates_b["family_accuracy"],
    }


def paired_bootstrap(
    pairs: list[dict[str, Any]],
    iterations: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    rng = random.Random(seed)
    count = len(pairs)
    samples: dict[str, list[float]] = {"accuracy_delta": [], "family_accuracy_delta": []}
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        deltas = _deltas_for_sample(pairs, indices)
        for key, value in deltas.items():
            samples[key].append(value)

    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    result = {}
    for key, values in samples.items():
        values.sort()
        result[key] = {
            "ci_lower": _quantile(values, lower_q),
            "ci_upper": _quantile(values, upper_q),
        }
    return result


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sample.")
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_sign_flip_test(
    pairs: list[dict[str, Any]],
    metric: str,
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    if metric == "accuracy":
        per_family = [
            pair["a"]["variant_correct"] / pair["a"]["variant_total"]
            - pair["b"]["variant_correct"] / pair["b"]["variant_total"]
            for pair in pairs
        ]
    elif metric == "family_accuracy":
        per_family = [
            pair["a"]["all_family_correct"] - pair["b"]["all_family_correct"] for pair in pairs
        ]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    observed = sum(per_family) / len(per_family)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        flipped = sum(delta if rng.random() < 0.5 else -delta for delta in per_family)
        if abs(flipped / len(per_family)) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "observed_mean_delta": observed,
        "p_value": (extreme + 1) / (iterations + 1),
    }


def compare_eval_files(
    path_a: Path,
    path_b: Path,
    label_a: str = "a",
    label_b: str = "b",
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    pairs = join_by_family(read_jsonl(path_a), read_jsonl(path_b))
    estimates_a = point_estimates(pairs, "a")
    estimates_b = point_estimates(pairs, "b")
    bootstrap = paired_bootstrap(pairs, iterations=iterations, seed=seed)
    report: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "paired_families": len(pairs),
        label_a: estimates_a,
        label_b: estimates_b,
        "accuracy_delta": estimates_a["accuracy"] - estimates_b["accuracy"],
        "family_accuracy_delta": estimates_a["family_accuracy"] - estimates_b["family_accuracy"],
        "bootstrap": bootstrap,
        "sign_flip": {
            "accuracy": paired_sign_flip_test(pairs, "accuracy", iterations=iterations, seed=seed),
            "family_accuracy": paired_sign_flip_test(
                pairs, "family_accuracy", iterations=iterations, seed=seed
            ),
        },
    }

    by_type: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_type.setdefault(pair["family_type"], []).append(pair)
    report["by_family_type"] = {
        family_type: {
            "paired_families": len(type_pairs),
            label_a: point_estimates(type_pairs, "a"),
            label_b: point_estimates(type_pairs, "b"),
        }
        for family_type, type_pairs in sorted(by_type.items())
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="Eval output jsonl for arm A.")
    parser.add_argument("--b", type=Path, required=True, help="Eval output jsonl for arm B.")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = compare_eval_files(
        args.a,
        args.b,
        label_a=args.label_a,
        label_b=args.label_b,
        iterations=args.iterations,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
