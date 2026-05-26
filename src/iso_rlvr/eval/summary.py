from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from iso_rlvr.rewards.iso import ScoredResponse, summarize


def scored_from_eval_rows(rows: list[dict[str, Any]]) -> list[ScoredResponse]:
    scored = []
    for row in rows:
        scored.append(
            ScoredResponse(
                family_id=str(row["family_id"]),
                variant_id=str(row["variant_id"]),
                gold=str(row["answer"]),
                response=str(row.get("model_response", row.get("response", ""))),
                correct=bool(row["correct"]),
                extracted_answer=row.get("extracted_answer"),
                token_count=int(row.get("response_tokens", 0)),
            )
        )
    return scored


def summarize_by_family_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["family_type"])].append(row)

    out: dict[str, dict[str, float | int]] = {}
    for family_type in sorted(by_type):
        type_rows = by_type[family_type]
        type_summary = summarize(scored_from_eval_rows(type_rows))
        family_ids = {str(row["family_id"]) for row in type_rows}
        type_summary.update(
            {
                "examples": len(type_rows),
                "families": len(family_ids),
            }
        )
        out[family_type] = type_summary
    return out


def family_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["family_type"]) for row in rows).items()))
