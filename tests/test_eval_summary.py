from iso_rlvr.eval.audit_dataset_overlap import audit_overlap
from iso_rlvr.eval.summary import summarize_by_family_type
from iso_rlvr.eval.summarize_eval_output import summarize_eval_rows


def test_summarize_by_family_type_reports_type_local_family_accuracy():
    rows = [
        {
            "family_id": "f1",
            "variant_id": "f1_v0",
            "family_type": "linear",
            "answer": "1",
            "model_response": "Answer: 1",
            "correct": True,
            "extracted_answer": "1",
            "response_tokens": 5,
        },
        {
            "family_id": "f1",
            "variant_id": "f1_v1",
            "family_type": "linear",
            "answer": "2",
            "model_response": "Answer: 0",
            "correct": False,
            "extracted_answer": "0",
            "response_tokens": 7,
        },
        {
            "family_id": "f2",
            "variant_id": "f2_v0",
            "family_type": "average",
            "answer": "3",
            "model_response": "Answer: 3",
            "correct": True,
            "extracted_answer": "3",
            "response_tokens": 4,
        },
    ]

    by_type = summarize_by_family_type(rows)

    assert by_type["linear"]["examples"] == 2
    assert by_type["linear"]["families"] == 1
    assert by_type["linear"]["accuracy"] == 0.5
    assert by_type["linear"]["family_accuracy"] == 0.0
    assert by_type["average"]["family_accuracy"] == 1.0


def test_audit_overlap_uses_problem_answer_type_fingerprint():
    train_rows = [
        {
            "family_id": "train_f1",
            "variant_id": "train_f1_v0",
            "family_type": "linear",
            "problem": "Solve x + 1 = 3.",
            "answer": "2",
        }
    ]
    heldout_rows = [
        {
            "family_id": "heldout_f9",
            "variant_id": "heldout_f9_v0",
            "family_type": "linear",
            "problem": "Solve x + 1 = 3.",
            "answer": "2",
        }
    ]

    report = audit_overlap(train_rows, heldout_rows)

    assert report["overlap_count"] == 1
    assert report["train_family_type_counts"] == {"linear": 1}


def test_summarize_eval_rows_matches_global_and_type_metrics():
    rows = [
        {
            "family_id": "f1",
            "variant_id": "f1_v0",
            "family_type": "linear",
            "answer": "1",
            "model_response": "Answer: 1",
            "correct": True,
            "extracted_answer": "1",
            "response_tokens": 5,
        }
    ]

    summary = summarize_eval_rows(rows)

    assert summary["accuracy"] == 1.0
    assert summary["by_family_type"]["linear"]["accuracy"] == 1.0
