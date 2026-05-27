from __future__ import annotations

from dataclasses import dataclass
import re


NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:/\d+)?(?:\.\d+)?"
LATEX_FRAC_PATTERN = r"\\frac\{[-+]?\d+\}\{\d+\}"
VALUE_PATTERN = rf"(?:{NUMBER_PATTERN}|{LATEX_FRAC_PATTERN})"

INDEXED_PATTERNS = [
    re.compile(
        rf"\bAnswer\s+(\d+)\s*(?::|=|\bis\b)\s*(?:\\boxed\{{)?({VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bProblem\s+(\d+)\s*(?::|=|\bis\b)\s*(?:\\boxed\{{)?({VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(rf"(?m)^\s*(\d+)\s*[\).:-]\s*(?:\\boxed\{{)?({VALUE_PATTERN})"),
]

BOXED_PATTERN = re.compile(rf"\\boxed\{{({VALUE_PATTERN})\}}")
PROBLEM_HEADING_PATTERN = re.compile(r"\bProblem\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PackedAnswerParse:
    answers: list[str | None]
    missing_indices: list[int]
    extra_answers: list[str]
    mode: str

    @property
    def complete(self) -> bool:
        return not self.missing_indices and not self.extra_answers


def _empty_parse(expected_count: int, mode: str) -> PackedAnswerParse:
    return PackedAnswerParse(
        answers=[None] * expected_count,
        missing_indices=list(range(1, expected_count + 1)),
        extra_answers=[],
        mode=mode,
    )


def _parse_indexed_answers(text: str, expected_count: int) -> PackedAnswerParse | None:
    found: list[tuple[int, str]] = []
    for pattern in INDEXED_PATTERNS:
        found = [
            (int(match.group(1)), _clean_value(match.group(2)))
            for match in pattern.finditer(text)
        ]
        if found:
            break

    if not found:
        found = _parse_problem_sections(text)

    if not found:
        return None

    answers: list[str | None] = [None] * expected_count
    extra_answers: list[str] = []
    for idx, value in found:
        if 1 <= idx <= expected_count and answers[idx - 1] is None:
            answers[idx - 1] = value
        else:
            extra_answers.append(value)

    return PackedAnswerParse(
        answers=answers,
        missing_indices=[idx for idx, value in enumerate(answers, start=1) if value is None],
        extra_answers=extra_answers,
        mode="indexed",
    )


def _parse_boxed_answers(text: str, expected_count: int) -> PackedAnswerParse | None:
    found = [_clean_value(match.group(1)) for match in BOXED_PATTERN.finditer(text)]
    if not found:
        return None

    answers: list[str | None] = list(found[:expected_count])
    answers.extend([None] * (expected_count - len(answers)))
    return PackedAnswerParse(
        answers=answers,
        missing_indices=[idx for idx, value in enumerate(answers, start=1) if value is None],
        extra_answers=found[expected_count:],
        mode="boxed",
    )


def parse_packed_answers(text: str, expected_count: int) -> PackedAnswerParse:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive.")

    indexed = _parse_indexed_answers(text, expected_count)
    if indexed is not None:
        return indexed

    boxed = _parse_boxed_answers(text, expected_count)
    if boxed is not None:
        return boxed

    return _empty_parse(expected_count, mode="missing")


def _parse_problem_sections(text: str) -> list[tuple[int, str]]:
    headings = list(PROBLEM_HEADING_PATTERN.finditer(text))
    found = []
    for pos, heading in enumerate(headings):
        start = heading.end()
        end = headings[pos + 1].start() if pos + 1 < len(headings) else len(text)
        section = text[start:end]
        boxed_values = [_clean_value(match.group(1)) for match in BOXED_PATTERN.finditer(section)]
        if boxed_values:
            found.append((int(heading.group(1)), boxed_values[-1]))
    return found


def _clean_value(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{(\d+)\}", value)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return value
