from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import random


@dataclass(frozen=True)
class ProblemVariant:
    family_id: str
    variant_id: str
    family_type: str
    problem: str
    answer: str
    metadata: dict[str, int | str]


def _format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _signed(value: int | Fraction) -> str:
    return f"+ {_format_number(Fraction(value))}" if value >= 0 else f"- {_format_number(-Fraction(value))}"


def proportional_family(rng: random.Random, family_id: str, variants: int) -> list[ProblemVariant]:
    unit_price = rng.randint(2, 12)
    objects = rng.choice(
        [
            ("apples", "dollars"),
            ("notebooks", "dollars"),
            ("tickets", "dollars"),
            ("meters of fabric", "dollars"),
        ]
    )
    out = []
    for idx in range(variants):
        quantity_a = rng.randint(2, 12)
        quantity_b = rng.randint(3, 20)
        cost_a = quantity_a * unit_price
        answer = quantity_b * unit_price
        problem = (
            f"If {quantity_a} {objects[0]} cost {cost_a} {objects[1]}, "
            f"how much do {quantity_b} {objects[0]} cost?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="proportional",
                problem=problem,
                answer=str(answer),
                metadata={
                    "unit_price": unit_price,
                    "quantity_a": quantity_a,
                    "quantity_b": quantity_b,
                    "cost_a": cost_a,
                },
            )
        )
    return out


def affine_family(rng: random.Random, family_id: str, variants: int) -> list[ProblemVariant]:
    start = rng.randint(3, 30)
    rate = rng.randint(2, 9)
    out = []
    for idx in range(variants):
        days = rng.randint(3, 18)
        answer = start + rate * days
        problem = (
            f"A tank starts with {start} liters of water. It gains {rate} liters each day. "
            f"How many liters are in the tank after {days} days?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="affine",
                problem=problem,
                answer=str(answer),
                metadata={"start": start, "rate": rate, "days": days},
            )
        )
    return out


def linear_equation_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    solution = rng.randint(-12, 20)
    out = []
    for idx in range(variants):
        a = rng.choice([x for x in range(-9, 10) if x not in (0, 1, -1)])
        b = rng.randint(-30, 30)
        c = a * solution + b
        problem = f"Solve for x: {a}x + {b} = {c}."
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="linear_equation",
                problem=problem,
                answer=str(solution),
                metadata={"a": a, "b": b, "c": c, "solution": solution},
            )
        )
    return out


def modular_family(rng: random.Random, family_id: str, variants: int) -> list[ProblemVariant]:
    modulus = rng.randint(5, 17)
    residue = rng.randint(0, modulus - 1)
    out = []
    for idx in range(variants):
        multiplier = rng.randint(2, 12)
        offset = rng.randint(0, 50)
        value = multiplier * modulus + residue + offset * modulus
        problem = f"What is the remainder when {value} is divided by {modulus}?"
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="modular",
                problem=problem,
                answer=str(residue),
                metadata={"modulus": modulus, "residue": residue, "value": value},
            )
        )
    return out


def unit_conversion_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    conversions = [
        ("hours", "minutes", 60),
        ("kilometers", "meters", 1000),
        ("meters", "centimeters", 100),
        ("dollars", "cents", 100),
    ]
    src, dst, factor = rng.choice(conversions)
    out = []
    for idx in range(variants):
        amount = rng.randint(2, 25)
        answer = amount * factor
        problem = f"Convert {amount} {src} into {dst}."
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="unit_conversion",
                problem=problem,
                answer=str(answer),
                metadata={"source": src, "target": dst, "factor": factor, "amount": amount},
            )
        )
    return out


def rational_linear_equation_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    solution = Fraction(rng.choice([2, 3, 4, 5, 7, 8, 9, 11]), rng.choice([2, 3, 4, 5, 6]))
    if solution.denominator == 1:
        solution += Fraction(1, rng.choice([2, 3, 4, 5]))

    out = []
    for idx in range(variants):
        a = rng.choice([x for x in range(-9, 10) if x not in (0, 1, -1)])
        b = Fraction(rng.randint(-25, 25), rng.choice([2, 3, 4, 5]))
        c = a * solution + b
        problem = (
            f"Solve for x: {a}x {_signed(b)} = {_format_number(c)}. "
            "Give x as a reduced fraction if needed."
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="rational_linear_equation",
                problem=problem,
                answer=_format_number(solution),
                metadata={
                    "a": a,
                    "b": _format_number(b),
                    "c": _format_number(c),
                    "solution": _format_number(solution),
                },
            )
        )
    return out


def nested_linear_equation_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    solution = rng.randint(-15, 25)
    out = []
    for idx in range(variants):
        a = rng.choice([x for x in range(-8, 9) if x not in (0, 1, -1)])
        shift = rng.randint(-12, 12)
        b = rng.randint(-30, 30)
        c = a * (solution + shift) + b
        problem = f"Solve for x: {a}(x {_signed(shift)}) {_signed(b)} = {c}."
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="nested_linear_equation",
                problem=problem,
                answer=str(solution),
                metadata={"a": a, "shift": shift, "b": b, "c": c, "solution": solution},
            )
        )
    return out


def two_variable_system_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    x = rng.randint(-8, 12)
    y = rng.randint(-8, 12)
    target = x + y
    out = []
    for idx in range(variants):
        while True:
            a, b, c, d = [rng.choice([v for v in range(-6, 7) if v != 0]) for _ in range(4)]
            if a * d - b * c != 0:
                break
        e = a * x + b * y
        f = c * x + d * y
        problem = (
            f"If {a}x {_signed(b)}y = {e} and {c}x {_signed(d)}y = {f}, "
            "what is x + y?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="two_variable_system",
                problem=problem,
                answer=str(target),
                metadata={"x": x, "y": y, "target": target, "a": a, "b": b, "c": c, "d": d},
            )
        )
    return out


def quadratic_root_family(rng: random.Random, family_id: str, variants: int) -> list[ProblemVariant]:
    root_a = rng.randint(-9, 9)
    root_b = rng.randint(-9, 9)
    while root_b == root_a:
        root_b = rng.randint(-9, 9)
    smaller = min(root_a, root_b)

    out = []
    scales = [1, 2, 3, 4, 5, -1, -2, -3]
    selected_scales = rng.sample(scales, k=min(variants, len(scales)))
    while len(selected_scales) < variants:
        selected_scales.append(rng.choice(scales))

    for idx, scale in enumerate(selected_scales):
        b = -scale * (root_a + root_b)
        c = scale * root_a * root_b
        problem = (
            f"The equation {scale}x^2 {_signed(b)}x {_signed(c)} = 0 has two integer roots. "
            "What is the smaller root?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="quadratic_root",
                problem=problem,
                answer=str(smaller),
                metadata={"root_a": root_a, "root_b": root_b, "scale": scale, "smaller": smaller},
            )
        )
    return out


def rational_system_target_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    x = Fraction(rng.randint(-12, 12), rng.choice([2, 3, 4, 5]))
    y = Fraction(rng.randint(-12, 12), rng.choice([2, 3, 4, 5]))
    target_name, target = rng.choice(
        [
            ("x + y", x + y),
            ("x - y", x - y),
            ("2x + y", 2 * x + y),
            ("3y - x", 3 * y - x),
        ]
    )

    out = []
    for idx in range(variants):
        while True:
            a, b, c, d = [rng.choice([v for v in range(-9, 10) if v != 0]) for _ in range(4)]
            if abs(a * d - b * c) > 1:
                break
        e = a * x + b * y
        f = c * x + d * y
        problem = (
            f"If {a}x {_signed(b)}y = {_format_number(e)} and "
            f"{c}x {_signed(d)}y = {_format_number(f)}, what is {target_name}? "
            "Give the answer as a reduced fraction if needed."
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="rational_system_target",
                problem=problem,
                answer=_format_number(target),
                metadata={
                    "x": _format_number(x),
                    "y": _format_number(y),
                    "target_name": target_name,
                    "target": _format_number(target),
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                },
            )
        )
    return out


def chinese_remainder_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    answer = rng.randint(20, 180)
    modulus_pairs: list[tuple[int, int]] = []
    candidates = list(range(5, 23))
    rng.shuffle(candidates)
    for first in candidates:
        for second in candidates:
            if first >= second:
                continue
            if math.gcd(first, second) == 1 and first * second > answer:
                modulus_pairs.append((first, second))
    rng.shuffle(modulus_pairs)
    if len(modulus_pairs) < variants:
        return modular_family(rng, family_id, variants)

    out = []
    for idx, (mod_a, mod_b) in enumerate(modulus_pairs[:variants]):
        rem_a = answer % mod_a
        rem_b = answer % mod_b
        problem = (
            f"What is the least nonnegative integer x such that x has remainder {rem_a} "
            f"when divided by {mod_a}, and remainder {rem_b} when divided by {mod_b}?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="chinese_remainder",
                problem=problem,
                answer=str(answer),
                metadata={"answer": answer, "mod_a": mod_a, "mod_b": mod_b},
            )
        )
    return out


def missing_average_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    missing = rng.randint(12, 95)
    out = []
    for idx in range(variants):
        count = rng.randint(4, 7)
        known = [rng.randint(10, 95) for _ in range(count)]
        final_average = Fraction(sum(known) + missing, count + 1)
        known_text = ", ".join(str(value) for value in known)
        problem = (
            f"The numbers {known_text}, and one unknown number have an average of "
            f"{_format_number(final_average)}. What is the unknown number?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="missing_average",
                problem=problem,
                answer=str(missing),
                metadata={
                    "known": ",".join(str(value) for value in known),
                    "final_average": _format_number(final_average),
                    "missing": missing,
                },
            )
        )
    return out


def affine_composition_family(
    rng: random.Random, family_id: str, variants: int
) -> list[ProblemVariant]:
    solution = rng.randint(-12, 18)
    out = []
    for idx in range(variants):
        a = rng.choice([v for v in range(-7, 8) if v not in (0, 1, -1)])
        c = rng.choice([v for v in range(-7, 8) if v not in (0, 1, -1)])
        b = rng.randint(-20, 20)
        d = rng.randint(-20, 20)
        target = a * (c * solution + d) + b
        problem = (
            f"Let f(t) = {a}t {_signed(b)} and g(t) = {c}t {_signed(d)}. "
            f"If f(g(x)) = {target}, what is x?"
        )
        out.append(
            ProblemVariant(
                family_id=family_id,
                variant_id=f"{family_id}_v{idx}",
                family_type="affine_composition",
                problem=problem,
                answer=str(solution),
                metadata={"a": a, "b": b, "c": c, "d": d, "target": target, "solution": solution},
            )
        )
    return out


EASY_FAMILY_BUILDERS = [
    proportional_family,
    affine_family,
    linear_equation_family,
    modular_family,
    unit_conversion_family,
]

HARD_FAMILY_BUILDERS = [
    rational_linear_equation_family,
    nested_linear_equation_family,
    two_variable_system_family,
    quadratic_root_family,
]

CHALLENGE_FAMILY_BUILDERS = [
    rational_system_target_family,
    chinese_remainder_family,
    missing_average_family,
    affine_composition_family,
]

CALIBRATED_FAMILY_BUILDERS = [
    rational_linear_equation_family,
    rational_linear_equation_family,
    rational_linear_equation_family,
    rational_linear_equation_family,
    rational_linear_equation_family,
    missing_average_family,
    missing_average_family,
    missing_average_family,
    rational_system_target_family,
    chinese_remainder_family,
]

FAMILY_BUILDER_PROFILES = {
    "easy": EASY_FAMILY_BUILDERS,
    "harder": HARD_FAMILY_BUILDERS,
    "mixed": EASY_FAMILY_BUILDERS + HARD_FAMILY_BUILDERS,
    "challenge": CHALLENGE_FAMILY_BUILDERS,
    "calibrated": CALIBRATED_FAMILY_BUILDERS,
}


def make_family(
    rng: random.Random, family_id: str, variants: int, profile: str = "mixed"
) -> list[ProblemVariant]:
    try:
        builders = FAMILY_BUILDER_PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(FAMILY_BUILDER_PROFILES))
        raise ValueError(f"Unknown dataset profile {profile!r}; expected one of: {valid}") from exc
    builder = rng.choice(builders)
    return builder(rng, family_id, variants)

