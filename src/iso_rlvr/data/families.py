from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
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


FAMILY_BUILDERS = [
    proportional_family,
    affine_family,
    linear_equation_family,
    modular_family,
    unit_conversion_family,
]


def make_family(rng: random.Random, family_id: str, variants: int) -> list[ProblemVariant]:
    builder = rng.choice(FAMILY_BUILDERS)
    return builder(rng, family_id, variants)

