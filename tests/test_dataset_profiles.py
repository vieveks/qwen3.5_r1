from fractions import Fraction
import random

import pytest

from iso_rlvr.data.families import (
    CALIBRATED_FAMILY_BUILDERS,
    CHALLENGE_FAMILY_BUILDERS,
    HARD_FAMILY_BUILDERS,
    make_family,
)
from iso_rlvr.rewards.answer import normalize_number


def test_harder_profile_uses_hard_family_types():
    rng = random.Random(7)
    family = make_family(rng, "fam_test", variants=4, profile="harder")
    hard_types = {builder.__name__.removesuffix("_family") for builder in HARD_FAMILY_BUILDERS}
    assert family[0].family_type in hard_types
    assert len(family) == 4


def test_hard_family_answers_are_numeric():
    rng = random.Random(13)
    for idx, builder in enumerate(HARD_FAMILY_BUILDERS + CHALLENGE_FAMILY_BUILDERS):
        family = builder(rng, f"fam_{idx}", variants=4)
        assert family
        for variant in family:
            assert normalize_number(variant.answer) is not None
            assert isinstance(normalize_number(variant.answer), Fraction)


def test_challenge_profile_uses_challenge_family_types():
    rng = random.Random(17)
    family = make_family(rng, "fam_test", variants=4, profile="challenge")
    challenge_types = {
        builder.__name__.removesuffix("_family") for builder in CHALLENGE_FAMILY_BUILDERS
    }
    assert family[0].family_type in challenge_types


def test_calibrated_profile_uses_calibration_family_types():
    rng = random.Random(23)
    family = make_family(rng, "fam_test", variants=4, profile="calibrated")
    calibrated_types = {
        builder.__name__.removesuffix("_family") for builder in CALIBRATED_FAMILY_BUILDERS
    }
    assert family[0].family_type in calibrated_types


def test_unknown_profile_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown dataset profile"):
        make_family(random.Random(1), "fam_bad", variants=4, profile="missing")
