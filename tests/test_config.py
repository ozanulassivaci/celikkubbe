from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.types import TargetClass


def test_unknown_class_range_matches_expected_intersection() -> None:
    # F16 (10-15) has the highest lower bound; every rule shares 15 as the
    # upper bound.
    assert config.UNKNOWN_CLASS_RANGE == (10.0, 15.0)


def test_derive_unknown_class_range_is_the_narrowest_band_for_any_rule_table() -> None:
    mutated_rules = {
        TargetClass.F16: (2.0, 20.0),
        TargetClass.HELICOPTER: (5.0, 12.0),
        TargetClass.UAV: (0.0, 9.0),
    }
    assert config.derive_unknown_class_range(mutated_rules) == (5.0, 9.0)


def test_derive_unknown_class_range_tracks_range_rules_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "RANGE_RULES",
        {
            TargetClass.F16: (3.0, 8.0),
            TargetClass.UAV: (1.0, 6.0),
        },
    )
    assert config.derive_unknown_class_range(config.RANGE_RULES) == (3.0, 6.0)
