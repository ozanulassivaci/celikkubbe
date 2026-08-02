from __future__ import annotations

from celikkubbe.core.strings import REASON_CODE_TR, UI_LABEL_TR, describe
from celikkubbe.core.types import ReasonCode


def test_every_reason_code_has_a_translation() -> None:
    for reason in ReasonCode:
        assert reason in REASON_CODE_TR
        assert isinstance(REASON_CODE_TR[reason], str)
        assert REASON_CODE_TR[reason]


def test_describe_returns_the_mapped_string() -> None:
    assert describe(ReasonCode.ESTOP_ACTIVE) == REASON_CODE_TR[ReasonCode.ESTOP_ACTIVE]


def test_every_ui_label_is_a_non_empty_string() -> None:
    for label in UI_LABEL_TR.values():
        assert isinstance(label, str)
        assert label
