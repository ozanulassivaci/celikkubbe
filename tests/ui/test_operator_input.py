from __future__ import annotations

from celikkubbe.ui.operator_input import OperatorInputBuilder


def test_build_defaults_to_nothing_held_and_no_target() -> None:
    builder = OperatorInputBuilder()
    result = builder.build()
    assert result.fire_requested is False
    assert result.arm_held is False
    assert result.manual_target_id is None


def test_one_source_holding_fire_is_enough() -> None:
    builder = OperatorInputBuilder()
    builder.set_fire("keyboard", True)
    assert builder.build().fire_requested is True


def test_releasing_one_source_does_not_clear_another_still_held() -> None:
    builder = OperatorInputBuilder()
    builder.set_fire("keyboard", True)
    builder.set_fire("gamepad", True)

    builder.set_fire("gamepad", False)

    assert builder.build().fire_requested is True


def test_all_sources_released_clears_fire() -> None:
    builder = OperatorInputBuilder()
    builder.set_fire("keyboard", True)
    builder.set_fire("gamepad", True)

    builder.set_fire("keyboard", False)
    builder.set_fire("gamepad", False)

    assert builder.build().fire_requested is False


def test_arm_sources_are_independent_of_fire_sources() -> None:
    builder = OperatorInputBuilder()
    builder.set_arm("gamepad", True)
    result = builder.build()
    assert result.arm_held is True
    assert result.fire_requested is False


def test_manual_target_persists_until_changed() -> None:
    builder = OperatorInputBuilder()
    builder.set_manual_target(7)
    assert builder.build().manual_target_id == 7
    builder.set_fire("keyboard", True)
    assert builder.build().manual_target_id == 7


def test_manual_target_can_be_cleared() -> None:
    builder = OperatorInputBuilder()
    builder.set_manual_target(7)
    builder.set_manual_target(None)
    assert builder.build().manual_target_id is None


def test_a_dropped_gamepad_forcing_its_own_sources_false_does_not_affect_keyboard() -> None:
    """The scenario request_estop's own docstring and the gamepad spec
    both care about: a disconnect must force *that source's* arm/fire to
    False without touching what another source is independently holding.
    """
    builder = OperatorInputBuilder()
    builder.set_fire("keyboard", True)
    builder.set_arm("keyboard", True)
    builder.set_fire("gamepad", True)
    builder.set_arm("gamepad", True)

    builder.set_fire("gamepad", False)
    builder.set_arm("gamepad", False)

    result = builder.build()
    assert result.fire_requested is True
    assert result.arm_held is True
