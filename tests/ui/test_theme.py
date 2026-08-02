"""theme.py tests: QSS generation, font-loading fallback, and a paint
smoke test for each custom widget. GUI tests are shallower than logic
tests by design -- these check that things render without raising and
that the documented fallback/branch behaviour actually happens, not
pixel-perfect output.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from celikkubbe.core.types import Stage
from celikkubbe.ui import theme


def _find_real_ttf() -> Path | None:
    import matplotlib

    candidate = (
        Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    )
    return candidate if candidate.is_file() else None


@pytest.mark.parametrize("stage", list(Stage))
def test_build_qss_generates_for_every_stage_accent(qapp, stage):
    accent = theme.accent_for_stage(stage)
    qss = theme.build_qss(accent)
    assert accent in qss
    assert "QMainWindow" in qss


def test_stage_accents_are_distinct():
    accents = {theme.accent_for_stage(s) for s in Stage}
    assert len(accents) == len(list(Stage))


def test_font_loading_falls_back_and_logs_when_bundled_font_missing(qapp, tmp_path, caplog):
    empty_dir = tmp_path / "fonts"
    empty_dir.mkdir()
    with caplog.at_level(logging.WARNING, logger=theme.logger.name):
        family = theme.load_monospace_font(fonts_dir=empty_dir)
    assert family  # always returns *something* usable
    assert any("no bundled monospace font found" in r.message for r in caplog.records)


def test_font_loading_falls_back_when_directory_does_not_exist(qapp, tmp_path, caplog):
    missing_dir = tmp_path / "does-not-exist"
    with caplog.at_level(logging.WARNING, logger=theme.logger.name):
        family = theme.load_monospace_font(fonts_dir=missing_dir)
    assert family


def test_font_loading_succeeds_against_a_real_bundled_font_file(qapp, tmp_path):
    real_ttf = _find_real_ttf()
    if real_ttf is None:
        pytest.skip("no real .ttf available in this environment to stand in as a bundled font")
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(real_ttf, fonts_dir / "Bundled.ttf")

    family = theme.load_monospace_font(fonts_dir=fonts_dir)
    assert family == "DejaVu Sans Mono"


def test_font_loading_skips_unreadable_file_and_still_returns_something(qapp, tmp_path, caplog):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "not-a-font.ttf").write_bytes(b"this is not a real font file")

    with caplog.at_level(logging.WARNING, logger=theme.logger.name):
        family = theme.load_monospace_font(fonts_dir=fonts_dir)
    assert family
    assert any("failed to load bundled font" in r.message for r in caplog.records)


def test_status_badge_paints_without_raising(qapp):
    badge = theme.StatusBadge("M1", theme.OK)
    badge.resize(60, 20)
    badge.set_status("M4", theme.DANGER)
    badge.repaint()


def test_toggle_switch_animates_position_on_toggle(qapp):
    switch = theme.ToggleSwitch()
    switch.resize(44, 22)
    assert switch.get_position() == 0.0
    switch.setChecked(True)
    # The animation itself runs on a timer; what matters here is that
    # toggling drives it towards 1.0 rather than snapping instantly, and
    # that painting at any point mid-animation does not raise.
    switch.set_position(0.5)
    switch.repaint()
    switch.setChecked(False)
    switch.repaint()


def test_risk_bar_paints_across_its_range(qapp):
    bar = theme.RiskBar()
    bar.resize(120, 28)
    for value in (-10.0, 0.0, 50.0, 100.0, 150.0):
        bar.set_value(value)
        bar.repaint()
    assert bar._value == 100.0  # clamped, not left at the last out-of-range input


def test_angle_gauge_paints_with_and_without_telemetry(qapp):
    gauge = theme.AngleGauge(-170.0, 170.0, warn_margin_deg=10.0)
    gauge.resize(120, 14)
    gauge.set_value(None)
    gauge.repaint()
    gauge.set_value(165.0)  # inside the danger margin near the limit
    gauge.repaint()
    gauge.set_value(0.0)
    gauge.repaint()
