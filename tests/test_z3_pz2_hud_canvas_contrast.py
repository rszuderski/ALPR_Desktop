from pathlib import Path

from auto_annotation_tool.gui.z3_inline_hud import (
    _HUD_CANVAS_BG,
    _HUD_LABEL_COLOR,
    _HUD_NEUTRAL_VALUE_COLOR,
    _hud_contrast_ratio,
    _hud_parse_hex_color,
    ensure_inline_hud_canvas_contrast,
)


def _ratio(fg):
    return _hud_contrast_ratio(
        _hud_parse_hex_color(fg),
        _hud_parse_hex_color(_HUD_CANVAS_BG),
    )


def test_label_color_is_readable_on_dark_canvas():
    resolved = ensure_inline_hud_canvas_contrast(
        _HUD_LABEL_COLOR,
        fallback=_HUD_LABEL_COLOR,
        min_ratio=5.0,
    )
    assert _ratio(resolved) >= 5.0


def test_neutral_value_is_brighter_than_label_and_readable():
    label = ensure_inline_hud_canvas_contrast(
        _HUD_LABEL_COLOR,
        fallback=_HUD_LABEL_COLOR,
        min_ratio=5.0,
    )
    value = ensure_inline_hud_canvas_contrast(
        _HUD_NEUTRAL_VALUE_COLOR,
        fallback=_HUD_NEUTRAL_VALUE_COLOR,
        min_ratio=5.3,
    )
    assert _ratio(value) >= 5.3
    assert value.lower() != label.lower()


def test_dark_semantic_green_is_brightened_without_losing_green_hue():
    resolved = ensure_inline_hud_canvas_contrast("#235b2a", min_ratio=5.3)
    r, g, b = _hud_parse_hex_color(resolved)
    assert _ratio(resolved) >= 5.3
    assert g >= r
    assert g >= b


def test_dark_semantic_red_is_brightened_without_losing_red_hue():
    resolved = ensure_inline_hud_canvas_contrast("#7a2424", min_ratio=5.3)
    r, g, b = _hud_parse_hex_color(resolved)
    assert _ratio(resolved) >= 5.3
    assert r >= g
    assert r >= b


def test_invalid_palette_color_falls_back_to_readable_neutral():
    resolved = ensure_inline_hud_canvas_contrast(
        "not-a-color",
        fallback=_HUD_NEUTRAL_VALUE_COLOR,
        min_ratio=5.3,
    )
    assert _ratio(resolved) >= 5.3


def test_renderer_applies_contrast_to_every_hud_value():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_inline_hud.py"
    ).read_text(encoding="utf-8-sig")
    assert "requested_value_color =" in source
    assert "ensure_inline_hud_canvas_contrast(" in source
    assert "min_ratio=5.3" in source
