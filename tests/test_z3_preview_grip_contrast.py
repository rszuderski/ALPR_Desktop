from types import SimpleNamespace

from auto_annotation_tool.gui.z3_preview_ui import (
    _get_preview_character_edit_grip_style,
)


def test_corner_grip_style_is_high_contrast_on_light_plate():
    host = SimpleNamespace(
        app=SimpleNamespace(
            palette={
                "panel": "#f5f5f5",
                "warning": "#7c651e",
                "info": "#9bc78a",
                "accent_alt": "#14b8a6",
            }
        ),
        _preview_char_geometry_inherit_down=False,
    )

    style = _get_preview_character_edit_grip_style(host)

    assert style["idle_outline"] == "#101820"
    assert style["idle_fill"] == "#ffd166"
    assert style["active_outline"] == "#ff8a00"
    assert style["active_fill"] == "#101820"
    assert style["idle_width"] >= 2
    assert style["active_width"] >= 3


def test_group_mode_grips_keep_same_high_contrast_contract():
    host = SimpleNamespace(
        app=SimpleNamespace(
            palette={
                "success": "#22c55e",
                "accent": "#22c55e",
            }
        ),
        _preview_char_geometry_inherit_down=True,
    )

    style = _get_preview_character_edit_grip_style(host)

    assert style["idle_outline"] == "#101820"
    assert style["idle_fill"] == "#d9ff66"
    assert style["active_fill"] == "#101820"
    assert style["idle_width"] >= 2
    assert style["active_width"] >= 3
