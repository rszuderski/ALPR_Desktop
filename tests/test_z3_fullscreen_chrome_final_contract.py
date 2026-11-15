from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fullscreen_source_legend_split_does_not_depend_on_button_render_order():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert 'or bool(getattr(drawers, "detached", False))' in source
    assert "gap_width = max(112.0, button_width + 32.0)" in source
    assert "abs(left_width - right_width)" in source
    assert "_preview_source_legend_gap_bounds = (" in source


def test_fullscreen_tools_dock_height_accounts_for_all_visible_rows():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_dock_ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "visible_row_count = 0" in source
    assert "expected_full_height = 34 + (visible_row_count * 29)" in source
    assert "height = max(height, expected_full_height)" in source
