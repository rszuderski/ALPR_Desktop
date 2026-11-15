from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_legend_split_is_driven_by_visible_actions_tab():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert 'str(bottom_button.winfo_manager()) == "place"' in source
    assert "if actions_tab_visible and len(prepared) >= 7:" in source
    assert "left_group = prepared[:4]" in source
    assert "right_group = prepared[4:7]" in source


def test_source_legend_reserves_explicit_actions_gap():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert "gap_width = max(132.0, button_width + 50.0)" in source
    assert 'host._preview_source_legend_layout_mode = "actions_gap"' in source
    assert "host._preview_source_legend_gap_bounds = (" in source


def test_source_legend_does_not_collapse_gap_on_right_overflow():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert "left_start -= overflow" in source
    assert "right_start -= overflow" in source
    assert "gap_left -= overflow" in source
    assert "gap_right -= overflow" in source
