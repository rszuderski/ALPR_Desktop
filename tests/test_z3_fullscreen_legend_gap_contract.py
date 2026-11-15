from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_side_tabs_are_slightly_lower_in_fullscreen():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_workspace_drawers.py"
    ).read_text(encoding="utf-8-sig")

    assert source.count("rely=.60") >= 2
    assert "rely=.52" not in source


def test_bottom_actions_use_reserved_source_legend_gap():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_workspace_drawers.py"
    ).read_text(encoding="utf-8-sig")

    assert "_preview_source_legend_gap_bounds" in source
    assert "gap_center = (left + right) / 2.0" in source


def test_source_legend_builds_balanced_center_gap_in_fullscreen():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert "gap_width = max(96.0, button_width + 28.0)" in source
    assert "for split_index in range(1, len(prepared)):" in source
    assert "balance = abs(left_width - right_width)" in source
    assert "host._preview_source_legend_gap_bounds = (" in source
