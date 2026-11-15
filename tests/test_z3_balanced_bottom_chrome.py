from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bottom_actions_uses_same_sidebar_toggle_component():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_workspace_drawers.py"
    ).read_text(encoding="utf-8-sig")

    assert 'collapsed_text="Akcje"' in source
    assert 'expanded_text="Akcje"' in source
    assert 'orientation="horizontal_tab"' in source
    assert 'side="bottom"' in source
    assert 'button.set_collapsed(not bool(self.target[side]))' in source
    assert 'button.configure(text="▼ Ukryj akcje"' not in source


def test_horizontal_tab_is_visual_sibling_of_vertical_tabs():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "campaign_sidebar.py"
    ).read_text(encoding="utf-8-sig")

    assert 'elif self.orientation == "horizontal_tab":' in source
    assert 'height = 30' in source
    assert 'tags=("direction",)' in source
    assert 'tags=("label",)' in source


def test_badge_groups_are_centered_in_symmetric_lanes():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_badge_legend.py"
    ).read_text(encoding="utf-8-sig")

    assert "left_lane_left = margin" in source
    assert "left_lane_right = gap_left - inner_gap" in source
    assert "right_lane_left = gap_right + inner_gap" in source
    assert "right_lane_right = float(width) - margin" in source
    assert "(left_capacity - left_width) / 2.0" in source
    assert "(right_capacity - right_width) / 2.0" in source
