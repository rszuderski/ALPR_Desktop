from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_character_badge_legend_owns_bottom_provenance_strip():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_detection_tab_ui.py"
    ).read_text(encoding="utf-8-sig")
    assert "self.preview_badge_legend = CharacterBadgeLegend(" in source
    assert "self.preview_badge_legend.grid(row=1, column=0, sticky=\"ew\")" in source


def test_fullscreen_badge_legend_reserves_real_actions_tab_lane():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_badge_legend.py"
    ).read_text(encoding="utf-8-sig")

    assert "def _actions_gap_bounds(self, width):" in source
    assert 'drawers.buttons.get("bottom")' in source
    assert 'str(button.winfo_manager() or "") != "place"' in source
    assert "button.winfo_rootx()" in source
    assert "self.winfo_rootx()" in source
    assert "left_items = prepared[:4]" in source
    assert "right_items = prepared[4:]" in source


def test_fullscreen_badge_legend_keeps_expected_group_order():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_badge_legend.py"
    ).read_text(encoding="utf-8-sig")

    assert '"yolo_box", "generated_box", "manual_box",' in source
    assert '"yolo_symbol", "ocr_symbol", "manual_sign", "gt_assisted",' in source
    assert "YB / GB / MB / YS | [ Actions tab ] | OS / MS / GT" in source


def test_workspace_refreshes_badge_legend_after_drawer_animation():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_workspace_drawers.py"
    ).read_text(encoding="utf-8-sig")

    assert 'legend = getattr(self.owner, "preview_badge_legend", None)' in source
    assert 'refresh = getattr(legend, "refresh", None)' in source
    assert "self.host.after(80, refresh)" in source
    assert "_debug_dump_bottom_canvas_items" not in source
