from pathlib import Path

from auto_annotation_tool.gui import z3_preview_badges


ROOT = Path(__file__).resolve().parents[1]


def test_source_legend_order_matches_fullscreen_visual_groups():
    items = z3_preview_badges.get_preview_source_component_legend_items(None)
    assert [item["component"] for item in items] == [
        "yolo_box",
        "generated_box",
        "manual_box",
        "yolo_symbol",
        "ocr_symbol",
        "manual_sign",
        "gt_assisted",
    ]


def test_fullscreen_gap_uses_real_canvas_bboxes_not_only_text_estimates():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert 'canvas.bbox("preview_source_legend_left")' in source
    assert 'canvas.bbox("preview_source_legend_right")' in source
    assert 'canvas.move("preview_source_legend_left"' in source
    assert 'canvas.move("preview_source_legend_right"' in source
    assert "gap_width = max(124.0, button_width + 42.0)" in source
    assert "actual_left = float(left_bbox[2])" in source
    assert "actual_right = float(right_bbox[0])" in source


def test_actions_gap_is_real_empty_space_between_rendered_groups():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_badges.py"
    ).read_text(encoding="utf-8-sig")

    assert "_preview_source_legend_gap_bounds = (" in source
    assert "actual_left," in source
    assert "actual_right," in source
