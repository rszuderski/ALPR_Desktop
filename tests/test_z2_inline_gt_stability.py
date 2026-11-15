import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_canvas_interaction
from auto_annotation_tool.gui import z2_plate_gt_inline
from auto_annotation_tool.gui import z2_preview_editor
from auto_annotation_tool.gui.zoomable_canvas import ZoomableCanvas


def test_resolver_never_returns_none_for_extreme_zoom():
    pos = z2_plate_gt_inline._resolve_non_overlapping_card_position(
        320, 180, 90, 60, (0, 0, 320, 180)
    )
    assert pos is not None
    x, y = pos
    assert 0 <= x <= 320
    assert 0 <= y <= 180


def test_light_overlay_uses_relocation_not_suspend():
    source = inspect.getsource(
        z2_plate_gt_inline.render_inline_plate_gt_editors
    )
    assert "if light_overlay:" in source
    assert "relocate_inline_plate_gt_editors" in source
    light = source.split("if light_overlay:", 1)[1].split(
        "if not gt_mode_enabled", 1
    )[0]
    assert "suspend_inline_plate_gt_editors" not in light


def test_preview_drag_uses_relocation_not_suspend():
    source = inspect.getsource(
        z2_preview_editor.on_zoomable_canvas_drag
    )
    assert "relocate_inline_plate_gt_editors" in source
    assert "suspend_inline_plate_gt_editors" not in source


def test_pan_motion_emits_post_pan_delegate():
    source = inspect.getsource(ZoomableCanvas._on_pan_motion)
    assert '"pan_applied"' in source


def test_pan_applied_delegate_relocates_gt():
    source = inspect.getsource(
        z2_canvas_interaction.on_zoomable_canvas_pan_applied
    )
    assert "relocate_inline_plate_gt_editors" in source


def test_relocate_moves_existing_card_without_recreating_it():
    shell = Mock()
    det = SimpleNamespace(
        polygon=[(40, 70), (140, 70), (140, 100), (40, 100)]
    )
    record = {
        "shell": shell,
        "det": det,
        "ann": SimpleNamespace(),
        "plate_idx": 0,
    }

    class Canvas:
        def delete(self, *_args, **_kwargs):
            return None
        def winfo_width(self):
            return 400
        def winfo_height(self):
            return 240
        def canvasx(self, value):
            return float(value)
        def canvasy(self, value):
            return float(value)
        def image_to_canvas_coords(self, x, y):
            return float(x), float(y)
        def create_line(self, *_args, **_kwargs):
            return 1
        def create_oval(self, *_args, **_kwargs):
            return 2

    canvas = Canvas()
    ann = SimpleNamespace()
    host = SimpleNamespace(
        preview_canvas=canvas,
        _plate_gt_inline_widgets={"plate-key": record},
        _plate_gt_inline_offsets={},
        _plate_gt_inline_enabled=True,
        _plate_gt_inline_mode_target="char",
        _campaign_graph_entry_context={},
        app=SimpleNamespace(palette={}),
    )
    host._get_preview_annotation = lambda: ann
    host._get_plate_detections = lambda _ann: [det]
    host._get_selected_plate_index_for_ann = lambda _ann: 0
    host._detection_polygon = lambda item: list(item.polygon)

    old_key = z2_plate_gt_inline._key_for
    z2_plate_gt_inline._key_for = lambda _det: "plate-key"
    try:
        z2_plate_gt_inline.relocate_inline_plate_gt_editors(
            host,
            canvas=canvas,
        )
    finally:
        z2_plate_gt_inline._key_for = old_key

    shell.place.assert_called()
    shell.lift.assert_called()
