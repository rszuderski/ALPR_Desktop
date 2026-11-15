import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_plate_gt_inline


def test_plate_visibility_rejects_offscreen_plate():
    assert not z2_plate_gt_inline._plate_bbox_meaningfully_visible(
        400, 240, (450, 70, 550, 100)
    )


def test_plate_visibility_rejects_tiny_edge_sliver():
    assert not z2_plate_gt_inline._plate_bbox_meaningfully_visible(
        400, 240, (398, 70, 498, 100)
    )


def test_plate_visibility_accepts_partial_plate():
    assert z2_plate_gt_inline._plate_bbox_meaningfully_visible(
        400, 240, (350, 70, 450, 100)
    )


def test_plate_visibility_accepts_huge_zoomed_plate():
    assert z2_plate_gt_inline._plate_bbox_meaningfully_visible(
        400, 240, (-1000, -1000, 1000, 1000)
    )


def test_fast_relocate_hides_only_offscreen_plate_card():
    visible_shell = Mock()
    hidden_shell = Mock()

    visible_det = SimpleNamespace(
        polygon=[(40, 70), (140, 70), (140, 100), (40, 100)]
    )
    hidden_det = SimpleNamespace(
        polygon=[(500, 70), (600, 70), (600, 100), (500, 100)]
    )

    records = {
        "visible": {
            "shell": visible_shell,
            "det": visible_det,
            "ann": SimpleNamespace(),
            "plate_idx": 0,
        },
        "hidden": {
            "shell": hidden_shell,
            "det": hidden_det,
            "ann": SimpleNamespace(),
            "plate_idx": 1,
        },
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
        _plate_gt_inline_widgets=records,
        _plate_gt_inline_offsets={},
        _plate_gt_inline_enabled=True,
        app=SimpleNamespace(palette={}),
    )
    host._get_preview_annotation = lambda: ann
    host._get_plate_detections = lambda _ann: [visible_det, hidden_det]
    host._detection_polygon = lambda det: list(det.polygon)

    old_key = z2_plate_gt_inline._key_for
    def key_for(det):
        return "visible" if det is visible_det else "hidden"
    z2_plate_gt_inline._key_for = key_for
    try:
        z2_plate_gt_inline.relocate_inline_plate_gt_editors(
            host,
            canvas=canvas,
        )
    finally:
        z2_plate_gt_inline._key_for = old_key

    visible_shell.place.assert_called()
    visible_shell.place_forget.assert_not_called()

    hidden_shell.place_forget.assert_called()
    hidden_shell.place.assert_not_called()


def test_full_renderer_checks_plate_visibility_before_clamp():
    source = inspect.getsource(
        z2_plate_gt_inline.render_inline_plate_gt_editors
    )
    assert "_plate_bbox_meaningfully_visible" in source
    assert "shell.place_forget()" in source


def test_no_zoom_viewport_freeze_policy_present():
    source = inspect.getsource(
        z2_plate_gt_inline.render_inline_plate_gt_editors
    )
    assert "zoom_animation_active" not in source
    assert "_zoom_animation_target_zoom" not in source
