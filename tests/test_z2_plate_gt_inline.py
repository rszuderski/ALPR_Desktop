from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gui import z2_plate_gt_inline as inline
from auto_annotation_tool.plate_ground_truth import (
    GROUND_TRUTH_SOURCE_ATTR,
    GROUND_TRUTH_TEXT_ATTR,
)


def plate(gt="", confidence=0.91):
    attributes = {}
    if gt:
        attributes[GROUND_TRUTH_TEXT_ATTR] = gt
        attributes[GROUND_TRUTH_SOURCE_ATTR] = "manual_z2"
    return Detection(
        label="plate",
        confidence=confidence,
        bbox=(0.0, 0.0, 10.0, 5.0),
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        attributes=attributes,
    )


def host_for(save_ok=True):
    host = SimpleNamespace()
    host._mark_preview_image_dirty = Mock()
    host._save_preview_edits = Mock(return_value=save_ok)
    host._update_preview_edit_status = Mock()
    host._refresh_step2_action_states = Mock()
    return host


def test_live_input_is_always_uppercase():
    assert inline.uppercase_registration_input("wi905pw") == "WI905PW"
    assert inline.uppercase_registration_input("kr 12-ab") == "KR 12-AB"


def test_count_plate_gt_counts_only_nonempty_gt():
    assert inline.count_plate_gt(
        [plate("AA111"), plate(""), plate("BB222")]
    ) == 2


def test_card_position_is_pushed_outside_plate_bbox():
    plate_bbox = (100.0, 100.0, 260.0, 160.0)
    position = inline._resolve_non_overlapping_card_position(
        640,
        480,
        120,
        110,
        plate_bbox,
    )
    assert position is not None
    x, y = position
    assert not inline._card_overlaps_plate(
        x,
        y,
        plate_bbox,
    )


def test_card_position_keeps_safe_requested_position():
    plate_bbox = (200.0, 200.0, 320.0, 250.0)
    position = inline._resolve_non_overlapping_card_position(
        640,
        480,
        40,
        40,
        plate_bbox,
    )
    assert position == (40, 40)


def test_char_route_defaults_gt_mode_on(monkeypatch):
    monkeypatch.setattr(
        inline,
        "CAMPAIGN",
        SimpleNamespace(get_iteration_target=lambda: "char"),
    )
    host = SimpleNamespace(_is_free_mode_session_context=lambda: False)
    assert inline.gt_mode_enabled(host) is True
    assert inline.gt_required_for_current_route(host) is False


def test_plate_route_keeps_gt_available_but_optional(monkeypatch):
    monkeypatch.setattr(
        inline,
        "CAMPAIGN",
        SimpleNamespace(get_iteration_target=lambda: "plate"),
    )
    host = SimpleNamespace(_is_free_mode_session_context=lambda: False)
    assert inline.gt_mode_enabled(host) is True
    assert inline.gt_required_for_current_route(host) is False


def test_inline_save_normalizes_and_persists_gt():
    host = host_for()
    ann = SimpleNamespace(filename="image.jpg")
    det = plate()

    ok, normalized = inline.save_inline_plate_gt_value(
        host,
        ann,
        det,
        " wi 905-pw ",
    )

    assert ok is True
    assert normalized == "WI905PW"
    assert det.attributes[GROUND_TRUTH_TEXT_ATTR] == "WI905PW"
    assert det.attributes[GROUND_TRUTH_SOURCE_ATTR] == "manual_z2"
    assert det.attributes["plate_annotation_id"].startswith("plate-ann-")
    host._save_preview_edits.assert_called_once()


class _ViewportCanvas:
    def winfo_width(self):
        return 800

    def winfo_height(self):
        return 600

    def canvasx(self, value):
        return float(value) + 120.0

    def canvasy(self, value):
        return float(value) + 80.0


def test_canvas_viewport_bounds_include_scroll_origin():
    assert inline.canvas_viewport_bounds(_ViewportCanvas()) == (
        120.0,
        80.0,
        920.0,
        680.0,
    )

def test_inline_xml_success_does_not_rollback_when_pack_sync_is_queued(monkeypatch):
    host = host_for()
    ann = SimpleNamespace(filename="image.jpg")
    det = plate()

    monkeypatch.setattr(
        inline.z2_gt_pack_runtime,
        "sync_plate_gt_after_xml_save",
        Mock(
            return_value={
                "ok": False,
                "enabled": True,
                "queued": True,
            }
        ),
    )

    ok, normalized = inline.save_inline_plate_gt_value(
        host,
        ann,
        det,
        "ABC123",
    )

    assert ok is True
    assert normalized == "ABC123"
    assert det.attributes[GROUND_TRUTH_TEXT_ATTR] == "ABC123"
