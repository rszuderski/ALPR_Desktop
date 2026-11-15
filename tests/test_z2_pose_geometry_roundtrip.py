"""Geometry provenance remains independent of preview and explicit edits."""
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
from auto_annotation_tool.gui import z2_preview_state as state
from auto_annotation_tool.gui import z2_preview_workflow as workflow
from auto_annotation_tool.gui import z2_preview_editor as editor
from auto_annotation_tool.gui import z2_canvas_interaction as interaction
from auto_annotation_tool.gui import z2_preview_autosave as autosave
from auto_annotation_tool.pose_geometry import (
    attach_pose_geometry, hydrate_stored_geometry, geometry_warning,
    geometry_display_polygon, finalize_manual_geometry, set_manual_display_point,
    GEOMETRY_XML_ATTR,
)


RAW = [(0., 0.), (40., 0.), (50., 20.), (60., 130.)]
BOWTIE = [RAW[0], RAW[1], RAW[3], RAW[2]]


def parser():
    return SimpleNamespace(_bbox_from_polygon=state._bbox_from_polygon,
                           _keypoints_from_polygon=state._keypoints_from_polygon)


def test_legacy_xml_load_preserves_original_order_without_fabricating_raw_pose(tmp_path):
    path = tmp_path / "legacy.xml"
    path.write_text('<annotations><image name="one.jpg" width="200" height="200">'
                    '<polygon label="plate" source="auto" points="0,0;40,0;60,130;50,20"/>'
                    '</image></annotations>', encoding="utf-8")
    original = path.read_bytes()
    detection = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert detection.polygon == BOWTIE
    assert path.read_bytes() == original
    assert detection.keypoints is None
    assert detection.geometry["geometry_validity"]["status"] == "SELF_INTERSECTING"
    assert detection.geometry["raw_pose_keypoints"] is None
    assert geometry_display_polygon(detection) == RAW
    assert geometry_warning(detection)
    detection.attributes["ground_truth_text"] = "WX12345"
    destination = tmp_path / "text_edit.xml"
    assert CVATExporter().export([ImageAnnotation("one.jpg", 200, 200, [detection])], destination)
    assert ET.parse(destination).getroot().find("image/polygon").get("points") == "0,0;40,0;60,130;50,20"
    reloaded = workflow._parse_cvat_preview_annotations(parser(), destination)[0].detections[0]
    assert reloaded.polygon == BOWTIE
    assert reloaded.geometry["raw_pose_keypoints"] is None
    assert geometry_warning(reloaded)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_raw_pose_permutation_confidence_and_bbox_survive_xml_manual_and_undo(tmp_path, dimensions):
    raw = [[x + 10.123456789, y + 7.123456789] + ([.25 + i * .13] if dimensions == 3 else [])
           for i, (x, y) in enumerate(BOWTIE)]
    det = Detection("plate", .83, (5, 6, 95, 150),
                    keypoints=[(p[0], p[1], p[2] if dimensions == 3 else 1.) for p in raw])
    attach_pose_geometry(det, raw, image_size=(250, 250))
    assert det.geometry["reorder_permutation"] == [0, 1, 3, 2]
    assert det.bbox == (5, 6, 95, 150)
    original = copy.deepcopy(det)
    path = tmp_path / "pose.xml"
    assert CVATExporter().export([ImageAnnotation("one.jpg", 250, 250, [det])], path)
    restored = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert restored.raw_pose_keypoints == raw
    assert restored.bbox == det.bbox
    assert restored.polygon == det.polygon
    assert restored.display_polygon == det.display_polygon
    assert restored.keypoints == det.keypoints
    snapshot = state._clone_preview_history_detection(restored)
    polygon = geometry_display_polygon(restored)
    polygon[1] = (polygon[1][0] + 2, polygon[1][1])
    set_manual_display_point(restored, polygon)
    finalize_manual_geometry(restored, image_size=(250, 250))
    assert restored.raw_pose_keypoints == raw
    assert restored.keypoints == original.keypoints
    assert snapshot.raw_pose_keypoints == raw
    assert snapshot.polygon == original.polygon
    assert not snapshot.geometry["manual_edit"]
    assert restored.geometry["manual_edit"]
    assert restored.attributes.get("pose_corner_identity") != "confirmed"
    assert CVATExporter().export([ImageAnnotation("one.jpg", 250, 250, [restored])], path)
    after = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert after.raw_pose_keypoints == raw
    assert after.polygon == restored.polygon
    assert after.geometry["manual_edit"]
    assert len(ET.parse(path).findall(f".//attribute[@name='{GEOMETRY_XML_ATTR}']")) == 1


def test_roi_translation_keeps_dimensions_and_confidence():
    raw = [[x, y, .2 + i*.15] for i, (x, y) in enumerate(RAW)]
    det = Detection("plate", .8, (100, 200, 170, 340))
    attach_pose_geometry(det, raw, offset=(100, 200), image_size=(500, 500))
    assert det.geometry["raw_pose_roi_keypoints"] == raw
    assert det.raw_pose_keypoints == [[x + 100, y + 200, conf] for x, y, conf in raw]
    assert det.bbox == (100, 200, 170, 340)


@pytest.mark.parametrize("raw", [None, [[float("nan"), 0], [40, 0], [50, 20], [60, 130]],
                                  [[0, 0], [0, 0], [50, 20], [60, 130]]])
def test_invalid_pose_has_explicit_fallback_and_portable_metadata(tmp_path, raw):
    det = Detection("plate", .9, (0, 0, 70, 140))
    attach_pose_geometry(det, raw, image_size=(200, 200))
    assert det.geometry["bbox_fallback"]
    assert geometry_warning(det)
    path = tmp_path / "fallback.xml"
    assert CVATExporter().export([ImageAnnotation("one.jpg", 200, 200, [det])], path)
    restored = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert restored.geometry["bbox_fallback"]
    encoded = ET.parse(path).find(f".//attribute[@name='{GEOMETRY_XML_ATTR}']").text
    json.loads(encoded, parse_constant=lambda v: pytest.fail(f"Nonstandard JSON: {v}"))


def drag_owner(*, moved=True, sequence=False):
    det = Detection("plate", .9, (0, 0, 70, 140), keypoints=[(x, y, .7) for x, y in RAW])
    attach_pose_geometry(det, [[x, y, .7] for x, y in RAW], image_size=(200, 200))
    other = Detection("plate", .7, (90, 20, 150, 150), polygon=[(90+x, 20+y) for x, y in BOWTIE])
    ann = ImageAnnotation("one.jpg", 200, 200, [det, other])
    host = SimpleNamespace(
        _preview_drag_state={"plate_idx": 0, "was_moved": moved},
        _preview_corner_drag_modifier_down=sequence, _preview_pending_vertex_hit=None,
        _preview_dirty_images={"another.jpg"}, current_preview_index=0,
        _get_preview_annotation=lambda: ann, _get_plate_detections=lambda a: a.detections,
        _detection_polygon=state._detection_polygon, preview_canvas=Mock(),
    )
    host._preview_modifier_active = lambda: host._preview_corner_drag_modifier_down
    for name in ["_cancel_preview_drag_refresh", "_refresh_preview_canvas", "_update_preview_edit_status",
                 "_mark_preview_image_dirty", "_refresh_preview_list_row_for_actual_index",
                 "_cancel_preview_post_interaction_refresh", "_refresh_preview_canvas_light",
                 "_refresh_preview_canvas_interactive", "_schedule_preview_autosave",
                 "_sync_preview_canvas_cursor", "_invalidate_preview_runtime_caches", "_push_preview_debug_event"]:
        setattr(host, name, Mock())
    return host, ann


def test_no_motion_release_does_not_reorder_or_mark_manual():
    host, ann = drag_owner(moved=False)
    before = copy.deepcopy(ann)
    editor._finish_preview_vertex_drag(host, mark_dirty=False)
    assert ann == before
    host._mark_preview_image_dirty.assert_not_called()


def test_w_release_only_finalizes_touched_detection_and_preserves_raw():
    host, ann = drag_owner(sequence=True)
    untouched = copy.deepcopy(ann.detections[1])
    raw = copy.deepcopy(ann.detections[0].raw_pose_keypoints)
    editor._finish_preview_vertex_drag(host, mark_dirty=True)
    assert ann.detections[0].geometry["manual_pending"]
    interaction._set_preview_corner_drag_modifier(host, False)
    assert not ann.detections[0].geometry["manual_pending"]
    assert ann.detections[0].raw_pose_keypoints == raw
    assert ann.detections[1] == untouched


def test_w_release_with_other_image_dirty_does_not_touch_current_geometry():
    host, ann = drag_owner(sequence=True)
    host._preview_drag_state = None
    before = copy.deepcopy(ann)
    interaction._set_preview_corner_drag_modifier(host, False)
    assert ann == before
    host._mark_preview_image_dirty.assert_not_called()


def test_pending_background_save_cannot_clear_w_finalization(tmp_path, monkeypatch):
    host, ann = drag_owner(sequence=True)
    ann.detections = ann.detections[:1]
    editor._finish_preview_vertex_drag(host, mark_dirty=True)
    path = tmp_path / "pending.xml"
    assert CVATExporter().export([ann], path)
    entered, release, posted = threading.Event(), threading.Event(), threading.Event()
    callbacks = []
    original_update = autosave.update_saved_images

    def blocked_update(*args):
        entered.set()
        assert release.wait(5)
        original_update(*args)

    monkeypatch.setattr(autosave, "update_saved_images", blocked_update)
    host.current_annotations = [ann]
    host.current_input_dir = tmp_path
    host._preview_image_edit_versions = {ann.filename: 1}
    host._preview_dirty_images = {ann.filename}
    host._preview_background_save = None
    host._get_current_annotation_xml_path = lambda: path
    host._post_to_ui = lambda callback: (callbacks.append(callback), posted.set())
    host._load_annotation_run_manifest = lambda run: {}
    host._collect_preview_resume_manifest_fields = lambda: {}
    for name in ["_update_annotation_run_manifest", "_remember_campaign_manual_plate_source",
                 "_queue_free_mode_session_save", "_update_preview_toolbar_state"]:
        setattr(host, name, Mock())
    host._mark_preview_image_dirty = lambda a, **kw: state._mark_preview_image_dirty(host, a, **kw)
    assert autosave.start_preview_autosave(host)
    job = host._preview_background_save
    try:
        assert entered.wait(2)
        assert job["annotations"][0].detections[0].geometry["manual_pending"]
        interaction._set_preview_corner_drag_modifier(host, False)
        assert not ann.detections[0].geometry["manual_pending"]
        assert host._preview_image_edit_versions[ann.filename] == 2
    finally:
        release.set()
    assert posted.wait(5)
    callbacks.pop(0)()
    assert host._preview_dirty_images == {ann.filename}
    host._schedule_preview_autosave.assert_called()


def test_w_release_during_second_drag_finalizes_previous_touched_plate():
    host, ann = drag_owner(sequence=True)
    editor._finish_preview_vertex_drag(host, mark_dirty=True)
    assert ann.detections[0].geometry["manual_pending"]
    host._preview_drag_state = {"plate_idx": 1, "was_moved": True}
    host._finish_preview_vertex_drag = lambda **kw: editor._finish_preview_vertex_drag(host, **kw)
    interaction._set_preview_corner_drag_modifier(host, False)
    assert not ann.detections[0].geometry["manual_pending"]
    assert not ann.detections[1].geometry["manual_pending"]
    assert not host._preview_geometry_touched.get(ann.filename)


def test_w_release_after_image_change_finalizes_only_prior_touched_detection():
    host, first = drag_owner(sequence=True)
    editor._finish_preview_vertex_drag(host, mark_dirty=True)
    second = ImageAnnotation("second.jpg", 200, 200, [copy.deepcopy(first.detections[1])])
    original_second = copy.deepcopy(second)
    host.current_annotations = [first, second]
    host._get_preview_annotation = lambda: second
    interaction._set_preview_corner_drag_modifier(host, False)
    assert not first.detections[0].geometry["manual_pending"]
    assert second == original_second
    assert not host._preview_geometry_touched


def test_nonfinite_legacy_points_never_reach_tk_coordinates(tmp_path):
    path = tmp_path / "nonfinite.xml"
    path.write_text('<annotations><image name="one.jpg" width="200" height="200">'
                    '<polygon label="plate" source="auto" points="nan,0;40,0;60,130;50,20"/>'
                    '</image></annotations>', encoding="utf-8")
    det = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert det.geometry["geometry_validity"]["status"] == "DEGENERATE"
    assert geometry_display_polygon(det) == []
    assert geometry_warning(det)


@pytest.mark.parametrize("held_w", [False, True])
def test_history_restore_rebinds_pending_w_geometry_to_cloned_detections(held_w):
    host, ann = drag_owner(sequence=True)
    editor._finish_preview_vertex_drag(host, mark_dirty=True)
    snapshot = {"annotation": state._clone_preview_history_detection(ann.detections[0])}
    pending_ann = ImageAnnotation(ann.filename, ann.width, ann.height, [snapshot["annotation"]])
    snapshot = {"annotation": pending_ann, "selected_plate_idx": 0, "selected_vehicle_idx": None}
    host.current_annotations = [ann]
    host._preview_corner_drag_modifier_down = held_w
    host._get_preview_annotation = lambda: host.current_annotations[0]
    host._get_preview_history_image_key = lambda a=None: (a or host.current_annotations[0]).filename
    host._clone_preview_history_annotation = lambda a: copy.deepcopy(a)
    for name in ["_set_selected_plate_index_for_ann", "_set_selected_vehicle_index_for_ann",
                 "_update_preview_toolbar_state", "_remember_annotation_run_resume_state",
                 "_queue_free_mode_session_save"]:
        setattr(host, name, Mock())
    host._save_preview_edits = Mock(return_value=True)
    assert state._restore_preview_annotation_history_snapshot(host, snapshot, action_label="Undo/Redo")
    restored = host.current_annotations[0].detections[0]
    assert id(restored) != id(ann.detections[0])
    if held_w:
        assert restored.geometry["manual_pending"]
        interaction._set_preview_corner_drag_modifier(host, False)
    assert not restored.geometry["manual_pending"]
    assert restored.raw_pose_keypoints == ann.detections[0].raw_pose_keypoints
    assert not host._preview_geometry_touched


def test_legacy_bbox_only_export_marks_fallback_without_mutating_detection(tmp_path):
    det = Detection("plate", .8, (10, 10, 40, 30))
    original = copy.deepcopy(det)
    path = tmp_path / "bbox.xml"
    assert CVATExporter().export([ImageAnnotation("one.jpg", 200, 200, [det])], path)
    restored = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert restored.geometry["bbox_fallback"]
    assert restored.raw_pose_keypoints is None
    assert det.geometry == original.geometry
    assert det.polygon == original.polygon


def test_direct_polygon_change_cannot_reuse_stale_xml_or_display_metadata(tmp_path):
    path = tmp_path / "legacy.xml"
    path.write_text('<annotations><image name="one.jpg" width="200" height="200">'
                    '<polygon label="plate" source="auto" points="10,10;100,10;100,50;10,50"/>'
                    '</image></annotations>', encoding="utf-8")
    ann = workflow._parse_cvat_preview_annotations(parser(), path)[0]
    ann.detections[0].polygon[0] = (12., 11.)
    assert CVATExporter().export([ann], path)
    restored = workflow._parse_cvat_preview_annotations(parser(), path)[0].detections[0]
    assert restored.polygon[0] == (12., 11.)
    assert geometry_display_polygon(restored)[0] == (12., 11.)
    assert restored.geometry["geometry_edit_untracked"]
    assert geometry_warning(restored)


def test_new_invalid_manual_polygon_export_fails_without_replacing_file(tmp_path):
    path = tmp_path / "manual.xml"
    path.write_text("existing", encoding="utf-8")
    det = Detection("plate", 1., (0, 0, 70, 140), polygon=BOWTIE)
    assert not CVATExporter().export([ImageAnnotation("one.jpg", 200, 200, [det])], path)
    assert path.read_text(encoding="utf-8") == "existing"
