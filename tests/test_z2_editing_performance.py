from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import copy
import threading
import xml.etree.ElementTree as ET

import pytest

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
from auto_annotation_tool.gui import z2_preview_autosave as autosave
from auto_annotation_tool.gui import z2_plate_gt_inline as inline
from auto_annotation_tool.gui import z2_session_runtime as session
from auto_annotation_tool.gui import z2_panel_workflow as panel
from auto_annotation_tool.gui import z2_annotation_process as process
from auto_annotation_tool.gui.z2_auto_run_result import merge_completed_view
from auto_annotation_tool.campaign_manager import CampaignManager
from auto_annotation_tool.config import CONFIG
from auto_annotation_tool.plate_ground_truth import get_plate_ground_truth, set_plate_ground_truth


def annotation(name="one.jpg", x=1):
    return ImageAnnotation(name, 100, 60, [Detection(
        "plate", .9, (x, 2, 40, 20), polygon=[(x, 2), (40, 2), (40, 20), (x, 20)],
        attributes={"manually_edited": "true", "audit_note": "Zażółć & <test>"},
    )])


def owner(tmp_path):
    annotations = [annotation(), annotation("two.jpg", 3)]
    path = tmp_path / "annotations.xml"
    assert CVATExporter().export(annotations, path, only_successful=False)
    callbacks = []
    host = SimpleNamespace(
        current_annotations=annotations, current_input_dir=tmp_path,
        _preview_dirty_images={"one.jpg"}, _preview_image_edit_versions={"one.jpg": 1},
        _preview_background_save=None, _preview_draw_mode=False,
        _get_current_annotation_xml_path=lambda: path,
        _post_to_ui=callbacks.append,
        _load_annotation_run_manifest=lambda run: {},
        _update_annotation_run_manifest=Mock(),
        _collect_preview_resume_manifest_fields=lambda: {},
        _remember_campaign_manual_plate_source=Mock(),
        _queue_free_mode_session_save=Mock(), _update_preview_toolbar_state=Mock(),
        _update_preview_edit_status=Mock(), _schedule_preview_autosave=Mock(),
    )
    return host, path, callbacks


def finish(host, callbacks):
    assert host._preview_background_save["done"].wait(5)
    # Worker sets the event before posting; wait for the thread's UI queue item.
    import time
    deadline = time.monotonic() + 2
    while not callbacks and time.monotonic() < deadline:
        time.sleep(.001)
    assert callbacks
    callbacks.pop(0)()


def test_incremental_save_preserves_other_images_gt_and_stable_ids(tmp_path):
    host, path, callbacks = owner(tmp_path)
    det = host.current_annotations[0].detections[0]
    set_plate_ground_truth(det.attributes, "WX12345")
    old_other = ET.tostring(ET.parse(path).getroot().findall("image")[1])
    det.polygon[0] = (8, 9)
    before_attrs = copy.deepcopy(det.attributes)
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    root = ET.parse(path).getroot()
    assert root.find("image/polygon").get("points").startswith("8.00,9.00;")
    assert ET.tostring(root.findall("image")[1]) == old_other
    assert get_plate_ground_truth(det.attributes) == "WX12345"
    assert det.attributes == before_attrs
    assert host._preview_dirty_images == set()
    assert not list(tmp_path.glob("*.tmp"))


def test_background_write_never_clears_a_newer_edit(tmp_path, monkeypatch):
    host, path, callbacks = owner(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = autosave.update_saved_images
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        original(*args)
    monkeypatch.setattr(autosave, "update_saved_images", blocked)
    assert autosave.start_preview_autosave(host)
    assert entered.wait(2)
    host.current_annotations[0].detections[0].polygon[0] = (12, 13)
    host._preview_image_edit_versions["one.jpg"] += 1
    release.set()
    finish(host, callbacks)
    assert host._preview_dirty_images == {"one.jpg"}
    assert not ET.parse(path).getroot().find("image/polygon").get("points").startswith("12.00")
    host._schedule_preview_autosave.assert_called_once()
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    assert ET.parse(path).getroot().find("image/polygon").get("points").startswith("12.00,13.00")


def test_explicit_save_supersedes_queued_autosave_completion(tmp_path):
    host, path, callbacks = owner(tmp_path)
    assert autosave.start_preview_autosave(host)
    autosave.wait_for_pending_save(host)
    # Keep a convex quad: this test exercises save ordering, not invalid geometry.
    host.current_annotations[0].detections[0].polygon[0] = (22, 5)
    assert CVATExporter().export(host.current_annotations, path, only_successful=False)
    for callback in callbacks:
        callback()
    assert ET.parse(path).getroot().find("image/polygon").get("points").startswith("22.00,5.00")
    host._update_annotation_run_manifest.assert_not_called()


def test_failed_autosave_keeps_dirty_data_and_original_file(tmp_path, monkeypatch):
    host, path, callbacks = owner(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(autosave, "update_saved_images", Mock(side_effect=OSError("locked")))
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    assert path.read_bytes() == before
    assert host._preview_dirty_images == {"one.jpg"}
    host._update_preview_edit_status.assert_called_once()


def test_missing_xml_image_does_not_replace_file(tmp_path):
    host, path, _ = owner(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        autosave.update_saved_images(path, [annotation("missing.jpg")])
    assert path.read_bytes() == before


def test_autosave_bookkeeping_waits_for_corner_drag_to_finish(tmp_path):
    host, path, callbacks = owner(tmp_path)
    host._preview_drag_state = {"vertex_idx": 0}
    host.frame = SimpleNamespace(after=lambda delay, callback: callbacks.append(callback))
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    host._remember_campaign_manual_plate_source.assert_not_called()
    assert host._preview_dirty_images
    host._preview_drag_state = None
    callbacks.pop(0)()
    assert not host._preview_dirty_images


def test_protected_bundle_uses_one_snapshot_and_worker_keeps_it_intact():
    from auto_annotation_tool.gui.z2_auto_run_result import protected_view_bundle
    previous = {"annotations": [annotation(), annotation("two.jpg")],
                "image_map": {"one.jpg": Path("one.jpg")}, "input_dir": "images"}
    bundle = protected_view_bundle(previous, {"one.jpg"})
    assert set(bundle) == {"one.jpg"}
    assert bundle["one.jpg"][0] is previous["annotations"][0]
    from auto_annotation_tool.gui.z2_preview_state import _merge_annotation_bundle_into_payload
    result, _, _ = _merge_annotation_bundle_into_payload([], {}, bundle)
    result[0].detections[0].polygon[0] = (22, 23)
    assert previous["annotations"][0].detections[0].polygon[0] == (1, 2)


def test_atomic_export_failure_preserves_original(tmp_path, monkeypatch):
    from auto_annotation_tool.exporters import cvat_exporter
    host, path, _ = owner(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(cvat_exporter.os, "replace", Mock(side_effect=OSError("locked")))
    assert not CVATExporter().export([annotation("replacement.jpg")], path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_unchanged_gt_focus_out_does_not_save_or_refresh_gate():
    det = annotation().detections[0]
    var = SimpleNamespace(get=lambda: "", set=lambda value: None)
    record = {"ann": annotation(), "det": det, "var": var}
    host = SimpleNamespace(_plate_gt_inline_widgets={"key": record}, _plate_gt_inline_save_after={})
    with patch.object(inline, "queue_inline_plate_gt_value") as save, \
         patch.object(inline, "_schedule_inline_gt_gate_refresh") as gate, \
         patch.object(inline, "_apply_record_style"):
        inline._commit_record(host, "key", final=True)
    save.assert_not_called()
    gate.assert_not_called()


def test_gt_edit_refreshes_gate_once_when_editing_finishes():
    det = annotation().detections[0]
    var = SimpleNamespace(get=lambda: "WX12345", set=lambda value: None)
    record = {"ann": annotation(), "det": det, "var": var}
    host = SimpleNamespace(_plate_gt_inline_widgets={"key": record}, _plate_gt_inline_save_after={})
    def save(*args, **kwargs):
        set_plate_ground_truth(det.attributes, "WX12345")
        return True, "WX12345"
    with patch.object(inline, "queue_inline_plate_gt_value", side_effect=save), \
         patch.object(inline, "_schedule_inline_gt_gate_refresh") as gate, \
         patch.object(inline, "_apply_record_style"):
        inline._commit_record(host, "key", final=False)
        gate.assert_not_called()
        inline._commit_record(host, "key", final=True)
        inline._commit_record(host, "key", final=True)
        gate.assert_called_once()


def test_content_edit_keeps_decoded_image_cache():
    cache = {"image-stat-key": object()}
    host = SimpleNamespace(_preview_render_image_cache=cache)
    session._invalidate_preview_runtime_caches(host)
    assert host._preview_render_image_cache is cache
    assert host._preview_counter_version == 1


def test_root_lookup_does_not_recreate_all_directories_and_repairs_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "DIR_9_PROJECTS", tmp_path)
    host = CampaignManager.__new__(CampaignManager)
    host.state = {"projects": {"demo": {"folder_name": "project"}}}
    real = host._ensure_project_workspace_tree
    with patch.object(host, "_ensure_project_workspace_tree", wraps=real) as ensure:
        root = host.get_project_root_dir("demo")
        host.get_project_root_dir("demo")
        ensure.assert_called_once()
        # A new configured root must get its own workspace tree.
        monkeypatch.setattr(CONFIG, "DIR_9_PROJECTS", tmp_path / "other")
        assert host.get_project_root_dir("demo").is_dir()
        assert ensure.call_count == 2


def test_worker_full_view_preserves_unprocessed_rows_without_changing_snapshot():
    old = [annotation(), annotation("two.jpg")]
    replacement = annotation(x=18)
    merged, paths = merge_completed_view([replacement], {}, {"annotations": old, "image_map": {"two.jpg": Path("two.jpg")}})
    assert merged[0] is replacement
    assert merged[1] is not old[1]
    assert [ann.filename for ann in merged] == ["one.jpg", "two.jpg"]
    assert paths["two.jpg"] == Path("two.jpg")


def test_completed_view_keeps_no_detection_result_instead_of_old_plate(tmp_path):
    old = [annotation()]
    missing = ImageAnnotation("one.jpg", 100, 60, [])
    merged, _ = merge_completed_view([missing], {}, {"annotations": old})
    assert merged == [missing]
    path = tmp_path / "annotations.xml"
    assert CVATExporter().export(merged, path, only_successful=False)
    image = ET.parse(path).getroot().find("image")
    assert image.get("name") == "one.jpg"
    assert image.find("polygon") is None


def test_finish_unlocks_list_before_publishing_worker_result():
    host = Mock()
    host._progress_update_lock = threading.Lock()
    host.current_annotations = [annotation()]
    host._annotation_stop_requested = False
    host._current_run_manual_template = False
    host._is_free_mode_session_context.return_value = True
    host._get_workflow_route.return_value = "auto"
    events = []
    host._set_preview_processing_overlay.side_effect = lambda active: events.append(("locked", active))
    host._finalize_successful_annotation_run_ui.side_effect = lambda *a, **kw: events.append(("publish", kw["result_loaded"]))
    process._finish(host, True, "done", run_dir=Path("new-run"))
    assert events == [("locked", False), ("publish", True)]


def test_loaded_result_does_not_reparse_or_merge_xml_again():
    from test_z2_auto_result_list import AutoResultListTests
    host = AutoResultListTests().owner(False)
    host._annotation_source_input_dir = None
    host._preview_dirty_images = {"one.jpg"}
    with patch.object(panel, "_merge_pre_run_visible_state_after_auto") as merge:
        panel._finalize_successful_annotation_run_ui(host, Path("new-run"), result_loaded=True)
    host._restore_preview_from_annotation_run.assert_not_called()
    merge.assert_not_called()
    assert host._preview_dirty_images == set()
    host._populate_preview_list_async.assert_called_once()


def test_shutdown_flushes_dirty_edits_after_background_write(tmp_path):
    from auto_annotation_tool.gui.tab_annotation import AnnotationTab
    host, path, callbacks = owner(tmp_path)
    host._cancel_preview_autosave = Mock()
    host._save_preview_edits = Mock(return_value=True)
    assert autosave.start_preview_autosave(host)
    assert AnnotationTab._force_save_all(host)
    assert host._preview_background_save is None
    host._save_preview_edits.assert_called_once_with(
        interactive=False, refresh_list=False, refresh_workflow=False, refresh_export_sources=False)


def test_hidden_campaign_refresh_waits_until_return_to_graph():
    frame = Mock()
    frame.winfo_viewable.return_value = False
    frame.bind.return_value = "binding"
    tab = SimpleNamespace(frame=frame, _refresh_dashboard=Mock())
    process._refresh_campaign_dashboard_when_visible(tab)
    process._refresh_campaign_dashboard_when_visible(tab)
    tab._refresh_dashboard.assert_not_called()
    frame.bind.assert_called_once()
    frame.bind.call_args.args[1](SimpleNamespace(widget=frame))
    frame.unbind.assert_called_once_with("<Map>", "binding")
    frame.after_idle.assert_called_once_with(tab._refresh_dashboard)
