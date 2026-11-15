from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch
import threading
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gui import z2_campaign_runtime as campaign
from auto_annotation_tool.gui import z2_gt_readiness as readiness
from auto_annotation_tool.gui import z2_gt_pack_runtime as runtime
from auto_annotation_tool.gui import z2_manifest_runtime as manifest
from auto_annotation_tool.gui import z2_plate_gt_inline as inline
from auto_annotation_tool.gui import z2_preview_autosave as autosave
from auto_annotation_tool.gui.tab_annotation import AnnotationTab
from auto_annotation_tool.plate_ground_truth import get_plate_ground_truth
from test_z2_editing_performance import owner as autosave_owner, finish


def plate(pid, gt="ABC123", polygon=None):
    return Detection("plate", .9, (10, 8, 80, 35),
                     polygon=polygon or [(10, 8), (80, 8), (80, 35), (10, 35)],
                     attributes={"plate_annotation_id": pid, "ground_truth_text": gt})


@pytest.mark.parametrize("primary_has_one", [False, True])
def test_counter_uses_scope_files_when_source_root_has_no_matching_images(tmp_path, monkeypatch, primary_has_one):
    source, scope, run = (tmp_path / name for name in ("source", "scope", "run"))
    for path in (source, scope, run):
        path.mkdir()
    annotations = [ImageAnnotation(name, 100, 60, [plate(name)]) for name in ("one.jpg", "two.jpg")]
    for ann in annotations:
        (scope / ann.filename).write_bytes(b"image")
    if primary_has_one:
        (source / "one.jpg").write_bytes(b"image")
    assert CVATExporter().export(annotations, run / "annotations.xml", only_successful=False)
    approved = {ann.filename for ann in annotations}
    payload = {"source_input_dir": str(source), "input_dir": str(scope), "approved_filenames": sorted(approved)}
    host = SimpleNamespace(
        current_annotation_run_dir=run, current_input_dir=scope, current_annotations=annotations,
        _is_free_mode_session_context=lambda: False,
        _resolve_safe_annotation_run_dir=lambda path, **kw: Path(path),
        _paths_equivalent=lambda a, b: Path(a) == Path(b),
        _load_annotation_run_manifest=lambda p: payload,
        _get_preview_approved_filenames=lambda: approved,
        _resolve_existing_dir=lambda p: Path(p) if p and Path(p).is_dir() else None,
        _annotation_run_manifest_has_manual_value=lambda m: False,
        _get_plate_detections=lambda ann: ann.plates,
        _detection_polygon=lambda det: det.polygon,
        _build_campaign_plate_entry_key=manifest._build_campaign_plate_entry_key,
    )
    host._build_campaign_plate_approved_entries_from_run = MethodType(campaign._build_campaign_plate_approved_entries_from_run, host)
    monkeypatch.setattr(readiness, "CAMPAIGN", SimpleNamespace(list_plate_approved_entries=lambda: []))
    result = readiness.build_char_gt_readiness(host, run_dir=run)
    assert result["gt_plates"] == result["total_plates"] == 2
    assert result["ready"]
    entries = host._build_campaign_plate_approved_entries_from_run(run)
    assert Path(entries[1]["source_image_path"]) == scope / "two.jpg"
    assert Path(entries[0]["source_image_path"]) == (source if primary_has_one else scope) / "one.jpg"


def test_live_gt_counter_avoids_export_builder_and_replaces_same_run_metadata(tmp_path, monkeypatch):
    run = tmp_path / "run"
    previous = {"entry_key": "original-key", "image_name": "one.jpg", "approved_from_run": str(run),
                "source_image_path": str(tmp_path / "source" / "one.jpg"),
                "plates": [{"attributes": {"ground_truth_text": ""}}]}
    unrelated = {"entry_key": "other-key", "image_name": "one.jpg", "approved_from_run": str(tmp_path / "old-run"),
                 "plates": [{"attributes": {"ground_truth_text": "OLD123"}}]}
    host = SimpleNamespace(current_annotation_run_dir=run,
                           current_annotations=[ImageAnnotation("one.jpg", 100, 60, [plate("p")])],
                           _get_preview_approved_filenames=lambda: {"one.jpg"},
                           _get_plate_detections=lambda ann: ann.plates,
                           _build_campaign_plate_approved_entries_from_run=Mock(side_effect=AssertionError("must not scan files")))
    monkeypatch.setattr(readiness, "CAMPAIGN", SimpleNamespace(list_plate_approved_entries=lambda: [previous, unrelated]))
    result = readiness.build_char_gt_readiness(host, run_dir=run)
    assert result["gt_plates"] == result["total_plates"] == 2
    assert result["missing_gt"] == 0
    host._build_campaign_plate_approved_entries_from_run.assert_not_called()


def test_final_gt_validation_still_reads_saved_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    host = SimpleNamespace(current_annotation_run_dir=run,
                           current_annotations=[ImageAnnotation("one.jpg", 100, 60, [plate("p")])],
                           _get_preview_approved_filenames=lambda: {"one.jpg"},
                           _get_plate_detections=lambda ann: ann.plates,
                           _build_campaign_plate_approved_entries_from_run=Mock(return_value=[{
                               "entry_key": "saved", "image_name": "one.jpg",
                               "plates": [{"attributes": {"ground_truth_text": ""}}]}]))
    monkeypatch.setattr(readiness, "CAMPAIGN", SimpleNamespace(list_plate_approved_entries=lambda: []))
    assert not readiness.build_char_gt_readiness(host, run_dir=run, force_parse_xml=True)["ready"]
    host._build_campaign_plate_approved_entries_from_run.assert_called_once_with(run, force_parse_xml=True)


def hydration_host(tmp_path, local_gt="ABC123"):
    image = tmp_path / "image.png"
    Image.new("RGB", (200, 100), "gray").save(image)
    pack = ALPRGTPack.create(tmp_path / "work.alprgt")
    record = pack.add_image(image)
    # Different identity, slightly shifted manual correction, same image bytes.
    pack.ensure_plate(image_id=record["image_id"], polygon=[(12, 9), (82, 9), (82, 36), (12, 36)], plate_id="old-alias")
    pack.set_ground_truth("old-alias", "ABC123")
    pack.ensure_plate(image_id=record["image_id"], polygon=[(120, 50), (190, 50), (190, 80), (120, 80)], plate_id="other-plate")
    pack.set_ground_truth("other-plate", "DEF456")
    host = SimpleNamespace(
        _resolve_preview_image_path=lambda ann: image,
        _get_plate_detections=lambda ann: ann.plates,
        _detection_polygon=lambda det: det.polygon,
    )
    ann = ImageAnnotation("renamed.png", 200, 100, [plate("current", local_gt)])
    return host, ann, pack


@pytest.mark.parametrize("local_gt", ["ABC123", "DIFFERENT", ""])
def test_hydration_skips_overlapping_alias_without_rebinding_gt(tmp_path, monkeypatch, local_gt):
    host, ann, pack = hydration_host(tmp_path, local_gt)
    monkeypatch.setattr(runtime, "_open_mounted_packs", lambda h: [(pack.root, pack)])
    original = ann.plates[0]
    original_polygon = list(original.polygon)
    report = runtime.hydrate_annotations_from_gt_pack(host, ann)
    assert report["skipped_overlapping"] == ["old-alias"]
    assert report["created"] == 1  # The separate second plate still hydrates.
    assert len(ann.plates) == 2
    assert ann.plates[0] is original
    assert original.polygon == original_polygon
    assert get_plate_ground_truth(original.attributes) == local_gt
    assert original.attributes["plate_annotation_id"] == "current"
    assert runtime.hydrate_annotations_from_gt_pack(host, ann)["created"] == 0


def gt_owner(tmp_path):
    host, xml, callbacks = autosave_owner(tmp_path)
    for ann in host.current_annotations:
        Image.new("RGB", (100, 60), "gray").save(tmp_path / ann.filename)
    host._z2_gt_working_pack_path = str(tmp_path / "work.alprgt")
    host._z2_gt_pack_source_paths = []
    host._z2_gt_mount_config_loaded = True
    host._z2_gt_mount_context_key = runtime._mount_context_key()
    host._resolve_preview_image_path = lambda ann: tmp_path / ann.filename
    host._detection_polygon = lambda det: det.polygon
    host._refresh_preview_canvas_light = Mock()
    host._refresh_preview_list_row_for_actual_index = Mock()
    host._save_preview_edits = Mock(side_effect=AssertionError("UI must not perform a full synchronous save"))
    def mark(ann, **kwargs):
        host._preview_dirty_images.add(ann.filename)
        versions = host._preview_image_edit_versions
        versions[ann.filename] = versions.get(ann.filename, 0) + 1
    host._mark_preview_image_dirty = mark
    return host, xml, callbacks


def xml_gt(xml):
    return ET.parse(xml).getroot().findtext("image/polygon/attribute[@name='ground_truth_text']", default="")


def test_gt_pack_io_runs_off_ui_and_later_edit_wins(tmp_path, monkeypatch):
    host, xml, callbacks = gt_owner(tmp_path)
    ann = host.current_annotations[0]
    det = ann.plates[0]
    entered, release = threading.Event(), threading.Event()
    original = runtime._sync_item_to_pack
    thread_ids = []
    def blocked(item):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(item)
    monkeypatch.setattr(runtime, "_sync_item_to_pack", blocked)
    try:
        assert inline.queue_inline_plate_gt_value(host, ann, det, "AA123")[0]
        assert entered.wait(2)
        # The first write is still blocked, but the UI can accept another edit.
        assert inline.queue_inline_plate_gt_value(host, ann, det, "BB234")[0]
        assert get_plate_ground_truth(det.attributes) == "BB234"
    finally:
        release.set()
    finish(host, callbacks)
    assert host._preview_gt_sync_pending
    assert ann.filename in host._preview_dirty_images
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    assert xml_gt(xml) == "BB234"
    pack = ALPRGTPack.open(tmp_path / "work.alprgt")
    assert pack.resolve_ground_truth(det.attributes["plate_annotation_id"])["text"] == "BB234"
    assert not host._preview_gt_sync_pending
    assert not host._preview_dirty_images
    assert all(tid != threading.get_ident() for tid in thread_ids)
    host._save_preview_edits.assert_not_called()


def test_xml_failure_keeps_typed_gt_and_pending_sync_for_retry(tmp_path, monkeypatch):
    host, xml, callbacks = gt_owner(tmp_path)
    ann, det = host.current_annotations[0], host.current_annotations[0].plates[0]
    original = autosave.update_saved_images
    monkeypatch.setattr(autosave, "update_saved_images", Mock(side_effect=OSError("locked")))
    assert inline.queue_inline_plate_gt_value(host, ann, det, "AA123")[0]
    finish(host, callbacks)
    assert get_plate_ground_truth(det.attributes) == "AA123"
    assert host._preview_gt_sync_pending and host._preview_dirty_images
    assert xml_gt(xml) == ""
    monkeypatch.setattr(autosave, "update_saved_images", original)
    assert autosave.start_preview_autosave(host)
    finish(host, callbacks)
    assert xml_gt(xml) == "AA123"
    assert not host._preview_gt_sync_pending


def test_failed_pack_set_is_not_replayed_after_a_later_clear(tmp_path, monkeypatch):
    host, xml, callbacks = gt_owner(tmp_path)
    ann, det = host.current_annotations[0], host.current_annotations[0].plates[0]
    original = runtime._sync_item_to_pack
    monkeypatch.setattr(runtime, "_sync_item_to_pack", Mock(side_effect=OSError("locked pack")))
    inline.queue_inline_plate_gt_value(host, ann, det, "AA123")
    finish(host, callbacks)
    assert xml_gt(xml) == "AA123"
    assert (tmp_path / "gt_sync_pending.json").is_file()
    monkeypatch.setattr(runtime, "_sync_item_to_pack", original)
    inline.queue_inline_plate_gt_value(host, ann, det, "")
    finish(host, callbacks)
    assert xml_gt(xml) == ""
    assert not (tmp_path / "gt_sync_pending.json").exists()
    pack = ALPRGTPack.open(tmp_path / "work.alprgt")
    assert pack.resolve_ground_truth(det.attributes["plate_annotation_id"])["operation"] == "clear"


def test_explicit_save_flushes_latest_gt_after_waiting_for_older_worker(tmp_path, monkeypatch):
    host, xml, callbacks = gt_owner(tmp_path)
    ann, det = host.current_annotations[0], host.current_annotations[0].plates[0]
    entered, release = threading.Event(), threading.Event()
    original = runtime._sync_item_to_pack
    def blocked(item):
        entered.set()
        assert release.wait(5)
        return original(item)
    monkeypatch.setattr(runtime, "_sync_item_to_pack", blocked)
    try:
        inline.queue_inline_plate_gt_value(host, ann, det, "AA123")
        assert entered.wait(2)
        inline.queue_inline_plate_gt_value(host, ann, det, "CC345")
    finally:
        release.set()
    autosave.wait_for_pending_save(host)
    assert CVATExporter().export(host.current_annotations, xml, only_successful=False)
    assert autosave.flush_pending_gt_after_xml_save(host, xml)["ok"]
    assert xml_gt(xml) == "CC345"
    pack = ALPRGTPack.open(tmp_path / "work.alprgt")
    assert pack.resolve_ground_truth(det.attributes["plate_annotation_id"])["text"] == "CC345"
    for callback in callbacks:
        callback()
    assert not host._preview_gt_sync_pending


def test_shutdown_commits_text_even_before_focus_out_or_debounce(tmp_path):
    host, xml, callbacks = gt_owner(tmp_path)
    ann, det = host.current_annotations[0], host.current_annotations[0].plates[0]
    record = {"ann": ann, "det": det, "var": SimpleNamespace(get=lambda: "LAST123", set=lambda value: None)}
    host._plate_gt_inline_widgets = {"key": record}
    host._plate_gt_inline_save_after = {}
    host._cancel_preview_autosave = Mock()
    host._save_preview_edits = Mock(return_value=True)
    with patch.object(inline, "_apply_record_style"), patch.object(inline, "_schedule_inline_gt_gate_refresh"):
        assert AnnotationTab._force_save_all(host)
    assert xml_gt(xml) == "LAST123"
    pack = ALPRGTPack.open(tmp_path / "work.alprgt")
    assert pack.resolve_ground_truth(det.attributes["plate_annotation_id"])["text"] == "LAST123"


def test_keypresses_defer_ui_maintenance_without_reconfiguring_unchanged_widgets():
    host = SimpleNamespace(app=SimpleNamespace(palette={}), _mark_preview_user_interaction=Mock())
    record = {"det": plate("p"), "editing": True, "confirmed": True,
              **{name: Mock() for name in ("shell", "info_row", "gt_row", "gt_label", "entry")}}
    host._plate_gt_inline_widgets = {"key": record}
    with patch.object(inline, "_update_record_layout_ui"):
        inline._apply_record_style(host, record, editing=True)
        for _ in range(10):
            inline._on_key_press(host, "key")
    assert record["entry"].configure.call_count == 1
    assert host._mark_preview_user_interaction.call_count == 10
