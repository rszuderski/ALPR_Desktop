import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z3_metadata_cache as cache
from auto_annotation_tool.gui import z3_preview_metadata_runtime as runtime
from auto_annotation_tool.gui import z3_readiness as readiness
from auto_annotation_tool.gui.z3_goldpack_ui import collect_gold_export_plate_candidates
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


def source(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "metadata.json"
    data = {"first": {"status": "perfect", "source_image": "ABC1234.jpg",
                      "source_info": {"bucket": "auto_preview"}, "fusion_strategy": "yolo_exact",
                      "characters": [{"character": "A", "bbox": [0, 0, 10, 20], "method": "yolo",
                                      "box_source": "yolo_box", "sign_source": "yolo_symbol"}],
                      "yolo_raw_detections": [{"bbox": [i, 2, i + 3, 20], "confidence": 0.2345} for i in range(300)]},
            "second": {"status": "needs_fix", "characters": [], "note": 'Zażółć: "{}", / \\ путь'}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp_path / "images").mkdir()
    (tmp_path / "images/first.jpg").touch()
    return path, data


@pytest.mark.parametrize("text", ['{}', ' {"a":1,"b":[true,false,null,{"x":"}\\\""}]} ',
                                 '{"a":{"v":"zażółć"},"a":null}', '{"a":NaN}'])
def test_fragment_decoder_keeps_standard_json_semantics(text):
    payload, fragments = cache.decode_preview_records(text)
    expected = json.loads(text)
    assert json.dumps(payload, sort_keys=True) == json.dumps(expected, sort_keys=True)
    rebuilt = "{" + ",".join(json.dumps(key) + ":" + value for key, value in fragments.items()) + "}"
    assert json.dumps(json.loads(rebuilt), sort_keys=True) == json.dumps(expected, sort_keys=True)


@pytest.mark.parametrize("text", ['{"a":1,}', '{1:2}', '{"a" 2}', '{} garbage', '{"a":1 "b":2}', ''])
def test_fragment_decoder_rejects_malformed_json(text):
    with pytest.raises(ValueError):
        cache.decode_preview_records(text)


def test_repeated_read_uses_one_parse_and_observes_external_file_change(tmp_path):
    path, original = source(tmp_path)
    host = SimpleNamespace()
    with patch.object(cache, "decode_preview_records", wraps=cache.decode_preview_records) as decode:
        first = cache.read_preview_metadata(host, path)
        assert cache.read_preview_metadata(host, path) is first
        decode.assert_called_once()
        path.write_text('{"external":{"changed":true}}', encoding="utf-8")
        assert cache.read_preview_metadata(host, path) == {"external": {"changed": True}}
        assert decode.call_count == 2
    assert first == original


def test_active_editor_values_override_disk_until_explicit_reload(tmp_path):
    path, data = source(tmp_path)
    host = SimpleNamespace(preview_metadata=data, _loaded_meta_path=path)
    data["second"]["status"] = "perfect"
    with patch.object(cache, "decode_preview_records", wraps=cache.decode_preview_records) as decode:
        assert cache.read_preview_metadata(host, path) is data
        decode.assert_not_called()
        assert cache.read_preview_metadata(host, path, prefer_live=False)["second"]["status"] == "needs_fix"


def test_readiness_normalizes_final_characters_without_mutating_diagnostics(tmp_path):
    path, data = source(tmp_path)
    before = copy.deepcopy(data)
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_metadata = data
    host._loaded_meta_path = path
    host._get_gold_export_meta_candidates = lambda: [path]
    with patch.object(host, "_normalize_character_source_kind", wraps=host._normalize_character_source_kind) as normalize:
        entries, *_ = collect_gold_export_plate_candidates(host, set(), set())
    assert len(entries) == 1
    assert normalize.call_count == 1
    assert data == before
    assert entries[0]["data"]["yolo_raw_detections"] == before["first"]["yolo_raw_detections"]


def test_readiness_cache_invalidates_on_edit_filter_and_file_change(tmp_path):
    path, data = source(tmp_path)
    collect = Mock(return_value=([{"data": data["first"]}], {}, {}, {}, {}))
    host = SimpleNamespace(_preview_metadata_revision=0, preview_metadata=data, _loaded_meta_path=path,
                           _get_gold_export_meta_candidates=lambda: [path],
                           _get_selected_gold_export_strategy_buckets=lambda: {"yolo_exact"},
                           _get_selected_gold_export_source_buckets=lambda: {"auto_preview"},
                           _collect_gold_export_plate_candidates=collect,
                           _is_exportable_character_record=lambda rec: bool(rec.get("character")))
    with patch.object(readiness.CAMPAIGN, "get_active_project_name", return_value=""):
        assert readiness.get_step3_yolo_export_readiness_snapshot(host)["ok"]
        readiness.get_step3_yolo_export_readiness_snapshot(host)
        assert collect.call_count == 1
        data["first"]["characters"].clear()
        cache.mark_preview_metadata_changed(host)
        assert not readiness.get_step3_yolo_export_readiness_snapshot(host)["ok"]
        assert collect.call_count == 2
        readiness.get_step3_yolo_export_readiness_snapshot(host, selected_sources={"local_manual"})
        assert collect.call_count == 3
        host._loaded_meta_path = None
        readiness.get_step3_yolo_export_readiness_snapshot(host)
        path.write_text('{"new":{}}', encoding="utf-8")
        readiness.get_step3_yolo_export_readiness_snapshot(host)
        assert collect.call_count == 5


def test_autosave_uses_immutable_snapshots_and_finishes_with_latest_edits(tmp_path):
    path, _ = source(tmp_path)
    data, fragments = cache.decode_preview_records(path.read_text(encoding="utf-8"))
    writer = cache.PreviewMetadataAutosave(path, data, fragments)
    started, release = threading.Event(), threading.Event()
    writes = []
    original_write = writer._write
    def delayed(snapshot):
        started.set()
        assert release.wait(5)
        result = original_write(snapshot)
        writes.append(json.loads(path.read_text(encoding="utf-8")))
        return result
    try:
        with patch.object(writer, "_write", delayed):
            data["first"]["characters"][0]["character"] = "B"
            writer.submit({"first"})
            assert started.wait(5)
            data["first"]["characters"][0]["character"] = "C"
            data["first"]["yolo_raw_detections"][0]["confidence"] = 0.9
            data["second"]["note"] = "kolejna edycja"
            writer.submit({"first", "second"})
            release.set()
            writer.flush()
        assert writes[0]["first"]["characters"][0]["character"] == "B"
        assert writes[0]["first"]["yolo_raw_detections"][0]["confidence"] == 0.2345
        assert writes[-1] == data
    finally:
        release.set()
        writer.close()


def test_atomic_autosave_failure_preserves_previous_file(tmp_path):
    path, data = source(tmp_path)
    before = path.read_bytes()
    _, fragments = cache.decode_preview_records(path.read_text(encoding="utf-8"))
    writer = cache.PreviewMetadataAutosave(path, data, fragments)
    with patch.object(Path, "replace", side_effect=OSError("locked")):
        writer.submit({"first"})
        with pytest.raises(OSError, match="locked"):
            writer.close()
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


class Timers:
    def __init__(self):
        self.jobs = {}
        self.serial = 0
    def after(self, _ms, callback):
        self.serial += 1
        self.jobs[self.serial] = callback
        return self.serial
    def after_cancel(self, job):
        self.jobs.pop(job, None)


def test_autosave_keeps_edited_plate_when_user_navigates_before_timer(tmp_path):
    path, _ = source(tmp_path)
    host = SimpleNamespace(frame=Timers(), _get_preview_metadata_path=lambda: path,
                           _preview_active_pid="first", _log_preview_edit_flow=Mock(),
                           _schedule_preview_info_refresh=Mock(), _update_preview_edit_status=Mock())
    host.preview_metadata = cache.read_preview_metadata(host, path)
    host._loaded_meta_path = path
    host.preview_metadata["first"]["characters"][0]["character"] = "Z"
    runtime._schedule_preview_metadata_save(host)
    host._preview_active_pid = "second"
    job = host._preview_metadata_save_after_id
    host.frame.jobs.pop(job)()
    try:
        runtime._flush_scheduled_preview_metadata_save(host)
        assert json.loads(path.read_text(encoding="utf-8")) == host.preview_metadata
        assert not host.frame.jobs
    finally:
        host._preview_autosave_writer.close()


def test_clean_preview_does_not_write_on_flush():
    host = SimpleNamespace(preview_metadata={"plate": {}}, frame=Timers(), _persist_preview_metadata=Mock())
    runtime._flush_scheduled_preview_metadata_save(host)
    host._persist_preview_metadata.assert_not_called()


def test_flush_keeps_edits_when_next_drag_cancelled_the_save_timer(tmp_path):
    path, _ = source(tmp_path)
    host = SimpleNamespace(frame=Timers(), _get_preview_metadata_path=lambda: path,
                           _preview_active_pid="first", _log_preview_edit_flow=Mock(),
                           _schedule_preview_info_refresh=Mock(), _update_preview_edit_status=Mock())
    host.preview_metadata = cache.read_preview_metadata(host, path)
    host._loaded_meta_path = path
    host.preview_metadata["first"]["characters"][0]["character"] = "Q"
    runtime._schedule_preview_metadata_save(host)
    runtime._cancel_scheduled_preview_metadata_save(host)
    try:
        runtime._flush_scheduled_preview_metadata_save(host)
        assert json.loads(path.read_text(encoding="utf-8")) == host.preview_metadata
        assert not host.frame.jobs
    finally:
        host._preview_autosave_writer.close()


def test_pending_save_never_overwrites_next_selected_run(tmp_path):
    first, _ = source(tmp_path / "first")
    second, _ = source(tmp_path / "second")
    before_second = second.read_bytes()
    selected = [first]
    host = SimpleNamespace(frame=Timers(), _get_preview_metadata_path=lambda: selected[0],
                           _preview_active_pid="first", _log_preview_edit_flow=Mock(),
                           _schedule_preview_info_refresh=Mock(), _update_preview_edit_status=Mock())
    host.preview_metadata = cache.read_preview_metadata(host, first)
    host._loaded_meta_path = first
    host.preview_metadata["first"]["characters"][0]["character"] = "X"
    runtime._schedule_preview_metadata_save(host)
    selected[0] = second
    try:
        runtime._flush_scheduled_preview_metadata_save(host)
        assert json.loads(first.read_text(encoding="utf-8"))["first"]["characters"][0]["character"] == "X"
        assert second.read_bytes() == before_second
    finally:
        host._preview_autosave_writer.close()


def test_readiness_and_prepared_export_select_the_same_plates(tmp_path):
    path, data = source(tmp_path)
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_metadata = data
    host._loaded_meta_path = path
    host._get_gold_export_meta_candidates = lambda: [path]
    prepared = collect_gold_export_plate_candidates(host, set(), set())
    readonly = collect_gold_export_plate_candidates(host, set(), set(), prepare_records=False)
    assert [entry["pid"] for entry in prepared[0]] == [entry["pid"] for entry in readonly[0]]
    assert prepared[1:] == readonly[1:]


def test_theme_paint_does_not_scan_preview_or_recalculate_readiness():
    from auto_annotation_tool.gui.z3_preview_status_ui import apply_preview_info_stats_style
    host = SimpleNamespace(app=SimpleNamespace(palette={}), _update_preview_repair_progress_ui=Mock())
    apply_preview_info_stats_style(host, type("Progress", (), {}))
    host._update_preview_repair_progress_ui.assert_not_called()


def test_hidden_campaign_graph_refresh_waits_until_return_from_pz2():
    from auto_annotation_tool.gui import campaign_stage_ui as stage
    host = Mock()
    host.frame.winfo_viewable.return_value = False
    host._refresh_wizard_transition_graph = lambda **kw: stage._refresh_wizard_transition_graph(host, **kw)
    stage._refresh_wizard_transition_graph(host, allow_pending_actions=False)
    stage._refresh_wizard_transition_graph(host, allow_pending_actions=False)
    host._render_step1_route_actions.assert_not_called()
    host.frame.bind.assert_called_once()
    callback = host.frame.bind.call_args.args[1]
    callback(SimpleNamespace(widget=host.frame))
    host.frame.winfo_viewable.return_value = True
    host.frame.after_idle.call_args.args[0]()
    host._render_step1_route_actions.assert_called_once_with(host.wizard_transition_graph_body, allow_pending_actions=False)


def test_row_still_revalidates_stale_perfect_status_after_loading(tmp_path):
    _, data = source(tmp_path)
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = data["first"]
    record["source_expected_texts"] = ["Z"]
    record["source_expected_text_source"] = "manual"
    host._get_plate_listbox_ordinal = lambda _pid: 1
    label = host._format_plate_listbox_label("first", record, ordinal=1)
    assert record["status"] == "needs_fix"
    assert "OK|" not in label


def test_hidden_canvas_waits_for_map_before_computing_image_transform():
    from auto_annotation_tool.gui.z3_preview_ui import on_preview_select
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_canvas = Mock()
    host.preview_canvas.winfo_ismapped.return_value = False
    host._schedule_preview_select_render = Mock()
    host._preview_render_state = {}
    on_preview_select(host)
    assert host._preview_render_when_visible
    assert host._preview_render_state == {}
    host._on_preview_canvas_configure(SimpleNamespace(width=800, height=600))
    host._schedule_preview_select_render.assert_called_once_with(delay_ms=35)


def test_campaign_preview_reuse_requires_same_iteration_source_and_file(tmp_path):
    from auto_annotation_tool.gui import z3_campaign_flow as flow
    path, data = source(tmp_path)
    xml = tmp_path / "annotations.xml"
    xml.write_text("<annotations/>", encoding="utf-8")
    images = tmp_path / "images"
    with patch.object(flow.CAMPAIGN, "get_step3_preview_dir", return_value=str(tmp_path)), \
            patch.object(flow.CAMPAIGN, "get_active_project_name", return_value="project"), \
            patch.object(flow.CAMPAIGN, "get_current_iteration_num", return_value=2):
        key = flow._campaign_preview_entry_key(xml, images)
        assert key is not None
        host = SimpleNamespace(_last_campaign_preview_entry_key=key, _loaded_meta_path=path,
                               _detect_tab_built=True, preview_metadata=data, _listbox_pid_by_index=["first"])
        assert flow._can_reuse_campaign_preview(host, key)
        with patch.object(flow.CAMPAIGN, "get_current_iteration_num", return_value=3):
            assert not flow._can_reuse_campaign_preview(host, flow._campaign_preview_entry_key(xml, images))
        xml.write_text("<annotations>changed</annotations>", encoding="utf-8")
        assert not flow._can_reuse_campaign_preview(host, flow._campaign_preview_entry_key(xml, images))
        host._last_campaign_preview_entry_key = flow._campaign_preview_entry_key(xml, images)
        path.write_text("{}", encoding="utf-8")
        assert not flow._can_reuse_campaign_preview(host, flow._campaign_preview_entry_key(xml, images))
