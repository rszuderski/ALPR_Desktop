import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z3_paths as paths
from auto_annotation_tool.gui import z3_detection_model_ui as devices
from auto_annotation_tool.gui import z3_extraction_sources as sources
from auto_annotation_tool.gui import z3_cvat_tab_ui as cvat
from auto_annotation_tool.gui import z3_goldpack_ui as gold
from auto_annotation_tool.gui import z3_navigation_runtime as navigation
from auto_annotation_tool.gui import z3_shared_ui as shared


class Value:
    def __init__(self, value=""):
        self.value = value
        self.writes = 0

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self.writes += 1


def owner(free=True):
    host = SimpleNamespace(
        app=SimpleNamespace(campaign_free_mode=free),
        _step3_linear_mode=not free,
        preview_dir_var=Value(), xml_path_var=Value(), images_dir_var=Value(),
        _preview_dir_plate_count_cache={}, _save_local_setting=Mock(),
        _is_campaign_char_step3_context=Mock(return_value=False),
    )
    host._get_preferred_step3_preview_dir = lambda **kw: paths.get_preferred_step3_preview_dir(host, **kw)
    host._preview_dir_has_plate_entries = lambda p: paths.preview_dir_has_plate_entries(host, p)
    host._get_preview_dir_plate_count = lambda p: paths.get_preview_dir_plate_count(host, p)
    host._is_usable_step3_preview_dir = lambda p, **kw: paths.is_usable_step3_preview_dir(host, p, **kw)
    return host


def make_run(root, name, count=2):
    run = root / name
    (run / "images").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({f"plate_{i}": {} for i in range(count)}))
    return run


@pytest.mark.parametrize("active_project", ["", "pisto"])
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_free_status_has_no_implicit_run_or_campaign_lookup(active_project, allow_fallback):
    host = owner()
    with patch.object(paths.CAMPAIGN, "get_active_project_name", return_value=active_project), \
         patch.object(paths.CAMPAIGN, "get_step3_preview_dir") as saved, \
         patch.object(paths, "find_latest_preview_run_dir") as scan:
        assert paths.get_preferred_step3_preview_dir(host, True, allow_fallback) == ""
        assert paths.get_saved_step3_preview_dir(host, True) == ""
    saved.assert_not_called()
    scan.assert_not_called()
    assert host.preview_dir_var.writes == 0


@pytest.mark.parametrize("free", [True, False])
def test_explicit_preview_remains_available_without_history_scan(tmp_path, free):
    run = make_run(tmp_path, "selected")
    host = owner(free)
    host.preview_dir_var = Value(str(run))
    with patch.object(paths.CAMPAIGN, "get_active_project_name", return_value="pisto"), \
         patch.object(paths.CAMPAIGN, "get_step3_preview_dir", return_value=""), \
         patch.object(paths, "find_latest_preview_run_dir") as scan:
        assert paths.get_preferred_step3_preview_dir(host, True) == str(run)
        assert cvat.get_active_preview_context(host)["plate_count"] == 2
    scan.assert_not_called()
    assert host.preview_dir_var.writes == 0


def test_free_empty_cvat_status_does_not_write_or_choose_a_run():
    host = owner()
    with patch.object(paths, "find_latest_preview_run_dir") as scan:
        assert cvat.get_active_preview_context(host)["reason"] == "missing_preview"
        assert gold.get_gold_export_meta_candidates(host) == []
    scan.assert_not_called()
    host._save_local_setting.assert_not_called()
    assert host.preview_dir_var.writes == 0


def test_free_gold_counts_do_not_rewrite_same_preview_and_trigger_traces(tmp_path):
    host = owner()
    run = make_run(tmp_path, "selected")
    host.preview_dir_var = Value(str(run))
    for _ in range(10):
        assert gold.get_gold_export_meta_candidates(host) == [run / "metadata.json"]
    assert host.preview_dir_var.writes == 0
    host._save_local_setting.assert_not_called()


@pytest.mark.parametrize("iteration", [1, 2, 12])
def test_campaign_saved_preview_has_priority_over_current_and_no_rescan(tmp_path, iteration):
    host = owner(False)
    host._campaign_graph_entry_context = {"graph_gate_id": "T05", "iteration": iteration}
    saved = make_run(tmp_path, "campaign")
    host.preview_dir_var.set(str(make_run(tmp_path, "old")))
    with patch.object(paths.CAMPAIGN, "get_active_project_name", return_value="pisto"), \
         patch.object(paths.CAMPAIGN, "get_step3_preview_dir", return_value=str(saved)), \
         patch.object(paths, "find_latest_preview_run_dir") as scan:
        assert paths.get_preferred_step3_preview_dir(host, True) == str(saved)
    scan.assert_not_called()


def test_campaign_fallback_is_still_available_when_saved_run_is_missing():
    host = owner(False)
    with patch.object(paths.CAMPAIGN, "get_active_project_name", return_value="pisto"), \
         patch.object(paths.CAMPAIGN, "get_step3_preview_dir", return_value=""), \
         patch.object(paths, "find_latest_preview_run_dir", return_value="campaign-run") as scan:
        assert paths.get_preferred_step3_preview_dir(host, True, False) == ""
        assert paths.get_preferred_step3_preview_dir(host, True, True) == "campaign-run"
    scan.assert_called_once_with(host, require_plates=True)


@pytest.mark.parametrize("xml,images", [("", ""), ("annotations.xml", ""), ("", "images")])
def test_empty_free_source_never_scans_extract_history(xml, images):
    host = owner()
    host.xml_path_var.set(xml)
    host.images_dir_var.set(images)
    host._get_step3_chars_root_dir = Mock(side_effect=AssertionError("unrequested disk scan"))
    assert sources.find_latest_extract_preview_run_dir(host, True) == ""
    host._get_step3_chars_root_dir.assert_not_called()


@pytest.mark.parametrize("free", [True, False])
def test_source_bound_scan_uses_metadata_and_keeps_matching_latest_run(tmp_path, free):
    host = owner(free)
    host.xml_path_var.set("chosen.xml")
    host.images_dir_var.set("chosen-images")
    host._get_step3_chars_root_dir = Mock(return_value=tmp_path)
    older = make_run(tmp_path, "older")
    latest = make_run(tmp_path, "latest")
    wrong = make_run(tmp_path, "wrong-source")
    for stamp, run in enumerate((older, latest, wrong), start=1):
        os.utime(run, (stamp, stamp))
        for i in range(10):
            (run / "images" / f"{i}.jpg").touch()
    patterns = []
    original = Path.rglob
    def rglob(path, pattern):
        patterns.append(pattern)
        return original(path, pattern)
    with patch.object(paths.CAMPAIGN, "get_active_project_name", return_value="pisto"), \
         patch.object(Path, "rglob", rglob), \
         patch.object(sources, "preview_matches_current_extract_source", side_effect=lambda h, p: p != wrong):
        assert sources.find_latest_extract_preview_run_dir(host, True) == str(latest)
    assert patterns == ["metadata.json"]


def test_generic_campaign_scan_rejects_empty_and_inflated_runs(tmp_path):
    host = owner(False)
    host._get_step3_chars_root_dir = Mock(return_value=tmp_path)
    valid = make_run(tmp_path, "valid")
    inflated = make_run(tmp_path, "inflated", 100)
    empty = make_run(tmp_path, "empty", 0)
    for stamp, run in enumerate((valid, inflated, empty), start=1):
        os.utime(run, (stamp, stamp))
    with patch.object(paths, "preview_dir_is_campaign_inflated", side_effect=lambda h, p: p == inflated):
        assert paths.find_latest_preview_run_dir(host, True) == str(valid)


def device_owner(options=None, ready=False):
    host = SimpleNamespace(
        app=SimpleNamespace(get_available_yolo_devices=Mock(return_value=options or ["Auto", "CPU"]),
                            _global_yolo_devices_cache_ready=ready),
        yolo_device_var=Value("Auto"), _startup_ui_ready=True,
        _auto_device_label=devices.auto_device_label,
    )
    host._get_available_devices = lambda: devices.get_available_devices(host)
    host._normalize_selected_device = lambda **kw: devices.normalize_selected_device(host, **kw)
    return host


def test_device_ui_uses_shared_cache_without_importing_torch_or_probing_cuda():
    host = device_owner(["Auto", "CPU", "cuda:0 (GPU)"])
    original = builtins.__import__
    def guarded(name, *args, **kw):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise AssertionError("UI attempted to load inference framework")
        return original(name, *args, **kw)
    with patch.object(builtins, "__import__", guarded):
        assert devices.get_available_devices(host) == ["Auto", "CPU", "cuda:0 (GPU)"]
    host.app.get_available_yolo_devices.assert_called_once_with(allow_probe=False)


@pytest.mark.parametrize("value,expected", [("CPU", "CPU"), ("Auto", "Auto"),
    ("cuda:0", "cuda:0 (GPU)"), ("cuda:1", "Auto"), ("cuda:10", "cuda:10 (Other)")])
def test_detection_keeps_the_selected_gpu_from_shared_cache(value, expected):
    host = device_owner(["Auto", "CPU", "cuda:0 (GPU)", "cuda:10 (Other)"])
    host.app.get_global_yolo_device_choice = lambda: value
    assert devices.get_effective_detection_device_choice(host) == expected


@pytest.mark.parametrize("ready,fragment", [(False, "Konfiguracji"), (True, "CPU")])
def test_device_hint_distinguishes_unknown_hardware_from_no_gpu(ready, fragment):
    host = device_owner(ready=ready)
    host.det_device_hint_lbl = object()
    host._set_themed_label_state = Mock()
    devices.update_device_hint(host)
    text = host._set_themed_label_state.call_args.kwargs["text"]
    assert fragment in text
    if not ready:
        assert "nie wykry" not in text


@pytest.mark.parametrize("iteration", [1, 12])
def test_current_campaign_pz2_approval_still_controls_access_to_pz3(iteration):
    host = owner(False)
    host._campaign_graph_entry_context = {"force_pz2": True}
    with patch.object(navigation.CAMPAIGN, "get_active_project_name", return_value="pisto"), \
         patch.object(navigation.CAMPAIGN, "get_current_iteration_num", return_value=iteration), \
         patch.object(navigation.CAMPAIGN, "get_iteration_state") as state:
        for source_iteration, ready in ((iteration - 1, False), (iteration, True)):
            state.return_value = {"t06_contracts": {"pz2_char_boxes": {
                "fulfilled": True, "source_iteration": source_iteration}}}
            assert navigation.campaign_step3_can_open_pz3_from_current_context(host) is ready


@pytest.mark.parametrize("detect", [True, False])
def test_existing_subtabs_are_not_rebuilt_on_reentry(detect):
    host = SimpleNamespace(_detect_tab_built=True, _dataset_tab_built=True,
                           _build_detection_tab=Mock(), _build_cvat_tab=Mock())
    func = shared.ensure_detect_tab_built if detect else shared.ensure_dataset_tab_built
    assert func(host)
    host._build_detection_tab.assert_not_called()
    host._build_cvat_tab.assert_not_called()
