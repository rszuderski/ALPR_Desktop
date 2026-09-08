import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui.tab_training import TrainingTab
from auto_annotation_tool.gui import z4_dataset_sources as sources
from auto_annotation_tool.gui import z4_device_runtime as devices
from auto_annotation_tool.gui import z4_training_metrics as metrics
from auto_annotation_tool.gui import z4_dataset_builder as builder
from auto_annotation_tool.gui import z4_shared_ui as shared


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def hardware_owner(ready=False, profiles=(), choice="Auto", error=""):
    host = TrainingTab.__new__(TrainingTab)
    host.app = SimpleNamespace(
        _global_yolo_devices_cache_ready=ready, _global_yolo_devices_last_error=error,
        get_available_yolo_device_profiles=Mock(side_effect=lambda: [dict(p) for p in profiles]),
        get_global_yolo_device_choice=lambda: choice,
    )
    host._training_device_profiles = []
    host._training_device_label_map = {"Auto": "auto", "CPU": "cpu"}
    host._training_auto_device_label = "Auto"
    host._training_cpu_device_label = "CPU"
    host.device_var = Value(choice)
    host._get_selected_training_target = Mock(return_value="plate")
    host._get_training_dataset_profile = Mock(return_value={"train_images": 800, "total_images": 1000})
    host._resolve_selected_training_base_model_profile = Mock(return_value={"bucket": "n", "label": "yolo26n"})
    host._refresh_training_recommendation_table = Mock()
    host._refresh_training_start_state = Mock()
    host.epochs_var, host.batch_var, host.imgsz_var, host.lr0_var = Value(123), Value(7), Value(416), Value(0.002)
    return host


GPU = {"raw": "cuda:0", "index": 0, "name": "Test GPU", "memory_gb": 4.0}


def test_device_profile_lookup_never_imports_torch_or_touches_cuda():
    host = hardware_owner(True, [GPU])
    with patch.object(devices, "get_torch_module", side_effect=AssertionError("CUDA on UI thread")) as probe:
        assert host._scan_training_cuda_devices() == [GPU]
        assert host._get_effective_training_device_profile("cuda:0") == ("cuda:0", GPU)
        assert host._get_effective_training_device_profile("CPU") == ("cpu", None)
    probe.assert_not_called()


def test_updated_shared_profile_replaces_stale_local_profile():
    profiles = [dict(GPU)]
    host = hardware_owner(True, profiles)
    assert host._get_effective_training_device_profile("Auto")[1]["memory_gb"] == 4.0
    profiles[0]["memory_gb"] = 12.0
    assert host._get_effective_training_device_profile("Auto")[1]["memory_gb"] == 12.0
    profiles.clear()
    assert host._get_effective_training_device_profile("Auto") == ("cpu", None)


@pytest.mark.parametrize("ready,error,profiles", [(False, "", []), (True, "driver timeout", [GPU]),
    (True, "", [{**GPU, "memory_gb": 0.0}])])
def test_unknown_hardware_does_not_overwrite_parameters_or_read_model_dataset(ready, error, profiles):
    host = hardware_owner(ready, profiles, error=error)
    assert host._get_training_device_recommendation("Auto")["ready"] is False
    host._apply_training_recommended_start_params()
    assert [var.get() for var in (host.epochs_var, host.batch_var, host.imgsz_var, host.lr0_var)] == [123, 7, 416, 0.002]
    host._get_training_dataset_profile.assert_not_called()
    host._resolve_selected_training_base_model_profile.assert_not_called()


@pytest.mark.parametrize("choice,ready,profiles,expected", [
    ("CPU", False, [], "cpu"), ("Auto", True, [], "cpu"),
    ("Auto", True, [GPU], "cuda:0"), ("cuda:0", True, [GPU], "cuda:0"),
])
def test_explicit_recommendation_keeps_cpu_gpu_semantics(choice, ready, profiles, expected):
    host = hardware_owner(ready, profiles, choice)
    result = host._get_training_device_recommendation(choice)
    assert result.get("ready", True)
    assert result["effective_raw"] == expected
    host._apply_training_recommended_start_params()
    assert host.epochs_var.get() == 123
    assert host.batch_var.get() == result["batch"]
    assert host.imgsz_var.get() == result["imgsz"]
    assert host.lr0_var.get() == result["lr0"]


def test_hint_does_no_work_before_pz2_exists():
    host = hardware_owner()
    host._step4_train_tab_built = False
    host._get_available_devices = Mock(side_effect=AssertionError("hidden view query"))
    host.device_combo = Mock()
    devices._refresh_training_device_hint(host)
    host._get_available_devices.assert_not_called()
    host._refresh_training_recommendation_table.assert_not_called()


def test_no_recommendation_calculation_without_table_widgets():
    host = SimpleNamespace(_train_recommendation_cells=[],
                           _build_training_recommendation_rows=Mock(side_effect=AssertionError("hidden table")))
    metrics._refresh_training_recommendation_table(host)
    host._build_training_recommendation_rows.assert_not_called()


@pytest.mark.parametrize("built", [False, True])
def test_deferred_model_refresh_is_lazy_and_does_not_apply_parameters(built):
    scheduled = []
    host = Mock()
    host._step4_train_tab_built = built
    host._step4_deferred_model_refresh_job = None
    host.frame.after.side_effect = lambda delay, callback: scheduled.append(callback) or 1
    metrics._schedule_step4_deferred_model_refresh(host)
    scheduled.pop()()
    assert host._refresh_base_model_choices.call_count == int(built)
    assert host._refresh_training_recommendation_table.call_count == int(built)
    host._apply_training_recommended_start_params.assert_not_called()


def test_model_refresh_preserves_user_parameters():
    host = Mock()
    builder._on_base_model_change(host)
    host._refresh_training_recommendation_table.assert_called_once()
    host._apply_training_recommended_start_params.assert_not_called()


@pytest.mark.parametrize("campaign,target", [(False, "plate"), (False, "char"), (True, "plate"), (True, "char")])
def test_accepting_dataset_keeps_target_and_does_not_apply_training_recommendation(campaign, target):
    host = Mock()
    host._get_locked_campaign_training_target.return_value = target if campaign else ""
    host.dataset_var = Value()
    with patch.object(shared.CAMPAIGN, "get_active_project_name", return_value="pisto" if campaign else ""):
        assert shared.accept_training_input_context(host, source="pz1", target=target, dataset_path="selected-dataset")
    assert host.dataset_var.get() == "selected-dataset"
    assert host._step4_dataset_mode == target
    assert host._step4_train_unlocked
    host._apply_training_recommended_start_params.assert_not_called()


def test_same_storage_keeps_existing_trainer_and_history(tmp_path):
    host = TrainingTab.__new__(TrainingTab)
    history = SimpleNamespace(history_dir=tmp_path)
    trainer = SimpleNamespace(is_training=True)
    host.history, host.trainer = history, trainer
    host._bind_trainer_callbacks = Mock()
    with patch("auto_annotation_tool.gui.tab_training.CAMPAIGN.get_active_project_name", return_value=""), \
         patch("auto_annotation_tool.gui.tab_training.CONFIG.get_training_runs_dir", return_value=tmp_path), \
         patch("auto_annotation_tool.gui.tab_training.TrainingHistory", side_effect=AssertionError("unneeded history construction")):
        host._rebind_free_mode_training_storage("plate", reload_history=False)
    assert host.history is history and host.trainer is trainer
    host._bind_trainer_callbacks.assert_not_called()


def make_dataset(root):
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        for name in ("a.jpg", "b.PNG", "skip.txt"):
            (root / "images" / split / name).touch()
        (root / "images" / split / "directory.jpg").mkdir()
    (root / "data.yaml").write_text("names: {0: A}\ntrain: images/train\nval: images/val\n")
    return root


def test_split_counts_are_exact_cached_and_invalidated_on_directory_change(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    host = SimpleNamespace()
    expected = {"train": 2, "val": 2, "test": 2, "total": 6}
    assert sources._get_dataset_split_image_counts(host, root) == expected
    with patch.object(sources.os, "scandir", side_effect=AssertionError("cache should avoid scanning")):
        result = sources._get_dataset_split_image_counts(host, root / "data.yaml")
        assert result == expected
        result["train"] = 999
        assert sources._get_dataset_split_image_counts(host, root) == expected
    split = root / "images" / "train"
    stamp = split.stat().st_mtime_ns
    (split / "new.jpeg").touch()
    os.utime(split, ns=(stamp + 10000000, stamp + 10000000))
    assert sources._get_dataset_split_image_counts(host, root) == {**expected, "train": 3, "total": 7}


def test_scandir_errors_do_not_poison_count_cache(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    host = SimpleNamespace()
    with patch.object(sources.os, "scandir", side_effect=PermissionError("temporary")):
        assert sources._get_dataset_split_image_counts(host, root)["total"] == 0
    assert sources._get_dataset_split_image_counts(host, root)["total"] == 6


def test_variant_list_still_blocks_incomplete_mz_and_does_not_select_another_dataset(tmp_path):
    valid = make_dataset(tmp_path / "valid")
    blocked = make_dataset(tmp_path / "blocked")
    (blocked / "mz_training_variant_manifest.json").write_text(json.dumps({"ready_for_training": False}))
    host = Mock()
    host._get_selected_training_target.return_value = "char"
    host._find_ready_dataset_candidates.return_value = [(valid, "char", 1), (blocked, "char", 2)]
    host._get_dataset_split_image_counts.side_effect = lambda path: sources._get_dataset_split_image_counts(SimpleNamespace(), path)
    host._format_workspace_relative_path.side_effect = str
    with patch.object(sources.CAMPAIGN, "get_active_project_name", return_value=""):
        variants = sources._get_free_dataset_variant_choices(host)
    assert [item["path"] for item in variants] == [str(valid)]
    host.dataset_var.set.assert_not_called()


def profile_owner(path, ui_queue):
    host = SimpleNamespace()
    host.path = path
    host._resolve_training_dataset_yaml_path = lambda: host.path
    host._ui = lambda callback: ui_queue.append(callback)
    host._refresh_training_dataset_quality_summary = Mock()
    host._refresh_training_recommendation_table = Mock()
    host._refresh_training_execution_summary = Mock()
    return host


def test_dataset_profile_counts_in_worker_and_publishes_once_on_ui_queue(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    for split in ("train", "val", "test"):
        label_dir = dataset / "labels" / split
        label_dir.mkdir(parents=True)
        (label_dir / "a.txt").write_text("0 .5 .5 .2 .2\n0 .6 .6 .2 .2\n")
    ui, workers = [], []
    host = profile_owner(dataset / "data.yaml", ui)
    with patch.object(sources.threading, "Thread", side_effect=lambda **kw: SimpleNamespace(start=lambda: workers.append(kw["target"]))):
        assert sources._get_training_dataset_profile(host)["pending"]
        assert sources._get_training_dataset_profile(host)["pending"]
        assert len(workers) == 1
        workers.pop()()
        assert not hasattr(host, "_training_dataset_profile_cache")
        assert len(ui) == 1
        ui.pop()()
        result = sources._get_training_dataset_profile(host)
    assert result["total_images"] == 6
    assert result["total_objects"] == 6
    assert result["total_label_files"] == 3
    assert not result.get("pending")
    assert not host._training_dataset_profile_job


def test_stale_dataset_profile_does_not_replace_changed_selection(tmp_path):
    a = make_dataset(tmp_path / "a") / "data.yaml"
    b = make_dataset(tmp_path / "b") / "data.yaml"
    ui, workers = [], []
    host = profile_owner(a, ui)
    with patch.object(sources.threading, "Thread", side_effect=lambda **kw: SimpleNamespace(start=lambda: workers.append(kw["target"]))):
        sources._get_training_dataset_profile(host)
        host.path = b
        sources._get_training_dataset_profile(host)
        assert len(workers) == 1
        workers.pop()()
        ui.pop()()
        assert not hasattr(host, "_training_dataset_profile_cache")
        assert sources._get_training_dataset_profile(host)["pending"]
        workers.pop()()
        ui.pop()()
        assert not sources._get_training_dataset_profile(host).get("pending")
    assert str(b.parent.resolve()) in host._training_dataset_profile_cache_key


def test_cleared_selection_discards_in_flight_profile(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    ui, workers = [], []
    host = profile_owner(dataset / "data.yaml", ui)
    with patch.object(sources.threading, "Thread", side_effect=lambda **kw: SimpleNamespace(start=lambda: workers.append(kw["target"]))):
        sources._get_training_dataset_profile(host)
        host.path = None
        workers.pop()()
        ui.pop()()
        assert sources._get_training_dataset_profile(host)["total_images"] == 0
    assert not hasattr(host, "_training_dataset_profile_cache")
    assert not host._training_dataset_profile_job


def test_profile_thread_start_failure_finishes_without_retry_loop(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    host = profile_owner(dataset / "data.yaml", [])
    with patch.object(sources.threading.Thread, "start", side_effect=RuntimeError("cannot start")) as start:
        result = sources._get_training_dataset_profile(host)
        assert result["error"] == "cannot start"
        assert not result.get("pending")
        assert sources._get_training_dataset_profile(host) == result
        start.assert_called_once()
    assert not host._training_dataset_profile_job


def test_pending_material_is_not_reported_as_an_empty_or_bad_dataset():
    host = TrainingTab.__new__(TrainingTab)
    host._get_selected_training_target = lambda: "char"
    host._get_training_dataset_quality_thresholds = lambda target: {}
    host._resolve_training_dataset_yaml_path = lambda: Path("dataset/data.yaml")
    host._get_training_dataset_profile = lambda path: {"pending": True}
    result = host._build_training_dataset_quality_summary()
    assert result["status"] == "Analiza w toku"
    assert result["tone"] == "muted"


def test_pending_profile_prevents_applying_incomplete_gpu_recommendation():
    host = hardware_owner(True, [GPU])
    host._get_training_dataset_profile.return_value = {"pending": True}
    result = host._get_training_device_recommendation("Auto")
    assert result["ready"] is False
    assert result["effective_raw"] == "cuda:0"
    host._apply_training_recommended_start_params()
    assert host.batch_var.get() == 7
    host._resolve_selected_training_base_model_profile.assert_not_called()
