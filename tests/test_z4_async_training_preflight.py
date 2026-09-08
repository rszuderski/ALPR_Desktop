import tempfile
import threading
import time
import queue
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z4_training_runtime, z4_training_metrics, z4_ui_runtime
from auto_annotation_tool.training import trainer as trainer_module
from auto_annotation_tool.training.training_history import TrainingHistory, TrainingStatus


class _Var:
    def __init__(self, value=""):
        self.value = value
        self.owner = threading.get_ident()

    def get(self):
        assert threading.get_ident() == self.owner
        return self.value

    def set(self, value):
        assert threading.get_ident() == self.owner
        self.value = value


class _Widget:
    def __init__(self):
        self.config = {}
        self.owner = threading.get_ident()

    def configure(self, **kwargs):
        assert threading.get_ident() == self.owner
        self.config.update(kwargs)


class _Frame:
    # Keep the interpreter on the test runner's main thread for the whole suite.
    # Per-test Tcl instances can otherwise be collected by a later Python worker.
    _shared_tcl = None

    def __init__(self):
        self.after_calls = []
        if _Frame._shared_tcl is None:
            _Frame._shared_tcl = tk.Tcl()
        self.tk = _Frame._shared_tcl
        self.owner = threading.get_ident()
        self.alive = True
        self.bindings = {}

    def after(self, delay_ms, callback=None, *args):
        assert threading.get_ident() == self.owner
        self.after_calls.append((delay_ms, callback, args))
        return self.tk.after(delay_ms, callback, *args)

    def after_cancel(self, job):
        assert threading.get_ident() == self.owner
        self.tk.after_cancel(job)

    def bind(self, name, callback, **_kwargs):
        self.bindings[name] = callback

    def winfo_exists(self):
        assert threading.get_ident() == self.owner
        return self.alive

    def pump(self):
        self.tk.update()

    def destroy(self):
        if not self.alive:
            return
        self.alive = False
        self.bindings["<Destroy>"](SimpleNamespace(widget=self))
        for job in self.tk.tk.call("after", "info"):
            self.tk.after_cancel(job)
        self.bindings.clear()
        self.after_calls.clear()


class _App:
    tabs = {}


class _SlowTrainer:
    def __init__(self, delay_s=0.35):
        self.delay_s = delay_s
        self.calls = 0
        self.done = threading.Event()
        self.validation_threads = []
        self.finished_at = None
        self.started_at = None
        self.error = None
        self.shutdown = Mock()

    def validate_dataset(self, _dataset):
        self.validation_threads.append(threading.get_ident())
        time.sleep(self.delay_s)
        return True, "Dataset OK", {"train_images": 1, "val_images": 1, "test_images": 0}

    def start_training(self, **kwargs):
        self.calls += 1
        self.started_at = time.perf_counter()
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback("Wolny testowy preflight", 45.0, "unit-test")
        self.validate_dataset(kwargs["dataset_path"])
        self.finished_at = time.perf_counter()
        self.done.set()
        if self.error:
            raise self.error
        return "run_async_001"

    def resume_training(self, run_id, *, progress_callback=None):
        return self.start_training(dataset_path="fixture", progress_callback=progress_callback)


class _Host:
    def __init__(self, dataset_root: Path):
        self.frame = _Frame()
        self.app = _App()
        self.trainer = _SlowTrainer()
        self.name_var = _Var("Async run")
        self.base_model_var = _Var("yolo11n")
        self.base_custom_var = _Var("")
        self.device_var = _Var("cpu")
        self.dataset_var = _Var(str(dataset_root))
        self._dataset_variant_choices = [{"path": str(dataset_root)}]
        self.btn_start_train = _Widget()
        self.btn_pause_train = _Widget()
        self.btn_stop_train = _Widget()
        self.train_progress_label = _Widget()
        self.train_resource_label = _Widget()
        self._training_completion_poll_job = None
        self._logs = []
        self._dataset_root = dataset_root
        self._ui_dispatch_queue = queue.Queue()
        self._ui_dispatch_after_id = None
        z4_ui_runtime._ensure_ui_dispatch_pump(self)

    def _ui(self, fn):
        return z4_ui_runtime._ui(self, fn)

    def _drain_ui_dispatch_queue(self):
        return z4_ui_runtime._drain_ui_dispatch_queue(self)

    def _get_pinned_step4_result_state(self):
        return {}

    def _clear_step4_guidance(self):
        return None

    def _set_step4_process_console_text(self, text):
        self._logs.append(text)

    def _set_training_metric_interpretation(self, text):
        self._metric_interpretation = text

    def _set_training_widget_text(self, widget, text):
        widget.configure(text=text)

    def _set_training_resource_sample(self, _sample):
        return None

    def _validate_active_training_source_for_pz2(self):
        raise AssertionError("Full GUI validation must not run on Start")

    def _looks_like_char_classification_dataset(self, _path):
        return False

    def _get_pose_dataset_size_warning(self, _dataset_root, _stats):
        return ""

    def _infer_dataset_target(self, _dataset_root):
        return "char"

    def _get_selected_training_target(self):
        return "char"

    def _format_training_target_label(self, target):
        return target

    def _rebind_free_mode_training_storage(self, target):
        self._target = target

    def _is_custom_base_model_key(self, _key):
        return False

    def _resolve_selected_training_base_model_display(self):
        return "YOLO test"

    def _resolve_selected_training_base_model_info(self, **_kwargs):
        return None, {}

    def _device_to_ultralytics(self, value):
        return value

    def _validate_training_base_model_target_compatibility(self, **_kwargs):
        return True, ""

    def _is_pose_base_model(self, _base_key, _base_model):
        return False

    def _resolve_step4_fine_tune_parent_run(self):
        return None

    def _safe_training_int_value(self, name, default=0, minimum=0):
        values = {"epochs_var": 1, "batch_var": 1, "imgsz_var": 320}
        return max(int(values.get(name, default)), int(minimum))

    def _safe_training_float_value(self, _name, default=0.01, minimum=0.0):
        return max(float(default), float(minimum))

    def _normalize_training_device_choice(self, value):
        return value

    def _get_effective_training_device_profile(self, value):
        return value, None

    def _get_training_gpu_capacity_block_reason(self, **_kwargs):
        return ""

    def _append_train_log(self, message):
        self._logs.append(message)

    def _release_gpu_resources_before_training(self):
        self._logs.append("[TEST] release gpu")

    def _begin_step4_operation(self, _owner, _label):
        self._locked = True
        return True

    def _end_step4_operation(self, _owner):
        self._ended = True
        self._locked = False

    def _is_training_configuration_ready(self):
        return z4_training_metrics._validate_training_source_lightweight(self)["ok"]

    def _refresh_training_start_state(self):
        return z4_training_metrics._refresh_training_start_state(self)

    def _set_train_progress_values(self, **kwargs):
        self._progress = kwargs

    def _remember_campaign_plate_training_source(self, _dataset_root):
        return None

    def _reset_training_runtime_progress(self):
        return None

    def _set_train_live_metrics(self, _metrics):
        return None

    def _set_training_running_ui_state(self, run_id, *, status_text=None):
        self._running_state = (run_id, status_text)

    def _remember_campaign_training_run_in_registry(self, **_kwargs):
        return None

    def _refresh_step4_campaign_navigation_ui(self):
        return None

    def get_campaign_training_target(self):
        return "char"

    def _poll_training_completion(self):
        return None


class Z4AsyncTrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name) / "dataset"
        self.dataset.mkdir()
        (self.dataset / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: char\n", encoding="utf-8",
        )
        self.host = _Host(self.dataset)
        self.addCleanup(self.host.frame.destroy)
        for target, replacement in (
            ("YOLO_AVAILABLE", True),
            ("cleanup_gpu_memory", Mock()),
        ):
            patcher = patch.object(z4_training_runtime, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(z4_training_runtime.CAMPAIGN, "get_active_project_name", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(z4_training_runtime.messagebox, "showerror")
        self.showerror = patcher.start()
        self.addCleanup(patcher.stop)

    def _wait_for_result(self):
        deadline = time.perf_counter() + 3
        while getattr(self.host, "_training_start_in_progress", False) and time.perf_counter() < deadline:
            self.host.frame.pump()
            time.sleep(0.005)
        self.assertFalse(self.host._training_start_in_progress)

    def test_start_training_returns_immediately_and_blocks_double_start(self):
        host = self.host
        started_at = time.perf_counter()
        z4_training_runtime._start_training(host)
        elapsed = time.perf_counter() - started_at
        z4_training_runtime._start_training(host)
        self.assertLess(elapsed, host.trainer.delay_s)
        self.assertTrue(host._training_start_in_progress)
        self.assertEqual(host.btn_start_train.config.get("state"), "disabled")
        self._wait_for_result()
        self.assertEqual(host.trainer.calls, 1)
        self.assertEqual(host.current_run_id, "run_async_001")
        self.assertIsNone(host._training_preflight_thread)

    def test_start_does_not_enumerate_on_ui_thread_and_validates_once(self):
        host = self.host
        main_thread = threading.get_ident()
        def no_scan_on_main(*_args, **_kwargs):
            self.assertNotEqual(threading.get_ident(), main_thread)
            return iter(())
        with patch.object(Path, "rglob", side_effect=no_scan_on_main), patch.object(Path, "iterdir", side_effect=no_scan_on_main):
            z4_training_runtime._start_training(host)
            self._wait_for_result()
        self.assertEqual(len(host.trainer.validation_threads), 1)
        self.assertNotEqual(host.trainer.validation_threads[0], main_thread)

    def test_real_dispatch_queues_callback_for_main_thread_exactly_once(self):
        host = self.host
        callback_threads, worker_threads = [], []
        def worker():
            worker_threads.append(threading.get_ident())
            host._ui(lambda: callback_threads.append(threading.get_ident()))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(1)
        self.assertEqual(callback_threads, [])
        host._drain_ui_dispatch_queue()
        host._drain_ui_dispatch_queue()
        self.assertEqual(callback_threads, [threading.get_ident()])
        self.assertNotEqual(callback_threads[0], worker_threads[0])

    def test_heartbeat_runs_during_one_second_preflight(self):
        host = self.host
        host.trainer.delay_s = 1.0
        heartbeats = []
        host.frame.after(100, lambda: heartbeats.append(time.perf_counter()))
        z4_training_runtime._start_training(host)
        self._wait_for_result()
        self.assertEqual(len(heartbeats), 1)
        self.assertLess(host.trainer.started_at, heartbeats[0])
        self.assertLess(heartbeats[0], host.trainer.finished_at)

    def test_async_failure_restores_start_and_releases_operation_lock(self):
        host = self.host
        host.trainer.error = RuntimeError("snapshot failed")
        z4_training_runtime._start_training(host)
        self._wait_for_result()
        self.assertIsNone(host._training_preflight_thread)
        self.assertFalse(host._locked)
        self.assertEqual(host.btn_start_train.config.get("state"), "normal")
        self.assertFalse(hasattr(host, "current_run_id"))
        self.showerror.assert_called_once()
        self.assertIn("snapshot failed", self.showerror.call_args.args[1])

    def test_destroyed_host_discards_worker_result_and_shuts_down_started_trainer(self):
        host = self.host
        z4_training_runtime._start_training(host)
        thread = host._training_preflight_thread
        host.frame.destroy()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(host._ui_dispatch_queue.empty())
        self.assertIsNone(host._ui_dispatch_after_id)
        host.trainer.shutdown.assert_called_once()
        self.assertFalse(hasattr(host, "current_run_id"))
        self.showerror.assert_not_called()

    def test_main_thread_callback_is_also_discarded_after_destroy(self):
        callback = Mock()
        self.host._ui(callback)
        self.host.frame.destroy()
        self.host.frame.pump()
        callback.assert_not_called()
        self.assertFalse(self.host._ui(callback))

    def test_progress_logs_stage_changes_not_percentage_updates(self):
        host = self.host
        for pct in (10, 11, 12, 20):
            z4_training_runtime._update_training_preflight_progress(host, "Dataset", pct, str(pct))
        z4_training_runtime._update_training_preflight_progress(host, "Model", 30)
        self.assertEqual(len([line for line in host._logs if line.startswith("[PREFLIGHT]")]), 2)
        self.assertEqual(host._progress["overall"], 30)

    def test_detach_is_main_thread_and_unload_is_in_worker(self):
        host = self.host
        main_thread = threading.get_ident()
        unload_threads = []
        model = SimpleNamespace(unload_models=lambda: unload_threads.append(threading.get_ident()))
        annotation = SimpleNamespace(annotator=model)
        host.app.tabs = {"annotation": annotation}
        z4_training_runtime._start_training(host)
        self.assertTrue(host._locked)
        self.assertIsNone(annotation.annotator)
        self._wait_for_result()
        self.assertEqual(len(unload_threads), 1)
        self.assertNotEqual(unload_threads[0], main_thread)

    def test_thread_start_failure_restores_start(self):
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start thread")):
            z4_training_runtime._start_training(self.host)
        self.assertFalse(self.host._training_start_in_progress)
        self.assertIsNone(self.host._training_preflight_thread)
        self.assertFalse(self.host._locked)
        self.assertEqual(self.host.trainer.calls, 0)
        self.assertEqual(self.host.btn_start_train.config.get("state"), "normal")

    def test_real_preflight_validates_once_and_rolls_back_failed_spawn(self):
        host = self.host
        for split in ("train", "val", "test"):
            images = self.dataset / "images" / split
            labels = self.dataset / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / f"{split}.jpg").write_bytes(b"image fixture")
            (labels / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="ascii")
        checkpoint = Path(self.temp.name) / "base.pt"
        checkpoint.write_bytes(b"checkpoint fixture")
        history = TrainingHistory(Path(self.temp.name) / "history")
        trainer = trainer_module.YOLOPoseTrainer(history=history)
        host.trainer = trainer
        main_thread = threading.get_ident()
        validation_threads = []
        actual_validate = trainer.validate_dataset
        def validate(path):
            validation_threads.append(threading.get_ident())
            return actual_validate(path)
        trainer.validate_dataset = Mock(side_effect=validate)
        trainer._reset_runtime_state = Mock()
        trainer._materialize_official_pretrained_checkpoint = Mock(return_value=str(checkpoint))
        with patch.object(trainer_module, "YOLO_AVAILABLE", True), \
             patch.object(trainer_module, "get_yolo_class", return_value=object), \
             patch.object(trainer_module, "_patch_ultralytics_save_model_closed_file_bug"), \
             patch.object(trainer_module.subprocess, "Popen", side_effect=OSError("spawn failed")) as spawn:
            z4_training_runtime._start_training(host)
            self._wait_for_result()

        trainer.validate_dataset.assert_called_once()
        self.assertNotEqual(validation_threads[0], main_thread)
        spawn.assert_called_once()
        self.assertTrue(spawn.call_args.kwargs["stdout"].closed)
        self.assertFalse(trainer.is_training)
        self.assertIsNone(trainer._worker_process)
        self.assertEqual(len(history.runs), 1)
        run = next(iter(history.runs.values()))
        self.assertEqual(run.status, TrainingStatus.FAILED.value)
        self.assertIn("spawn failed", run.error_message)
        self.assertTrue(run.training_dataset_snapshot)
        self.assertTrue(run.input_checkpoint_snapshot["sha256"])
        self.assertEqual(host._last_training_source.stats.split_counts()["total"], 3)
        self.assertEqual(host.btn_start_train.config["state"], "normal")

    def test_shared_operation_lock_blocks_z2_and_z3_during_training(self):
        from auto_annotation_tool.gui.app import AutoAnnotationApp
        from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
        app = AutoAnnotationApp.__new__(AutoAnnotationApp)
        app._processing_state_lock = threading.Lock()
        app._exclusive_operation = None
        app._refresh_processing_state = Mock()
        app.append_global_terminal = Mock()
        self.host.app = app
        self.host._begin_step4_operation = lambda owner, label: z4_ui_runtime._begin_step4_operation(self.host, owner, label)
        z4_training_runtime._start_training(self.host)
        for owner in ("z2.annotation.run", "pz2.fast_test.run", "pz2.ocr_ranking.run"):
            ok, message = app.try_begin_exclusive_operation(owner, "test inference")
            self.assertFalse(ok)
            self.assertTrue(message)
        z3 = SimpleNamespace(app=app)
        with patch("auto_annotation_tool.gui.tab_character_annotation.messagebox.showinfo"):
            self.assertFalse(CharacterAnnotationTab._lock_ui_for_testing(z3, "pz2.fast_test.run", "test"))
        self._wait_for_result()
        ok, _ = app.try_begin_exclusive_operation("z2.annotation.run", "test inference")
        self.assertFalse(ok)
        app.end_exclusive_operation("z4.training.run")
        ok, _ = app.try_begin_exclusive_operation("z2.annotation.run", "test inference")
        self.assertTrue(ok)

    def test_preparing_cockpit_does_not_scan_or_describe_models_again(self):
        self.host._training_start_in_progress = True
        self.host._last_training_cockpit_summary = {"cards": [("Dataset", "DS-1", "fixture")]}
        self.host._resolve_training_dataset_yaml_path = Mock(side_effect=AssertionError("UI read during preflight"))
        summary = z4_training_metrics._build_training_cockpit_summary(self.host, ready=False)
        self.assertEqual(summary["status"], "Przygotowanie treningu")
        self.assertEqual(summary["cards"], self.host._last_training_cockpit_summary["cards"])

    def test_resume_uses_async_dispatch_and_keeps_double_resume_blocked(self):
        checkpoint = Path(self.temp.name) / "last.pt"
        checkpoint.write_bytes(b"fixture")
        run = SimpleNamespace(id="paused_run", last_weights=str(checkpoint), metrics_history=[])
        host = self.host
        host._selected_run = lambda: run
        host._is_history_run_resume_allowed = lambda _run: True
        host._does_history_run_match_active_campaign_target = lambda _run: True
        z4_training_runtime._resume_selected_run(host)
        z4_training_runtime._resume_selected_run(host)
        self.assertTrue(host._training_start_in_progress)
        self._wait_for_result()
        self.assertEqual(host.trainer.calls, 1)
        self.assertEqual(host.current_run_id, "run_async_001")
        self.assertIsNone(host._training_preflight_thread)

    def test_lightweight_validation_preserves_variant_and_target_contract(self):
        host = self.host
        host._dataset_variant_choices = []
        self.assertFalse(z4_training_metrics._validate_training_source_lightweight(host)["ok"])
        with patch.object(z4_training_metrics.CAMPAIGN, "get_active_project_name", return_value="demo"):
            self.assertTrue(z4_training_metrics._validate_training_source_lightweight(host)["ok"])
            host._infer_dataset_target = lambda _path: "plate"
            self.assertFalse(z4_training_metrics._validate_training_source_lightweight(host)["ok"])

    def test_lightweight_validation_does_not_bypass_failed_augmentation_manifest(self):
        (self.dataset / "mz_training_variant_manifest.json").write_text(
            '{"ready_for_training": false, "completion_status": "PARTIAL_AUGMENTATION"}', encoding="utf-8",
        )
        z4_training_runtime._start_training(self.host)
        self.assertEqual(self.host.trainer.calls, 0)
        self.showerror.assert_called_once()

    def test_unknown_checkpoint_is_not_loaded_by_lightweight_model_check(self):
        checkpoint = Path(self.temp.name) / "unknown.pt"
        checkpoint.write_bytes(b"fixture")
        with patch("auto_annotation_tool.validators.get_yolo_class", side_effect=AssertionError("heavy model load")):
            task = z4_training_metrics._training_base_model_task_lightweight(self.host, "Custom", str(checkpoint))
        self.assertIsNone(task)


if __name__ == "__main__":
    unittest.main()
