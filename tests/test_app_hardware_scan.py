import threading
import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui.app import AutoAnnotationApp
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


class _Scheduler:
    def __init__(self):
        self.owner = threading.get_ident()
        self.jobs = {}
        self.serial = 0

    def after(self, _delay, callback):
        assert threading.get_ident() == self.owner, "Tk called from worker"
        self.serial += 1
        self.jobs[self.serial] = callback
        return self.serial

    def after_cancel(self, job):
        assert threading.get_ident() == self.owner
        self.jobs.pop(job, None)

    def tick(self):
        job = next(iter(self.jobs))
        self.jobs.pop(job)()


class _Var:
    def __init__(self, value):
        self.owner = threading.get_ident()
        self.value = value

    def get(self):
        assert threading.get_ident() == self.owner, "Tk variable read from worker"
        return self.value

    def set(self, value):
        assert threading.get_ident() == self.owner, "Tk variable written from worker"
        self.value = value


class HardwareScanTests(unittest.TestCase):
    def setUp(self):
        self.app = AutoAnnotationApp.__new__(AutoAnnotationApp)
        self.app.root = _Scheduler()
        self.app.global_yolo_device_var = _Var("Auto")
        self.app._global_yolo_devices_cache = ["Auto", "CPU"]
        self.app._global_yolo_devices_cache_ready = False
        self.app._global_yolo_devices_scan_in_progress = False
        self.app._global_yolo_devices_last_error = ""
        self.app._global_yolo_device_saved_preference_raw = "Auto"
        self.app.update_status = Mock()
        self.app._save_global_yolo_device_preference = Mock(side_effect=self._remember)
        self.app._configuration_menu_button = object()
        self.app._menu_dropdown_owner = self.app._configuration_menu_button
        self.app._menu_dropdown_update_items = Mock()

    def _remember(self, value):
        self.app._global_yolo_device_saved_preference_raw = value

    def _complete(self):
        self.app._global_yolo_devices_scan_thread.join(timeout=2)
        self.assertFalse(self.app._global_yolo_devices_scan_thread.is_alive())
        self.app.root.tick()

    def test_completion_refreshes_open_menu_even_for_silent_startup_scan(self):
        app = self.app
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU", "cuda:0 (Test GPU)"])
        initial_items = app._build_configuration_menu_items()
        self.assertTrue(initial_items[-1]["disabled"])
        self.assertTrue(app._global_yolo_devices_scan_in_progress)
        self._complete()

        self.assertFalse(app._global_yolo_devices_scan_in_progress)
        self.assertEqual(app.root.jobs, {})
        app._menu_dropdown_update_items.assert_called_once()
        items = app._menu_dropdown_update_items.call_args.args[0]
        self.assertFalse(items[-1]["disabled"])
        self.assertTrue(any(row.get("label") == "GPU/CUDA 0 - Test GPU" for row in items))
        app._build_configuration_menu_items()
        app._scan_available_yolo_devices_sync.assert_called_once()
        app.update_status.assert_not_called()

    def test_scan_does_not_block_ui_or_start_duplicate_workers(self):
        release = threading.Event()
        entered = threading.Event()
        app = self.app

        def scan(progress):
            progress("cuda_availability")
            entered.set()
            release.wait(timeout=2)
            return ["Auto", "CPU"]

        app._scan_available_yolo_devices_sync = Mock(side_effect=scan)
        try:
            app._refresh_global_yolo_devices_async()
            self.assertTrue(entered.wait(timeout=1))
            app._refresh_global_yolo_devices_async()
            app._build_configuration_menu_items()
            app.root.tick()
            self.assertTrue(app._global_yolo_devices_scan_in_progress)
            self.assertEqual(len(app.root.jobs), 1)
            app._scan_available_yolo_devices_sync.assert_called_once()
        finally:
            release.set()
            self._complete()

    def test_timeout_ends_wait_and_late_result_cannot_overwrite_new_state(self):
        release = threading.Event()
        app = self.app
        app._scan_available_yolo_devices_sync = Mock(
            side_effect=lambda **_: release.wait(timeout=2) and ["Auto", "CPU", "cuda:0 (Late GPU)"],
        )
        try:
            with patch("auto_annotation_tool.gui.app.time.monotonic", return_value=100.0):
                app._refresh_global_yolo_devices_async(silent=True)
            with patch("auto_annotation_tool.gui.app.time.monotonic", return_value=116.0):
                app.root.tick()

            self.assertFalse(app._global_yolo_devices_scan_in_progress)
            self.assertIn("Timeout", app._global_yolo_devices_last_error)
            self.assertEqual(app.root.jobs, {})
            self.assertEqual(app.get_available_yolo_devices(), ["Auto", "CPU"])
            items = app._menu_dropdown_update_items.call_args.args[0]
            self.assertFalse(items[-1]["disabled"])
            self.assertTrue(any("nie udało" in row.get("label", "") for row in items))
            app._build_configuration_menu_items()
            app._refresh_global_yolo_devices_async()
            app._scan_available_yolo_devices_sync.assert_called_once()
        finally:
            release.set()
            app._global_yolo_devices_scan_thread.join(timeout=2)

        self.assertEqual(app.get_available_yolo_devices(), ["Auto", "CPU"])
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU"])
        app._refresh_global_yolo_devices_async()
        self._complete()
        self.assertEqual(app._global_yolo_devices_last_error, "")

    def test_error_keeps_confirmed_devices_and_preference(self):
        app = self.app
        app._global_yolo_devices_cache.append("cuda:0 (Confirmed GPU)")
        app.global_yolo_device_var.set("cuda:0 (Confirmed GPU)")
        app._scan_available_yolo_devices_sync = Mock(side_effect=RuntimeError("driver error"))
        app._refresh_global_yolo_devices_async()
        self._complete()

        self.assertFalse(app._global_yolo_devices_scan_in_progress)
        self.assertIn("driver error", app._global_yolo_devices_last_error)
        self.assertEqual(app.global_yolo_device_var.get(), "cuda:0 (Confirmed GPU)")
        self.assertIn("cuda:0 (Confirmed GPU)", app.get_available_yolo_devices())
        app._save_global_yolo_device_preference.assert_not_called()
        self.assertEqual(app.update_status.call_args.args[1], "warning")

    def test_confirmed_no_gpu_disables_gpu_and_clears_stale_saved_preference(self):
        app = self.app
        app._global_yolo_device_saved_preference_raw = "cuda:0 (Old GPU)"
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU"])
        app._refresh_global_yolo_devices_async()
        self._complete()
        items = app._build_configuration_menu_items()
        gpu = next(row for row in items if row.get("label", "").startswith("GPU/CUDA"))
        self.assertTrue(gpu["disabled"])
        self.assertIn("niedostępne", gpu["label"])
        self.assertEqual(app._global_yolo_device_saved_preference_raw, "Auto")
        self.assertEqual(app._global_yolo_devices_last_error, "")

    def test_restores_saved_gpu_only_after_confirmation(self):
        app = self.app
        app._global_yolo_device_saved_preference_raw = "cuda:0 (Saved GPU)"
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU", "cuda:0 (Current GPU)"])
        app._refresh_global_yolo_devices_async(silent=True)
        self.assertEqual(app.global_yolo_device_var.get(), "Auto")
        self._complete()
        self.assertEqual(app.global_yolo_device_var.get(), "cuda:0 (Current GPU)")

    def test_explicit_cpu_choice_during_scan_is_preserved(self):
        app = self.app
        app._global_yolo_device_saved_preference_raw = "cuda:0 (Saved GPU)"
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU", "cuda:0 (Current GPU)"])
        app._refresh_global_yolo_devices_async(silent=True)
        app.global_yolo_device_var.set("CPU")
        self._complete()
        self.assertEqual(app.global_yolo_device_var.get(), "CPU")

    def test_scan_does_not_replace_another_open_menu(self):
        app = self.app
        app._menu_dropdown_owner = object()
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU"])
        app._refresh_global_yolo_devices_async(silent=True)
        self._complete()
        app._menu_dropdown_update_items.assert_not_called()

    def test_shutdown_stops_polling_without_worker_touching_tk(self):
        app = self.app
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU"])
        app._refresh_global_yolo_devices_async()
        app._closing_in_progress = True
        self._complete()
        self.assertFalse(app._global_yolo_devices_scan_in_progress)
        self.assertEqual(app.root.jobs, {})
        app._menu_dropdown_update_items.assert_not_called()

    def test_thread_start_failure_unlocks_refresh(self):
        app = self.app
        with patch("auto_annotation_tool.gui.app.threading.Thread.start", side_effect=RuntimeError("thread start failed")):
            app._refresh_global_yolo_devices_async(silent=True)
        self.assertFalse(app._global_yolo_devices_scan_in_progress)
        self.assertEqual(app.root.jobs, {})
        self.assertIn("thread start failed", app._global_yolo_devices_last_error)

    def test_real_menu_updates_in_place_when_scan_finishes(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        root.withdraw()
        app = self.app
        app.root = root
        app.palette = get_theme_palette("dark_visual_cs")
        app._configuration_menu_button = tk.Button(root, text="Konfiguracja")
        app._menu_dropdown_owner = None
        app._menu_dropdown_update_items = None
        app._scan_available_yolo_devices_sync = Mock(return_value=["Auto", "CPU", "cuda:0 (Test GPU)"])
        native_toplevel = tk.Toplevel

        def hidden_toplevel(*args, **kwargs):
            popup = native_toplevel(*args, **kwargs)
            popup.withdraw()
            return popup

        try:
            items = app._build_configuration_menu_items()
            with patch("auto_annotation_tool.gui.app_menu_dropdown.tk.Toplevel", side_effect=hidden_toplevel):
                app._open_menu_dropdown(app._configuration_menu_button, items)
            popup = app._menu_dropdown
            body = popup.winfo_children()[0].winfo_children()[0]
            self.assertTrue(any("w toku" in str(row.cget("text")) for row in body.winfo_children() if isinstance(row, tk.Button)))
            deadline = time.monotonic() + 2
            while app._global_yolo_devices_scan_in_progress and time.monotonic() < deadline:
                root.update()
                time.sleep(0.01)

            self.assertFalse(app._global_yolo_devices_scan_in_progress)
            self.assertIs(app._menu_dropdown, popup)
            self.assertFalse(popup.winfo_ismapped())
            rows = [row for row in body.winfo_children() if isinstance(row, tk.Button)]
            gpu = next(row for row in rows if "GPU/CUDA 0" in str(row.cget("text")))
            self.assertEqual(str(gpu.cget("state")), "normal")
            self.assertFalse(any("w toku" in str(row.cget("text")) for row in rows))
            app._close_menu_dropdown()
            self.assertIsNone(app._menu_dropdown_update_items)
            self.assertFalse(popup.winfo_exists())
        finally:
            root.destroy()

    def test_profiles_are_published_on_ui_thread_only_after_success(self):
        app = self.app
        profile = {"raw": "cuda:0", "index": 0, "name": "GPU", "memory_gb": 8.0}
        def scan(progress):
            progress({"device_profiles": [profile]})
            return ["Auto", "CPU", "cuda:0 (GPU)"]
        app._scan_available_yolo_devices_sync = scan
        refreshed = []
        app.tabs = {"training": SimpleNamespace(_step4_train_tab_built=True,
                    _refresh_training_device_hint=lambda: refreshed.append(threading.get_ident()))}
        app._refresh_global_yolo_devices_async(silent=True)
        self.assertEqual(app.get_available_yolo_device_profiles(), [])
        self._complete()
        self.assertEqual(app.get_available_yolo_device_profiles(), [profile])
        self.assertEqual(refreshed, [threading.get_ident()])
        returned = app.get_available_yolo_device_profiles()
        returned[0]["memory_gb"] = 99
        self.assertEqual(app.get_available_yolo_device_profiles()[0]["memory_gb"], 8.0)

    def test_failed_scan_keeps_confirmed_profiles(self):
        app = self.app
        profile = {"raw": "cuda:0", "memory_gb": 4.0}
        app._global_yolo_device_profiles_cache = [profile]
        def scan(progress):
            progress({"device_profiles": [{"raw": "cuda:1", "memory_gb": 12.0}]})
            raise RuntimeError("driver failed")
        app._scan_available_yolo_devices_sync = scan
        app._refresh_global_yolo_devices_async(silent=True)
        self._complete()
        self.assertEqual(app.get_available_yolo_device_profiles(), [profile])

    def test_successful_no_gpu_scan_clears_old_profiles(self):
        app = self.app
        app._global_yolo_device_profiles_cache = [{"raw": "cuda:0", "memory_gb": 8.0}]
        def scan(progress):
            progress({"device_profiles": []})
            return ["Auto", "CPU"]
        app._scan_available_yolo_devices_sync = scan
        app._refresh_global_yolo_devices_async(silent=True)
        self._complete()
        self.assertEqual(app.get_available_yolo_device_profiles(), [])

    def test_shared_scan_collects_vram_without_writing_cache_from_worker(self):
        app = self.app
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               get_device_name=lambda i: "GPU",
                               get_device_properties=lambda i: SimpleNamespace(total_memory=8 * 1024 ** 3))
        events = []
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            labels = app._scan_available_yolo_devices_sync(progress=events.append)
        self.assertEqual(labels, ["Auto", "CPU", "cuda:0 (GPU)"])
        self.assertEqual(events[-1]["device_profiles"][0]["memory_gb"], 8.0)
        self.assertEqual(app.get_available_yolo_device_profiles(), [])


if __name__ == "__main__":
    unittest.main()
