#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Główna aplikacja GUI.
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from datetime import datetime
import threading
import time
import queue

from ..config import CONFIG, logger, TK_AVAILABLE, SESSION
from ..icons import IconManager
from .web_slim_scrollbar import blend_hex_colors

# Importy zakładek
from .tab_annotation import AnnotationTab
from .tab_character_annotation import CharacterAnnotationTab
from .tab_training import TrainingTab
from .tab_campaign import CampaignTab
from .help_manager import HELP
from .free_mode_assistant import (
    FreeModeAssistantContext,
    FreeModeAssistantOverlay,
    get_mobile_export_assistant_context,
)
from .lazy_notebook_tab import _LazyNotebookTab
from .app_delegates import bind_app_delegates
from .app_theme_runtime import bind_app_theme_runtime
from .app_theme_definitions import THEME_DEFINITIONS, get_theme_palette

try:
    from .tab_help import HelpTab
except ImportError:
    HelpTab = None


APP_AUTHOR = "R. Szuderski"
HARDWARE_SCAN_TIMEOUT_SECONDS = 15.0


class _MobileExportMenuHost:
    """Lightweight adapter for opening mobile export without building the Z4 UI."""

    def __init__(self, app):
        self.app = app
        self.frame = app.root
        self.history = None
        self._mobile_export_center_dialog = None

    @staticmethod
    def _safe_model_export_slug(value: str, fallback: str = "model") -> str:
        import re

        text = str(value or "").strip() or fallback
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
        return (slug or fallback)[:48]

    @staticmethod
    def _format_training_target_label(target: str) -> str:
        normalized = CONFIG.normalize_task_target(target)
        labels = {
            "plate": "tablice (YOLO Pose)",
            "char": "znaki tablic (YOLO Detect)",
            "vehicle": "pojazdy (YOLO Detect)",
        }
        return labels.get(normalized, "znaki tablic (YOLO Detect)")

    @staticmethod
    def _json_safe_training_value(value):
        from pathlib import Path

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _MobileExportMenuHost._json_safe_training_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_MobileExportMenuHost._json_safe_training_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _training_metric_float(value) -> float | None:
        if value in (None, "", "-"):
            return None
        try:
            numeric = float(str(value).strip().replace(",", "."))
        except Exception:
            return None
        return numeric if numeric == numeric else None

    @classmethod
    def _training_metric_value(cls, mapping: dict | None, keys: tuple[str, ...]) -> float | None:
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            if key in mapping:
                value = cls._training_metric_float(mapping.get(key))
                if value is not None:
                    return value
        lowered = {str(k or "").strip().lower(): v for k, v in mapping.items()}
        for key in keys:
            value = cls._training_metric_float(lowered.get(str(key or "").strip().lower()))
            if value is not None:
                return value
        return None

    def _append_train_log(self, message: str) -> None:
        logger.info(str(message or ""))

    def _infer_dataset_target(self, dataset_path) -> str:
        from pathlib import Path

        text = str(dataset_path or "").strip().lower().replace("\\", "/")
        if not text:
            return ""
        try:
            path = Path(dataset_path)
            if path.is_file() and path.name.lower() == "data.yaml":
                text = f"{text} {path.parent.name.lower()}"
        except Exception:
            pass
        if any(token in text for token in ("plate", "plates", "tablic", "pose", "mt-")):
            return "plate"
        if any(token in text for token in ("char", "chars", "znak", "mz-")):
            return "char"
        if any(token in text for token in ("vehicle", "vehicles", "pojazd", "mp-")):
            return "vehicle"
        return ""

    def _find_best_weights_for_run(self, run_id: str):
        from pathlib import Path
        from ..campaign_manager import CAMPAIGN

        safe_run_id = str(run_id or "").strip()
        if not safe_run_id:
            return None
        roots = []
        try:
            project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
            project_root = CAMPAIGN.get_active_project_root_dir() if project_name else None
            if project_root:
                roots.append(Path(project_root) / "5_training_runs")
        except Exception:
            pass
        for target in ("plate", "char", "vehicle"):
            try:
                roots.append(Path(CONFIG.get_training_runs_dir(target)))
            except Exception:
                pass
        seen = set()
        for root in roots:
            try:
                root = Path(root)
                key = str(root.resolve()).lower()
            except Exception:
                key = str(root).lower()
            if key in seen or not root.exists():
                continue
            seen.add(key)
            try:
                candidates = list(root.rglob("best.pt"))
            except Exception:
                candidates = []
            matching = []
            for candidate in candidates:
                text = str(candidate).lower()
                if safe_run_id.lower() in text:
                    matching.append(candidate)
            if matching:
                matching.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                return matching[0]
        return None

    def _infer_history_run_target(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._infer_history_run_target(self, *args, **kwargs)

    def _resolve_history_run_best_weights(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._resolve_history_run_best_weights(self, *args, **kwargs)

    def _build_history_run_metric_summary(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._build_history_run_metric_summary(self, *args, **kwargs)

    def _build_mobile_model_export_path(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._build_mobile_model_export_path(self, *args, **kwargs)

    def _build_mobile_alpr_package_export_path(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._build_mobile_alpr_package_export_path(self, *args, **kwargs)

    def _build_mobile_export_metadata(self, *args, **kwargs):
        from . import z4_model_export

        return z4_model_export._build_mobile_export_metadata(self, *args, **kwargs)


class AutoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self._faulthandler_stream = None
        self.root.title(f"{CONFIG.APP_NAME} ver. {CONFIG.VERSION}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind("<Unmap>", self._on_root_unmap, add="+")
        self.root.bind("<Map>", self._on_root_map, add="+")
        self.root.bind("<Visibility>", self._on_root_visibility, add="+")
        self.root.bind("<FocusIn>", self._on_root_focus_in, add="+")
        self._enable_fatal_crash_logging()
        
        self.icon_manager = IconManager
        self.icon_manager.test_emoji_support(root)
        
        self.themes = THEME_DEFINITIONS
        self.current_theme_key = self._load_theme_preference()
        self.current_theme_name = self.themes[self.current_theme_key]["label"]
        self.palette = get_theme_palette(self.current_theme_key, self.themes)
        self.theme_var = tk.StringVar(master=root, value=self.current_theme_key)
        self.global_yolo_device_var = tk.StringVar(
            master=root,
            value=self._load_global_yolo_device_preference(),
        )
        self._global_yolo_devices_cache = self._initial_global_yolo_device_options()
        self._global_yolo_devices_cache_ready = False
        self._global_yolo_devices_scan_in_progress = False
        self._global_yolo_devices_last_error = ""
        self.menu_bar_frame = None
        self.menu_theme_badge = None
        self._menu_dropdown = None
        self._menu_dropdown_owner = None
        self._menu_outside_click_bind_id = None
        self._menu_escape_bind_id = None
        self._menu_buttons = []
        self._theme_refresh_after_id = None
        self._theme_refresh_after_ids = []
        self._help_panel_default_height = 1
        self._help_panel_expanded = False
        self._help_panel_apply_in_progress = False
        self._help_overlay_place_after_id = None
        self._help_overlay_forced_visible = False
        self._help_overlay_forced_text = ""
        self._free_mode_assistant_overlay = None
        self._free_mode_assistant_place_after_id = None
        self._free_mode_assistant_refresh_after_id = None
        self._free_mode_assistant_toggle_btn = None
        self._free_mode_assistant_enabled = False
        self._free_mode_assistant_overlay_owner = None
        self._free_mode_assistant_context_override_key = ""
        self._free_mode_assistant_context_override = None
        self._free_mode_assistant_context_override_owner = None
        self._simple_tooltip_window = None
        self._simple_tooltip_after_id = None
        self._simple_tooltip_target = None
        self._help_panel_message_prefix = "HELP:"
        self._status_full_text = ""
        self._global_terminal_lines = ["[APP] Terminal globalny gotowy. Tutaj trafiaja logi procesow z Z2, PZ2 i Z4."]
        self._global_terminal_entries = [
            {
                "text": "[APP] Terminal globalny gotowy. Tutaj trafiaja logi procesow z Z2, PZ2 i Z4.",
                "tag": "terminal_info",
            }
        ]
        self._global_terminal_max_lines = 1600
        self._global_terminal_window = None
        self._global_terminal_shell = None
        self._global_terminal_header = None
        self._global_terminal_title_lbl = None
        self._global_terminal_clear_btn = None
        self._global_terminal_close_btn = None
        self._global_terminal_body = None
        self._global_terminal_text = None
        self._global_terminal_scrollbar = None
        self._global_terminal_hscrollbar = None
        self._global_terminal_toggle_btn = None
        self._global_terminal_visible = False
        self._global_terminal_geometry_initialized = False
        self._global_terminal_hold_position = False
        self._window_restore_after_id = None
        self._window_restore_topmost_after_id = None
        self._window_restore_attempts = 0
        self._window_restore_pending = False
        self.tabs = {}
        self._closing_in_progress = False
        self.startup_overlay_frame = None
        self.startup_overlay_card = None
        self.startup_overlay_title_lbl = None
        self.startup_overlay_status_lbl = None
        self.startup_overlay_progress = None
        self.startup_overlay_window = None
        self.startup_overlay_shown_at = None
        self.startup_progress_var = tk.DoubleVar(master=root, value=0.0)
        self.startup_status_var = tk.StringVar(master=root, value="Przygotowanie aplikacji...")
        self._startup_finalize_after_id = None
        self._startup_finalize_attempts = 0
        self._startup_ready_streak = 0
        self._startup_tabs_present_since = None
        self._startup_progress_peak = 0.0
        self._main_window_hidden_for_startup = False
        self._main_window_revealed = False

        self.style = ttk.Style()
        self._setup_style(self.current_theme_key)
        self._prepare_main_window_for_startup()
        self._show_startup_overlay()
        self._set_startup_progress(8, "Uruchamianie interfejsu...")
        self.is_processing = False
        self._manual_processing = False
        self._exclusive_operation = None
        self._processing_state_lock = threading.RLock()
        # Lokalna flaga aktywnego trybu kampanii.
        self.campaign_mode_active = False
        # Ręczne wyjście z projektu ma pierwszeństwo nad automatycznym trybem kampanii.
        self.campaign_free_mode = False
        self._finish_app_init(root)

    # Delegates from app recovery and tooltip modules are bound after class creation.

    def _finish_app_init(self, root):
        self._set_startup_progress(16, "Budowanie menu...")
        self._create_menu()
        self._set_startup_progress(24, "Konfiguracja okien dialogowych...")
        self._install_themed_dialog_hooks()

        self.default_status_message = HELP.default_message
        
        # Panel pomocy musi powstać przed notebookiem, aby poprawnie zakotwiczyć go na dole okna.
        self._set_startup_progress(34, "Inicjalizacja panelu pomocy...")
        status_bg = self.palette["console_bg"]
        self.info_panel_frame = tk.Frame(root, bg=status_bg, bd=0, highlightthickness=0)
        self.info_panel_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._global_terminal_toggle_btn = tk.Button(
            self.info_panel_frame,
            text=">_",
            command=self.toggle_global_terminal,
            width=3,
            cursor="hand2",
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            padx=6,
            pady=2,
            font=("Consolas", 9, "bold"),
            bg=status_bg,
            activebackground=status_bg,
            fg=self.palette["guide"],
            activeforeground=self.palette["guide"],
        )
        self._global_terminal_toggle_btn.pack(side=tk.LEFT, padx=(8, 0), pady=4)

        self._free_mode_assistant_toggle_btn = tk.Button(
            self.info_panel_frame,
            text="AS",
            command=self.toggle_free_mode_assistant,
            width=3,
            cursor="hand2",
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            padx=6,
            pady=2,
            font=("Segoe UI", 9, "bold"),
            bg=status_bg,
            activebackground=status_bg,
            fg=self.palette["guide"],
            activeforeground=self.palette["guide"],
        )
        self._free_mode_assistant_toggle_btn.pack(side=tk.LEFT, padx=(4, 0), pady=4)
        self._bind_simple_tooltip(self._global_terminal_toggle_btn, "Terminal")
        self._bind_simple_tooltip(self._free_mode_assistant_toggle_btn, "Asystent")
        HELP.bind_help(self._global_terminal_toggle_btn, "app_global_terminal")
        HELP.bind_help(self._free_mode_assistant_toggle_btn, "app_free_assistant")

        self.status_text = tk.Text(
            self.info_panel_frame, height=1, wrap=tk.NONE, 
            bg=status_bg,
            bd=0,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            fg=self.palette["console_fg"],
            insertbackground=self.palette["console_fg"],
            highlightthickness=0
        )
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 10), pady=4)
        self.status_text.insert(tk.END, self._format_help_panel_message(self.default_status_message))
        self.status_text.config(state=tk.DISABLED)
        self.status_text.bind("<Configure>", self._on_help_panel_text_configure, add="+")
        HELP.bind_help(self.status_text, "app_help_panel")
        self.help_overlay_frame = tk.Frame(
            root,
            bg=self.palette["surface_info"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.palette["accent"],
            highlightcolor=self.palette["accent"]
        )
        self.help_overlay_title_lbl = tk.Label(
            self.help_overlay_frame,
            text="Rozwinięta pomoc  |  ESC",
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 9, "bold"),
            bg=self.palette["surface_info"],
            fg=self.palette["fg"],
            bd=0,
            highlightthickness=0
        )
        self.help_overlay_title_lbl.pack(fill=tk.X, padx=12, pady=(10, 4))
        self.help_overlay_text = tk.Message(
            self.help_overlay_frame,
            text=self._format_help_panel_message(self.default_status_message),
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 10),
            bg=self.palette["surface_info"],
            fg=self.palette["fg"],
            width=560,
            padx=0,
            pady=0
        )
        self.help_overlay_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        for widget in (self.help_overlay_frame, self.help_overlay_title_lbl, self.help_overlay_text):
            try:
                widget.bind("<ButtonPress-1>", self._on_help_overlay_primary_click, add="+")
            except Exception:
                pass
        self._status_full_text = self._format_help_panel_message(self.default_status_message)
        self._bind_help_panel_shortcuts()
        self._apply_help_panel_visual_state()
        
        # Podpinamy globalny menedżer pomocy
        HELP.status_updater = self.update_status
        HELP.overlay_presenter = self.show_context_help_overlay
        HELP.overlay_dismisser = self.hide_context_help_overlay
        
        # 2. Tworzenie Notatnika z zakładkami
        self._set_startup_progress(48, "Tworzenie struktury zakładek...")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))
        
        self._create_tabs(progress_callback=self._set_startup_progress)
        self._set_startup_progress(92, "Nakładanie motywu...")
        self._apply_theme_to_tabs()

        # główna blokada działa przez disabled tabs
        self.notebook.bind("<ButtonPress-1>", self._on_main_notebook_button_press, add="+")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_notebook_tab_changed)
        self._free_mode_assistant_overlay = FreeModeAssistantOverlay(
            root,
            on_close=self._close_free_mode_assistant_overlay,
        )
        self._free_mode_assistant_overlay_owner = root
        self.root.bind("<Configure>", lambda _event: self._schedule_free_mode_assistant_placement(), add="+")

        # początkowa synchronizacja stanów zakładek
        self.update_campaign_tab_access()
        self._restore_active_main_tab_preference()
        self._refresh_free_mode_assistant()

        # Po starcie pokaż informację o aktywnym projekcie, jeśli aplikacja wznawia tryb kampanii.
        try:
            from ..campaign_manager import CAMPAIGN
            active_proj = CAMPAIGN.get_active_project_name()
            if active_proj:
                try:
                    annotation_tab = getattr(self, "tabs", {}).get("annotation")
                    ensure_context = (
                        getattr(annotation_tab, "ensure_campaign_context_ready_for_active_project", None)
                        if annotation_tab is not None
                        else None
                    )
                    if callable(ensure_context) and self._get_selected_tab_key() == "annotation":
                        self.root.after_idle(ensure_context)
                except Exception as restore_err:
                    logger.debug(f"Nie udalo sie przywrocic kontekstu Z2 dla aktywnego projektu po starcie: {restore_err}")

                self.update_status(
                    f"Aktywny projekt: {active_proj}. Aplikacja działa w trybie kampanii — aby wrócić do trybu swobodnego, użyj „Wyjdź z projektu” w Wizardzie.",
                    "warning"
                )
        except Exception:
            pass

        self._schedule_startup_finalize()

    # Delegates from app_startup are bound after class creation.

    def _load_theme_preference(self) -> str:
        try:
            if SESSION:
                saved = SESSION.get("ui", "theme", "dark_visual_cs")
                if saved in self.themes:
                    return saved
        except Exception:
            pass
        return "dark_visual_cs"

    def _save_theme_preference(self, theme_key: str):
        try:
            if SESSION:
                SESSION.set("ui", "theme", theme_key)
                SESSION.save_session()
        except Exception:
            pass

    def _auto_device_label(self) -> str:
        return "Auto"

    def _cpu_device_label(self) -> str:
        return "CPU"

    @staticmethod
    def _device_cuda_prefix(device_label: str | None) -> str:
        raw = str(device_label or "").strip().lower()
        if not raw.startswith("cuda:"):
            return ""
        return raw.split()[0]

    def _format_global_yolo_device_menu_label(self, device_label: str) -> str:
        raw = str(device_label or "").strip()
        lower = raw.lower()
        if lower.startswith("auto"):
            return "Auto"
        if lower.startswith("cpu"):
            return "CPU"
        if lower.startswith("cuda:"):
            prefix = raw.split()[0]
            index = prefix.split(":", 1)[1] if ":" in prefix else "0"
            detail = raw[len(prefix):].strip()
            if detail.startswith("(") and detail.endswith(")"):
                detail = detail[1:-1].strip()
            return f"GPU/CUDA {index}" + (f" - {detail}" if detail else "")
        return raw

    def _initial_global_yolo_device_options(self) -> list[str]:
        return [self._auto_device_label(), self._cpu_device_label()]

    def _normalize_global_yolo_device_options(self, devices: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in [self._auto_device_label(), self._cpu_device_label(), *(devices or [])]:
            label = str(item or "").strip()
            if not label:
                continue
            key = self._device_cuda_prefix(label) or label.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(label)
        return normalized

    def _scan_available_yolo_devices_sync(self, progress=None) -> list[str]:
        devices = [self._auto_device_label(), self._cpu_device_label()]
        profiles = []
        if progress:
            progress("torch_import")
        import torch

        if progress:
            progress("cuda_availability")
        if torch.cuda.is_available():
            if progress:
                progress("cuda_properties")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({name})")
                try:
                    memory = float(torch.cuda.get_device_properties(i).total_memory) / (1024 ** 3)
                except Exception:
                    memory = 0.0
                profiles.append({"raw": f"cuda:{i}", "index": i, "name": name, "memory_gb": memory})
        if progress:
            progress({"device_profiles": profiles})
        return self._normalize_global_yolo_device_options(devices)

    def get_available_yolo_devices(self, *, allow_probe: bool = False) -> list[str]:
        if allow_probe:
            profiles = []
            def collect_profiles(event):
                if isinstance(event, dict):
                    profiles.extend(event.get("device_profiles", []))
            devices = self._scan_available_yolo_devices_sync(progress=collect_profiles)
            self._global_yolo_devices_cache = list(devices)
            self._global_yolo_device_profiles_cache = profiles
            self._global_yolo_devices_cache_ready = True
            return devices
        return list(
            getattr(self, "_global_yolo_devices_cache", None)
            or self._initial_global_yolo_device_options()
        )

    def get_available_yolo_device_profiles(self) -> list[dict]:
        """Return confirmed hardware metadata without accessing torch or the driver."""
        return [dict(item) for item in getattr(self, "_global_yolo_device_profiles_cache", [])]

    def _refresh_global_yolo_devices_async(self, *, silent: bool = False) -> None:
        if getattr(self, "_closing_in_progress", False):
            return
        if bool(getattr(self, "_global_yolo_devices_scan_in_progress", False)):
            if not silent:
                self.update_status("Wykrywanie urządzeń GPU/CUDA już trwa.", "info")
            return

        previous_worker = getattr(self, "_global_yolo_devices_scan_thread", None)
        if previous_worker is not None and previous_worker.is_alive():
            # A timed-out driver call cannot be cancelled safely inside a thread.
            if not silent:
                self.update_status(
                    "Sprawdzanie GPU nie odpowiedziało w wyznaczonym czasie. "
                    "Możesz wybrać CPU; kolejna próba będzie możliwa po zwolnieniu sterownika.",
                    "warning",
                )
            return

        self._global_yolo_devices_scan_in_progress = True
        self._global_yolo_devices_last_error = ""
        results = queue.SimpleQueue()
        started_at = time.monotonic()
        phase = "start"
        device_profiles = None
        logger.info("[HARDWARE SCAN] started timeout=%.0fs", HARDWARE_SCAN_TIMEOUT_SECONDS)
        if not silent:
            self.update_status("Wykrywam dostępne urządzenia GPU/CUDA w tle...", "info")

        def worker() -> None:
            try:
                devices = self._scan_available_yolo_devices_sync(
                    progress=lambda stage: results.put(("phase", stage, "", time.monotonic())),
                )
            except Exception as exc:
                results.put(("done", None, f"{type(exc).__name__}: {exc}", time.monotonic()))
            else:
                results.put(("done", devices, "", time.monotonic()))

        def poll() -> None:
            nonlocal phase, device_profiles
            self._global_yolo_devices_scan_after_id = None
            if getattr(self, "_closing_in_progress", False):
                self._global_yolo_devices_scan_in_progress = False
                return
            while True:
                try:
                    kind, payload, error_text, event_time = results.get_nowait()
                except queue.Empty:
                    break
                if kind == "phase":
                    if isinstance(payload, dict) and "device_profiles" in payload:
                        device_profiles = payload["device_profiles"]
                        continue
                    phase = payload
                    logger.info(
                        "[HARDWARE SCAN] phase=%s elapsed=%.0fms",
                        phase, (event_time - started_at) * 1000,
                    )
                    continue
                self._finish_global_yolo_device_scan(
                    payload, error_text, silent=silent, device_profiles=device_profiles,
                )
                logger.info(
                    "[HARDWARE SCAN] finished probe=%.0fms ui_delivery=%.0fms error=%s",
                    (event_time - started_at) * 1000,
                    (time.monotonic() - event_time) * 1000, error_text or "none",
                )
                return
            if time.monotonic() - started_at >= HARDWARE_SCAN_TIMEOUT_SECONDS:
                error_text = f"Timeout ({HARDWARE_SCAN_TIMEOUT_SECONDS:.0f}s), phase={phase}"
                logger.warning("[HARDWARE SCAN] %s", error_text)
                self._finish_global_yolo_device_scan(None, error_text, silent=silent)
                return
            self._global_yolo_devices_scan_after_id = self.root.after(100, poll)

        self._global_yolo_devices_scan_thread = threading.Thread(
            target=worker, daemon=True, name="hardware-scan",
        )
        try:
            # Tk callbacks are scheduled only by the UI thread; the worker just writes to a queue.
            self._global_yolo_devices_scan_after_id = self.root.after(100, poll)
            self._global_yolo_devices_scan_thread.start()
        except Exception as exc:
            pending = getattr(self, "_global_yolo_devices_scan_after_id", None)
            if pending:
                self.root.after_cancel(pending)
                self._global_yolo_devices_scan_after_id = None
            self._finish_global_yolo_device_scan(None, str(exc), silent=silent)

    def _finish_global_yolo_device_scan(self, devices, error_text: str, *, silent: bool, device_profiles=None) -> None:
        self._global_yolo_devices_scan_in_progress = False
        self._global_yolo_devices_cache_ready = True
        self._global_yolo_devices_last_error = error_text
        if not error_text:
            self._global_yolo_devices_cache = self._normalize_global_yolo_device_options(devices)
            self._global_yolo_device_profiles_cache = [dict(item) for item in (device_profiles or [])]
            current_before = str(self.global_yolo_device_var.get() or "").strip()
            saved_before = str(
                getattr(self, "_global_yolo_device_saved_preference_raw", "") or ""
            ).strip()
            restore_candidate = (
                saved_before
                if current_before.lower().startswith("auto") and saved_before.lower().startswith("cuda:")
                else current_before
            )
            normalized_current = self.normalize_global_yolo_device_choice(
                restore_candidate, devices=self._global_yolo_devices_cache,
            )
            if current_before != normalized_current:
                self.global_yolo_device_var.set(normalized_current)
                self._save_global_yolo_device_preference(normalized_current)
            elif saved_before.lower().startswith("cuda:") and normalized_current == self._auto_device_label():
                self._save_global_yolo_device_preference(normalized_current)

            training_tab = getattr(self, "tabs", {}).get("training")
            if bool(getattr(training_tab, "_step4_train_tab_built", False)):
                try:
                    training_tab._refresh_training_device_hint()
                except Exception as exc:
                    logger.debug("Nie udało się odświeżyć informacji o sprzęcie Z4: %s", exc)

        owner = getattr(self, "_configuration_menu_button", None)
        refresh_menu = getattr(self, "_menu_dropdown_update_items", None)
        if owner is not None and getattr(self, "_menu_dropdown_owner", None) is owner and callable(refresh_menu):
            refresh_menu(self._build_configuration_menu_items())
        if silent:
            return
        gpu_count = sum(1 for item in self.get_available_yolo_devices() if str(item).lower().startswith("cuda:"))
        if error_text:
            self.update_status(
                "Nie udało się sprawdzić sprzętu. Zachowano ostatnią potwierdzoną listę urządzeń. "
                "Możesz ponowić sprawdzanie w menu Konfiguracja.",
                "warning",
            )
        elif gpu_count:
            self.update_status(f"Wykryto urządzenia CUDA: {gpu_count}.", "success")
        else:
            self.update_status("GPU/CUDA niedostępne. Dostępne są Auto i CPU.", "info")

    def _global_yolo_device_menu_selected(self, device_label: str) -> bool:
        current = self.normalize_global_yolo_device_choice()
        candidate = self.normalize_global_yolo_device_choice(device_label)
        return str(current or "").strip().lower() == str(candidate or "").strip().lower()

    def _build_configuration_menu_items(self) -> list[dict]:
        if (
            not bool(getattr(self, "_global_yolo_devices_cache_ready", False))
            and not bool(getattr(self, "_global_yolo_devices_scan_in_progress", False))
        ):
            self._refresh_global_yolo_devices_async(silent=True)

        devices = self.get_available_yolo_devices()
        items = []
        for device_label in devices:
            items.append(
                {
                    "kind": "radio",
                    "label": self._format_global_yolo_device_menu_label(device_label),
                    "selected": self._global_yolo_device_menu_selected(device_label),
                    "command": (
                        lambda value=device_label: self.set_global_yolo_device_choice(value)
                    ),
                }
            )

        gpu_count = sum(1 for item in devices if str(item).lower().startswith("cuda:"))
        if not gpu_count:
            if bool(getattr(self, "_global_yolo_devices_scan_in_progress", False)):
                gpu_label = "GPU/CUDA - sprawdzam dostępność..."
            elif getattr(self, "_global_yolo_devices_last_error", ""):
                gpu_label = "GPU/CUDA - nie udało się sprawdzić"
            elif bool(getattr(self, "_global_yolo_devices_cache_ready", False)):
                gpu_label = "GPU/CUDA - niedostępne"
            else:
                gpu_label = "GPU/CUDA - jeszcze niesprawdzone"
            items.append(
                {
                    "kind": "command",
                    "label": gpu_label,
                    "disabled": True,
                    "command": None,
                }
            )

        items.append({"kind": "separator"})
        if gpu_count and getattr(self, "_global_yolo_devices_last_error", ""):
            items.append({"label": "Nie udało się odświeżyć stanu sprzętu", "disabled": True})
        if bool(getattr(self, "_global_yolo_devices_scan_in_progress", False)):
            refresh_label = "Odświeżanie stanu sprzętu w toku..."
            refresh_command = lambda: None
            refresh_disabled = True
        else:
            refresh_label = "Odśwież stan sprzętu"
            refresh_command = lambda: self._refresh_global_yolo_devices_async(silent=False)
            refresh_disabled = False
        items.append(
            {
                "kind": "command",
                "label": refresh_label,
                "disabled": refresh_disabled,
                "command": refresh_command,
            }
        )
        return items

    def normalize_global_yolo_device_choice(
        self,
        raw_value: str | None = None,
        devices: list[str] | None = None,
    ) -> str:
        # CUDA is valid only when the current hardware scan confirms it.
        if devices is None:
            available = list(getattr(self, "_global_yolo_devices_cache", None) or [])
        else:
            available = list(devices or [])
        current = str(
            raw_value if raw_value is not None else self.global_yolo_device_var.get() or ""
        ).strip()
        current_lower = current.lower()
        fallback = self._auto_device_label()

        if not current or current_lower.startswith("auto"):
            return self._auto_device_label()
        if current_lower.startswith("cpu"):
            return self._cpu_device_label()
        if current_lower.startswith("cuda:"):
            prefix = current.split()[0]
            for option in available:
                if str(option or "").lower().startswith(prefix):
                    return option
            return fallback

        for option in available:
            if str(option or "").strip().lower() == current_lower:
                return str(option or "").strip()
        return fallback

    def _load_global_yolo_device_preference(self) -> str:
        try:
            saved = SESSION.get("ui", "global_yolo_device", "auto") if SESSION else "auto"
        except Exception:
            saved = "auto"
        self._global_yolo_device_saved_preference_raw = str(saved or "auto").strip() or "auto"
        return self.normalize_global_yolo_device_choice(saved)

    def _save_global_yolo_device_preference(self, value: str):
        self._global_yolo_device_saved_preference_raw = value
        try:
            if not SESSION:
                return
            SESSION.set("ui", "global_yolo_device", value)
            SESSION.save_session()
        except Exception:
            pass

    def get_global_yolo_device_choice(self) -> str:
        normalized = self.normalize_global_yolo_device_choice()
        try:
            if self.global_yolo_device_var.get() != normalized:
                self.global_yolo_device_var.set(normalized)
        except Exception:
            pass
        return normalized

    def set_global_yolo_device_choice(self, value: str, *, persist: bool = True):
        normalized = self.normalize_global_yolo_device_choice(value)
        try:
            self.global_yolo_device_var.set(normalized)
        except Exception:
            pass

        if persist:
            self._save_global_yolo_device_preference(normalized)

        for tab_key in ("annotation", "characters", "training"):
            try:
                tab = self.tabs.get(tab_key)
                if tab is not None and hasattr(tab, "apply_global_yolo_device_choice"):
                    tab.apply_global_yolo_device_choice(normalized)
            except Exception:
                pass

        try:
            if bool(getattr(self, "is_processing", False)):
                active_label = self.get_active_exclusive_operation_label()
                suffix = f" ({active_label})" if active_label else ""
                self.update_status(
                    f"Globalne urządzenie YOLO ustawione na: {normalized}. "
                    f"Aktywny proces{suffix} używa urządzenia wybranego przy starcie; zmiana zadziała od następnego uruchomienia.",
                    "warning",
                )
            else:
                self.update_status(f"Globalne urządzenie YOLO: {normalized}", "info")
        except Exception:
            pass

    def _is_free_mode_session_context(self) -> bool:
        try:
            from ..campaign_manager import CAMPAIGN
            active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        except Exception:
            active_project = ""

        # Aktywny projekt jest źródłem prawdy dla kontekstu kampanii.
        # Flaga campaign_free_mode bywa stanem przejściowym po wyjściu z projektu
        # i nie może nadpisywać aktywnego projektu.
        if active_project:
            if bool(getattr(self, "campaign_free_mode", False)):
                self.campaign_free_mode = False
            return False

        return True

    def _load_active_main_tab_preference(self) -> str:
        return "campaign" if "campaign" in self.tabs else "annotation"

    def _save_active_main_tab_preference(self, tab_key: str | None = None):
        try:
            if not SESSION or not self._is_free_mode_session_context():
                return

            selected_key = tab_key or self._get_selected_tab_key()
            if not selected_key or selected_key not in self.tabs:
                return

            SESSION.set("ui", "active_main_tab", selected_key)
            SESSION.save_session()
        except Exception as e:
            logger.debug(f"Nie udalo sie zapisac ostatniej zakladki: {e}")

    def _restore_active_main_tab_preference(self):
        tab_key = self._load_active_main_tab_preference()
        if tab_key not in self.tabs:
            return

        try:
            self.notebook.select(str(self.tabs[tab_key].frame))
        except Exception as e:
            logger.debug(f"Nie udalo sie przywrocic ostatniej zakladki: {e}")

    def get_main_tab_label(self, tab_key: str) -> str:
        labels = {
            "campaign": "[Z1] Wizard",
            "annotation": "[Z2] Anotacja tablic",
            "characters": "[Z3] Autoanotacja znaków tablic",
            "training": "[Z4] Trening i analiza",
            "help": "[Z5] Instrukcja i architektura",
        }
        return labels.get(tab_key, tab_key)

    def refresh_main_tab_labels(self, active_tab_key: str = None):
        for key, tab in self.tabs.items():
            try:
                label = self.get_main_tab_label(key)
                self.notebook.tab(str(tab.frame), text=label)
            except Exception:
                pass

    def _format_project_created_at(self, created_at: str) -> str:
        value = str(created_at or "").strip()
        if not value:
            return ""

        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return value.replace("T", " ")

    def refresh_window_title(self):
        base_title = f"{CONFIG.APP_NAME} ver. {CONFIG.VERSION}"

        try:
            from ..campaign_manager import CAMPAIGN

            active_project = CAMPAIGN.get_active_project_name()
            if not active_project:
                self._refresh_menu_badge()
                self.root.title(base_title)
                return

            project_title = str(active_project).strip()
            self._refresh_menu_badge()
            self.root.title(f"{base_title} | {project_title}" if project_title else base_title)
        except Exception:
            self._refresh_menu_badge()
            self.root.title(base_title)

    def _get_lazy_tab_factory(self, tab_key: str):
        tab_key = str(tab_key or "").strip()
        if tab_key == "annotation":
            from .tab_annotation import AnnotationTab as LazyAnnotationTab

            return LazyAnnotationTab
        if tab_key == "characters":
            from .tab_character_annotation import CharacterAnnotationTab as LazyCharacterAnnotationTab

            return LazyCharacterAnnotationTab
        if tab_key == "training":
            from .tab_training import TrainingTab as LazyTrainingTab

            return LazyTrainingTab
        if HelpTab:
            return HelpTab if tab_key == "help" else None
        return None

    def _is_lazy_tab_key(self, tab_key: str) -> bool:
        return isinstance(getattr(self, "tabs", {}).get(tab_key), _LazyNotebookTab)

    def _iter_loaded_tabs(self):
        for tab in getattr(self, "tabs", {}).values():
            if isinstance(tab, _LazyNotebookTab):
                continue
            yield tab

    def _add_lazy_tab(self, tab_key: str):
        placeholder = _LazyNotebookTab(self, tab_key)
        self.tabs[tab_key] = placeholder
        self.notebook.add(placeholder.frame, text=self.get_main_tab_label(tab_key))
        return placeholder

    def _apply_theme_to_single_tab(self, tab):
        try:
            apply_theme = getattr(tab, "apply_theme", None)
            if callable(apply_theme):
                apply_theme()
        except Exception:
            pass

    def _ensure_tab_loaded(self, tab_key: str, *, select: bool = False):
        tab_key = str(tab_key or "").strip()
        current = getattr(self, "tabs", {}).get(tab_key)
        if current is None:
            raise KeyError(f"Unknown tab key: {tab_key}")
        if not isinstance(current, _LazyNotebookTab):
            if select:
                try:
                    self.notebook.select(str(current.frame))
                except Exception:
                    pass
            return current

        if bool(getattr(self, "_lazy_tab_load_in_progress", False)):
            return current

        factory = self._get_lazy_tab_factory(tab_key)
        if factory is None:
            return current

        placeholder_frame = current.frame
        placeholder_widget = str(placeholder_frame)
        try:
            tab_index = self.notebook.index(str(placeholder_frame))
        except Exception:
            tab_index = "end"
        try:
            tab_state = str(self.notebook.tab(str(placeholder_frame), "state") or "normal")
        except Exception:
            tab_state = "normal"

        self._lazy_tab_load_in_progress = True
        self._lazy_tab_loading_key = tab_key
        self._lazy_tab_loading_placeholder_widget = placeholder_widget
        self._lazy_tab_deferred_select_key = ""
        if select:
            self._lazy_tab_pending_select_key = tab_key
        started = time.perf_counter()
        phase_started = started

        def _log_lazy_phase(name: str) -> None:
            nonlocal phase_started
            try:
                now = time.perf_counter()
                elapsed_ms = (now - phase_started) * 1000.0
                total_ms = (now - started) * 1000.0
                if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                    logger.info(
                        "[LAZY PERF] "
                        f"{tab_key}.{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                    )
                phase_started = now
            except Exception:
                pass

        try:
            try:
                self.update_status(f"Ładuję {self.get_main_tab_label(tab_key)}...", "info")
            except Exception:
                pass
            try:
                current.set_loading_state()
                self.root.update_idletasks()
            except Exception:
                pass
            _log_lazy_phase("placeholder_paint")

            try:
                real_tab = factory(self.notebook, self)
                self.tabs[tab_key] = real_tab
                self._lazy_tab_pending_real_widget = str(real_tab.frame)
                self._lazy_tab_pending_placeholder_widget = placeholder_widget
            except Exception:
                raise
            _log_lazy_phase("factory")

            real_widget = str(real_tab.frame)
            try:
                self.notebook.insert(tab_index, real_tab.frame, text=self.get_main_tab_label(tab_key), state="hidden")
            except Exception:
                try:
                    self.notebook.insert(tab_index, real_tab.frame, text=self.get_main_tab_label(tab_key))
                except Exception:
                    self.notebook.add(real_tab.frame, text=self.get_main_tab_label(tab_key))
                try:
                    self.notebook.hide(real_widget)
                except Exception:
                    pass
            _log_lazy_phase("insert_hidden")

            def _reveal_real_tab():
                try:
                    desired_state = tab_state if tab_state in {"normal", "disabled"} else "normal"
                    if select and desired_state == "disabled":
                        desired_state = "normal"
                    self.notebook.tab(real_widget, state=desired_state)
                except Exception:
                    pass

            if select:
                try:
                    # Realną zakładkę trzymamy ukrytą aż do pierwszego paintu.
                    # Dzięki temu pasek Notebooka nie pokazuje przez moment
                    # podwójnej zakładki: placeholder + właściwy frame.
                    self.notebook.select(str(placeholder_frame))
                    self.root.update_idletasks()
                except Exception:
                    pass
            _log_lazy_phase("keep_placeholder_selected")

            if tab_key == "annotation":
                def _apply_annotation_theme_later(tab_ref=real_tab):
                    try:
                        if self.tabs.get(tab_key) is tab_ref and tab_ref.frame.winfo_exists():
                            self._apply_theme_to_single_tab(tab_ref)
                    except Exception:
                        pass

                try:
                    real_tab.frame.after(1200, _apply_annotation_theme_later)
                except Exception:
                    pass
            else:
                self._apply_theme_to_single_tab(real_tab)
            _log_lazy_phase("apply_theme")
            try:
                first_paint = getattr(real_tab, "prepare_lazy_tab_first_paint", None)
                if callable(first_paint):
                    first_paint()
            except Exception:
                pass
            _log_lazy_phase("prepare_first_paint")
            try:
                first_paint_wait = getattr(real_tab, "wait_for_lazy_tab_first_paint", None)
                if callable(first_paint_wait):
                    first_paint_wait(timeout_ms=4500)
            except Exception:
                pass
            _log_lazy_phase("wait_first_paint")
            if tab_key == "characters":
                try:
                    real_tab.frame.update_idletasks()
                    self.root.update_idletasks()
                except Exception:
                    pass

            def _dispose_placeholder():
                should_keep_selection = False
                try:
                    selected_widget = str(self.notebook.select())
                    should_keep_selection = selected_widget in {str(real_tab.frame), placeholder_widget}
                except Exception:
                    should_keep_selection = False

                previous_dispose_guard = bool(getattr(self, "_notebook_placeholder_dispose_in_progress", False))
                self._notebook_placeholder_dispose_in_progress = True
                try:
                    try:
                        self.notebook.forget(str(placeholder_frame))
                    except Exception:
                        pass
                    try:
                        placeholder_frame.destroy()
                    except Exception:
                        pass
                    _reveal_real_tab()
                    if select and should_keep_selection:
                        try:
                            self.notebook.select(str(real_tab.frame))
                        except Exception:
                            pass
                finally:
                    self._notebook_placeholder_dispose_in_progress = previous_dispose_guard
                if select and should_keep_selection:
                    try:
                        self.notebook.select(str(real_tab.frame))
                        self.root.update_idletasks()
                        if str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip() == tab_key:
                            self._lazy_tab_pending_select_key = ""
                    except Exception:
                        pass
                elif select and str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip() == tab_key:
                    self._lazy_tab_pending_select_key = ""

            if select:
                try:
                    self.root.after_idle(_dispose_placeholder)
                except Exception:
                    _dispose_placeholder()
            else:
                _dispose_placeholder()

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(f"Leniwie załadowano zakładkę {tab_key} w {elapsed_ms:.0f} ms")
            try:
                self.update_status(f"Załadowano {self.get_main_tab_label(tab_key)}.", "success")
            except Exception:
                pass
            return real_tab
        except Exception as e:
            self.tabs[tab_key] = current
            if select and str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip() == tab_key:
                self._lazy_tab_pending_select_key = ""
            logger.error(f"Nie udało się leniwie załadować zakładki {tab_key}: {e}")
            try:
                self.update_status(f"Nie udało się załadować {self.get_main_tab_label(tab_key)}.", "error")
            except Exception:
                pass
            return current
        finally:
            self._lazy_tab_load_in_progress = False
            self._lazy_tab_loading_key = ""
            self._lazy_tab_loading_placeholder_widget = ""
            self._lazy_tab_pending_real_widget = ""
            self._lazy_tab_pending_placeholder_widget = ""
            deferred_key = str(getattr(self, "_lazy_tab_deferred_select_key", "") or "").strip()
            self._lazy_tab_deferred_select_key = ""
            if deferred_key and deferred_key != tab_key and deferred_key in getattr(self, "tabs", {}):
                try:
                    self.root.after_idle(lambda key=deferred_key: self.select_tab(key))
                except Exception:
                    try:
                        self.select_tab(deferred_key)
                    except Exception:
                        pass

    def _restore_lazy_tab_selection(self, tab_key: str, tab) -> None:
        pending_key = str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip()
        tab_key = str(tab_key or "").strip()
        if pending_key != tab_key:
            return
        try:
            selected_widget = str(self.notebook.select())
        except Exception:
            selected_widget = ""
        allowed_widgets = {
            str(getattr(tab, "frame", "") or ""),
            str(getattr(self, "_lazy_tab_pending_real_widget", "") or ""),
            str(getattr(self, "_lazy_tab_pending_placeholder_widget", "") or ""),
        }
        allowed_widgets.discard("")
        if allowed_widgets and selected_widget not in allowed_widgets:
            try:
                self._lazy_tab_pending_select_key = ""
            except Exception:
                pass
            return
        try:
            if self._get_selected_tab_key() != tab_key:
                self.notebook.select(str(tab.frame))
        except Exception:
            pass
        try:
            if str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip() == tab_key:
                self._lazy_tab_pending_select_key = ""
        except Exception:
            pass

    def _create_tabs(self, progress_callback=None):
        def _progress(value, message):
            if callable(progress_callback):
                try:
                    progress_callback(value, message)
                except Exception:
                    pass
        try:
            try:
                _progress(56, "Ładowanie zakładki Z1...")
                self.tabs['campaign'] = CampaignTab(self.notebook, self)
                self.notebook.add(self.tabs['campaign'].frame, text=self.get_main_tab_label("campaign"))

            except Exception as e:
                logger.error(f"Nie udało się załadować zakładki Kampanii: {e}")

            _progress(66, "Rejestrowanie zakładki Z2...")
            self._add_lazy_tab("annotation")

            _progress(76, "Rejestrowanie zakładki Z3...")
            self._add_lazy_tab("characters")

            _progress(84, "Rejestrowanie zakładki Z4...")
            self._add_lazy_tab("training")

            if HelpTab:
                _progress(89, "Rejestrowanie zakładki Z5...")
                self._add_lazy_tab("help")

            self.notebook.select(0)
        except Exception as e:
            logger.error(f"Krytyczny błąd budowania zakładek GUI: {e}")

    def _create_menu(self):
        palette = self.palette

        self._close_menu_dropdown()

        try:
            self.root.config(menu="")
        except Exception:
            pass

        if self.menu_bar_frame is not None:
            try:
                self.menu_bar_frame.destroy()
            except Exception:
                pass

        self.menu_bar_frame = tk.Frame(
            self.root,
            bg=palette["panel"],
            bd=0,
            highlightthickness=1,
            highlightbackground=palette["border"]
        )
        pack_kwargs = {"side": tk.TOP, "fill": tk.X}
        if getattr(self, "notebook", None) is not None:
            pack_kwargs["before"] = self.notebook
        self.menu_bar_frame.pack(**pack_kwargs)
        try:
            self.menu_bar_frame.lift()
        except Exception:
            pass

        left = tk.Frame(self.menu_bar_frame, bg=palette["panel"])
        left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=1)
        self._menu_buttons = []

        def make_menu_button(label: str, items_factory, min_width: int = 220):
            shell = tk.Frame(left, bg=palette["panel"], bd=0, highlightthickness=0)
            shell.pack(side=tk.LEFT, padx=(0, 2), pady=0)
            normal_bg = palette["panel"]
            hover_bg = blend_hex_colors(
                palette["panel_alt"],
                palette["accent"],
                0.08,
            )
            active_fg = palette["fg"]

            btn = tk.Button(
                shell,
                text=label,
                bg=normal_bg,
                fg=active_fg,
                activebackground=hover_bg,
                activeforeground=active_fg,
                relief=tk.FLAT,
                bd=0,
                padx=8,
                pady=2,
                font=("Segoe UI", 9, "normal"),
                highlightthickness=0,
                command=lambda: self._toggle_menu_dropdown(btn, items_factory(), min_width=min_width)
            )
            btn.pack(side=tk.TOP, fill=tk.X)

            underline = tk.Frame(
                shell,
                bg=normal_bg,
                height=1,
                bd=0,
                highlightthickness=0
            )
            underline.pack(side=tk.TOP, fill=tk.X, padx=3)

            def _set_hover_line(active: bool):
                try:
                    bg = hover_bg if active else normal_bg
                    shell.configure(bg=bg)
                    btn.configure(bg=bg, activebackground=hover_bg)
                    underline.configure(bg=(palette["accent"] if active else bg))
                except Exception:
                    pass

            def _sync_hover_on_enter(_event=None):
                _set_hover_line(True)

            def _sync_hover_on_leave(_event=None):
                if getattr(self, "_menu_dropdown_owner", None) is btn:
                    return
                _set_hover_line(False)

            for widget in (shell, btn):
                try:
                    widget.bind("<Enter>", _sync_hover_on_enter, add="+")
                    widget.bind("<Leave>", _sync_hover_on_leave, add="+")
                except Exception:
                    pass

            self._menu_buttons.append({"button": btn, "shell": shell, "underline": underline, "set_hover": _set_hover_line})
            return btn

        make_menu_button(
            "Plik",
            lambda: [
                {"kind": "command", "label": "Historia projektu", "command": self.show_project_history_dialog},
                {"kind": "separator"},
                {"kind": "command", "label": "Wyjdź z projektu / trybu kampanii", "command": self._exit_campaign_mode_anytime},
                {"kind": "separator"},
                {"kind": "command", "label": "Wyjście", "command": self._on_closing},
            ],
            min_width=260
        )

        self._configuration_menu_button = make_menu_button(
            "Konfiguracja",
            self._build_configuration_menu_items,
            min_width=320
        )

        make_menu_button(
            "Integracje",
            lambda: [
                {
                    "kind": "command",
                    "label": "Eksport mobilny",
                    "command": self._open_mobile_model_export_center_from_menu,
                },
                {
                    "kind": "command",
                    "label": "Analiza i import raportów mobilnych",
                    "command": self._open_mobile_report_browser_from_menu,
                },
            ],
            min_width=300
        )

        make_menu_button(
            "Styl",
            lambda: [
                {
                    "kind": "radio",
                    "label": theme_data["label"],
                    "selected": (theme_key == self.current_theme_key),
                    "command": (lambda value=theme_key: self.set_theme(value)),
                }
                for theme_key, theme_data in self.themes.items()
            ],
            min_width=240
        )

        make_menu_button(
            "Pomoc",
            lambda: [
                {"kind": "command", "label": f"Wersja: {CONFIG.VERSION}", "command": self.show_about_dialog},
                {"kind": "command", "label": f"Autor: {APP_AUTHOR}", "command": self.show_about_dialog},
                {"kind": "separator"},
                {"kind": "command", "label": "O programie", "command": self.show_about_dialog},
            ],
            min_width=220
        )

        self.menu_theme_badge = tk.Label(
            self.menu_bar_frame,
            text=self._get_menu_badge_text(),
            bg=palette["panel"],
            fg=palette["muted"],
            font=("Segoe UI", 8, "normal"),
            padx=8,
            pady=3
        )
        self.menu_theme_badge.pack(side=tk.RIGHT)

    def _paint_mobile_export_menu_loader(self, loader: tk.Toplevel, progress_canvas: tk.Canvas, progress: float) -> None:
        try:
            palette = getattr(self, "palette", {})
            bg = palette["panel"]
            fg = palette["fg"]
            accent = palette["accent"]
            success = palette["success"]
            border = palette["panel_border"]
            width = max(320, int(progress_canvas.winfo_width() or 420))
            height = max(12, int(progress_canvas.winfo_height() or 14))
            value = max(0.0, min(100.0, float(progress or 0.0)))
            fill_width = int(width * value / 100.0)
            progress_canvas.delete("all")
            progress_canvas.create_rectangle(
                0,
                0,
                width,
                height,
                fill=blend_hex_colors(bg, fg, 0.08),
                outline=blend_hex_colors(border, bg, 0.25),
                width=1,
            )
            if fill_width > 1:
                progress_canvas.create_rectangle(
                    1,
                    1,
                    max(1, fill_width - 1),
                    max(1, height - 1),
                    fill=blend_hex_colors(accent, success, 0.35),
                    outline="",
                )
            loader.update()
        except Exception:
            pass

    def _show_mobile_export_menu_loader(self, message: str = "Przygotowuję eksport mobilny...", progress: float = 8.0) -> None:
        try:
            existing = getattr(self, "_mobile_export_menu_loader", None)
            if existing is not None and existing.winfo_exists():
                status_var = getattr(self, "_mobile_export_menu_loader_status_var", None)
                progress_canvas = getattr(self, "_mobile_export_menu_loader_progress_canvas", None)
                if status_var is not None:
                    status_var.set(message)
                if progress_canvas is not None:
                    self._paint_mobile_export_menu_loader(existing, progress_canvas, progress)
                return
        except Exception:
            pass
        try:
            palette = getattr(self, "palette", {})
            bg = palette["panel"]
            fg = palette["fg"]
            muted = palette["muted"]
            accent = palette["accent"]
            card_bg = blend_hex_colors(bg, accent, 0.045)
            border = blend_hex_colors(accent, bg, 0.38)
            loader = tk.Toplevel(self.root)
            loader.withdraw()
            loader.overrideredirect(True)
            try:
                loader.attributes("-topmost", True)
            except Exception:
                pass
            loader.configure(bg=bg)
            card = tk.Frame(
                loader,
                bg=card_bg,
                padx=18,
                pady=15,
                highlightthickness=1,
                highlightbackground=border,
            )
            card.pack(fill=tk.BOTH, expand=True)
            tk.Label(
                card,
                text="Ładowanie eksportu mobilnego",
                bg=card_bg,
                fg=fg,
                font=("Segoe UI", 11, "bold"),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(0, 8))
            progress_canvas = tk.Canvas(card, width=420, height=14, bg=card_bg, highlightthickness=0, bd=0)
            progress_canvas.pack(fill=tk.X)
            status_var = tk.StringVar(value=message)
            tk.Label(
                card,
                textvariable=status_var,
                bg=card_bg,
                fg=muted,
                font=("Segoe UI", 9),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(8, 0))
            loader.update_idletasks()
            width = max(420, int(card.winfo_reqwidth() or 460))
            height = max(116, int(card.winfo_reqheight() or 128))
            try:
                root_x = int(self.root.winfo_rootx())
                root_y = int(self.root.winfo_rooty())
                root_w = int(self.root.winfo_width())
                root_h = int(self.root.winfo_height())
                x = root_x + int((root_w - width) / 2)
                y = root_y + int((root_h - height) / 2)
            except Exception:
                screen_w = int(loader.winfo_screenwidth() or 1360)
                screen_h = int(loader.winfo_screenheight() or 840)
                x = int((screen_w - width) / 2)
                y = int((screen_h - height) / 2)
            loader.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
            self._mobile_export_menu_loader = loader
            self._mobile_export_menu_loader_status_var = status_var
            self._mobile_export_menu_loader_progress_canvas = progress_canvas
            loader.deiconify()
            loader.lift()
            self._paint_mobile_export_menu_loader(loader, progress_canvas, progress)
        except Exception:
            self._mobile_export_menu_loader = None

    def _close_mobile_export_menu_loader(self) -> None:
        loader = getattr(self, "_mobile_export_menu_loader", None)
        self._mobile_export_menu_loader = None
        self._mobile_export_menu_loader_status_var = None
        self._mobile_export_menu_loader_progress_canvas = None
        try:
            if loader is not None and loader.winfo_exists():
                loader.destroy()
        except Exception:
            pass

    def _open_mobile_model_export_center_from_menu_legacy_training_loader(self):
        self._show_mobile_export_menu_loader("Ładuję moduł treningu przed eksportem mobilnym...", 12.0)
        try:
            training_tab = self._ensure_tab_loaded("training", select=False)
        except Exception as e:
            self._close_mobile_export_menu_loader()
            logger.error(f"Nie udało się przygotować eksportu mobilnego: {e}")
            return self.themed_info(
                "Eksport mobilny",
                f"Nie udało się przygotować modułu eksportu mobilnego:\n{e}",
                parent=self.root,
                tone="error",
            )

        if isinstance(training_tab, _LazyNotebookTab):
            attempts = int(getattr(self, "_mobile_export_menu_load_attempts", 0) or 0) + 1
            self._mobile_export_menu_load_attempts = attempts
            if attempts > 20:
                self._mobile_export_menu_load_attempts = 0
                self._close_mobile_export_menu_loader()
                return self.themed_info(
                    "Eksport mobilny",
                    "Nie udało się załadować modułu treningu. Otwórz kartę Z4 ręcznie i spróbuj ponownie.",
                    parent=self.root,
                    tone="warning",
                )
            try:
                self._show_mobile_export_menu_loader(
                    f"Ładuję moduł treningu przed eksportem mobilnym... próba {attempts}/20",
                    min(72.0, 12.0 + attempts * 3.0),
                )
            except Exception:
                pass
            try:
                self.root.after(250, self._open_mobile_model_export_center_from_menu)
            except Exception:
                pass
            return None

        self._mobile_export_menu_load_attempts = 0
        opener = getattr(training_tab, "_open_mobile_model_export_center", None)
        if not callable(opener):
            self._close_mobile_export_menu_loader()
            return self.themed_info(
                "Eksport mobilny",
                "Moduł treningu nie udostępnia jeszcze centrum eksportu mobilnego.",
                parent=self.root,
                tone="warning",
            )
        self._show_mobile_export_menu_loader("Uruchamiam centrum eksportu mobilnego...", 86.0)
        self._close_mobile_export_menu_loader()
        return opener()

    def _get_mobile_export_menu_host(self):
        host = getattr(self, "_mobile_export_menu_host", None)
        if host is None or getattr(host, "app", None) is not self:
            host = _MobileExportMenuHost(self)
            self._mobile_export_menu_host = host
        return host

    def _open_mobile_model_export_center_from_menu(self):
        for owner in (getattr(self, "_mobile_export_menu_host", None), self):
            try:
                existing_dialog = getattr(owner, "_mobile_export_center_dialog", None)
                if existing_dialog is not None and existing_dialog.winfo_exists():
                    existing_dialog.deiconify()
                    existing_dialog.lift()
                    existing_dialog.focus_force()
                    return existing_dialog
            except Exception:
                pass

        if bool(getattr(self, "_mobile_export_menu_opening", False)):
            return None
        self._mobile_export_menu_opening = True

        try:
            self._close_mobile_export_menu_loader()
        except Exception:
            pass
        try:
            self._close_menu_dropdown()
        except Exception:
            pass

        def _open_after_menu_event():
            try:
                host = self._get_mobile_export_menu_host()
                from . import z4_model_export

                self._mobile_export_menu_load_attempts = 0
                return z4_model_export._open_mobile_model_export_center(host)
            except Exception as e:
                logger.error(f"Nie udalo sie przygotowac eksportu mobilnego: {e}")
                return self.themed_info(
                    "Eksport mobilny",
                    f"Nie udalo sie przygotowac modulu eksportu mobilnego:\n{e}",
                    parent=self.root,
                    tone="error",
                )
            finally:
                self._mobile_export_menu_opening = False
                try:
                    self._close_mobile_export_menu_loader()
                except Exception:
                    pass

        try:
            self.root.after_idle(_open_after_menu_event)
        except Exception:
            _open_after_menu_event()
        return None

    def _open_mobile_report_browser_from_menu(self):
        try:
            host = self._get_mobile_export_menu_host()
            from . import z4_mobile_report_browser
        except Exception as e:
            logger.error(f"Nie udało się przygotować importu raportów mobilnych: {e}")
            return self.themed_info(
                "Analiza i import raportów mobilnych",
                f"Nie udało się przygotować przeglądarki raportów:\n{e}",
                parent=self.root,
                tone="error",
            )

        return z4_mobile_report_browser.open_mobile_report_browser(host, parent=self.root)

    def _get_menu_badge_text(self) -> str:
        try:
            from ..campaign_manager import CAMPAIGN
            active_project = (CAMPAIGN.get_active_project_name() or "").strip()
            active_iteration = int(CAMPAIGN.get_current_iteration_num() or 1) if active_project else 0
            active_step = int(getattr(CAMPAIGN, "get_current_step", lambda: 1)() or 1) if active_project else 0
            active_target = str(getattr(CAMPAIGN, "get_iteration_target", lambda: "")() or "").strip().lower()
        except Exception:
            active_project = ""
            active_iteration = 0
            active_step = 0
            active_target = ""

        if active_project:
            target_label = self._format_iteration_target_badge_label(active_target)
            stage_label = self._format_campaign_step_badge_label(active_step)
            gate_label = self._get_active_campaign_work_gate_badge_label()
            gate_part = f" | Bramka: {gate_label}" if gate_label else ""
            return (
                f"Projekt: {active_project} | Iteracja {max(1, int(active_iteration or 1))} | {stage_label} | "
                f"Tor: {target_label}{gate_part}"
            )
        return "Projekt: tryb swobodny"

    def _get_active_campaign_work_gate_badge_label(self) -> str:
        try:
            from ..campaign_manager import CAMPAIGN

            if not (CAMPAIGN.get_active_project_name() or "").strip():
                return ""
        except Exception:
            return ""

        tab_key = str(self._get_selected_tab_key() or "").strip()
        tab = getattr(self, "tabs", {}).get(tab_key) if tab_key else None
        gate_id = ""

        if tab_key == "campaign":
            try:
                graph_gate_getter = getattr(tab, "get_active_campaign_graph_gate_badge_label", None)
                graph_gate_label = str(graph_gate_getter() if callable(graph_gate_getter) else "").strip()
                if graph_gate_label:
                    return graph_gate_label
            except Exception:
                pass

        try:
            graph_context = dict(getattr(tab, "_campaign_graph_entry_context", {}) or {})
        except Exception:
            graph_context = {}
        if graph_context:
            gate_id = str(
                graph_context.get("graph_gate_id")
                or graph_context.get("gate_id")
                or graph_context.get("working_gate_id")
                or ""
            ).strip().upper()

        if not gate_id and tab_key == "characters":
            gate_id = "T06"

        if not gate_id and tab_key == "training":
            # Z4 jest obecnie pracą domykającą iterację: wynik treningu/decyzja
            # wraca do ostatniej bramki grafu.
            gate_id = "T06"

        return self._format_campaign_gate_badge_label(gate_id)

    @staticmethod
    def _format_campaign_gate_badge_label(gate_id: str | None) -> str:
        normalized = str(gate_id or "").strip().upper()
        if not normalized:
            return ""
        try:
            from .z2_shared_ui import campaign_visible_gate_id

            normalized = str(campaign_visible_gate_id(normalized) or normalized).strip().upper()
        except Exception:
            pass
        return normalized

    @staticmethod
    def _format_campaign_step_badge_label(step: int | str | None) -> str:
        try:
            step_num = int(step or 1)
        except Exception:
            step_num = 1
        if step_num < 1:
            step_num = 1
        if step_num > 4:
            step_num = 4
        return f"Etap E{step_num}"

    @staticmethod
    def _format_iteration_target_badge_label(target: str | None) -> str:
        normalized = str(target or "").strip().lower()
        if normalized == "plate":
            return "tablice"
        if normalized == "char":
            return "znaki"
        return "jeszcze nie wybrany"

    def _refresh_menu_badge(self):
        badge = getattr(self, "menu_theme_badge", None)
        if badge is None:
            return

        try:
            badge.configure(text=self._get_menu_badge_text())
        except Exception:
            pass

    def _exit_campaign_mode_anytime(self):
        try:
            from ..campaign_manager import CAMPAIGN
            active_project = (CAMPAIGN.get_active_project_name() or "").strip()
        except Exception:
            active_project = ""

        if not active_project:
            self.themed_info(
                "Tryb swobodny",
                "Nie ma aktywnego projektu. Aplikacja działa już w trybie swobodnym.",
                parent=self.root,
                tone="info",
            )
            return

        campaign_tab = self.tabs.get("campaign")
        if campaign_tab is not None and hasattr(campaign_tab, "_exit_project_mode"):
            try:
                campaign_tab._exit_project_mode()
                return
            except Exception as e:
                logger.error(f"Nie udało się wyjść z projektu przez menu główne: {e}")

        self.themed_info(
            "Wyjście z projektu",
            "Nie udało się uruchomić wyjścia z projektu z poziomu menu. Spróbuj użyć przycisku „Wyjdź z projektu” w wizardzie.",
            parent=self.root,
            tone="warning",
        )

    def _widget_contains_point(self, widget, x_root: int, y_root: int) -> bool:
        if widget is None:
            return False

        try:
            wx = widget.winfo_rootx()
            wy = widget.winfo_rooty()
            return wx <= x_root < (wx + widget.winfo_width()) and wy <= y_root < (wy + widget.winfo_height())
        except Exception:
            return False

    def _bind_help_panel_shortcuts(self):
        try:
            self.root.bind_all("<ButtonPress-1>", self._dismiss_forced_help_overlay, add="+")
        except Exception:
            pass

        try:
            self.root.bind_all("<Escape>", self._dismiss_forced_help_overlay, add="+")
        except Exception:
            pass

    def _dismiss_forced_help_overlay(self, event=None):
        if bool(getattr(self, "_help_overlay_forced_visible", False)):
            self.hide_context_help_overlay()
            return "break"

    def _on_help_overlay_primary_click(self, event=None):
        if not bool(getattr(self, "_help_overlay_forced_visible", False)):
            return None
        self.hide_context_help_overlay()
        return "break"

    def show_context_help_overlay(self, message: str, icon: str = "help"):
        formatted_message = self._format_help_panel_message(message, icon=icon)
        self._status_full_text = formatted_message
        self._help_overlay_forced_text = formatted_message
        self._help_overlay_forced_visible = True
        self._apply_help_panel_visual_state()
        try:
            self.help_overlay_frame.grab_set()
        except Exception:
            pass

    def hide_context_help_overlay(self):
        if not bool(getattr(self, "_help_overlay_forced_visible", False)):
            return

        self._help_overlay_forced_visible = False
        self._help_overlay_forced_text = ""
        try:
            current_grab = self.root.grab_current()
        except Exception:
            current_grab = None

        try:
            if current_grab is self.help_overlay_frame:
                self.help_overlay_frame.grab_release()
        except Exception:
            pass
        self._apply_help_panel_visual_state()

    def _format_help_panel_message(self, message: str, icon: str | None = None) -> str:
        base_prefix = str(getattr(self, "_help_panel_message_prefix", "HELP:")).strip()
        clean_message = str(message or "").strip()
        default_message = str(getattr(self, "default_status_message", "") or "").strip()

        if not clean_message:
            clean_message = default_message

        if clean_message.startswith(base_prefix):
            return clean_message

        return f"{base_prefix} {clean_message}".strip()

    def _get_help_strip_available_width(self) -> int:
        widget = getattr(self, "status_text", None)
        if widget is None:
            return 320

        try:
            width = int(widget.winfo_width())
        except Exception:
            width = 0

        return max(180, width - 24)

    def _measure_help_text_width(self, text: str) -> int:
        widget = getattr(self, "status_text", None)
        if widget is None:
            return len(str(text or "")) * 7

        try:
            font_obj = tkfont.Font(font=widget.cget("font"))
        except Exception:
            font_obj = tkfont.Font(self.root, family="Segoe UI", size=10)

        return int(font_obj.measure(str(text or "")))

    def _build_help_strip_text(self, text: str) -> str:
        full_text = str(text or "").replace("\r", " ").replace("\n", " ").strip()
        if not full_text:
            return ""

        compact_prefix = str(getattr(self, "_help_panel_message_prefix", "HELP:")).strip()
        hint_suffix = "..."
        available_width = self._get_help_strip_available_width()
        full_width = self._measure_help_text_width(full_text)
        if full_width <= available_width:
            return full_text

        suffix = hint_suffix
        suffix_width = self._measure_help_text_width(suffix)
        if suffix_width >= available_width:
            return compact_prefix

        low = 0
        high = len(full_text)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = full_text[:mid].rstrip() + suffix
            if self._measure_help_text_width(candidate) <= available_width:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best or compact_prefix

    def _set_help_overlay_visible(self, visible: bool):
        overlay = getattr(self, "help_overlay_frame", None)
        if overlay is None:
            return

        if visible:
            self._schedule_help_overlay_placement()
        else:
            try:
                overlay.place_forget()
            except Exception:
                pass

    def _schedule_help_overlay_placement(self):
        pending = getattr(self, "_help_overlay_place_after_id", None)
        if pending:
            try:
                self.root.after_cancel(pending)
            except Exception:
                pass
        try:
            self._help_overlay_place_after_id = self.root.after_idle(self._place_help_overlay)
        except Exception:
            self._help_overlay_place_after_id = None

    def _place_help_overlay(self):
        self._help_overlay_place_after_id = None
        overlay = getattr(self, "help_overlay_frame", None)
        message = getattr(self, "help_overlay_text", None)
        info_panel = getattr(self, "info_panel_frame", None)
        if overlay is None or message is None or info_panel is None:
            return

        if not bool(getattr(self, "_help_panel_expanded", False)):
            try:
                overlay.place_forget()
            except Exception:
                pass
            return

        try:
            self.root.update_idletasks()
        except Exception:
            pass

        root_w = max(640, int(self.root.winfo_width() or 0))
        overlay_w = min(760, max(420, int(root_w * 0.52)))
        wrap_w = max(320, overlay_w - 24)

        try:
            message.configure(width=wrap_w)
        except Exception:
            pass

        try:
            overlay.update_idletasks()
        except Exception:
            pass

        try:
            overlay_h = int(overlay.winfo_reqheight())
        except Exception:
            overlay_h = 180

        try:
            info_panel_h = int(info_panel.winfo_height() or 0)
        except Exception:
            info_panel_h = 0

        x = max(12, int((root_w - overlay_w) / 2))
        y = max(12, int(self.root.winfo_height() - info_panel_h - overlay_h - 14))

        try:
            overlay.place(x=x, y=y, width=overlay_w, height=overlay_h)
            overlay.lift()
        except Exception:
            pass

    def _render_help_panel_text(self):
        widget = getattr(self, "status_text", None)
        if widget is None:
            return

        display_text = self._build_help_strip_text(getattr(self, "_status_full_text", ""))
        try:
            current_text = widget.get("1.0", tk.END).strip()
        except Exception:
            current_text = None

        try:
            if current_text != display_text:
                widget.config(state=tk.NORMAL)
                widget.delete("1.0", tk.END)
                widget.insert(tk.END, display_text)
                widget.yview_moveto(0.0)
                widget.config(state=tk.DISABLED)
        except Exception:
            pass

    def _apply_help_panel_visual_state(self):
        widget = getattr(self, "status_text", None)
        frame = getattr(self, "info_panel_frame", None)
        overlay = getattr(self, "help_overlay_frame", None)
        overlay_title = getattr(self, "help_overlay_title_lbl", None)
        overlay_text = getattr(self, "help_overlay_text", None)
        if widget is None or frame is None or overlay is None or bool(getattr(self, "_help_panel_apply_in_progress", False)):
            return

        self._help_panel_apply_in_progress = True
        palette = self.palette
        expanded = bool(getattr(self, "_help_overlay_forced_visible", False))
        self._help_panel_expanded = expanded

        panel_bg = palette["surface_info"] if expanded else palette["console_bg"]
        overlay_bg = palette["surface_info"]
        overlay_border = palette["accent"]
        text_fg = palette["console_fg"]
        try:
            try:
                frame.configure(bg=panel_bg)
            except Exception:
                pass

            try:
                widget.configure(
                    bg=panel_bg,
                    fg=text_fg,
                    insertbackground=text_fg,
                    height=max(1, int(self._help_panel_default_height)),
                )
                widget.yview_moveto(0.0)
            except Exception:
                pass

            try:
                overlay.configure(
                    bg=overlay_bg,
                    highlightbackground=overlay_border,
                    highlightcolor=overlay_border,
                )
            except Exception:
                pass

            try:
                if overlay_title is not None:
                    overlay_title.configure(
                        bg=overlay_bg,
                        fg=palette["fg"],
                        text="Rozwinięta pomoc  |  ESC",
                    )
            except Exception:
                pass

            try:
                if overlay_text is not None:
                    overlay_body = str(
                        getattr(self, "_help_overlay_forced_text", "")
                        if bool(getattr(self, "_help_overlay_forced_visible", False))
                        else (getattr(self, "_status_full_text", "") or self.default_status_message)
                    )
                    overlay_text.configure(
                        bg=overlay_bg,
                        fg=palette["fg"],
                        text=overlay_body,
                    )
            except Exception:
                pass

            self._sync_global_terminal_toggle_state()
            self._sync_free_mode_assistant_toggle_state()
            self._apply_global_terminal_visual_state()
            self._render_help_panel_text()
            self._set_help_overlay_visible(expanded)
        finally:
            self._help_panel_apply_in_progress = False

    def _sync_global_terminal_toggle_state(self):
        btn = getattr(self, "_global_terminal_toggle_btn", None)
        if btn is None:
            return

        palette = self.palette
        base_bg = palette["console_bg"]
        border = palette["panel_border"]
        fg = palette["guide"]

        try:
            btn.configure(
                bg=base_bg,
                fg=fg,
                activebackground=base_bg,
                activeforeground=fg,
                highlightbackground=border,
                highlightcolor=border,
                relief=tk.FLAT,
            )
        except Exception:
            pass

    def _is_free_mode_assistant_available(self, context: FreeModeAssistantContext | None = None) -> bool:
        try:
            if context is not None:
                return not context.is_empty()
            if not self._is_free_mode_assistant_context():
                return False
            return not self._get_free_mode_assistant_context().is_empty()
        except Exception:
            return False

    def _sync_free_mode_assistant_toggle_state(
        self,
        context: FreeModeAssistantContext | None = None,
        *,
        available: bool | None = None,
    ):
        btn = getattr(self, "_free_mode_assistant_toggle_btn", None)
        if btn is None:
            return

        palette = self.palette
        base_bg = palette["console_bg"]
        border = palette["panel_border"]
        success = palette["success"]
        guide = palette["guide"]
        muted = palette["muted_dim"]
        if available is None:
            available = self._is_free_mode_assistant_available(context)
        active = bool(getattr(self, "_free_mode_assistant_enabled", False)) and available
        bg = success if active else base_bg
        fg = palette["accent_text"] if active else (guide if available else muted)

        try:
            btn.configure(
                text="AS",
                state=tk.NORMAL if available else tk.DISABLED,
                bg=bg,
                fg=fg,
                activebackground=bg,
                activeforeground=fg,
                disabledforeground=muted,
                highlightbackground=border,
                highlightcolor=border,
                highlightthickness=1 if active else 0,
                relief=tk.SUNKEN if active else tk.FLAT,
            )
        except Exception:
            pass

    def _close_free_mode_assistant_overlay(self):
        self._free_mode_assistant_enabled = False
        overlay = getattr(self, "_free_mode_assistant_overlay", None)
        if overlay is not None:
            try:
                overlay.hide()
            except Exception:
                pass
        try:
            self._sync_free_mode_assistant_toggle_state()
        except Exception:
            pass

    def toggle_free_mode_assistant(self):
        if not self._is_free_mode_assistant_context():
            self._free_mode_assistant_enabled = False
            overlay = getattr(self, "_free_mode_assistant_overlay", None)
            if overlay is not None:
                try:
                    overlay.hide()
                except Exception:
                    pass
            self._sync_free_mode_assistant_toggle_state(FreeModeAssistantContext())
            return

        context = self._get_free_mode_assistant_context()
        if context.is_empty():
            self._free_mode_assistant_enabled = False
            overlay = getattr(self, "_free_mode_assistant_overlay", None)
            if overlay is not None:
                try:
                    overlay.hide()
                except Exception:
                    pass
            self._sync_free_mode_assistant_toggle_state(context)
            return

        self._free_mode_assistant_enabled = not bool(getattr(self, "_free_mode_assistant_enabled", False))
        self._refresh_free_mode_assistant(context=context)

    # Delegates from app_global_terminal are bound after class creation.

    def _on_help_panel_text_configure(self, event=None):
        if bool(getattr(self, "_help_panel_apply_in_progress", False)):
            return
        self._render_help_panel_text()
        if bool(getattr(self, "_help_panel_expanded", False)):
            self._schedule_help_overlay_placement()

    # Delegates from app_menu_dropdown are bound after class creation.

    def show_assistant_message(self, message: str, icon: str = "info"):
        """Confirm a small action in the global strip and the open AS panel."""
        self.update_status(message, icon)
        overlay = getattr(self, "_free_mode_assistant_overlay", None)
        if overlay is not None and bool(getattr(overlay, "_visible", False)):
            overlay.show_notice(message)

    def update_status(self, message: str, icon: str = "info"):
        """Aktualizuje główny panel wskazówek (zapobiega migotaniu)."""
        new_text = self._format_help_panel_message(message, icon=icon)
        try:
            previous_text = str(getattr(self, "_status_full_text", "") or "")
            self._status_full_text = new_text
            if previous_text != new_text:
                self.root.update_idletasks()
        except Exception:
            pass
        try:
            self._apply_help_panel_visual_state()
        except Exception:
            pass
    
    def set_processing(self, processing: bool):
        with self._processing_state_lock:
            self._manual_processing = bool(processing)
        self._refresh_processing_state()

    def _refresh_processing_state(self):
        with self._processing_state_lock:
            processing = bool(self._manual_processing or self._exclusive_operation)
            self.is_processing = processing

        def update():
            try:
                self.root.config(cursor="wait" if processing else "")
            except Exception:
                pass

        try:
            self.root.after(0, update)
        except Exception:
            try:
                update()
            except Exception:
                pass

    def get_active_exclusive_operation_label(self) -> str:
        with self._processing_state_lock:
            active = dict(self._exclusive_operation or {})
        return str(active.get("label") or "").strip()

    def try_begin_exclusive_operation(self, owner: str, label: str) -> tuple[bool, str]:
        owner_key = str(owner or "").strip()
        label_text = str(label or "").strip() or owner_key or "operacja"
        denial_message = ""
        started = False

        with self._processing_state_lock:
            active = dict(self._exclusive_operation or {})
            active_owner = str(active.get("owner") or "").strip()
            active_label = str(active.get("label") or "").strip()

            if active_owner:
                denial_message = (
                    f"W aplikacji trwa juz: {active_label or 'inna operacja'}.\n\n"
                    f"Poczekaj na jej zakonczenie, zanim uruchomisz: {label_text}."
                )
            else:
                self._exclusive_operation = {"owner": owner_key, "label": label_text}
                started = not active_owner

        if denial_message:
            try:
                self.append_global_terminal(
                    f"[BUSY] Odrzucono start: {label_text}. Trwa: {active_label or 'inna operacja'}.",
                    source="APP",
                    tag="terminal_warning",
                )
            except Exception:
                pass
            self._refresh_processing_state()
            return False, denial_message

        self._refresh_processing_state()
        if started:
            try:
                self.append_global_terminal(
                    f"[LOCK] Rozpoczeto: {label_text}. Pozostale procesy sa chwilowo zablokowane.",
                    source="APP",
                    tag="terminal_info",
                )
            except Exception:
                pass
        return True, ""

    def end_exclusive_operation(self, owner: str):
        owner_key = str(owner or "").strip()
        finished_label = ""
        with self._processing_state_lock:
            active = dict(self._exclusive_operation or {})
            active_owner = str(active.get("owner") or "").strip()
            if active_owner and active_owner == owner_key:
                finished_label = str(active.get("label") or "").strip()
                self._exclusive_operation = None
        self._refresh_processing_state()
        if finished_label:
            try:
                self.append_global_terminal(
                    f"[LOCK] Zakonczono: {finished_label}. Mozesz uruchomic kolejny proces.",
                    source="APP",
                    tag="terminal_info",
                )
            except Exception:
                pass

    def set_campaign_mode(self, active: bool):
        """
        Przełącza tryb kampanii.
        Uwaga: jeśli użytkownik ręcznie wszedł w tryb swobodny, nie nadpisujemy tego automatem.
        """
        self.campaign_mode_active = bool(active)
        self.update_campaign_tab_access()

    def set_free_mode_assistant_context_override(self, key: str, context, owner: tk.Misc | None = None) -> None:
        self._free_mode_assistant_context_override_key = str(key or "").strip()
        self._free_mode_assistant_context_override = FreeModeAssistantContext.from_value(context)
        self._free_mode_assistant_context_override_owner = self._normalize_free_mode_assistant_owner(owner)
        self.notify_free_mode_assistant_context_changed()

    def clear_free_mode_assistant_context_override(self, key: str | None = None) -> None:
        current_key = str(getattr(self, "_free_mode_assistant_context_override_key", "") or "")
        if key is not None and str(key or "").strip() != current_key:
            return
        self._free_mode_assistant_context_override_key = ""
        self._free_mode_assistant_context_override = None
        self._free_mode_assistant_context_override_owner = None
        self.notify_free_mode_assistant_context_changed()

    def _normalize_free_mode_assistant_owner(self, owner: tk.Misc | None = None) -> tk.Misc | None:
        if owner is None:
            return None
        try:
            top = owner.winfo_toplevel()
        except Exception:
            top = owner
        try:
            if top is None or not bool(top.winfo_exists()):
                return None
        except Exception:
            return None
        return top

    def _get_free_mode_assistant_owner(self) -> tk.Misc:
        try:
            override = FreeModeAssistantContext.from_value(
                getattr(self, "_free_mode_assistant_context_override", None)
            )
            owner = self._normalize_free_mode_assistant_owner(
                getattr(self, "_free_mode_assistant_context_override_owner", None)
            )
            if not override.is_empty() and owner is not None:
                return owner
        except Exception:
            pass
        return self.root

    def _ensure_free_mode_assistant_overlay(self, owner: tk.Misc | None = None) -> FreeModeAssistantOverlay | None:
        owner = self._normalize_free_mode_assistant_owner(owner) or self.root
        overlay = getattr(self, "_free_mode_assistant_overlay", None)
        current_owner = getattr(self, "_free_mode_assistant_overlay_owner", None)
        if overlay is not None and current_owner is owner:
            return overlay

        if overlay is not None:
            try:
                overlay.hide()
            except Exception:
                pass

        try:
            overlay = FreeModeAssistantOverlay(
                owner,
                on_close=self._close_free_mode_assistant_overlay,
            )
            self._free_mode_assistant_overlay = overlay
            self._free_mode_assistant_overlay_owner = owner
            try:
                owner.bind("<Configure>", lambda _event: self._schedule_free_mode_assistant_placement(), add="+")
            except Exception:
                pass
            return overlay
        except Exception:
            return None

    def _restore_free_mode_assistant_owner(self, owner: tk.Misc | None = None) -> None:
        owner = self._normalize_free_mode_assistant_owner(owner)
        if owner is None or owner is self.root:
            return
        try:
            if bool(getattr(owner, "_aat_skip_window_recovery", False)) and owner.state() in {"iconic", "withdrawn"}:
                return
        except tk.TclError:
            return
        try:
            if str(owner.state() or "") == "iconic":
                owner.deiconify()
        except Exception:
            pass
        try:
            owner.lift()
        except Exception:
            pass
        try:
            owner.focus_force()
        except Exception:
            pass

    def _get_selected_tab_key(self):
        try:
            selected_widget = str(self.notebook.select())
        except Exception:
            return None

        loading_placeholder = str(getattr(self, "_lazy_tab_loading_placeholder_widget", "") or "").strip()
        if loading_placeholder and selected_widget == loading_placeholder:
            loading_key = str(getattr(self, "_lazy_tab_loading_key", "") or "").strip()
            if loading_key:
                return loading_key

        for key, tab in self.tabs.items():
            try:
                if str(tab.frame) == selected_widget:
                    return key
            except Exception:
                pass

        return None

    def _is_free_mode_assistant_context(self) -> bool:
        try:
            override = FreeModeAssistantContext.from_value(
                getattr(self, "_free_mode_assistant_context_override", None)
            )
            if not override.is_empty():
                return True
        except Exception:
            pass
        tab_key = self._get_selected_tab_key()
        return bool(tab_key)

    def _get_default_free_mode_assistant_context(self, tab_key: str | None) -> FreeModeAssistantContext:
        if tab_key == "campaign":
            return FreeModeAssistantContext(
                location="[Z1] Kampania i graf przejść",
                goal=(
                    "Z1 prowadzi projekt przez graf bramek T01-T06. Tutaj wybierasz aktywną ścieżkę, "
                    "sprawdzasz zasoby, uruchamiasz pracę bramki i formalnie domykasz przejścia."
                ),
                current=(
                    "Źródłem prawdy dla gotowości bramki jest kontrakt zasobów i zatwierdzony przyrost pracy, "
                    "nie sam kolor pola w UI."
                ),
                workflow=(
                    "Wybierz jedną bramkę elektrodą, jeśli etap ma kilka możliwych przejść.",
                    "W Zasobach sprawdź kontrakt: wymagane, opcjonalne, do kontroli, spełnione albo brakujące.",
                    "W Pracy wybierz akcję prowadzącą do właściwej karty roboczej: Z2, Z3/PZ2, Z3/PZ3 albo Z4.",
                    "Po powrocie do grafu zatwierdź bramkę tylko wtedy, gdy rozumiesz, co zostanie dodane do projektu.",
                    "T06 domyka tor treningowy albo świadome pominięcie treningu i przenosi projekt do kolejnej iteracji.",
                ),
                glossary=(
                    "Z1 = graf kampanii",
                    "T01-T06 = bramki przejść między etapami",
                    "kontrakt zasobu = wymaganie bramki policzone na danych projektu",
                    "przyrost iteracji = materiał zatwierdzony w bieżącym cyklu",
                    "zasób dziedziczony = materiał z poprzednich iteracji",
                ),
                caution="T01 i T02 są alternatywami startowymi iteracji. Po zatwierdzonym przyroście jednej ścieżki druga nie powinna przejmować tego samego cyklu pracy.",
                references=("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
            )
        if tab_key == "help":
            return FreeModeAssistantContext(
                location="[Z5] Instrukcja i architektura",
                goal="Ta zakładka jest miejscem dokumentacji: wyjaśnia przepływ programu, pojęcia, architekturę, eksport i metodykę badań.",
                current="Z5 opisuje obowiązujący model pracy aplikacji i odsyła do dokumentów źródłowych w katalogu docs.",
                workflow=(
                    "Użyj Z5, gdy chcesz sprawdzić znaczenie zakładek, etapów albo pojęć używanych w aplikacji.",
                    "W części Kampania sprawdzisz sens bramek T01-T06 i kontraktów O/AT/AZ.",
                    "W części Dane i kod znajdziesz mapę najważniejszych modułów.",
                    "W części Eksport i badania znajdziesz zasady pakietu .alprmodel, kalibracji i eksperymentów.",
                ),
                glossary=(
                    "Z5 = instrukcja i architektura",
                    "architektura = opis odpowiedzialności modułów i przepływów",
                    "workflow = kolejność pracy użytkownika",
                    "docs = katalog dokumentacji projektu",
                ),
                caution="Z5 nie wykonuje pracy na danych. To mapa i dokumentacja, a właściwe operacje robisz w Z1-Z4.",
                references=(
                    "docs/mapa_funkcji_i_kodu.md",
                    "docs/eksport_mobilny_kwantyzacja.md",
                    "docs/siatka_eksperymentow_mobilnych_alpr.md",
                ),
            )
        if tab_key == "annotation":
            return FreeModeAssistantContext(
                location="[Z2] Kontrola i anotacje tablic",
                goal=(
                    "Z2 pracuje na obrazach i ramkach tablic. Może tworzyć anotacje ręcznie, wspierać się modelem "
                    "albo kontrolować importowane AT przed dodaniem ich do puli projektu."
                ),
                current=(
                    "Materiał staje się zasobem projektu dopiero po kontroli i statusie [OK]. Import AT jest wejściem do kontroli, "
                    "nie gotowym wynikiem."
                ),
                workflow=(
                    "Wybierz albo potwierdź zbiór obrazów O.",
                    "Utwórz anotacje tablic ręcznie, uruchom autoanotację MT albo skontroluj importowane AT.",
                    "Popraw ramki/poligony tablic i oznacz poprawne pozycje statusem [OK].",
                    "Zapisz wynik kontroli i wróć do bramki kampanii albo wyeksportuj dataset tablic w trybie swobodnym.",
                ),
                glossary=(
                    "O = zbiór obrazów wejściowych",
                    "AT = anotacje tablic zgodne z O",
                    "do kontroli = materiał wymagający sprawdzenia w Z2",
                    "status [OK] = obraz/tablica zaakceptowana do dalszych etapów",
                    "MT = model tablic używany do autoanotacji",
                ),
                caution="Nie zatwierdzaj obrazów bez poprawnych ramek tablic. Tylko [OK] zasila pulę projektową i liczy się jako przyrost iteracji.",
                references=("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
            )
        if tab_key == "characters":
            return FreeModeAssistantContext(
                location="[Z3] Znaki na wyodrębnionych tablicach",
                goal=(
                    "Z3 prowadzi pracę nad znakami: od cropów tablic, przez pipeline YB/YS/OCR i korekty manualne, "
                    "do źródłowego datasetu znaków eksportowanego w PZ3."
                ),
                current=(
                    "PZ2 przygotowuje i kontroluje znaki. PZ3 tworzy artefakt datasetu. Dopiero eksport PZ3 jest zasobem "
                    "do treningu modelu MZ."
                ),
                workflow=(
                    "PZ1 wyodrębnia tablice z zatwierdzonego źródła.",
                    "PZ2 wykrywa i koryguje znaki blokami YB, YS i OCR.",
                    "Ręczne ramki i ręcznie wpisane znaki mają pierwszeństwo przed wynikiem automatu.",
                    "Status perfect oznacza zgodność odczytu, ramek i układu tablicy z regułami projektu.",
                    "PZ3 zbiera perfecty, obsługuje opcjonalny CVAT i eksportuje źródłowy dataset znaków.",
                ),
                glossary=(
                    "YB = detekcja ramek znaków",
                    "YS = rozpoznanie klas znaków na istniejących ramkach",
                    "OCR = odczyt tekstu całej tablicy",
                    "MB/MS = ręczna ramka lub ręcznie wpisany znak",
                    "AZ = anotacje znaków do datasetu MZ",
                ),
                caution="Detekcja ma pomagać, ale nie powinna nadpisywać manuali. Przy zdjęciu z kilkoma rejestracjami odczyt może pasować do jednego z kandydatów zapisanych w nazwie pliku.",
                references=("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
            )
        if tab_key == "training":
            mobile_export = get_mobile_export_assistant_context()
            return FreeModeAssistantContext(
                location="[Z4] Trening i analiza",
                goal=(
                    "Z4 przygotowuje warianty datasetów, uruchamia trening, porównuje modele, wybiera wynik bramki "
                    "i eksportuje model mobilny lub kompletny pakiet ALPR .alprmodel."
                ),
                current=(
                    "Model startowy treningu i wynik bramki to dwa różne wybory. Model startowy rozpoczyna run, "
                    "a wynik bramki wskazuje zatwierdzony model projektu."
                ),
                workflow=(
                    "W PZ1 wybierz źródło i utwórz wariant treningowy datasetu.",
                    "W PZ2 wybierz wariant, model startowy, parametry i uruchom trening.",
                    "Po treningu sprawdź historię, ranking i porównania na wspólnym torze testowym.",
                    "Jako wynik bramki wskaż model świadomie: z historii, rankingu albo sekcji wyboru wyniku.",
                    "Do Androida eksportuj pojedynczy MP, MT lub MZ do diagnostyki i podmiany modelu albo pakiet MT+MZ / MP+MT+MZ do testu end-to-end.",
                ),
                glossary=(
                    "wariant = konkretna wersja datasetu",
                    "split = train / val / test",
                    "model startowy = checkpoint użyty na wejściu treningu",
                    "wynik bramki = model zatwierdzony jako rezultat pracy",
                    "data.yaml = opis datasetu YOLO; przy INT8 służy do kalibracji",
                    "model mobilny = jeden MP, MT albo MZ; pakiet ALPR = MT+MZ lub MP+MT+MZ",
                    "kalibracja = pomiar zakresów aktywacji na reprezentatywnych obrazach",
                    "INT8 = wariant kwantyzowany wymagający kalibracji",
                ),
                caution=mobile_export.get("caution", "") or "Źródła datasetu przygotowuje PZ1; PZ2 trenuje na wybranym wariancie.",
                references=(
                    "docs/eksport_mobilny_kwantyzacja.md",
                    "docs/siatka_eksperymentow_mobilnych_alpr.md",
                    "docs/podbudowa_literaturowa_metodyki_testow_alpr.md",
                ),
            )
        return FreeModeAssistantContext(
            location="Kontekst roboczy aplikacji",
            goal="AS opisuje bieżący ekran, kolejny sensowny krok, pojęcia i ryzyka wynikające z przepływu danych.",
            current="W kampanii obowiązują bramki T01-T06 i kontrakty zasobów. W trybie swobodnym użytkownik sam pilnuje spójności artefaktów.",
            workflow=(
                "Z2 przygotowuje i kontroluje tablice.",
                "Z3 przygotowuje znaki oraz źródłowy dataset znaków.",
                "Z4 tworzy warianty datasetu, trenuje, rankinguje i eksportuje modele.",
                "Z5 dokumentuje przepływy, architekturę i metodykę badań.",
            ),
            glossary=("Z2 = tablice", "Z3 = znaki", "Z4 = dataset, trening i eksport", "Z5 = instrukcja"),
            caution="Przy pracy poza kampanią nie mieszaj artefaktów z różnych źródeł bez świadomej kontroli zgodności.",
            references=("docs/mapa_funkcji_i_kodu.md",),
        )

    def _get_free_mode_assistant_context(self) -> FreeModeAssistantContext:
        try:
            override = FreeModeAssistantContext.from_value(
                getattr(self, "_free_mode_assistant_context_override", None)
            )
            if not override.is_empty():
                return override
        except Exception:
            pass

        tab_key = self._get_selected_tab_key()
        if tab_key is None:
            return FreeModeAssistantContext()

        tab = self.tabs.get(tab_key) if isinstance(getattr(self, "tabs", None), dict) else None
        provider = getattr(tab, "get_free_mode_assistant_context", None)
        if callable(provider):
            try:
                provided = FreeModeAssistantContext.from_value(provider())
                if not provided.is_empty():
                    return provided
            except Exception:
                pass
        return self._get_default_free_mode_assistant_context(tab_key)

    def _schedule_free_mode_assistant_placement(self):
        if getattr(self, "_free_mode_assistant_place_after_id", None):
            return
        try:
            self._free_mode_assistant_place_after_id = self.root.after_idle(self._place_free_mode_assistant)
        except Exception:
            self._free_mode_assistant_place_after_id = None

    def _place_free_mode_assistant(self):
        self._free_mode_assistant_place_after_id = None
        if not bool(getattr(self, "_free_mode_assistant_enabled", False)):
            return
        owner = self._get_free_mode_assistant_owner()
        try:
            if bool(getattr(owner, "_aat_skip_window_recovery", False)) and owner.state() in {"iconic", "withdrawn"}:
                return
        except tk.TclError:
            return
        overlay = self._ensure_free_mode_assistant_overlay(owner)
        if overlay is None:
            return
        try:
            moving_callback = getattr(owner, "_aat_window_is_moving_callback", None)
            if callable(moving_callback) and bool(moving_callback()):
                self._free_mode_assistant_place_after_id = self.root.after(240, self._place_free_mode_assistant)
                return
        except Exception:
            pass
        is_root_owner = owner is self.root
        try:
            overlay.place(
                notebook=getattr(self, "notebook", None) if is_root_owner else None,
                info_panel=getattr(self, "info_panel_frame", None) if is_root_owner else None,
            )
            self._restore_free_mode_assistant_owner(owner)
        except Exception:
            pass

    def _refresh_free_mode_assistant(self, context: FreeModeAssistantContext | None = None):
        overlay = getattr(self, "_free_mode_assistant_overlay", None)

        context_possible = self._is_free_mode_assistant_context()
        if not context_possible:
            self._sync_free_mode_assistant_toggle_state(FreeModeAssistantContext(), available=False)
            if overlay is not None:
                overlay.hide()
            return

        if not bool(getattr(self, "_free_mode_assistant_enabled", False)):
            self._sync_free_mode_assistant_toggle_state(available=True)
            if overlay is not None:
                overlay.hide()
            return

        if context is None:
            context = self._get_free_mode_assistant_context()
        self._sync_free_mode_assistant_toggle_state(context)
        if context.is_empty():
            if overlay is not None:
                overlay.hide()
            return

        owner = self._get_free_mode_assistant_owner()
        overlay = self._ensure_free_mode_assistant_overlay(owner)
        if overlay is None:
            return

        overlay.update_context(context, palette=getattr(self, "palette", {}))
        overlay.show()
        self._restore_free_mode_assistant_owner(owner)
        self._schedule_free_mode_assistant_placement()

    def notify_free_mode_assistant_context_changed(self):
        if getattr(self, "_free_mode_assistant_refresh_after_id", None):
            return
        try:
            self._free_mode_assistant_refresh_after_id = self.root.after_idle(self._refresh_free_mode_assistant_from_after)
        except Exception:
            self._free_mode_assistant_refresh_after_id = None

    def _refresh_free_mode_assistant_from_after(self):
        self._free_mode_assistant_refresh_after_id = None
        self._refresh_free_mode_assistant()


    def update_campaign_tab_access(self):
        """
        Steruje dostępnością głównych zakładek.

        Zasada:
        - tryb swobodny: wszystko dostępne
        - tryb aktywnego projektu:
        * Help jest zawsze dostępny
        * aktualnie otwarta zakładka robocza pozostaje aktywna
        * Wizard Z1 wraca tylko przez dedykowane CTA w module
        """

        from ..campaign_manager import CAMPAIGN

        active = str(CAMPAIGN.get_active_project_name() or "").strip()
        active_step_tab_key = None

        if active and bool(getattr(self, "campaign_free_mode", False)):
            self.campaign_free_mode = False

        # Brak aktywnego projektu = pełna swoboda.
        if not active:
            self.campaign_mode_active = False

            for key, tab in self.tabs.items():
                try:
                    self.notebook.tab(str(tab.frame), state="normal")
                except Exception:
                    pass

            self.refresh_main_tab_labels()
            self.refresh_window_title()
            self.notify_free_mode_assistant_context_changed()
            return

        # aktywny projekt
        self.campaign_mode_active = True
        step_to_tab = {
            1: "campaign",
            2: "annotation",
            3: "characters",
            4: "training",
        }
        active_step_tab_key = step_to_tab.get(CAMPAIGN.get_current_step(), "campaign")

        selected_key = self._get_selected_tab_key()

        allowed = {"help"}
        if selected_key == "campaign" or active_step_tab_key == "campaign":
            allowed.add("campaign")

        # Jeżeli użytkownik jest już w zakładce roboczej, zostaw ją aktywną,
        # ale nie odblokowuj Z1 do ręcznego kliknięcia.
        if selected_key in {"annotation", "characters", "training"}:
            allowed.add(selected_key)

        for key, tab in self.tabs.items():
            try:
                state = "normal" if key in allowed else "disabled"
                self.notebook.tab(str(tab.frame), state=state)
            except Exception:
                pass

        self.refresh_main_tab_labels(active_tab_key=active_step_tab_key)
        self.refresh_window_title()
        self.notify_free_mode_assistant_context_changed()

    def get_tab_index(self, tab_key: str) -> int:
        if tab_key not in self.tabs:
            raise KeyError(f"Unknown tab key: {tab_key}")

        target_widget = str(self.tabs[tab_key].frame)

        for i, widget_name in enumerate(self.notebook.tabs()):
            if str(widget_name) == target_widget:
                return i

        raise KeyError(f"Tab widget not found in notebook for key: {tab_key}")


    def select_tab(self, tab_key: str):
        if tab_key not in self.tabs:
            raise KeyError(f"Unknown tab key: {tab_key}")

        previous_target = str(getattr(self, "_controlled_tab_target_key", "") or "").strip()
        self._controlled_tab_target_key = tab_key
        try:
            current = self.tabs.get(tab_key)
            if isinstance(current, _LazyNotebookTab):
                try:
                    self.notebook.tab(str(current.frame), state="normal")
                    self.notebook.select(str(current.frame))
                except Exception:
                    pass
                tab = self._ensure_tab_loaded(tab_key, select=True)
            else:
                tab = self._ensure_tab_loaded(tab_key, select=False)
            if tab_key == "campaign":
                self._allow_campaign_tab_once = True
            self.notebook.select(str(tab.frame))
            self._refresh_menu_badge()
        finally:
            self._controlled_tab_target_key = previous_target

    def open_controlled_tab(self, tab_key: str):
        if tab_key not in self.tabs:
            raise KeyError(f"Unknown tab key: {tab_key}")

        previous_target = str(getattr(self, "_controlled_tab_target_key", "") or "").strip()
        self._controlled_tab_target_key = tab_key
        try:
            current = self.tabs.get(tab_key)
            if isinstance(current, _LazyNotebookTab):
                silent_char_graph_entry = bool(
                    tab_key == "characters"
                    and getattr(self, "_suppress_characters_lazy_first_paint_overlay_once", False)
                )
                if not silent_char_graph_entry:
                    try:
                        # Programowe wejscie do Z3/Z4 musi wskazac loaderowi wlasciwy
                        # cel. Inaczej straznik kampanii widzi Z1 i moze dociagnac
                        # fallback, np. Z2, rownolegle do ladowania Z3.
                        self.notebook.tab(str(current.frame), state="normal")
                        self.notebook.select(str(current.frame))
                    except Exception:
                        pass
                tab = self._ensure_tab_loaded(tab_key, select=not silent_char_graph_entry)
            else:
                tab = self._ensure_tab_loaded(tab_key, select=False)

            tab_widget = str(tab.frame)
            self.notebook.tab(tab_widget, state="normal")
            if tab_key == "campaign":
                self._allow_campaign_tab_once = True
            self.notebook.select(tab_widget)
            self.update_campaign_tab_access()
            self._refresh_menu_badge()
            return
        finally:
            self._controlled_tab_target_key = previous_target

    def _guard_campaign_navigation(self, event=None):
        try:
            from ..campaign_manager import CAMPAIGN
        except Exception:
            return

        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        if not active_project:
            return
        if bool(getattr(self, "campaign_free_mode", False)):
            self.campaign_free_mode = False

        if bool(getattr(self, "_campaign_nav_guard_in_progress", False)):
            return
        if str(getattr(self, "_controlled_tab_target_key", "") or "").strip():
            return

        selected_key = self._get_selected_tab_key()
        if selected_key != "campaign":
            return

        if bool(getattr(self, "_allow_campaign_tab_once", False)):
            self._allow_campaign_tab_once = False
            return

        fallback_key = str(getattr(self, "_last_allowed_main_tab_key", "") or "").strip()
        if fallback_key not in {"annotation", "characters", "training"}:
            step_to_tab = {
                2: "annotation",
                3: "characters",
                4: "training",
            }
            try:
                fallback_key = step_to_tab.get(int(CAMPAIGN.get_current_step() or 0), "annotation")
            except Exception:
                fallback_key = "annotation"

        if fallback_key not in self.tabs or fallback_key == "campaign":
            return

        def _restore_previous_tab():
            self._campaign_nav_guard_in_progress = True
            try:
                tab = self._ensure_tab_loaded(fallback_key, select=False)
                self.notebook.select(str(tab.frame))
                self.update_status(
                    "Do wizarda kampanii wracaj przez dedykowany przycisk w module, a nie przez klikniecie zakladki Z1.",
                    "warning",
                )
            except Exception:
                pass
            finally:
                self._campaign_nav_guard_in_progress = False

        try:
            self.notebook.after_idle(_restore_previous_tab)
        except Exception:
            _restore_previous_tab()

    def _on_main_notebook_button_press(self, event=None):
        notebook = getattr(self, "notebook", None)
        if notebook is None or event is None:
            return

        try:
            tab_index = notebook.index(f"@{int(event.x)},{int(event.y)}")
            tab_widget = str(notebook.tabs()[tab_index])
        except Exception:
            return

        requested_key = None
        loading_placeholder = str(getattr(self, "_lazy_tab_loading_placeholder_widget", "") or "").strip()
        if loading_placeholder and tab_widget == loading_placeholder:
            requested_key = str(getattr(self, "_lazy_tab_loading_key", "") or "").strip() or None
        for key, tab in getattr(self, "tabs", {}).items():
            if requested_key:
                break
            try:
                if str(tab.frame) == tab_widget:
                    requested_key = key
                    break
            except Exception:
                pass

        if not requested_key:
            return

        self._last_user_requested_main_tab_key = requested_key
        self._last_user_requested_main_tab_at = time.perf_counter()

        if bool(getattr(self, "_lazy_tab_load_in_progress", False)):
            loading_key = str(getattr(self, "_lazy_tab_loading_key", "") or "").strip()
            if requested_key != loading_key:
                # Last requested tab wins. FIFO would replay stale clicks after a slow lazy-load.
                self._lazy_tab_deferred_select_key = requested_key

    def _on_main_notebook_tab_changed(self, event=None):
        if bool(getattr(self, "_notebook_placeholder_dispose_in_progress", False)):
            return
        if bool(getattr(self, "_lazy_tab_load_in_progress", False)):
            loading_key = str(getattr(self, "_lazy_tab_loading_key", "") or "").strip()
            selected_key = self._get_selected_tab_key()
            if loading_key and selected_key != loading_key:
                last_user_key = str(getattr(self, "_last_user_requested_main_tab_key", "") or "").strip()
                try:
                    last_user_age = time.perf_counter() - float(getattr(self, "_last_user_requested_main_tab_at", 0.0) or 0.0)
                except Exception:
                    last_user_age = 9999.0
                is_recent_user_request = bool(selected_key and selected_key == last_user_key and last_user_age <= 2.0)
                if is_recent_user_request and selected_key in getattr(self, "tabs", {}):
                    # Keep only the newest user intent while another tab is still loading.
                    self._lazy_tab_deferred_select_key = selected_key
                else:
                    try:
                        loading_tab = getattr(self, "tabs", {}).get(loading_key)
                        loading_frame = getattr(loading_tab, "frame", None)
                        if loading_frame is not None:
                            self.notebook.select(str(loading_frame))
                    except Exception:
                        pass
                return
        self._guard_campaign_navigation(event)
        selected_key = self._get_selected_tab_key()
        pending_lazy_key = str(getattr(self, "_lazy_tab_pending_select_key", "") or "").strip()
        if pending_lazy_key and selected_key != pending_lazy_key:
            pending_tab = getattr(self, "tabs", {}).get(pending_lazy_key)
            if pending_tab is not None and not isinstance(pending_tab, _LazyNotebookTab):
                try:
                    self.root.after_idle(
                        lambda key=pending_lazy_key, tab=pending_tab: self._restore_lazy_tab_selection(key, tab)
                    )
                except Exception:
                    self._restore_lazy_tab_selection(pending_lazy_key, pending_tab)
                return
        elif pending_lazy_key and selected_key == pending_lazy_key:
            self._lazy_tab_pending_select_key = ""
        if selected_key and self._is_lazy_tab_key(selected_key):
            self._ensure_tab_loaded(selected_key, select=True)
            return
        if selected_key == "annotation":
            try:
                annotation_tab = getattr(self, "tabs", {}).get("annotation")
                from ..campaign_manager import CAMPAIGN
                active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
                if active_project:
                    if bool(getattr(self, "campaign_free_mode", False)):
                        self.campaign_free_mode = False
                    ensure_context = (
                        getattr(annotation_tab, "ensure_campaign_context_ready_for_active_project", None)
                        if annotation_tab is not None
                        else None
                    )
                    if callable(ensure_context):
                        self.root.after_idle(ensure_context)
                    reset_selection = (
                        getattr(annotation_tab, "reset_preview_selection_to_first_visible_on_tab_entry", None)
                        if annotation_tab is not None
                        else None
                    )
                    if callable(reset_selection):
                        self.root.after(
                            240,
                            lambda reset=reset_selection: reset(reason="campaign-tab-entry"),
                        )
                else:
                    ensure_preview = (
                        getattr(annotation_tab, "ensure_free_mode_session_preview_ready", None)
                        if annotation_tab is not None
                        else None
                    )
                    if callable(ensure_preview):
                        self.root.after_idle(ensure_preview)
                    reset_selection = (
                        getattr(annotation_tab, "reset_preview_selection_to_first_visible_on_tab_entry", None)
                        if annotation_tab is not None
                        else None
                    )
                    if callable(reset_selection):
                        self.root.after(
                            240,
                            lambda reset=reset_selection: reset(reason="free-tab-entry"),
                        )
            except Exception:
                pass
        elif selected_key == "characters":
            try:
                from ..campaign_manager import CAMPAIGN

                active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
            except Exception:
                active_project = ""
            if not active_project:
                try:
                    character_tab = getattr(self, "tabs", {}).get("characters")
                    if character_tab is not None and not isinstance(character_tab, _LazyNotebookTab):
                        ensure_free_mode = getattr(character_tab, "ensure_free_mode_context_ready", None)
                        if callable(ensure_free_mode):
                            self.root.after_idle(ensure_free_mode)
                except Exception:
                    pass
        elif selected_key == "training":
            try:
                training_tab = getattr(self, "tabs", {}).get("training")
                ensure_layout = (
                    getattr(training_tab, "ensure_visible_layout_ready", None)
                    if training_tab is not None
                    else None
                )
                if callable(ensure_layout):
                    self.root.after_idle(ensure_layout)
            except Exception:
                pass
        if selected_key in {"annotation", "characters", "training"}:
            self._last_allowed_main_tab_key = selected_key
        self._refresh_menu_badge()
        self._save_active_main_tab_preference()
        self.notify_free_mode_assistant_context_changed()


    
    # Delegates from app_shutdown are bound after class creation.

bind_app_theme_runtime(AutoAnnotationApp)
bind_app_delegates(AutoAnnotationApp)
