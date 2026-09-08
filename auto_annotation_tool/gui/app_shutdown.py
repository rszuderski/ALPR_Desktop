#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sprzątanie runtime i bezpieczne zamykanie aplikacji."""

import gc
import tkinter as tk
from tkinter import messagebox

from ..config import logger
from ..utils import cleanup_gpu_memory
from .help_manager import HELP
from .lazy_notebook_tab import _LazyNotebookTab

def _cancel_after_handle_safely(self, scheduler, handle) -> None:
    if scheduler is None or not handle:
        return
    try:
        scheduler.after_cancel(handle)
    except Exception:
        pass

def _cancel_dynamic_after_callbacks(self) -> None:
    owners = [self]
    try:
        owners.extend(list(self._iter_loaded_tabs()))
    except Exception:
        pass

    suffixes = (
        "_after_id",
        "_after_ids",
        "_after_job",
        "_poll_job",
        "_refresh_job",
        "_layout_after_id",
        "_watchdog_job",
    )
    for owner in owners:
        scheduler = getattr(owner, "frame", None) or getattr(owner, "root", None) or self.root
        for attr_name, value in list(vars(owner).items()):
            if not attr_name.endswith(suffixes):
                continue
            handles = value if isinstance(value, (list, tuple, set)) else [value]
            for handle in list(handles):
                self._cancel_after_handle_safely(scheduler, handle)
            try:
                setattr(owner, attr_name, [] if isinstance(value, list) else None)
            except Exception:
                pass

def _release_runtime_references(self) -> None:
    try:
        self._cancel_dynamic_after_callbacks()
    except Exception:
        pass

    for tab in list(self._iter_loaded_tabs()):
        queue_obj = getattr(tab, "_ui_dispatch_queue", None)
        if queue_obj is not None:
            try:
                while True:
                    queue_obj.get_nowait()
            except Exception:
                pass

        try:
            release_gpu = getattr(tab, "release_gpu_resources_for_training", None)
            if callable(release_gpu):
                release_gpu()
        except Exception:
            pass

        for attr_name in ("annotator", "detector", "char_detector", "ocr_engine", "plate_ocr"):
            obj = getattr(tab, attr_name, None)
            if obj is None:
                continue
            for method_name in ("shutdown", "stop", "unload_models", "unload", "close"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
                    break
            try:
                setattr(tab, attr_name, None)
            except Exception:
                pass

        for attr_name, value in list(vars(tab).items()):
            try:
                if attr_name in {"preview_metadata", "_preview_legend_image_cache", "_preview_image_meta_cache"}:
                    if hasattr(value, "clear"):
                        value.clear()
                elif attr_name.endswith("_cache") and hasattr(value, "clear"):
                    value.clear()
                elif attr_name in {"_current_photo", "current_photo", "preview_photo", "_wizard_header_metro_photo"}:
                    setattr(tab, attr_name, None)
            except Exception:
                pass

        for attr_name in (
            "current_annotations",
            "_pending_source_image_map",
            "_preview_image_path_map",
            "_preview_list_display_indices",
            "_preview_list_display_index_map",
            "_preview_list_frozen_filename_order",
            "_preview_list_frozen_bucket_snapshot",
            "preview_plate_ids",
            "_preview_base_plate_ids",
            "_listbox_pid_by_index",
        ):
            value = getattr(tab, attr_name, None)
            if value is None:
                continue
            try:
                if hasattr(value, "clear"):
                    value.clear()
                else:
                    setattr(tab, attr_name, None)
            except Exception:
                try:
                    setattr(tab, attr_name, None)
                except Exception:
                    pass

    try:
        if hasattr(HELP, "_cache") and hasattr(HELP._cache, "clear"):
            HELP._cache.clear()
    except Exception:
        pass

    try:
        cleanup_gpu_memory()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def _flush_loaded_tab_runtime_state(self) -> None:
    for tab_name, tab in getattr(self, "tabs", {}).items():
        if isinstance(tab, _LazyNotebookTab):
            continue
        try:
            force_save = getattr(tab, "_force_save_all", None)
            if callable(force_save):
                force_save()
        except Exception as e:
            logger.debug(f"Nie udalo sie wymusic pelnego zapisu zakladki {tab_name}: {e}")

        try:
            on_app_close = getattr(tab, "_on_app_close", None)
            if callable(on_app_close):
                try:
                    on_app_close(None)
                except TypeError:
                    on_app_close()
        except Exception as e:
            logger.debug(f"Nie udalo sie wykonac haka zamkniecia zakladki {tab_name}: {e}")

        try:
            flush_session = getattr(tab, "flush_free_mode_session_state", None)
            if callable(flush_session):
                flush_session()
        except Exception as e:
            logger.debug(f"Nie udalo sie zapisac stanu zakladki {tab_name}: {e}")

        try:
            flush_preview = getattr(tab, "_flush_scheduled_preview_metadata_save", None)
            if callable(flush_preview):
                flush_preview()
        except Exception as e:
            logger.debug(f"Nie udalo sie zapisac odlozonych anotacji Z3/PZ2 ({tab_name}): {e}")

        try:
            sync_campaign_ok = getattr(tab, "_sync_campaign_char_repair_approved_run_to_project_source", None)
            if callable(sync_campaign_ok):
                sync_campaign_ok(reason="shutdown", refresh_effective_source=True)
        except Exception as e:
            logger.debug(f"Nie udalo sie zsynchronizowac zatwierdzen T06 przy zamykaniu ({tab_name}): {e}")

        try:
            preview_metadata = getattr(tab, "preview_metadata", None)
            persist_preview = getattr(tab, "_persist_preview_metadata", None)
            if preview_metadata and callable(persist_preview):
                persist_preview(success_message=None, refresh_list=False)
        except Exception as e:
            logger.debug(f"Nie udalo sie wymusic zapisu metadata Z3/PZ2 ({tab_name}): {e}")

def _on_closing(self):
    if getattr(self, "_closing_in_progress", False):
        return

    self._closing_in_progress = True

    try:
        busy = bool(getattr(self, "is_processing", False))
        for tab in self._iter_loaded_tabs():
            try:
                if getattr(tab, "is_processing", False):
                    busy = True
                    break
                trainer = getattr(tab, "trainer", None)
                if trainer is not None and getattr(trainer, "is_training", False):
                    busy = True
                    break
            except Exception:
                pass

        if busy:
            if not messagebox.askokcancel("Zamknij", "Przetwarzanie w toku. Na pewno zamknąć?"):
                self._closing_in_progress = False
                return

        try:
            self._save_active_main_tab_preference()
        except Exception:
            pass

        try:
            self._flush_loaded_tab_runtime_state()
        except Exception:
            pass

        for attr_name in (
            "_window_restore_after_id",
            "_window_restore_topmost_after_id",
            "_theme_refresh_after_id",
            "_startup_finalize_after_id",
            "_help_overlay_place_after_id",
            "_free_mode_assistant_place_after_id",
            "_free_mode_assistant_refresh_after_id",
        ):
            pending = getattr(self, attr_name, None)
            if not pending:
                continue
            try:
                self.root.after_cancel(pending)
            except Exception:
                pass
            try:
                setattr(self, attr_name, None)
            except Exception:
                pass

        try:
            self._cancel_dynamic_after_callbacks()
        except Exception:
            pass

        for tab_name, tab in getattr(self, "tabs", {}).items():
            if isinstance(tab, _LazyNotebookTab):
                continue
            try:
                flush_session = getattr(tab, "flush_free_mode_session_state", None)
                if callable(flush_session):
                    flush_session()
            except Exception as e:
                logger.debug(f"Nie udalo sie zapisac stanu zakladki {tab_name}: {e}")

            try:
                if hasattr(tab, "is_processing"):
                    tab.is_processing = False
            except Exception:
                pass

            try:
                stop_event = getattr(tab, "fast_test_stop", None)
                if stop_event is not None:
                    stop_event.set()
            except Exception:
                pass

            try:
                on_app_close = getattr(tab, "_on_app_close", None)
                if callable(on_app_close):
                    try:
                        on_app_close(None)
                    except TypeError:
                        on_app_close()
            except Exception:
                pass

            try:
                trainer = getattr(tab, "trainer", None)
                if trainer is not None:
                    shutdown = getattr(trainer, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                    elif hasattr(trainer, "stop_training"):
                        trainer.stop_training()
            except Exception:
                pass

            try:
                annotator = getattr(tab, "annotator", None)
                if annotator:
                    annotator.stop()
                    annotator.unload_models()
            except Exception:
                pass

        try:
            self._release_runtime_references()
        except Exception as e:
            logger.debug(f"Nie udało się wykonać pełnego cleanupu runtime podczas zamykania: {e}")

        for widget in list(self.root.winfo_children()):
            try:
                if isinstance(widget, tk.Toplevel):
                    try:
                        widget.grab_release()
                    except Exception:
                        pass
                    widget.destroy()
            except Exception:
                pass

        try:
            self.root.grab_release()
        except Exception:
            pass

        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception:
                pass

        try:
            self.root.update_idletasks()
        except Exception:
            pass

        try:
            self.root.quit()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            cleanup_gpu_memory()
            gc.collect()
        except Exception:
            pass
    finally:
        try:
            if self.root.winfo_exists():
                self._closing_in_progress = False
        except Exception:
            self._closing_in_progress = False
