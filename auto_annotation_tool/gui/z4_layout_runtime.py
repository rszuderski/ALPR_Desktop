#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 transient layout, logs and scroll helpers extracted from tab_training.py."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import datetime
import time
import webbrowser
import csv
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import (
    CONFIG,
    YOLO_AVAILABLE,
    AVAILABLE_POSE_MODELS,
    AVAILABLE_DETECT_MODELS,
    PIL_AVAILABLE,
    get_torch_module,
    get_yolo_class,
    is_cuda_available,
    logger,
)
from ..icons import IconManager
from ..validators import validate_model_file, format_yolo_model_identity
from ..training import YOLOPoseTrainer, TrainingHistory, TrainingStatus, DatasetCreator, DatasetSplitter
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .modal_scroll_guard import event_is_over_foreign_toplevel
from .zoomable_canvas import ZoomableCanvas
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    finish_campaign_step4,
    get_campaign_training_target as campaign_flow_get_training_target,
    open_campaign_step4_entry,
    poll_training_completion,
    restore_step4_campaign_project_state,
    set_campaign_context,
    set_campaign_training_target as campaign_flow_set_training_target,
)
from .z4_flow_models import (
    CharYoloDatasetSourceAdapter,
    PlateXmlImagesSourceAdapter,
    TrainingSource,
    TrainingSourceStats,
)
from .z4_free_mode_flow import (
    refresh_free_training_route_cards,
    refresh_free_training_route_ui,
    update_step4_notebook_mode,
)
from .z4_shared_ui import (
    accept_training_input_context,
    clear_step4_guidance,
    guide_step4_builder_action,
    guide_step4_finish_action,
    guide_step4_next_action,
    guide_step4_route_selection,
    mark_step4_dataset_ready,
    open_step4_dataset_stage,
    refresh_step4_campaign_builder_inputs_ui,
    refresh_step4_analysis_tab_visibility,
    refresh_step4_campaign_navigation_ui,
    refresh_step4_dataset_mode_ui,
    refresh_step4_training_inputs_mode_ui,
    set_step4_dataset_mode,
    sync_step4_analysis_nav_buttons,
    step4_dataset_go_back,
    step4_dataset_go_next,
    step4_train_go_back,
)
from . import z4_dataset_sources
from . import z4_training_metrics
from . import z4_dataset_builder
from . import z4_analysis_ranking
from . import z4_training_runtime
from . import z4_campaign_state
from . import z4_dataset_validation
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None



def _reset_step4_transient_ui(
    self,
    datasets_dir: Path | None = None,
    target: str | None = None,
    clear_builder_inputs: bool = False,
    require_route_selection: bool | None = None,
):
    if target is not None:
        target = str(target).strip().lower()
        if target not in ("char", "plate"):
            target = "char"
        self._campaign_training_target = target
        self._step4_dataset_mode = target

    if require_route_selection is None:
        require_route_selection = bool(CAMPAIGN.get_active_project_name())

    self._step4_route_selected = not bool(require_route_selection)
    self._step4_train_unlocked = False

    split_out_default = (
        str(datasets_dir / "[NAZWA_ZRODLA]_Split_[DATA_I_CZAS]")
        if datasets_dir is not None
        else str(self._get_datasets_base_dir() / "[NAZWA_ZRODLA]_Split_[DATA_I_CZAS]")
    )
    ds_out_default = (
        str(datasets_dir / "Plates_CVAT_[DATA_I_CZAS]")
        if datasets_dir is not None
        else str(self._get_datasets_base_dir() / "Plates_CVAT_[DATA_I_CZAS]")
    )

    try:
        self.split_src_var.set("")
    except Exception:
        pass

    try:
        self.split_out_var.set(split_out_default)
    except Exception:
        pass

    try:
        self.dataset_var.set("")
    except Exception:
        pass

    try:
        self.ds_out_var.set(ds_out_default)
    except Exception:
        pass

    if clear_builder_inputs:
        try:
            self.cvat_xml_var.set("")
        except Exception:
            pass

        try:
            self.cvat_images_var.set("")
        except Exception:
            pass

    self.current_run_id = None
    self._plots_paths = []
    self._plot_original_path = None
    self._plot_photo = None
    self._plot_img_id = None
    try:
        self._close_analysis_dialog()
    except Exception:
        pass
    self._pending_campaign_model_type = None
    self._step4_builder_log_visible = False
    self._step4_train_log_visible = False
    self._current_training_dataset_is_pose = None
    self._step4_campaign_finish_ready = False

    try:
        self.plots_list.delete(0, tk.END)
    except Exception:
        pass

    try:
        if hasattr(self, "plot_canvas"):
            self.plot_canvas.original_image = None
            self.plot_canvas.photo_image = None
            self.plot_canvas.image_id = None
            self.plot_canvas.delete("all")
    except Exception:
        pass

    try:
        self.tree.selection_remove(*self.tree.selection())
    except Exception:
        pass

    try:
        self._set_train_progress_values(overall=0.0, epoch=0.0)
    except Exception:
        pass

    try:
        self._set_step4_process_console_text(
            "Oczekuję na rozpoczęcie treningu lub walidacji...\n"
            "Terminal procesu jest gotowy na dane z Ultralytics.\n"
        )
    except Exception:
        pass

    try:
        self.ds_progress_var.set(0.0)
    except Exception:
        pass

    try:
        self._style_training_success_label(self.ds_status)
        self.ds_status.configure(text="Gotowy")
    except Exception:
        pass

    try:
        self.split_progress_var.set(0.0)
    except Exception:
        pass

    try:
        self._style_training_success_label(self.split_status)
        self.split_status.configure(text="Gotowy")
    except Exception:
        pass

    try:
        self._set_split_feedback_visibility(False)
    except Exception:
        pass

    try:
        self.step4_builder_log_text.configure(state=tk.NORMAL)
        self.step4_builder_log_text.delete(1.0, tk.END)
        self.step4_builder_log_text.configure(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self.val_model_var.set("")
    except Exception:
        pass

    try:
        self.val_data_var.set("")
    except Exception:
        pass

    try:
        self.val_split_var.set("val")
    except Exception:
        pass

    try:
        self.val_status.configure(text="Gotowy", foreground="gray")
    except Exception:
        pass

    try:
        self.rank_progress_var.set(0.0)
    except Exception:
        pass

    try:
        self.rank_status.configure(text="Gotowy", foreground="gray")
    except Exception:
        pass

    try:
        self.rank_models_dir.set(str(self._get_ranking_models_default_dir()))
    except Exception:
        pass

    try:
        self.rank_data_dir.set("")
    except Exception:
        pass

    if self._training_completion_poll_job is not None:
        try:
            self.frame.after_cancel(self._training_completion_poll_job)
        except Exception:
            pass
        self._training_completion_poll_job = None

    try:
        self._load_history()
    except Exception:
        pass

    try:
        self._set_training_ui_idle_state()
    except Exception:
        pass

    try:
        self._set_step4_builder_log_visibility(False)
    except Exception:
        pass

    try:
        self._set_step4_train_log_visibility(False)
    except Exception:
        pass

    try:
        target_tab = self.tab_dataset if CAMPAIGN.get_active_project_name() else self.tab_train
        self.main_nb.select(target_tab)
    except Exception:
        pass

    try:
        self.right_nb.select(self.hist_tab)
    except Exception:
        pass

    try:
        self._sync_step4_analysis_nav_buttons()
    except Exception:
        pass

    try:
        self._refresh_step4_dataset_mode_ui()
    except Exception:
        pass

    try:
        self._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        self._update_training_dataset_hint()
    except Exception:
        pass

    try:
        if require_route_selection:
            self._guide_step4_route_selection()
        else:
            self._clear_step4_guidance()
    except Exception:
        pass


def set_campaign_training_target(self, target: str):
    campaign_flow_set_training_target(self, target)


def get_campaign_training_target(self) -> str:
    return campaign_flow_get_training_target(self)


def _append_step4_builder_log(self, message: str):
    if not hasattr(self, "step4_builder_log_text"):
        return

    try:
        self.step4_builder_log_text.configure(state=tk.NORMAL)
        self.step4_builder_log_text.insert(tk.END, message.rstrip() + "\n")
        self.step4_builder_log_text.see(tk.END)
        self.step4_builder_log_text.configure(state=tk.DISABLED)
    except Exception:
        pass


def _set_step4_builder_log_visibility(self, visible: bool):
    if not hasattr(self, "step4_builder_log_frame"):
        return

    self._step4_builder_log_visible = bool(visible)
    try:
        self.step4_builder_log_frame.pack_forget()
    except Exception:
        pass
    try:
        if hasattr(self, "step4_builder_tools"):
            self.step4_builder_tools.pack_forget()
    except Exception:
        pass

    if self._step4_builder_log_visible:
        try:
            if hasattr(self.app, "show_global_terminal"):
                self.app.show_global_terminal()
        except Exception:
            pass
    return


def _toggle_step4_builder_log(self):
    try:
        if hasattr(self.app, "toggle_global_terminal"):
            self.app.toggle_global_terminal()
            return
    except Exception:
        pass
    self._set_step4_builder_log_visibility(
        not getattr(self, "_step4_builder_log_visible", False)
    )


def _set_split_feedback_visibility(self, visible: bool):
    if not hasattr(self, "split_feedback_frame"):
        return

    if visible:
        try:
            self.split_feedback_frame.pack_forget()
        except Exception:
            pass
        try:
            parent = getattr(self.split_feedback_frame, "master", None)
            action_inner = getattr(self, "step4_split_action_inner", None)
            if parent is action_inner:
                self.split_feedback_frame.pack(fill=tk.X, pady=(8, 0))
            else:
                self.split_feedback_frame.pack(
                    fill=tk.X,
                    pady=(0, 8),
                    after=self.btn_step4_split_frame
                )
        except Exception:
            try:
                self.split_feedback_frame.pack(fill=tk.X, pady=(8, 0))
            except Exception:
                pass
        try:
            self._configure_train_progress_styles()
        except Exception:
            pass
    else:
        self.split_feedback_frame.pack_forget()


def _set_step4_process_console_text(self, message: str):
    if not hasattr(self, "train_log_console"):
        return

    try:
        self.train_log_console.config(state=tk.NORMAL)
        self.train_log_console.delete(1.0, tk.END)
        if message:
            self.train_log_console.insert(tk.END, message)
        self.train_log_console.config(state=tk.DISABLED)
    except Exception:
        pass


def _sync_train_left_scrollregion(self, event=None):
    canvas = getattr(self, "train_left_canvas", None)
    if canvas is None:
        return

    try:
        canvas.configure(scrollregion=canvas.bbox("all"))
    except Exception:
        pass


def _sync_dataset_mode_scrollregion(self, event=None):
    canvas = getattr(self, "ds_mode_canvas", None)
    if canvas is None:
        return

    try:
        canvas.configure(scrollregion=canvas.bbox("all"))
    except Exception:
        pass


def _sync_train_left_canvas_width(self, event=None):
    canvas = getattr(self, "train_left_canvas", None)
    if canvas is None:
        return

    try:
        inset = max(0, int(getattr(self, "_train_left_content_inset", 0)))
        available_width = max(50, int(canvas.winfo_width()) - (2 * inset))
        max_width = max(50, int(getattr(self, "_train_left_content_max_width", available_width)))
        width = min(available_width, max_width)
        canvas.coords(self.train_left_content_window, inset, 0)
        canvas.itemconfigure(self.train_left_content_window, width=width)
    except Exception:
        pass

    try:
        self._update_training_dataset_hint_wraplength()
    except Exception:
        pass

    try:
        content = getattr(self, "train_left_content", None)
        if content is not None:
            content.update_idletasks()
    except Exception:
        pass

    try:
        self._sync_train_left_scrollregion()
    except Exception:
        pass


def _sync_dataset_mode_canvas_width(self, event=None):
    canvas = getattr(self, "ds_mode_canvas", None)
    if canvas is None:
        return

    try:
        width = max(50, int(canvas.winfo_width()) - 2)
        canvas.itemconfigure(self.ds_mode_host_window, width=width)
    except Exception:
        pass

    try:
        content = getattr(self, "ds_mode_host", None)
        if content is not None:
            content.update_idletasks()
    except Exception:
        pass

    try:
        self._sync_dataset_mode_scrollregion()
    except Exception:
        pass


def ensure_visible_layout_ready(self, *, force: bool = False) -> bool:
    canvas = getattr(self, "train_left_canvas", None)
    if canvas is None:
        return False

    try:
        if not force and not bool(self.frame.winfo_ismapped()):
            return False
    except Exception:
        if not force:
            return False

    def _run_sync():
        self._training_visible_layout_after_id = None

        try:
            self.frame.update_idletasks()
        except Exception:
            pass

        try:
            self._sync_train_left_canvas_width()
        except Exception:
            pass

        try:
            self._sync_train_left_scrollregion()
        except Exception:
            pass

        try:
            self._init_train_pane_layout()
        except Exception:
            pass

        try:
            self._refresh_step4_analysis_tab_visibility()
        except Exception:
            pass

        try:
            self._sync_step4_analysis_nav_buttons()
        except Exception:
            pass

        try:
            self.frame.update_idletasks()
        except Exception:
            pass

        try:
            self._sync_train_left_scrollregion()
        except Exception:
            pass

    try:
        pending = getattr(self, "_training_visible_layout_after_id", None)
        if pending is not None:
            self.frame.after_cancel(pending)
    except Exception:
        pass
    self._training_visible_layout_after_id = None

    _run_sync()

    try:
        self._training_visible_layout_after_id = self.frame.after(90, _run_sync)
    except Exception:
        pass

    return True


def _widget_contains_point(self, widget, x_root: int, y_root: int) -> bool:
    if widget is None:
        return False
    try:
        wx = int(widget.winfo_rootx())
        wy = int(widget.winfo_rooty())
        return wx <= x_root < (wx + int(widget.winfo_width())) and wy <= y_root < (wy + int(widget.winfo_height()))
    except Exception:
        return False


def _mousewheel_units(self, event) -> int:
    return self._inertial_scroll.mousewheel_units(event)


def _train_left_canvas_overflows(self) -> bool:
    canvas = getattr(self, "train_left_canvas", None)
    return self._scroll_canvas_overflows(canvas)


def _dataset_mode_canvas_overflows(self) -> bool:
    canvas = getattr(self, "ds_mode_canvas", None)
    return self._scroll_canvas_overflows(canvas)


def _scroll_canvas_overflows(self, canvas) -> bool:
    if canvas is None:
        return False

    try:
        bbox = canvas.bbox("all")
        if not bbox:
            return False
        content_height = int(bbox[3]) - int(bbox[1])
        viewport_height = int(canvas.winfo_height())
        return content_height > viewport_height + 1
    except Exception:
        return False


def _on_train_left_global_mousewheel(self, event):
    if event_is_over_foreign_toplevel(getattr(self, "frame", None), event):
        return "break"

    canvas = getattr(self, "train_left_canvas", None)
    if canvas is None:
        return None

    if self._training_combobox_scroll_guard_active():
        return "break"
    if self._mousewheel_event_from_combobox(event):
        return "break"

    if self._inertial_scroll.scroll_canvas_if_targeted(
        canvas,
        event,
        pointer_widget=self.frame,
        overflow_checker=self._train_left_canvas_overflows,
    ):
        return "break"
    return None


def _on_dataset_mode_global_mousewheel(self, event):
    if event_is_over_foreign_toplevel(getattr(self, "frame", None), event):
        return "break"

    canvas = getattr(self, "ds_mode_canvas", None)
    if canvas is None:
        return None

    if self._training_combobox_scroll_guard_active():
        return "break"
    if self._mousewheel_event_from_combobox(event):
        return "break"

    if self._inertial_scroll.scroll_canvas_if_targeted(
        canvas,
        event,
        pointer_widget=self.frame,
        overflow_checker=self._dataset_mode_canvas_overflows,
    ):
        return "break"
    return None


def _restore_scroll_canvas_focus(self, canvas):
    if canvas is None:
        return
    try:
        canvas.focus_set()
    except Exception:
        pass


def _redirect_child_mousewheel_to_canvas(self, event, canvas):
    if canvas is None:
        return None

    if self._inertial_scroll.redirect_child_mousewheel_to_canvas(
        event,
        canvas,
        pointer_widget=self.frame,
        overflow_checker=lambda c=canvas: self._scroll_canvas_overflows(c),
    ):
        self._restore_scroll_canvas_focus(canvas)
        return "break"

    self._restore_scroll_canvas_focus(canvas)
    return "break"


def _redirect_combobox_mousewheel_to_canvas(self, event, canvas):
    # Zamknięty Combobox w Tk potrafi zmienić wartość samym kółkiem myszy.
    # W Z4 to zbyt ryzykowne: scroll nad polem ma przewijać panel, a wybór
    # modelu ma następować dopiero po świadomym rozwinięciu listy.
    widget = getattr(event, "widget", None)
    if self._combobox_popdown_visible(widget):
        self._mark_training_combobox_scroll_guard(widget)
        return "break"

    self._mark_training_combobox_scroll_guard(widget, hold_ms=250)
    if self._inertial_scroll.redirect_child_mousewheel_to_canvas(
        event,
        canvas,
        pointer_widget=self.frame,
        overflow_checker=lambda c=canvas: self._scroll_canvas_overflows(c),
    ):
        self._restore_scroll_canvas_focus(canvas)
        return "break"

    self._restore_scroll_canvas_focus(canvas)
    return "break"


def _register_training_scroll_guard_combobox(self, widget) -> None:
    if widget is None or bool(getattr(widget, "_training_scroll_guard_bound", False)):
        return

    try:
        widget._training_scroll_guard_bound = True
    except Exception:
        pass

    registered = list(getattr(self, "_training_scroll_guard_comboboxes", []) or [])
    if widget not in registered:
        registered.append(widget)
        self._training_scroll_guard_comboboxes = registered

    def _mark(_event=None, w=widget):
        self._mark_training_combobox_scroll_guard(w)
        return None

    def _release(_event=None):
        self._schedule_training_combobox_guard_poll(delay_ms=120)
        return None

    for sequence in ("<ButtonPress-1>", "<KeyPress-Down>", "<Alt-Down>", "<FocusIn>"):
        try:
            widget.bind(sequence, _mark, add="+")
        except Exception:
            pass
    for sequence in ("<<ComboboxSelected>>", "<Escape>", "<Return>", "<FocusOut>"):
        try:
            widget.bind(sequence, _release, add="+")
        except Exception:
            pass


def _mark_training_combobox_scroll_guard(self, widget=None, *, hold_ms: int = 1200) -> None:
    try:
        self._training_combobox_scroll_guard_widget = widget
        self._training_combobox_scroll_guard_until = time.perf_counter() + (max(150, int(hold_ms)) / 1000.0)
    except Exception:
        return
    self._schedule_training_combobox_guard_poll(delay_ms=120)


def _schedule_training_combobox_guard_poll(self, *, delay_ms: int = 180) -> None:
    frame = getattr(self, "frame", None)
    if frame is None:
        return

    pending = getattr(self, "_training_combobox_scroll_guard_after_id", None)
    if pending is not None:
        try:
            frame.after_cancel(pending)
        except Exception:
            pass

    try:
        self._training_combobox_scroll_guard_after_id = frame.after(
            max(50, int(delay_ms)),
            self._poll_training_combobox_scroll_guard,
        )
    except Exception:
        self._training_combobox_scroll_guard_after_id = None


def _poll_training_combobox_scroll_guard(self) -> None:
    self._training_combobox_scroll_guard_after_id = None
    if self._any_training_combobox_popdown_visible():
        self._mark_training_combobox_scroll_guard(
            getattr(self, "_training_combobox_scroll_guard_widget", None),
            hold_ms=900,
        )
        return

    try:
        if time.perf_counter() >= float(getattr(self, "_training_combobox_scroll_guard_until", 0.0) or 0.0):
            self._training_combobox_scroll_guard_until = 0.0
            self._training_combobox_scroll_guard_widget = None
            return
    except Exception:
        self._training_combobox_scroll_guard_until = 0.0
        self._training_combobox_scroll_guard_widget = None
        return

    self._schedule_training_combobox_guard_poll(delay_ms=180)


def _combobox_popdown_visible(self, widget) -> bool:
    if widget is None:
        return False
    try:
        if not bool(widget.winfo_exists()):
            return False
    except Exception:
        return False

    try:
        popdown = widget.tk.call("ttk::combobox::PopdownWindow", str(widget))
        return bool(int(widget.tk.call("winfo", "viewable", popdown)))
    except Exception:
        return False


def _any_training_combobox_popdown_visible(self) -> bool:
    for widget in list(getattr(self, "_training_scroll_guard_comboboxes", []) or []):
        if self._combobox_popdown_visible(widget):
            return True
    return False


def _training_combobox_scroll_guard_active(self) -> bool:
    if self._any_training_combobox_popdown_visible():
        return True
    try:
        return time.perf_counter() < float(getattr(self, "_training_combobox_scroll_guard_until", 0.0) or 0.0)
    except Exception:
        return False


def _widget_is_combobox_or_popdown(self, widget) -> bool:
    visited = set()
    while widget is not None and id(widget) not in visited:
        visited.add(id(widget))
        try:
            class_name = str(widget.winfo_class())
        except Exception:
            class_name = ""
        try:
            widget_path = str(widget).lower()
        except Exception:
            widget_path = ""

        if class_name in {"TCombobox", "ComboboxPopdown"}:
            return True
        if class_name in {"Listbox", "Toplevel", "Frame"} and (
            "popdown" in widget_path or "combobox" in widget_path
        ):
            return True

        try:
            widget = widget.master
        except Exception:
            widget = None
    return False


def _mousewheel_event_from_combobox(self, event) -> bool:
    if self._widget_is_combobox_or_popdown(getattr(event, "widget", None)):
        self._mark_training_combobox_scroll_guard(getattr(event, "widget", None))
        return True

    try:
        x_root = int(getattr(event, "x_root", 0) or self.frame.winfo_pointerx())
        y_root = int(getattr(event, "y_root", 0) or self.frame.winfo_pointery())
        root = self.frame.winfo_toplevel()
        pointer_widget = root.winfo_containing(x_root, y_root)
    except Exception:
        pointer_widget = None

    if self._widget_is_combobox_or_popdown(pointer_widget):
        self._mark_training_combobox_scroll_guard(pointer_widget)
        return True
    return False


def _bind_scroll_canvas_children(self, root, canvas):
    if root is None or canvas is None:
        return

    release_focus_classes = {
        "TButton",
        "Button",
        "TCheckbutton",
        "Checkbutton",
        "TRadiobutton",
        "Radiobutton",
        "TScale",
        "Scale",
        "TCombobox",
        "Spinbox",
    }

    def _walk(widget):
        try:
            class_name = str(widget.winfo_class())
        except Exception:
            class_name = ""

        if class_name != "TCombobox":
            try:
                widget.bind("<MouseWheel>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
                widget.bind("<Button-4>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
                widget.bind("<Button-5>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
            except Exception:
                pass

        if class_name in release_focus_classes:
            try:
                widget.configure(takefocus=0)
            except Exception:
                pass
            try:
                widget.bind("<ButtonRelease-1>", lambda _e, c=canvas: self._restore_scroll_canvas_focus(c), add="+")
            except Exception:
                pass
            if class_name == "TCombobox":
                try:
                    self._register_training_scroll_guard_combobox(widget)
                except Exception:
                    pass
                try:
                    widget.bind("<MouseWheel>", lambda e, c=canvas: self._redirect_combobox_mousewheel_to_canvas(e, c), add="+")
                    widget.bind("<Button-4>", lambda e, c=canvas: self._redirect_combobox_mousewheel_to_canvas(e, c), add="+")
                    widget.bind("<Button-5>", lambda e, c=canvas: self._redirect_combobox_mousewheel_to_canvas(e, c), add="+")
                except Exception:
                    pass
                try:
                    widget.bind("<<ComboboxSelected>>", lambda _e, c=canvas: self._restore_scroll_canvas_focus(c), add="+")
                except Exception:
                    pass

        for child in widget.winfo_children():
            _walk(child)

    _walk(root)
