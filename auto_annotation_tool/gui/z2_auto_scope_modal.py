#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 auto-annotation scope modal extracted from AnnotationTab."""

import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from ..config import AVAILABLE_DETECT_MODELS
from ..validators import format_yolo_model_identity, validate_model_file
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


def prompt_plate_auto_scope_choice(host, *, candidate_image_paths: list[Path] | None = None) -> dict | None:
    self = host
    scope_open_visible_state = {
        "annotations": copy.deepcopy(list(getattr(self, "current_annotations", []) or [])),
        "image_map": dict(getattr(self, "_preview_image_path_map", {}) or {}),
        "preview_index": getattr(self, "current_preview_index", None),
        "approved_filenames": set(getattr(self, "_preview_approved_filenames", set()) or set()),
        "campaign_pending_approved": set(getattr(self, "_campaign_pending_approved_filenames", set()) or set()),
        "hidden_project_approved": set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set()),
        "hidden_char_effective": set(getattr(self, "_campaign_hidden_char_effective_filenames", set()) or set()),
    }
    scope_info = self._collect_plate_auto_scope_candidates(
        candidate_image_paths=candidate_image_paths,
        protect_existing=True,
    )
    raw_all_paths = list(scope_info.get("raw_all_paths") or [])
    all_paths = list(scope_info.get("all_paths") or [])
    scope_protected_filenames = self._get_preview_auto_scope_protected_filenames()
    all_source_paths = raw_all_paths or all_paths
    scope_candidate_path_map = {
        str(path.name or "").strip().lower(): path
        for path in all_source_paths
        if str(getattr(path, "name", "") or "").strip()
    }
    protected_skip_count = int(scope_info.get("protected_skip_count", 0) or 0)
    selected_count = int(scope_info.get("selected_count", 0) or 0)
    selected_total_count = int(scope_info.get("selected_total_count", selected_count) or 0)
    selected_row_count = int(scope_info.get("selected_row_count", selected_total_count) or 0)
    bucket_paths = self._collect_plate_auto_scope_bucket_paths(
        candidate_image_paths=all_source_paths,
        protect_existing=False,
    )
    manual_bucket_paths = list(bucket_paths.get("manual") or [])
    auto_bucket_paths = list(bucket_paths.get("auto") or [])
    problem_bucket_paths = list(bucket_paths.get("problem") or [])

    if not all_source_paths:
        if protected_skip_count > 0:
            try:
                messagebox.showinfo(
                    "Brak obrazów do autoanotacji",
                    (
                        "W wybranym zakresie nie ma obrazów gotowych do autoanotacji.\n\n"
                        f"Pominięto {protected_skip_count} pozycji, bo są ręcznie anotowane albo mają status [OK]. "
                        "To mechanizm ochronny przed nadpisaniem gotowej pracy."
                    ),
                )
            except Exception:
                pass
        return None

    palette = getattr(self.app, "palette", {}) or {}
    result = {"choice": ""}

    dialog = tk.Toplevel(self.frame)
    dialog_done_var = tk.BooleanVar(master=dialog, value=False)
    self._plate_auto_scope_active_dialog = dialog
    try:
        owner_window = self.frame.winfo_toplevel()
    except Exception:
        owner_window = None
    try:
        dialog.withdraw()
    except Exception:
        pass
    dialog_styler = getattr(self.app, "style_dialog_window", None)
    if callable(dialog_styler):
        try:
            dialog_styler(dialog, title="Zakres autoanotacji", geometry="760x620", parent=self.frame)
            try:
                dialog.grab_release()
            except Exception:
                pass
        except Exception:
            try:
                dialog.title("Zakres autoanotacji")
                dialog.resizable(False, False)
            except Exception:
                pass
    else:
        try:
            dialog.title("Zakres autoanotacji")
            dialog.resizable(False, False)
        except Exception:
            pass
    try:
        if owner_window is not None:
            dialog.transient(owner_window)
    except Exception:
        pass

    panel_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f39c12")

    try:
        dialog.configure(
            bg=panel_bg,
            highlightbackground=panel_bg,
            highlightcolor=panel_bg,
        )
    except Exception:
        pass

    self._set_plate_auto_scope_modal_ui_lock(True)

    body_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(body_surface_builder):
        try:
            body_surface = body_surface_builder(dialog, tone="info")
        except Exception:
            body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            body_surface.pack(fill=tk.BOTH, expand=True)
    else:
        body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        body_surface.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(body_surface, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

    scroll_host = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    scroll_host.pack(fill=tk.BOTH, expand=True)

    scroll_canvas = tk.Canvas(
        scroll_host,
        bg=panel_bg,
        bd=0,
        highlightthickness=0,
    )
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll_bar = WebSlimScrollbar(
        scroll_host,
        command=scroll_canvas.yview,
        auto_hide=False,
    )
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_canvas.configure(yscrollcommand=scroll_bar.set)

    body_content = tk.Frame(scroll_canvas, bg=panel_bg, bd=0, highlightthickness=0)
    body_window_id = scroll_canvas.create_window((0, 0), window=body_content, anchor="nw")

    def _sync_scope_modal_scrollregion(_event=None):
        try:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        except Exception:
            pass

    def _sync_scope_modal_canvas_width(event=None):
        try:
            width = int(getattr(event, "width", 0) or scroll_canvas.winfo_width() or 0)
            if width > 0:
                scroll_canvas.itemconfigure(body_window_id, width=width)
        except Exception:
            pass

    def _capture_scope_modal_scroll_state() -> dict:
        try:
            yview = tuple(scroll_canvas.yview() or (0.0, 1.0))
        except Exception:
            yview = (0.0, 1.0)
        try:
            top_px = float(scroll_canvas.canvasy(0))
        except Exception:
            top_px = 0.0
        return {
            "fraction": float(yview[0] if yview else 0.0),
            "top_px": top_px,
        }

    def _restore_scope_modal_scroll_state(scroll_state: dict | None) -> None:
        if not isinstance(scroll_state, dict):
            return
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        try:
            scroll_canvas.update_idletasks()
            region = str(scroll_canvas.cget("scrollregion") or "").split()
            if len(region) >= 4:
                y1 = float(region[1])
                y2 = float(region[3])
                viewport_h = max(1.0, float(scroll_canvas.winfo_height() or 1.0))
                scroll_h = max(1.0, y2 - y1)
                max_top = max(0.0, scroll_h - viewport_h)
                top_px = min(max(float(scroll_state.get("top_px", 0.0) or 0.0), y1), y1 + max_top)
                scroll_canvas.yview_moveto(max(0.0, min(1.0, (top_px - y1) / scroll_h)))
            else:
                scroll_canvas.yview_moveto(
                    max(0.0, min(1.0, float(scroll_state.get("fraction", 0.0) or 0.0)))
                )
        except Exception:
            pass

    try:
        body_content.bind("<Configure>", _sync_scope_modal_scrollregion, add="+")
        scroll_canvas.bind("<Configure>", _sync_scope_modal_canvas_width, add="+")
    except Exception:
        pass

    def _scope_modal_restore_focus():
        try:
            scroll_canvas.focus_set()
        except Exception:
            pass

    def _scope_modal_redirect_wheel(event):
        try:
            if self._inertial_scroll.redirect_child_mousewheel_to_canvas(
                event,
                scroll_canvas,
                pointer_widget=dialog,
                overflow_checker=lambda: self._panel_canvas_overflows(scroll_canvas),
            ):
                _scope_modal_restore_focus()
                return "break"
        except Exception:
            pass
        _scope_modal_restore_focus()
        return "break"

    def _bind_scope_modal_scroll_children(widget):
        already_bound = bool(getattr(widget, "_scope_modal_scroll_bound", False))
        try:
            if not already_bound:
                widget.bind("<MouseWheel>", _scope_modal_redirect_wheel, add="+")
                widget.bind("<Button-4>", _scope_modal_redirect_wheel, add="+")
                widget.bind("<Button-5>", _scope_modal_redirect_wheel, add="+")
                widget._scope_modal_scroll_bound = True
        except Exception:
            pass
        try:
            children = list(widget.winfo_children())
        except Exception:
            children = []
        for child in children:
            _bind_scope_modal_scroll_children(child)

    def _restore_scope_modal_visibility(_event=None):
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        if bool(getattr(self, "_plate_auto_scope_selection_mode_active", False)):
            return
        try:
            if owner_window is not None and str(owner_window.state()) == "iconic":
                return
        except Exception:
            pass

        def _do_restore():
            try:
                if not dialog.winfo_exists():
                    return
            except Exception:
                return
            try:
                if str(dialog.state()) == "iconic":
                    dialog.deiconify()
            except Exception:
                pass
            try:
                dialog.lift()
            except Exception:
                pass
            try:
                dialog.focus_force()
            except Exception:
                pass
            _scope_modal_restore_focus()

        try:
            dialog.after(30, _do_restore)
        except Exception:
            _do_restore()

    owner_map_bind_id = None
    owner_focus_bind_id = None
    try:
        if owner_window is not None:
            owner_map_bind_id = owner_window.bind("<Map>", _restore_scope_modal_visibility, add="+")
            owner_focus_bind_id = owner_window.bind("<FocusIn>", _restore_scope_modal_visibility, add="+")
    except Exception:
        owner_map_bind_id = None
        owner_focus_bind_id = None

    title_lbl = tk.Label(
        body_content,
        text="Najpierw wybierz model tablic, potem zakres autoanotacji",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    title_lbl.pack(anchor=tk.W, fill=tk.X)

    tk.Label(
        body_content,
        text=(
            "Najważniejsze i wymagane ustawienie tego okna to model tablic (YOLO Pose). "
            "Bez niego autoanotacja nie wystartuje. Dopiero po wyborze modelu ustawisz zakres pracy, "
            "próg pewności, dopasowanie oraz opcjonalne wsparcie pojazdami."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 14))

    tk.Label(
        body_content,
        text=(
            "To okno możesz zostawić otwarte. Jeśli chcesz użyć zaznaczenia z listy, "
            "zaznacz obrazy po lewej stronie, a przycisk „Uruchom autoanotację” odblokuje się od razu."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9, "italic"),
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 14))

    progress_var = tk.DoubleVar(master=dialog, value=0.0)
    progress_status_var = tk.StringVar(
        master=dialog,
        value="Gotowe do uruchomienia autoanotacji.",
    )
    progress_counts_var = tk.StringVar(master=dialog, value="oczekiwanie na start")
    progress_file_var = tk.StringVar(master=dialog, value="")

    selected_mode_var = tk.StringVar(value="all")
    include_manual_var = tk.BooleanVar(value=False)
    include_auto_var = tk.BooleanVar(value=False)
    include_problem_var = tk.BooleanVar(value=False)
    protect_existing_var = tk.BooleanVar(value=True)

    def _count_scope_existing_plates(paths: list[Path] | tuple[Path, ...] | None) -> int:
        names = {
            str(getattr(path, "name", "") or "").strip().lower()
            for path in list(paths or [])
            if str(getattr(path, "name", "") or "").strip()
        }
        if not names:
            return 0
        seen_names: set[str] = set()
        plate_count = 0
        for ann in list(getattr(self, "current_annotations", []) or []):
            filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
            if not filename_key or filename_key in seen_names or filename_key not in names:
                continue
            seen_names.add(filename_key)
            try:
                plate_count += int(len(self._get_plate_detections(ann)))
            except Exception:
                pass
        return int(plate_count)

    def _format_scope_count(image_count: int, plate_count: int) -> str:
        return f"{int(image_count)} zdjęć / {int(plate_count)} tablic w obecnych anotacjach"

    def build_modal_toggle(
        parent,
        *,
        kind: str,
        variable,
        text: str,
        value=None,
        bg: str,
        fg_color: str,
        font=None,
        wraplength: int = 520,
        enabled_getter=None,
    ):
        row = tk.Frame(parent, bg=bg, bd=0, highlightthickness=0)
        indicator = tk.Canvas(
            row,
            width=18,
            height=18,
            bg=bg,
            bd=0,
            highlightthickness=0,
        )
        indicator.pack(side=tk.LEFT, anchor="n", pady=(1, 0))

        label = tk.Label(
            row,
            text=text,
            bg=bg,
            fg=fg_color,
            font=font or ("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=wraplength,
            bd=0,
            highlightthickness=0,
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        def _is_selected() -> bool:
            if kind == "radio":
                return str(variable.get() or "").strip() == str(value or "").strip()
            try:
                return bool(variable.get())
            except Exception:
                return False

        def _is_enabled() -> bool:
            if not callable(enabled_getter):
                return True
            try:
                return bool(enabled_getter())
            except Exception:
                return True

        def _redraw(*_args):
            indicator.delete("all")
            enabled = _is_enabled()
            outline = blend_hex_colors(border, bg, 0.15)
            active_fg = fg_color if enabled else muted
            active_success = success if enabled else blend_hex_colors(success, bg, 0.55)
            try:
                label.configure(fg=active_fg)
            except Exception:
                pass
            if kind == "radio":
                indicator.create_oval(2, 2, 16, 16, outline=outline, width=1.5, fill=bg)
                if _is_selected():
                    indicator.create_oval(6, 6, 12, 12, outline=active_success, fill=active_success, width=1.0)
            else:
                indicator.create_rectangle(2, 2, 16, 16, outline=outline, width=1.5, fill=bg)
                if _is_selected():
                    indicator.create_line(
                        4, 10, 7, 13, 14, 5,
                        fill=active_success,
                        width=2.4,
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND,
                    )

        def _choose(_event=None):
            if not _is_enabled():
                _redraw()
                return "break"
            if kind == "radio":
                variable.set(value)
            else:
                try:
                    variable.set(not bool(variable.get()))
                except Exception:
                    pass
            _redraw()

        for widget in (row, indicator, label):
            try:
                widget.bind("<Button-1>", _choose, add="+")
            except Exception:
                pass
        try:
            variable.trace_add("write", _redraw)
        except Exception:
            pass
        try:
            row.after_idle(_redraw)
        except Exception:
            _redraw()

        return {
            "row": row,
            "indicator": indicator,
            "label": label,
            "refresh": _redraw,
            "is_enabled": _is_enabled,
        }

    def build_option_card(parent, *, value: str, title: str, description: str):
        card = tk.Frame(
            parent,
            bg=field_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=12,
            pady=10,
        )
        card.pack(fill=tk.X, pady=(0, 10))
        toggle = build_modal_toggle(
            card,
            kind="radio",
            variable=selected_mode_var,
            value=value,
            text=title,
            bg=field_bg,
            fg_color=fg,
            font=("Segoe UI", 9, "bold"),
            wraplength=520,
        )
        toggle["row"].pack(fill=tk.X)

        description_var = tk.StringVar(value=description)
        tk.Label(
            card,
            textvariable=description_var,
            bg=field_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=500,
            padx=26,
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
        return card, description_var

    section_header_bg = blend_hex_colors(field_bg, panel_bg, 0.24)
    section_border = blend_hex_colors(success, panel_bg, 0.42)

    def build_section_card(parent, number: str, title: str, subtitle: str = "", *, pady=(0, 14)):
        shell = tk.Frame(
            parent,
            bg=field_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=section_border,
            highlightcolor=section_border,
        )
        shell.pack(fill=tk.X, pady=pady)
        header = tk.Frame(shell, bg=section_header_bg, bd=0, highlightthickness=0, padx=12, pady=8)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=str(number),
            bg=blend_hex_colors(success, section_header_bg, 0.12),
            fg=fg,
            font=("Segoe UI", 8, "bold"),
            width=3,
            anchor="center",
            justify=tk.CENTER,
        ).pack(side=tk.LEFT, anchor="n", padx=(0, 10))
        title_col = tk.Frame(header, bg=section_header_bg, bd=0, highlightthickness=0)
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_col,
            text=title,
            bg=section_header_bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        if str(subtitle or "").strip():
            tk.Label(
                title_col,
                text=subtitle,
                bg=section_header_bg,
                fg=muted,
                font=("Segoe UI", 8),
                anchor="w",
                justify=tk.LEFT,
                wraplength=540,
            ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        content = tk.Frame(shell, bg=field_bg, bd=0, highlightthickness=0, padx=12, pady=12)
        content.pack(fill=tk.X)
        return content

    settings_card = build_section_card(
        body_content,
        "1",
        "Model do autoanotacji",
        "Sprawdź status modeli i wybierz model tablic dla bieżącego uruchomienia.",
        pady=(2, 14),
    )

    tk.Label(
        settings_card,
        text="Model tablic (YOLO Pose)",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    tk.Label(
        settings_card,
        text=(
            "Autoanotacja nie ruszy bez aktywnego modelu tablic. Model projektu jest domyślny, "
            "a opcje awaryjne służą do jednorazowego wskazania pliku .pt."
        ),
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 10))

    project_model_path = (
        None
        if self._is_free_mode_session_context()
        else self._get_campaign_project_plate_model_path()
    )
    current_plate_path = str(self.plate_custom_var.get() or "").strip()
    if not current_plate_path:
        try:
            runtime_plate_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
            runtime_plate_path = str(runtime_plate_meta.get("path") or "").strip()
            if runtime_plate_path and Path(runtime_plate_path).exists():
                current_plate_path = runtime_plate_path
                self.plate_custom_var.set(runtime_plate_path)
        except Exception:
            pass
    default_plate_mode = "custom"
    try:
        if project_model_path is not None and current_plate_path:
            if Path(current_plate_path).resolve() == project_model_path.resolve():
                default_plate_mode = "project"
    except Exception:
        pass
    if project_model_path is not None and not current_plate_path:
        default_plate_mode = "project"
        current_plate_path = str(project_model_path)

    plate_mode_var = tk.StringVar(
        value=("project" if (project_model_path is not None and default_plate_mode == "project") else "custom")
    )
    modal_plate_path_var = tk.StringVar(value=current_plate_path)
    modal_plate_adopt_var = tk.BooleanVar(value=False)
    default_picker_scope = "free" if self._is_free_mode_session_context() else "project"
    modal_plate_picker_scope_var = tk.StringVar(value=default_picker_scope)
    plate_emergency_visible_var = tk.BooleanVar(value=False)
    modal_vehicle_picker_scope_var = tk.StringVar(value=default_picker_scope)
    modal_conf_var = tk.DoubleVar(value=float(self.conf_var.get() or 0.25))
    modal_use_vehicle_var = tk.BooleanVar(value=bool(self._get_auto_vehicle_choice() == "use"))
    modal_vehicle_model_var = tk.StringVar(value=str(self.vehicle_model_var.get() or "").strip())
    modal_vehicle_custom_var = tk.StringVar(value=str(self.vehicle_custom_var.get() or "").strip())

    if not str(modal_vehicle_model_var.get() or "").strip():
        try:
            available_vehicle_keys = sorted(list(AVAILABLE_DETECT_MODELS.keys()))
            if available_vehicle_keys:
                modal_vehicle_model_var.set("yolo11s" if "yolo11s" in available_vehicle_keys else available_vehicle_keys[0])
        except Exception:
            pass

    def _model_name(path_value) -> str:
        try:
            path_text = str(path_value or "").strip()
            return Path(path_text).name if path_text else ""
        except Exception:
            return str(path_value or "").strip()

    def _same_model_path(left, right) -> bool:
        try:
            if left is None or right is None:
                return False
            return Path(left).resolve() == Path(right).resolve()
        except Exception:
            return False

    def _active_scope_model_path() -> Path | None:
        mode = str(plate_mode_var.get() or "").strip().lower()
        if mode == "project":
            return project_model_path if project_model_path is not None and project_model_path.exists() else None
        path_text = str(modal_plate_path_var.get() or "").strip()
        if not path_text:
            return None
        try:
            candidate = Path(path_text)
        except Exception:
            return None
        return candidate if candidate.exists() else None

    def _show_scope_models_details() -> None:
        project_path = project_model_path if project_model_path is not None and project_model_path.exists() else None
        active_path = _active_scope_model_path()
        if project_path is None and active_path is None:
            messagebox.showwarning(
                "Brak modeli",
                "Nie ma modelu, którego parametry można wyświetlić.",
                parent=dialog,
            )
            return

        details = tk.Toplevel(dialog)
        try:
            details.withdraw()
        except Exception:
            pass
        title = "Szczegóły modeli autoanotacji"
        styler = getattr(self.app, "style_dialog_window", None)
        if callable(styler):
            try:
                styler(details, title=title, geometry="760x560", parent=dialog)
            except Exception:
                try:
                    details.title(title)
                    details.resizable(False, False)
                except Exception:
                    pass
        else:
            try:
                details.title(title)
                details.resizable(False, False)
            except Exception:
                pass
        try:
            details.transient(dialog)
            details.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
        except Exception:
            pass

        surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
        if callable(surface_builder):
            try:
                surface = surface_builder(details, tone="info")
            except Exception:
                surface = tk.Frame(details, bg=panel_bg, bd=0, highlightthickness=0)
                surface.pack(fill=tk.BOTH, expand=True)
        else:
            surface = tk.Frame(details, bg=panel_bg, bd=0, highlightthickness=0)
            surface.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(surface, bg=panel_bg, bd=0, highlightthickness=0)
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        tk.Label(
            content,
            text=title,
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            content,
            text=(
                "Tu porównujesz trwały model projektu z modelem, który zostanie użyty "
                "w bieżącej autoanotacji Z2."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=690,
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

        rendered_model_detail_paths: set[str] = set()

        def _model_detail_key(model_path: Path | None) -> str:
            if model_path is None:
                return ""
            try:
                return str(Path(model_path).resolve()).casefold()
            except Exception:
                return str(model_path).strip().casefold()

        def _add_details_block(label: str, model_path: Path | None, context_text: str) -> None:
            model_key = _model_detail_key(model_path)
            if model_key and model_key in rendered_model_detail_paths:
                return
            if model_key:
                rendered_model_detail_paths.add(model_key)
            block = tk.Frame(
                content,
                bg=field_bg,
                bd=0,
                highlightthickness=1,
                highlightbackground=border,
                highlightcolor=border,
                padx=10,
                pady=8,
            )
            block.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
            tk.Label(
                block,
                text=label,
                bg=field_bg,
                fg=fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                justify=tk.LEFT,
            ).pack(anchor=tk.W, fill=tk.X)
            tk.Label(
                block,
                text=context_text,
                bg=field_bg,
                fg=muted,
                font=("Segoe UI", 8),
                anchor="w",
                justify=tk.LEFT,
                wraplength=670,
            ).pack(anchor=tk.W, fill=tk.X, pady=(3, 8))
            if model_path is None:
                tk.Label(
                    block,
                    text="Brak modelu w tym miejscu.",
                    bg=field_bg,
                    fg=palette.get("warning", "#f39c12"),
                    font=("Segoe UI", 8, "bold"),
                    anchor="w",
                    justify=tk.LEFT,
                ).pack(anchor=tk.W, fill=tk.X)
                return
            try:
                rows, tone = self._build_auto_annotation_model_quality_rows(model_path)
            except Exception:
                rows, tone = [], "warning"
            self._render_auto_annotation_model_quality_table(
                block,
                rows,
                tone,
                bg=panel_bg,
                fg=fg,
                muted=muted,
                border=border,
                success=success,
                warning=palette.get("warning", "#f39c12"),
                wraplength=560,
                label_width=17,
                model_path=model_path,
            )

        _add_details_block(
            "Model projektu (MT)",
            project_path,
            "Trwały zasób kampanii. Jeśli jest aktywny, Z2 może użyć go bez ponownego wskazywania pliku.",
        )
        _add_details_block(
            "Aktywny model autoanotacji",
            active_path,
            "Model używany przez bieżące uruchomienie autoanotacji. Może być modelem projektu albo modelem jednorazowym.",
        )

        buttons = tk.Frame(content, bg=panel_bg, bd=0, highlightthickness=0)
        buttons.pack(fill=tk.X, pady=(2, 0), side=tk.BOTTOM)
        ttk.Button(buttons, text="Zamknij", command=details.destroy).pack(side=tk.RIGHT)

        try:
            self._fit_borderless_dialog(details, parent=dialog, min_width=760, min_height=520)
            details.update_idletasks()
            details.deiconify()
            details.lift()
            details.focus_force()
        except Exception:
            pass
        try:
            details.grab_set()
        except Exception:
            pass
        details.bind("<Escape>", lambda _e: details.destroy())
        try:
            details.wait_window()
        finally:
            try:
                if details.winfo_exists():
                    details.grab_release()
            except Exception:
                pass

    model_status_card = tk.Frame(
        settings_card,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=10,
        pady=8,
    )
    model_status_card.pack(fill=tk.X, pady=(0, 10))

    tk.Label(
        model_status_card,
        text="Status modeli autoanotacji",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    project_model_status_var = tk.StringVar(value="")
    active_model_status_var = tk.StringVar(value="")
    model_decision_status_var = tk.StringVar(value="")

    def _add_model_status_row(label_text: str, variable: tk.StringVar, *, tone: str = "muted") -> tk.Label:
        row = tk.Frame(model_status_card, bg=panel_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, pady=(5, 0))
        tk.Label(
            row,
            text=label_text,
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            width=28,
        ).pack(side=tk.LEFT)
        value_lbl = tk.Label(
            row,
            textvariable=variable,
            bg=panel_bg,
            fg=(success if tone == "success" else palette.get("warning", "#f39c12") if tone == "warning" else fg),
            font=("Segoe UI", 8, "bold" if tone in {"success", "warning"} else "normal"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=390,
        )
        value_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return value_lbl

    project_model_status_lbl = _add_model_status_row("Model projektu (MT)", project_model_status_var, tone="muted")
    active_model_status_lbl = _add_model_status_row("Aktywny model autoanotacji", active_model_status_var, tone="muted")
    model_decision_status_lbl = _add_model_status_row("Co zostanie użyte", model_decision_status_var, tone="muted")

    model_status_buttons = tk.Frame(model_status_card, bg=panel_bg, bd=0, highlightthickness=0)
    model_status_buttons.pack(fill=tk.X, pady=(8, 0))
    model_status_details_btn = ttk.Button(
        model_status_buttons,
        text="Szczegóły modeli",
        command=_show_scope_models_details,
    )
    model_status_details_btn.pack(side=tk.LEFT)

    plate_model_mode_host = tk.Frame(settings_card, bg=field_bg, bd=0, highlightthickness=0)
    plate_model_mode_host.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        plate_model_mode_host,
        text="Wybór aktywnego modelu tablic (YOLO Pose)",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    if project_model_path is not None:
        project_identity = self._get_model_identity_caption(project_model_path)
        project_label = f"Model projektu: {project_model_path.name}"
        if project_identity:
            project_label += f" | {project_identity}"
        project_plate_toggle = build_modal_toggle(
            plate_model_mode_host,
            kind="radio",
            variable=plate_mode_var,
            value="project",
            text=project_label,
            bg=field_bg,
            fg_color=fg,
            font=("Segoe UI", 9),
            wraplength=540,
        )
        project_plate_toggle["row"].pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    plate_emergency_header = tk.Frame(plate_model_mode_host, bg=field_bg, bd=0, highlightthickness=0)
    plate_emergency_header.pack(fill=tk.X, pady=(8, 0))
    plate_emergency_text = tk.Frame(plate_emergency_header, bg=field_bg, bd=0, highlightthickness=0)
    plate_emergency_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Label(
        plate_emergency_text,
        text="Sekcja awaryjna modelu tablic",
        bg=field_bg,
        fg=warning,
        font=("Segoe UI", 8, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    tk.Label(
        plate_emergency_text,
        text=(
            "Rozwiń tylko wtedy, gdy chcesz wskazać model jednorazowy .pt "
            "albo świadomie podmienić MT projektu."
        ),
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
        bd=0,
        highlightthickness=0,
    ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

    def toggle_plate_emergency_options() -> None:
        plate_emergency_visible_var.set(not bool(plate_emergency_visible_var.get()))
        try:
            refresh_scope_choices()
        except Exception:
            pass

    plate_emergency_btn = ttk.Button(
        plate_emergency_header,
        text="Pokaż opcje awaryjne",
        command=toggle_plate_emergency_options,
    )
    plate_emergency_btn.pack(side=tk.RIGHT, padx=(10, 0))

    custom_plate_toggle = build_modal_toggle(
        plate_model_mode_host,
        kind="radio",
        variable=plate_mode_var,
        value="custom",
        text="Wskaż model jednorazowy dla bieżącej autoanotacji",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=540,
    )

    plate_model_hint_lbl = tk.Label(
        plate_model_mode_host,
        text="",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
        bd=0,
        highlightthickness=0,
    )
    plate_model_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

    plate_model_quality_table = tk.Frame(
        plate_model_mode_host,
        bg=field_bg,
        bd=0,
        highlightthickness=0,
    )
    plate_model_quality_table.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    def render_plate_model_quality_table(rows: list[tuple[str, str, str]], tone: str) -> None:
        for child in list(plate_model_quality_table.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        summary = "Szczegółowe parametry modelu są dostępne na żądanie."
        for label, value, _row_tone in list(rows or []):
            if str(label or "").strip() in {"Ocena", "mAP50-95", "mAP50-90", "Status"} and str(value or "").strip():
                summary = f"{label}: {value}"
                break
        tk.Label(
            plate_model_quality_table,
            text=summary,
            bg=field_bg,
            fg=(success if str(tone or "").strip().lower() == "success" else palette.get("warning", "#f39c12")),
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        ).pack(anchor=tk.W, fill=tk.X)
        _bind_scope_modal_scroll_children(plate_model_quality_table)

    def refresh_model_status_card() -> None:
        active_path = _active_scope_model_path()
        if self._is_free_mode_session_context():
            project_text = "nie dotyczy trybu swobodnego"
            project_tone = "muted"
        elif project_model_path is not None and project_model_path.exists():
            project_identity = self._get_model_identity_caption(project_model_path)
            project_text = f"jest: {project_model_path.name}"
            if project_identity:
                project_text += f" | {project_identity}"
            project_tone = "success"
        else:
            project_text = "brak modelu projektowego MT"
            project_tone = "warning"

        if active_path is not None:
            active_identity = self._get_model_identity_caption(active_path)
            active_text = f"{active_path.name}"
            if active_identity:
                active_text += f" | {active_identity}"
            if _same_model_path(active_path, project_model_path):
                decision_text = "Z2 użyje modelu projektu jako aktywnego modelu autoanotacji."
                active_tone = "success"
            else:
                decision_text = "Z2 użyje modelu wskazanego tylko dla bieżącej autoanotacji, chyba że zapiszesz go także w projekcie."
                active_tone = "warning"
        else:
            active_text = "brak - wybierz model, aby uruchomić autoanotację"
            decision_text = "Autoanotacja nie wystartuje, dopóki nie będzie aktywnego modelu tablic."
            active_tone = "warning"

        project_model_status_var.set(project_text)
        active_model_status_var.set(active_text)
        model_decision_status_var.set(decision_text)
        project_color = success if project_tone == "success" else palette.get("warning", "#f39c12") if project_tone == "warning" else muted
        active_color = success if active_tone == "success" else palette.get("warning", "#f39c12")
        try:
            project_model_status_lbl.configure(fg=project_color)
            active_model_status_lbl.configure(fg=active_color)
            model_decision_status_lbl.configure(fg=(success if active_tone == "success" else muted))
            model_status_details_btn.configure(
                state=(
                    tk.NORMAL
                    if (
                        (project_model_path is not None and project_model_path.exists())
                        or active_path is not None
                    )
                    else tk.DISABLED
                )
            )
        except Exception:
            pass

    plate_path_row = tk.Frame(plate_model_mode_host, bg=field_bg, bd=0, highlightthickness=0)
    plate_path_entry = ttk.Entry(plate_path_row, textvariable=modal_plate_path_var)
    plate_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def prompt_model_picker_scope_for_file_dialog(model_label: str) -> str | None:
        if self._is_free_mode_session_context():
            return "free"
        choice: dict[str, str | None] = {"value": None}
        picker_dialog = tk.Toplevel(dialog)
        picker_dialog.title("Katalog startowy modelu")
        picker_dialog.transient(dialog)
        try:
            picker_dialog.grab_set()
        except Exception:
            pass
        picker_dialog.configure(bg=panel_bg)
        picker_dialog.withdraw()

        content = tk.Frame(picker_dialog, bg=panel_bg, padx=18, pady=16)
        content.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            content,
            text=f"Od którego katalogu zacząć wybór {model_label} .pt?",
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            content,
            text=(
                "To tylko skrót nawigacyjny dla okna wyboru modelu. "
                "Aktywny model zmieni się dopiero po wskazaniu konkretnego pliku."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=440,
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

        buttons = tk.Frame(content, bg=panel_bg, bd=0, highlightthickness=0)
        buttons.pack(fill=tk.X)

        def finish(value: str | None) -> None:
            choice["value"] = value
            try:
                picker_dialog.destroy()
            except Exception:
                pass

        ttk.Button(
            buttons,
            text="Modele projektu",
            command=lambda: finish("project"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            buttons,
            text="Główny katalog modeli",
            command=lambda: finish("free"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            buttons,
            text="Anuluj",
            command=lambda: finish(None),
        ).pack(side=tk.RIGHT)

        try:
            self._fit_borderless_dialog(picker_dialog, parent=dialog, min_width=520, min_height=190)
            picker_dialog.update_idletasks()
            picker_dialog.deiconify()
            picker_dialog.lift()
        except Exception:
            try:
                picker_dialog.update_idletasks()
                picker_dialog.deiconify()
                picker_dialog.lift()
            except Exception:
                pass
        picker_dialog.bind("<Escape>", lambda _event: finish(None))
        picker_dialog.wait_window()
        return choice.get("value")

    def choose_plate_model_for_modal():
        preferred = None
        try:
            preferred = Path(str(modal_plate_path_var.get() or "").strip()) if str(modal_plate_path_var.get() or "").strip() else project_model_path
        except Exception:
            preferred = project_model_path
        picker_scope = prompt_model_picker_scope_for_file_dialog("modelu tablic")
        if not picker_scope:
            return
        modal_plate_picker_scope_var.set(str(picker_scope))
        initial_dir = self._get_campaign_plate_model_picker_initial_dir(
            preferred_path=preferred,
            picker_scope=str(picker_scope or default_picker_scope),
        )
        chosen = filedialog.askopenfilename(
            initialdir=str(initial_dir.absolute()),
            filetypes=[("YOLO Model", "*.pt")],
            title="Wskaż model tablic dla autoanotacji Z2",
        )
        if chosen:
            modal_plate_path_var.set(str(chosen))
            plate_mode_var.set("custom")
            refresh_scope_choices()

    ttk.Button(
        plate_path_row,
        text="Wybierz",
        command=choose_plate_model_for_modal,
    ).pack(side=tk.RIGHT, padx=(5, 0))

    plate_adopt_check = build_modal_toggle(
        plate_model_mode_host,
        kind="check",
        variable=modal_plate_adopt_var,
        text="Zmień model projektu (MT) na wskazany plik",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=540,
    )
    plate_adopt_hint_lbl = tk.Label(
        plate_model_mode_host,
        text=(
            "Zaznacz tylko wtedy, gdy wskazany model ma zastąpić główne źródło MT projektu. "
            "To sensowne np. przy imporcie sprawdzonego modelu z głównego katalogu modeli albo z innej maszyny."
        ),
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
        bd=0,
        highlightthickness=0,
    )

    params_section = build_section_card(
        body_content,
        "2",
        "Parametry detekcji",
        "Ustaw próg pewności oraz opcjonalne wsparcie modelem pojazdów.",
    )

    conf_card = tk.Frame(params_section, bg=field_bg, bd=0, highlightthickness=0)
    conf_card.pack(fill=tk.X, pady=(0, 10))
    tk.Label(
        conf_card,
        text="Próg pewności detekcji tablic",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    conf_row = tk.Frame(conf_card, bg=field_bg, bd=0, highlightthickness=0)
    conf_row.pack(fill=tk.X, pady=(4, 0))
    ttk.Scale(
        conf_row,
        from_=0.1,
        to=0.9,
        variable=modal_conf_var,
        orient=tk.HORIZONTAL,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    conf_value_lbl = tk.Label(
        conf_row,
        text="0.25",
        width=4,
        anchor="e",
        justify=tk.RIGHT,
        bg=field_bg,
        fg=fg,
        bd=0,
        highlightthickness=0,
    )
    conf_value_lbl.pack(side=tk.RIGHT, padx=(8, 0))

    vehicle_card = tk.Frame(
        params_section,
        bg=field_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=12,
    )
    vehicle_card.pack(fill=tk.X, pady=(0, 0))
    tk.Label(
        vehicle_card,
        text="Opcjonalna asysta modelem pojazdów",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    vehicle_check = build_modal_toggle(
        vehicle_card,
        kind="check",
        variable=modal_use_vehicle_var,
        text="Użyj wykrywania pojazdów jako wsparcia dla tablic w bieżącej autoanotacji",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=540,
    )
    vehicle_check["row"].pack(anchor=tk.W, fill=tk.X)
    tk.Label(
        vehicle_card,
        text="Model pojazdów służy tylko do zawężenia szukania tablic do obszaru pojazdu. Boxy pojazdów nie są eksportowane do finalnego YOLO.",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
        bd=0,
        highlightthickness=0,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

    vehicle_model_host = tk.Frame(vehicle_card, bg=field_bg, bd=0, highlightthickness=0)
    tk.Label(
        vehicle_model_host,
        text="Model pojazdów do wsparcia tablic (YOLO Box)",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

    try:
        vehicle_values = list(getattr(self, "vehicle_combo", None)["values"])
    except Exception:
        try:
            vehicle_values = sorted(list(AVAILABLE_DETECT_MODELS.keys())) + ["Custom"]
        except Exception:
            vehicle_values = ["Custom"]
    if vehicle_values and str(modal_vehicle_model_var.get() or "").strip() not in vehicle_values:
        modal_vehicle_model_var.set(str(vehicle_values[0]))

    vehicle_combo = ttk.Combobox(
        vehicle_model_host,
        textvariable=modal_vehicle_model_var,
        values=vehicle_values,
        state="readonly",
    )
    vehicle_combo.pack(fill=tk.X, pady=(4, 0))

    vehicle_custom_row = tk.Frame(vehicle_model_host, bg=field_bg, bd=0, highlightthickness=0)
    vehicle_custom_entry = ttk.Entry(vehicle_custom_row, textvariable=modal_vehicle_custom_var)
    vehicle_custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def choose_vehicle_model_for_modal():
        try:
            preferred = Path(str(modal_vehicle_custom_var.get() or "").strip()) if str(modal_vehicle_custom_var.get() or "").strip() else None
        except Exception:
            preferred = None
        picker_scope = prompt_model_picker_scope_for_file_dialog("modelu pojazdów")
        if not picker_scope:
            return
        modal_vehicle_picker_scope_var.set(str(picker_scope))
        initial_dir = self._get_auto_model_picker_initial_dir(
            "vehicle",
            picker_scope=str(picker_scope or default_picker_scope),
            preferred_path=preferred,
        )
        chosen = filedialog.askopenfilename(
            initialdir=str(Path(initial_dir).absolute()),
            filetypes=[("YOLO Model", "*.pt")],
            title="Wskaż model pojazdów do wsparcia autoanotacji Z2",
        )
        if chosen:
            modal_vehicle_custom_var.set(str(chosen))
            modal_vehicle_model_var.set("Custom")
            refresh_scope_choices()

    ttk.Button(
        vehicle_custom_row,
        text="Wybierz",
        command=choose_vehicle_model_for_modal,
    ).pack(side=tk.RIGHT, padx=(5, 0))

    scope_section = build_section_card(
        body_content,
        "3",
        "Zakres pracy",
        "Wybierz obrazy do przetworzenia i ochronę istniejącej pracy.",
    )

    options_title_lbl = tk.Label(
        scope_section,
        text="Wariant zakresu",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    options_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

    tk.Label(
        scope_section,
        text=(
            "Dopiero po wyborze modelu tablic zdecyduj, na jakim zakresie obrazów uruchomić autoanotację."
        ),
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 10))

    protection_notice_var = tk.StringVar(value="")
    protection_notice_lbl = tk.Label(
        scope_section,
        textvariable=protection_notice_var,
        bg=field_bg,
        fg=palette.get("warning", "#f39c12"),
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
        bd=0,
        highlightthickness=0,
    )

    protection_card = tk.Frame(
        scope_section,
        bg=field_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=10,
    )
    protection_card.pack(fill=tk.X, pady=(0, 10))
    protection_toggle = build_modal_toggle(
        protection_card,
        kind="check",
        variable=protect_existing_var,
        text="Chroń obrazy [OK] oraz ręcznie anotowane/poprawione",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9, "bold"),
        wraplength=520,
    )
    protection_toggle["row"].pack(anchor=tk.W, fill=tk.X)
    protection_desc_var = tk.StringVar(
        value=(
            "Włączona ochrona pomija gotowe lub ręcznie przepracowane obrazy, nawet jeśli są zaznaczone. "
            "Wyłącz ją tylko wtedy, gdy świadomie chcesz pozwolić autoanotacji nadpisać istniejącą pracę."
        )
    )
    tk.Label(
        protection_card,
        textvariable=protection_desc_var,
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=500,
        padx=26,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    options_host = tk.Frame(scope_section, bg=field_bg, bd=0, highlightthickness=0)
    options_host.pack(fill=tk.BOTH, expand=True)

    all_card, all_desc_var = build_option_card(
        options_host,
        value="all",
        title="Cały zestaw zdjęć",
        description=(
            "Uruchom autoanotację na całym zakresie wejścia: "
            f"{_format_scope_count(len(all_paths), _count_scope_existing_plates(all_paths))}."
        ),
    )
    bucket_card, bucket_desc_var = build_option_card(
        parent=options_host,
        value="buckets",
        title="Wybrane grupy z listy",
        description=(
            "Zaznacz grupy M / A / --. Liczby przy checkboxach oznaczają zdjęcia; "
            f"obecnie w grupach zapisano tablice: M {_count_scope_existing_plates(manual_bucket_paths)}, "
            f"A {_count_scope_existing_plates(auto_bucket_paths)}, -- {_count_scope_existing_plates(problem_bucket_paths)}."
        ),
    )
    bucket_checks = tk.Frame(bucket_card, bg=field_bg, bd=0, highlightthickness=0)
    bucket_checks.pack(anchor=tk.W, fill=tk.X, padx=(24, 0), pady=(8, 0))

    bucket_manual_toggle = build_modal_toggle(
        bucket_checks,
        kind="check",
        variable=include_manual_var,
        text=f"M ({len(manual_bucket_paths)})",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=120,
        enabled_getter=lambda: str(selected_mode_var.get() or "").strip().lower() == "buckets",
    )
    bucket_manual_toggle["row"].pack(side=tk.LEFT, padx=(0, 10))
    bucket_auto_toggle = build_modal_toggle(
        bucket_checks,
        kind="check",
        variable=include_auto_var,
        text=f"A ({len(auto_bucket_paths)})",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=120,
        enabled_getter=lambda: str(selected_mode_var.get() or "").strip().lower() == "buckets",
    )
    bucket_auto_toggle["row"].pack(side=tk.LEFT, padx=(0, 10))
    bucket_problem_toggle = build_modal_toggle(
        bucket_checks,
        kind="check",
        variable=include_problem_var,
        text=f"-- ({len(problem_bucket_paths)})",
        bg=field_bg,
        fg_color=fg,
        font=("Segoe UI", 9),
        wraplength=120,
        enabled_getter=lambda: str(selected_mode_var.get() or "").strip().lower() == "buckets",
    )
    bucket_problem_toggle["row"].pack(side=tk.LEFT, padx=(0, 10))

    _selected_card, selected_desc_var = build_option_card(
        options_host,
        value="selected",
        title="Ręczne zaznaczanie na liście",
        description=(
            "Licznik poniżej pokazuje faktycznie zaznaczone wiersze listy i liczbę obrazów, które trafią do autoanotacji. Możesz zmienić zaznaczenie bez zamykania tego okna. Zdjęcia ręcznie anotowane i ze statusem [OK] zostaną pominięte mimo zaznaczenia - to mechanizm ochronny przed nadpisaniem gotowej pracy."
            if selected_row_count > 0
            else "To okno możesz zostawić otwarte. Zaznacz obrazy na liście po lewej już teraz, a przycisk „Uruchom autoanotację” odblokuje się automatycznie. Zdjęcia ręcznie anotowane i ze statusem [OK] zostaną pominięte mimo zaznaczenia - to mechanizm ochronny."
        ),
    )
    selected_rows_count_var = tk.StringVar(value=str(selected_row_count))
    selected_process_count_var = tk.StringVar(value=str(selected_count))
    selected_plate_count_var = tk.StringVar(
        value=str(_count_scope_existing_plates(list(scope_info.get("selected_paths") or [])))
    )
    selected_counter_row = tk.Frame(_selected_card, bg=field_bg, bd=0, highlightthickness=0)
    tk.Label(
        selected_counter_row,
        text="Zaznaczono: ",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side=tk.LEFT)
    selected_rows_count_lbl = tk.Label(
        selected_counter_row,
        textvariable=selected_rows_count_var,
        bg=field_bg,
        fg=(success if selected_row_count > 0 else muted),
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        width=4,
    )
    selected_rows_count_lbl.pack(side=tk.LEFT)
    tk.Label(
        selected_counter_row,
        text="pozycji   Do autoanotacji: ",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side=tk.LEFT)
    selected_process_count_lbl = tk.Label(
        selected_counter_row,
        textvariable=selected_process_count_var,
        bg=field_bg,
        fg=(success if selected_count > 0 else muted),
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        width=4,
    )
    selected_process_count_lbl.pack(side=tk.LEFT)
    tk.Label(
        selected_counter_row,
        text="zdjęć / ",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side=tk.LEFT)
    selected_plate_count_lbl = tk.Label(
        selected_counter_row,
        textvariable=selected_plate_count_var,
        bg=field_bg,
        fg=(success if int(selected_plate_count_var.get() or 0) > 0 else muted),
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        width=4,
    )
    selected_plate_count_lbl.pack(side=tk.LEFT)
    tk.Label(
        selected_counter_row,
        text="tablic",
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side=tk.LEFT)

    def refresh_selected_counter_visibility() -> None:
        selected_mode_active = str(selected_mode_var.get() or "").strip().lower() == "selected"
        self._set_widget_packed(
            selected_counter_row,
            bool(selected_mode_active),
            anchor=tk.W,
            fill=tk.X,
            padx=(26, 0),
            pady=(8, 0),
        )

    refresh_selected_counter_visibility()

    footer = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    footer.pack(fill=tk.X, pady=(4, 0))

    footer_progress_panel = tk.Frame(
        footer,
        bg=field_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=section_border,
        highlightcolor=section_border,
        padx=12,
        pady=8,
    )
    footer_progress_panel.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    tk.Label(
        footer_progress_panel,
        text="4. Start i postęp autoanotacji",
        bg=field_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    ttk.Progressbar(
        footer_progress_panel,
        variable=progress_var,
        maximum=100,
        mode="determinate",
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 5))
    tk.Label(
        footer_progress_panel,
        textvariable=progress_status_var,
        bg=field_bg,
        fg=success,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
    ).pack(anchor=tk.W, fill=tk.X)
    footer_progress_meta = tk.Frame(footer_progress_panel, bg=field_bg, bd=0, highlightthickness=0)
    footer_progress_meta.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
    tk.Label(
        footer_progress_meta,
        textvariable=progress_counts_var,
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
    ).pack(side=tk.LEFT)
    tk.Label(
        footer_progress_meta,
        textvariable=progress_file_var,
        bg=field_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

    buttons = tk.Frame(footer, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X)

    def _cancel_scope_modal_refresh_jobs() -> None:
        if getattr(self, "_plate_auto_scope_modal_refresh_callback", None) is refresh_scope_choices:
            self._plate_auto_scope_modal_refresh_callback = None
        if getattr(self, "_plate_auto_scope_modal_selection_refresh_callback", None) is refresh_selected_scope_counter:
            self._plate_auto_scope_modal_selection_refresh_callback = None
        try:
            pending_refresh_after_id = getattr(self, "_plate_auto_scope_modal_refresh_after_id", None)
            if pending_refresh_after_id:
                self.frame.after_cancel(pending_refresh_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_refresh_after_id = None
        try:
            pending_selection_after_id = getattr(self, "_plate_auto_scope_modal_selection_refresh_after_id", None)
            if pending_selection_after_id:
                self.frame.after_cancel(pending_selection_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_selection_refresh_after_id = None
        try:
            pending_watch_after_id = getattr(self, "_plate_auto_scope_modal_selection_watch_after_id", None)
            if pending_watch_after_id:
                dialog.after_cancel(pending_watch_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_selection_watch_after_id = None
        self._plate_auto_scope_modal_last_selection_signature = ()
        try:
            self._set_plate_auto_scope_selection_mode(False)
        except Exception:
            pass

    def _update_scope_progress_modal(
        *,
        pct: float | None = None,
        current: int | None = None,
        total: int | None = None,
        successful: int | None = None,
        filename: str = "",
        meta_text: str = "",
    ) -> None:
        def _clean_progress_status_text(raw_text: str) -> str:
            text = str(raw_text or "").strip()
            while "|" in text:
                prefix, rest = text.split("|", 1)
                marker = prefix.strip().lower()
                if marker.endswith("%") or "/" in marker or marker.startswith("ok"):
                    text = rest.strip()
                    continue
                break
            if "/" in text and "%" in text:
                return ""
            return text

        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        try:
            safe_pct = 0.0 if pct is None else max(0.0, min(100.0, float(pct)))
            progress_var.set(safe_pct)
        except Exception:
            pass
        try:
            safe_current = int(current or 0)
            safe_total = int(total or 0)
            safe_successful = int(successful or 0)
            if safe_total > 0:
                progress_counts_var.set(f"OK: {safe_successful} | {safe_current} / {safe_total}")
            else:
                progress_counts_var.set("przygotowanie")
        except Exception:
            pass
        try:
            clean_filename = str(filename or "").strip()
            progress_file_var.set(Path(clean_filename).name if clean_filename else "")
        except Exception:
            progress_file_var.set("")
        try:
            clean_meta_text = _clean_progress_status_text(str(meta_text or ""))
            if clean_meta_text:
                progress_status_var.set(clean_meta_text)
            elif total:
                progress_status_var.set("Przetwarzam obrazy...")
            else:
                progress_status_var.set("Przygotowuję autoanotację...")
        except Exception:
            pass

    def _close_scope_progress_modal(*_args, **_kwargs) -> None:
        try:
            self._plate_auto_scope_progress_modal_active = False
            self._plate_auto_scope_progress_dialog = None
            self._update_plate_auto_scope_progress_modal = None
            self._close_plate_auto_scope_progress_modal = None
        except Exception:
            pass
        try:
            if dialog.winfo_exists():
                dialog.destroy()
        except Exception:
            pass

    def _request_stop_from_progress() -> None:
        try:
            if bool(getattr(self, "is_processing", False)):
                self._stop_annotation()
                progress_status_var.set("Zatrzymuję autoanotację...")
                return
        except Exception:
            pass
        _close_scope_progress_modal()

    def _show_scope_progress_view() -> None:
        _cancel_scope_modal_refresh_jobs()
        for old_progress_widget in (
            getattr(self, "progress_info_row", None),
            getattr(self, "progress", None),
        ):
            try:
                old_progress_widget.pack_forget()
            except Exception:
                pass
        try:
            old_splash = getattr(self, "_campaign_step2_splash_overlay", None)
            if old_splash is not None:
                old_splash.place_forget()
            self._campaign_step2_splash_visible = False
        except Exception:
            pass
        try:
            dialog.protocol("WM_DELETE_WINDOW", _request_stop_from_progress)
        except Exception:
            pass
        try:
            next_btn.configure(text="Autoanotacja trwa...", state=tk.DISABLED)
        except Exception:
            pass
        try:
            cancel_btn.configure(text="Zatrzymaj autoanotację", command=_request_stop_from_progress)
        except Exception:
            pass
        try:
            self._plate_auto_scope_progress_modal_active = True
            self._plate_auto_scope_progress_dialog = dialog
            self._update_plate_auto_scope_progress_modal = _update_scope_progress_modal
            self._close_plate_auto_scope_progress_modal = _close_scope_progress_modal
        except Exception:
            pass
        _update_scope_progress_modal(pct=0.0, current=0, total=0, meta_text="Przygotowuję start autoanotacji...")
        try:
            dialog.update_idletasks()
            dialog.lift()
        except Exception:
            pass

    def choose_cancel() -> None:
        if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)) or bool(getattr(self, "is_processing", False)):
            _request_stop_from_progress()
            return
        result["choice"] = "cancel"
        try:
            dialog_done_var.set(True)
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    cancel_btn = ttk.Button(buttons, text="Anuluj", command=choose_cancel)
    cancel_btn.pack(side=tk.RIGHT)
    next_btn = ttk.Button(buttons, text="Uruchom autoanotację")
    next_btn.pack(side=tk.RIGHT, padx=(0, 8))

    def resolve_bucket_paths() -> list[Path]:
        live_bucket_paths = self._collect_plate_auto_scope_bucket_paths(
            candidate_image_paths=all_source_paths,
            protect_existing=bool(protect_existing_var.get()),
        )
        merged: list[Path] = []
        seen_names: set[str] = set()
        if bool(include_manual_var.get()):
            for path in list(live_bucket_paths.get("manual") or []):
                key = str(path.name or "").strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                    merged.append(path)
        if bool(include_auto_var.get()):
            for path in list(live_bucket_paths.get("auto") or []):
                key = str(path.name or "").strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                    merged.append(path)
        if bool(include_problem_var.get()):
            for path in list(live_bucket_paths.get("problem") or []):
                key = str(path.name or "").strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                merged.append(path)
        return merged

    def _scope_model_already_confirmed_for_start(
        model_path,
        *,
        adopt_to_project: bool = False,
    ) -> bool:
        if self._is_free_mode_session_context() or adopt_to_project:
            return False
        try:
            confirmed = str(getattr(self, "_campaign_plate_auto_model_confirmed_for_scope", "") or "").strip()
            if not confirmed:
                return False
            return Path(confirmed).resolve() == Path(model_path).resolve()
        except Exception:
            return False

    def _consume_scope_model_confirmation() -> None:
        try:
            self._campaign_plate_auto_model_confirmed_for_scope = ""
        except Exception:
            pass

    def choose_next() -> None:
        mode = str(selected_mode_var.get() or "").strip().lower()
        if mode == "selected":
            live_scope = self._collect_plate_auto_scope_candidates(
                candidate_image_paths=all_source_paths,
                protect_existing=bool(protect_existing_var.get()),
            )
            live_selected_paths = list(live_scope.get("selected_paths") or [])
            if not live_selected_paths:
                return
            result["choice"] = "selected"
            result["payload"] = {
                "mode": "selected",
                "label": "Zaznaczone obrazy",
                "image_paths": live_selected_paths,
                "protect_existing": bool(protect_existing_var.get()),
            }
        elif mode == "buckets":
            bucket_choice_paths = resolve_bucket_paths()
            if not bucket_choice_paths:
                return
            chosen_labels = []
            if bool(include_manual_var.get()):
                chosen_labels.append("M")
            if bool(include_auto_var.get()):
                chosen_labels.append("A")
            if bool(include_problem_var.get()):
                chosen_labels.append("--")
            result["choice"] = "buckets"
            result["payload"] = {
                "mode": "buckets",
                "label": f"Grupy: {', '.join(chosen_labels)}",
                "image_paths": bucket_choice_paths,
                "protect_existing": bool(protect_existing_var.get()),
            }
        else:
            live_scope = self._collect_plate_auto_scope_candidates(
                candidate_image_paths=all_source_paths,
                protect_existing=bool(protect_existing_var.get()),
            )
            live_all_paths = list(live_scope.get("all_paths") or [])
            if not live_all_paths:
                return
            result["choice"] = "all"
            result["payload"] = {
                "mode": "all",
                "label": "Cały zestaw zdjęć",
                "image_paths": live_all_paths,
                "protect_existing": bool(protect_existing_var.get()),
            }

        plate_mode = str(plate_mode_var.get() or "").strip().lower()
        use_project_model = bool(plate_mode == "project")
        use_vehicle = bool(modal_use_vehicle_var.get())

        try:
            if use_project_model:
                if project_model_path is None or not project_model_path.exists():
                    messagebox.showerror("Brak modelu", "Model projektu tablic nie jest dostępny.", parent=dialog)
                    return
                if not self._is_free_mode_session_context():
                    if _scope_model_already_confirmed_for_start(project_model_path, adopt_to_project=False):
                        apply_result = True
                    else:
                        apply_result = self._apply_campaign_plate_auto_model_choice(project_model_path, adopt_to_project=False)
                        if apply_result == "retry":
                            return
                        if not apply_result:
                            return
                else:
                    self.plate_custom_var.set(str(project_model_path))
                    self._remember_plate_model_runtime_meta(
                        model_path=project_model_path,
                        source="project",
                        scope="project",
                    )
            else:
                chosen_plate_path = str(modal_plate_path_var.get() or "").strip()
                if not chosen_plate_path or not Path(chosen_plate_path).exists():
                    messagebox.showerror(
                        "Brak modelu",
                        "W tym modalu wskaż model tablic (.pt) dla bieżącej autoanotacji Z2.",
                        parent=dialog,
                    )
                    return
                if not self._is_free_mode_session_context():
                    adopt_plate_model = bool(modal_plate_adopt_var.get())
                    if _scope_model_already_confirmed_for_start(
                        Path(chosen_plate_path),
                        adopt_to_project=adopt_plate_model,
                    ):
                        apply_result = True
                    else:
                        apply_result = self._apply_campaign_plate_auto_model_choice(
                            Path(chosen_plate_path),
                            adopt_to_project=adopt_plate_model,
                        )
                        if apply_result == "retry":
                            return
                        if not apply_result:
                            return
                else:
                    ok, message, info = validate_model_file(Path(chosen_plate_path))
                    if not ok:
                        messagebox.showerror("Nieprawidłowy model", f"Nie udało się użyć wybranego modelu:\n{message}", parent=dialog)
                        return
                    self.plate_custom_var.set(str(chosen_plate_path))
                    self._remember_plate_model_runtime_meta(
                        model_path=chosen_plate_path,
                        identity=format_yolo_model_identity(info),
                        source="external",
                        scope="run",
                    )

            self.conf_var.set(float(modal_conf_var.get() or 0.25))
            self._set_auto_vehicle_choice_state(
                "use" if use_vehicle else "skip",
                campaign_context=(not self._is_free_mode_session_context()),
            )
            self.mode_var.set("C: Pojazdy + tablice" if use_vehicle else "B: Tylko tablice")
            self.vehicle_model_var.set(str(modal_vehicle_model_var.get() or "").strip())
            self.vehicle_custom_var.set(str(modal_vehicle_custom_var.get() or "").strip())
            self._auto_route_settings_pending = False
            self._on_vehicle_model_change(refresh_workflow=False)
            self._refresh_detection_configuration_ui()
            self._refresh_plate_model_runtime_info_ui()
        except Exception as e:
            messagebox.showerror(
                "Błąd ustawień autoanotacji",
                f"Nie udało się zastosować ustawień bieżącej autoanotacji:\n{e}",
                parent=dialog,
            )
            return

        try:
            _show_scope_progress_view()
        except Exception:
            pass
        try:
            dialog_done_var.set(True)
        except Exception:
            pass

    next_btn.configure(command=choose_next)

    def _scope_model_inputs_ready(can_continue: bool) -> bool:
        if not can_continue:
            return False
        use_project_model = str(plate_mode_var.get() or "").strip().lower() == "project"
        if use_project_model:
            can_continue = bool(project_model_path is not None and project_model_path.exists())
        else:
            chosen_plate_path = str(modal_plate_path_var.get() or "").strip()
            can_continue = bool(chosen_plate_path and Path(chosen_plate_path).exists())

        use_vehicle = bool(modal_use_vehicle_var.get())
        if can_continue and use_vehicle and str(modal_vehicle_model_var.get() or "").strip() == "Custom":
            chosen_vehicle_path = str(modal_vehicle_custom_var.get() or "").strip()
            can_continue = bool(chosen_vehicle_path and Path(chosen_vehicle_path).exists())
        return bool(can_continue)

    def refresh_selected_scope_counter() -> None:
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
            try:
                next_btn.configure(text="Autoanotacja trwa...", state=tk.DISABLED)
                cancel_btn.configure(text="Zatrzymaj autoanotację", command=_request_stop_from_progress)
            except Exception:
                pass
            return

        mode = str(selected_mode_var.get() or "").strip().lower()
        if mode != "selected":
            try:
                refresh_selected_counter_visibility()
            except Exception:
                pass
            return

        annotations = list(getattr(self, "current_annotations", []) or [])
        try:
            selected_display_indices = list(self.preview_listbox.curselection() or ())
        except Exception:
            selected_display_indices = []
        display_map = getattr(self, "_preview_list_display_indices", []) or []
        selected_indices: list[int] = []
        seen_selected_indices: set[int] = set()
        for display_index in selected_display_indices:
            try:
                safe_display_index = int(display_index)
                actual_index = int(display_map[safe_display_index])
            except Exception:
                continue
            if actual_index in seen_selected_indices:
                continue
            seen_selected_indices.add(actual_index)
            selected_indices.append(actual_index)
        active_protected_filenames = (
            scope_protected_filenames
            if bool(protect_existing_var.get())
            else set()
        )
        selected_paths = self._resolve_preview_scope_paths(
            selected_indices,
            candidate_path_map=scope_candidate_path_map,
            protected_filenames=active_protected_filenames,
        )
        live_selected_row_count = len(selected_display_indices)
        live_selected_total_count = self._count_unique_preview_scope_indices(
            annotations,
            selected_indices,
        )
        live_selected_protected_skip_count = self._count_protected_preview_scope_indices(
            annotations,
            selected_indices,
            active_protected_filenames,
        )
        live_selected_count = len(selected_paths)
        live_selected_plate_count = _count_scope_existing_plates(selected_paths)
        live_selected_duplicate_skip_count = max(0, int(live_selected_row_count) - int(live_selected_total_count))
        live_selected_unresolved_skip_count = max(
            0,
            int(live_selected_total_count) - int(live_selected_protected_skip_count) - int(live_selected_count),
        )

        try:
            refresh_selected_counter_visibility()
            selected_desc_var.set(
                (
                    "Licznik poniżej rozdziela zaznaczone wiersze, zdjęcia kierowane do autoanotacji oraz liczbę tablic już zapisanych w tych zdjęciach. "
                    "Ochrona jest włączona, więc obrazy ręcznie anotowane/poprawione albo ze statusem [OK] zostaną pominięte mimo zaznaczenia."
                    if bool(protect_existing_var.get())
                    else "Licznik poniżej rozdziela zaznaczone wiersze, zdjęcia kierowane do autoanotacji oraz liczbę tablic już zapisanych w tych zdjęciach. Ochrona jest wyłączona, więc autoanotacja może nadpisać także obrazy [OK] oraz ręcznie anotowane/poprawione."
                )
            )
            selected_rows_count_var.set(str(live_selected_row_count))
            selected_process_count_var.set(str(live_selected_count))
            selected_plate_count_var.set(str(live_selected_plate_count))
            selected_rows_count_lbl.configure(fg=(success if live_selected_row_count > 0 else muted))
            selected_process_count_lbl.configure(
                fg=(
                    success
                    if live_selected_count > 0
                    else (palette.get("warning", "#f39c12") if live_selected_row_count > 0 else muted)
                )
            )
            selected_plate_count_lbl.configure(fg=(success if live_selected_plate_count > 0 else muted))
        except Exception:
            pass

        try:
            notice_parts = []
            if live_selected_protected_skip_count > 0:
                notice_parts.append(
                    f"W zaznaczeniu pominięte zostanie {live_selected_protected_skip_count} zdjęć ręcznie anotowanych/poprawionych albo ze statusem [OK]."
                )
            if live_selected_duplicate_skip_count > 0:
                notice_parts.append(
                    f"{live_selected_duplicate_skip_count} zaznaczonych pozycji ma powtórzoną nazwę pliku i zostanie potraktowanych jako ten sam obraz."
                )
            if live_selected_unresolved_skip_count > 0:
                notice_parts.append(
                    f"{live_selected_unresolved_skip_count} zaznaczonych pozycji nie ma dostępnego pliku obrazu w tym zakresie."
                )
            notice_text = (
                f"Do autoanotacji trafi {live_selected_count} zdjęć / {live_selected_plate_count} zapisanych tablic. " + " ".join(notice_parts)
                if notice_parts
                else ""
            )
            protection_notice_var.set(notice_text)
            self._set_widget_packed(
                protection_notice_lbl,
                bool(notice_text),
                anchor=tk.W,
                fill=tk.X,
                pady=(0, 10),
                before=options_host,
            )
        except Exception:
            protection_notice_var.set("")
            self._set_widget_packed(protection_notice_lbl, False)

        try:
            next_btn.configure(
                text="Uruchom autoanotację",
                state=(
                    tk.NORMAL
                    if _scope_model_inputs_ready(live_selected_count > 0)
                    else tk.DISABLED
                )
            )
        except Exception:
            pass

    def refresh_scope_choices() -> None:
        scroll_state = _capture_scope_modal_scroll_state()
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
            try:
                next_btn.configure(text="Autoanotacja trwa...", state=tk.DISABLED)
                cancel_btn.configure(text="Zatrzymaj autoanotację", command=_request_stop_from_progress)
            except Exception:
                pass
            return

        protection_enabled = bool(protect_existing_var.get())
        live_scope = self._collect_plate_auto_scope_candidates(
            candidate_image_paths=all_source_paths,
            protect_existing=protection_enabled,
        )
        live_selected_count = int(live_scope.get("selected_count", 0) or 0)
        live_selected_total_count = int(live_scope.get("selected_total_count", live_selected_count) or 0)
        live_selected_row_count = int(live_scope.get("selected_row_count", live_selected_total_count) or 0)
        live_selected_duplicate_skip_count = int(live_scope.get("selected_duplicate_skip_count", 0) or 0)
        live_selected_unresolved_skip_count = int(live_scope.get("selected_unresolved_skip_count", 0) or 0)
        live_selected_plate_count = _count_scope_existing_plates(list(live_scope.get("selected_paths") or []))
        live_all_paths = list(live_scope.get("all_paths") or [])
        live_all_plate_count = _count_scope_existing_plates(live_all_paths)
        live_bucket_paths = self._collect_plate_auto_scope_bucket_paths(
            candidate_image_paths=all_source_paths,
            protect_existing=False,
        )
        live_bucket_protected_counts = self._collect_plate_auto_scope_bucket_protected_counts(
            candidate_image_paths=all_source_paths,
            protect_existing=protection_enabled,
        )
        live_manual_count = len(list(live_bucket_paths.get("manual") or []))
        live_auto_count = len(list(live_bucket_paths.get("auto") or []))
        live_problem_count = len(list(live_bucket_paths.get("problem") or []))

        try:
            protection_desc_var.set(
                (
                    "Ochrona jest włączona: autoanotacja pominie obrazy [OK] oraz ręcznie anotowane/poprawione. "
                    "Wyłącz ją tylko wtedy, gdy świadomie chcesz pozwolić modelowi nadpisać istniejące ramki."
                    if protection_enabled
                    else "Ochrona jest wyłączona: do autoanotacji mogą trafić także obrazy [OK] oraz ręcznie anotowane/poprawione. To tryb świadomego nadpisywania."
                )
            )
            all_desc_var.set(
                "Uruchom autoanotację na całym zakresie wejścia: "
                f"{_format_scope_count(len(live_all_paths), live_all_plate_count)}."
            )
            refresh_selected_counter_visibility()
            selected_desc_var.set(
                (
                    "Licznik poniżej rozdziela zaznaczone wiersze, zdjęcia kierowane do autoanotacji oraz tablice już zapisane w tych zdjęciach. Ochrona pominie obrazy [OK] oraz ręcznie anotowane/poprawione."
                    if str(selected_mode_var.get() or "").strip().lower() == "selected" and protection_enabled
                    else "Licznik poniżej rozdziela zaznaczone wiersze, zdjęcia kierowane do autoanotacji oraz tablice już zapisane w tych zdjęciach. Ochrona jest wyłączona, więc istniejące ramki mogą zostać nadpisane."
                    if str(selected_mode_var.get() or "").strip().lower() == "selected"
                    else "To okno możesz zostawić otwarte. Zaznacz obrazy na liście po lewej, a przycisk „Uruchom autoanotację” odblokuje się automatycznie. Liczniki rozdzielają zdjęcia od istniejących tablic."
                )
            )
            selected_rows_count_var.set(str(live_selected_row_count))
            selected_process_count_var.set(str(live_selected_count))
            selected_plate_count_var.set(str(live_selected_plate_count))
            selected_rows_count_lbl.configure(fg=(success if live_selected_row_count > 0 else muted))
            selected_process_count_lbl.configure(
                fg=(
                    success
                    if live_selected_count > 0
                    else (palette.get("warning", "#f39c12") if live_selected_row_count > 0 else muted)
                )
            )
            selected_plate_count_lbl.configure(fg=(success if live_selected_plate_count > 0 else muted))
        except Exception:
            pass

        try:
            bucket_desc_var.set(
                "Zaznacz grupy M / A / --. Liczby przy checkboxach oznaczają zdjęcia; "
                f"obecnie w grupach zapisano tablice: M {_count_scope_existing_plates(live_bucket_paths.get('manual') or [])}, "
                f"A {_count_scope_existing_plates(live_bucket_paths.get('auto') or [])}, "
                f"-- {_count_scope_existing_plates(live_bucket_paths.get('problem') or [])}."
            )
        except Exception:
            pass

        try:
            mode_for_notice = str(selected_mode_var.get() or "").strip().lower()
            if mode_for_notice == "selected":
                protected_for_notice = int(live_scope.get("selected_protected_skip_count", 0) or 0)
                process_for_notice = int(live_selected_count or 0)
                plates_for_notice = int(live_selected_plate_count or 0)
                scope_word = "zaznaczeniu"
            elif mode_for_notice == "buckets":
                protected_for_notice = 0
                if protection_enabled:
                    if bool(include_manual_var.get()):
                        protected_for_notice += int(live_bucket_protected_counts.get("manual", 0) or 0)
                    if bool(include_auto_var.get()):
                        protected_for_notice += int(live_bucket_protected_counts.get("auto", 0) or 0)
                    if bool(include_problem_var.get()):
                        protected_for_notice += int(live_bucket_protected_counts.get("problem", 0) or 0)
                bucket_notice_paths = resolve_bucket_paths()
                process_for_notice = int(len(bucket_notice_paths) or 0)
                plates_for_notice = int(_count_scope_existing_plates(bucket_notice_paths))
                scope_word = "wybranych grupach"
            else:
                protected_for_notice = int(live_scope.get("protected_skip_count", 0) or 0)
                process_for_notice = int(len(live_all_paths) or 0)
                plates_for_notice = int(live_all_plate_count or 0)
                scope_word = "całym zestawie"
            if (
                protected_for_notice > 0
                or (
                    mode_for_notice == "selected"
                    and (live_selected_unresolved_skip_count > 0 or live_selected_duplicate_skip_count > 0)
                )
            ):
                notice_parts = []
                if protected_for_notice > 0:
                    notice_parts.append(
                        f"W {scope_word} pominięte zostanie {protected_for_notice} zdjęć ręcznie anotowanych/poprawionych albo ze statusem [OK]."
                    )
                if mode_for_notice == "selected" and live_selected_duplicate_skip_count > 0:
                    notice_parts.append(
                        f"{live_selected_duplicate_skip_count} zaznaczonych pozycji ma powtórzoną nazwę pliku i zostanie potraktowanych jako ten sam obraz."
                    )
                if mode_for_notice == "selected" and live_selected_unresolved_skip_count > 0:
                    notice_parts.append(
                        f"{live_selected_unresolved_skip_count} zaznaczonych pozycji nie ma dostępnego pliku obrazu w tym zakresie."
                    )
                notice_text = (
                    f"Do autoanotacji trafi {process_for_notice} zdjęć / {plates_for_notice} zapisanych tablic. "
                    + " ".join(notice_parts)
                )
                protection_notice_var.set(notice_text)
            else:
                notice_text = ""
                protection_notice_var.set("")
            self._set_widget_packed(
                protection_notice_lbl,
                bool(notice_text),
                anchor=tk.W,
                fill=tk.X,
                pady=(0, 10),
                before=options_host,
            )
        except Exception:
            protection_notice_var.set("")
            self._set_widget_packed(protection_notice_lbl, False)

        try:
            conf_value_lbl.configure(text=f"{float(modal_conf_var.get() or 0.0):.2f}")
        except Exception:
            pass

        use_project_model = str(plate_mode_var.get() or "").strip().lower() == "project"
        plate_custom_active = not use_project_model
        emergency_visible = bool(plate_emergency_visible_var.get())
        try:
            plate_emergency_btn.configure(
                text=("Ukryj opcje awaryjne" if emergency_visible else "Pokaż opcje awaryjne")
            )
        except Exception:
            pass
        try:
            refresh_model_status_card()
        except Exception:
            pass
        try:
            plate_path_entry.configure(state=("disabled" if not plate_custom_active else "normal"))
        except Exception:
            pass
        self._set_widget_packed(
            custom_plate_toggle["row"],
            emergency_visible,
            anchor=tk.W,
            fill=tk.X,
            pady=(4, 0),
        )
        self._set_widget_packed(
            plate_path_row,
            bool(emergency_visible and plate_custom_active),
            fill=tk.X,
            pady=(4, 0),
        )
        self._set_widget_packed(
            plate_adopt_check["row"],
            bool((not self._is_free_mode_session_context()) and emergency_visible and plate_custom_active),
            anchor=tk.W,
            pady=(6, 0),
        )
        self._set_widget_packed(
            plate_adopt_hint_lbl,
            bool((not self._is_free_mode_session_context()) and emergency_visible and plate_custom_active),
            anchor=tk.W,
            fill=tk.X,
            pady=(3, 0),
        )
        try:
            if use_project_model:
                if project_model_path is not None and project_model_path.exists():
                    quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(project_model_path)
                    plate_model_hint_lbl.configure(
                        text="To wymagany model dla bieżącej autoanotacji. Użyjesz aktywnego modelu projektu tablic.",
                        fg=muted,
                    )
                    render_plate_model_quality_table(quality_rows, quality_tone)
                else:
                    quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(None)
                    plate_model_hint_lbl.configure(
                        text="Model projektu tablic nie jest dostępny. Aby uruchomić autoanotację, przełącz na model wskazany tylko dla bieżącej autoanotacji.",
                        fg=palette.get("warning", "#f39c12"),
                    )
                    render_plate_model_quality_table(quality_rows, quality_tone)
            elif str(modal_plate_path_var.get() or "").strip():
                quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(
                    str(modal_plate_path_var.get() or "").strip()
                )
                if (not self._is_free_mode_session_context()) and bool(modal_plate_adopt_var.get()):
                    hint_text = (
                        "Po zatwierdzeniu ten plik stanie się modelem projektu (MT) i będzie głównym "
                        "modelem tablic dla kolejnych uruchomień Z2 w tej kampanii."
                    )
                else:
                    hint_text = (
                        "Ten plik zostanie użyty tylko w bieżącej autoanotacji Z2. "
                        "Model projektu (MT) pozostanie bez zmian."
                    )
                plate_model_hint_lbl.configure(
                    text=hint_text,
                    fg=muted,
                )
                render_plate_model_quality_table(quality_rows, quality_tone)
            else:
                quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(None)
                plate_model_hint_lbl.configure(
                    text="Najpierw wskaż wymagany model tablic (.pt). Bez tego autoanotacja nie wystartuje i przycisk „Uruchom autoanotację” pozostanie zablokowany.",
                    fg=palette.get("warning", "#f39c12"),
                )
                render_plate_model_quality_table(quality_rows, quality_tone)
        except Exception:
            pass

        use_vehicle = bool(modal_use_vehicle_var.get())
        self._set_widget_packed(
            vehicle_model_host,
            use_vehicle,
            fill=tk.X,
            pady=(4, 0),
        )
        vehicle_custom_active = use_vehicle and str(modal_vehicle_model_var.get() or "").strip() == "Custom"
        self._set_widget_packed(
            vehicle_custom_row,
            vehicle_custom_active,
            fill=tk.X,
            pady=(6, 0),
        )

        try:
            for widget, text in (
                (bucket_manual_toggle["label"], f"M ({live_manual_count})"),
                (bucket_auto_toggle["label"], f"A ({live_auto_count})"),
                (bucket_problem_toggle["label"], f"-- ({live_problem_count})"),
            ):
                widget.configure(text=text)
            for toggle in (bucket_manual_toggle, bucket_auto_toggle, bucket_problem_toggle):
                refresh = toggle.get("refresh")
                if callable(refresh):
                    refresh()
        except Exception:
            pass

        mode = str(selected_mode_var.get() or "").strip().lower()
        self._set_plate_auto_scope_selection_mode(mode == "selected", dialog=dialog)
        if mode == "selected":
            can_continue = live_selected_count > 0
        elif mode == "buckets":
            can_continue = bool(resolve_bucket_paths())
        else:
            can_continue = bool(live_all_paths)

        can_continue = _scope_model_inputs_ready(can_continue)

        try:
            next_btn.configure(
                text="Uruchom autoanotację",
                state=(tk.NORMAL if can_continue else tk.DISABLED),
            )
        except Exception:
            pass

        try:
            dialog.after_idle(lambda state=scroll_state: _restore_scope_modal_scroll_state(state))
            dialog.after(35, lambda state=scroll_state: _restore_scope_modal_scroll_state(state))
        except Exception:
            _restore_scope_modal_scroll_state(scroll_state)

    self._plate_auto_scope_modal_refresh_callback = refresh_scope_choices
    self._plate_auto_scope_modal_selection_refresh_callback = refresh_selected_scope_counter
    self._plate_auto_scope_modal_last_selection_signature = self._get_preview_listbox_selection_signature()

    def _schedule_scope_selection_watch(delay_ms: int = 120) -> None:
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        if getattr(self, "_plate_auto_scope_modal_refresh_callback", None) is not refresh_scope_choices:
            return
        try:
            self._plate_auto_scope_modal_selection_watch_after_id = dialog.after(
                int(delay_ms),
                _poll_scope_selection,
            )
        except Exception:
            self._plate_auto_scope_modal_selection_watch_after_id = None

    def _poll_scope_selection() -> None:
        self._plate_auto_scope_modal_selection_watch_after_id = None
        try:
            if not dialog.winfo_exists():
                return
        except Exception:
            return
        if getattr(self, "_plate_auto_scope_modal_refresh_callback", None) is not refresh_scope_choices:
            return

        try:
            current_signature = self._get_preview_listbox_selection_signature()
            previous_signature = tuple(getattr(self, "_plate_auto_scope_modal_last_selection_signature", ()) or ())
        except Exception:
            current_signature = previous_signature = ()

        mode = str(selected_mode_var.get() or "").strip().lower()
        if mode == "selected" and current_signature != previous_signature:
            self._plate_auto_scope_modal_last_selection_signature = current_signature
            self._refresh_plate_auto_scope_modal_selection_state()
        elif mode != "selected":
            self._plate_auto_scope_modal_last_selection_signature = current_signature

        _schedule_scope_selection_watch(450 if mode == "selected" else 900)

    _schedule_scope_selection_watch(450)

    cleanup_done = {"value": False}

    def _restore_scope_open_preview_state_if_cancelled() -> None:
        choice = str(result.get("choice") or "").strip().lower()
        if choice and choice != "cancel":
            return
        try:
            if bool(getattr(self, "is_processing", False)):
                return
            if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
                return
        except Exception:
            return

        visible_annotations = scope_open_visible_state.get("annotations")
        if not isinstance(visible_annotations, list):
            return
        try:
            self.current_annotations = copy.deepcopy(visible_annotations)
            self._preview_image_path_map = dict(scope_open_visible_state.get("image_map") or {})
            self.current_preview_index = scope_open_visible_state.get("preview_index")
            self._preview_approved_filenames = set(scope_open_visible_state.get("approved_filenames") or set())
            if not self._is_free_mode_session_context():
                self._campaign_pending_approved_filenames = set(
                    scope_open_visible_state.get("campaign_pending_approved") or set()
                )
                self._campaign_hidden_project_approved_filenames = set(
                    scope_open_visible_state.get("hidden_project_approved") or set()
                )
                self._campaign_hidden_char_effective_filenames = set(
                    scope_open_visible_state.get("hidden_char_effective") or set()
                )
            try:
                self._invalidate_preview_runtime_caches()
            except Exception:
                pass
            try:
                self._refresh_preview_list(preserve_selection=True, render_current=False)
            except Exception:
                pass
        except Exception:
            pass

    def _cleanup_scope_modal(event=None, *_args):
        try:
            if event is not None and getattr(event, "widget", None) is not dialog:
                return
        except Exception:
            pass
        if cleanup_done.get("value"):
            return
        cleanup_done["value"] = True
        if getattr(self, "_plate_auto_scope_modal_refresh_callback", None) is refresh_scope_choices:
            self._plate_auto_scope_modal_refresh_callback = None
        if getattr(self, "_plate_auto_scope_modal_selection_refresh_callback", None) is refresh_selected_scope_counter:
            self._plate_auto_scope_modal_selection_refresh_callback = None
        try:
            pending_refresh_after_id = getattr(self, "_plate_auto_scope_modal_refresh_after_id", None)
            if pending_refresh_after_id:
                self.frame.after_cancel(pending_refresh_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_refresh_after_id = None
        try:
            pending_selection_after_id = getattr(self, "_plate_auto_scope_modal_selection_refresh_after_id", None)
            if pending_selection_after_id:
                self.frame.after_cancel(pending_selection_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_selection_refresh_after_id = None
        try:
            pending_watch_after_id = getattr(self, "_plate_auto_scope_modal_selection_watch_after_id", None)
            if pending_watch_after_id:
                dialog.after_cancel(pending_watch_after_id)
        except Exception:
            pass
        self._plate_auto_scope_modal_selection_watch_after_id = None
        self._plate_auto_scope_modal_last_selection_signature = ()
        self._plate_auto_scope_active_dialog = None
        _consume_scope_model_confirmation()
        self._set_plate_auto_scope_selection_mode(False)
        self._set_plate_auto_scope_modal_ui_lock(False)
        _restore_scope_open_preview_state_if_cancelled()

        def _refresh_auto_cta_after_scope_close() -> None:
            try:
                self._refresh_step2_action_states(lightweight=False)
            except TypeError:
                try:
                    self._refresh_step2_action_states()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                self._refresh_free_mode_workflow_ui()
            except Exception:
                pass

        try:
            self.frame.after_idle(_refresh_auto_cta_after_scope_close)
        except Exception:
            _refresh_auto_cta_after_scope_close()
        try:
            if owner_window is not None and owner_map_bind_id:
                owner_window.unbind("<Map>", owner_map_bind_id)
        except Exception:
            pass
        try:
            if owner_window is not None and owner_focus_bind_id:
                owner_window.unbind("<FocusIn>", owner_focus_bind_id)
        except Exception:
            pass

    try:
        dialog.bind("<Destroy>", _cleanup_scope_modal, add="+")
    except Exception:
        pass

    def _schedule_scope_choices_refresh(*_args, delay_ms: int = 90):
        try:
            pending_refresh_after_id = getattr(self, "_plate_auto_scope_modal_refresh_after_id", None)
            if pending_refresh_after_id:
                self.frame.after_cancel(pending_refresh_after_id)
        except Exception:
            pass

        def _run_refresh():
            self._plate_auto_scope_modal_refresh_after_id = None
            refresh_scope_choices()

        try:
            self._plate_auto_scope_modal_refresh_after_id = self.frame.after(
                max(20, int(delay_ms or 90)),
                _run_refresh,
            )
        except Exception:
            self._plate_auto_scope_modal_refresh_after_id = None
            refresh_scope_choices()

    try:
        selected_mode_var.trace_add("write", _schedule_scope_choices_refresh)
        protect_existing_var.trace_add("write", _schedule_scope_choices_refresh)
        include_manual_var.trace_add("write", _schedule_scope_choices_refresh)
        include_auto_var.trace_add("write", _schedule_scope_choices_refresh)
        include_problem_var.trace_add("write", _schedule_scope_choices_refresh)
        plate_mode_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_plate_picker_scope_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_plate_path_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_plate_adopt_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_conf_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_use_vehicle_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_vehicle_model_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_vehicle_picker_scope_var.trace_add("write", _schedule_scope_choices_refresh)
        modal_vehicle_custom_var.trace_add("write", _schedule_scope_choices_refresh)
    except Exception:
        pass
    refresh_scope_choices()
    try:
        _bind_scope_modal_scroll_children(body_content)
        scroll_canvas.bind("<MouseWheel>", _scope_modal_redirect_wheel, add="+")
        scroll_canvas.bind("<Button-4>", _scope_modal_redirect_wheel, add="+")
        scroll_canvas.bind("<Button-5>", _scope_modal_redirect_wheel, add="+")
    except Exception:
        pass
    try:
        self._fit_borderless_dialog(dialog, parent=self.frame, min_width=760, min_height=620)
        dialog.update_idletasks()
        try:
            scroll_canvas.yview_moveto(0.0)
        except Exception:
            pass
        dialog.deiconify()
        dialog.lift()
        _restore_scope_modal_visibility()
    except Exception:
        pass
    try:
        dialog.protocol("WM_DELETE_WINDOW", choose_cancel)
    except Exception:
        pass
    dialog.bind("<Escape>", lambda _e: choose_cancel())
    dialog.wait_variable(dialog_done_var)

    choice = str(result.get("choice") or "").strip().lower()
    if not choice or choice == "cancel":
        _cleanup_scope_modal()
        return None
    payload = result.get("payload")
    if isinstance(payload, dict) and payload.get("image_paths"):
        return payload
    return {"mode": "all", "label": "Cały zestaw zdjęć", "image_paths": all_paths}
