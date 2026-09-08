#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3/PZ2 detection tab UI builder extracted from CharacterAnnotationTab."""

import time
import tkinter as tk
from tkinter import ttk

from ..config import logger
from .canvas_progress_overlay import CanvasProgressOverlay
from .z3_detection_guard_dialog import prompt_pz2_detection_guard_options
from .z3_detection_runtime import (
    confirm_last_detection_result,
    refresh_detection_review_controls,
    run_detection_stage,
    run_fast_ocr_test,
    undo_last_detection_result,
    unlock_ui_after_testing,
)
from .z3_detection_controls_ui import (
    apply_detection_advanced_section_style,
    apply_detection_param_row_style,
    bind_detect_mode_card,
    compose_detection_method_status,
    format_last_detection_summary_line,
    format_plate_last_detection_line,
    get_detection_pipeline_short_label,
    get_detection_advanced_toggle_label,
    load_last_detection_summary,
    get_detection_method_key,
    get_detection_method_status_label,
    get_hybrid_detection_status_text,
    get_hybrid_rescue_max_chars,
    get_yolo_rescue_enabled,
    get_yolo_box_ocr_status_text,
    handle_detect_mode_selection,
    normalize_detection_method_key,
    on_yolo_option_var_write,
    refresh_detect_mode_cards,
    refresh_detection_active_model_label,
    refresh_detection_refiner_guard_label,
    refresh_last_detection_status_label,
    save_last_detection_summary,
    refresh_detection_advanced_sections,
    set_detection_method_key,
    set_detection_process_log_visibility,
    set_test_progress_counter,
    set_test_progress_detail,
    set_test_status,
    show_last_detection_details,
    toggle_detection_advanced_panel,
    toggle_detection_process_log,
    update_detection_progress_ui,
)
from .z3_detection_pipeline_ui import (
    get_detection_pipeline_blocks,
    compile_detection_pipeline_blocks,
    get_detection_workflow_text,
    apply_detection_pipeline_blocks,
    refresh_detection_workflow_info_label,
    get_detection_pipeline_block_meta,
    get_detection_pipeline_block_style,
    get_detection_pipeline_builder_blocks,
    set_detection_pipeline_builder_blocks,
    get_saved_detection_pipeline_blocks,
    save_detection_pipeline_blocks,
    select_detection_pipeline_builder_block,
    set_detection_pipeline_builder_preset,
    append_detection_pipeline_builder_block,
    move_detection_pipeline_builder_selected_block,
    remove_detection_pipeline_builder_selected_block,
    clear_detection_pipeline_builder,
    close_detection_pipeline_builder,
    commit_detection_pipeline_builder,
    refresh_detection_pipeline_builder,
    refresh_detection_pipeline_model_row,
    refresh_detection_pipeline_builder_property_panel,
    pick_detection_pipeline_yolo_model,
    show_detection_pipeline_model_details,
    open_detection_pipeline_advanced_modal,
    draw_detection_pipeline_builder_canvas,
    open_detection_pipeline_builder,
)
from .z3_detection_model_ui import (
    get_detection_active_model_status,
    has_configured_yolo_detection_model,
    get_campaign_detection_yolo_model_path,
    get_campaign_char_model_path,
    get_effective_yolo_model_path,
    sync_yolo_model_binding,
    infer_yolo_arch_from_model_path,
    auto_device_label,
    get_available_devices,
    normalize_selected_device,
    get_effective_detection_device_choice,
    refresh_device_options,
    device_to_ultralytics,
    device_to_ocr,
    update_device_hint,
    apply_global_yolo_device_choice,
    set_button_emphasis,
    ensure_yolo_model_checkpoint,
    pick_yolo_model,
    update_yolo_visibility,
    refresh_yolo_model_picker_state,
)
from .z3_detection_algorithms import (
    apply_final_truth_count_guard,
    apply_yolo_box_backend,
    best_text_distance,
    build_pz2_detection_guard_counts,
    find_best_yolo_box_backend_index,
    find_best_yolo_rescue_index,
    fit_detection_count_to_truths,
    get_text_mismatch_positions,
    get_yolo_runtime_settings,
    levenshtein_distance,
    pick_best_true_text,
    repair_ocr_with_yolo_boxes,
    resolve_canonical_detections,
)
from .help_manager import HELP
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


def build_detection_tab(
    host,
    parent,
    SlimProgressBar,
    nav_button_width,
    detection_method_card_meta,
    preview_box_mode_options,
    preview_layout_filter_options,
    preview_sort_options,
):
    self = host
    build_profile_start = time.perf_counter()
    build_profile_last = build_profile_start
    build_profile_marks: list[str] = []

    def _mark_build_phase(label: str, *, threshold_ms: float = 180.0) -> None:
        nonlocal build_profile_last
        now = time.perf_counter()
        delta_ms = (now - build_profile_last) * 1000.0
        total_ms = (now - build_profile_start) * 1000.0
        build_profile_last = now
        if delta_ms >= threshold_ms:
            build_profile_marks.append(f"{label}={delta_ms:.0f}ms/{total_ms:.0f}ms")

    NAV_BUTTON_WIDTH = nav_button_width
    DETECTION_METHOD_CARD_META = detection_method_card_meta
    PREVIEW_BOX_MODE_OPTIONS = preview_box_mode_options
    PREVIEW_LAYOUT_FILTER_OPTIONS = preview_layout_filter_options
    PREVIEW_SORT_OPTIONS = preview_sort_options
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_rowconfigure(1, weight=0)
    parent.grid_columnconfigure(0, weight=1)

    content_frame = ttk.Frame(parent, style="Panel.TFrame")
    self.detect_content_frame = content_frame
    content_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 0))
    content_frame.grid_rowconfigure(0, weight=1)
    content_frame.grid_rowconfigure(1, weight=0)
    content_frame.grid_rowconfigure(2, weight=0)
    content_frame.grid_columnconfigure(0, weight=1)

    palette = getattr(self.app, "palette", {})
    split = tk.PanedWindow(
        content_frame,
        orient=tk.HORIZONTAL,
        bd=0,
        sashwidth=7,
        sashpad=2,
        sashrelief=tk.FLAT,
        opaqueresize=False,
        bg=str(palette.get("panel", "#252526")),
    )
    self.detect_split = split
    split.grid(row=0, column=0, sticky="nsew")
    split.bind("<ButtonPress-1>", self._begin_pz2_panel_resize, add="+")
    split.bind("<ButtonRelease-1>", self._end_pz2_panel_resize, add="+")

    left_panel = ttk.Frame(split, style="Panel.TFrame")
    right_panel = ttk.Frame(split, style="Panel.TFrame", padding=0)
    self.detect_left_panel = left_panel
    self.detect_right_panel = right_panel
    split.add(left_panel, minsize=560, stretch="always")
    split.add(right_panel, minsize=300, stretch="never")

    left_panel.grid_rowconfigure(0, weight=0)
    left_panel.grid_rowconfigure(1, weight=1)
    left_panel.grid_columnconfigure(0, weight=1)

    self.detect_left_header = ttk.Frame(left_panel, style="Panel.TFrame")
    self.detect_left_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    self.detect_left_header.grid_columnconfigure(0, weight=1)

    self.detect_left_title_lbl = SectionHeaderLabel(
        self.detect_left_header,
        self.app,
        text="PZ2. Wykrywanie znaków i analiza",
    )
    self.detect_left_title_lbl.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    self.detect_left_intro_lbl = tk.Label(
        self.detect_left_header,
        text=(
            "Pracujesz na cropach tablic przygotowanych w PZ1. "
            "Po lewej wybierasz tablicę, a na canvasie rysujesz i poprawiasz boxy znaków. "
            "Status perfect program nada automatycznie, gdy zapisane znaki będą zgodne z nazwą pliku źródłowego."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=900,
        bd=0,
        highlightthickness=0,
    )
    self.detect_left_intro_lbl.grid(row=1, column=0, sticky="ew")
    self._set_inline_status_label_state(
        self.detect_left_intro_lbl,
        text=self.detect_left_intro_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    self.detect_left_header.bind(
        "<Configure>",
        lambda event: self.detect_left_intro_lbl.configure(wraplength=max(280, int(getattr(event, "width", 0) or 0) - 24)),
        add="+",
    )
    self.detect_left_header.grid_remove()

    old_splash_overlay = getattr(self, "campaign_detect_splash_overlay", None)
    old_splash_card = getattr(self, "campaign_detect_splash_card", None)
    old_splash_title_lbl = getattr(self, "campaign_detect_splash_title_lbl", None)
    old_splash_body_lbl = getattr(self, "campaign_detect_splash_body_lbl", None)
    old_splash_progress = getattr(self, "campaign_detect_splash_progress", None)
    old_splash_return_btn = getattr(self, "campaign_detect_splash_return_btn", None)
    old_splash_surface = getattr(self, "_campaign_detect_splash_surface", None)
    keep_existing_splash = False
    try:
        keep_existing_splash = bool(
            getattr(self, "_campaign_detect_splash_visible", False)
            and old_splash_overlay is not None
            and old_splash_overlay.winfo_exists()
        )
    except Exception:
        keep_existing_splash = False
    try:
        if old_splash_overlay is not None and old_splash_overlay.winfo_exists() and not keep_existing_splash:
            old_splash_overlay.destroy()
    except Exception:
        pass

    self.campaign_detect_splash_overlay = tk.Frame(
        content_frame,
        bd=0,
        highlightthickness=0,
    )
    self._campaign_detect_splash_surface = content_frame
    self.campaign_detect_splash_card = tk.Frame(
        self.campaign_detect_splash_overlay,
        bd=0,
        highlightthickness=1,
        padx=22,
        pady=20,
    )
    self.campaign_detect_splash_card.place(relx=0.5, rely=0.34, anchor="n")
    self.campaign_detect_splash_card.grid_columnconfigure(0, weight=1, minsize=560)

    self.campaign_detect_splash_title_lbl = tk.Label(
        self.campaign_detect_splash_card,
        text="Przygotowuję wyodrębnione tablice dla Z3",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 13),
        bd=0,
        highlightthickness=0,
    )
    self.campaign_detect_splash_title_lbl.grid(row=0, column=0, sticky="ew")

    self.campaign_detect_splash_body_lbl = tk.Label(
        self.campaign_detect_splash_card,
        text="To automatyczny krok pośredni przed pracą nad znakami.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
        bd=0,
        highlightthickness=0,
    )
    self.campaign_detect_splash_body_lbl.grid(row=1, column=0, sticky="ew", pady=(10, 0))

    self.campaign_detect_splash_progress = ttk.Progressbar(
        self.campaign_detect_splash_card,
        mode="determinate",
        maximum=100.0,
    )
    self.campaign_detect_splash_progress.grid(row=2, column=0, sticky="ew", pady=(14, 0))

    self.campaign_detect_splash_return_btn = ttk.Button(
        self.campaign_detect_splash_card,
        text="Wróć do E3",
        command=self._return_to_wizard_for_step3_rework,
    )
    self.campaign_detect_splash_return_btn.grid(row=3, column=0, sticky="w", pady=(14, 0))
    self.campaign_detect_splash_return_btn.grid_remove()
    self.campaign_detect_splash_overlay.place_forget()
    if keep_existing_splash:
        new_splash_overlay = self.campaign_detect_splash_overlay
        try:
            if new_splash_overlay is not None and new_splash_overlay.winfo_exists():
                new_splash_overlay.destroy()
        except Exception:
            pass
        self.campaign_detect_splash_overlay = old_splash_overlay
        self.campaign_detect_splash_card = old_splash_card
        self.campaign_detect_splash_title_lbl = old_splash_title_lbl
        self.campaign_detect_splash_body_lbl = old_splash_body_lbl
        self.campaign_detect_splash_progress = old_splash_progress
        self.campaign_detect_splash_return_btn = old_splash_return_btn
        self._campaign_detect_splash_surface = old_splash_surface
    _mark_build_phase("shell")

    preview_panel_bg = palette.get("panel", "#252526")
    preview_panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    self.preview_vertical_split = tk.PanedWindow(
        left_panel,
        orient=tk.HORIZONTAL,
        sashwidth=6,
        sashrelief=tk.FLAT,
        sashcursor="sb_h_double_arrow",
        cursor="arrow",
        showhandle=False,
        opaqueresize=False,
        bd=0,
        relief=tk.FLAT,
        background=preview_panel_bg,
    )
    self.preview_vertical_split.grid(row=1, column=0, sticky="nsew")
    self.preview_vertical_split.bind("<ButtonPress-1>", self._begin_pz2_panel_resize, add="+")
    self.preview_vertical_split.bind("<ButtonRelease-1>", self._end_pz2_panel_resize, add="+")

    preview_lf = tk.Frame(
        self.preview_vertical_split,
        bg=preview_panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=preview_panel_border,
        highlightcolor=preview_panel_border,
        padx=8,
        pady=8,
        cursor="arrow",
    )
    self.preview_lf = preview_lf
    preview_lf.grid_rowconfigure(0, weight=0)
    preview_lf.grid_rowconfigure(1, weight=0)
    preview_lf.grid_rowconfigure(2, weight=1)
    preview_lf.grid_rowconfigure(3, weight=0)
    preview_lf.grid_columnconfigure(0, weight=1)

    self.preview_canvas_host = tk.Frame(
        preview_lf,
        bg=palette.get("field", preview_panel_bg),
        bd=0,
        highlightthickness=0,
    )
    self.preview_canvas_host.grid(row=2, column=0, sticky="nsew")
    self.preview_canvas_host.grid_rowconfigure(0, weight=1)
    self.preview_canvas_host.grid_columnconfigure(0, weight=1)
    self.preview_canvas_host.bind(
        "<Configure>",
        lambda _event: self._schedule_preview_overlay_relayout(delay_ms=55),
        add="+",
    )

    self.preview_canvas = tk.Canvas(
        self.preview_canvas_host,
        bg=palette.get("field", preview_panel_bg),
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
        takefocus=1,
    )
    self.preview_canvas.grid(row=0, column=0, sticky="nsew")
    self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)
    self.preview_canvas.bind("<Map>", self._on_preview_canvas_configure, add="+")
    self.preview_canvas.bind("<Enter>", self._on_preview_canvas_enter, add="+")
    self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_canvas_press, add="+")
    self.preview_canvas.bind("<ButtonPress-3>", self._on_preview_canvas_secondary_press, add="+")
    self.preview_canvas.bind("<B1-Motion>", self._on_preview_canvas_drag, add="+")
    self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_canvas_release, add="+")
    self.preview_canvas.bind(
        "<Double-Button-1>",
        lambda event: self._on_preview_canvas_press(event)
        if self._extract_preview_action_from_current_item() == "toggle_plate_rows"
        else self._edit_selected_preview_char_symbol(event),
        add="+",
    )
    self.preview_canvas.bind("<Motion>", self._on_preview_canvas_motion, add="+")
    self.preview_canvas.bind("<Leave>", self._on_preview_canvas_leave, add="+")
    self.preview_canvas.bind("<MouseWheel>", self._on_preview_canvas_mousewheel, add="+")
    self.preview_canvas.bind("<Button-4>", self._on_preview_canvas_mousewheel, add="+")
    self.preview_canvas.bind("<Button-5>", self._on_preview_canvas_mousewheel, add="+")
    self.preview_canvas.bind("<Alt-w>", self._on_preview_char_label_shortcut, add="+")
    self.preview_canvas.bind("<Alt-W>", self._on_preview_char_label_shortcut, add="+")
    self.preview_canvas.bind("<KeyPress>", self._on_preview_canvas_keypress, add="+")
    self.preview_canvas.bind("<KeyRelease>", self._on_preview_canvas_keyrelease, add="+")

    self.preview_processing_overlay_controller = CanvasProgressOverlay(
        self.preview_canvas_host,
        palette=palette,
        overlay_bg=blend_hex_colors(palette.get("panel", "#252526"), "#000000", 0.34),
        on_cancel=lambda: self.fast_test_stop.set(),
    )

    self.preview_tools = ttk.Frame(preview_lf, style="Panel.TFrame")
    self.preview_tools.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    self.preview_tools.grid_columnconfigure(0, weight=1)
    self.preview_edit_toolbar = self.preview_tools

    self.preview_title_lbl = SectionHeaderLabel(
        self.preview_tools,
        self.app,
        text="Podgląd tablicy i boxów znaków",
    )
    self.preview_title_lbl.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    self.preview_intro_lbl = tk.Label(
        self.preview_tools,
        text=(
            "Pracujesz na cropach tablic przygotowanych w PZ1. "
            "Wybierz metodę OCR/YOLO, uruchom detekcję, popraw boxy znaków na canvasie "
            "i oznacz poprawne tablice jako perfect przed przejściem do PZ3."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=780,
        bd=0,
        highlightthickness=0,
    )
    self.preview_intro_lbl.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    self.preview_focus_prompt_shell = tk.Frame(
        self.preview_tools,
        bd=0,
        highlightthickness=1,
        padx=10,
        pady=8,
        cursor="hand2",
    )
    self.preview_focus_prompt_shell.grid(row=2, column=0, sticky="ew")
    self.preview_focus_prompt_shell.grid_columnconfigure(0, weight=1)

    self.preview_focus_prompt_lbl = tk.Label(
        self.preview_focus_prompt_shell,
        text="ENTER - pełny canvas. Najwygodniej rysować i poprawiać boxy znaków w pełnym widoku.",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.preview_focus_prompt_lbl.grid(row=0, column=0, sticky="w")

    self.preview_focus_prompt_key_lbl = tk.Label(
        self.preview_focus_prompt_shell,
        text="Enter",
        anchor="center",
        justify=tk.CENTER,
        font=("Consolas", 9, "bold"),
        bd=0,
        highlightthickness=1,
        padx=10,
        pady=3,
        cursor="hand2",
    )
    self.preview_focus_prompt_key_lbl.grid(row=0, column=1, sticky="e", padx=(10, 0))

    for widget in (
        self.preview_focus_prompt_shell,
        self.preview_focus_prompt_lbl,
        self.preview_focus_prompt_key_lbl,
    ):
        widget.bind("<Button-1>", self._on_preview_enter_fullscreen_shortcut, add="+")

    self.preview_shortcuts_lbl = tk.Label(
        self.preview_tools,
        text="Skróty: Q/E przełącza poprzednią/następną tablicę na liście, D uzbraja rysowanie nowego boxa, S zaznacza lub odznacza hoverowany box, PPM usuwa zaznaczony box, Alt+W włącza tryb wpisywania znaków, LPM albo strzałki lewo/prawo wybierają pole, 0-9/A-Z wpisuje znak, Esc wychodzi z wpisywania, Enter przełącza pełny ekran, Ctrl+Z/Y cofa i ponawia.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=780,
        bd=0,
        highlightthickness=0,
    )
    self.preview_shortcuts_lbl.grid(row=3, column=0, sticky="ew", pady=(4, 0))
    self.preview_shortcuts_lbl.grid_remove()
    self.preview_tools.bind("<Configure>", self._sync_preview_intro_wraplength, add="+")
    self._set_inline_status_label_state(
        self.preview_intro_lbl,
        text=self.preview_intro_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    self._set_inline_status_label_state(
        self.preview_shortcuts_lbl,
        text=self.preview_shortcuts_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    self._preview_tools_hidden_by_design = True
    # PZ2 canvas ma startować bez opisowej karty nad podglądem.
    # Opisy przepływu trzymamy w AS/helpie, nie nad obszarem roboczym.
    for widget in (
        self.preview_title_lbl,
        self.preview_intro_lbl,
        self.preview_focus_prompt_shell,
        self.preview_shortcuts_lbl,
    ):
        try:
            widget.grid_remove()
        except Exception:
            pass
    try:
        self.preview_tools.grid_remove()
    except Exception:
        pass

    self.preview_record_overlay = tk.Frame(
        self.preview_canvas,
        bd=0,
        highlightthickness=0,
        padx=0,
        pady=0,
    )
    self.preview_record_overlay.place_forget()
    self.preview_record_overlay.grid_columnconfigure(0, weight=0)
    self.preview_record_source_lbl = None

    self.preview_hidden_controls = ttk.Frame(self.preview_tools, style="Panel.TFrame")

    self.preview_prev_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Poprzednie (Q)",
        command=self._on_preview_prev_shortcut,
        state=tk.DISABLED,
    )

    self.preview_next_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Następne (E)",
        command=self._on_preview_next_shortcut,
        state=tk.DISABLED,
    )

    self.preview_fit_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Dopasuj tablicę do okna (F)",
        command=self._on_preview_fit_shortcut,
        state=tk.DISABLED,
    )

    self.preview_fullscreen_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Pełny ekran (Enter)",
        command=self._toggle_preview_fullscreen,
        state=tk.DISABLED,
    )

    self.preview_mode_overlay = tk.Frame(self.preview_canvas, bd=0, highlightthickness=1, padx=8, pady=8)
    self.preview_mode_overlay.place_forget()

    self.preview_typing_overlay = tk.Frame(self.preview_canvas, bd=0, highlightthickness=1, padx=6, pady=4)
    self.preview_typing_overlay.place_forget()
    self.preview_typing_overlay_lbl = tk.Text(
        self.preview_typing_overlay,
        width=26,
        height=2,
        wrap=tk.WORD,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
        relief=tk.FLAT,
        cursor="arrow",
        takefocus=0,
        padx=0,
        pady=0,
    )
    self.preview_typing_overlay_lbl.pack(fill=tk.X)

    self.preview_mode_eye_btn = tk.Canvas(
        self.preview_mode_overlay,
        width=34,
        height=24,
        bd=0,
        relief=tk.FLAT,
        cursor="hand2",
        takefocus=0,
        highlightthickness=0,
    )
    self.preview_mode_eye_btn.bind("<ButtonPress-1>", self._on_preview_mode_eye_press)
    self.preview_mode_eye_btn.bind("<B1-Motion>", self._on_preview_mode_eye_drag)
    self.preview_mode_eye_btn.bind("<ButtonRelease-1>", self._on_preview_mode_eye_release)
    self.preview_mode_overlay_toggle_btn = self.preview_mode_eye_btn

    self.preview_mode_overlay_content = tk.Frame(self.preview_mode_overlay, bd=0, highlightthickness=0)
    self.preview_mode_overlay_content.pack(fill=tk.X)

    self.preview_mode_overlay_title_lbl = tk.Label(
        self.preview_mode_overlay_content,
        text="Źródło ramek znaków",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bd=0,
        highlightthickness=0,
    )
    self.preview_mode_overlay_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.preview_mode_overlay_current_lbl = tk.Label(
        self.preview_mode_overlay_content,
        text="Aktualnie pokazujesz: Auto (wg etapu)",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
    )
    self.preview_mode_overlay_current_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.preview_mode_overlay_body = tk.Frame(self.preview_mode_overlay_content, bd=0, highlightthickness=0)
    self.preview_mode_overlay_body.pack(fill=tk.X)

    self.preview_edit_toggle_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Edytuj ramki",
        command=self._toggle_preview_char_edit_mode,
    )

    self.preview_add_box_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Nowa ramka",
        command=self._toggle_preview_char_add_mode,
    )

    self.preview_edit_char_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Zmień znak",
        command=self._edit_selected_preview_char_symbol,
    )

    self.preview_delete_char_btn = ttk.Button(
        self.preview_hidden_controls,
        text="Usuń ramkę",
        command=self._delete_selected_preview_char_box,
    )

    self.preview_box_mode_toggle_btn = ttk.Menubutton(
        self.preview_record_overlay,
        text="Źródło ramek znaków",
        direction="below",
        style="PreviewOverlayFlat.TMenubutton",
    )
    self.preview_box_mode_toggle_btn.grid(row=0, column=0, sticky="w")
    self.preview_box_mode_toggle_menu = tk.Menu(self.preview_box_mode_toggle_btn, tearoff=False)
    self.preview_box_mode_toggle_btn["menu"] = self.preview_box_mode_toggle_menu
    for mode_key, mode_label in PREVIEW_BOX_MODE_OPTIONS:
        self.preview_box_mode_toggle_menu.add_radiobutton(
            label=str(mode_label),
            value=str(mode_label),
            variable=self.preview_box_mode_var,
            command=self._on_preview_box_mode_change,
        )
    self.preview_box_mode_toggle_menu.add_separator()
    self.preview_box_mode_toggle_menu.add_command(
        label="Wyczyść wybór (Auto)",
        command=self._reset_preview_box_mode_selection,
    )
    self._apply_preview_source_actions_style()

    self.preview_hint_frame = ttk.Frame(self.preview_canvas_host, style="Panel.TFrame")
    self.preview_hint_frame.bind("<ButtonPress-1>", self._on_preview_controls_legend_press, add="+")
    self.preview_hint_frame.bind("<B1-Motion>", self._on_preview_controls_legend_drag, add="+")
    self.preview_hint_frame.bind("<ButtonRelease-1>", self._on_preview_controls_legend_release, add="+")
    self.preview_hint_frame.bind("<Motion>", self._on_preview_controls_legend_motion, add="+")
    self.preview_hint_frame.bind("<Leave>", self._on_preview_controls_legend_leave, add="+")
    self.preview_hint_frame.grid_columnconfigure(0, weight=1)
    self.preview_hint_frame.grid_rowconfigure(0, weight=1)
    self.preview_controls_canvas = tk.Canvas(
        self.preview_hint_frame,
        height=72,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.preview_controls_canvas.grid(row=0, column=0, sticky="nsew")
    self.preview_controls_canvas.bind(
        "<Configure>",
        self._on_preview_controls_legend_configure,
        add="+",
    )
    self.preview_controls_canvas.bind("<ButtonPress-1>", self._on_preview_controls_legend_press, add="+")
    self.preview_controls_canvas.bind("<B1-Motion>", self._on_preview_controls_legend_drag, add="+")
    self.preview_controls_canvas.bind("<ButtonRelease-1>", self._on_preview_controls_legend_release, add="+")
    self.preview_controls_canvas.bind("<Motion>", self._on_preview_controls_legend_motion, add="+")
    self.preview_controls_canvas.bind("<Leave>", self._on_preview_controls_legend_leave, add="+")
    self.preview_controls_canvas.bind("<MouseWheel>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_canvas.bind("<Button-4>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_canvas.bind("<Button-5>", self._on_preview_controls_legend_mousewheel, add="+")
    _mark_build_phase("preview_canvas")
    self.preview_controls_vbar = WebSlimScrollbar(
        self.preview_hint_frame,
        orient=tk.VERTICAL,
        command=self.preview_controls_canvas.yview,
        auto_hide=False,
        thickness=6,
        thumb_scale=0.42,
        track_color=palette.get("panel", "#252526"),
        thumb_color="#2ecc71",
        thumb_hover_color="#56f29d",
    )
    self.preview_controls_canvas.configure(yscrollcommand=self.preview_controls_vbar.set)

    self.preview_overlay_dock = tk.Frame(
        self.preview_canvas_host,
        bd=0,
        highlightthickness=1,
        padx=6,
        pady=5,
        cursor="arrow",
    )
    self.preview_overlay_dock_header = tk.Frame(self.preview_overlay_dock, bd=0, highlightthickness=0, cursor="arrow")
    self.preview_overlay_dock_header.pack(fill=tk.X)
    self.preview_overlay_dock_title_lbl = tk.Label(
        self.preview_overlay_dock_header,
        text="OL",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    self.preview_overlay_dock_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.preview_overlay_dock_toggle_lbl = tk.Label(
        self.preview_overlay_dock_header,
        text=">",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    self.preview_overlay_dock_toggle_lbl.pack(side=tk.RIGHT)
    self.preview_overlay_dock_body = tk.Frame(self.preview_overlay_dock, bd=0, highlightthickness=0)
    self._preview_overlay_dock_tool_rows = {}
    row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="hand2")
    icon = tk.Label(
        row,
        text="K",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    label = tk.Label(
        row,
        text="Kompas",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    status = tk.Label(
        row,
        text="ON",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="hand2",
    )
    icon.pack(side=tk.LEFT, padx=(4, 5), pady=4)
    label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["legend"] = {
        "row": row,
        "icon": icon,
        "label": label,
        "status": status,
    }
    assistant_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="hand2")
    assistant_icon = tk.Label(
        assistant_row,
        text="AS",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    assistant_label = tk.Label(
        assistant_row,
        text="Asysta pracy",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    assistant_status = tk.Label(
        assistant_row,
        text="ON",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="hand2",
    )
    assistant_icon.pack(side=tk.LEFT, padx=(4, 5), pady=4)
    assistant_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    assistant_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["assistant"] = {
        "row": assistant_row,
        "icon": assistant_icon,
        "label": assistant_label,
        "status": assistant_status,
    }
    layout_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="hand2")
    layout_icon = tk.Label(
        layout_row,
        text="RZ",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    layout_label = tk.Label(
        layout_row,
        text="Układ tablicy",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    layout_status = tk.Label(
        layout_row,
        text="AUTO",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="hand2",
    )
    layout_icon.pack(side=tk.LEFT, padx=(4, 5), pady=4)
    layout_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    layout_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["layout"] = {
        "row": layout_row,
        "icon": layout_icon,
        "label": layout_label,
        "status": layout_status,
    }
    plate_status_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    plate_status_icon = tk.Label(
        plate_status_row,
        text="ST",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    plate_status_label = tk.Label(
        plate_status_row,
        text="Ocena tablicy",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    plate_status_value = tk.Label(
        plate_status_row,
        text="KOREKTA",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    plate_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    plate_status_value.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["plate_status"] = {
        "row": plate_status_row,
        "icon": plate_status_icon,
        "label": plate_status_label,
        "status": plate_status_value,
    }
    export_condition_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    export_condition_icon = tk.Label(
        export_condition_row,
        text="EX",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    export_condition_label = tk.Label(
        export_condition_row,
        text="MINIMUM ZBIORU",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    export_condition_status = tk.Label(
        export_condition_row,
        text="BRAK",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    export_condition_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    export_condition_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["export_condition"] = {
        "row": export_condition_row,
        "icon": export_condition_icon,
        "label": export_condition_label,
        "status": export_condition_status,
    }
    gate_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    gate_icon = tk.Label(
        gate_row,
        text="PZ2",
        width=3,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    gate_label = tk.Label(
        gate_row,
        text="Cel PZ2",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    gate_status = tk.Label(
        gate_row,
        text="ZAMKNIĘTA",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    gate_icon.pack(side=tk.LEFT, padx=(4, 5), pady=4)
    gate_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    gate_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["gate"] = {
        "row": gate_row,
        "icon": gate_icon,
        "label": gate_label,
        "status": gate_status,
    }
    quality_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    quality_icon = tk.Label(
        quality_row,
        text="",
        width=0,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_label = tk.Label(
        quality_row,
        text="JAKOŚĆ ZBIORU",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_status = tk.Label(
        quality_row,
        text="SŁABY",
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    quality_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    quality_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["quality"] = {
        "row": quality_row,
        "icon": quality_icon,
        "label": quality_label,
        "status": quality_status,
    }
    quality_have_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    quality_have_icon = tk.Label(
        quality_have_row,
        text="",
        width=0,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_have_label = tk.Label(
        quality_have_row,
        text="W ZBIORZE",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_have_status = tk.Label(
        quality_have_row,
        text="0",
        anchor="e",
        justify=tk.RIGHT,
        font=("Segoe UI", 7, "normal"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    quality_have_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    quality_have_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["quality_have"] = {
        "row": quality_have_row,
        "icon": quality_have_icon,
        "label": quality_have_label,
        "status": quality_have_status,
    }
    quality_missing_row = tk.Frame(self.preview_overlay_dock_body, bd=0, highlightthickness=1, cursor="arrow")
    quality_missing_icon = tk.Label(
        quality_missing_row,
        text="",
        width=0,
        anchor="center",
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_missing_label = tk.Label(
        quality_missing_row,
        text="DO KOLEJNEGO POZIOMU",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="arrow",
    )
    quality_missing_status = tk.Label(
        quality_missing_row,
        text="0",
        anchor="e",
        justify=tk.RIGHT,
        font=("Segoe UI", 7, "normal"),
        bd=0,
        highlightthickness=0,
        padx=5,
        pady=1,
        cursor="arrow",
    )
    quality_missing_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
    quality_missing_status.pack(side=tk.RIGHT, padx=(6, 4), pady=4)
    self._preview_overlay_dock_tool_rows["quality_missing"] = {
        "row": quality_missing_row,
        "icon": quality_missing_icon,
        "label": quality_missing_label,
        "status": quality_missing_status,
    }
    for widget in (row, icon, label, status):
        widget.bind("<Button-1>", lambda _event: self._toggle_preview_overlay_dock_tool("legend"), add="+")
    for widget in (assistant_row, assistant_icon, assistant_label, assistant_status):
        widget.bind("<Button-1>", lambda _event: self._toggle_preview_overlay_dock_tool("assistant"), add="+")
    for widget in (layout_row, layout_icon, layout_label, layout_status):
        widget.bind("<Button-1>", lambda event: self._cycle_preview_plate_layout_override(event), add="+")
    self.preview_overlay_dock.place_forget()
    self.frame.after_idle(
        lambda: (
            self._place_preview_overlay_dock(force_render=True),
            self._place_preview_hint_overlay(refresh=True),
        )
    )
    _mark_build_phase("preview_overlay_dock")

    self.preview_edit_status_lbl = tk.Label(
        preview_lf,
        textvariable=self.preview_edit_status_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=780,
        bd=0,
        highlightthickness=0,
    )
    self.preview_edit_status_lbl.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    self.preview_edit_status_lbl.grid_remove()
    self._set_inline_status_label_state(self.preview_edit_status_lbl, text=self.preview_edit_status_var.get(), tone="muted", emphasis=False)
    self._apply_preview_focus_prompt_style()
    self._refresh_preview_editor_toolbar()
    self._update_preview_edit_status()

    list_lf = tk.Frame(
        self.preview_vertical_split,
        bg=preview_panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=preview_panel_border,
        highlightcolor=preview_panel_border,
        padx=8,
        pady=8,
        cursor="arrow",
    )
    self.preview_list_lf = list_lf
    list_lf.grid_rowconfigure(5, weight=1)
    list_lf.grid_columnconfigure(0, weight=1)
    list_lf.grid_columnconfigure(1, minsize=16)

    self.preview_vertical_split.add(list_lf, minsize=240)
    self.preview_vertical_split.add(preview_lf, minsize=320)

    self.preview_list_header_lbl = SectionHeaderLabel(
        list_lf,
        self.app,
        text="PZ2. Wykrywanie znaków i analiza",
    )
    self.preview_list_header_lbl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

    self.preview_list_intro_lbl = tk.Label(
        list_lf,
        text=(
            "PZ2 przygotowuje anotacje znaków na wyodrębnionych tablicach: poprawiasz ramki, wpisujesz znaki "
            "i doprowadzasz tablice do statusu perfect. Gdy zbiór PZ2 jest sensowny, przejdź do PZ3 i utwórz źródłowy dataset znaków."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=260,
        bd=0,
        highlightthickness=0,
    )
    self.preview_list_intro_lbl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    self._set_inline_status_label_state(
        self.preview_list_intro_lbl,
        text=self.preview_list_intro_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    list_lf.bind(
        "<Configure>",
        lambda event: self.preview_list_intro_lbl.configure(wraplength=max(180, int(getattr(event, "width", 0) or 0) - 28)),
        add="+",
    )

    self.plates_list_title_lbl = SectionHeaderLabel(
        list_lf,
        self.app,
        text="Lista tablic do korekty",
    )
    self.plates_list_title_lbl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    # PZ2 ma jeden główny nagłówek w lewym panelu; dodatkowy tytuł samej listy jest zbędny.
    for widget in (
        self.plates_list_title_lbl,
    ):
        try:
            widget.grid_remove()
        except Exception:
            pass

    self.plates_legend_frame = tk.Frame(list_lf, bd=0, highlightthickness=0)
    self.plates_legend_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    for column, minsize in enumerate((50, 36, 38, 36, 34, 30, 30)):
        self.plates_legend_frame.grid_columnconfigure(column, weight=0, minsize=minsize)

    self.plates_legend_scope_header_lbl = tk.Label(self.plates_legend_frame, text="Typ", bd=0, highlightthickness=0)
    self.plates_legend_scope_header_lbl.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_total_header_lbl = tk.Label(self.plates_legend_frame, text="Raz.", bd=0, highlightthickness=0)
    self.plates_legend_total_header_lbl.grid(row=0, column=1, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_perfect_header_lbl = tk.Label(self.plates_legend_frame, text="Perf.", bd=0, highlightthickness=0)
    self.plates_legend_perfect_header_lbl.grid(row=0, column=2, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_manual_header_lbl = tk.Label(self.plates_legend_frame, text="Man.", bd=0, highlightthickness=0)
    self.plates_legend_manual_header_lbl.grid(row=0, column=3, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_yolo_header_lbl = tk.Label(self.plates_legend_frame, text="YOLO", bd=0, highlightthickness=0)
    self.plates_legend_yolo_header_lbl.grid(row=0, column=4, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_ocr_header_lbl = tk.Label(self.plates_legend_frame, text="OCR", bd=0, highlightthickness=0)
    self.plates_legend_ocr_header_lbl.grid(row=0, column=5, sticky="w", padx=(0, 4), pady=(0, 1))
    self.plates_legend_hybrid_header_lbl = tk.Label(self.plates_legend_frame, text="Hyb.", bd=0, highlightthickness=0)
    self.plates_legend_hybrid_header_lbl.grid(row=0, column=6, sticky="w", pady=(0, 1))

    self.plates_legend_tab_title_lbl = tk.Label(self.plates_legend_frame, text="Tablice", bd=0, highlightthickness=0)
    self.plates_legend_tab_title_lbl.grid(row=1, column=0, sticky="w", padx=(0, 4))
    self.plates_legend_tab_total_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_total_lbl.grid(row=1, column=1, sticky="w", padx=(0, 4))
    self.plates_legend_tab_perfect_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_perfect_lbl.grid(row=1, column=2, sticky="w", padx=(0, 4))
    self.plates_legend_tab_manual_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_manual_lbl.grid(row=1, column=3, sticky="w", padx=(0, 4))
    self.plates_legend_tab_yolo_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_yolo_lbl.grid(row=1, column=4, sticky="w", padx=(0, 4))
    self.plates_legend_tab_ocr_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_ocr_lbl.grid(row=1, column=5, sticky="w", padx=(0, 4))
    self.plates_legend_tab_hybrid_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_tab_hybrid_lbl.grid(row=1, column=6, sticky="w")

    self.plates_legend_box_title_lbl = tk.Label(self.plates_legend_frame, text="Boxy", bd=0, highlightthickness=0)
    self.plates_legend_box_title_lbl.grid(row=2, column=0, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_total_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_box_total_lbl.grid(row=2, column=1, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_perfect_lbl = tk.Label(self.plates_legend_frame, text="-", bd=0, highlightthickness=0)
    self.plates_legend_box_perfect_lbl.grid(row=2, column=2, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_manual_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_box_manual_lbl.grid(row=2, column=3, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_yolo_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_box_yolo_lbl.grid(row=2, column=4, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_ocr_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_box_ocr_lbl.grid(row=2, column=5, sticky="w", padx=(0, 4), pady=(1, 0))
    self.plates_legend_box_hybrid_lbl = tk.Label(self.plates_legend_frame, text="0", bd=0, highlightthickness=0)
    self.plates_legend_box_hybrid_lbl.grid(row=2, column=6, sticky="w", pady=(1, 0))

    self._apply_plates_legend_style()
    self._set_plates_legend_info()

    self.preview_import_focus_frame = ttk.Frame(list_lf, style="Panel.TFrame")
    self.preview_import_focus_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    self.preview_import_focus_frame.grid_columnconfigure(1, weight=1)

    self.preview_import_focus_btn = ttk.Button(
        self.preview_import_focus_frame,
        text="Pokaż tylko zaimportowane",
        command=self._toggle_preview_import_focus,
        style="WorkflowCard.TButton",
        state=tk.DISABLED,
    )
    self.preview_import_focus_btn.grid(row=0, column=0, sticky="w")

    self.preview_import_focus_hint_lbl = tk.Label(
        self.preview_import_focus_frame,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=220,
        bd=0,
        highlightthickness=0,
    )
    self.preview_import_focus_hint_lbl.grid(row=0, column=1, sticky="ew", padx=(8, 0))
    self._set_inline_status_label_state(
        self.preview_import_focus_hint_lbl,
        text="",
        tone="muted",
        emphasis=False,
    )
    self.preview_import_focus_frame.grid_remove()

    self.preview_list_host = tk.Frame(list_lf, bd=0, highlightthickness=0, cursor="arrow")
    self.preview_list_host.grid(row=5, column=0, columnspan=2, sticky="nsew")
    self.preview_list_host.grid_rowconfigure(1, weight=1)
    self.preview_list_host.grid_columnconfigure(0, weight=1)

    self.preview_sort_bar = tk.Frame(
        self.preview_list_host,
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
    )
    self.preview_sort_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
    self.preview_sort_bar.grid_columnconfigure(0, weight=1)

    self.preview_sort_title_lbl = tk.Label(
        self.preview_sort_bar,
        text="Sortowanie:",
        anchor="w",
        bd=0,
        highlightthickness=0,
        padx=0,
        font=("Segoe UI", 8),
    )
    self.preview_sort_title_lbl.grid(row=0, column=0, sticky="w", padx=(0, 2), pady=(0, 2))

    self.preview_sort_buttons_frame = tk.Frame(self.preview_sort_bar, bd=0, highlightthickness=0)
    self.preview_sort_buttons_frame.grid(row=1, column=0, sticky="ew")
    buttons_per_row = 4
    for button_column in range(buttons_per_row):
        self.preview_sort_buttons_frame.grid_columnconfigure(button_column, weight=1)

    self.preview_sort_buttons = {}
    for idx, (mode_key, mode_label) in enumerate(PREVIEW_SORT_OPTIONS):
        btn_row = idx // buttons_per_row
        btn_column = idx % buttons_per_row
        btn = tk.Button(
            self.preview_sort_buttons_frame,
            text=mode_label,
            font=("Segoe UI", 8),
            padx=6,
            pady=1,
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            takefocus=0,
            command=lambda target_key=mode_key: self._on_preview_sort_mode_change(target_key),
        )
        btn.grid(
            row=btn_row,
            column=btn_column,
            sticky="ew",
            padx=(0, 4) if btn_column < (buttons_per_row - 1) else 0,
            pady=(0, 4) if btn_row == 0 else 0,
        )
        btn.bind("<Enter>", lambda _event, target_key=mode_key: self._set_preview_sort_hover(target_key, True))
        btn.bind("<Leave>", lambda _event, target_key=mode_key: self._set_preview_sort_hover(target_key, False))
        self.preview_sort_buttons[mode_key] = btn

    self.preview_layout_filter_title_lbl = None
    self.preview_layout_filter_buttons_frame = None
    self.preview_layout_filter_buttons = {}
    self._apply_preview_sort_bar_style()

    self.plates_listbox = tk.Listbox(
        self.preview_list_host,
        font=("Consolas", 10),
        selectmode=tk.BROWSE,
        exportselection=False,
        activestyle="none",
        cursor="arrow",
    )
    self.plates_listbox.grid(row=1, column=0, sticky="nsew", pady=0)

    scroll = WebSlimScrollbar(self.preview_list_host, command=self.plates_listbox.yview)
    scroll.grid(row=1, column=1, sticky="ns", padx=(6, 0))

    self.plates_listbox.config(yscrollcommand=scroll.set)
    self.plates_listbox.bind("<Button-1>", self._on_preview_list_mouse_primary, add=False)
    self.plates_listbox.bind("<B1-Motion>", lambda _event: "break", add=False)
    self.plates_listbox.bind(
        "<<ListboxSelect>>",
        lambda _event: (self._schedule_preview_select_render(delay_ms=1), "break")[1],
    )
    self.plates_listbox.bind("<Up>", lambda _event: self._handle_preview_list_arrow_nav(-1), add=False)
    self.plates_listbox.bind("<Down>", lambda _event: self._handle_preview_list_arrow_nav(1), add=False)
    self.plates_listbox.bind("<KP_Up>", lambda _event: self._handle_preview_list_arrow_nav(-1), add=False)
    self.plates_listbox.bind("<KP_Down>", lambda _event: self._handle_preview_list_arrow_nav(1), add=False)
    def _handle_preview_list_qe_nav(event):
        try:
            self.plates_listbox.focus_set()
        except Exception:
            pass
        result = self._on_preview_canvas_keypress(event)
        return result or "break"

    self.plates_listbox.bind("<KeyPress-q>", _handle_preview_list_qe_nav, add=False)
    self.plates_listbox.bind("<KeyPress-Q>", _handle_preview_list_qe_nav, add=False)
    self.plates_listbox.bind("<KeyPress-e>", _handle_preview_list_qe_nav, add=False)
    self.plates_listbox.bind("<KeyPress-E>", _handle_preview_list_qe_nav, add=False)
    self.plates_listbox.bind("<Escape>", self._on_preview_escape_shortcut, add=False)
    self.plates_listbox.bind("<MouseWheel>", self._on_plates_listbox_mousewheel, add="+")
    self.plates_listbox.bind("<Button-4>", self._on_plates_listbox_mousewheel, add="+")
    self.plates_listbox.bind("<Button-5>", self._on_plates_listbox_mousewheel, add="+")
    _mark_build_phase("preview_list")

    self.preview_box_mode_rows = []
    for mode_key, mode_label in PREVIEW_BOX_MODE_OPTIONS:
        row = tk.Frame(self.preview_mode_overlay_body, bd=0, highlightthickness=0, cursor="hand2")
        row.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        indicator = tk.Canvas(
            row,
            width=16,
            height=16,
            bd=0,
            highlightthickness=0,
            cursor="hand2"
        )
        indicator.pack(side=tk.LEFT, padx=(0, 6))
        label = tk.Label(
            row,
            text=mode_label,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            cursor="hand2"
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _select_preview_mode(_event=None, target_label=mode_label):
            self.preview_box_mode_var.set(target_label)
            self._on_preview_box_mode_change()
            self._preview_mode_overlay_expanded = False
            self._refresh_preview_mode_overlay_visibility()
            return "break"

        for widget in (row, indicator, label):
            widget.bind("<Button-1>", _select_preview_mode)

        row_info = {
            "kind": "radio",
            "frame": row,
            "indicator": indicator,
            "label": label,
            "selected_getter": (lambda target_label=mode_label: self.preview_box_mode_var.get() == target_label),
            "hovered": False,
        }
        for widget in (row, indicator, label):
            widget.bind("<Enter>", lambda _event, info=row_info: self._set_selection_row_hover(info, True))
            widget.bind("<Leave>", lambda _event, info=row_info: self._set_selection_row_hover(info, False))
        self.preview_box_mode_rows.append(row_info)
    self._apply_preview_mode_radio_style()
    self._apply_preview_mode_overlay_style()
    self._refresh_preview_mode_overlay_visibility()

    right_panel.grid_rowconfigure(0, weight=0)
    right_panel.grid_rowconfigure(1, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)
    try:
        right_panel.grid_anchor("n")
    except Exception:
        pass
    try:
        right_panel.configure(width=368)
    except Exception:
        pass

    self.detect_right_pinned_status_host = ttk.Frame(right_panel, style="Panel.TFrame")
    self.detect_right_pinned_status_host.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    self.detect_right_pinned_status_host.grid_columnconfigure(0, weight=1)

    palette = getattr(self.app, "palette", {})
    self.detect_right_scroll_host = ttk.Frame(right_panel, style="Panel.TFrame")
    self.detect_right_scroll_host.grid(row=1, column=0, sticky="nsew")
    self.detect_right_scroll_host.grid_rowconfigure(0, weight=1)
    self.detect_right_scroll_host.grid_columnconfigure(0, weight=1)

    self.detect_right_canvas = tk.Canvas(
        self.detect_right_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0
    )
    self.detect_right_canvas.grid(row=0, column=0, sticky="nsew")

    self.detect_right_scrollbar = WebSlimScrollbar(
        self.detect_right_scroll_host,
        command=self.detect_right_canvas.yview
    )
    self.detect_right_scrollbar.grid(row=0, column=1, sticky="ns")
    self.detect_right_canvas.configure(yscrollcommand=self.detect_right_scrollbar.set)

    def _on_detect_right_canvas_mousewheel(event):
        return self._redirect_child_mousewheel_to_canvas(
            event,
            self.detect_right_canvas,
            self._detect_right_canvas_overflows,
        )

    for widget in (right_panel, self.detect_right_scroll_host, self.detect_right_canvas):
        try:
            widget.bind("<MouseWheel>", _on_detect_right_canvas_mousewheel, add="+")
            widget.bind("<Button-4>", _on_detect_right_canvas_mousewheel, add="+")
            widget.bind("<Button-5>", _on_detect_right_canvas_mousewheel, add="+")
        except Exception:
            pass

    self.detect_right_content = ttk.Frame(self.detect_right_canvas, style="Panel.TFrame")
    self.detect_right_content.grid_columnconfigure(0, weight=1)
    self.detect_right_content_window = self.detect_right_canvas.create_window(
        (0, 0),
        window=self.detect_right_content,
        anchor="nw"
    )
    self.detect_right_content.bind("<Configure>", self._sync_detect_right_scrollregion, add="+")
    _mark_build_phase("right_canvas")
    self.detect_right_canvas.bind("<Configure>", self._sync_detect_right_canvas_width, add="+")

    set_lf = ttk.LabelFrame(self.detect_right_content, text="", padding=(12, 4, 12, 10))
    set_lf.grid(row=0, column=0, sticky="ew", pady=(0, 10))

    self.detect_settings_title_lbl = SectionHeaderLabel(
        set_lf,
        self.app,
        text="Odczyt znaków i ramki",
        pady=1,
        min_height=22,
    )
    self.detect_settings_title_lbl.pack(fill=tk.X, pady=(0, 6))

    detect_mode_cards_frame = tk.Frame(set_lf, bd=0, highlightthickness=0)
    # Presety są wybierane dopiero w budowniczym pipeline. Prawy panel pokazuje
    # aktualny pipeline i jedno jawne CTA do jego zmiany.
    detect_mode_cards_frame.grid_columnconfigure(0, weight=1)

    self._detect_mode_cards = {}
    detect_mode_specs = ("OCR", "YOLO", "BOTH", "YOLO_OCR")
    for idx, mode_key in enumerate(detect_mode_specs):
        meta = DETECTION_METHOD_CARD_META.get(mode_key, {})
        title_text = str(meta.get("title", mode_key) or mode_key)
        desc_text = str(meta.get("desc", "") or "")
        card = tk.Frame(detect_mode_cards_frame, bd=0, highlightthickness=1, padx=12, pady=10)
        card.grid(row=idx, column=0, sticky="ew", pady=(0, 8) if idx < len(detect_mode_specs) - 1 else 0)
        card.grid_columnconfigure(1, weight=1)

        indicator = tk.Canvas(
            card,
            width=16,
            height=16,
            bd=0,
            highlightthickness=0,
        )
        indicator.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 10))

        title_lbl = tk.Label(
            card,
            text=title_text,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
            bd=0,
            highlightthickness=0,
        )
        title_lbl.grid(row=0, column=1, sticky="ew")

        desc_lbl = tk.Label(
            card,
            text=desc_text,
            anchor="w",
            justify=tk.LEFT,
            wraplength=250,
            bd=0,
            highlightthickness=0,
        )
        desc_lbl.grid(row=1, column=1, sticky="ew", pady=(4, 0))

        self._detect_mode_cards[mode_key] = {
            "frame": card,
            "indicator": indicator,
            "title": title_lbl,
            "desc": desc_lbl,
            "default_desc": desc_text,
            "enabled": True,
        }

    self._refresh_detect_mode_cards()

    pipeline_shell_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.74,
    )
    pipeline_shell_fill = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.90,
    )
    self.detect_pipeline_summary_shell = tk.Frame(
        set_lf,
        bg=pipeline_shell_border,
        bd=0,
        highlightthickness=1,
        highlightbackground=pipeline_shell_border,
        highlightcolor=pipeline_shell_border,
    )
    self.detect_pipeline_summary_shell.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
    self.detect_pipeline_summary_inner = tk.Frame(
        self.detect_pipeline_summary_shell,
        bg=pipeline_shell_fill,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=9,
    )
    self.detect_pipeline_summary_inner.pack(fill=tk.X, padx=1, pady=1)
    self.detect_pipeline_summary_inner.grid_columnconfigure(0, weight=1)

    self.detect_pipeline_summary_title_lbl = tk.Label(
        self.detect_pipeline_summary_inner,
        text="Aktywny pipeline detekcji",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bg=pipeline_shell_fill,
        fg=palette.get("fg", "#f3f3f3"),
        bd=0,
        highlightthickness=0,
    )
    self.detect_pipeline_summary_title_lbl.grid(row=0, column=0, sticky="ew")

    self.detect_run_model_info_lbl = tk.Label(
        self.detect_pipeline_summary_inner,
        text="OCR (O) | YOLO: bez YOLO | mAP50-95: -",
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        font=("Segoe UI", 9, "bold"),
        bg=pipeline_shell_fill,
        bd=0,
        highlightthickness=0,
        padx=0,
        pady=0,
    )
    self.detect_run_model_info_lbl._inline_status_font = ("Segoe UI", 9, "bold")
    self.detect_run_model_info_lbl._inline_status_bg = pipeline_shell_fill
    self.detect_run_model_info_lbl.grid(row=1, column=0, sticky="ew", pady=(4, 8))

    self.detect_workflow_info_lbl = ttk.Label(
        set_lf,
        text=self._get_detection_workflow_text(),
        style="Muted.TLabel",
        wraplength=0,
        justify=tk.LEFT,
    )

    self.detect_active_model_lbl = ttk.Label(
        set_lf,
        text="",
        style="Muted.TLabel",
        wraplength=320,
        justify=tk.LEFT,
    )
    self.yolo_model_status_lbl = self.detect_active_model_lbl

    self.yolo_model_row = tk.Frame(self.detect_pipeline_summary_inner, bg=pipeline_shell_fill, bd=0, highlightthickness=0)
    self.yolo_model_row.grid(row=2, column=0, sticky="ew")
    self.yolo_model_row.grid_columnconfigure(0, weight=1)
    self.yolo_model_row.grid_columnconfigure(1, weight=0)
    self.yolo_model_row.grid_columnconfigure(2, weight=0)

    self.detect_pipeline_builder_btn = ttk.Button(
        self.yolo_model_row,
        text="Zmień pipeline",
        command=self._open_detection_pipeline_builder,
        style="WorkflowCard.TButton",
    )
    self.detect_pipeline_builder_btn.grid(row=0, column=0, sticky="ew")
    self.detect_pipeline_builder_btn.configure(padding=(8, 6))

    self.btn_detection_last_details = ttk.Button(
        self.yolo_model_row,
        text="Szczegóły",
        command=self._show_last_detection_details,
        style="WorkflowCard.TButton",
        width=9,
    )
    self.btn_detection_last_details.grid(row=0, column=1, sticky="e", padx=(8, 0))
    self.btn_detection_last_details.configure(padding=(8, 6))

    self.det_yolo_model_browse_btn = ttk.Button(
        self.yolo_model_row,
        text="Wybierz model detekcji",
        command=self._pick_yolo_model,
        style="WorkflowCard.TButton"
    )
    self.det_yolo_model_browse_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))
    self.det_yolo_model_browse_btn.configure(padding=(8, 2))
    self._refresh_yolo_model_picker_state()
    self._refresh_detection_active_model_label()

    self._refresh_device_options()

    self.detection_advanced_toggle_btn = ttk.Button(
        self.detect_right_content,
        text=self._get_detection_advanced_toggle_label(),
        command=self._toggle_detection_advanced_panel,
        style="WorkflowCard.TButton",
    )
    self.detection_advanced_toggle_btn.grid(row=2, column=0, sticky="w", padx=(12, 12), pady=(0, 10))
    self.detection_advanced_toggle_btn.configure(padding=(8, 2))

    self.yolo_advanced_shell = tk.Frame(self.detect_right_content, bd=0, highlightthickness=1)
    self.yolo_advanced_body = ttk.Frame(self.yolo_advanced_shell, padding=(12, 10))
    self.yolo_advanced_body.pack(fill=tk.X)

    self.detect_yolo_advanced_title_lbl = tk.Label(
        self.yolo_advanced_body,
        text="Zaawansowane YOLO",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.detect_yolo_advanced_title_lbl.pack(fill=tk.X)

    self.detect_yolo_advanced_title_lbl.pack_configure(pady=(0, 8))

    def make_detection_param_row(parent, *, cursor: str = "", pady=(0, 4), padx: int = 10, inner_pady: int = 6):
        row = tk.Frame(
            parent,
            bd=0,
            highlightthickness=1,
            padx=padx,
            pady=inner_pady,
            cursor=cursor,
        )
        row.pack(fill=tk.X, pady=pady)
        return row

    def make_detection_stage_header(parent, text: str, *, pady=(0, 8)):
        label = tk.Label(
            parent,
            text=text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
        )
        label.pack(fill=tk.X, pady=(0, 4))
        divider = tk.Frame(parent, height=1, bd=0, highlightthickness=0)
        divider.pack(fill=tk.X, pady=pady)
        self._register_detection_section_widgets(labels=[label], dividers=[divider])
        return label, divider

    self.hybrid_rescue_frame = tk.Frame(self.yolo_advanced_body, bd=0, highlightthickness=0)

    self.hybrid_rescue_stage_lbl, self.hybrid_rescue_stage_divider = make_detection_stage_header(
        self.hybrid_rescue_frame,
        "2. Ratunek brakujących znaków",
    )

    self.hybrid_rescue_title_lbl = tk.Label(
        self.hybrid_rescue_frame,
        text="Rescue YOLO",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        wraplength=320,
    )
    self.hybrid_rescue_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
    try:
        self.app.ensure_adaptive_wrap(self.hybrid_rescue_title_lbl, container=self.hybrid_rescue_frame, padding=24, min_wrap=180)
    except Exception:
        pass

    self.hybrid_rescue_info_lbl = ttk.Label(
        self.hybrid_rescue_frame,
        text="Gdy normalny próg confidence zostawi mniej znaków niż wynika z napisu tablicy, system może zrobić kontrolny przebieg do 0.10. Wynik przechodzi tylko wtedy, gdy poprawia dopasowanie.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT,
    )
    self.hybrid_rescue_info_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    try:
        self.app.ensure_adaptive_wrap(self.hybrid_rescue_info_lbl, container=self.hybrid_rescue_frame, padding=24, min_wrap=180)
    except Exception:
        pass

    self.hybrid_rescue_toggle_row = make_detection_param_row(
        self.hybrid_rescue_frame,
        cursor="hand2",
        pady=(0, 8),
    )
    self.hybrid_rescue_toggle_indicator = tk.Canvas(
        self.hybrid_rescue_toggle_row,
        width=16,
        height=16,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.hybrid_rescue_toggle_indicator.pack(side=tk.LEFT, padx=(0, 6))
    self.hybrid_rescue_toggle_lbl = tk.Label(
        self.hybrid_rescue_toggle_row,
        text="Odzysk niskopewnych znaków",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.hybrid_rescue_toggle_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.hybrid_rescue_value_lbl = tk.Label(
        self.hybrid_rescue_toggle_row,
        width=5,
        anchor="e",
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.hybrid_rescue_value_lbl.pack(side=tk.RIGHT, padx=(8, 0))

    def _toggle_hybrid_rescue(_event=None):
        if not getattr(self, "hybrid_rescue_row_info", {}).get("enabled", True):
            return "break"
        try:
            enabled = self._get_hybrid_rescue_max_chars() > 0
            self.hybrid_rescue_max_chars_var.set(0 if enabled else 1)
        except Exception:
            self.hybrid_rescue_max_chars_var.set(1)
        return "break"

    for widget in (self.hybrid_rescue_toggle_row, self.hybrid_rescue_toggle_indicator, self.hybrid_rescue_toggle_lbl, self.hybrid_rescue_value_lbl):
        widget.bind("<Button-1>", _toggle_hybrid_rescue)

    hybrid_rescue_row_info = {
        "kind": "check",
        "frame": self.hybrid_rescue_toggle_row,
        "indicator": self.hybrid_rescue_toggle_indicator,
        "label": self.hybrid_rescue_toggle_lbl,
        "extra_widgets": [self.hybrid_rescue_value_lbl],
        "selected_getter": lambda: self._get_hybrid_rescue_max_chars() > 0,
        "enabled": True,
        "hovered": False,
    }
    for widget in (self.hybrid_rescue_toggle_row, self.hybrid_rescue_toggle_indicator, self.hybrid_rescue_toggle_lbl, self.hybrid_rescue_value_lbl):
        widget.bind("<Enter>", lambda _event, info=hybrid_rescue_row_info: self._set_selection_row_hover(info, True))
        widget.bind("<Leave>", lambda _event, info=hybrid_rescue_row_info: self._set_selection_row_hover(info, False))
    self.hybrid_rescue_row_info = hybrid_rescue_row_info
    self.yolo_option_rows.append(hybrid_rescue_row_info)

    def _refresh_hybrid_rescue_value(*_args):
        try:
            current_value = self._get_hybrid_rescue_max_chars()
            self.hybrid_rescue_value_lbl.configure(text=("OFF" if current_value <= 0 else "AUTO"))
            self._apply_yolo_option_check_style()
            self._save_local_setting("char_hybrid_rescue_max_chars", 1 if current_value > 0 else 0)
            self._refresh_detection_workflow_info_label()
            if self._get_detection_method_key() == "BOTH" and hasattr(self, "test_status_lbl"):
                self._set_test_status(
                    self._compose_detection_method_status(self._get_hybrid_detection_status_text()),
                    "info"
                )
        except Exception:
            self.hybrid_rescue_value_lbl.configure(text="AUTO")

    try:
        self.hybrid_rescue_max_chars_var.trace_add("write", _refresh_hybrid_rescue_value)
    except Exception:
        pass
    _refresh_hybrid_rescue_value()

    self.hybrid_box_backend_stage_lbl, self.hybrid_box_backend_stage_divider = make_detection_stage_header(
        self.hybrid_rescue_frame,
        "3. Źródło końcowych ramek",
    )

    self.hybrid_yolo_box_backend_row = make_detection_param_row(self.hybrid_rescue_frame, cursor="hand2")
    self.hybrid_yolo_box_backend_indicator = tk.Canvas(
        self.hybrid_yolo_box_backend_row,
        width=16,
        height=16,
        bd=0,
        highlightthickness=0,
        cursor="hand2"
    )
    self.hybrid_yolo_box_backend_indicator.pack(side=tk.LEFT, padx=(0, 6))
    self.hybrid_yolo_box_backend_lbl = tk.Label(
        self.hybrid_yolo_box_backend_row,
        text="Końcowe ramki bierz z YOLO",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        cursor="hand2"
    )
    self.hybrid_yolo_box_backend_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    self.hybrid_box_backend_info_lbl = ttk.Label(
        self.hybrid_rescue_frame,
        text=(
            "Włącz, jeśli finalny box do treningu ma pochodzić z YOLO, a nie z technicznego podziału OCR. "
            "Ramki końcowe z YOLO nie nadpisują manuali: ręczne boxy są chronione, a automat uzupełnia tylko pozostałe miejsca."
        ),
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT,
    )
    self.hybrid_box_backend_info_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    try:
        self.app.ensure_adaptive_wrap(self.hybrid_box_backend_info_lbl, container=self.hybrid_rescue_frame, padding=24, min_wrap=180)
    except Exception:
        pass

    def _toggle_hybrid_yolo_box_backend(_event=None):
        if not getattr(self, "hybrid_yolo_box_backend_row_info", {}).get("enabled", True):
            return "break"
        try:
            self.hybrid_yolo_box_backend_var.set(not bool(self.hybrid_yolo_box_backend_var.get()))
            if self._get_detection_method_key() == "BOTH" and hasattr(self, "test_status_lbl"):
                self._set_test_status(
                    self._compose_detection_method_status(self._get_hybrid_detection_status_text()),
                    "info"
                )
        except Exception:
            self.hybrid_yolo_box_backend_var.set(True)
        return "break"

    for widget in (self.hybrid_yolo_box_backend_row, self.hybrid_yolo_box_backend_indicator, self.hybrid_yolo_box_backend_lbl):
        widget.bind("<Button-1>", _toggle_hybrid_yolo_box_backend)

    hybrid_yolo_box_backend_row_info = {
        "kind": "check",
        "frame": self.hybrid_yolo_box_backend_row,
        "indicator": self.hybrid_yolo_box_backend_indicator,
        "label": self.hybrid_yolo_box_backend_lbl,
        "selected_getter": lambda: bool(self.hybrid_yolo_box_backend_var.get()),
        "enabled": True,
        "hovered": False,
    }
    for widget in (self.hybrid_yolo_box_backend_row, self.hybrid_yolo_box_backend_indicator, self.hybrid_yolo_box_backend_lbl):
        widget.bind("<Enter>", lambda _event, info=hybrid_yolo_box_backend_row_info: self._set_selection_row_hover(info, True))
        widget.bind("<Leave>", lambda _event, info=hybrid_yolo_box_backend_row_info: self._set_selection_row_hover(info, False))
    self.hybrid_yolo_box_backend_row_info = hybrid_yolo_box_backend_row_info
    self.yolo_option_rows.append(hybrid_yolo_box_backend_row_info)
    self._apply_yolo_option_check_style()

    self.yolo_panel = ttk.Frame(self.yolo_advanced_body)

    self.yolo_tuning_lf = ttk.LabelFrame(self.yolo_panel, text=" Strojenie ramek YOLO ", padding=8)
    self.yolo_tuning_lf.pack(fill=tk.X, pady=(10, 0))

    def add_yolo_scale(attr_name, parent, label_text, variable, from_, to_, digits=2):
        row = make_detection_param_row(parent)

        label = tk.Label(
            row,
            text=label_text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
        )
        label.pack(side=tk.LEFT)

        scale = ttk.Scale(row, from_=from_, to=to_, variable=variable)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        setattr(self, attr_name, scale)

        value_lbl = tk.Label(
            row,
            width=6,
            anchor="e",
            bd=0,
            highlightthickness=0,
        )
        value_lbl.pack(side=tk.RIGHT)

        fmt = "{:." + str(int(digits)) + "f}"

        def refresh_value(*_args):
            try:
                value_lbl.config(text=fmt.format(float(variable.get())))
            except Exception:
                value_lbl.config(text=str(variable.get()))

        try:
            variable.trace_add("write", refresh_value)
        except Exception:
            pass
        refresh_value()
        self._register_detection_param_row(row, labels=[label, value_lbl], scales=[scale])

    def pack_adaptive_yolo_info(parent, text, *, style="PanelMuted.TLabel", wraplength=320, pady=(0, 6), min_wrap=180, padding=24):
        info_lbl = ttk.Label(
            parent,
            text=text,
            style=style,
            wraplength=wraplength,
            justify=tk.LEFT
        )
        info_lbl.pack(anchor=tk.W, fill=tk.X, pady=pady)
        try:
            self.app.ensure_adaptive_wrap(info_lbl, container=parent, padding=padding, min_wrap=min_wrap)
        except Exception:
            pass
        return info_lbl

    def register_adaptive_wrap_descendants(container, *, padding=24, min_wrap=180):
        if container is None:
            return
        try:
            descendants = list(container.winfo_children())
        except Exception:
            descendants = []
        while descendants:
            widget = descendants.pop(0)
            try:
                descendants.extend(widget.winfo_children())
            except Exception:
                pass
            try:
                wraplength = int(float(widget.cget("wraplength") or 0))
            except Exception:
                wraplength = 0
            if wraplength <= 0:
                continue
            try:
                self.app.ensure_adaptive_wrap(widget, container=container, padding=padding, min_wrap=min_wrap)
            except Exception:
                pass

    def defer_adaptive_wrap_descendants(container, *, padding=24, min_wrap=180, delay_ms=900):
        if container is None:
            return

        def _run():
            try:
                if not bool(container.winfo_exists()):
                    return
            except Exception:
                return
            register_adaptive_wrap_descendants(container, padding=padding, min_wrap=min_wrap)

        try:
            self.frame.after(int(delay_ms), _run)
        except Exception:
            _run()

    def ensure_self_adaptive_wrap(widget, *, wraplength=None, padding=6, min_wrap=120):
        if widget is None:
            return
        if wraplength is not None:
            try:
                widget.configure(wraplength=wraplength)
            except Exception:
                pass
        try:
            self.app.ensure_adaptive_wrap(widget, container=widget, padding=padding, min_wrap=min_wrap)
        except Exception:
            pass

    ttk.Label(
        self.yolo_tuning_lf,
        text="Te progi pomagają odsiać słabe ramki, usuwać duplikaty i utrzymać wiarygodną geometrię znaków.",
        style="Muted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    self.yolo_conf_row = make_detection_param_row(self.yolo_tuning_lf)
    self.yolo_conf_lbl = tk.Label(
        self.yolo_conf_row,
        text="Confidence YB:",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.yolo_conf_lbl.pack(side=tk.LEFT)
    self.yolo_conf_spin = ttk.Spinbox(
        self.yolo_conf_row,
        from_=0.00001,
        to=1.0,
        increment=0.05,
        textvariable=self.yolo_box_conf_var,
        width=9,
        format="%.5f",
    )
    self.yolo_conf_spin.pack(side=tk.RIGHT, padx=(6, 0))
    self._register_detection_param_row(self.yolo_conf_row, labels=[self.yolo_conf_lbl])

    self.yolo_symbol_conf_row = make_detection_param_row(self.yolo_tuning_lf)
    self.yolo_symbol_conf_lbl = tk.Label(
        self.yolo_symbol_conf_row,
        text="Confidence YS:",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.yolo_symbol_conf_lbl.pack(side=tk.LEFT)
    self.yolo_symbol_conf_spin = ttk.Spinbox(
        self.yolo_symbol_conf_row,
        from_=0.00001,
        to=1.0,
        increment=0.05,
        textvariable=self.yolo_symbol_conf_var,
        width=9,
        format="%.5f",
    )
    self.yolo_symbol_conf_spin.pack(side=tk.RIGHT, padx=(6, 0))
    self._register_detection_param_row(self.yolo_symbol_conf_row, labels=[self.yolo_symbol_conf_lbl])

    ttk.Label(
        self.yolo_tuning_lf,
        text="Wyżej: mniej słabych i fałszywych boxów, ale można zgubić trudne znaki. Niżej: więcej trafień, ale rośnie ryzyko dubli i szumu.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    self.yolo_iou_row = make_detection_param_row(self.yolo_tuning_lf)
    self.yolo_iou_lbl = tk.Label(
        self.yolo_iou_row,
        text="NMS IoU:",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.yolo_iou_lbl.pack(side=tk.LEFT)
    self.yolo_iou_spin = ttk.Spinbox(
        self.yolo_iou_row,
        from_=0.01,
        to=0.99,
        increment=0.05,
        textvariable=self.yolo_iou_var,
        width=9,
        format="%.5f",
    )
    self.yolo_iou_spin.pack(side=tk.RIGHT, padx=(6, 0))
    self._register_detection_param_row(self.yolo_iou_row, labels=[self.yolo_iou_lbl])

    ttk.Label(
        self.yolo_tuning_lf,
        text="Niżej: NMS agresywniej scala podobne ramki i mocniej wycina duble. Wyżej: zostawia więcej zbliżonych boxów, co pomaga przy ciasnych znakach, ale może dublować.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    self.yolo_overlap_row = make_detection_param_row(self.yolo_tuning_lf)
    self.yolo_overlap_lbl = tk.Label(
        self.yolo_overlap_row,
        text="Nakładanie boxów:",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.yolo_overlap_lbl.pack(side=tk.LEFT)
    self.yolo_overlap_spin = ttk.Spinbox(
        self.yolo_overlap_row,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=self.yolo_overlap_var,
        width=9,
        format="%.5f",
    )
    self.yolo_overlap_spin.pack(side=tk.RIGHT, padx=(6, 0))
    self._register_detection_param_row(self.yolo_overlap_row, labels=[self.yolo_overlap_lbl])

    ttk.Label(
        self.yolo_tuning_lf,
        text="Niżej: system szybciej uzna dwa boxy za ten sam znak i odrzuci słabszy. Wyżej: dwa boxy muszą się mocniej pokrywać, więc więcej dubli może zostać.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    ttk.Label(
        self.yolo_tuning_lf,
        text="Próg liczony jako wspólna część powierzchni mniejszego boxa dla dwóch detekcji.",
        style="Muted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    self.yolo_agnostic_nms_row = make_detection_param_row(self.yolo_tuning_lf, cursor="hand2", pady=(0, 0))
    self.yolo_agnostic_nms_indicator = tk.Canvas(
        self.yolo_agnostic_nms_row,
        width=16,
        height=16,
        bd=0,
        highlightthickness=0,
        cursor="hand2"
    )
    self.yolo_agnostic_nms_indicator.pack(side=tk.LEFT, padx=(0, 6))
    self.yolo_agnostic_nms_lbl = tk.Label(
        self.yolo_agnostic_nms_row,
        text="Class agnostic",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        cursor="hand2"
    )
    self.yolo_agnostic_nms_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _toggle_yolo_agnostic(_event=None):
        if not getattr(self, "yolo_agnostic_row_info", {}).get("enabled", True):
            return "break"
        try:
            self.yolo_agnostic_nms_var.set(not bool(self.yolo_agnostic_nms_var.get()))
        except Exception:
            self.yolo_agnostic_nms_var.set(True)
        return "break"

    for widget in (self.yolo_agnostic_nms_row, self.yolo_agnostic_nms_indicator, self.yolo_agnostic_nms_lbl):
        widget.bind("<Button-1>", _toggle_yolo_agnostic)

    yolo_agnostic_row_info = {
        "kind": "check",
        "frame": self.yolo_agnostic_nms_row,
        "indicator": self.yolo_agnostic_nms_indicator,
        "label": self.yolo_agnostic_nms_lbl,
        "selected_getter": lambda: bool(self.yolo_agnostic_nms_var.get()),
        "enabled": True,
        "hovered": False,
    }
    for widget in (self.yolo_agnostic_nms_row, self.yolo_agnostic_nms_indicator, self.yolo_agnostic_nms_lbl):
        widget.bind("<Enter>", lambda _event, info=yolo_agnostic_row_info: self._set_selection_row_hover(info, True))
        widget.bind("<Leave>", lambda _event, info=yolo_agnostic_row_info: self._set_selection_row_hover(info, False))
    self.yolo_agnostic_row_info = yolo_agnostic_row_info
    self.yolo_option_rows.append(yolo_agnostic_row_info)
    self._apply_yolo_option_check_style()

    ttk.Label(
        self.yolo_tuning_lf,
        text="Włącz, gdy ten sam znak dostaje kilka klas naraz, np. B i 8. Wyłącz, gdy model zbyt mocno skleja sąsiednie, podobne znaki.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(4, 0))

    self.yolo_seq_lf = ttk.LabelFrame(self.yolo_tuning_lf, text=" Filtr wiarygodnej sekwencji YOLO ", padding=8)
    self.yolo_seq_lf.pack(fill=tk.X, pady=(10, 0))

    ttk.Label(
        self.yolo_seq_lf,
        text="Po NMS system dodatkowo sprawdza, czy ramki YOLO układają się w wiarygodną sekwencję znaków na tablicy.",
        style="Muted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_center_y_scale", self.yolo_seq_lf, "Tolerancja osi Y:", self.yolo_seq_center_y_var, 0.10, 1.50, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: znaki muszą leżeć bliżej jednej linii. Wyżej: filtr jest bardziej wyrozumiały dla krzywych lub nierównych tablic.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_min_h_scale", self.yolo_seq_lf, "Min. zgodność wysokości:", self.yolo_seq_min_h_ratio_var, 0.20, 1.00, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: łatwiej przepuścić małe lub uszkodzone boxy. Wyżej: filtr mocniej odrzuca znaki o wyraźnie innej wysokości.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_max_h_scale", self.yolo_seq_lf, "Max. wysokość względem mediany:", self.yolo_seq_max_h_ratio_var, 1.00, 3.50, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: szybciej wylatują podejrzanie wysokie boxy. Wyżej: łatwiej zostawić znaki z dużym marginesem lub przeskalowaniem.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_max_w_scale", self.yolo_seq_lf, "Max. szerokość względem mediany:", self.yolo_seq_max_w_ratio_var, 1.00, 4.50, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: system mocniej odcina szerokie śmieci lub zlane znaki. Wyżej: zostawia więcej nietypowych, szerokich liter.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_soft_overlap_scale", self.yolo_seq_lf, "Miękki konflikt nakładania:", self.yolo_seq_soft_overlap_var, 0.00, 1.00, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: nawet lekkie wchodzenie jednego boxa w drugi uruchamia rywalizację sąsiednich znaków. Wyżej: filtr rzadziej uznaje konflikt.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 6))

    add_yolo_scale("yolo_seq_hard_overlap_scale", self.yolo_seq_lf, "Twardy konflikt nakładania:", self.yolo_seq_hard_overlap_var, 0.00, 1.00, digits=2)
    ttk.Label(
        self.yolo_seq_lf,
        text="Niżej: mocno nachodzące boxy są szybciej traktowane jako dubel lub błąd. Wyżej: system dłużej toleruje ciężkie nakładanie sąsiednich znaków.",
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(0, 2))

    defer_adaptive_wrap_descendants(self.yolo_tuning_lf, padding=28, min_wrap=180, delay_ms=1200)
    defer_adaptive_wrap_descendants(self.yolo_seq_lf, padding=28, min_wrap=180, delay_ms=1200)
    self._apply_detection_param_row_style()
    self._update_yolo_visibility()
    _mark_build_phase("advanced_yolo")

    self.actions_lf = tk.Frame(self.detect_right_content, bd=0, highlightthickness=1)
    self.actions_lf.grid(row=4, column=0, sticky="nsew", pady=(0, 10), padx=(12, 12))

    self.ocr_advanced_body = ttk.Frame(self.actions_lf, padding=(12, 10))
    self.ocr_advanced_body.pack(fill=tk.X)

    self.detect_ocr_stage_lbl, self.detect_ocr_stage_divider = make_detection_stage_header(
        self.ocr_advanced_body,
        "1. Odczyt znaków",
    )

    self.detect_ocr_title_lbl = tk.Label(
        self.ocr_advanced_body,
        text="Zaawansowane OCR",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.detect_ocr_title_lbl.pack(fill=tk.X)

    self.detect_ocr_title_lbl.pack_configure(pady=(0, 8))

    add_yolo_scale(
        "ocr_min_height_ratio_scale",
        self.ocr_advanced_body,
        "Min. wysokość boxa OCR:",
        self.ocr_min_height_ratio_var,
        0.20,
        1.00,
        digits=2,
    )
    self.ocr_min_height_hint_lbl = ttk.Label(
        self.ocr_advanced_body,
        text=(
            "Chroni znaki typu Y/1 przed zbyt niskim boxem po czystym OCR. "
            "Wyżej: ramki mocniej trzymają wysokość wiersza; niżej: są ciaśniej docięte."
        ),
        style="PanelMuted.TLabel",
        wraplength=320,
        justify=tk.LEFT,
    )
    self.ocr_min_height_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    try:
        self.app.ensure_adaptive_wrap(self.ocr_min_height_hint_lbl, container=self.ocr_advanced_body, padding=24, min_wrap=180)
    except Exception:
        pass
    self._apply_detection_advanced_section_style()
    _mark_build_phase("advanced_ocr_header")

    self.ocr_summary_store = tk.Frame(self.actions_lf, bg=palette.get("panel", "#252526"), bd=0, highlightthickness=0)

    actions_block = ttk.Frame(self.ocr_advanced_body)
    actions_block.pack(fill=tk.X)
    actions_block.grid_columnconfigure(0, weight=1)

    self.btn_ocr_summary = ttk.Button(
        actions_block,
        text="Podsumowanie OCR",
        command=self._open_ocr_summary_modal,
        style="WorkflowCard.TButton"
    )
    self.btn_ocr_summary.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    self.btn_ocr_summary.configure(padding=(8, 2))

    ttk.Label(
        self.ocr_summary_store,
        text="Lider OCR:",
        font=("Segoe UI", 9, "bold")
    ).pack(anchor=tk.W, pady=(0, 2))

    self.winner_name_lbl = tk.Label(
        self.ocr_summary_store,
        text="BRAK DANYCH",
        width=30,
        anchor="w",
        justify="left",
        wraplength=340,
        bd=0,
        highlightthickness=0
    )
    self.winner_name_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_status_label_state(self.winner_name_lbl, text="BRAK DANYCH", tone="neutral", emphasis=True)

    ttk.Label(
        self.ocr_summary_store,
        text="Status rankingu OCR:",
        style="PanelMuted.TLabel"
    ).pack(anchor=tk.W, pady=(0, 2))

    self.winner_acc_lbl = tk.Label(
        self.ocr_summary_store,
        text="0.0%",
        width=30,
        anchor="w",
        justify="left",
        wraplength=340,
        bd=0,
        highlightthickness=0
    )
    self.winner_acc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_status_label_state(self.winner_acc_lbl, text="0.0%", tone="muted", emphasis=False)

    self.btn_ocr_lab = ttk.Button(
        actions_block,
        text="Laboratorium OCR",
        command=self._open_filter_lab,
        style="WorkflowCard.TButton"
    )
    self.btn_ocr_lab.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    self.btn_ocr_lab.configure(padding=(8, 2))

    self.btn_rank_presets = ttk.Button(
        actions_block,
        text="Ranking presetów OCR",
        command=self._open_ocr_ranking_modal,
        style="WorkflowCard.TButton"
    )
    self.btn_rank_presets.grid(row=2, column=0, sticky="ew")
    self.btn_rank_presets.configure(padding=(8, 2))

    preview_status_lf = ttk.LabelFrame(self.detect_right_pinned_status_host, text="", padding=(12, 4, 12, 10))
    preview_status_lf.grid(row=0, column=0, sticky="ew")
    preview_status_lf.grid_columnconfigure(0, weight=1)
    self.preview_status_lf = preview_status_lf

    self.preview_status_title_lbl = SectionHeaderLabel(
        preview_status_lf,
        self.app,
        text="Status pracy",
        pady=1,
        min_height=22,
    )
    self.preview_status_title_lbl.grid(row=0, column=0, sticky="ew", pady=(0, 4))

    self.preview_load_note_lbl = tk.Label(
        preview_status_lf,
        text="Status pracy, liczniki i jakość zbioru są widoczne tutaj. Szuflada na canvasie służy jako szybki skrót w trakcie edycji.",
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.preview_load_note_lbl.grid(row=1, column=0, sticky="ew", pady=(0, 1))
    self._set_inline_status_label_state(
        self.preview_load_note_lbl,
        text="Status pracy, liczniki i jakość zbioru są widoczne tutaj. Szuflada na canvasie służy jako szybki skrót w trakcie edycji.",
        tone="muted",
        emphasis=False
    )
    self.preview_info_lbl = None

    self.preview_counts_frame = tk.Frame(preview_status_lf, bd=0, highlightthickness=0)
    self.preview_counts_frame.grid(row=2, column=0, sticky="w", pady=(4, 0))
    for column, minsize in enumerate((82, 102, 70)):
        self.preview_counts_frame.grid_columnconfigure(column, weight=0, minsize=minsize)

    self.preview_perfect_title_lbl = tk.Label(
        self.preview_counts_frame,
        text="Perfect",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_perfect_title_lbl.grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 1))
    self.preview_error_title_lbl = tk.Label(
        self.preview_counts_frame,
        text="Do poprawy",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_error_title_lbl.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(0, 1))
    self.preview_total_title_lbl = tk.Label(
        self.preview_counts_frame,
        text="Razem",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_total_title_lbl.grid(row=0, column=2, sticky="w", pady=(0, 1))

    self.preview_perfect_count_lbl = tk.Label(
        self.preview_counts_frame,
        text="0",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_perfect_count_lbl.grid(row=1, column=0, sticky="w", padx=(0, 12))
    self.preview_error_count_lbl = tk.Label(
        self.preview_counts_frame,
        text="0",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_error_count_lbl.grid(row=1, column=1, sticky="w", padx=(0, 12))
    self.preview_unknown_count_lbl = tk.Label(
        self.preview_counts_frame,
        text="0",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
    )
    self.preview_unknown_count_lbl.grid(row=1, column=2, sticky="w")
    self.preview_total_count_lbl = self.preview_unknown_count_lbl
    self._apply_preview_info_stats_style()

    self.preview_layout_summary_lbl = tk.Label(
        preview_status_lf,
        text="Układ tablic: brak danych.",
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.preview_layout_summary_lbl.grid(row=3, column=0, sticky="ew", pady=(6, 0))
    self._set_inline_status_label_state(
        self.preview_layout_summary_lbl,
        text="Układ tablic: brak danych.",
        tone="muted",
        emphasis=False,
    )

    self.preview_repair_progress_title_lbl = tk.Label(
        preview_status_lf,
        text="Postęp poprawek: czekam na zestaw wyodrębnionych tablic.",
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.preview_repair_progress_title_lbl.grid(row=4, column=0, sticky="ew", pady=(6, 0))

    self.preview_repair_progress = SlimProgressBar(
        preview_status_lf,
        maximum=100.0,
        value=0.0,
        thickness=4,
        height=10,
    )
    self.preview_repair_progress.grid(row=5, column=0, sticky="ew", pady=(4, 0))

    self.preview_repair_progress_status_lbl = tk.Label(
        preview_status_lf,
        text="Ocena zbioru pojawi się po wczytaniu zestawu i pierwszych poprawkach.",
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.preview_repair_progress_status_lbl.grid(row=6, column=0, sticky="ew", pady=(4, 0))
    self._set_inline_status_label_state(
        self.preview_repair_progress_status_lbl,
        text="Ocena zbioru pojawi się po wczytaniu zestawu i pierwszych poprawkach.",
        tone="muted",
        emphasis=False,
    )
    self._update_preview_repair_progress_ui()

    self.preview_fusion_info_lbl = tk.Label(
        preview_status_lf,
        text=self._format_perfect_strategy_counts(self._empty_perfect_strategy_counts()),
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.preview_fusion_info_lbl.grid(row=7, column=0, sticky="ew", pady=(4, 0))
    self._set_inline_status_label_state(
        self.preview_fusion_info_lbl,
        text=self._format_perfect_strategy_counts(self._empty_perfect_strategy_counts()),
        tone="muted",
        emphasis=False
    )

    self.preview_box_mode_info_lbl = tk.Label(
        preview_status_lf,
        text="Źródło ramek znaków: brak zaznaczonej tablicy",
        justify="left",
        wraplength=320,
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.preview_box_mode_info_lbl.grid(row=8, column=0, sticky="ew", pady=(4, 0))
    self._set_inline_status_label_state(
        self.preview_box_mode_info_lbl,
        text="Źródło ramek znaków: brak zaznaczonej tablicy",
        tone="muted",
        emphasis=False
    )
    try:
        if bool(getattr(self, "_campaign_pz2_sync_loading", False)):
            self.preview_repair_progress_title_lbl.grid_remove()
            self.preview_repair_progress.grid_remove()
        self.preview_repair_progress_status_lbl.grid_remove()
        self.preview_fusion_info_lbl.grid_remove()
        self.preview_box_mode_info_lbl.grid_remove()
    except Exception:
        pass
    _mark_build_phase("right_status")

    preview_source_lf = ttk.LabelFrame(self.detect_right_content, text="", padding=(12, 10))
    preview_source_lf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    preview_source_lf.grid_columnconfigure(0, weight=1)
    self.preview_source_lf = preview_source_lf

    self.preview_dir_hint_lbl = ttk.Label(
        preview_source_lf,
        text="Źródło PZ2",
        style="Muted.TLabel",
        wraplength=320,
        justify=tk.LEFT,
    )
    self.preview_dir_hint_lbl.grid(row=0, column=0, sticky="ew", pady=(0, 4))

    self.preview_source_status_lbl = tk.Label(
        preview_source_lf,
        text="Brak aktywnego preview runu z PZ1",
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        bd=0,
        highlightthickness=0,
    )
    self.preview_source_status_lbl.grid(row=1, column=0, sticky="ew", pady=(0, 3))

    self.preview_source_detail_lbl = tk.Label(
        preview_source_lf,
        text="Najpierw uruchom wyodrębnianie w PZ1.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        bd=0,
        highlightthickness=0,
    )
    self.preview_source_detail_lbl.grid(row=2, column=0, sticky="ew", pady=(0, 8))

    self._refresh_preview_source_panel()
    _mark_build_phase("preview_source_panel")

    self.detection_log_frame = ttk.LabelFrame(content_frame, text=" Terminal procesu ", padding=10)
    self.detection_log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

    self.detection_log_host = ttk.Frame(self.detection_log_frame, style="Panel.TFrame")
    self.detection_log_host.pack(fill=tk.BOTH, expand=True)

    self.test_log_text = tk.Text(
        self.detection_log_host,
        height=7,
        wrap=tk.WORD,
        font=("Consolas", 9),
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
    )
    self.test_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.detection_log_scrollbar = WebSlimScrollbar(
        self.detection_log_host,
        orient=tk.VERTICAL,
        command=self.test_log_text.yview,
        auto_hide=False,
    )
    self.detection_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.test_log_text.configure(yscrollcommand=self.detection_log_scrollbar.set)
    self.test_log_text.web_vbar = self.detection_log_scrollbar

    try:
        self.test_log_text.insert(tk.END, "Gotowy do uruchomienia detekcji znaków.\n")
        self.test_log_text.configure(state="disabled")
    except Exception:
        pass

    self._set_detection_process_log_visibility(False)
    _mark_build_phase("process_log")

    footer_nav = ttk.Frame(content_frame)
    self.detect_footer_nav = footer_nav
    footer_nav.grid(row=1, column=0, sticky="sew", pady=(0, 0))
    footer_nav.grid_rowconfigure(0, weight=0)
    footer_nav.grid_rowconfigure(1, weight=0)
    footer_nav.grid_rowconfigure(2, weight=0)
    footer_nav.grid_rowconfigure(3, weight=0)
    footer_nav.grid_columnconfigure(0, weight=1)

    self.detect_nav_row = ttk.Frame(footer_nav)
    self.detect_nav_row.grid(row=3, column=0, sticky="sew", pady=(0, 0))
    self.detect_nav_row.grid_columnconfigure(0, weight=0)
    self.detect_nav_row.grid_columnconfigure(1, weight=1)
    self.detect_nav_row.grid_columnconfigure(2, weight=0)

    detection_action_button_padding = (10, 13)

    self.detect_actions_row = ttk.Frame(footer_nav)
    self.detect_actions_row.grid(row=0, column=0, sticky="ew", pady=(4, 4))
    self.detect_actions_row.grid_rowconfigure(0, weight=0)
    self.detect_actions_row.grid_columnconfigure(0, weight=0)
    self.detect_actions_row.grid_columnconfigure(1, weight=1)
    self.detect_actions_row.grid_columnconfigure(2, weight=0)

    self.btn_back_to_extract = ttk.Button(
        self.detect_nav_row,
        text="← Wstecz do Wycinania Tablic",
        command=self.back_to_substep_1,
        style="WorkflowCard.TButton"
    )
    self.btn_back_to_extract.grid(row=0, column=0, sticky="sw")
    self.btn_back_to_extract.configure(text="Wstecz do PZ1", padding=(6, 0), width=NAV_BUTTON_WIDTH)

    self.btn_return_to_graph_pz2_frame = tk.Frame(self.detect_nav_row, bd=0, highlightthickness=0)
    self.btn_return_to_graph_pz2_frame.grid(row=0, column=0, sticky="sw")
    self.btn_return_to_graph_pz2 = ttk.Button(
        self.btn_return_to_graph_pz2_frame,
        text="Zapisz PZ2 i wróć do grafu",
        command=self._return_to_wizard_from_step3_pz2,
        style="WorkflowCard.TButton",
    )
    self.btn_return_to_graph_pz2.pack(anchor=tk.SW)
    self.btn_return_to_graph_pz2.configure(padding=(8, 0), width=28)
    self.btn_return_to_graph_pz2_frame.grid_remove()

    self.btn_run_detection_frame = tk.Frame(self.detect_actions_row, bd=0, highlightthickness=0)
    self.btn_run_detection_frame.grid(row=0, column=0, sticky="w")

    self.btn_run_detection_pulse_frame = tk.Frame(
        self.btn_run_detection_frame,
        bd=0,
        highlightthickness=0
    )
    self.btn_run_detection_pulse_frame.pack(side=tk.LEFT, anchor=tk.CENTER)

    self.btn_run_detection = ttk.Button(
        self.btn_run_detection_pulse_frame,
        text="Uruchom detekcję",
        command=self._run_detection_stage,
        style="Accent.TButton"
    )
    self.btn_run_detection.pack(fill=tk.X)
    self.btn_run_detection.configure(
        text="Uruchom detekcję",
        padding=detection_action_button_padding,
        width=34,
    )

    def _sync_run_detection_button_to_list_width(event=None):
        try:
            raw_width = int(getattr(event, "width", 0) or self.preview_list_host.winfo_width() or 0)
        except Exception:
            raw_width = 0
        if raw_width <= 0:
            return
        width_chars = max(26, min(56, int(round(raw_width / 9.0))))
        try:
            self.btn_run_detection.configure(width=width_chars)
        except Exception:
            pass

    try:
        self.preview_list_host.bind("<Configure>", _sync_run_detection_button_to_list_width, add="+")
        self.btn_run_detection.after_idle(_sync_run_detection_button_to_list_width)
    except Exception:
        pass

    self.detection_review_actions_frame = ttk.Frame(self.btn_run_detection_frame)
    self.detection_review_actions_frame.pack(side=tk.LEFT, anchor=tk.CENTER, padx=(8, 0))
    self.btn_undo_detection_result = ttk.Button(
        self.detection_review_actions_frame,
        text="Cofnij detekcję",
        command=self._undo_last_detection_result,
        style="WorkflowCard.TButton",
    )
    self.btn_undo_detection_result.pack(side=tk.LEFT)
    self.btn_undo_detection_result.configure(padding=detection_action_button_padding, width=16, state=tk.DISABLED)
    self.btn_confirm_detection_result = None

    self.detect_run_status_frame = ttk.Frame(self.detect_actions_row)
    self.detect_run_status_frame.grid_rowconfigure(0, weight=0)
    self.detect_run_status_frame.grid_columnconfigure(0, weight=1)
    self.detect_run_status_frame.grid_columnconfigure(1, weight=0)

    self.detect_run_info_stack = ttk.Frame(self.detect_run_status_frame)
    self.detect_run_info_stack.grid(row=0, column=0, sticky="ew")

    self.test_status_lbl = tk.Label(
        self.detect_run_status_frame,
        text="Gotowy do testów",
        anchor="w",
        justify="left",
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0
    )
    self.test_status_lbl._inline_status_font = ("Segoe UI", 8)
    self._set_inline_status_label_state(
        self.test_status_lbl,
        text="Tryb pracy: OCR | gotowa do uruchomienia",
        tone="neutral",
        emphasis=False
    )
    self.test_status_lbl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
    self.test_status_lbl.grid_remove()

    manual_guard_text = "Ochrona manuali: detekcja uzupełnia lub odświeża tylko pozostałe znaki."
    self.detect_manual_guard_lbl = tk.Label(
        self.detect_run_status_frame,
        text=manual_guard_text,
        anchor="w",
        justify="left",
        wraplength=0,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
    )
    self.detect_manual_guard_lbl._inline_status_font = ("Segoe UI", 8)
    self._set_inline_status_label_state(
        self.detect_manual_guard_lbl,
        text=manual_guard_text,
        tone="muted",
        emphasis=False,
    )

    refiner_guard_text = "Refiner perfect: niedostępny w pipeline OCR. Poprawki boxów wymagają modelu YOLO."
    self.detect_refiner_guard_lbl = tk.Label(
        self.detect_run_status_frame,
        text=refiner_guard_text,
        anchor="w",
        justify="left",
        wraplength=0,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
    )
    self.detect_refiner_guard_lbl._inline_status_font = ("Segoe UI", 8)
    self._set_inline_status_label_state(
        self.detect_refiner_guard_lbl,
        text=refiner_guard_text,
        tone="muted",
        emphasis=False,
    )

    self.detect_last_run_lbl = tk.Label(
        self.detect_run_info_stack,
        text="Ostatnia detekcja: brak zapisanego przebiegu",
        anchor="w",
        justify="left",
        wraplength=0,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
    )
    self.detect_last_run_lbl._inline_status_font = ("Segoe UI", 8)
    self.detect_last_run_lbl.pack(anchor=tk.W, fill=tk.X, pady=0, ipady=0)
    self.detect_last_run_lbl.pack_forget()
    self._refresh_last_detection_status_label()
    self._refresh_detection_refiner_guard_label()
    self._refresh_detection_review_controls()

    self.test_progress_row = tk.Frame(
        self.detect_run_status_frame,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.test_progress = SlimProgressBar(
        self.test_progress_row,
        maximum=100,
        value=0,
        thickness=2,
        trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
        fill_color=palette.get("success", "#2ecc71"),
        bg=palette.get("panel", "#252526"),
        width=240,
        height=6,
    )
    self.test_progress.pack(side=tk.LEFT)

    self.test_progress_count_lbl = tk.Label(
        self.test_progress_row,
        text="",
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.test_progress_count_lbl.pack(side=tk.LEFT, padx=(6, 0))
    self._set_inline_status_label_state(self.test_progress_count_lbl, text="", tone="muted", emphasis=True)

    self.footer_test_status_lbl = tk.Label(
        footer_nav,
        text="Gotowy do testów",
        anchor="w",
        justify="left",
        wraplength=460,
        bd=0,
        highlightthickness=0
    )
    self.footer_test_status_lbl.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    self._set_inline_status_label_state(self.footer_test_status_lbl, text="Gotowy do testów", tone="neutral", emphasis=True)
    self.footer_test_status_lbl.grid_remove()

    self.detect_nav_divider = tk.Frame(
        footer_nav,
        height=1,
        bg=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        bd=0,
        highlightthickness=0,
    )
    self.detect_nav_divider.grid(row=2, column=0, sticky="ew", pady=(2, 2))

    self.btn_toggle_detection_log = ttk.Button(
        self.detect_actions_row,
        text="Pokaż terminal",
        command=self._toggle_detection_process_log,
        style="WorkflowCard.TButton"
    )
    self.btn_toggle_detection_log.configure(text="Terminal", padding=(6, 1))

    self.btn_to_dataset_frame = tk.Frame(self.detect_nav_row, bd=0, highlightthickness=0)
    self.btn_to_dataset_frame.grid(row=0, column=2, sticky="se", padx=(10, 0))

    self.btn_to_dataset_pulse_frame = tk.Frame(
        self.btn_to_dataset_frame,
        bd=0,
        highlightthickness=0
    )
    self.btn_to_dataset_pulse_frame.pack(anchor=tk.SE)

    self.btn_to_dataset = ttk.Button(
        self.btn_to_dataset_pulse_frame,
        text="Dalej → Integracje i dataset",
        command=self.go_to_substep_3,
        state=tk.DISABLED,
        style="WorkflowCardPrimary.TButton"
    )
    self.btn_to_dataset.pack(anchor=tk.SE)
    self.btn_to_dataset.configure(text="Krok 2: dataset PZ3", padding=(6, 2, 6, 0), width=NAV_BUTTON_WIDTH)

    # =========================
    # HELP BINDS
    # =========================
    HELP.bind_help(self.btn_run_detection, "t2_fast_test")
    HELP.bind_help(self.btn_rank_presets, "t2_rank")
    HELP.bind_help(detect_mode_cards_frame, "t2_method")
    HELP.bind_help(self.btn_ocr_lab, "t2_lab_btn")
    HELP.bind_help(self.yolo_model_row, "t2_yolo_model")
    HELP.bind_help(self.yolo_tuning_lf, "t2_yolo_tuning")
    HELP.bind_help(self.hybrid_yolo_box_backend_row, "t2_yolo_box_backend")
    HELP.bind_help(self.hybrid_yolo_box_backend_indicator, "t2_yolo_box_backend")
    HELP.bind_help(self.hybrid_yolo_box_backend_lbl, "t2_yolo_box_backend")
    det_device_combo = getattr(self, "det_device_combo", None)
    det_device_hint_lbl = getattr(self, "det_device_hint_lbl", None)
    HELP.bind_help(self.btn_toggle_detection_log, "t2_cut_logs")
    HELP.bind_help(self.plates_listbox, "t2_listbox")
    HELP.bind_help(self.detect_left_title_lbl, "t2_canvas")
    HELP.bind_help(self.detect_left_intro_lbl, "t2_canvas")
    HELP.bind_help(self.preview_list_header_lbl, "t2_canvas")
    HELP.bind_help(self.preview_list_intro_lbl, "t2_canvas")
    HELP.bind_help(self.preview_title_lbl, "t2_canvas")
    HELP.bind_help(self.preview_intro_lbl, "t2_canvas")
    HELP.bind_help(self.preview_focus_prompt_shell, "t2_canvas")
    HELP.bind_help(self.preview_canvas, "t2_canvas")
    HELP.bind_help(det_device_combo, "t2_device")
    HELP.bind_help(det_device_hint_lbl, "t2_device")
    HELP.bind_help(self.preview_dir_hint_lbl, "t2_preview_run")
    HELP.bind_help(self.preview_source_status_lbl, "t2_preview_run")
    HELP.bind_help(self.preview_source_detail_lbl, "t2_preview_run")
    HELP.bind_help(self.preview_mode_overlay, "t2_preview_mode")
    HELP.bind_help(self.preview_mode_eye_btn, "t2_preview_mode")
    HELP.bind_help(self.preview_edit_toolbar, "t2_canvas")
    HELP.bind_help(self.preview_controls_canvas, "t2_canvas")
    HELP.bind_help(self.preview_prev_btn, "t2_preview_nav")
    HELP.bind_help(self.preview_next_btn, "t2_preview_nav")
    HELP.bind_help(self.preview_fit_btn, "t2_preview_nav")
    HELP.bind_help(self.preview_fullscreen_btn, "t2_preview_nav")
    HELP.bind_help(self.preview_edit_toggle_btn, "t2_preview_edit")
    HELP.bind_help(self.preview_add_box_btn, "t2_preview_edit")
    HELP.bind_help(self.preview_edit_char_btn, "t2_preview_edit")
    HELP.bind_help(self.preview_delete_char_btn, "t2_preview_edit")
    HELP.bind_help(self.preview_box_mode_toggle_btn, "t2_preview_mode")
    HELP.bind_help(self.preview_load_note_lbl, "t2_preview_info")
    HELP.bind_help(self.btn_to_dataset_frame, "t2_to_dataset")
    HELP.bind_help(self.btn_to_dataset, "t2_to_dataset")

    for entry in getattr(self, "_detect_mode_cards", {}).values():
        ensure_self_adaptive_wrap(entry.get("desc"), padding=6, min_wrap=130)

    ensure_self_adaptive_wrap(getattr(self, "detect_active_model_lbl", None), padding=6, min_wrap=160)
    ensure_self_adaptive_wrap(getattr(self, "yolo_model_status_lbl", None), padding=6, min_wrap=120)
    ensure_self_adaptive_wrap(getattr(self, "detect_run_model_info_lbl", None), padding=6, min_wrap=220)
    ensure_self_adaptive_wrap(getattr(self, "preview_dir_hint_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "preview_load_note_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "preview_record_source_lbl", None), padding=6, min_wrap=180)
    ensure_self_adaptive_wrap(getattr(self, "preview_layout_summary_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "preview_fusion_info_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "preview_box_mode_info_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "preview_import_focus_hint_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "winner_name_lbl", None), padding=6, min_wrap=140)
    ensure_self_adaptive_wrap(getattr(self, "winner_acc_lbl", None), padding=6, min_wrap=140)
    defer_adaptive_wrap_descendants(self.hybrid_rescue_frame, padding=24, min_wrap=160, delay_ms=1400)
    defer_adaptive_wrap_descendants(self.ocr_summary_store, padding=24, min_wrap=160, delay_ms=1400)
    try:
        self.frame.after(1400, self._schedule_detect_right_adaptive_wrap_refresh)
    except Exception:
        self._schedule_detect_right_adaptive_wrap_refresh()

    self._refresh_detect_mode_cards()
    self._refresh_detection_advanced_sections()
    self._update_yolo_visibility()

    def _deferred_bind_detect_right_scroll_children():
        try:
            if not bool(self.detect_right_content.winfo_exists()):
                return
        except Exception:
            return
        self._bind_scroll_canvas_children(
            self.detect_right_content,
            self.detect_right_canvas,
            self._detect_right_canvas_overflows
        )
        self._bind_scroll_canvas_children(
            self.detect_right_pinned_status_host,
            self.detect_right_canvas,
            self._detect_right_canvas_overflows
        )

    try:
        self.frame.after_idle(_deferred_bind_detect_right_scroll_children)
    except Exception:
        _deferred_bind_detect_right_scroll_children()
    try:
        self.frame.after(900, _deferred_bind_detect_right_scroll_children)
    except Exception:
        pass
    _mark_build_phase("help_and_adaptive")
    self.frame.after_idle(self._sync_detect_right_scrollregion)
    self.frame.after_idle(self._sync_detect_right_canvas_width)
    self.frame.after_idle(self._init_preview_vertical_split)
    _mark_build_phase("final_bindings", threshold_ms=80.0)

    if build_profile_marks:
        try:
            logger.info(
                "[Z3/PZ2 UI build phases] %s total=%.1fms",
                " ".join(build_profile_marks),
                (time.perf_counter() - build_profile_start) * 1000.0,
            )
        except Exception:
            pass
