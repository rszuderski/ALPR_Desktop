#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main Z2 widget builder extracted from AnnotationTab."""

import tkinter as tk
import time
from tkinter import ttk
from pathlib import Path

from ..config import CONFIG
from ..config import logger
from .canvas_progress_overlay import CanvasProgressOverlay
from .help_manager import HELP
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z2_main_widget_bindings import bind_annotation_widget_help_and_events
from .z2_right_panel_widgets import build_annotation_right_panel


def create_annotation_widgets(host, SlimProgressBar, nav_button_width):
    self = host
    _build_started_at = time.perf_counter()
    _phase_started_at = _build_started_at

    def _log_build_phase(name: str) -> None:
        nonlocal _phase_started_at
        try:
            now = time.perf_counter()
            elapsed_ms = (now - _phase_started_at) * 1000.0
            total_ms = (now - _build_started_at) * 1000.0
            if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                logger.info(
                    "[Z2 PERF] create_annotation_widgets "
                    f"{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                )
            _phase_started_at = now
        except Exception:
            pass

    NAV_BUTTON_WIDTH = nav_button_width
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    try:
        from ..campaign_manager import CAMPAIGN

        initial_campaign_context = bool(CAMPAIGN.get_active_project_name()) and not bool(
            getattr(self.app, "campaign_free_mode", False)
        )
    except Exception:
        initial_campaign_context = False

    pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL, cursor="arrow")
    pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
    self.main_pane = pane

    left_frame = ttk.Frame(pane, style="Panel.TFrame")
    center_frame = ttk.Frame(pane, style="Panel.TFrame")
    right_frame = ttk.Frame(pane, style="Panel.TFrame")
    self.main_left_frame = left_frame
    self.main_center_frame = center_frame
    self.main_right_frame = right_frame
    self._annotation_right_panel_visible = True

    pane.add(left_frame, weight=2)
    pane.add(center_frame, weight=6)
    pane.add(right_frame, weight=1)
    pane.bind("<Configure>", self._on_main_pane_configure, add="+")
    pane.bind("<ButtonPress-1>", self._on_main_pane_button_press, add="+")
    pane.bind("<B1-Motion>", self._on_main_pane_drag_motion, add="+")
    pane.bind("<ButtonRelease-1>", self._on_main_pane_drag_release, add="+")

    # --- LEWA KOLUMNA ---
    left_frame.grid_columnconfigure(0, weight=1)
    left_frame.grid_rowconfigure(0, weight=0)
    left_frame.grid_rowconfigure(1, weight=1)
    self.left_column_pane = None

    left_scroll_shell = tk.Frame(
        left_frame,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=panel_border,
        highlightcolor=panel_border,
    )
    left_scroll_shell.grid(row=0, column=0, sticky="nsew")
    self.left_scroll_shell = left_scroll_shell

    left_scroll_host = ttk.Frame(left_scroll_shell, style="Panel.TFrame")
    left_scroll_host.pack(fill=tk.BOTH, expand=True)
    self.left_scroll_host = left_scroll_host

    self.left_settings_canvas = tk.Canvas(left_scroll_host, bg=panel_bg, highlightthickness=0, bd=0)
    self.left_settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    self.left_settings_scrollbar = WebSlimScrollbar(
        left_scroll_host,
        command=self.left_settings_canvas.yview
    )
    self.left_settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.left_settings_canvas.configure(yscrollcommand=self.left_settings_scrollbar.set)

    self.left_settings_content = ttk.Frame(self.left_settings_canvas, style="Panel.TFrame")
    self._left_settings_window_id = self.left_settings_canvas.create_window(
        (0, 0),
        window=self.left_settings_content,
        anchor="nw"
    )
    self.left_settings_content.bind("<Configure>", self._sync_left_panel_scrollregion)
    self.left_settings_canvas.bind("<Configure>", self._sync_left_panel_canvas_width)
    _log_build_phase("base_panes")

    self.preview_left_list_shell = tk.Frame(
        left_frame,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=panel_border,
        highlightcolor=panel_border,
    )
    self.preview_left_list_resize_grip = None
    self.preview_left_list_resize_grip_line = None
    self.preview_left_list_host = ttk.Frame(self.preview_left_list_shell, style="Panel.TFrame")
    self.preview_left_list_host.pack(fill=tk.BOTH, expand=True)
    self.preview_left_list_shell.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    settings_col = ttk.Frame(self.left_settings_content, style="Panel.TFrame")
    settings_col.pack(fill=tk.X, expand=False, padx=12, pady=(14, 20))
    self.left_settings_col = settings_col

    workflow_shell_border = blend_hex_colors(
        palette.get("accent", "#4f8de3"),
        panel_border,
        0.62,
    )
    workflow_shell_fill = blend_hex_colors(
        palette.get("surface_info", panel_bg),
        panel_bg,
        0.80,
    )
    self.workflow_entry_shell = tk.Frame(
        settings_col,
        bg=workflow_shell_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
    )
    if not initial_campaign_context:
        self.workflow_entry_shell.pack(fill=tk.X, pady=(0, 10))

    self.workflow_entry_shell_inner = tk.Frame(
        self.workflow_entry_shell,
        bg=workflow_shell_fill,
        bd=0,
        highlightthickness=0,
        padx=3,
        pady=5,
    )
    self.workflow_entry_shell_inner.pack(fill=tk.X, expand=False)

    self.workflow_entry_section = ttk.Frame(self.workflow_entry_shell_inner, style="Panel.TFrame")
    self.workflow_entry_section.pack(fill=tk.X)
    self.workflow_entry_title_lbl = None
    self.workflow_intro_lbl = None
    self.workflow_cards_frame = None
    self.auto_route_card = None
    self.auto_route_card_title = None
    self.auto_route_card_desc = None
    self.manual_route_card = None
    self.manual_route_card_title = None
    self.manual_route_card_desc = None

    if not initial_campaign_context:
        self.workflow_entry_title_lbl = SectionHeaderLabel(
            self.workflow_entry_section,
            self.app,
            text="Co chcesz zrobic?",
        )
        self.workflow_entry_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        self.workflow_intro_lbl = tk.Label(
            self.workflow_entry_section,
            textvariable=self.workflow_intro_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=360,
            bd=0,
            highlightthickness=0
        )
        self.workflow_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        self._set_inline_label_state(self.workflow_intro_lbl, tone="muted", emphasis=False)

        self.workflow_cards_frame = ttk.Frame(self.workflow_entry_section, style="Panel.TFrame")
        self.workflow_cards_frame.pack(fill=tk.X)

        palette = getattr(self.app, "palette", {})
        workflow_card_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
        workflow_card_border = blend_hex_colors(
            palette.get("success", "#4ec9b0"),
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            0.18,
        )
        workflow_card_fg = palette.get("fg", "#f3f3f3")
        workflow_card_muted = palette.get("muted", "#c7c7c7")

        self.auto_route_card = tk.Frame(
        self.workflow_cards_frame,
        bd=0,
        highlightthickness=1,
        highlightbackground=workflow_card_border,
        highlightcolor=workflow_card_border,
        bg=workflow_card_bg,
        padx=14,
        pady=12,
        cursor="hand2",
    )
        self.auto_route_card.pack(fill=tk.X, pady=(0, 8))
        self.auto_route_card_title = tk.Label(
        self.auto_route_card,
        text="Autoanotacja tablic",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        cursor="hand2",
        bd=0,
        highlightthickness=0,
        bg=workflow_card_bg,
        fg=workflow_card_fg,
    )
        self.auto_route_card_title.pack(anchor=tk.W, fill=tk.X)
        self.auto_route_card_desc = tk.Label(
        self.auto_route_card,
        text="Uruchom YOLO, zapisz run anotacji Z2 w workspace i przejdź potem do korekty oraz splitu.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=336,
        cursor="hand2",
        bd=0,
        highlightthickness=0,
        bg=workflow_card_bg,
        fg=workflow_card_muted,
    )
        self.auto_route_card_desc.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

        self.manual_route_card = tk.Frame(
        self.workflow_cards_frame,
        bd=0,
        highlightthickness=1,
        highlightbackground=workflow_card_border,
        highlightcolor=workflow_card_border,
        bg=workflow_card_bg,
        padx=14,
        pady=12,
        cursor="hand2",
    )
        self.manual_route_card.pack(fill=tk.X)
        self.manual_route_card_title = tk.Label(
        self.manual_route_card,
        text="Praca ręczna na runie Z2",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        cursor="hand2",
        bd=0,
        highlightthickness=0,
        bg=workflow_card_bg,
        fg=workflow_card_fg,
    )
        self.manual_route_card_title.pack(anchor=tk.W, fill=tk.X)
        self.manual_route_card_desc = tk.Label(
        self.manual_route_card,
        text="Utwórz nowy run ręczny, otwórz lokalny run z historii albo wskaż dowolny run Z2 do korekty.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=336,
        cursor="hand2",
        bd=0,
        highlightthickness=0,
        bg=workflow_card_bg,
        fg=workflow_card_muted,
    )
        self.manual_route_card_desc.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

        self._bind_workflow_card(self.auto_route_card, "auto")
        self._bind_workflow_card(self.manual_route_card, "manual")
        self._refresh_workflow_route_cards(refresh_content=True)
    _log_build_phase("workflow_cards")
    self.workflow_entry_separator = self._build_left_section_separator(settings_col, pady=(16, 20))
    _log_build_phase("workflow_entry")

    workflow_nav_panel = ttk.Frame(self.workflow_entry_shell_inner, style="Panel.TFrame")
    workflow_nav_panel.pack(fill=tk.X, pady=(12, 0))
    self.workflow_nav_panel = workflow_nav_panel

    self.workflow_progress_canvas = tk.Canvas(
        workflow_nav_panel,
        height=68,
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
    )
    self.workflow_progress_canvas.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
    self.workflow_progress_canvas.bind(
        "<Configure>",
        lambda _e: self._refresh_z2_miniflow_progress(),
        add="+",
    )

    self.workflow_nav_bottom_panel = ttk.Frame(self.workflow_entry_shell_inner, style="Panel.TFrame")

    source_section = ttk.Frame(settings_col, style="Panel.TFrame")
    self.source_section = source_section
    if not initial_campaign_context:
        source_section.pack(fill=tk.X)
    self.sources_title_lbl = SectionHeaderLabel(
        source_section,
        self.app,
        text="1. Wejście i zapis runu anotacji Z2",
    )
    self.sources_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.input_dir_title_lbl = ttk.Label(
        source_section,
        text="Folder obrazów wejściowych",
        style="Panel.TLabel"
    )
    self.input_dir_title_lbl.pack(anchor=tk.W, fill=tk.X)
    row_in, self.input_dir_entry, self.input_dir_browse_btn = self._build_left_path_row(
        source_section,
        self.input_dir_var,
        button_text="Wybierz obrazy",
        button_command=self._select_input_dir,
    )

    self.project_paths_info_lbl = tk.Label(
        source_section,
        textvariable=self.project_paths_info_var,
        font=("Segoe UI", 9),
        wraplength=360,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.project_paths_info_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
    self._set_inline_label_state(self.project_paths_info_lbl, tone="muted", emphasis=False)

    self.project_paths_rel_lbl = tk.Label(
        source_section,
        textvariable=self.project_paths_rel_var,
        wraplength=360,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0
    )
    self.project_paths_rel_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 10))
    self._set_inline_label_state(self.project_paths_rel_lbl, tone="muted", emphasis=False)

    self.output_dir_title_lbl = ttk.Label(
        source_section,
        text="Folder, w którym Z2 zapisuje runy anotacji",
        style="Panel.TLabel"
    )
    self.output_dir_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.output_dir_var.set(str(Path(CONFIG.get_auto_annotations_dir("plate"))))
    row_out, self.output_dir_entry, _ = self._build_left_path_row(
        source_section,
        self.output_dir_var,
        state="readonly",
    )
    self._enable_compact_path_entry(
        self.output_dir_entry,
        self.output_dir_var,
        title="Pełna ścieżka katalogu runów anotacji Z2",
    )

    self.source_section_separator = self._build_left_section_separator(settings_col, pady=(16, 20))
    if initial_campaign_context:
        try:
            self.source_section_separator.pack_forget()
        except Exception:
            pass

    actions_lf = ttk.Frame(self.workflow_entry_shell_inner, style="Panel.TFrame")
    self.actions_section = actions_lf
    actions_lf.pack(fill=tk.X)
    self.run_title_lbl = SectionHeaderLabel(
        actions_lf,
        self.app,
        text="2. Wybierz tor i uruchom Z2",
    )
    self.run_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.run_intro_lbl = tk.Label(
        actions_lf,
        textvariable=self.run_intro_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.run_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.run_intro_lbl, tone="muted", emphasis=False)

    self.route_badge_lbl = tk.Label(
        actions_lf,
        textvariable=self.route_badge_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.route_badge_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
    self._set_inline_label_state(self.route_badge_lbl, tone="success", emphasis=True)

    self.route_summary_lbl = tk.Label(
        actions_lf,
        textvariable=self.route_summary_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.route_summary_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.route_summary_lbl, tone="muted", emphasis=False)

    self.workflow_action_hint_lbl = tk.Label(
        actions_lf,
        textvariable=self.workflow_action_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.workflow_action_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.workflow_action_hint_lbl, tone="muted", emphasis=False)

    self.return_to_campaign_btn = ttk.Button(
        actions_lf,
        text="Wróć do grafu",
        style="WorkflowCard.TButton",
        command=self._return_to_campaign_wizard,
    )
    self.return_to_campaign_btn.configure(padding=(6, 1))

    self.route_selector_frame = ttk.Frame(actions_lf, style="Panel.TFrame")
    self.route_selector_frame.pack(fill=tk.X, pady=(2, 2))

    self.auto_route_radio = ttk.Radiobutton(
        self.route_selector_frame,
        text="Autoanotacja runu anotacji Z2",
        variable=self.manual_xml_template_var,
        value=False,
        command=self._update_manual_xml_template_ui
    )
    self.auto_route_radio.pack(anchor=tk.W)

    self.manual_route_radio = ttk.Radiobutton(
        self.route_selector_frame,
        text="Ręczna anotacja tablic",
        variable=self.manual_xml_template_var,
        value=True,
        command=self._update_manual_xml_template_ui
    )
    self.manual_route_radio.pack(anchor=tk.W, pady=(2, 0))

    self.manual_vehicle_assist_check = ttk.Checkbutton(
        self.route_selector_frame,
        text="Dodaj tylko boxy pojazdów jako pomoc",
        variable=self.manual_vehicle_assist_var,
        command=self._update_manual_xml_template_ui
    )

    self.manual_vehicle_assist_hint_lbl = tk.Label(
        self.route_selector_frame,
        textvariable=self.manual_vehicle_assist_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        bd=0,
        highlightthickness=0
    )

    self.manual_xml_template_hint_lbl = tk.Label(
        actions_lf,
        textvariable=self.manual_xml_template_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.manual_xml_template_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    self.auto_plate_model_section = self._build_workflow_step_card(actions_lf)
    self.auto_plate_model_title_lbl = tk.Label(
        self.auto_plate_model_section,
        text="1. Wskaz model tablic (YOLO Pose)",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.auto_plate_model_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.auto_plate_model_hint_lbl = tk.Label(
        self.auto_plate_model_section,
        text="Ten model jest wymagany, aby uruchomić autoanotację tablic.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.auto_plate_model_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.auto_plate_model_hint_lbl, tone="muted", emphasis=False)
    (
        self.workflow_plate_path_row,
        self.workflow_plate_path_entry,
        self.workflow_plate_browse_btn,
    ) = self._build_left_path_row(
        self.auto_plate_model_section,
        self.plate_custom_var,
        button_text="Wybierz",
        button_command=self._select_plate_custom,
    )
    self.workflow_plate_model_info_box = tk.Frame(
        self.auto_plate_model_section,
        bd=0,
        highlightthickness=1,
        padx=10,
        pady=8,
    )
    self.workflow_plate_model_info_title_lbl = tk.Label(
        self.workflow_plate_model_info_box,
        text="Model projektu i bieżącego runu",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_plate_model_info_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.workflow_plate_model_info_var = tk.StringVar(value="")
    self.workflow_plate_model_info_lbl = tk.Label(
        self.workflow_plate_model_info_box,
        textvariable=self.workflow_plate_model_info_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_plate_model_info_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
    self.workflow_plate_path_row.configure(style="WorkflowPlatePath.TFrame")
    self._register_workflow_step_card(
        "auto_plate_model",
        self.auto_plate_model_section,
        title=self.auto_plate_model_title_lbl,
        labels=[self.auto_plate_model_hint_lbl, self.workflow_plate_model_info_title_lbl, self.workflow_plate_model_info_lbl],
        child_frames=[self.workflow_plate_model_info_box],
        style_targets=[
            {"widget": self.workflow_plate_path_row, "kind": "frame", "style": "WorkflowPlatePath.TFrame"},
            {"widget": self.workflow_plate_path_entry, "kind": "entry", "style": "WorkflowPlatePath.TEntry"},
        ],
        step_keys={"auto_plate_model"},
    )

    self.workflow_conf_section = self._build_workflow_step_card(actions_lf)
    self.workflow_conf_title_lbl = tk.Label(
        self.workflow_conf_section,
        textvariable=self.workflow_conf_title_var,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_conf_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.workflow_conf_hint_lbl = tk.Label(
        self.workflow_conf_section,
        textvariable=self.workflow_conf_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_conf_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.workflow_conf_hint_lbl, tone="muted", emphasis=False)
    self.workflow_conf_row = tk.Frame(self.workflow_conf_section, bd=0, highlightthickness=0)
    self.workflow_conf_row.pack(fill=tk.X)
    self.workflow_conf_scale = ttk.Scale(
        self.workflow_conf_row,
        from_=0.1,
        to=0.9,
        variable=self.conf_var,
        orient=tk.HORIZONTAL,
        style="WorkflowConf.Horizontal.TScale",
    )
    self.workflow_conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.workflow_conf_value_lbl = tk.Label(
        self.workflow_conf_row,
        width=4,
        anchor="e",
        justify=tk.RIGHT,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_conf_value_lbl.pack(side=tk.RIGHT, padx=(5, 0))
    self.conf_var.trace_add("write", lambda *a: self._refresh_confidence_value_labels())
    self._refresh_confidence_value_labels()
    self._register_workflow_step_card(
        "workflow_conf",
        self.workflow_conf_section,
        title=self.workflow_conf_title_lbl,
        labels=[self.workflow_conf_hint_lbl, self.workflow_conf_value_lbl],
        child_frames=[self.workflow_conf_row],
        style_targets=[
            {"widget": self.workflow_conf_scale, "kind": "scale", "style": "WorkflowConf.Horizontal.TScale"},
        ],
        step_keys={"auto_conf", "manual_conf"},
    )

    self.auto_vehicle_choice_section = self._build_workflow_step_card(actions_lf)
    self.auto_vehicle_choice_title_lbl = tk.Label(
        self.auto_vehicle_choice_section,
        text="Czy dodać model pojazdów (YOLO Box)?",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.auto_vehicle_choice_title_lbl.pack(anchor=tk.W, fill=tk.X)

    self.auto_vehicle_choice_row = tk.Frame(self.auto_vehicle_choice_section, bd=0, highlightthickness=0)
    self.auto_vehicle_choice_row.columnconfigure(0, weight=1)
    self.auto_vehicle_choice_skip_check = ttk.Checkbutton(
        self.auto_vehicle_choice_row,
        text="Pomijam model pojazdów i wykrywam tylko tablice",
        style="WorkflowAutoVehicleChoice.TCheckbutton",
        variable=self.auto_vehicle_choice_var,
        onvalue="skip",
        offvalue="use",
        command=self._on_auto_vehicle_skip_toggle,
    )
    self.auto_vehicle_choice_skip_check.grid(row=0, column=0, sticky="w")
    self.auto_vehicle_choice_row.pack(fill=tk.X, pady=(6, 0))

    self.auto_vehicle_choice_hint_lbl = tk.Label(
        self.auto_vehicle_choice_section,
        textvariable=self.auto_vehicle_choice_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.auto_vehicle_choice_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self._set_inline_label_state(self.auto_vehicle_choice_hint_lbl, tone="muted", emphasis=False)
    self._register_workflow_step_card(
        "auto_vehicle_choice",
        self.auto_vehicle_choice_section,
        title=self.auto_vehicle_choice_title_lbl,
        labels=[self.auto_vehicle_choice_hint_lbl],
        child_frames=[self.auto_vehicle_choice_row],
        style_targets=[
            {
                "widget": self.auto_vehicle_choice_skip_check,
                "kind": "checkbutton",
                "style": "WorkflowAutoVehicleChoice.TCheckbutton",
            },
        ],
        step_keys={"auto_vehicle_choice"},
    )

    self.workflow_vehicle_model_section = self._build_workflow_step_card(actions_lf)
    self.workflow_vehicle_model_title_lbl = tk.Label(
        self.workflow_vehicle_model_section,
        textvariable=self.workflow_vehicle_model_title_var,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_vehicle_model_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.workflow_vehicle_model_hint_lbl = tk.Label(
        self.workflow_vehicle_model_section,
        textvariable=self.workflow_vehicle_model_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_vehicle_model_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.workflow_vehicle_model_hint_lbl, tone="muted", emphasis=False)
    self.workflow_vehicle_combo = ttk.Combobox(
        self.workflow_vehicle_model_section,
        textvariable=self.vehicle_model_var,
        style="WorkflowVehicleModel.TCombobox",
        state="readonly",
    )
    self.workflow_vehicle_combo.pack(fill=tk.X, pady=(0, 2))
    self.workflow_vehicle_combo.bind("<<ComboboxSelected>>", self._on_vehicle_model_change)
    self.workflow_vehicle_custom_row = tk.Frame(self.workflow_vehicle_model_section, bd=0, highlightthickness=0)
    self.workflow_vehicle_custom_entry = ttk.Entry(
        self.workflow_vehicle_custom_row,
        textvariable=self.vehicle_custom_var,
        style="WorkflowVehicleCustom.TEntry",
    )
    self.workflow_vehicle_custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.workflow_vehicle_custom_browse_btn = ttk.Button(
        self.workflow_vehicle_custom_row,
        text="Wybierz",
        style="WorkflowCard.TButton",
        command=self._select_vehicle_custom,
    )
    self.workflow_vehicle_custom_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
    self._register_workflow_step_card(
        "workflow_vehicle_model",
        self.workflow_vehicle_model_section,
        title=self.workflow_vehicle_model_title_lbl,
        labels=[self.workflow_vehicle_model_hint_lbl],
        child_frames=[self.workflow_vehicle_custom_row],
        style_targets=[
            {"widget": self.workflow_vehicle_combo, "kind": "combobox", "style": "WorkflowVehicleModel.TCombobox"},
            {"widget": self.workflow_vehicle_custom_entry, "kind": "entry", "style": "WorkflowVehicleCustom.TEntry"},
        ],
        step_keys={"auto_vehicle_model", "manual_vehicle_model"},
    )

    self.manual_entry_section = self._build_workflow_step_card(actions_lf)
    self.manual_entry_mode_row = tk.Frame(self.manual_entry_section, bd=0, highlightthickness=0)
    self.manual_entry_mode_row.columnconfigure(0, weight=1)
    self.manual_entry_title_lbl = tk.Label(
        self.manual_entry_section,
        textvariable=self.manual_entry_title_var,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.manual_entry_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.manual_entry_new_radio = ttk.Radiobutton(
        self.manual_entry_mode_row,
        text="Utwórz nowy run ręczny",
        style="WorkflowManualEntry.TRadiobutton",
        value="new",
        variable=self.manual_entry_mode_var,
        command=self._on_manual_entry_mode_change,
    )
    self.manual_entry_new_radio.grid(row=0, column=0, sticky="w")
    self.manual_entry_continue_radio = ttk.Radiobutton(
        self.manual_entry_mode_row,
        text="Otwórz run z historii",
        style="WorkflowManualEntry.TRadiobutton",
        value="continue",
        variable=self.manual_entry_mode_var,
        command=self._on_manual_entry_mode_change,
    )
    self.manual_entry_continue_radio.grid(row=1, column=0, sticky="w", pady=(4, 0))
    self.manual_entry_import_radio = ttk.Radiobutton(
        self.manual_entry_mode_row,
        text="Wskaż dowolny run Z2",
        style="WorkflowManualEntry.TRadiobutton",
        value="import",
        variable=self.manual_entry_mode_var,
        command=self._on_manual_entry_mode_change,
    )
    self.manual_entry_import_radio.grid(row=2, column=0, sticky="w", pady=(4, 0))
    self.manual_entry_mode_row.pack(fill=tk.X, pady=(6, 0))

    self.manual_entry_hint_lbl = tk.Label(
        self.manual_entry_section,
        textvariable=self.manual_entry_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.manual_entry_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self._set_inline_label_state(self.manual_entry_hint_lbl, tone="muted", emphasis=False)
    self._register_workflow_step_card(
        "manual_entry",
        self.manual_entry_section,
        title=self.manual_entry_title_lbl,
        labels=[self.manual_entry_hint_lbl],
        child_frames=[self.manual_entry_mode_row],
        style_targets=[
            {
                "widget": self.manual_entry_new_radio,
                "kind": "radiobutton",
                "style": "WorkflowManualEntry.TRadiobutton",
            },
            {
                "widget": self.manual_entry_continue_radio,
                "kind": "radiobutton",
                "style": "WorkflowManualEntry.TRadiobutton",
            },
            {
                "widget": self.manual_entry_import_radio,
                "kind": "radiobutton",
                "style": "WorkflowManualEntry.TRadiobutton",
            },
        ],
        step_keys={"manual_entry"},
    )

    self.manual_history_section = self._build_workflow_step_card(actions_lf)
    self.manual_history_title_lbl = tk.Label(
        self.manual_history_section,
        text="Historia runów Z2",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.manual_history_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.manual_history_combo = ttk.Combobox(
        self.manual_history_section,
        textvariable=self.manual_history_run_var,
        style="WorkflowManualHistory.TCombobox",
        state="readonly",
    )
    self.manual_history_combo.pack(fill=tk.X, pady=(4, 4))
    self.manual_history_combo.bind("<<ComboboxSelected>>", self._on_manual_history_selection_changed)
    self.manual_history_open_btn = ttk.Button(
        self.manual_history_section,
        text="Otwórz run Z2 z historii",
        style="WorkflowCardPrimary.TButton",
        command=self._open_selected_manual_review_history_run,
    )
    self.manual_history_open_btn.pack(fill=tk.X)
    self.manual_history_import_btn = ttk.Button(
        self.manual_history_section,
        text="Wskaż run Z2",
        style="WorkflowCard.TButton",
        command=self._import_or_open_manual_review_run_from_dialog,
    )
    self.manual_history_import_btn.pack(fill=tk.X, pady=(0, 4))
    self.manual_history_hint_lbl = tk.Label(
        self.manual_history_section,
        textvariable=self.manual_history_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.manual_history_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self._set_inline_label_state(self.manual_history_hint_lbl, tone="muted", emphasis=False)
    for widget in (
        self.manual_history_section,
        self.manual_history_title_lbl,
        self.manual_history_combo,
        self.manual_history_open_btn,
        self.manual_history_import_btn,
        self.manual_history_hint_lbl,
    ):
        HELP.bind_help(widget, "tab1_manual_history")
    self._register_workflow_step_card(
        "manual_history",
        self.manual_history_section,
        title=self.manual_history_title_lbl,
        labels=[self.manual_history_hint_lbl],
        style_targets=[
            {"widget": self.manual_history_combo, "kind": "combobox", "style": "WorkflowManualHistory.TCombobox"},
        ],
        step_keys={"manual_history"},
    )

    self.workflow_input_section = self._build_workflow_step_card(actions_lf)
    self.workflow_input_title_lbl = tk.Label(
        self.workflow_input_section,
        text="Folder obrazów",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_input_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.workflow_input_hint_lbl = tk.Label(
        self.workflow_input_section,
        text="Wybierz folder z obrazami, na których ma pracować aktualny tor Z2.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_input_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.workflow_input_hint_lbl, tone="muted", emphasis=False)
    self.workflow_manual_vehicle_assist_check = ttk.Checkbutton(
        self.workflow_input_section,
        text="Opcjonalnie: dodaj boxy pojazdów jako pomoc",
        style="WorkflowManualAssist.TCheckbutton",
        variable=self.manual_vehicle_assist_var,
        command=self._update_manual_xml_template_ui,
    )
    self.workflow_manual_vehicle_assist_hint_lbl = tk.Label(
        self.workflow_input_section,
        textvariable=self.manual_vehicle_assist_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        bd=0,
        highlightthickness=0,
    )
    self._set_inline_label_state(self.workflow_manual_vehicle_assist_hint_lbl, tone="muted", emphasis=False)
    (
        self.workflow_input_row,
        self.workflow_input_entry,
        self.workflow_input_browse_btn,
    ) = self._build_left_path_row(
        self.workflow_input_section,
        self.input_dir_var,
        button_text="Wybierz obrazy",
        button_command=self._select_input_dir,
    )
    self.workflow_input_row.configure(style="WorkflowInputPath.TFrame")

    self.campaign_reuse_manual_check = ttk.Checkbutton(
        self.workflow_input_section,
        text="Dołącz ręcznie anotowane zdjęcia z wcześniejszych iteracji",
        variable=self.campaign_reuse_manual_var,
        command=self._on_campaign_reuse_manual_toggle,
    )
    self.campaign_reuse_manual_hint_lbl = tk.Label(
        self.workflow_input_section,
        textvariable=self.campaign_reuse_manual_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        bd=0,
        highlightthickness=0,
    )
    self._set_inline_label_state(self.campaign_reuse_manual_hint_lbl, tone="warning", emphasis=False)
    self._register_workflow_step_card(
        "workflow_input",
        self.workflow_input_section,
        title=self.workflow_input_title_lbl,
        labels=[
            self.workflow_input_hint_lbl,
            self.workflow_manual_vehicle_assist_hint_lbl,
            self.campaign_reuse_manual_hint_lbl,
        ],
        child_frames=[self.workflow_input_row],
        style_targets=[
            {
                "widget": self.workflow_manual_vehicle_assist_check,
                "kind": "checkbutton",
                "style": "WorkflowManualAssist.TCheckbutton",
            },
            {
                "widget": self.campaign_reuse_manual_check,
                "kind": "checkbutton",
                "style": "WorkflowManualAssist.TCheckbutton",
            },
            {"widget": self.workflow_input_row, "kind": "frame", "style": "WorkflowInputPath.TFrame"},
            {"widget": self.workflow_input_entry, "kind": "entry", "style": "WorkflowInputPath.TEntry"},
        ],
        step_keys={"auto_input", "manual_input"},
    )

    self.workflow_nav_row = ttk.Frame(self.workflow_nav_bottom_panel, style="Panel.TFrame")
    self.workflow_nav_row.columnconfigure(0, weight=0)
    self.workflow_nav_row.columnconfigure(1, weight=1)
    self.workflow_nav_row.pack(fill=tk.X, pady=(2, 2))
    self.workflow_back_btn = ttk.Button(
        self.workflow_nav_row,
        text="Wstecz",
        style="WorkflowCard.TButton",
        command=self._go_to_previous_workflow_step,
    )
    self.workflow_back_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))
    self.workflow_back_btn.configure(padding=(8, 3))
    self.workflow_next_btn = ttk.Button(
        self.workflow_nav_row,
        text="Dalej",
        style="Accent.TButton",
        command=self._go_to_next_workflow_step,
    )
    self.workflow_next_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
    self.workflow_next_btn.configure(padding=(8, 3))

    self.workflow_start_section = self._build_workflow_step_card(actions_lf)
    self.workflow_start_title_lbl = tk.Label(
        self.workflow_start_section,
        text="Uruchom proces Z2",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        bd=0,
        highlightthickness=0,
    )
    self.workflow_start_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.workflow_start_intro_lbl = tk.Label(
        self.workflow_start_section,
        textvariable=self.workflow_start_intro_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_start_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
    self._set_inline_label_state(self.workflow_start_intro_lbl, tone="muted", emphasis=False)
    self._bind_label_wrap_to_container(
        self.workflow_start_intro_lbl,
        self.workflow_start_section,
        padding_px=54,
        min_px=220,
    )

    self.workflow_start_manual_vehicle_assist_check = ttk.Checkbutton(
        self.workflow_start_section,
        text="Opcjonalnie: dodaj pomocnicze boxy pojazdów do XML",
        style="WorkflowManualAssist.TCheckbutton",
        variable=self.manual_vehicle_assist_var,
        command=self._update_manual_xml_template_ui,
    )
    self.workflow_start_manual_vehicle_assist_hint_lbl = tk.Label(
        self.workflow_start_section,
        textvariable=self.manual_vehicle_assist_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        bd=0,
        highlightthickness=0,
    )
    self._set_inline_label_state(self.workflow_start_manual_vehicle_assist_hint_lbl, tone="muted", emphasis=False)

    self.start_btn_row = tk.Frame(self.workflow_start_section, bd=0, highlightthickness=0)
    self.start_btn_row.pack(fill=tk.X, pady=(4, 0))
    self.start_btn_row.columnconfigure(0, weight=5)
    self.start_btn_row.columnconfigure(1, weight=1)

    self.start_btn_frame = tk.Frame(self.start_btn_row, bd=0, highlightthickness=0)
    self.start_btn_frame.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    self.start_btn = ttk.Button(
        self.start_btn_frame,
        text="Wybierz tor",
        command=self._start_annotation,
        style="WorkflowCardPrimary.TButton"
    )
    self.start_btn.pack(fill=tk.X)

    self.stop_btn = ttk.Button(
        self.start_btn_row,
        text="ZATRZYMAJ",
        style="WorkflowCard.TButton",
        command=self._stop_annotation,
        state=tk.DISABLED
    )
    self.stop_btn.grid(row=0, column=1, sticky="ew")
    self.stop_btn.grid_remove()

    self.workflow_start_action_hint_lbl = tk.Label(
        self.workflow_start_section,
        textvariable=self.workflow_action_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    self.workflow_start_action_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self._set_inline_label_state(self.workflow_start_action_hint_lbl, tone="muted", emphasis=False)
    self._bind_label_wrap_to_container(
        self.workflow_start_action_hint_lbl,
        self.workflow_start_section,
        padding_px=54,
        min_px=220,
    )

    progress_info_row = tk.Frame(self.workflow_start_section, bd=0, highlightthickness=0)
    self.progress_info_row = progress_info_row
    progress_info_row.pack(fill=tk.X, pady=(10, 4))
    progress_info_row.pack_forget()

    self.status_label = tk.Label(
        progress_info_row,
        text="Gotowy",
        anchor="w",
        justify=tk.LEFT,
        wraplength=340,
        font=("Segoe UI", 9),
        bd=0,
        highlightthickness=0
    )
    self.status_label.pack(anchor=tk.W, fill=tk.X)
    self._set_inline_label_state(self.status_label, text="Gotowy", tone="neutral", emphasis=True)
    self._bind_label_wrap_to_container(
        self.status_label,
        self.workflow_start_section,
        padding_px=54,
        min_px=220,
    )

    self.progress_counts_lbl = tk.Label(
        progress_info_row,
        textvariable=self.progress_counts_var,
        anchor="e",
        justify=tk.RIGHT,
        font=("Segoe UI", 9),
        bd=0,
        highlightthickness=0
    )
    self.progress_counts_lbl.pack(anchor=tk.E, pady=(2, 0))
    self._set_inline_label_state(self.progress_counts_lbl, tone="muted", emphasis=False)

    self.progress = SlimProgressBar(
        self.workflow_start_section,
        maximum=100,
        value=0,
        thickness=2
    )
    self.progress.pack_forget()
    self._refresh_workflow_progress_style()
    self._set_progress_counters(0, 0, 0)
    self._register_workflow_step_card(
        "workflow_start",
        self.workflow_start_section,
        title=self.workflow_start_title_lbl,
        labels=[
            self.workflow_start_intro_lbl,
            self.workflow_start_manual_vehicle_assist_hint_lbl,
            self.status_label,
            self.progress_counts_lbl,
            self.workflow_start_action_hint_lbl,
        ],
        child_frames=[self.start_btn_row, self.start_btn_frame, self.progress_info_row],
        style_targets=[
            {
                "widget": self.workflow_start_manual_vehicle_assist_check,
                "kind": "checkbutton",
                "style": "WorkflowManualAssist.TCheckbutton",
            },
        ],
        step_keys={"auto_start", "manual_start"},
    )

    self.actions_section_separator = self._build_left_section_separator(self.workflow_entry_shell_inner, pady=(16, 20))
    _log_build_phase("actions_sections")

    self.followup_section = self._build_workflow_step_card(self.workflow_entry_shell_inner)
    self.followup_section.pack(fill=tk.X)
    self.followup_title_lbl = SectionHeaderLabel(
        self.followup_section,
        self.app,
        text="3. Reczne poprawki i iteracje",
    )
    self.followup_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.followup_intro_lbl = tk.Label(
        self.followup_section,
        textvariable=self.followup_intro_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.followup_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_label_state(self.followup_intro_lbl, tone="muted", emphasis=False)

    self.followup_actions_row = ttk.Frame(self.followup_section, style="Panel.TFrame")
    self.followup_actions_row.pack(fill=tk.X, pady=(0, 8))
    self.followup_actions_row.columnconfigure(0, weight=1)
    self.followup_actions_row.columnconfigure(1, weight=1)
    self.enter_manual_review_btn = ttk.Button(
        self.followup_actions_row,
        text="Przegladaj i koryguj",
        style="WorkflowCardPrimary.TButton",
        command=lambda: self._activate_z2_secondary_action("review"),
    )
    self.enter_manual_review_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    self.jump_to_export_btn = ttk.Button(
        self.followup_actions_row,
        text="Utworz dataset",
        style="WorkflowCard.TButton",
        command=lambda: self._activate_z2_secondary_action("export"),
    )
    self.jump_to_export_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    self.run_output_info_lbl = tk.Label(
        self.followup_section,
        textvariable=self.run_output_info_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self._set_inline_label_state(self.run_output_info_lbl, tone="muted", emphasis=False)
    self._bind_full_path_dialog_on_click(
        self.run_output_info_lbl,
        lambda: self._get_preferred_annotation_run_dir(require_xml=True),
        title="Pełna ścieżka runu anotacji Z2",
    )

    self.open_run_dir_btn = ttk.Button(
        self.followup_section,
        text="Otwórz folder runu",
        style="WorkflowCard.TButton",
        command=self._open_current_run_dir,
    )

    self.post_annotation_hint_lbl = tk.Label(
        self.followup_section,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self._register_workflow_step_card(
        "workflow_followup",
        self.followup_section,
        labels=[
            self.followup_intro_lbl,
            self.run_output_info_lbl,
            self.post_annotation_hint_lbl,
        ],
        style_targets=[
            {
                "widget": self.followup_actions_row,
                "kind": "frame",
                "style": "WorkflowFollowupActions.TFrame",
            },
        ],
    )

    self.manual_stage_section = self._build_workflow_step_card(self.workflow_entry_shell_inner)
    self.manual_stage_section.pack(fill=tk.X)
    self.manual_stage_title_lbl = SectionHeaderLabel(
        self.manual_stage_section,
        self.app,
        text="Stage kolejnej iteracji",
    )
    self.manual_stage_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    self.manual_stage_help_lbl = tk.Label(
        self.manual_stage_section,
        text=(
            "Stage to pomocnicza pula zdjec do kolejnej iteracji recznej. "
            "Po eksporcie moga trafia tu nieoznaczone obrazy, a recznie mozesz tez "
            "dolozyc nowy zestaw zdjec bez mieszania z gotowym runem."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.manual_stage_help_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_inline_label_state(self.manual_stage_help_lbl, tone="muted", emphasis=False)
    self._bind_label_wrap_to_container(
        self.manual_stage_help_lbl,
        self.manual_stage_section,
        padding_px=54,
        min_px=220,
    )

    self.manual_stage_path_title_lbl = ttk.Label(
        self.manual_stage_section,
        text="Folder stage (kolejna pula do anotacji):",
        style="Panel.TLabel"
    )
    self.manual_stage_path_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.manual_stage_path_row, self.manual_stage_path_entry, _ = self._build_left_path_row(
        self.manual_stage_section,
        self.manual_stage_dir_var,
        state="readonly",
    )
    self._enable_compact_path_entry(
        self.manual_stage_path_entry,
        self.manual_stage_dir_var,
        title="Pełna ścieżka folderu stage",
    )

    self.manual_stage_status_lbl = tk.Label(
        self.manual_stage_section,
        textvariable=self.manual_stage_status_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.manual_stage_status_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_label_state(self.manual_stage_status_lbl, tone="muted", emphasis=False)
    self._bind_label_wrap_to_container(
        self.manual_stage_status_lbl,
        self.manual_stage_section,
        padding_px=54,
        min_px=220,
    )

    self.manual_stage_buttons_row = ttk.Frame(self.manual_stage_section, style="Panel.TFrame")
    self.manual_stage_buttons_row.pack(fill=tk.X, pady=(0, 2))
    self.manual_stage_buttons_row.columnconfigure(0, weight=1)
    self.manual_stage_buttons_row.columnconfigure(1, weight=1)

    self.manual_stage_use_btn = ttk.Button(
        self.manual_stage_buttons_row,
        text="Uzyj stage jako wejscia Z2",
        style="WorkflowCard.TButton",
        command=self._use_manual_plate_stage_as_input,
    )
    self.manual_stage_use_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

    self.manual_stage_add_btn = ttk.Button(
        self.manual_stage_buttons_row,
        text="Dodaj zdjecia do stage",
        style="WorkflowCard.TButton",
        command=self._add_images_to_manual_plate_stage,
    )
    self.manual_stage_add_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    self._register_workflow_step_card(
        "workflow_manual_stage",
        self.manual_stage_section,
        labels=[
            self.manual_stage_help_lbl,
            self.manual_stage_status_lbl,
        ],
        style_targets=[
            {
                "widget": self.manual_stage_path_title_lbl,
                "kind": "title_label",
                "style": "WorkflowManualStagePathTitle.TLabel",
            },
            {
                "widget": self.manual_stage_path_row,
                "kind": "frame",
                "style": "WorkflowManualStagePath.TFrame",
            },
            {
                "widget": self.manual_stage_path_entry,
                "kind": "entry",
                "style": "WorkflowManualStagePath.TEntry",
            },
            {
                "widget": self.manual_stage_buttons_row,
                "kind": "frame",
                "style": "WorkflowManualStageButtons.TFrame",
            },
        ],
    )

    self.manual_stage_separator = self._build_left_section_separator(self.workflow_entry_shell_inner, pady=(18, 20))
    _log_build_phase("manual_stage_section")

    self.manual_stage_export_box = tk.Frame(
        self.workflow_entry_shell_inner,
        bd=0,
        highlightthickness=1,
        padx=14,
        pady=12,
    )
    self.manual_stage_export_box.pack(fill=tk.X, pady=(0, 12))

    self.manual_stage_export_title_lbl = tk.Label(
        self.manual_stage_export_box,
        text="Trening modelu YOLO Pose",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=0,
        pady=0,
        font=("Segoe UI", 9),
    )
    self.manual_stage_export_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))

    self.manual_stage_export_help_lbl = tk.Label(
        self.manual_stage_export_box,
        text=(
            "Jesli chcesz od razu trenowac model YOLO Pose wykrywajacy obrys tablic "
            "rejestracyjnych, ustaw split i wyeksportuj dataset. Potem przejdz do Z4, "
            "gdzie uruchomisz trening."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
        padx=0,
        pady=0,
    )
    self.manual_stage_export_help_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._bind_label_wrap_to_container(
        self.manual_stage_export_help_lbl,
        self.manual_stage_export_box,
        padding_px=36,
        min_px=220,
    )

    self.manual_stage_export_btn = ttk.Button(
        self.manual_stage_export_box,
        text="Split i eksport",
        style="WorkflowCard.TButton",
        command=self._jump_to_export_section,
    )
    self.manual_stage_export_btn.pack(anchor=tk.W, pady=(2, 0))

    export_lf = self._build_workflow_step_card(self.workflow_entry_shell_inner)
    self.export_section = export_lf
    export_lf.pack(fill=tk.X)
    self.export_title_lbl = tk.Label(
        export_lf,
        text="4. Split i eksport datasetu z gotowego runu anotacji",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        bd=0,
        highlightthickness=0,
    )
    self.export_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    _log_build_phase("export_section_start")

    self.export_intro_lbl = tk.Label(
        export_lf,
        textvariable=self.export_intro_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.export_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_label_state(self.export_intro_lbl, tone="muted", emphasis=False)

    run_row = ttk.Frame(export_lf, style="Panel.TFrame")
    run_row.pack(fill=tk.X, pady=(0, 4))
    self.plate_dataset_run_title_lbl = ttk.Label(
        run_row,
        text="Folder runu anotacji Z2",
        style="Panel.TLabel"
    )
    self.plate_dataset_run_title_lbl.pack(anchor=tk.W)
    run_input, self.plate_dataset_run_entry, self.plate_dataset_run_btn = self._build_left_path_row(
        run_row,
        self.plate_dataset_run_var,
        button_text="Wskaż inny run anotacji",
        button_command=self._select_plate_dataset_run_dir,
    )
    if self.plate_dataset_run_btn is not None:
        self.plate_dataset_run_btn.configure(style="WorkflowCard.TButton")

    img_row = ttk.Frame(export_lf, style="Panel.TFrame")
    img_row.pack(fill=tk.X, pady=(0, 4))
    self.plate_dataset_images_title_lbl = ttk.Label(
        img_row,
        text="Folder obrazów dla wybranego runu anotacji",
        style="Panel.TLabel"
    )
    self.plate_dataset_images_title_lbl.pack(anchor=tk.W)
    img_input, self.plate_dataset_images_entry, self.plate_dataset_images_btn = self._build_left_path_row(
        img_row,
        self.plate_dataset_images_var,
        button_text="Wskaż obrazy",
        button_command=self._select_plate_dataset_images_dir,
    )
    if self.plate_dataset_images_btn is not None:
        self.plate_dataset_images_btn.configure(style="WorkflowCard.TButton")

    out_row = ttk.Frame(export_lf, style="Panel.TFrame")
    out_row.pack(fill=tk.X, pady=(0, 8))
    self.plate_dataset_out_title_lbl = ttk.Label(
        out_row,
        text="Docelowy katalog datasetu",
        style="Panel.TLabel"
    )
    self.plate_dataset_out_title_lbl.pack(anchor=tk.W)
    out_input, self.plate_dataset_out_entry, _ = self._build_left_path_row(
        out_row,
        self.plate_dataset_out_var,
        state="readonly",
    )
    self._enable_compact_path_entry(
        self.plate_dataset_out_entry,
        self.plate_dataset_out_var,
        title="Pełna ścieżka katalogu datasetu",
    )

    split_lf = ttk.Frame(export_lf, style="Panel.TFrame")
    self.plate_export_split_frame = split_lf
    split_lf.pack(fill=tk.X, pady=(0, 8))
    self.split_title_lbl = tk.Label(
        split_lf,
        text="Podzial train / val / test",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        bd=0,
        highlightthickness=0,
    )
    self.split_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    split_grid = ttk.Frame(split_lf, style="Panel.TFrame")
    split_grid.pack(fill=tk.X)

    self.plate_train_title_lbl = ttk.Label(split_grid, text="Train %", style="Panel.TLabel")
    self.plate_train_title_lbl.grid(row=0, column=0, sticky=tk.W)
    self.plate_train_scale = ttk.Scale(
        split_grid,
        from_=50,
        to=90,
        variable=self.plate_train_pct,
        command=lambda e: self._update_plate_dataset_ratio_labels()
    )
    self.plate_train_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
    self.plate_train_lbl = ttk.Label(split_grid, text="80%", style="Panel.TLabel")
    self.plate_train_lbl.grid(row=0, column=2, sticky=tk.W)

    self.plate_val_title_lbl = ttk.Label(split_grid, text="Val %", style="Panel.TLabel")
    self.plate_val_title_lbl.grid(row=1, column=0, sticky=tk.W)
    self.plate_val_scale = ttk.Scale(
        split_grid,
        from_=0,
        to=20,
        variable=self.plate_val_pct,
        command=lambda e: self._update_plate_dataset_ratio_labels()
    )
    self.plate_val_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
    self.plate_val_lbl = ttk.Label(split_grid, text="10%", style="Panel.TLabel")
    self.plate_val_lbl.grid(row=1, column=2, sticky=tk.W)

    self.plate_test_title_lbl = ttk.Label(split_grid, text="Test %", style="Panel.TLabel")
    self.plate_test_title_lbl.grid(row=2, column=0, sticky=tk.W)
    self.plate_test_auto_lbl = ttk.Label(
        split_grid,
        text="liczony automatycznie",
        style="PanelMuted.TLabel",
    )
    self.plate_test_auto_lbl.grid(row=2, column=1, sticky=tk.W, padx=5)
    self.plate_test_lbl = ttk.Label(split_grid, text="Test: 10%", style="Panel.TLabel")
    self.plate_test_lbl.grid(row=2, column=2, sticky=tk.W)
    split_grid.columnconfigure(1, weight=1)

    self.export_plate_dataset_btn = ttk.Button(
        export_lf,
        text="EKSPORT",
        style="WorkflowCardPrimary.TButton",
        command=self._start_z2_export_choice_flow
    )
    self.export_plate_dataset_btn.pack(fill=tk.X)

    self.export_plate_annotations_btn = ttk.Button(
        export_lf,
        text="EKSPORTUJ ANOTACJE TABLIC",
        style="WorkflowCard.TButton",
        command=self._start_plate_annotation_package_export,
    )
    self.export_plate_annotations_btn.pack(fill=tk.X, pady=(6, 0))

    self.plate_export_progress = None

    self.plate_export_status_lbl = tk.Label(
        export_lf,
        text="Wskaż run anotacji i obrazy do eksportu datasetu.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0
    )
    self.plate_export_status_lbl.pack(anchor=tk.W, fill=tk.X)
    self._set_inline_label_state(
        self.plate_export_status_lbl,
        tone="muted",
        emphasis=False
    )
    self._bind_label_wrap_to_container(
        self.plate_export_status_lbl,
        export_lf,
        padding_px=28,
        min_px=240,
    )

    self.export_back_btn = ttk.Button(
        export_lf,
        text="Wroc do podsumowania wynikow autoanotacji",
        style="WorkflowCard.TButton",
        command=self._close_export_followup,
    )
    self.export_back_btn.pack(fill=tk.X, pady=(10, 0))
    self._register_workflow_step_card(
        "workflow_export",
        self.export_section,
        labels=[
            self.export_intro_lbl,
            self.plate_export_status_lbl,
        ],
        style_targets=[
            {"widget": run_row, "kind": "frame", "style": "WorkflowExportRunRow.TFrame"},
            {
                "widget": self.plate_dataset_run_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportRunTitle.TLabel",
            },
            {"widget": run_input, "kind": "frame", "style": "WorkflowExportRunInput.TFrame"},
            {
                "widget": self.plate_dataset_run_entry,
                "kind": "entry",
                "style": "WorkflowExportRunInput.TEntry",
            },
            {"widget": img_row, "kind": "frame", "style": "WorkflowExportImagesRow.TFrame"},
            {
                "widget": self.plate_dataset_images_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportImagesTitle.TLabel",
            },
            {"widget": img_input, "kind": "frame", "style": "WorkflowExportImagesInput.TFrame"},
            {
                "widget": self.plate_dataset_images_entry,
                "kind": "entry",
                "style": "WorkflowExportImagesInput.TEntry",
            },
            {"widget": out_row, "kind": "frame", "style": "WorkflowExportOutRow.TFrame"},
            {
                "widget": self.plate_dataset_out_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportOutTitle.TLabel",
            },
            {"widget": out_input, "kind": "frame", "style": "WorkflowExportOutInput.TFrame"},
            {
                "widget": self.plate_dataset_out_entry,
                "kind": "entry",
                "style": "WorkflowExportOutInput.TEntry",
            },
            {"widget": split_lf, "kind": "frame", "style": "WorkflowExportSplit.TFrame"},
            {"widget": split_grid, "kind": "frame", "style": "WorkflowExportSplitGrid.TFrame"},
            {
                "widget": self.plate_train_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportTrainTitle.TLabel",
            },
            {
                "widget": self.plate_train_scale,
                "kind": "scale",
                "style": "WorkflowExportTrain.Horizontal.TScale",
            },
            {
                "widget": self.plate_train_lbl,
                "kind": "label",
                "style": "WorkflowExportTrainValue.TLabel",
            },
            {
                "widget": self.plate_val_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportValTitle.TLabel",
            },
            {
                "widget": self.plate_val_scale,
                "kind": "scale",
                "style": "WorkflowExportVal.Horizontal.TScale",
            },
            {
                "widget": self.plate_val_lbl,
                "kind": "label",
                "style": "WorkflowExportValValue.TLabel",
            },
            {
                "widget": self.plate_test_title_lbl,
                "kind": "title_label",
                "style": "WorkflowExportTestTitle.TLabel",
            },
            {
                "widget": self.plate_test_auto_lbl,
                "kind": "muted_label",
                "style": "WorkflowExportTestAuto.TLabel",
            },
            {
                "widget": self.plate_test_lbl,
                "kind": "label",
                "style": "WorkflowExportTestValue.TLabel",
            },
            {
                "widget": self.export_plate_annotations_btn,
                "kind": "button",
                "style": "WorkflowExportAnnotationsButton.TButton",
            },
        ],
    )

    _log_build_phase("export_section_complete")
    self._update_plate_dataset_ratio_labels()
    _log_build_phase("export_ratio_labels")
    try:
        self._campaign_deferred_plate_export_status_after_id = self.frame.after(
            900,
            lambda: self._set_plate_export_status(
                "Status eksportu zostanie sprawdzony po otwarciu lub utworzeniu runu anotacji Z2.",
                "muted",
            ),
        )
    except Exception:
        pass
    _log_build_phase("export_status_deferred")
    self._refresh_left_panel_route_copy()
    _log_build_phase("export_route_copy")
    _log_build_phase("export_refresh")

    # --- ŚRODKOWA KOLUMNA (PODGLĄD + TERMINAL PROCESU) ---
    preview_host = ttk.Frame(center_frame)
    self.preview_host = preview_host
    preview_host.pack(fill=tk.BOTH, expand=True, padx=5, pady=0)

    list_lf = ttk.LabelFrame(
        self.preview_left_list_host,
        text=" Lista wyników anotacji ",
        padding=(4, 1, 4, 4),
    )
    self.preview_list_lf = list_lf
    list_lf.pack(fill=tk.BOTH, expand=True)
    preview_list_meta = ttk.Frame(list_lf, style="Panel.TFrame")
    self.preview_list_meta = preview_list_meta
    preview_list_meta.pack(fill=tk.BOTH, expand=True, pady=(0, 1))
    self.preview_list_summary_lbl = tk.Label(
        preview_list_meta,
        textvariable=self.preview_list_summary_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=260,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 9),
    )
    self.preview_list_summary_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))

    preview_list_filter_shell = tk.Frame(
        preview_list_meta,
        bd=0,
        highlightthickness=1,
        padx=8,
        pady=6,
    )
    self.preview_list_filter_shell = preview_list_filter_shell
    preview_list_filter_shell.pack(fill=tk.X, pady=(0, 3))
    preview_list_controls = ttk.Frame(preview_list_filter_shell, style="Panel.TFrame")
    self.preview_list_controls = preview_list_controls
    preview_list_controls.pack(fill=tk.X)
    self.preview_list_filter_lbl = tk.Label(
        preview_list_controls,
        text="Filtr jakości tablic",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
    )
    self.preview_list_filter_lbl.pack(anchor=tk.W, pady=(0, 0))
    preview_list_filter_row = ttk.Frame(preview_list_controls, style="Panel.TFrame")
    self.preview_list_filter_row = preview_list_filter_row
    preview_list_filter_row.pack(fill=tk.X, pady=(4, 2))
    ttk.Label(
        preview_list_filter_row,
        text="Min. Det",
    ).pack(side=tk.LEFT)
    self.preview_filter_conf_spin = ttk.Spinbox(
        preview_list_filter_row,
        from_=0.0,
        to=1.0,
        increment=0.05,
        width=6,
        format="%.2f",
        textvariable=self.preview_filter_conf_var,
    )
    self.preview_filter_conf_spin.pack(side=tk.LEFT, padx=(4, 10))
    ttk.Label(
        preview_list_filter_row,
        text="Min. Fit",
    ).pack(side=tk.LEFT)
    self.preview_filter_fit_spin = ttk.Spinbox(
        preview_list_filter_row,
        from_=0.0,
        to=1.0,
        increment=0.05,
        width=6,
        format="%.2f",
        textvariable=self.preview_filter_fit_var,
    )
    self.preview_filter_fit_spin.pack(side=tk.LEFT, padx=(4, 0))
    preview_list_filter_actions_row = ttk.Frame(preview_list_controls, style="Panel.TFrame")
    self.preview_list_filter_actions_row = preview_list_filter_actions_row
    preview_list_filter_actions_row.pack(fill=tk.X, pady=(2, 2))
    self.preview_apply_filter_btn = ttk.Button(
        preview_list_filter_actions_row,
        text="Zastosuj filtr",
        command=self._apply_preview_metric_filters,
        padding=(8, 1),
    )
    self.preview_apply_filter_btn.pack(side=tk.LEFT)
    self.preview_filter_reset_btn = ttk.Button(
        preview_list_filter_actions_row,
        text="Wyczyść filtry",
        command=self._reset_preview_metric_filters,
        padding=(8, 1),
    )
    self.preview_filter_reset_btn.pack(side=tk.LEFT, padx=(8, 0))
    self.preview_filter_hint_lbl = tk.Label(
        preview_list_controls,
        text="Po zastosowaniu filtra detekcji na liście pozostaną tylko pozycje spełniające ustawione kryteria.",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 8),
        wraplength=260,
    )
    self.preview_filter_hint_lbl.pack(anchor=tk.W, pady=(0, 2))
    try:
        self.app.ensure_adaptive_wrap(
            self.preview_filter_hint_lbl,
            container=preview_list_controls,
            padding=18,
            min_wrap=220,
        )
    except Exception:
        pass
    try:
        preview_list_filter_shell.pack_forget()
    except Exception:
        pass
    preview_list_sort_grid = ttk.Frame(preview_list_controls, style="Panel.TFrame")
    self.preview_list_sort_grid = preview_list_sort_grid
    preview_list_sort_grid.pack_forget()
    preview_list_sort_grid.columnconfigure(0, weight=1)
    preview_list_sort_grid.columnconfigure(1, weight=1)

    def _build_preview_sort_tile(parent, row, column, sort_mode, title, desc):
        tile = tk.Frame(
            parent,
            bd=0,
            highlightthickness=1,
            padx=6,
            pady=4,
            cursor="hand2",
        )
        tile.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=(0 if column == 0 else 4, 4 if column == 0 else 0),
            pady=(0, 2),
        )
        title_lbl = tk.Label(
            tile,
            text=title,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        title_lbl.pack(anchor=tk.W, fill=tk.X)
        self._preview_list_sort_tiles[sort_mode] = {
            "tile": tile,
            "title": title_lbl,
            "desc": None,
        }
        self._bind_preview_sort_tile(tile, sort_mode)

    _build_preview_sort_tile(
        preview_list_sort_grid,
        0,
        0,
        "Status: ED, OK, problem",
        "ED, OK, problem",
        "Najpierw ręczne poprawki",
    )
    _build_preview_sort_tile(
        preview_list_sort_grid,
        0,
        1,
        "Status: A, M, OK, problem",
        "A, M, OK, problem",
        "Najpierw autoanotacja",
    )
    _build_preview_sort_tile(
        preview_list_sort_grid,
        1,
        0,
        "Status: problem, ED, OK",
        "problem, ED, OK",
        "Najpierw braki i błędy",
    )
    _build_preview_sort_tile(
        preview_list_sort_grid,
        1,
        1,
        "Nazwa pliku A-Z",
        "Nazwa A-Z",
        "Kolejność alfabetyczna",
    )

    preview_list_section = ttk.Frame(preview_list_meta, style="Panel.TFrame")
    self.preview_list_section = preview_list_section
    preview_list_section.pack(fill=tk.BOTH, expand=True, pady=(3, 0))
    self.preview_list_sort_lbl = tk.Label(
        preview_list_section,
        text="Legenda i sortowanie listy",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
    )
    self.preview_list_sort_lbl.pack(anchor=tk.W, pady=(0, 2))

    preview_list_legend = ttk.Frame(preview_list_section, style="Panel.TFrame")
    self.preview_list_legend = preview_list_legend
    preview_list_legend.pack(fill=tk.X, pady=(0, 3))
    preview_list_legend_grid = ttk.Frame(preview_list_legend, style="Panel.TFrame")
    self.preview_list_legend_grid = preview_list_legend_grid
    preview_list_legend_grid.pack(fill=tk.X)
    preview_list_legend_grid.columnconfigure(0, weight=1, uniform="preview_legend")
    preview_list_legend_grid.columnconfigure(1, weight=1, uniform="preview_legend")
    preview_list_legend_grid.columnconfigure(2, weight=0, minsize=0)

    def _build_preview_list_legend_item(parent, row, column, badge_text, text, *, rowspan: int = 1):
        item = tk.Frame(parent, bd=0, highlightthickness=1, padx=5, pady=2)
        item.grid(
            row=row,
            column=column,
            rowspan=max(1, int(rowspan)),
            sticky="ew",
            padx=(0 if column == 0 else 4, 4 if column < 2 else 0),
            pady=(0, 2),
        )
        badge = tk.Label(
            item,
            text=badge_text,
            width=3,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7, "bold"),
            padx=3,
            pady=0,
        )
        badge.pack(side=tk.LEFT)
        label = tk.Label(
            item,
            text=text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            wraplength=96,
            font=("Segoe UI", 8),
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        count_lbl = tk.Label(
            item,
            text="0",
            anchor="e",
            justify=tk.RIGHT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 9),
            width=3,
        )
        count_lbl.pack(side=tk.RIGHT, padx=(6, 0))
        return item, badge, label, count_lbl

    (
        self.preview_list_legend_ok_item,
        self.preview_list_legend_ok_badge,
        self.preview_list_legend_ok_lbl,
        self.preview_list_legend_ok_count_lbl,
    ) = _build_preview_list_legend_item(preview_list_legend_grid, 0, 0, "A", "Auto")
    (
        self.preview_list_legend_corrected_item,
        self.preview_list_legend_corrected_badge,
        self.preview_list_legend_corrected_lbl,
        self.preview_list_legend_corrected_count_lbl,
    ) = _build_preview_list_legend_item(preview_list_legend_grid, 0, 1, "M", "Manual")
    (
        self.preview_list_legend_problem_item,
        self.preview_list_legend_problem_badge,
        self.preview_list_legend_problem_lbl,
        self.preview_list_legend_problem_count_lbl,
    ) = _build_preview_list_legend_item(preview_list_legend_grid, 1, 0, "--", "Brak")
    (
        self.preview_list_legend_dirty_item,
        self.preview_list_legend_dirty_badge,
        self.preview_list_legend_dirty_lbl,
        self.preview_list_legend_dirty_count_lbl,
    ) = _build_preview_list_legend_item(preview_list_legend_grid, 1, 1, "OK", "Zatwierdzone")
    self.preview_list_legend_total_item = tk.Frame(
        preview_list_legend_grid,
        bd=0,
        highlightthickness=1,
        padx=6,
        pady=2,
    )
    self.preview_list_legend_total_item.grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=(0, 0),
        pady=(1, 1),
    )
    self.preview_list_legend_total_lbl = tk.Label(
        self.preview_list_legend_total_item,
        text="Na liście",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 8),
    )
    self.preview_list_legend_total_lbl.pack(side=tk.LEFT, anchor=tk.W)
    self.preview_list_legend_total_count_lbl = tk.Label(
        self.preview_list_legend_total_item,
        text="0 / 0",
        anchor="e",
        justify=tk.RIGHT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
    )
    self.preview_list_legend_total_count_lbl.pack(side=tk.RIGHT, anchor=tk.E, padx=(6, 0))

    self._bind_preview_sort_tile(self.preview_list_legend_ok_item, "Status: A, M, OK, problem")
    self._bind_preview_sort_tile(self.preview_list_legend_corrected_item, "Status: ED, OK, problem")
    self._bind_preview_sort_tile(self.preview_list_legend_problem_item, "Status: problem, ED, OK")
    self._bind_preview_sort_tile(self.preview_list_legend_dirty_item, "Status: OK, ED, problem")
    list_frame = ttk.Frame(preview_list_section)
    self.preview_list_frame = list_frame
    list_frame.pack(fill=tk.BOTH, expand=True)
    self.preview_listbox = tk.Listbox(
        list_frame,
        font=("Consolas", 9),
        height=9,
        activestyle="none",
        selectmode=tk.EXTENDED,
        exportselection=False,
    )
    self.preview_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll = WebSlimScrollbar(list_frame, command=self.preview_listbox.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.preview_listbox.config(yscrollcommand=scroll.set)

    self.preview_filter_bar = ttk.Frame(preview_list_section, style="Panel.TFrame")
    self.preview_filter_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0), before=list_frame)
    self.preview_filter_bar_grid = ttk.Frame(self.preview_filter_bar, style="Panel.TFrame")
    self.preview_filter_bar_grid.pack(fill=tk.X)
    self.preview_filter_bar_grid.columnconfigure(0, weight=1)
    self.preview_filter_bar_grid.columnconfigure(1, weight=1)
    self.preview_filter_bar_grid.columnconfigure(2, weight=1)

    def _build_preview_filter_action_item(parent, column, badge_text, text):
        item = tk.Frame(parent, bd=0, highlightthickness=1, padx=5, pady=2, cursor="hand2")
        item.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=(0 if column == 0 else 4, 4 if column < 2 else 0),
            pady=(0, 2),
        )
        badge = tk.Label(
            item,
            text=badge_text,
            width=4,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7, "bold"),
            padx=3,
            pady=0,
        )
        badge.pack(side=tk.LEFT)
        label = tk.Label(
            item,
            text=text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            wraplength=92,
            font=("Segoe UI", 8),
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        value_lbl = tk.Label(
            item,
            text="",
            anchor="e",
            justify=tk.RIGHT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 8),
            width=13,
        )
        value_lbl.pack(side=tk.RIGHT, padx=(6, 0))
        return item, badge, label, value_lbl

    (
        self.preview_filter_bar_setup_item,
        self.preview_filter_bar_setup_badge,
        self.preview_filter_bar_setup_lbl,
        self.preview_filter_bar_setup_value_lbl,
    ) = _build_preview_filter_action_item(self.preview_filter_bar_grid, 0, "SET", "Ustaw")
    (
        self.preview_filter_bar_apply_item,
        self.preview_filter_bar_apply_badge,
        self.preview_filter_bar_apply_lbl,
        self.preview_filter_bar_apply_value_lbl,
    ) = _build_preview_filter_action_item(self.preview_filter_bar_grid, 1, "GO", "Zastosuj")
    (
        self.preview_filter_bar_reset_item,
        self.preview_filter_bar_reset_badge,
        self.preview_filter_bar_reset_lbl,
        self.preview_filter_bar_reset_value_lbl,
    ) = _build_preview_filter_action_item(self.preview_filter_bar_grid, 2, "CLR", "Wyczyść")
    self.preview_list_context_menu = tk.Menu(self.preview_listbox, tearoff=0)
    self.preview_list_context_menu.add_command(
        label="Zaznacz obrazy po filtrze",
        command=self._select_all_visible_preview_images,
    )
    for widget in (
        getattr(self, "preview_filter_bar_setup_item", None),
        getattr(self, "preview_filter_bar_setup_badge", None),
        getattr(self, "preview_filter_bar_setup_lbl", None),
        getattr(self, "preview_filter_bar_setup_value_lbl", None),
    ):
        if widget is not None:
            try:
                widget.bind("<Button-1>", lambda _e: self._open_preview_metric_filter_modal(), add="+")
            except Exception:
                pass
    for widget in (
        getattr(self, "preview_filter_bar_apply_item", None),
        getattr(self, "preview_filter_bar_apply_badge", None),
        getattr(self, "preview_filter_bar_apply_lbl", None),
        getattr(self, "preview_filter_bar_apply_value_lbl", None),
    ):
        if widget is not None:
            try:
                widget.bind("<Button-1>", lambda _e: self._apply_preview_metric_filters(), add="+")
            except Exception:
                pass
    for widget in (
        getattr(self, "preview_filter_bar_reset_item", None),
        getattr(self, "preview_filter_bar_reset_badge", None),
        getattr(self, "preview_filter_bar_reset_lbl", None),
        getattr(self, "preview_filter_bar_reset_value_lbl", None),
    ):
        if widget is not None:
            try:
                widget.bind("<Button-1>", lambda _e: self._reset_preview_metric_filters(), add="+")
            except Exception:
                pass
    self.preview_list_context_menu.add_separator()
    self.preview_list_context_menu.add_command(
        label="Oznacz zaznaczone jako OK",
        command=lambda: self._set_selected_preview_images_approved(
            True,
            persist_immediately=False,
            refresh_export_sources=False,
            schedule_followup_refresh=True,
        ),
    )
    self.preview_list_context_menu.add_command(
        label="Cofnij OK",
        command=lambda: self._set_selected_preview_images_approved(
            False,
            persist_immediately=False,
            refresh_export_sources=False,
            schedule_followup_refresh=True,
        ),
    )
    self.preview_list_context_menu.add_command(
        label="Zmień nazwę pliku...",
        command=self._rename_selected_preview_image_file,
    )
    self.preview_list_context_menu.add_separator()
    self.preview_list_context_menu.add_command(
        label="Wyczyść auto w całym runie",
        command=self._clear_selected_preview_auto_plates,
    )
    self.preview_listbox.bind("<Button-1>", self._on_preview_list_mouse_primary, add=False)
    self.preview_listbox.bind("<ButtonRelease-1>", lambda _event: self._refresh_plate_auto_scope_modal_selection_state(), add="+")
    self.preview_listbox.bind("<Button-3>", self._on_preview_list_mouse_secondary, add=False)
    self.preview_listbox.bind("<B1-Motion>", lambda _event: "break", add=False)
    self.preview_listbox.bind("<<ListboxSelect>>", self._on_preview_select)
    self.preview_listbox.bind("<KeyRelease>", lambda _event: self._refresh_plate_auto_scope_modal_selection_state(), add="+")
    self.preview_listbox.bind("<Shift-F10>", self._open_preview_list_context_menu_from_keyboard, add="+")
    self.preview_listbox.bind("<Menu>", self._open_preview_list_context_menu_from_keyboard, add="+")
    self.preview_listbox.bind("<Up>", self._on_preview_arrow_up_shortcut, add=False)
    self.preview_listbox.bind("<Down>", self._on_preview_arrow_down_shortcut, add=False)
    self.preview_listbox.bind("<KP_Up>", self._on_preview_arrow_up_shortcut, add=False)
    self.preview_listbox.bind("<KP_Down>", self._on_preview_arrow_down_shortcut, add=False)
    self.preview_listbox.bind("<MouseWheel>", self._on_preview_listbox_mousewheel, add="+")
    self.preview_listbox.bind("<Button-4>", self._on_preview_listbox_mousewheel, add="+")
    self.preview_listbox.bind("<Button-5>", self._on_preview_listbox_mousewheel, add="+")
    self._refresh_preview_list_legend_theme()
    self._refresh_preview_list_summary()
    _log_build_phase("preview_list")
    preview_lf = ttk.LabelFrame(preview_host, text=" Podgląd anotacji ", padding=8)
    self.preview_lf = preview_lf
    preview_lf.pack(fill=tk.BOTH, expand=True)
    preview_tools = ttk.Frame(preview_lf, style="Panel.TFrame")
    self.preview_tools = preview_tools
    self.preview_tools_primary_row = ttk.Frame(preview_tools, style="Panel.TFrame")
    self.preview_tools_primary_row.pack(fill=tk.X)
    self.preview_tools_primary_left = ttk.Frame(self.preview_tools_primary_row, style="Panel.TFrame")
    self.preview_tools_primary_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.preview_tools_primary_right = ttk.Frame(self.preview_tools_primary_row, style="Panel.TFrame")
    self.preview_tools_primary_right.pack(side=tk.RIGHT)
    self.preview_tools_secondary_row = ttk.Frame(preview_tools, style="Panel.TFrame")
    self.preview_tools_secondary_row.pack(fill=tk.X, pady=(6, 0))

    self.preview_prev_btn = ttk.Button(
        self.preview_tools_primary_left,
        text="Poprzednie (Q)",
        command=lambda: self._select_preview_relative(-1),
        state=tk.DISABLED
    )
    self.preview_prev_btn.pack(side=tk.LEFT)

    self.preview_next_btn = ttk.Button(
        self.preview_tools_primary_left,
        text="Nastepne (E)",
        command=lambda: self._select_preview_relative(1),
        state=tk.DISABLED
    )
    self.preview_next_btn.pack(side=tk.LEFT, padx=(6, 0))

    self.preview_fit_btn = ttk.Button(
        self.preview_tools_primary_left,
        text="Dopasuj obraz do okna (F)",
        command=self._fit_preview_image_to_view,
        state=tk.DISABLED
    )
    self.preview_fit_btn.pack(side=tk.LEFT, padx=(10, 0))

    self.preview_draw_btn = ttk.Button(
        self.preview_tools_primary_left,
        text="Nowy polygon 4 pkt (D)",
        command=self._toggle_preview_draw_mode,
        state=tk.DISABLED
    )
    self.preview_draw_btn.pack(side=tk.LEFT, padx=(10, 0))

    self.preview_fullscreen_btn = ttk.Button(
        self.preview_tools_primary_right,
        text="Pełny ekran (Enter)",
        command=self._toggle_preview_fullscreen,
        state=tk.DISABLED
    )
    self.preview_fullscreen_btn.pack(side=tk.RIGHT, padx=(0, 8))

    self.preview_move_stage_btn = ttk.Button(
        self.preview_tools_secondary_row,
        text="Przenies do stage",
        command=self._move_current_preview_image_to_stage,
        state=tk.DISABLED
    )
    self.preview_move_stage_btn.pack(side=tk.LEFT)

    self.preview_delete_image_btn = ttk.Button(
        self.preview_tools_secondary_row,
        text="Usun zdjecie (Del)",
        command=self._delete_current_preview_image_hard,
        state=tk.DISABLED
    )
    self.preview_delete_image_btn.pack(side=tk.LEFT, padx=(10, 0))

    self.preview_fullscreen_hint_lbl = tk.Label(
        preview_lf,
        textvariable=self.preview_fullscreen_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=780,
        bd=0,
        highlightthickness=0
    )
    self.preview_fullscreen_hint_var.set("")
    self._set_inline_label_state(self.preview_fullscreen_hint_lbl, tone="muted", emphasis=False)

    self.preview_save_btn = ttk.Button(
        self.preview_tools_primary_right,
        text="Zapisz (Ctrl+S)",
        command=self._save_preview_edits,
        state=tk.DISABLED
    )
    self.preview_save_btn.pack(side=tk.RIGHT)

    canvas_frame = tk.Frame(
        preview_lf,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.canvas_frame = canvas_frame
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    canvas_frame.bind(
        "<Configure>",
        self._on_preview_canvas_frame_configure,
        add="+",
    )
    preview_hint_frame = tk.Frame(
        canvas_frame,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=1,
    )
    self.preview_hint_frame = preview_hint_frame
    preview_hint_frame.bind("<ButtonPress-1>", self._on_preview_controls_legend_press, add="+")
    preview_hint_frame.bind("<B1-Motion>", self._on_preview_controls_legend_drag, add="+")
    preview_hint_frame.bind("<ButtonRelease-1>", self._on_preview_controls_legend_release, add="+")
    preview_hint_frame.bind("<Motion>", self._on_preview_controls_legend_motion, add="+")
    preview_hint_frame.bind("<Enter>", self._on_preview_controls_legend_enter, add="+")
    preview_hint_frame.bind("<Leave>", self._on_preview_controls_legend_leave, add="+")
    preview_hint_frame.bind("<MouseWheel>", self._on_preview_controls_legend_mousewheel, add="+")
    preview_hint_frame.bind("<Button-4>", self._on_preview_controls_legend_mousewheel, add="+")
    preview_hint_frame.bind("<Button-5>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_canvas = tk.Canvas(
        preview_hint_frame,
        height=84,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0
    )
    self.preview_controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.preview_controls_vbar = WebSlimScrollbar(
        preview_hint_frame,
        orient=tk.VERTICAL,
        command=self._on_preview_controls_scrollbar_command,
        auto_hide=False,
        thickness=6,
        thumb_scale=0.42,
        track_color=palette.get("scrollbar_track", palette.get("panel", "#252526")),
        thumb_color=palette.get("scrollbar_thumb", palette.get("success", "#2ecc71")),
        thumb_hover_color=palette.get("scrollbar_thumb_hover", palette.get("accent", "#56f29d")),
    )
    self.preview_controls_vbar.bind("<MouseWheel>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_vbar.bind("<Button-4>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_vbar.bind("<Button-5>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_vbar.bind("<Enter>", self._on_preview_controls_legend_enter, add="+")
    self.preview_controls_canvas.configure(takefocus=1)
    self.preview_controls_canvas.bind(
        "<Configure>",
        self._on_preview_controls_legend_configure,
        add="+"
    )
    self.preview_controls_canvas.bind("<ButtonPress-1>", self._on_preview_controls_legend_press, add="+")
    self.preview_controls_canvas.bind("<B1-Motion>", self._on_preview_controls_legend_drag, add="+")
    self.preview_controls_canvas.bind("<ButtonRelease-1>", self._on_preview_controls_legend_release, add="+")
    self.preview_controls_canvas.bind("<Motion>", self._on_preview_controls_legend_motion, add="+")
    self.preview_controls_canvas.bind("<Enter>", self._on_preview_controls_legend_enter, add="+")
    self.preview_controls_canvas.bind("<Leave>", self._on_preview_controls_legend_leave, add="+")
    self.preview_controls_canvas.bind("<MouseWheel>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_canvas.bind("<Button-4>", self._on_preview_controls_legend_mousewheel, add="+")
    self.preview_controls_canvas.bind("<Button-5>", self._on_preview_controls_legend_mousewheel, add="+")
    if not bool(getattr(self, "_preview_controls_global_wheel_bound", False)):
        self.frame.bind_all("<MouseWheel>", self._on_preview_controls_legend_mousewheel, add="+")
        self.frame.bind_all("<Button-4>", self._on_preview_controls_legend_mousewheel, add="+")
        self.frame.bind_all("<Button-5>", self._on_preview_controls_legend_mousewheel, add="+")
        self._preview_controls_global_wheel_bound = True
    self.preview_canvas = ZoomableCanvas(
        canvas_frame,
        bg=palette.get("panel_alt", palette.get("panel", "#1e1e1e")),
        highlightthickness=0,
    )
    self.preview_canvas.pack(fill=tk.BOTH, expand=True)
    self.preview_canvas.show_info = False
    self.preview_canvas.reset_shortcut_enabled = False
    self.preview_canvas.max_zoom = 12.0
    self.preview_canvas.set_overlay_renderer(self._draw_annotation_preview_overlay)
    self.preview_canvas.set_interaction_delegate(self)
    self.preview_canvas.bind("<Button-1>", lambda _e: self.preview_canvas.focus_set(), add="+")
    self.preview_canvas.bind("<Motion>", self._on_preview_canvas_motion, add="+")
    self.preview_canvas.bind("<Leave>", self._on_preview_canvas_leave, add="+")
    self.preview_canvas.bind("<Button-3>", self._on_preview_canvas_right_click, add="+")
    self.preview_canvas.bind("<FocusOut>", self._on_preview_canvas_focus_out, add="+")
    self._bind_preview_shortcuts()
    _log_build_phase("preview_canvas")

    self.preview_metrics_frame = tk.Frame(
        canvas_frame,
        bd=0,
        highlightthickness=1,
        cursor="arrow",
    )
    self.preview_metrics_header = tk.Frame(
        self.preview_metrics_frame,
        bd=0,
        highlightthickness=0,
    )
    self.preview_metrics_header.pack(fill=tk.X)
    self.preview_metrics_icon_lbl = tk.Label(
        self.preview_metrics_header,
        text="PX",
        width=3,
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        padx=4,
        pady=4,
        cursor="hand2",
    )
    self.preview_metrics_icon_lbl.pack(side=tk.LEFT)
    self.preview_metrics_title_lbl = tk.Label(
        self.preview_metrics_header,
        text="Parametry obrazu i ramki",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        padx=6,
        pady=4,
        cursor="hand2",
    )
    self.preview_metrics_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.preview_metrics_toggle_lbl = tk.Label(
        self.preview_metrics_header,
        text="-",
        width=2,
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
        padx=2,
        pady=4,
        cursor="hand2",
    )
    self.preview_metrics_toggle_lbl.pack(side=tk.LEFT)
    self.preview_metrics_grab_lbl = tk.Canvas(
        self.preview_metrics_header,
        bd=0,
        highlightthickness=0,
        width=22,
        height=22,
        cursor="fleur",
    )
    self.preview_metrics_grab_lbl.pack(side=tk.LEFT, padx=(3, 5), pady=3)
    self.preview_metrics_body = tk.Frame(
        self.preview_metrics_frame,
        bd=0,
        highlightthickness=0,
    )
    self.preview_metrics_body.pack(fill=tk.BOTH, expand=True)
    for widget in (
        self.preview_metrics_icon_lbl,
        self.preview_metrics_title_lbl,
        self.preview_metrics_toggle_lbl,
    ):
        try:
            widget.bind("<Button-1>", self._toggle_preview_metrics_overlay, add="+")
        except Exception:
            pass
    for widget in (self.preview_metrics_grab_lbl,):
        try:
            widget.bind("<ButtonPress-1>", self._on_preview_metrics_overlay_press, add="+")
            widget.bind("<B1-Motion>", self._on_preview_metrics_overlay_drag, add="+")
            widget.bind("<ButtonRelease-1>", self._on_preview_metrics_overlay_release, add="+")
        except Exception:
            pass
    self.preview_metrics_frame.place_forget()

    self.preview_overlay_dock = tk.Frame(
        canvas_frame,
        bd=0,
        highlightthickness=1,
        cursor="arrow",
    )
    self.preview_overlay_dock_header = tk.Frame(
        self.preview_overlay_dock,
        bd=0,
        highlightthickness=0,
    )
    self.preview_overlay_dock_header.pack(fill=tk.X)
    self.preview_overlay_dock_title_lbl = tk.Label(
        self.preview_overlay_dock_header,
        text="SZUFLADA",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 7),
        padx=7,
        pady=5,
        cursor="hand2",
    )
    self.preview_overlay_dock_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.preview_overlay_dock_toggle_lbl = tk.Label(
        self.preview_overlay_dock_header,
        text="<",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        width=2,
        padx=3,
        pady=5,
        cursor="hand2",
    )
    self.preview_overlay_dock_toggle_lbl.pack(side=tk.RIGHT)
    self.preview_overlay_dock_body = tk.Frame(
        self.preview_overlay_dock,
        bd=0,
        highlightthickness=0,
    )
    self.preview_overlay_dock_body.pack(fill=tk.X)
    self._preview_overlay_dock_tool_rows = {}
    self._preview_overlay_dock_status_rows = {}
    self.preview_overlay_dock_actions_title_lbl = tk.Label(
        self.preview_overlay_dock_body,
        text="AKCJE",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        padx=10,
        pady=7,
        cursor="arrow",
    )
    self.preview_overlay_dock_actions_title_lbl.pack(fill=tk.X, padx=6, pady=(0, 6))
    self.preview_overlay_dock_actions_frame = tk.Frame(
        self.preview_overlay_dock_body,
        bd=0,
        highlightthickness=0,
    )
    self.preview_overlay_dock_actions_frame.pack(fill=tk.X)
    self.preview_overlay_dock_status_title_lbl = tk.Label(
        self.preview_overlay_dock_body,
        text="STATUSY",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        padx=10,
        pady=7,
        cursor="arrow",
    )
    # Status pracy bramki jest renderowany niżej przez gate overlay.
    # Nie pakujemy tu statusów przełączników, żeby szuflada nie dublowała treści.
    self.preview_overlay_dock_status_frame = tk.Frame(
        self.preview_overlay_dock_body,
        bd=0,
        highlightthickness=0,
    )
    for tool_key, icon_text, label_text in (
        ("legend", "KP", "Kompas"),
        ("super", "SK", "Super"),
        ("metrics", "PX", "Parametry"),
    ):
        row = tk.Frame(self.preview_overlay_dock_actions_frame, bd=0, highlightthickness=1, cursor="hand2")
        row.pack(fill=tk.X, padx=6, pady=(0, 6))
        icon_lbl = tk.Label(
            row,
            text=icon_text,
            width=3,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 8),
            padx=4,
            pady=7,
            cursor="hand2",
        )
        icon_lbl.pack(side=tk.LEFT)
        text_lbl = tk.Label(
            row,
            text=label_text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 8),
            padx=6,
            pady=7,
            cursor="hand2",
        )
        text_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        action_status_lbl = tk.Label(
            row,
            text="",
            anchor="e",
            justify=tk.RIGHT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7),
            padx=4,
            pady=7,
            cursor="hand2",
        )
        action_status_lbl.pack(side=tk.RIGHT)
        self._preview_overlay_dock_tool_rows[tool_key] = {
            "row": row,
            "icon": icon_lbl,
            "label": text_lbl,
            "status": action_status_lbl,
        }
        for widget in (row, icon_lbl, text_lbl, action_status_lbl):
            try:
                widget.bind(
                    "<Button-1>",
                    lambda _event, key=tool_key: self._toggle_preview_overlay_dock_tool(key),
                    add="+",
                )
            except Exception:
                pass
        status_row = tk.Frame(self.preview_overlay_dock_status_frame, bd=0, highlightthickness=0, cursor="arrow")
        status_dot_lbl = tk.Label(
            status_row,
            text="●",
            width=2,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7),
            padx=1,
            pady=2,
            cursor="arrow",
        )
        status_dot_lbl.pack(side=tk.LEFT)
        status_name_lbl = tk.Label(
            status_row,
            text=label_text,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7),
            padx=3,
            pady=2,
            cursor="arrow",
        )
        status_name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        status_value_lbl = tk.Label(
            status_row,
            text="OFF",
            anchor="e",
            justify=tk.RIGHT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 7),
            padx=3,
            pady=2,
            cursor="arrow",
        )
        status_value_lbl.pack(side=tk.RIGHT)
        self._preview_overlay_dock_status_rows[tool_key] = {
            "row": status_row,
            "dot": status_dot_lbl,
            "label": status_name_lbl,
            "status": status_value_lbl,
        }
    for widget in (
        self.preview_overlay_dock_header,
        self.preview_overlay_dock_title_lbl,
        self.preview_overlay_dock_toggle_lbl,
    ):
        try:
            widget.bind("<Button-1>", self._toggle_preview_overlay_dock, add="+")
        except Exception:
            pass
    self.preview_overlay_dock.place_forget()

    self.preview_image_status_frame = tk.Frame(
        canvas_frame,
        bd=0,
        highlightthickness=1,
        cursor="arrow",
    )
    self.preview_image_status_lbl = tk.Label(
        self.preview_image_status_frame,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 7),
        padx=7,
        pady=4,
    )
    self.preview_image_status_lbl.pack(fill=tk.BOTH, expand=True)
    self.preview_image_status_frame.place_forget()

    self.preview_campaign_gate_frame = tk.Frame(
        canvas_frame,
        bd=0,
        highlightthickness=1,
        cursor="arrow",
    )
    self.preview_campaign_gate_header = tk.Frame(
        self.preview_campaign_gate_frame,
        bd=0,
        highlightthickness=0,
    )
    self.preview_campaign_gate_header.pack(fill=tk.X)
    self.preview_campaign_gate_title_lbl = tk.Label(
        self.preview_campaign_gate_header,
        text="BRAMKA GRAFU",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 8),
        padx=8,
        pady=3,
    )
    self.preview_campaign_gate_title_lbl.pack(fill=tk.X, expand=True)
    self.preview_campaign_gate_status_lbl = tk.Label(
        self.preview_campaign_gate_header,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=1,
        font=("Segoe UI Semibold", 8),
        padx=6,
        pady=2,
    )
    self.preview_campaign_gate_status_lbl.pack(fill=tk.X, padx=7, pady=(0, 6))
    self.preview_campaign_gate_body = tk.Frame(
        self.preview_campaign_gate_frame,
        bd=0,
        highlightthickness=0,
    )
    self.preview_campaign_gate_body.pack(fill=tk.X)
    self.preview_campaign_gate_counters_frame = tk.Frame(
        self.preview_campaign_gate_body,
        bd=0,
        highlightthickness=0,
    )
    self.preview_campaign_gate_counters_frame.pack(fill=tk.X, padx=5, pady=(5, 3))
    self.preview_campaign_gate_count_lbl = tk.Label(
        self.preview_campaign_gate_counters_frame,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
        padx=4,
        pady=4,
    )
    self.preview_campaign_gate_count_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
    self.preview_campaign_gate_missing_lbl = tk.Label(
        self.preview_campaign_gate_counters_frame,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI Semibold", 9),
        padx=4,
        pady=4,
    )
    self.preview_campaign_gate_missing_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
    self.preview_campaign_gate_quality_lbl = tk.Label(
        self.preview_campaign_gate_body,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=1,
        font=("Segoe UI Semibold", 8),
        padx=7,
        pady=3,
    )
    self.preview_campaign_gate_quality_lbl.pack(fill=tk.X, padx=5, pady=(0, 3))
    self.preview_campaign_gate_info_lbl = tk.Label(
        self.preview_campaign_gate_body,
        text="",
        anchor="center",
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 7),
        padx=7,
        pady=3,
    )
    self.preview_campaign_gate_info_lbl.pack(fill=tk.X, padx=5, pady=(0, 5))
    self.preview_campaign_gate_frame.place_forget()
    _log_build_phase("preview_overlays")

    self.frame.after_idle(
        lambda: (
            self._place_preview_legend_overlay(refresh=True),
            self._place_preview_overlay_dock(force_render=True),
            self._place_preview_campaign_gate_overlay(force_render=True),
        )
    )

    self.preview_edit_status_lbl = ttk.Label(
        preview_lf,
        textvariable=self.preview_edit_status_var,
        style="PanelMuted.TLabel",
        justify=tk.LEFT,
        wraplength=780
    )
    try:
        self.app.ensure_adaptive_wrap(self.preview_fullscreen_hint_lbl, container=preview_lf, padding=28, min_wrap=280)
        self.app.ensure_adaptive_wrap(self.preview_edit_status_lbl, container=preview_lf, padding=28, min_wrap=280)
    except Exception:
        pass
    try:
        self.frame.after_idle(
            lambda: (
                self._place_preview_legend_overlay(),
                self._place_preview_overlay_dock(),
                self._place_preview_campaign_gate_overlay(),
            )
        )
    except Exception:
        pass

    self.preview_debug_lbl = tk.Label(
        preview_lf,
        textvariable=self.preview_debug_var,
        font=("Consolas", 8),
        justify=tk.LEFT,
        anchor="w",
        wraplength=780,
        bd=0,
        highlightthickness=0
    )
    try:
        self.app.ensure_adaptive_wrap(self.preview_debug_lbl, container=preview_lf, padding=28, min_wrap=320)
    except Exception:
        pass
    if self._preview_debug_enabled:
        self.preview_debug_lbl.pack(fill=tk.X, pady=(6, 0))
        self._set_inline_label_state(self.preview_debug_lbl, tone="muted", emphasis=False)
    self._refresh_preview_debug_status()

    self.preview_processing_overlay_controller = CanvasProgressOverlay(
        preview_host,
        palette=palette,
        overlay_bg=blend_hex_colors(palette.get("panel", "#252526"), "#000000", 0.34),
        on_cancel=self._stop_annotation,
    )

    self._campaign_step2_splash_overlay = tk.Frame(
        self.frame,
        bd=0,
        highlightthickness=0,
    )
    self._campaign_step2_splash_card = tk.Frame(
        self._campaign_step2_splash_overlay,
        bd=0,
        highlightthickness=1,
        padx=20,
        pady=16,
    )
    self._campaign_step2_splash_card.place(relx=0.5, rely=0.18, anchor="n")
    self._campaign_step2_splash_card.grid_columnconfigure(0, weight=1)
    self._campaign_step2_splash_title_lbl = tk.Label(
        self._campaign_step2_splash_card,
        text="Przygotowuję Z2",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 12),
        bd=0,
        highlightthickness=0,
    )
    self._campaign_step2_splash_title_lbl.grid(row=0, column=0, sticky="ew")
    self._campaign_step2_splash_body_lbl = tk.Label(
        self._campaign_step2_splash_card,
        text="Ładuję kontekst wejścia i listę obrazów tego zestawu.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
        bd=0,
        highlightthickness=0,
    )
    self._campaign_step2_splash_body_lbl.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    self._campaign_step2_splash_progress_var = tk.DoubleVar(value=0.0)
    self._campaign_step2_splash_progress = tk.Canvas(
        self._campaign_step2_splash_card,
        width=520,
        height=16,
        bd=0,
        highlightthickness=0,
    )
    self._campaign_step2_splash_progress.grid(row=2, column=0, sticky="ew", pady=(11, 0))
    self._campaign_step2_splash_progress_pct_lbl = tk.Label(
        self._campaign_step2_splash_card,
        text="0%",
        anchor="e",
        justify=tk.RIGHT,
        font=("Segoe UI", 8, "bold"),
        bd=0,
        highlightthickness=0,
    )
    self._campaign_step2_splash_progress_pct_lbl.grid(row=3, column=0, sticky="e", pady=(5, 0))
    self._campaign_step2_splash_overlay.place_forget()

    log_tools = ttk.Frame(center_frame)
    self.log_tools = log_tools
    self.btn_toggle_annotation_log = None

    self.btn_toggle_annotation_log = ttk.Button(
        log_tools,
        text="Pokaż terminal",
        command=self._toggle_annotation_process_log
    )
    self.btn_toggle_annotation_log.pack_forget()

    ttk.Label(
        log_tools,
        text="Terminal procesu jest dostępny na żądanie użytkownika.",
        style="Muted.TLabel"
    ).pack(side=tk.LEFT, padx=(8, 0))

    self.annotation_log_overlay = tk.Frame(
        preview_host,
        bg="#0b0f14",
        bd=1,
        highlightthickness=1,
        highlightbackground="#313b46",
        highlightcolor="#313b46",
    )
    self.annotation_log_frame = ttk.LabelFrame(self.annotation_log_overlay, text=" Terminal procesu ", padding=6)
    self.annotation_log_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
    self.annotation_log_host = ttk.Frame(self.annotation_log_frame, style="Panel.TFrame")
    self.annotation_log_host.pack(fill=tk.BOTH, expand=True)

    self.log_text = tk.Text(
        self.annotation_log_host,
        wrap=tk.WORD,
        font=("Consolas", 9),
        bg="#161616",
        fg="#f3f3f3",
        insertbackground="#f3f3f3",
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
    )
    self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.annotation_log_scrollbar = WebSlimScrollbar(
        self.annotation_log_host,
        orient=tk.VERTICAL,
        command=self.log_text.yview,
        auto_hide=False,
    )
    self.annotation_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.log_text.configure(yscrollcommand=self.annotation_log_scrollbar.set)
    self.log_text.web_vbar = self.annotation_log_scrollbar
    self._redirect_logs()
    self._set_annotation_process_log_visibility(False)
    try:
        self.log_tools.pack_forget()
    except Exception:
        pass

    _log_build_phase("process_log")

    row_conf = build_annotation_right_panel(
        self,
        right_frame,
        panel_bg=panel_bg,
        panel_border=panel_border,
        nav_button_width=NAV_BUTTON_WIDTH,
    )
    _log_build_phase("right_panel")

    bind_annotation_widget_help_and_events(
        self,
        row_in=row_in,
        row_out=row_out,
        run_row=run_row,
        img_row=img_row,
        split_lf=split_lf,
        row_conf=row_conf,
    )
    _log_build_phase("bindings")
