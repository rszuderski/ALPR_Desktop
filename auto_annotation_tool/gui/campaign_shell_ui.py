#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign tab shell layout helpers."""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from ..config import CONFIG
from . import campaign_ui_helpers
from .help_manager import HELP
from .app_theme_definitions import CAMPAIGN_SIDEBAR_STYLE
from .campaign_sidebar import ProjectSidebarToggle, SlidingProjectSidebar
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors

def _build_ui(self):
    palette = getattr(self.app, "palette", {})

    # ---------------- HEADER ----------------
    header_bg = palette.get("panel", "#252526")
    header_f = tk.Frame(self.frame, bg=header_bg, bd=0, highlightthickness=0, padx=10, pady=4)
    header_f.columnconfigure(0, weight=1)
    header_f.columnconfigure(1, weight=1)
    header_f.columnconfigure(2, weight=0)
    self.header_frame = header_f

    self.lbl_title = tk.Label(
        header_f,
        text="PANEL KAMPANII",
        font=("Segoe UI", 14, "bold"),
        fg=palette.get("fg", "#f3f3f3"),
        bg=header_bg
    )
    # Tytul zostaje w obiekcie dla odswiezania motywu, ale nie zajmuje juz wiersza UI.
    self.lbl_title.grid_forget()

    proj_frame = tk.Frame(header_f, bg=header_bg, bd=0, highlightthickness=0)
    self.proj_frame = proj_frame

    self.btn_open_proj = ttk.Button(proj_frame, text="Otwórz projekt", command=self._open_selected_project)

    self.btn_del_proj = ttk.Button(proj_frame, text="Usuń projekt", command=self._delete_project)

    self.campaign_banner_shell = None
    self.banner_progress_row = None
    self.wizard_header_metro_canvas = None
    self.wizard_exit_button_canvas = None

    # ---------------- MAIN 2-COLUMN GRID ----------------
    main_container = ttk.Frame(self.frame, padding=(10, 8, 10, 10))
    main_container.pack(fill=tk.BOTH, expand=True)
    self.main_container = main_container

    self.project_loading_overlay = tk.Frame(
        main_container,
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_card = tk.Frame(
        self.project_loading_overlay,
        bd=0,
        highlightthickness=1,
        padx=0,
        pady=0,
    )
    self.project_loading_card.place(relx=0.5, rely=0.17, anchor="n", width=560, height=188)
    self.project_loading_card.grid_columnconfigure(1, weight=1)
    self.project_loading_card.grid_rowconfigure(0, weight=1)
    self.project_loading_accent_bar = tk.Frame(
        self.project_loading_card,
        width=2,
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_accent_bar.grid(row=0, column=0, sticky="nsw")
    self.project_loading_content = tk.Frame(
        self.project_loading_card,
        bd=0,
        highlightthickness=0,
        padx=18,
        pady=14,
    )
    self.project_loading_content.grid(row=0, column=1, sticky="nsew")
    self.project_loading_content.grid_columnconfigure(0, weight=1)
    self.project_loading_eyebrow_lbl = tk.Label(
        self.project_loading_content,
        text="ODTWARZANIE PROJEKTU",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 7, "bold"),
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_eyebrow_lbl.grid(row=0, column=0, sticky="ew")
    self.project_loading_title_lbl = tk.Label(
        self.project_loading_content,
        text="Ładuję projekt",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 12),
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_title_lbl.grid(row=1, column=0, sticky="ew", pady=(3, 0))
    self.project_loading_body_lbl = tk.Label(
        self.project_loading_content,
        text="Odtwarzam stan bramek i kontekst roboczy projektu.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_body_lbl.grid(row=2, column=0, sticky="ew", pady=(6, 0))
    self.project_loading_progress = ttk.Progressbar(
        self.project_loading_content,
        mode="indeterminate",
        style="Horizontal.TProgressbar",
    )
    self.project_loading_progress.grid(row=3, column=0, sticky="ew", pady=(9, 0))
    self.project_loading_step_lbl = tk.Label(
        self.project_loading_content,
        text="Proszę chwilę poczekać...",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
        bd=0,
        highlightthickness=0,
    )
    self.project_loading_step_lbl.grid(row=4, column=0, sticky="ew", pady=(5, 0))
    self.project_loading_overlay.place_forget()

    # Stały układ dwukolumnowy.
    main_container.columnconfigure(0, weight=1, minsize=860)
    main_container.columnconfigure(1, weight=0, minsize=CAMPAIGN_SIDEBAR_STYLE["width"])
    main_container.rowconfigure(0, weight=1)

    # RIGHT SIDEBAR: projekty i modele
    left_panel_host = ttk.Frame(main_container, width=CAMPAIGN_SIDEBAR_STYLE["width"])
    left_panel_host.grid(row=0, column=1, sticky="nsew", padx=(CAMPAIGN_SIDEBAR_STYLE["gap"], 0))
    try:
        left_panel_host.grid_propagate(False)
        left_panel_host.columnconfigure(0, weight=1)
        left_panel_host.rowconfigure(0, weight=1)
    except Exception:
        pass
    self.left_panel_host = left_panel_host

    left_panel = ttk.Frame(
        left_panel_host,
        padding=(15, 54, 15, 15),
    )
    left_panel.grid(row=0, column=0, sticky="nsew")
    self.left_panel = left_panel
    left_panel.columnconfigure(0, weight=1)
    left_panel.rowconfigure(0, weight=1)

    self.project_sidebar_title_lbl = tk.Label(
        left_panel_host, text="Projekty", bg=palette["panel"], fg=palette["fg"],
        font=CAMPAIGN_SIDEBAR_STYLE["button_font"], anchor="w",
    )
    self.project_sidebar_title_lbl.place(x=15, y=14)

    self.right_sidebar_tab_host = ProjectSidebarToggle(
        main_container, palette, lambda: _toggle_project_side_panel(self),
    )
    self.right_sidebar_tab_canvas = self.right_sidebar_tab_host

    settled_refresh = {"after_id": None}

    def _cancel_settled_refresh(event=None):
        if event is not None and event.widget is not main_container:
            return
        if settled_refresh["after_id"] is not None:
            main_container.after_cancel(settled_refresh["after_id"])
            settled_refresh["after_id"] = None

    def _panel_settled():
        _cancel_settled_refresh()
        def _refresh():
            settled_refresh["after_id"] = None
            if not main_container.winfo_exists() or bool(getattr(self, "_project_sidebar_animating", False)):
                return
            self._sync_left_panel_canvas_width()
            self._sync_right_panel_canvas_width()
            refresh_graph = getattr(self, "_campaign_graph_refresh_after_layout_change", None)
            if callable(refresh_graph):
                refresh_graph()
        settled_refresh["after_id"] = main_container.after_idle(_refresh)

    main_container.bind("<Destroy>", _cancel_settled_refresh, add="+")

    self._project_sidebar_motion = SlidingProjectSidebar(
        main_container, left_panel_host, self.right_sidebar_tab_host,
        on_motion=lambda active: setattr(self, "_project_sidebar_animating", active),
        on_settled=_panel_settled,
    )

    self.left_scroll_host = ttk.Frame(left_panel)
    self.left_scroll_host.grid(row=0, column=0, sticky="nsew")
    self.left_scroll_host.columnconfigure(0, weight=1)
    self.left_scroll_host.rowconfigure(0, weight=1)

    self.left_panel_canvas = tk.Canvas(
        self.left_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0
    )
    self.left_panel_canvas.grid(row=0, column=0, sticky="nsew")

    self.left_panel_scrollbar = WebSlimScrollbar(
        self.left_scroll_host,
        command=self.left_panel_canvas.yview
    )
    self.left_panel_scrollbar.grid(row=0, column=1, sticky="ns")
    try:
        green = self._get_campaign_green_accent()
        self.left_panel_scrollbar.configure_style(
            track_color=palette.get("panel", "#252526"),
            thumb_color=green,
            thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
        )
    except Exception:
        pass
    self.left_panel_canvas.configure(yscrollcommand=self.left_panel_scrollbar.set)

    self.left_content = ttk.Frame(self.left_panel_canvas)
    self.left_content_window = self.left_panel_canvas.create_window(
        (0, 0),
        window=self.left_content,
        anchor="nw"
    )
    self.left_content.bind("<Configure>", self._sync_left_panel_scrollregion, add="+")
    self.left_panel_canvas.bind("<Configure>", self._sync_left_panel_canvas_width, add="+")

    self.left_footer = ttk.Frame(left_panel)
    self.left_footer.grid(row=1, column=0, sticky="ew", pady=(10, 0))

    self.left_panel_hint_lbl = tk.Label(
        self.left_content,
        text="",
        justify=tk.LEFT,
        fg=palette.get("muted", "#b8b8b8"),
        bg=palette.get("panel", "#252526")
    )
    self.left_panel_hint_lbl.pack(anchor=tk.W, pady=(0, 0))
    self._build_projects_browser(self.left_content)

    self._build_model_status(self.left_content, "Model Pojazdów (Detect):", "vehicle", Path(CONFIG.DIR_6_MODELS))
    self._build_model_status(self.left_content, "Model Tablic (Pose):", "plate", Path(CONFIG.DIR_6_MODELS))
    self._build_model_status(self.left_content, "Model Znaków (OCR/YOLO):", "char", Path(CONFIG.DIR_6_MODELS))

    self.btn_complete_project = ttk.Button(
        self.left_footer,
        text="Zakończ projekt",
        command=self._toggle_project_completion
    )
    self.btn_complete_project.pack(fill=tk.X, pady=(8, 0))

    self.btn_exit_project = ttk.Button(
        self.left_footer,
        text="Wyjdź z projektu",
        command=self._exit_project_mode,
    )
    self.btn_exit_project.pack(fill=tk.X, pady=(6, 0))

    HELP.bind_help(self.btn_open_proj, "camp_open_project")
    HELP.bind_help(self.btn_del_proj, "camp_delete_project")
    HELP.bind_help(self.btn_exit_project, "camp_exit_project")
    HELP.bind_help(left_panel, "camp_models")
    HELP.bind_help(self.btn_complete_project, "camp_advance")

    # LEFT MAIN PANEL: workflow
    self.right_panel = ttk.Frame(
        main_container,
        padding=0
    )
    self.right_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    self.right_panel.columnconfigure(0, weight=1)
    # The campaign graph is now the primary workflow surface.  The old
    # scrollable stage-card area remains only as a compatibility placeholder.
    self.right_panel.rowconfigure(0, weight=1)
    self.right_panel.rowconfigure(1, weight=0)

    self.right_graph_host = ttk.Frame(self.right_panel)
    self.right_graph_host.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
    self.right_graph_host.columnconfigure(0, weight=1)
    self.right_graph_host.rowconfigure(0, weight=1)
    self.right_graph_content = tk.Frame(
        self.right_graph_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.right_graph_content.grid(row=0, column=0, sticky="nsew")

    self.right_scroll_host = ttk.Frame(self.right_panel)
    self.right_scroll_host.grid(row=1, column=0, sticky="nsew")
    self.right_scroll_host.columnconfigure(0, weight=1)
    self.right_scroll_host.rowconfigure(0, weight=1)

    self.right_panel_canvas = tk.Canvas(
        self.right_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0
    )
    self.right_panel_canvas.grid(row=0, column=0, sticky="nsew")

    self.right_panel_scrollbar = WebSlimScrollbar(
        self.right_scroll_host,
        command=self.right_panel_canvas.yview
    )
    self.right_panel_scrollbar.grid(row=0, column=1, sticky="ns")
    try:
        green = self._get_campaign_green_accent()
        self.right_panel_scrollbar.configure_style(
            track_color=palette.get("panel", "#252526"),
            thumb_color=green,
            thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
        )
    except Exception:
        pass
    self.right_panel_canvas.configure(yscrollcommand=self.right_panel_scrollbar.set)

    self.right_content = ttk.Frame(self.right_panel_canvas)
    self.right_content_window = self.right_panel_canvas.create_window(
        (0, 0),
        window=self.right_content,
        anchor="nw"
    )
    self.right_content.bind("<Configure>", self._sync_right_panel_scrollregion, add="+")
    self.right_panel_canvas.bind("<Configure>", self._sync_right_panel_canvas_width, add="+")
    self.right_scroll_host.grid_remove()

    self._rebuild_wizard_stage_ui()
    self.frame.after_idle(self._sync_left_panel_scrollregion)
    self.frame.after_idle(self._sync_left_panel_canvas_width)
    self.frame.after_idle(self._sync_right_panel_scrollregion)
    self.frame.after_idle(self._sync_right_panel_canvas_width)
    self.frame.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
    self.frame.bind_all("<Button-4>", self._on_global_mousewheel, add="+")
    self.frame.bind_all("<Button-5>", self._on_global_mousewheel, add="+")
    campaign_ui_helpers._repair_polish_widget_texts(self.frame)

def _set_project_side_panel_collapsed(self, collapsed: bool, *, from_graph: bool = False):
    collapsed = bool(collapsed)
    motion = getattr(self, "_project_sidebar_motion", None)
    if motion is None:
        return
    if collapsed and not bool(getattr(self, "_project_side_panel_collapsed", False)):
        # Preserve the graph's established world origin as its viewport widens.
        graph_view = getattr(self, "_campaign_graph_view", None)
        graph_canvas = getattr(self, "campaign_transition_graph_canvas", None)
        if isinstance(graph_view, dict) and graph_canvas is not None and graph_canvas.winfo_exists():
            canvas_width = float(max(graph_canvas.winfo_width(), 720))
            layout_width = float(graph_view.get("layout_width", 0.0) or 0.0)
            if layout_width <= 0:
                layout_width = min(max(860.0, canvas_width * 0.96), 1480.0)
                graph_view["layout_width"] = layout_width
            graph_view.setdefault("layout_offset_x", (canvas_width-layout_width) / 2.0)
    self._project_side_panel_collapsed = collapsed
    motion.set_collapsed(collapsed, animate=bool(self.frame.winfo_ismapped()))
    if from_graph and collapsed:
        self.app.update_status("Prawy panel projektu schowany, aby powiększyć graf.", "info")

def _toggle_project_side_panel(self):
    _set_project_side_panel_collapsed(
        self,
        not bool(getattr(self, "_project_side_panel_collapsed", False)),
    )

def _apply_campaign_graph_fullscreen_layout(self):
    expanded = bool(getattr(self, "_campaign_graph_fullscreen", False))
    right_panel = getattr(self, "right_panel", None)
    graph_host = getattr(self, "right_graph_host", None)
    graph_content = getattr(self, "right_graph_content", None)
    scroll_host = getattr(self, "right_scroll_host", None)
    canvas = getattr(self, "campaign_transition_graph_canvas", None)

    try:
        if right_panel is not None:
            right_panel.rowconfigure(0, weight=1)
            right_panel.rowconfigure(1, weight=0)
    except Exception:
        pass
    try:
        if graph_host is not None:
            graph_host.grid_configure(sticky="nsew")
            graph_host.rowconfigure(0, weight=1)
            graph_host.columnconfigure(0, weight=1)
    except Exception:
        pass
    try:
        if graph_content is not None:
            graph_content.grid_configure(sticky="nsew")
    except Exception:
        pass
    try:
        if scroll_host is not None:
            scroll_host.grid_remove()
    except Exception:
        pass
    def _safe_generate_configure(widget):
        try:
            if widget is not None and widget.winfo_exists():
                widget.event_generate("<Configure>")
        except tk.TclError:
            pass
        except Exception:
            pass

    try:
        if canvas is not None and canvas.winfo_exists():
            canvas.configure(height=(820 if expanded else 580))
            canvas.pack_configure(fill=tk.BOTH, expand=True)
            _safe_generate_configure(canvas)
            self.frame.after_idle(lambda widget=canvas: _safe_generate_configure(widget))
    except tk.TclError:
        pass
    except Exception:
        pass

def _toggle_campaign_graph_fullscreen(self):
    entering = not bool(getattr(self, "_campaign_graph_fullscreen", False))
    if entering:
        self._campaign_graph_prev_side_panel_collapsed = bool(getattr(self, "_project_side_panel_collapsed", False))
    self._campaign_graph_fullscreen = entering
    if entering:
        _set_project_side_panel_collapsed(self, True, from_graph=True)
    else:
        _set_project_side_panel_collapsed(
            self,
            bool(getattr(self, "_campaign_graph_prev_side_panel_collapsed", False)),
        )
    _apply_campaign_graph_fullscreen_layout(self)
    try:
        self.main_container.update_idletasks()
        self.frame.after_idle(self._apply_campaign_graph_fullscreen_layout)
    except Exception:
        pass

def _sync_left_panel_scrollregion(self, event=None):
    if not hasattr(self, "left_panel_canvas") or self.left_panel_canvas is None:
        return

    try:
        self.left_panel_canvas.configure(scrollregion=self.left_panel_canvas.bbox("all"))
    except Exception:
        pass

def _sync_left_panel_canvas_width(self, event=None):
    if not hasattr(self, "left_panel_canvas") or self.left_panel_canvas is None:
        return

    try:
        width = max(50, int(self.left_panel_canvas.winfo_width()))
        self.left_panel_canvas.itemconfigure(self.left_content_window, width=width)
    except Exception:
        pass

def _sync_right_panel_scrollregion(self, event=None):
    if not hasattr(self, "right_panel_canvas") or self.right_panel_canvas is None:
        return

    try:
        self.right_panel_canvas.configure(scrollregion=self.right_panel_canvas.bbox("all"))
    except Exception:
        pass

def _sync_right_panel_canvas_width(self, event=None):
    if not hasattr(self, "right_panel_canvas") or self.right_panel_canvas is None:
        return

    try:
        width = max(50, int(self.right_panel_canvas.winfo_width()))
        self.right_panel_canvas.itemconfigure(self.right_content_window, width=width)
    except Exception:
        pass
