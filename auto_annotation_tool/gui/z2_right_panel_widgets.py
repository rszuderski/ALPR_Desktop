#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Right-side Z2 status/configuration panel widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .z2_shared_ui import get_campaign_return_to_graph_copy


def build_annotation_right_panel(
    host,
    right_frame,
    *,
    panel_bg: str,
    panel_border: str,
    nav_button_width: int,
):
    self = host
    NAV_BUTTON_WIDTH = nav_button_width
    # --- PRAWA KOLUMNA ---
    right_scroll_shell = tk.Frame(
        right_frame,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=panel_border,
        highlightcolor=panel_border,
    )
    right_scroll_shell.pack(fill=tk.BOTH, expand=True)
    self.right_scroll_shell = right_scroll_shell
    return_copy = get_campaign_return_to_graph_copy(self)

    # Legacy right-side detection configuration was replaced by the graph flow,
    # scoped modals, and the global device menu. Keep the old attributes as
    # inert placeholders so older refresh paths can no-op safely.
    self.right_scroll_host = None
    self.right_settings_canvas = None
    self.right_settings_scrollbar = None
    self.right_settings_content = None
    self._right_settings_window_id = None
    self.detection_settings_lf = None
    self._legacy_detection_panel_visible = False

    self.approve_btn_row = ttk.LabelFrame(
        right_scroll_shell,
        text=str(return_copy.get("section") or " Powrót do grafu "),
        padding=10,
    )
    self.approve_btn_row.bind("<Configure>", self._sync_approve_hint_wraplength, add="+")

    self.approve_context_box = tk.Frame(
        self.approve_btn_row,
        bd=0,
        highlightthickness=1
    )
    self.approve_context_lbl = tk.Label(
        self.approve_context_box,
        textvariable=self.approve_context_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0,
    )
    self.approve_context_lbl.pack(fill=tk.X, pady=(4, 8))
    self._set_inline_label_state(self.approve_context_lbl, tone="info", emphasis=True)
    self._set_approve_context_box_state("info")

    self.approve_hint_box = tk.Frame(
        self.approve_btn_row,
        bd=0,
        highlightthickness=1
    )

    self.approve_hint_title_lbl = tk.Label(
        self.approve_hint_box,
        textvariable=self.approve_hint_title_var,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0,
        font=("Segoe UI", 9, "bold"),
    )
    self._set_inline_label_state(self.approve_hint_title_lbl, tone="muted", emphasis=True)

    self.approve_hint_table_frame = ttk.Frame(self.approve_hint_box, style="Panel.TFrame")

    self.approve_gate_hint_lbl = tk.Label(
        self.approve_hint_box,
        textvariable=self.approve_gate_hint_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0
    )
    self._set_inline_label_state(self.approve_gate_hint_lbl, tone="muted", emphasis=False)
    self._set_approve_hint_box_state("muted")

    self.approve_breakdown_var = tk.StringVar(value="")
    self.approve_breakdown_box = tk.Frame(
        self.approve_btn_row,
        bd=0,
        highlightthickness=1
    )

    self.approve_breakdown_title_lbl = tk.Label(
        self.approve_breakdown_box,
        textvariable=self.approve_breakdown_title_var,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0,
        font=("Segoe UI", 9, "bold"),
    )
    self._set_inline_label_state(self.approve_breakdown_title_lbl, tone="muted", emphasis=True)

    self.approve_breakdown_lbl = tk.Label(
        self.approve_breakdown_box,
        textvariable=self.approve_breakdown_var,
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0,
    )
    self.approve_breakdown_lbl.pack(fill=tk.X, pady=(4, 4))
    self._set_inline_label_state(self.approve_breakdown_lbl, tone="muted", emphasis=False)

    self.approve_breakdown_table_frame = ttk.Frame(self.approve_breakdown_box, style="Panel.TFrame")

    self.approve_breakdown_canvas = tk.Canvas(
        self.approve_breakdown_box,
        height=46,
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
    )
    self.approve_breakdown_canvas.pack(fill=tk.X)
    self.approve_breakdown_canvas.bind(
        "<Configure>",
        lambda _event: self._refresh_approval_breakdown_canvas(),
        add="+",
    )

    self.mode_title_lbl = None
    self.mode_combo = None
    self.mode_hint_lbl = None
    self.veh_frame = None
    self.vehicle_combo = None
    self.veh_custom_row = None
    self.pla_frame = None
    self.pla_custom_row = None
    self.plate_path_entry = None
    self.plate_browse_btn = None
    self.param_frame = None
    self.conf_value_lbl = None
    self.device_combo = None
    self.device_hint_lbl = None
    row_conf = None

    self.approve_btn_frame = tk.Frame(self.approve_btn_row, bd=0, highlightthickness=0)
    self.approve_btn_frame.pack(fill=tk.X)

    self.approve_btn = ttk.Button(
        self.approve_btn_frame,
        text="Zatwierdź",
        command=self._approve_annotation_stage,
        state=tk.DISABLED
    )
    # Campaign gates are closed from the graph badge. Keep the legacy button
    # alive for older code paths, but do not show it in the right panel by default.

    self.return_to_campaign_right_btn = ttk.Button(
        self.approve_btn_frame,
        text=str(return_copy.get("button") or "Wróć do grafu"),
        style="WorkflowCard.TButton",
        command=self._return_to_campaign_wizard,
    )
    self.return_to_campaign_right_btn.configure(
        padding=(8, 2),
        width=max(NAV_BUTTON_WIDTH, int(return_copy.get("width", NAV_BUTTON_WIDTH) or NAV_BUTTON_WIDTH)),
    )

    # Zatrzymywanie autoanotacji należy do modala procesu, nie do prawego
    # panelu bramki. Atrybut zostaje tylko jako bezpieczny placeholder dla
    # starszych ścieżek, które mogą go sprawdzać przez getattr().
    self.stop_annotation_right_btn = None

    return row_conf
