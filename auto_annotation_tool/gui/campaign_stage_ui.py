#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign graph UI helpers extracted from tab_campaign.py.

The historical ``wizard_stage`` names are kept as compatibility delegates while
the visible campaign workflow is migrated to the state-machine graph.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from datetime import datetime
from time import perf_counter
import xml.etree.ElementTree as ET
import shutil
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from . import campaign_model_status
from . import campaign_step1_assets
from . import campaign_step1_ingest
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
from .campaign_models import WizardStageStatus


def _rebuild_wizard_stage_ui(self):
    wizard_parent = self.right_content if self.right_content is not None else self.right_panel
    for widget in wizard_parent.winfo_children():
        widget.destroy()
    graph_parent = getattr(self, "right_graph_content", None) or wizard_parent
    if graph_parent is not wizard_parent:
        try:
            for widget in graph_parent.winfo_children():
                widget.destroy()
        except Exception:
            pass
    self.step1_ingest_host_item = None
    self.wizard_stage_cards = {}

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")

    self.wizard_empty_state_card = {"shell": None}
    self.wizard_stage_cards_host = None
    self.wizard_header_shell = None
    self.wizard_header_title_lbl = None
    self.wizard_header_summary_lbl = None

    try:
        right_panel = getattr(self, "right_panel", None)
        if right_panel is not None:
            right_panel.rowconfigure(0, weight=1)
            right_panel.rowconfigure(1, weight=0)
    except Exception:
        pass
    try:
        graph_host = getattr(self, "right_graph_host", None)
        if graph_host is not None:
            graph_host.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
            graph_host.columnconfigure(0, weight=1)
            graph_host.rowconfigure(0, weight=1)
    except Exception:
        pass
    try:
        if graph_parent is not None and graph_parent.winfo_exists():
            graph_parent.grid(row=0, column=0, sticky="nsew")
            graph_parent.columnconfigure(0, weight=1)
            graph_parent.rowconfigure(0, weight=1)
    except Exception:
        pass
    try:
        scroll_host = getattr(self, "right_scroll_host", None)
        if scroll_host is not None:
            scroll_host.grid_remove()
    except Exception:
        pass

    graph_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    self.wizard_transition_graph_shell = tk.Frame(
        graph_parent,
        bg=panel_bg,
        bd=0,
        highlightthickness=0,
        highlightbackground=graph_border,
        highlightcolor=graph_border,
    )
    self.wizard_transition_graph_shell.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
    self.wizard_transition_graph_body = tk.Frame(
        self.wizard_transition_graph_shell,
        bg=panel_bg,
        bd=0,
        highlightthickness=0,
    )
    self.wizard_transition_graph_body.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    self.frame.after_idle(self._sync_right_panel_scrollregion)
    self.frame.after_idle(self._sync_right_panel_canvas_width)

def _refresh_wizard_transition_graph(self, *, allow_pending_actions: bool = True):
    if not self.frame.winfo_viewable():
        pending = getattr(self, "_wizard_graph_deferred_request", None)
        self._wizard_graph_deferred_request = bool(allow_pending_actions or pending is True)
        if getattr(self, "_wizard_graph_map_bound", None) is not True:
            def on_map(event):
                if event.widget is not self.frame:
                    return
                def refresh_when_visible():
                    request = getattr(self, "_wizard_graph_deferred_request", None)
                    if isinstance(request, bool) and self.frame.winfo_viewable():
                        self._refresh_wizard_transition_graph(allow_pending_actions=request)
                self.frame.after_idle(refresh_when_visible)
            self.frame.bind("<Map>", on_map, add="+")
            self._wizard_graph_map_bound = True
        return
    self._wizard_graph_deferred_request = None
    refresh_started = perf_counter()
    try:
        ensure_ready = getattr(self, "_ensure_wizard_stage_ui_ready", None)
        if callable(ensure_ready):
            ensure_ready()
    except Exception as e:
        logger.debug(f"Nie udalo sie odbudowac kontenera grafu kampanii: {e}")

    body = getattr(self, "wizard_transition_graph_body", None)
    try:
        if body is None or not body.winfo_exists():
            logger.warning("Graf kampanii nie ma aktywnego kontenera body po probie odbudowy UI.")
            return
    except Exception:
        logger.warning("Graf kampanii ma nieaktywny kontener body po probie odbudowy UI.")
        return
    try:
        self._render_step1_route_actions(body, allow_pending_actions=allow_pending_actions)
    except Exception as e:
        logger.exception("Nie udało się odświeżyć grafu przejść kampanii")
        try:
            for widget in body.winfo_children():
                widget.destroy()
            palette = getattr(self.app, "palette", {})
            tk.Label(
                body,
                text="Mapa przejść kampanii",
                anchor="w",
                justify=tk.LEFT,
                font=("Segoe UI", 10, "bold"),
                fg=palette.get("fg", "#f3f3f3"),
                bg=palette.get("panel", "#252526"),
            ).pack(fill=tk.X)
            tk.Label(
                body,
                text=f"Graf nie został zbudowany. Renderer zgłosił błąd: {e}",
                anchor="w",
                justify=tk.LEFT,
                fg=palette.get("warning", "#f39c12"),
                bg=palette.get("panel", "#252526"),
            ).pack(fill=tk.X, pady=(4, 0))
        except Exception:
            pass
    finally:
        try:
            self._log_perf(
                "refresh_wizard_transition_graph",
                refresh_started,
                threshold_ms=20.0,
            )
        except Exception:
            pass


def _show_wizard_transition_graph(self):
    try:
        ensure_ready = getattr(self, "_ensure_wizard_stage_ui_ready", None)
        if callable(ensure_ready):
            ensure_ready()
    except Exception as e:
        logger.debug(f"Nie udalo sie przygotowac UI grafu kampanii: {e}")

    graph_shell = getattr(self, "wizard_transition_graph_shell", None)
    try:
        if graph_shell is None or not graph_shell.winfo_exists():
            logger.warning("Graf kampanii nie zostal pokazany, bo shell grafu nie istnieje.")
            return
    except Exception:
        logger.warning("Graf kampanii nie zostal pokazany, bo shell grafu jest nieaktywny.")
        return

    try:
        graph_host = getattr(self, "right_graph_host", None)
        if graph_host is not None:
            graph_host.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
            graph_host.columnconfigure(0, weight=1)
            graph_host.rowconfigure(0, weight=1)
            try:
                graph_content = getattr(self, "right_graph_content", None)
                if graph_content is not None and graph_content.winfo_exists():
                    graph_content.grid(row=0, column=0, sticky="nsew")
                    graph_content.columnconfigure(0, weight=1)
                    graph_content.rowconfigure(0, weight=1)
            except Exception:
                pass
    except Exception:
        pass
    try:
        right_panel = getattr(self, "right_panel", None)
        if right_panel is not None:
            right_panel.rowconfigure(0, weight=1)
            right_panel.rowconfigure(1, weight=0)
        scroll_host = getattr(self, "right_scroll_host", None)
        if scroll_host is not None:
            scroll_host.grid_remove()
    except Exception:
        pass

    try:
        if str(graph_shell.winfo_manager()) == "pack":
            graph_shell.pack_forget()
    except Exception:
        pass

    pack_kwargs = {"fill": tk.BOTH, "expand": True, "pady": (0, 0)}

    try:
        graph_shell.pack(**pack_kwargs)
    except Exception:
        try:
            graph_shell.pack(**pack_kwargs)
        except Exception:
            return
    self._refresh_wizard_transition_graph()
    try:
        apply_graph_layout = getattr(self, "_apply_campaign_graph_fullscreen_layout", None)
        if callable(apply_graph_layout):
            apply_graph_layout()
    except Exception:
        pass
    try:
        if not bool(getattr(self, "_project_open_lightweight_refresh", False)):
            self.frame.after_idle(self._refresh_wizard_transition_graph)
            self.frame.after(120, self._refresh_wizard_transition_graph)
        self.frame.after_idle(self._sync_right_panel_scrollregion)
        self.frame.after_idle(self._sync_right_panel_canvas_width)
    except Exception:
        pass


def _set_pack_visibility(self, widget, visible: bool, **pack_kwargs):
    if widget is None:
        return
    try:
        manager = str(widget.winfo_manager())
    except Exception:
        manager = ""

    if visible:
        if manager != "pack":
            try:
                widget.pack(**pack_kwargs)
            except Exception:
                try:
                    fallback_kwargs = {
                        key: value
                        for key, value in dict(pack_kwargs or {}).items()
                        if key not in {"before", "after", "in_"}
                    }
                    widget.pack(**fallback_kwargs)
                except Exception:
                    pass
        return

    if manager == "pack":
        try:
            widget.pack_forget()
        except Exception:
            pass


def _set_grid_visibility(self, widget, visible: bool):
    if widget is None:
        return
    try:
        manager = str(widget.winfo_manager())
    except Exception:
        manager = ""

    if visible:
        if manager != "grid":
            try:
                widget.grid()
            except Exception:
                pass
        return

    if manager == "grid":
        try:
            widget.grid_remove()
        except Exception:
            pass


def _cancel_wizard_stage_curtain_animation(self, card: dict):
    if not isinstance(card, dict):
        return
    pending = card.get("curtain_after_id")
    if not pending:
        return
    try:
        self.frame.after_cancel(pending)
    except Exception:
        pass
    card["curtain_after_id"] = None


def _set_wizard_stage_curtain_height(self, card: dict, height: int):
    if not isinstance(card, dict):
        return
    clip = card.get("curtain_clip")
    toggle = card.get("curtain_toggle")
    if clip is None:
        return
    normalized_height = max(0, int(height))
    try:
        manager = str(clip.winfo_manager())
    except Exception:
        manager = ""
    if normalized_height <= 0:
        if manager == "pack":
            try:
                clip.pack_forget()
            except Exception:
                pass
    elif manager != "pack":
        try:
            clip.pack(fill=tk.X, pady=(0, 0), after=toggle)
        except Exception:
            try:
                clip.pack(fill=tk.X, pady=(0, 0))
            except Exception:
                pass
    try:
        clip.configure(height=normalized_height)
    except Exception:
        pass
    try:
        if normalized_height > 0:
            clip.pack_configure(pady=(6, 0))
    except Exception:
        pass
    try:
        self.frame.after_idle(self._sync_right_panel_scrollregion)
    except Exception:
        pass


def _animate_wizard_stage_curtain(self, card: dict, expand: bool):
    if not isinstance(card, dict):
        return

    clip = card.get("curtain_clip")
    inner = card.get("curtain_inner")
    shell = card.get("curtain_shell")
    action_row = card.get("action_row")
    if clip is None or inner is None or shell is None:
        return

    self._cancel_wizard_stage_curtain_animation(card)
    self._set_pack_visibility(shell, True, fill=tk.X, pady=(10, 0), before=action_row)
    if expand:
        try:
            if str(clip.winfo_manager()) != "pack":
                clip.pack(fill=tk.X, pady=(0, 0), after=card.get("curtain_toggle"))
        except Exception:
            try:
                clip.pack(fill=tk.X, pady=(0, 0))
            except Exception:
                pass

    try:
        inner.update_idletasks()
        clip.update_idletasks()
    except Exception:
        pass

    start_height = 0
    try:
        start_height = int(clip.cget("height") or 0)
    except Exception:
        try:
            start_height = int(clip.winfo_height() or 0)
        except Exception:
            start_height = 0

    target_height = 0
    if expand:
        try:
            target_height = max(0, int(inner.winfo_reqheight() or 0))
        except Exception:
            target_height = 0

    if start_height == target_height:
        self._set_wizard_stage_curtain_height(card, target_height)
        card["curtain_expanded"] = bool(expand)
        return

    total_steps = 10
    interval_ms = 18
    delta = target_height - start_height

    def _tick(step_index: int = 0):
        ratio = float(step_index + 1) / float(total_steps)
        eased = ratio * ratio * (3.0 - (2.0 * ratio))
        current_height = int(round(start_height + (delta * eased)))
        self._set_wizard_stage_curtain_height(card, current_height)
        if (step_index + 1) < total_steps:
            try:
                card["curtain_after_id"] = self.frame.after(interval_ms, lambda: _tick(step_index + 1))
            except Exception:
                card["curtain_after_id"] = None
        else:
            card["curtain_after_id"] = None
            self._set_wizard_stage_curtain_height(card, target_height)
            card["curtain_expanded"] = bool(expand)

    _tick(0)


def _toggle_wizard_stage_curtain(self, stage_key: str):
    card = self.wizard_stage_cards.get(str(stage_key or "").strip())
    if not isinstance(card, dict):
        return "break"
    if not bool(card.get("curtain_visible", False)):
        return "break"

    card["curtain_user_touched"] = True
    expand = not bool(card.get("curtain_expanded", False))
    card["curtain_expanded"] = expand
    self._refresh_wizard_stage_curtain_style(card)
    self._animate_wizard_stage_curtain(card, expand)
    return "break"


def _collapse_wizard_stage_curtain(self, stage_key: str, *, reset_user_touched: bool = True):
    card = self.wizard_stage_cards.get(str(stage_key or "").strip())
    if not isinstance(card, dict):
        return

    card["curtain_expanded"] = False
    if reset_user_touched:
        card["curtain_user_touched"] = False

    self._cancel_wizard_stage_curtain_animation(card)
    self._set_wizard_stage_curtain_height(card, 0)
    self._refresh_wizard_stage_curtain_style(card)


def _collapse_all_wizard_stage_curtains(self, *, reset_user_touched: bool = True):
    for stage_key in list(getattr(self, "wizard_stage_cards", {}).keys()):
        try:
            self._collapse_wizard_stage_curtain(stage_key, reset_user_touched=reset_user_touched)
        except Exception:
            pass


def _refresh_wizard_stage_curtain_style(self, card: dict):
    if not isinstance(card, dict):
        return

    palette = getattr(self.app, "palette", {})
    card_bg = palette.get("panel", "#252526")
    muted = palette.get("muted", "#c7c7c7")
    fg = palette.get("fg", "#f3f3f3")
    success = palette.get("success", "#27ae60")
    border = str(card.get("curtain_border", "") or palette.get("panel_border", palette.get("border", "#3c3c3c")))
    surface = str(card.get("curtain_surface", "") or palette.get("panel_alt", card_bg))
    expanded = bool(card.get("curtain_expanded", False))
    indicator_text = "▾" if expanded else "▸"

    for widget_name in ("curtain_shell", "curtain_clip", "curtain_inner"):
        widget = card.get(widget_name)
        if widget is None:
            continue
        try:
            widget.configure(bg=card_bg)
        except Exception:
            pass

    toggle = card.get("curtain_toggle")
    if toggle is not None:
        try:
            toggle.configure(
                bg=surface,
                highlightbackground=border,
                highlightcolor=border,
            )
        except Exception:
            pass

    for widget_name in ("curtain_text_col", "curtain_table"):
        widget = card.get(widget_name)
        if widget is None:
            continue
        try:
            widget.configure(bg=surface, highlightbackground=border, highlightcolor=border)
        except Exception:
            pass

    indicator = card.get("curtain_indicator")
    if indicator is not None:
        try:
            indicator.configure(text=indicator_text, bg=surface, fg=success)
        except Exception:
            pass

    title = card.get("curtain_title")
    if title is not None:
        try:
            title.configure(bg=surface, fg=fg)
        except Exception:
            pass

    meta = card.get("curtain_meta")
    if meta is not None:
        try:
            meta.configure(bg=surface, fg=muted)
        except Exception:
            pass


def _format_step1_selection_mode_label(*args, **kwargs):
    return campaign_step1_ingest._format_step1_selection_mode_label(*args, **kwargs)


def _get_step1_manifest_context(self, *args, **kwargs):
    return campaign_step1_ingest._get_step1_manifest_context(self, *args, **kwargs)


def _get_step1_manifest_context_lightweight(self, *args, **kwargs):
    return campaign_step1_ingest._get_step1_manifest_context_lightweight(self, *args, **kwargs)


def _build_step1_summary_payload(self, *args, **kwargs):
    return campaign_step1_ingest._build_step1_summary_payload(self, *args, **kwargs)


def _render_step1_stage_curtain(self, card: dict, status: WizardStageStatus, style: dict):
    if not isinstance(card, dict):
        return

    payload = {}
    if str(getattr(status, "key", "") or "").strip() == "step1":
        payload = self._build_step1_summary_payload()

    rows = list(payload.get("rows", []) or [])
    step1_mode = ""
    if str(getattr(status, "key", "") or "").strip() == "step1":
        step1_mode = self._get_step1_presentation_mode()
    visible = bool(rows) and step1_mode != "operational_assets"
    card["curtain_visible"] = visible

    shell = card.get("curtain_shell")
    clip = card.get("curtain_clip")
    inner = card.get("curtain_inner")
    table = card.get("curtain_table")
    action_row = card.get("action_row")
    content = card.get("content")
    if shell is None or clip is None or inner is None or table is None or content is None:
        return

    if not visible:
        card["curtain_expanded"] = False
        card["curtain_user_touched"] = False
        self._cancel_wizard_stage_curtain_animation(card)
        self._set_wizard_stage_curtain_height(card, 0)
        self._set_pack_visibility(shell, False)
        return

    if not bool(card.get("curtain_user_touched", False)):
        card["curtain_expanded"] = False

    border = str(style.get("border", "") or getattr(self.app, "palette", {}).get("panel_border", "#3c3c3c"))
    surface = blend_hex_colors(border, getattr(self.app, "palette", {}).get("panel", "#252526"), 0.92)
    card["curtain_border"] = border
    card["curtain_surface"] = surface

    title_lbl = card.get("curtain_title")
    meta_lbl = card.get("curtain_meta")
    if title_lbl is not None:
        try:
            title_lbl.configure(text=campaign_ui_helpers._repair_polish_text(str(payload.get("title", "Podsumowanie etapu") or "Podsumowanie etapu")))
        except Exception:
            pass
    if meta_lbl is not None:
        try:
            meta_lbl.configure(text=campaign_ui_helpers._repair_polish_text(str(payload.get("meta", "") or "")))
        except Exception:
            pass
        try:
            self._set_pack_visibility(meta_lbl, bool(str(payload.get("meta", "") or "").strip()), fill=tk.X, pady=(4, 0))
        except Exception:
            pass

    for child in list(table.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass

    palette = getattr(self.app, "palette", {})
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    panel_bg = palette.get("panel", "#252526")
    grid_border = blend_hex_colors(border, panel_bg, 0.34)
    header_bg = blend_hex_colors(surface, panel_bg, 0.18)

    try:
        table.grid_columnconfigure(0, weight=1)
        table.grid_columnconfigure(1, weight=0, minsize=120)
    except Exception:
        pass
    try:
        table.configure(
            bg=grid_border,
            highlightbackground=grid_border,
            highlightcolor=grid_border,
            highlightthickness=1,
        )
    except Exception:
        pass

    header_left = tk.Label(
        table,
        text="Pozycja",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        fg=muted,
        bg=header_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=grid_border,
        highlightcolor=grid_border,
        padx=10,
        pady=6,
    )
    header_left.grid(row=0, column=0, sticky="ew")

    header_right = tk.Label(
        table,
        text="Wartość",
        anchor="e",
        justify=tk.LEFT,
        font=("Segoe UI", 8, "bold"),
        fg=muted,
        bg=header_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=grid_border,
        highlightcolor=grid_border,
        padx=10,
        pady=6,
    )
    header_right.grid(row=0, column=1, sticky="ew")

    for row_idx, (label_text, value_text) in enumerate(rows, start=1):
        row_bg = blend_hex_colors(surface, panel_bg, 0.10 if ((row_idx - 1) % 2 == 0) else 0.18)

        key_lbl = tk.Label(
            table,
            text=campaign_ui_helpers._repair_polish_text(str(label_text or "").strip() or "-"),
            anchor="nw",
            justify=tk.LEFT,
            width=18,
            font=("Segoe UI", 9, "bold"),
            fg=muted,
            bg=row_bg,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=grid_border,
            highlightcolor=grid_border,
            padx=10,
            pady=8,
        )
        key_lbl.grid(row=row_idx, column=0, sticky="nsew")

        val_lbl = tk.Label(
            table,
            text=campaign_ui_helpers._repair_polish_text(str(value_text or "").strip() or "-"),
            anchor="e",
            justify=tk.RIGHT,
            font=("Segoe UI", 9, "bold"),
            fg=fg,
            bg=row_bg,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=grid_border,
            highlightcolor=grid_border,
            padx=10,
            pady=8,
        )
        val_lbl.grid(row=row_idx, column=1, sticky="nsew")

    self._set_pack_visibility(shell, True, fill=tk.X, pady=(10, 0), before=action_row)
    self._refresh_wizard_stage_curtain_style(card)
    try:
        inner.update_idletasks()
    except Exception:
        pass
    self._set_wizard_stage_curtain_height(
        card,
        int(inner.winfo_reqheight() or 0) if bool(card.get("curtain_expanded", False)) else 0,
    )


def _refresh_wizard_stage_metro(self, statuses: list[WizardStageStatus] | None = None):
    self._wizard_header_metro_statuses = list(statuses or [])
    try:
        notify = getattr(self.app, "notify_free_mode_assistant_context_changed", None)
        if callable(notify):
            notify()
    except Exception:
        pass
    return

    filtered_statuses = []
    for key in ("step1", "step2", "step3", "step4"):
        matched = None
        if statuses:
            for status in statuses:
                if getattr(status, "key", "") == key:
                    matched = status
                    break
        if matched is None:
            matched = WizardStageStatus(
                key=key,
                title="",
                state="locked",
                summary="",
                details="",
                is_current=False,
            )
        filtered_statuses.append(matched)

    self._wizard_header_metro_statuses = filtered_statuses
    self.frame.after_idle(self._draw_wizard_stage_metro)
    try:
        notify = getattr(self.app, "notify_free_mode_assistant_context_changed", None)
        if callable(notify):
            notify()
    except Exception:
        pass


def _draw_wizard_stage_metro(self, _event=None):
    return

    canvas = getattr(self, "wizard_header_metro_canvas", None)
    if canvas is None:
        return

    palette = getattr(self.app, "palette", {})
    panel_bg = str(canvas.cget("bg") or palette.get("panel", "#252526"))
    panel_alt = palette.get("panel_alt", "#2d2d30")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    muted_dim = palette.get("muted_dim", "#9a9a9a")
    border = blend_hex_colors(
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        panel_bg,
        0.28,
    )
    success = palette.get("success", "#27ae60")
    current_fill = blend_hex_colors(success, panel_bg, 0.16)
    neutral_outline = blend_hex_colors(fg, panel_bg, 0.34)
    soft_label = blend_hex_colors(muted, panel_bg, 0.18)

    canvas.delete("all")

    statuses = list(getattr(self, "_wizard_header_metro_statuses", []) or [])
    if not statuses:
        statuses = [
            WizardStageStatus(key=f"step{idx}", title="", state="locked")
            for idx in range(1, 5)
        ]

    width = max(int(canvas.winfo_width() or 720), 360)
    height = max(int(canvas.winfo_height() or 84), 84)
    line_y = 31
    circle_r = 14
    top_y = 0
    label_y = 58
    repair_text = campaign_ui_helpers._repair_polish_text
    label_map = {
        "step1": repair_text("Wejście"),
        "step2": "Tor",
        "step3": "Znaki",
        "step4": "Trening",
    }

    usable_left = 46
    usable_right = width - 46
    if usable_right <= usable_left:
        usable_left = 28
        usable_right = width - 28
    count = max(len(statuses), 2)
    step_gap = (usable_right - usable_left) / max(count - 1, 1)
    positions = [usable_left + idx * step_gap for idx in range(len(statuses))]

    current_index = next((idx for idx, status in enumerate(statuses) if bool(getattr(status, "is_current", False))), None)
    if current_index is None and statuses and all(str(getattr(status, "state", "") or "").strip().lower() in {"done", "skipped"} for status in statuses):
        current_index = len(statuses) - 1

    marker_text = "W TOKU"
    if current_index is not None and 0 <= current_index < len(statuses):
        current_state = str(getattr(statuses[current_index], "state", "") or "").strip().lower()
        if current_state == "skipped":
            marker_text = "POMINIETY"
        elif current_state == "done":
            marker_text = "GOTOWE"

    def station_style(status: WizardStageStatus, *, highlighted: bool) -> dict:
        state_key = str(getattr(status, "state", "") or "").strip().lower()
        if state_key == "skipped":
            return {"fill": panel_bg, "outline": muted_dim, "text": muted_dim, "label": muted_dim, "dash": (3, 2)}
        if highlighted:
            return {"fill": current_fill, "outline": success, "text": fg, "label": success, "dash": None}
        if state_key == "done":
            return {"fill": panel_bg, "outline": success, "text": fg, "label": success, "dash": None}
        return {"fill": panel_bg, "outline": neutral_outline, "text": fg, "label": soft_label, "dash": None}

    if PIL_AVAILABLE and Image is not None and ImageTk is not None and ImageDraw is not None and ImageFont is not None:
        hi_pixel_budget = 4_500_000
        scale = 4 if (width * height) <= 180_000 else 2
        hi_w = max(width * scale, 4)
        hi_h = max(height * scale, 4)
        can_use_pil_metro = (hi_w * hi_h) <= hi_pixel_budget and width <= 2400 and height <= 280
        if can_use_pil_metro:
            try:
                image = Image.new("RGBA", (hi_w, hi_h), self._hex_to_rgba(panel_bg))
                draw = ImageDraw.Draw(image, "RGBA")
                font_marker = self._get_pil_font(10 * scale, bold=False)
                font_step = self._get_pil_font(10 * scale, bold=False)
                font_label = self._get_pil_font(10 * scale, bold=False)
                font_skipped = self._get_pil_font(9 * scale, bold=False)

                def s(value: float) -> int:
                    return int(round(float(value) * scale))

                for idx in range(len(positions) - 1):
                    segment_color = border
                    if current_index is not None:
                        if idx < current_index:
                            segment_color = success
                    elif statuses and all(str(getattr(status, "state", "") or "").strip().lower() in {"done", "skipped"} for status in statuses):
                        segment_color = success

                    draw.line(
                        [(s(positions[idx] + circle_r), s(line_y)), (s(positions[idx + 1] - circle_r), s(line_y))],
                        fill=self._hex_to_rgba(segment_color),
                        width=max(2, s(2)),
                    )

                marker_bbox = None
                if current_index is not None and 0 <= current_index < len(positions):
                    current_status = statuses[current_index]
                    current_style = station_style(current_status, highlighted=True)
                    marker_x = s(positions[current_index])
                    marker_y = s(top_y)
                    try:
                        bbox = draw.textbbox((0, 0), marker_text, font=font_marker)
                        text_w = max(1, bbox[2] - bbox[0])
                        text_h = max(1, bbox[3] - bbox[1])
                    except Exception:
                        text_w = 24 * scale
                        text_h = 8 * scale
                    text_pos = (marker_x - text_w // 2, marker_y)
                    draw.text(text_pos, marker_text, font=font_marker, fill=self._hex_to_rgba(current_style["outline"]))
                    marker_bbox = (text_pos[0], text_pos[1], text_pos[0] + text_w, text_pos[1] + text_h)
                    marker_line_top = marker_bbox[3] + (6 * scale)
                    marker_line_bottom = s(line_y - circle_r - 4)
                    if marker_line_bottom > marker_line_top:
                        draw.line(
                            [(marker_x, marker_line_top), (marker_x, marker_line_bottom)],
                            fill=self._hex_to_rgba(current_style["outline"]),
                            width=max(2, s(2)),
                        )

                for idx, status in enumerate(statuses):
                    state_key = str(getattr(status, "state", "") or "").strip().lower()
                    if state_key != "skipped" or idx == current_index:
                        continue
                    skipped_style = station_style(status, highlighted=False)
                    skipped_text = "POMINIETO"
                    try:
                        bbox = draw.textbbox((0, 0), skipped_text, font=font_skipped)
                        text_w = max(1, bbox[2] - bbox[0])
                    except Exception:
                        text_w = 30 * scale
                    draw.text(
                        (s(positions[idx]) - text_w // 2, s(top_y + 2)),
                        skipped_text,
                        font=font_skipped,
                        fill=self._hex_to_rgba(skipped_style["label"]),
                    )

                for idx, status in enumerate(statuses):
                    x = positions[idx]
                    highlighted = bool(current_index == idx)
                    style = station_style(status, highlighted=highlighted)

                    draw.ellipse(
                        [s(x - circle_r), s(line_y - circle_r), s(x + circle_r), s(line_y + circle_r)],
                        fill=self._hex_to_rgba(style["fill"]),
                        outline=self._hex_to_rgba(style["outline"]),
                        width=max(2, s(2)),
                    )

                    step_text = f"E{idx + 1}"
                    try:
                        bbox = draw.textbbox((0, 0), step_text, font=font_step)
                        step_x = s(x) - ((bbox[0] + bbox[2]) / 2.0)
                        step_y = s(line_y) - ((bbox[1] + bbox[3]) / 2.0)
                    except Exception:
                        step_x = s(x) - (7 * scale)
                        step_y = s(line_y) - (4 * scale)
                    draw.text(
                        (step_x, step_y),
                        step_text,
                        font=font_step,
                        fill=self._hex_to_rgba(style["text"]),
                    )

                    label_text_local = label_map.get(getattr(status, "key", ""), f"E{idx + 1}")
                    label_color = style["label"] if highlighted or str(getattr(status, "state", "") or "").strip().lower() in {"done", "in_progress", "needs_attention"} else soft_label
                    try:
                        bbox = draw.textbbox((0, 0), label_text_local, font=font_label)
                        label_w = max(1, bbox[2] - bbox[0])
                    except Exception:
                        label_w = 24 * scale
                    draw.text(
                        (s(x) - label_w // 2, s(label_y)),
                        label_text_local,
                        font=font_label,
                        fill=self._hex_to_rgba(label_color),
                    )

                try:
                    resampling = Image.Resampling.LANCZOS
                except Exception:
                    resampling = Image.LANCZOS
                image = image.resize((width, height), resampling)
                self._wizard_header_metro_photo = ImageTk.PhotoImage(image)
                canvas.create_image(0, 0, anchor=tk.NW, image=self._wizard_header_metro_photo)
                return
            except MemoryError:
                logger.warning(
                    "Wizard metro PIL fallback po MemoryError: width=%s height=%s scale=%s hi=%sx%s",
                    width,
                    height,
                    scale,
                    hi_w,
                    hi_h,
                )
            except Exception as exc:
                logger.debug("Wizard metro PIL fallback do Canvas: %s", exc)

    for idx in range(len(positions) - 1):
        segment_color = border
        if current_index is not None:
            if idx < current_index:
                segment_color = success
        elif statuses and all(str(getattr(status, "state", "") or "").strip().lower() in {"done", "skipped"} for status in statuses):
            segment_color = success

        canvas.create_line(
            positions[idx] + circle_r,
            line_y,
            positions[idx + 1] - circle_r,
            line_y,
            fill=segment_color,
            width=2,
            capstyle=tk.ROUND,
        )

    if current_index is not None and 0 <= current_index < len(positions):
        current_status = statuses[current_index]
        current_style = station_style(current_status, highlighted=True)
        marker_id = canvas.create_text(
            positions[current_index],
            top_y,
            text=marker_text,
            fill=current_style["outline"],
            font=("Segoe UI", 12, "normal"),
            anchor=tk.N,
        )
        try:
            marker_bbox = canvas.bbox(marker_id)
        except Exception:
            marker_bbox = None

        marker_line_top = (marker_bbox[3] + 6) if marker_bbox else (top_y + 14)
        marker_line_bottom = line_y - circle_r - 4
        if marker_line_bottom > marker_line_top:
            canvas.create_line(
                positions[current_index],
                marker_line_top,
                positions[current_index],
                marker_line_bottom,
                fill=current_style["outline"],
                width=2,
            )

    for idx, status in enumerate(statuses):
        state_key = str(getattr(status, "state", "") or "").strip().lower()
        if state_key != "skipped" or idx == current_index:
            continue
        skipped_style = station_style(status, highlighted=False)
        canvas.create_text(
            positions[idx],
            top_y + 2,
            text="POMINIETO",
            fill=skipped_style["label"],
            font=("Segoe UI", 11, "normal"),
            anchor=tk.N,
        )

    for idx, status in enumerate(statuses):
        x = positions[idx]
        highlighted = bool(current_index == idx)
        style = station_style(status, highlighted=highlighted)

        oval_id = canvas.create_oval(
            x - circle_r,
            line_y - circle_r,
            x + circle_r,
            line_y + circle_r,
            fill=style["fill"],
            outline=style["outline"],
            width=2,
        )
        if style["dash"]:
            canvas.itemconfigure(oval_id, dash=style["dash"])

        canvas.create_text(
            x,
            line_y,
            text=f"E{idx + 1}",
            fill=style["text"],
            font=("Segoe UI", 11, "normal"),
        )
        canvas.create_text(
            x,
            label_y,
            text=label_map.get(getattr(status, "key", ""), f"E{idx + 1}"),
            fill=(style["label"] if highlighted or str(getattr(status, "state", "") or "").strip().lower() in {"done", "in_progress", "needs_attention"} else soft_label),
            font=("Segoe UI", 11, "normal"),
        )


def _build_wizard_stage_card(self, parent, key: str, *, help_key: str | None = None):
    palette = getattr(self.app, "palette", {})
    card_bg = palette.get("panel", "#252526")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    shell = tk.Frame(parent, bg=border, bd=0, highlightthickness=0)
    shell.pack(fill=tk.X, pady=(0, 12))

    content = tk.Frame(shell, bg=card_bg, padx=14, pady=12)
    content.pack(fill=tk.X, padx=1, pady=1)

    header_row = tk.Frame(content, bg=card_bg, bd=0, highlightthickness=0)
    header_row.pack(fill=tk.X)

    title_lbl = tk.Label(
        header_row,
        text="",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 12, "bold"),
        fg=palette.get("fg", "#f3f3f3"),
        bg=card_bg,
    )
    title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    header_status_col = tk.Frame(header_row, bg=card_bg, bd=0, highlightthickness=0)
    header_status_col.pack(side=tk.RIGHT, padx=(10, 0))
    header_status_col.grid_columnconfigure(0, weight=1)

    badge_lbl = tk.Label(
        header_status_col,
        text="",
        anchor="e",
        justify=tk.RIGHT,
        font=("Segoe UI", 8, "bold"),
        padx=8,
        pady=3,
        bd=0,
        highlightthickness=0,
    )
    badge_lbl.grid(row=0, column=0, sticky="e")

    summary_lbl = tk.Label(
        content,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=760,
        fg=palette.get("fg", "#f3f3f3"),
        bg=card_bg,
    )
    summary_lbl.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    details_lbl = tk.Label(
        content,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=760,
        fg=palette.get("muted", "#c7c7c7"),
        bg=card_bg,
    )
    details_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

    action_row = tk.Frame(content, bg=card_bg, bd=0, highlightthickness=0)
    action_row.pack(fill=tk.X, pady=(10, 0))

    primary_btn = ttk.Button(action_row, text="")
    secondary_btn = ttk.Button(action_row, text="")

    curtain_shell = tk.Frame(content, bg=card_bg, bd=0, highlightthickness=0)

    curtain_toggle = tk.Frame(
        curtain_shell,
        bg=card_bg,
        bd=0,
        highlightthickness=1,
        cursor="hand2",
        padx=10,
        pady=8,
    )
    curtain_toggle.pack(fill=tk.X)

    curtain_indicator = tk.Label(
        curtain_toggle,
        text="▸",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 10, "bold"),
        bg=card_bg,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    curtain_indicator.pack(side=tk.LEFT)

    curtain_text_col = tk.Frame(curtain_toggle, bg=card_bg, bd=0, highlightthickness=0, cursor="hand2")
    curtain_text_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    curtain_title_lbl = tk.Label(
        curtain_text_col,
        text="",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bg=card_bg,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    curtain_title_lbl.pack(anchor=tk.W, fill=tk.X)

    curtain_meta_lbl = tk.Label(
        curtain_text_col,
        text="",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
        bg=card_bg,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    curtain_meta_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

    curtain_clip = tk.Frame(curtain_shell, bg=card_bg, bd=0, highlightthickness=0, height=0)
    curtain_clip.pack(fill=tk.X, pady=(6, 0))
    curtain_clip.pack_propagate(False)

    curtain_inner = tk.Frame(curtain_clip, bg=card_bg, bd=0, highlightthickness=0)
    curtain_inner.pack(fill=tk.X)

    curtain_table = tk.Frame(
        curtain_inner,
        bg=card_bg,
        bd=0,
        highlightthickness=1,
    )
    curtain_table.pack(fill=tk.X)

    body = tk.Frame(content, bg=card_bg, bd=0, highlightthickness=0)

    badge_cta_shell = tk.Frame(
        action_row,
        bg=card_bg,
        bd=0,
        highlightthickness=1,
        padx=1,
        pady=1,
    )
    badge_cta_btn = tk.Label(
        badge_cta_shell,
        text="ZATWIERDŹ ETAP",
        bd=0,
        highlightthickness=1,
        padx=14,
        pady=5,
        cursor="hand2",
        font=("Segoe UI Semibold", 9),
        anchor="center",
        justify=tk.CENTER,
    )
    badge_cta_btn.pack(anchor=tk.E)

    if help_key:
        for widget in (shell, content, header_row, header_status_col, title_lbl, summary_lbl, details_lbl, body, curtain_shell, curtain_toggle, curtain_table, badge_cta_shell, badge_cta_btn):
            HELP.bind_help(widget, help_key)

    for widget in (
        shell,
        content,
        header_row,
        header_status_col,
        title_lbl,
        summary_lbl,
        details_lbl,
        action_row,
        body,
        curtain_shell,
        curtain_toggle,
        badge_cta_shell,
        badge_cta_btn,
    ):
        try:
            widget.bind("<Enter>", lambda _event, stage_key=key: self._set_wizard_assistant_stage_context(stage_key), add="+")
        except Exception:
            pass

    card = {
        "key": key,
        "shell": shell,
        "content": content,
        "header_row": header_row,
        "header_status_col": header_status_col,
        "title": title_lbl,
        "badge": badge_lbl,
        "badge_cta_shell": badge_cta_shell,
        "badge_cta_btn": badge_cta_btn,
        "summary": summary_lbl,
        "details": details_lbl,
        "action_row": action_row,
        "primary_btn": primary_btn,
        "secondary_btn": secondary_btn,
        "body": body,
        "curtain_shell": curtain_shell,
        "curtain_toggle": curtain_toggle,
        "curtain_indicator": curtain_indicator,
        "curtain_text_col": curtain_text_col,
        "curtain_title": curtain_title_lbl,
        "curtain_meta": curtain_meta_lbl,
        "curtain_clip": curtain_clip,
        "curtain_inner": curtain_inner,
        "curtain_table": curtain_table,
        "curtain_visible": False,
        "curtain_expanded": False,
        "curtain_user_touched": False,
        "curtain_after_id": None,
        "status": None,
    }

    for widget in (curtain_toggle, curtain_indicator, curtain_text_col, curtain_title_lbl, curtain_meta_lbl):
        try:
            widget.bind("<Button-1>", lambda _event, stage_key=key: self._toggle_wizard_stage_curtain(stage_key), add="+")
        except Exception:
            pass

    return card


def _is_wizard_stage_emphasized(self, status: WizardStageStatus) -> bool:
    state_key = str(getattr(status, "state", "") or "").strip().lower()
    if bool(getattr(status, "is_current", False)):
        return True
    return state_key in {"in_progress", "needs_attention"}


def _get_wizard_stage_state_style(self, state: str, *, is_current: bool = False, emphasized: bool = True) -> dict:
    palette = getattr(self.app, "palette", {})
    state_key = str(state or "").strip().lower()
    style_map = {
        "locked": {
            "label": "ZABLOKOWANE",
            "border": palette.get("panel_border", palette.get("border", "#3c3c3c")),
            "badge_bg": palette.get("panel_alt", "#2f3136"),
            "badge_fg": palette.get("muted", "#c7c7c7"),
            "title_fg": palette.get("muted", "#c7c7c7"),
            "summary_fg": palette.get("muted", "#c7c7c7"),
            "details_fg": palette.get("muted_dim", "#9a9a9a"),
        },
        "ready": {
            "label": "GOTOWE",
            "border": palette.get("success", "#27ae60"),
            "badge_bg": palette.get("surface_success", palette.get("panel_alt", "#1f3320")),
            "badge_fg": palette.get("success", "#27ae60"),
            "title_fg": palette.get("fg", "#f3f3f3"),
            "summary_fg": palette.get("fg", "#f3f3f3"),
            "details_fg": palette.get("muted", "#c7c7c7"),
        },
        "in_progress": {
            "label": "W TOKU",
            "border": palette.get("success", "#27ae60"),
            "badge_bg": palette.get("surface_success", palette.get("panel_alt", "#1f3320")),
            "badge_fg": palette.get("success", "#27ae60"),
            "title_fg": palette.get("fg", "#f3f3f3"),
            "summary_fg": palette.get("fg", "#f3f3f3"),
            "details_fg": palette.get("muted", "#c7c7c7"),
        },
        "needs_attention": {
            "label": "UWAGA",
            "border": palette.get("warning", "#d35400"),
            "badge_bg": palette.get("surface_warning", palette.get("panel_alt", "#3a2323")),
            "badge_fg": palette.get("warning", "#d35400"),
            "title_fg": palette.get("fg", "#f3f3f3"),
            "summary_fg": palette.get("fg", "#f3f3f3"),
            "details_fg": palette.get("muted", "#c7c7c7"),
        },
        "done": {
            "label": "GOTOWE",
            "border": blend_hex_colors(
                palette.get("success", "#27ae60"),
                palette.get("panel_border", palette.get("border", "#3c3c3c")),
                0.78,
            ),
            "badge_bg": blend_hex_colors(
                palette.get("surface_success", palette.get("panel_alt", "#1f3320")),
                palette.get("panel_alt", "#2f3136"),
                0.42,
            ),
            "badge_fg": palette.get("success", "#27ae60"),
            "title_fg": palette.get("fg", "#f3f3f3"),
            "summary_fg": palette.get("fg", "#f3f3f3"),
            "details_fg": palette.get("muted", "#c7c7c7"),
        },
        "skipped": {
            "label": "POMINIETE",
            "border": palette.get("panel_border", palette.get("border", "#3c3c3c")),
            "badge_bg": palette.get("panel_alt", "#2f3136"),
            "badge_fg": palette.get("muted_dim", "#9a9a9a"),
            "title_fg": palette.get("muted", "#c7c7c7"),
            "summary_fg": palette.get("muted", "#c7c7c7"),
            "details_fg": palette.get("muted_dim", "#9a9a9a"),
        },
    }
    base = dict(style_map.get(state_key, style_map["locked"]))
    if is_current and state_key in {"ready", "in_progress", "needs_attention"}:
        base["border"] = palette.get("success", "#27ae60")
    if not emphasized:
        panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        panel_alt = palette.get("panel_alt", "#2f3136")
        base["border"] = blend_hex_colors(base.get("border", panel_border), panel_border, 0.82)
        base["badge_bg"] = blend_hex_colors(base.get("badge_bg", panel_alt), panel_alt, 0.42)
        base["badge_fg"] = palette.get("muted_dim", "#9a9a9a")
        base["title_fg"] = palette.get("muted", "#c7c7c7")
        base["summary_fg"] = palette.get("muted_dim", "#9a9a9a")
        base["details_fg"] = palette.get("muted_dim", "#9a9a9a")
    return base


def _configure_wizard_stage_button(self, button, label: str, command, *, side=tk.LEFT, padx=(0, 0), debug_id: str = ""):
    if button is None:
        return
    label_text = campaign_ui_helpers._repair_polish_text(str(label or "").strip())
    if not label_text or command is None:
        try:
            button.configure(text="", command=(lambda: None), state="disabled")
        except Exception:
            pass
        try:
            button.unbind("<ButtonPress-1>")
            button.unbind("<ButtonPress-3>")
        except Exception:
            pass
        self._set_pack_visibility(button, False)
        return

    debug_suffix = f" ({debug_id})" if str(debug_id or "").strip() else ""

    def _invoke_stage_button(event=None, _command=command, _button=button, _debug_id=debug_id):
        now = perf_counter()
        try:
            last_invoke = float(getattr(_button, "_wizard_stage_last_invoke_at", 0.0) or 0.0)
        except Exception:
            last_invoke = 0.0
        if now - last_invoke < 0.25:
            return "break" if event is not None else None
        try:
            setattr(_button, "_wizard_stage_last_invoke_at", now)
        except Exception:
            pass
        try:
            _command()
        except Exception as e:
            logger.error(f"Nie udało się wykonać akcji panelu wizarda {_debug_id}: {e}")
        return "break" if event is not None else None

    try:
        button.configure(text=campaign_ui_helpers._repair_polish_text(f"{label_text}{debug_suffix}"), command=_invoke_stage_button, state="normal")
        # Standardowy ttk.Button odpala komendę dopiero przy puszczeniu LPM.
        # Przy intensywnym odświeżaniu kart pierwszy klik potrafił zostać
        # zużyty przez focus/przerysowanie, więc dla CTA etapów startujemy
        # akcję już na naciśnięciu. PPM traktujemy tak samo dla tych CTA.
        button.bind("<ButtonPress-1>", _invoke_stage_button)
        button.bind("<ButtonPress-3>", _invoke_stage_button)
    except Exception:
        pass
    self._set_pack_visibility(button, True, side=side, padx=padx)


def _configure_wizard_stage_badge_cta(self, card: dict, *, visible: bool, command=None, label: str = ""):
    if not card:
        return
    shell = card.get("badge_cta_shell")
    button = card.get("badge_cta_btn")
    if shell is None or button is None:
        return

    if not visible or command is None:
        try:
            button.unbind("<Button-1>")
            button.unbind("<Enter>")
            button.unbind("<Leave>")
        except Exception:
            pass
        try:
            shell.pack_forget()
        except Exception:
            pass
        try:
            shell.grid_remove()
        except Exception:
            pass
        return

    palette = getattr(self.app, "palette", {})
    success = palette.get("success", "#27ae60")
    surface_success = palette.get("surface_success", "#1f3320")
    panel_bg = palette.get("panel", "#252526")
    glow = blend_hex_colors(success, surface_success, 0.10)
    border = blend_hex_colors(success, panel_bg, 0.08)
    fill = blend_hex_colors(surface_success, panel_bg, 0.08)
    hover_fill = blend_hex_colors(fill, success, 0.18)
    hover_border = blend_hex_colors(success, palette.get("fg", "#f3f3f3"), 0.18)
    text_color = palette.get("fg", "#f3f3f3")
    button_label = campaign_ui_helpers._repair_polish_text(str(label or "").strip() or "Zatwierdź etap").upper()

    try:
        shell.configure(
            bg=glow,
            highlightbackground=border,
            highlightcolor=border,
            highlightthickness=1,
        )
    except Exception:
        pass
    try:
        button.configure(
            text=button_label,
            bg=fill,
            fg=text_color,
            activebackground=hover_fill,
            activeforeground=text_color,
            highlightbackground=border,
            highlightcolor=border,
            highlightthickness=1,
            padx=14,
            pady=5,
            font=("Segoe UI Semibold", 9),
        )
    except Exception:
        pass
    try:
        button.unbind("<Button-1>")
        button.unbind("<Enter>")
        button.unbind("<Leave>")
    except Exception:
        pass
    try:
        button.bind("<Button-1>", lambda _event, cmd=command: cmd(), add="+")
        button.bind(
            "<Enter>",
            lambda _event, widget=button, shell_widget=shell: (
                shell_widget.configure(highlightbackground=hover_border, highlightcolor=hover_border),
                widget.configure(bg=hover_fill),
            ),
            add="+",
        )
        button.bind(
            "<Leave>",
            lambda _event, widget=button, shell_widget=shell: (
                shell_widget.configure(highlightbackground=border, highlightcolor=border),
                widget.configure(bg=fill),
            ),
            add="+",
        )
    except Exception:
        pass
    try:
        if str(shell.winfo_manager()) == "pack":
            shell.pack_forget()
        elif str(shell.winfo_manager()) == "grid":
            shell.grid_remove()
        shell.pack(side=tk.RIGHT, padx=(8, 0))
        shell.lift()
    except Exception:
        pass


def _get_step2_render_disk_approval_fallback(self, iteration_target: str = "") -> dict:
    target = self._normalize_iteration_target(iteration_target)
    if target not in {"plate", "char"}:
        try:
            target = self._normalize_iteration_target(CAMPAIGN.get_iteration_target())
        except Exception:
            target = ""
    if target not in {"plate", "char"}:
        return {}

    try:
        return dict(
            self._get_step2_disk_approval_fallback(
                iteration_target=target,
                plate_ready_source={},
                char_ready_source={},
            )
            or {}
        )
    except Exception as e:
        logger.debug(f"Nie udało się sprawdzić trwałej bramki E2 z dysku: {e}")
        return {}


def _force_current_step2_badge_from_disk(self) -> None:
    try:
        if not str(CAMPAIGN.get_active_project_name() or "").strip():
            return
        if str(CAMPAIGN.get_project_status() or "active").strip().lower() in {"completed", "paused"}:
            return
        if int(CAMPAIGN.get_current_step() or 0) != 2:
            return
        if str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved":
            return
        if str(CAMPAIGN.get_step2_status() or "").strip().lower() == "approved":
            return
        iteration_target = self._normalize_iteration_target(CAMPAIGN.get_iteration_target())
    except Exception:
        return

    if iteration_target not in {"plate", "char"}:
        return

    disk_fallback = self._get_step2_render_disk_approval_fallback(iteration_target)
    if not bool(disk_fallback.get("ready")):
        return

    next_stage_label = "E4T" if iteration_target == "plate" else "E3"
    summary = "Etap 2 możesz już zamknąć, ale warto jeszcze rozważyć dopisanie tablic w Z2."
    details = (
        f"Minimalny próg projektu jest już spełniony, więc możesz od razu zamknąć E2 "
        f"i odblokować {next_stage_label}. Jeśli chcesz wzmocnić zbiór projektu, dopisz jeszcze "
        "kilka poprawnych anotacji tablic w Z2."
    )
    forced_status = None
    statuses = [
        status
        for status in list(getattr(self, "_wizard_header_metro_statuses", []) or [])
        if isinstance(status, WizardStageStatus)
    ]
    for index, status in enumerate(statuses):
        if str(getattr(status, "key", "") or "").strip().lower() != "step2":
            continue
        forced_status = WizardStageStatus(
            key="step2",
            title=str(getattr(status, "title", "") or "E2. Tablice"),
            state="ready",
            summary=summary,
            details=details,
            primary_label="Dodaj jeszcze tablice w Z2",
            primary_command=self._step_goto_auto_annotation,
            badge_action_label="Zatwierdź etap",
            badge_action_command=self._approve_step2_from_wizard,
            visible=bool(getattr(status, "visible", True)),
            is_current=True,
        )
        statuses[index] = forced_status
        break

    if forced_status is None:
        forced_status = WizardStageStatus(
            key="step2",
            title="E2. Tablice",
            state="ready",
            summary=summary,
            details=details,
            primary_label="Dodaj jeszcze tablice w Z2",
            primary_command=self._step_goto_auto_annotation,
            badge_action_label="Zatwierdź etap",
            badge_action_command=self._approve_step2_from_wizard,
            is_current=True,
        )
        statuses.append(forced_status)

    if statuses:
        self._wizard_header_metro_statuses = statuses
        try:
            self._refresh_wizard_stage_metro(statuses)
        except Exception:
            pass

    card = dict(getattr(self, "wizard_stage_cards", {}) or {}).get("step2")
    if card is not None:
        try:
            self._apply_wizard_stage_status(card, forced_status)
        except Exception as e:
            logger.debug(f"Nie udało się wymusić trwałego badge'a E2 na karcie wizarda: {e}")


def _get_persistent_wizard_stage_badge_override(self, status: WizardStageStatus) -> tuple[str, object | None]:
    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        return "", None
    key = str(getattr(status, "key", "") or "").strip().lower()
    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
        step1_status = str(CAMPAIGN.get_step1_status() or "").strip().lower()
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
        step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
        iteration_target = self._normalize_iteration_target(CAMPAIGN.get_iteration_target())
        project_status = str(CAMPAIGN.get_project_status() or "active").strip().lower()
    except Exception:
        return "", None

    if project_status in {"completed", "paused"}:
        return "", None

    if key == "step1":
        if current_step != 1 or step1_status == "approved" or iteration_target not in {"plate", "char"}:
            return "", None
        try:
            summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
            summary_count = int(summary.get("selected_total") or summary.get("raw_total") or 0)
            if summary_count > 0:
                has_images = True
            else:
                master_pool = CAMPAIGN.get_master_pool_dir()
                has_images = bool(master_pool and Path(master_pool).exists() and Path(master_pool).is_dir())
        except Exception:
            has_images = False
        return ("Zatwierdź etap", self._approve_step1_from_wizard) if has_images else ("", None)

    if key == "step2":
        if (
            current_step != 2
            or step1_status != "approved"
            or step2_status == "approved"
            or iteration_target not in {"plate", "char"}
        ):
            return "", None

        try:
            disk_fallback = self._get_step2_render_disk_approval_fallback(iteration_target)
        except Exception as e:
            logger.debug(f"Nie udało się sprawdzić trwałej bramki E2 przy renderze badge'a: {e}")
            disk_fallback = {}

        if bool(disk_fallback.get("ready")):
            return "Zatwierdź etap", self._approve_step2_from_wizard
        return "", None

    if key == "step3":
        if current_step != 3 or iteration_target != "char" or step2_status != "approved":
            return "", None
        if step3_status == "ready":
            return "Zatwierdź etap", self._approve_step3_from_wizard
        if step3_status in {"pending", "needs_rework"}:
            try:
                readiness = self._detect_campaign_char_ready_dataset_state()
                if bool(readiness.get("ok")) and int(readiness.get("perfect_count", 0) or 0) > 0:
                    return "Zatwierdź etap", self._approve_step3_from_wizard
            except Exception as e:
                logger.debug(f"Nie udało się sprawdzić trwałej bramki E3 przy renderze badge'a: {e}")
        return "", None

    if key == "step4":
        if current_step != 4 or iteration_target not in {"plate", "char"}:
            return "", None
        try:
            training_tab = self.app.tabs.get("training") if getattr(self.app, "tabs", None) else None
            if training_tab is not None and hasattr(training_tab, "get_campaign_step4_finish_state"):
                finish_state = training_tab.get_campaign_step4_finish_state(iteration_target=iteration_target) or {}
            else:
                finish_state = CAMPAIGN.get_step4_finish_state() or {}
        except Exception:
            finish_state = {}
        try:
            finish_iteration = int(finish_state.get("iteration", 0) or 0)
        except Exception:
            finish_iteration = 0
        finish_target = self._normalize_iteration_target(finish_state.get("target", "")) or iteration_target
        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            current_iteration = 0
        if (
            bool(finish_state.get("ready", False))
            and finish_target == iteration_target
            and finish_iteration == current_iteration
        ):
            return "Zatwierdź etap", self._finish_step4_iteration
        return "Zakończ etap bez treningu", self._finish_step4_without_training

    return "", None


def _stabilize_wizard_stage_badges(self) -> None:
    return

    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        # Lekkie otwarcie projektu ma już świeży snapshot statusów z
        # _refresh_wizard_active_dashboard(). Pełna stabilizacja ponownie liczy
        # bramki, źródła i gotowość zakładek, co przy NEON potrafiło zamrażać UI.
        return
    if not str(CAMPAIGN.get_active_project_name() or "").strip():
        return

    statuses = []
    try:
        self._clear_dashboard_perf_cache()
        active_state = self._build_active_project_dashboard_state()
        if active_state:
            statuses = self._get_wizard_stage_statuses(
                active_project=str(active_state.get("active_project", "") or "").strip(),
                current_step=int(active_state.get("current_step", 1) or 1),
                iteration_target=str(active_state.get("iteration_target", "") or "").strip().lower(),
                project_status=str(active_state.get("project_status", "active") or "active").strip().lower(),
                project_paused_at=str(active_state.get("project_paused_at", "") or "").strip(),
                project_completed_at=str(active_state.get("project_completed_at", "") or "").strip(),
                step1_status=str(active_state.get("step1_status", "") or "").strip().lower(),
                step2_status=str(active_state.get("step2_status", "") or "").strip().lower(),
                step3_status=str(active_state.get("step3_status", "") or "").strip().lower(),
            )
    except Exception as e:
        logger.debug(f"Nie udało się ponownie przeliczyć badge'y etapów wizarda: {e}")

    if not statuses:
        statuses = [
            status
            for status in list(getattr(self, "_wizard_header_metro_statuses", []) or [])
            if isinstance(status, WizardStageStatus)
        ]
    if not statuses:
        return

    self._wizard_header_metro_statuses = list(statuses)
    try:
        self._refresh_wizard_stage_metro(statuses)
    except Exception:
        pass

    for status in statuses:
        card = self.wizard_stage_cards.get(str(getattr(status, "key", "") or ""))
        if card is None:
            continue
        try:
            self._apply_wizard_stage_status(card, status)
            continue
        except Exception as e:
            logger.debug(f"Nie udało się ustabilizować karty etapu {getattr(status, 'key', '')}: {e}")
        badge_label = str(getattr(status, "badge_action_label", "") or "").strip()
        badge_command = getattr(status, "badge_action_command", None)
        if not badge_label or badge_command is None:
            badge_label, badge_command = self._get_persistent_wizard_stage_badge_override(status)
        self._configure_wizard_stage_badge_cta(
            card,
            visible=bool(getattr(status, "visible", True) and badge_label and badge_command is not None),
            command=badge_command,
            label=badge_label,
        )

    try:
        self.frame.update_idletasks()
    except Exception:
        pass
    try:
        self.frame.after_idle(self._sync_right_panel_scrollregion)
    except Exception:
        pass
    try:
        self._force_current_step2_badge_from_disk()
    except Exception as e:
        logger.debug(f"Nie udało się wymusić badge'a E2 z trwałego stanu projektu: {e}")


def _schedule_wizard_stage_badge_stabilization(self, delay_ms: int = 140) -> None:
    return

    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        return
    try:
        pending = getattr(self, "_wizard_stage_badge_stabilize_after_id", None)
        if pending:
            self.frame.after_cancel(pending)
    except Exception:
        pass

    def _run() -> None:
        self._wizard_stage_badge_stabilize_after_id = None
        try:
            self._stabilize_wizard_stage_badges()
        except Exception as e:
            logger.debug(f"Nie udało się ustabilizować badge zatwierdzania etapu: {e}")

    try:
        self._wizard_stage_badge_stabilize_after_id = self.frame.after(max(1, int(delay_ms or 1)), _run)
    except Exception:
        self._wizard_stage_badge_stabilize_after_id = None


def _apply_wizard_stage_status(self, card: dict, status: WizardStageStatus):
    return

    if not card:
        return
    card["status"] = status
    repair_text = campaign_ui_helpers._repair_polish_text

    self._set_pack_visibility(card.get("shell"), bool(status.visible), fill=tk.X, pady=(0, 12))
    if not status.visible:
        return

    palette = getattr(self.app, "palette", {})
    card_bg = palette.get("panel", "#252526")
    emphasized = self._is_wizard_stage_emphasized(status)
    style = self._get_wizard_stage_state_style(status.state, is_current=status.is_current, emphasized=emphasized)

    badge_action_label = repair_text(str(getattr(status, "badge_action_label", "") or "").strip())
    badge_action_command = getattr(status, "badge_action_command", None)
    if (
        (not badge_action_label or badge_action_command is None)
        and not bool(getattr(self, "_project_open_lightweight_refresh", False))
    ):
        badge_action_label, badge_action_command = self._get_persistent_wizard_stage_badge_override(status)

    inline_approve = bool(
        badge_action_label
        and badge_action_command is not None
    )
    badge_text = style["label"]
    if inline_approve:
        badge_text = "GOTOWE"
    elif (
        str(getattr(status, "key", "") or "").strip().lower() == "step2"
        and bool(getattr(status, "is_current", False))
        and str(getattr(status, "state", "") or "").strip().lower() in {"ready", "in_progress", "needs_attention"}
    ):
        badge_text = "W TOKU"

    try:
        card["shell"].config(bg=style["border"])
        card["content"].config(bg=card_bg)
        card["header_row"].config(bg=card_bg)
        card["header_status_col"].config(bg=card_bg)
        card["action_row"].config(bg=card_bg)
        card["body"].config(bg=card_bg)
        card["title"].config(text=repair_text(status.title), fg=style["title_fg"], bg=card_bg)
        card["badge"].config(
            text=repair_text(badge_text),
            fg=style["badge_fg"],
            bg=style["badge_bg"],
        )
        card["summary"].config(text=repair_text(str(status.summary or "").strip()), fg=style["summary_fg"], bg=card_bg)
    except Exception:
        pass

    header_badge = card.get("badge")
    if header_badge is not None:
        try:
            header_badge.unbind("<Button-1>")
            header_badge.unbind("<Enter>")
            header_badge.unbind("<Leave>")
        except Exception:
            pass

        if inline_approve:
            success = palette.get("success", "#27ae60")
            surface_success = palette.get("surface_success", "#1f3320")
            badge_fill = blend_hex_colors(surface_success, card_bg, 0.08)
            badge_border = blend_hex_colors(success, card_bg, 0.08)
            badge_fg = palette.get("fg", "#f3f3f3")
            try:
                header_badge.configure(
                    bg=badge_fill,
                    fg=badge_fg,
                    cursor="",
                    font=("Segoe UI Semibold", 8),
                    padx=10,
                    pady=4,
                    highlightthickness=1,
                    highlightbackground=badge_border,
                    highlightcolor=badge_border,
                )
            except Exception:
                pass
        else:
            try:
                header_badge.configure(
                    cursor="",
                    font=("Segoe UI", 8, "bold"),
                    padx=8,
                    pady=3,
                    highlightthickness=0,
                )
            except Exception:
                pass

    summary_text = repair_text(str(status.summary or "").strip())
    show_summary = bool(summary_text)
    self._set_pack_visibility(
        card.get("summary"),
        show_summary,
        anchor=tk.W,
        fill=tk.X,
        pady=(8, 0),
        before=card.get("details"),
    )

    details_text = repair_text(str(status.details or "").strip())
    try:
        card["details"].config(text=details_text, bg=card_bg, fg=style.get("details_fg", palette.get("muted", "#c7c7c7")))
    except Exception:
        pass
    self._set_pack_visibility(
        card.get("details"),
        bool(details_text),
        anchor=tk.W,
        fill=tk.X,
        pady=(6, 0),
        before=card.get("action_row"),
    )

    self._render_step1_stage_curtain(card, status, style)

    stage_key = str(status.key or "").strip().upper() or "STAGE"
    row_primary_label = repair_text(str(status.primary_label or ""))
    row_primary_command = status.primary_command
    self._configure_wizard_stage_button(
        card.get("primary_btn"),
        row_primary_label,
        row_primary_command,
        side=tk.LEFT,
        padx=(0, 8),
        debug_id=f"{stage_key}-P1",
    )
    self._configure_wizard_stage_button(
        card.get("secondary_btn"),
        repair_text(status.secondary_label),
        status.secondary_command,
        side=tk.LEFT,
        padx=(0, 0),
        debug_id=f"{stage_key}-P2",
    )
    footer_actions_visible = bool(
        str(row_primary_label or "").strip()
        or repair_text(str(status.secondary_label or "").strip())
        or inline_approve
    )
    self._set_pack_visibility(
        card.get("action_row"),
        footer_actions_visible,
        fill=tk.X,
        pady=(10, 0),
    )

    body = card.get("body")
    if body is not None:
        if status.body_mode == "step1_ingest":
            try:
                self._theme_step1_ingest_panel()
            except Exception:
                pass
        if status.body_mode == "step3_rework":
            self._set_pack_visibility(body, False)
        elif status.body_mode == "step1_ingest":
            step1_mode = self._get_step1_presentation_mode()
            if step1_mode == "operational_summary":
                self._set_pack_visibility(
                    body,
                    bool(status.body_visible),
                    fill=tk.X,
                    pady=(12, 0),
                    before=card.get("curtain_shell"),
                )
            else:
                self._set_pack_visibility(
                    body,
                    bool(status.body_visible),
                    fill=tk.X,
                    pady=(12, 0),
                    before=card.get("action_row"),
                )
        else:
            self._set_pack_visibility(body, bool(status.body_visible), fill=tk.X, pady=(12, 0))

    self._configure_wizard_stage_badge_cta(
        card,
        visible=inline_approve,
        command=badge_action_command,
        label=badge_action_label,
    )
