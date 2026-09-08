#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render helpers for the Z2 canvas overlay dock."""

import tkinter as tk
from pathlib import Path

from ..config import CONFIG, logger
from .z2_shared_ui import (
    build_campaign_gate_focus_state,
    campaign_gate_id_for_edge,
    campaign_visible_gate_id,
)
from .web_slim_scrollbar import blend_hex_colors


def get_preview_overlay_dock_theme(owner) -> dict:
    try:
        legend_theme = owner._get_preview_legend_theme()
    except Exception:
        legend_theme = {}
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    fill = str(legend_theme.get("panel_fill", palette.get("panel", "#101419")))
    outline = str(legend_theme.get("panel_outline", palette.get("panel_border", "#4b5563")))
    text_fill = str(legend_theme.get("entry_text", palette.get("fg", "#f8fafc")))
    muted = str(legend_theme.get("section_muted", palette.get("muted", "#c7c7c7")))
    accent = str(legend_theme.get("badge_plate_outline", palette.get("accent", "#f1c40f")))
    active = str(palette.get("success", "#2fbf71"))
    hidden = str(palette.get("muted", "#8b949e"))
    row_fill = blend_hex_colors(outline, fill, 0.07)
    active_fill = blend_hex_colors(active, fill, 0.22)
    hidden_fill = blend_hex_colors(outline, fill, 0.06)
    status_on_fill = blend_hex_colors(active, fill, 0.40)
    status_off_fill = blend_hex_colors(outline, fill, 0.32)
    try:
        fill_is_light = bool(owner._legend_color_is_light(fill))
    except Exception:
        fill_is_light = False
    section_fill = blend_hex_colors(accent, fill, 0.42 if fill_is_light else 0.30)
    try:
        section_is_light = bool(owner._legend_color_is_light(section_fill))
    except Exception:
        section_is_light = fill_is_light
    section_text = "#07130d" if section_is_light else "#f8fafc"
    return {
        "fill": fill,
        "outline": outline,
        "text": text_fill,
        "muted": muted,
        "accent": accent,
        "active": active,
        "hidden": hidden,
        "row_fill": row_fill,
        "active_fill": active_fill,
        "hidden_fill": hidden_fill,
        "status_on_fill": status_on_fill,
        "status_on_text": text_fill,
        "status_off_fill": status_off_fill,
        "status_off_text": muted,
        "action_fill": row_fill,
        "action_outline": blend_hex_colors(outline, fill, 0.18),
        "action_hover_fill": blend_hex_colors(accent, fill, 0.28),
        "action_hover_outline": blend_hex_colors(accent, fill, 0.62),
        "section_fill": section_fill,
        "section_text": section_text,
    }


def _pointer_inside_widget(widget) -> bool:
    if widget is None:
        return False
    try:
        px = int(widget.winfo_pointerx())
        py = int(widget.winfo_pointery())
        x1 = int(widget.winfo_rootx())
        y1 = int(widget.winfo_rooty())
        x2 = x1 + int(widget.winfo_width())
        y2 = y1 + int(widget.winfo_height())
        return bool(x1 <= px <= x2 and y1 <= py <= y2)
    except Exception:
        return False


def _apply_preview_dock_row_hover_style(owner, row_key: str, hovered: bool) -> None:
    key = str(row_key or "").strip()
    if not key:
        return
    widgets = (getattr(owner, "_preview_overlay_dock_tool_rows", {}) or {}).get(key)
    if not isinstance(widgets, dict):
        return
    row = widgets.get("row")
    if row is None:
        return
    theme = get_preview_overlay_dock_theme(owner)
    tool_states = owner._get_preview_overlay_dock_tools_state()
    visible, status = tool_states.get(key, (False, "OFF"))
    fill = theme["fill"]
    row_bg = theme["action_hover_fill"] if hovered else fill
    row_fg = theme["text"] if visible else theme["muted"]
    icon_bg = blend_hex_colors(theme["accent"], fill, 0.60) if hovered else fill
    icon_fg = theme["text"] if hovered else theme["muted"]
    status_bg = (
        blend_hex_colors(theme["active"], fill, 0.52 if hovered else 0.40)
        if visible
        else theme["status_off_fill"]
    )
    status_fg = theme["status_on_text"] if visible else theme["status_off_text"]
    outline = theme["action_hover_outline"] if hovered else theme["action_outline"]
    try:
        row.configure(bg=row_bg, highlightbackground=outline, highlightcolor=outline)
        widgets["icon"].configure(bg=icon_bg, fg=icon_fg)
        widgets["label"].configure(bg=row_bg, fg=row_fg)
        widgets["status"].configure(bg=status_bg, fg=status_fg, text=str(status))
    except Exception:
        pass


def _set_preview_dock_row_hover(owner, row_key: str, active: bool, row=None) -> None:
    key = str(row_key or "").strip()
    if not key:
        return
    if not bool(active) and _pointer_inside_widget(row):
        return
    hover_rows = getattr(owner, "_preview_overlay_dock_hover_rows", None)
    if not isinstance(hover_rows, set):
        hover_rows = set()
    before = set(hover_rows)
    if bool(active):
        hover_rows = {key}
    else:
        hover_rows.discard(key)
    if hover_rows == before:
        return
    owner._preview_overlay_dock_hover_rows = hover_rows
    for old_key in before - hover_rows:
        _apply_preview_dock_row_hover_style(owner, old_key, False)
    for new_key in hover_rows - before:
        _apply_preview_dock_row_hover_style(owner, new_key, True)


def _bind_preview_dock_hover(owner, row_key: str, widgets: dict) -> None:
    if widgets.get("_hover_bound"):
        return
    row = widgets.get("row")
    for widget in (row, widgets.get("icon"), widgets.get("label"), widgets.get("status")):
        if widget is None:
            continue
        try:
            widget.bind(
                "<Enter>",
                lambda _event, key=row_key: _set_preview_dock_row_hover(owner, key, True),
                add="+",
            )
            widget.bind(
                "<Leave>",
                lambda _event, key=row_key, target=row: _set_preview_dock_row_hover(owner, key, False, target),
                add="+",
            )
        except Exception:
            pass
    widgets["_hover_bound"] = True


def _ensure_preview_dock_gate_widgets(owner, body):
    if body is None:
        return None
    frame = getattr(owner, "preview_overlay_dock_gate_frame", None)
    if frame is not None:
        return frame
    try:
        frame = tk.Frame(body, bd=0, highlightthickness=0)
        title = tk.Label(
            frame,
            text="",
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 8),
            padx=10,
            pady=7,
        )
        title.pack(fill=tk.X, padx=6, pady=(0, 6))
        approval = tk.Label(
            frame,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=1,
            font=("Segoe UI Semibold", 7),
            padx=6,
            pady=3,
        )
        approval.pack(fill=tk.X, padx=6, pady=(0, 5))
        status = tk.Label(
            frame,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=1,
            font=("Segoe UI Semibold", 7),
            padx=6,
            pady=3,
        )
        status.pack(fill=tk.X, padx=6, pady=(0, 5))
        counters = tk.Frame(frame, bd=0, highlightthickness=0)
        counters.pack(fill=tk.X, padx=6, pady=(0, 5))
        have = tk.Label(
            counters,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 7),
            padx=4,
            pady=4,
        )
        have.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        missing = tk.Label(
            counters,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 7),
            padx=4,
            pady=4,
        )
        missing.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        quality = tk.Label(
            frame,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=1,
            font=("Segoe UI Semibold", 7),
            padx=6,
            pady=3,
        )
        quality.pack(fill=tk.X, padx=6, pady=(0, 4))
        info = tk.Label(
            frame,
            text="",
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 7),
            padx=6,
            pady=3,
        )
        info.pack(fill=tk.X, padx=6, pady=(0, 6))
        owner.preview_overlay_dock_gate_frame = frame
        owner.preview_overlay_dock_gate_title_lbl = title
        owner.preview_overlay_dock_gate_approval_lbl = approval
        owner.preview_overlay_dock_gate_status_lbl = status
        owner.preview_overlay_dock_gate_counters_frame = counters
        owner.preview_overlay_dock_gate_have_lbl = have
        owner.preview_overlay_dock_gate_missing_lbl = missing
        owner.preview_overlay_dock_gate_quality_lbl = quality
        owner.preview_overlay_dock_gate_info_lbl = info
        return frame
    except Exception:
        return None


def _build_preview_dock_fallback_gate(owner) -> dict:
    """Build a compact progress snapshot for fullscreen when the main gate state is not ready yet."""
    try:
        from ..campaign_manager import CAMPAIGN

        if not bool(CAMPAIGN.get_active_project_name()):
            return {}
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        current_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        return {}

    graph_context = {}
    try:
        graph_context = dict(getattr(owner, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) or graph_gate_id or ""
    if iteration_target not in {"plate", "char"}:
        iteration_target = "char" if graph_display_gate_id in {"T05", "T06"} else "plate"

    project_approved_images = int(approved_stats.get("images", 0) or 0)
    project_approved_plates = int(approved_stats.get("plates", 0) or 0)
    run_ok_images = 0
    run_ok_plates = 0
    try:
        if getattr(owner, "current_annotations", None):
            run_ok_images, run_ok_plates = owner._get_current_preview_plate_approved_counts()
    except Exception:
        run_ok_images, run_ok_plates = 0, 0
    if int(run_ok_images or 0) <= 0 and int(run_ok_plates or 0) <= 0:
        try:
            run_ok_images, run_ok_plates = owner._get_run_plate_approved_counts(
                getattr(owner, "current_annotation_run_dir", None)
            )
        except Exception:
            run_ok_images, run_ok_plates = 0, 0

    effective_images = max(0, project_approved_images + int(run_ok_images or 0))
    effective_plates = max(0, project_approved_plates + int(run_ok_plates or 0))
    required_plates = int(
        getattr(
            CONFIG,
            "CAMPAIGN_MIN_CHAR_PLATES" if iteration_target == "char" else "CAMPAIGN_MIN_PLATE_ANNOTATIONS",
            10,
        )
        or 10
    )
    missing_plates = max(0, required_plates - effective_plates)
    xml_required = bool(current_iteration_num <= 1)
    xml_exists = False
    try:
        xml_path = getattr(owner, "current_annotation_xml_path", None)
        xml_exists = bool(xml_path and Path(xml_path).exists())
    except Exception:
        xml_exists = False
    if not xml_exists:
        try:
            run_dir = getattr(owner, "current_annotation_run_dir", None)
            xml_exists = bool(run_dir and (Path(run_dir) / "annotations.xml").exists())
        except Exception:
            xml_exists = False
    ready = bool(effective_plates >= required_plates and (xml_exists or not xml_required))
    quality_info = CONFIG.describe_yolo_pose_dataset_quality(max(0, effective_plates))
    focus_state = build_campaign_gate_focus_state(missing_plates, quality_info)
    gate_title = f"BRAMKA {graph_display_gate_id}" if graph_display_gate_id else "BRAMKA GRAFU"
    instruction = "" if ready else "Zatwierdzaj obrazy z poprawnymi ramkami tablic jako OK."
    return {
        "visible": True,
        "ready": ready,
        "tone": "success" if ready else "warning",
        "title": gate_title,
        "gate_id": graph_display_gate_id,
        "source_gate_id": graph_gate_id,
        "status": "OTWARTA" if ready else "W TRAKCIE",
        "instruction": instruction,
        "approved_images": int(effective_images),
        "approved_plates": int(effective_plates),
        "required_images": 0,
        "required_plates": int(required_plates),
        "missing_images": 0,
        "missing_plates": int(missing_plates),
        "missing_to_open": int(missing_plates),
        "missing_focus_label": str(focus_state.get("label") or "DO MIN."),
        "missing_focus_row_label": str(focus_state.get("row_label") or "Do otwarcia bramki brakuje"),
        "missing_focus_value": int(focus_state.get("value") or 0),
        "missing_focus_text": str(focus_state.get("text") or ""),
        "missing_focus_tone": str(focus_state.get("tone") or ("success" if ready else "warning")),
        "gate_metric": "plates",
        "xml": "XML: OK" if xml_exists else ("XML: wymagany" if xml_required else "XML: brak"),
        "iteration": int(current_iteration_num),
    }


def _build_preview_dock_inline_gate(owner) -> dict:
    if not bool(getattr(owner, "_preview_fullscreen_active", False)):
        return {"visible": False}
    try:
        state = dict(owner._build_campaign_z2_gate_overlay_state() or {})
    except Exception:
        state = {}
    if bool(state.get("visible")):
        try:
            owner._campaign_step2_gate_overlay_state = dict(state)
        except Exception:
            pass
        return state
    try:
        cached_state = dict(getattr(owner, "_campaign_step2_gate_overlay_state", {}) or {})
    except Exception:
        cached_state = {}
    if bool(cached_state.get("visible")):
        return cached_state
    state = _build_preview_dock_fallback_gate(owner)
    if not bool(state.get("visible")):
        return {"visible": False}
    return state


def render_preview_overlay_dock(owner, *, force_render: bool = False) -> tuple[int, int]:
    dock = getattr(owner, "preview_overlay_dock", None)
    header = getattr(owner, "preview_overlay_dock_header", None)
    title = getattr(owner, "preview_overlay_dock_title_lbl", None)
    toggle = getattr(owner, "preview_overlay_dock_toggle_lbl", None)
    body = getattr(owner, "preview_overlay_dock_body", None)
    actions_title = getattr(owner, "preview_overlay_dock_actions_title_lbl", None)
    actions_frame = getattr(owner, "preview_overlay_dock_actions_frame", None)
    status_title = getattr(owner, "preview_overlay_dock_status_title_lbl", None)
    status_frame = getattr(owner, "preview_overlay_dock_status_frame", None)
    if dock is None:
        return 0, 0

    expanded = True
    owner._preview_overlay_dock_expanded = True
    theme = get_preview_overlay_dock_theme(owner)
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    tool_states = owner._get_preview_overlay_dock_tools_state()
    hover_rows = getattr(owner, "_preview_overlay_dock_hover_rows", set())
    if not isinstance(hover_rows, set):
        hover_rows = set()
    valid_hover_rows = {str(key) for key in tool_states}
    hover_rows = {str(key) for key in hover_rows if str(key) in valid_hover_rows}
    if len(hover_rows) > 1:
        hover_rows = {sorted(hover_rows)[0]}
    owner._preview_overlay_dock_hover_rows = hover_rows
    tool_state_key = tuple((key, int(value[0]), value[1]) for key, value in sorted(tool_states.items()))
    hover_key = tuple(sorted(str(key) for key in hover_rows))
    theme_key = tuple(sorted((str(k), str(v)) for k, v in theme.items()))
    approval_key = ()
    if bool(getattr(owner, "_preview_fullscreen_active", False)):
        try:
            ann = owner._get_preview_annotation()
            approval_key = (
                str(getattr(ann, "filename", "") or ""),
                int(bool(owner._preview_annotation_is_explicitly_approved(ann))) if ann is not None else 0,
                int(getattr(owner, "_preview_approval_version", 0) or 0),
            )
        except Exception:
            approval_key = ("", 0, 0)

    fullscreen = bool(getattr(owner, "_preview_fullscreen_active", False))
    cached_inline_gate = getattr(owner, "_preview_overlay_dock_inline_gate_state", None)
    inline_gate_source_key = (
        int(fullscreen),
        approval_key,
        str(getattr(owner, "current_annotation_run_dir", "") or ""),
        int(len(getattr(owner, "current_annotations", []) or [])),
    )
    cached_inline_gate_source_key = getattr(owner, "_preview_overlay_dock_inline_gate_source_key", None)
    if not fullscreen:
        inline_gate = {"visible": False}
        owner._preview_overlay_dock_inline_gate_state = {}
        owner._preview_overlay_dock_inline_gate_source_key = None
    elif (
        bool(force_render)
        or not isinstance(cached_inline_gate, dict)
        or not bool(cached_inline_gate.get("visible"))
        or cached_inline_gate_source_key != inline_gate_source_key
    ):
        inline_gate = _build_preview_dock_inline_gate(owner)
        owner._preview_overlay_dock_inline_gate_state = dict(inline_gate or {})
        owner._preview_overlay_dock_inline_gate_source_key = inline_gate_source_key
    else:
        inline_gate = dict(cached_inline_gate)
    inline_gate_key = ()
    if bool(inline_gate.get("visible")):
        inline_gate_key = tuple(
            (str(key), str(inline_gate.get(key)))
            for key in (
                "title",
                "status",
                "ready",
                "tone",
                "approved_images",
                "approved_plates",
                "missing_images",
                "missing_to_open",
                "missing_plates",
                "missing_focus_label",
                "missing_focus_row_label",
                "missing_focus_value",
                "missing_focus_text",
                "missing_focus_tone",
                "required_plates",
                "gate_metric",
                "instruction",
                "xml",
            )
        )
    pre_gate_key = (int(expanded), tool_state_key, hover_key, approval_key, inline_gate_key, theme_key)
    cached_size = getattr(owner, "_preview_overlay_dock_size", None)
    size_key = (
        int(expanded),
        tool_state_key,
        inline_gate_key,
        approval_key,
        theme_key,
    )
    render_key = (
        int(expanded),
        tool_state_key,
        hover_key,
        inline_gate_key,
        approval_key,
        theme_key,
    )
    gate_render_key = (inline_gate_key, approval_key, theme_key)
    render_changed = bool(force_render) or render_key != getattr(owner, "_preview_overlay_dock_render_key", None)
    size_changed = bool(force_render) or size_key != getattr(owner, "_preview_overlay_dock_size_key", None)
    if not render_changed and not size_changed and isinstance(cached_size, (tuple, list)) and len(cached_size) >= 2:
        try:
            return int(cached_size[0]), int(cached_size[1])
        except Exception:
            pass

    if render_changed:
        fill = theme["fill"]
        outline = theme["outline"]
        text_fill = theme["text"]
        muted = theme["muted"]
        try:
            dock.configure(bg=fill, highlightbackground=outline, highlightcolor=outline)
            if header is not None:
                header.configure(bg=fill)
            if title is not None:
                title.configure(
                    text="SZUFLADA Z2",
                    bg=fill,
                    fg=text_fill,
                    anchor="w",
                    cursor="arrow",
                )
            if toggle is not None:
                toggle.configure(text="", bg=fill, fg=muted)
                if str(toggle.winfo_manager()):
                    toggle.pack_forget()
            if body is not None:
                body.configure(bg=fill)
                if not str(body.winfo_manager()):
                    body.pack(fill="x")
            if actions_title is not None:
                actions_title.configure(
                    bg=theme["section_fill"],
                    fg=theme.get("section_text", text_fill),
                    text="PRZEŁĄCZNIKI",
                    anchor="w",
                    font=("Segoe UI Semibold", 8),
                    padx=10,
                    pady=7,
                )
                if not str(actions_title.winfo_manager()):
                    actions_title.pack(fill="x", padx=6, pady=(0, 6))
                else:
                    actions_title.pack_configure(padx=6, pady=(0, 6))
            if actions_frame is not None:
                actions_frame.configure(bg=fill)
                if not str(actions_frame.winfo_manager()):
                    actions_frame.pack(fill="x")
            for status_widget in (status_title, status_frame):
                if status_widget is not None and str(status_widget.winfo_manager()):
                    status_widget.pack_forget()
        except Exception:
            pass

        for key, widgets in dict(getattr(owner, "_preview_overlay_dock_tool_rows", {}) or {}).items():
            visible, status = tool_states.get(str(key), (False, "OFF"))
            hovered = str(key) in hover_rows
            row_bg = theme["action_hover_fill"] if hovered else theme["fill"]
            row_fg = theme["text"] if visible else theme["muted"]
            icon_bg = blend_hex_colors(theme["accent"], theme["fill"], 0.60) if hovered else theme["fill"]
            icon_fg = theme["text"] if hovered else theme["muted"]
            status_bg = (
                blend_hex_colors(theme["active"], theme["fill"], 0.52 if hovered else 0.40)
                if visible
                else theme["status_off_fill"]
            )
            status_fg = theme["status_on_text"] if visible else theme["status_off_text"]
            try:
                _bind_preview_dock_hover(owner, str(key), widgets)
                widgets["row"].configure(
                    bg=row_bg,
                    highlightbackground=theme["action_hover_outline"] if hovered else theme["action_outline"],
                    highlightcolor=theme["action_hover_outline"] if hovered else theme["action_outline"],
                )
                widgets["icon"].configure(bg=icon_bg, fg=icon_fg)
                widgets["label"].configure(bg=row_bg, fg=row_fg)
                widgets["status"].configure(bg=status_bg, fg=status_fg, text=str(status))
            except Exception:
                pass

        for key, widgets in dict(getattr(owner, "_preview_overlay_dock_status_rows", {}) or {}).items():
            visible, status = tool_states.get(str(key), (False, "OFF"))
            row_bg = theme["fill"]
            status_fg = theme["active"] if visible else theme["muted"]
            label_fg = theme["text"] if visible else theme["muted"]
            try:
                widgets["row"].configure(bg=row_bg)
                widgets["dot"].configure(bg=row_bg, fg=status_fg)
                widgets["label"].configure(bg=row_bg, fg=label_fg)
                widgets["status"].configure(bg=row_bg, fg=status_fg, text=str(status))
            except Exception:
                pass

        gate_frame = _ensure_preview_dock_gate_widgets(owner, body)
        if gate_frame is not None:
            if bool(inline_gate.get("visible")):
                try:
                    gate_changed = bool(force_render) or gate_render_key != getattr(
                        owner, "_preview_overlay_dock_gate_render_key", None
                    )
                    if not gate_changed:
                        owner._preview_overlay_dock_render_key = render_key
                        owner._preview_overlay_dock_size_key = size_key
                        owner._preview_overlay_dock_pre_gate_key = pre_gate_key
                        if isinstance(cached_size, (tuple, list)) and len(cached_size) >= 2:
                            return int(cached_size[0]), int(cached_size[1])
                    success = str(theme["active"])
                    error = str(getattr(getattr(owner, "app", None), "palette", {}).get("error", "#c7422f"))
                    warning = str(getattr(getattr(owner, "app", None), "palette", {}).get("warning", "#f39c12"))
                    fill = theme["fill"]
                    body_fill = blend_hex_colors(theme["outline"], fill, 0.06)
                    text_fill = theme["text"]
                    muted = theme["muted"]
                    ready = bool(inline_gate.get("ready"))
                    approved_images = int(inline_gate.get("approved_images", 0) or 0)
                    approved_plates = int(inline_gate.get("approved_plates", 0) or 0)
                    missing_images = int(inline_gate.get("missing_images", 0) or 0)
                    missing_to_open = int(
                        inline_gate.get(
                            "missing_to_open",
                            inline_gate.get("missing_plates", missing_images),
                        )
                        or 0
                    )
                    gate_metric = str(inline_gate.get("gate_metric") or "").strip().lower()
                    if gate_metric not in {"images", "plates"}:
                        gate_metric = "plates" if (
                            approved_plates
                            or int(inline_gate.get("required_plates", 0) or 0)
                            or int(inline_gate.get("missing_plates", 0) or 0)
                        ) else "images"
                    missing_focus_label = str(inline_gate.get("missing_focus_label") or "BRAKUJE").strip().upper()
                    missing_focus_value_raw = inline_gate.get("missing_focus_value", missing_to_open)
                    missing_focus_tone = str(inline_gate.get("missing_focus_tone") or "").strip().lower()
                    try:
                        missing_focus_value = int(missing_focus_value_raw or 0)
                    except Exception:
                        missing_focus_value = int(missing_to_open or 0)
                    have_value = approved_plates if gate_metric == "plates" else approved_images
                    have_label = "TABLICE" if gate_metric == "plates" else "OBRAZY"
                    quality_score = max(0, approved_plates) if gate_metric == "plates" else min(max(0, approved_images), max(0, approved_plates))
                    quality_info = CONFIG.describe_yolo_pose_dataset_quality(quality_score)
                    quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
                    quality_tone = str(quality_info.get("tone", "error") or "error").strip().lower()
                    quality_color = success if quality_tone == "success" else warning if quality_tone == "warning" else error
                    gate_status_fill = success if ready else error
                    status_text = "#111111" if owner._legend_color_is_light(gate_status_fill) else "#ffffff"
                    have_fill = blend_hex_colors(success, fill, 0.36)
                    info_color = str(palette.get("info", palette.get("accent", "#4aa3ff")))
                    missing_base = (
                        success
                        if missing_focus_tone == "success"
                        else warning
                        if missing_focus_tone == "warning"
                        else info_color
                        if missing_focus_tone == "info"
                        else error
                    )
                    missing_fill = blend_hex_colors(missing_base, fill, 0.36)
                    have_text = "#111111" if owner._legend_color_is_light(have_fill) else "#ffffff"
                    missing_text = "#111111" if owner._legend_color_is_light(missing_fill) else "#ffffff"
                    quality_text = "#111111" if owner._legend_color_is_light(quality_color) else "#ffffff"
                    instruction = str(inline_gate.get("instruction") or "").strip()
                    if not instruction and not ready:
                        instruction = "Oznaczaj dalej."

                    ann = owner._get_preview_annotation()
                    approved = bool(owner._preview_annotation_is_explicitly_approved(ann)) if ann is not None else False
                    approval_fill = success if approved else error
                    approval_text = "#111111" if owner._legend_color_is_light(approval_fill) else "#ffffff"
                    getattr(owner, "preview_overlay_dock_gate_frame").configure(bg=fill)
                    gate_title_lbl = getattr(owner, "preview_overlay_dock_gate_title_lbl")
                    gate_title_lbl.configure(
                        bg=theme["section_fill"],
                        fg=theme.get("section_text", muted),
                        font=("Segoe UI Semibold", 8),
                        padx=10,
                        pady=7,
                        text="STATUS PRACY",
                    )
                    try:
                        gate_title_lbl.pack_configure(padx=6, pady=(0, 6))
                    except Exception:
                        pass
                    getattr(owner, "preview_overlay_dock_gate_approval_lbl").configure(
                        bg=approval_fill,
                        fg=approval_text,
                        highlightbackground=blend_hex_colors(approval_fill, fill, 0.12),
                        highlightcolor=blend_hex_colors(approval_fill, fill, 0.12),
                        text="ZDJĘCIE OK" if approved else "ZDJĘCIE NIEZATWIERDZONE",
                    )
                    getattr(owner, "preview_overlay_dock_gate_status_lbl").configure(
                        bg=gate_status_fill,
                        fg=status_text,
                        highlightbackground=blend_hex_colors(gate_status_fill, fill, 0.12),
                        highlightcolor=blend_hex_colors(gate_status_fill, fill, 0.12),
                        text=(
                            f"{str(inline_gate.get('title') or 'BRAMKA GRAFU').strip().upper()}\n"
                            f"{str(inline_gate.get('status') or '').strip().upper()}"
                        ),
                    )
                    getattr(owner, "preview_overlay_dock_gate_counters_frame").configure(bg=fill)
                    getattr(owner, "preview_overlay_dock_gate_have_lbl").configure(
                        bg=have_fill,
                        fg=have_text,
                        text=f"{have_label}\n{have_value}",
                    )
                    getattr(owner, "preview_overlay_dock_gate_missing_lbl").configure(
                        bg=missing_fill,
                        fg=missing_text,
                        text=f"{missing_focus_label}\n{'MAX' if missing_focus_tone == 'success' and missing_focus_value <= 0 else missing_focus_value}",
                    )
                    getattr(owner, "preview_overlay_dock_gate_quality_lbl").configure(
                        bg=quality_color,
                        fg=quality_text,
                        highlightbackground=blend_hex_colors(quality_color, fill, 0.12),
                        highlightcolor=blend_hex_colors(quality_color, fill, 0.12),
                        text=f"JAKOŚĆ ZBIORU\n{quality_label}",
                    )
                    info_lbl = getattr(owner, "preview_overlay_dock_gate_info_lbl")
                    if ready or not instruction:
                        if str(info_lbl.winfo_manager()):
                            info_lbl.pack_forget()
                    else:
                        if not str(info_lbl.winfo_manager()):
                            info_lbl.pack(fill=tk.X, padx=6, pady=(0, 6))
                        info_lbl.configure(
                            bg=body_fill,
                            fg=text_fill,
                            text=instruction,
                            wraplength=214,
                        )
                    if str(gate_frame.winfo_manager()):
                        try:
                            gate_frame.pack_forget()
                        except Exception:
                            pass
                    if not str(gate_frame.winfo_manager()):
                        pack_options = {"fill": "x", "pady": (6, 0)}
                        if actions_frame is not None and str(actions_frame.winfo_manager()):
                            pack_options["after"] = actions_frame
                        gate_frame.pack(**pack_options)
                    owner._preview_overlay_dock_gate_render_key = gate_render_key
                except Exception:
                    try:
                        logger.debug("Nie udało się wyrenderować statusu pracy w szufladzie Z2.", exc_info=True)
                    except Exception:
                        pass
                    try:
                        if str(gate_frame.winfo_manager()):
                            gate_frame.pack_forget()
                    except Exception:
                        pass
                    owner._preview_overlay_dock_gate_render_key = None
            elif str(gate_frame.winfo_manager()):
                try:
                    gate_frame.pack_forget()
                    owner._preview_overlay_dock_gate_render_key = None
                except Exception:
                    pass
        owner._preview_overlay_dock_render_key = render_key

    try:
        if size_changed or not isinstance(cached_size, (tuple, list)):
            dock.update_idletasks()
        width = int(dock.winfo_reqwidth() or (118 if expanded else 42))
        height = int(dock.winfo_reqheight() or (178 if expanded else 32))
    except Exception:
        width = 118 if expanded else 42
        height = 178 if expanded else 32
    if bool(fullscreen):
        debug_key = (
            bool(inline_gate.get("visible")),
            str(inline_gate.get("title") or ""),
            str(inline_gate.get("status") or ""),
            str(inline_gate.get("approved_images") or ""),
            str(inline_gate.get("approved_plates") or ""),
            str(inline_gate.get("missing_to_open") or inline_gate.get("missing_plates") or ""),
        )
        if debug_key != getattr(owner, "_preview_overlay_dock_inline_gate_debug_key", None):
            owner._preview_overlay_dock_inline_gate_debug_key = debug_key
            try:
                logger.info(
                    "[Z2 DOCK] fullscreen=%s gate_visible=%s title=%s status=%s ok_images=%s ok_plates=%s missing=%s",
                    bool(fullscreen),
                    bool(inline_gate.get("visible")),
                    str(inline_gate.get("title") or ""),
                    str(inline_gate.get("status") or ""),
                    str(inline_gate.get("approved_images") or ""),
                    str(inline_gate.get("approved_plates") or ""),
                    str(inline_gate.get("missing_to_open") or inline_gate.get("missing_plates") or ""),
                )
            except Exception:
                pass
    max_height = 520 if bool(inline_gate.get("visible")) else 230
    max_width = 270 if bool(inline_gate.get("visible")) else 174
    min_width = 250 if bool(inline_gate.get("visible")) else 174
    size = int(max(min_width, min(max_width, width))), int(max(30, min(max_height, height)))
    owner._preview_overlay_dock_size = size
    owner._preview_overlay_dock_size_key = size_key
    owner._preview_overlay_dock_pre_gate_key = pre_gate_key
    return size


def render_preview_image_status_overlay(owner, *, force_render: bool = False) -> tuple[int, int]:
    frame = getattr(owner, "preview_image_status_frame", None)
    label = getattr(owner, "preview_image_status_lbl", None)
    if frame is None or label is None:
        return 0, 0

    ann = owner._get_preview_annotation()
    if ann is None:
        return 0, 0

    try:
        legend_theme = owner._get_preview_legend_theme()
    except Exception:
        legend_theme = {}
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    panel_fill = str(legend_theme.get("panel_fill", palette.get("panel", "#101419")))
    success = str(palette.get("success", "#2fbf71"))
    error = str(palette.get("error", "#c7422f"))

    approved = bool(owner._preview_annotation_is_explicitly_approved(ann))
    status_fill = success if approved else error
    status_text = "#111111" if owner._legend_color_is_light(status_fill) else "#ffffff"
    outline = blend_hex_colors(status_fill, panel_fill, 0.08)
    filename = str(getattr(ann, "filename", "") or "").strip()
    label_text = "ZDJĘCIE\nZATWIERDZONE [OK]" if approved else "ZDJĘCIE\nNIEZATWIERDZONE"

    render_key = (
        int(bool(getattr(owner, "_preview_fullscreen_active", False))),
        int(approved),
        int(getattr(owner, "_preview_approval_version", 0) or 0),
        filename.lower(),
        panel_fill,
        status_fill,
        status_text,
        outline,
    )
    if bool(force_render) or render_key != getattr(owner, "_preview_image_status_overlay_render_key", None):
        try:
            frame.configure(bg=status_fill, highlightbackground=outline, highlightcolor=outline)
            label.configure(
                bg=status_fill,
                fg=status_text,
                text=label_text,
                wraplength=126,
            )
        except Exception:
            pass
        owner._preview_image_status_overlay_render_key = render_key

    try:
        frame.update_idletasks()
        width = int(frame.winfo_reqwidth() or 126)
        height = int(frame.winfo_reqheight() or 38)
    except Exception:
        width = 126
        height = 38
    return int(max(96, min(170, width))), int(max(30, min(72, height)))


def render_preview_campaign_gate_overlay(owner, state: dict, *, force_render: bool = False) -> tuple[int, int]:
    frame = getattr(owner, "preview_campaign_gate_frame", None)
    header = getattr(owner, "preview_campaign_gate_header", None)
    title = getattr(owner, "preview_campaign_gate_title_lbl", None)
    status = getattr(owner, "preview_campaign_gate_status_lbl", None)
    body = getattr(owner, "preview_campaign_gate_body", None)
    counters = getattr(owner, "preview_campaign_gate_counters_frame", None)
    count = getattr(owner, "preview_campaign_gate_count_lbl", None)
    missing = getattr(owner, "preview_campaign_gate_missing_lbl", None)
    quality = getattr(owner, "preview_campaign_gate_quality_lbl", None)
    info = getattr(owner, "preview_campaign_gate_info_lbl", None)
    if frame is None:
        return 0, 0

    try:
        legend_theme = owner._get_preview_legend_theme()
    except Exception:
        legend_theme = {}
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    fill = str(legend_theme.get("panel_fill", palette.get("panel", "#101419")))
    outline = str(legend_theme.get("panel_outline", palette.get("panel_border", "#4b5563")))
    text_fill = str(legend_theme.get("entry_text", palette.get("fg", "#f8fafc")))
    muted = str(legend_theme.get("section_muted", palette.get("muted", "#c7c7c7")))
    success = str(palette.get("success", "#2fbf71"))
    warning = str(palette.get("warning", "#f39c12"))
    error = str(palette.get("error", "#c7422f"))
    tone = str(state.get("tone") or "muted").strip().lower()
    ready = bool(state.get("ready"))
    accent = success if ready else error
    header_fill = blend_hex_colors(accent, fill, 0.18)
    body_fill = (
        blend_hex_colors(fill, "#000000", 0.18)
        if owner._legend_color_is_light(fill)
        else blend_hex_colors(fill, "#ffffff", 0.08)
    )
    body_text_fill = owner._get_preview_legend_text_color(body_fill)
    status_fill = success if ready else error
    status_text_fill = "#111111" if owner._legend_color_is_light(status_fill) else "#ffffff"
    status_outline = blend_hex_colors(accent, fill, 0.08)
    have_fill = blend_hex_colors(success, body_fill, 0.32)
    have_text_fill = "#111111" if owner._legend_color_is_light(have_fill) else "#ffffff"
    approved_images = int(state.get("approved_images", 0) or 0)
    approved_plates = int(state.get("approved_plates", 0) or 0)
    missing_images = int(state.get("missing_images", 0) or 0)
    missing_to_open = int(
        state.get("missing_to_open", state.get("missing_plates", missing_images)) or 0
    )
    missing_focus_label = str(state.get("missing_focus_label") or "BRAKUJE").strip().upper()
    missing_focus_tone = str(state.get("missing_focus_tone") or "").strip().lower()
    try:
        missing_focus_value = int(state.get("missing_focus_value", missing_to_open) or 0)
    except Exception:
        missing_focus_value = int(missing_to_open or 0)
    info_color = str(palette.get("info", palette.get("accent", "#4aa3ff")))
    missing_base = (
        success
        if missing_focus_tone == "success"
        else warning
        if missing_focus_tone == "warning"
        else info_color
        if missing_focus_tone == "info"
        else error
    )
    missing_fill = blend_hex_colors(missing_base, body_fill, 0.32)
    missing_text_fill = "#111111" if owner._legend_color_is_light(missing_fill) else "#ffffff"
    gate_metric = str(state.get("gate_metric") or "").strip().lower()
    if gate_metric not in {"images", "plates"}:
        gate_metric = "plates" if (
            approved_plates
            or int(state.get("required_plates", 0) or 0)
            or int(state.get("missing_plates", 0) or 0)
        ) else "images"
    xml_text = str(state.get("xml") or "").strip()
    quality_score = max(0, approved_plates) if gate_metric == "plates" else min(max(0, approved_images), max(0, approved_plates))
    quality_info = CONFIG.describe_yolo_pose_dataset_quality(quality_score)
    quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
    quality_tone = str(quality_info.get("tone", "error") or "error").strip().lower()
    if quality_tone == "success":
        quality_color = success
    elif quality_tone == "warning":
        quality_color = warning
    else:
        quality_color = error
    quality_fill = quality_color
    quality_text_fill = "#111111" if owner._legend_color_is_light(quality_fill) else "#ffffff"
    quality_outline = blend_hex_colors(quality_color, fill, 0.08)
    quality_text = f"JAKOŚĆ ZBIORU\n{quality_label}"
    title_text = str(state.get("title") or "BRAMKA GRAFU").strip()
    instruction_text = str(state.get("instruction") or "").strip()
    if not instruction_text:
        if ready:
            instruction_text = ""
        elif "wymagany" in xml_text.lower() and missing_to_open > 0:
            instruction_text = "Utwórz XML i oznaczaj dalej."
        elif "wymagany" in xml_text.lower():
            instruction_text = "Utwórz XML."
        elif gate_metric == "plates" and missing_to_open > 0:
            instruction_text = "Oznaczaj tablice dalej."
        elif approved_plates <= 0:
            instruction_text = "Dodaj ramkę tablicy."
        else:
            instruction_text = "Oznaczaj dalej."
    have_value = approved_plates if gate_metric == "plates" else approved_images
    have_label = "TABLICE" if gate_metric == "plates" else "OBRAZY"
    have_text = f"{have_label}\n{have_value}"
    missing_text = f"{missing_focus_label}\n{'MAX' if missing_focus_tone == 'success' and missing_focus_value <= 0 else missing_focus_value}"
    render_key = (
        str(state.get("status") or ""),
        title_text,
        have_text,
        missing_text,
        quality_text,
        instruction_text,
        tone,
        fill,
        outline,
        text_fill,
        muted,
        accent,
        body_fill,
        body_text_fill,
        status_fill,
        have_fill,
        missing_fill,
        quality_fill,
    )
    if bool(force_render) or render_key != getattr(owner, "_preview_campaign_gate_overlay_render_key", None):
        try:
            frame.configure(bg=fill, highlightbackground=accent, highlightcolor=accent)
            if header is not None:
                header.configure(bg=header_fill)
            if body is not None:
                body.configure(bg=body_fill)
            if counters is not None:
                counters.configure(bg=body_fill)
            if title is not None:
                title.configure(bg=header_fill, fg=text_fill, text=title_text)
            if status is not None:
                status.configure(
                    bg=status_fill,
                    fg=status_text_fill,
                    highlightbackground=status_outline,
                    highlightcolor=status_outline,
                    text=str(state.get("status") or ""),
                )
            if count is not None:
                if not str(count.winfo_manager()):
                    count.pack(side="left", fill="both", expand=True, padx=(0, 2))
                count.configure(
                    bg=have_fill,
                    fg=have_text_fill,
                    text=have_text,
                    wraplength=54,
                )
            if missing is not None:
                if not str(missing.winfo_manager()):
                    missing.pack(side="left", fill="both", expand=True, padx=(2, 0))
                missing.configure(
                    bg=missing_fill,
                    fg=missing_text_fill,
                    text=missing_text,
                    wraplength=58,
                )
            if quality is not None:
                if not str(quality.winfo_manager()):
                    quality.pack(fill="x", padx=5, pady=(0, 3))
                quality.configure(
                    bg=quality_fill,
                    fg=quality_text_fill,
                    highlightbackground=quality_outline,
                    highlightcolor=quality_outline,
                    text=quality_text,
                    wraplength=128,
                )
            if info is not None:
                if instruction_text:
                    if not str(info.winfo_manager()):
                        info.pack(fill="x", padx=5, pady=(0, 5))
                    info.configure(
                        bg=body_fill,
                        fg=body_text_fill,
                        text=instruction_text,
                        wraplength=128,
                    )
                elif str(info.winfo_manager()):
                    info.pack_forget()
        except Exception:
            pass
        owner._preview_campaign_gate_overlay_render_key = render_key

    try:
        frame.update_idletasks()
        width = int(frame.winfo_reqwidth() or 126)
        height = int(frame.winfo_reqheight() or 72)
    except Exception:
        width = 126
        height = 72
    return int(max(112, min(170, width))), int(max(102, min(260, height)))
