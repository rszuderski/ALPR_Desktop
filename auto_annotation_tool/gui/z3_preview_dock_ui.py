from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from .web_slim_scrollbar import blend_hex_colors
from .z3_preview_compass_ui import get_preview_legend_theme
from .z2_drawer_slide import PreviewDrawerSlide, suspend_drawer

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def get_preview_overlay_dock_theme(host: "CharacterAnnotationTab") -> dict:
    try:
        legend_theme = get_preview_legend_theme(host)
    except Exception:
        legend_theme = {}
    palette = getattr(host.app, "palette", {}) if getattr(host, "app", None) is not None else {}
    fill = str(legend_theme.get("panel_fill", palette.get("panel", "#101419")))
    outline = str(legend_theme.get("panel_outline", palette.get("panel_border", "#4b5563")))
    text = str(legend_theme.get("entry_text", palette.get("fg", "#f8fafc")))
    muted = str(legend_theme.get("section_muted", palette.get("muted", "#c7c7c7")))
    active = str(palette.get("success", legend_theme.get("badge_plate_outline", "#2fbf71")))
    accent = str(legend_theme.get("badge_plate_outline", active))
    info = str(palette.get("info", accent))
    warning = str(palette.get("warning", "#f4c27a"))
    error = str(palette.get("error", "#e74c3c"))
    return {
        "fill": fill,
        "outline": outline,
        "text": text,
        "muted": muted,
        "active": active,
        "accent": accent,
        "info": info,
        "warning": warning,
        "error": error,
        "row_fill": blend_hex_colors(outline, fill, 0.12),
        "active_fill": blend_hex_colors(active, fill, 0.22),
        "hover_fill": blend_hex_colors(accent, fill, 0.28),
        "hover_outline": blend_hex_colors(accent, fill, 0.62),
        "hidden_fill": blend_hex_colors(outline, fill, 0.06),
        "status_on_fill": blend_hex_colors(active, fill, 0.40),
        "status_off_fill": blend_hex_colors(outline, fill, 0.32),
    }


def _forget_preview_dock_row(widgets: dict) -> None:
    row = widgets.get("row")
    try:
        if row is not None and str(row.winfo_manager()):
            row.pack_forget()
    except Exception:
        pass


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


def _get_preview_dock_readable_text(host: "CharacterAnnotationTab", background: str, *, preferred: str | None = None) -> str:
    try:
        return str(host._get_readable_text_color(str(background), preferred=preferred))
    except Exception:
        pass
    bg = str(background or "#000000").strip().lstrip("#")
    if len(bg) == 3:
        bg = "".join(ch * 2 for ch in bg)
    try:
        r = int(bg[0:2], 16)
        g = int(bg[2:4], 16)
        b = int(bg[4:6], 16)
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "#111827" if luminance >= 150 else "#f8fafc"
    except Exception:
        return str(preferred or "#f8fafc")


def _set_preview_dock_row_hover(host: "CharacterAnnotationTab", row_key: str, active: bool, row=None) -> None:
    key = str(row_key or "").strip()
    if not key:
        return
    if not bool(active) and _pointer_inside_widget(row):
        return
    hover_rows = getattr(host, "_preview_overlay_dock_hover_rows", None)
    if not isinstance(hover_rows, set):
        hover_rows = set()
    before = set(hover_rows)
    if bool(active):
        hover_rows.add(key)
    else:
        hover_rows.discard(key)
    if hover_rows == before:
        return
    host._preview_overlay_dock_hover_rows = hover_rows
    _apply_preview_dock_row_hover_style(host, key, bool(active))


def _get_preview_dock_row_runtime_state(host: "CharacterAnnotationTab", row_key: str, theme: dict, *, gate_state=None) -> dict:
    key = str(row_key or "").strip()
    if key == "legend":
        visible = bool(getattr(host, "_preview_controls_legend_visible", True))
        return {
            "visible": visible,
            "tone": "success" if visible else "muted",
            "status_text": "ON" if visible else "OFF",
            "label_text": None,
            "icon_text": None,
            "interactive": True,
        }
    if key == "assistant":
        visible = bool(getattr(host, "_preview_operation_assistant_visible", True))
        return {
            "visible": visible,
            "tone": "success" if visible else "muted",
            "status_text": "ON" if visible else "OFF",
            "label_text": None,
            "icon_text": None,
            "interactive": True,
        }
    if key == "layout":
        data = host._get_preview_active_data(create=False)
        layout_text, layout_tone = host._get_preview_plate_layout_dock_text(data)
        return {
            "visible": True,
            "tone": layout_tone,
            "status_text": layout_text,
            "label_text": None,
            "icon_text": None,
            "interactive": True,
        }
    if key == "plate_status":
        data = host._get_preview_active_data(create=False)
        status_meta = host._get_preview_status_presentation(data=data if isinstance(data, dict) else None)
        status_raw = str(status_meta.get("status", "unknown") or "unknown").strip().lower()
        severity = str(status_meta.get("severity", "muted") or "muted").strip().lower()
        status_text = "PERFECT" if status_raw == "perfect" else "KOREKTA"
        if status_text == "PERFECT":
            severity = "success"
        elif severity not in {"warning", "error"}:
            severity = "warning"
        return {
            "visible": True,
            "tone": severity,
            "status_text": status_text,
            "label_text": None,
            "icon_text": None,
            "interactive": False,
        }

    if gate_state is None:
        gate_state = host._get_preview_step3_gate_overlay_state()
    gate_visible = bool(gate_state.get("visible"))
    gate_id = str(gate_state.get("gate_id", "") or "").strip().upper()
    if key == "gate":
        if gate_id == "T06":
            return {
                "visible": False,
                "tone": "muted",
                "status_text": "",
                "label_text": None,
                "icon_text": None,
                "interactive": False,
            }
        return {
            "visible": gate_visible,
            "tone": str(gate_state.get("gate_tone", "muted") or "muted").strip().lower(),
            "status_text": str(gate_state.get("gate_text", "-") or "-").strip(),
            "label_text": f"Bramka {gate_id}" if gate_id else "Bramka grafu",
            "icon_text": gate_id or "BG",
            "interactive": False,
        }
    if key == "export_condition":
        return {
            "visible": gate_visible,
            "tone": str(gate_state.get("export_condition_tone", "muted") or "muted").strip().lower(),
            "status_text": str(gate_state.get("export_condition_text", "-") or "-").strip(),
            "label_text": None,
            "icon_text": None,
            "interactive": False,
        }
    if key == "quality":
        return {
            "visible": gate_visible,
            "tone": str(gate_state.get("quality_tone", "muted") or "muted").strip().lower(),
            "status_text": str(gate_state.get("quality_text", "-") or "-").strip(),
            "label_text": None,
            "icon_text": None,
            "interactive": False,
        }
    if key == "quality_have":
        return {
            "visible": gate_visible,
            "tone": str(gate_state.get("quality_tone", "muted") or "muted").strip().lower(),
            "status_text": str(gate_state.get("quality_have", "0") or "0").strip(),
            "label_text": None,
            "icon_text": None,
            "interactive": False,
        }
    if key == "quality_missing":
        return {
            "visible": gate_visible,
            "tone": "success" if bool(gate_state.get("quality_missing_ready", False)) else str(gate_state.get("quality_tone", "warning") or "warning").strip().lower(),
            "status_text": str(gate_state.get("quality_missing", "0") or "0").strip(),
            "label_text": None,
            "icon_text": None,
            "interactive": False,
        }
    return {
        "visible": False,
        "tone": "muted",
        "status_text": "OFF",
        "label_text": None,
        "icon_text": None,
        "interactive": False,
    }


def _apply_preview_dock_row_hover_style(host: "CharacterAnnotationTab", row_key: str, hovered: bool) -> None:
    widgets = (getattr(host, "_preview_overlay_dock_tool_rows", {}) or {}).get(str(row_key or "").strip())
    if not isinstance(widgets, dict):
        return
    row = widgets.get("row")
    if row is None:
        return
    theme = get_preview_overlay_dock_theme(host)
    fill = theme["fill"]
    state = _get_preview_dock_row_runtime_state(host, row_key, theme)
    visible = bool(state.get("visible"))
    interactive = bool(state.get("interactive"))
    if not interactive:
        return
    tone = str(state.get("tone", "muted") or "muted").strip().lower()
    tone_color = {
        "success": theme["active"],
        "info": theme["info"],
        "warning": theme["warning"],
        "error": theme["error"],
    }.get(tone, theme["outline"])
    row_bg = theme["hover_fill"] if hovered else blend_hex_colors(theme["accent"], fill, 0.18)
    row_fg = _get_preview_dock_readable_text(host, row_bg, preferred=theme["text"] if visible else theme["muted"])
    icon_bg = blend_hex_colors(theme["accent"], fill, 0.68 if hovered else 0.52)
    status_bg = blend_hex_colors(tone_color, fill, 0.52 if hovered else 0.40) if visible else theme["status_off_fill"]
    icon_fg = _get_preview_dock_readable_text(host, icon_bg, preferred=theme["text"])
    status_fg = _get_preview_dock_readable_text(host, status_bg, preferred=theme["text"] if visible else theme["muted"])
    outline = theme["hover_outline"] if hovered else tone_color
    try:
        row.configure(bg=row_bg, highlightbackground=outline, highlightcolor=outline)
        widgets["icon"].configure(bg=icon_bg, fg=icon_fg, font=("Segoe UI", 8, "bold"))
        widgets["label"].configure(bg=row_bg, fg=row_fg)
        widgets["status"].configure(bg=status_bg, fg=status_fg, text=str(state.get("status_text", "") or ""))
    except Exception:
        pass


def refresh_preview_overlay_dock_tool_row(host: "CharacterAnnotationTab", row_key: str) -> None:
    """Refresh a single interactive dock row without rebuilding the whole drawer."""
    key = str(row_key or "").strip()
    if not key:
        return
    hover_rows = getattr(host, "_preview_overlay_dock_hover_rows", set())
    if not isinstance(hover_rows, set):
        hover_rows = set()
    _apply_preview_dock_row_hover_style(host, key, bool(key in hover_rows))


def _bind_preview_dock_hover(host: "CharacterAnnotationTab", row_key: str, widgets: dict) -> None:
    if widgets.get("_hover_bound"):
        return
    row = widgets.get("row")
    for widget in (row, widgets.get("icon"), widgets.get("label"), widgets.get("status")):
        if widget is None:
            continue
        try:
            widget.bind(
                "<Enter>",
                lambda _event, key=row_key: _set_preview_dock_row_hover(host, key, True),
                add="+",
            )
            widget.bind(
                "<Leave>",
                lambda _event, key=row_key, target=row: _set_preview_dock_row_hover(host, key, False, target),
                add="+",
            )
        except Exception:
            pass
    widgets["_hover_bound"] = True


def render_preview_overlay_dock(host: "CharacterAnnotationTab", *, force_render: bool = False) -> tuple[int, int]:
    dock = getattr(host, "preview_overlay_dock", None)
    header = getattr(host, "preview_overlay_dock_header", None)
    title = getattr(host, "preview_overlay_dock_title_lbl", None)
    toggle = getattr(host, "preview_overlay_dock_toggle_lbl", None)
    body = getattr(host, "preview_overlay_dock_body", None)
    if dock is None:
        return 0, 0

    expanded = True
    compact = not bool(getattr(host, "_preview_fullscreen_active", False))
    legend_visible = bool(getattr(host, "_preview_controls_legend_visible", True))
    assistant_visible = bool(getattr(host, "_preview_operation_assistant_visible", True))
    theme = get_preview_overlay_dock_theme(host)
    data = host._get_preview_active_data(create=False)
    gate_state = host._get_preview_step3_gate_overlay_state()
    gate_visible = bool(gate_state.get("visible"))
    gate_id = str(gate_state.get("gate_id", "") or "").strip().upper()
    gate_label = str(gate_state.get("gate_label", "") or "").strip()
    gate_text = str(gate_state.get("gate_text", "-") or "-").strip()
    gate_tone = str(gate_state.get("gate_tone", "muted") or "muted").strip().lower()
    export_condition_text = str(gate_state.get("export_condition_text", "-") or "-").strip()
    export_condition_tone = str(gate_state.get("export_condition_tone", "muted") or "muted").strip().lower()
    quality_text = str(gate_state.get("quality_text", "-") or "-").strip()
    quality_tone = str(gate_state.get("quality_tone", "muted") or "muted").strip().lower()
    quality_have = str(gate_state.get("quality_have", "0") or "0").strip()
    quality_missing = str(gate_state.get("quality_missing", "0") or "0").strip()
    quality_missing_ready = bool(gate_state.get("quality_missing_ready", False))
    gate_missing_tone = str(gate_state.get("gate_missing_tone", "warning") or "warning").strip().lower()
    hover_rows = getattr(host, "_preview_overlay_dock_hover_rows", set())
    if not isinstance(hover_rows, set):
        hover_rows = set()
    row_states = {key: _get_preview_dock_row_runtime_state(host, key, theme, gate_state=gate_state)
                  for key in (getattr(host, "_preview_overlay_dock_tool_rows", {}) or {})}
    layout_state = _get_preview_dock_row_runtime_state(host, "layout", theme, gate_state=gate_state)
    plate_state = _get_preview_dock_row_runtime_state(host, "plate_status", theme, gate_state=gate_state)
    render_key = (
        int(expanded),
        int(compact),
        int(legend_visible),
        int(assistant_visible),
        str(layout_state.get("status_text", "")),
        str(layout_state.get("tone", "")),
        str(plate_state.get("status_text", "")),
        str(plate_state.get("tone", "")),
        int(gate_visible),
        gate_id,
        gate_label,
        gate_text,
        gate_tone,
        export_condition_text,
        export_condition_tone,
        quality_text,
        quality_tone,
        quality_have,
        quality_missing,
        int(quality_missing_ready),
        gate_missing_tone,
        int(gate_state.get("perfect_count", 0) or 0),
        int(gate_state.get("exportable_plate_count", 0) or 0),
        int(gate_state.get("exportable_char_count", 0) or 0),
        tuple(sorted((str(key), str(value)) for key, value in theme.items())),
    )
    if bool(force_render) or render_key != getattr(host, "_preview_overlay_dock_render_key", None):
        fill = theme["fill"]
        outline = theme["active"]
        header_fg = _get_preview_dock_readable_text(host, fill, preferred=theme["text"])
        toggle_fg = _get_preview_dock_readable_text(host, fill, preferred=theme["accent"])
        try:
            dock.configure(bg=fill, highlightbackground=outline, highlightcolor=outline)
            if header is not None:
                header.configure(bg=fill)
            if title is not None:
                title.configure(
                    text="NARZĘDZIA",
                    bg=fill,
                    fg=header_fg,
                    anchor="w",
                )
            if toggle is not None:
                toggle.configure(text="", bg=fill, fg=toggle_fg)
                try:
                    toggle.pack_forget()
                except Exception:
                    pass
            if body is not None:
                body.configure(bg=fill)
                if not str(body.winfo_manager()):
                    body.pack(fill=tk.X, pady=(5, 0))
        except Exception:
            pass

        for key, widgets in dict(getattr(host, "_preview_overlay_dock_tool_rows", {}) or {}).items():
            row_key = str(key)
            if compact and row_key not in {"legend", "assistant", "layout", "plate_status", "gate"}:
                _forget_preview_dock_row(widgets)
                continue
            row_state = row_states[row_key]
            if not bool(row_state.get("visible")) and row_key not in {"legend", "assistant"}:
                _forget_preview_dock_row(widgets)
                continue
            visible = bool(row_state.get("visible"))
            tone = str(row_state.get("tone", "muted") or "muted").strip().lower()
            status_text = str(row_state.get("status_text", "OFF") or "OFF")
            icon_text = row_state.get("icon_text")
            label_text = row_state.get("label_text")

            tone_color = {
                "success": theme["active"],
                "info": theme["info"],
                "warning": theme["warning"],
                "error": theme["error"],
            }.get(tone, theme["outline"])
            interactive = bool(row_state.get("interactive"))
            hovered = bool(interactive and row_key in hover_rows)
            if hovered:
                row_bg = theme["hover_fill"]
            else:
                row_bg = blend_hex_colors(theme["accent"], fill, 0.18) if interactive else fill
            row_fg = _get_preview_dock_readable_text(host, row_bg, preferred=theme["text"] if visible else theme["muted"])
            icon_bg = blend_hex_colors(theme["accent"], fill, 0.68 if hovered else 0.52) if interactive else fill
            status_bg = blend_hex_colors(tone_color, fill, 0.52 if hovered else 0.40) if visible else theme["status_off_fill"]
            icon_fg = _get_preview_dock_readable_text(host, icon_bg, preferred=theme["text"] if interactive else theme["muted"])
            status_fg = _get_preview_dock_readable_text(host, status_bg, preferred=theme["text"] if visible else theme["muted"])
            row = widgets.get("row")
            try:
                if interactive:
                    _bind_preview_dock_hover(host, row_key, widgets)
                if row is not None and not str(row.winfo_manager()):
                    row.pack(fill=tk.X)
                row.configure(
                    bg=row_bg,
                    highlightbackground=theme["hover_outline"] if hovered else (tone_color if interactive else theme["outline"]),
                    highlightcolor=theme["hover_outline"] if hovered else (tone_color if interactive else theme["outline"]),
                )
                widgets["icon"].configure(
                    bg=icon_bg,
                    fg=icon_fg,
                    text=icon_text if row_key == "gate" else widgets["icon"].cget("text"),
                    font=("Segoe UI", 8, "bold"),
                )
                widgets["label"].configure(
                    bg=row_bg,
                    fg=row_fg,
                    text=label_text if row_key == "gate" else widgets["label"].cget("text"),
                )
                widgets["status"].configure(bg=status_bg, fg=status_fg, text=status_text)
            except Exception:
                pass
        host._preview_overlay_dock_render_key = render_key

    try:
        width = int(
            dock.winfo_reqwidth()
            or dock.winfo_width()
            or getattr(host, "_preview_overlay_dock_last_width", 118 if expanded else 36)
            or (118 if expanded else 36)
        )
        height = int(
            dock.winfo_reqheight()
            or dock.winfo_height()
            or getattr(host, "_preview_overlay_dock_last_height", 96 if expanded else 30)
            or (96 if expanded else 30)
        )
    except Exception:
        width = int(getattr(host, "_preview_overlay_dock_last_width", 118 if expanded else 36) or (118 if expanded else 36))
        height = int(getattr(host, "_preview_overlay_dock_last_height", 96 if expanded else 30) or (96 if expanded else 30))
    min_width = 118 if compact else 138
    max_width = 178 if compact else 220
    min_height = 74 if compact else 136
    max_height = 210 if compact else 318
    width = int(max(min_width, min(max_width, width)))
    height = int(max(min_height, min(max_height, height)))
    host._preview_overlay_dock_last_width = width
    host._preview_overlay_dock_last_height = height
    return width, height


def place_preview_overlay_dock(host: "CharacterAnnotationTab", *, force_render: bool = False) -> None:
    dock = getattr(host, "preview_overlay_dock", None)
    canvas_host = getattr(host, "preview_canvas_host", None)
    if dock is None or canvas_host is None:
        return
    if not bool(getattr(host, "_preview_render_state", None)):
        suspend_drawer(host)
        try:
            dock.place_forget()
        except Exception:
            pass
        return

    try:
        host_width = int(canvas_host.winfo_width() or 0)
        host_height = int(canvas_host.winfo_height() or 0)
    except Exception:
        host_width = host_height = 0
    if host_width <= 180 or host_height <= 120:
        suspend_drawer(host)
        try:
            dock.place_forget()
        except Exception:
            pass
        return

    dock_width, dock_height = render_preview_overlay_dock(host, force_render=force_render)
    margin = 12.0
    toggle_rect = getattr(host, "_preview_fullscreen_toggle_rect", None)
    if isinstance(toggle_rect, tuple) and len(toggle_rect) == 4:
        try:
            _icon_x1, _icon_y1, icon_x2, icon_y2 = [float(value) for value in toggle_rect]
            x = icon_x2 - float(dock_width)
            y = icon_y2 + 6.0
        except Exception:
            x = float(host_width) - float(dock_width) - margin
            y = max(46.0, float(getattr(host, "_preview_overlay_top_bar_height", 38.0) or 38.0) + 12.0)
    else:
        x = float(host_width) - float(dock_width) - margin
        y = max(46.0, float(getattr(host, "_preview_overlay_top_bar_height", 38.0) or 38.0) + 12.0)
    x = min(max(margin, x), max(margin, float(host_width) - float(dock_width) - margin))
    y = min(max(margin, y), max(margin, float(host_height) - float(dock_height) - margin))
    try:
        slide = getattr(host, "_preview_drawer_slide", None)
        if slide is None:
            slide = host._preview_drawer_slide = PreviewDrawerSlide(host, host=canvas_host)
        slide.place(
            host_width, int(round(x)), int(round(y)), int(dock_width), int(dock_height),
            fullscreen=True,
            toggle_y=int(max(46, float(getattr(host, "_preview_overlay_top_bar_height", 38)) + 6)),
        )
    except Exception:
        pass
