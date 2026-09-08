from __future__ import annotations

"""
Zakładka: Panel kampanii i etapow projektu.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
from pathlib import Path
from textwrap import shorten
from collections import Counter
from dataclasses import dataclass
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
from .help_manager import HELP
from .app_theme_definitions import CAMPAIGN_SIDEBAR_STYLE
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel

_MOJIBAKE_MARKERS = ("Ä", "Ĺ", "Ă", "Å", "Â", "â€", "â†", "â")
_MOJIBAKE_FALLBACKS = {
    "Ä…": "ą",
    "Ä‡": "ć",
    "Ä™": "ę",
    "Ĺ‚": "ł",
    "Ĺ„": "ń",
    "Ăł": "ó",
    "Ĺ›": "ś",
    "Ĺş": "ź",
    "ĹĽ": "ż",
    "Ä„": "Ą",
    "Ä†": "Ć",
    "Ä": "Ę",
    "Ĺ": "Ł",
    "Ĺ": "Ń",
    "Ă“": "Ó",
    "Ĺš": "Ś",
    "Ĺą": "Ź",
    "Ĺ»": "Ż",
    "â€ž": "„",
    "â€ť": "”",
    "â€™": "’",
    "â€“": "-",
    "â€”": "-",
    "â†’": "→",
    "â†": "←",
    "â’": "−",
}


def _repair_polish_text(value):
    """Repair UTF-8 text that was accidentally decoded as Windows-1250."""
    if not isinstance(value, str) or not value:
        return value
    if not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value
    try:
        repaired = value.encode("cp1250").decode("utf-8")
        if repaired and sum(marker in repaired for marker in _MOJIBAKE_MARKERS) < sum(marker in value for marker in _MOJIBAKE_MARKERS):
            return repaired
    except Exception:
        pass
    repaired = value
    for wrong, right in _MOJIBAKE_FALLBACKS.items():
        repaired = repaired.replace(wrong, right)
    return repaired


def _repair_polish_widget_texts(root):
    if root is None:
        return
    try:
        text = root.cget("text")
        repaired = _repair_polish_text(text)
        if repaired != text:
            root.configure(text=repaired)
    except Exception:
        pass
    try:
        for child in root.winfo_children():
            _repair_polish_widget_texts(child)
    except Exception:
        pass


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
    event_num = getattr(event, "num", None)
    if event_num == 4:
        return -1
    if event_num == 5:
        return 1

    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        units = -int(delta / 120)
    else:
        units = -1 if delta > 0 else 1
    return units if units != 0 else (-1 if delta > 0 else 1)

def _mousewheel_magnitude(self, event) -> float:
    event_num = getattr(event, "num", None)
    if event_num in {4, 5}:
        return 1.0
    try:
        delta = abs(int(getattr(event, "delta", 0) or 0))
    except Exception:
        delta = 0
    if delta <= 0:
        return 1.0
    return max(1.0, min(3.0, float(delta) / 120.0))

def _compute_canvas_scroll_delta(self, canvas, units: int, *, magnitude: float = 1.0) -> float:
    if canvas is None or units == 0:
        return 0.0
    try:
        first, last = canvas.yview()
        span = max(0.02, float(last) - float(first))
        base_step = max(0.0035, min(0.018, span * 0.08))
        step = base_step * max(1.0, min(3.0, float(magnitude or 1.0)))
        return float(units) * step
    except Exception:
        return 0.0

def _compute_listbox_scroll_delta(self, listbox, units: int, *, magnitude: float = 1.0) -> float:
    if listbox is None or units == 0:
        return 0.0
    try:
        first, last = listbox.yview()
        span = max(0.02, float(last) - float(first))
        base_step = max(0.008, min(0.035, span * 0.12))
        step = base_step * max(1.0, min(3.0, float(magnitude or 1.0)))
        return float(units) * step
    except Exception:
        return 0.0

def _apply_canvas_scroll_delta(self, canvas, delta_fraction: float) -> bool:
    if canvas is None or abs(float(delta_fraction or 0.0)) < 1e-6:
        return False
    try:
        first, last = canvas.yview()
        first = float(first)
        last = float(last)
        span = max(0.02, last - first)
        target = max(0.0, min(max(0.0, 1.0 - span), first + float(delta_fraction)))
        if abs(target - first) < 1e-6:
            return False
        canvas.yview_moveto(target)
        return True
    except Exception:
        return False

def _apply_listbox_scroll_delta(self, listbox, delta_fraction: float) -> bool:
    if listbox is None or abs(float(delta_fraction or 0.0)) < 1e-6:
        return False
    try:
        first, last = listbox.yview()
        first = float(first)
        last = float(last)
        span = max(0.02, last - first)
        target = max(0.0, min(max(0.0, 1.0 - span), first + float(delta_fraction)))
        if abs(target - first) < 1e-6:
            return False
        listbox.yview_moveto(target)
        return True
    except Exception:
        return False

def _queue_scroll_inertia(self, widget, *, mode: str, delta_fraction: float) -> bool:
    if widget is None:
        return False
    delta_fraction = float(delta_fraction or 0.0)
    if abs(delta_fraction) < 1e-6:
        return False

    key = f"{mode}:{str(widget)}"
    state = self._scroll_inertia_jobs.get(key)
    if not isinstance(state, dict):
        state = {
            "widget": widget,
            "mode": str(mode or "").strip().lower(),
            "velocity": 0.0,
            "after_id": None,
        }
        self._scroll_inertia_jobs[key] = state

    velocity = float(state.get("velocity", 0.0) or 0.0) + delta_fraction
    state["velocity"] = max(-0.14, min(0.14, velocity))

    if state.get("after_id") is None:
        try:
            state["after_id"] = self.frame.after(14, lambda scroll_key=key: self._advance_scroll_inertia(scroll_key))
        except Exception:
            state["after_id"] = None
            return False
    return True

def _advance_scroll_inertia(self, key: str):
    state = self._scroll_inertia_jobs.get(str(key or "").strip())
    if not isinstance(state, dict):
        return

    state["after_id"] = None
    widget = state.get("widget")
    mode = str(state.get("mode", "") or "").strip().lower()
    velocity = float(state.get("velocity", 0.0) or 0.0)
    if abs(velocity) < 0.0007:
        self._scroll_inertia_jobs.pop(key, None)
        return

    if mode == "canvas":
        moved = self._apply_canvas_scroll_delta(widget, velocity)
    else:
        moved = self._apply_listbox_scroll_delta(widget, velocity)

    if not moved:
        self._scroll_inertia_jobs.pop(key, None)
        return

    state["velocity"] = velocity * 0.78
    if abs(float(state.get("velocity", 0.0) or 0.0)) < 0.0007:
        self._scroll_inertia_jobs.pop(key, None)
        return

    try:
        state["after_id"] = self.frame.after(14, lambda scroll_key=key: self._advance_scroll_inertia(scroll_key))
    except Exception:
        self._scroll_inertia_jobs.pop(key, None)

def _canvas_can_scroll(self, canvas, units: int) -> bool:
    if canvas is None or units == 0:
        return False
    try:
        first, last = canvas.yview()
        if units < 0 and float(first) <= 0.0:
            return False
        if units > 0 and float(last) >= 1.0:
            return False
        return True
    except Exception:
        return False

def _listbox_can_scroll(self, listbox, units: int) -> bool:
    if listbox is None or units == 0:
        return False
    try:
        first, last = listbox.yview()
        if units < 0 and float(first) <= 0.0:
            return False
        if units > 0 and float(last) >= 1.0:
            return False
        return True
    except Exception:
        return False

def _on_listbox_mousewheel(self, event, listbox):
    units = self._mousewheel_units(event)
    magnitude = self._mousewheel_magnitude(event)
    if listbox is None or units == 0:
        return None
    host_canvas = None
    if listbox is self.project_listbox:
        host_canvas = self.left_panel_canvas
    elif listbox is self.ingest_plan_listbox:
        host_canvas = self.right_panel_canvas
    if host_canvas is not None and self._canvas_can_scroll(host_canvas, units):
        try:
            delta_fraction = self._compute_canvas_scroll_delta(host_canvas, units, magnitude=magnitude)
            self._queue_scroll_inertia(host_canvas, mode="canvas", delta_fraction=delta_fraction)
        except Exception:
            pass
        return "break"
    if not self._listbox_can_scroll(listbox, units):
        return None
    try:
        delta_fraction = self._compute_listbox_scroll_delta(listbox, units, magnitude=magnitude)
        self._queue_scroll_inertia(listbox, mode="listbox", delta_fraction=delta_fraction)
    except Exception:
        pass
    return "break"

def _on_global_mousewheel(self, event):
    units = self._mousewheel_units(event)
    magnitude = self._mousewheel_magnitude(event)
    if units == 0:
        return None

    try:
        x_root = int(getattr(event, "x_root", 0) or self.frame.winfo_pointerx())
        y_root = int(getattr(event, "y_root", 0) or self.frame.winfo_pointery())
    except Exception:
        return None

    for listbox in (self.ingest_plan_listbox, self.project_listbox):
        if not self._widget_contains_point(listbox, x_root, y_root):
            continue
        try:
            host_canvas = None
            if listbox is self.project_listbox:
                host_canvas = self.left_panel_canvas
            elif listbox is self.ingest_plan_listbox:
                host_canvas = self.right_panel_canvas

            if host_canvas is not None and self._canvas_can_scroll(host_canvas, units):
                delta_fraction = self._compute_canvas_scroll_delta(host_canvas, units, magnitude=magnitude)
                self._queue_scroll_inertia(host_canvas, mode="canvas", delta_fraction=delta_fraction)
                return "break"
            if self._listbox_can_scroll(listbox, units):
                delta_fraction = self._compute_listbox_scroll_delta(listbox, units, magnitude=magnitude)
                self._queue_scroll_inertia(listbox, mode="listbox", delta_fraction=delta_fraction)
            return "break"
        except Exception:
            return "break"

    for canvas in (self.right_panel_canvas, self.left_panel_canvas):
        if not self._widget_contains_point(canvas, x_root, y_root):
            continue
        if not self._canvas_can_scroll(canvas, units):
            continue
        try:
            delta_fraction = self._compute_canvas_scroll_delta(canvas, units, magnitude=magnitude)
            self._queue_scroll_inertia(canvas, mode="canvas", delta_fraction=delta_fraction)
            return "break"
        except Exception:
            return None
    return None

def _bind_icon_button(self, canvas, *, role: str, command):
    if canvas is None:
        return

    canvas.bind("<Configure>", lambda _e, r=role: self._draw_icon_button(r), add="+")
    canvas.bind("<Enter>", lambda _e, r=role: self._set_icon_button_visual(r, hover=True), add="+")
    canvas.bind("<Leave>", lambda _e, r=role: self._set_icon_button_visual(r, hover=False, pressed=False), add="+")
    canvas.bind("<ButtonPress-1>", lambda _e, r=role: self._set_icon_button_visual(r, pressed=True), add="+")
    canvas.bind(
        "<ButtonRelease-1>",
        lambda e, r=role, cmd=command: self._on_icon_button_release(e, role=r, command=cmd),
        add="+",
    )
    self._draw_icon_button(role)

def _set_icon_button_enabled(self, role: str, enabled: bool):
    state = self._icon_button_state.setdefault(role, {"hover": False, "pressed": False, "enabled": True})
    state["enabled"] = bool(enabled)
    if not enabled:
        state["hover"] = False
        state["pressed"] = False
    self._draw_icon_button(role)

def _set_icon_button_visual(self, role: str, *, hover: bool | None = None, pressed: bool | None = None):
    state = self._icon_button_state.setdefault(role, {"hover": False, "pressed": False, "enabled": True})
    if not state.get("enabled", True):
        hover = False
        pressed = False
    if hover is not None:
        state["hover"] = bool(hover)
    if pressed is not None:
        state["pressed"] = bool(pressed)
    self._draw_icon_button(role)
    return "break"

def _on_icon_button_release(self, event, *, role: str, command):
    state = self._icon_button_state.setdefault(role, {"hover": False, "pressed": False, "enabled": True})
    state["pressed"] = False
    self._draw_icon_button(role)
    if not state.get("enabled", True) or command is None:
        return "break"
    try:
        if self._widget_contains_point(event.widget, int(event.x_root), int(event.y_root)):
            command()
    except Exception:
        pass
    return "break"

def _get_pil_font(self, size: int, *, bold: bool = False):
    cache_key = (int(size), bool(bold))
    cached = self._pil_font_cache.get(cache_key)
    if cached is not None:
        return cached

    font_names = (
        ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    )
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name, int(size))
            self._pil_font_cache[cache_key] = font
            return font
        except Exception:
            continue

    font = ImageFont.load_default()
    self._pil_font_cache[cache_key] = font
    return font

def _hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = str(color or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (0, 0, 0, int(alpha))
    try:
        rgb = tuple(int(value[idx:idx + 2], 16) for idx in (0, 2, 4))
        return (rgb[0], rgb[1], rgb[2], int(alpha))
    except Exception:
        return (0, 0, 0, int(alpha))

def _pick_readable_text_color(cls, background: str, *, dark_text: str = "#111111", light_text: str = "#f3f3f3") -> str:
    red, green, blue, _alpha = cls._hex_to_rgba(background, 255)
    luminance = ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)) / 255.0
    return dark_text if luminance >= 0.62 else light_text

def _format_tail_text(text: str, width: int = 42) -> str:
    value = str(text or "").replace("\\", "/").strip()
    if not value:
        return ""
    max_width = max(12, int(width or 42))
    if len(value) <= max_width:
        return value
    return "..." + value[-max(1, max_width - 3):]

def _style_project_start_badge_button(self, widget, *, host_bg: str | None = None):
    if widget is None:
        return

    palette = getattr(self.app, "palette", {})
    try:
        enabled = bool(getattr(widget, "_project_start_badge_enabled", True))
    except Exception:
        enabled = True
    try:
        hover = bool(getattr(widget, "_project_start_badge_hover", False))
    except Exception:
        hover = False
    try:
        tone = str(getattr(widget, "_project_start_badge_tone", "info") or "info").strip().lower()
    except Exception:
        tone = "info"

    base_bg = str(host_bg or palette.get("panel", "#252526"))
    crt_tone = palette.get("success", "#27ae60")
    tone_color = {
        "primary": palette.get("accent", "#2980b9"),
        "info": palette.get("accent", "#2980b9"),
        "success": palette.get("success", "#27ae60"),
        "crt": crt_tone,
        "warning": palette.get("warning", "#d35400"),
        "danger": palette.get("error", "#c0392b"),
        "muted": palette.get("muted", "#c7c7c7"),
    }.get(tone, palette.get("accent", "#2980b9"))
    if tone == "crt":
        field_bg = palette.get("field", "#1a1a1a")
        if enabled:
            badge_bg = blend_hex_colors(
                base_bg,
                crt_tone,
                0.24 if hover else 0.20,
            )
            badge_border = blend_hex_colors(crt_tone, base_bg, 0.18)
            badge_fg = palette.get("fg", "#f3f3f3")
        else:
            badge_bg = blend_hex_colors(base_bg, field_bg, 0.12)
            badge_border = blend_hex_colors(base_bg, field_bg, 0.26)
            badge_fg = palette.get("muted", "#c7c7c7")
    elif not enabled:
        tone_color = palette.get("muted_dim", "#777777")
        bg_mix = 0.30 if hover and enabled else 0.18
        badge_bg = blend_hex_colors(tone_color, base_bg, bg_mix)
        badge_border = blend_hex_colors(tone_color, base_bg, 0.08 if enabled else 0.42)
        badge_fg = self._pick_readable_text_color(badge_bg)
        if not enabled:
            badge_fg = palette.get("muted", "#999999")
    else:
        bg_mix = 0.30 if hover and enabled else 0.18
        badge_bg = blend_hex_colors(tone_color, base_bg, bg_mix)
        badge_border = blend_hex_colors(tone_color, base_bg, 0.08 if enabled else 0.42)
        badge_fg = self._pick_readable_text_color(badge_bg)

    try:
        widget.config(
            bg=badge_bg,
            fg=badge_fg,
            activebackground=badge_bg,
            activeforeground=badge_fg,
            highlightthickness=1,
            highlightbackground=badge_border,
            highlightcolor=badge_border,
            cursor=("hand2" if enabled else ""),
        )
    except Exception:
        pass

def _create_project_start_badge_button(self, parent, *, text: str, command, tone: str = "info"):
    palette = getattr(self.app, "palette", {})
    widget = tk.Label(
        parent,
        text=_repair_polish_text(str(text or "").strip()),
        font=("Segoe UI", 8),
        padx=7,
        pady=2,
        bd=0,
        relief=tk.FLAT,
        bg=palette.get("panel", "#252526"),
        fg=palette.get("fg", "#f3f3f3"),
        anchor="center",
        justify=tk.CENTER,
        takefocus=0,
    )
    widget._project_start_badge_command = command
    widget._project_start_badge_enabled = True
    widget._project_start_badge_tone = str(tone or "info").strip().lower() or "info"
    widget._project_start_badge_hover = False

    def _on_click(_event=None, target=widget):
        if not bool(getattr(target, "_project_start_badge_enabled", True)):
            return
        callback = getattr(target, "_project_start_badge_command", None)
        if callable(callback):
            callback()

    def _on_enter(_event=None, target=widget):
        target._project_start_badge_hover = True
        self._style_project_start_badge_button(target)

    def _on_leave(_event=None, target=widget):
        target._project_start_badge_hover = False
        self._style_project_start_badge_button(target)

    widget.bind("<Button-1>", _on_click, add="+")
    widget.bind("<Enter>", _on_enter, add="+")
    widget.bind("<Leave>", _on_leave, add="+")
    self._style_project_start_badge_button(widget)
    return widget

def _set_project_start_badge_button_state(self, widget, *, enabled: bool | None = None, text: str | None = None, tone: str | None = None):
    if widget is None:
        return
    if enabled is not None:
        try:
            widget._project_start_badge_enabled = bool(enabled)
        except Exception:
            pass
    if text is not None:
        try:
            widget.config(text=_repair_polish_text(str(text or "").strip()))
        except Exception:
            pass
    if tone is not None:
        try:
            widget._project_start_badge_tone = str(tone or "info").strip().lower() or "info"
        except Exception:
            pass
    self._style_project_start_badge_button(widget)

def _draw_project_add_action(canvas, palette, state):
    width = int(canvas.winfo_width())
    if width < 2:
        return
    enabled = bool(state.get("enabled", True))
    font_spec = CAMPAIGN_SIDEBAR_STYLE["project_action_font"]
    metrics = tkfont.Font(root=canvas, font=font_spec)
    padding, gap = 8, 10
    diameter = max(28, round(metrics.metrics("linespace") * 1.2))
    label_x = padding + diameter + gap
    available = max(1, width - label_x - padding)
    color = palette["success"] if enabled else palette["muted"]
    icon_color = palette["fg"] if enabled else palette["muted"]
    outline = blend_hex_colors(icon_color, str(canvas.cget("bg")), 0.34 if enabled else 0.60)
    canvas.delete("all")
    canvas.configure(cursor="hand2" if enabled else "arrow")
    label = canvas.create_text(
        label_x, padding, text="Utwórz projekt", anchor="nw", width=available,
        font=font_spec, fill=color, tags=("project_action_label",),
    )
    bounds = canvas.bbox(label)
    text_height = bounds[3] - bounds[1]
    height = max(38, text_height + 2*padding, diameter + 2*padding)
    if int(canvas.cget("height")) != height:
        canvas.configure(height=height)
    canvas.itemconfigure(label, anchor="w")
    canvas.coords(label, label_x, height/2)
    cx, cy = padding + diameter/2, height/2
    canvas.create_oval(padding, cy-diameter/2, padding+diameter, cy+diameter/2,
                       outline=outline, width=2, tags=("project_action_icon",))
    half = diameter * 0.22
    canvas.create_line(cx-half, cy, cx+half, cy, fill=icon_color, width=2,
                       capstyle=tk.ROUND, tags=("project_action_icon",))
    canvas.create_line(cx, cy-half, cx, cy+half, fill=icon_color, width=2,
                       capstyle=tk.ROUND, tags=("project_action_icon",))


def _draw_icon_button(self, role: str):
    canvas = (
        self.project_add_button_canvas
        if role == "project_add"
        else self.wizard_exit_button_canvas
    )
    if canvas is None:
        return

    palette = getattr(self.app, "palette", {})
    bg = str(canvas.cget("bg") or palette.get("panel", "#252526"))
    state = self._icon_button_state.setdefault(role, {"hover": False, "pressed": False, "enabled": True})
    if role == "project_add":
        # Native text measures DPI and wraps within the actual sidebar viewport.
        _draw_project_add_action(canvas, palette, state)
        return
    enabled = bool(state.get("enabled", True))
    hover = bool(state.get("hover", False))
    pressed = bool(state.get("pressed", False))
    width = max(int(canvas.winfo_width() or int(canvas.cget("width") or 34)), 24)
    height = max(int(canvas.winfo_height() or int(canvas.cget("height") or 34)), 24)

    canvas.delete("all")
    cursor = "hand2" if enabled else "arrow"
    try:
        canvas.config(cursor=cursor)
    except Exception:
        pass

    label_text = _repair_polish_text("Utwórz projekt" if role == "project_add" else "Wyjdź z projektu")

    if PIL_AVAILABLE and Image is not None and ImageTk is not None and ImageDraw is not None and ImageFont is not None:
        scale = 4
        hi_w = max(width * scale, 4)
        hi_h = max(height * scale, 4)
        image = Image.new("RGBA", (hi_w, hi_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")

        if role == "project_add":
            neutral = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted_dim", "#8c8c8c")
            success = palette.get("success", "#27ae60") if enabled else palette.get("muted", "#c7c7c7")
            outline = blend_hex_colors(neutral, bg, 0.34 if enabled else 0.60)
            icon_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")
            label_color = success
        else:
            neutral = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted_dim", "#8c8c8c")
            outline = blend_hex_colors(neutral, bg, 0.34 if enabled else 0.60)
            icon_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")
            label_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")

        circle_d = min(hi_h - (6 * scale), 28 * scale)
        circle_left = 6 * scale
        circle_top = max(scale, (hi_h - circle_d) // 2)
        circle_box = [circle_left, circle_top, circle_left + circle_d, circle_top + circle_d]
        inner_box = [circle_box[0] + scale, circle_box[1] + scale, circle_box[2] - scale, circle_box[3] - scale]

        if role == "project_add":
            draw.ellipse(circle_box, fill=self._hex_to_rgba(bg, 0), outline=self._hex_to_rgba(outline), width=max(scale, 2))
        else:
            draw.ellipse(circle_box, fill=self._hex_to_rgba(bg, 0), outline=self._hex_to_rgba(outline), width=max(scale, 2))

        cx = (inner_box[0] + inner_box[2]) / 2.0
        cy = (inner_box[1] + inner_box[3]) / 2.0
        if role == "project_add":
            line_w = max(scale, 3)
            line_len = 6 * scale
            draw.line(
                (cx - line_len, cy, cx + line_len, cy),
                fill=self._hex_to_rgba(icon_color),
                width=line_w,
            )
            draw.line(
                (cx, cy - line_len, cx, cy + line_len),
                fill=self._hex_to_rgba(icon_color),
                width=line_w,
            )
        else:
            stop_half = 4.6 * scale
            draw.rectangle(
                [cx - stop_half, cy - stop_half, cx + stop_half, cy + stop_half],
                outline=self._hex_to_rgba(icon_color),
                fill=self._hex_to_rgba(bg, 0),
                width=max(scale, 2),
            )

        font = self._get_pil_font(13 * scale, bold=False)
        text_x = circle_box[2] + (8 * scale)
        try:
            text_box = draw.textbbox((0, 0), label_text, font=font)
            text_h = max(1, text_box[3] - text_box[1])
        except Exception:
            text_h = 10 * scale
        text_y = max(0, int((hi_h - text_h) / 2) - (scale // 2))
        draw.text((text_x, text_y), label_text, font=font, fill=self._hex_to_rgba(label_color))

        try:
            resampling = Image.Resampling.LANCZOS
        except Exception:
            resampling = Image.LANCZOS
        image = image.resize((width, height), resampling)
        photo = ImageTk.PhotoImage(image)
        self._icon_button_images[role] = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        return

    if role == "project_add":
        neutral = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted_dim", "#8c8c8c")
        success = palette.get("success", "#27ae60") if enabled else palette.get("muted", "#c7c7c7")
        outline = blend_hex_colors(neutral, bg, 0.34 if enabled else 0.60)
        icon_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")
        label_color = success
    else:
        neutral = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted_dim", "#8c8c8c")
        outline = blend_hex_colors(neutral, bg, 0.34 if enabled else 0.60)
        icon_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")
        label_color = palette.get("fg", "#f3f3f3") if enabled else palette.get("muted", "#c7c7c7")

    circle_top = max(3, int((height - 28) / 2))
    circle_box = (4, circle_top, 32, circle_top + 28)
    if role == "project_add":
        canvas.create_oval(*circle_box, fill="", outline=outline, width=2)
    else:
        canvas.create_oval(*circle_box, fill="", outline=outline, width=2)

    cx = (circle_box[0] + circle_box[2]) / 2.0
    cy = (circle_box[1] + circle_box[3]) / 2.0
    if role == "project_add":
        canvas.create_line(cx - 6, cy, cx + 6, cy, fill=icon_color, width=2, capstyle=tk.ROUND)
        canvas.create_line(cx, cy - 6, cx, cy + 6, fill=icon_color, width=2, capstyle=tk.ROUND)
    else:
        canvas.create_rectangle(cx - 4.5, cy - 4.5, cx + 4.5, cy + 4.5, outline=icon_color, width=2)
    canvas.create_text(42, height / 2.0, text=label_text, anchor=tk.W, fill=label_color, font=("Segoe UI", 12, "normal"))
