#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theme, styling and themed dialog runtime for the main GUI app."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, simpledialog

from ..config import CONFIG, logger
from .app_theme_definitions import THEME_PALETTE_CONTRACT
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors

APP_AUTHOR = "R. Szuderski"


def _is_guided_style(self, style_name: str) -> bool:
    return style_name in {
        "GuidedNeutral.TButton",
        "GuidedAccent.TButton",
    }

def _remember_guided_base_style(self, button, base_style: str = None) -> str:
    style_name = base_style or getattr(button, "_guided_base_style", None) or str(button.cget("style") or "").strip() or "TButton"
    if self._is_guided_style(style_name):
        style_name = getattr(button, "_guided_base_style", "TButton")
    button._guided_base_style = style_name
    return style_name

def _get_guided_styles_for_button(self, button, base_style: str = None):
    style_name = self._remember_guided_base_style(button, base_style)
    is_accent = ("Accent" in style_name) or style_name == "Accent.TButton"
    emphasis_style = "GuidedAccent.TButton" if is_accent else "GuidedNeutral.TButton"
    return style_name, emphasis_style

def set_button_emphasis(self, button, enabled: bool, base_style: str = None):
    if button is None:
        return

    try:
        base_name, emphasis_style = self._get_guided_styles_for_button(button, base_style)
        button.configure(style=(emphasis_style if enabled else base_name))
    except Exception as e:
        logger.debug(f"Nie udało się ustawić podświetlenia przycisku: {e}")

def style_guidance_frame(self, frame, background: str = None, emphasized: bool = None):
    if frame is None:
        return

    palette = self.palette
    base_border = palette["panel_border"]
    emphasis_border = blend_hex_colors(
        palette["success"],
        palette["panel_border"],
        0.18,
    )
    bg = background or getattr(frame, "_guided_frame_bg", None) or palette["panel"]
    is_emphasized = getattr(frame, "_guided_frame_emphasized", False) if emphasized is None else bool(emphasized)
    border = emphasis_border if is_emphasized else base_border

    frame._guided_frame_bg = bg
    frame._guided_frame_emphasized = is_emphasized

    try:
        frame.configure(
            bg=bg,
            bd=0,
            relief=tk.FLAT,
            highlightthickness=(1 if is_emphasized else 0),
            highlightbackground=border,
            highlightcolor=border
        )
    except Exception as e:
        logger.debug(f"Nie udało się wystylizować ramki prowadzenia: {e}")

def set_frame_emphasis(self, frame, enabled: bool, background: str = None):
    if frame is None:
        return

    try:
        self.style_guidance_frame(frame, background=background, emphasized=enabled)
    except Exception as e:
        logger.debug(f"Nie udało się ustawić podświetlenia ramki: {e}")

def set_theme(self, theme_key: str, persist: bool = True, announce: bool = True):
    if theme_key not in self.themes:
        return

    self._setup_style(theme_key)
    self._restyle_shell()
    self._schedule_theme_refresh_finalize()

    if persist:
        self._save_theme_preference(theme_key)

    if announce:
        self.update_status(
            f"Aktywny styl: {self.current_theme_name}.",
            "info"
        )

def _restyle_shell(self):
    palette = self.palette

    try:
        self.root.configure(bg=palette["bg"])
    except Exception:
        pass

    try:
        self._create_menu()
    except Exception:
        pass

    try:
        self._apply_help_panel_visual_state()
    except Exception:
        pass

    try:
        overlay = getattr(self, "_free_mode_assistant_overlay", None)
        if overlay is not None:
            overlay.refresh_theme(palette)
            self._schedule_free_mode_assistant_placement()
        self._sync_free_mode_assistant_toggle_state()
    except Exception:
        pass

    try:
        self.refresh_window_title()
    except Exception:
        pass

    self._apply_theme_to_tabs()

def _apply_theme_to_tabs(self):
    try:
        for tab in getattr(self, "tabs", {}).values():
            apply_theme = getattr(tab, "apply_theme", None)
            if callable(apply_theme):
                apply_theme()
    except Exception:
        pass

def _apply_theme_to_widget_tree(self, root):
    if root is None:
        return

    visited: set[int] = set()

    def walk(widget):
        if widget is None:
            return

        widget_id = id(widget)
        if widget_id in visited:
            return
        visited.add(widget_id)

        apply_theme = getattr(widget, "apply_theme", None)
        if callable(apply_theme):
            try:
                apply_theme()
            except Exception:
                pass

        if isinstance(widget, WebSlimScrollbar):
            try:
                track = self._resolve_widget_background(getattr(widget, "master", None))
                self.style_web_scrollbar(widget, track_color=track)
            except Exception:
                pass

        try:
            for child in widget.winfo_children():
                walk(child)
        except Exception:
            pass

    walk(root)

def _emit_theme_repaint_pulse(self):
    """Force custom drawn widgets to repaint after a palette switch."""
    visited: set[int] = set()

    def _safe_event_generate(widget, sequence: str):
        try:
            widget.event_generate(sequence)
        except Exception:
            pass

    def _safe_configure_event(widget):
        try:
            width = max(1, int(widget.winfo_width() or 1))
            height = max(1, int(widget.winfo_height() or 1))
            widget.event_generate("<Configure>", width=width, height=height)
        except Exception:
            _safe_event_generate(widget, "<Configure>")

    def walk(widget):
        if widget is None:
            return

        widget_id = id(widget)
        if widget_id in visited:
            return
        visited.add(widget_id)

        try:
            if isinstance(widget, tk.Canvas):
                _safe_configure_event(widget)
                _safe_event_generate(widget, "<Expose>")
            else:
                class_name = str(widget.winfo_class() or "")
                if class_name in {"Panedwindow", "TPanedwindow", "TNotebook"}:
                    _safe_configure_event(widget)
        except Exception:
            pass

        try:
            for child in widget.winfo_children():
                walk(child)
        except Exception:
            pass

    walk(self.root)

    try:
        self.root.event_generate("<<AppThemeChanged>>")
    except Exception:
        pass

def _finalize_theme_refresh(self):
    self._theme_refresh_after_id = None

    try:
        self._apply_theme_to_widget_tree(self.root)
    except Exception:
        pass

    try:
        self._emit_theme_repaint_pulse()
    except Exception:
        pass

    try:
        self.root.update_idletasks()
    except Exception:
        pass

def _schedule_theme_refresh_finalize(self):
    pending = getattr(self, "_theme_refresh_after_id", None)
    if pending is not None:
        try:
            self.root.after_cancel(pending)
        except Exception:
            pass
        self._theme_refresh_after_id = None

    for pending_id in list(getattr(self, "_theme_refresh_after_ids", []) or []):
        try:
            self.root.after_cancel(pending_id)
        except Exception:
            pass
    self._theme_refresh_after_ids = []

    self._finalize_theme_refresh()

    def _queue_refresh(delay_ms: int | None = None):
        token = {"id": None}

        def _run():
            pending_id = token.get("id")
            try:
                if pending_id in self._theme_refresh_after_ids:
                    self._theme_refresh_after_ids.remove(pending_id)
            except Exception:
                pass
            self._finalize_theme_refresh()

        try:
            if delay_ms is None:
                token["id"] = self.root.after_idle(_run)
            else:
                token["id"] = self.root.after(int(delay_ms), _run)
            self._theme_refresh_after_ids.append(token["id"])
            self._theme_refresh_after_id = token["id"]
        except Exception:
            self._theme_refresh_after_id = None

    _queue_refresh(None)
    _queue_refresh(60)
    try:
        _queue_refresh(180)
    except Exception:
        self._theme_refresh_after_id = None

def style_native_scrollbar(self, scrollbar, background: str = None, troughcolor: str = None, bordercolor: str = None):
    if scrollbar is None:
        return

    palette = getattr(self, "palette", {})
    trough = troughcolor or palette["panel"]
    _track, thumb, thumb_hover = self._get_scrollbar_colors(track_color=trough)
    bg = thumb if self.is_dark_theme() else (background or thumb)
    border = bordercolor or palette["panel_border"]
    active_bg = thumb_hover

    options = {
        "bg": bg,
        "activebackground": active_bg,
        "troughcolor": trough,
        "highlightbackground": border,
        "highlightcolor": border,
        "highlightthickness": 0,
        "bd": 0,
        "borderwidth": 0,
        "relief": tk.FLAT,
        "activerelief": tk.FLAT,
        "elementborderwidth": 0,
        "width": 8,
    }

    for option_name, option_value in options.items():
        try:
            scrollbar.configure(**{option_name: option_value})
        except Exception:
            pass

def is_dark_theme(self) -> bool:
    return str(getattr(self, "current_theme_key", "") or "").strip().lower().startswith("dark")

def _get_scrollbar_colors(self, track_color: str = None) -> tuple[str, str, str]:
    palette = getattr(self, "palette", {})
    track = track_color or palette["scrollbar_track"]
    return track, palette["scrollbar_thumb"], palette["scrollbar_thumb_hover"]

def style_text_widget(self, widget, role: str = "default"):
    if widget is None:
        return

    palette = getattr(self, "palette", {})
    role_key = str(role or "default").strip().lower()

    if role_key == "console":
        bg = palette["console_bg"]
        fg = palette["console_fg"]
        border = palette["console_border"]
    elif role_key == "doc":
        bg = palette["doc_bg"]
        fg = palette["doc_fg"]
        border = palette["console_border"]
    else:
        bg = palette["field"]
        fg = palette["fg"]
        border = palette["border"]

    options = {
        "bg": bg,
        "fg": fg,
        "insertbackground": fg,
        "bd": 0,
        "relief": tk.FLAT,
        "highlightthickness": 1,
        "highlightbackground": border,
        "highlightcolor": border,
    }

    for option_name, option_value in options.items():
        try:
            widget.configure(**{option_name: option_value})
        except Exception:
            pass

    seen_scrollbars: set[int] = set()
    for attr_name in ("vbar", "web_vbar"):
        scrollbar = getattr(widget, attr_name, None)
        if scrollbar is None:
            continue

        scrollbar_id = id(scrollbar)
        if scrollbar_id in seen_scrollbars:
            continue
        seen_scrollbars.add(scrollbar_id)

        if isinstance(scrollbar, WebSlimScrollbar):
            self.style_web_scrollbar(scrollbar, track_color=bg)
        else:
            self.style_native_scrollbar(
                scrollbar,
                background=palette["panel_alt"],
                troughcolor=bg,
                bordercolor=border
            )

def style_listbox_widget(self, widget, bordercolor: str = None):
    if widget is None:
        return

    palette = getattr(self, "palette", {})
    border = bordercolor or palette["panel_border"]
    select_bg, select_fg = self.get_list_selection_colors()

    options = {
        "bg": palette["field"],
        "fg": palette["fg"],
        "selectbackground": select_bg,
        "selectforeground": select_fg,
        "disabledforeground": palette["muted_dim"],
        "highlightthickness": 1,
        "highlightbackground": border,
        "highlightcolor": border,
        "bd": 0,
        "relief": tk.FLAT,
    }

    for option_name, option_value in options.items():
        try:
            widget.configure(**{option_name: option_value})
        except Exception:
            pass

def style_web_scrollbar(self, scrollbar, track_color: str = None):
    if scrollbar is None or not isinstance(scrollbar, WebSlimScrollbar):
        return

    track, thumb, thumb_hover = self._get_scrollbar_colors(track_color=track_color)

    try:
        scrollbar.configure_style(
            track_color=track,
            thumb_color=thumb,
            thumb_hover_color=thumb_hover,
        )
    except Exception:
        pass

def style_canvas_widget(self, widget, background: str = None, bordercolor: str = None):
    if widget is None:
        return

    palette = getattr(self, "palette", {})
    bg = background or palette["panel"]
    border = bordercolor or palette["panel_border"]

    options = {
        "bg": bg,
        "highlightthickness": 1,
        "highlightbackground": border,
        "highlightcolor": border,
        "bd": 0,
        "relief": tk.FLAT,
    }

    for option_name, option_value in options.items():
        try:
            widget.configure(**{option_name: option_value})
        except Exception:
            pass

def _get_scale_colors(self, background: str = None) -> tuple[str, str, str, str, str]:
    palette = getattr(self, "palette", {})
    bg = self._coerce_color_hex(
        background,
        fallback=palette["panel"],
    )
    success = palette["progress_fill"]
    track = blend_hex_colors(success, bg, 0.45)
    thumb = success
    thumb_hover = blend_hex_colors(thumb, palette["fg"], 0.18)
    # Keep disabled sliders visually consistent with enabled ones; the non-interactive
    # state is conveyed by behavior, not by washing out the green marker.
    thumb_disabled = thumb
    return bg, track, thumb, thumb_hover, thumb_disabled

@staticmethod
def _paint_scale_track_image(image, color: str):
    if image is None:
        return
    try:
        image.blank()
        image.put(str(color), to=(0, 0, int(image.width()), int(image.height())))
    except Exception:
        pass

@staticmethod
def _paint_scale_thumb_image(image, fill_color: str, outline_color: str = None):
    if image is None:
        return

    try:
        image.blank()
        width = int(image.width())
        height = int(image.height())
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        radius = max(2.0, (min(width, height) / 2.0) - 1.0)
        outline_threshold = radius - 1.15

        for y in range(height):
            for x in range(width):
                dist = ((float(x) - cx) ** 2 + (float(y) - cy) ** 2) ** 0.5
                if dist > radius:
                    continue
                color = outline_color if outline_color and dist >= outline_threshold else fill_color
                image.put(str(color), to=(x, y, x + 1, y + 1))
    except Exception:
        pass

@staticmethod
def _style_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "default"
    return "".join(ch if ch.isalnum() else "_" for ch in raw) or "default"

def get_list_selection_colors(self) -> tuple[str, str]:
    """High-contrast selection colors for native list widgets."""
    palette = getattr(self, "palette", {})
    return str(palette["selection_bg"]), str(palette["selection_fg"])

def _coerce_color_hex(self, color: str = None, fallback: str = None) -> str:
    palette = getattr(self, "palette", None) or THEME_PALETTE_CONTRACT
    default = str(
        fallback
        or palette["panel"]
    ).strip()
    raw = str(color or "").strip() or default

    def _normalize_hex(value: str):
        candidate = str(value or "").strip()
        if not candidate.startswith("#"):
            return None
        token = candidate[1:]
        if len(token) == 3:
            token = "".join(ch * 2 for ch in token)
        if len(token) != 6:
            return None
        try:
            int(token, 16)
        except Exception:
            return None
        return f"#{token.lower()}"

    normalized = _normalize_hex(raw)
    if normalized:
        return normalized

    try:
        red, green, blue = self.root.winfo_rgb(raw)
        return f"#{red // 256:02x}{green // 256:02x}{blue // 256:02x}"
    except Exception:
        pass

    normalized_default = _normalize_hex(default)
    if normalized_default:
        return normalized_default
    return THEME_PALETTE_CONTRACT["panel"]

def _resolve_widget_background(self, widget, fallback: str = None) -> str:
    palette = getattr(self, "palette", None) or THEME_PALETTE_CONTRACT
    default_bg = self._coerce_color_hex(fallback, fallback=palette["panel"])
    current = widget
    visited: set[int] = set()

    while current is not None:
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)

        for option_name in ("bg", "background"):
            try:
                value = str(current.cget(option_name) or "").strip()
            except Exception:
                value = ""
            if value:
                resolved = self._coerce_color_hex(value, fallback=None)
                if resolved:
                    return resolved

        try:
            class_name = str(current.winfo_class() or "")
        except Exception:
            class_name = ""
        if class_name.startswith("T"):
            style_name = ""
            try:
                style_name = str(current.cget("style") or "").strip()
            except Exception:
                style_name = ""
            for lookup_style in (style_name, class_name):
                if not lookup_style:
                    continue
                try:
                    value = str(self.style.lookup(lookup_style, "background") or "").strip()
                except Exception:
                    value = ""
                if value:
                    resolved = self._coerce_color_hex(value, fallback=None)
                    if resolved:
                        return resolved

        current = getattr(current, "master", None)

    return default_bg

def _ensure_horizontal_scale_style_assets(self, background: str = None, style_name: str = "Horizontal.TScale"):
    if not hasattr(self, "_horizontal_scale_style_assets"):
        self._horizontal_scale_style_assets = {}

    normalized_style = str(style_name or "Horizontal.TScale").strip() or "Horizontal.TScale"
    style_token = self._style_token(normalized_style)
    assets = self._horizontal_scale_style_assets.setdefault(normalized_style, {})
    if not assets:
        assets["track"] = tk.PhotoImage(master=self.root, width=16, height=4)
        assets["thumb"] = tk.PhotoImage(master=self.root, width=14, height=14)
        assets["thumb_active"] = tk.PhotoImage(master=self.root, width=14, height=14)
        assets["thumb_disabled"] = tk.PhotoImage(master=self.root, width=14, height=14)
        assets["trough_element"] = f"{style_token}.Horizontal.Scale.trough"
        assets["slider_element"] = f"{style_token}.Horizontal.Scale.slider"

        try:
            self.style.element_create(
                assets["trough_element"],
                "image",
                assets["track"],
                border=0,
                sticky="ew",
            )
        except Exception:
            pass

        try:
            self.style.element_create(
                assets["slider_element"],
                "image",
                assets["thumb"],
                ("active", assets["thumb_active"]),
                ("pressed", assets["thumb_active"]),
                ("disabled", assets["thumb_disabled"]),
                border=0,
                sticky="",
            )
        except Exception:
            pass

    bg, track, thumb, thumb_hover, thumb_disabled = self._get_scale_colors(background)
    thumb_outline = blend_hex_colors(thumb, bg, 0.35)
    thumb_disabled_outline = blend_hex_colors(thumb_disabled, bg, 0.45)

    self._paint_scale_track_image(assets.get("track"), track)
    self._paint_scale_thumb_image(assets.get("thumb"), thumb, thumb_outline)
    self._paint_scale_thumb_image(assets.get("thumb_active"), thumb_hover, thumb_outline)
    self._paint_scale_thumb_image(assets.get("thumb_disabled"), thumb_disabled, thumb_disabled_outline)

    try:
        self.style.layout(
            normalized_style,
            [
                (
                    assets["trough_element"],
                    {
                        "sticky": "ew",
                        "children": [
                            (assets["slider_element"], {"side": "left", "sticky": ""})
                        ],
                    },
                )
            ],
        )
    except Exception:
        pass

    try:
        self.style.configure(
            normalized_style,
            background=bg,
            borderwidth=0,
            relief=tk.FLAT,
            sliderlength=14,
            troughcolor=track,
            lightcolor=track,
            darkcolor=track,
        )
    except Exception:
        pass

    try:
        self.style.map(
            normalized_style,
            background=[
                ("disabled", bg),
                ("active", bg),
            ],
            troughcolor=[
                ("disabled", track),
                ("active", track),
            ],
            lightcolor=[
                ("disabled", track),
                ("active", track),
            ],
            darkcolor=[
                ("disabled", track),
                ("active", track),
            ],
        )
    except Exception:
        pass

def style_ttk_scale_widget(self, widget, background: str = None, base_style: str = None) -> str:
    if widget is None:
        return str(base_style or "Horizontal.TScale")

    try:
        current_style = str(widget.cget("style") or "").strip()
    except Exception:
        current_style = ""

    resolved_base_style = str(
        base_style
        or getattr(widget, "_base_ttk_scale_style", "")
        or current_style
        or "Horizontal.TScale"
    ).strip() or "Horizontal.TScale"
    if ".AutoBg_" in resolved_base_style:
        resolved_base_style = resolved_base_style.split(".AutoBg_", 1)[0] or "Horizontal.TScale"

    resolved_bg = self._coerce_color_hex(
        background,
        fallback=self._resolve_widget_background(getattr(widget, "master", None), fallback=background),
    )
    style_name = f"{resolved_base_style}.AutoBg_{self._style_token(resolved_bg)}"
    self._ensure_horizontal_scale_style_assets(background=resolved_bg, style_name=style_name)

    try:
        widget._base_ttk_scale_style = resolved_base_style
    except Exception:
        pass

    try:
        widget.configure(style=style_name, cursor="hand2", takefocus=0)
    except Exception:
        pass

    return style_name

def style_ttk_frame_widget(self, widget, background: str = None, base_style: str = None) -> str:
    if widget is None:
        return str(base_style or "TFrame")

    try:
        current_style = str(widget.cget("style") or "").strip()
    except Exception:
        current_style = ""

    resolved_base_style = str(
        base_style
        or getattr(widget, "_base_ttk_frame_style", "")
        or current_style
        or "TFrame"
    ).strip() or "TFrame"
    if resolved_base_style.startswith("AutoBg_") and "." in resolved_base_style:
        resolved_base_style = resolved_base_style.split(".", 1)[1] or "TFrame"

    resolved_bg = self._coerce_color_hex(
        background,
        fallback=self._resolve_widget_background(getattr(widget, "master", None), fallback=background),
    )
    style_name = f"AutoBg_{self._style_token(resolved_bg)}.{resolved_base_style}"

    try:
        self.style.configure(style_name, background=resolved_bg)
    except Exception:
        pass

    try:
        widget._base_ttk_frame_style = resolved_base_style
    except Exception:
        pass

    try:
        widget.configure(style=style_name)
    except Exception:
        pass

    return style_name

def style_ttk_panedwindow_widget(self, widget, background: str = None, base_style: str = None) -> str:
    if widget is None:
        return str(base_style or "TPanedwindow")

    try:
        current_style = str(widget.cget("style") or "").strip()
    except Exception:
        current_style = ""

    resolved_base_style = str(
        base_style
        or getattr(widget, "_base_ttk_panedwindow_style", "")
        or current_style
        or "TPanedwindow"
    ).strip() or "TPanedwindow"
    if resolved_base_style.startswith("AutoBg_") and "." in resolved_base_style:
        resolved_base_style = resolved_base_style.split(".", 1)[1] or "TPanedwindow"

    resolved_bg = self._coerce_color_hex(
        background,
        fallback=self._resolve_widget_background(getattr(widget, "master", None), fallback=background),
    )
    style_name = f"AutoBg_{self._style_token(resolved_bg)}.{resolved_base_style}"

    try:
        self.style.configure(style_name, background=resolved_bg)
    except Exception:
        pass

    try:
        widget._base_ttk_panedwindow_style = resolved_base_style
    except Exception:
        pass

    try:
        widget.configure(style=style_name)
    except Exception:
        pass

    return style_name

def style_ttk_labelframe_widget(self, widget, background: str = None, base_style: str = None) -> str:
    if widget is None:
        return str(base_style or "TLabelframe")

    try:
        current_style = str(widget.cget("style") or "").strip()
    except Exception:
        current_style = ""

    resolved_base_style = str(
        base_style
        or getattr(widget, "_base_ttk_labelframe_style", "")
        or current_style
        or "TLabelframe"
    ).strip() or "TLabelframe"
    if resolved_base_style.startswith("AutoBg_") and "." in resolved_base_style:
        resolved_base_style = resolved_base_style.split(".", 1)[1] or "TLabelframe"

    resolved_bg = self._coerce_color_hex(
        background,
        fallback=self._resolve_widget_background(getattr(widget, "master", None), fallback=background),
    )
    palette = getattr(self, "palette", {})
    border = palette["panel_border"]
    fg = palette["fg"]
    style_name = f"AutoBg_{self._style_token(resolved_bg)}.{resolved_base_style}"
    label_style_name = f"{style_name}.Label"

    try:
        self.style.configure(
            style_name,
            background=resolved_bg,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            borderwidth=1,
            relief=tk.SOLID,
        )
    except Exception:
        pass

    try:
        current_font = self.style.lookup(f"{resolved_base_style}.Label", "font") or ("Segoe UI", 10, "bold")
    except Exception:
        current_font = ("Segoe UI", 10, "bold")

    try:
        self.style.configure(
            label_style_name,
            background=resolved_bg,
            foreground=fg,
            font=current_font,
        )
    except Exception:
        pass

    try:
        widget._base_ttk_labelframe_style = resolved_base_style
    except Exception:
        pass

    try:
        widget.configure(style=style_name)
    except Exception:
        pass

    return style_name

def _update_adaptive_wraplength(self, widget):
    if widget is None:
        return

    if bool(getattr(self, "_main_window_hidden_for_startup", False)) and not bool(getattr(self, "_main_window_revealed", False)):
        return

    try:
        base_wrap = int(float(getattr(widget, "_adaptive_wrap_base", 0) or 0))
    except Exception:
        base_wrap = 0
    if base_wrap <= 0:
        return

    container = getattr(widget, "_adaptive_wrap_container", None) or getattr(widget, "master", None)
    mapped = False
    for candidate in (container, widget):
        if candidate is None:
            continue
        try:
            if bool(candidate.winfo_ismapped()):
                mapped = True
                break
        except Exception:
            pass
    if not mapped:
        return

    width = 0
    for candidate in (container, widget):
        if candidate is None:
            continue
        try:
            width = int(candidate.winfo_width() or 0)
        except Exception:
            width = 0
        if width > 1:
            break

    if width <= 1:
        return

    try:
        padding = int(getattr(widget, "_adaptive_wrap_padding", 18))
    except Exception:
        padding = 18
    try:
        min_wrap = int(getattr(widget, "_adaptive_wrap_min", 80))
    except Exception:
        min_wrap = 80
    try:
        max_wrap = int(getattr(widget, "_adaptive_wrap_max", base_wrap) or base_wrap)
    except Exception:
        max_wrap = base_wrap

    target = max(min_wrap, int(width) - padding)
    if max_wrap > 0:
        target = min(max_wrap, target)

    try:
        current_wrap = int(float(widget.cget("wraplength") or 0))
    except Exception:
        current_wrap = 0

    if abs(current_wrap - target) <= 2:
        return

    try:
        widget.configure(wraplength=target)
    except Exception:
        pass

def _refresh_adaptive_wraps(self, root):
    if root is None:
        return

    visited: set[int] = set()

    def walk(widget):
        if widget is None:
            return

        widget_id = id(widget)
        if widget_id in visited:
            return
        visited.add(widget_id)

        if hasattr(widget, "_adaptive_wrap_base"):
            try:
                self._update_adaptive_wraplength(widget)
            except Exception:
                pass

        try:
            for child in widget.winfo_children():
                walk(child)
        except Exception:
            pass

    walk(root)

def _refresh_adaptive_wraps_for_container(self, container):
    if container is None:
        return

    try:
        container._adaptive_wrap_after_id = None
    except Exception:
        pass

    targets = list(getattr(container, "_adaptive_wrap_targets", []) or [])
    seen: set[int] = set()
    for widget in targets:
        if widget is None:
            continue
        widget_id = id(widget)
        if widget_id in seen:
            continue
        seen.add(widget_id)
        try:
            if bool(widget.winfo_exists()):
                self._update_adaptive_wraplength(widget)
        except Exception:
            pass

def _schedule_adaptive_wrap_refresh(self, container):
    if container is None:
        return

    pending = getattr(container, "_adaptive_wrap_after_id", None)
    if pending is not None:
        return

    try:
        container._adaptive_wrap_after_id = container.after_idle(
            lambda target=container: self._refresh_adaptive_wraps_for_container(target)
        )
    except Exception:
        try:
            container._adaptive_wrap_after_id = None
        except Exception:
            pass

def ensure_adaptive_wrap(self, widget, container=None, *, padding: int = 18, min_wrap: int = 80):
    if widget is None:
        return

    try:
        configured_wrap = int(float(widget.cget("wraplength") or 0))
    except Exception:
        configured_wrap = 0
    if configured_wrap <= 0:
        return

    if not hasattr(widget, "_adaptive_wrap_base"):
        try:
            widget._adaptive_wrap_base = int(configured_wrap)
        except Exception:
            return
    try:
        widget._adaptive_wrap_max = max(int(getattr(widget, "_adaptive_wrap_max", 0) or 0), int(configured_wrap))
    except Exception:
        widget._adaptive_wrap_max = int(configured_wrap)
    widget._adaptive_wrap_container = container or getattr(widget, "master", None)
    widget._adaptive_wrap_padding = int(padding)
    widget._adaptive_wrap_min = int(min_wrap)

    container_widget = getattr(widget, "_adaptive_wrap_container", None)
    if container_widget is not None:
        targets = list(getattr(container_widget, "_adaptive_wrap_targets", []) or [])
        if all(existing is not widget for existing in targets):
            targets.append(widget)
            try:
                container_widget._adaptive_wrap_targets = targets
            except Exception:
                pass
        if not bool(getattr(container_widget, "_adaptive_wrap_bound", False)):
            try:
                container_widget.bind(
                    "<Configure>",
                    lambda _event, target=container_widget: self._schedule_adaptive_wrap_refresh(target),
                    add="+",
                )
                container_widget._adaptive_wrap_bound = True
            except Exception:
                pass
        self._schedule_adaptive_wrap_refresh(container_widget)

    self._update_adaptive_wraplength(widget)

def style_classic_scale_widget(self, widget, background: str = None):
    if widget is None:
        return

    bg, track, thumb, thumb_hover, thumb_disabled = self._get_scale_colors(background)
    is_disabled = False
    try:
        is_disabled = str(widget.cget("state") or "").lower() == "disabled"
    except Exception:
        is_disabled = False

    thumb_color = thumb_disabled if is_disabled else thumb
    active_thumb = thumb_disabled if is_disabled else thumb_hover

    options = {
        "bg": bg,
        "troughcolor": track,
        "activebackground": active_thumb,
        "highlightbackground": bg,
        "highlightcolor": bg,
        "highlightthickness": 0,
        "bd": 0,
        "relief": tk.FLAT,
        "sliderrelief": tk.FLAT,
        "sliderlength": 14,
        "width": 4,
        "fg": thumb_color,
    }

    for option_name, option_value in options.items():
        try:
            widget.configure(**{option_name: option_value})
        except Exception:
            pass

def _get_panel_label_style(self, style_name: str | None) -> str | None:
    style_name = str(style_name or "").strip()
    if style_name.startswith("Panel"):
        return style_name

    mapping = {
        "": "Panel.TLabel",
        "TLabel": "Panel.TLabel",
        "Muted.TLabel": "PanelMuted.TLabel",
        "Info.TLabel": "PanelInfo.TLabel",
    }
    return mapping.get(style_name)

def _get_panel_control_style(self, class_name: str, style_name: str | None) -> str | None:
    current_style = str(style_name or "").strip()
    if current_style.startswith("Panel"):
        return current_style

    mappings = {
        "TCheckbutton": {
            "": "Panel.TCheckbutton",
            "TCheckbutton": "Panel.TCheckbutton",
        },
        "TRadiobutton": {
            "": "Panel.TRadiobutton",
            "TRadiobutton": "Panel.TRadiobutton",
        },
    }
    return mappings.get(str(class_name or "").strip(), {}).get(current_style)

def style_panel_surface(self, root, background: str = None):
    if root is None:
        return

    palette = getattr(self, "palette", {})
    bg = self._coerce_color_hex(
        background,
        fallback=palette["panel"],
    )
    visited: set[int] = set()

    def walk(widget):
        if widget is None:
            return

        widget_id = id(widget)
        if widget_id in visited:
            return
        visited.add(widget_id)

        try:
            class_name = str(widget.winfo_class())
        except Exception:
            class_name = ""

        if widget is root:
            local_bg = bg
        else:
            local_bg = self._resolve_widget_background(getattr(widget, "master", None), fallback=bg)

        if isinstance(widget, WebSlimScrollbar):
            self.style_web_scrollbar(widget, track_color=local_bg)
        elif class_name == "TScale":
            self.style_ttk_scale_widget(widget, background=local_bg)
        elif class_name == "Scale":
            self.style_classic_scale_widget(widget, background=local_bg)
        elif class_name == "TFrame":
            try:
                current_style = str(widget.cget("style") or "").strip()
            except Exception:
                current_style = ""
            self.style_ttk_frame_widget(widget, background=local_bg, base_style=(current_style or "TFrame"))
        elif class_name == "TPanedwindow":
            try:
                current_style = str(widget.cget("style") or "").strip()
            except Exception:
                current_style = ""
            self.style_ttk_panedwindow_widget(widget, background=local_bg, base_style=(current_style or "TPanedwindow"))
        elif class_name == "TLabelframe":
            try:
                current_style = str(widget.cget("style") or "").strip()
            except Exception:
                current_style = ""
            self.style_ttk_labelframe_widget(widget, background=local_bg, base_style=(current_style or "TLabelframe"))
        elif class_name == "TLabel":
            try:
                current_style = str(widget.cget("style") or "").strip()
            except Exception:
                current_style = ""
            target_style = self._get_panel_label_style(current_style)
            if target_style and target_style != current_style:
                try:
                    widget.configure(style=target_style)
                except Exception:
                    pass
        elif class_name in {"TCheckbutton", "TRadiobutton"}:
            try:
                current_style = str(widget.cget("style") or "").strip()
            except Exception:
                current_style = ""
            target_style = self._get_panel_control_style(class_name, current_style)
            if target_style and target_style != current_style:
                try:
                    widget.configure(style=target_style)
                except Exception:
                    pass
        elif class_name == "Frame":
            try:
                widget.configure(bg=local_bg)
            except Exception:
                pass
        elif class_name == "Label":
            try:
                widget.configure(bg=local_bg)
            except Exception:
                pass
        elif class_name == "Canvas":
            try:
                widget.configure(bg=local_bg)
            except Exception:
                pass
        elif class_name == "Labelframe":
            try:
                widget.configure(
                    bg=local_bg,
                    fg=palette["fg"],
                    highlightbackground=palette["panel_border"],
                    highlightcolor=palette["panel_border"],
                )
            except Exception:
                pass

        try:
            for child in widget.winfo_children():
                walk(child)
        except Exception:
            pass

    walk(root)

def _center_dialog_window(self, dialog, parent=None, width: int | None = None, height: int | None = None, margin: int = 24):
    try:
        dialog.update_idletasks()

        screen_width = int(dialog.winfo_screenwidth())
        screen_height = int(dialog.winfo_screenheight())
        final_width = int(width or dialog.winfo_width() or dialog.winfo_reqwidth())
        final_height = int(height or dialog.winfo_height() or dialog.winfo_reqheight())

        anchor = parent or self.root
        x = (screen_width - final_width) // 2
        y = (screen_height - final_height) // 2
        try:
            anchor.update_idletasks()
            if anchor.winfo_ismapped():
                anchor_x = int(anchor.winfo_rootx())
                anchor_y = int(anchor.winfo_rooty())
                anchor_w = int(anchor.winfo_width())
                anchor_h = int(anchor.winfo_height())
                if anchor_w > 80 and anchor_h > 80:
                    x = anchor_x + ((anchor_w - final_width) // 2)
                    y = anchor_y + ((anchor_h - final_height) // 2)
        except Exception:
            pass

        x = max(margin, min(x, screen_width - final_width - margin))
        y = max(margin, min(y, screen_height - final_height - margin))
        dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")
    except Exception:
        pass

def style_dialog_window(self, dialog, title: str = "", geometry: str = None, parent=None):
    palette = self.palette
    dialog.configure(bg=palette["bg"])
    if title:
        dialog.title(title)
    width = None
    height = None
    if geometry:
        dialog.geometry(geometry)
        try:
            size_part = str(geometry).split("+", 1)[0]
            if "x" in size_part:
                width_text, height_text = size_part.lower().split("x", 1)
                width = int(width_text)
                height = int(height_text)
        except Exception:
            width = None
            height = None
    dialog.transient(parent or self.root)
    dialog.resizable(False, False)
    self._center_dialog_window(dialog, parent=parent, width=width, height=height)
    try:
        dialog.grab_set()
    except Exception:
        pass
    return dialog

def _fit_dialog_to_content(
    self,
    dialog,
    parent=None,
    min_width: int = 520,
    min_height: int = 220,
    margin: int = 24,
):
    try:
        dialog.update_idletasks()

        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        final_width = max(min_width, dialog.winfo_reqwidth())
        final_height = max(min_height, dialog.winfo_reqheight())

        max_width = max(min_width, screen_width - (margin * 2))
        max_height = max(min_height, screen_height - (margin * 2))
        final_width = min(final_width, max_width)
        final_height = min(final_height, max_height)

        anchor = parent or self.root
        x = (screen_width - final_width) // 2
        y = (screen_height - final_height) // 2

        try:
            anchor.update_idletasks()
            if anchor.winfo_ismapped():
                x = anchor.winfo_rootx() + max(0, (anchor.winfo_width() - final_width) // 2)
                y = anchor.winfo_rooty() + max(0, (anchor.winfo_height() - final_height) // 2)
        except Exception:
            pass

        x = max(margin, min(x, screen_width - final_width - margin))
        y = max(margin, min(y, screen_height - final_height - margin))

        dialog.minsize(min_width, min_height)
        dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")
    except Exception:
        pass

def _build_themed_dialog_surface(self, dialog, *, tone: str = "info"):
    palette = self.palette
    tone_key = str(tone or "info").strip().lower()
    tone_surface = {
        "info": palette["surface_info"],
        "success": palette["surface_success"],
        "warning": palette["surface_warning"],
        "error": blend_hex_colors(
            palette["surface_warning"],
            palette["error"],
            0.08,
        ),
    }.get(tone_key, palette["panel"])
    border = {
        "info": palette["accent"],
        "success": palette["success"],
        "warning": palette["warning"],
        "error": palette["error"],
    }.get(tone_key, palette["panel_border"])
    dialog.configure(bg=tone_surface)

    body = tk.Frame(
        dialog,
        bg=tone_surface,
        bd=0,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(border, tone_surface, 0.32),
        highlightcolor=blend_hex_colors(border, tone_surface, 0.32),
    )
    body.pack(fill=tk.BOTH, expand=True)
    return body

def themed_message_dialog(
    self,
    title: str,
    message: str,
    parent=None,
    buttons=None,
    default_button: str = None,
    tone: str = "info",
    wraplength: int = 440,
):
    buttons = list(buttons or ["OK"])
    default_button = default_button or buttons[0]
    palette = self.palette

    dialog = tk.Toplevel(parent or self.root)
    self.style_dialog_window(dialog, title=title, geometry="520x240", parent=parent)
    body = self._build_themed_dialog_surface(dialog, tone=tone)

    tk.Label(
        body,
        text=message,
        bg=str(body.cget("bg") or palette["panel"]),
        fg=palette["fg"],
        font=("Segoe UI", 10),
        wraplength=wraplength,
        justify=tk.LEFT,
        anchor="w",
    ).pack(fill=tk.BOTH, expand=True, padx=18, pady=(18, 16))

    result = {"value": None}

    btn_row = tk.Frame(body, bg=str(body.cget("bg") or palette["panel"]))
    btn_row.pack(fill=tk.X, padx=18, pady=(0, 18))

    def close_with(value):
        result["value"] = value
        dialog.destroy()

    for label in reversed(buttons):
        style_name = "Accent.TButton" if label == default_button else "TButton"
        ttk.Button(
            btn_row,
            text=label,
            command=lambda value=label: close_with(value),
            style=style_name,
        ).pack(side=tk.RIGHT, padx=(8, 0))

    self._fit_dialog_to_content(dialog, parent=parent, min_width=520, min_height=240)
    dialog.bind("<Escape>", lambda _e: close_with(None))
    dialog.bind("<Return>", lambda _e: close_with(default_button))
    dialog.wait_window()
    return result["value"]

def themed_confirm(
    self,
    title: str,
    message: str,
    parent=None,
    confirm_label: str = "OK",
    cancel_label: str = "Anuluj",
    tone: str = "warning"
) -> bool:
    result = self.themed_message_dialog(
        title=title,
        message=message,
        parent=parent,
        buttons=[cancel_label, confirm_label],
        default_button=confirm_label,
        tone=tone,
    )
    return result == confirm_label

def themed_info(self, title: str, message: str, parent=None, tone: str = "info"):
    self.themed_message_dialog(
        title=title,
        message=message,
        parent=parent,
        buttons=["OK"],
        default_button="OK",
        tone=tone,
    )

def themed_error(self, title: str, message: str, parent=None):
    self.themed_info(title=title, message=message, parent=parent, tone="error")

def _install_themed_dialog_hooks(self):
    def _extract_parent(kwargs):
        return kwargs.get("parent", self.root)

    def showinfo(title, message, **kwargs):
        self.themed_info(title, message, parent=_extract_parent(kwargs), tone="info")
        return "ok"

    def showwarning(title, message, **kwargs):
        self.themed_info(title, message, parent=_extract_parent(kwargs), tone="warning")
        return "ok"

    def showerror(title, message, **kwargs):
        self.themed_error(title, message, parent=_extract_parent(kwargs))
        return "ok"

    def askyesno(title, message, **kwargs):
        return self.themed_confirm(
            title,
            message,
            parent=_extract_parent(kwargs),
            confirm_label="Tak",
            cancel_label="Nie",
            tone="warning"
        )

    def askokcancel(title, message, **kwargs):
        return self.themed_confirm(
            title,
            message,
            parent=_extract_parent(kwargs),
            confirm_label="OK",
            cancel_label="Anuluj",
            tone="warning"
        )

    def askstring(title, prompt, **kwargs):
        return self.themed_ask_string(
            title,
            prompt,
            parent=_extract_parent(kwargs),
            action_label="OK",
            initial_value=kwargs.get("initialvalue", "") or ""
        )

    messagebox.showinfo = showinfo
    messagebox.showwarning = showwarning
    messagebox.showerror = showerror
    messagebox.askyesno = askyesno
    messagebox.askokcancel = askokcancel
    simpledialog.askstring = askstring

def themed_ask_string(
    self,
    title: str,
    prompt: str,
    parent=None,
    action_label: str = "OK",
    initial_value: str = "",
):
    palette = self.palette
    dialog = tk.Toplevel(parent or self.root)
    self.style_dialog_window(dialog, title=title, geometry="520x240", parent=parent)
    body = self._build_themed_dialog_surface(dialog, tone="info")

    tk.Label(
        body,
        text=prompt,
        bg=str(body.cget("bg") or palette["panel"]),
        fg=palette["fg"],
        font=("Segoe UI", 10),
        wraplength=440,
        justify=tk.LEFT,
        anchor="w",
    ).pack(fill=tk.X, padx=18, pady=(18, 8))

    value_var = tk.StringVar(value=initial_value)
    entry = ttk.Entry(body, textvariable=value_var)
    entry.pack(fill=tk.X, padx=18, pady=(0, 16))
    entry.focus_set()
    entry.selection_range(0, tk.END)

    result = {"value": None}

    def accept():
        result["value"] = value_var.get().strip()
        dialog.destroy()

    def cancel():
        dialog.destroy()

    btn_row = tk.Frame(body, bg=str(body.cget("bg") or palette["panel"]))
    btn_row.pack(fill=tk.X, padx=18, pady=(0, 18))
    ttk.Button(btn_row, text=action_label, command=accept, style="Accent.TButton").pack(side=tk.RIGHT)
    ttk.Button(btn_row, text="Anuluj", command=cancel).pack(side=tk.RIGHT, padx=(0, 8))

    self._fit_dialog_to_content(dialog, parent=parent, min_width=520, min_height=240)
    dialog.bind("<Return>", lambda _e: accept())
    dialog.bind("<Escape>", lambda _e: cancel())
    dialog.wait_window()
    return result["value"]

def show_about_dialog(self):
    details = (
        f"{CONFIG.APP_NAME}\n"
        f"Wersja: {CONFIG.VERSION}\n"
        f"Autor: {APP_AUTHOR}\n"
        f"Styl: {self.current_theme_name}"
    )
    self.themed_info("O aplikacji", details, parent=self.root, tone="info")

# Delegate from app_style_setup is bound after class creation.



def bind_app_theme_runtime(cls):
    """Bind extracted theme/runtime methods back onto ``AutoAnnotationApp``."""
    for name in (
        '_is_guided_style',
        '_remember_guided_base_style',
        '_get_guided_styles_for_button',
        'set_button_emphasis',
        'style_guidance_frame',
        'set_frame_emphasis',
        'set_theme',
        '_restyle_shell',
        '_apply_theme_to_tabs',
        '_apply_theme_to_widget_tree',
        '_emit_theme_repaint_pulse',
        '_finalize_theme_refresh',
        '_schedule_theme_refresh_finalize',
        'style_native_scrollbar',
        'is_dark_theme',
        '_get_scrollbar_colors',
        'style_text_widget',
        'style_listbox_widget',
        'style_web_scrollbar',
        'style_canvas_widget',
        '_get_scale_colors',
        'get_list_selection_colors',
        '_coerce_color_hex',
        '_resolve_widget_background',
        '_ensure_horizontal_scale_style_assets',
        'style_ttk_scale_widget',
        'style_ttk_frame_widget',
        'style_ttk_panedwindow_widget',
        'style_ttk_labelframe_widget',
        '_update_adaptive_wraplength',
        '_refresh_adaptive_wraps',
        '_refresh_adaptive_wraps_for_container',
        '_schedule_adaptive_wrap_refresh',
        'ensure_adaptive_wrap',
        'style_classic_scale_widget',
        '_get_panel_label_style',
        '_get_panel_control_style',
        'style_panel_surface',
        '_center_dialog_window',
        'style_dialog_window',
        '_fit_dialog_to_content',
        '_build_themed_dialog_surface',
        'themed_message_dialog',
        'themed_confirm',
        'themed_info',
        'themed_error',
        '_install_themed_dialog_hooks',
        'themed_ask_string',
        'show_about_dialog'
    ):
        setattr(cls, name, globals()[name])
    for name in (
        '_paint_scale_track_image',
        '_paint_scale_thumb_image',
        '_style_token'
    ):
        setattr(cls, name, staticmethod(globals()[name]))
    return cls
