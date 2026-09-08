#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theme application helpers for the Z3 character annotation tab."""

import tkinter as tk
from tkinter import ttk

from .web_slim_scrollbar import blend_hex_colors


def panel_style_name(tone: str = "neutral", emphasis: bool = False) -> str:
    tone_key = str(tone or "").strip().lower()
    if emphasis:
        return {
            "neutral": "PanelStatusNeutral.TLabel",
            "info": "PanelStatusInfo.TLabel",
            "success": "PanelStatusSuccess.TLabel",
            "warning": "PanelStatusWarning.TLabel",
            "error": "PanelStatusError.TLabel",
            "muted": "PanelStatusNeutral.TLabel",
        }.get(tone_key, "PanelStatusNeutral.TLabel")

    return {
        "neutral": "PanelMuted.TLabel",
        "muted": "PanelMuted.TLabel",
        "info": "PanelInfo.TLabel",
        "success": "PanelSuccess.TLabel",
        "warning": "PanelStatusWarning.TLabel",
        "error": "PanelError.TLabel",
    }.get(tone_key, "PanelMuted.TLabel")


def get_readable_text_color(background: str, preferred: str = None) -> str:
    def _normalize_hex(color_value):
        raw = str(color_value or "").strip()
        if not raw.startswith("#"):
            return None
        hex_part = raw[1:]
        if len(hex_part) == 3:
            hex_part = "".join(ch * 2 for ch in hex_part)
        if len(hex_part) != 6:
            return None
        try:
            return tuple(int(hex_part[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return None

    def _relative_luminance(rgb):
        channels = []
        for channel in rgb:
            normalized = channel / 255.0
            if normalized <= 0.03928:
                channels.append(normalized / 12.92)
            else:
                channels.append(((normalized + 0.055) / 1.055) ** 2.4)
        return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2])

    def _contrast_ratio(color_a, color_b):
        lum_a = _relative_luminance(color_a)
        lum_b = _relative_luminance(color_b)
        lighter = max(lum_a, lum_b)
        darker = min(lum_a, lum_b)
        return (lighter + 0.05) / (darker + 0.05)

    bg_rgb = _normalize_hex(background)
    if bg_rgb is None:
        return preferred or "#ffffff"

    preferred_rgb = _normalize_hex(preferred)
    if preferred_rgb is not None and _contrast_ratio(bg_rgb, preferred_rgb) >= 4.5:
        return preferred

    dark_text = "#111111"
    light_text = "#ffffff"
    dark_rgb = _normalize_hex(dark_text)
    light_rgb = _normalize_hex(light_text)
    if dark_rgb is None or light_rgb is None:
        return preferred or light_text

    dark_ratio = _contrast_ratio(bg_rgb, dark_rgb)
    light_ratio = _contrast_ratio(bg_rgb, light_rgb)
    return dark_text if dark_ratio >= light_ratio else light_text


def set_themed_label_state(host, widget, text: str | None = None, tone: str = "neutral", emphasis: bool = False):
    if widget is None:
        return

    config_kwargs = {"style": panel_style_name(tone=tone, emphasis=emphasis)}
    if text is not None:
        config_kwargs["text"] = text

    widget.config(**config_kwargs)


def mark_inline_status_contrast_boost(host, widget) -> None:
    if widget is None or not isinstance(widget, tk.Label):
        return

    try:
        widget._inline_status_contrast_boost = True
    except Exception:
        return

    try:
        set_inline_status_label_state(
            host,
            widget,
            text=widget.cget("text"),
            tone=getattr(widget, "_inline_status_tone", "muted"),
            emphasis=bool(getattr(widget, "_inline_status_emphasis", False)),
        )
    except Exception:
        pass


def set_inline_status_label_state(
    host,
    widget,
    text: str | None = None,
    tone: str = "neutral",
    emphasis: bool = False,
) -> bool:
    if widget is None or not isinstance(widget, tk.Label):
        return False

    palette = getattr(host.app, "palette", {})
    bg = palette.get("bg", "#1f1f1f")
    bg_override = str(getattr(widget, "_inline_status_bg", "") or "").strip()
    if bg_override:
        bg = bg_override
    try:
        parent = widget.nametowidget(widget.winfo_parent())
    except Exception:
        parent = None

    if not bg_override:
        for candidate in (parent, widget):
            if candidate is None:
                continue
            try:
                bg_candidate = candidate.cget("bg")
                if bg_candidate:
                    bg = bg_candidate
                    break
            except Exception:
                pass
            try:
                bg_candidate = candidate.cget("background")
                if bg_candidate:
                    bg = bg_candidate
                    break
            except Exception:
                pass
            try:
                style_name = str(candidate.cget("style") or "").strip()
                if style_name:
                    bg_candidate = host.app.style.lookup(style_name, "background")
                    if bg_candidate:
                        bg = bg_candidate
                        break
            except Exception:
                pass
            try:
                style_name = candidate.winfo_class()
                bg_candidate = ttk.Style().lookup(style_name, "background")
                if bg_candidate:
                    bg = bg_candidate
                    break
            except Exception:
                pass

    tone_key = str(tone or "").strip().lower()
    fg = {
        "default": palette.get("fg", "#f3f3f3"),
        "neutral": palette.get("muted", "#9a9a9a"),
        "muted": palette.get("muted", "#9a9a9a"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f39c12"),
        "error": palette.get("error", "#e74c3c"),
    }.get(tone_key, palette.get("muted", "#9a9a9a"))

    if tone_key in {"neutral", "muted"} and bool(getattr(widget, "_inline_status_contrast_boost", False)):
        is_light_background = get_readable_text_color(bg, preferred="#111111") == "#111111"
        preferred_fg = blend_hex_colors(
            palette.get("fg", "#f3f3f3"),
            bg,
            0.10 if is_light_background else 0.28,
        )
        fg = get_readable_text_color(bg, preferred=preferred_fg)

    custom_font = getattr(widget, "_inline_status_font", None)
    config_kwargs = {
        "bg": bg,
        "fg": fg,
        "font": custom_font or (("Segoe UI", 10, "bold") if emphasis else ("Segoe UI", 9)),
    }
    if text is not None:
        config_kwargs["text"] = text

    widget._inline_status_tone = tone_key
    widget._inline_status_emphasis = bool(emphasis)
    widget.config(**config_kwargs)
    return True


def get_inline_status_widget_snapshot(widget, *, fallback_text: str = "", fallback_tone: str = "muted") -> tuple[str, str]:
    if widget is None:
        return str(fallback_text or ""), str(fallback_tone or "muted")

    try:
        current_text = str(widget.cget("text") or "").strip()
    except Exception:
        current_text = ""

    current_tone = str(getattr(widget, "_inline_status_tone", "") or "").strip().lower()
    if not current_tone:
        current_tone = str(fallback_tone or "muted")

    return (current_text if current_text else str(fallback_text or "")), current_tone


def draw_selection_indicator(host, canvas, kind: str, selected: bool, background: str) -> None:
    if canvas is None:
        return

    palette = getattr(host.app, "palette", {})
    outline = palette.get("border", "#5a5a5a")
    success = palette.get("success", "#4ec9b0")

    try:
        canvas.configure(bg=background)
        canvas.delete("all")
    except Exception:
        return

    if kind == "radio":
        canvas.create_oval(2, 2, 14, 14, outline=outline, width=2, fill=background)
        if selected:
            canvas.create_oval(5, 5, 11, 11, outline=success, width=1, fill=success)
        else:
            canvas.create_oval(5, 5, 11, 11, outline=background, width=1, fill=background)
        return

    if kind == "summary":
        canvas.create_line(3, 5, 13, 5, fill=success, width=2, capstyle=tk.ROUND)
        canvas.create_line(3, 11, 13, 11, fill=success, width=2, capstyle=tk.ROUND)
        return

    canvas.create_rectangle(2, 2, 14, 14, outline=(success if selected else outline), width=2, fill=background)
    if selected:
        canvas.create_line(
            4, 8, 7, 11, 12, 5,
            fill=success,
            width=2,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )


def set_selection_row_hover(host, row_info: dict, hovered: bool) -> None:
    if not isinstance(row_info, dict):
        return
    if bool(getattr(host, "_selection_hover_suppressed", False)):
        return
    row_info["hovered"] = bool(hovered)
    refresh_selection_row(host, row_info)


def refresh_selection_row(host, row_info: dict) -> None:
    if not isinstance(row_info, dict):
        return

    frame = row_info.get("frame")
    label = row_info.get("label")
    indicator = row_info.get("indicator")
    if frame is None or label is None or indicator is None:
        return

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted_dim", palette.get("muted", "#a0a0a0"))
    selected = bool(row_info.get("selected_getter", lambda: False)())
    enabled = bool(row_info.get("enabled", True))
    hovered = bool(row_info.get("hovered", False))
    try:
        parent_bg = host.app._resolve_widget_background(getattr(frame, "master", None), fallback=panel_bg)
    except Exception:
        parent_bg = panel_bg
    base_bg = row_info.get("base_bg") or parent_bg
    hover_bg = row_info.get("hover_bg")
    if not hover_bg:
        hover_bg = base_bg
    row_bg = hover_bg if hovered else base_bg
    row_fg = (row_info.get("fg") or fg) if enabled else (row_info.get("disabled_fg") or muted_fg)
    border_color = row_info.get("border_color") or row_bg

    try:
        frame.configure(bg=row_bg, highlightbackground=border_color, highlightcolor=border_color)
    except Exception:
        pass
    try:
        label.configure(bg=row_bg, fg=row_fg)
    except Exception:
        pass
    for widget in list(row_info.get("extra_widgets", []) or []):
        if widget is None:
            continue
        try:
            widget.configure(bg=row_bg, fg=row_fg)
        except Exception:
            pass
    badge = row_info.get("badge")
    if badge is not None:
        badge_bg = row_info.get("badge_bg") or row_bg
        badge_fg = row_info.get("badge_fg") or row_fg
        try:
            badge.configure(bg=badge_bg, fg=badge_fg)
        except Exception:
            pass

    draw_selection_indicator(host, indicator, row_info.get("kind", "radio"), selected, row_bg)


def apply_preview_mode_radio_style(host) -> None:
    for row_info in getattr(host, "preview_box_mode_rows", []):
        refresh_selection_row(host, row_info)


def redraw_preview_mode_eye_icon(host) -> None:
    eye_canvas = getattr(host, "preview_mode_eye_btn", None)
    if eye_canvas is None:
        return

    palette = getattr(host.app, "palette", {})
    panel_alt = palette.get("panel_alt", palette.get("panel", "#252526"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    gold = palette.get("warning", "#d4af37")
    active_bg = blend_hex_colors(gold, panel_alt, 0.82)
    btn_bg = active_bg if bool(getattr(host, "_preview_mode_overlay_expanded", False)) else panel_alt
    eye_outline = blend_hex_colors(gold, border, 0.35)
    sclera_fill = "#f8fbff"
    pupil_fill = "#111111"

    try:
        eye_canvas.configure(
            width=34,
            height=24,
            bg=btn_bg,
            highlightbackground=blend_hex_colors(gold, border, 0.55),
            highlightcolor=blend_hex_colors(gold, border, 0.55),
            bd=0,
            relief=tk.FLAT,
        )
        eye_canvas.delete("all")
        eye_canvas.create_polygon(
            4, 12,
            9, 6,
            17, 4,
            25, 6,
            30, 12,
            25, 18,
            17, 20,
            9, 18,
            smooth=True,
            splinesteps=24,
            fill=sclera_fill,
            outline=eye_outline,
            width=1.5,
        )
        eye_canvas.create_oval(
            13, 8,
            21, 16,
            fill=pupil_fill,
            outline=pupil_fill,
            width=1,
        )
        eye_canvas.create_oval(
            16, 10,
            18, 12,
            fill="#ffffff",
            outline="",
        )
    except Exception:
        pass


def refresh_preview_mode_overlay_visibility(host) -> None:
    overlay = getattr(host, "preview_mode_overlay", None)
    external_toggle_btn = getattr(host, "preview_box_mode_toggle_btn", None)
    current_lbl = getattr(host, "preview_mode_overlay_current_lbl", None)
    if overlay is None:
        return

    current_mode = host.PREVIEW_BOX_MODE_LABELS.get(
        host._get_preview_box_mode_key(),
        host.PREVIEW_BOX_MODE_LABELS.get("AUTO", "Auto"),
    ) if hasattr(host, "PREVIEW_BOX_MODE_LABELS") else None
    if current_mode is None:
        current_mode = globals().get("PREVIEW_BOX_MODE_LABELS", {}).get("AUTO", "Auto")

    if current_lbl is not None:
        try:
            current_lbl.configure(text=f"Aktualnie pokazujesz: {current_mode}")
        except Exception:
            pass
    if external_toggle_btn is not None:
        try:
            external_toggle_btn.configure(text=f"Źródło końcowych ramek: {current_mode}")
        except Exception:
            pass

    try:
        overlay.place_forget()
    except Exception:
        pass


def apply_preview_mode_overlay_style(host, preview_box_mode_labels: dict) -> None:
    palette = getattr(host.app, "palette", {})
    overlay = getattr(host, "preview_mode_overlay", None)
    content = getattr(host, "preview_mode_overlay_content", None)
    title = getattr(host, "preview_mode_overlay_title_lbl", None)
    current = getattr(host, "preview_mode_overlay_current_lbl", None)
    body = getattr(host, "preview_mode_overlay_body", None)
    toggle_btn = getattr(host, "preview_mode_eye_btn", None) or getattr(host, "preview_mode_overlay_toggle_btn", None)
    panel_alt = palette.get("panel_alt", palette.get("panel", "#252526"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#a0a0a0")
    gold = palette.get("warning", "#d4af37")
    active_bg = blend_hex_colors(gold, panel_alt, 0.82)
    gold_border = blend_hex_colors(gold, border, 0.55)

    if overlay is not None:
        try:
            overlay.configure(
                bg=panel_alt,
                highlightbackground=border,
                highlightcolor=border,
            )
        except Exception:
            pass

    if content is not None:
        try:
            content.configure(bg=panel_alt)
        except Exception:
            pass

    if title is not None:
        try:
            title.configure(bg=panel_alt, fg=fg)
        except Exception:
            pass

    if current is not None:
        try:
            current.configure(bg=panel_alt, fg=muted)
        except Exception:
            pass

    if body is not None:
        try:
            body.configure(bg=panel_alt)
        except Exception:
            pass

    if toggle_btn is not None:
        try:
            toggle_btn.configure(
                bg=(active_bg if bool(getattr(host, "_preview_mode_overlay_expanded", False)) else panel_alt),
                highlightbackground=gold_border,
                highlightcolor=gold_border,
                relief=tk.FLAT,
                bd=0,
            )
        except Exception:
            pass
        redraw_preview_mode_eye_icon(host)

    refresh_preview_mode_overlay_visibility_with_labels(host, preview_box_mode_labels)


def refresh_preview_mode_overlay_visibility_with_labels(host, preview_box_mode_labels: dict) -> None:
    overlay = getattr(host, "preview_mode_overlay", None)
    external_toggle_btn = getattr(host, "preview_box_mode_toggle_btn", None)
    current_lbl = getattr(host, "preview_mode_overlay_current_lbl", None)
    if overlay is None:
        return

    current_mode = preview_box_mode_labels.get(
        host._get_preview_box_mode_key(),
        preview_box_mode_labels.get("AUTO", "Auto"),
    )

    if current_lbl is not None:
        try:
            current_lbl.configure(text=f"Aktualnie pokazujesz: {current_mode}")
        except Exception:
            pass
    if external_toggle_btn is not None:
        try:
            external_toggle_btn.configure(text=f"Źródło końcowych ramek: {current_mode}")
        except Exception:
            pass

    try:
        overlay.place_forget()
    except Exception:
        pass


def apply_yolo_option_check_style(host) -> None:
    for row_info in getattr(host, "yolo_option_rows", []):
        refresh_selection_row(host, row_info)


def apply_gold_export_filter_check_style(host) -> None:
    for row_info in getattr(host, "gold_export_filter_rows", []):
        refresh_selection_row(host, row_info)


def apply_gold_export_source_check_style(host) -> None:
    for row_info in getattr(host, "gold_export_source_rows", []):
        refresh_selection_row(host, row_info)


def apply_gold_export_split_check_style(host) -> None:
    row_info = getattr(host, "gold_export_split_row_info", None)
    if isinstance(row_info, dict):
        refresh_selection_row(host, row_info)


def apply_plates_legend_style(host) -> None:
    palette = getattr(host.app, "palette", {})
    bg = palette.get("panel", palette.get("bg", "#252526"))
    shell_bg = blend_hex_colors(palette.get("panel_alt", bg), bg, 0.25)
    fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted", "#a0a0a0")
    success_fg = palette.get("success", "#2ecc71")
    warning_fg = palette.get("warning", "#f4c27a")
    info_fg = palette.get("info", palette.get("accent", "#4aa3ff"))
    hybrid_fg = blend_hex_colors(info_fg, success_fg, 0.5)
    shell_border = blend_hex_colors(palette.get("border", "#3c3c3c"), shell_bg, 0.35)

    frame = getattr(host, "plates_legend_frame", None)
    if frame is not None:
        try:
            frame.configure(
                bg=shell_bg,
                highlightthickness=1,
                highlightbackground=shell_border,
                highlightcolor=shell_border,
                padx=3,
                pady=5,
            )
        except Exception:
            pass
    label_configs = (
        ("plates_legend_scope_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_total_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_perfect_header_lbl", {"bg": shell_bg, "fg": success_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_manual_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_yolo_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_ocr_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_hybrid_header_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7), "anchor": "w"}),
        ("plates_legend_tab_title_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_total_lbl", {"bg": shell_bg, "fg": fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_perfect_lbl", {"bg": shell_bg, "fg": success_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_manual_lbl", {"bg": shell_bg, "fg": warning_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_yolo_lbl", {"bg": shell_bg, "fg": info_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_ocr_lbl", {"bg": shell_bg, "fg": success_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_tab_hybrid_lbl", {"bg": shell_bg, "fg": hybrid_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_title_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_total_lbl", {"bg": shell_bg, "fg": fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_perfect_lbl", {"bg": shell_bg, "fg": muted_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_manual_lbl", {"bg": shell_bg, "fg": warning_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_yolo_lbl", {"bg": shell_bg, "fg": info_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_ocr_lbl", {"bg": shell_bg, "fg": success_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
        ("plates_legend_box_hybrid_lbl", {"bg": shell_bg, "fg": hybrid_fg, "font": ("Segoe UI", 7, "bold"), "anchor": "w"}),
    )
    for name, config in label_configs:
        widget = getattr(host, name, None)
        if widget is None:
            continue
        try:
            widget.configure(**config)
        except Exception:
            pass


def apply_character_annotation_theme(host, progress_bar_cls) -> None:
    self = host
    palette = getattr(self.app, "palette", {})
    console_border = palette.get("console_border", palette.get("border", "#3c3c3c"))

    try:
        self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
    except Exception:
        pass

    self._apply_preview_sort_bar_style()
    self._apply_preview_mode_radio_style()
    self._apply_yolo_option_check_style()
    self._apply_gold_export_filter_check_style()
    self._apply_gold_export_split_check_style()
    self._apply_plates_legend_style()
    self._apply_preview_source_actions_style()
    self._apply_preview_info_stats_style()
    self._apply_preview_surface_style()
    self._apply_detection_advanced_section_style()
    self._refresh_preview_source_panel()
    self._refresh_extract_entry_cards()
    try:
        status_label = getattr(self, "source_binding_status_lbl", None)
        self._set_source_binding_status(
            status_label.cget("text") if status_label is not None else "",
            getattr(self, "_source_binding_status_tone", "success"),
        )
    except Exception:
        pass

    for widget_name in ("ext_log", "test_log_text", "export_console", "import_console"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            self.app.style_text_widget(widget, role="console")
        except Exception:
            pass

    for header_name in (
        "extract_entry_title_lbl",
        "extract_source_title_lbl",
        "extract_start_title_lbl",
        "detect_left_title_lbl",
        "preview_list_header_lbl",
        "preview_title_lbl",
        "detect_source_title_lbl",
        "preview_dir_hint_lbl",
        "preview_status_title_lbl",
        "detect_settings_title_lbl",
        "plates_list_title_lbl",
        "dataset_title_lbl",
        "review_title_lbl",
        "dataset_export_title_lbl",
        "import_section_title_lbl",
        "dataset_status_title_lbl",
    ):
        header = getattr(self, header_name, None)
        if header is None:
            continue
        try:
            header.apply_theme()
        except Exception:
            pass

    try:
        if hasattr(self, "ext_log_scrollbar"):
            self.app.style_web_scrollbar(
                self.ext_log_scrollbar,
                track_color=palette.get("console_bg", "#252526"),
            )
    except Exception:
        pass

    try:
        self.app.style_listbox_widget(self.plates_listbox, bordercolor=console_border)
        for idx, pid in enumerate(getattr(self, "_listbox_pid_by_index", [])):
            status = str(self.preview_metadata.get(pid, {}).get("status", "unknown")).strip().lower()
            self._apply_plate_listbox_row_style(idx, status)
    except Exception:
        pass

    try:
        self.app.style_canvas_widget(
            self.preview_canvas,
            background=palette.get("panel", "#1e1e1e"),
            bordercolor=console_border,
        )
    except Exception:
        pass

    try:
        if hasattr(self, "extract_left_canvas"):
            self.extract_left_canvas.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=console_border,
                highlightcolor=console_border,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "extract_left_scrollbar"):
            self.app.style_web_scrollbar(
                self.extract_left_scrollbar,
                track_color=palette.get("panel", "#252526"),
            )
        if hasattr(self, "extract_left_scroll_host"):
            self.extract_left_scroll_host.configure(style="Panel.TFrame")
        if hasattr(self, "extract_left_content"):
            self.extract_left_content.configure(style="Panel.TFrame")
    except Exception:
        pass

    try:
        export_shell_border = blend_hex_colors(
            palette.get("accent", "#4f8de3"),
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            0.62,
        )
        export_shell_fill = blend_hex_colors(
            palette.get("surface_info", palette.get("panel", "#252526")),
            palette.get("panel", "#252526"),
            0.80,
        )
        self.cvat_export_canvas.configure(
            bg=export_shell_fill,
            highlightthickness=0,
            bd=0,
        )
        if hasattr(self, "pz3_export_shell"):
            self.pz3_export_shell.configure(bg=export_shell_border)
        if hasattr(self, "pz3_export_shell_inner"):
            self.pz3_export_shell_inner.configure(bg=export_shell_fill)
            self.app.style_panel_surface(
                self.pz3_export_shell_inner,
                background=export_shell_fill,
            )
        if hasattr(self, "pz3_export_host"):
            self.pz3_export_host.configure(style="Panel.TFrame")
        if hasattr(self, "pz3_export_nav"):
            self.pz3_export_nav.configure(bg=export_shell_fill)
        if hasattr(self, "cvat_export_scrollbar"):
            self.app.style_web_scrollbar(
                self.cvat_export_scrollbar,
                track_color=export_shell_fill,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "preview_vertical_split"):
            self.preview_vertical_split.configure(
                background=palette.get("panel", "#252526"),
                sashwidth=6,
                sashrelief=tk.FLAT,
                sashcursor="sb_h_double_arrow",
                showhandle=False,
            )
    except Exception:
        pass

    for frame_name in ("detect_content_frame", "detect_left_panel", "detect_right_panel"):
        frame = getattr(self, frame_name, None)
        if frame is None:
            continue
        try:
            self.app.style_ttk_frame_widget(frame, background=palette.get("panel", "#252526"))
        except Exception:
            pass

    for pane_name in ("detect_split",):
        pane = getattr(self, pane_name, None)
        if pane is None:
            continue
        try:
            self.app.style_ttk_panedwindow_widget(pane, background=palette.get("panel", "#252526"))
        except Exception:
            pass

    try:
        if hasattr(self, "test_progress") and isinstance(self.test_progress, progress_bar_cls):
            self.test_progress.configure(
                bg=palette.get("panel", "#252526"),
                trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
                fill_color=palette.get("success", "#2ecc71"),
                thickness=2,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "ext_progress") and isinstance(self.ext_progress, progress_bar_cls):
            self.ext_progress.configure(
                bg=palette.get("panel", "#252526"),
                trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
                fill_color=palette.get("success", "#2ecc71"),
                thickness=2,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "detect_right_canvas"):
            self.detect_right_canvas.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=console_border,
                highlightcolor=console_border,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "detect_nav_divider"):
            self.detect_nav_divider.configure(bg=palette.get("panel_border", palette.get("border", "#3c3c3c")))
    except Exception:
        pass

    try:
        if hasattr(self, "extract_nav_divider"):
            self.extract_nav_divider.configure(bg=palette.get("panel_border", palette.get("border", "#3c3c3c")))
    except Exception:
        pass

    try:
        workflow_shell_border = blend_hex_colors(
            palette.get("accent", "#4f8de3"),
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            0.62,
        )
        workflow_shell_fill = blend_hex_colors(
            palette.get("surface_info", palette.get("panel", "#252526")),
            palette.get("panel", "#252526"),
            0.80,
        )
        if hasattr(self, "extract_workflow_shell"):
            self.extract_workflow_shell.configure(bg=workflow_shell_border)
        if hasattr(self, "extract_workflow_shell_inner"):
            self.extract_workflow_shell_inner.configure(bg=workflow_shell_fill)
            self.app.style_panel_surface(
                self.extract_workflow_shell_inner,
                background=workflow_shell_fill,
            )
        if hasattr(self, "extract_entry_section"):
            self.extract_entry_section.configure(bg=workflow_shell_fill)
        if hasattr(self, "extract_entry_cards_frame"):
            self.extract_entry_cards_frame.configure(bg=workflow_shell_fill)
    except Exception:
        pass

    try:
        if hasattr(self, "extract_start_section"):
            start_border = blend_hex_colors(
                palette.get("surface_info", palette.get("panel", "#252526")),
                palette.get("panel", "#252526"),
                0.80,
            )
            start_fill = start_border
            self.extract_start_section.configure(
                bg=start_border,
            )
        if hasattr(self, "extract_start_section_inner"):
            self.extract_start_section_inner.configure(bg=start_fill)
            self.app.style_panel_surface(
                self.extract_start_section_inner,
                background=start_fill,
            )
        self._refresh_extract_start_card_style()
    except Exception:
        pass

    for frame_name in (
        "btn_to_detect_frame",
        "btn_run_detection_frame",
        "btn_to_dataset_frame",
        "pz3_nav_actions_frame",
    ):
        frame = getattr(self, frame_name, None)
        if frame is None:
            continue
        try:
            frame.configure(bg=palette.get("panel", "#252526"))
        except Exception:
            pass

    try:
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
        shell = getattr(self, "detect_pipeline_summary_shell", None)
        if shell is not None:
            shell.configure(
                bg=pipeline_shell_border,
                highlightbackground=pipeline_shell_border,
                highlightcolor=pipeline_shell_border,
            )
        for frame_name in (
            "detect_pipeline_summary_inner",
            "yolo_model_row",
        ):
            frame = getattr(self, frame_name, None)
            if frame is not None:
                frame.configure(bg=pipeline_shell_fill)
        title = getattr(self, "detect_pipeline_summary_title_lbl", None)
        if title is not None:
            title.configure(
                bg=pipeline_shell_fill,
                fg=palette.get("fg", "#f3f3f3"),
            )
        run_label = getattr(self, "detect_run_model_info_lbl", None)
        if run_label is not None:
            run_label._inline_status_bg = pipeline_shell_fill
            self._set_inline_status_label_state(
                run_label,
                text=run_label.cget("text"),
                tone=getattr(run_label, "_inline_status_tone", "success"),
                emphasis=bool(getattr(run_label, "_inline_status_emphasis", False)),
            )
    except Exception:
        pass

    try:
        export_shell_fill = blend_hex_colors(
            palette.get("surface_info", palette.get("panel", "#252526")),
            palette.get("panel", "#252526"),
            0.80,
        )
        nav_actions = getattr(self, "pz3_nav_actions_frame", None)
        if nav_actions is not None:
            nav_actions.configure(bg=export_shell_fill)
        action_card = getattr(self, "step3_finish_action_card", None)
        if action_card is not None:
            action_card.refresh_theme(container_background=export_shell_fill)
    except Exception:
        pass

    for label_name, style_name in (
        ("cvat_option1_title_lbl", "PanelStatusError.TLabel"),
        ("cvat_option2_title_lbl", "PanelStatusSuccess.TLabel"),
        ("cvat_option1_desc_lbl", "PanelMuted.TLabel"),
        ("cvat_option2_desc_lbl", "PanelMuted.TLabel"),
        ("cvat_import_desc_lbl", "PanelMuted.TLabel"),
    ):
        label = getattr(self, label_name, None)
        if label is None or isinstance(label, tk.Label):
            continue
        try:
            label.configure(style=style_name)
        except Exception:
            pass

    inline_label_defaults = {
        "ext_status": ("neutral", True),
        "detect_run_model_info_lbl": ("success", False),
        "test_status_lbl": ("neutral", False),
        "test_progress_count_lbl": ("muted", True),
        "winner_name_lbl": ("neutral", True),
        "winner_acc_lbl": ("muted", False),
        "preview_info_lbl": ("neutral", False),
        "detect_left_intro_lbl": ("muted", False),
        "preview_list_intro_lbl": ("muted", False),
        "preview_intro_lbl": ("muted", False),
        "preview_shortcuts_lbl": ("muted", False),
        "preview_record_source_lbl": ("muted", False),
        "preview_source_status_lbl": ("warning", True),
        "preview_source_detail_lbl": ("muted", False),
        "preview_edit_status_lbl": ("muted", False),
        "preview_load_note_lbl": ("muted", False),
        "preview_fusion_info_lbl": ("muted", False),
        "preview_box_mode_info_lbl": ("muted", False),
        "preview_import_focus_hint_lbl": ("muted", False),
        "footer_test_status_lbl": ("neutral", True),
        "extract_source_intro_lbl": ("muted", False),
        "extract_run_hint_lbl": ("muted", False),
        "extract_continue_xml_value_lbl": ("muted", False),
        "extract_continue_images_value_lbl": ("muted", False),
        "extract_xml_hint_lbl": ("muted", False),
        "extract_images_hint_lbl": ("muted", False),
        "extract_source_fields_hint_lbl": ("muted", False),
        "extract_start_hint_lbl": ("muted", False),
        "dataset_intro_lbl": ("muted", False),
        "pz3_flow_lbl": ("success", True),
        "review_intro_lbl": ("muted", False),
        "cvat_option1_title_lbl": ("error", False),
        "cvat_option2_title_lbl": ("success", True),
        "cvat_option1_desc_lbl": ("muted", False),
        "cvat_option2_desc_lbl": ("muted", False),
        "pz3_dataset_source_intro_lbl": ("muted", False),
        "pz3_source_preview_status_lbl": ("muted", False),
        "pz3_source_pool_status_lbl": ("muted", False),
        "pz3_source_next_status_lbl": ("muted", False),
        "pz3_existing_dataset_title_lbl": ("default", False),
        "pz3_existing_dataset_desc_lbl": ("muted", False),
        "pz3_existing_dataset_status_lbl": ("muted", False),
        "pz3_dataset_action_hint_lbl": ("muted", False),
        "pz3_goldpack_step_title_lbl": ("success", True),
        "pz3_goldpack_step_desc_lbl": ("muted", False),
        "pz3_export_step_title_lbl": ("success", True),
        "pz3_export_step_desc_lbl": ("muted", False),
        "cvat_import_title_lbl": ("default", False),
        "cvat_import_desc_lbl": ("muted", False),
        "dataset_status_intro_lbl": ("muted", False),
        "gold_export_filters_title_lbl": ("default", False),
        "gold_export_sources_title_lbl": ("default", False),
        "gold_export_scope_lbl": ("muted", False),
        "gold_cvat_source_status_lbl": ("muted", False),
        "gold_export_sources_hint_lbl": ("muted", False),
    }

    for label_name, (default_tone, default_emphasis) in inline_label_defaults.items():
        label = getattr(self, label_name, None)
        if label is None or not isinstance(label, tk.Label):
            continue
        try:
            self._set_inline_status_label_state(
                label,
                text=label.cget("text"),
                tone=getattr(label, "_inline_status_tone", default_tone),
                emphasis=getattr(label, "_inline_status_emphasis", default_emphasis),
            )
        except Exception:
            pass

    try:
        refresh = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            refresh()
    except Exception:
        pass

    try:
        refresh = getattr(self, "_refresh_pz3_cards_ui", None)
        if callable(refresh):
            refresh()
    except Exception:
        pass
