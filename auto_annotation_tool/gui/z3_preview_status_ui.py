from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from .web_slim_scrollbar import blend_hex_colors

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def apply_preview_source_actions_style(host: "CharacterAnnotationTab") -> None:
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    panel_alt = palette.get("panel_alt", panel_bg)
    fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted_dim", palette.get("muted", "#a0a0a0"))
    hover_bg = palette.get("button_hover", panel_alt)
    active_bg = palette.get("surface_info", hover_bg)
    arrow_color = palette.get("success", "#2ecc71")
    subtle_green = blend_hex_colors(arrow_color, panel_bg, 0.72)
    active_green = blend_hex_colors(arrow_color, panel_bg, 0.38)

    overlay = getattr(host, "preview_record_overlay", None)
    if overlay is not None:
        try:
            overlay.configure(
                bg=panel_bg,
                highlightbackground=panel_bg,
                highlightcolor=panel_bg,
            )
        except Exception:
            pass

    style = getattr(host.app, "style", None)
    if style is not None:
        try:
            style.configure(
                "PreviewOverlayFlat.TMenubutton",
                background=panel_bg,
                foreground=fg,
                bordercolor=subtle_green,
                lightcolor=subtle_green,
                darkcolor=subtle_green,
                arrowcolor=arrow_color,
                padding=(3, 0),
                borderwidth=1,
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
            )
            style.map(
                "PreviewOverlayFlat.TMenubutton",
                background=[
                    ("active", blend_hex_colors(arrow_color, panel_bg, 0.78)),
                    ("pressed", active_bg),
                    ("disabled", panel_bg),
                ],
                foreground=[("disabled", muted_fg)],
                bordercolor=[
                    ("active", active_green),
                    ("pressed", active_green),
                    ("disabled", blend_hex_colors(arrow_color, panel_bg, 0.82)),
                ],
                lightcolor=[
                    ("active", active_green),
                    ("pressed", active_green),
                    ("disabled", blend_hex_colors(arrow_color, panel_bg, 0.82)),
                ],
                darkcolor=[
                    ("active", active_green),
                    ("pressed", active_green),
                    ("disabled", blend_hex_colors(arrow_color, panel_bg, 0.82)),
                ],
                arrowcolor=[
                    ("active", arrow_color),
                    ("pressed", arrow_color),
                    ("!disabled", arrow_color),
                    ("disabled", muted_fg),
                ],
            )
        except Exception:
            pass

    for btn in [getattr(host, "preview_box_mode_toggle_btn", None)]:
        if btn is None:
            continue
        try:
            btn.configure(style="PreviewOverlayFlat.TMenubutton")
        except Exception:
            pass

    for menu in [getattr(host, "preview_box_mode_toggle_menu", None)]:
        if menu is None:
            continue
        try:
            menu.configure(
                background=panel_alt,
                foreground=arrow_color,
                activebackground=blend_hex_colors(arrow_color, panel_bg, 0.82),
                activeforeground=arrow_color,
                disabledforeground=muted_fg,
                bd=1,
                relief=tk.FLAT,
                borderwidth=1,
                activeborderwidth=0,
                selectcolor=panel_alt,
            )
        except Exception:
            pass


def apply_preview_info_stats_style(host: "CharacterAnnotationTab", progress_bar_cls) -> None:
    palette = getattr(host.app, "palette", {})
    bg = palette.get("panel", palette.get("bg", "#252526"))
    fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted", "#a0a0a0")
    success_fg = palette.get("success", "#2ecc71")
    error_fg = palette.get("error", "#e74c3c")
    info_fg = palette.get("info", palette.get("accent", "#4aa3ff"))

    frame = getattr(host, "preview_counts_frame", None)
    if frame is not None:
        try:
            frame.configure(bg=bg)
        except Exception:
            pass

    label_configs = (
        ("preview_perfect_title_lbl", {"bg": bg, "fg": muted_fg, "font": ("Segoe UI", 8)}),
        ("preview_error_title_lbl", {"bg": bg, "fg": muted_fg, "font": ("Segoe UI", 8)}),
        ("preview_total_title_lbl", {"bg": bg, "fg": muted_fg, "font": ("Segoe UI", 8)}),
        ("preview_perfect_count_lbl", {"bg": bg, "fg": success_fg, "font": ("Segoe UI", 12, "bold")}),
        ("preview_error_count_lbl", {"bg": bg, "fg": error_fg, "font": ("Segoe UI", 12, "bold")}),
        ("preview_unknown_count_lbl", {"bg": bg, "fg": info_fg, "font": ("Segoe UI", 12, "bold")}),
        ("preview_layout_summary_lbl", {"bg": bg, "fg": muted_fg, "font": ("Segoe UI", 8)}),
        ("preview_repair_progress_title_lbl", {"bg": bg, "fg": fg, "font": ("Segoe UI", 9, "bold")}),
    )
    for name, config in label_configs:
        widget = getattr(host, name, None)
        if widget is None:
            continue
        try:
            widget.configure(**config)
        except Exception:
            pass

    progress = getattr(host, "preview_repair_progress", None)
    if isinstance(progress, progress_bar_cls):
        try:
            progress.configure(bg=bg, background=bg, trough_color=palette.get("border", "#4b4b4b"))
        except Exception:
            pass

def apply_preview_typing_overlay_style(host: "CharacterAnnotationTab") -> None:
    palette = getattr(host.app, "palette", {})
    overlay = getattr(host, "preview_typing_overlay", None)
    label = getattr(host, "preview_typing_overlay_lbl", None)
    if overlay is None or label is None:
        return

    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", panel)
    accent = palette.get("accent", palette.get("success", "#2ecc71"))
    border = blend_hex_colors(accent, panel_alt, 0.28)
    fill = blend_hex_colors(panel_alt, panel, 0.22)
    fg = palette.get("muted", palette.get("fg", "#f3f3f3"))
    key_fg = palette.get("success", accent)

    try:
        overlay.configure(
            bg=fill,
            highlightbackground=border,
            highlightcolor=border,
            padx=6,
            pady=4,
        )
    except Exception:
        pass
    try:
        label.configure(
            bg=fill,
            fg=fg,
            font=("Segoe UI", 8, "normal"),
            insertbackground=fg,
            selectbackground=blend_hex_colors(accent, fill, 0.42),
            selectforeground=palette.get("fg", "#f3f3f3"),
        )
        if isinstance(label, tk.Text):
            label.tag_configure("body", foreground=fg, font=("Segoe UI", 8, "normal"))
            label.tag_configure("key", foreground=key_fg, font=("Segoe UI", 8, "bold"))
    except Exception:
        pass


def apply_preview_surface_style(host: "CharacterAnnotationTab") -> None:
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    for widget_name in ("preview_lf", "preview_list_lf"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(
                bg=panel_bg,
                highlightbackground=panel_border,
                highlightcolor=panel_border,
            )
            host.app.style_panel_surface(widget, background=panel_bg)
        except Exception:
            pass

    canvas = getattr(host, "preview_controls_canvas", None)
    if canvas is not None:
        try:
            canvas.configure(bg=panel_bg)
        except Exception:
            pass

    for widget_name in (
        "detect_left_intro_lbl",
        "preview_list_intro_lbl",
        "preview_intro_lbl",
        "preview_shortcuts_lbl",
        "preview_record_source_lbl",
        "preview_edit_status_lbl",
        "preview_load_note_lbl",
        "preview_layout_summary_lbl",
        "preview_fusion_info_lbl",
        "preview_box_mode_info_lbl",
        "preview_import_focus_hint_lbl",
    ):
        widget = getattr(host, widget_name, None)
        if widget is None or not isinstance(widget, tk.Label):
            continue
        try:
            widget._inline_status_bg = panel_bg
        except Exception:
            pass

    host._apply_preview_focus_prompt_style()
    host._apply_preview_typing_overlay_style()


def sync_preview_intro_wraplength(host: "CharacterAnnotationTab", event=None) -> None:
    try:
        width = int(getattr(event, "width", 0) or getattr(host, "preview_tools", None).winfo_width() or 0)
    except Exception:
        width = 0
    wraplength = max(240, width - 24) if width > 0 else 780
    for widget_name in ("preview_intro_lbl", "preview_shortcuts_lbl"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(wraplength=wraplength)
        except Exception:
            pass


def apply_preview_focus_prompt_style(host: "CharacterAnnotationTab") -> None:
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    panel_alt = palette.get("panel_alt", panel_bg)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent = palette.get("accent", "#0e639c")
    prompt_fill = blend_hex_colors(panel_alt, panel_bg, 0.34)
    prompt_outline = blend_hex_colors(accent, border, 0.48)
    key_fill = blend_hex_colors(palette.get("field", panel_alt), panel_bg, 0.14)
    key_fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted_dim", palette.get("muted", "#8f98a3"))

    shell = getattr(host, "preview_focus_prompt_shell", None)
    if shell is not None:
        try:
            shell.configure(bg=prompt_fill, highlightbackground=prompt_outline, highlightcolor=prompt_outline)
        except Exception:
            pass

    text_lbl = getattr(host, "preview_focus_prompt_lbl", None)
    if text_lbl is not None:
        try:
            text_lbl.configure(bg=prompt_fill, fg=muted_fg)
        except Exception:
            pass

    key_lbl = getattr(host, "preview_focus_prompt_key_lbl", None)
    if key_lbl is not None:
        try:
            key_lbl.configure(
                bg=key_fill,
                fg=key_fg,
                highlightbackground=prompt_outline,
                highlightcolor=prompt_outline,
            )
        except Exception:
            pass


def set_preview_processing_overlay(
    host: "CharacterAnnotationTab",
    active: bool,
    *,
    title: str = "",
    details: str = "",
    cancel_command=None,
    cancel_visible: bool = True,
    cancel_text: str = "Anuluj",
) -> None:
    controller = getattr(host, "preview_processing_overlay_controller", None)
    if controller is None:
        return

    if active:
        try:
            if hasattr(controller, "configure_cancel"):
                resolved_cancel = cancel_command
                if resolved_cancel is None and cancel_visible:
                    resolved_cancel = lambda: host.fast_test_stop.set()
                controller.configure_cancel(
                    command=resolved_cancel,
                    text=cancel_text,
                    visible=cancel_visible,
                    enabled=callable(resolved_cancel),
                )
            controller.show(title=str(title or ""), details=str(details or ""))
        except Exception:
            pass
    else:
        try:
            controller.hide()
            if hasattr(controller, "configure_cancel"):
                controller.configure_cancel(
                    command=lambda: host.fast_test_stop.set(),
                    text="Anuluj",
                    visible=True,
                    enabled=True,
                )
        except Exception:
            pass


def update_preview_processing_overlay_progress(
    host: "CharacterAnnotationTab",
    *,
    pct: float | None = None,
    current: int | None = None,
    total: int | None = None,
    filename: str = "",
    meta_text: str = "",
) -> None:
    controller = getattr(host, "preview_processing_overlay_controller", None)
    if controller is None:
        return
    try:
        controller.update_progress(
            pct=pct,
            current=current,
            total=total,
            filename=filename,
            meta_text=meta_text,
        )
    except Exception:
        pass


def set_preview_info(host: "CharacterAnnotationTab", text: str, tone: str = "neutral") -> None:
    label = getattr(host, "preview_info_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if label is None:
        return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def set_preview_counts_info(
    host: "CharacterAnnotationTab",
    perfect: int = 0,
    needs_fix: int = 0,
    unknown: int = 0,
    char_boxes: int = 0,
) -> None:
    ok_count = max(0, int(perfect))
    bad_count = max(0, int(needs_fix) + int(unknown))
    total_count = max(0, ok_count + bad_count)
    for widget_name in ("preview_counts_frame", "preview_layout_summary_lbl"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            if not str(widget.winfo_manager()):
                widget.grid()
        except Exception:
            pass
    label_map = (
        ("preview_perfect_count_lbl", str(ok_count)),
        ("preview_error_count_lbl", str(bad_count)),
        ("preview_unknown_count_lbl", str(total_count)),
    )
    for attr_name, value_text in label_map:
        widget = getattr(host, attr_name, None)
        if widget is None:
            continue
        try:
            widget.configure(text=value_text)
        except Exception:
            pass


def format_preview_layout_summary(counts: dict | None = None) -> str:
    source = counts if isinstance(counts, dict) else {}
    layout_counts = source.get("layout_counts", {}) if isinstance(source.get("layout_counts", {}), dict) else {}
    perfect_counts = source.get("layout_perfect_counts", {}) if isinstance(source.get("layout_perfect_counts", {}), dict) else {}

    order = ("1R", "1R*", "2R", "2R?", "2R*", "?")
    ordered_keys = [key for key in order if int(layout_counts.get(key, 0) or 0) > 0]
    ordered_keys.extend(
        sorted(
            key for key, value in layout_counts.items()
            if key not in ordered_keys and int(value or 0) > 0
        )
    )

    if not ordered_keys:
        return "Układ tablic: brak danych."

    parts = []
    for key in ordered_keys:
        total = int(layout_counts.get(key, 0) or 0)
        perfect = int(perfect_counts.get(key, 0) or 0)
        if perfect > 0:
            parts.append(f"{key}: {total} (OK {perfect})")
        else:
            parts.append(f"{key}: {total}")
    return "Układ tablic: " + " | ".join(parts)


def set_preview_layout_summary_info(host: "CharacterAnnotationTab", counts: dict | None = None) -> None:
    label = getattr(host, "preview_layout_summary_lbl", None)
    if label is None:
        return
    text = format_preview_layout_summary(counts)
    host._set_inline_status_label_state(label, text=text, tone="muted", emphasis=False)


def build_plates_list_legend_counts(host: "CharacterAnnotationTab", counts: dict | None = None) -> dict:
    metadata_map = host.preview_metadata if isinstance(getattr(host, "preview_metadata", None), dict) else {}
    status_counts = counts if isinstance(counts, dict) else host._count_preview_statuses()

    plate_total = max(0, int(status_counts.get("total", 0) or 0))
    plate_manual = 0
    plate_yolo = 0
    plate_ocr = 0
    plate_hybrid = 0

    box_total = 0
    box_manual = 0
    box_yolo = 0
    box_ocr = 0
    box_hybrid = 0

    for _pid, data in (metadata_map or {}).items():
        if not isinstance(data, dict):
            continue

        chars = list(data.get("characters", []) or [])
        source_counts = host._count_character_sources(chars, data=data)
        manual_count = max(
            int(source_counts.get("manual", 0) or 0),
            int(source_counts.get("manual_box", 0) or 0),
            int(source_counts.get("manual_sign", 0) or 0),
        )
        ocr_count = max(
            int(source_counts.get("ocr", 0) or 0),
            int(source_counts.get("generated_box", 0) or 0),
            int(source_counts.get("ocr_symbol", 0) or 0),
        )
        yolo_count = max(
            int(source_counts.get("yolo", 0) or 0),
            int(source_counts.get("yolo_box", 0) or 0),
            int(source_counts.get("yolo_symbol", 0) or 0),
        )
        hybrid_count = int(source_counts.get("yolo_box_ocr", 0) or 0) + int(source_counts.get("yolo_rescue", 0) or 0)

        box_total += int(len(chars))
        box_manual += manual_count
        box_ocr += ocr_count
        box_yolo += yolo_count
        box_hybrid += hybrid_count

        if manual_count > 0 or host._get_plate_source_bucket(data) in {"local_manual", "cvat_manual"}:
            plate_manual += 1
        elif hybrid_count > 0:
            plate_hybrid += 1
        elif yolo_count > 0:
            plate_yolo += 1
        elif ocr_count > 0:
            plate_ocr += 1

    return {
        "plate_total": int(plate_total),
        "plate_perfect": int(status_counts.get("perfect", 0) or 0),
        "plate_manual": int(plate_manual),
        "plate_yolo": int(plate_yolo),
        "plate_ocr": int(plate_ocr),
        "plate_hybrid": int(plate_hybrid),
        "box_total": int(box_total),
        "box_manual": int(box_manual),
        "box_yolo": int(box_yolo),
        "box_ocr": int(box_ocr),
        "box_hybrid": int(box_hybrid),
    }


def set_plates_legend_info(host: "CharacterAnnotationTab", counts: dict | None = None) -> None:
    snapshot = build_plates_list_legend_counts(host, counts)
    label_map = (
        ("plates_legend_tab_total_lbl", str(int(snapshot.get("plate_total", 0) or 0))),
        ("plates_legend_tab_perfect_lbl", str(int(snapshot.get("plate_perfect", 0) or 0))),
        ("plates_legend_tab_manual_lbl", str(int(snapshot.get("plate_manual", 0) or 0))),
        ("plates_legend_tab_yolo_lbl", str(int(snapshot.get("plate_yolo", 0) or 0))),
        ("plates_legend_tab_ocr_lbl", str(int(snapshot.get("plate_ocr", 0) or 0))),
        ("plates_legend_tab_hybrid_lbl", str(int(snapshot.get("plate_hybrid", 0) or 0))),
        ("plates_legend_box_total_lbl", str(int(snapshot.get("box_total", 0) or 0))),
        ("plates_legend_box_perfect_lbl", "-"),
        ("plates_legend_box_manual_lbl", str(int(snapshot.get("box_manual", 0) or 0))),
        ("plates_legend_box_yolo_lbl", str(int(snapshot.get("box_yolo", 0) or 0))),
        ("plates_legend_box_ocr_lbl", str(int(snapshot.get("box_ocr", 0) or 0))),
        ("plates_legend_box_hybrid_lbl", str(int(snapshot.get("box_hybrid", 0) or 0))),
    )
    for attr_name, value_text in label_map:
        widget = getattr(host, attr_name, None)
        if widget is None:
            continue
        try:
            widget.configure(text=value_text)
        except Exception:
            pass


def set_preview_repair_progress_status(host: "CharacterAnnotationTab", text: str, tone: str = "muted") -> None:
    label = getattr(host, "preview_repair_progress_status_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def sync_preview_edit_status_visibility(host) -> None:
    label = getattr(host, "preview_edit_status_lbl", None)
    if label is None:
        return

    # Status operacji PZ2 jest nakładką canvasa; label w siatce potrafił pompować canvas.
    try:
        label.grid_remove()
    except Exception:
        pass


def update_preview_edit_status(
    host,
    extra_message: str | None = None,
    tone: str = "muted",
    emphasis: bool | None = None,
) -> None:
    label_mode_active, label_input_active = host._get_preview_typing_state()
    message = str(extra_message or "").strip()
    overlay_only_message = ""
    overlay_available = host._preview_operation_overlay_available()

    if not message:
        add_state = getattr(host, "_preview_char_add_state", None)
        if isinstance(add_state, dict) and bool(add_state.get("click_draw")):
            message = "Rysowanie boxa: przesuń mysz i kliknij drugi narożnik."
            tone = "info"
            overlay_only_message = message if overlay_available else ""
        elif bool(getattr(host, "_preview_char_add_click_armed", False)):
            message = "D uzbrojone: kliknij pierwszy narożnik boxa na tablicy."
            tone = "info"
            overlay_only_message = message if overlay_available else ""
        elif host._preview_char_add_requested():
            message = "Narysuj nowy box przeciągnięciem LPM po tablicy."
            tone = "info"
            overlay_only_message = message if overlay_available else ""
        elif label_mode_active:
            overlay_only_message = host._format_preview_typing_status(persistent=True)
            message = "Tryb wpisywania znaków aktywny."
            tone = "info"
        elif label_input_active:
            overlay_only_message = host._format_preview_typing_status(persistent=False)
            message = "Aktywne pole znaku."
            tone = "info"
        elif bool(getattr(host, "_preview_char_edit_mode", False)):
            message = "Tryb edycji: S zaznacza lub odznacza hoverowany box, LPM+drag przesuwa zaznaczony box, uchwyty rogów zmieniają rozmiar, Alt+W włącza wpisywanie znaków, a strzałki lewo/prawo przełączają boxy."
            tone = "muted"
            overlay_only_message = message if overlay_available else ""
        else:
            message = (
                "Q/E przełącza poprzednią/następną tablicę na liście, D uzbraja nowy box, "
                "S zaznacza lub odznacza hoverowany box, Alt+W włącza wpisywanie znaków, "
                "Enter przełącza pełny ekran. "
                f"{host._format_preview_layout_semantics(short=True)}"
            )
            tone = "muted"
            overlay_only_message = ""
    elif label_mode_active:
        overlay_only_message = host._format_preview_typing_status(message, persistent=True)
        message = "Tryb wpisywania znaków aktywny."
    elif label_input_active:
        overlay_only_message = host._format_preview_typing_status(message, persistent=False)
        message = "Aktywne pole znaku."
    elif overlay_available:
        overlay_only_message = message

    if emphasis is None:
        emphasis = False

    label = getattr(host, "preview_edit_status_lbl", None)
    use_canvas_overlay = bool(overlay_available)
    if use_canvas_overlay and label is not None:
        try:
            label.grid_remove()
        except Exception:
            pass

    next_overlay_text = str(overlay_only_message or "")
    try:
        previous_overlay_text = str(getattr(host, "_preview_typing_overlay_text", "") or "")
    except Exception:
        previous_overlay_text = ""
    overlay_text_changed = next_overlay_text != previous_overlay_text
    try:
        host._preview_typing_overlay_text = next_overlay_text
    except Exception:
        pass

    if not use_canvas_overlay:
        if hasattr(host, "preview_edit_status_var"):
            try:
                host.preview_edit_status_var.set(message)
            except Exception:
                pass
        if label is not None:
            host._set_inline_status_label_state(label, text=message, tone=tone, emphasis=bool(emphasis))

    host._sync_preview_edit_status_visibility()
    if use_canvas_overlay and not overlay_text_changed:
        return
    host._refresh_preview_typing_overlay_visibility()


def refresh_preview_typing_overlay_visibility(host) -> None:
    overlay = getattr(host, "preview_typing_overlay", None)
    label = getattr(host, "preview_typing_overlay_lbl", None)
    canvas = getattr(host, "preview_canvas", None)
    if overlay is None or label is None or canvas is None:
        return

    overlay_text = host._get_preview_typing_overlay_text()
    if not overlay_text:
        try:
            overlay.place_forget()
        except Exception:
            pass
        return

    canvas_w, canvas_h = host._get_preview_canvas_size()
    if canvas_w <= 0.0 or canvas_h <= 0.0:
        return

    wraplength = max(130, min(185, int(canvas_w * 0.18)))
    layout_key = (str(overlay_text), int(wraplength))
    host._configure_preview_typing_overlay_text(overlay_text, wraplength)

    layout_cache = getattr(host, "_preview_typing_overlay_layout_cache", None)
    if not isinstance(layout_cache, dict):
        layout_cache = {}
        try:
            host._preview_typing_overlay_layout_cache = layout_cache
        except Exception:
            pass
    cached_size = layout_cache.get(layout_key)
    if isinstance(cached_size, tuple) and len(cached_size) == 2:
        try:
            overlay_w = float(cached_size[0])
            overlay_h = float(cached_size[1])
        except Exception:
            overlay_w = overlay_h = 0.0
    else:
        overlay_w = overlay_h = 0.0

    try:
        if overlay_w <= 0.0:
            overlay_w = float(max(0, int(overlay.winfo_reqwidth() or overlay.winfo_width() or 0)))
        if overlay_h <= 0.0:
            overlay_h = float(max(0, int(overlay.winfo_reqheight() or overlay.winfo_height() or 0)))
    except Exception:
        overlay_w = 0.0
        overlay_h = 0.0

    if overlay_w <= 0.0 or overlay_h <= 0.0:
        overlay_w = float(max(160, int(wraplength) + 22))
        overlay_h = 64.0
    if layout_key not in layout_cache:
        if len(layout_cache) > 18:
            try:
                layout_cache.pop(next(iter(layout_cache)))
            except Exception:
                layout_cache.clear()
        layout_cache[layout_key] = (float(overlay_w), float(overlay_h))

    anchor = host._resolve_preview_typing_overlay_anchor(overlay_w, overlay_h)
    if anchor is None:
        try:
            overlay.place_forget()
        except Exception:
            pass
        return
    x, y = anchor

    try:
        overlay.place(in_=canvas, x=x, y=y, anchor="nw")
        overlay.lift()
    except Exception:
        pass


def get_preview_repair_progress_snapshot(host, counts: dict | None = None) -> dict:
    counts = dict(counts or host._count_preview_statuses() or {})
    perfect = max(0, int(counts.get("perfect", 0) or 0))
    needs_fix = max(0, int(counts.get("needs_fix", 0) or 0))
    unknown = max(0, int(counts.get("unknown", 0) or 0))
    total = max(0, int(counts.get("total", 0) or 0))
    exportable_plate_count = 0
    exportable_char_count = 0
    try:
        export_pool = host._get_step3_yolo_export_readiness_snapshot(
            selected_strategies=host._get_selected_gold_export_strategy_buckets(),
            selected_sources=host._get_selected_gold_export_source_buckets(),
        )
        exportable_plate_count = max(0, int(export_pool.get("selected_plate_count", 0) or 0))
        exportable_char_count = max(0, int(export_pool.get("selected_char_count", 0) or 0))
        min_required_perfect = max(1, int(export_pool.get("min_exportable_plate_count", 1) or 1))
        export_ready = bool(export_pool.get("ok"))
    except Exception:
        exportable_plate_count = host._count_exportable_perfect_plates_in_metadata(host.preview_metadata)
        try:
            exportable_char_count = int(
                sum(int(v or 0) for v in dict(counts.get("strategy_char_counts", {}) or {}).values())
            )
        except Exception:
            exportable_char_count = 0
        min_required_perfect = 10 if bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()) else 1
        export_ready = bool(exportable_plate_count >= int(min_required_perfect) and exportable_char_count > 0)
    progress_value = (float(perfect) / float(total) * 100.0) if total > 0 else 0.0

    train_count = 0
    val_count = 0
    test_count = 0
    train_pct = 0.0
    val_pct = 0.0
    test_pct = 0.0
    split_ready = bool(export_ready)
    missing_for_split = max(0, int(min_required_perfect) - int(exportable_plate_count or 0))
    in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    dataset_ready = False
    dataset_message = ""
    if in_campaign:
        try:
            training_readiness = host._get_campaign_step3_training_readiness()
            dataset_ready = bool(training_readiness.get("ok"))
            dataset_message = str(training_readiness.get("message", "") or "").strip()
        except Exception:
            dataset_ready = False
            dataset_message = ""

    if total <= 0:
        tone = "muted"
        summary = "Warunek eksportu PZ3: czekam na tablice."
        details = (
            f"Warunek eksportu PZ3 w kampanii: co najmniej {min_required_perfect} tablic perfect, "
            "z przynajmniej jednym poprawnym boxem znaku i etykietą, objęta aktywnym zakresem gold packa."
        )
    elif not split_ready:
        tone = "error"
        summary = f"Warunek eksportu PZ3 niespełniony: do datasetu przechodzi {exportable_plate_count} tablic / {exportable_char_count} znaków."
        if perfect > 0:
            details = (
                f"W PZ2 są tablice perfect ({perfect}), ale minimum kampanii to {min_required_perfect} "
                "tablic eksportowalnych do PZ3. Każda musi mieć zapisany przynajmniej jeden poprawny box znaku "
                "z etykietą i przechodzić przez aktualne strategie oraz źródła gold packa."
            )
        else:
            details = (
                f"Aby wejście do PZ3 miało sens, przygotuj w PZ2 co najmniej {min_required_perfect} tablic perfect: "
                "oznacz wszystkie znaki, zapisz boxy z etykietami i doprowadź tablicę do statusu perfect."
            )
    else:
        ready_target = (
            "Materiał spełnia warunek E3."
            if getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()
            else "Materiał jest gotowy do eksportu źródłowego datasetu."
        )
        ratio = (float(perfect) / float(total)) if total > 0 else 0.0
        if ratio < 0.6:
            tone = "warning"
            details = (
                f"Warunek eksportu PZ3 jest spełniony: minimum {min_required_perfect} eksportowalnych tablic perfect jest osiągnięte. "
                "Dataset można utworzyć, ale im więcej tablic perfect, tym większy sens treningu w Z4."
            )
        else:
            tone = "success"
            details = f"{ready_target} Wariant treningowy i split train / val / test przygotujesz w Z4."
        summary = f"Warunek eksportu PZ3 spełniony: do datasetu przechodzi {exportable_plate_count} tablic / {exportable_char_count} znaków."

    if in_campaign:
        if not split_ready:
            if missing_for_split > 0:
                gate_summary = f"Krok 1/2: zbiór PZ2 nie jest jeszcze gotowy. Brakuje {missing_for_split} tablic perfect."
            else:
                gate_summary = "Krok 1/2: zbiór PZ2 nie jest jeszcze gotowy. Brakuje eksportowalnych ramek znaków."
            gate_details = (
                f"Cel PZ2: przygotować anotacje znaków do późniejszego datasetu. Masz {exportable_plate_count}/{min_required_perfect} "
                f"eksportowalnych tablic perfect i {exportable_char_count} znaków. "
                "Po spełnieniu tego minimum użyj przycisku „Krok 2: dataset PZ3”; w PZ3 powstanie właściwy dataset."
            )
            if perfect != exportable_plate_count:
                gate_details += (
                    f" Na liście jest {perfect} tablic perfect, ale do bramki liczy się {exportable_plate_count}; "
                    "sprawdź zakres strategii i źródeł gold packa."
                )
            summary = gate_summary
            details = f"{gate_details} {details}".strip()
        elif not dataset_ready:
            summary = "Krok 1/2 gotowy: zbiór PZ2 spełnia minimum. Krok 2/2: utwórz dataset znaków w PZ3."
            gate_details = (
                f"Zbiór PZ2 zawiera {exportable_plate_count}/{min_required_perfect} eksportowalnych tablic perfect "
                f"i {exportable_char_count} znaków. To odblokowuje przejście do PZ3, ale nie otwiera jeszcze T06. "
                "Dopiero eksport źródłowego datasetu znaków w PZ3 domyka warunek bramki."
            )
            if dataset_message:
                gate_details += f" {dataset_message}"
            details = gate_details

    return {
        "perfect": perfect,
        "needs_fix": needs_fix,
        "unknown": unknown,
        "total": total,
        "exportable_plate_count": int(exportable_plate_count),
        "exportable_char_count": int(exportable_char_count),
        "progress_value": max(0.0, min(100.0, progress_value)),
        "tone": tone,
        "summary": summary,
        "details": details,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_pct": float(train_pct),
        "val_pct": float(val_pct),
        "test_pct": float(test_pct),
        "split_ready": split_ready,
        "min_required_perfect": int(min_required_perfect),
        "missing_for_split": int(missing_for_split),
    }


def update_preview_repair_progress_ui(host, counts: dict | None = None):
    snapshot = get_preview_repair_progress_snapshot(host, counts)
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    trough = palette.get("border", "#4b4b4b")
    tone = str(snapshot.get("tone", "muted") or "muted").strip().lower()

    fill_map = {
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f0b44c"),
        "error": palette.get("error", "#e74c3c"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "muted": palette.get("muted", "#9aa0a6"),
        "neutral": palette.get("accent", "#4aa3ff"),
    }
    fill = fill_map.get(tone, palette.get("accent", "#4aa3ff"))

    title = getattr(host, "preview_repair_progress_title_lbl", None)
    if title is not None:
        try:
            title.configure(
                text=str(snapshot.get("summary") or "Postęp poprawek"),
                bg=panel_bg,
                fg=palette.get("fg", "#f3f3f3"),
            )
        except Exception:
            pass

    progress = getattr(host, "preview_repair_progress", None)
    if progress is not None and hasattr(progress, "configure"):
        try:
            progress.configure(
                maximum=100.0,
                value=float(snapshot.get("progress_value", 0.0) or 0.0),
                bg=panel_bg,
                background=panel_bg,
                trough_color=trough,
                fill_color=fill,
            )
        except Exception:
            pass

    detail = str(snapshot.get("details") or "").strip()
    host._set_preview_repair_progress_status(detail, tone=tone)
    try:
        if bool(getattr(host, "_step3_linear_mode", False)) and CAMPAIGN.get_active_project_name():
            host.frame.after_idle(lambda: host._place_preview_overlay_dock(force_render=False))
    except Exception:
        pass


def set_preview_fusion_info(host: "CharacterAnnotationTab", text: str, tone: str = "muted") -> None:
    label = getattr(host, "preview_fusion_info_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def set_preview_box_info(host: "CharacterAnnotationTab", text: str, tone: str = "muted") -> None:
    label = getattr(host, "preview_box_mode_info_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def update_preview_box_info_label(
    host: "CharacterAnnotationTab",
    plate_id=None,
    mode_key=None,
    shown_count=None,
    yolo_raw_count=None,
    yolo_nms_count=None,
    yolo_filtered_count=None,
) -> None:
    if not plate_id:
        host._set_preview_box_info("Źródło ramek znaków: brak zaznaczonej tablicy", "muted")
        return

    label = host._get_preview_box_mode_label(mode_key)
    details = [f"{plate_id}", f"ramki znaków: {label}"]
    data = host.preview_metadata.get(str(plate_id), {}) if plate_id else {}
    status_meta = host._get_preview_status_presentation(data=data, plate_id=plate_id)
    status = str(status_meta.get("status", "unknown") or "unknown").strip().lower()
    details.append(str(status_meta.get("info_text", "status: nieoceniona")))

    if shown_count is not None:
        details.append(f"widoczne ramki: {int(shown_count)}")

    yolo_parts = []
    if yolo_raw_count is not None:
        yolo_parts.append(f"raw={int(yolo_raw_count)}")
    if yolo_nms_count is not None:
        yolo_parts.append(f"nms={int(yolo_nms_count)}")
    if yolo_filtered_count is not None:
        yolo_parts.append(f"filtr={int(yolo_filtered_count)}")
    if yolo_parts:
        details.append("YOLO: " + ", ".join(yolo_parts))

    try:
        plate_detection_line = host._format_plate_last_detection_line(data)
    except Exception:
        plate_detection_line = ""
    if plate_detection_line:
        details.append(plate_detection_line)

    if bool(getattr(host, "_preview_char_add_mode", False)):
        details.append("korekta: nowy box")
    elif bool(getattr(host, "_preview_char_edit_mode", False)):
        details.append("korekta: edycja")

    severity = str(status_meta.get("severity", "muted") or "muted").strip().lower()
    if severity == "success":
        tone = "success"
    elif severity == "error":
        tone = "error"
    elif severity == "warning" or status == "needs_fix":
        tone = "warning"
    else:
        tone = "info" if shown_count else "warning"
    host._set_preview_box_info(" | ".join(details), tone)


def update_preview_info_label(host: "CharacterAnnotationTab") -> None:
    try:
        counts = host._count_preview_statuses()
        host._preview_status_counts_snapshot = dict(counts)
        host._set_preview_info(f"Wczytano tablic: {counts['total']}", "neutral")
        host._set_preview_counts_info(
            perfect=counts["perfect"],
            needs_fix=counts["needs_fix"],
            unknown=counts["unknown"],
            char_boxes=counts.get("char_boxes", 0),
        )
        host._set_preview_layout_summary_info(counts)
        host._set_plates_legend_info(counts)
        host._update_preview_repair_progress_ui(counts)
        host._set_preview_fusion_info(
            host._format_perfect_strategy_counts(counts.get("strategy_counts", {})),
            "muted",
        )
        host._refresh_gold_export_filter_labels()
        host._refresh_gold_export_source_labels()
        host._refresh_gold_export_scope_label()
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć preview_info_lbl: {e}")


def refresh_preview_live_metadata_ui(
    host,
    *,
    status_message: str | None = None,
    status_tone: str = "muted",
    render_preview: bool = True,
    refresh_row: bool = True,
) -> None:
    pid = str(getattr(host, "_preview_active_pid", "") or "").strip()
    data = host._get_preview_active_data(create=False)
    if pid and isinstance(data, dict):
        if bool(refresh_row):
            host._refresh_preview_listbox_row(pid)
        if render_preview:
            host._update_preview_info_label()
        else:
            host._schedule_preview_info_refresh()

        if render_preview:
            box_chars, box_source = host._get_preview_box_records(data)
            yolo_raw_count = len(data.get("yolo_raw_detections", []) or [])
            yolo_nms_count = len(data.get("yolo_nms_detections", []) or [])
            yolo_filtered_count = len(data.get("yolo_detections", []) or [])
        else:
            box_chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
            box_source = "FINAL"
            yolo_raw_count = len(data.get("yolo_raw_detections", []) or [])
            yolo_nms_count = len(data.get("yolo_nms_detections", []) or [])
            yolo_filtered_count = len(data.get("yolo_detections", []) or [])
        host._update_preview_box_info_label(
            plate_id=pid,
            mode_key=box_source,
            shown_count=len(box_chars),
            yolo_raw_count=yolo_raw_count,
            yolo_nms_count=yolo_nms_count,
            yolo_filtered_count=yolo_filtered_count,
        )
        if render_preview:
            host._sync_step3_access_from_preview_state(host.preview_metadata)

    host._refresh_preview_editor_toolbar()
    if status_message:
        host._update_preview_edit_status(status_message, tone=status_tone)
    else:
        host._update_preview_edit_status()
    if render_preview:
        host._on_preview_select(None)
    host._focus_preview_canvas()


def refresh_preview_layout_override_ui_light(host, *, message: str, tone: str = "info") -> None:
    pid = str(getattr(host, "_preview_active_pid", "") or "").strip()
    data = host._get_preview_active_data(create=False)
    if not pid or not isinstance(data, dict):
        return

    try:
        host._refresh_preview_listbox_row(pid)
    except Exception:
        pass

    try:
        box_chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
        host._update_preview_box_info_label(
            plate_id=pid,
            mode_key="FINAL",
            shown_count=len(box_chars),
            yolo_raw_count=len(data.get("yolo_raw_detections", []) or []),
            yolo_nms_count=len(data.get("yolo_nms_detections", []) or []),
            yolo_filtered_count=len(data.get("yolo_detections", []) or []),
        )
    except Exception:
        pass

    canvas = getattr(host, "preview_canvas", None)
    if canvas is not None and getattr(host, "_preview_render_state", None):
        try:
            canvas.delete("preview_overlay")
            canvas.delete("preview_overlay_action")
            canvas.delete("preview_char")
            host._preview_char_runtime = {}
            host._preview_char_record_render_tags = {}
            box_chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
            host._draw_preview_plate_status_frame(data)
            host._draw_preview_canvas_info_overlay(
                canvas,
                max(50, int(canvas.winfo_width() or 50)),
                data=data,
                box_chars=box_chars,
                has_boxes=bool(box_chars),
            )
            for idx, _rec in enumerate(host._get_preview_active_character_records(create=False)):
                host._redraw_preview_character_overlay_only(idx)
            host._draw_preview_layout_separator(data)
            try:
                canvas.tag_raise("preview_overlay")
                canvas.tag_raise("preview_overlay_action")
            except Exception:
                pass
            try:
                host._place_preview_overlay_dock()
                host._refresh_preview_typing_overlay_visibility()
            except Exception:
                pass
            host._apply_preview_canvas_cursor()
        except Exception as exc:
            logger.debug(f"Lekki refresh układu tablicy nie powiódł się: {exc}")
            host._on_preview_select(None)

    host._refresh_preview_editor_toolbar()
    host._update_preview_edit_status(message, tone=tone)
    host._focus_preview_canvas()

