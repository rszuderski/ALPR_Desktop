"""Rendering helpers for the Z2 canvas controls legend.

The owner object still owns state, hit boxes and cached image assets. This
module keeps the repetitive canvas drawing code out of the main Z2 tab class.
"""

from __future__ import annotations

import math
import tkinter as tk

from .web_slim_scrollbar import blend_hex_colors


def get_preview_legend_theme(owner) -> dict:
    palette = getattr(owner.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", panel)
    field = palette.get("field", panel)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent = palette.get("accent", "#0e639c")
    success = palette.get("success", "#2fbf71")
    surface_success = palette.get("surface_success", blend_hex_colors(success, panel_alt, 0.72))
    warning = palette.get("warning", "#f59e0b")
    error = palette.get("error", "#dc2626")
    surface_error = palette.get("surface_error", blend_hex_colors(error, panel_alt, 0.72))
    try:
        panel_rgb = tuple(int(str(panel).lstrip("#")[idx:idx + 2], 16) for idx in (0, 2, 4))
        panel_luma = ((0.2126 * panel_rgb[0]) + (0.7152 * panel_rgb[1]) + (0.0722 * panel_rgb[2])) / 255.0
    except Exception:
        panel_luma = 0.0
    if panel_luma >= 0.6:
        canvas_bg = blend_hex_colors(str(panel), str(success), 0.05)
        shell_fill = blend_hex_colors(str(panel), "#ffffff", 0.34)
        entry_fill = blend_hex_colors(shell_fill, str(success), 0.06)
        section_fill = blend_hex_colors(shell_fill, str(success), 0.04)
        label_fill = "#172033"
        muted_fill = "#516070"
        entry_text = "#111827"
        section_title = "#172033"
        section_muted = "#4b5b6b"
        token_fill = blend_hex_colors("#ffffff", str(success), 0.05)
        token_text = "#111827"
        panel_outline = str(
            palette.get(
                "panel_border_strong",
                blend_hex_colors("#111827", str(border), 0.42),
            )
        )
        compass_guide = "#7b8796"
        compass_dot = "#111827"
    else:
        canvas_bg = blend_hex_colors(str(success), str(panel), 0.18)
        shell_fill = blend_hex_colors(str(panel_alt), "#0b1016", 0.24)
        entry_fill = blend_hex_colors(shell_fill, "#ffffff", 0.08)
        section_fill = blend_hex_colors(shell_fill, "#ffffff", 0.05)
        label_fill = "#f4f7fb"
        muted_fill = "#c7d0db"
        entry_text = "#f8fafc"
        section_title = "#eef4fb"
        section_muted = "#e3ebf4"
        token_fill = blend_hex_colors(field, shell_fill, 0.72)
        token_text = "#f8fafc"
        panel_outline = border
        compass_guide = "#d7dde4"
        compass_dot = "#f3f3f3"

    return {
        "canvas_bg": canvas_bg,
        "panel_fill": shell_fill,
        "panel_outline": panel_outline,
        "shell_outline": success,
        "label_fill": label_fill,
        "muted_fill": muted_fill,
        "entry_fill": entry_fill,
        "entry_text": entry_text,
        "section_fill": section_fill,
        "section_title": section_title,
        "section_muted": section_muted,
        "token_fill": token_fill,
        "token_text": token_text,
        "badge_file_fill": palette.get("surface_info", blend_hex_colors(accent, shell_fill, 0.72)),
        "badge_file_outline": accent,
        "badge_image_fill": palette.get("surface_warning", blend_hex_colors(warning, shell_fill, 0.72)),
        "badge_image_outline": warning,
        "badge_plate_fill": surface_success,
        "badge_plate_outline": success,
        "badge_plate_multi_fill": surface_error,
        "badge_plate_multi_outline": error,
        "compass_guide": compass_guide,
        "compass_dot": compass_dot,
        "compass_north": "#d64545",
        "compass_south": "#183a63",
    }


def build_preview_controls_context_rows(owner) -> list[tuple[str, str, str, str]]:
    context = owner._get_preview_legend_context()
    theme = owner._get_preview_legend_theme()
    plate_total = int(context.get("plate_total", 0) or 0)
    plate_fill = theme["badge_plate_multi_fill"] if plate_total > 1 else theme["badge_plate_fill"]
    plate_outline = theme["badge_plate_multi_outline"] if plate_total > 1 else theme["badge_plate_outline"]
    rows = [
        (
            "Zdjęcie",
            str(context.get("image_text") or "").replace("Zdjęcie: ", ""),
            theme["badge_image_fill"],
            theme["badge_image_outline"],
        ),
        (
            "Tablica",
            str(context.get("plate_text") or "").replace("Tablica: ", ""),
            plate_fill,
            plate_outline,
        ),
        (
            "Pojazd",
            str(context.get("vehicle_text") or "").replace("Pojazd: ", ""),
            theme["badge_image_fill"],
            theme["badge_image_outline"],
        ),
    ]
    if str(context.get("pool_text") or "").strip():
        rows.append(
            (
                "Pula",
                str(context.get("pool_text") or "").replace("Pula: ", ""),
                theme["entry_fill"],
                theme["panel_outline"],
            )
        )
    return rows


def preview_controls_row_is_multi_plate_badge(row_label: str, row_value: str) -> bool:
    if str(row_label or "").strip().lower() != "tablica":
        return False
    value = str(row_value or "").strip()
    if "/" not in value:
        return False
    try:
        _current, total = value.split("/", 1)
        return int(total.strip()) > 1
    except Exception:
        return False


def _preview_controls_context_fonts(owner):
    compact = not owner._is_preview_controls_legend_expanded()
    return (
        owner._get_preview_legend_font(11 if compact else 8, "bold"),
        owner._get_preview_legend_font(9 if compact else 8, "normal"),
        owner._get_preview_legend_font(10 if compact else 8, "bold"),
    )


def _compact_preview_controls_layout(owner, width):
    """Measure wrapped text with the same Canvas fonts used for painting."""
    context = owner._get_preview_legend_context()
    rows = owner._build_preview_controls_context_rows()
    file_font, meta_font, value_font = _preview_controls_context_fonts(owner)
    canvas = getattr(owner, "preview_controls_canvas", None)
    key = (int(width), str(context.get("filename", "")), tuple((r[0], r[1]) for r in rows),
           tuple(tuple(sorted(font.actual().items())) for font in (file_font, meta_font, value_font)),
           canvas is not None)
    cached = getattr(owner, "_compact_preview_controls_layout_cache", None)
    if cached and cached[0] == key:
        return cached[1]

    def text_height(text, font, available_width):
        available_width = max(40.0, float(available_width))
        if canvas is not None:
            item = None
            try:
                item = canvas.create_text(-10000, -10000, anchor="nw", text=text,
                                          font=font, width=available_width)
                bbox = canvas.bbox(item)
                if bbox:
                    return float(bbox[3] - bbox[1])
            finally:
                if item is not None:
                    canvas.delete(item)
        return max(1, math.ceil(font.measure(str(text)) / available_width)) * font.metrics("linespace")

    value_x = 12.0 + max([64.0] + [float(meta_font.measure(row[0])) + 12 for row in rows])
    file_h = text_height(f"Plik: {context.get('filename', '')}", file_font, width - 24)
    row_heights = tuple(
        max(float(meta_font.metrics("linespace")),
            float(value_font.metrics("linespace")) + 6 if owner._preview_controls_row_is_multi_plate_badge(row[0], row[1])
            else text_height(row[1], value_font, width - value_x - 20)) + 6
        for row in rows
    )
    title_font = owner._get_preview_legend_font(10, "bold")
    state_font = owner._get_preview_legend_font(9, "normal")
    header_h = max(38.0, 5 + title_font.metrics("linespace") + 4 + state_font.metrics("linespace"))
    layout = {"value_x": value_x, "file_h": file_h, "row_heights": row_heights,
              "height": 10 + header_h + 8 + file_h + 10 + sum(row_heights) + 18}
    owner._compact_preview_controls_layout_cache = (key, layout)
    return layout


def get_preview_controls_legend_target_width(owner) -> int:
    canvas_frame = getattr(owner, "canvas_frame", None)
    try:
        frame_width = int(canvas_frame.winfo_width() or canvas_frame.winfo_reqwidth() or 0) if canvas_frame is not None else 0
    except Exception:
        frame_width = 0
    if frame_width <= 0:
        frame_width = 420

    expanded = owner._is_preview_controls_legend_expanded()
    if expanded:
        if bool(getattr(owner, "_preview_fullscreen_active", False)):
            return int(max(318, min(352, frame_width - 24)))
        return int(max(276, min(306, frame_width - 24)))

    context = owner._get_preview_legend_context()
    rows = owner._build_preview_controls_context_rows()
    file_font, _meta_font, value_font = _preview_controls_context_fonts(owner)
    file_width = float(file_font.measure(f"Plik: {context.get('filename', '')}")) + 18.0
    value_width = 0.0
    for row_label, row_value, _fill, _outline in rows:
        measured = float(value_font.measure(str(row_value or "")))
        if owner._preview_controls_row_is_multi_plate_badge(str(row_label), str(row_value)):
            measured += 20.0
        value_width = max(value_width, measured)
    context_width = 12.0 + 64.0 + value_width + 14.0
    if bool(getattr(owner, "_preview_fullscreen_active", False)):
        compact_min = 284.0
        compact_max = 352.0
    else:
        compact_min = 274.0
        compact_max = 330.0
    header_width = float(owner._get_preview_legend_font(10, "bold").measure("Skróty podglądu")) + 124.0
    compact_min = max(compact_min, header_width)
    compact_max = max(compact_max, compact_min)
    collapsed_width = max(compact_min, min(max(file_width, context_width), compact_max))
    return int(min(max(collapsed_width, compact_min), max(int(compact_min), frame_width - 24)))


def get_preview_controls_legend_target_height(owner, width: float | None = None) -> int:
    safe_width = max(194.0, float(width or owner._get_preview_controls_legend_target_width()))
    expanded = owner._is_preview_controls_legend_expanded()
    rows = owner._build_preview_controls_context_rows()

    if not expanded:
        return int(math.ceil(_compact_preview_controls_layout(owner, safe_width)["height"]))

    top_pad = 7.0
    header_h = 34.0
    context_h = 18.0 + (len(rows) * 18.0) + 4.0
    compact_height = top_pad + header_h + context_h + 6.0

    sections = owner._build_preview_legend_sections()
    shortcuts_h = 0.0
    for section in sections:
        items = section.get("items", []) or []
        section_cols = 2 if safe_width >= 316.0 else 1
        section_cols = max(1, min(section_cols, int(section.get("columns", section_cols) or section_cols)))
        section_rows = max(1, math.ceil(len(items) / float(section_cols)))
        shortcuts_h += 42.0 + (section_rows * 31.0)
    return int(max(compact_height + shortcuts_h + 16.0, 240.0))


def build_preview_legend_sections(owner):
    nav_label = (
        "poprz./nast. tablica"
        if bool(getattr(owner, "_preview_super_correction_active", False))
        else "poprz./nast. zdjęcie"
    )
    return [
        {
            "title": "Nawigacja",
            "accent": "#2f80ed",
            "columns": 2,
            "items": [
                {"tokens": ["Q", "E"], "connector": "/", "modes": ["tap", "tap"], "label": nav_label},
                {"tokens": ["Up", "Down"], "connector": "/", "modes": ["tap", "tap"], "label": "lista"},
                {"tokens": ["F"], "modes": ["tap"], "label": "dopasuj"},
                {"tokens": ["MMB"], "modes": ["tap"], "label": "zoom"},
                {"tokens": ["R", "LPM"], "connector": "+", "modes": ["hold", "hold"], "label": "płynny zoom x2"},
                {"tokens": ["R", "PPM"], "connector": "+", "modes": ["hold", "tap"], "label": "cofnij zoom"},
                {"tokens": ["Enter"], "modes": ["tap"], "label": "pełny ekran"},
                {"tokens": ["Y"], "modes": ["tap"], "label": "super korekta"},
            ],
        },
        {
            "title": "Fokus",
            "accent": "#14b8a6",
            "columns": 2,
            "items": [
                {"tokens": ["A"], "modes": ["tap"], "label": "tablica +/-"},
                {"tokens": ["Spacja"], "modes": ["tap"], "label": "OK/NOK"},
                {"tokens": ["R"], "modes": ["tap"], "label": "kadr tablicy"},
            ],
        },
        {
            "title": "Edycja",
            "accent": "#f59e0b",
            "columns": 2,
            "items": [
                {"tokens": ["W", "LPM"], "connector": "+", "modes": ["hold", "hold"], "label": "róg"},
                {"tokens": ["D"], "modes": ["tap"], "label": "nowa tablica"},
                {"tokens": ["S"], "modes": ["tap"], "label": "zaznacz polygon"},
                {"tokens": ["PPM"], "modes": ["tap"], "label": "usuń aktywną"},
                {"tokens": ["Delete"], "modes": ["tap"], "label": "usuń zdjęcie"},
                {"tokens": ["Esc"], "modes": ["tap"], "label": "anuluj / wyjdź z FS"},
                {"tokens": ["Ctrl+Z", "Ctrl+Y"], "connector": "/", "modes": ["tap", "tap"], "label": "historia"},
                {"tokens": ["Ctrl+S"], "modes": ["tap"], "label": "zapisz"},
            ],
        },
    ]


def _preview_controls_context_shape(owner, context_rows: list[tuple[str, str, str, str]]) -> tuple:
    return tuple(
        (
            str(row_label),
            int(owner._preview_controls_row_is_multi_plate_badge(str(row_label), str(row_value))),
        )
        for row_label, row_value, _fill, _outline in context_rows
    )


def _preview_controls_static_key(
    owner,
    *,
    legend_mode: str,
    expanded: bool,
    width: float,
    scroll_offset: float,
    legend_theme: dict,
    sections: list[dict],
    context_rows: list[tuple[str, str, str, str]],
) -> tuple:
    return (
        legend_mode,
        int(bool(expanded)),
        int(width),
        int(round(float(scroll_offset or 0.0))),
        str(legend_theme.get("canvas_bg", "")),
        str(legend_theme.get("panel_fill", "")),
        str(legend_theme.get("panel_outline", "")),
        str(legend_theme.get("section_fill", "")),
        str(legend_theme.get("section_title", "")),
        str(legend_theme.get("section_muted", "")),
        _preview_controls_context_shape(owner, context_rows),
        (() if expanded else tuple(_compact_preview_controls_layout(owner, width)[key]
                                  for key in ("height", "row_heights"))),
        tuple(
            (
                str(section.get("title", "")),
                tuple(
                    (
                        tuple(str(token) for token in item.get("tokens", [])),
                        str(item.get("connector", "") or ""),
                        tuple(str(mode) for mode in item.get("modes", []) or []),
                        str(item.get("label", "") or ""),
                    )
                    for item in section.get("items", []) or []
                ),
            )
            for section in sections
        ),
    )


def _update_preview_controls_context_only(
    owner,
    canvas,
    *,
    context: dict,
    context_rows: list[tuple[str, str, str, str]],
    legend_theme: dict,
    shell_width: float,
) -> bool:
    item_ids = getattr(owner, "_preview_controls_legend_context_item_ids", None)
    if not isinstance(item_ids, dict):
        return False
    row_items = item_ids.get("rows")
    if not isinstance(row_items, list) or len(row_items) != len(context_rows):
        return False

    inner_pad_x = 12.0
    label_fill = str(legend_theme.get("label_fill", "#f4f7fb"))
    value_fill = str(legend_theme.get("entry_text", "#ffffff"))
    _file_font, _meta_font, value_font = _preview_controls_context_fonts(owner)
    file_id = item_ids.get("file")
    if file_id is None:
        return False

    try:
        previous_file_bbox = canvas.bbox(file_id)
    except Exception:
        previous_file_bbox = None
    if not previous_file_bbox:
        return False
    try:
        canvas.itemconfigure(
            file_id,
            text=f"Plik: {context['filename']}",
            fill=label_fill,
            width=max(120.0, shell_width - (inner_pad_x * 2.0)),
        )
    except Exception:
        return False
    try:
        next_file_bbox = canvas.bbox(file_id)
    except Exception:
        next_file_bbox = None
    if previous_file_bbox and next_file_bbox:
        previous_h = int(previous_file_bbox[3] - previous_file_bbox[1])
        next_h = int(next_file_bbox[3] - next_file_bbox[1])
        if previous_h != next_h:
            return False

    for row_info, (row_label, row_value, row_fill, row_outline) in zip(row_items, context_rows):
        if not isinstance(row_info, dict) or str(row_info.get("label", "")) != str(row_label):
            return False
        row_kind = str(row_info.get("kind", ""))
        is_badge = owner._preview_controls_row_is_multi_plate_badge(str(row_label), str(row_value))
        if is_badge and row_kind != "badge":
            return False
        if not is_badge and row_kind != "text":
            return False
        if is_badge:
            rect_id = row_info.get("rect")
            text_id = row_info.get("text")
            if rect_id is None or text_id is None:
                return False
            if not canvas.bbox(rect_id) or not canvas.bbox(text_id):
                return False
            try:
                badge_text = str(row_value)
                badge_x = float(row_info.get("x", 0.0))
                badge_y = float(row_info.get("y", 0.0))
                badge_h = float(row_info.get("h", 17.0))
                badge_w = max(40.0, float(value_font.measure(badge_text)) + 18.0)
                canvas.coords(rect_id, badge_x, badge_y, badge_x + badge_w, badge_y + badge_h)
                canvas.itemconfigure(rect_id, fill=str(row_fill), outline=str(row_outline))
                canvas.coords(text_id, badge_x + (badge_w / 2.0), badge_y + (badge_h / 2.0) - 0.5)
                canvas.itemconfigure(
                    text_id,
                    text=badge_text,
                    fill=owner._get_preview_legend_text_color(str(row_fill)),
                    font=value_font,
                )
            except Exception:
                return False
        else:
            value_id = row_info.get("value")
            if value_id is None:
                return False
            if not canvas.bbox(value_id):
                return False
            try:
                canvas.itemconfigure(
                    value_id,
                    text=str(row_value),
                    fill=value_fill,
                    font=value_font,
                    width=max(80.0, shell_width - float(row_info.get("x", 0.0)) - inner_pad_x - 8.0),
                )
            except Exception:
                return False
    return True


def _remember_preview_legend_photo(owner, photo):
    refs = getattr(owner, "_preview_controls_legend_photo_refs", None)
    if not isinstance(refs, list):
        refs = []
        owner._preview_controls_legend_photo_refs = refs
    refs.append(photo)
    return photo


def _preview_controls_bbox_ready(owner, canvas) -> bool:
    def _bbox_ok(raw) -> bool:
        if not (isinstance(raw, tuple) and len(raw) == 4):
            return False
        try:
            x1, y1, x2, y2 = [float(value) for value in raw]
        except Exception:
            return False
        return x2 > x1 and y2 > y1

    if not _bbox_ok(getattr(owner, "_preview_controls_legend_grab_bbox", None)):
        return False
    if not _bbox_ok(getattr(owner, "_preview_controls_legend_toggle_bbox", None)):
        return False
    try:
        if not canvas.find_withtag("preview_legend_grab"):
            return False
        if not canvas.find_withtag("preview_legend_toggle"):
            return False
    except Exception:
        return False
    return True


def _bind_preview_controls_interactive_items(owner, canvas) -> None:
    try:
        for tag_name in ("preview_legend_grab", "preview_legend_toggle"):
            canvas.tag_bind(tag_name, "<ButtonPress-1>", owner._on_preview_controls_legend_press)
            canvas.tag_bind(tag_name, "<B1-Motion>", owner._on_preview_controls_legend_drag)
            canvas.tag_bind(tag_name, "<ButtonRelease-1>", owner._on_preview_controls_legend_release)
            canvas.tag_bind(tag_name, "<Motion>", owner._on_preview_controls_legend_motion)
            canvas.tag_bind(tag_name, "<Leave>", owner._on_preview_controls_legend_leave)
    except Exception:
        pass


def refresh_preview_controls_legend(owner):
    if not bool(getattr(owner, "_startup_ui_ready", False)):
        return
    if not bool(getattr(owner, "_preview_controls_legend_visible", True)):
        owner._hide_preview_controls_legend_overlay()
        return

    canvas = getattr(owner, "preview_controls_canvas", None)
    if canvas is None:
        return

    render_width_override = float(getattr(owner, "_preview_controls_legend_render_width_override", 0.0) or 0.0)
    render_height_override = float(getattr(owner, "_preview_controls_legend_render_height_override", 0.0) or 0.0)
    target_width = float(owner._get_preview_controls_legend_target_width())
    current_width = float(getattr(owner, "_preview_controls_legend_current_width", 0.0) or 0.0)
    if render_width_override:
        width = float(render_width_override)
    elif current_width > 0.0:
        width = float(current_width)
    else:
        width = float(target_width)
    owner._preview_controls_legend_current_width = float(width)
    legend_theme = owner._get_preview_legend_theme()
    plus_fill = legend_theme["muted_fill"]
    shell_outline = legend_theme["panel_outline"]
    try:
        canvas.configure(bg=legend_theme["panel_fill"])
        host_frame = getattr(owner, "preview_hint_frame", None)
        if host_frame is not None:
            host_frame.configure(
                bg=legend_theme["panel_fill"],
                highlightbackground=legend_theme["panel_outline"],
                highlightcolor=legend_theme["panel_outline"],
            )
        vbar = getattr(owner, "preview_controls_vbar", None)
        if vbar is not None and hasattr(vbar, "configure_style"):
            vbar.configure_style(
                track_color=legend_theme["panel_fill"],
                thumb_color=legend_theme["badge_plate_outline"],
                thumb_hover_color=legend_theme["shell_outline"],
            )
    except Exception:
        pass
    sections = owner._build_preview_legend_sections()
    legend_mode = "fullscreen" if bool(getattr(owner, "_preview_fullscreen_active", False)) else "inline"
    expanded = owner._is_preview_controls_legend_expanded()
    requested_scroll_offset = (
        float(getattr(owner, "_preview_controls_legend_scroll_offset", 0.0) or 0.0)
        if expanded
        else 0.0
    )
    context = owner._get_preview_legend_context()
    context_rows = owner._build_preview_controls_context_rows()
    static_key = _preview_controls_static_key(
        owner,
        legend_mode=legend_mode,
        expanded=expanded,
        width=width,
        scroll_offset=requested_scroll_offset,
        legend_theme=legend_theme,
        sections=sections,
        context_rows=context_rows,
    )
    legend_key = (
        static_key,
        str(context.get("filename", "")),
        str(context.get("image_text", "")),
        str(context.get("plate_text", "")),
        str(context.get("vehicle_text", "")),
        str(context.get("pool_text", "")),
    )
    bboxes_ready = _preview_controls_bbox_ready(owner, canvas)
    if legend_key == getattr(owner, "_preview_controls_legend_render_key", None) and bboxes_ready:
        _bind_preview_controls_interactive_items(owner, canvas)
        return
    if bboxes_ready and static_key == getattr(owner, "_preview_controls_legend_static_key", None):
        if _update_preview_controls_context_only(
            owner,
            canvas,
            context=context,
            context_rows=context_rows,
            legend_theme=legend_theme,
            shell_width=float(width),
        ):
            owner._preview_controls_legend_render_key = legend_key
            _bind_preview_controls_interactive_items(owner, canvas)
            return

    shell_width = width
    owner._preview_controls_legend_grab_bbox = None
    owner._preview_controls_legend_toggle_bbox = None
    owner._preview_controls_legend_context_item_ids = None
    owner._preview_controls_legend_photo_refs = []
    target_shell_height = float(owner._get_preview_controls_legend_target_height(width))
    current_shell_height = float(getattr(owner, "_preview_controls_legend_current_height", 0.0) or 0.0)
    shell_height = max(
        120.0,
        render_height_override or current_shell_height or target_shell_height,
    )
    if not expanded and not render_height_override:
        shell_height = target_shell_height
    shell_photo = owner._get_preview_legend_shell_photo(
        legend_theme,
        width=int(shell_width),
        height=int(shell_height),
        overlay_x=10,
        overlay_y=10,
    )
    _remember_preview_legend_photo(owner, shell_photo)
    canvas.delete("all")
    shell_id = canvas.create_image(
        0.0,
        0.0,
        image=shell_photo,
        anchor="nw",
        tags=("preview_legend", "preview_legend_shell"),
    )

    toggle_x = 12.0
    toggle_y = 10.0
    grab_x = max(8.0, shell_width - 30.0)
    grab_y = 10.0
    expand_x = max(toggle_x + 112.0, grab_x - 25.0)
    expand_y = grab_y
    _toggle_block_w, toggle_block_h = draw_preview_legend_compass_toggle(
        owner,
        canvas,
        toggle_x,
        toggle_y,
        expanded=expanded,
        theme=legend_theme,
        toggle_icon_x=expand_x,
        toggle_icon_y=expand_y,
        title_max_width=max(54.0, expand_x - (toggle_x + 48.0) - 8.0),
    )
    draw_preview_legend_grab_handle(
        owner,
        canvas,
        grab_x,
        grab_y,
        theme=legend_theme,
    )
    inner_pad_x = 12.0
    label_fill = str(legend_theme.get("label_fill", "#f4f7fb"))
    muted_fill = str(legend_theme.get("muted_fill", "#c7d0db"))
    value_fill = str(legend_theme.get("entry_text", "#ffffff"))
    current_y = toggle_y + toggle_block_h + (4.0 if expanded else 8.0)

    file_font, meta_font, value_font = _preview_controls_context_fonts(owner)

    file_text = f"Plik: {context['filename']}"
    context_item_ids = {"file": None, "rows": []}
    file_id = canvas.create_text(
        inner_pad_x,
        current_y,
        text=file_text,
        anchor="nw",
        fill=label_fill,
        width=max(120.0, shell_width - (inner_pad_x * 2.0)),
        justify=tk.LEFT,
        font=file_font,
        tags=("preview_legend", "preview_legend_context"),
    )
    context_item_ids["file"] = file_id
    file_bbox = canvas.bbox("preview_legend_context")
    current_y = float(file_bbox[3] + (6.0 if expanded else 10.0)) if file_bbox else current_y + 18.0

    row_h = 18.0
    compact_layout = _compact_preview_controls_layout(owner, width) if not expanded else None
    context_value_x = compact_layout["value_x"] if compact_layout else inner_pad_x + 64.0
    for idx, (row_label, row_value, row_fill, row_outline) in enumerate(context_rows):
        block_y = current_y
        canvas.create_text(
            inner_pad_x,
            block_y,
            text=str(row_label),
            anchor="nw",
            fill=muted_fill,
            font=meta_font,
            tags=("preview_legend", "preview_legend_context"),
        )
        if owner._preview_controls_row_is_multi_plate_badge(str(row_label), str(row_value)):
            badge_text = str(row_value)
            badge_w = max(40.0, float(value_font.measure(badge_text)) + 18.0)
            badge_h = float(value_font.metrics("linespace") + 6) if not expanded else 17.0
            badge_x = context_value_x
            badge_y = block_y - 1.0
            rect_id = canvas.create_rectangle(
                badge_x,
                badge_y,
                badge_x + badge_w,
                badge_y + badge_h,
                fill=str(row_fill),
                outline=str(row_outline),
                width=1,
                tags=("preview_legend", "preview_legend_context"),
            )
            text_id = canvas.create_text(
                badge_x + (badge_w / 2.0),
                badge_y + (badge_h / 2.0) - 0.5,
                text=badge_text,
                anchor="center",
                fill=owner._get_preview_legend_text_color(str(row_fill)),
                font=value_font,
                tags=("preview_legend", "preview_legend_context"),
            )
            try:
                canvas.tag_lower(rect_id, text_id)
            except Exception:
                pass
            context_item_ids["rows"].append(
                {
                    "label": str(row_label),
                    "kind": "badge",
                    "rect": rect_id,
                    "text": text_id,
                    "x": float(badge_x),
                    "y": float(badge_y),
                    "h": float(badge_h),
                }
            )
        else:
            value_id = canvas.create_text(
                context_value_x,
                block_y,
                text=str(row_value),
                anchor="nw",
                fill=value_fill,
                font=value_font,
                width=max(80.0, shell_width - context_value_x - inner_pad_x - 8.0),
                justify=tk.LEFT,
                tags=("preview_legend", "preview_legend_context"),
            )
            context_item_ids["rows"].append(
                {
                    "label": str(row_label),
                    "kind": "text",
                    "value": value_id,
                    "x": float(context_value_x),
                }
            )
        current_y += compact_layout["row_heights"][idx] if compact_layout else row_h


    if not expanded:
        try:
            final_height = int(math.ceil(shell_height))
            final_width = int(math.ceil(shell_width))
            canvas.configure(
                height=final_height,
                scrollregion=(0, 0, final_width, final_height),
            )
            canvas.yview_moveto(0.0)
            owner._preview_controls_legend_current_width = float(final_width)
            owner._preview_controls_legend_current_height = float(final_height)
            owner._preview_controls_legend_scroll_offset = 0.0
        except Exception:
            pass
        owner._preview_controls_legend_context_item_ids = context_item_ids
        try:
            canvas.tag_lower(shell_id)
        except Exception:
            pass
        owner._preview_controls_legend_rendered_scroll_offset = 0.0
        owner._preview_controls_legend_static_key = static_key
        owner._preview_controls_legend_render_key = legend_key
        try:
            canvas.tag_raise("preview_legend_toggle")
            canvas.tag_raise("preview_legend_grab")
        except Exception:
            pass
        _bind_preview_controls_interactive_items(owner, canvas)
        return

    separator_y = current_y + 6.0
    canvas.create_line(
        inner_pad_x,
        separator_y,
        shell_width - inner_pad_x - 8.0,
        separator_y,
        fill=blend_hex_colors(shell_outline, legend_theme["canvas_bg"], 0.30),
        width=1,
        tags=("preview_legend",),
    )

    outer_pad_x = 9.0
    outer_pad_y = separator_y + 8.0
    section_gap_y = 10.0
    token_gap = 8.0
    label_gap_x = 7.0
    title_font = owner._get_preview_legend_font(7, "bold")
    desc_font = owner._get_preview_legend_font(8, "normal")
    max_bottom = outer_pad_y
    current_y = outer_pad_y
    content_w = shell_width - (outer_pad_x * 2.0) - 2.0
    base_shortcut_cols = 2 if shell_width >= 316.0 else 1
    shortcut_col_gap = 12.0
    item_row_h = 31.0
    shortcut_layout = []
    shortcuts_total_h = 0.0
    scroll_tags = ("preview_legend", "preview_legend_scroll_content")
    for section in sections:
        shortcut_cols = max(1, min(base_shortcut_cols, int(section.get("columns", base_shortcut_cols) or base_shortcut_cols)))
        shortcut_col_w = max(136.0, (content_w - (shortcut_col_gap * (shortcut_cols - 1))) / float(shortcut_cols))
        section_rows = max(1, math.ceil(len(section.get("items", [])) / float(shortcut_cols)))
        title_h = float(title_font.metrics("linespace"))
        section_box_h = 10.0 + (section_rows * item_row_h) + 8.0
        shortcut_layout.append((section, shortcut_cols, shortcut_col_w, section_rows, title_h, section_box_h))
        shortcuts_total_h += title_h + 4.0 + section_box_h + section_gap_y

    estimated_total_height = max(float(shell_height), 160.0, outer_pad_y + shortcuts_total_h + 16.0)
    try:
        estimated_viewport_height = int(
            math.ceil(float(getattr(owner, "_preview_controls_legend_viewport_height", 0.0) or shell_height))
        )
    except Exception:
        estimated_viewport_height = int(math.ceil(shell_height))
    estimated_viewport_height = max(80, min(estimated_viewport_height, int(math.ceil(estimated_total_height))))
    max_scroll = max(0.0, float(estimated_total_height - estimated_viewport_height))
    scroll_offset = max(0.0, min(float(requested_scroll_offset), max_scroll))
    owner._preview_controls_legend_scroll_offset = float(scroll_offset)
    current_y = outer_pad_y - scroll_offset

    for section, shortcut_cols, shortcut_col_w, section_rows, title_h, section_box_h in shortcut_layout:
        section_box_x = outer_pad_x
        section_title_y = current_y
        section_box_y = section_title_y + title_h + 4.0
        canvas.create_text(
            section_box_x + 10.0,
            section_title_y,
            text=str(section.get("title", "")),
            fill=legend_theme["section_title"],
            anchor="nw",
            font=title_font,
            tags=scroll_tags,
        )
        section_shell = owner._get_preview_legend_group_shell_photo(
            legend_theme,
            width=int(content_w),
            height=int(section_box_h),
            outline=str(legend_theme.get("badge_plate_outline", "#2fbf71")),
        )
        _remember_preview_legend_photo(owner, section_shell)
        canvas.create_image(
            section_box_x,
            section_box_y,
            image=section_shell,
            anchor="nw",
            tags=scroll_tags,
        )

        row_base_y = section_box_y + 9.0

        for item_idx, item in enumerate(section.get("items", [])):
            local_col = item_idx % shortcut_cols
            local_row = item_idx // shortcut_cols
            item_x = section_box_x + 8.0 + (local_col * (shortcut_col_w + shortcut_col_gap))
            row_y = row_base_y + (local_row * item_row_h)
            tokens = [str(token) for token in item.get("tokens", [])]
            connector = str(item.get("connector", "") or "")
            label = str(item.get("label", "") or "")
            modes = [str(mode) for mode in item.get("modes", []) or []]
            token_x = item_x
            prev_right = None
            key_y = row_y + 6.0
            label_y = row_y + 9.0

            for token_idx, token_text in enumerate(tokens):
                if token_idx > 0 and connector:
                    connector_x = float(prev_right) + (token_gap / 2.0)
                    canvas.create_text(
                        connector_x,
                        label_y + 5.0,
                        text=connector,
                        fill=plus_fill,
                        anchor="center",
                        font=owner._get_preview_legend_font((8 if connector == "+" else 6), "bold"),
                        tags=scroll_tags,
                    )
                token_mode = modes[token_idx] if token_idx < len(modes) else str(item.get("mode", "") or "")
                token_w, _token_h = draw_preview_legend_keycap(
                    owner,
                    canvas,
                    token_x,
                    key_y,
                    token_text,
                    fill=legend_theme["token_fill"],
                    outline=str(section.get("accent", "#3498db")),
                    text_fill=legend_theme["token_text"],
                    interaction=token_mode,
                    tags=scroll_tags,
                )
                prev_right = token_x + float(token_w)
                if token_idx < (len(tokens) - 1):
                    token_x = prev_right + token_gap

            label_x = float(prev_right or token_x) + max(5.0, label_gap_x - 3.0)
            label_width = max(40.0, (item_x + shortcut_col_w) - label_x - 2.0)
            canvas.create_text(
                label_x,
                label_y,
                text=label,
                fill=str(legend_theme.get("section_title", "#eef4fb")),
                anchor="nw",
                width=label_width,
                font=desc_font,
                tags=scroll_tags,
            )

        current_y += title_h + 4.0 + section_box_h + section_gap_y
        max_bottom = max(max_bottom, current_y)

    total_height = max(float(shell_height), 160.0, estimated_total_height)
    if total_height > float(shell_height) + 1.0:
        shell_photo = owner._get_preview_legend_shell_photo(
            legend_theme,
            width=int(shell_width),
            height=int(total_height),
            overlay_x=10,
            overlay_y=10,
        )
        _remember_preview_legend_photo(owner, shell_photo)
        try:
            canvas.itemconfigure(shell_id, image=shell_photo)
        except Exception:
            pass
    try:
        content_height = int(math.ceil(total_height))
        final_width = int(math.ceil(shell_width))
        viewport_height = int(math.ceil(float(getattr(owner, "_preview_controls_legend_viewport_height", 0.0) or content_height)))
        viewport_height = max(80, min(viewport_height, content_height))
        canvas.configure(
            height=viewport_height,
            scrollregion=(0, 0, final_width, content_height),
        )
        max_scroll = max(0.0, float(content_height - viewport_height))
        scroll_offset = max(0.0, min(float(getattr(owner, "_preview_controls_legend_scroll_offset", 0.0) or 0.0), max_scroll))
        owner._preview_controls_legend_scroll_offset = float(scroll_offset)
        canvas.yview_moveto(0.0)
        if max_scroll > 0.0:
            header_mask_bottom = max(0.0, outer_pad_y)
            canvas.create_rectangle(
                0.0,
                0.0,
                final_width,
                header_mask_bottom,
                fill=legend_theme["panel_fill"],
                outline="",
                tags=("preview_legend", "preview_legend_header_mask"),
            )
            canvas.create_line(
                inner_pad_x,
                separator_y,
                shell_width - inner_pad_x - 8.0,
                separator_y,
                fill=blend_hex_colors(shell_outline, legend_theme["canvas_bg"], 0.30),
                width=1,
                tags=("preview_legend", "preview_legend_header_mask"),
            )
            try:
                vbar = getattr(owner, "preview_controls_vbar", None)
                if vbar is not None:
                    content_denominator = max(float(content_height), 1.0)
                    first = max(0.0, min(1.0, scroll_offset / content_denominator))
                    last = max(first, min(1.0, (scroll_offset + float(viewport_height)) / content_denominator))
                    vbar.set(first, last)
            except Exception:
                pass
        owner._preview_controls_legend_current_width = float(final_width)
        owner._preview_controls_legend_current_height = float(viewport_height)
        owner._preview_controls_legend_content_height = float(content_height)
        owner._preview_controls_legend_rendered_scroll_offset = float(scroll_offset)
        owner._preview_controls_legend_scroll_enabled = bool(
            expanded
            and content_height > viewport_height + 2
        )
    except Exception:
        pass
    canvas.tag_lower(shell_id)
    owner._preview_controls_legend_context_item_ids = context_item_ids
    owner._preview_controls_legend_static_key = static_key
    owner._preview_controls_legend_render_key = legend_key
    try:
        canvas.tag_raise("preview_legend_header_mask")
        canvas.tag_raise("preview_legend_context")
        canvas.tag_raise("preview_legend_toggle")
        canvas.tag_raise("preview_legend_grab")
    except Exception:
        pass
    _bind_preview_controls_interactive_items(owner, canvas)


def draw_preview_legend_interaction_marker(
    owner,
    canvas,
    x: float,
    y: float,
    width: float,
    *,
    interaction: str | None,
    fill: str,
    outline: str,
    text_fill: str,
    tags: tuple[str, ...] | None = None,
) -> None:
    mode = owner._normalize_preview_legend_interaction(interaction)
    if not mode:
        return
    item_tags = tuple(tags or ("preview_legend",))

    center_x = float(x) + (float(width) / 2.0)
    if mode == "tap":
        marker_w = 8.0
        marker_h = 8.0
        marker_x = center_x - (marker_w / 2.0)
        marker_y = float(y) - marker_h - 3.0
        photo = owner._get_preview_legend_interaction_marker_photo(
            mode,
            width=int(math.ceil(marker_w)),
            height=int(math.ceil(marker_h)),
            fill=fill,
            outline=outline,
            text_fill=text_fill,
        )
        _remember_preview_legend_photo(owner, photo)
        canvas.create_image(
            marker_x,
            marker_y,
            image=photo,
            anchor="nw",
            tags=item_tags,
        )
        return

    marker_w = int(math.ceil(min(22.0, max(14.0, float(width) - 4.0))))
    marker_h = 6
    marker_x = center_x - (float(marker_w) / 2.0)
    marker_y = float(y) - marker_h - 4.0
    photo = owner._get_preview_legend_interaction_marker_photo(
        mode,
        width=marker_w,
        height=marker_h,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )
    _remember_preview_legend_photo(owner, photo)
    canvas.create_image(
        marker_x,
        marker_y,
        image=photo,
        anchor="nw",
        tags=item_tags,
    )


def draw_preview_legend_keycap(
    owner,
    canvas,
    x: float,
    y: float,
    text: str,
    *,
    fill: str,
    outline: str,
    text_fill: str,
    interaction: str | None = None,
    tags: tuple[str, ...] | None = None,
):
    item_tags = tuple(tags or ("preview_legend",))
    photo, width, height = owner._get_preview_legend_keycap_photo(
        str(text),
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )
    _remember_preview_legend_photo(owner, photo)
    canvas.create_image(
        x,
        y,
        image=photo,
        anchor="nw",
        tags=item_tags,
    )
    draw_preview_legend_interaction_marker(
        owner,
        canvas,
        x,
        y,
        width,
        interaction=interaction,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
        tags=item_tags,
    )
    font_obj = owner._get_preview_legend_font(7, "bold")
    canvas.create_text(
        x + (width / 2.0),
        y + (height / 2.0) + 0.5,
        text=str(text),
        fill=text_fill,
        anchor="center",
        font=font_obj,
        tags=item_tags,
    )
    return float(width), float(height)


def draw_preview_legend_mousecap(
    owner,
    canvas,
    x: float,
    y: float,
    text: str,
    *,
    fill: str,
    outline: str,
    text_fill: str,
):
    # W Z2 nie udajemy ksztaltu myszy. LPM/PPM sa zwyklymi tokenami jak reszta skrotow.
    return draw_preview_legend_keycap(
        owner,
        canvas,
        x,
        y,
        text,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )


def draw_preview_legend_compass_toggle(
    owner,
    canvas,
    x: float,
    y: float,
    *,
    expanded: bool,
    theme: dict,
    toggle_icon_x: float | None = None,
    toggle_icon_y: float | None = None,
    title_max_width: float | None = None,
) -> tuple[float, float]:
    tags = ("preview_legend", "preview_legend_header", "preview_legend_toggle")
    toggle_tags = ("preview_legend", "preview_legend_toggle")
    size = 38.0
    text_fill = str(theme.get("token_text", "#f3f3f3"))
    compass_photo = owner._get_preview_legend_compass_photo(theme, size=int(size))
    _remember_preview_legend_photo(owner, compass_photo)
    canvas.create_image(
        x,
        y,
        image=compass_photo,
        anchor="nw",
        tags=tags,
    )
    if not bool(expanded):
        accent = str(theme.get("shell_outline", theme.get("badge_plate_outline", "#2fbf71")))
        canvas.create_rectangle(
            x - 2.0,
            y - 2.0,
            x + size + 2.0,
            y + size + 2.0,
            outline=accent,
            width=2,
            tags=tags,
        )
    state_text = "Zwiń skróty" if expanded else "Rozwiń skróty"
    title_x = x + size + 10.0
    title_font = owner._get_preview_legend_font(8 if expanded else 10, "bold")
    state_font = owner._get_preview_legend_font(7 if expanded else 9, "normal")
    safe_title_width = float(title_max_width or 96.0)
    title_text = owner._fit_preview_text_to_width("Skróty podglądu", safe_title_width, title_font)
    state_text = owner._fit_preview_text_to_width(state_text, safe_title_width, state_font)
    canvas.create_text(
        title_x,
        y + 5.0,
        text=title_text,
        anchor="nw",
        fill=theme.get("section_title", text_fill),
        font=title_font,
        tags=tags,
    )
    canvas.create_text(
        title_x,
        y + (19.0 if expanded else 5.0 + title_font.metrics("linespace") + 4.0),
        text=state_text,
        anchor="nw",
        fill=theme.get("section_muted", "#d8e2ee"),
        font=state_font,
        tags=tags,
    )
    icon_x = float(toggle_icon_x if toggle_icon_x is not None else x + 158.0)
    icon_y = float(toggle_icon_y if toggle_icon_y is not None else y + 1.0)
    icon_w = 20.0
    icon_h = 20.0
    fill = str(theme.get("entry_fill", "#3a3f46"))
    outline = str(theme.get("panel_outline", "#4b5563"))
    accent = str(theme.get("shell_outline", theme.get("badge_plate_outline", "#2fbf71")))
    canvas.create_rectangle(
        icon_x,
        icon_y,
        icon_x + icon_w,
        icon_y + icon_h,
        fill=fill,
        outline=accent if expanded else outline,
        width=1,
        tags=toggle_tags,
    )
    cx = icon_x + (icon_w / 2.0)
    cy = icon_y + (icon_h / 2.0)
    arrow_fill = str(theme.get("section_title", text_fill))
    if expanded:
        arrow_points = (
            cx - 5.0, cy - 2.0,
            cx, cy + 4.0,
            cx + 5.0, cy - 2.0,
        )
    else:
        arrow_points = (
            cx - 5.0, cy + 2.0,
            cx, cy - 4.0,
            cx + 5.0, cy + 2.0,
        )
    canvas.create_line(
        *arrow_points,
        fill=arrow_fill,
        width=2,
        capstyle=tk.ROUND,
        joinstyle=tk.ROUND,
        tags=toggle_tags,
    )
    header_h = size if expanded else max(size, 5 + title_font.metrics("linespace") + 4 + state_font.metrics("linespace"))
    owner._preview_controls_legend_toggle_bbox = (
        float(min(x, title_x, icon_x) - 5.0),
        float(min(y, icon_y) - 5.0),
        float(max(x + size, title_x + safe_title_width, icon_x + icon_w) + 5.0),
        float(max(y + header_h, icon_y + icon_h) + 5.0),
    )
    return 152.0, header_h


def draw_preview_legend_grab_handle(owner, canvas, x: float, y: float, *, theme: dict) -> tuple[float, float]:
    handle_w = 18.0
    handle_h = 20.0
    fill = str(theme.get("entry_fill", "#3a3f46"))
    outline = str(theme.get("panel_outline", "#4b5563"))
    dot_fill = str(theme.get("section_muted", "#d7e1ec"))
    canvas.create_rectangle(
        x,
        y,
        x + handle_w,
        y + handle_h,
        fill=fill,
        outline=outline,
        width=1,
        tags=("preview_legend", "preview_legend_grab"),
    )
    dot_r = 1.2
    dot_x = x + (handle_w / 2.0)
    for row_y in (y + 6.0, y + 10.0, y + 14.0):
        for col_dx in (-3.0, 3.0):
            canvas.create_oval(
                dot_x + col_dx - dot_r,
                row_y - dot_r,
                dot_x + col_dx + dot_r,
                row_y + dot_r,
                fill=dot_fill,
                outline="",
                tags=("preview_legend", "preview_legend_grab"),
            )
    owner._preview_controls_legend_grab_bbox = (float(x), float(y), float(x + handle_w), float(y + handle_h))
    return handle_w, handle_h
