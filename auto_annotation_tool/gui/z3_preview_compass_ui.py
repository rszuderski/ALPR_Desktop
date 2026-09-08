from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont
from typing import TYPE_CHECKING

from ..config import logger as _COMPASS_LOG
from .web_slim_scrollbar import blend_hex_colors

try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def _hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    try:
        return (
            int(raw[0:2], 16),
            int(raw[2:4], 16),
            int(raw[4:6], 16),
            int(alpha),
        )
    except Exception:
        return 32, 36, 42, int(alpha)


def _measure_preview_legend_token(token_text: str, font_obj) -> float:
    return max(24.0, float(font_obj.measure(str(token_text))) + 12.0)


def _normalize_preview_legend_interaction(interaction: str | None) -> str:
    value = str(interaction or "").strip().lower()
    if value in {"tap", "click", "single", "1x", "once"}:
        return "tap"
    if value in {"hold", "press", "held", "down"}:
        return "hold"
    return ""


def _preview_legend_image_cache(host: "CharacterAnnotationTab") -> dict:
    cache = getattr(host, "_preview_legend_image_cache", None)
    if cache is None:
        cache = {}
        host._preview_legend_image_cache = cache
    return cache


def get_preview_legend_theme(host: "CharacterAnnotationTab") -> dict:
    palette = getattr(host.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", panel)
    field = palette.get("field", panel)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent = palette.get("accent", "#0e639c")
    success = palette.get("success", "#2fbf71")
    warning = palette.get("warning", "#f59e0b")

    return {
        "canvas_bg": panel,
        "panel_fill": panel_alt,
        "panel_outline": border,
        "label_fill": palette.get("fg", "#f3f3f3"),
        "muted_fill": palette.get("muted_dim", palette.get("muted", "#8f98a3")),
        "entry_fill": blend_hex_colors(panel_alt, panel, 0.28),
        "entry_text": palette.get("fg", "#f3f3f3"),
        "section_fill": blend_hex_colors(panel_alt, panel, 0.18),
        "section_title": palette.get("fg", "#f3f3f3"),
        "section_muted": blend_hex_colors(palette.get("fg", "#f3f3f3"), panel_alt, 0.18),
        "token_fill": field,
        "token_text": palette.get("fg", "#f8fafc"),
        "badge_file_fill": palette.get("surface_info", blend_hex_colors(accent, panel_alt, 0.72)),
        "badge_file_outline": accent,
        "badge_image_fill": palette.get("surface_warning", blend_hex_colors(warning, panel_alt, 0.72)),
        "badge_image_outline": warning,
        "badge_plate_fill": palette.get("surface_success", blend_hex_colors(success, panel_alt, 0.72)),
        "badge_plate_outline": success,
    }


def get_preview_legend_compass_photo(host: "CharacterAnnotationTab", theme: dict, *, size: int = 38):
    if not PIL_AVAILABLE:
        return None
    cache = _preview_legend_image_cache(host)
    key = ("compass-thin-v2", int(size), str(theme.get("badge_plate_outline", "")), str(theme.get("token_fill", "")))
    photo = cache.get(key)
    if photo is not None:
        return photo

    scale = 6
    safe_size = max(18, int(size))
    px = int(safe_size * scale)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    accent = _hex_to_rgba(str(theme.get("badge_plate_outline", "#2fbf71")), 255)
    token_fill = _hex_to_rgba(str(theme.get("token_fill", "#3c3c3c")), 245)
    north_fill = _hex_to_rgba("#d64545", 255)
    south_fill = _hex_to_rgba("#183a63", 248)
    guide_fill = _hex_to_rgba("#d7dde4", 96)

    center = px / 2.0
    margin = max(1, int(2.0 * scale))
    outer = (margin, margin, px - margin - 1, px - margin - 1)
    inner_r = px * 0.31
    inner = (center - inner_r, center - inner_r, center + inner_r, center + inner_r)
    draw.ellipse(outer, outline=accent, width=max(1, int(scale * 0.65)))
    draw.ellipse(inner, fill=token_fill)
    draw.line((center, center - inner_r + (1.0 * scale), center, center + inner_r - (1.0 * scale)), fill=guide_fill, width=max(1, scale - 1))
    draw.line((center - inner_r + (1.0 * scale), center, center + inner_r - (1.0 * scale), center), fill=guide_fill, width=max(1, scale - 1))

    needle_len = px * 0.29
    needle_half = px * 0.075
    waist = px * 0.055
    north = [
        (center, center - needle_len),
        (center + needle_half, center + waist),
        (center, center - (waist * 1.05)),
        (center - needle_half, center + waist),
    ]
    south = [
        (center, center + needle_len),
        (center + needle_half, center - waist),
        (center, center + (waist * 1.05)),
        (center - needle_half, center - waist),
    ]
    draw.polygon(north, fill=north_fill)
    draw.polygon(south, fill=south_fill)
    dot_r = px * 0.047
    draw.ellipse((center - dot_r, center - dot_r, center + dot_r, center + dot_r), fill=_hex_to_rgba("#f3f3f3", 255))

    img = img.resize((safe_size, safe_size), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img, master=host.frame)
    cache[key] = photo
    return photo


def get_preview_legend_interaction_marker_photo(
    host: "CharacterAnnotationTab",
    mode: str,
    *,
    width: int,
    height: int,
    fill: str,
    outline: str,
    text_fill: str,
):
    if not PIL_AVAILABLE:
        return None
    cache = _preview_legend_image_cache(host)
    mode_value = _normalize_preview_legend_interaction(mode)
    safe_w = max(6, int(width))
    safe_h = max(6, int(height))
    key = (
        "interaction_marker",
        mode_value,
        safe_w,
        safe_h,
        str(fill),
        str(outline),
        str(text_fill),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    scale = 6
    px_w = safe_w * scale
    px_h = safe_h * scale
    img = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    marker_fill = _hex_to_rgba(outline, 238)
    if mode_value == "tap":
        margin = max(1, int(1.2 * scale))
        draw.ellipse(
            (margin, margin, px_w - margin - 1, px_h - margin - 1),
            fill=marker_fill,
        )
    else:
        line_h = max(2, int(2.6 * scale))
        y1 = max(1, int((px_h - line_h) / 2))
        y2 = min(px_h - 1, y1 + line_h)
        x1 = max(1, int(1.2 * scale))
        x2 = min(px_w - 1, px_w - x1)
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=max(1, int(line_h / 2)),
            fill=marker_fill,
        )

    photo = ImageTk.PhotoImage(img.resize((safe_w, safe_h), Image.Resampling.LANCZOS), master=host.frame)
    cache[key] = photo
    while len(cache) > 64:
        try:
            cache.pop(next(iter(cache)), None)
        except Exception:
            break
    return photo


def draw_preview_legend_interaction_marker(
    host: "CharacterAnnotationTab",
    canvas,
    x: float,
    y: float,
    width: float,
    *,
    interaction: str | None,
    fill: str,
    outline: str,
    text_fill: str,
) -> None:
    mode = _normalize_preview_legend_interaction(interaction)
    if not mode:
        return

    center_x = float(x) + (float(width) / 2.0)
    if mode == "tap":
        marker_w = 8.0
        marker_h = 8.0
        marker_x = center_x - (marker_w / 2.0)
        marker_y = float(y) - marker_h - 3.0
        photo = get_preview_legend_interaction_marker_photo(
            host,
            mode,
            width=int(math.ceil(marker_w)),
            height=int(math.ceil(marker_h)),
            fill=fill,
            outline=outline,
            text_fill=text_fill,
        )
        if photo is not None:
            canvas.create_image(
                marker_x,
                marker_y,
                image=photo,
                anchor="nw",
                tags=("preview_legend",),
            )
        else:
            dot_r = marker_w / 2.0
            canvas.create_oval(
                marker_x,
                marker_y,
                marker_x + (dot_r * 2.0),
                marker_y + (dot_r * 2.0),
                fill=outline,
                outline="",
                width=0,
                tags=("preview_legend",),
            )
        return

    marker_w = int(math.ceil(min(22.0, max(14.0, float(width) - 4.0))))
    marker_h = 6
    marker_x = center_x - (float(marker_w) / 2.0)
    marker_y = float(y) - marker_h - 4.0
    photo = get_preview_legend_interaction_marker_photo(
        host,
        mode,
        width=marker_w,
        height=marker_h,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )
    if photo is not None:
        canvas.create_image(
            marker_x,
            marker_y,
            image=photo,
            anchor="nw",
            tags=("preview_legend",),
        )
        return

    canvas.create_rectangle(
        marker_x,
        marker_y + 2.0,
        marker_x + marker_w,
        marker_y + 4.0,
        fill=str(outline),
        outline="",
        width=0,
        tags=("preview_legend",),
    )


def draw_preview_legend_keycap(
    host: "CharacterAnnotationTab",
    canvas,
    x: float,
    y: float,
    text: str,
    *,
    fill: str,
    outline: str,
    text_fill: str,
    interaction: str | None = None,
):
    font_obj = host._get_preview_legend_font(8, "bold")
    width = _measure_preview_legend_token(text, font_obj)
    height = 21.0
    draw_preview_legend_interaction_marker(
        host,
        canvas,
        x,
        y,
        width,
        interaction=interaction,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )
    canvas.create_rectangle(
        x,
        y,
        x + width,
        y + height,
        fill=fill,
        outline=outline,
        width=1,
        tags=("preview_legend",),
    )
    canvas.create_line(
        x + 1,
        y + 1,
        x + width - 1,
        y + 1,
        fill="#ffffff",
        width=1,
        tags=("preview_legend",),
    )
    canvas.create_text(
        x + (width / 2.0),
        y + (height / 2.0) + 0.5,
        text=str(text),
        fill=text_fill,
        anchor="center",
        font=font_obj,
        tags=("preview_legend",),
    )
    return float(width), float(height)


def draw_preview_legend_compass_toggle(
    host: "CharacterAnnotationTab",
    canvas,
    x: float,
    y: float,
    *,
    expanded: bool,
    theme: dict,
    width: float = 360.0,
) -> tuple[float, float]:
    tags = ("preview_legend", "preview_legend_toggle")
    compact = not bool(expanded)
    size = 34.0 if width >= 300 else 28.0
    text_fill = str(theme.get("token_text", "#f3f3f3"))
    compass_photo = get_preview_legend_compass_photo(host, theme, size=int(size))
    if compass_photo is not None:
        canvas.create_image(
            x,
            y,
            image=compass_photo,
            anchor="nw",
            tags=tags,
        )
    else:
        canvas.create_oval(x, y, x + size, y + size, fill=theme.get("token_fill", "#3c3c3c"), outline=theme.get("badge_plate_outline", "#2fbf71"), width=1, tags=tags)
        canvas.create_text(x + (size / 2.0), y + (size / 2.0), text="◎", fill=text_fill, font=host._get_preview_legend_font(12, "bold"), tags=tags)
    state_text = "Zwiń ▴" if expanded else "Rozwiń ▾"
    title_x = x + size + 10.0
    canvas.create_text(
        title_x,
        y + 7.0,
        text="Kompas podglądu" if width >= 340 else "Kompas",
        anchor="nw",
        fill=theme.get("section_title", text_fill),
        font=host._get_preview_legend_font(10 if width >= 340 else 9, "bold"),
        tags=tags,
    )
    button_w = 70.0
    button_x = width - button_w - (42 if expanded else 12)
    button_tags = tags + ("preview_legend_action",)
    canvas.create_rectangle(button_x, y + 3, button_x + button_w, y + 30,
                            fill=theme["entry_fill"], outline=theme["panel_outline"], tags=button_tags)
    canvas.create_text(
        button_x + button_w / 2,
        y + 16,
        text=state_text,
        anchor="center",
        fill=theme.get("section_muted", "#d8e2ee"),
        font=host._get_preview_legend_font(9, "bold"),
        tags=button_tags,
    )
    return 206.0, size


def draw_preview_legend_grab_handle(host: "CharacterAnnotationTab", canvas, x: float, y: float, *, theme: dict) -> tuple[float, float]:
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
    center_x = x + (handle_w / 2.0)
    start_y = y + 5.0
    gap_y = 4.0
    gap_x = 4.0
    dot_r = 1.15
    for row_idx in range(3):
        for col_idx in (-1, 1):
            cx = center_x + (col_idx * gap_x / 2.0)
            cy = start_y + (row_idx * gap_y)
            canvas.create_oval(
                cx - dot_r,
                cy - dot_r,
                cx + dot_r,
                cy + dot_r,
                fill=dot_fill,
                outline=dot_fill,
                tags=("preview_legend", "preview_legend_grab"),
            )
    host._preview_controls_legend_grab_bbox = (float(x), float(y), float(x + handle_w), float(y + handle_h))
    return handle_w, handle_h


def clamp_preview_controls_legend_offsets(
    host: "CharacterAnnotationTab",
    offset_x: float,
    offset_y: float,
    *,
    width: float | None = None,
    height: float | None = None,
) -> tuple[float, float]:
    host_widget = getattr(host, "preview_canvas_host", None)
    try:
        frame_w = float(host_widget.winfo_width() or host_widget.winfo_reqwidth() or 0) if host_widget is not None else 0.0
        frame_h = float(host_widget.winfo_height() or host_widget.winfo_reqheight() or 0) if host_widget is not None else 0.0
    except Exception:
        frame_w = 0.0
        frame_h = 0.0
    top_clearance = 8.0
    if bool(getattr(host, "_preview_fullscreen_active", False)) or bool(getattr(host, "_preview_render_state", None)):
        top_clearance = max(38.0, float(getattr(host, "_preview_overlay_top_bar_height", 38.0) or 38.0)) + 8.0
    safe_w = float(width or getattr(host, "_preview_controls_legend_current_width", 0.0) or 280.0)
    safe_h = float(height or getattr(host, "_preview_controls_legend_current_height", 0.0) or 120.0)
    clamped_x = min(max(0.0, float(offset_x)), max(0.0, frame_w - safe_w))
    max_y = max(0.0, frame_h - safe_h)
    min_y = min(max_y, max(0.0, top_clearance))
    clamped_y = min(max(min_y, float(offset_y)), max_y)
    return float(clamped_x), float(clamped_y)


def toggle_preview_controls_legend(host: "CharacterAnnotationTab", event=None):
    host._preview_controls_legend_drag_state = None
    host._preview_controls_legend_click_state = None
    was_expanded = bool(host._is_preview_controls_legend_expanded())
    collapsed_anchor = None
    if not was_expanded:
        collapsed_anchor = {
            "x": float(getattr(host, "_preview_controls_legend_offset_x", 10.0)),
            "y": float(getattr(host, "_preview_controls_legend_offset_y", 10.0)),
            "manual": bool(getattr(host, "_preview_controls_legend_position_manual", False)),
        }
        host._preview_controls_legend_last_collapsed_offset = dict(collapsed_anchor)
    else:
        stored_anchor = getattr(host, "_preview_controls_legend_last_collapsed_offset", None)
        if isinstance(stored_anchor, dict):
            collapsed_anchor = dict(stored_anchor)

    host._preview_controls_legend_expanded = not was_expanded
    host._preview_controls_legend_position_manual = False
    if was_expanded and isinstance(collapsed_anchor, dict):
        host._preview_controls_legend_offset_x = float(collapsed_anchor.get("x", 10.0))
        host._preview_controls_legend_offset_y = float(collapsed_anchor.get("y", 10.0))
        host._preview_controls_legend_position_manual = bool(collapsed_anchor.get("manual", False))
    place_preview_hint_overlay(host, refresh=True)
    try:
        _COMPASS_LOG.info(
            "[Z3 COMPASS] toggle expanded %s->%s fullscreen=%s visible=%s",
            int(was_expanded),
            int(bool(host._is_preview_controls_legend_expanded())),
            int(bool(getattr(host, "_preview_fullscreen_active", False))),
            int(bool(getattr(host, "_preview_controls_legend_visible", True))),
        )
    except Exception:
        pass
    return "break"


def update_preview_controls_legend_cursor(
    host: "CharacterAnnotationTab",
    local_x: float | None = None,
    local_y: float | None = None,
):
    cursor = "hand2"
    if isinstance(getattr(host, "_preview_controls_legend_drag_state", None), dict):
        cursor = "fleur"
    elif local_x is not None and local_y is not None and host._is_preview_controls_legend_grab_hit(local_x, local_y):
        cursor = "fleur"
    for widget_name in ("preview_hint_frame", "preview_controls_canvas"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(cursor=cursor)
        except Exception:
            pass


def _preview_controls_legend_event_xy(host: "CharacterAnnotationTab", event=None) -> tuple[float, float]:
    try:
        local_x = float(getattr(event, "x", 0.0) or 0.0)
        local_y = float(getattr(event, "y", 0.0) or 0.0)
    except Exception:
        return 0.0, 0.0
    canvas = getattr(host, "preview_controls_canvas", None)
    if canvas is not None and getattr(event, "widget", None) is canvas:
        try:
            local_x = float(canvas.canvasx(local_x))
            local_y = float(canvas.canvasy(local_y))
        except Exception:
            pass
    return local_x, local_y


def _preview_controls_legend_compact_hit(host: "CharacterAnnotationTab", local_x: float, local_y: float) -> bool:
    if bool(host._is_preview_controls_legend_expanded()):
        return False
    try:
        width = float(getattr(host, "_preview_controls_legend_current_width", 32.0) or 32.0)
        height = float(getattr(host, "_preview_controls_legend_current_height", 32.0) or 32.0)
    except Exception:
        width = height = 32.0
    return 0.0 <= float(local_x) <= max(32.0, width) and 0.0 <= float(local_y) <= max(32.0, height)


def on_preview_controls_legend_press(host: "CharacterAnnotationTab", event=None):
    if event is None:
        return None
    try:
        local_x, local_y = _preview_controls_legend_event_xy(host, event)
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        return "break"
    if host._is_preview_controls_legend_grab_hit(local_x, local_y):
        try:
            _COMPASS_LOG.info("[Z3 COMPASS] press grab x=%.1f y=%.1f", local_x, local_y)
        except Exception:
            pass
        host._preview_controls_legend_drag_state = {
            "press_root_x": root_x,
            "press_root_y": root_y,
            "start_x": float(getattr(host, "_preview_controls_legend_offset_x", 10.0)),
            "start_y": float(getattr(host, "_preview_controls_legend_offset_y", 10.0)),
        }
        host._preview_controls_legend_click_state = None
        update_preview_controls_legend_cursor(host, local_x, local_y)
        return "break"
    host._preview_controls_legend_drag_state = None
    host._preview_controls_legend_click_state = {
        "press_root_x": root_x,
        "press_root_y": root_y,
        "start_x": float(getattr(host, "_preview_controls_legend_offset_x", 10.0)),
        "start_y": float(getattr(host, "_preview_controls_legend_offset_y", 10.0)),
        "compact_drag": bool(_preview_controls_legend_compact_hit(host, local_x, local_y)),
        "toggle_hit": bool(canvas_toggle_hit(host, local_x, local_y)),
    }
    try:
        _COMPASS_LOG.info(
            "[Z3 COMPASS] press click x=%.1f y=%.1f compact_drag=%s expanded=%s",
            local_x,
            local_y,
            int(bool(host._preview_controls_legend_click_state.get("compact_drag"))),
            int(bool(host._is_preview_controls_legend_expanded())),
        )
    except Exception:
        pass
    return "break"


def on_preview_controls_legend_drag(host: "CharacterAnnotationTab", event=None):
    drag_state = getattr(host, "_preview_controls_legend_drag_state", None)
    click_state = getattr(host, "_preview_controls_legend_click_state", None)
    if event is None:
        return None
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        return "break"
    if not isinstance(drag_state, dict):
        if not (isinstance(click_state, dict) and bool(click_state.get("compact_drag"))):
            return None
        move_x = root_x - float(click_state.get("press_root_x", root_x))
        move_y = root_y - float(click_state.get("press_root_y", root_y))
        if abs(move_x) <= 4.0 and abs(move_y) <= 4.0:
            return "break"
        drag_state = {
            "press_root_x": float(click_state.get("press_root_x", root_x)),
            "press_root_y": float(click_state.get("press_root_y", root_y)),
            "start_x": float(click_state.get("start_x", getattr(host, "_preview_controls_legend_offset_x", 10.0) or 10.0)),
            "start_y": float(click_state.get("start_y", getattr(host, "_preview_controls_legend_offset_y", 10.0) or 10.0)),
        }
        host._preview_controls_legend_drag_state = drag_state
        host._preview_controls_legend_click_state = None
    next_x = float(drag_state.get("start_x", 10.0)) + (root_x - float(drag_state.get("press_root_x", root_x)))
    next_y = float(drag_state.get("start_y", 10.0)) + (root_y - float(drag_state.get("press_root_y", root_y)))
    clamped_x, clamped_y = clamp_preview_controls_legend_offsets(host, next_x, next_y)
    host._preview_controls_legend_offset_x = clamped_x
    host._preview_controls_legend_offset_y = clamped_y
    host._preview_controls_legend_position_manual = True
    place_preview_hint_overlay(host, refresh=False)
    return "break"


def on_preview_controls_legend_motion(host: "CharacterAnnotationTab", event=None):
    if event is None:
        return None
    try:
        local_x, local_y = _preview_controls_legend_event_xy(host, event)
        update_preview_controls_legend_cursor(
            host,
            local_x,
            local_y,
        )
    except Exception:
        update_preview_controls_legend_cursor(host)
    return None


def on_preview_controls_legend_leave(host: "CharacterAnnotationTab", event=None):
    for widget_name in ("preview_hint_frame", "preview_controls_canvas"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(cursor="arrow")
        except Exception:
            pass
    return None


def on_preview_controls_legend_release(host: "CharacterAnnotationTab", event=None):
    drag_state = getattr(host, "_preview_controls_legend_drag_state", None)
    click_state = getattr(host, "_preview_controls_legend_click_state", None)
    host._preview_controls_legend_drag_state = None
    host._preview_controls_legend_click_state = None
    update_preview_controls_legend_cursor(host)
    if isinstance(drag_state, dict):
        return "break"
    if not isinstance(click_state, dict) or event is None:
        return None
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        return "break"
    if abs(root_x - float(click_state.get("press_root_x", root_x))) > 4.0 or abs(root_y - float(click_state.get("press_root_y", root_y))) > 4.0:
        return "break"
    local_x, local_y = _preview_controls_legend_event_xy(host, event)
    if not (click_state.get("compact_drag") or click_state.get("toggle_hit")
            or canvas_toggle_hit(host, local_x, local_y)):
        return "break"
    try:
        _COMPASS_LOG.info("[Z3 COMPASS] release click -> toggle")
    except Exception:
        pass
    return toggle_preview_controls_legend(host, event)


def on_preview_controls_legend_mousewheel(host: "CharacterAnnotationTab", event=None):
    canvas = getattr(host, "preview_controls_canvas", None)
    if canvas is None or event is None:
        return None
    try:
        content_h = float(getattr(host, "_preview_controls_legend_content_height", 0.0) or 0.0)
        visible_h = float(getattr(host, "_preview_controls_legend_current_height", 0.0) or 0.0)
        if content_h <= visible_h + 1.0:
            return None
    except Exception:
        return None

    delta_units = 0
    try:
        if hasattr(event, "delta") and int(getattr(event, "delta", 0) or 0) != 0:
            delta_units = -1 if int(event.delta) > 0 else 1
        else:
            num = int(getattr(event, "num", 0) or 0)
            if num == 4:
                delta_units = -1
            elif num == 5:
                delta_units = 1
    except Exception:
        delta_units = 0
    if delta_units == 0:
        return None
    try:
        canvas.yview_scroll(delta_units, "units")
        return "break"
    except Exception:
        return None


def build_preview_legend_sections(host: "CharacterAnnotationTab"):
    return [
        {
            "title": "Nawigacja",
            "accent": "#2f80ed",
            "items": [
                {"tokens": ["Q", "E"], "connector": "/", "modes": ["tap", "tap"], "label": "poprz./nast. tablica"},
                {"tokens": ["F"], "modes": ["tap"], "label": "tablica do okna"},
                {"tokens": ["Rolka"], "label": "zoom in / out"},
                {"tokens": ["Enter"], "modes": ["tap"], "label": "pełny ekran / wyjście"},
            ],
        },
        {
            "title": "Boxy",
            "accent": "#22c55e",
            "items": [
                {"tokens": ["D", "LPM"], "connector": "→", "modes": ["tap", "tap"], "label": "nowy box klik-klik"},
                {"tokens": ["S"], "modes": ["tap"], "label": "select / off"},
                {"tokens": ["LPM"], "modes": ["hold"], "label": "przesuń lub resize"},
                {"tokens": ["PPM"], "modes": ["tap"], "label": "usuń aktywny"},
                {"tokens": ["Ctrl+Z", "Ctrl+Y"], "connector": "/", "modes": ["tap", "tap"], "label": "historia"},
            ],
        },
        {
            "title": "Znaki",
            "accent": "#f59e0b",
            "items": [
                {"tokens": ["Alt+W"], "modes": ["tap"], "label": "tryb wpisywania"},
                {"tokens": ["LPM"], "modes": ["tap"], "label": "wybierz pole"},
                {"tokens": ["←", "→"], "connector": "/", "modes": ["tap", "tap"], "label": "pole +/-"},
                {"tokens": ["0-9/A-Z"], "modes": ["tap"], "label": "wpisz znak"},
                {"tokens": ["Esc"], "modes": ["tap"], "label": "wyjdz z wpisywania"},
            ],
        },
    ]


def refresh_preview_controls_legend(host: "CharacterAnnotationTab"):
    place_preview_hint_overlay(host, refresh=True)


def canvas_toggle_hit(host, x: float, y: float) -> bool:
    canvas = getattr(host, "preview_controls_canvas", None)
    bounds = canvas.bbox("preview_legend_toggle") if canvas is not None else None
    return bool(bounds and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3])


def _sync_preview_compass_header(host):
    canvas = host.preview_controls_canvas
    offset = float(canvas.canvasy(0))
    previous = float(getattr(host, "_preview_compass_header_offset", 0.0))
    if offset != previous:
        canvas.move("preview_legend_header", 0, offset - previous)
    host._preview_compass_header_offset = offset
    canvas.tag_raise("preview_legend_header")
    host._preview_controls_legend_grab_bbox = canvas.bbox("preview_legend_grab")


def _draw_preview_compass_context(host, canvas, width: float, y: float, theme: dict) -> float:
    """Collapsed compass keeps the current file and navigation context, as in Z2."""
    context = host._get_preview_legend_context()
    tags = ("preview_legend", "preview_legend_context")
    item = canvas.create_text(14, y, text=str(context.get("filename") or "Brak obrazu"),
                             anchor="nw", width=max(1, width - 28), fill=theme["entry_text"],
                             font=host._get_preview_legend_font(11, "bold"), tags=tags)
    bounds = canvas.bbox(item)
    y = float(bounds[3] + 10) if bounds else y + 26
    cell_w = (width - 36) / 2
    for index, (title, value) in enumerate((("TABLICE", context.get("image_text", "0/0")),
                                           ("BOXY", context.get("plate_text", "0")))):
        x = 14 + index * (cell_w + 8)
        value = str(value).split(":", 1)[-1].strip()
        canvas.create_rectangle(x, y, x + cell_w, y + 56, fill=theme["entry_fill"],
                                outline=theme["panel_outline"], tags=tags)
        canvas.create_text(x + 9, y + 7, text=title, anchor="nw", fill=theme["section_muted"],
                           font=host._get_preview_legend_font(8, "bold"), tags=tags)
        canvas.create_text(x + 9, y + 23, text=value, anchor="nw", fill=theme["entry_text"],
                           font=host._get_preview_legend_font(15 if width >= 300 else 12, "bold"),
                           tags=tags + ("preview_legend_counter",))
    y += 66
    canvas.create_text(14, y, text=str(context.get("vehicle_text") or ""), anchor="nw",
                       fill=theme["section_muted"], font=host._get_preview_legend_font(9, "normal"), tags=tags)
    return y + 24


def _draw_preview_controls_legend(host: "CharacterAnnotationTab", width: float) -> None:
    canvas = getattr(host, "preview_controls_canvas", None)
    if canvas is None:
        return

    expanded = host._is_preview_controls_legend_expanded()
    compact = not bool(expanded)
    canvas.delete("all")
    legend_theme = get_preview_legend_theme(host)
    bg_fill = legend_theme["panel_fill"]
    bg_outline = legend_theme["panel_outline"]
    plus_fill = legend_theme["muted_fill"]
    try:
        canvas.configure(bg=legend_theme["canvas_bg"])
    except Exception:
        pass

    host._preview_controls_legend_grab_bbox = None
    background_height = 72.0
    if compact:
        bg_outline = str(legend_theme.get("badge_plate_outline", "#2fbf71"))
    background_id = canvas.create_rectangle(
        2 if compact else 1,
        2 if compact else 1,
        width - (2 if compact else 2),
        background_height - (2 if compact else 0),
        fill=bg_fill,
        outline=bg_outline,
        width=1,
        tags=("preview_legend",)
    )
    toggle_x = 12.0
    toggle_y = 8.0
    toggle_w, toggle_h = draw_preview_legend_compass_toggle(
        host,
        canvas,
        toggle_x,
        toggle_y,
        expanded=expanded,
        theme=legend_theme,
        width=width,
    )
    if not compact:
        draw_preview_legend_grab_handle(
            host,
            canvas,
            max(12.0, width - 28.0),
            10.0,
            theme=legend_theme,
        )
    current_y = _draw_preview_compass_context(host, canvas, width, toggle_y + toggle_h + 10.0, legend_theme)
    header_height = current_y

    if expanded:
        sections = build_preview_legend_sections(host)
        outer_pad_x = 14.0
        outer_pad_y = current_y + 4.0
        section_gap_x = 10.0
        section_gap_y = 8.0
        section_pad_x = 10.0
        section_pad_y = 8.0
        title_block_y = 15.0
        item_gap_y = 5.0
        token_gap = 12.0
        label_gap_x = 8.0
        item_row_height = 32.0
        title_font = host._get_preview_legend_font(8, "bold")
        desc_font = host._get_preview_legend_font(8, "normal")

        if width >= 920.0:
            column_count = 3
        elif width >= 600.0:
            column_count = 2
        else:
            column_count = 1

        column_count = max(1, min(column_count, len(sections)))
        section_width = max(
            180.0,
            (width - (outer_pad_x * 2.0) - (section_gap_x * (column_count - 1))) / float(column_count),
        )

        section_heights = []
        for section in sections:
            item_count = len(section.get("items", []))
            section_height = (
                title_block_y
                + section_pad_y
                + max(1, item_count) * item_row_height
                + max(0, item_count - 1) * item_gap_y
                + section_pad_y
            )
            section_heights.append(section_height)

        row_heights = []
        for start_idx in range(0, len(sections), column_count):
            row_heights.append(max(section_heights[start_idx:start_idx + column_count]))

        y_offsets = []
        y_cursor = outer_pad_y
        for row_height in row_heights:
            y_offsets.append(y_cursor)
            y_cursor += row_height + section_gap_y

        max_bottom = current_y
        for idx, section in enumerate(sections):
            row_idx = idx // column_count
            col_idx = idx % column_count
            section_x = outer_pad_x + (col_idx * (section_width + section_gap_x))
            section_y = y_offsets[row_idx]
            section_height = section_heights[idx]
            section_body_y = section_y + title_block_y
            section_body_h = section_height - title_block_y
            accent = str(section.get("accent", "#3498db"))

            canvas.create_text(
                section_x + section_pad_x,
                section_y,
                text=str(section.get("title", "")),
                fill=legend_theme["section_title"],
                anchor="nw",
                font=title_font,
                tags=("preview_legend",)
            )
            section_rect = canvas.create_rectangle(
                section_x,
                section_body_y,
                section_x + section_width,
                section_body_y + section_body_h,
                fill=legend_theme["section_fill"],
                outline=accent,
                width=1,
                tags=("preview_legend",)
            )
            canvas.tag_lower(section_rect)

            item_y = section_body_y + section_pad_y
            for item in section.get("items", []):
                tokens = [str(token) for token in item.get("tokens", [])]
                connector = str(item.get("connector", "") or "")
                label = str(item.get("label", "") or "")
                modes = [str(mode) for mode in item.get("modes", [])]
                token_x = section_x + section_pad_x
                prev_right = None
                key_y = item_y + 7.0
                label_y = item_y + 17.0

                for token_idx, token_text in enumerate(tokens):
                    if token_idx > 0 and connector:
                        connector_x = float(prev_right) + (token_gap / 2.0)
                        canvas.create_text(
                            connector_x,
                            label_y,
                            text=connector,
                            fill=plus_fill,
                            anchor="center",
                            font=host._get_preview_legend_font(7, "bold"),
                            tags=("preview_legend",)
                        )
                    token_mode = modes[token_idx] if token_idx < len(modes) else str(item.get("mode", "") or "")
                    token_w, _token_h = draw_preview_legend_keycap(
                        host,
                        canvas,
                        token_x,
                        key_y,
                        token_text,
                        fill=legend_theme["token_fill"],
                        outline=accent,
                        text_fill=legend_theme["token_text"],
                        interaction=token_mode,
                    )
                    prev_right = token_x + float(token_w)
                    if token_idx < (len(tokens) - 1):
                        token_x = prev_right + token_gap

                label_x = min(
                    section_x + section_width - section_pad_x - 56.0,
                    max(section_x + section_pad_x + 88.0, float(prev_right or token_x) + label_gap_x),
                )
                canvas.create_text(
                    label_x,
                    label_y,
                    text=label,
                    fill=legend_theme["section_muted"],
                    anchor="w",
                    width=max(48.0, (section_x + section_width - section_pad_x) - label_x),
                    font=desc_font,
                    tags=("preview_legend",)
                )
                item_y += item_row_height + item_gap_y

            max_bottom = max(max_bottom, section_y + section_height)

        total_height = max(92.0, max_bottom + 12.0)
    else:
        total_height = max(56.0, current_y + 4.0)
    # Keep the close toggle and grab available while the shortcuts scroll.
    header_background = canvas.create_rectangle(2, 2, width - 2, header_height, fill=bg_fill,
                                                outline="", tags=("preview_legend", "preview_legend_header"))
    canvas.tag_lower(header_background, "preview_legend_toggle")
    for tag in ("preview_legend_toggle", "preview_legend_grab", "preview_legend_context"):
        canvas.addtag_withtag("preview_legend_header", tag)
    host._preview_compass_header_offset = 0.0
    canvas.coords(
        background_id,
        2 if compact else 1,
        2 if compact else 1,
        width - (3 if compact else 2),
        total_height - (3 if compact else 2),
    )
    try:
        canvas.configure(scrollregion=(0, 0, int(max(1.0, width)), int(max(1.0, total_height))))
    except Exception:
        pass
    host._preview_controls_legend_content_height = float(total_height)
    _sync_preview_compass_header(host)

def place_preview_hint_overlay(host: "CharacterAnnotationTab", refresh: bool = False):
    host_widget = getattr(host, "preview_canvas_host", None)
    overlay = getattr(host, "preview_hint_frame", None)
    canvas = getattr(host, "preview_controls_canvas", None)
    vbar = getattr(host, "preview_controls_vbar", None)
    if host_widget is None or overlay is None or canvas is None:
        return
    if getattr(host, "_preview_compass_scroll_canvas", None) is not canvas:
        def on_scroll(first, last):
            if vbar is not None:
                vbar.set(first, last)
            _sync_preview_compass_header(host)
        canvas.configure(yscrollcommand=on_scroll)
        host._preview_compass_scroll_canvas = canvas
    if not bool(getattr(host, "_preview_controls_legend_visible", True)):
        overlay.place_forget()
        host._preview_controls_legend_current_bounds = None
        host._preview_controls_legend_drag_state = None
        host._preview_controls_legend_click_state = None
        return

    host_width = max(0, int(host_widget.winfo_width() or 0))
    host_height = max(0, int(host_widget.winfo_height() or 0))
    if host_width <= 1 or host_height <= 1:
        return

    expanded = host._is_preview_controls_legend_expanded()
    fullscreen = bool(getattr(host, "_preview_fullscreen_active", False))
    compact = not bool(expanded)
    top_clearance = 0.0
    if fullscreen or bool(getattr(host, "_preview_render_state", None)):
        top_clearance = max(38.0, float(getattr(host, "_preview_overlay_top_bar_height", 38.0) or 38.0)) + 8.0

    margin = 12.0
    left, top = margin, max(margin, top_clearance + 4.0)
    right, bottom = max(left + 1.0, host_width - margin), max(top + 1.0, host_height - margin)
    dock = getattr(host, "preview_overlay_dock", None)
    dock_rect = None
    if dock is not None and str(dock.winfo_manager()) == "place":
        try:
            # Read the requested placement, which is already current even before
            # Tk handles Configure. Never enter the event loop while laying out.
            info = dock.place_info()
            dock_x, dock_y = float(info["x"]), float(info["y"])
            dock_w, dock_h = float(info["width"]), float(info["height"])
            if dock_w > 0 and dock_h > 0:
                dock_rect = (dock_x, dock_y, dock_x + dock_w, dock_y + dock_h)
        except (tk.TclError, KeyError, ValueError):
            pass
    if dock_rect is not None:
        dx1, dy1, dx2, dy2 = dock_rect
        gap = 8.0
        # Prefer the space beside the drawer; on a narrow preview use the
        # space below it and scroll the legend instead of covering the drawer.
        candidates = (
            (left, top, min(right, dx1 - gap), bottom),
            (max(left, dx2 + gap), top, right, bottom),
            (left, max(top, dy2 + gap), right, bottom),
            (left, top, right, min(bottom, dy1 - gap)),
        )
        for area in candidates:
            if area[2] - area[0] >= 220.0 and area[3] - area[1] >= 56.0:
                left, top, right, bottom = area
                break

    overlay_width = min(360.0 if compact else (520.0 if fullscreen else 620.0), right - left)
    # The slim scrollbar overlays the inside edge of the card. It never leaves
    # a differently coloured gutter, or changes the canvas width on refresh.
    canvas_width = max(1.0, overlay_width)
    context = host._get_preview_legend_context()
    render_key = (bool(expanded), canvas_width, tuple(sorted(context.items())))
    if refresh or render_key != getattr(host, "_preview_controls_legend_render_key", None):
        _draw_preview_controls_legend(host, canvas_width)
        host._preview_controls_legend_render_key = render_key
    content_height = float(getattr(host, "_preview_controls_legend_content_height", 32.0))
    overlay_height = min(content_height, bottom - top)
    manual = bool(getattr(host, "_preview_controls_legend_position_manual", False))
    offset_x, offset_y = left, top
    if manual:
        offset_x = float(getattr(host, "_preview_controls_legend_offset_x", offset_x))
        offset_y = float(getattr(host, "_preview_controls_legend_offset_y", offset_y))
    offset_x, offset_y = clamp_preview_controls_legend_offsets(
        host, offset_x, offset_y, width=overlay_width, height=overlay_height,
    )
    host._preview_controls_legend_offset_x = offset_x
    host._preview_controls_legend_offset_y = offset_y
    host._preview_controls_legend_current_width = float(overlay_width)
    host._preview_controls_legend_current_height = float(overlay_height)
    host._preview_controls_legend_current_bounds = (
        float(offset_x),
        float(offset_y),
        float(offset_x) + float(overlay_width),
        float(offset_y) + float(overlay_height),
    )

    canvas.configure(width=int(canvas_width), height=int(math.ceil(overlay_height)))
    overlay.grid_columnconfigure(1, minsize=0)
    if content_height > overlay_height + 1.0:
        if vbar is not None:
            if str(vbar.winfo_manager()) == "grid":
                vbar.grid_forget()
            vbar.place(x=int(overlay_width - 8), y=2, width=6, height=max(1, int(overlay_height - 4)))
            tk.Misc.tkraise(vbar)
    else:
        if vbar is not None and str(vbar.winfo_manager()):
            vbar.place_forget()
        canvas.yview_moveto(0.0)

    overlay.place(
        in_=host_widget,
        x=int(round(offset_x)),
        y=int(round(offset_y)),
        width=int(math.ceil(overlay_width)),
        height=int(math.ceil(overlay_height)),
        anchor="nw",
    )
    overlay.lift()
