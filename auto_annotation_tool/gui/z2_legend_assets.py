#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bitmap asset helpers for the Z2 canvas shortcut legend."""

import math

from PIL import Image, ImageDraw, ImageFilter, ImageTk

from .web_slim_scrollbar import blend_hex_colors


def legend_color_is_light(color: str) -> bool:
    value = str(color or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return False
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except Exception:
        return False
    luminance = ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)) / 255.0
    return luminance >= 0.62


def get_preview_legend_text_color(owner, fill: str) -> str:
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    if legend_color_is_light(fill):
        return palette.get("fg", "#111111")
    return palette.get("accent_text", "#ffffff")


def hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = str(color or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return 32, 36, 42, int(alpha)
    try:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
            int(alpha),
        )
    except Exception:
        return 32, 36, 42, int(alpha)


def _legend_image_cache(owner) -> dict:
    cache = getattr(owner, "_preview_legend_image_cache", None)
    if cache is None:
        cache = {}
        owner._preview_legend_image_cache = cache
    return cache


def trim_preview_legend_image_cache(owner, *, max_entries: int = 48):
    cache = getattr(owner, "_preview_legend_image_cache", None)
    if not isinstance(cache, dict):
        return
    try:
        limit = max(8, int(max_entries))
    except Exception:
        limit = 48
    while len(cache) > limit:
        try:
            oldest_key = next(iter(cache))
        except Exception:
            break
        cache.pop(oldest_key, None)


def clear_preview_legend_image_cache(owner):
    cache = getattr(owner, "_preview_legend_image_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def get_preview_legend_compass_photo(owner, theme: dict, *, size: int = 38):
    cache = _legend_image_cache(owner)
    key = (
        "compass",
        int(size),
        str(theme.get("badge_plate_outline", "")),
        str(theme.get("token_fill", "")),
        str(theme.get("compass_guide", "")),
        str(theme.get("compass_dot", "")),
        str(theme.get("compass_north", "")),
        str(theme.get("compass_south", "")),
    )
    photo = cache.get(key)
    if photo is not None:
        return photo

    scale = 6
    px = int(size * scale)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    accent = hex_to_rgba(str(theme.get("badge_plate_outline", "#2fbf71")), 255)
    token_fill = hex_to_rgba(str(theme.get("token_fill", "#3c3c3c")), 245)
    north_fill = hex_to_rgba(str(theme.get("compass_north", "#d64545")), 255)
    south_fill = hex_to_rgba(str(theme.get("compass_south", "#183a63")), 248)
    guide_fill = hex_to_rgba(str(theme.get("compass_guide", "#d7dde4")), 96)

    margin = int(2.5 * scale)
    outer = (margin, margin, px - margin, px - margin)
    inner_margin = int(10 * scale / 2.0)
    inner = (inner_margin, inner_margin, px - inner_margin, px - inner_margin)
    draw.ellipse(outer, outline=accent, width=max(2, scale))
    draw.ellipse(
        (margin + scale, margin + scale, px - margin - scale, px - margin - scale),
        outline=hex_to_rgba(blend_hex_colors(str(theme.get("badge_plate_outline", "#2fbf71")), "#ffffff", 0.25), 180),
        width=max(1, scale - 1),
    )
    draw.ellipse(inner, fill=token_fill)
    center = px / 2.0
    inner_r = (inner[2] - inner[0]) / 2.0
    draw.line((center, center - inner_r + (1.0 * scale), center, center + inner_r - (1.0 * scale)), fill=guide_fill, width=max(1, scale - 1))
    draw.line((center - inner_r + (1.0 * scale), center, center + inner_r - (1.0 * scale), center), fill=guide_fill, width=max(1, scale - 1))

    north = [
        (center, center - (14.5 * scale / 2.0)),
        (center + (2.9 * scale), center + (1.2 * scale)),
        (center, center - (2.4 * scale)),
        (center - (2.9 * scale), center + (1.2 * scale)),
    ]
    south = [
        (center, center + (14.5 * scale / 2.0)),
        (center + (2.6 * scale), center - (1.2 * scale)),
        (center, center + (2.4 * scale)),
        (center - (2.6 * scale), center - (1.2 * scale)),
    ]
    draw.polygon(north, fill=north_fill)
    draw.polygon(south, fill=south_fill)
    dot_r = 1.8 * scale
    draw.ellipse(
        (center - dot_r, center - dot_r, center + dot_r, center + dot_r),
        fill=hex_to_rgba(str(theme.get("compass_dot", "#f3f3f3")), 255),
    )

    img = img.resize((int(size), int(size)), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img, master=owner.frame)
    cache[key] = photo
    return photo


def get_preview_legend_shell_photo(owner, theme: dict, *, width: int, height: int, overlay_x: int = 10, overlay_y: int = 10):
    cache = _legend_image_cache(owner)
    safe_w = max(40, int(width))
    safe_h = max(40, int(height))
    key = (
        "shell",
        safe_w,
        safe_h,
        str(theme.get("panel_fill", "")),
        str(theme.get("canvas_bg", "")),
        str(theme.get("shell_outline", "")),
    )
    photo = cache.get(key)
    if photo is not None:
        return photo

    scale = 2
    px_w = safe_w * scale
    px_h = safe_h * scale

    panel_fill = blend_hex_colors(
        str(theme.get("panel_fill", "#2d2d30")),
        str(theme.get("canvas_bg", "#0b1016")),
        0.10,
    )
    outer_base = Image.new("RGBA", (px_w, px_h), hex_to_rgba(panel_fill, 255))
    panel_base = Image.new("RGBA", (px_w, px_h), hex_to_rgba(panel_fill, 255))

    outline_alpha = 156
    inset = 0

    top_sheen = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    sheen_draw = ImageDraw.Draw(top_sheen)
    sheen_fill = hex_to_rgba(blend_hex_colors(str(theme.get("panel_fill", "#2d2d30")), "#ffffff", 0.08), 20)
    sheen_draw.rounded_rectangle(
        (inset, inset, px_w - inset - 1, min(px_h - inset - 1, inset + int(26 * scale))),
        radius=max(2, int(3 * scale)),
        fill=sheen_fill,
    )
    top_sheen = top_sheen.filter(ImageFilter.GaussianBlur(radius=2 * scale))
    panel_base.alpha_composite(top_sheen)

    mask = Image.new("L", (px_w, px_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = max(2, int(3 * scale))
    mask_draw.rounded_rectangle((inset, inset, px_w - inset - 1, px_h - inset - 1), radius=radius, fill=255)
    rounded = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    rounded.paste(panel_base, (0, 0), mask)

    outline_draw = ImageDraw.Draw(rounded)
    outline_rgba = hex_to_rgba(
        blend_hex_colors(
            str(theme.get("shell_outline", theme.get("badge_plate_outline", "#2fbf71"))),
            str(theme.get("panel_fill", "#2d2d30")),
            0.12,
        ),
        outline_alpha,
    )
    outline_draw.rounded_rectangle(
        (inset, inset, px_w - inset - 1, px_h - inset - 1),
        radius=radius,
        outline=outline_rgba,
        width=max(1, scale - 1),
    )

    final_img = outer_base.copy()
    final_img.alpha_composite(rounded)
    img = final_img.resize((safe_w, safe_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img, master=owner.frame)
    cache[key] = photo
    trim_preview_legend_image_cache(owner)
    return photo


def get_preview_legend_keycap_photo(owner, text: str, *, fill: str, outline: str, text_fill: str):
    cache = _legend_image_cache(owner)
    text_value = str(text or "")
    font_obj = owner._get_preview_legend_font(7, "bold")
    width = int(math.ceil(max(22.0, float(font_obj.measure(text_value)) + 10.0)))
    height = 19
    key = ("keycap", text_value, width, height, str(fill), str(outline), str(text_fill))
    cached = cache.get(key)
    if cached is not None:
        return cached

    scale = 5
    px_w = width * scale
    px_h = height * scale
    img = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int((height / 2.0) * scale)
    fill_rgba = hex_to_rgba(fill, 248)
    outline_rgba = hex_to_rgba(outline, 228)

    draw.rounded_rectangle(
        (1, 1, px_w - 2, px_h - 2),
        radius=radius,
        fill=fill_rgba,
        outline=outline_rgba,
        width=max(2, scale - 1),
    )

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img, master=owner.frame)
    cache[key] = (photo, float(width), float(height))
    trim_preview_legend_image_cache(owner)
    return cache[key]


def get_preview_legend_group_shell_photo(owner, theme: dict, *, width: int, height: int, outline: str):
    cache = _legend_image_cache(owner)
    safe_w = max(32, int(width))
    safe_h = max(32, int(height))
    key = ("legend_group_shell", safe_w, safe_h, str(theme.get("panel_fill", "")), str(outline or ""))
    photo = cache.get(key)
    if photo is not None:
        return photo

    scale = 3
    px_w = safe_w * scale
    px_h = safe_h * scale
    img = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = max(2, int(2 * scale))

    outline_rgba = hex_to_rgba(outline or str(theme.get("badge_plate_outline", "#2fbf71")), 120)
    fill_rgba = hex_to_rgba(blend_hex_colors(str(theme.get("panel_fill", "#2d2d30")), "#0b1016", 0.18), 92)
    glow_rgba = hex_to_rgba(outline or str(theme.get("badge_plate_outline", "#2fbf71")), 26)
    radius = max(6, int(6 * scale))

    glow = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle(
        (inset, inset, px_w - inset - 1, px_h - inset - 1),
        radius=radius,
        fill=glow_rgba,
        outline=None,
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=2 * scale))
    img.alpha_composite(glow)

    draw.rounded_rectangle(
        (inset, inset, px_w - inset - 1, px_h - inset - 1),
        radius=radius,
        fill=fill_rgba,
        outline=outline_rgba,
        width=max(2, scale - 1),
    )

    photo = ImageTk.PhotoImage(img.resize((safe_w, safe_h), Image.Resampling.LANCZOS), master=owner.frame)
    cache[key] = photo
    trim_preview_legend_image_cache(owner)
    return photo


def normalize_preview_legend_interaction(interaction: str | None) -> str:
    value = str(interaction or "").strip().lower()
    if value in {"tap", "click", "single", "1x", "once"}:
        return "tap"
    if value in {"hold", "press", "held", "down"}:
        return "hold"
    return ""


def get_preview_legend_interaction_marker_photo(
    owner,
    mode: str,
    *,
    width: int,
    height: int,
    fill: str,
    outline: str,
    text_fill: str,
):
    cache = _legend_image_cache(owner)
    mode_value = normalize_preview_legend_interaction(mode)
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

    marker_fill = hex_to_rgba(outline, 238)
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

    photo = ImageTk.PhotoImage(img.resize((safe_w, safe_h), Image.Resampling.LANCZOS), master=owner.frame)
    cache[key] = photo
    trim_preview_legend_image_cache(owner)
    return photo
