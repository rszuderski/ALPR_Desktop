from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

from .web_slim_scrollbar import blend_hex_colors


def measure_preview_text_badge(host, canvas, text: str, *, font=("Segoe UI", 8, "bold"), pad_x: int = 4, pad_y: int = 2):
    if canvas is None or not text:
        return 0, 0

    temp_id = None
    try:
        temp_id = canvas.create_text(-1000, -1000, text=text, font=font, anchor=tk.NW)
        bbox = canvas.bbox(temp_id)
        if not bbox:
            return 0, 0
        width = max(0, int(bbox[2] - bbox[0])) + (pad_x * 2)
        height = max(0, int(bbox[3] - bbox[1])) + (pad_y * 2)
        return width, height
    except Exception:
        return 0, 0
    finally:
        if temp_id is not None:
            try:
                canvas.delete(temp_id)
            except Exception:
                pass


def measure_preview_badge_stack(
    host,
    canvas,
    badge_layers,
    *,
    font=("Segoe UI", 8, "bold"),
    pad_x: int = 4,
    pad_y: int = 2,
    layer_gap: int = 1,
) -> tuple[float, float]:
    prepared_layers = list(badge_layers or [])
    if canvas is None or not prepared_layers:
        return 0.0, 0.0

    total_height = 0.0
    max_width = 0.0
    for idx, layer in enumerate(prepared_layers):
        text = str(layer.get("text", "") or "").strip()
        if not text:
            continue
        width, height = measure_preview_text_badge(host, canvas, text, font=font, pad_x=pad_x, pad_y=pad_y)
        if width <= 0 or height <= 0:
            continue
        max_width = max(max_width, float(width))
        total_height += float(height)
        if idx > 0:
            total_height += float(layer_gap)

    return float(max_width), float(total_height)


def draw_preview_text_badge(
    host,
    canvas,
    x: float,
    y: float,
    text: str,
    fill_color: str,
    *,
    anchor=tk.NW,
    font=("Segoe UI", 8, "bold"),
    outline_color: str = None,
    text_color: str = None,
    pad_x: int = 4,
    pad_y: int = 2,
    tags=None,
):
    if canvas is None or not text:
        return None, None

    resolved_fill = str(fill_color or "#3c3c3c")
    resolved_outline = str(outline_color or resolved_fill)
    resolved_text = str(text_color or host._get_readable_text_color(resolved_fill, preferred="#ffffff"))

    try:
        text_id = canvas.create_text(x, y, text=text, fill=resolved_text, font=font, anchor=anchor)
        text_bbox = canvas.bbox(text_id)
        if not text_bbox:
            return text_id, None

        bg_id = canvas.create_rectangle(
            text_bbox[0] - pad_x,
            text_bbox[1] - pad_y,
            text_bbox[2] + pad_x,
            text_bbox[3] + pad_y,
            fill=resolved_fill,
            outline=resolved_outline,
            width=1,
        )
        canvas.tag_raise(text_id, bg_id)
        if tags:
            for tag in list(tags):
                try:
                    canvas.addtag_withtag(str(tag), text_id)
                    canvas.addtag_withtag(str(tag), bg_id)
                except Exception:
                    pass
        return text_id, bg_id
    except Exception:
        return None, None


def draw_preview_badge_stack(
    host,
    canvas,
    x: float,
    y: float,
    badge_layers,
    *,
    font=("Segoe UI", 8, "bold"),
    pad_x: int = 4,
    pad_y: int = 2,
    layer_gap: int = 1,
    tags=None,
) -> tuple[float, float]:
    prepared_layers = list(badge_layers or [])
    if canvas is None or not prepared_layers:
        return 0.0, 0.0

    stack_width, stack_height = measure_preview_badge_stack(
        host, canvas, prepared_layers, font=font, pad_x=pad_x, pad_y=pad_y, layer_gap=layer_gap
    )
    current_y = float(y)
    for idx, layer in enumerate(prepared_layers):
        text = str(layer.get("text", "") or "").strip()
        if not text:
            continue
        width, height = measure_preview_text_badge(host, canvas, text, font=font, pad_x=pad_x, pad_y=pad_y)
        if width <= 0 or height <= 0:
            continue
        layer_x = float(x) + max(0.0, (float(stack_width) - float(width)) / 2.0)
        draw_preview_text_badge(
            host,
            canvas,
            layer_x,
            current_y,
            text,
            fill_color=str(layer.get("fill_color", "#3c3c3c")),
            outline_color=str(layer.get("outline_color", layer.get("fill_color", "#3c3c3c"))),
            text_color=str(layer.get("text_color", "#ffffff")),
            font=font,
            anchor=tk.NW,
            pad_x=pad_x,
            pad_y=pad_y,
            tags=tags,
        )
        current_y += float(height)
        if idx < len(prepared_layers) - 1:
            current_y += float(layer_gap)

    return float(stack_width), float(stack_height)


def fit_preview_text_to_width(host, text: str, max_width: float, *, font=("Segoe UI", 8, "bold")) -> str:
    content = str(text or "")
    if not content:
        return ""
    try:
        font_obj = font if isinstance(font, tkfont.Font) else tkfont.Font(font=font)
        limit = max(0.0, float(max_width or 0.0))
        if limit <= 0.0 or float(font_obj.measure(content)) <= limit:
            return content
        ellipsis = "..."
        if float(font_obj.measure(ellipsis)) > limit:
            return ""
        trimmed = content
        while trimmed and float(font_obj.measure(f"{trimmed}{ellipsis}")) > limit:
            trimmed = trimmed[:-1]
        return f"{trimmed.rstrip()}{ellipsis}" if trimmed else ellipsis
    except Exception:
        return content


def draw_preview_fixed_text_badge(
    host,
    canvas,
    x: float,
    y: float,
    width: float,
    text: str,
    fill_color: str,
    *,
    font=("Segoe UI", 8, "bold"),
    outline_color: str = None,
    text_color: str = None,
    pad_x: int = 6,
    pad_y: int = 2,
    text_align: str = "left",
    tags=None,
):
    if canvas is None or float(width or 0.0) <= 0.0:
        return None, None

    resolved_fill = str(fill_color or "#3c3c3c")
    resolved_outline = str(outline_color or resolved_fill)
    resolved_text = str(text_color or host._get_readable_text_color(resolved_fill, preferred="#ffffff"))
    resolved_font = font if isinstance(font, tkfont.Font) else tkfont.Font(font=font)
    badge_width = max(24.0, float(width))
    badge_height = max(18.0, float(resolved_font.metrics("linespace")) + (pad_y * 2.0) + 2.0)
    visible_text = fit_preview_text_to_width(host, text, badge_width - (pad_x * 2.0), font=resolved_font)

    bg_id = None
    text_id = None
    try:
        bg_id = canvas.create_rectangle(
            x,
            y,
            x + badge_width,
            y + badge_height,
            fill=resolved_fill,
            outline=resolved_outline,
            width=1,
        )
        normalized_align = str(text_align or "left").strip().lower()
        if normalized_align == "center":
            text_anchor = tk.CENTER
            text_x = x + (badge_width / 2.0)
        elif normalized_align == "right":
            text_anchor = tk.E
            text_x = x + badge_width - pad_x
        else:
            text_anchor = tk.W
            text_x = x + pad_x

        text_id = canvas.create_text(
            text_x,
            y + (badge_height / 2.0),
            text=visible_text,
            fill=resolved_text,
            font=resolved_font,
            anchor=text_anchor,
        )
        canvas.tag_raise(text_id, bg_id)
        if tags:
            for tag in list(tags):
                try:
                    canvas.addtag_withtag(str(tag), text_id)
                    canvas.addtag_withtag(str(tag), bg_id)
                except Exception:
                    pass
        return text_id, bg_id
    except Exception:
        return text_id, bg_id


def get_preview_badge_layout_metrics(host):
    return {
        "font": ("Segoe UI", 6, "normal"),
        "pad_x": 2,
        "pad_y": 1,
        "col_gap": 3,
        "row_gap": 2,
    }


def estimate_preview_badge_layout(host, canvas, badge_specs, image_width: float):
    if canvas is None or image_width <= 0:
        return 1, 12

    metrics = get_preview_badge_layout_metrics(host)
    prepared = []
    max_height = 0
    for spec in sorted(list(badge_specs or []), key=lambda item: float(item.get("center_ref", item.get("center_x", 0.0)))):
        badge_layers = list(spec.get("badge_layers", []) or [])
        text = str(spec.get("text", "") or "").strip()
        if badge_layers:
            width, height = measure_preview_badge_stack(
                host,
                canvas,
                badge_layers,
                font=metrics["font"],
                pad_x=metrics["pad_x"],
                pad_y=metrics["pad_y"],
                layer_gap=metrics["row_gap"],
            )
        else:
            if not text:
                continue
            width, height = measure_preview_text_badge(
                host, canvas, text, font=metrics["font"], pad_x=metrics["pad_x"], pad_y=metrics["pad_y"]
            )
        if width <= 0 or height <= 0:
            continue

        try:
            center_ref = float(spec.get("center_ref", spec.get("center_x", 0.0)))
        except Exception:
            center_ref = 0.0
        prepared.append({"center_ref": center_ref, "width": float(width)})
        max_height = max(max_height, int(height))

    if not prepared:
        return 1, max(10, max_height or 12)

    row_last_right = []
    for spec in prepared:
        width = float(spec["width"])
        desired_left = (float(spec["center_ref"]) * float(image_width)) - (width / 2.0)
        desired_left = max(0.0, min(desired_left, float(image_width) - width))

        placed = False
        for row_idx in range(len(row_last_right)):
            candidate_left = max(desired_left, row_last_right[row_idx] + metrics["col_gap"])
            if candidate_left + width <= float(image_width):
                row_last_right[row_idx] = candidate_left + width
                placed = True
                break
        if not placed:
            row_last_right.append(desired_left + width)

    return max(1, len(row_last_right)), max(10, max_height or 12)


def measure_preview_overlay_text_width(host, text: str, font) -> float:
    content = str(text or "")
    if not content:
        return 0.0
    try:
        font_obj, key = _preview_overlay_measurement_font(host, font)
        cache = getattr(host, "_preview_overlay_text_width_cache", None)
        if not isinstance(cache, dict):
            cache = host._preview_overlay_text_width_cache = {}
        cache_key = (key, content)
        if cache_key not in cache:
            if len(cache) >= 512:
                cache.clear()
            cache[cache_key] = float(font_obj.measure(content))
        return cache[cache_key]
    except Exception:
        return float(max(0, len(content)) * 7)


def measure_preview_overlay_font_height(host, font) -> float:
    try:
        font_obj, key = _preview_overlay_measurement_font(host, font)
        cache = getattr(host, "_preview_overlay_font_height_cache", None)
        if not isinstance(cache, dict):
            cache = host._preview_overlay_font_height_cache = {}
        if key not in cache:
            if len(cache) >= 32:
                cache.clear()
            cache[key] = float(max(10, int(font_obj.metrics("linespace") or 0)))
        return cache[key]
    except Exception:
        return 12.0


def _preview_overlay_measurement_font(host, font):
    master = getattr(host, "preview_canvas", None) or host.frame
    fonts = getattr(host, "_preview_overlay_measurement_fonts", None)
    if not isinstance(fonts, dict):
        fonts = host._preview_overlay_measurement_fonts = {}
    if isinstance(font, tkfont.Font):
        font_obj = font
        description = tuple(sorted(font.actual().items()))
    else:
        description = str(font)
        if description not in fonts:
            if len(fonts) >= 32:
                fonts.clear()
            fonts[description] = tkfont.Font(root=master, font=font)
        font_obj = fonts[description]
    return font_obj, (description, float(master.tk.call("tk", "scaling")))


def estimate_preview_source_legend_height(host) -> float:
    badge_font = ("Segoe UI", 7, "bold")
    label_font = ("Segoe UI", 7, "normal")
    badge_pad_x = 2
    badge_pad_y = 1
    max_item_height = measure_preview_overlay_font_height(host, label_font)

    for item in host._get_preview_source_component_legend_items():
        style = host._get_preview_badge_component_style(item.get("component"))
        _badge_width, badge_height = measure_preview_text_badge(
            host,
            getattr(host, "preview_canvas", None),
            str(style.get("label", "") or "").strip(),
            font=badge_font,
            pad_x=badge_pad_x,
            pad_y=badge_pad_y,
        )
        max_item_height = max(max_item_height, float(badge_height or 0.0))

    return float(max(12.0, max_item_height))


def estimate_preview_source_legend_width(host) -> float:
    badge_font = ("Segoe UI", 7, "bold")
    label_font = ("Segoe UI", 7, "normal")
    badge_pad_x = 2
    badge_pad_y = 1
    item_gap = 8
    total_width = 0.0

    legend_items = host._get_preview_source_component_legend_items()
    for idx, item in enumerate(legend_items):
        style = host._get_preview_badge_component_style(item.get("component"))
        badge_text = str(style.get("label", "") or "").strip()
        legend_text = str(style.get("legend", "") or badge_text).strip()
        badge_width, _badge_height = measure_preview_text_badge(
            host,
            getattr(host, "preview_canvas", None),
            badge_text,
            font=badge_font,
            pad_x=badge_pad_x,
            pad_y=badge_pad_y,
        )
        if badge_width <= 0:
            badge_width = measure_preview_overlay_text_width(host, badge_text, badge_font) + (badge_pad_x * 2.0)
        total_width += float(badge_width)
        total_width += 5.0
        total_width += measure_preview_overlay_text_width(host, legend_text, label_font)
        if idx < (len(legend_items) - 1):
            total_width += item_gap

    return float(total_width)


def get_preview_source_visual_style(host, source_tag: str):
    palette = getattr(host.app, "palette", {})
    normalized = host._normalize_character_source_tag(raw_tag=source_tag)

    styles = {
        "manual": {
            "outline": "#d000a8",
            "guide": "#ff6ce4",
            "char": "#fff5d6",
            "badge_fill": "#4f3a0f",
            "badge_outline": "#6b4a10",
            "badge_fg": "#ffe9a8",
            "label": "M",
            "legend": "Ręczne",
        },
        "ocr": {
            "outline": palette.get("info", "#56b6ff"),
            "guide": "#9fd8ff",
            "char": "#dff4ff",
            "badge_fill": "#163346",
            "badge_outline": "#275674",
            "badge_fg": "#bfe9ff",
            "label": "O",
            "legend": "OCR znak",
        },
        "yolo": {
            "outline": "#0064e0",
            "guide": "#ffd79c",
            "char": "#fff1d6",
            "badge_fill": "#4d3310",
            "badge_outline": "#6b4a10",
            "badge_fg": "#ffe3af",
            "label": "YS",
            "legend": "YOLO znak",
        },
        "yolo_box": {
            # Canvas geometry needs contrast against the photo, independently
            # of the theme's warning colour (olive on light plates is too faint).
            "outline": "#0064e0",
            "guide": "#ffd79c",
            "char": "#fff1d6",
            "badge_fill": "#4d3310",
            "badge_outline": "#6b4a10",
            "badge_fg": "#ffe3af",
            "label": "YB",
            "legend": "YOLO box",
        },
        "yolo_symbol": {
            "outline": palette.get("success", "#2ecc71"),
            "guide": "#8be7b1",
            "char": "#ddffe9",
            "badge_fill": "#123b25",
            "badge_outline": "#1b6a42",
            "badge_fg": "#bff5d1",
            "label": "YS",
            "legend": "YOLO znak",
        },
        "yolo_box_ocr": {
            "outline": blend_hex_colors(palette.get("info", "#56b6ff"), palette.get("warning", "#f4c27a"), 0.38),
            "guide": blend_hex_colors("#9fd8ff", "#ffd79c", 0.45),
            "char": "#e8f7ff",
            "badge_fill": "#17384a",
            "badge_outline": "#2b5f7d",
            "badge_fg": "#d7f2ff",
            "label": "O",
            "legend": "YOLO box + OCR znak",
        },
        "yolo_rescue": {
            "outline": palette.get("success", "#2ecc71"),
            "guide": "#8be7b1",
            "char": "#ddffe9",
            "badge_fill": "#123b25",
            "badge_outline": "#1b6a42",
            "badge_fg": "#bff5d1",
            "label": "YS",
            "legend": "OCR znak + YOLO znak",
        },
    }
    return styles.get(normalized, styles["ocr"])


def get_preview_badge_component_style(host, component_key: str) -> dict:
    normalized = str(component_key or "").strip().lower().replace("-", "_")
    mapping = {
        "manual": ("manual", "M", "Ręczne"),
        "ocr_symbol": ("ocr", "O", "OCR znak"),
        "yolo_box": ("yolo_box", "YB", "YOLO box"),
        "yolo_symbol": ("yolo_symbol", "YS", "YOLO znak"),
    }
    source_key, label, legend = mapping.get(normalized, mapping["ocr_symbol"])
    base_style = dict(get_preview_source_visual_style(host, source_key))
    base_style["label"] = label
    base_style["legend"] = legend
    return base_style


def get_preview_source_component_legend_items(host) -> list[dict]:
    return [
        {"component": "manual"},
        {"component": "ocr_symbol"},
        {"component": "yolo_box"},
        {"component": "yolo_symbol"},
    ]


def get_preview_source_badge_layers(
    host,
    source_tag: str,
    *,
    confidence: float | None = None,
    box_backend_confidence: float | None = None,
    include_confidence: bool = True,
    has_symbol: bool | None = None,
    uses_yolo_box_backend: bool = False,
) -> list[dict]:
    normalized = host._normalize_character_source_tag(raw_tag=source_tag)

    def _build_layer(component_key: str, text: str | None = None, conf_value: float | None = None) -> dict:
        style = get_preview_badge_component_style(host, component_key)
        layer_text = str(text or style.get("label", "") or "").strip()
        if include_confidence and conf_value is not None:
            try:
                layer_text = f"{layer_text} {float(conf_value):.2f}"
            except Exception:
                pass
        fill = str(style.get("badge_fill", style.get("outline", "#3c3c3c")))
        outline = str(style.get("badge_outline", fill))
        text_color = str(host._get_readable_text_color(fill, preferred=style.get("badge_fg", "#ffffff")))
        return {"text": layer_text, "fill_color": fill, "outline_color": outline, "text_color": text_color}

    if normalized == "yolo_box_ocr":
        yolo_confidence = box_backend_confidence if box_backend_confidence is not None else confidence
        if has_symbol is False:
            return [_build_layer("yolo_box", "YB", yolo_confidence)]
        return [_build_layer("yolo_box", "YB", yolo_confidence), _build_layer("ocr_symbol", "O")]
    if normalized == "yolo_rescue":
        return list(reversed([_build_layer("ocr_symbol", "O"), _build_layer("yolo_symbol", "YS", confidence)]))
    if normalized == "yolo_box":
        return [_build_layer("yolo_box", "YB", box_backend_confidence if box_backend_confidence is not None else confidence)]
    if normalized == "yolo_symbol":
        return [_build_layer("yolo_symbol", "YS", confidence)]
    if normalized == "yolo":
        return list(reversed([_build_layer("yolo_box", "YB"), _build_layer("yolo_symbol", "YS", confidence)]))
    if normalized == "manual":
        layers = [_build_layer("manual", "M")]
        if uses_yolo_box_backend:
            layers.insert(0, _build_layer("yolo_box", "YB", box_backend_confidence))
        return layers
    layers = [_build_layer("ocr_symbol", "O", confidence)]
    if uses_yolo_box_backend:
        layers.insert(0, _build_layer("yolo_box", "YB", box_backend_confidence))
    return layers


def draw_preview_source_legend(host, canvas, x: float, y: float):
    if canvas is None:
        return float(x)

    text_fg = getattr(host.app, "palette", {}).get("muted", "#b0b0b0")
    badge_font = ("Segoe UI", 7, "bold")
    label_font = ("Segoe UI", 7, "normal")
    badge_pad_x = 2
    badge_pad_y = 1
    item_gap = 8
    current_x = float(x)

    legend_items = host._get_preview_source_component_legend_items()
    for item in legend_items:
        style = host._get_preview_badge_component_style(item.get("component"))
        badge_text = str(style.get("label", "") or "").strip()
        legend_text = str(style.get("legend", "") or badge_text).strip()
        badge_width, badge_height = measure_preview_text_badge(
            host, canvas, badge_text, font=badge_font, pad_x=badge_pad_x, pad_y=badge_pad_y
        )
        draw_preview_text_badge(
            host,
            canvas,
            current_x,
            y,
            badge_text,
            fill_color=str(style.get("badge_fill", style.get("outline", "#3c3c3c"))),
            outline_color=str(style.get("badge_outline", style.get("badge_fill", style.get("outline", "#3c3c3c")))),
            text_color=str(host._get_readable_text_color(
                str(style.get("badge_fill", style.get("outline", "#3c3c3c"))),
                preferred=style.get("badge_fg", "#ffffff"),
            )),
            font=badge_font,
            anchor=tk.NW,
            pad_x=badge_pad_x,
            pad_y=badge_pad_y,
            tags=("preview_overlay", "preview_source_legend"),
        )
        label_x = current_x + badge_width + 5
        label_y = y + max(0, badge_height / 2.0)
        label_id = canvas.create_text(
            label_x,
            label_y,
            text=legend_text,
            fill=text_fg,
            font=label_font,
            anchor=tk.W,
            tags=("preview_overlay", "preview_source_legend"),
        )
        label_bbox = canvas.bbox(label_id)
        current_x = float(label_bbox[2]) + item_gap if label_bbox else label_x + item_gap

    return current_x


# Split provenance badges.  The earlier single source_tag model is kept above
# for metadata compatibility, but UI should render box and sign provenance
# independently when those fields are available.
def get_preview_badge_component_style(host, component_key: str) -> dict:
    normalized = str(component_key or "").strip().lower().replace("-", "_")
    mapping = {
        "manual": ("manual", "M", "Reczne"),
        "manual_box": ("manual", "MB", "Manual box"),
        "manual_sign": ("manual", "MS", "Manual znak"),
        "generated_box": ("ocr", "GB", "Box segmentowany"),
        "ocr_symbol": ("ocr", "OS", "OCR znak"),
        "yolo_box": ("yolo_box", "YB", "YOLO box"),
        "yolo_symbol": ("yolo_symbol", "YS", "YOLO znak"),
    }
    source_key, label, legend = mapping.get(normalized, mapping["ocr_symbol"])
    base_style = dict(get_preview_source_visual_style(host, source_key))
    base_style["label"] = label
    base_style["legend"] = legend
    return base_style


def get_preview_source_component_legend_items(host) -> list[dict]:
    return [
        {"component": "manual_box"},
        {"component": "manual_sign"},
        {"component": "generated_box"},
        {"component": "ocr_symbol"},
        {"component": "yolo_box"},
        {"component": "yolo_symbol"},
    ]


def get_preview_source_badge_layers(
    host,
    source_tag: str,
    *,
    confidence: float | None = None,
    box_backend_confidence: float | None = None,
    include_confidence: bool = True,
    has_symbol: bool | None = None,
    uses_yolo_box_backend: bool = False,
    box_source: str | None = None,
    sign_source: str | None = None,
) -> list[dict]:
    normalized = host._normalize_character_source_tag(raw_tag=source_tag)

    def _build_layer(component_key: str, text: str | None = None, conf_value: float | None = None) -> dict:
        style = get_preview_badge_component_style(host, component_key)
        layer_text = str(text or style.get("label", "") or "").strip()
        if include_confidence and conf_value is not None:
            try:
                layer_text = f"{layer_text} {float(conf_value):.2f}"
            except Exception:
                pass
        fill = str(style.get("badge_fill", style.get("outline", "#3c3c3c")))
        outline = str(style.get("badge_outline", fill))
        text_color = str(host._get_readable_text_color(fill, preferred=style.get("badge_fg", "#ffffff")))
        return {"text": layer_text, "fill_color": fill, "outline_color": outline, "text_color": text_color}

    normalized_box = str(box_source or "").strip().lower().replace("-", "_")
    normalized_sign = str(sign_source or "").strip().lower().replace("-", "_")
    if normalized_box or normalized_sign:
        layers = []
        if normalized_box == "manual_box":
            layers.append(_build_layer("manual_box", "MB"))
        elif normalized_box == "yolo_box":
            layers.append(_build_layer("yolo_box", "YB", box_backend_confidence if box_backend_confidence is not None else confidence))
        elif normalized_box == "generated_box":
            layers.append(_build_layer("generated_box", "GB"))

        if normalized_sign == "manual_sign":
            layers.append(_build_layer("manual_sign", "MS"))
        elif normalized_sign == "yolo_symbol":
            layers.append(_build_layer("yolo_symbol", "YS", confidence))
        elif normalized_sign == "ocr_symbol":
            layers.append(_build_layer("ocr_symbol", "OS", confidence if normalized_box != "yolo_box" else None))
        return layers

    if normalized == "yolo_box_ocr":
        yolo_confidence = box_backend_confidence if box_backend_confidence is not None else confidence
        if has_symbol is False:
            return [_build_layer("yolo_box", "YB", yolo_confidence)]
        return [_build_layer("yolo_box", "YB", yolo_confidence), _build_layer("ocr_symbol", "OS")]
    if normalized == "yolo_rescue":
        return list(reversed([_build_layer("ocr_symbol", "OS"), _build_layer("yolo_symbol", "YS", confidence)]))
    if normalized == "yolo_box":
        return [_build_layer("yolo_box", "YB", box_backend_confidence if box_backend_confidence is not None else confidence)]
    if normalized == "yolo_symbol":
        return [_build_layer("yolo_symbol", "YS", confidence)]
    if normalized == "yolo":
        return list(reversed([_build_layer("yolo_box", "YB"), _build_layer("yolo_symbol", "YS", confidence)]))
    if normalized == "manual":
        layers = [_build_layer("manual", "M")]
        if uses_yolo_box_backend:
            layers.insert(0, _build_layer("yolo_box", "YB", box_backend_confidence))
        return layers
    layers = [_build_layer("ocr_symbol", "OS", confidence)]
    if uses_yolo_box_backend:
        layers.insert(0, _build_layer("yolo_box", "YB", box_backend_confidence))
    return layers
