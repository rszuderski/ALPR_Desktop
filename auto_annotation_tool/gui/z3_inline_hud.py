"""Text-only plate context, anchored to the canvas in both preview modes."""
from pathlib import Path
import tkinter.font as tkfont

from .z2_inline_hud import fit_text


_HUD_CANVAS_BG = "#1f1f1f"
_HUD_LABEL_COLOR = "#b7c0c8"
_HUD_NEUTRAL_VALUE_COLOR = "#f3f6f8"


def _hud_parse_hex_color(value: str):
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return None
    try:
        return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))
    except Exception:
        return None


def _hud_channel_luminance(channel: int) -> float:
    value = max(0.0, min(255.0, float(channel))) / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _hud_relative_luminance(rgb) -> float:
    r, g, b = rgb
    return (
        0.2126 * _hud_channel_luminance(r)
        + 0.7152 * _hud_channel_luminance(g)
        + 0.0722 * _hud_channel_luminance(b)
    )


def _hud_contrast_ratio(fg_rgb, bg_rgb) -> float:
    fg_l = _hud_relative_luminance(fg_rgb)
    bg_l = _hud_relative_luminance(bg_rgb)
    lighter = max(fg_l, bg_l)
    darker = min(fg_l, bg_l)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_inline_hud_canvas_contrast(
    color: str,
    *,
    fallback: str = _HUD_NEUTRAL_VALUE_COLOR,
    min_ratio: float = 5.0,
) -> str:
    """Keep semantic hue but guarantee readable text on the dark PZ2 canvas."""
    bg_rgb = _hud_parse_hex_color(_HUD_CANVAS_BG)
    rgb = _hud_parse_hex_color(color)
    fallback_rgb = _hud_parse_hex_color(fallback)

    if bg_rgb is None:
        return str(color or fallback)
    if rgb is None:
        rgb = fallback_rgb
    if rgb is None:
        return "#f3f6f8"

    if _hud_contrast_ratio(rgb, bg_rgb) >= float(min_ratio):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    original = rgb
    for step in range(1, 21):
        amount = step / 20.0
        candidate = tuple(
            int(round(channel + (255 - channel) * amount))
            for channel in original
        )
        if _hud_contrast_ratio(candidate, bg_rgb) >= float(min_ratio):
            return "#{:02x}{:02x}{:02x}".format(*candidate)

    return str(fallback or "#f3f6f8")



_FILENAME_HINT_SOURCES = {
    "filename_order",
    "filename_order_backfill",
}


def resolve_inline_hud_filename_hint(data: dict | None) -> str:
    """Return only an explicitly mapped filename-derived plate hint.

    Filename text is assistance/provenance, never operator GT.
    Ambiguous filename tokens are deliberately not collapsed into one value.
    """
    source = data if isinstance(data, dict) else {}
    attrs = source.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}

    expected_source = str(
        source.get("source_expected_text_source")
        or attrs.get("source_expected_text_source")
        or ""
    ).strip().lower()
    if expected_source not in _FILENAME_HINT_SOURCES:
        return ""

    values = []
    seen = set()
    for container in (source, attrs):
        value = str(container.get("source_expected_text", "") or "").strip().upper()
        if value and value not in seen:
            seen.add(value)
            values.append(value)

        raw_values = container.get("source_expected_texts")
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if isinstance(raw_values, (list, tuple, set)):
            for raw in raw_values:
                value = str(raw or "").strip().upper()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)

    return values[0] if len(values) == 1 else ""


def resolve_inline_hud_plate_status(host, data: dict | None) -> tuple[str, str]:
    source = data if isinstance(data, dict) else {}
    palette = getattr(getattr(host, "app", None), "palette", {}) or {}

    gold_state = source.get("gold_state")
    if isinstance(gold_state, dict) and bool(gold_state.get("excluded", False)):
        return "WYKLUCZONA", str(palette.get("muted", "#8b949e"))

    chars = source.get("characters", [])
    if not isinstance(chars, list):
        chars = []

    try:
        presentation = host._get_preview_status_presentation(
            data=source,
            chars=chars,
            plate_id=str(
                source.get("plate_id", "")
                or getattr(host, "_preview_active_pid", "")
                or ""
            ),
        )
    except Exception:
        presentation = {}

    status = str(
        presentation.get("status", source.get("status", "unknown")) or "unknown"
    ).strip().lower()
    ready = bool(presentation.get("ready_for_approval", False))
    inherited_z2 = bool(
        str(presentation.get("inherited_z2_number", "") or "").strip()
    )

    if status == "perfect":
        return "OK", str(palette.get("success", "#27ae60"))

    pending_color = str(palette.get("warning", "#f59e0b"))
    if ready and inherited_z2:
        return "ZATWIERDŹ", pending_color
    if ready:
        return "DO KONTROLI", pending_color

    return "DO KOREKTY", str(palette.get("error", "#c0392b"))

def apply_inline_hud_plate_status(items, status_text: str, status_color: str):
    prepared = [dict(item) for item in list(items or [])]
    for item in prepared:
        current = str(item.get("text", "") or "")
        if current.startswith("Decyzja:") or current.startswith("Status tablicy:"):
            item["text"] = f"Decyzja: {status_text}"
            item["outline"] = str(status_color)
            break
    return prepared


def plan_inline_hud(host, width, data, status_layout):
    fonts = getattr(host, "_preview_inline_hud_fonts", None)
    if fonts is None:
        fonts = (
            tkfont.Font(host.frame, family="Segoe UI", size=14, weight="bold"),
            tkfont.Font(host.frame, family="Segoe UI", size=11, weight="bold"),
        )
        host._preview_inline_hud_fonts = fonts
    file_font, count_font = fonts
    data = data if isinstance(data, dict) else {}
    index = host._get_current_preview_list_index()
    total = len(getattr(host, "_listbox_pid_by_index", []) or [])
    number = int(index) + 1 if index is not None and total else 0
    prefix = f"{number:0{max(2, len(str(total)))}d}. "
    filename = (
        Path(str(data.get("source_image") or "")).name
        or str(data.get("plate_id") or "Brak pliku")
    )
    left, top, right = 12, 8, max(60, width - 12)
    file_limit = max(20, right - 80 - left - file_font.measure(prefix))
    lines = [
        (
            prefix,
            left,
            top,
            file_font,
            "#ffd166",
            ("preview_inline_hud_number",),
        ),
        (
            fit_text(filename, file_font, file_limit),
            left + file_font.measure(prefix),
            top,
            file_font,
            "#7ee7ff",
            (
                "preview_inline_hud_filename",
                "preview_overlay_action",
                "preview_action::edit_source_filename",
            ),
        ),
    ]

    attrs = data.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    number_source = str(
        data.get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip().lower()
    try:
        saved_number = str(host._get_preview_ground_truth_text(data) or "").strip()
    except Exception:
        saved_number = str(
            data.get("ground_truth_text")
            or attrs.get("ground_truth_text")
            or ""
        ).strip()

    review_state = data.get("review_state") if isinstance(data, dict) else {}
    review_status = (
        str(review_state.get("status", "") or "").strip().lower()
        if isinstance(review_state, dict)
        else ""
    )

    if saved_number and number_source == "manual_z2":
        number_text = f"GT z Z2: {saved_number}"
    elif saved_number and review_status == "approved":
        number_text = f"GT Z3: {saved_number}"
    else:
        number_text = "GT z Z2: brak"

    palette = getattr(getattr(host, "app", None), "palette", {}) or {}

    # Jeden, stały kolor wszystkich etykiet HUD.
    # Semantyka koloru dotyczy wyłącznie wartości po dwukropku.
    hud_label_color = ensure_inline_hud_canvas_contrast(
        _HUD_LABEL_COLOR,
        fallback=_HUD_LABEL_COLOR,
        min_ratio=5.0,
    )

    filename_hint = (
        resolve_inline_hud_filename_hint(data)
        if number_source != "manual_z2"
        else ""
    )

    items = [
        {
            "text": f"Tablica: {number}/{total}",
            "outline": str(palette.get("fg", "#f3f3f3")),
        },
        {
            "text": number_text,
            "outline": str(
                palette.get("accent_alt", "#7ee7ff")
                if saved_number and number_source == "manual_z2"
                else palette.get("fg", "#f3f3f3")
            ),
            "tags": (
                "preview_overlay_action",
                "preview_action::edit_plate_gt",
            ),
        },
    ]
    if filename_hint:
        items.append(
            {
                "text": f"Podpowiedź z nazwy: {filename_hint}",
                "outline": str(
                    palette.get(
                        "info",
                        palette.get("accent_alt", "#7ee7ff"),
                    )
                ),
                "tags": ("preview_overlay", "preview_filename_hint"),
            }
        )
    items.extend(dict(item) for item in status_layout.get("neutral_badges", []))
    items.extend(dict(item) for item in status_layout.get("badges", []))

    hud_status, hud_color = resolve_inline_hud_plate_status(host, data)
    items = apply_inline_hud_plate_status(items, hud_status, hud_color)

    x, y = left, top + file_font.metrics("linespace") + 5
    row_height = count_font.metrics("linespace") + 5
    item_gap = 20
    label_value_gap = 5

    for item in items:
        raw_text = str(item.get("text", "") or "").strip()
        tags = tuple(item.get("tags", ())) + ("preview_inline_hud_counts",)
        requested_value_color = str(item.get("outline", _HUD_NEUTRAL_VALUE_COLOR))
        value_color = ensure_inline_hud_canvas_contrast(
            requested_value_color,
            fallback=_HUD_NEUTRAL_VALUE_COLOR,
            min_ratio=5.3,
        )
        if value_color.strip().lower() == hud_label_color.strip().lower():
            value_color = _HUD_NEUTRAL_VALUE_COLOR

        if ":" in raw_text:
            label_raw, value_raw = raw_text.split(":", 1)
            label_text = f"{label_raw.strip()}:"
            value_text = value_raw.strip()

            label_width = count_font.measure(label_text)
            value_width = count_font.measure(value_text)
            item_width = label_width + label_value_gap + value_width

            if x > left and x + item_width > right:
                x, y = left, y + row_height

            max_value_width = max(
                20,
                int(right - x - label_width - label_value_gap),
            )
            value_text = fit_text(value_text, count_font, max_value_width)
            value_width = count_font.measure(value_text)

            lines.append(
                (
                    label_text,
                    x,
                    y,
                    count_font,
                    hud_label_color,
                    tags + ("preview_inline_hud_label",),
                )
            )
            value_x = x + label_width + label_value_gap
            lines.append(
                (
                    value_text,
                    value_x,
                    y,
                    count_font,
                    value_color,
                    tags + ("preview_inline_hud_value",),
                )
            )
            x = value_x + value_width + item_gap
            continue

        text = fit_text(raw_text, count_font, right - left)
        text_width = count_font.measure(text)
        if x > left and x + text_width > right:
            x, y = left, y + row_height
        lines.append(
            (
                text,
                x,
                y,
                count_font,
                value_color,
                tags,
            )
        )
        x += text_width + item_gap

    return {
        "lines": lines,
        "bar_height": float(y + row_height + 6),
        "label_color": hud_label_color,
    }

def draw_inline_hud(host, canvas, plan):
    for text, x, y, font, color, extra_tags in plan["lines"]:
        tags = ("preview_overlay", "preview_inline_hud") + extra_tags
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            canvas.create_text(x + dx, y + dy, anchor="nw", text=text, font=font,
                               fill="#101820", tags=tags + ("preview_inline_hud_outline",))
        canvas.create_text(x, y, anchor="nw", text=text, font=font, fill=color, tags=tags)
    host._preview_hud_bottom_in_view = plan["bar_height"]
