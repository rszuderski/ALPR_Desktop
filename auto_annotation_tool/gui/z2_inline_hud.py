"""Text-only context shared by windowed and fullscreen previews."""

import tkinter.font as tkfont


def fit_text(text, font, width):
    """Middle ellipsis keeps both the registration prefix and file suffix."""
    text = str(text)
    if font.measure(text) <= width:
        return text
    low, high = 0, len(text)
    while low < high:
        count = (low + high + 1) // 2
        head = (count + 1) // 2
        tail = count // 2
        candidate = text[:head] + "…" + (text[-tail:] if tail else "")
        if font.measure(candidate) <= width:
            low = count
        else:
            high = count - 1
    head, tail = (low + 1) // 2, low // 2
    return text[:head] + "…" + (text[-tail:] if tail else "")


def draw_inline_context(owner, canvas, *, left, top, right, combo_width, combo_height):
    """Draw context at the left edge; return a collision-free status position."""
    context = owner._get_preview_legend_context()
    annotation = owner._get_preview_annotation()
    filename = str(getattr(annotation, "filename", "") or context.get("filename", ""))
    image_count = str(context.get("image_text", "")).partition(":")[2].strip()
    image_number = image_count.partition("/")[0] or "0"
    # Use the same visible-list number as the list, including its zero padding.
    total = image_count.partition("/")[2]
    image_number = image_number.zfill(max(2, len(total)))
    fonts = getattr(owner, "_preview_inline_hud_fonts", None)
    if fonts is None:
        fonts = (
            tkfont.Font(owner.frame, family="Segoe UI", size=14, weight="bold"),
            tkfont.Font(owner.frame, family="Segoe UI", size=11, weight="bold"),
        )
        owner._preview_inline_hud_fonts = fonts
    file_font, count_font = fonts
    line_height = file_font.metrics("linespace")
    height = line_height + 3 + count_font.metrics("linespace")
    x, y = left + 12, top + 10
    combo_x = max(left + 12, right - combo_width - 12)
    available = combo_x - x - 20
    combo_y = y + max(0, (height - combo_height) / 2)
    if available < 210:
        # Narrow panes wrap the controls, never the filename or the HUD anchor.
        available = max(40, right - x - 12)
        combo_y = y + height + 8

    tags = ("preview_overlay", "preview_plate_combo_overlay", "preview_inline_hud",
            getattr(canvas, "VIEWPORT_FIXED_TAG", "preview_viewport_fixed"))

    def lettering(text, px, py, font, color, role):
        # A glyph outline is legible on both white plates and dark photographs.
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            canvas.create_text(px + dx, py + dy, anchor="nw", text=text, font=font,
                               fill="#101820", tags=tags + ("preview_inline_hud_outline",))
        canvas.create_text(px, py, anchor="nw", text=text, font=font, fill=color,
                           tags=tags + (role,))

    number = f"{image_number}. "
    number_width = file_font.measure(number)
    lettering(number, x, y, file_font, "#ffd166", "preview_inline_hud_number")
    lettering(fit_text(filename, file_font, available - number_width), x + number_width,
              y, file_font, "#7ee7ff", "preview_inline_hud_filename")
    counters = [f"Zdjęcie {image_count}"]
    vehicle = str(context.get("vehicle_text", "")).partition(":")[2].strip()
    if vehicle and vehicle != "0/0":
        counters.append(f"Pojazd {vehicle}")
    pool = str(context.get("pool_text", "")).strip()
    if pool:
        counters.append(pool)
    lettering(fit_text("  ·  ".join(counters), count_font, available), x,
              y + line_height + 3, count_font, "#b6f5ce", "preview_inline_hud_counts")
    owner._preview_hud_bottom_in_view = max(y + height, combo_y + combo_height) - top
    return combo_x, combo_y
