"""Canvas-native row-layout control attached to the plate frame."""

import tkinter as tk

from .web_slim_scrollbar import blend_hex_colors


CONTROL_TAG = "preview_plate_layout_control"
TIP_TAG = "preview_plate_layout_tip"
ACTION = "toggle_plate_rows"


def hide_layout_tip(host):
    canvas = host.preview_canvas
    pending = getattr(host, "_preview_layout_tip_after_id", None)
    if pending:
        canvas.after_cancel(pending)
    host._preview_layout_tip_after_id = None
    canvas.delete(TIP_TAG)


def draw_plate_layout_control(host, data, *, right, top):
    canvas = host.preview_canvas
    palette = getattr(host.app, "palette", {})
    background = palette.get("field", "#202830")
    foreground = palette.get("fg", "#f3f3f3")
    border = blend_hex_colors(background, foreground, 0.55)
    fill = blend_hex_colors(background, "#0064e0", 0.16)
    hover_fill = blend_hex_colors(background, "#0064e0", 0.36)
    width, height = 54, 34
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    two_rows = bool(host._should_preview_use_two_row_layers(data))
    signature = (id(canvas), id(data), right, top, canvas_width, canvas_height, two_rows, fill, foreground, border)
    if (getattr(host, "_preview_layout_control_signature", None) == signature
            and canvas.find_withtag(CONTROL_TAG)):
        canvas.tag_raise(CONTROL_TAG)
        canvas.tag_raise(TIP_TAG)
        return
    hide_layout_tip(host)
    canvas.delete(CONTROL_TAG)
    for sequence, binding in getattr(host, "_preview_layout_control_bindings", ()):
        canvas.tag_unbind(CONTROL_TAG, sequence, binding)
    host._preview_layout_control_signature = signature
    left = max(4.0, min(right - width / 2, canvas_width - width - 4.0))
    upper = max(4.0, min(top - height / 2, canvas_height - height - 4.0))
    right, lower = left + width, upper + height
    # Keep this control outside transient HUD tags: status pulses redraw the
    # header, but must not remove the hovered control or reset its tooltip timer.
    tags = (CONTROL_TAG, f"preview_action::{ACTION}")
    radius = 7
    shell = canvas.create_polygon(
        left + radius, upper, right - radius, upper, right, upper, right, upper + radius,
        right, lower - radius, right, lower, right - radius, lower, left + radius, lower,
        left, lower, left, lower - radius, left, upper + radius, left, upper,
        smooth=True, splinesteps=12, fill=fill, outline=border, width=1, tags=tags,
    )
    row_count = 2 if two_rows else 1
    canvas.addtag_withtag(f"preview_plate_layout_rows::{row_count}", shell)
    # A miniature plate with actual glyph rows, plus a compact change arrow.
    x, y = left + 7, upper + 8
    canvas.create_rectangle(x, y, x + 27, y + 18, outline=foreground, width=1, tags=tags)
    rows = ((y + 4, 4), (y + 11, 4)) if two_rows else ((y + 6, 7),)
    for row_y, glyph_height in rows:
        for column in range(4):
            glyph_x = x + 4 + column * 5
            canvas.create_rectangle(glyph_x, row_y, glyph_x + 2, row_y + glyph_height,
                                    fill=foreground, outline="", tags=tags)
    arrow_x, arrow_y = left + 43, upper + 17
    canvas.create_line(arrow_x - 3, arrow_y - 7, arrow_x + 3, arrow_y - 7,
                       arrow_x + 3, arrow_y + 7, arrow_x - 3, arrow_y + 7,
                       fill=foreground, width=1.5, arrow=tk.LAST, arrowshape=(4, 5, 2), tags=tags)

    def show_tip():
        host._preview_layout_tip_after_id = None
        if not canvas.find_withtag(CONTROL_TAG):
            return
        tip_y = upper - 4 if upper >= 54 else lower + 8
        anchor = tk.SE if upper >= 54 else tk.NE
        text = canvas.create_text(
            right, tip_y, text=f"Zmień na {'1 rząd' if two_rows else '2 rzędy'}\nPPM: wybór / AUTO",
            anchor=anchor, justify=tk.LEFT, font=("Segoe UI", 9), fill=foreground, tags=(TIP_TAG,),
        )
        box = canvas.bbox(text)
        if box[0] < 6:
            canvas.move(text, 6 - box[0], 0)
            box = canvas.bbox(text)
        backdrop = canvas.create_rectangle(box[0] - 6, box[1] - 5, box[2] + 6, box[3] + 5,
                                            fill=background, outline=border, tags=(TIP_TAG,))
        canvas.tag_raise(text, backdrop)

    def enter(_event):
        hide_layout_tip(host)
        canvas.itemconfigure(shell, fill=hover_fill, outline=foreground)
        host._preview_layout_tip_after_id = canvas.after(350, show_tip)

    def leave(_event):
        hide_layout_tip(host)
        canvas.itemconfigure(shell, fill=fill, outline=border)

    host._preview_layout_control_bindings = [
        ("<Enter>", canvas.tag_bind(CONTROL_TAG, "<Enter>", enter)),
        ("<Leave>", canvas.tag_bind(CONTROL_TAG, "<Leave>", leave)),
    ]
    canvas.tag_raise(CONTROL_TAG)


def toggle_plate_rows(host):
    data = host._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return "break"
    from .z3_plate_layout_runtime import _apply_preview_plate_layout_override

    hide_layout_tip(host)
    next_layout = "single_row" if host._should_preview_use_two_row_layers(data) else "two_row"
    return _apply_preview_plate_layout_override(host, next_layout, source="frame_toggle")
