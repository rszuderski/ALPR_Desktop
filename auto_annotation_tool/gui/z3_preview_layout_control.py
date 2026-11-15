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
    """Draw a retained 1R/2R selector physically anchored to the plate frame."""
    canvas = host.preview_canvas
    palette = getattr(host.app, "palette", {})

    background = palette.get("field", "#202830")
    foreground = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#aeb7bf")
    accent = palette.get("accent", "#4f8de3")
    border = blend_hex_colors(background, foreground, 0.52)
    inactive_fill = blend_hex_colors(background, foreground, 0.06)
    active_fill = blend_hex_colors(background, accent, 0.44)

    # Pionowy przełącznik siedzi po PRAWEJ stronie ramki tablicy.
    # Dzięki temu nie przykrywa badge'y nad ramką.
    width = 44.0
    height = 58.0
    gap = 6.0

    # `right` i `top` to prawy-górny narożnik ramki statusu.
    # Nie przycinamy do viewportu: podczas pan/zoom kontrolka ma pozostać
    # fizycznie przyklejona do ramki tablicy.
    frame_right = float(right)
    anchor_top = float(top)
    left = frame_right + gap
    upper = anchor_top
    control_right = left + width
    lower = upper + height
    middle_y = upper + (height / 2.0)
    center_x = left + (width / 2.0)

    two_rows = bool(host._should_preview_use_two_row_layers(data))
    row_count = 2 if two_rows else 1

    items = getattr(host, "_preview_layout_control_items", None)
    live = bool(
        isinstance(items, dict)
        and items.get("canvas") is canvas
        and items.get("shell") is not None
        and canvas.type(items.get("shell"))
    )

    if not live:
        hide_layout_tip(host)
        try:
            canvas.delete(CONTROL_TAG)
        except Exception:
            pass

        for sequence, binding in getattr(
            host,
            "_preview_layout_control_bindings",
            (),
        ):
            try:
                canvas.tag_unbind(CONTROL_TAG, sequence, binding)
            except Exception:
                pass

        tags = (CONTROL_TAG, f"preview_action::{ACTION}")
        shell = canvas.create_rectangle(
            left,
            upper,
            control_right,
            lower,
            fill=background,
            outline=border,
            width=1,
            tags=tags,
        )
        left_fill = canvas.create_rectangle(
            left + 2,
            upper + 2,
            control_right - 2,
            middle_y - 1,
            fill=inactive_fill,
            outline="",
            tags=tags,
        )
        right_fill = canvas.create_rectangle(
            left + 2,
            middle_y + 1,
            control_right - 2,
            lower - 2,
            fill=inactive_fill,
            outline="",
            tags=tags,
        )
        divider = canvas.create_line(
            left + 4,
            middle_y,
            control_right - 4,
            middle_y,
            fill=border,
            width=1,
            tags=tags,
        )
        left_text = canvas.create_text(
            center_x,
            upper + (height * 0.25),
            text="1R",
            fill=foreground,
            font=("Segoe UI", 9, "bold"),
            tags=tags,
        )
        right_text = canvas.create_text(
            center_x,
            upper + (height * 0.75),
            text="2R",
            fill=foreground,
            font=("Segoe UI", 9, "bold"),
            tags=tags,
        )
        items = {
            "canvas": canvas,
            "shell": shell,
            "left_fill": left_fill,
            "right_fill": right_fill,
            "divider": divider,
            "left_text": left_text,
            "right_text": right_text,
        }
        host._preview_layout_control_items = items

        def show_tip():
            host._preview_layout_tip_after_id = None
            if not canvas.find_withtag(CONTROL_TAG):
                return
            tip_y = upper - 5.0
            anchor = tk.SE
            if tip_y < 42.0:
                tip_y = lower + 7.0
                anchor = tk.NE
            text_id = canvas.create_text(
                control_right,
                tip_y,
                text=(
                    f"Układ tablicy: {'2 rzędy' if two_rows else '1 rząd'}\n"
                    "LPM: przełącz 1R ↔ 2R   •   PPM: wybór / AUTO"
                ),
                anchor=anchor,
                justify=tk.LEFT,
                font=("Segoe UI", 9),
                fill=foreground,
                tags=(TIP_TAG,),
            )
            box = canvas.bbox(text_id)
            if box:
                backdrop = canvas.create_rectangle(
                    box[0] - 6,
                    box[1] - 5,
                    box[2] + 6,
                    box[3] + 5,
                    fill=background,
                    outline=border,
                    tags=(TIP_TAG,),
                )
                canvas.tag_raise(text_id, backdrop)

        def enter(_event):
            hide_layout_tip(host)
            try:
                canvas.itemconfigure(
                    items["shell"],
                    outline=foreground,
                    width=2,
                )
            except Exception:
                pass
            host._preview_layout_tip_after_id = canvas.after(350, show_tip)

        def leave(_event):
            hide_layout_tip(host)
            try:
                canvas.itemconfigure(
                    items["shell"],
                    outline=border,
                    width=1,
                )
            except Exception:
                pass

        host._preview_layout_control_bindings = [
            (
                "<Enter>",
                canvas.tag_bind(CONTROL_TAG, "<Enter>", enter),
            ),
            (
                "<Leave>",
                canvas.tag_bind(CONTROL_TAG, "<Leave>", leave),
            ),
        ]

    # Retained-mode update: same item IDs, only geometry and state change.
    canvas.coords(items["shell"], left, upper, control_right, lower)
    canvas.coords(
        items["left_fill"],
        left + 2,
        upper + 2,
        control_right - 2,
        middle_y - 1,
    )
    canvas.coords(
        items["right_fill"],
        left + 2,
        middle_y + 1,
        control_right - 2,
        lower - 2,
    )
    canvas.coords(
        items["divider"],
        left + 4,
        middle_y,
        control_right - 4,
        middle_y,
    )
    canvas.coords(
        items["left_text"],
        center_x,
        upper + (height * 0.25),
    )
    canvas.coords(
        items["right_text"],
        center_x,
        upper + (height * 0.75),
    )

    canvas.itemconfigure(
        items["left_fill"],
        fill=active_fill if not two_rows else inactive_fill,
    )
    canvas.itemconfigure(
        items["right_fill"],
        fill=active_fill if two_rows else inactive_fill,
    )
    canvas.itemconfigure(
        items["left_text"],
        fill=foreground if not two_rows else muted,
    )
    canvas.itemconfigure(
        items["right_text"],
        fill=foreground if two_rows else muted,
    )
    canvas.itemconfigure(items["divider"], fill=border)

    try:
        canvas.dtag(items["shell"], "preview_plate_layout_rows::1")
        canvas.dtag(items["shell"], "preview_plate_layout_rows::2")
    except Exception:
        pass
    canvas.addtag_withtag(
        f"preview_plate_layout_rows::{row_count}",
        items["shell"],
    )

    host._preview_layout_control_signature = (
        id(canvas),
        id(data),
        float(frame_right),
        float(anchor_top),
        int(row_count),
        active_fill,
        inactive_fill,
        foreground,
        muted,
        border,
    )
    host._preview_layout_control_anchor = {
        "right": float(frame_right),
        "top": float(anchor_top),
        "left": float(left),
        "upper": float(upper),
        "control_right": float(control_right),
        "lower": float(lower),
        "width": float(width),
        "height": float(height),
        "gap": float(gap),
    }
    canvas.tag_raise(CONTROL_TAG)
    canvas.tag_raise(TIP_TAG)


def toggle_plate_rows(host):
    data = host._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return "break"
    from .z3_plate_layout_runtime import _apply_preview_plate_layout_override

    hide_layout_tip(host)
    next_layout = "single_row" if host._should_preview_use_two_row_layers(data) else "two_row"
    return _apply_preview_plate_layout_override(host, next_layout, source="frame_toggle")
