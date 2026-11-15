"""Compact, wrapping provenance legend outside the zoomable plate canvas."""

import tkinter as tk
from tkinter import font as tkfont

from .z3_preview_badges import get_preview_badge_component_style


class CharacterBadgeLegend(tk.Canvas):
    COMPONENTS = (
        "yolo_box", "generated_box", "manual_box",
        "yolo_symbol", "ocr_symbol", "manual_sign", "gt_assisted",
    )

    def __init__(self, parent, host):
        super().__init__(parent, background="#1e1e1e", height=28, width=1,
                         bd=0, highlightthickness=0, takefocus=0)
        self.host = host
        self._badge_font = tkfont.Font(root=self, family="Segoe UI", size=9, weight="bold")
        self._label_font = tkfont.Font(root=self, family="Segoe UI", size=9)
        self._last_width = None
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        if event.width != self._last_width:
            self.refresh()

    def _actions_gap_bounds(self, width):
        """Return the horizontal lane occupied by the fullscreen Actions tab."""
        if not bool(getattr(self.host, "_preview_fullscreen_active", False)):
            return None

        drawers = getattr(self.host, "_preview_workspace_drawers", None)
        if drawers is None:
            return None
        try:
            button = drawers.buttons.get("bottom")
        except Exception:
            button = None
        if button is None:
            return None

        try:
            if str(button.winfo_manager() or "") != "place":
                return None
            self.update_idletasks()
            button.update_idletasks()
            button_left = float(button.winfo_rootx()) - float(self.winfo_rootx())
            button_width = float(max(1, button.winfo_width(), button.winfo_reqwidth()))
        except Exception:
            return None

        pad = 16.0
        min_gap = max(116.0, button_width + (pad * 2.0))
        center = button_left + (button_width / 2.0)
        left = center - (min_gap / 2.0)
        right = center + (min_gap / 2.0)

        margin = 8.0
        if left < margin:
            right += margin - left
            left = margin
        if right > float(width) - margin:
            shift = right - (float(width) - margin)
            left -= shift
            right -= shift

        if left <= margin or right >= float(width) - margin or right <= left:
            return None
        return float(left), float(right)

    def refresh(self):
        width = max(1, self.winfo_width())
        self._last_width = width
        self.delete("all")

        margin = 8.0
        item_gap = 14.0
        inner_gap = 12.0
        y = 5.0
        row_height = max(
            self._badge_font.metrics("linespace"),
            self._label_font.metrics("linespace"),
        ) + 8.0

        prepared = []
        for component in self.COMPONENTS:
            style = get_preview_badge_component_style(self.host, component)
            badge_width = float(self._badge_font.measure(style["label"]) + 10)
            label_width = float(self._label_font.measure(style["legend"]))
            prepared.append(
                {
                    "component": component,
                    "style": style,
                    "badge_width": badge_width,
                    "label_width": label_width,
                    "item_width": badge_width + 5.0 + label_width,
                }
            )

        def draw_item(item, x, row_y):
            style = item["style"]
            badge_width = float(item["badge_width"])
            tags = ("legend_item", item["component"])
            self.create_rectangle(
                x,
                row_y,
                x + badge_width,
                row_y + row_height - 4,
                fill=style["badge_fill"],
                outline=style["badge_outline"],
                tags=tags,
            )
            self.create_text(
                x + badge_width / 2.0,
                row_y + (row_height - 4) / 2.0,
                text=style["label"],
                fill=style["badge_fg"],
                font=self._badge_font,
                tags=tags,
            )
            self.create_text(
                x + badge_width + 5.0,
                row_y + (row_height - 4) / 2.0,
                text=style["legend"],
                fill="#e1e3e8",
                font=self._label_font,
                anchor="w",
                tags=tags,
            )
            return x + float(item["item_width"])

        hint = "Liczba 0–1: pewność modelu"
        hint_width = float(self._label_font.measure(hint))
        actions_gap = self._actions_gap_bounds(width)

        # Fullscreen composition:
        # YB / GB / MB / YS | [ Actions tab ] | OS / MS / GT / confidence hint
        if actions_gap is not None and len(prepared) == 7:
            gap_left, gap_right = actions_gap
            left_items = prepared[:4]
            right_items = prepared[4:]

            left_width = (
                sum(float(item["item_width"]) for item in left_items)
                + item_gap * max(0, len(left_items) - 1)
            )
            right_width = (
                sum(float(item["item_width"]) for item in right_items)
                + item_gap * max(0, len(right_items) - 1)
                + item_gap
                + hint_width
            )

            left_lane_left = margin
            left_lane_right = gap_left - inner_gap
            right_lane_left = gap_right + inner_gap
            right_lane_right = float(width) - margin

            left_capacity = left_lane_right - left_lane_left
            right_capacity = right_lane_right - right_lane_left

            if left_width <= left_capacity and right_width <= right_capacity:
                left_start = (
                    left_lane_left
                    + max(0.0, (left_capacity - left_width) / 2.0)
                )
                right_start = (
                    right_lane_left
                    + max(0.0, (right_capacity - right_width) / 2.0)
                )

                x = left_start
                for idx, item in enumerate(left_items):
                    x = draw_item(item, x, y)
                    if idx < len(left_items) - 1:
                        x += item_gap

                x = right_start
                for idx, item in enumerate(right_items):
                    x = draw_item(item, x, y)
                    x += item_gap

                self.create_text(
                    x,
                    y + (row_height - 4) / 2.0,
                    text=hint,
                    fill="#bfc4cc",
                    font=self._label_font,
                    anchor="w",
                    tags="legend_hint",
                )
                self.configure(height=int(y + row_height + 1))
                return

        # Normal mode / narrow fallback: original wrapping behaviour.
        x = margin
        for item in prepared:
            item_width = float(item["item_width"])
            if x > margin and x + item_width > float(width) - margin:
                x, y = margin, y + row_height
            x = draw_item(item, x, y) + item_gap

        if x > margin and x + hint_width > float(width) - margin:
            x, y = margin, y + row_height
        self.create_text(
            x,
            y + (row_height - 4) / 2.0,
            text=hint,
            fill="#bfc4cc",
            font=self._label_font,
            anchor="w",
            tags="legend_hint",
        )
        self.configure(height=int(y + row_height + 1))
