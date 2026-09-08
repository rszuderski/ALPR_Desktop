#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wspólny helper bezwładności scrolla dla zakładek GUI.
"""

from __future__ import annotations


class InertialScrollController:
    def __init__(self, scheduler_widget, *, decay: float = 0.78, interval_ms: int = 14):
        self.scheduler_widget = scheduler_widget
        self.decay = float(decay)
        self.interval_ms = int(interval_ms)
        self._jobs: dict[str, dict] = {}

    @staticmethod
    def mousewheel_units(event) -> int:
        event_num = getattr(event, "num", None)
        if event_num == 4:
            return -1
        if event_num == 5:
            return 1

        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        if abs(delta) >= 120:
            units = -int(delta / 120)
        else:
            units = -1 if delta > 0 else 1
        return units if units != 0 else (-1 if delta > 0 else 1)

    @staticmethod
    def mousewheel_magnitude(event) -> float:
        event_num = getattr(event, "num", None)
        if event_num in {4, 5}:
            return 1.0
        try:
            delta = abs(int(getattr(event, "delta", 0) or 0))
        except Exception:
            delta = 0
        if delta <= 0:
            return 1.0
        return max(1.0, min(3.0, float(delta) / 120.0))

    @staticmethod
    def widget_contains_point(widget, x_root: int, y_root: int) -> bool:
        if widget is None:
            return False
        try:
            wx = int(widget.winfo_rootx())
            wy = int(widget.winfo_rooty())
            return wx <= x_root < (wx + int(widget.winfo_width())) and wy <= y_root < (wy + int(widget.winfo_height()))
        except Exception:
            return False

    @staticmethod
    def _resolve_pointer_root(event, pointer_widget):
        try:
            x_root = int(getattr(event, "x_root", 0) or pointer_widget.winfo_pointerx())
            y_root = int(getattr(event, "y_root", 0) or pointer_widget.winfo_pointery())
            return x_root, y_root
        except Exception:
            return None, None

    @staticmethod
    def canvas_overflows(canvas) -> bool:
        if canvas is None:
            return False
        try:
            bbox = canvas.bbox("all")
            if not bbox:
                return False
            content_height = int(bbox[3]) - int(bbox[1])
            viewport_height = int(canvas.winfo_height())
            return content_height > viewport_height + 1
        except Exception:
            return False

    @staticmethod
    def canvas_can_scroll(canvas, units: int) -> bool:
        if canvas is None or units == 0:
            return False
        try:
            first, last = canvas.yview()
            if units < 0 and float(first) <= 0.0:
                return False
            if units > 0 and float(last) >= 1.0:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def listbox_can_scroll(listbox, units: int) -> bool:
        if listbox is None or units == 0:
            return False
        try:
            first, last = listbox.yview()
            if units < 0 and float(first) <= 0.0:
                return False
            if units > 0 and float(last) >= 1.0:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _compute_canvas_scroll_delta(canvas, units: int, *, magnitude: float = 1.0) -> float:
        if canvas is None or units == 0:
            return 0.0
        try:
            first, last = canvas.yview()
            span = max(0.02, float(last) - float(first))
            base_step = max(0.0035, min(0.018, span * 0.08))
            step = base_step * max(1.0, min(3.0, float(magnitude or 1.0)))
            return float(units) * step
        except Exception:
            return 0.0

    @staticmethod
    def _compute_listbox_scroll_delta(listbox, units: int, *, magnitude: float = 1.0) -> float:
        if listbox is None or units == 0:
            return 0.0
        try:
            total_items = max(1, int(listbox.size() or 0))
            first, last = listbox.yview()
            span = max(1e-6, float(last) - float(first))
            visible_rows = max(1.0, float(span) * float(total_items))

            # Dla dużych list chcemy przewijać raczej o kilka wierszy,
            # a nie o stały procent całej zawartości. W przeciwnym razie
            # paczka 3000 obrazów skacze o dziesiątki pozycji na jeden ruch rolką.
            if total_items <= 80:
                base_rows = 1.35
            elif total_items <= 300:
                base_rows = 1.15
            elif total_items <= 1200:
                base_rows = 1.0
            else:
                base_rows = 0.85

            if visible_rows <= 8.0:
                base_rows *= 0.9
            elif visible_rows >= 22.0:
                base_rows *= 1.1

            rows_per_tick = max(0.75, min(2.25, base_rows * max(1.0, min(3.0, float(magnitude or 1.0)))))
            step = rows_per_tick / float(total_items)
            return float(units) * step
        except Exception:
            return 0.0

    @staticmethod
    def _apply_canvas_scroll_delta(canvas, delta_fraction: float) -> bool:
        if canvas is None or abs(float(delta_fraction or 0.0)) < 1e-6:
            return False
        try:
            first, last = canvas.yview()
            first = float(first)
            last = float(last)
            span = max(0.02, last - first)
            target = max(0.0, min(max(0.0, 1.0 - span), first + float(delta_fraction)))
            if abs(target - first) < 1e-6:
                return False
            canvas.yview_moveto(target)
            return True
        except Exception:
            return False

    @staticmethod
    def _apply_listbox_scroll_delta(listbox, delta_fraction: float) -> bool:
        if listbox is None or abs(float(delta_fraction or 0.0)) < 1e-6:
            return False
        try:
            first, last = listbox.yview()
            first = float(first)
            last = float(last)
            # Listbox może mieć bardzo mały visible span dla dużych paczek;
            # sztuczne podbijanie go do 0.02 blokowało dojazd do końca listy.
            span = max(1e-6, last - first)
            target = max(0.0, min(max(0.0, 1.0 - span), first + float(delta_fraction)))
            if abs(target - first) < 1e-6:
                return False
            listbox.yview_moveto(target)
            return True
        except Exception:
            return False

    def _queue_inertia(self, widget, *, mode: str, delta_fraction: float) -> bool:
        if widget is None:
            return False
        delta_fraction = float(delta_fraction or 0.0)
        if abs(delta_fraction) < 1e-6:
            return False

        key = f"{str(mode or '').strip().lower()}:{str(widget)}"
        state = self._jobs.get(key)
        if not isinstance(state, dict):
            state = {
                "widget": widget,
                "mode": str(mode or "").strip().lower(),
                "velocity": 0.0,
                "after_id": None,
            }
            self._jobs[key] = state

        velocity = float(state.get("velocity", 0.0) or 0.0) + delta_fraction
        state["velocity"] = max(-0.14, min(0.14, velocity))

        if state.get("after_id") is None:
            try:
                state["after_id"] = self.scheduler_widget.after(
                    self.interval_ms,
                    lambda scroll_key=key: self._advance_inertia(scroll_key),
                )
            except Exception:
                state["after_id"] = None
                return False
        return True

    def _advance_inertia(self, key: str):
        state = self._jobs.get(str(key or "").strip())
        if not isinstance(state, dict):
            return

        state["after_id"] = None
        widget = state.get("widget")
        mode = str(state.get("mode", "") or "").strip().lower()
        velocity = float(state.get("velocity", 0.0) or 0.0)
        min_velocity = 0.0007 if mode == "canvas" else 0.00008
        if abs(velocity) < min_velocity:
            self._jobs.pop(key, None)
            return

        if mode == "canvas":
            moved = self._apply_canvas_scroll_delta(widget, velocity)
        else:
            moved = self._apply_listbox_scroll_delta(widget, velocity)

        if not moved:
            self._jobs.pop(key, None)
            return

        state["velocity"] = velocity * self.decay
        if abs(float(state.get("velocity", 0.0) or 0.0)) < min_velocity:
            self._jobs.pop(key, None)
            return

        try:
            state["after_id"] = self.scheduler_widget.after(
                self.interval_ms,
                lambda scroll_key=key: self._advance_inertia(scroll_key),
            )
        except Exception:
            self._jobs.pop(key, None)

    def queue_canvas_by_units(self, canvas, units: int, *, magnitude: float = 1.0) -> bool:
        delta = self._compute_canvas_scroll_delta(canvas, units, magnitude=magnitude)
        return self._queue_inertia(canvas, mode="canvas", delta_fraction=delta)

    def queue_listbox_by_units(self, listbox, units: int, *, magnitude: float = 1.0) -> bool:
        delta = self._compute_listbox_scroll_delta(listbox, units, magnitude=magnitude)
        return self._queue_inertia(listbox, mode="listbox", delta_fraction=delta)

    def scroll_canvas_if_targeted(self, canvas, event, *, pointer_widget, overflow_checker=None) -> bool:
        if canvas is None:
            return False
        units = self.mousewheel_units(event)
        if units == 0:
            return False

        x_root, y_root = self._resolve_pointer_root(event, pointer_widget)
        if x_root is None or not self.widget_contains_point(canvas, x_root, y_root):
            return False

        can_overflow = self.canvas_overflows(canvas)
        if callable(overflow_checker):
            try:
                can_overflow = bool(overflow_checker())
            except Exception:
                pass
        if not can_overflow or not self.canvas_can_scroll(canvas, units):
            return False

        return self.queue_canvas_by_units(canvas, units, magnitude=self.mousewheel_magnitude(event))

    def redirect_child_mousewheel_to_canvas(self, event, canvas, *, pointer_widget, overflow_checker=None) -> bool:
        if canvas is None:
            return False
        units = self.mousewheel_units(event)
        if units == 0:
            return False

        x_root, y_root = self._resolve_pointer_root(event, pointer_widget)
        if x_root is None or not self.widget_contains_point(canvas, x_root, y_root):
            return False

        can_overflow = True
        if callable(overflow_checker):
            try:
                can_overflow = bool(overflow_checker())
            except Exception:
                can_overflow = True
        if not can_overflow or not self.canvas_can_scroll(canvas, units):
            return False

        return self.queue_canvas_by_units(canvas, units, magnitude=self.mousewheel_magnitude(event))
