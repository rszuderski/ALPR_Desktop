#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canvas z obsługą zoom'u i pan'u dla podglądu obrazów.
Zastosowanie: Podgląd w zakładce Prostowania Tablic.
"""

import math
import logging
import time
import tkinter as tk
from types import SimpleNamespace
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class ZoomableCanvas(tk.Canvas):
    """Canvas z obsługą zoom'u (kółko myszy) i pan'u (LPM + przeciąganie)."""
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        try:
            self.configure(takefocus=True)
        except Exception:
            pass
        
        # Stan zoom'u
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 1.12

        # Oryginalne dane
        self.original_image = None  # PIL Image
        self.photo_image = None     # ImageTk.PhotoImage
        self.image_id = None        # Canvas image ID
        self.overlay_renderer = None
        self.interaction_delegate = None
        self._render_region = None
        self._pan_buffered_move_active = False
        self._deferred_display_after_id = None
        self._deferred_display_fast_mode = False
        self._final_quality_after_id = None
        self._zoom_animation_after_id = None
        self._zoom_animation_target_zoom = None
        self._interaction_fast_rendering = False
        self._middle_click_zoom_restore_state = None
        self._middle_click_zoom_last_trigger_at = 0.0
        self._middle_click_zoom_ignore_release_until = 0.0
        
        # Pan'u - przesuwanie widoku
        self.pan_data = {
            'x': 0,
            'y': 0,
            'press_x': None,
            'press_y': None
        }
        
        # Flaga pokazania tekstu Info
        self.show_info = True
        self.reset_shortcut_enabled = True
        
        # Bind'y zdarzeń - Zoom
        self.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.bind("<Button-4>", self._on_mousewheel)     # Linux scroll up
        self.bind("<Button-5>", self._on_mousewheel)     # Linux scroll down
        self.bind("<Button-2>", self._on_middle_click_zoom)
        self.bind("<ButtonRelease-2>", self._on_middle_click_zoom)
        
        # Bind'y - Pan (przeciąganie LPM)
        self.bind("<Button-1>", self._on_pan_press)
        self.bind("<B1-Motion>", self._on_pan_motion)
        self.bind("<ButtonRelease-1>", self._on_pan_release)
        
        # Bind'y - klawiatura
        self.bind("<Home>", self._on_reset_view)  # Home = reset zoom i pan
        self.bind("<r>", self._on_reset_view)     # r = reset
        self.bind("<R>", self._on_reset_view)
        self.bind("<i>", self._on_toggle_info)    # i = toggle info
        self.bind("<I>", self._on_toggle_info)
        
        # Kursor
        self.current_cursor = "arrow"

    def _perf_probe_active(self) -> bool:
        try:
            return time.perf_counter() <= float(getattr(self, "_perf_probe_until", 0.0) or 0.0)
        except Exception:
            return False

    def _perf_probe_label(self) -> str:
        return str(getattr(self, "_perf_probe_label_value", "") or "-")

    def _log_perf(self, operation: str, elapsed_ms: float, *, threshold_ms: float = 80.0, **details) -> None:
        if not self._perf_probe_active() and float(elapsed_ms) < float(threshold_ms):
            return
        try:
            suffix = " ".join(f"{key}={value}" for key, value in details.items())
            if suffix:
                suffix = " " + suffix
            logger.info(
                "[Z2 CANVAS PERF] %s %.1fms label=%s%s",
                operation,
                float(elapsed_ms),
                self._perf_probe_label(),
                suffix,
            )
        except Exception:
            pass

    def _get_canvas_size(self):
        return (
            max(1, int(self.winfo_width() or 1)),
            max(1, int(self.winfo_height() or 1)),
        )

    def _get_full_image_size(self):
        if self.original_image is None:
            return 0.0, 0.0
        return (
            max(1.0, float(self.original_image.width) * float(self.zoom_level)),
            max(1.0, float(self.original_image.height) * float(self.zoom_level)),
        )

    def _clamp_origin(self, origin_x: float, origin_y: float):
        if self.original_image is None:
            return float(origin_x), float(origin_y)

        canvas_width, canvas_height = self._get_canvas_size()
        full_width, full_height = self._get_full_image_size()
        clamped_x = float(origin_x)
        clamped_y = float(origin_y)

        if full_width <= float(canvas_width):
            clamped_x = (float(canvas_width) - full_width) / 2.0
        else:
            min_x = float(canvas_width) - full_width
            clamped_x = min(0.0, max(min_x, clamped_x))

        if full_height <= float(canvas_height):
            clamped_y = (float(canvas_height) - full_height) / 2.0
        else:
            min_y = float(canvas_height) - full_height
            clamped_y = min(0.0, max(min_y, clamped_y))

        return clamped_x, clamped_y

    def _apply_clamped_pan(self):
        clamped_x, clamped_y = self._clamp_origin(
            float(self.pan_data.get('x', 0.0)),
            float(self.pan_data.get('y', 0.0)),
        )
        self.pan_data = {
            'x': float(clamped_x),
            'y': float(clamped_y),
            'press_x': self.pan_data.get('press_x'),
            'press_y': self.pan_data.get('press_y'),
        }
        return float(clamped_x), float(clamped_y)

    def _get_exact_visible_image_bounds(self):
        if self.original_image is None:
            return None

        canvas_width, canvas_height = self._get_canvas_size()
        zoom = max(1e-9, float(self.zoom_level))
        origin_x = float(self.pan_data.get('x', 0.0))
        origin_y = float(self.pan_data.get('y', 0.0))
        image_width = int(self.original_image.width)
        image_height = int(self.original_image.height)

        visible_left = max(0.0, (0.0 - origin_x) / zoom)
        visible_top = max(0.0, (0.0 - origin_y) / zoom)
        visible_right = min(float(image_width), (float(canvas_width) - origin_x) / zoom)
        visible_bottom = min(float(image_height), (float(canvas_height) - origin_y) / zoom)

        if visible_right <= visible_left or visible_bottom <= visible_top:
            return None

        return (
            float(visible_left),
            float(visible_top),
            float(visible_right),
            float(visible_bottom),
        )

    def _get_visible_image_region(self, *, interaction_fast: bool = False):
        if self.original_image is None:
            return None

        canvas_width, canvas_height = self._get_canvas_size()
        zoom = max(1e-9, float(self.zoom_level))
        origin_x = float(self.pan_data.get('x', 0.0))
        origin_y = float(self.pan_data.get('y', 0.0))
        image_width = int(self.original_image.width)
        image_height = int(self.original_image.height)

        exact_bounds = self._get_exact_visible_image_bounds()
        if exact_bounds is None:
            return None

        visible_left, visible_top, visible_right, visible_bottom = exact_bounds

        if interaction_fast:
            buffer_canvas_x = min(max(28.0, float(canvas_width) * 0.12), 96.0)
            buffer_canvas_y = min(max(28.0, float(canvas_height) * 0.12), 96.0)
        else:
            buffer_canvas_x = min(max(64.0, float(canvas_width) * 0.35), 240.0)
            buffer_canvas_y = min(max(64.0, float(canvas_height) * 0.35), 240.0)
        buffer_img_x = buffer_canvas_x / zoom
        buffer_img_y = buffer_canvas_y / zoom

        crop_left = max(0, int(math.floor(visible_left - buffer_img_x)))
        crop_top = max(0, int(math.floor(visible_top - buffer_img_y)))
        crop_right = min(image_width, int(math.ceil(visible_right + buffer_img_x)))
        crop_bottom = min(image_height, int(math.ceil(visible_bottom + buffer_img_y)))

        if crop_right <= crop_left or crop_bottom <= crop_top:
            return None

        return {
            "crop_box": (crop_left, crop_top, crop_right, crop_bottom),
            "draw_x": origin_x + (float(crop_left) * zoom),
            "draw_y": origin_y + (float(crop_top) * zoom),
            "draw_width": max(1, int(math.ceil((crop_right - crop_left) * zoom))),
            "draw_height": max(1, int(math.ceil((crop_bottom - crop_top) * zoom))),
            "visible_bounds": exact_bounds,
        }
    
    def set_image(self, pil_image):
        """Ustaw nowy obraz (PIL.Image) i wyczyść zoom/pan."""
        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._middle_click_zoom_restore_state = None
        self.original_image = pil_image
        self.zoom_level = 1.0
        self.pan_data = {'x': 0, 'y': 0, 'press_x': None, 'press_y': None}
        self._update_display()

    def set_image_fit_to_view(self, pil_image, *, interaction_fast: bool = False):
        """Set a new image and render it already fitted to the canvas."""
        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._middle_click_zoom_restore_state = None
        self.original_image = pil_image
        self.zoom_level = 1.0
        self.pan_data = {'x': 0, 'y': 0, 'press_x': None, 'press_y': None}

        if not self._apply_fit_to_view_geometry():
            try:
                self.after(25, self.fit_to_view)
            except Exception:
                pass
            return

        self._update_display(interaction_fast=bool(interaction_fast))
        if interaction_fast:
            self._schedule_final_quality_display(delay_ms=500)

    def clear_image(self):
        """Wyczysc aktualny obraz i zresetuj stan widoku."""
        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._middle_click_zoom_restore_state = None
        self.original_image = None
        self.photo_image = None
        self.image_id = None
        self._render_region = None
        self._pan_buffered_move_active = False
        self.zoom_level = 1.0
        self.pan_data = {'x': 0, 'y': 0, 'press_x': None, 'press_y': None}
        try:
            self.delete("all")
        except Exception:
            pass
        try:
            self.configure(scrollregion=(0, 0, 0, 0))
        except Exception:
            pass

    def set_image_preserve_view(self, pil_image, *, redraw: bool = True):
        """Ustaw nowy obraz bez zerowania zoomu i pan'u."""
        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._middle_click_zoom_restore_state = None
        self.original_image = pil_image
        self._apply_clamped_pan()
        if redraw:
            self._update_display()

    def set_overlay_renderer(self, renderer):
        """Ustaw callback rysujący overlay po wyrenderowaniu obrazu."""
        self.overlay_renderer = renderer
        if self.original_image is not None:
            self._update_display()

    def set_interaction_delegate(self, delegate):
        """Ustaw delegata przejmującego zdarzenia LPM przed domyślnym panem."""
        self.interaction_delegate = delegate

    def _delegate_interaction(self, action: str, event) -> bool:
        delegate = getattr(self, "interaction_delegate", None)
        if delegate is None:
            return False

        method = getattr(delegate, f"on_zoomable_canvas_{action}", None)
        if not callable(method):
            return False

        try:
            return bool(method(self, event))
        except Exception:
            return False

    def _delegate_final_quality_delay_ms(self) -> int:
        delegate = getattr(self, "interaction_delegate", None)
        if delegate is None:
            return 0

        method = getattr(delegate, "on_zoomable_canvas_final_quality_delay_ms", None)
        if not callable(method):
            return 0

        try:
            return max(0, int(method(self) or 0))
        except Exception:
            return 0

    def _delegate_blocks_pan(self, event) -> bool:
        return self._delegate_interaction("should_block_pan", event)

    def _normalize_pointer_event(self, event):
        raw_x = float(getattr(event, "x", 0.0) or 0.0)
        raw_y = float(getattr(event, "y", 0.0) or 0.0)
        norm_x = raw_x
        norm_y = raw_y
        corrected = False
        norm_source = "raw"
        root_x = 0.0
        root_y = 0.0
        event_local_x = None
        event_local_y = None
        pointer_local_x = None
        pointer_local_y = None

        try:
            root_x = float(self.winfo_rootx())
            root_y = float(self.winfo_rooty())
        except Exception:
            root_x = 0.0
            root_y = 0.0

        try:
            if hasattr(event, "x_root") and hasattr(event, "y_root"):
                event_local_x = float(getattr(event, "x_root", 0.0) or 0.0) - root_x
                event_local_y = float(getattr(event, "y_root", 0.0) or 0.0) - root_y
        except Exception:
            event_local_x = None
            event_local_y = None

        # Querying the global pointer position can be surprisingly expensive on
        # Windows/Tk and sits directly in the first-corner drag hot path.  Use it
        # only as a real fallback when the event does not provide root coords.
        if event_local_x is None or event_local_y is None:
            try:
                pointer_local_x = float(self.winfo_pointerx()) - root_x
                pointer_local_y = float(self.winfo_pointery()) - root_y
            except Exception:
                pointer_local_x = None
                pointer_local_y = None

        if event_local_x is not None and event_local_y is not None:
            if abs(event_local_x - raw_x) > 1.5 or abs(event_local_y - raw_y) > 1.5:
                norm_x = event_local_x
                norm_y = event_local_y
                corrected = True
                norm_source = "event_root"
        elif pointer_local_x is not None and pointer_local_y is not None:
            # Korzystamy z biezacej pozycji kursora tylko jako ostatecznego fallbacku.
            # Przy dragowaniu zalegle eventy musza zachowac historyczne wspolrzedne
            # zdarzenia; podstawianie tu "zywego" kursora powodowalo skoki uchwytu
            # i przestawienie punktu juz po puszczeniu myszy.
            if abs(pointer_local_x - raw_x) > 1.5 or abs(pointer_local_y - raw_y) > 1.5:
                norm_x = pointer_local_x
                norm_y = pointer_local_y
                corrected = True
                norm_source = "pointer"

        data = {}
        try:
            data.update(getattr(event, "__dict__", {}) or {})
        except Exception:
            pass

        for attr in ("widget", "type", "state", "delta", "num", "x_root", "y_root"):
            if attr not in data and hasattr(event, attr):
                try:
                    data[attr] = getattr(event, attr)
                except Exception:
                    pass

        data["x"] = norm_x
        data["y"] = norm_y
        data["raw_x"] = raw_x
        data["raw_y"] = raw_y
        data["event_local_x"] = event_local_x
        data["event_local_y"] = event_local_y
        data["pointer_local_x"] = pointer_local_x
        data["pointer_local_y"] = pointer_local_y
        data["canvas_root_x"] = root_x
        data["canvas_root_y"] = root_y
        try:
            data["canvas_x"] = float(self.canvasx(norm_x))
            data["canvas_y"] = float(self.canvasy(norm_y))
            data["canvas_offset_x"] = float(data["canvas_x"] - norm_x)
            data["canvas_offset_y"] = float(data["canvas_y"] - norm_y)
        except Exception:
            data["canvas_x"] = norm_x
            data["canvas_y"] = norm_y
            data["canvas_offset_x"] = 0.0
            data["canvas_offset_y"] = 0.0
        data["norm_source"] = norm_source
        data["normalized_from_root"] = corrected
        return SimpleNamespace(**data)

    def _cancel_deferred_display(self):
        pending = getattr(self, "_deferred_display_after_id", None)
        if pending:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        self._deferred_display_after_id = None
        self._deferred_display_fast_mode = False
        self._cancel_final_quality_display()

    def _cancel_final_quality_display(self):
        pending = getattr(self, "_final_quality_after_id", None)
        if pending:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        self._final_quality_after_id = None

    def _schedule_final_quality_display(self, delay_ms: int = 110):
        if self.original_image is None:
            return

        self._cancel_final_quality_display()
        scheduled_at = time.perf_counter()
        probe_active = self._perf_probe_active()
        if probe_active:
            self._log_perf(
                "final_quality.schedule",
                0.0,
                threshold_ms=0.0,
                delay=max(0, int(delay_ms)),
            )

        def _run():
            run_started_at = time.perf_counter()
            self._final_quality_after_id = None
            wait_ms = max(0.0, (run_started_at - scheduled_at) * 1000.0)
            if probe_active or self._perf_probe_active() or wait_ms >= (max(0, int(delay_ms)) + 80):
                self._log_perf(
                    "final_quality.run_wait",
                    wait_ms,
                    threshold_ms=0.0 if probe_active else 120.0,
                    delay=max(0, int(delay_ms)),
                )
            defer_ms = self._delegate_final_quality_delay_ms()
            if defer_ms > 0:
                try:
                    self._final_quality_after_id = self.after(max(80, int(defer_ms)), _run)
                except Exception:
                    self._final_quality_after_id = None
                return
            self._update_display(interaction_fast=False)
            elapsed_ms = max(0.0, (time.perf_counter() - run_started_at) * 1000.0)
            self._log_perf(
                "final_quality.run_total",
                elapsed_ms,
                threshold_ms=0.0 if probe_active else 120.0,
            )

        try:
            self._final_quality_after_id = self.after(max(0, int(delay_ms)), _run)
        except Exception:
            self._final_quality_after_id = None
            self._update_display(interaction_fast=False)

    def _cancel_zoom_animation(self):
        pending = getattr(self, "_zoom_animation_after_id", None)
        if pending:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        self._zoom_animation_after_id = None
        self._zoom_animation_target_zoom = None

    def _get_wheel_zoom_base(self) -> float:
        zoom = max(0.01, float(self.zoom_level))
        if zoom < 0.35:
            return 1.22
        if zoom < 0.70:
            return 1.18
        if zoom < 1.20:
            return 1.14
        if zoom < 2.00:
            return 1.10
        if zoom < 3.20:
            return 1.08
        return 1.06

    def _schedule_deferred_display(self, delay_ms: int = 16, *, interaction_fast: bool = False):
        if self.original_image is None:
            return
        if interaction_fast:
            self._schedule_final_quality_display(delay_ms=110)
        else:
            self._cancel_final_quality_display()

        if getattr(self, "_deferred_display_after_id", None):
            self._deferred_display_fast_mode = bool(
                getattr(self, "_deferred_display_fast_mode", False) and interaction_fast
            )
            return

        self._deferred_display_fast_mode = bool(interaction_fast)

        def _run():
            interaction_fast_local = bool(getattr(self, "_deferred_display_fast_mode", False))
            self._deferred_display_after_id = None
            self._deferred_display_fast_mode = False
            self._update_display(interaction_fast=interaction_fast_local)

        try:
            self._deferred_display_after_id = self.after(max(0, int(delay_ms)), _run)
        except Exception:
            self._deferred_display_after_id = None
            self._deferred_display_fast_mode = False
            self._update_display(interaction_fast=bool(interaction_fast))
    
    def _on_mousewheel(self, event):
        """Obsługa zoom'u kółkiem myszy."""
        if self.original_image is None:
            return

        delegate = getattr(self, "interaction_delegate", None)
        overlay_wheel_handler = getattr(delegate, "_handle_preview_canvas_overlay_mousewheel", None)
        if callable(overlay_wheel_handler):
            try:
                overlay_result = overlay_wheel_handler(event)
                if overlay_result == "break":
                    return "break"
            except Exception:
                pass

        try:
            self.focus_set()
        except Exception:
            pass
        event = self._normalize_pointer_event(event)
        anchor_x = float(getattr(event, "canvas_x", getattr(event, "x", 0.0)))
        anchor_y = float(getattr(event, "canvas_y", getattr(event, "y", 0.0)))

        raw_delta = event.delta if hasattr(event, 'delta') else (-event.num + 5) * 120
        steps = max(1, int(abs(raw_delta) / 120)) if raw_delta else 1
        steps = min(4, steps)
        zoom_factor = float(self._get_wheel_zoom_base()) ** steps
        base_zoom = float(self.zoom_level)
        pending_target = getattr(self, "_zoom_animation_target_zoom", None)
        if pending_target is not None:
            try:
                base_zoom = float(pending_target)
            except (TypeError, ValueError):
                base_zoom = float(self.zoom_level)

        if raw_delta < 0:
            new_zoom = base_zoom / zoom_factor
        else:
            new_zoom = base_zoom * zoom_factor

        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        if abs(new_zoom - float(self.zoom_level)) < 1e-9:
            return "break"

        duration_ms = min(140, 82 + (steps * 14))
        self._animate_zoom_to(anchor_x, anchor_y, new_zoom, duration_ms=duration_ms)
        self._delegate_interaction("zoom", event)
        return "break"

    def _get_middle_click_target_zoom(self) -> float:
        zoom = max(0.01, float(self.zoom_level))
        if zoom < 0.45:
            return min(self.max_zoom, 1.8)
        if zoom < 0.90:
            return min(self.max_zoom, 2.6)
        if zoom < 1.50:
            return min(self.max_zoom, 3.3)
        if zoom < 2.50:
            return min(self.max_zoom, 4.1)
        return min(self.max_zoom, zoom * 1.18)

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        clamped = min(1.0, max(0.0, float(t)))
        return 1.0 - ((1.0 - clamped) ** 3)

    @staticmethod
    def _event_is_button_release(event) -> bool:
        try:
            event_type = getattr(event, "type", None)
            if str(event_type) in {"5", "ButtonRelease"}:
                return True
            return int(event_type) == 5
        except Exception:
            return False

    def _animate_zoom_to(self, anchor_canvas_x: float, anchor_canvas_y: float, target_zoom: float, *, duration_ms: int = 170):
        if self.original_image is None:
            return

        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._cancel_final_quality_display()

        start_zoom = float(self.zoom_level)
        target_zoom = max(self.min_zoom, min(self.max_zoom, float(target_zoom)))
        if abs(target_zoom - start_zoom) < 1e-6:
            return

        anchor_img_x, anchor_img_y = self.canvas_to_image_coords(anchor_canvas_x, anchor_canvas_y, clamp=False)
        frame_interval_ms = 12
        frame_count = max(7, min(20, int(max(1, duration_ms) / frame_interval_ms)))
        frame_index = 0
        self._zoom_animation_target_zoom = float(target_zoom)

        def _run_frame():
            nonlocal frame_index
            frame_index += 1
            progress = float(frame_index) / float(frame_count)
            eased = self._ease_out_cubic(progress)
            current_zoom = start_zoom + ((target_zoom - start_zoom) * eased)
            self.zoom_level = max(self.min_zoom, min(self.max_zoom, float(current_zoom)))
            self.pan_data['x'] = float(anchor_canvas_x) - (float(anchor_img_x) * float(self.zoom_level))
            self.pan_data['y'] = float(anchor_canvas_y) - (float(anchor_img_y) * float(self.zoom_level))
            self._apply_clamped_pan()

            if frame_index >= frame_count:
                self._zoom_animation_after_id = None
                self._zoom_animation_target_zoom = None
                self._update_display(interaction_fast=False)
                return

            self._update_display(interaction_fast=True)
            try:
                self._zoom_animation_after_id = self.after(frame_interval_ms, _run_frame)
            except Exception:
                self._zoom_animation_after_id = None
                self._zoom_animation_target_zoom = None
                self._update_display(interaction_fast=False)

        _run_frame()

    def _get_view_state_center(self, view_state):
        if not isinstance(view_state, dict):
            return None

        try:
            zoom_level = float(view_state.get("zoom_level", self.zoom_level))
        except Exception:
            zoom_level = float(self.zoom_level)
        zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))

        if "center_img_x" in view_state and "center_img_y" in view_state:
            try:
                return (
                    float(view_state.get("center_img_x", 0.0)),
                    float(view_state.get("center_img_y", 0.0)),
                    float(zoom_level),
                )
            except Exception:
                return None

        try:
            canvas_width = max(1.0, float(self.winfo_width()))
            canvas_height = max(1.0, float(self.winfo_height()))
            origin_x = float(view_state.get("origin_x", self.pan_data.get('x', 0.0)))
            origin_y = float(view_state.get("origin_y", self.pan_data.get('y', 0.0)))
            center_img_x = ((canvas_width / 2.0) - origin_x) / max(1e-9, float(zoom_level))
            center_img_y = ((canvas_height / 2.0) - origin_y) / max(1e-9, float(zoom_level))
            return (float(center_img_x), float(center_img_y), float(zoom_level))
        except Exception:
            return None

    def _animate_to_view_state(self, view_state, *, duration_ms: int = 170):
        if self.original_image is None:
            return

        current_state = self.get_view_state()
        current_center = self._get_view_state_center(current_state)
        target_center = self._get_view_state_center(view_state)
        if current_center is None or target_center is None:
            try:
                self.set_view_state(view_state, redraw=True)
            except Exception:
                pass
            return

        self._cancel_zoom_animation()
        self._cancel_deferred_display()
        self._cancel_final_quality_display()

        start_center_x, start_center_y, start_zoom = current_center
        target_center_x, target_center_y, target_zoom = target_center
        frame_interval_ms = 12
        frame_count = max(7, min(20, int(max(1, duration_ms) / frame_interval_ms)))
        frame_index = 0
        self._zoom_animation_target_zoom = float(target_zoom)

        def _run_frame():
            nonlocal frame_index
            frame_index += 1
            progress = float(frame_index) / float(frame_count)
            eased = self._ease_out_cubic(progress)
            current_zoom = start_zoom + ((target_zoom - start_zoom) * eased)
            current_center_x2 = start_center_x + ((target_center_x - start_center_x) * eased)
            current_center_y2 = start_center_y + ((target_center_y - start_center_y) * eased)

            self.zoom_level = max(self.min_zoom, min(self.max_zoom, float(current_zoom)))
            canvas_width = max(1.0, float(self.winfo_width()))
            canvas_height = max(1.0, float(self.winfo_height()))
            self.pan_data['x'] = (canvas_width / 2.0) - (float(current_center_x2) * float(self.zoom_level))
            self.pan_data['y'] = (canvas_height / 2.0) - (float(current_center_y2) * float(self.zoom_level))
            self._apply_clamped_pan()

            if frame_index >= frame_count:
                self._zoom_animation_after_id = None
                self._zoom_animation_target_zoom = None
                self._update_display(interaction_fast=False)
                return

            self._update_display(interaction_fast=True)
            try:
                self._zoom_animation_after_id = self.after(frame_interval_ms, _run_frame)
            except Exception:
                self._zoom_animation_after_id = None
                self._zoom_animation_target_zoom = None
                self._update_display(interaction_fast=False)

        _run_frame()

    def _on_middle_click_zoom(self, event):
        if self.original_image is None:
            return "break"

        now = time.monotonic()
        is_release = self._event_is_button_release(event)
        if is_release:
            ignore_until = float(getattr(self, "_middle_click_zoom_ignore_release_until", 0.0) or 0.0)
            if now <= ignore_until:
                self._middle_click_zoom_ignore_release_until = 0.0
                return "break"

        if (now - float(getattr(self, "_middle_click_zoom_last_trigger_at", 0.0) or 0.0)) < 0.12:
            return "break"
        self._middle_click_zoom_last_trigger_at = now
        if not is_release:
            self._middle_click_zoom_ignore_release_until = now + 0.45

        event = self._normalize_pointer_event(event)
        restore_state = getattr(self, "_middle_click_zoom_restore_state", None)
        if isinstance(restore_state, dict):
            self._middle_click_zoom_restore_state = None
            self._animate_to_view_state(restore_state, duration_ms=180)
            self._delegate_interaction("zoom", event)
            return "break"

        anchor_x = float(getattr(event, "canvas_x", getattr(event, "x", 0.0)))
        anchor_y = float(getattr(event, "canvas_y", getattr(event, "y", 0.0)))
        self._middle_click_zoom_restore_state = self.get_view_state()
        target_zoom = self._get_middle_click_target_zoom()
        self._animate_zoom_to(anchor_x, anchor_y, target_zoom, duration_ms=170)
        self._delegate_interaction("zoom", event)
        return "break"
    
    def _on_pan_press(self, event):
        """Początek przeciągania (naciśnięcie LPM)."""
        started_at = time.perf_counter()
        self._cancel_zoom_animation()
        after_cancel = time.perf_counter()
        event = self._normalize_pointer_event(event)
        after_normalize = time.perf_counter()
        if self._delegate_interaction("press", event):
            elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
            self._log_perf(
                "pan_press.delegate",
                elapsed_ms,
                threshold_ms=0.0 if self._perf_probe_active() else 80.0,
                cancel=f"{(after_cancel - started_at) * 1000.0:.1f}",
                normalize=f"{(after_normalize - after_cancel) * 1000.0:.1f}",
            )
            return
        try:
            self.focus_set()
        except Exception:
            pass
        if self._delegate_blocks_pan(event):
            self.pan_data['press_x'] = None
            self.pan_data['press_y'] = None
            return

        self.pan_data['press_x'] = event.x
        self.pan_data['press_y'] = event.y
        self.config(cursor="hand2")
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        self._log_perf(
            "pan_press.pan",
            elapsed_ms,
            threshold_ms=0.0 if self._perf_probe_active() else 80.0,
            cancel=f"{(after_cancel - started_at) * 1000.0:.1f}",
            normalize=f"{(after_normalize - after_cancel) * 1000.0:.1f}",
        )
    
    def _on_pan_motion(self, event):
        """Przeciąganie widoku (ruch myszy z LPM)."""
        event = self._normalize_pointer_event(event)
        if self._delegate_interaction("drag", event):
            return
        if self._delegate_blocks_pan(event):
            self.pan_data['press_x'] = None
            self.pan_data['press_y'] = None
            return

        if self.pan_data['press_x'] is None:
            return
        
        # Oblicz deltę i przytnij ruch tak, aby obraz nie wypadal poza viewport.
        dx = float(event.x - self.pan_data['press_x'])
        dy = float(event.y - self.pan_data['press_y'])
        previous_x = float(self.pan_data.get('x', 0.0))
        previous_y = float(self.pan_data.get('y', 0.0))
        next_x, next_y = self._clamp_origin(previous_x + dx, previous_y + dy)
        applied_dx = float(next_x - previous_x)
        applied_dy = float(next_y - previous_y)
        self.pan_data['x'] = float(next_x)
        self.pan_data['y'] = float(next_y)

        # Aktualizuj pozycję
        self.pan_data['press_x'] = event.x
        self.pan_data['press_y'] = event.y

        if abs(applied_dx) < 1e-9 and abs(applied_dy) < 1e-9:
            return

        if not self._apply_buffered_pan_move(applied_dx, applied_dy):
            self._schedule_deferred_display(delay_ms=16, interaction_fast=True)
    
    def _on_pan_release(self, event):
        """Koniec przeciągania (zwolnienie LPM)."""
        event = self._normalize_pointer_event(event)
        if self._delegate_interaction("release", event):
            return
        if self._delegate_blocks_pan(event):
            self.pan_data['press_x'] = None
            self.pan_data['press_y'] = None
            return

        self.pan_data['press_x'] = None
        self.pan_data['press_y'] = None
        self.config(cursor=self.current_cursor)
        if self._pan_buffered_move_active:
            self._schedule_deferred_display(delay_ms=0, interaction_fast=False)
    
    def _on_reset_view(self, event):
        """Reset zoom i pan (naciśnięcie Home lub R)."""
        if event is not None and not bool(getattr(self, "reset_shortcut_enabled", True)):
            return
        self.zoom_level = 1.0
        self.pan_data = {'x': 0, 'y': 0, 'press_x': None, 'press_y': None}
        self._update_display()
    
    def _on_toggle_info(self, event):
        """Toggle wyświetlania tekstu info (naciśnięcie I)."""
        self.show_info = not self.show_info
        self._update_display()
    
    def _update_display(self, interaction_fast: bool = False):
        """Aktualizuj canvas z nowym zoom'em i pan'em."""
        if self.original_image is None:
            return
        self._update_display_visible_region(interaction_fast=interaction_fast)
        return
        
        # Skaluj obraz
        new_width = int(self.original_image.width * self.zoom_level)
        new_height = int(self.original_image.height * self.zoom_level)
        
        if new_width <= 0 or new_height <= 0:
            return
        
        # BILINEAR zapewnia płynniejszy zoom niż LANCZOS w interaktywnym podglądzie.
        scaled = self.original_image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )
        
        # Konwertuj do PhotoImage
        self.photo_image = ImageTk.PhotoImage(scaled)
        
        # Wyczyść canvas
        self.delete("all")
        
        # Umieść obraz na canvas (z offsetem z pan'u)
        self.image_id = self.create_image(
            self.pan_data['x'],
            self.pan_data['y'],
            image=self.photo_image,
            anchor="nw"  # Northwest = górny lewy róg
        )
        
        # Aktualizuj scroll region
        self.configure(scrollregion=self.bbox("all"))

        self._draw_overlay()

    def _update_display_visible_region(self, interaction_fast: bool = False):
        """Renderuj tylko widoczny fragment obrazu zamiast skalowac calosc."""
        if self.original_image is None:
            return

        started_at = time.perf_counter()
        phase_at = started_at
        phase_ms: dict[str, float] = {}

        def mark_phase(name: str) -> None:
            nonlocal phase_at
            now = time.perf_counter()
            phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
            phase_at = now

        self._apply_clamped_pan()
        mark_phase("clamp")

        visible_region = self._get_visible_image_region(interaction_fast=interaction_fast)
        mark_phase("region")
        full_width, full_height = self._get_full_image_size()
        origin_x = float(self.pan_data.get('x', 0.0))
        origin_y = float(self.pan_data.get('y', 0.0))

        next_photo_image = None
        next_image_coords = None
        next_render_region = None

        if visible_region is not None:
            crop_box = visible_region["crop_box"]
            cropped = self.original_image.crop(crop_box)
            mark_phase("crop")
            resampling = Image.Resampling.NEAREST if interaction_fast else getattr(
                self,
                "resampling_quality",
                Image.Resampling.BILINEAR,
            )
            scaled = cropped.resize(
                (int(visible_region["draw_width"]), int(visible_region["draw_height"])),
                resampling
            )
            mark_phase("resize")
            next_photo_image = ImageTk.PhotoImage(scaled)
            mark_phase("photo")
            next_image_coords = (
                float(visible_region["draw_x"]),
                float(visible_region["draw_y"]),
            )
            next_render_region = {
                "crop_box": tuple(crop_box),
                "draw_x": float(visible_region["draw_x"]),
                "draw_y": float(visible_region["draw_y"]),
                "draw_width": int(visible_region["draw_width"]),
                "draw_height": int(visible_region["draw_height"]),
                "zoom_level": float(self.zoom_level),
            }

        # Prepare the resized frame before clearing the canvas. The old image
        # stays visible during resize, so Q/E navigation does not flash blank.
        self.delete("all")
        mark_phase("delete")
        self.photo_image = None
        self.image_id = None
        self._render_region = None
        self._pan_buffered_move_active = False

        if next_photo_image is not None and next_image_coords is not None:
            self.photo_image = next_photo_image
            self.image_id = self.create_image(
                float(next_image_coords[0]),
                float(next_image_coords[1]),
                image=self.photo_image,
                anchor="nw"
            )
            self._render_region = next_render_region
        mark_phase("create")

        self.configure(
            scrollregion=(
                origin_x,
                origin_y,
                origin_x + full_width,
                origin_y + full_height,
            )
        )
        mark_phase("scrollregion")
        previous_fast_rendering = bool(getattr(self, "_interaction_fast_rendering", False))
        self._interaction_fast_rendering = bool(interaction_fast)
        try:
            self._draw_overlay()
        finally:
            self._interaction_fast_rendering = previous_fast_rendering
        mark_phase("overlay")
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if self._perf_probe_active() or elapsed_ms >= 90.0:
            details = {
                "fast": int(bool(interaction_fast)),
                "clamp": f"{phase_ms.get('clamp', 0.0):.1f}",
                "region": f"{phase_ms.get('region', 0.0):.1f}",
                "crop": f"{phase_ms.get('crop', 0.0):.1f}",
                "resize": f"{phase_ms.get('resize', 0.0):.1f}",
                "photo": f"{phase_ms.get('photo', 0.0):.1f}",
                "delete": f"{phase_ms.get('delete', 0.0):.1f}",
                "create": f"{phase_ms.get('create', 0.0):.1f}",
                "scroll": f"{phase_ms.get('scrollregion', 0.0):.1f}",
                "overlay": f"{phase_ms.get('overlay', 0.0):.1f}",
            }
            try:
                if visible_region is not None:
                    details["draw"] = f"{int(visible_region.get('draw_width', 0))}x{int(visible_region.get('draw_height', 0))}"
            except Exception:
                pass
            self._log_perf(
                "display.visible_region",
                elapsed_ms,
                threshold_ms=0.0 if self._perf_probe_active() else 90.0,
                **details,
            )

    def _can_reuse_buffered_pan(self):
        region = self._render_region if isinstance(self._render_region, dict) else None
        if region is None or self.image_id is None or self.photo_image is None:
            return False

        if abs(float(region.get("zoom_level", 0.0)) - float(self.zoom_level)) > 1e-9:
            return False

        exact_bounds = self._get_exact_visible_image_bounds()
        if exact_bounds is None:
            return False

        crop_left, crop_top, crop_right, crop_bottom = [float(v) for v in region.get("crop_box", (0, 0, 0, 0))]
        visible_left, visible_top, visible_right, visible_bottom = exact_bounds
        return (
            crop_left <= visible_left
            and crop_top <= visible_top
            and crop_right >= visible_right
            and crop_bottom >= visible_bottom
        )

    def _apply_buffered_pan_move(self, dx: float, dy: float) -> bool:
        if not self._can_reuse_buffered_pan():
            return False

        try:
            self.move("all", float(dx), float(dy))
        except Exception:
            return False

        if isinstance(self._render_region, dict):
            self._render_region["draw_x"] = float(self._render_region.get("draw_x", 0.0)) + float(dx)
            self._render_region["draw_y"] = float(self._render_region.get("draw_y", 0.0)) + float(dy)

        full_width = max(1.0, float(self.original_image.width) * float(self.zoom_level))
        full_height = max(1.0, float(self.original_image.height) * float(self.zoom_level))
        origin_x = float(self.pan_data.get('x', 0.0))
        origin_y = float(self.pan_data.get('y', 0.0))
        self.configure(
            scrollregion=(
                origin_x,
                origin_y,
                origin_x + full_width,
                origin_y + full_height,
            )
        )
        self._pan_buffered_move_active = True
        return True

    def _draw_overlay(self, *, skip_info: bool = False):
        if self.original_image is None:
            return

        renderer = getattr(self, "overlay_renderer", None)
        if callable(renderer):
            try:
                renderer(self)
            except Exception:
                pass

        if self.show_info and not skip_info:
            vx = self.canvasx(10)
            vy = self.canvasy(10)
            
            info_text = f"Zoom: {self.zoom_level:.2f}x"
            help_text = "[Rolka: zoom] [LPM+drag: pan] [ŚPM: zoom/powrót] [R/Home: reset] [I: info]"
            vy_help = self.canvasy(30)

            offsets = [(-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2)]
            
            for dx, dy in offsets:
                self.create_text(vx + dx, vy + dy, text=info_text, fill="black", font=("Arial", 11, "bold"), anchor="nw", tags="info")
            self.create_text(vx, vy, text=info_text, fill="#f1c40f", font=("Arial", 11, "bold"), anchor="nw", tags="info")
            
            for dx, dy in offsets:
                self.create_text(vx + dx, vy_help + dy, text=help_text, fill="black", font=("Arial", 9, "bold"), anchor="nw", tags="info")
            self.create_text(vx, vy_help, text=help_text, fill="white", font=("Arial", 9, "bold"), anchor="nw", tags="info")

    def refresh_overlay_only(self, *, skip_info: bool = False):
        """Przerysuj tylko overlay bez ponownego skalowania całego obrazu."""
        if self.original_image is None:
            return

        if self.image_id is None or self.photo_image is None:
            self._update_display()
            return

        self.delete("preview_overlay")
        if not skip_info:
            self.delete("info")
        self._draw_overlay(skip_info=skip_info)
    
    def get_zoom_level(self):
        """Zwróć obecny poziom zoom'u."""
        return self.zoom_level
    
    def set_zoom_level(self, level):
        """Ustaw poziom zoom'u bezpośrednio."""
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, level))
        self._update_display()

    def get_view_state(self):
        """Zwróć bieżący stan widoku (zoom + pozycja obrazu)."""
        state = {
            "zoom_level": float(self.zoom_level),
            "origin_x": float(self.pan_data.get('x', 0.0)),
            "origin_y": float(self.pan_data.get('y', 0.0)),
        }
        try:
            self.update_idletasks()
            canvas_width = max(1.0, float(self.winfo_width()))
            canvas_height = max(1.0, float(self.winfo_height()))
            center_img_x, center_img_y = self.canvas_to_image_coords(canvas_width / 2.0, canvas_height / 2.0, clamp=False)
            state["center_img_x"] = float(center_img_x)
            state["center_img_y"] = float(center_img_y)
            state["canvas_width"] = float(canvas_width)
            state["canvas_height"] = float(canvas_height)
        except Exception:
            pass
        return state

    def set_view_state(self, view_state, redraw: bool = True):
        """Ustaw kompletny stan widoku (zoom + pozycja obrazu)."""
        if not isinstance(view_state, dict):
            return False

        try:
            zoom_level = float(view_state.get("zoom_level", self.zoom_level))
        except (TypeError, ValueError):
            return False

        self.zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
        origin_x = float(self.pan_data.get('x', 0.0))
        origin_y = float(self.pan_data.get('y', 0.0))
        try:
            if "center_img_x" in view_state and "center_img_y" in view_state:
                self.update_idletasks()
                canvas_width = max(1.0, float(self.winfo_width()))
                canvas_height = max(1.0, float(self.winfo_height()))
                center_img_x = float(view_state.get("center_img_x", 0.0))
                center_img_y = float(view_state.get("center_img_y", 0.0))
                origin_x = (canvas_width / 2.0) - (center_img_x * self.zoom_level)
                origin_y = (canvas_height / 2.0) - (center_img_y * self.zoom_level)
            else:
                origin_x = float(view_state.get("origin_x", self.pan_data.get('x', 0.0)))
                origin_y = float(view_state.get("origin_y", self.pan_data.get('y', 0.0)))
        except (TypeError, ValueError):
            return False

        self.pan_data = {
            'x': origin_x,
            'y': origin_y,
            'press_x': None,
            'press_y': None
        }
        self._apply_clamped_pan()
        if redraw:
            self._update_display()
        return True

    def set_view(self, zoom_level: float, origin_x: float, origin_y: float, redraw: bool = True):
        """Ustaw zoom i origin obrazu w jednym kroku."""
        return self.set_view_state(
            {
                "zoom_level": float(zoom_level),
                "origin_x": float(origin_x),
                "origin_y": float(origin_y),
            },
            redraw=redraw,
        )
    
    def reset_view(self):
        """Reset widoku (zoom + pan)."""
        self._on_reset_view(None)

    def _apply_fit_to_view_geometry(self):
        if self.original_image is None:
            return False

        self.update_idletasks()
        canvas_width = max(1, int(self.winfo_width()))
        canvas_height = max(1, int(self.winfo_height()))

        if canvas_width <= 1 or canvas_height <= 1:
            return False

        scale_x = canvas_width / max(1, self.original_image.width)
        scale_y = canvas_height / max(1, self.original_image.height)
        fitted_zoom = min(scale_x, scale_y)
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, fitted_zoom))

        scaled_width = float(self.original_image.width) * float(self.zoom_level)
        scaled_height = float(self.original_image.height) * float(self.zoom_level)

        self.pan_data = {
            'x': (float(canvas_width) - scaled_width) / 2.0,
            'y': (float(canvas_height) - scaled_height) / 2.0,
            'press_x': None,
            'press_y': None
        }
        self._apply_clamped_pan()
        return True

    def fit_to_view(self):
        """Dopasuj cały obraz do aktualnego rozmiaru canvasa i wycentruj go."""
        if self.original_image is None:
            return

        if not self._apply_fit_to_view_geometry():
            self.after(25, self.fit_to_view)
            return
        self._update_display()
        
    def update_image_preserve_zoom(self, pil_image):
        """
        Aktualizuj obraz, zachowując obecny zoom i pan.
        
        Args:
            pil_image: Nowy obraz (PIL.Image)
        """
        self.original_image = pil_image
        self._apply_clamped_pan()
        self._update_display()

    def get_image_origin(self):
        """Zwróć pozycję lewego górnego rogu obrazu na canvasie."""
        return float(self.pan_data.get('x', 0.0)), float(self.pan_data.get('y', 0.0))

    def get_display_image_size(self):
        """Zwróć rozmiar obrazu po uwzględnieniu bieżącego zoomu."""
        if self.original_image is None:
            return 0.0, 0.0
        return (
            float(self.original_image.width) * float(self.zoom_level),
            float(self.original_image.height) * float(self.zoom_level),
        )

    def image_to_canvas_coords(self, x: float, y: float):
        """Przelicz współrzędne obrazu na współrzędne canvasa."""
        origin_x, origin_y = self.get_image_origin()
        return (
            origin_x + (float(x) * float(self.zoom_level)),
            origin_y + (float(y) * float(self.zoom_level)),
        )

    def canvas_to_image_coords(self, x: float, y: float, clamp: bool = False):
        """Przelicz współrzędne canvasa na współrzędne obrazu."""
        origin_x, origin_y = self.get_image_origin()
        if float(self.zoom_level) == 0.0:
            img_x, img_y = 0.0, 0.0
        else:
            img_x = (float(x) - origin_x) / float(self.zoom_level)
            img_y = (float(y) - origin_y) / float(self.zoom_level)

        if clamp and self.original_image is not None:
            img_x = min(max(0.0, img_x), max(0.0, float(self.original_image.width) - 1.0))
            img_y = min(max(0.0, img_y), max(0.0, float(self.original_image.height) - 1.0))
        return img_x, img_y

    def point_is_inside_image(self, x: float, y: float) -> bool:
        """Sprawdź, czy punkt canvasa leży na obszarze obrazu."""
        if self.original_image is None:
            return False
        img_x, img_y = self.canvas_to_image_coords(x, y, clamp=False)
        return (
            0.0 <= img_x <= max(0.0, float(self.original_image.width) - 1.0)
            and 0.0 <= img_y <= max(0.0, float(self.original_image.height) - 1.0)
        )
