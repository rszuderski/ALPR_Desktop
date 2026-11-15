"""Coalesced pointer-following pan; no inertia or changes to zoom semantics."""
import math
import time
import tkinter as tk
from .mobile_canvas_zoom import paint_viewport, cancel_zoom

FRAME_MS = 16


def request_pan(host, center):
    if getattr(host, '_closed', False) or host._image is None or host._view_drag_at is None:
        return
    if not getattr(host, '_smooth_pan', None):
        host._smooth_pan = SmoothPan(host)
    host._smooth_pan.request(center)


def finish_pan(host):
    pan = getattr(host, '_smooth_pan', None)
    if pan:
        pan.finish()


def cancel_pan(host):
    pan = getattr(host, '_smooth_pan', None)
    if pan:
        pan.cancel()


class SmoothPan:
    def __init__(self, host, clock=time.perf_counter):
        self.h, self.canvas, self.clock = host, host.canvas, clock
        self.active = False
        self.after_id = None
        self.pending = None
        self.last_paint = float('-inf')
        self.tag = 'SmoothPan' + str(id(self))
        self.canvas.bindtags((self.tag, *self.canvas.bindtags()))
        for sequence in ('<ButtonPress>', '<KeyPress>', '<FocusOut>'):
            self.canvas.bind_class(self.tag, sequence, self.interrupt)
        self.canvas.bind('<Destroy>', self.destroy, add='+')

    def context(self):
        h = self.h
        return (id(h._image), getattr(h, '_image_token', None), h._image_entry,
                getattr(h, 'current_invocation', None), h._view_zoom,
                self.canvas.winfo_width(), self.canvas.winfo_height())

    def valid(self):
        h = self.h
        return (self.active and not getattr(h, '_closed', False)
                and h._view_drag_at is not None and self.context() == self.key
                and h._view_center == self.last_center)

    def cancel_timer(self):
        if self.after_id is not None:
            try:
                self.canvas.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def cancel(self):
        self.cancel_timer()
        if self.active:
            self.h._view_drag_at = None
        self.active = False
        self.pending = None

    def request(self, center):
        if not all(math.isfinite(v) for v in center):
            return
        if self.active and not self.valid():
            self.cancel()
            return
        if not self.active:
            if center == self.h._view_center:
                return
            cancel_zoom(self.h)
            self.active = True
            self.key = self.context()
            self.last_center = self.h._view_center
            self.last_paint = float('-inf')
        self.pending = center
        if self.after_id is None:
            elapsed = (self.clock() - self.last_paint) * 1000
            delay = 1 if elapsed >= FRAME_MS else max(1, round(FRAME_MS - elapsed))
            self.after_id = self.canvas.after(delay, self.tick)

    def tick(self):
        self.after_id = None
        if not self.valid():
            self.cancel()
            return
        if self.pending is not None:
            self.last_paint = self.clock()
            self.h._view_center, self.pending = self.pending, None
            paint_viewport(self.h, preview=True)
            self.last_center = self.h._view_center  # Includes viewport clamping.

    def finish(self):
        if not self.active:
            return
        if not self.valid():
            self.cancel()
            return
        self.cancel_timer()
        if self.pending is not None:
            self.h._view_center = self.pending
        # Exactly one full-quality frame at the final pointer position.
        # The same viewport-only path keeps scientific panels out of the drag.
        try:
            paint_viewport(self.h)
        finally:
            self.cancel()

    def interrupt(self, event):
        if getattr(event, 'num', None) not in (4, 5):
            self.finish()

    def destroy(self, event):
        if event.widget is self.canvas:
            self.cancel()
            for sequence in ('<ButtonPress>', '<KeyPress>', '<FocusOut>'):
                self.canvas.unbind_class(self.tag, sequence)
