"""Transient viewport animation. No review, geometry or source-data writes."""
import math
import time
import tkinter as tk

MIN_ZOOM, MAX_ZOOM = 1.0, 256.0  # Same upper bound as the existing A/plate fit.
FRAME_MS = 16

def wheel_factor(event, *, step_factor=1.25):
    delta = float(getattr(event, 'delta', 0) or 0)
    if not math.isfinite(delta):
        return None
    if delta:
        steps = delta / 120.0
    else:
        steps = {4: 1, 5: -1}.get(getattr(event, 'num', None), 0)
    return step_factor ** max(-24, min(24, steps)) if steps else None

def paint_viewport(host, *, preview=False):
    host._viewport_paint = True
    host._zoom_preview = preview
    try:
        host._paint_image()
    finally:
        host._viewport_paint = False
        host._zoom_preview = False

def cancel_zoom(host, *, refine=False):
    zoom = getattr(host, '_smooth_zoom', None)
    if zoom:
        zoom.cancel(refine=refine)

def request_zoom(host, factor, anchor):
    if getattr(host, '_closed', False) or host._image is None:
        return
    editor = getattr(host, '_canvas_editor', None)
    if getattr(editor, 'drag', None) or getattr(host, '_view_drag_at', None):
        return  # Never move the coordinate system during a corner/pan drag.
    if not getattr(host, '_smooth_zoom', None):
        host._smooth_zoom = SmoothZoom(host)
    host._smooth_zoom.request(factor, anchor)

class SmoothZoom:
    def __init__(self, host, clock=time.perf_counter):
        self.h, self.canvas, self.clock = host, host.canvas, clock
        self.after_id = None
        self.active = False
        self.tag = 'SmoothViewport' + str(id(self))
        self.canvas.bindtags((self.tag, *self.canvas.bindtags()))
        for sequence in ('<ButtonPress>', '<KeyPress>', '<FocusOut>'):
            self.canvas.bind_class(self.tag, sequence, self.interrupt)
        self.canvas.bind('<Destroy>', self.destroy, add='+')

    def context(self):
        h = self.h
        return (id(h._image), getattr(h, '_image_token', None), h._image_entry,
                getattr(h, 'current_invocation', None), self.canvas.winfo_width(), self.canvas.winfo_height())

    def interrupt(self, event):
        if getattr(event, 'num', None) not in (4, 5):
            self.cancel(refine=True)

    def destroy(self, event):
        if event.widget is self.canvas:
            self.cancel()
            for sequence in ('<ButtonPress>', '<KeyPress>', '<FocusOut>'):
                self.canvas.unbind_class(self.tag, sequence)

    def cancel(self, *, refine=False):
        active = self.active
        self.active = False
        if self.after_id is not None:
            try:
                self.canvas.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if active and refine and not getattr(self.h, '_closed', False) and self.h._image is not None:
            paint_viewport(self.h)

    def request(self, factor, anchor):
        if not math.isfinite(factor) or factor <= 0 or factor == 1:
            return
        h = self.h
        context = self.context()
        direction = 1 if factor > 1 else -1
        if self.active and (context != self.key or (h._view_zoom, h._view_center) != self.last_view):
            self.cancel()
        if not self.active:
            self.target = h._view_zoom
            self.direction = direction
        elif direction != self.direction:
            self.target = h._view_zoom  # Reverse now; do not drain old input first.
        target = max(MIN_ZOOM, min(MAX_ZOOM, self.target * factor))
        if abs(target - h._view_zoom) < 1e-8:
            self.cancel(refine=True)
            return
        self.target, self.direction, self.key = target, direction, context
        self.anchor = anchor
        width, height = context[-2:]
        center = h._view_center or (h._image.width / 2, h._image.height / 2)
        scale = max(1e-8, h._view_fit_scale * h._view_zoom)
        self.image_point = (center[0] + (anchor[0]-width/2)/scale,
                            center[1] + (anchor[1]-height/2)/scale)
        self.last_view = (h._view_zoom, h._view_center)
        if not self.active:
            self.active = True
            self.last_time = self.clock() - FRAME_MS / 1000
            self.after_id = self.canvas.after(1, self.tick)

    def tick(self):
        self.after_id = None
        h = self.h
        if not self.active:
            return
        if (getattr(h, '_closed', False) or self.context() != self.key
                or (h._view_zoom, h._view_center) != self.last_view):
            self.cancel()
            return
        start = self.clock()
        elapsed = max(.001, start-self.last_time)
        self.last_time = start
        distance = math.log(self.target/h._view_zoom)
        finished = abs(distance) < .001
        zoom = self.target if finished else h._view_zoom * math.exp(distance * (1-math.exp(-elapsed/.040)))
        width, height = self.key[-2:]
        scale = max(1e-8, h._view_fit_scale * zoom)
        h._view_zoom = zoom
        h._view_center = (self.image_point[0]-(self.anchor[0]-width/2)/scale,
                          self.image_point[1]-(self.anchor[1]-height/2)/scale)
        paint_viewport(h, preview=not finished)
        self.last_view = (h._view_zoom, h._view_center)
        if finished:
            self.active = False
        else:
            delay = max(1, round(FRAME_MS-(self.clock()-start)*1000))
            self.after_id = self.canvas.after(delay, self.tick)
