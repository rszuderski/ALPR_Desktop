"""Fullscreen drawer translation, independent of annotation/render state."""
from time import perf_counter

from .app_theme_definitions import CAMPAIGN_SIDEBAR_STYLE as STYLE, get_runtime_palette
from .campaign_sidebar import ProjectSidebarToggle


class PreviewDrawerSlide:
    def __init__(self, owner, *, clock=perf_counter, host=None):
        self.owner = owner
        self.panel = owner.preview_overlay_dock
        self.host = host if host is not None else owner.canvas_frame
        self.clock = clock
        self.hidden = False
        self.active = False
        self.visible = 1.0
        self.target = 1.0
        self._job = None
        self._geometry = None
        self._destroyed = False
        self.button = ProjectSidebarToggle(
            self.host, get_runtime_palette(owner), self.toggle,
            collapsed_text="Szuflada", expanded_text="Ukryj szufladę",
        )
        self.button.place_forget()
        self.panel.bind("<Destroy>", self._destroy, add="+")
        self.panel.bind("<Configure>", self._configure, add="+")

    def _configure(self, event):
        if event.widget is self.panel and self._job is not None:
            return "break"

    def place(self, frame_width, x, y, width, height, *, fullscreen, toggle_y=6):
        if self._destroyed:
            return
        if fullscreen:
            y = max(y, toggle_y + 6 + self.button._height)
        self._geometry = (frame_width, x, y, width, height)
        if fullscreen != self.active:
            self._cancel()
            self.active = fullscreen
            self.visible = self.target = 0.0 if fullscreen and self.hidden else 1.0
        palette = get_runtime_palette(self.owner)
        if palette != self.button.palette:
            self.button.set_palette(palette)
        if fullscreen:
            self.button.set_collapsed(self.hidden)
            self.button.place(relx=1, x=-10, y=toggle_y, anchor="ne")
            self.button.lift()
        else:
            self.button.place_forget()
        self._paint()

    def toggle(self):
        if not self.active or self._destroyed:
            return
        self.hidden = not self.hidden
        self._cancel()
        self.target = 0.0 if self.hidden else 1.0
        self._from = self.visible
        self._started = self.clock()
        self.button.set_collapsed(self.hidden)
        self._tick()
        self.owner.preview_canvas.focus_set()

    def _paint(self):
        if self._geometry is None:
            return
        frame_width, x, y, width, height = self._geometry
        if self.visible <= 0:
            self.panel.place_forget()
        else:
            offset = (frame_width - x + 1) * (1 - self.visible)
            self.panel.place(x=round(x + offset), y=y, width=width, height=height, anchor="nw")
            self.panel.lift()
        if self.active:
            self.button.lift()

    def _tick(self):
        self._job = None
        if self._destroyed:
            return
        t = min(1.0, max(0.0, (self.clock() - self._started) * 1000 / STYLE["animation_ms"]))
        self.visible = self._from + (self.target - self._from) * t * t * (3 - 2 * t)
        self._paint()
        if t < 1:
            self._job = self.host.after(STYLE["frame_ms"], self._tick)

    def suspend(self):
        self._cancel()
        self.active = False
        self.panel.place_forget()
        self.button.place_forget()

    def _cancel(self):
        if self._job is not None:
            self.host.after_cancel(self._job)
            self._job = None

    def _destroy(self, event):
        if event.widget is self.panel:
            self._cancel()
            self._destroyed = True
            self.button.destroy()


def place_drawer(owner, frame_width, x, y, width, height):
    slide = getattr(owner, "_preview_drawer_slide", None)
    if slide is None:
        slide = owner._preview_drawer_slide = PreviewDrawerSlide(owner)
    slide.place(frame_width, x, y, width, height,
                fullscreen=bool(getattr(owner, "_preview_fullscreen_active", False)))


def suspend_drawer(owner):
    slide = getattr(owner, "_preview_drawer_slide", None)
    if slide is not None:
        slide.suspend()
