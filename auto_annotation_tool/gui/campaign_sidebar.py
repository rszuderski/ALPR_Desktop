"""Project sidebar controls and reversible, nonblocking slide animation."""

from time import perf_counter
import tkinter as tk
from tkinter import font as tkfont

from .app_theme_definitions import CAMPAIGN_SIDEBAR_STYLE as STYLE


class ProjectSidebarToggle(tk.Canvas):
    """One stationary, keyboard-accessible control for both sidebar states."""

    def __init__(self, parent, palette, command, *, collapsed_text="Pokaż projekty", expanded_text="Ukryj panel"):
        self.collapsed_text, self.expanded_text = collapsed_text, expanded_text
        self._font = tkfont.Font(root=parent, font=STYLE["button_font"])
        width = max(self._font.measure(text) for text in (collapsed_text, expanded_text)) + 66
        height = max(STYLE["button_height"], self._font.metrics("linespace") + 18)
        super().__init__(parent, width=width, height=height, bd=0, highlightthickness=0,
                         takefocus=True, cursor="hand2")
        self.palette = palette
        self.command = command
        self.collapsed = False
        self._hover = False
        self._focused = False
        self._width, self._height = width, height
        self.bind("<Button-1>", self._invoke)
        self.bind("<Return>", self._invoke)
        self.bind("<space>", self._invoke)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.bind("<FocusOut>", lambda _e: self._set_focus(False))
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(sequence, lambda _e: "break")
        self._draw()
        self.place(relx=1.0, x=-STYLE["inset_right"]-6, y=STYLE["inset_top"]+4, anchor="ne")

    def _draw(self):
        self.delete("all")
        p = self.palette
        self.configure(bg=p["panel"])
        self.create_rectangle(1, 1, self._width-1, self._height-1, width=2, tags=("shell",))
        cy = self._height / 2
        self.create_rectangle(10, cy-10, 30, cy+10, outline=p["accent"], width=1.5)
        self.create_rectangle(24, cy-8, 28, cy+8, fill=p["accent"], outline="")
        points = (45, cy-6, 39, cy, 45, cy+6) if self.collapsed else (39, cy-6, 45, cy, 39, cy+6)
        self.create_line(*points, fill=p["fg"], width=2.5, capstyle="round", joinstyle="round", tags=("direction",))
        self.create_text(56, cy, anchor="w", text=self.collapsed_text if self.collapsed else self.expanded_text,
                         font=self._font, fill=p["fg"], tags=("label",))
        self._paint_state()

    def _paint_state(self):
        self.itemconfigure("shell", fill=self.palette["button_hover"] if self._hover else self.palette["field"],
                           outline=self.palette["fg"] if self._focused else self.palette["accent"])

    def _set_hover(self, value):
        self._hover = value
        self._paint_state()

    def _set_focus(self, value):
        self._focused = value
        self._paint_state()

    def _invoke(self, _event=None):
        self.focus_set()
        self.command()
        return "break"

    def set_collapsed(self, collapsed):
        if self.collapsed != bool(collapsed):
            self.collapsed = bool(collapsed)
            self._draw()
        self.lift()

    def set_palette(self, palette):
        self.palette = palette
        self._draw()

    def lift(self, aboveThis=None):
        self.tk.call("raise", self._w)


class SlidingProjectSidebar:
    """Translate the existing panel, never resize/rebuild its contents per frame."""

    def __init__(self, container, panel, toggle, *, on_motion, on_settled, clock=perf_counter):
        self.container, self.panel, self.toggle = container, panel, toggle
        self.on_motion, self.on_settled = on_motion, on_settled
        self.clock = clock
        self.visible = 1.0
        self.target = 1.0
        self.width = STYLE["width"]
        self.running = False
        self._job = None
        self._destroyed = False
        self._from = self.visible
        self._started = self.clock()
        self._duration = STYLE["animation_ms"] / 1000.0
        self.panel.bind("<Destroy>", self._on_destroy, add="+")
        self.panel.bind("<Configure>", self._on_configure, add="+")

    def _on_configure(self, _event):
        # Position-only animation must not repeatedly relayout root overlays.
        return "break" if self.running else None

    def set_collapsed(self, collapsed, *, animate=True):
        if self._destroyed:
            return
        target = 0.0 if collapsed else 1.0
        self.toggle.set_collapsed(collapsed)
        if self.running and target == self.target:
            return
        if not self.running and self.visible == target:
            return
        if self.running:
            self._sample()
        self._cancel_tick()
        self.target = target
        self.on_motion(True)
        if not animate:
            self._finish()
            return
        if self.panel.winfo_manager() == "grid":
            self.width = max(STYLE["width"], self.panel.winfo_width())
        self.running = True
        self._from = self.visible
        self._started = self.clock()
        self._duration = max(0.08, STYLE["animation_ms"] / 1000.0 * abs(self.target-self.visible))
        self.panel.grid_remove()
        self.container.columnconfigure(1, weight=0, minsize=0)
        self._place()
        self.panel.lift()
        self.toggle.lift()
        self._job = self.container.after(STYLE["frame_ms"], self._tick)

    def _sample(self):
        t = max(0.0, min(1.0, (self.clock()-self._started) / self._duration))
        ease = t*t*(3.0-2.0*t)
        self.visible = self._from + (self.target-self._from)*ease
        return t

    def _place(self):
        self.panel.place(relx=1.0, x=-round((self.width+STYLE["inset_right"])*self.visible),
                         y=STYLE["inset_top"], width=self.width, relheight=1.0,
                         height=-STYLE["inset_top"]-STYLE["inset_bottom"], anchor="nw")

    def _tick(self):
        self._job = None
        if self._destroyed:
            return
        if self._sample() >= 1.0:
            self._finish()
        else:
            self._place()
            self._job = self.container.after(STYLE["frame_ms"], self._tick)

    def _finish(self):
        self.visible = self.target
        self.running = False
        self.panel.place_forget()
        if self.target:
            self.panel.grid(row=0, column=1, sticky="nsew", padx=(STYLE["gap"], 0))
            self.container.columnconfigure(1, weight=0, minsize=STYLE["width"])
        else:
            self.panel.grid_remove()
            self.container.columnconfigure(1, weight=0, minsize=0)
        self.toggle.lift()
        self.on_motion(False)
        self.on_settled()

    def _cancel_tick(self):
        if self._job is not None:
            self.container.after_cancel(self._job)
            self._job = None

    def _on_destroy(self, event):
        if event.widget is not self.panel:
            return
        self._destroyed = True
        self._cancel_tick()
        self.running = False
        self.on_motion(False)
