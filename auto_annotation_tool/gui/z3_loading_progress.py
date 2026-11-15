"""Visible loading progress independent of the platform's ttk progress style."""

import tkinter as tk


class LoadingProgressBar(tk.Canvas):
    def __init__(self, master, **kwargs):
        self._mode = "determinate"
        self._maximum = 100.0
        self._value = 0.0
        self._phase = 0.0
        self._job = None
        self._interval = 40
        self._fill_color = "#4c7a3b"
        self._trough_color = "#e6ead7"
        self._border_color = "#c5ccb0"
        super().__init__(master, width=280, height=18, bd=0, highlightthickness=0, **kwargs)
        self._trough = self.create_rectangle(0, 0, 0, 0)
        self._fill = self.create_rectangle(0, 0, 0, 0, outline="")
        self.bind("<Configure>", self._redraw, add="+")
        self.bind("<Destroy>", lambda event: self.stop() if event.widget is self else None, add="+")

    def _redraw(self, *_args):
        width = max(4, self.winfo_width())
        height = max(8, self.winfo_height())
        left, right = 1, width - 2
        top, bottom = 3, height - 3
        self.coords(self._trough, left, top, right, bottom)
        self.itemconfigure(self._trough, fill=self._trough_color, outline=self._border_color)
        if self._mode == "indeterminate":
            segment = (right - left) * .24
            start = left + (right - left - segment) * self._phase
            end = start + segment
        else:
            start = left
            end = left + (right - left) * min(1., self._value / self._maximum)
        self.coords(self._fill, start, top + 1, end, bottom - 1)
        self.itemconfigure(self._fill, fill=self._fill_color,
                           state=tk.NORMAL if end > start else tk.HIDDEN)

    def configure(self, cnf=None, **kwargs):
        if cnf is not None and not isinstance(cnf, dict):
            return super().configure(cnf, **kwargs)
        options = dict(cnf or {})
        options.update(kwargs)
        if "mode" in options:
            self._mode = str(options.pop("mode"))
            if self._mode != "indeterminate":
                self.stop()
        if "maximum" in options:
            self._maximum = max(1., float(options.pop("maximum")))
        if "value" in options:
            self._value = max(0., float(options.pop("value")))
        for option, attr in (("fill_color", "_fill_color"), ("trough_color", "_trough_color"),
                             ("border_color", "_border_color")):
            if option in options:
                setattr(self, attr, options.pop(option))
        result = super().configure(**options) if options else None
        self._redraw()
        return result

    config = configure

    def cget(self, key):
        if key in {"value", "maximum", "mode"}:
            return getattr(self, "_" + key)
        return super().cget(key)

    def start(self, interval=40):
        self._interval = max(20, int(interval))
        if self._job is None:
            self._animate()

    def _animate(self):
        self._job = None
        if self._mode != "indeterminate":
            return
        self._phase = (self._phase + .04) % 1.
        self._redraw()
        self._job = self.after(self._interval, self._animate)

    def stop(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
