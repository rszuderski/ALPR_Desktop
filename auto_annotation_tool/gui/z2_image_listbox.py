"""Native Listbox selection and scrolling with separately colored filenames.

Only visible rows are painted. The native widget remains the source of truth
for indices, selection, bindings and scrollbars, including bulk selection.
"""

import tkinter as tk
import tkinter.font as tkfont

from .z2_inline_hud import fit_text


class ImageListbox(tk.Listbox):
    def __init__(self, master, *, owner, **kwargs):
        super().__init__(master, **kwargs)
        self.owner = owner
        self._paint_job = None
        self._row_font = None
        self._surface = tk.Canvas(self, borderwidth=0, highlightthickness=0, takefocus=0)
        self._surface.place(x=0, y=0, relwidth=1, relheight=1)
        # Forward pointer events exactly once, to the native list and its existing
        # application bindings. Keyboard focus stays on the Listbox itself.
        self._surface.bindtags((str(self._surface),))
        events = ["<Motion>", "<Enter>", "<Leave>", "<MouseWheel>"]
        events += [f"<{kind}-{button}>" for kind in ("ButtonPress", "ButtonRelease")
                   for button in range(1, 6)]
        for sequence in events:
            self._surface.bind(sequence, lambda event, seq=sequence: self._forward(seq, event))
        self.bind("<Configure>", self._queue_paint, add="+")
        self.bind("<Destroy>", self._destroy_painter, add="+")
        # Observe Tcl-level mutations too (native keyboard bindings and the fast
        # selection-clear path bypass Python method overrides).
        self._native_command = self._w + "_native"
        self.tk.call("rename", self._w, self._native_command)
        self.tk.createcommand(self._w, self._dispatch)
        self._queue_paint()

    def _dispatch(self, *args):
        result = self.tk.call(self._native_command, *args)
        command = args[0] if args else ""
        if (command in ("insert", "delete", "activate", "see", "scan")
                or command in ("xview", "yview") and len(args) > 1
                or command in ("configure", "config") and len(args) > 2
                or command in ("itemconfigure", "itemconfig") and len(args) > 3
                or command == "selection" and len(args) > 1 and args[1] != "includes"):
            self._queue_paint()
        return result

    def _forward(self, sequence, event):
        options = dict(x=event.x, y=event.y, rootx=event.x_root, rooty=event.y_root,
                       state=event.state, time=event.time)
        if sequence == "<MouseWheel>":
            options["delta"] = event.delta
        self.event_generate(sequence, **options)
        self._queue_paint()
        return "break"

    def _queue_paint(self, _event=None):
        if self._paint_job is None:
            self._paint_job = self.after_idle(self._paint_rows)

    def _destroy_painter(self, event):
        if event.widget is not self:
            return
        if self._paint_job is not None:
            self.after_cancel(self._paint_job)
            self._paint_job = None
        self.tk.deletecommand(self._w)

    def _filename(self, index, label):
        indices = getattr(self.owner, "_preview_list_display_indices", ())
        annotations = getattr(self.owner, "current_annotations", ())
        try:
            filename = str(annotations[indices[index]].filename)
        except (IndexError, KeyError, TypeError, AttributeError):
            return ""
        return filename if filename and label.endswith(filename) else ""

    def _paint_rows(self):
        self._paint_job = None
        surface = self._surface
        background = self.cget("background")
        surface.configure(background=background, cursor=self.cget("cursor"))
        surface.delete("all")
        font_spec = self.cget("font")
        if self._row_font is None or self._row_font[0] != font_spec:
            self._row_font = (font_spec, tkfont.Font(self, font=font_spec))
        font = self._row_font[1]
        red, green, blue = self.winfo_rgb(background)
        dark = (red * .2126 + green * .7152 + blue * .0722) < 32768
        filename_color = "#8be3ff" if dark else "#164e83"
        width, height = self.winfo_width(), self.winfo_height()
        if not self.size() or width < 4 or height < 4:
            return
        selected = set(self.curselection())
        for index in range(self.nearest(0), min(self.size(), self.nearest(height) + 1)):
            box = self.bbox(index)
            if box is None:
                continue
            _native_x, y, _text_width, row_height = box
            label = self.get(index)
            filename = self._filename(index, label)
            prefix = label[:-len(filename)] if filename else label
            row_bg = self.itemcget(index, "background") or background
            color = self.itemcget(index, "foreground") or self.cget("foreground")
            file_color = filename_color
            if index in selected:
                row_bg = self.itemcget(index, "selectbackground") or self.cget("selectbackground")
                file_color = self.itemcget(index, "selectforeground") or self.cget("selectforeground")
            surface.create_rectangle(0, y, width, y + row_height + 1, fill=row_bg, outline="")
            # Keep the warning/status color and give the filename its own readable
            # column even when confidence values would otherwise push it offscreen.
            if filename:
                prefix_width = max(20, width * .55)
                if font.measure(prefix) > prefix_width and "]" in prefix:
                    # Drop optional confidence/reuse details before abbreviating
                    # an actionable status such as BRAK GT: 1.
                    prefix = prefix[:prefix.index("]") + 1] + " "
                prefix = fit_text(prefix, font, max(prefix_width, min(font.measure(prefix), width - 90)))
            surface.create_text(5, y, anchor="nw", text=prefix, font=font,
                                fill=color, tags=("image_list_status", f"row_{index}"))
            if filename:
                file_x = 5 + font.measure(prefix) + 8
                surface.create_text(file_x, y, anchor="nw",
                                    text=fit_text(filename, font, max(10, width - file_x - 6)),
                                    font=font, fill=file_color,
                                    tags=("image_list_filename", f"row_{index}"))
