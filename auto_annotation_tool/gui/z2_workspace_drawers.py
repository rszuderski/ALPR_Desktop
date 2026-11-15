"""Slide the existing Z2 panels over the preview without rebuilding their contents."""

from time import perf_counter
import tkinter as tk


class WorkspaceDrawers:
    duration = .18

    def __init__(self, owner, *, clock=perf_counter):
        self.owner = owner
        self.host = owner.main_pane
        self.clock = clock
        self.detached = False
        self.destroyed = False
        self.panels = {"left": owner.main_left_frame, "right": owner.main_right_frame}
        self.widths = {}
        self.visible = {"left": 1.0, "right": 1.0, "bottom": 1.0}
        self.target = dict(self.visible)
        self.buttons = {}
        self._job = None
        self._restore_job = None
        self._complete = None
        self._bottom_origin = None
        self._bottom_offset = 0.0
        self._bottom_view_top = 0.0
        for side in self.panels:
            button = tk.Button(self.host, command=lambda key=side: self.toggle(key),
                               takefocus=False, bd=0, padx=6, pady=5, cursor="hand2",
                               bg="#233342", fg="#b8edff", activebackground="#344e64",
                               activeforeground="#ffffff", font=("Segoe UI", 9, "bold"))
            self.buttons[side] = button
        self.bottom_button = tk.Button(
            owner.canvas_frame, text="⌃ Skróty", command=lambda: self.toggle("bottom"),
            takefocus=False, bd=0, padx=8, pady=3, cursor="hand2",
            bg="#233342", fg="#b8edff", activebackground="#344e64",
            activeforeground="#ffffff", font=("Segoe UI", 9, "bold"),
        )
        self.host.bind("<Configure>", self._configure, add="+")
        self.host.bind("<Destroy>", self._destroy, add="+")

    def enter(self):
        self._cancel()
        if not self.detached:
            pane_ids = tuple(str(item) for item in self.host.panes())
            self.original_right = str(self.panels["right"]) in pane_ids
            self.widths = {side: max(180, panel.winfo_width()) for side, panel in self.panels.items()}
            self.widths["right"] = min(280, self.widths["right"])
            self.right_inset = (self.host.winfo_width() - self.host.sashpos(1)
                                if self.original_right else self.widths["right"])
            self.right_inset = min(285, self.right_inset)
            self.visible.update(left=1.0, right=float(self.original_right))
            self.right_allowed = self.original_right
            self.detached = True
            for panel in self.panels.values():
                if str(panel) in pane_ids:
                    self.host.forget(panel)
        self._paint()

    def animate_mode(self, fullscreen, complete):
        self._cancel()
        self._complete = complete
        self.target = {"left": float(not fullscreen),
                       "right": float(not fullscreen and self.right_allowed),
                       "bottom": float(not fullscreen)}
        self._start = dict(self.visible)
        self._started = self.clock()
        self._tick()

    def toggle(self, side):
        if (side != "bottom" and not self.detached) or getattr(self.owner, "_preview_fullscreen_transition_active", False):
            return
        if side == "right" and not self.right_allowed:
            return
        self._cancel()
        self._complete = None
        self.target[side] = 0.0 if self.target[side] else 1.0
        self._start = dict(self.visible)
        self._started = self.clock()
        if side == "left" and self.target[side]:
            self.sync_selection()
        if side == "right" and self.target[side]:
            self.owner._refresh_preview_list_summary(lightweight=True)
        if side == "bottom" and self.target[side]:
            self.owner._draw_preview_bottom_hint(self.owner.preview_canvas)
        self._tick()
        self.owner.preview_canvas.focus_set()

    def sync_selection(self):
        owner = self.owner
        if not getattr(owner, "_preview_list_selection_sync_pending", False):
            return
        index = owner._get_preview_display_index(owner.current_preview_index)
        if index is not None:
            owner._clear_listbox_selection_fast(owner.preview_listbox)
            owner.preview_listbox.selection_set(index)
            owner.preview_listbox.activate(index)
            owner.preview_listbox.see(index)
        owner._preview_list_selection_sync_pending = False

    def sync_right_visibility(self, allowed):
        self.right_allowed = bool(allowed)
        if not allowed:
            self.visible["right"] = self.target["right"] = 0.0
        self._paint()

    def _configure(self, event):
        if event.widget is self.host and self.detached:
            self._paint()

    def _paint(self):
        if self.destroyed:
            return
        self._paint_bottom()
        if not self.detached:
            return
        width, height = self.host.winfo_width(), self.host.winfo_height()
        for side, panel in self.panels.items():
            panel_width = min(self.widths[side], max(180, width - 80))
            visible = self.visible[side] if side != "right" or self.right_allowed else 0.0
            x = round(-panel_width * (1 - visible)) if side == "left" else round(width - panel_width * visible)
            panel.place(x=x, y=0, width=panel_width, height=height)
            panel.lift()
            button = self.buttons[side]
            if not getattr(self.owner, "_preview_fullscreen_active", False) or (side == "right" and not self.right_allowed):
                button.place_forget()
                continue
            if side == "left":
                button.configure(text="‹ Obrazy" if self.target[side] else "Obrazy ›")
                button.place(x=max(0, x + panel_width), y=0, rely=.52, anchor="w")
            else:
                button.configure(text="Status ›" if self.target[side] else "‹ Status")
                self.place_status_toggle()
            button.lift()

    def place_status_toggle(self):
        """Align both side tabs below the tool drawer, outside its content."""
        if not (self.detached and self.right_allowed
                and getattr(self.owner, "_preview_fullscreen_active", False)):
            return
        frame = self.owner.canvas_frame
        width = self.host.winfo_width()
        panel_width = min(self.widths["right"], max(180, width - 80))
        edge = round(width - panel_width * self.visible["right"])
        height = self.host.winfo_height()
        half_height = max(button.winfo_reqheight() for button in self.buttons.values()) / 2
        y = height * .52
        x = edge
        slide = getattr(self.owner, "_preview_drawer_slide", None)
        if slide is not None and slide.active and slide.visible > 0 and slide._geometry:
            _, dock_x, dock_y, _, dock_height = slide._geometry
            bottom = frame.winfo_rooty() - self.host.winfo_rooty() + dock_y + dock_height
            y = max(y, bottom + 8 + half_height)
            if y + half_height + 4 > height:
                # On unusually short windows keep the handle accessible next
                # to the drawer rather than over its last row or off-screen.
                x = min(x, frame.winfo_rootx() - self.host.winfo_rootx() + dock_x - 8)
        y = min(y, height - half_height - 4)
        self.buttons["right"].place(x=x, y=round(y), rely=0, anchor="e")
        self.buttons["left"].place_configure(y=round(y), rely=0)
        self.buttons["right"].lift()

    def bottom_rendered(self, canvas):
        self._bottom_origin = canvas.bbox("preview_bottom_hint")
        self._bottom_offset = 0.0
        self._bottom_view_top = float(canvas.canvasy(0))
        canvas.addtag_withtag(getattr(canvas, "VIEWPORT_FIXED_TAG", "preview_viewport_fixed"),
                              "preview_bottom_hint")
        self.owner._preview_bottom_hint_move_bbox = None
        self._paint_bottom()

    def _paint_bottom(self):
        canvas = self.owner.preview_canvas
        if self._bottom_origin is None or getattr(canvas, "original_image", None) is None:
            self.bottom_button.place_forget()
            return
        height = canvas.winfo_height()
        view_top = float(canvas.canvasy(0))
        view_delta = view_top - self._bottom_view_top
        if view_delta:
            x1, y1, x2, y2 = self._bottom_origin
            self._bottom_origin = (x1, y1 + view_delta, x2, y2 + view_delta)
            self._bottom_view_top = view_top
        bottom = float(canvas.canvasy(height))
        distance = max(0, bottom - self._bottom_origin[1] + 2)
        offset = distance * (1 - self.visible["bottom"])
        dy = offset - self._bottom_offset
        if canvas.find_withtag("preview_bottom_hint") and abs(dy) > .01:
            canvas.move("preview_bottom_hint", 0, dy)
            for attr in ("_preview_bottom_hint_bbox", "_preview_bottom_hint_collapse_bbox"):
                box = getattr(self.owner, attr, None)
                if box is not None:
                    setattr(self.owner, attr, (box[0], box[1] + dy, box[2], box[3] + dy))
        self._bottom_offset = offset
        y = min(height - 4, self._bottom_origin[1] + offset - float(canvas.canvasy(0)) - 2)
        self.bottom_button.configure(text="⌄ Ukryj skróty" if self.target["bottom"] else "⌃ Skróty")
        self.bottom_button.place(relx=.5, y=max(26, y), anchor="s")
        self.bottom_button.lift()

    def _tick(self):
        self._job = None
        if self.destroyed:
            return
        t = min(1.0, max(0.0, (self.clock() - self._started) / self.duration))
        eased = t * t * (3 - 2 * t)
        self.visible = {side: self._start[side] + (target - self._start[side]) * eased
                        for side, target in self.target.items()}
        self._paint()
        if t < 1:
            self._job = self.host.after(16, self._tick)
        else:
            complete, self._complete = self._complete, None
            if not getattr(self.owner, "_preview_fullscreen_active", False):
                self.attach()
            if complete is not None:
                complete()

    def attach(self):
        self._cancel()
        if not self.detached:
            return
        for side, panel in self.panels.items():
            panel.place_forget()
            self.buttons[side].place_forget()
        self.host.insert(0, self.panels["left"], weight=2)
        if self.right_allowed:
            self.host.add(self.panels["right"], weight=0)
        self.detached = False
        # Restore the operator's widths, rather than applying layout defaults.
        def restore_widths():
            self._restore_job = None
            if self.destroyed or self.detached:
                return
            width = self.host.winfo_width()
            left = min(self.widths["left"], max(180, width - 260))
            if self.right_allowed:
                self.host.sashpos(1, max(left + 160, width - self.right_inset))
            # Setting the right sash redistributes the earlier panes in ttk.
            # Set the left boundary last so that its saved width survives.
            self.host.sashpos(0, left)
        # ttk first requests child sizes and then allocates the panes in a
        # second idle pass. Restore only after both, without nested update().
        def after_allocation():
            self._restore_job = self.host.after_idle(restore_widths)
        self._restore_job = self.host.after_idle(after_allocation)

    def _cancel(self):
        if self._job is not None:
            self.host.after_cancel(self._job)
            self._job = None
        if self._restore_job is not None:
            self.host.after_cancel(self._restore_job)
            self._restore_job = None

    def _destroy(self, event):
        if event.widget is self.host:
            self._cancel()
            self._complete = None
            self.destroyed = True


def workspace_drawers(owner):
    controller = getattr(owner, "_preview_workspace_drawers", None)
    if controller is None:
        controller = owner._preview_workspace_drawers = WorkspaceDrawers(owner)
    return controller
