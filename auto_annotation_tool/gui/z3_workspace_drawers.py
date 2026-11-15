"""Slide the existing PZ2 panels without rebuilding controls or the preview."""
from time import perf_counter
import tkinter as tk

from .app_theme_definitions import get_runtime_palette
from .campaign_sidebar import ProjectSidebarToggle


class WorkspaceDrawers:
    duration = .18

    def __init__(self, owner, *, clock=perf_counter):
        self.owner = owner
        self.host = owner.detect_content_frame
        self.clock = clock
        self.detached = False
        self.destroyed = False
        self.mode_transition = False
        self.panels = {"left": owner.preview_list_lf, "right": owner.detect_right_panel,
                       "bottom": owner.detect_footer_nav}
        self.parents = {"left": owner.preview_vertical_split, "right": owner.detect_split,
                        "bottom": self.host}
        self.visible = dict.fromkeys(self.panels, 1.0)
        self.target = dict(self.visible)
        self.buttons = {}
        self._job = self._restore_job = self._complete = None
        palette = get_runtime_palette(owner)
        for side, parent in self.parents.items():
            if side in {"left", "right"}:
                label = "Tablice" if side == "left" else "Ustawienia"
                self.buttons[side] = ProjectSidebarToggle(
                    parent,
                    palette,
                    lambda key=side: self.toggle(key),
                    collapsed_text=label,
                    expanded_text=label,
                    orientation="vertical",
                    side=side,
                )
                self.buttons[side].place_forget()
            else:
                self.buttons[side] = ProjectSidebarToggle(
                    parent,
                    palette,
                    lambda key=side: self.toggle(key),
                    collapsed_text="Akcje",
                    expanded_text="Akcje",
                    orientation="horizontal_tab",
                    side="bottom",
                )
                self.buttons[side].place_forget()
            parent.bind("<Configure>", self._configure, add="+")
        self.host.bind("<Destroy>", self._destroy, add="+")

    def enter(self):
        self._cancel()
        if not self.detached:
            self.saved = {}
            for side, panel in self.panels.items():
                parent = self.parents[side]
                if side == "bottom":
                    allowed = panel.winfo_manager() == "grid"
                    options = panel.grid_info() if allowed else {}
                    extent = panel.winfo_height()
                else:
                    allowed = str(panel) in tuple(map(str, parent.panes()))
                    options = {key: value[-1] for key, value in parent.paneconfigure(panel).items()
                               if key not in {"after", "before"}} if allowed else {}
                    extent = panel.winfo_width()
                self.saved[side] = {"allowed": allowed, "options": options, "extent": max(1, extent)}
                self.visible[side] = self.target[side] = float(allowed)
                if allowed:
                    if side == "bottom":
                        panel.grid_remove()
                    else:
                        self.saved[side]["sash"] = parent.sash_coord(0)[0]
                        self.saved[side]["inset"] = parent.winfo_width() - parent.sash_coord(0)[0]
                        parent.forget(panel)
            self.detached = True
        self._paint()

    def animate_mode(self, fullscreen, complete):
        self._cancel()
        self.mode_transition = True
        self._complete = complete
        self.target = {side: float(not fullscreen and self.saved[side]["allowed"])
                       for side in self.panels}
        self._start_animation()

    def toggle(self, side):
        if not self.detached or self.mode_transition:
            return
        if not self.saved[side]["allowed"]:
            return
        self._cancel()
        self._complete = None
        self.target[side] = 0.0 if self.target[side] else 1.0
        self._start_animation()
        self.owner._focus_preview_canvas()

    def _start_animation(self):
        self._start = dict(self.visible)
        self._started = self.clock()
        self._tick()

    def _configure(self, event):
        if event.widget in self.parents.values() and self.detached:
            self._paint()

    def _place_bottom_button(self, button, width, height, drawer_y):
        """Place Actions in the reserved gap of the fullscreen source legend."""
        tab_y = float(min(height - 4, drawer_y - 4))
        tab_x = float(width) / 2.0

        gap = getattr(self.owner, "_preview_source_legend_gap_bounds", None)
        canvas = getattr(self.owner, "preview_canvas", None)
        if gap and canvas is not None:
            try:
                left, _top, right, _bottom = [float(value) for value in gap]
                host_root_x = float(self.host.winfo_rootx())
                canvas_root_x = float(canvas.winfo_rootx())
                gap_center = (left + right) / 2.0
                tab_x = canvas_root_x - host_root_x + gap_center
            except (tk.TclError, TypeError, ValueError):
                pass

        tab_x = max(8.0, min(float(width) - 8.0, tab_x))
        tab_y = max(float(max(18, button.winfo_reqheight())), tab_y)
        button.place(x=round(tab_x), y=round(tab_y), anchor="s")

    def place_status_toggle(self):
        """Re-anchor the bottom Actions tab after canvas overlays are redrawn."""
        if self.destroyed or not self.detached:
            return
        saved = getattr(self, "saved", {}).get("bottom", {})
        if not saved or not saved.get("allowed"):
            return

        button = self.buttons.get("bottom")
        panel = self.panels.get("bottom")
        parent = self.parents.get("bottom")
        if button is None or panel is None or parent is None:
            return

        try:
            if str(button.winfo_manager()) != "place":
                return
            width = int(parent.winfo_width())
            height = int(parent.winfo_height())
            info = panel.place_info()
            drawer_y = int(float(info.get("y", height))) if info else height
        except (tk.TclError, TypeError, ValueError):
            return

        self._place_bottom_button(button, width, height, drawer_y)
        button.lift()

    def _paint(self):
        if self.destroyed or not self.detached:
            return
        for side, panel in self.panels.items():
            saved = self.saved[side]
            if not saved["allowed"]:
                self.buttons[side].place_forget()
                continue
            parent = self.parents[side]
            width, height = parent.winfo_width(), parent.winfo_height()
            fraction = self.visible[side]
            button = self.buttons[side]
            if side == "bottom":
                extent = min(max(saved["extent"], panel.winfo_reqheight()), max(40, height - 100))
                y = round(height - extent * fraction)
                panel.place(x=0, y=y, width=width, height=extent)
                button.set_collapsed(not bool(self.target[side]))
                self._place_bottom_button(button, width, height, y)
            else:
                extent = min(saved["extent"], max(180, width - 80))
                x = round(-extent * (1 - fraction)) if side == "left" else round(width - extent * fraction)
                panel.place(x=x, y=0, width=extent, height=height)

                panel_visible_target = bool(self.target[side])
                button.set_collapsed(not panel_visible_target)

                if side == "left":
                    boundary = max(0, x + extent)
                    # Zakładka zawsze leży POZA panelem:
                    # - otwarty panel: po jego prawej stronie, nad workspace,
                    # - schowany panel: przy lewej krawędzi workspace.
                    button.place(
                        x=boundary + 2,
                        rely=.60,
                        anchor="w",
                    )
                else:
                    boundary = min(width, x)
                    # Analogicznie po prawej:
                    # - otwarty panel: zakładka leży po jego lewej stronie,
                    # - schowany panel: przy prawej krawędzi workspace.
                    button.place(
                        x=boundary - 2,
                        rely=.60,
                        anchor="e",
                    )
            panel.lift()
            button.lift()
            if not getattr(self.owner, "_preview_fullscreen_active", False):
                button.place_forget()

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
        elif not getattr(self.owner, "_preview_fullscreen_active", False):
            self.attach()
        else:
            self._finish()

    def _finish(self):
        self.mode_transition = False
        complete, self._complete = self._complete, None
        if complete is not None:
            complete()

        if bool(getattr(self.owner, "_preview_fullscreen_active", False)):
            legend = getattr(self.owner, "preview_badge_legend", None)
            refresh = getattr(legend, "refresh", None)
            if callable(refresh):
                try:
                    self.host.after(80, refresh)
                except Exception:
                    pass

    def attach(self):
        self._cancel()
        if not self.detached:
            return
        for side, panel in self.panels.items():
            panel.place_forget()
            self.buttons[side].place_forget()
            saved = self.saved[side]
            if not saved["allowed"]:
                continue
            if side == "bottom":
                panel.grid(**saved["options"])
            else:
                options = dict(saved["options"], width=saved["extent"])
                if side == "left":
                    options["before"] = self.owner.preview_lf
                self.parents[side].add(panel, **options)
        self.detached = False

        def restore():
            self._restore_job = None
            if self.destroyed or self.detached:
                return
            for side in ("right", "left"):
                saved, parent = self.saved[side], self.parents[side]
                if saved["allowed"]:
                    x = parent.winfo_width() - saved["inset"] if side == "right" else saved["sash"]
                    parent.sash_place(0, max(0, x), 1)
            self._finish()

        def allocated():
            self._restore_job = self.host.after_idle(restore)
        self._restore_job = self.host.after_idle(allocated)

    def _cancel(self):
        for name in ("_job", "_restore_job"):
            job = getattr(self, name)
            if job is not None:
                self.host.after_cancel(job)
                setattr(self, name, None)

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
