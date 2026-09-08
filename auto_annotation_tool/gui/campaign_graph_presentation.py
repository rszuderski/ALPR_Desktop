"""Screen-space graph controls and presentation-only workflow focus."""

from dataclasses import dataclass
import tkinter as tk
from tkinter import font as tkfont

from ..campaign_transition_graph import campaign_stage_step
from .app_theme_definitions import CAMPAIGN_GRAPH_STYLE
from .web_slim_scrollbar import blend_hex_colors


GRAPH_SIGNAL_SPEED_PX_S = 120.0
GRAPH_SIGNAL_TAIL_PX = 29.0


@dataclass
class GraphSignalMotion:
    """Distance in screen pixels; never normalize travel by path length."""

    last_time: float
    distance: float = -4.0
    paused: bool = False
    speed: float = GRAPH_SIGNAL_SPEED_PX_S

    def advance(self, now: float, *, paused: bool = False) -> float:
        elapsed = max(0.0, now - self.last_time)
        self.last_time = now
        if not paused and not self.paused:
            self.distance += elapsed * self.speed
        self.paused = paused
        return self.distance

    def finished(self, path_length: float) -> bool:
        return self.distance > path_length + GRAPH_SIGNAL_TAIL_PX


def graph_signal_color(palette):
    background = palette.get("campaign_graph_bg", "#071009").lstrip("#")
    light = sum(int(background[index:index + 2], 16) for index in (0, 2, 4)) > 3 * 150
    return "#007e9b" if light else "#55dcf5"


def gate_electrode_style(palette, *, selected=False, enabled=True):
    background = palette.get("campaign_gate_surface", palette.get("panel", "#141e12"))
    if not enabled:
        muted = palette.get("campaign_card_disabled", "#4b5c48")
        return {"fill": background, "border": blend_hex_colors(background, muted, 0.55),
                "ink": muted, "rod": blend_hex_colors(background, muted, 0.75), "hover": background}
    if selected:
        return {"fill": "#14684f", "border": "#6be5b6", "ink": "#e0fff1",
                "rod": "#a0ecd0", "hover": "#20825f"}
    return {"fill": "#09556c", "border": "#4cd2ee", "ink": "#e0faff",
            "rod": "#89d9e9", "hover": "#10768d"}


def close_graph_dialogs(parent):
    """Discard graph snapshots before leaving for an editor."""
    try:
        children = list(parent.winfo_children())
    except (AttributeError, tk.TclError):
        return
    for child in children:
        if bool(getattr(child, "_campaign_graph_dialog", False)):
            try:
                child.destroy()
            except tk.TclError:
                pass
        elif not isinstance(child, tk.Toplevel):
            close_graph_dialogs(child)


@dataclass(frozen=True)
class WorkflowFocus:
    edges: frozenset[str]
    stages: frozenset[str]


def resolve_workflow_focus(graph, *, current_step, selected_edge="", selected_path="", operable_edges=None):
    candidates = [edge for edge in graph.edges if campaign_stage_step(edge.source) == current_step]
    if operable_edges is not None:
        candidates = [edge for edge in candidates if edge.key in operable_edges]
    if selected_path:
        candidates = [edge for edge in candidates if not edge.paths or selected_path in edge.paths]
    selected = next((edge for edge in candidates if edge.key == selected_edge), None)
    if selected is not None:
        candidates = [selected]
    stages = {stage for edge in candidates for stage in (edge.source, edge.target)}
    if not stages:
        stages = {node.key for node in graph.nodes.values() if campaign_stage_step(node.key) == current_step}
    return WorkflowFocus(frozenset(edge.key for edge in candidates), frozenset(stages))


def _item_in_focus(tags, scope):
    if "graph_overlay_fixed" in tags or "graph_redraw_previous_frame" in tags:
        return None
    # An available selector remains a visible way to switch routes even when
    # its gate's informational fields belong to the dimmed branch.
    if "gate_button" in tags and any(tag.startswith("gate:") and tag.endswith(":select") for tag in tags):
        return True
    for prefix in ("gate-group:", "gate-connector:", "edge-line:", "graph_edge_arrow:", "gate_selector_arc_anim:"):
        for tag in tags:
            if tag.startswith(prefix):
                key = tag[len(prefix):]
                if key == "return_trunk":
                    return bool(scope.edges & {"e4t_to_e1", "e4z_to_e1"})
                return key in scope.edges
    for tag in tags:
        if tag.startswith("node-junction:"):
            return set(tag.split(":")[1:]).issubset(scope.stages)
    for prefix in ("node-group:", "node-output-connector:"):
        for tag in tags:
            if tag.startswith(prefix):
                return tag[len(prefix):] in scope.stages
    if "graph_return_trunk" in tags or "graph_return_junction" in tags:
        return bool(scope.edges & {"e4t_to_e1", "e4z_to_e1"})
    return None


class CanvasWorkflowFocus:
    def __init__(self, canvas, palette):
        self.canvas = canvas
        self.background = palette["campaign_graph_bg"]
        self._colors = {}
        self._dimmed_colors = {}
        self._scope = None

    def begin_frame(self):
        self._colors.clear()
        self._scope = None

    def apply(self, scope, *, refresh_links=False):
        canvas = self.canvas
        items = canvas.find_all()
        current_ids = set(items)
        changed_scope = scope != self._scope
        self._colors = {key: value for key, value in self._colors.items() if key in current_ids}
        link_items = set()
        if refresh_links:
            for tag in ("edge_line", "graph_edge_arrow", "gate_connector", "graph_node_output_connector"):
                link_items.update(canvas.find_withtag(tag))
        for item in items:
            cached = self._colors.get(item)
            if cached is not None and not changed_scope and item not in link_items:
                continue
            active = _item_in_focus(canvas.gettags(item), scope)
            if active is None:
                continue
            kind = canvas.type(item)
            if kind not in {"line", "text", "rectangle", "oval", "arc", "polygon"}:
                continue
            attributes = ("fill",) if kind in {"line", "text"} else ("fill", "outline")
            cached = cached or {}
            updates = {}
            for attr in attributes:
                current = canvas.itemcget(item, attr)
                original, painted = cached.get(attr, (current, current))
                if current != painted:
                    original = current
                desired = original
                if not active and original:
                    if original not in self._dimmed_colors:
                        self._dimmed_colors[original] = blend_hex_colors(
                            original, self.background, CAMPAIGN_GRAPH_STYLE["inactive_blend"],
                        )
                    desired = self._dimmed_colors[original]
                if current != desired:
                    updates[attr] = desired
                cached[attr] = (original, desired)
            if updates:
                canvas.itemconfigure(item, **updates)
            self._colors[item] = cached
        self._scope = scope

    def raise_scope(self, scope):
        canvas = self.canvas
        for edge in scope.edges:
            for prefix in ("edge-line:", "graph_edge_arrow:", "gate-connector:"):
                canvas.tag_raise(prefix + edge)
        if scope.edges & {"e4t_to_e1", "e4z_to_e1"}:
            for tag in ("graph_return_trunk", "graph_return_junction", "graph_edge_arrow:return_trunk"):
                canvas.tag_raise(tag)
        for stage in scope.stages:
            canvas.tag_raise("node-output-connector:" + stage)
            canvas.tag_raise("node-group:" + stage)
        for edge in scope.edges:
            canvas.tag_raise("gate-group:" + edge)


class GraphToolbar(tk.Canvas):
    """A child widget, never an item scaled/deleted with the world canvas."""

    def __init__(self, parent, palette, commands):
        super().__init__(parent, height=CAMPAIGN_GRAPH_STYLE["toolbar_height"],
                         bg=palette["panel"], bd=0, highlightthickness=1,
                         highlightbackground=palette["panel_border"], takefocus=True)
        self.palette = palette
        self.commands = commands
        self._font = tkfont.Font(root=parent, font=CAMPAIGN_GRAPH_STYLE["toolbar_font"])
        self._height = max(CAMPAIGN_GRAPH_STYLE["toolbar_height"], self._font.metrics("linespace") + 18)
        self.configure(height=self._height)
        self.zoom = 1.0
        self.context = ""
        self._layout_width = 0
        self._focused = 0
        self._actions = ["zoom_out", "zoom_in", "reset", "history", "guide"]
        self._parent_bind = parent.bind("<Configure>", self._resize, add="+")
        self.bind("<Left>", lambda _e: self._focus_action(-1))
        self.bind("<Right>", lambda _e: self._focus_action(1))
        self.bind("<Return>", lambda _e: self.invoke(self._actions[self._focused]))
        self.bind("<space>", lambda _e: self.invoke(self._actions[self._focused]))
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(event, lambda _e: "break")
        self._resize()
        for action in self._actions:
            self.tag_bind(action, "<Enter>", lambda _e, a=action: self._hover(a, True))
            self.tag_bind(action, "<Leave>", lambda _e, a=action: self._hover(a, False))
            self.tag_bind(action, "<Button-1>", lambda _e, a=action: self.invoke(a))

    def _resize(self, event=None):
        available = int(event.width) if event is not None else self.master.winfo_width()
        width = min(750, max(270, available - 18))
        if width == self._layout_width:
            return
        self._layout_width = width
        self.configure(width=width)
        self.place(x=8, y=6, width=width)
        self._draw()

    def _icon(self, action, x, y, color, tags):
        def line(*xy, **kwargs):
            self.create_line(*xy, fill=color, width=1.5, tags=tags, **kwargs)
        if action.startswith("zoom"):
            self.create_oval(x+1, y+1, x+14, y+14, outline=color, width=1.5, tags=tags)
            line(x+12, y+12, x+19, y+19)
            line(x+4, y+7.5, x+11, y+7.5)
            if action == "zoom_in":
                line(x+7.5, y+4, x+7.5, y+11)
        elif action == "reset":
            self.create_arc(x+2, y+2, x+18, y+18, start=45, extent=290, style="arc", outline=color, width=1.5, tags=tags)
            line(x+17, y+1, x+17, y+7, x+11, y+7)
        elif action == "history":
            line(x+4, y+3, x+4, y+17, x+16, y+17)
            line(x+4, y+9, x+16, y+9)
            for cx, cy in ((4, 3), (16, 9), (16, 17)):
                self.create_oval(x+cx-2, y+cy-2, x+cx+2, y+cy+2,
                                 fill=color, outline=color, tags=tags)
        elif action == "guide":
            self.create_rectangle(x+1, y+2, x+19, y+18, outline=color, width=1.5, tags=tags)
            line(x+10, y+2, x+10, y+18)
            for yy in (6, 10, 14):
                line(x+4, y+yy, x+7, y+yy)
                line(x+13, y+yy, x+16, y+yy)

    def _draw(self):
        self.delete("all")
        p = self.palette
        font = self._font
        specs = (
            ("zoom_out", "", 34), ("level", "", max(54, font.measure("240%") + 8)), ("zoom_in", "", 34),
            ("reset", "Reset", font.measure("Reset") + 44),
            ("history", "Ślad projektu", font.measure("Ślad projektu") + 44),
            ("guide", "Poradnik", font.measure("Poradnik") + 44),
        )
        compact = self._layout_width < sum(width + 4 for _, _, width in specs) + 12
        center_y = self._height / 2
        x = 6
        for action, label, full_width in specs:
            width = 34 if compact and action not in {"level"} else full_width
            if action == "level":
                self.create_text(x+width/2, center_y, text=f"{self.zoom:.0%}", font=font,
                                 fill=p["fg"], tags=("zoom_value",))
            else:
                self.create_rectangle(x, 4, x+width, self._height-4, fill=p["field"], outline=p["border"], tags=(action, f"button:{action}"))
                self._icon(action, x+7, center_y-10, p["accent"], (action,))
                if label and not compact:
                    self.create_text(x+34, center_y, text=label, anchor="w", font=font, fill=p["fg"], tags=(action,))
            x += width + 4
        if not compact and self._layout_width - x > font.measure(self.context) + 20:
            self.create_text(x+8, center_y, text=self.context, anchor="w", font=font,
                             fill=p["muted"], tags=("focus_context",))

    def _hover(self, action, active):
        self.itemconfigure(f"button:{action}", fill=self.palette["button_hover"] if active else self.palette["field"])
        self.configure(cursor="hand2" if active else "")

    def _focus_action(self, delta):
        self._hover(self._actions[self._focused], False)
        self._focused = (self._focused + delta) % len(self._actions)
        self._hover(self._actions[self._focused], True)
        return "break"

    def invoke(self, action):
        self.focus_set()
        if (action == "zoom_out" and self.zoom <= 0.65) or (action == "zoom_in" and self.zoom >= 2.4):
            return "break"
        callback = self.commands.get(action)
        if callable(callback):
            callback()
        return "break"

    def sync(self, zoom, context):
        if zoom != self.zoom:
            self.zoom = zoom
            self.itemconfigure("zoom_value", text=f"{zoom:.0%}")
        if context != self.context:
            self.context = context
            self._draw()
        self.lift()

    def lift(self, aboveThis=None):
        # Canvas.lift normally raises drawing items, not the child widget.
        self.tk.call("raise", self._w)

    def destroy(self):
        if self._parent_bind:
            self.master.unbind("<Configure>", self._parent_bind)
            self._parent_bind = None
        super().destroy()
