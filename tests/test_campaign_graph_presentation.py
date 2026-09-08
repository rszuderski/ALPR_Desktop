"""Regression tests for fixed controls and presentation-only workflow focus."""

import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock

from auto_annotation_tool.campaign_transition_graph import CAMPAIGN_TRANSITION_GRAPH as GRAPH
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette, THEME_DEFINITIONS
from auto_annotation_tool.gui.campaign_graph_presentation import (
    CanvasWorkflowFocus, GraphToolbar, WorkflowFocus, resolve_workflow_focus,
)


def scope(edge):
    transition = GRAPH.get_edge(edge)
    return WorkflowFocus(frozenset({edge}), frozenset({transition.source, transition.target}))


class WorkflowFocusTests(unittest.TestCase):
    def test_all_routes_and_stages(self):
        for path, step, edge in (
            ("plate_training", 1, "e1_to_e2"),
            ("char_from_images", 1, "e1_to_e2"),
            ("char_from_ready_plates", 1, "e1_to_e3"),
            ("plate_training", 2, "e2_to_e4"),
            ("char_from_images", 2, "e2_to_e3"),
            ("char_from_images", 3, "e3_to_e4"),
            ("char_from_ready_plates", 3, "e3_to_e4"),
            ("plate_training", 4, "e4t_to_e1"),
            ("char_from_images", 4, "e4z_to_e1"),
            ("char_from_ready_plates", 4, "e4z_to_e1"),
        ):
            with self.subTest(path=path, step=step):
                self.assertEqual(resolve_workflow_focus(GRAPH, current_step=step, selected_path=path), scope(edge))

    def test_before_choice_both_entry_gates_remain_visible(self):
        self.assertEqual(resolve_workflow_focus(GRAPH, current_step=1), WorkflowFocus(
            frozenset({"e1_to_e2", "e1_to_e3"}), frozenset({"E1", "E2", "E3"}),
        ))

    def test_explicit_entry_choice(self):
        self.assertEqual(resolve_workflow_focus(GRAPH, current_step=1, selected_edge="e1_to_e3"), scope("e1_to_e3"))

    def test_old_iteration_selection_does_not_focus_old_stage(self):
        self.assertEqual(resolve_workflow_focus(GRAPH, current_step=1,
            selected_edge="e4z_to_e1", selected_path="plate_training"), scope("e1_to_e2"))

    def test_stale_selection_on_another_route(self):
        self.assertEqual(resolve_workflow_focus(GRAPH, current_step=4,
            selected_edge="e4z_to_e1", selected_path="plate_training"), scope("e4t_to_e1"))

    def test_committed_entry_excludes_other_gate(self):
        for active, other in (("e1_to_e2", "e1_to_e3"), ("e1_to_e3", "e1_to_e2")):
            self.assertEqual(resolve_workflow_focus(GRAPH, current_step=1,
                selected_edge=other, operable_edges={active}), scope(active))

    def test_no_available_gate_keeps_current_stage(self):
        self.assertEqual(resolve_workflow_focus(GRAPH, current_step=3, operable_edges=set()),
                         WorkflowFocus(frozenset(), frozenset({"E3"})))


class GraphPresentationTkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One interpreter, held on the main thread for the entire test class.
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk unavailable: {exc}")
        cls.root.withdraw()
        source = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/campaign_dashboard_ui.py"
        module = ast.parse(source.read_text(encoding="utf-8-sig"))
        cls.renderer = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                            and node.name == "_render_step1_route_actions")

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.palette = get_theme_palette()
        self.canvas = tk.Canvas(self.root, width=900, height=700, bg=self.palette["campaign_graph_bg"])
        self.style = CanvasWorkflowFocus(self.canvas, self.palette)
        self.commands = {name: Mock() for name in ("zoom_in", "zoom_out", "reset", "history", "guide")}
        self.toolbar = GraphToolbar(self.canvas, self.palette, self.commands)
        self.toolbar._resize(SimpleNamespace(width=900))
        self.toolbar.sync(1.0, "T04 | E2 -> E4T")

    def tearDown(self):
        self.canvas.destroy()
        self.root.update_idletasks()

    def callback_namespace(self, **extra):
        """Run the actual outer renderer callbacks on a small real graph."""
        namespace = {
            "tk": tk, "canvas": self.canvas, "graph_toolbar": self.toolbar,
            "workflow_focus_style": self.style, "_workflow_focus_scope": lambda: scope("e2_to_e4"),
            "_sync_graph_toolbar": lambda: self.toolbar.sync(1.0, "T04 | E2 -> E4T"),
            "_graph_zoom": lambda: 1.4, "_graph_pan": lambda: (40.0, -20.0),
            "graph_render_helpers": {}, "_raise_graph_interactive_layers": Mock(),
            "_scale_graph_preview_text_items": Mock(),
        }
        wanted = {"_apply_graph_view", "_redraw_graph_fixed_overlay", "_preview_graph_zoom_transform",
                  "_raise_graph_overlay_layers", "_flush_graph_pan_move", "_request_graph_redraw"}
        body = [node for node in self.renderer.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        self.assertEqual(len(body), len(wanted))
        exec(compile(ast.Module(body=body, type_ignores=[]), "graph_callbacks", "exec"), namespace)
        namespace.update(extra)
        return namespace

    def test_toolbar_survives_world_zoom_pan_and_frame_deletion(self):
        node = self.canvas.create_rectangle(100, 100, 180, 170, tags=("node-group:E2",))
        fixed = self.canvas.create_text(10, 680, text="legend", tags=("graph_overlay_fixed",))
        before = self.toolbar.coords("button:reset"), self.toolbar.place_info(), self.toolbar.find_all()
        fixed_coords = self.canvas.coords(fixed)
        ns = self.callback_namespace()
        for _ in range(15):
            ns["_apply_graph_view"](900, 700)
            ns["_preview_graph_zoom_transform"](450, 350, 1.035)
            ns["_raise_graph_overlay_layers"]()
        self.assertNotEqual(self.canvas.coords(node), [100, 100, 180, 170])
        self.assertEqual(self.canvas.coords(fixed), fixed_coords)
        self.assertEqual((self.toolbar.coords("button:reset"), self.toolbar.place_info(), self.toolbar.find_all()), before)
        self.canvas.addtag_withtag("graph_redraw_previous_frame", "all")
        self.canvas.delete("graph_redraw_previous_frame")
        self.canvas.delete("all")
        self.assertEqual(self.toolbar.find_all(), before[2])
        self.assertEqual(self.toolbar.winfo_manager(), "place")
        self.toolbar.invoke("history")
        self.commands["history"].assert_called_once()

    def test_partial_overlay_redraw_uses_registered_legend(self):
        old = self.canvas.create_text(10, 680, text="old", tags=("graph_overlay_fixed",))
        def legend():
            self.canvas.create_text(10, 680, text="new", tags=("graph_overlay_fixed",))
        ns = self.callback_namespace(graph_render_helpers={"draw_fixed_legend": legend})
        ns["_redraw_graph_fixed_overlay"]()
        self.assertNotIn(old, self.canvas.find_all())
        items = self.canvas.find_withtag("graph_overlay_fixed")
        self.assertEqual(len(items), 1)
        self.assertEqual(self.canvas.itemcget(items[0], "text"), "new")
        ns["_raise_graph_interactive_layers"].assert_called_once()

    def test_signal_is_occluded_by_all_cards_after_each_frame_and_drag(self):
        edge = self.canvas.create_line(0, 50, 500, 50, fill="#55dcf5", tags=("edge-line:e2_to_e4", "edge_line"))
        node = self.canvas.create_rectangle(100, 20, 200, 80, fill="#141e12", tags=("node-group:E2",))
        gate = self.canvas.create_rectangle(300, 20, 400, 80, fill="#141e12", tags=("gate-group:e3_to_e4",))
        tooltip = self.canvas.create_rectangle(310, 5, 410, 30, fill="#ffffff", tags=("gate_selector_tooltip",))
        ns = self.callback_namespace()
        for dragged in (None, node, gate):
            self.canvas.delete("graph_attention_flow")
            signal = self.canvas.create_line(0, 50, 500, 50, fill="#55dcf5", width=3,
                                             tags=("graph_attention_flow",))
            if dragged is not None:
                self.canvas.tag_raise(dragged)
            order_before = list(self.canvas.find_all())
            ns["_raise_graph_overlay_layers"]()
            order = list(self.canvas.find_all())
            self.assertLess(order.index(edge), order.index(signal))
            for card in (node, gate, tooltip):
                self.assertLess(order.index(signal), order.index(card))
            self.assertEqual(order_before.index(node) < order_before.index(gate), order.index(node) < order.index(gate))
            # At an exposed section of the edge the signal is visible; at a
            # crossing the opaque card is the top item on the real Tk canvas.
            for x, expected in ((50, signal), (150, node), (350, gate)):
                self.assertEqual(self.canvas.find_overlapping(x, 50, x, 50)[-1], expected)

    def test_pan_only_moves_world_items(self):
        node = self.canvas.create_rectangle(100, 100, 180, 170, tags=("node-group:E2",))
        legend = self.canvas.create_text(10, 680, text="legend", tags=("graph_overlay_fixed",))
        ns = self.callback_namespace(graph_pan_state={"active": True, "start_x": 100, "start_y": 80,
            "pending_x": 120, "pending_y": 110, "orig_x": 40, "orig_y": -20},
            gate_drag_state={}, node_drag_state={}, graph_view={})
        ns["_flush_graph_pan_move"]()
        self.assertEqual(self.canvas.coords(node), [120, 130, 200, 200])
        self.assertEqual(self.canvas.coords(legend), [10, 680])

    def gate_layout_namespace(self, *, width=900, zoom=1.0, local_zoom=1.0, title="T02 TRENUJ MODEL ZNAKÓW", status="WYBIERZ"):
        from tkinter import font as tkfont
        draw = next(node for node in self.renderer.body if isinstance(node, ast.FunctionDef) and node.name == "_draw")
        gate = next(node for node in draw.body if isinstance(node, ast.FunctionDef) and node.name == "_draw_gate")
        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        start = next(index for index, node in enumerate(gate.body) if assigns(node, "title_font"))
        end = next(index for index, node in enumerate(gate.body) if assigns(node, "gate_h"))
        fonts = [node for node in draw.body if isinstance(node, ast.FunctionDef) and node.name in {"_gate_font", "_gate_layout_font"}]
        ns = {"tkfont": tkfont, "canvas": self.canvas, "width": width, "local_zoom": local_zoom,
              "zoom_for_fonts": zoom, "graph_zoom_scale": zoom, "graph_base_font_scale": 1.22,
              "gate_title_text": title, "status_text": status, "resource_status_value": "OK",
              "work_status_value": "KONTROLUJ AT", "approve_display_label": "ZATWIERDŹ",
              "rows": [(label, "", None, False) for label in ("BRAMKA", "ZASOBY", "PRACA", "ZATWIERDŹ")]}
        exec(compile(ast.Module(body=fonts + gate.body[start:end + 1], type_ignores=[]), "gate_dimensions", "exec"), ns)
        ns["status_body"] = next(node.body for node in ast.walk(gate)
                                 if isinstance(node, ast.If) and ast.unparse(node.test) == "idx == 0")
        return ns

    def test_gate_dimensions_stay_fixed_while_sidebar_changes_viewport(self):
        from auto_annotation_tool.campaign_transition_specs import TRANSITION_SPECS
        for spec in TRANSITION_SPECS:
            for local_zoom in (1.0, 1.4):
                with self.subTest(gate=spec.badge_id, local_zoom=local_zoom):
                    sizes = []
                    for width in (900, 980, 1100, 1240, 1100, 900):
                        ns = self.gate_layout_namespace(width=width, local_zoom=local_zoom,
                            title=f"{spec.badge_id} {spec.badge_label}".upper())
                        sizes.append((ns["gate_w"], ns["gate_h"], ns["title_h"], ns["row_heights"]))
                    self.assertTrue(all(size == sizes[0] for size in sizes))
                    if local_zoom > 1:
                        normal = self.gate_layout_namespace(title=f"{spec.badge_id} {spec.badge_label}".upper())
                        self.assertGreater(sizes[0][0], normal["gate_w"])

    def test_gate_status_leaves_row_border_visible_at_all_zoom_levels(self):
        for status in ("WYBIERZ", "OTWARTA", "ZAMKNIĘTA", "PRZERWANE", "WYBIERZ WYNIK", "DO KONTROLI"):
            for zoom, local_zoom in ((0.7, 1.0), (1.0, 1.0), (1.4, 1.0), (1.0, 1.4)):
                with self.subTest(status=status, zoom=zoom, local_zoom=local_zoom):
                    self.canvas.delete("all")
                    ns = self.gate_layout_namespace(zoom=zoom, local_zoom=local_zoom, status=status)
                    row_width, row_height = ns["gate_w"], ns["row_heights"]["BRAMKA"]
                    row = self.canvas.create_rectangle(20, 30, 20 + row_width, 30 + row_height,
                        fill="#141e12", outline="#81ecd0", width=1)
                    ns.update(x=20, y0=30, current_row_h=row_height, value=status,
                              row_fill="#141e12", graph_card_muted="#a0b19a", value_color="#81ecd0", row_tags=("status",))
                    exec(compile(ast.Module(body=ns["status_body"], type_ignores=[]), "gate_status", "exec"), ns)
                    self.canvas.scale("all", 0, 0, zoom, zoom)
                    for y in (30 * zoom, (30 + row_height) * zoom):
                        for fraction in (0.6, 0.8, 0.95):
                            x = (20 + row_width * fraction) * zoom
                            self.assertEqual(self.canvas.find_overlapping(x, y, x, y)[-1], row)
                    status_item = next(item for item in self.canvas.find_withtag("status") if self.canvas.type(item) == "text")
                    box = self.canvas.bbox(status_item)
                    self.assertGreater(box[1], 30 * zoom)
                    self.assertLess(box[3], (30 + row_height) * zoom)

    def test_fading_preserves_geometry_bindings_and_visibility(self):
        for theme in THEME_DEFINITIONS:
            with self.subTest(theme=theme):
                self.canvas.delete("all")
                palette = get_theme_palette(theme)
                style = CanvasWorkflowFocus(self.canvas, palette)
                inactive = self.canvas.create_rectangle(20, 30, 200, 90, fill=palette["campaign_gate_surface"],
                    outline=palette["campaign_card_text"], tags=("gate-group:e3_to_e4",))
                label = self.canvas.create_text(40, 50, text="T05", fill=palette["campaign_card_text"],
                    font=("Segoe UI", 12), tags=("gate-group:e3_to_e4",))
                active = self.canvas.create_rectangle(300, 40, 400, 80, fill=palette["campaign_gate_surface"],
                    tags=("gate-group:e2_to_e4",))
                self.canvas.tag_bind("gate-group:e3_to_e4", "<Button-1>", lambda _e: None)
                before = (self.canvas.coords(inactive), self.canvas.itemcget(label, "font"),
                          self.canvas.tag_bind("gate-group:e3_to_e4", "<Button-1>"))
                style.apply(scope("e2_to_e4"))
                dim = self.canvas.itemcget(inactive, "fill")
                self.assertNotEqual(dim, palette["campaign_gate_surface"])
                self.assertEqual(self.canvas.itemcget(active, "fill"), palette["campaign_gate_surface"])
                for _ in range(5):
                    style.apply(scope("e2_to_e4"), refresh_links=True)
                self.assertEqual(self.canvas.itemcget(inactive, "fill"), dim)
                self.assertEqual(self.canvas.itemcget(inactive, "state"), "")
                self.assertEqual((self.canvas.coords(inactive), self.canvas.itemcget(label, "font"),
                    self.canvas.tag_bind("gate-group:e3_to_e4", "<Button-1>")), before)
                style.apply(scope("e3_to_e4"))
                self.assertEqual(self.canvas.itemcget(inactive, "fill"), palette["campaign_gate_surface"])
                self.assertEqual(self.canvas.itemcget(label, "fill"), palette["campaign_card_text"])

    def test_partial_link_redraw_and_new_elements_are_dimmed(self):
        color = self.palette["campaign_edge"]
        line = self.canvas.create_line(10, 50, 500, 50, fill=color, tags=("edge-line:e3_to_e4", "edge_line"))
        self.style.apply(scope("e2_to_e4"))
        expected = self.canvas.itemcget(line, "fill")
        self.canvas.itemconfigure(line, fill=color)
        arrow = self.canvas.create_polygon(490, 40, 500, 50, 490, 60, fill=color,
            tags=("graph_edge_arrow:e3_to_e4", "graph_edge_arrow"))
        self.style.apply(scope("e2_to_e4"), refresh_links=True)
        self.assertEqual(self.canvas.itemcget(line, "fill"), expected)
        self.assertEqual(self.canvas.itemcget(arrow, "fill"), expected)
        self.style.apply(scope("e3_to_e4"))
        self.assertEqual(self.canvas.itemcget(line, "fill"), color)

    def test_clickable_alternative_electrode_stays_visible_when_branch_is_dimmed(self):
        from auto_annotation_tool.gui.campaign_graph_presentation import gate_electrode_style
        for theme in THEME_DEFINITIONS:
            with self.subTest(theme=theme):
                self.canvas.delete("all")
                palette = get_theme_palette(theme)
                color = gate_electrode_style(palette)["fill"]
                alternative = self.canvas.create_rectangle(10, 10, 42, 38, fill=color,
                    tags=("gate-group:e1_to_e2", "gate:e1_to_e2:select", "gate_button"))
                unavailable = self.canvas.create_rectangle(50, 10, 82, 38, fill=color,
                    tags=("gate-group:e2_to_e4",))
                style = CanvasWorkflowFocus(self.canvas, palette)
                style.apply(scope("e1_to_e3"))
                self.assertEqual(self.canvas.itemcget(alternative, "fill"), color)
                self.assertNotEqual(self.canvas.itemcget(unavailable, "fill"), color)

    def test_electrode_hover_tooltip_and_restore_in_all_themes(self):
        from auto_annotation_tool.gui.campaign_graph_presentation import gate_electrode_style
        from auto_annotation_tool.gui.web_slim_scrollbar import blend_hex_colors
        wanted = {"_clear_gate_selector_tooltip", "_show_gate_selector_tooltip"}
        body = [node for node in self.renderer.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        self.assertEqual(len(body), len(wanted))
        def luminance(color):
            channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                      for value in channels]
            return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
        for theme in THEME_DEFINITIONS:
            with self.subTest(theme=theme):
                self.canvas.delete("all")
                palette = get_theme_palette(theme)
                style = gate_electrode_style(palette)
                face = self.canvas.create_rectangle(10, 100, 42, 128, fill=style["fill"],
                    tags=("gate_selector_face:e1_to_e2", "gate:e1_to_e2:select", "gate_button"))
                tag = "gate:e1_to_e2:select"
                geom = dict(x0=10, y0=100, x1=42, y1=128, hover_fill=style["hover"],
                            color=style["border"], gate_id="T01", gate_label="Trenuj model", selected=False)
                ns = {"canvas": self.canvas, "tk": tk, "gate_selector_hover": {}, "gate_geometry": {},
                      "gate_selector_geometry": {tag: geom}, "_screen_x": lambda x, w: x,
                      "_screen_y": lambda y, h: y, "accent": palette["accent"], "palette": palette,
                      "card_bg": palette["panel"], "fg": palette["fg"], "blend_hex_colors": blend_hex_colors,
                      "CAMPAIGN_TRANSITION_GRAPH": GRAPH, "_can_clear_step1_gate_selection": lambda edge: True}
                exec(compile(ast.Module(body=body, type_ignores=[]), "selector_callbacks", "exec"), ns)
                ns["_show_gate_selector_tooltip"](tag)
                self.assertEqual(self.canvas.itemcget(face, "fill"), style["hover"])
                tooltip = self.canvas.find_withtag("gate_selector_tooltip")
                self.assertEqual(len(tooltip), 4)
                background = next(item for item in tooltip if self.canvas.type(item) == "rectangle")
                title = next(item for item in tooltip if self.canvas.type(item) == "text")
                levels = sorted(luminance(self.canvas.itemcget(item, "fill")) for item in (background, title))
                self.assertGreaterEqual((levels[1] + 0.05) / (levels[0] + 0.05), 4.5)
                geom["selected"] = True
                ns["_show_gate_selector_tooltip"](tag)
                texts = [self.canvas.itemcget(item, "text") for item in self.canvas.find_withtag("gate_selector_tooltip")
                         if self.canvas.type(item) == "text"]
                self.assertIn("wyczyścić wybór", " ".join(texts))
                ns["_clear_gate_selector_tooltip"]()
                self.assertEqual(self.canvas.itemcget(face, "fill"), style["fill"])
                self.assertFalse(self.canvas.find_withtag("gate_selector_tooltip"))

    def test_return_edge_is_bright_only_for_training_transition(self):
        color = self.palette["campaign_edge"]
        trunk = self.canvas.create_line(10, 50, 500, 50, fill=color, tags=("graph_return_trunk",))
        self.style.apply(scope("e3_to_e4"))
        self.assertNotEqual(self.canvas.itemcget(trunk, "fill"), color)
        for edge in ("e4t_to_e1", "e4z_to_e1"):
            self.style.apply(scope(edge))
            self.assertEqual(self.canvas.itemcget(trunk, "fill"), color)

    def test_workflow_layer_allows_explicit_drag_to_front(self):
        focused = self.canvas.create_rectangle(10, 10, 200, 100, tags=("gate-group:e2_to_e4",))
        other = self.canvas.create_rectangle(10, 10, 200, 100, tags=("gate-group:e3_to_e4",))
        self.style.raise_scope(scope("e2_to_e4"))
        self.assertEqual(self.canvas.find_all()[-1], focused)
        self.canvas.tag_raise("gate-group:e3_to_e4")
        self.callback_namespace()["_raise_graph_overlay_layers"]()
        self.assertEqual(self.canvas.find_all()[-1], other)

    def test_compact_and_full_toolbar_bounds_and_no_callback_growth(self):
        self.assertGreater(len(self.toolbar.find_withtag("history")), 3)
        initial_commands = len(self.toolbar._tclCommands)
        for width in (300, 400, 600, 900, 1600) * 3:
            self.toolbar._resize(SimpleNamespace(width=width))
            for action in self.commands:
                bounds = self.toolbar.bbox(action)
                self.assertGreaterEqual(bounds[0], 0)
                self.assertLessEqual(bounds[2], self.toolbar._layout_width)
                self.assertLessEqual(bounds[3], self.toolbar._height)
        self.assertEqual(len(self.toolbar._tclCommands), initial_commands)

    def test_toolbar_zoom_does_not_rebuild_icons(self):
        original = self.toolbar.find_all()
        for zoom in (0.65, 1.1, 1.035, 1.8, 2.4):
            self.toolbar.sync(zoom, self.toolbar.context)
            self.assertEqual(self.toolbar.find_all(), original)
            self.assertEqual(self.toolbar.itemcget("zoom_value", "text"), f"{zoom:.0%}")

    def test_toolbar_commands_and_zoom_limits(self):
        for action in self.commands:
            self.assertEqual(self.toolbar.invoke(action), "break")
            self.commands[action].assert_called_once()
        self.toolbar.sync(0.65, self.toolbar.context)
        self.toolbar.invoke("zoom_out")
        self.commands["zoom_out"].assert_called_once()
        self.toolbar.sync(2.4, self.toolbar.context)
        self.toolbar.invoke("zoom_in")
        self.commands["zoom_in"].assert_called_once()

    def test_resize_updates_toolbar_but_destroy_leaves_parent_binding(self):
        callback = Mock()
        binding = self.canvas.bind("<Configure>", callback, add="+")
        self.toolbar.destroy()
        self.assertIn(binding, self.canvas.bind("<Configure>"))

    def test_sidebar_slide_defers_graph_render_until_settled(self):
        owner = SimpleNamespace(_project_sidebar_animating=True)
        draw = Mock()
        redraw_state = {"after_id": None}
        ns = self.callback_namespace(graph_redraw_state=redraw_state,
            graph_pan_state={}, gate_drag_state={}, node_drag_state={},
            _graph_modal_activity_active=lambda: False, _draw=draw, _bind_gate_tags=Mock())
        ns["self"] = owner
        for _ in range(5):
            ns["_request_graph_redraw"](delay_ms=1)
            self.root.tk.call("after", 5)
            self.root.update()
        draw.assert_not_called()
        self.assertIsNone(redraw_state["after_id"])
        owner._project_sidebar_animating = False
        ns["_request_graph_redraw"](delay_ms=1)
        self.root.tk.call("after", 5)
        self.root.update()
        draw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
