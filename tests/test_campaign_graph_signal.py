"""Exercise the graph's actual animation callbacks with a controlled clock."""

import ast
from math import hypot
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock

import pytest

from auto_annotation_tool.campaign_transition_graph import CAMPAIGN_TRANSITION_GRAPH as GRAPH
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.campaign_graph_presentation import GraphSignalMotion, graph_signal_color
from auto_annotation_tool.gui.web_slim_scrollbar import blend_hex_colors


@pytest.fixture
def animation():
    source = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/campaign_dashboard_ui.py"
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    renderer = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_render_step1_route_actions")
    draw = next(node for node in renderer.body if isinstance(node, ast.FunctionDef) and node.name == "_draw")
    wanted = {"_polyline_length", "_point_on_polyline", "_draw_attention_particles",
              "_maybe_start_graph_attention_flow"}
    body = [node for node in draw.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(body) == len(wanted)
    state = SimpleNamespace(now=0.0, modal=False, lines=[], pending=None, zoom=1.0)
    paths = {"e1_to_e2": [(0.0, 0.0), (100.0, 0.0)],
             "e1_to_e3": [(0.0, 100.0), (1000.0, 100.0)]}
    canvas = Mock()
    canvas.winfo_exists.return_value = True
    canvas.delete.side_effect = lambda *args: state.lines.clear()
    canvas.create_line.side_effect = lambda *coords, **kw: state.lines.append((coords, kw))
    def after(ms, callback):
        state.pending = callback
        return "timer"
    canvas.after.side_effect = after
    palette = get_theme_palette()
    namespace = {
        "tk": tk, "hypot": hypot, "canvas": canvas, "palette": palette,
        "card_bg": palette["panel"], "blend_hex_colors": blend_hex_colors,
        "GraphSignalMotion": GraphSignalMotion, "graph_signal_color": graph_signal_color,
        "CAMPAIGN_TRANSITION_GRAPH": GRAPH, "active_gate_keys": list(paths),
        "gate_geometry": dict.fromkeys(paths, {}), "attention_seen": set(), "attention_state": {},
        "active_project_name": "signal-test", "current_iteration": 1, "current_step": 1,
        "_graph_attention_modal_open": lambda: False, "_cancel_graph_attention_flow": Mock(),
        "_graph_modal_activity_active": lambda: state.modal, "perf_counter": lambda: state.now,
        "_attention_path_for_edge": lambda edge: paths[edge.key],
        "_raise_graph_interactive_layers": Mock(), "_graph_zoom": lambda: state.zoom,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "graph_signal_callbacks", "exec"), namespace)
    namespace["_maybe_start_graph_attention_flow"]()
    assert state.pending is not None
    def frame(now, *, modal=False):
        state.now, state.modal = now, modal
        callback, state.pending = state.pending, None
        assert callable(callback)
        callback()
        return [(sum(coords[::2]) / 2, sum(coords[1::2]) / 2) for coords, kw in state.lines]
    state.frame, state.namespace, state.paths = frame, namespace, paths
    return state


@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
def test_short_and_long_edges_move_same_screen_distance(animation, zoom):
    animation.zoom = zoom
    centers = animation.frame(0.5)
    assert max(x for x, y in centers if y == 0) == pytest.approx(56)
    assert max(x for x, y in centers if y == 100) == pytest.approx(56)
    centers = animation.frame(0.7)
    assert max(x for x, y in centers if y == 0) == pytest.approx(80)
    assert max(x for x, y in centers if y == 100) == pytest.approx(80)


def test_short_edge_finishes_while_long_edge_keeps_traveling(animation):
    centers = animation.frame(1.2)
    assert centers and all(y == 100 for x, y in centers)
    assert animation.pending is not None
    assert not animation.namespace["attention_seen"]
    assert animation.frame(8.7) == []
    assert animation.pending is None
    assert animation.namespace["attention_seen"]


def test_signal_follows_polyline_distance_through_corner(animation):
    animation.paths["e1_to_e2"] = [(0, 0), (40, 0), (40, 100)]
    centers = animation.frame(0.5)
    assert centers[0] == pytest.approx((40, 16))


def test_modal_pauses_signal_without_jump_on_resume(animation):
    before = animation.frame(0.5)
    assert animation.frame(1.0, modal=True) == []
    assert animation.frame(20.0, modal=True) == []
    assert animation.frame(21.0) == before
    centers = animation.frame(21.2)
    assert max(x for x, y in centers if y == 100) == pytest.approx(80)


@pytest.mark.parametrize("cadence", [1 / 60, 1 / 30, 0.25])
def test_speed_independent_of_frame_cadence(cadence):
    motion = GraphSignalMotion(0)
    for step in range(1, round(1 / cadence) + 1):
        motion.advance(step * cadence)
    assert motion.distance == pytest.approx(116)
