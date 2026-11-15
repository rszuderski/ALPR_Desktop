import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import font as tkfont

import pytest

from auto_annotation_tool.gui.campaign_gate_fields import gate_field_copy, field_layout, draw_field, text_height
from auto_annotation_tool.gui.campaign_graph_presentation import gate_electrode_style
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.web_slim_scrollbar import blend_hex_colors
from auto_annotation_tool.campaign_transition_graph import CAMPAIGN_TRANSITION_GRAPH


@pytest.fixture(scope="module")
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def gate_renderer(canvas, *, theme="dark_graphite", zoom=1.0, local_zoom=1.0):
    """Actual graph drawing function with detached, read-only state examples."""
    source = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/campaign_dashboard_ui.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_render_step1_route_actions")
    draw = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "_draw")
    functions = [node for node in draw.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_draw_gate", "_gate_font", "_gate_layout_font"}]
    case = {"status": "ZAMKNIĘTA", "resources": "Brak: tablice [OK]", "work": "PRACUJ W Z2",
            "approve": "ZATWIERDŹ", "ready": False, "enabled": True, "x": 60, "y": 30,
            "title": "Przejście do znaków", "gate": "T03", "interrupted": 0}
    palette = get_theme_palette(theme)
    ns = dict(tk=tk, tkfont=tkfont, canvas=canvas, palette=palette,
              gate_field_copy=gate_field_copy, field_layout=field_layout, draw_field=draw_field,
              text_height=text_height, blend_hex_colors=blend_hex_colors, gate_electrode_style=gate_electrode_style,
              CAMPAIGN_TRANSITION_GRAPH=CAMPAIGN_TRANSITION_GRAPH, stage_pos={}, node_y=450,
              width=1500, graph_zoom_scale=zoom, zoom_for_fonts=zoom, graph_base_font_scale=1.22,
              gate_offsets={}, free_drag_extent=10000, active_project_name="read-only-preview",
              current_iteration=4, current_step=2, selected_path="char_from_images",
              approve_blink_seen=set(), approve_blink_active_keys=set(), pending_gate_zoom_badges=[])
    for key in ("gate_geometry", "gate_connector_items", "gate_selector_geometry", "gate_selector_arc_geometry", "gate_field_geometry"):
        ns[key] = {}
    for key in ("graph_gate_surface", "graph_gate_outline", "graph_shadow", "graph_card_text",
                "graph_card_disabled", "graph_card_success", "graph_card_accent", "graph_card_muted",
                "graph_card_warning", "graph_card_error"):
        ns[key] = palette.get("campaign_" + key.removeprefix("graph_"), palette.get("fg", "#eeeeee"))
    ns.update(card_bg=palette["campaign_graph_bg"], active_edge_color=palette["accent"],
              muted_dim=palette["muted"], warning=palette["warning"])
    ns.update({name: lambda *args: True for name in ("_t07_edge_visible", "_edge_gate_active", "_edge_operable", "_edge_route_highlighted")})
    ns.update({name: lambda *args: case["enabled"] for name in ("_edge_selected", "_edge_fields_enabled", "_edge_resources_enabled", "_edge_actions_enabled", "_edge_select_enabled")})
    ns.update({name: lambda *args: False for name in ("_edge_completed", "_is_t07_graph_edge")})
    ns.update({name: lambda: {} for name in ("_get_t07_repair_interruption_state", "_current_t07_training_finish_state",
              "_current_step4_work_interruption_state", "_current_t07_training_candidate_state", "_get_t07_pending_repair_approved_state")})
    ns.update(
        _t06_interrupted_work_state=lambda: {"unpromoted_approved_images": case["interrupted"]} if case["interrupted"] else {},
        _edge_status=lambda edge: (case["status"], palette["warning"]),
        _edge_anchor_point=lambda edge: (case["x"] + 100, case["y"] + 400),
        _edge_spec=lambda edge: SimpleNamespace(badge_label=case["title"], badge_id=case["gate"]),
        _visible_badge_id=lambda value: value, _gate_local_zoom=lambda edge: local_zoom,
        _edge_resource_badge_status=lambda edge: case["resources"],
        _edge_work_compact_status=lambda edge: case["work"],
        _edge_approve_label=lambda edge: case["approve"], _edge_approve_enabled=lambda edge: case["ready"],
        _stable_gate_position=lambda key, *args: (case["x"], case["y"]),
        _graph_card_contrast_color=lambda color: color,
        _connector_anchor_point_for_edge=lambda edge, x, y: (x, y + 400),
        _connector_coords_for_edge=lambda edge, x0, y0, x1, y1: (x0, y0, x1, y1),
        _draw_corner_drag_handle=lambda *args, **kwargs: None,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_gate_renderer", "exec"), ns)
    return ns, case


def test_dynamic_copy_keeps_counts_destinations_and_iteration_provenance():
    copy = gate_field_copy("PRACA", "PRZERWANE +1128 OK", enabled=True)
    assert copy.primary == "Praca przerwana" and "1128" in copy.detail
    assert "Z3" in gate_field_copy("PRACA", "ZATWIERDŹ -> Z3").detail
    approved = gate_field_copy("ZATWIERDŹ", "", enabled=True,
                               approve_label="ZATWIERDŹ\nAT(IT4)  MZ(IT2)")
    assert approved.detail == "Anotacje tablic · iteracja 4\nModel znaków · iteracja 2"
    assert gate_field_copy("ZASOBY", "MODEL_CUSTOM_V3.pt").primary == "MODEL_CUSTOM_V3.pt"
    assert gate_field_copy("ZASOBY", "OK: AZ z IT12").detail == "AZ z iteracji 12"


@pytest.mark.parametrize("zoom,local_zoom", [(0.7, 1.0), (1.0, 1.0), (1.4, 1.0), (1.0, 1.4)])
def test_real_gate_dynamic_labels_fit_their_fields_and_keep_action_tags(root, zoom, local_zoom):
    canvas = tk.Canvas(root)
    ns, case = gate_renderer(canvas, zoom=zoom, local_zoom=local_zoom)
    case.update(resources="OK: AZ z IT12 (z poprzednich iteracji)", ready=True,
                approve="ZATWIERDŹ\nAT(IT12)  MZ(IT3)", interrupted=123456,
                status="PRZERWANE", title="Przygotowanie datasetu znaków do treningu")
    edge = CAMPAIGN_TRANSITION_GRAPH.get_edge("e3_to_e4")
    ns["_draw_gate"](edge)
    canvas.scale("all", 0, 0, zoom, zoom)
    for suffix in ("resources", "actions", "approve"):
        tag = f"gate:{edge.key}:{suffix}"
        items = canvas.find_withtag(tag)
        assert items, tag
        rect = next(item for item in items if canvas.type(item) == "rectangle")
        left, top, right, bottom = canvas.coords(rect)
        for item in items:
            if canvas.type(item) != "text":
                continue
            x0, y0, x1, y1 = canvas.bbox(item)
            assert x0 >= left and x1 <= right, (suffix, canvas.itemcget(item, "text"))
            assert y0 >= top and y1 <= bottom, (suffix, canvas.itemcget(item, "text"))
    assert any("123456" in canvas.itemcget(item, "text") for item in canvas.find_withtag("gate_field_detail"))
    assert not canvas.find_overlapping(-10010, -10010, -9000, -9000)
    canvas.destroy()


def test_disabled_fields_cannot_invoke_actions_and_measurement_cache_is_reused(root):
    canvas = tk.Canvas(root)
    ns, case = gate_renderer(canvas)
    case.update(enabled=False, status="WYBIERZ")
    edge = CAMPAIGN_TRANSITION_GRAPH.get_edge("e2_to_e3")
    ns["_draw_gate"](edge)
    cache = dict(canvas._gate_field_text_heights)
    canvas.delete("all")
    ns["_draw_gate"](edge)
    assert cache == canvas._gate_field_text_heights
    for suffix in ("resources", "actions", "approve"):
        assert not canvas.find_withtag(f"gate:{edge.key}:{suffix}")
    assert canvas.find_withtag("gate_field_primary")
    canvas.destroy()
