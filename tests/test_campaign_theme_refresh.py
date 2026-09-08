"""A palette switch must not dispatch pending campaign navigation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock

import pytest

from auto_annotation_tool.campaign_transition_graph import CAMPAIGN_TRANSITION_GRAPH
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.tab_campaign import CampaignTab
from auto_annotation_tool.gui import campaign_stage_ui


@pytest.fixture(scope="module")
def renderer():
    source = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/campaign_dashboard_ui.py"
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == "_render_step1_route_actions")


def pending_callback(renderer, *, visible=True, canvas_exists=True, busy=False):
    host = SimpleNamespace(
        frame=Mock(), _pending_gate_action_modal_edge_key="e3_to_e4",
        _pending_gate_action_modal_reason="pz1_ready_return_to_t05_work",
        _pending_gate_action_modal_attempts=12,
    )
    host.frame.winfo_viewable.return_value = visible
    canvas = Mock()
    canvas.winfo_exists.return_value = canvas_exists
    namespace = {
        "self": host, "canvas": canvas, "tk": tk, "CAMPAIGN_TRANSITION_GRAPH": CAMPAIGN_TRANSITION_GRAPH,
        "_graph_modal_activity_active": Mock(return_value=busy),
        "_edge_actions_enabled": Mock(return_value=True), "_open_actions": Mock(), "logger": Mock(),
    }
    node = next(node for node in renderer.body if isinstance(node, ast.FunctionDef)
                and node.name == "_open_pending_gate_action_modal")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pending_gate_action", "exec"), namespace)
    return host, canvas, namespace


@pytest.mark.parametrize("visible,exists", [(False, True), (True, False)])
def test_pending_work_cannot_open_over_pz2_or_from_replaced_graph(renderer, visible, exists):
    host, canvas, ns = pending_callback(renderer, visible=visible, canvas_exists=exists)
    ns["_open_pending_gate_action_modal"]()
    ns["_open_actions"].assert_not_called()
    canvas.after.assert_not_called()
    assert host._pending_gate_action_modal_edge_key == "e3_to_e4"
    assert host._pending_gate_action_modal_attempts == 12


def test_explicit_return_to_visible_campaign_opens_work_once(renderer):
    host, _, ns = pending_callback(renderer)
    ns["_open_pending_gate_action_modal"]()
    ns["_open_pending_gate_action_modal"]()
    ns["_open_actions"].assert_called_once_with("e3_to_e4")
    assert host._pending_gate_action_modal_edge_key == ""


def test_switching_to_pz2_while_waiting_for_another_dialog_cancels_retry(renderer):
    host, canvas, ns = pending_callback(renderer, busy=True)
    ns["_open_pending_gate_action_modal"]()
    retry = canvas.after.call_args.args[1]
    host.frame.winfo_viewable.return_value = False
    ns["_graph_modal_activity_active"].return_value = False
    retry()
    ns["_open_actions"].assert_not_called()
    assert canvas.after.call_count == 1


@pytest.mark.parametrize("theme", ["light_visual_cs", "dark_visual_cs"])
def test_campaign_theme_repaints_graph_without_refreshing_workflow(theme):
    host = Mock()
    host.app.palette = get_theme_palette(theme)
    host._model_status_title_labels = []
    host.wizard_transition_graph_shell.winfo_manager.return_value = "pack"
    CampaignTab.apply_theme(host)
    host._refresh_dashboard.assert_not_called()
    host._refresh_wizard_transition_graph.assert_called_once_with(allow_pending_actions=False)


def test_theme_does_not_build_or_reveal_hidden_campaign_graph():
    host = Mock()
    host.app.palette = get_theme_palette("dark_visual_cs")
    host._model_status_title_labels = []
    host.wizard_transition_graph_shell.winfo_manager.return_value = ""
    CampaignTab.apply_theme(host)
    host._refresh_dashboard.assert_not_called()
    host._refresh_wizard_transition_graph.assert_not_called()


@pytest.mark.parametrize("allow_pending_actions", [False, True])
def test_graph_refresh_preserves_navigation_policy(allow_pending_actions):
    host = Mock()
    campaign_stage_ui._refresh_wizard_transition_graph(host, allow_pending_actions=allow_pending_actions)
    host._render_step1_route_actions.assert_called_once_with(
        host.wizard_transition_graph_body, allow_pending_actions=allow_pending_actions,
    )


@pytest.mark.parametrize("allow_pending_actions", [False, True])
def test_only_workflow_refresh_schedules_pending_work(renderer, allow_pending_actions):
    host, frame, ns = pending_callback(renderer)
    ns.update(frame=frame, allow_pending_actions=allow_pending_actions)
    # Execute the real scheduling block at the end of the graph renderer.
    node = next(node for node in renderer.body if isinstance(node, ast.Try)
                and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "after" and child.args
                        and isinstance(child.args[0], ast.Constant) and child.args[0].value == 220
                        for child in ast.walk(node)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "schedule_gate_work", "exec"), ns)
    if allow_pending_actions:
        frame.after.assert_called_once_with(220, ns["_open_pending_gate_action_modal"])
    else:
        frame.after.assert_not_called()
    assert host._pending_gate_action_modal_edge_key == "e3_to_e4"
