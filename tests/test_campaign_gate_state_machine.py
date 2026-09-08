import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.campaign_transition_evaluator import CampaignTransitionEvalContext, is_transition_completed, is_transition_ready
from auto_annotation_tool.campaign_transition_specs import get_transition_specs_for_edge
from auto_annotation_tool.campaign_resource_state import CampaignResourceSnapshot
from auto_annotation_tool.gui import campaign_graph_actions as actions
from auto_annotation_tool.gui import campaign_navigation as navigation
from auto_annotation_tool.gui.campaign_t02_source import resolve_t02_review_source
from auto_annotation_tool.gui import z2_manifest_runtime
from auto_annotation_tool.gui import campaign_step1_ingest


@pytest.fixture
def campaign(monkeypatch):
    state = {"step": 1, "iteration": 3, "path": "char_from_ready_plates", "target": "char", "project": "demo"}
    manager = Mock()
    manager.get_current_step.side_effect = lambda: state["step"]
    manager.get_current_iteration_num.side_effect = lambda: state["iteration"]
    manager.get_active_project_name.side_effect = lambda: state["project"]
    manager.get_iteration_path.side_effect = lambda: state["path"]
    manager.get_iteration_target.side_effect = lambda: state["target"]
    manager.set_current_step.side_effect = lambda value: state.update(step=value)
    for step in range(1, 5):
        getattr(manager, f"get_step{step}_status").side_effect = lambda step=step: state.get(f"status{step}", "pending")
        getattr(manager, f"approve_step{step}").side_effect = lambda step=step: state.update({f"status{step}": "approved"})
    manager.get_project_start_plate_source.return_value = {}
    manager.get_iteration_state.return_value = {}
    manager.get_plate_approved_set_stats.return_value = {"plates": 14, "images": 13}
    for module in (actions, navigation):
        monkeypatch.setattr(module, "CAMPAIGN", manager)
    return manager, state


def source_files(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "images").mkdir()
    (run / "annotations.xml").write_text("<annotations/>", encoding="utf-8")
    return {"run_dir": str(run), "xml_path": str(run / "annotations.xml"), "images_dir": str(run / "images")}


def test_t02_uses_approved_pool_and_resumes_same_work(campaign, tmp_path):
    manager, _ = campaign
    builder = Mock(return_value=source_files(tmp_path))
    assert resolve_t02_review_source(manager)["needs_approved_source"]
    result = resolve_t02_review_source(manager, build_approved_source=builder)
    assert result["ok"] and result["context"]["input_source"] == "t02_project_approved"
    manager.get_iteration_state.return_value = manager.upsert_iteration_state.call_args.kwargs["updates"]
    resumed = resolve_t02_review_source(manager, build_approved_source=builder)
    assert resumed["context"] == result["context"]
    builder.assert_called_once()
    manager.approve_step1.assert_not_called()


def test_t02_import_has_priority_and_broken_import_is_not_replaced(campaign, tmp_path):
    manager, _ = campaign
    source = source_files(tmp_path)
    manager.get_project_start_plate_source.return_value = {"source_xml_path": source["xml_path"], "source_input_path": source["images_dir"]}
    builder = Mock()
    result = resolve_t02_review_source(manager, build_approved_source=builder)
    assert result["context"]["xml_path"] == source["xml_path"]
    Path(source["xml_path"]).unlink()
    assert resolve_t02_review_source(manager, build_approved_source=builder)["reason"] == "missing_t02_xml"
    builder.assert_not_called()


def test_t02_missing_source_does_not_log_success_or_change_path(campaign):
    manager, state = campaign
    manager.get_plate_approved_set_stats.return_value = {}
    host = SimpleNamespace(app=Mock())
    host._step_goto_auto_annotation = lambda **kwargs: navigation._step_goto_auto_annotation(host, **kwargs)
    result = actions.execute_campaign_graph_action(host, "open_z2_campaign_context", payload={"context": {"z2_work_mode": "t02_at_review"}})
    assert not result.ok and "Wskaż import AT" in result.message
    assert manager.append_project_history_event.call_args.kwargs["status"] == "error"
    manager.set_iteration_path.assert_not_called()
    assert state["step"] == 1


def test_t02_keeps_already_approved_images_visible_even_with_warm_hide_cache():
    host = SimpleNamespace(_is_free_mode_session_context=lambda: False,
        _campaign_graph_entry_context={"graph_edge_key": "e1_to_e3"},
        _campaign_hidden_project_approved_filenames={"old.jpg"},
        _campaign_hidden_project_approved_runtime_cache={"names": {"old.jpg"}, "expires_at": float("inf")},
        _get_campaign_plate_approved_filenames=Mock(side_effect=AssertionError("T02 must not hide the pool")))
    assert z2_manifest_runtime._get_campaign_hidden_project_approved_filenames_runtime(host) == set()


def test_t02_missing_images_reports_problem_without_rebuilding(campaign, tmp_path):
    manager, _ = campaign
    source = source_files(tmp_path)
    Path(source["images_dir"]).rmdir()
    manager.get_project_start_plate_source.return_value = {"source_run_path": source["run_dir"], "source_input_path": source["images_dir"]}
    result = resolve_t02_review_source(manager)
    assert not result["ok"] and result["reason"] == "missing_t02_images"


@pytest.mark.parametrize("confirmed", [False, True])
def test_t02_actual_approval_preserves_e1_on_cancel_and_skips_e2_on_confirm(campaign, monkeypatch, confirmed):
    manager, state = campaign
    monkeypatch.setattr(campaign_step1_ingest, "CAMPAIGN", manager)
    host = SimpleNamespace(app=Mock(), frame=None, _get_iteration_target=lambda: "char",
        _get_step1_char_route_preflight_state=lambda: {"material_ready": True, "plate_material_count": 14,
            "min_plates": 10, "source_image_count": 13}, _refresh_active_project_wizard_only=Mock())
    host.app.themed_confirm.return_value = confirmed
    host._approve_step1_with_existing_char_material = lambda: campaign_step1_ingest._approve_step1_with_existing_char_material(host)
    result = actions.execute_campaign_graph_action(host, "approve_step1_ready_plates", payload={"context": {"graph_edge_key": "e1_to_e3"}})
    assert result.ok == confirmed
    assert state["step"] == (3 if confirmed else 1)
    assert manager.get_step1_status() == ("approved" if confirmed else "pending")
    assert manager.get_step2_status() == ("approved" if confirmed else "pending")


GATES = [
    ("e1_to_e2", "plate_training", "approve_step1", "_approve_step1_from_wizard", 1, 2, "plate"),
    ("e1_to_e2", "char_from_images", "approve_step1", "_approve_step1_from_wizard", 1, 2, "char"),
    ("e1_to_e3", "char_from_ready_plates", "approve_step1_ready_plates", "_approve_step1_with_existing_char_material", 1, 3, "char"),
    ("e2_to_e3", "char_from_images", "approve_step2", "_approve_step2_from_wizard", 2, 3, "char"),
    ("e2_to_e4", "plate_training", "approve_step2", "_approve_step2_from_wizard", 2, 4, "plate"),
    ("e3_to_e4", "char_from_ready_plates", "approve_step3", "_approve_step3_from_wizard", 3, 4, "char"),
    ("e4_to_e1", "plate_training", "approve_step4", "_finish_step4_iteration", 4, 5, "plate"),
    ("e4_to_e1", "char_from_images", "approve_step4_without_training", "_finish_step4_without_training", 4, 5, "char"),
]


@pytest.mark.parametrize("edge,path,action,method,step,next_step,target", GATES)
@pytest.mark.parametrize("outcome", ["cancel", "noop", "exception", "approve"])
def test_approval_records_only_completed_transitions(campaign, monkeypatch, edge, path, action, method, step, next_step, target, outcome):
    manager, state = campaign
    state.update(step=step, path=path, target=target)
    monkeypatch.setattr(actions, "_current_step4_without_training_decision_ready", lambda: False)
    monkeypatch.setattr(actions, "_current_step4_training_finish_ready", lambda host: True)
    monkeypatch.setattr(actions, "_record_graph_action_history", history := Mock())

    def invoke(*args, **kwargs):
        if outcome == "exception":
            raise RuntimeError("backend failure")
        if outcome == "approve":
            state.update({"step": next_step, f"status{step}": "approved"})
            return True
        return False if outcome == "cancel" else None

    host = SimpleNamespace(app=Mock(), **{method: Mock(side_effect=invoke)})
    spec = get_transition_specs_for_edge(edge)[0]
    result = actions.execute_campaign_graph_action(host, action, payload={"context": {
        "graph_edge_key": edge, "graph_gate_id": spec.badge_id, "graph_transition_target": spec.target}})
    assert result.ok == (outcome == "approve")
    assert history.call_args.args[2].ok == result.ok
    assert history.call_args.kwargs["iteration_num"] == 3
    if outcome != "approve":
        assert state["step"] == step


@pytest.mark.parametrize("path", ["plate_training", "char_from_images", "char_from_ready_plates"])
def test_completed_gates_belong_only_to_selected_path(path):
    ctx = CampaignTransitionEvalContext(selected_path=path, explicit_selected_path=path, current_step=4,
        stage_status={"step1": "approved", "step2": "approved", "step3": "approved"})
    for edge in ("e1_to_e2", "e1_to_e3", "e2_to_e3", "e2_to_e4", "e3_to_e4"):
        spec = get_transition_specs_for_edge(edge)[0]
        expected = path in {"plate_training", "char_from_images"} if edge == "e1_to_e2" else spec.path_key == path
        if edge == "e3_to_e4":
            expected = path in {"char_from_images", "char_from_ready_plates"}
        assert is_transition_completed(spec, ctx) == expected


@pytest.mark.parametrize("iteration", [1, 2, 3])
@pytest.mark.parametrize("path", ["plate_training", "char_from_images", "char_from_ready_plates"])
def test_old_resources_do_not_unlock_future_stages(iteration, path):
    snapshots = {key: CampaignResourceSnapshot(key, key, key.upper(), key, source="previous iteration",
                 counter_text="1000", tone="success") for key in ("approved_plates", "char_dataset", "training_result")}
    ctx = CampaignTransitionEvalContext(selected_path=path, current_step=1, current_iteration=iteration,
        material_ready=True, plate_material_count=1000, image_count=1000, resource_snapshots=snapshots)
    for edge in ("e2_to_e3", "e2_to_e4", "e3_to_e4", "e4_to_e1"):
        assert not is_transition_ready(get_transition_specs_for_edge(edge)[0], ctx)


@pytest.mark.parametrize("kind", ["training", "skip"])
@pytest.mark.parametrize("record_iteration,record_target,expected", [(2, "char", False), (3, "plate", False), (3, "char", True)])
def test_t06_decision_must_match_iteration_and_target(campaign, tmp_path, kind, record_iteration, record_target, expected):
    manager, state = campaign
    state["step"] = 4
    model = tmp_path / "best.pt"
    model.touch()
    record = {"ready": True, "iteration": record_iteration, "target": record_target,
              "run_id": "run_003", "model_path": str(model), "selection_confirmed": True}
    if kind == "skip":
        manager.get_step4_without_training_decision.return_value = record
        assert actions._current_step4_without_training_decision_ready() == expected
    else:
        manager.get_step4_finish_state.return_value = record
        host = SimpleNamespace(app=SimpleNamespace(tabs={}))
        assert actions._current_step4_training_finish_ready(host) == expected


@pytest.mark.parametrize("step,path", [(1, "plate_training"), (2, "char_from_images"), (4, "plate_training")])
def test_stale_or_wrong_route_approval_does_not_call_backend(campaign, step, path):
    _, state = campaign
    state.update(step=step, path=path)
    host = SimpleNamespace(_approve_step2_from_wizard=Mock())
    result = actions.execute_campaign_graph_action(host, "approve_step2", payload={"context": {"graph_edge_key": "e2_to_e4"}})
    assert not result.ok
    host._approve_step2_from_wizard.assert_not_called()


@pytest.mark.parametrize("ok", [False, True])
def test_deferred_navigation_history_uses_completion_not_request(campaign, monkeypatch, ok):
    _, state = campaign
    monkeypatch.setattr(actions, "_record_graph_action_history", history := Mock())
    host = SimpleNamespace(_step_goto_auto_annotation=Mock(return_value={"ok": True, "pending": True}))
    result = actions.execute_campaign_graph_action(host, "open_z2_campaign_context")
    assert result.pending
    history.assert_not_called()
    state["iteration"] = 4
    host._step_goto_auto_annotation.call_args.kwargs["on_complete"]({"ok": ok, "message": "wynik"})
    assert history.call_args.args[2].ok == ok
    assert history.call_args.kwargs["iteration_num"] == 3


class Frame:
    def __init__(self):
        self.callbacks = []

    def after(self, delay, callback):
        self.callbacks.append(callback)


@pytest.mark.parametrize("failure", ["entry", "switch", "no_switch", "project_changed", "none"])
def test_navigation_reports_async_result_and_keeps_t02_image_source(campaign, tmp_path, failure):
    manager, state = campaign
    source = source_files(tmp_path)
    manager.get_project_start_plate_source.return_value = {"source_run_path": source["run_dir"], "source_input_path": source["images_dir"]}
    manager.get_dir.return_value = tmp_path
    manager.get_staging_dir.return_value = tmp_path
    manager.get_iteration_image_source_dir.return_value = tmp_path
    manager.get_global_model.return_value = ""
    annotation = Mock()
    annotation.open_campaign_step2_entry.return_value = {"ok": True, "reused_loaded_t02_run": True}
    if failure == "entry":
        annotation.open_campaign_step2_entry.side_effect = RuntimeError("entry failed")
    app = Mock(tabs={"annotation": annotation})
    app._get_selected_tab_key.return_value = "campaign" if failure == "no_switch" else "annotation"
    if failure == "switch":
        app.open_controlled_tab.side_effect = RuntimeError("switch failed")
    host = SimpleNamespace(app=app, frame=Frame(), _get_iteration_target=lambda: "char",
        _get_annotation_step2_source_state=lambda target: {}, _get_char_route_source_state=lambda: {},
        _get_char_route_ready_source=lambda: {})
    done = Mock()
    result = navigation._step_goto_auto_annotation(host, preferred_source_context={"z2_work_mode": "t02_at_review"}, on_complete=done)
    assert result["pending"]
    done.assert_not_called()
    if failure == "project_changed":
        state["project"] = "other"
    host.frame.callbacks.pop(0)()
    assert done.call_args.args[0]["ok"] == (failure == "none")
    if failure != "project_changed":
        assert annotation.open_campaign_step2_entry.call_args.kwargs["source_context"]["input_dir"] == source["images_dir"]
    else:
        annotation.open_campaign_step2_entry.assert_not_called()
