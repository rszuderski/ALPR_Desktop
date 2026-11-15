from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_context_runtime as runtime


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyWidget:
    def delete(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        pass


class DummyFrame:
    def __init__(self):
        self.after_idle_calls = 0

    def after_idle(self, callback):
        self.after_idle_calls += 1


def _host(*, snapshot_raises=False, screen=""):
    host = SimpleNamespace(
        _pre_campaign_free_mode_snapshot=None,
        free_mode_screen_var=Var(screen),
        workflow_route_var=Var("manual"),
        workflow_step_var=Var("manual_start"),
        input_dir_var=Var(""),
        project_paths_info_var=Var(""),
        project_paths_rel_var=Var(""),
        current_annotations=[],
        is_processing=False,
        preview_listbox=DummyWidget(),
        preview_canvas=DummyWidget(),
        log_text=DummyWidget(),
        progress=DummyWidget(),
        start_btn=DummyWidget(),
        stop_btn=DummyWidget(),
        approve_btn=DummyWidget(),
        frame=DummyFrame(),
        _manual_review_active=True,
        _manual_review_from_auto=True,
        _manual_review_origin_route="manual",
        _manual_review_export_ready=True,
        _reset_campaign_step2_runtime_state=Mock(),
        _hide_campaign_step2_splash=Mock(),
        _set_campaign_paths_lock_state=Mock(),
        _clear_preview_editor_state=Mock(),
        _set_status_label_state=Mock(),
        _set_post_annotation_hint=Mock(),
        _set_progress_counters=Mock(),
        _set_annotation_process_log_visibility=Mock(),
        _is_free_mode_session_context=lambda: True,
        _normalize_free_mode_screen_value=lambda value: (
            str(value or "").strip().lower()
            if str(value or "").strip().lower()
            in {"route_choice", "workflow", "auto_summary", "manual_review", "export"}
            else ""
        ),
        _refresh_left_panel_route_copy=Mock(),
        _refresh_detection_configuration_ui=Mock(),
        _refresh_step2_action_states=Mock(),
        _refresh_free_mode_workflow_ui=Mock(),
        flush_free_mode_session_state=Mock(),
    )

    def apply_snapshot(*args, **kwargs):
        if snapshot_raises:
            raise RuntimeError("simulated snapshot failure")
        if not host.free_mode_screen_var.get():
            host.free_mode_screen_var.set("route_choice")

    host._apply_free_mode_session_snapshot = apply_snapshot
    return host


def test_clear_campaign_context_rebuilds_free_mode_even_when_snapshot_restore_fails():
    host = _host(snapshot_raises=True, screen="")

    runtime.clear_campaign_context(host, restore_free_mode_preview=False)

    assert host.free_mode_screen_var.get() == "route_choice"
    assert host.workflow_route_var.get() == ""
    assert host.workflow_step_var.get() == ""
    assert host._manual_review_active is False
    assert host._manual_review_from_auto is False
    host._refresh_left_panel_route_copy.assert_called()
    host._refresh_detection_configuration_ui.assert_called()
    host._refresh_step2_action_states.assert_called()
    host._refresh_free_mode_workflow_ui.assert_called()
    assert host.frame.after_idle_calls == 1


def test_valid_free_mode_workflow_state_is_not_forced_back_to_route_choice():
    host = _host(snapshot_raises=False, screen="workflow")

    runtime.clear_campaign_context(host, restore_free_mode_preview=False)

    assert host.free_mode_screen_var.get() == "workflow"
    assert host.workflow_route_var.get() == "manual"
    assert host.workflow_step_var.get() == "manual_start"
    host._refresh_free_mode_workflow_ui.assert_called()
