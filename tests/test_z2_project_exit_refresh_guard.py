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
    def after_idle(self, callback):
        self.after_idle_callback = callback


def _host():
    host = SimpleNamespace(
        _workflow_ui_refresh_in_progress=True,
        _workflow_ui_refresh_pending=True,
        _pre_campaign_free_mode_snapshot=None,
        free_mode_screen_var=Var("route_choice"),
        workflow_route_var=Var(""),
        workflow_step_var=Var(""),
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
        _manual_review_active=False,
        _manual_review_from_auto=False,
        _manual_review_origin_route="",
        _manual_review_export_ready=False,
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
        return None

    host._apply_free_mode_session_snapshot = apply_snapshot
    return host


def test_project_exit_clears_stale_workflow_refresh_guard():
    host = _host()

    runtime.clear_campaign_context(host, restore_free_mode_preview=False)

    assert host._workflow_ui_refresh_in_progress is False
    assert host._workflow_ui_refresh_pending is False
    host._refresh_free_mode_workflow_ui.assert_called()


def test_first_free_mode_refresh_is_not_blocked_by_old_campaign_guard():
    host = _host()
    observed = []

    def refresh():
        observed.append((
            host._workflow_ui_refresh_in_progress,
            host._workflow_ui_refresh_pending,
        ))

    host._refresh_free_mode_workflow_ui = refresh

    runtime.clear_campaign_context(host, restore_free_mode_preview=False)

    assert observed
    assert observed[0] == (False, False)
