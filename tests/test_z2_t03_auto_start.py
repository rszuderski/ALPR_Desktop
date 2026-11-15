from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import copy
import threading

import pytest

from auto_annotation_tool.gui import z2_annotation_process as process
from auto_annotation_tool.gui import z2_model_runtime as models
from auto_annotation_tool.gui.z2_auto_run_result import remove_vehicle_assistance
from auto_annotation_tool.gui import campaign_navigation as navigation
from auto_annotation_tool.gui import z2_context_runtime as context


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FactoryReached(BaseException):
    pass


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("use_vehicle", [False, True])
def test_modal_choice_controls_actual_annotator_even_if_ui_choice_changes(tmp_path, monkeypatch, free, use_vehicle):
    image = tmp_path / "sample.jpg"
    image.touch()
    owner = Mock()
    owner._get_workflow_route.return_value = "auto"
    owner._get_manual_entry_mode.return_value = "new"
    owner._coerce_workflow_step.return_value = "auto_start"
    owner._is_free_mode_session_context.return_value = free
    owner.input_dir_var = Value(str(tmp_path))
    owner.output_dir_var = Value(str(tmp_path))
    owner.mode_var = Value("C: Pojazdy + tablice")
    owner.conf_var = Value(0.25)
    owner.campaign_reuse_manual_var = Value(False)
    owner._normalize_mode_value.side_effect = lambda value=None: models._normalize_mode_value(owner, value)
    owner._mode_uses_vehicle.side_effect = lambda value=None: models._mode_uses_vehicle(owner, value)
    owner._mode_uses_plate.return_value = True
    owner._get_auto_vehicle_choice.return_value = "skip" if use_vehicle else "use"
    owner._resolve_annotation_input_images_dir.return_value = tmp_path
    owner._manual_xml_template_enabled.return_value = False
    owner._manual_vehicle_assist_enabled.return_value = False
    owner._collect_campaign_auto_annotation_sources.return_value = {"image_paths": [image], "image_map": {}}
    owner._collect_preview_manually_touched_filenames.return_value = set()
    owner._campaign_auto_manual_overlay_bundle = {}
    owner._prompt_plate_auto_scope_choice.return_value = {
        "mode": "all", "image_paths": [image], "use_vehicle": use_vehicle,
    }
    owner._dedupe_image_paths_by_name.side_effect = list
    owner._get_campaign_iteration_manifest_image_count.return_value = 0
    owner._get_campaign_iteration_manifest_image_paths.return_value = []
    owner._get_effective_yolo_device_choice.return_value = "CPU"
    owner._device_to_ultralytics.return_value = "cpu"
    owner._confirm_and_download_missing_models.return_value = True
    owner._get_model_path.side_effect = lambda role: tmp_path / f"{role}.pt"
    owner._coerce_annotation_output_dir.return_value = tmp_path
    owner.app.try_begin_exclusive_operation.return_value = (True, "")
    owner._progress_update_lock = threading.Lock()
    owner._plate_auto_scope_progress_modal_active = True
    owner.annotator = None
    plate_factory = Mock(side_effect=FactoryReached)
    combined_factory = Mock(side_effect=FactoryReached)
    monkeypatch.setattr(process, "create_plate_annotator", plate_factory)
    monkeypatch.setattr(process, "create_combined_plate_annotator", combined_factory)
    with pytest.raises(FactoryReached):
        process._start_annotation(owner)
    assert combined_factory.called == use_vehicle
    assert plate_factory.called != use_vehicle
    assert owner._current_run_vehicle_assist == use_vehicle
    roles = [call.args[0] for call in owner._get_model_path.call_args_list]
    assert ("vehicle" in roles) == use_vehicle
    if not free:
        assert owner._collect_campaign_auto_annotation_sources.call_args.kwargs["exclude_manual_touched"] is False


def test_removing_inherited_vehicle_boxes_preserves_plate_geometry_and_rollback():
    plate = SimpleNamespace(label="plate", polygon=[(1, 2), (3, 4)], attributes={"gt": "ABC", "manually_edited": "true"})
    previous = [SimpleNamespace(detections=[plate, SimpleNamespace(label="vehicle")])]
    result = copy.deepcopy(previous)
    assert remove_vehicle_assistance(result) == 1
    assert result[0].detections[0].__dict__ == plate.__dict__
    assert len(previous[0].detections) == 2
    assert remove_vehicle_assistance(result) == 0


def test_t03_navigation_defers_preview_and_does_not_scan_sources_twice(tmp_path, monkeypatch):
    campaign = Mock()
    campaign.get_active_project_name.return_value = "MZ"
    campaign.get_current_iteration_num.return_value = 1
    campaign.get_current_step.return_value = 2
    campaign.get_step1_status.return_value = "approved"
    campaign.get_iteration_path.return_value = "char_from_images"
    campaign.get_dir.return_value = tmp_path
    campaign.get_staging_dir.return_value = tmp_path
    campaign.get_iteration_image_source_dir.return_value = tmp_path
    campaign.get_global_model.return_value = ""
    monkeypatch.setattr(navigation, "CAMPAIGN", campaign)
    tab = Mock()
    tab.open_campaign_step2_entry.return_value = {"ok": True, "reused_loaded_run": True}
    app = Mock(tabs={"annotation": tab})
    app._get_selected_tab_key.return_value = "annotation"
    jobs = []
    host = SimpleNamespace(app=app, frame=SimpleNamespace(after=lambda delay, cb: jobs.append(cb)),
        _get_iteration_target=lambda: "char", _get_annotation_step2_source_state=Mock(),
        _get_char_route_source_state=Mock(), _get_char_route_ready_source=Mock())
    result = navigation._step_goto_auto_annotation(host, preferred_source_context={
        "graph_gate_id": "T03", "graph_edge_key": "e2_to_e3"})
    assert result["pending"]
    jobs.pop(0)()
    kwargs = tab.open_campaign_step2_entry.call_args.kwargs
    assert kwargs["restore_preview"] is False
    assert kwargs["defer_preview_load"] is True
    host._get_annotation_step2_source_state.assert_not_called()
    host._get_char_route_source_state.assert_not_called()
    tab._start_annotation.assert_not_called()


def test_entry_waits_for_deferred_source_before_scanning_action_state():
    owner = Mock()
    owner._campaign_step2_transition_refresh_pending = True
    owner._campaign_step2_transition_defer_source_preview = True
    owner._campaign_step2_transition_skip_heavy_finalize = False
    context._end_campaign_step2_transition(owner)
    assert owner._campaign_step2_transition_in_progress is False
    assert owner._campaign_step2_transition_defer_source_preview is False
    owner._refresh_step2_action_states.assert_not_called()
    owner._refresh_free_mode_workflow_ui.assert_not_called()
