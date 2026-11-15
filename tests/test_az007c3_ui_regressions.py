from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool import campaign_manager
from auto_annotation_tool.campaign_transition_evaluator import is_transition_path_active
from auto_annotation_tool.campaign_transition_specs import get_transition_specs_for_edge
from auto_annotation_tool.gui import z2_campaign_flow


def test_t01_path_is_active_for_both_supported_routes_and_t02_is_specific():
    t01 = get_transition_specs_for_edge("e1_to_e2")[0]
    t02 = get_transition_specs_for_edge("e1_to_e3")[0]

    assert is_transition_path_active(t01, "plate_training")
    assert is_transition_path_active(t01, "char_from_images")
    assert not is_transition_path_active(t01, "char_from_ready_plates")

    assert is_transition_path_active(t02, "char_from_ready_plates")
    assert not is_transition_path_active(t02, "char_from_images")


def test_resource_modal_reuses_transition_path_evaluator():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_dashboard_ui.py"
    )
    source = path.read_text(encoding="utf-8-sig")
    assert "is_transition_path_active(edge_spec, current_path)" in source


def test_manual_t03_without_xml_has_explicit_prepare_xml_fallback():
    start = Mock(name="start_annotation")
    host = SimpleNamespace(
        is_processing=False,
        _campaign_graph_entry_context={
            "graph_edge_key": "e2_to_e3",
            "graph_gate_id": "T03",
        },
        _start_annotation=start,
    )

    with (
        patch.object(campaign_manager.CAMPAIGN, "get_current_step", return_value=2),
        patch.object(campaign_manager.CAMPAIGN, "get_iteration_target", return_value="char"),
    ):
        state = z2_campaign_flow.build_z2_cta_state_campaign(
            host,
            route="manual",
            current_step="manual_start",
            show_workflow_steps=True,
            show_export_followup=False,
            auto_completed=False,
            manual_run_already_created=False,
            input_dir_ready=True,
            auto_setup_pending=False,
            auto_vehicle_choice="skip",
            manual_setup=True,
            campaign_reused_manual_count=0,
        )

    assert state.show_start_controls
    assert state.start_enabled
    assert state.start_text == "Przygotuj roboczy XML Z2"
    assert state.start_command is start
    assert not state.suppress_duplicate_start_cta
