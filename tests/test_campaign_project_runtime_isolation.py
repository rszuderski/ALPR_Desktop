from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import campaign_project_browser as browser
from auto_annotation_tool.gui import campaign_stage_logic as stage_logic
from auto_annotation_tool.gui import campaign_step1_ingest as ingest


def test_project_runtime_reset_clears_e1_transients_and_cache():
    owner = SimpleNamespace(
        _ingest_plan_generation_token=("old", 1, "C:/old", 1.0),
        current_ingest_plan={"project": "OLD", "iteration": 1, "selected_total": 232382},
        ingest_plan_items=[{"name": "old.jpg"}],
        last_ingest_snapshot={"old": True},
        _existing_iteration_ingest_plan_signature=("old",),
        _clear_dashboard_perf_cache=Mock(),
    )

    browser._reset_campaign_project_runtime_state(owner)

    assert owner._ingest_plan_generation_token is None
    assert owner.current_ingest_plan == {}
    assert owner.ingest_plan_items == []
    assert owner.last_ingest_snapshot == {}
    assert owner._existing_iteration_ingest_plan_signature is None
    owner._clear_dashboard_perf_cache.assert_called_once_with()


def test_char_preflight_ignores_stale_plan_from_previous_project():
    manager = Mock()
    manager.load_latest_ingest_plan_summary.return_value = {}
    manager.get_active_project_name.return_value = "NEW"
    manager.get_current_iteration_num.return_value = 1

    owner = SimpleNamespace(
        current_ingest_plan={
            "project": "OLD",
            "iteration": 1,
            "selected_total": 232382,
            "selected": [{"name": "old.jpg"}],
        },
        _get_active_step1_draft_plan=lambda: {},
        _get_project_start_effective_images_source=lambda: {
            "iteration_count": 0,
            "effective_count": 0,
            "master_count": 0,
        },
    )

    with patch.object(stage_logic, "CAMPAIGN", manager):
        assert stage_logic._get_step1_char_preflight_image_count(owner) == 0


def test_ensure_ingest_plan_rejects_runtime_plan_from_previous_project():
    manager = Mock()
    manager.get_active_project_name.return_value = "NEW"
    manager.get_current_iteration_num.return_value = 1
    manager.get_master_pool_dir.return_value = None

    owner = SimpleNamespace(
        current_ingest_plan={
            "project": "OLD",
            "iteration": 1,
            "selected_total": 232382,
            "selected": [{"name": "old.jpg"}],
        },
        ingest_plan_items=[{"name": "old.jpg"}],
        _existing_iteration_ingest_plan_signature=("old",),
        last_ingest_snapshot={},
        _load_latest_ingest_plan_for_current_iteration=lambda: {},
        _load_ingest_manifest_cached=lambda: {},
        _build_ingest_plan_from_manifest_for_display=lambda *args, **kwargs: {},
    )

    with patch.object(ingest, "CAMPAIGN", manager):
        assert ingest._ensure_current_ingest_plan_from_master_pool(owner) is False

    assert owner.current_ingest_plan == {}
    assert owner.ingest_plan_items == []
    assert owner._existing_iteration_ingest_plan_signature is None


def test_new_project_resets_e1_runtime_before_creation():
    manager = Mock()
    manager.get_active_project_name.return_value = "OLD"
    manager.create_project.return_value = True

    app = SimpleNamespace(
        themed_ask_string=Mock(return_value="NEW"),
        tabs={},
        campaign_free_mode=False,
        set_campaign_mode=Mock(),
        update_campaign_tab_access=Mock(),
        themed_info=Mock(),
        themed_error=Mock(),
    )
    owner = SimpleNamespace(
        app=app,
        frame=object(),
        _ensure_project_context_is_switchable=Mock(return_value=True),
        _reset_campaign_graph_runtime_state=Mock(),
        _reset_campaign_project_runtime_state=Mock(),
        _release_active_project_resources_before_switch=Mock(),
        _rebuild_wizard_stage_ui=Mock(),
        _refresh_dashboard=Mock(),
    )

    with patch.object(browser, "CAMPAIGN", manager):
        browser._add_new_project(owner)

    owner._reset_campaign_project_runtime_state.assert_called_once_with()
    owner._release_active_project_resources_before_switch.assert_called_once_with("OLD")
    manager.create_project.assert_called_once_with("NEW")
