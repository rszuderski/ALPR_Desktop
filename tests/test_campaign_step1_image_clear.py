from unittest.mock import Mock

from auto_annotation_tool.campaign_manager import CampaignManager


def _owner(tmp_path, *, package_iterations):
    manager = CampaignManager.__new__(CampaignManager)
    manager.state = {
        "active_project": "demo",
        "projects": {
            "demo": {
                "current_iteration": 1,
                "master_pool_dir": "old/source",
            }
        },
    }
    manager._resolve_project_name = lambda project_name=None: "demo"

    manager.clear_master_pool_dir = Mock(return_value=True)
    manager.clear_latest_ingest_plan = Mock(return_value=True)

    manifest = tmp_path / "iter_001_manifest.json"
    manifest.write_text('{"source_dir":"old/source"}', encoding="utf-8")
    manager.get_ingest_manifest_path = lambda *args, **kwargs: manifest

    registry = {
        "project": "demo",
        "packages": {
            "pkg_old": {
                "package_id": "pkg_old",
                "iterations": list(package_iterations),
                "image_source": {
                    "master_pool_dir": "old/source",
                    "image_set_count": 1000,
                },
            }
        },
        "iteration_index": {"1": "pkg_old"},
        "iteration_state": {},
    }
    manager.load_artifact_registry = Mock(return_value=registry)
    saved = {}

    def save(payload, project_name=None):
        saved.clear()
        saved.update(payload)
        return True

    manager.save_artifact_registry = Mock(side_effect=save)

    manager._ingest_manifest_cache = {"stale": 1}
    manager._latest_ingest_plan_summary_cache = {"stale": 1}
    manager._latest_ingest_plan_summary_runtime_cache = {"stale": 1}
    manager._iteration_image_count_cache = {"stale": 1}
    manager._iteration_image_source_dir_cache = {"stale": 1}

    return manager, manifest, saved


def test_clear_step1_image_source_removes_manifest_and_current_package(tmp_path):
    manager, manifest, saved = _owner(tmp_path, package_iterations=[1])

    result = CampaignManager.clear_step1_image_source_state(
        manager,
        iteration_num=1,
        project_name="demo",
    )

    assert result["ok"]
    assert result["manifest_removed"]
    assert result["artifact_binding_removed"]
    assert not manifest.exists()
    assert "1" not in saved["iteration_index"]
    assert "pkg_old" not in saved["packages"]
    manager.clear_master_pool_dir.assert_called_once_with("demo")
    manager.clear_latest_ingest_plan.assert_called_once_with("demo")

    for cache_name in (
        "_ingest_manifest_cache",
        "_latest_ingest_plan_summary_cache",
        "_latest_ingest_plan_summary_runtime_cache",
        "_iteration_image_count_cache",
        "_iteration_image_source_dir_cache",
    ):
        assert getattr(manager, cache_name) == {}


def test_clear_step1_image_source_keeps_package_used_by_other_iteration(tmp_path):
    manager, manifest, saved = _owner(tmp_path, package_iterations=[1, 2])

    result = CampaignManager.clear_step1_image_source_state(
        manager,
        iteration_num=1,
        project_name="demo",
    )

    assert result["ok"]
    assert "1" not in saved["iteration_index"]
    assert "pkg_old" in saved["packages"]
    assert saved["packages"]["pkg_old"]["iterations"] == [2]
