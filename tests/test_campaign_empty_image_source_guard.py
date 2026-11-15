from pathlib import Path

from auto_annotation_tool.campaign_manager import CampaignManager


def _manager_for_empty_project(tmp_path: Path) -> CampaignManager:
    manager = CampaignManager.__new__(CampaignManager)
    manager.state = {
        "active_project": "demo",
        "projects": {
            "demo": {
                "current_iteration": 1,
                "master_pool_dir": "",
            }
        },
    }
    manager._iteration_image_source_dir_cache = {}
    manager._resolve_project_name = lambda project_name=None: project_name or "demo"
    raw_dir = tmp_path / "project" / "1_raw_images" / "Iteracja_001"
    manager.get_iteration_raw_dir = lambda iteration_num=None, project_name=None: raw_dir
    manager.load_latest_ingest_plan_summary = lambda project_name=None: {}
    manager.load_ingest_manifest = lambda iteration_num=None, project_name=None: {}
    manager.count_images_in_dir = lambda path_like, recursive=False: 0
    return manager


def test_empty_source_never_resolves_to_process_working_directory(tmp_path, monkeypatch):
    manager = CampaignManager.__new__(CampaignManager)

    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / "stray.jpg").write_bytes(b"x")
    monkeypatch.chdir(cwd)

    assert manager._resolve_valid_image_source_dir("", reject_broad=False) == ""
    assert manager._resolve_valid_image_source_dir("   ", reject_broad=False) == ""


def test_empty_project_image_source_fallback_is_project_raw_dir_not_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / "stray.jpg").write_bytes(b"x")
    monkeypatch.chdir(cwd)

    manager = _manager_for_empty_project(tmp_path)
    expected = manager.get_iteration_raw_dir(1, "demo")

    resolved = manager.get_iteration_image_source_dir(1, "demo")

    assert resolved == expected
    assert resolved != cwd
    assert manager._iteration_image_source_dir_cache[("demo", 1)] == str(expected)
