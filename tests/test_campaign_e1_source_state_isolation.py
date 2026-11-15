import inspect
from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui import campaign_step1_assets as assets
from auto_annotation_tool.gui.tab_campaign import CampaignTab
from auto_annotation_tool.gui import campaign_step1_ingest as ingest


def test_collect_image_names_ignores_stale_plan_from_other_source(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    for name in ("AA11111_001.jpg", "BB22222_001.jpg"):
        (new_root / name).write_bytes(b"x")

    host = SimpleNamespace(
        current_ingest_plan={
            "master_pool_dir": str(old_root),
            "selected_total": 100_000,
            "selected": [
                {"name": f"STALE_{index:06d}.jpg"}
                for index in range(2500)
            ],
        }
    )
    progress = []
    names = assets._collect_project_start_image_names(
        host,
        images_dir=new_root,
        manifest={},
        progress_callback=lambda count, detail: progress.append((count, detail)),
    )

    assert sorted(names) == ["AA11111_001.jpg", "BB22222_001.jpg"]
    assert all(count <= 2 for count, _detail in progress)


def test_collect_image_names_uses_matching_plan_without_disk_scan(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    host = SimpleNamespace(
        current_ingest_plan={
            "master_pool_dir": str(root),
            "selected_total": 2,
            "selected": [
                {"name": "AA11111_001.jpg"},
                {"name": "BB22222_001.jpg"},
            ],
        }
    )

    names = assets._collect_project_start_image_names(host, images_dir=root, manifest={})
    assert names == ["AA11111_001.jpg", "BB22222_001.jpg"]


def test_new_O_state_is_cleared_before_new_master_pool_is_saved():
    source = inspect.getsource(CampaignTab._choose_master_pool_dir)
    clear_pos = source.index("clear_step1_image_source_state")
    set_pos = source.index("set_master_pool_dir")
    generate_pos = source.index("_generate_ingest_plan")
    sync_pos = source.index("_sync_iteration_artifact_registry_from_project_start")

    assert clear_pos < set_pos
    assert generate_pos < sync_pos


def test_artifact_sync_is_deferred_until_finished_plan_exists():
    source = inspect.getsource(ingest._finish_generated_ingest_plan)
    assign_pos = source.index("self.current_ingest_plan = plan")
    save_pos = source.index("save_latest_ingest_plan")
    sync_pos = source.index("_sync_iteration_artifact_registry_from_project_start")

    assert assign_pos < save_pos < sync_pos
