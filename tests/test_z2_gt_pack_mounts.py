import json
from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui import z2_gt_pack_runtime as runtime


class FakeSession:
    def __init__(self):
        self.data = {}
        self.saved = 0

    def get(self, tab, key, default=""):
        return self.data.get(tab, {}).get(key, default)

    def set(self, tab, key, value):
        self.data.setdefault(tab, {})[key] = value

    def save_session(self):
        self.saved += 1


def host():
    return SimpleNamespace()


def test_project_mount_config_roundtrip_uses_project_relative_refs(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.alprgt"
    inside = project / "gt" / "inside.alprgt"

    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_name",
        lambda: "demo",
    )
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_root_dir",
        lambda: project,
    )

    first = host()
    runtime.set_gt_pack_mounts(
        first,
        source_paths=[inside, external],
        working_path=project / "gt" / "work.alprgt",
        persist=True,
    )

    config = (
        project
        / "_campaign_state"
        / "ground_truth"
        / "mounts.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    assert payload["source_paths"][0].startswith("project://")
    assert payload["working_path"].startswith("project://")

    second = host()
    sources = runtime.get_configured_source_pack_paths(second)
    working = runtime.get_working_gt_pack_path(
        second,
        create_parent=False,
    )

    assert inside in sources
    assert external in sources
    assert working == project / "gt" / "work.alprgt"


def test_free_mode_mount_config_uses_session(
    tmp_path,
    monkeypatch,
):
    session = FakeSession()
    monkeypatch.setattr(runtime, "SESSION", session)
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_name",
        lambda: "",
    )
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_root_dir",
        lambda: None,
    )

    source = tmp_path / "source.alprgt"
    working = tmp_path / "work.alprgt"

    first = host()
    runtime.set_gt_pack_mounts(
        first,
        source_paths=[source],
        working_path=working,
        persist=True,
    )
    assert session.saved == 1

    second = host()
    assert runtime.get_configured_source_pack_paths(second) == [source]
    assert runtime.get_working_gt_pack_path(
        second,
        create_parent=False,
    ) == working


def test_status_reports_pending_and_last_restore_conflicts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_name",
        lambda: "",
    )
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_root_dir",
        lambda: None,
    )

    h = host()
    h.run_dir = str(tmp_path / "run")
    h._z2_gt_mount_config_loaded = True
    h._z2_gt_pack_source_paths = []
    h._z2_gt_working_pack_path = str(tmp_path / "work.alprgt")
    h._z2_gt_last_restore_report = {
        "conflicts": [{"reason": "test"}],
    }
    h._get_current_annotation_xml_path = (
        lambda: Path(h.run_dir) / "annotations.xml"
    )

    outbox = Path(h.run_dir) / "gt_sync_pending.json"
    outbox.parent.mkdir(parents=True)
    outbox.write_text(
        json.dumps(
            {
                "schema": runtime.OUTBOX_SCHEMA,
                "items": {
                    "plate-a|set": {"plate_id": "plate-a"}
                },
            }
        ),
        encoding="utf-8",
    )

    status = runtime.get_gt_pack_status(h)
    assert status["enabled"] is True
    assert status["pending_count"] == 1
    assert status["conflict_count"] == 1
    assert status["tone"] == "conflict"


def test_status_tile_hit():
    h = host()
    h._z2_gt_pack_status_bbox = (10.0, 10.0, 50.0, 30.0)
    assert runtime.is_gt_pack_status_hit(h, 20.0, 20.0)
    assert not runtime.is_gt_pack_status_hit(h, 70.0, 20.0)


def test_explicit_runtime_mounts_are_not_clobbered_by_persisted_config(
    tmp_path,
    monkeypatch,
):
    session = FakeSession()
    session.data = {
        "annotation": {
            "gt_pack_mounts": {
                "schema": runtime.MOUNTS_SCHEMA,
                "source_paths": [str(tmp_path / "persisted.alprgt")],
                "working_path": str(tmp_path / "persisted_work.alprgt"),
            }
        }
    }
    monkeypatch.setattr(runtime, "SESSION", session)
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_name",
        lambda: "",
    )
    monkeypatch.setattr(
        runtime.CAMPAIGN,
        "get_active_project_root_dir",
        lambda: None,
    )

    explicit_source = tmp_path / "explicit.alprgt"
    explicit_working = tmp_path / "explicit_work.alprgt"
    h = SimpleNamespace(
        _z2_gt_pack_source_paths=[str(explicit_source)],
        _z2_gt_working_pack_path=str(explicit_working),
    )

    assert runtime.get_configured_source_pack_paths(h) == [
        explicit_source
    ]
    assert runtime.get_working_gt_pack_path(
        h,
        create_parent=False,
    ) == explicit_working
