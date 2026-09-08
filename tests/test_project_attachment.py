import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auto_annotation_tool.campaign_manager import CampaignManager
from auto_annotation_tool import project_attachment as attach


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def manager(tmp_path, monkeypatch):
    workspace = tmp_path / "Workspace"
    projects = workspace / "9_projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr(attach, "CONFIG", SimpleNamespace(DIR_9_PROJECTS=projects))
    owner = CampaignManager.__new__(CampaignManager)
    owner.state_file = workspace / "campaigns_registry.json"
    owner.state = {"active_project": "existing", "projects": {"existing": {"folder_name": "existing_123ABC", "current_step": 2}}}
    owner._registered_project_names = {"existing"}
    write_json(owner.state_file, owner.state)
    return owner


def copied_project(manager, *, name="portable", folder="portable_ABC123", old_root=None):
    root = manager.state_file.parent / "9_projects" / folder
    root.mkdir()
    old = old_root or "C:/Other/Workspace/9_projects/portable_ABC123"
    preview = old + "/3_cropped_characters/run_001"
    artifact = {"project": name, "iteration_state": {"2": {
        "t06_work_session": {"active": True, "state": "active", "work_area": "z3", "substep": 2, "preview_dir": preview},
        "t06_contracts": {"pz2_char_boxes": {"fulfilled": True}},
    }}}
    write_json(root / "_campaign_state/artifact_registry.json", artifact)
    events = [{"project": name, "iteration": 2, "step": 3, "status": "ok", "created_at": "2026-09-05T12:00:00",
               "details": {"path": "char_from_ready_plates", "preview_dir": preview}}]
    (root / "_campaign_state/project_history.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
    weights = root / "5_training_runs/20260905_120000/train/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint that must not be changed")
    run = {"best_weights": old + "/5_training_runs/20260905_120000/train/weights/best.pt", "status": "completed",
           "provenance": {"best_epoch": 135, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                          "dataset_id": "original_dataset", "split_sha256": "original_split", "data_yaml_sha256": "original_yaml"}}
    write_json(root / "5_training_runs/training_history.json", {"runs": {"20260905_120000": run}})
    write_json(weights.with_suffix(".pt.metadata.json"), run)
    source = {"annotation_run_dir": old + "/_campaign_state/char_effective_source",
              "xml_path": old + "/_campaign_state/char_effective_source/annotations.xml",
              "images_dir": old + "/_campaign_state/char_effective_source/images"}
    write_json(root / "3_cropped_characters/run_001/extract_manifest.json", {"status": "completed", "source": source})
    write_json(root / "3_cropped_characters/run_001/metadata.json", {"p1": {}, "p2": {}})
    write_json(root / "_campaign_state/wizard_view_cache.json", {"old": old + "/some.json|123|456"})
    data_yaml = root / "4_training_datasets/dataset/data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("train: images/train\nval: images/val\nnc: 1\n", encoding="utf-8")
    return root, run


@pytest.mark.parametrize("old", ["C:/Other/Workspace/9_projects/portable_ABC123", "/home/alex/Workspace/9_projects/portable_ABC123"])
def test_inspection_is_read_only_and_recovers_current_iteration(manager, old):
    root, _ = copied_project(manager, old_root=old)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    registry = manager.state_file.read_bytes()
    plan = manager.inspect_existing_project(root)
    assert plan.state_source == "recovered"
    assert plan.name == "portable"
    assert plan.project_data["current_iteration"] == 2
    assert plan.project_data["current_step"] == 3
    assert plan.project_data["step3_substep"] == 2
    assert plan.project_data["step3_stage1_done"] and plan.project_data["step3_stage2_done"]
    assert plan.preview_count == 2 and plan.training_runs == 1
    assert plan.project_data["best_plate_model"] == ""
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert manager.state_file.read_bytes() == registry


def test_attachment_preserves_data_and_active_project_with_backups(manager):
    root, run = copied_project(manager)
    old_registry = json.loads(manager.state_file.read_text())
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = manager.attach_existing_project(manager.inspect_existing_project(root))
    state = json.loads(manager.state_file.read_text())
    assert state["active_project"] == "existing"
    assert state["projects"]["existing"] == old_registry["projects"]["existing"]
    assert manager.get_all_projects() == ["existing", "portable"]
    assert state["projects"]["portable"]["folder_name"] == root.name
    history = json.loads((root / "5_training_runs/training_history.json").read_text())
    updated = history["runs"]["20260905_120000"]
    assert updated["best_weights"] == str(root / "5_training_runs/20260905_120000/train/weights/best.pt")
    assert updated["provenance"] == run["provenance"]
    for path, content in before.items():
        if path.suffix == ".pt" or path.name == "data.yaml":
            assert path.read_bytes() == content
        elif path.read_bytes() != content:
            assert (result["backup_dir"] / "files" / path.relative_to(root)).read_bytes() == content
    assert json.loads((result["backup_dir"] / "attachment.json").read_text())["status"] == "attached"
    assert json.loads((root / attach.DESCRIPTOR).read_text())["project"] == state["projects"]["portable"]


def test_descriptor_retains_exact_state_when_folder_is_renamed(manager):
    root, _ = copied_project(manager, folder="renamed_ABC123")
    data = manager._get_default_project_template("Original")
    data.update(folder_name="portable_ABC123", current_iteration=7, current_step=4, best_epoch=135,
                step4_finish_ready=True, best_plate_model="C:/Other/Workspace/9_projects/portable_ABC123/6_models/chosen.pt")
    write_json(root / attach.DESCRIPTOR, {"schema": attach.PROJECT_SCHEMA, "name": "Original",
                                       "root": "C:/Other/Workspace/9_projects/portable_ABC123", "project": data})
    plan = manager.inspect_existing_project(root)
    assert plan.state_source == "descriptor"
    assert plan.project_data["current_iteration"] == 7
    assert plan.project_data["current_step"] == 4
    assert plan.project_data["step4_finish_ready"] is True
    assert plan.project_data["best_plate_model"] == str(root / "6_models/chosen.pt")


def test_external_paths_move_only_to_existing_resources_and_missing_pool_is_grouped(manager):
    root, _ = copied_project(manager)
    local_model = manager.state_file.parent / "6_models/base/pose/model.pt"
    local_model.parent.mkdir(parents=True)
    local_model.write_bytes(b"base model")
    path = root / "_campaign_state/resources.json"
    write_json(path, {"model": "C:/Other/Workspace/6_models/base/pose/model.pt",
                      "images": [f"C:/Other/Workspace/1_raw_images/Missing/{i}.jpg" for i in range(1000)]})
    plan = manager.inspect_existing_project(root)
    assert plan.unresolved_paths == ["C:/Other/Workspace/1_raw_images"]
    moved = json.loads(plan.changes[path])
    assert moved["model"] == str(local_model)
    assert moved["images"][0] == "C:/Other/Workspace/1_raw_images/Missing/0.jpg"


def test_collision_and_repeated_attachment_never_overwrite_project(manager):
    root, _ = copied_project(manager)
    plan = manager.inspect_existing_project(root)
    with pytest.raises(attach.ProjectAttachmentError, match="zajęta"):
        manager.attach_existing_project(plan, name="EXISTING")
    manager.attach_existing_project(plan)
    with pytest.raises(attach.ProjectAttachmentError, match="już podłączony"):
        manager.attach_existing_project(manager.inspect_existing_project(root), name="different")
    assert len(manager.state["projects"]) == 2


def test_changed_project_requires_new_inspection(manager):
    root, _ = copied_project(manager)
    plan = manager.inspect_existing_project(root)
    (root / "3_cropped_characters/run_001/metadata.json").write_text("{}")
    with pytest.raises(attach.ProjectAttachmentError, match="zmieniły"):
        manager.attach_existing_project(plan)
    assert not (root / "_campaign_state/attachments").exists()


def test_failed_registry_write_rolls_back_every_metadata_file(manager):
    root, _ = copied_project(manager)
    plan = manager.inspect_existing_project(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    registry = manager.state_file.read_bytes()
    original_write = attach._atomic_bytes
    def fail_registry(path, content):
        if path == manager.state_file:
            raise OSError("simulated disk error")
        original_write(path, content)
    with patch.object(attach, "_atomic_bytes", side_effect=fail_registry), pytest.raises(OSError):
        manager.attach_existing_project(plan)
    assert manager.state_file.read_bytes() == registry
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (root / attach.DESCRIPTOR).exists()
    assert "portable" not in manager.state["projects"]


def test_registry_change_during_attachment_is_not_lost(manager):
    root, _ = copied_project(manager)
    plan = manager.inspect_existing_project(root)
    original_write = attach._atomic_bytes
    def concurrent_write(path, content):
        original_write(path, content)
        if path == root / attach.DESCRIPTOR:
            state = json.loads(manager.state_file.read_text())
            state["projects"]["concurrent"] = {"folder_name": "concurrent_123456"}
            write_json(manager.state_file, state)
    with patch.object(attach, "_atomic_bytes", side_effect=concurrent_write), pytest.raises(attach.ProjectAttachmentError, match="Rejestr projektów zmienił"):
        manager.attach_existing_project(plan)
    state = json.loads(manager.state_file.read_text())
    assert "concurrent" in state["projects"] and "portable" not in state["projects"]


def test_normal_save_preserves_new_external_entries_and_creates_portable_descriptor(manager):
    existing = manager.state_file.parent / "9_projects/existing_123ABC"
    existing.mkdir()
    state = copy.deepcopy(manager.state)
    state["projects"]["external"] = {"folder_name": "external_ABC123"}
    write_json(manager.state_file, state)
    manager.state["projects"]["existing"]["current_step"] = 3
    assert manager.save_state()
    assert "external" in json.loads(manager.state_file.read_text())["projects"]
    descriptor = json.loads((existing / attach.DESCRIPTOR).read_text())
    assert descriptor["schema"] == attach.PROJECT_SCHEMA
    assert descriptor["project"]["current_step"] == 3


def test_discovery_ignores_empty_registered_and_nested_folders(manager):
    root, _ = copied_project(manager)
    parent = root.parent
    (parent / "empty").mkdir()
    write_json(parent / "existing_123ABC/_campaign_state/artifact_registry.json", {})
    assert manager.list_attachable_projects() == [root]
    with pytest.raises(attach.ProjectAttachmentError):
        manager.inspect_existing_project(parent / "empty")
    with pytest.raises(attach.ProjectAttachmentError):
        manager.inspect_existing_project(root / "5_training_runs")


def test_corrupt_metadata_and_traversal_paths_are_rejected(manager):
    root, _ = copied_project(manager)
    path = root / "_campaign_state/resources.json"
    path.write_text("not json")
    with pytest.raises(attach.ProjectAttachmentError, match="odczytać"):
        manager.inspect_existing_project(root)
    write_json(path, {"path": "C:/Other/Workspace/9_projects/portable_ABC123/../../outside"})
    with pytest.raises(attach.ProjectAttachmentError, match="poza projekt"):
        manager.inspect_existing_project(root)


def approved_cache(root):
    old = "C:/Other/Workspace/1_raw_images/Missing/image.jpg"
    entry = {"image_name": "image.jpg", "source_image_path": old, "width": 1280, "height": 853,
             "plates": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]}
    approved = root / "_campaign_state/plate_approved_set.json"
    write_json(approved, {"entries": {old: entry}})
    cache = root / "_campaign_state/char_effective_source"
    write_json(cache / "source_state.json", {
        "approved_manifest_token": "C:/Other/Workspace/9_projects/portable_ABC123/_campaign_state/plate_approved_set.json|123|" + str(approved.stat().st_size),
    })
    (cache / "images").mkdir()
    image = cache / "images/image.jpg"
    image.write_bytes(b"original transferred image")
    (cache / "annotations.xml").write_text('<annotations><image name="image.jpg" width="1280" height="853"><polygon label="plate" points="0,0;10,0;10,10;0,10"/></image></annotations>')
    return image


@pytest.mark.parametrize("hardlinks", [True, False])
def test_transferred_approved_images_survive_source_cache_rebuild(manager, hardlinks):
    root, _ = copied_project(manager)
    image = approved_cache(root)
    original_image = image.read_bytes()
    plan = manager.inspect_existing_project(root)
    assert len(plan.source_copies) == 1
    if hardlinks:
        result = manager.attach_existing_project(plan)
    else:
        with patch.object(attach.os, "link", side_effect=OSError("no hardlinks")):
            result = manager.attach_existing_project(plan)
    assert result["retained_images"] == 1
    entry = next(iter(json.loads((root / "_campaign_state/plate_approved_set.json").read_text())["entries"].values()))
    stable = Path(entry["source_image_path"])
    assert stable.is_relative_to(root / "1_raw_images")
    image.unlink()  # The application may rebuild the disposable source cache.
    assert stable.read_bytes() == original_image


def test_images_are_not_recovered_from_incompatible_cached_annotations(manager):
    root, _ = copied_project(manager)
    image = approved_cache(root)
    xml = image.parent.parent / "annotations.xml"
    xml.write_text(xml.read_text().replace("10,10", "100,100"))
    assert not manager.inspect_existing_project(root).source_copies


def test_image_retention_is_rolled_back_with_registration_failure(manager):
    root, _ = copied_project(manager)
    image = approved_cache(root)
    plan = manager.inspect_existing_project(root)
    write = attach._atomic_bytes
    def fail_registry(path, content):
        if path == manager.state_file:
            raise OSError("registry unavailable")
        write(path, content)
    with patch.object(attach, "_atomic_bytes", side_effect=fail_registry), pytest.raises(OSError):
        manager.attach_existing_project(plan)
    assert image.read_bytes() == b"original transferred image"
    assert all(not path.exists() for path in plan.source_copies)


def test_unresolved_external_resources_remain_visible_after_another_move(manager):
    root, _ = copied_project(manager)
    write_json(root / "_campaign_state/resources.json", {"images": "D:/Oldest/Workspace/1_raw_images/missing_pool/image.jpg"})
    manager.attach_existing_project(manager.inspect_existing_project(root))
    plan = manager.inspect_existing_project(root)
    assert plan.state_source == "descriptor"
    assert "D:/Oldest/Workspace/1_raw_images" in plan.unresolved_paths
