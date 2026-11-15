from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from auto_annotation_tool import campaign_image_identity as identity
from auto_annotation_tool.campaign_manager import CampaignManager


def source(tmp_path, name, content=b"same image"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def manifest(*paths, iteration=1, **kwargs):
    return {"iteration": iteration, "selected_images": [
        {"name": path.name, "source_path": str(path)} for path in paths], **kwargs}


def save(tmp_path, data):
    return identity.save_manifest_with_identity(data, tmp_path / "_campaign_state/ingest" / f"iter_{data['iteration']:03d}_manifest.json")


def test_same_bytes_different_names_are_deduplicated(tmp_path):
    result = save(tmp_path, manifest(source(tmp_path, "a.jpg"), source(tmp_path, "copy.jpg")))
    assert result["selected_count"] == 1
    assert result["identity_summary"]["duplicate_sha256"] == 1
    assert result["identity_summary"]["duplicates"][0]["existing"]["name"] == "a.jpg"
    assert len(result["selected_images"][0]["content_sha256"]) == 64
    assert result["image_set_token"].startswith("iset_00001_")


@pytest.mark.parametrize("mode", ["manual", "pool_reuse", "stage_reuse"])
def test_new_admission_cannot_bypass_project_sha_guard(tmp_path, mode):
    save(tmp_path, manifest(source(tmp_path, "a.jpg")))
    with pytest.raises(identity.IdentityRejected) as exc:
        save(tmp_path, manifest(source(tmp_path, "renamed.jpg"), iteration=2, selection_mode=mode))
    assert exc.value.summary["duplicate_sha256"] == 1
    assert not (tmp_path / "_campaign_state/ingest/iter_002_manifest.json").exists()


def test_same_name_different_bytes_is_name_collision_not_content_duplicate(tmp_path):
    a, b = source(tmp_path, "one/a.jpg", b"one"), source(tmp_path, "two/a.jpg", b"two")
    result = save(tmp_path, manifest(a, b))
    report = result["identity_summary"]
    assert report["duplicate_sha256"] == 0
    assert report["name_collisions"] == 1
    conflict = report["collisions"][0]
    assert conflict["content_sha256"] != conflict["existing_sha256"]


def test_cache_avoids_reread_and_invalidates_modified_file(tmp_path, monkeypatch):
    path = source(tmp_path, "a.jpg")
    index = identity.ImageHashIndex(tmp_path / "index.json")
    hash_file = Mock(wraps=identity.file_sha256)
    monkeypatch.setattr(identity, "file_sha256", hash_file)
    first = index.fingerprint(path)
    assert index.fingerprint(path) == first
    assert hash_file.call_count == 1
    path.write_bytes(b"changed")
    assert index.fingerprint(path) != first
    assert hash_file.call_count == 2


def test_legacy_manifest_is_readable_and_bootstraps_identity_without_rewriting(tmp_path):
    path = source(tmp_path, "a.jpg")
    old = tmp_path / "_campaign_state/ingest/iter_001_manifest.json"
    old.parent.mkdir(parents=True)
    original = json.dumps(manifest(path)).encode()
    old.write_bytes(original)
    with pytest.raises(identity.IdentityRejected):
        save(tmp_path, manifest(source(tmp_path, "copy.jpg"), iteration=2))
    assert old.read_bytes() == original
    index = identity.ImageHashIndex(tmp_path / "_campaign_state/image_hash_index.json")
    assert len(index.data["hashes"]) == 1


def test_repeated_save_and_explicit_iteration_reference_keep_existing_resource(tmp_path, monkeypatch):
    path = source(tmp_path, "a.jpg")
    original = save(tmp_path, manifest(path))
    monkeypatch.setattr(identity, "file_sha256", Mock(side_effect=AssertionError("cached file must not be re-read")))
    repeated = save(tmp_path, deepcopy(original))
    assert repeated["selected_images"] == original["selected_images"]
    reference = save(tmp_path, manifest(path, iteration=2, identity_mode="reference", reused_from_iteration=1))
    assert reference["selected_count"] == 1 and reference["identity_summary"]["reused"] == 1
    assert reference["proposal_summary"]["new_to_project_count"] == 0
    index = identity.ImageHashIndex(tmp_path / "_campaign_state/image_hash_index.json")
    assert len(index.data["hashes"]) == 1
    assert len(next(iter(index.data["hashes"].values()))["occurrences"]) == 2


def test_reuse_does_not_trust_a_hash_after_source_bytes_change(tmp_path):
    path = source(tmp_path, "a.jpg")
    saved = save(tmp_path, manifest(path))
    path.write_bytes(b"new bytes")
    saved.update(iteration=2, identity_mode="reference", reused_from_iteration=1)
    with pytest.raises(identity.IdentityRejected) as exc:
        save(tmp_path, saved)
    assert exc.value.summary["name_collisions"] == 1
    assert exc.value.summary["duplicate_sha256"] == 0


def test_failed_manifest_write_does_not_register_admission(tmp_path, monkeypatch):
    path = source(tmp_path, "a.jpg")
    monkeypatch.setattr(identity, "_atomic_write", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        save(tmp_path, manifest(path))
    assert not (tmp_path / "_campaign_state/image_hash_index.json").exists()


def test_recovery_from_index_write_failure_rebuilds_from_committed_manifest(tmp_path, monkeypatch):
    path = source(tmp_path, "a.jpg")
    writer = identity._atomic_write
    def fail_index(target, value):
        if target.name == "image_hash_index.json":
            raise OSError("index write interrupted")
        writer(target, value)
    monkeypatch.setattr(identity, "_atomic_write", fail_index)
    with pytest.raises(OSError):
        save(tmp_path, manifest(path))
    monkeypatch.setattr(identity, "_atomic_write", writer)
    assert save(tmp_path, manifest(path))["selected_count"] == 1
    assert len(identity.ImageHashIndex(tmp_path / "_campaign_state/image_hash_index.json").data["hashes"]) == 1


def test_parallel_admissions_cannot_register_the_same_image_twice(tmp_path):
    a, b = source(tmp_path, "a.jpg"), source(tmp_path, "copy.jpg")
    def admit(number, path):
        try:
            save(tmp_path, manifest(path, iteration=number))
            return True
        except identity.IdentityRejected:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(admit, 1, a), pool.submit(admit, 2, b)]
        assert sorted(f.result(timeout=10) for f in futures) == [False, True]


def test_record_iteration_ingest_uses_the_shared_guard(tmp_path):
    manager = CampaignManager.__new__(CampaignManager)
    manager.state = {"projects": {"demo": {"current_iteration": 1}}}
    manager._resolve_project_name = lambda name=None: "demo"
    manager.get_iteration_raw_dir = lambda *args: tmp_path / "raw"
    manager.get_master_pool_dir = lambda *args: tmp_path
    manager.get_ingest_manifest_path = lambda number=None, *args: tmp_path / "_campaign_state/ingest" / f"iter_{number or 1:03d}_manifest.json"
    manager.list_plate_approved_entries = lambda *args: []
    a, b = source(tmp_path, "a.jpg"), source(tmp_path, "copy.jpg")
    path = manager.record_iteration_ingest(tmp_path, [a, b])
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["selected_count"] == 1
    with pytest.raises(identity.IdentityRejected):
        manager.record_iteration_ingest(tmp_path, [b], iteration_num=2)
