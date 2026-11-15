import copy
import hashlib
import json
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image

from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from auto_annotation_tool.training.model_provenance import build_training_dataset_snapshot
from auto_annotation_tool.training.source_inventory import (
    FILENAME, SNAPSHOT_KEY, build_source_inventory, file_sha256, inventory_for_export, validate_archive_inventory,
)
from test_mobile_model_package_contract import export_fixture_model


def dataset(root, *, missing=False):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        path = root / "images" / split / "crop.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (60, 20), (i * 70, 90, 110)).save(path)
        rows.append({"image_path": path.relative_to(root).as_posix(), "split": split,
                     "provenance": {"source_image_id": "img-sha256-" + ("a" * 64 if i < 2 else "b" * 64)}})
    if missing:
        rows[-1]["provenance"] = {}
    (root / "metadata_manifest.json").write_text(json.dumps({"items": rows}), encoding="utf-8")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [A]\n", encoding="utf-8")
    return root


def test_mz_scene_deduplication_split_membership_and_determinism(tmp_path):
    root = dataset(tmp_path / "data")
    result = build_source_inventory(root, role="character", dataset_id="DS-test")
    assert result["source_scene_hashes"] == [{"sha256": "a" * 64, "splits": ["train", "val"]},
                                              {"sha256": "b" * 64, "splits": ["test"]}]
    assert result["sample_count"] == 3 and len(result["artifact_hashes"]) == 3
    assert result["status"] == "complete"
    assert result == build_source_inventory(root, role="character", dataset_id="DS-test")
    assert str(tmp_path) not in json.dumps(result)


def test_mz_unresolved_never_uses_crop_sha(tmp_path):
    result = build_source_inventory(dataset(tmp_path / "data", missing=True), role="character")
    assert result["unresolved_source_count"] == 1 and result["status"] == "partial"
    assert len(result["source_scene_hashes"]) == 1


@pytest.mark.parametrize("role", ["plate", "vehicle"])
def test_mt_mp_hash_actual_input(tmp_path, role):
    root = dataset(tmp_path / "data")
    (root / "metadata_manifest.json").unlink()
    result = build_source_inventory(root, role=role)
    assert {r["sha256"] for r in result["source_scene_hashes"]} == {file_sha256(p) for p in root.glob("images/*/*.png")}
    assert result["status"] == "complete"


def test_frozen_inventory_survives_dataset_removal_and_does_not_change_identity(tmp_path):
    root = dataset(tmp_path / "data")
    snapshot = build_training_dataset_snapshot(root, target="char")
    frozen = copy.deepcopy(snapshot[SNAPSHOT_KEY])
    renamed = root.with_name("unavailable")
    root.rename(renamed)
    payload = inventory_for_export({"training": {"dataset": snapshot}}, role="character")
    assert payload == frozen


def test_legacy_snapshot_reconstruction_requires_matching_content(tmp_path):
    root = dataset(tmp_path / "data")
    snapshot = build_training_dataset_snapshot(root, target="char")
    snapshot.pop(SNAPSHOT_KEY)
    metadata = {"training": {"dataset": snapshot}}
    result = inventory_for_export(metadata, role="character")
    assert result["status"] == "complete" and result["capture"] == "reconstructed_from_verified_snapshot"
    (root / "images/train/crop.png").write_bytes(b"changed")
    result = inventory_for_export(metadata, role="character")
    assert result["status"] == "unavailable" and result["capture"] == "dataset_changed_since_training"


def test_export_time_fingerprint_is_not_historical_evidence(tmp_path):
    root = dataset(tmp_path / "data")
    snapshot = build_training_dataset_snapshot(root, target="char")
    snapshot.pop(SNAPSHOT_KEY)
    snapshot.pop("snapshot_source")
    result = inventory_for_export({"training": {"dataset": snapshot, "provenance_capture": "reconstructed_at_export"}}, role="character")
    assert result["status"] == "unavailable"


def test_other_lineage_dataset_is_not_silently_certified(tmp_path):
    snapshot = build_training_dataset_snapshot(dataset(tmp_path / "data"), target="char")
    result = inventory_for_export({"training": {"dataset": snapshot, "lineage": [{"dataset_id": "DS-older"}]}}, role="character")
    assert result["status"] == "partial" and result["unresolved_dataset_count"] == 1


def test_assignment_manifest_source_id_fallback(tmp_path):
    root = dataset(tmp_path / "data")
    rows = json.loads((root / "metadata_manifest.json").read_text())["items"]
    (root / "metadata_manifest.json").unlink()
    for row in rows:
        row["group_id"] = "source_image_id:" + row.pop("provenance")["source_image_id"]
    (root / "split_assignment_manifest.json").write_text(json.dumps({"items": rows}))
    assert build_source_inventory(root, role="character")["status"] == "complete"


def test_export_writes_sidecar_checks_sha_and_keeps_legacy_compatible(tmp_path):
    result = build_source_inventory(dataset(tmp_path / "data"), role="character", dataset_id="DS-contract-fixture")
    with patch("auto_annotation_tool.training.source_inventory.inventory_for_export", return_value=result):
        package, _ = export_fixture_model(tmp_path / "export", "character")
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        data = archive.read(FILENAME)
        assert manifest["training_source_inventory"]["sha256"] == hashlib.sha256(data).hexdigest()
        assert "source_scene_hashes" not in archive.read("manifest.json").decode()
        files = {n: archive.read(n) for n in archive.namelist()}
    manifest.pop("training_source_inventory")
    legacy = tmp_path / "legacy.alprmodel"
    with zipfile.ZipFile(legacy, "w") as archive:
        for name, data in files.items():
            if name == FILENAME:
                continue
            archive.writestr(name, json.dumps(manifest) if name == "manifest.json" else data)
    mobile.MobileModelExporter().validate_package(legacy)
    with zipfile.ZipFile(package, "a") as archive:
        # A duplicate ZIP entry is rejected before it could shadow the inventory.
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(FILENAME, "{}")
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter().validate_package(package)


def test_full_bundle_root_stays_small_with_large_inventory(tmp_path):
    inventory = build_source_inventory(dataset(tmp_path / "data"), role="character", dataset_id="DS-contract-fixture")
    inventory["source_scene_hashes"] = [{"sha256": f"{i:064x}", "splits": ["train"]} for i in range(20000)]
    inventory["sample_count"] = 20000
    with patch("auto_annotation_tool.training.source_inventory.inventory_for_export", return_value=inventory):
        mz, _ = export_fixture_model(tmp_path / "mz", "character")
    mt, _ = export_fixture_model(tmp_path / "mt", "plate")
    package = mobile.MobileAlprPackageExporter().export(mobile.MobileAlprPackageRequest(
        destination=tmp_path / "full.alprmodel", plate_package=mt, character_package=mz))
    with zipfile.ZipFile(package) as archive:
        assert len(archive.read("manifest.json")) < mobile.MAX_ANDROID_ROOT_MANIFEST_BYTES
        assert b"source_scene_hashes" not in archive.read("manifest.json")


def test_inventory_hash_mismatch_rejected(tmp_path):
    package, _ = export_fixture_model(tmp_path / "export", "character")
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        manifest["training_source_inventory"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="SHA-256"):
            validate_archive_inventory(archive, manifest)


def test_identical_crop_bytes_are_still_two_artifacts(tmp_path):
    root = dataset(tmp_path / "data")
    original = root / "images/train/crop.png"
    duplicate = original.with_name("second_crop.png")
    duplicate.write_bytes(original.read_bytes())
    payload = json.loads((root / "metadata_manifest.json").read_text())
    row = copy.deepcopy(payload["items"][0])
    row["image_path"] = duplicate.relative_to(root).as_posix()
    payload["items"].append(row)
    (root / "metadata_manifest.json").write_text(json.dumps(payload))
    result = build_source_inventory(root, role="character")
    assert len(result["source_scene_hashes"]) == 2
    assert result["sample_count"] == len(result["artifact_hashes"]) == 4


@pytest.mark.parametrize("role", ["character", "plate"])
def test_augmentation_resolves_original_scene(tmp_path, role):
    root = dataset(tmp_path / "data")
    original = root / "images/train/crop.png"
    generated = original.with_name("crop__aug.png")
    Image.new("RGB", (60, 20), "red").save(generated)
    (root / "augmentation_manifest.json").write_text(json.dumps({"generated_files": [{
        "image": generated.relative_to(root).as_posix(), "source_image": original.relative_to(root).as_posix()}]}))
    result = build_source_inventory(root, role=role)
    assert result["status"] == "complete"
    assert len(result["source_scene_hashes"]) == 2
    assert len(result["artifact_hashes"]) == 4
    assert file_sha256(generated) not in {r["sha256"] for r in result["source_scene_hashes"]}


def test_embedded_frozen_arrays_are_removed_from_all_manifest_copies(tmp_path):
    snapshot = build_training_dataset_snapshot(dataset(tmp_path / "data"), target="char")
    original_class = mobile.MobileExportRequest
    def with_snapshot(**kwargs):
        kwargs["metadata"]["training"].update(dataset=snapshot, dataset_id=snapshot["dataset_id"])
        kwargs["metadata"]["candidate"] = {"nested": {SNAPSHOT_KEY: snapshot[SNAPSHOT_KEY]}}
        return original_class(**kwargs)
    with patch.object(mobile, "MobileExportRequest", side_effect=with_snapshot):
        path, _ = export_fixture_model(tmp_path / "export", "character")
    with zipfile.ZipFile(path) as archive:
        assert SNAPSHOT_KEY.encode() not in archive.read("manifest.json")
        assert b"source_scene_hashes" not in archive.read("manifest.json")
        assert json.loads(archive.read(FILENAME))["status"] == "complete"
