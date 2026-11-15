import json
from pathlib import Path

import pytest

from auto_annotation_tool.training.dataset_creator import DatasetCreator, PlateAnnotation
from auto_annotation_tool.training.dataset_splitter import DatasetSplitter
from auto_annotation_tool.training.scene_split import assign_scene_splits
from auto_annotation_tool.training.source_inventory import file_sha256, build_source_inventory


def samples(root, total=24):
    root.mkdir()
    rows = []
    for i in range(total):
        path = root / f"image_{i:03}.jpg"
        # Image decoding is irrelevant to the split; each pair represents the same scene.
        path.write_bytes(f"scene-{i // 2}".encode())
        rows.append({"name": path.name, "image": path})
    return rows


def assert_partition(manifest, expected):
    assert len(manifest["items"]) == expected
    for key in ("group_id", "artifact_sha256", "source_image_sha256"):
        groups = {}
        for row in manifest["items"]:
            groups.setdefault(row[key], set()).add(row["split"])
        assert all(len(splits) == 1 for splits in groups.values())
    assert manifest["cross_split_group_leakage"] == 0


def test_duplicate_scenes_stay_together_and_input_order_does_not_matter(tmp_path):
    rows = samples(tmp_path / "source")
    ratios = {"train": .8, "val": .1, "test": .1}
    _, assignment = assign_scene_splits(rows, ratios)
    assert_partition(assignment, 24)
    assert all(assignment["counts"][s] for s in ratios)
    assert assign_scene_splits(list(reversed(rows)), ratios)[1] == assignment


def test_one_unique_scene_cannot_fill_train_and_val(tmp_path):
    with pytest.raises(ValueError, match="niezależnych scen"):
        assign_scene_splits(samples(tmp_path / "source", 2), {"train": .8, "val": .2})


def test_creator_keeps_duplicate_annotations_and_does_not_modify_source(tmp_path):
    root, output = tmp_path / "source", tmp_path / "output"
    rows = samples(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    creator = DatasetCreator()
    creator.annotations = [PlateAnnotation(row["name"], 100, 100, [(10, 10), (90, 10), (90, 40), (10, 40)]) for row in rows]
    ok, message, stats = creator.create_dataset(root, output, {"train": .8, "val": .1, "test": .1})
    assert ok, message
    assert stats["total"] == 24
    assert_partition(json.loads((output / "scene_split_assignment.json").read_text()), 24)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert not creator.create_dataset(root, output)[0]


def test_resplit_includes_augmentation_ancestry_and_inventory(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    generated = []
    for i in range(12):
        name = f"p{i}.jpg"
        (source / "images/train" / name).write_bytes(f"pixels-{i}".encode())
        (source / "labels/train" / f"p{i}.txt").write_text("0 .5 .5 .3 .1")
        if i % 2:
            generated.append({"image": "images/train/" + name, "source_image": f"images/train/p{i-1}.jpg"})
    (source / "augmentation_manifest.json").write_text(json.dumps({"generated_files": generated}))
    ok, message, stats = DatasetSplitter().split_dataset(source, output, {"train": .8, "val": .1, "test": .1})
    assert ok, message
    manifest = json.loads((output / "scene_split_assignment.json").read_text())
    assert_partition(manifest, 12)
    assert manifest["groups_total"] == 6
    inventory = build_source_inventory(output, role="plate")
    assert inventory["status"] == "complete"
    assert len(inventory["source_scene_hashes"]) == 6
    assert all(len(row["splits"]) == 1 for row in inventory["source_scene_hashes"])


def test_ambiguous_destination_and_missing_ancestry_fail_before_writes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    (source / "images/train/a.jpg").write_bytes(b"x")
    (source / "labels/train/a.txt").write_text("label")
    (source / "augmentation_manifest.json").write_text(json.dumps({"generated_files": [
        {"image": "images/train/a.jpg", "source_image": "images/train/missing.jpg"}]}))
    ok, _, _ = DatasetSplitter().split_dataset(source, output, {"train": .8, "val": .2})
    assert not ok and not output.exists()


def test_same_artifact_with_different_declared_sources_joins_groups(tmp_path):
    rows = samples(tmp_path / "source", 6)
    declared = {r["image"].resolve(): ("a" if i < 3 else "b") * 64 for i, r in enumerate(rows)}
    _, result = assign_scene_splits(rows, {"train": 1.0}, source_hashes=declared)
    assert result["groups_total"] == 1
    assert_partition(result, 6)


def test_char_crops_never_become_source_scene_inventory(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    for i in range(6):
        (source / f"images/train/p{i}.jpg").write_bytes(f"crop-{i}".encode())
        (source / f"labels/train/p{i}.txt").write_text("0 .5 .5 .1 .1")
    (source / "data.yaml").write_text("names: " + json.dumps(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    ok, message, _ = DatasetSplitter().split_dataset(source, output, {"train": .8, "val": .2})
    assert ok, message
    inventory = build_source_inventory(output, role="character")
    assert inventory["status"] == "unavailable"
    assert not inventory["source_scene_hashes"]
