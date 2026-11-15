import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_annotation_tool.gui.dataset_display import build_dataset_display_ref
from auto_annotation_tool.gui import z4_dataset_sources as sources


def _make_dataset(root: Path, *, train=2, val=1, test=1):
    for split, count in (("train", train), ("val", val), ("test", test)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (image_dir / f"{split}_{index}.jpg").touch()
            (label_dir / f"{split}_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {0: A}\n",
        encoding="utf-8",
    )
    return root


def test_dataset_display_uses_last_timestamp_in_variant_name(tmp_path):
    root = _make_dataset(
        tmp_path / "YOLO_MegaDataset_Chars_20261004_141352_Split_20261007_204718_Aug_plus909_20261007_204734"
    )
    ref = build_dataset_display_ref(root, target_hint="char")
    assert ref.created_label == "2026-10-07 20:47"
    assert ref.id.startswith("DS-ZN-261007-2047-")


def test_pz2_hides_base_split_when_augmented_child_points_to_it(tmp_path):
    base = _make_dataset(tmp_path / "YOLO_MegaDataset_Chars_20261004_141352_Split_20261007_203519")
    aug = _make_dataset(
        tmp_path / "YOLO_MegaDataset_Chars_20261004_141352_Split_20261007_203519_Aug_plus909_20261007_203537",
        train=4,
    )
    (aug / "augmentation_scope.json").write_text(
        json.dumps({"source_dataset_dir": str(base.resolve()), "requested_extra": 909, "generated": 909}),
        encoding="utf-8",
    )
    host = SimpleNamespace(
        _get_selected_training_target=lambda: "char",
        _get_datasets_base_dir=lambda: tmp_path,
        _find_ready_dataset_candidates=lambda _root: [(base, "char", 10.0), (aug, "char", 20.0)],
        _get_dataset_split_image_counts=lambda path: sources._get_dataset_split_image_counts(SimpleNamespace(), path),
        _format_workspace_relative_path=lambda path: str(path),
    )
    with patch.object(sources.CAMPAIGN, "get_active_project_name", return_value=""):
        variants = sources._get_free_dataset_variant_choices(host)
    assert [Path(item["path"]).resolve() for item in variants] == [aug.resolve()]
    assert variants[0]["variant_role"] == "augmented"
    assert variants[0]["role_label"] == "AUG +909"
    assert "AUG +909" in variants[0]["label"]


def test_standalone_unaugmented_split_remains_selectable(tmp_path):
    base = _make_dataset(tmp_path / "YOLO_MegaDataset_Chars_20261004_141352_Split_20261007_173554")
    host = SimpleNamespace(
        _get_selected_training_target=lambda: "char",
        _get_datasets_base_dir=lambda: tmp_path,
        _find_ready_dataset_candidates=lambda _root: [(base, "char", 10.0)],
        _get_dataset_split_image_counts=lambda path: sources._get_dataset_split_image_counts(SimpleNamespace(), path),
        _format_workspace_relative_path=lambda path: str(path),
    )
    with patch.object(sources.CAMPAIGN, "get_active_project_name", return_value=""):
        variants = sources._get_free_dataset_variant_choices(host)
    assert len(variants) == 1
    assert variants[0]["variant_role"] == "base"
    assert variants[0]["role_label"] == "BAZA"
    assert "BAZA" in variants[0]["label"]
