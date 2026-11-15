from __future__ import annotations

import json
import tempfile
from pathlib import Path

from auto_annotation_tool.gui import z4_dataset_builder
from auto_annotation_tool.gui.z4_dataset_readiness import get_training_dataset_readiness
from auto_annotation_tool.training import AugmentationProfile
from auto_annotation_tool.training.character_class_distribution import (
    CHARACTER_BALANCE_ALPHABET,
    plan_character_train_augmentation,
)


def _names_yaml() -> str:
    return (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(
            f"  {idx}: '{symbol}'\n"
            for idx, symbol in enumerate(CHARACTER_BALANCE_ALPHABET)
        )
    )


def _write_sparse_split(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    (root / "data.yaml").write_text(_names_yaml(), encoding="utf-8")

    zero_id = CHARACTER_BALANCE_ALPHABET.index("0")
    one_id = CHARACTER_BALANCE_ALPHABET.index("1")

    for idx in range(8):
        stem = f"train_{idx}"
        (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"img")
        class_id = zero_id if idx % 2 == 0 else one_id
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    (root / "images" / "val" / "val_0.jpg").write_bytes(b"img")
    (root / "labels" / "val" / "val_0.txt").write_text(
        f"{zero_id} 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (root / "images" / "test" / "test_0.jpg").write_bytes(b"img")
    (root / "labels" / "test" / "test_0.txt").write_text(
        f"{one_id} 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )


def test_source_only_variant_is_ready_even_with_representation_deficits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "split"
        _write_sparse_split(root)

        plan = plan_character_train_augmentation(root, target_ratio=1.0)
        assert plan.feasible is False
        assert plan.planned_images == 0

        result = z4_dataset_builder._finalize_step4_mz_representation_variant(
            object(),
            dataset_path=root,
            plan=plan,
            profile=AugmentationProfile(enabled=False, extra_count=0),
            counts={"train": 8, "val": 1, "test": 1, "total": 10},
            generated=0,
            requested_extra=0,
            augmented=False,
            representation_required=False,
        )

        assert result["ok"] is True
        assert "ostrzeżenie jakościowe, nie blokada" in result["message"]

        manifest = json.loads(
            (root / "mz_training_variant_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["augmentation_mode"] == "source_only"
        assert manifest["representation_required"] is False
        assert manifest["representation_policy"] == "advisory"
        assert manifest["ready_for_training"] is True
        assert manifest["completion_status"] == "SOURCE_ONLY_READY_WITH_REPRESENTATION_WARNING"
        assert manifest["representation_status"] == "TARGET_NOT_REACHED"
        assert manifest["deficit_after"]

        readiness = get_training_dataset_readiness(root, target="char")
        assert readiness["ok"] is True


def test_split_workflow_checks_zero_synthetics_before_strict_representation_gate():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z4_dataset_builder.py"
    )
    source = path.read_text(encoding="utf-8-sig")
    split_segment = source[source.index("def _split_dataset_thread"):]

    assert (
        "augmentation_requested = bool(_step4_profile_requests_augmentation(augmentation_profile))"
        in split_segment
    )
    assert "if not augmentation_requested:" in split_segment
    assert "representation_required=False" in split_segment

    optional_pos = split_segment.index("if not augmentation_requested:")
    strict_pos = split_segment.index(
        "if not exact_feasible and exact_planned_images <= 0:"
    )
    assert optional_pos < strict_pos
