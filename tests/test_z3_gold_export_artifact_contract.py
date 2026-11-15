from __future__ import annotations

import json
from pathlib import Path

from auto_annotation_tool.gui import z3_goldpack_ui as gold


def _write_artifact(root: Path, *, label_rows: int = 2, include_image: bool = True):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("path: .\ntrain: images\nval: images\n", encoding="utf-8")

    if include_image:
        (root / "images" / "p1.jpg").write_bytes(b"not-empty")
    (root / "labels" / "p1.txt").write_text(
        "\n".join("0 0.5 0.5 0.2 0.2" for _ in range(label_rows)),
        encoding="utf-8",
    )

    manifest = {
        "schema": gold.GOLD_EXPORT_ARTIFACT_SCHEMA,
        "dataset_type": "char_yolo_detect",
        "plate_count": 1,
        "character_count": 2,
        "items": [
            {
                "pid": "p1",
                "image_path": "images/p1.jpg",
                "label_path": "labels/p1.txt",
                "characters": [
                    {"character": "A"},
                    {"character": "B"},
                ],
            }
        ],
    }
    (root / "metadata_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _readiness(*, plates=1, chars=2, minimum=1):
    return {
        "ok": True,
        "selected_plate_count": plates,
        "selected_char_count": chars,
        "min_exportable_plate_count": minimum,
    }


def test_artifact_contract_accepts_matching_readiness_manifest_and_files(tmp_path):
    _write_artifact(tmp_path)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(),
    )

    assert result["ok"] is True
    assert result["plate_count"] == 1
    assert result["character_count"] == 2


def test_artifact_contract_rejects_readiness_plate_count_drift(tmp_path):
    _write_artifact(tmp_path)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(plates=2),
    )

    assert result["ok"] is False
    assert "liczba tablic" in result["message"].lower()


def test_artifact_contract_rejects_readiness_character_count_drift(tmp_path):
    _write_artifact(tmp_path)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(chars=3),
    )

    assert result["ok"] is False
    assert "liczba znaków" in result["message"].lower()


def test_artifact_contract_rejects_missing_exported_image(tmp_path):
    _write_artifact(tmp_path, include_image=False)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(),
    )

    assert result["ok"] is False
    assert "brak wyeksportowanego obrazu" in result["message"].lower()


def test_artifact_contract_rejects_label_manifest_row_mismatch(tmp_path):
    _write_artifact(tmp_path, label_rows=1)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(),
    )

    assert result["ok"] is False
    assert "niespójna etykieta" in result["message"].lower()


def test_artifact_contract_rejects_below_minimum_even_if_counts_match(tmp_path):
    _write_artifact(tmp_path)

    result = gold.validate_gold_export_artifact_contract(
        tmp_path,
        _readiness(plates=1, chars=2, minimum=10),
    )

    assert result["ok"] is False
    assert "poniżej minimum" in result["message"].lower()


def test_runtime_validates_artifact_before_marking_success():
    source = Path(gold.__file__).read_text(encoding="utf-8-sig")
    start = source.index("def run_yolo_gold_export(")
    end = source.index("\ndef run_pz3_existing_dataset_split(", start)
    body = source[start:end]

    validate_pos = body.index("validate_gold_export_artifact_contract(")
    success_pos = body.index('success_summary["gold_dataset_valid"] = True')

    assert validate_pos < success_pos
    assert "CAMPAIGN.set_step3_needs_rework()" in body
