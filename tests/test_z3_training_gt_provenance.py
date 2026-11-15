import json
from pathlib import Path

from auto_annotation_tool.gui import z3_dataset_provenance
from auto_annotation_tool.training.model_provenance import (
    build_dataset_training_provenance,
    build_model_training_provenance,
    build_training_dataset_snapshot,
)


def raw_plate(
    *,
    image_id,
    plate_id,
    gt_hash,
    gt,
    prediction,
    edit_distance,
    exact,
):
    return {
        "source_image_id": image_id,
        "source_annotation_id": plate_id,
        "source_gt_hash": gt_hash,
        "ground_truth_text": gt,
        "raw_detection": {
            "contract": "gt_blind.v1",
            "prediction_text": prediction,
            "characters": [],
        },
        "raw_validation": {
            "schema": "alpr.pz2.raw_validation.v1",
            "raw_contract": "gt_blind.v1",
            "expected_source": "ground_truth",
            "ground_truth_text": gt,
            "prediction_text": prediction,
            "status": "perfect" if exact else "needs_fix",
            "exact_text_match": bool(exact),
            "exact_count_match": len(gt) == len(prediction),
            "geometry_ok": True,
            "edit_distance": edit_distance,
        },
    }


def test_raw_benchmark_exact_plate_and_corpus_cer():
    records = [
        raw_plate(
            image_id="img-a",
            plate_id="plate-a",
            gt_hash="gt-a",
            gt="ABC123",
            prediction="ABC123",
            edit_distance=0,
            exact=True,
        ),
        raw_plate(
            image_id="img-b",
            plate_id="plate-b",
            gt_hash="gt-b",
            gt="XYZ999",
            prediction="XY2999",
            edit_distance=1,
            exact=False,
        ),
    ]

    result = z3_dataset_provenance.build_gt_blind_raw_benchmark(
        records
    )

    assert result["evaluable_plates"] == 2
    assert result["exact_plate_count"] == 1
    assert result["exact_plate_rate"] == 0.5
    assert result["gt_characters"] == 12
    assert result["edit_distance_total"] == 1
    assert result["cer"] == round(1 / 12, 8)
    assert result["gt_hash_coverage"] == "full"
    assert len(result["gt_contract_fingerprint_sha256"]) == 64


def test_raw_benchmark_gt_fingerprint_changes_with_gt_version():
    first = raw_plate(
        image_id="img-a",
        plate_id="plate-a",
        gt_hash="gt-v1",
        gt="ABC123",
        prediction="ABC123",
        edit_distance=0,
        exact=True,
    )
    second = dict(first)
    second["source_gt_hash"] = "gt-v2"

    a = z3_dataset_provenance.build_gt_blind_raw_benchmark([first])
    b = z3_dataset_provenance.build_gt_blind_raw_benchmark([second])

    assert (
        a["gt_contract_fingerprint_sha256"]
        != b["gt_contract_fingerprint_sha256"]
    )


def make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / split / "one.jpg").write_bytes(b"image")
        (dataset / "labels" / split / "one.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    (dataset / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: 'A'\n",
        encoding="utf-8",
    )

    benchmark = {
        "schema": "alpr.pz2.raw_benchmark.v1",
        "raw_contract": "gt_blind.v1",
        "metric_policy": "exact_plate_and_corpus_cer.v1",
        "evaluable_plates": 10,
        "exact_plate_count": 8,
        "exact_plate_rate": 0.8,
        "gt_characters": 60,
        "edit_distance_total": 3,
        "cer": 0.05,
        "gt_contract_fingerprint_sha256": "a" * 64,
    }
    (dataset / "metadata_manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "char_yolo_detect",
                "provenance_schema": (
                    "alpr.dataset.sample_provenance.v1"
                ),
                "provenance_counts": {
                    "raw_model_exact": 8,
                    "manual": 2,
                },
                "raw_benchmark": benchmark,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (dataset / "mz_training_variant_manifest.json").write_text(
        json.dumps(
            {
                "schema": "alpr.mz_training_variant.v1",
                "ready_for_training": True,
                "completion_status": "REPRESENTATION_OK",
            }
        ),
        encoding="utf-8",
    )
    return dataset


def test_training_snapshot_includes_gold_provenance_and_raw_benchmark(
    tmp_path,
):
    dataset = make_dataset(tmp_path)

    snapshot = build_training_dataset_snapshot(
        dataset,
        target="char",
    )

    manifest_names = {
        row["name"]
        for row in snapshot["manifests"]
    }
    assert "metadata_manifest.json" in manifest_names
    assert "mz_training_variant_manifest.json" in manifest_names
    assert snapshot["source_sample_provenance_schema"] == (
        "alpr.dataset.sample_provenance.v1"
    )
    assert snapshot["source_sample_provenance_counts"] == {
        "raw_model_exact": 8,
        "manual": 2,
    }
    assert snapshot["source_raw_benchmark"]["exact_plate_rate"] == 0.8
    assert snapshot["source_gt_contract_fingerprint_sha256"] == "a" * 64


def test_dataset_id_changes_when_gold_provenance_manifest_changes(tmp_path):
    dataset = make_dataset(tmp_path)

    before = build_dataset_training_provenance(
        dataset,
        target="char",
    )
    manifest_path = dataset / "metadata_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["provenance_counts"]["manual"] = 3
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    after = build_dataset_training_provenance(
        dataset,
        target="char",
    )

    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert before["dataset_id"] != after["dataset_id"]


def test_model_training_provenance_keeps_frozen_raw_benchmark(tmp_path):
    dataset = make_dataset(tmp_path)
    snapshot = build_training_dataset_snapshot(
        dataset,
        target="char",
    )
    run = {
        "id": "20260920_180000",
        "name": "MZ test",
        "dataset_path": str(dataset),
        "training_target": "char",
        "epochs": 1,
        "current_epoch": 1,
        "training_dataset_snapshot": snapshot,
    }

    result = build_model_training_provenance(
        run,
        target="char",
        include_dataset_fingerprint=False,
    )

    assert (
        result["dataset"]["source_gt_contract_fingerprint_sha256"]
        == "a" * 64
    )
    assert result["dataset"]["source_raw_benchmark"]["cer"] == 0.05

def test_raw_benchmark_fingerprint_tracks_full_gt_revision_set():
    first = raw_plate(
        image_id="img-a",
        plate_id="plate-a",
        gt_hash="same-gt-hash",
        gt="ABC123",
        prediction="ABC123",
        edit_distance=0,
        exact=True,
    )
    first["source_gt_revision_ids"] = ["rev-a", "rev-b"]
    first["raw_validation"]["gt_revision_ids"] = [
        "rev-a",
        "rev-b",
    ]

    second = json.loads(json.dumps(first))
    second["source_gt_revision_ids"] = ["rev-a", "rev-c"]
    second["raw_validation"]["gt_revision_ids"] = [
        "rev-a",
        "rev-c",
    ]

    a = z3_dataset_provenance.build_gt_blind_raw_benchmark(
        [first]
    )
    b = z3_dataset_provenance.build_gt_blind_raw_benchmark(
        [second]
    )

    assert a["gt_revision_coverage"] == "full"
    assert b["gt_revision_coverage"] == "full"
    assert (
        a["gt_contract_fingerprint_sha256"]
        != b["gt_contract_fingerprint_sha256"]
    )


def test_raw_benchmark_excludes_stale_validation_revision():
    data = raw_plate(
        image_id="img-a",
        plate_id="plate-a",
        gt_hash="same-gt-hash",
        gt="ABC123",
        prediction="ABC123",
        edit_distance=0,
        exact=True,
    )
    data["source_gt_revision_ids"] = ["rev-current"]
    data["raw_validation"]["gt_revision_ids"] = ["rev-old"]

    result = z3_dataset_provenance.build_gt_blind_raw_benchmark(
        [data]
    )

    assert result["evaluable_plates"] == 0
    assert result["invalid_validation_count"] == 1

def test_raw_benchmark_reports_box_count_accuracy():
    records = [
        raw_plate(
            image_id="img-a",
            plate_id="plate-a",
            gt_hash="gt-a",
            gt="ABC123",
            prediction="ABC123",
            edit_distance=0,
            exact=True,
        ),
        raw_plate(
            image_id="img-b",
            plate_id="plate-b",
            gt_hash="gt-b",
            gt="XYZ999",
            prediction="XYZ99",
            edit_distance=1,
            exact=False,
        ),
    ]
    records[0]["raw_validation"].update(
        expected_char_count=6,
        detected_char_count=6,
        box_count_error=0,
    )
    records[1]["raw_validation"].update(
        expected_char_count=6,
        detected_char_count=5,
        box_count_error=-1,
    )

    result = z3_dataset_provenance.build_gt_blind_raw_benchmark(
        records
    )

    assert result["box_count_exact_count"] == 1
    assert result["box_count_accuracy"] == 0.5
    assert result["box_count_abs_error_total"] == 1
    assert result["mean_abs_box_count_error"] == 0.5


def test_training_snapshot_keeps_dataset_revision_sets(tmp_path):
    dataset = make_dataset(tmp_path)
    manifest_path = dataset / "metadata_manifest.json"
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    payload["gt_revision_ids"] = ["rev-b", "rev-a"]
    payload["geometry_revision_ids"] = [
        "geom-b",
        "geom-a",
    ]
    payload["gt_contract_fingerprint_sha256"] = "b" * 64
    manifest_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    snapshot = build_training_dataset_snapshot(
        dataset,
        target="char",
    )

    assert snapshot["source_gt_revision_ids"] == [
        "rev-a",
        "rev-b",
    ]
    assert snapshot["source_geometry_revision_ids"] == [
        "geom-a",
        "geom-b",
    ]
    assert (
        snapshot["source_gt_contract_fingerprint_sha256"]
        == "b" * 64
    )
