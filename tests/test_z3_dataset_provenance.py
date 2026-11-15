import copy
from pathlib import Path

from auto_annotation_tool.gui import z3_dataset_provenance as provenance


def chars(text, **extra):
    return [
        {
            "character": ch,
            "bbox": [idx * 10, 0, idx * 10 + 8, 20],
            **extra,
        }
        for idx, ch in enumerate(text)
    ]


def raw_exact_data():
    records = chars("ABC123")
    return {
        "source_image_id": "img-sha256-a",
        "source_annotation_id": "plate-ann-a",
        "source_geometry_hash": "geom",
        "source_gt_hash": "gt",
        "ground_truth_source": "manual_z2",
        "characters": records,
        "raw_detection": {
            "contract": "gt_blind.v1",
            "prediction_text": "ABC123",
            "characters": copy.deepcopy(records),
        },
        "raw_validation": {
            "status": "perfect",
            "exact_text_match": True,
            "exact_count_match": True,
            "geometry_ok": True,
        },
    }


def category(data, source_bucket=""):
    return provenance.classify_plate_dataset_provenance(
        None,
        data,
        meta_path=Path("metadata.json"),
        source_bucket=source_bucket,
    )["category"]


def test_raw_model_exact_requires_gt_blind_exact_and_unchanged_final():
    data = raw_exact_data()
    assert category(data) == provenance.RAW_MODEL_EXACT

    data["characters"][0]["bbox"][0] += 1
    assert category(data) == provenance.LEGACY_UNTRACKED


def test_manual_has_priority_over_raw_exact():
    data = raw_exact_data()
    data["characters"][0]["sign_source"] = "manual_sign"
    data["characters"][0]["source_kind"] = "local_manual"

    assert category(data) == provenance.MANUAL


def test_gt_assist_confirmed_has_highest_review_priority():
    data = raw_exact_data()
    data["correction_source"] = "gt_assist_confirmed"
    data["source_info"] = {
        "bucket": "local_manual",
        "origin": "preview_editor",
    }

    assert category(data) == provenance.GT_ASSIST_CONFIRMED


def test_cvat_import_is_explicit():
    data = raw_exact_data()
    data["source_info"] = {
        "bucket": "cvat_manual",
        "origin": "cvat_import",
    }

    assert category(data) == provenance.CVAT_IMPORT


def test_mobile_human_review_is_preserved():
    data = {
        "characters": chars("ABC123"),
        "source_expected_text_source": "mobile_crop_human_review",
        "mobile_acquisition": {
            "session_id": "capture-1",
        },
    }

    assert category(data) == provenance.MOBILE_HUMAN_REVIEW


def test_legacy_perfect_is_not_called_raw_model_exact():
    data = {
        "status": "perfect",
        "characters": chars("ABC123"),
    }

    assert category(data) == provenance.LEGACY_UNTRACKED


def test_provenance_summary_counts_categories():
    items = [
        {"provenance": {"category": provenance.RAW_MODEL_EXACT}},
        {"provenance": {"category": provenance.RAW_MODEL_EXACT}},
        {"provenance": {"category": provenance.MANUAL}},
    ]

    assert provenance.summarize_dataset_provenance(items) == {
        provenance.RAW_MODEL_EXACT: 2,
        provenance.MANUAL: 1,
    }

def test_dataset_provenance_preserves_revision_sets():
    data = raw_exact_data()
    data["source_geometry_revision_ids"] = ["geom-b", "geom-a"]
    data["source_gt_revision_ids"] = ["rev-b", "rev-a"]

    result = provenance.classify_plate_dataset_provenance(
        None,
        data,
        meta_path=Path("metadata.json"),
    )

    assert result["source_geometry_revision_ids"] == [
        "geom-a",
        "geom-b",
    ]
    assert result["source_gt_revision_ids"] == [
        "rev-a",
        "rev-b",
    ]


def test_raw_model_exact_rejects_validation_from_old_gt_revision():
    data = raw_exact_data()
    data["source_gt_revision_ids"] = ["rev-new"]
    data["raw_validation"]["gt_revision_ids"] = ["rev-old"]

    assert category(data) == provenance.LEGACY_UNTRACKED

def test_collect_dataset_revision_ids_deduplicates_all_heads():
    items = [
        {
            "provenance": {
                "source_gt_revision_ids": [
                    "rev-b",
                    "rev-a",
                ],
                "source_geometry_revision_ids": [
                    "geom-b",
                ],
            }
        },
        {
            "provenance": {
                "source_gt_revision_id": "rev-a",
                "source_geometry_revision_id": "geom-a",
            }
        },
    ]

    result = provenance.collect_dataset_revision_ids(items)

    assert result["gt_revision_ids"] == [
        "rev-a",
        "rev-b",
    ]
    assert result["geometry_revision_ids"] == [
        "geom-a",
        "geom-b",
    ]


def test_cross_project_az_reuse_provenance_is_self_contained():
    data = raw_exact_data()
    data["gold_state"] = {
        "approved": True,
        "excluded": False,
        "candidate": True,
    }
    data["review_state"] = {
        "schema": "alpr.pz2.review.v1",
        "status": "approved",
        "source": "manual_z3",
        "human_edited": False,
        "approved_at": "2026-09-28T21:00:00+02:00",
    }
    data["fusion_strategy"] = "az_reuse"
    data["fusion_details"] = {
        "source": "az_project_import",
        "az_revision_id": "AZREV-SOURCE-1",
        "effective_status": "imported_pending_review",
    }
    data["az_reuse"] = {
        "schema": "alpr.az_reuse.v1",
        "az_revision_id": "AZREV-SOURCE-1",
        "payload_sha256": "payload-sha",
        "source_kind": "pz2_detect",
        "trust_state": "auto",
        "origin_project_id": "PRJ-SOURCE",
        "origin_iteration": 1,
        "created_at": "2026-09-28T18:00:00+02:00",
        "project_id": "PRJ-SOURCE",
        "effective_status": "imported_pending_review",
        "requires_review": True,
    }

    result = provenance.classify_plate_dataset_provenance(
        None,
        data,
        meta_path=Path("metadata.json"),
        source_bucket="auto_preview",
    )
    reuse = result["az_reuse"]

    assert reuse["schema"] == provenance.AZ_REUSE_PROVENANCE_SCHEMA
    assert reuse["reused"] is True
    assert reuse["cross_project_import"] is True
    assert reuse["az_revision_id"] == "AZREV-SOURCE-1"
    assert reuse["payload_sha256"] == "payload-sha"
    assert reuse["origin_project_id"] == "PRJ-SOURCE"
    assert reuse["revision_project_id"] == "PRJ-SOURCE"
    assert reuse["source_effective_status_at_materialization"] == "imported_pending_review"
    assert reuse["required_local_review_at_materialization"] is True
    assert reuse["fusion_source"] == "az_project_import"
    assert reuse["local_review"]["status"] == "approved"
    assert reuse["local_review"]["approved"] is True
    assert reuse["local_review"]["human_edited"] is False
    assert reuse["local_review"]["gold_approved"] is True


def test_dataset_az_reuse_summary_distinguishes_import_from_local_trust():
    reused = {
        "provenance": {
            "az_reuse": {
                "reused": True,
                "cross_project_import": True,
                "required_local_review_at_materialization": True,
                "local_review": {
                    "approved": True,
                    "gold_approved": True,
                },
            }
        }
    }
    plain = {
        "provenance": {
            "category": provenance.RAW_MODEL_EXACT,
            "az_reuse": {},
        }
    }

    summary = provenance.summarize_dataset_az_reuse(
        [reused, reused, plain]
    )

    assert summary == {
        "schema": provenance.AZ_REUSE_SUMMARY_SCHEMA,
        "total_items": 3,
        "reused_items": 2,
        "cross_project_import_items": 2,
        "required_local_review_at_materialization_items": 2,
        "local_review_approved_items": 2,
        "local_gold_approved_items": 2,
    }


def test_non_reused_sample_has_empty_az_reuse_provenance():
    result = provenance.classify_plate_dataset_provenance(
        None,
        raw_exact_data(),
        meta_path=Path("metadata.json"),
    )
    assert result["az_reuse"] == {}
