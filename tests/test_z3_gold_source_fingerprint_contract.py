from __future__ import annotations

from pathlib import Path

from auto_annotation_tool.gui import z3_dataset_provenance as provenance
from auto_annotation_tool.gui import z3_readiness as readiness


def _entry(pid: str, char: str = "A", x1: float = 1.0):
    data = {
        "source_image": f"{pid}.jpg",
        "source_image_id": f"img-{pid}",
        "source_annotation_id": f"ann-{pid}",
        "source_gt_hash": f"gt-{pid}",
        "source_gt_revision_ids": [f"gt-r-{pid}"],
        "source_geometry_hash": f"geom-{pid}",
        "source_geometry_revision_ids": [f"geom-r-{pid}"],
        "ground_truth_text": char,
        "review_state": {
            "status": "approved",
            "approved_reference": {
                "gt_hash": f"gt-{pid}",
                "gt_revision_ids": [f"gt-r-{pid}"],
                "geometry_revision_ids": [f"geom-r-{pid}"],
                "ground_truth_text": char,
            },
        },
        "characters": [
            {
                "character": char,
                "bbox": [x1, 2.0, x1 + 10.0, 22.0],
            }
        ],
    }
    return {
        "pid": pid,
        "strategy_bucket": "manual",
        "source_bucket": "auto_preview",
        "data": data,
    }


def test_gold_source_fingerprint_is_order_stable():
    left = provenance.build_gold_source_contract_fingerprint(
        [_entry("p1"), _entry("p2", "B")]
    )
    right = provenance.build_gold_source_contract_fingerprint(
        [_entry("p2", "B"), _entry("p1")]
    )

    assert left["schema"] == provenance.GOLD_SOURCE_CONTRACT_SCHEMA
    assert left["sha256"] == right["sha256"]
    assert left["plate_count"] == 2
    assert left["character_count"] == 2


def test_gold_source_fingerprint_changes_on_character_or_geometry_edit():
    base = provenance.build_gold_source_contract_fingerprint(
        [_entry("p1", "A", 1.0)]
    )
    changed_char = provenance.build_gold_source_contract_fingerprint(
        [_entry("p1", "B", 1.0)]
    )
    changed_box = provenance.build_gold_source_contract_fingerprint(
        [_entry("p1", "A", 3.0)]
    )

    assert base["sha256"] != changed_char["sha256"]
    assert base["sha256"] != changed_box["sha256"]


def test_gold_source_fingerprint_changes_on_approved_reference_change():
    first = _entry("p1")
    second = _entry("p1")
    second["data"]["review_state"]["approved_reference"]["gt_hash"] = "gt-new"

    left = provenance.build_gold_source_contract_fingerprint([first])
    right = provenance.build_gold_source_contract_fingerprint([second])

    assert left["sha256"] != right["sha256"]


def test_new_summary_requires_matching_current_gold_contract():
    summary = {"gold_source_contract_sha256": "abc"}

    assert readiness.step3_export_summary_matches_current_gold_contract(
        summary,
        {"gold_source_contract_sha256": "abc"},
    )
    assert not readiness.step3_export_summary_matches_current_gold_contract(
        summary,
        {"gold_source_contract_sha256": "def"},
    )
    assert not readiness.step3_export_summary_matches_current_gold_contract(
        summary,
        {},
    )


def test_legacy_summary_without_fingerprint_stays_compatible():
    assert readiness.step3_export_summary_matches_current_gold_contract(
        {"gold_dataset_created": True},
        {},
    )


def test_campaign_return_does_not_bypass_training_readiness_with_summary_only():
    from auto_annotation_tool.gui import z3_campaign_flow as flow

    source = Path(flow.__file__).read_text(encoding="utf-8-sig")
    start = source.index("def return_to_wizard_for_step3_rework(")
    end = source.index("\ndef _step3_preview_ready_for_pz2(", start)
    body = source[start:end]

    ready_start = body.index("ready_for_approval = bool(")
    ready_end = body.index("\n\n            if current_status", ready_start)
    ready_block = body[ready_start:ready_end]

    assert "host._has_any_step3_export_outputs()" in ready_block
    assert 'readiness.get("ok")' in ready_block
    assert "or bool(ready_summary)" not in ready_block


def test_export_runtime_persists_gold_source_fingerprint():
    from auto_annotation_tool.gui import z3_goldpack_ui as gold

    source = Path(gold.__file__).read_text(encoding="utf-8-sig")
    start = source.index("def run_yolo_gold_export(")
    end = source.index("\ndef run_pz3_existing_dataset_split(", start)
    body = source[start:end]

    assert "build_gold_source_contract_fingerprint(" in body
    assert '"gold_source_contract_sha256"' in body
    assert "prepared_source_contract" in body
