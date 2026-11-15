"""Dataset provenance contract for Z3/PZ3 exports."""

from __future__ import annotations

from pathlib import Path

from .z3_gt_contract import (
    canonical_raw_detection_hash,
    revision_ids_from_data,
)


PROVENANCE_SCHEMA = "alpr.dataset.sample_provenance.v1"
AZ_REUSE_PROVENANCE_SCHEMA = "alpr.dataset.az_reuse_provenance.v1"
AZ_REUSE_SUMMARY_SCHEMA = "alpr.dataset.az_reuse_summary.v1"

RAW_MODEL_EXACT = "raw_model_exact"
MANUAL = "manual"
GT_ASSIST_CONFIRMED = "gt_assist_confirmed"
CVAT_IMPORT = "cvat_import"
MOBILE_HUMAN_REVIEW = "mobile_human_review"
LEGACY_UNTRACKED = "legacy_untracked"

KNOWN_CATEGORIES = (
    RAW_MODEL_EXACT,
    MANUAL,
    GT_ASSIST_CONFIRMED,
    CVAT_IMPORT,
    MOBILE_HUMAN_REVIEW,
    LEGACY_UNTRACKED,
)


def _norm(value) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _record_snapshot(record) -> tuple:
    if not isinstance(record, dict):
        return ("", ())
    character = str(record.get("character", "") or "").strip().upper()
    bbox = record.get("bbox", [])
    safe_bbox = []
    if isinstance(bbox, (list, tuple)):
        for value in list(bbox)[:4]:
            try:
                safe_bbox.append(round(float(value), 4))
            except Exception:
                safe_bbox.append(0.0)
    return character, tuple(safe_bbox)


def _final_matches_raw(data: dict) -> bool:
    raw = data.get("raw_detection")
    if not isinstance(raw, dict):
        return False

    raw_chars = raw.get("characters", [])
    final_chars = data.get("characters", [])
    if not isinstance(raw_chars, list) or not isinstance(final_chars, list):
        return False
    if len(raw_chars) != len(final_chars):
        return False

    return [
        _record_snapshot(rec)
        for rec in final_chars
    ] == [
        _record_snapshot(rec)
        for rec in raw_chars
    ]


def _has_gt_assist_confirmation(data: dict) -> bool:
    for field in ("correction_source", "review_source"):
        if _norm(data.get(field)) == GT_ASSIST_CONFIRMED:
            return True

    assist = data.get("gt_assist")
    if isinstance(assist, dict):
        if _norm(assist.get("status")) == "accepted":
            return True

    for rec in list(data.get("characters", []) or []):
        if not isinstance(rec, dict):
            continue
        if _norm(rec.get("correction_source")) == GT_ASSIST_CONFIRMED:
            return True
    return False


def _has_cvat_import(data: dict, source_bucket: str = "") -> bool:
    if _norm(source_bucket) == "cvat_manual":
        return True

    source_info = data.get("source_info")
    if isinstance(source_info, dict):
        if _norm(source_info.get("bucket")) == "cvat_manual":
            return True
        if _norm(source_info.get("origin")) == "cvat_import":
            return True

    for rec in list(data.get("characters", []) or []):
        if not isinstance(rec, dict):
            continue
        if _norm(rec.get("method")) in {"cvat", "cvat_manual", "cvat_import"}:
            return True
        if _norm(rec.get("source_kind")) == "cvat_manual":
            return True
    return False


def _has_local_manual(data: dict, source_bucket: str = "") -> bool:
    if _norm(source_bucket) == "local_manual":
        return True

    if _norm(data.get("fusion_strategy")) == "manual_correction":
        return True

    for field in ("correction_source", "review_source"):
        if _norm(data.get(field)) in {
            "manual",
            "local_manual",
            "manual_correction",
            "preview_editor",
        }:
            return True

    source_info = data.get("source_info")
    if isinstance(source_info, dict):
        if _norm(source_info.get("bucket")) == "local_manual":
            return True
        if _norm(source_info.get("origin")) in {
            "preview_editor",
            "manual_correction",
            "local_manual",
        }:
            return True

    for rec in list(data.get("characters", []) or []):
        if not isinstance(rec, dict):
            continue
        if _norm(rec.get("source_kind")) == "local_manual":
            return True
        if _norm(rec.get("method")) == "manual":
            return True
        if _norm(rec.get("box_source")) == "manual_box":
            return True
        if _norm(rec.get("sign_source")) == "manual_sign":
            return True

    return False


def _has_mobile_human_review(data: dict) -> bool:
    if _norm(data.get("source_expected_text_source")) == "mobile_crop_human_review":
        return True

    acquisition = data.get("mobile_acquisition")
    if isinstance(acquisition, dict):
        for field in (
            "human_review",
            "reviewed",
            "review_id",
            "review_revision",
        ):
            if acquisition.get(field):
                return True
    return False


def _raw_validation_matches_source_contract(
    data: dict,
) -> bool:
    raw = data.get("raw_detection")
    validation = data.get("raw_validation")
    if not isinstance(raw, dict) or not isinstance(validation, dict):
        return False

    validation_raw_hash = str(
        validation.get("raw_result_hash") or ""
    ).strip()
    if validation_raw_hash:
        current_raw_hash = canonical_raw_detection_hash(raw)
        if validation_raw_hash != current_raw_hash:
            return False

    validation_gt_hash = str(
        validation.get("gt_hash") or ""
    ).strip()
    current_gt_hash = str(
        data.get("source_gt_hash") or ""
    ).strip()
    if (
        validation_gt_hash
        and current_gt_hash
        and validation_gt_hash != current_gt_hash
    ):
        return False

    validation_revision_ids = revision_ids_from_data(
        validation,
        "gt_revision_ids",
        "gt_revision_id",
    )
    current_revision_ids = revision_ids_from_data(
        data,
        "source_gt_revision_ids",
        "source_gt_revision_id",
    )
    if (
        validation_revision_ids
        and validation_revision_ids != current_revision_ids
    ):
        return False

    return True


def _is_raw_model_exact(data: dict) -> bool:
    raw = data.get("raw_detection")
    validation = data.get("raw_validation")
    if not isinstance(raw, dict) or not isinstance(validation, dict):
        return False
    if str(raw.get("contract") or "").strip() != "gt_blind.v1":
        return False
    if not _raw_validation_matches_source_contract(data):
        return False
    if _norm(validation.get("status")) != "perfect":
        return False
    if not bool(validation.get("exact_text_match")):
        return False
    if not bool(validation.get("exact_count_match")):
        return False
    if not bool(validation.get("geometry_ok")):
        return False
    return _final_matches_raw(data)



def build_az_reuse_provenance(data: dict | None) -> dict:
    """Portable AZ reuse snapshot, separated from current local TARGET trust."""
    source_data = data if isinstance(data, dict) else {}
    az_reuse = (
        source_data.get("az_reuse")
        if isinstance(source_data.get("az_reuse"), dict)
        else {}
    )
    fusion_details = (
        source_data.get("fusion_details")
        if isinstance(source_data.get("fusion_details"), dict)
        else {}
    )
    review_state = (
        source_data.get("review_state")
        if isinstance(source_data.get("review_state"), dict)
        else {}
    )
    gold_state = (
        source_data.get("gold_state")
        if isinstance(source_data.get("gold_state"), dict)
        else {}
    )

    fusion_source = str(fusion_details.get("source") or "").strip()
    review_source = str(review_state.get("source") or "").strip()
    effective_status = str(az_reuse.get("effective_status") or "").strip()

    reused = bool(
        az_reuse
        or _norm(source_data.get("fusion_strategy")) == "az_reuse"
        or _norm(fusion_source) in {"az_registry", "az_project_import"}
    )
    if not reused:
        return {}

    cross_project_import = bool(
        _norm(fusion_source) == "az_project_import"
        or _norm(review_source) == "az_project_import"
        or _norm(effective_status) == "imported_pending_review"
    )
    local_review_status = str(review_state.get("status") or "").strip().lower()

    return {
        "schema": AZ_REUSE_PROVENANCE_SCHEMA,
        "reused": True,
        "cross_project_import": cross_project_import,
        "source_schema": str(az_reuse.get("schema") or "").strip(),
        "az_revision_id": str(
            az_reuse.get("az_revision_id")
            or fusion_details.get("az_revision_id")
            or ""
        ).strip(),
        "payload_sha256": str(az_reuse.get("payload_sha256") or "").strip(),
        "source_kind": str(az_reuse.get("source_kind") or "").strip(),
        "source_trust_state": str(az_reuse.get("trust_state") or "").strip(),
        "origin_project_id": (
            str(az_reuse.get("origin_project_id") or "").strip() or None
        ),
        "origin_iteration": az_reuse.get("origin_iteration"),
        "revision_project_id": (
            str(az_reuse.get("project_id") or "").strip() or None
        ),
        "revision_created_at": str(az_reuse.get("created_at") or "").strip(),
        "source_effective_status_at_materialization": effective_status or None,
        "required_local_review_at_materialization": bool(
            az_reuse.get("requires_review")
            or _norm(effective_status) == "imported_pending_review"
        ),
        "fusion_source": fusion_source,
        "local_review": {
            "status": local_review_status,
            "approved": local_review_status == "approved",
            "source": review_source,
            "human_edited": bool(review_state.get("human_edited", False)),
            "approved_at": (
                str(review_state.get("approved_at") or "").strip() or None
            ),
            "gold_approved": bool(gold_state.get("approved", False)),
        },
    }


def classify_plate_dataset_provenance(
    host,
    data: dict | None,
    *,
    meta_path: Path | None = None,
    source_bucket: str = "",
) -> dict:
    source_data = data if isinstance(data, dict) else {}
    evidence: list[str] = []

    if _has_gt_assist_confirmation(source_data):
        category = GT_ASSIST_CONFIRMED
        evidence.append("gt_assist_confirmed")
    elif _has_cvat_import(source_data, source_bucket):
        category = CVAT_IMPORT
        evidence.append("cvat_import")
    elif _has_local_manual(source_data, source_bucket):
        category = MANUAL
        evidence.append("manual_review")
    elif _has_mobile_human_review(source_data):
        category = MOBILE_HUMAN_REVIEW
        evidence.append("mobile_crop_human_review")
    elif _is_raw_model_exact(source_data):
        category = RAW_MODEL_EXACT
        evidence.append("gt_blind_raw_exact")
    else:
        category = LEGACY_UNTRACKED
        evidence.append("no_trusted_provenance_contract")

    raw = source_data.get("raw_detection")
    raw_validation = source_data.get("raw_validation")
    source_info = source_data.get("source_info")

    return {
        "schema": PROVENANCE_SCHEMA,
        "category": category,
        "evidence": evidence,
        "raw_contract": (
            str(raw.get("contract") or "").strip()
            if isinstance(raw, dict)
            else ""
        ),
        "raw_validation_status": (
            str(raw_validation.get("status") or "").strip()
            if isinstance(raw_validation, dict)
            else ""
        ),
        "source_image_id": str(
            source_data.get("source_image_id") or ""
        ).strip(),
        "source_annotation_id": str(
            source_data.get("source_annotation_id")
            or source_data.get("plate_annotation_id")
            or ""
        ).strip(),
        "source_geometry_hash": str(
            source_data.get("source_geometry_hash") or ""
        ).strip(),
        "source_gt_hash": str(
            source_data.get("source_gt_hash") or ""
        ).strip(),
        "source_geometry_revision_id": str(
            source_data.get("source_geometry_revision_id") or ""
        ).strip(),
        "source_geometry_revision_ids": revision_ids_from_data(
            source_data,
            "source_geometry_revision_ids",
            "source_geometry_revision_id",
        ),
        "source_gt_revision_id": str(
            source_data.get("source_gt_revision_id") or ""
        ).strip(),
        "source_gt_revision_ids": revision_ids_from_data(
            source_data,
            "source_gt_revision_ids",
            "source_gt_revision_id",
        ),
        "ground_truth_source": str(
            source_data.get("ground_truth_source") or ""
        ).strip(),
        "plate_source_bucket": str(source_bucket or "").strip(),
        "plate_source_origin": (
            str(source_info.get("origin") or "").strip()
            if isinstance(source_info, dict)
            else ""
        ),
        "meta_source": (
            str(meta_path)
            if meta_path is not None
            else ""
        ),
        "az_reuse": build_az_reuse_provenance(source_data),
    }


def collect_dataset_revision_ids(items) -> dict[str, list[str]]:
    """Aggregate GT/geometry revision provenance from exported samples."""
    gt_ids: set[str] = set()
    geometry_ids: set[str] = set()

    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        provenance = (
            item.get("provenance")
            if isinstance(item.get("provenance"), dict)
            else item
        )
        if not isinstance(provenance, dict):
            continue

        gt_ids.update(
            revision_ids_from_data(
                provenance,
                "source_gt_revision_ids",
                "source_gt_revision_id",
            )
        )
        geometry_ids.update(
            revision_ids_from_data(
                provenance,
                "source_geometry_revision_ids",
                "source_geometry_revision_id",
            )
        )

    return {
        "gt_revision_ids": sorted(gt_ids),
        "geometry_revision_ids": sorted(geometry_ids),
    }


def collect_dataset_revision_ids(items) -> dict[str, list[str]]:
    """Aggregate GT/geometry revision provenance from exported samples."""
    gt_ids: set[str] = set()
    geometry_ids: set[str] = set()

    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        provenance = (
            item.get("provenance")
            if isinstance(item.get("provenance"), dict)
            else item
        )
        if not isinstance(provenance, dict):
            continue

        gt_ids.update(
            revision_ids_from_data(
                provenance,
                "source_gt_revision_ids",
                "source_gt_revision_id",
            )
        )
        geometry_ids.update(
            revision_ids_from_data(
                provenance,
                "source_geometry_revision_ids",
                "source_geometry_revision_id",
            )
        )

    return {
        "gt_revision_ids": sorted(gt_ids),
        "geometry_revision_ids": sorted(geometry_ids),
    }


def summarize_dataset_provenance(items) -> dict[str, int]:
    counts = {category: 0 for category in KNOWN_CATEGORIES}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            continue
        category = str(
            provenance.get("category") or LEGACY_UNTRACKED
        ).strip()
        if category not in counts:
            category = LEGACY_UNTRACKED
        counts[category] = int(counts.get(category, 0) or 0) + 1
    return {
        key: int(value)
        for key, value in counts.items()
        if int(value) > 0
    }


def summarize_dataset_az_reuse(items) -> dict[str, int | str]:
    summary = {
        "schema": AZ_REUSE_SUMMARY_SCHEMA,
        "total_items": 0,
        "reused_items": 0,
        "cross_project_import_items": 0,
        "required_local_review_at_materialization_items": 0,
        "local_review_approved_items": 0,
        "local_gold_approved_items": 0,
    }

    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        summary["total_items"] += 1
        provenance = (
            item.get("provenance")
            if isinstance(item.get("provenance"), dict)
            else {}
        )
        reuse = (
            provenance.get("az_reuse")
            if isinstance(provenance.get("az_reuse"), dict)
            else {}
        )
        if not reuse:
            continue

        summary["reused_items"] += 1
        if bool(reuse.get("cross_project_import")):
            summary["cross_project_import_items"] += 1
        if bool(reuse.get("required_local_review_at_materialization")):
            summary["required_local_review_at_materialization_items"] += 1

        local_review = (
            reuse.get("local_review")
            if isinstance(reuse.get("local_review"), dict)
            else {}
        )
        if bool(local_review.get("approved")):
            summary["local_review_approved_items"] += 1
        if bool(local_review.get("gold_approved")):
            summary["local_gold_approved_items"] += 1

    return summary


RAW_BENCHMARK_SCHEMA = "alpr.pz2.raw_benchmark.v1"


def _canonical_json_sha256(value) -> str:
    import hashlib
    import json

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()



GOLD_SOURCE_CONTRACT_SCHEMA = "alpr.pz3.gold_source_contract.v1"


def build_gold_source_contract_fingerprint(entries) -> dict:
    """Fingerprint the exact canonical GOLD source selected for PZ3 export."""
    rows = []
    total_characters = 0

    for item in list(entries or []):
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        if not isinstance(data, dict):
            continue

        review_state = (
            data.get("review_state")
            if isinstance(data.get("review_state"), dict)
            else {}
        )
        approved_reference = (
            review_state.get("approved_reference")
            if isinstance(review_state.get("approved_reference"), dict)
            else {}
        )

        characters = []
        for rec in list(data.get("characters") or []):
            if not isinstance(rec, dict):
                continue
            try:
                bbox = [
                    round(float(value), 6)
                    for value in list(rec.get("bbox") or [])[:4]
                ]
            except Exception:
                bbox = []
            characters.append(
                {
                    "character": str(
                        rec.get("character") or ""
                    ).strip().upper(),
                    "bbox": bbox,
                }
            )

        characters.sort(
            key=lambda row: (
                tuple(row.get("bbox") or []),
                str(row.get("character") or ""),
            )
        )
        total_characters += len(characters)

        rows.append(
            {
                "pid": str(item.get("pid") or "").strip(),
                "strategy_bucket": str(
                    item.get("strategy_bucket") or ""
                ).strip(),
                "source_bucket": str(
                    item.get("source_bucket") or ""
                ).strip(),
                "source_image": str(
                    data.get("source_image") or ""
                ).strip(),
                "source_image_id": str(
                    data.get("source_image_id") or ""
                ).strip(),
                "source_annotation_id": str(
                    data.get("source_annotation_id")
                    or data.get("plate_annotation_id")
                    or ""
                ).strip(),
                "source_gt_hash": str(
                    data.get("source_gt_hash") or ""
                ).strip(),
                "source_gt_revision_ids": revision_ids_from_data(
                    data,
                    "source_gt_revision_ids",
                    "source_gt_revision_id",
                ),
                "source_geometry_hash": str(
                    data.get("source_geometry_hash") or ""
                ).strip(),
                "source_geometry_revision_ids": revision_ids_from_data(
                    data,
                    "source_geometry_revision_ids",
                    "source_geometry_revision_id",
                ),
                "ground_truth_text": str(
                    data.get("ground_truth_text") or ""
                ).strip().upper(),
                "review_status": str(
                    review_state.get("status") or ""
                ).strip().lower(),
                "approved_reference": approved_reference,
                "characters": characters,
            }
        )

    rows.sort(
        key=lambda row: (
            str(row.get("source_image_id") or ""),
            str(row.get("source_annotation_id") or ""),
            str(row.get("source_image") or ""),
            str(row.get("pid") or ""),
            _canonical_json_sha256(row),
        )
    )

    payload = {
        "schema": GOLD_SOURCE_CONTRACT_SCHEMA,
        "plates": rows,
    }
    return {
        "schema": GOLD_SOURCE_CONTRACT_SCHEMA,
        "sha256": _canonical_json_sha256(payload),
        "plate_count": len(rows),
        "character_count": int(total_characters),
    }


def _edit_distance(left: str, right: str) -> int:
    left = str(left or "")
    right = str(right or "")
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return int(previous[-1])


def build_gt_blind_raw_benchmark(records) -> dict:
    """Aggregate exact-plate and corpus CER from gt_blind.v1 RAW only.

    ``records`` may contain preview metadata dictionaries directly or
    Gold-export candidate dictionaries with the preview record in ``data``.
    """
    source_records = list(records or [])
    total_records = 0
    gt_blind_records = 0
    evaluable_records = 0
    exact_plate_count = 0
    exact_text_count = 0
    box_count_exact_count = 0
    box_count_abs_error_total = 0
    total_gt_characters = 0
    total_edit_distance = 0
    missing_ground_truth_count = 0
    excluded_non_blind_count = 0
    invalid_validation_count = 0
    gt_contract_rows = []
    hashed_gt_count = 0
    revisioned_gt_count = 0

    for item in source_records:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else item
        if not isinstance(data, dict):
            continue

        total_records += 1
        raw = data.get("raw_detection")
        validation = data.get("raw_validation")
        if (
            not isinstance(raw, dict)
            or str(raw.get("contract") or "").strip() != "gt_blind.v1"
        ):
            excluded_non_blind_count += 1
            continue

        gt_blind_records += 1
        if not isinstance(validation, dict):
            invalid_validation_count += 1
            continue
        if str(validation.get("raw_contract") or "").strip() not in {
            "",
            "gt_blind.v1",
        }:
            invalid_validation_count += 1
            continue
        if str(validation.get("expected_source") or "").strip() != "ground_truth":
            invalid_validation_count += 1
            continue
        if not _raw_validation_matches_source_contract(data):
            invalid_validation_count += 1
            continue

        gt_text = str(validation.get("ground_truth_text") or "").strip().upper()
        if not gt_text:
            missing_ground_truth_count += 1
            continue

        prediction_text = str(
            validation.get("prediction_text")
            or raw.get("prediction_text")
            or ""
        ).strip().upper()

        try:
            edit_distance = int(validation.get("edit_distance"))
        except Exception:
            edit_distance = _edit_distance(prediction_text, gt_text)

        evaluable_records += 1
        total_gt_characters += len(gt_text)
        total_edit_distance += max(0, edit_distance)

        text_exact = bool(
            validation.get("exact_text_match")
            and prediction_text == gt_text
        )
        if text_exact:
            exact_text_count += 1

        plate_exact = bool(
            str(validation.get("status") or "").strip().lower() == "perfect"
            and text_exact
            and bool(validation.get("exact_count_match"))
            and bool(validation.get("geometry_ok"))
        )
        if plate_exact:
            exact_plate_count += 1

        try:
            box_count_error = int(
                validation.get("box_count_error")
            )
        except Exception:
            try:
                expected_count = int(
                    validation.get("expected_char_count")
                    or len(gt_text)
                )
            except Exception:
                expected_count = len(gt_text)
            try:
                detected_count = int(
                    validation.get("detected_char_count")
                    or len(raw.get("characters", []) or [])
                )
            except Exception:
                detected_count = len(
                    raw.get("characters", []) or []
                )
            box_count_error = int(
                detected_count - expected_count
            )

        if box_count_error == 0:
            box_count_exact_count += 1
        box_count_abs_error_total += abs(box_count_error)

        source_gt_hash = str(data.get("source_gt_hash") or "").strip()
        if source_gt_hash:
            hashed_gt_count += 1
        source_gt_revision_ids = revision_ids_from_data(
            data,
            "source_gt_revision_ids",
            "source_gt_revision_id",
        )
        if source_gt_revision_ids:
            revisioned_gt_count += 1

        gt_contract_rows.append(
            {
                "source_image_id": str(
                    data.get("source_image_id") or ""
                ).strip(),
                "source_annotation_id": str(
                    data.get("source_annotation_id")
                    or data.get("plate_annotation_id")
                    or ""
                ).strip(),
                "source_gt_hash": source_gt_hash,
                "source_gt_revision_id": str(
                    data.get("source_gt_revision_id") or ""
                ).strip(),
                "source_gt_revision_ids": list(
                    source_gt_revision_ids
                ),
                "ground_truth_text": gt_text,
            }
        )

    if evaluable_records > 0:
        exact_plate_rate = (
            float(exact_plate_count) / float(evaluable_records)
        )
        exact_text_rate = (
            float(exact_text_count) / float(evaluable_records)
        )
        box_count_accuracy = (
            float(box_count_exact_count)
            / float(evaluable_records)
        )
        mean_abs_box_count_error = (
            float(box_count_abs_error_total)
            / float(evaluable_records)
        )
    else:
        exact_plate_rate = None
        exact_text_rate = None
        box_count_accuracy = None
        mean_abs_box_count_error = None

    cer = (
        float(total_edit_distance) / float(total_gt_characters)
        if total_gt_characters > 0
        else None
    )

    gt_contract_rows = sorted(
        gt_contract_rows,
        key=lambda row: (
            row["source_image_id"],
            row["source_annotation_id"],
            row["source_gt_hash"],
            row["source_gt_revision_id"],
            tuple(row["source_gt_revision_ids"]),
            row["ground_truth_text"],
        ),
    )
    gt_contract_fingerprint = (
        _canonical_json_sha256(gt_contract_rows)
        if gt_contract_rows
        else ""
    )

    if evaluable_records <= 0:
        gt_hash_coverage = "none"
        gt_revision_coverage = "none"
    elif hashed_gt_count >= evaluable_records:
        gt_hash_coverage = "full"
    elif hashed_gt_count > 0:
        gt_hash_coverage = "partial"
    else:
        gt_hash_coverage = "text_only"

    if evaluable_records > 0:
        if revisioned_gt_count >= evaluable_records:
            gt_revision_coverage = "full"
        elif revisioned_gt_count > 0:
            gt_revision_coverage = "partial"
        else:
            gt_revision_coverage = "none"

    return {
        "schema": RAW_BENCHMARK_SCHEMA,
        "raw_contract": "gt_blind.v1",
        "metric_policy": "exact_plate_and_corpus_cer.v1",
        "total_input_plates": int(total_records),
        "gt_blind_plates": int(gt_blind_records),
        "evaluable_plates": int(evaluable_records),
        "exact_plate_count": int(exact_plate_count),
        "exact_plate_rate": (
            round(float(exact_plate_rate), 8)
            if exact_plate_rate is not None
            else None
        ),
        "exact_text_count": int(exact_text_count),
        "exact_text_rate": (
            round(float(exact_text_rate), 8)
            if exact_text_rate is not None
            else None
        ),
        "box_count_exact_count": int(
            box_count_exact_count
        ),
        "box_count_accuracy": (
            round(float(box_count_accuracy), 8)
            if box_count_accuracy is not None
            else None
        ),
        "box_count_abs_error_total": int(
            box_count_abs_error_total
        ),
        "mean_abs_box_count_error": (
            round(float(mean_abs_box_count_error), 8)
            if mean_abs_box_count_error is not None
            else None
        ),
        "gt_characters": int(total_gt_characters),
        "edit_distance_total": int(total_edit_distance),
        "cer": (
            round(float(cer), 8)
            if cer is not None
            else None
        ),
        "missing_ground_truth_count": int(missing_ground_truth_count),
        "excluded_non_blind_count": int(excluded_non_blind_count),
        "invalid_validation_count": int(invalid_validation_count),
        "gt_hash_coverage": gt_hash_coverage,
        "gt_hash_count": int(hashed_gt_count),
        "gt_revision_coverage": gt_revision_coverage,
        "gt_revision_count": int(revisioned_gt_count),
        "gt_contract_fingerprint_sha256": gt_contract_fingerprint,
    }
