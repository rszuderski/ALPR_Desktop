"""ROIv1: evidence gates for *formal* per-plate MT science (never guessed).

Historical multi_plate_v1 stores an operator's correspondence on an MT input
image, not a frozen full-frame independent-GT experiment. This module records
what an actual invocation saw, separating ROI context from scientific approval.
It does not touch ZIP archives, GT sidecars or neural-network results.
"""
from __future__ import annotations

import math
from typing import Any

AUDIT_SCHEMA = "alpr.mobile.mt_roi_evidence_audit.v1"
FORMAL_SCHEMA = "alpr.mobile.multi_plate_formal_gate.v1"


def _num(value):
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None


def invocation_roi_audit(session, invocation_id: str) -> dict[str, Any]:
    """Describe an MT invocation, not an entire camera stream/frame.

    Image `samples/evidence/*` is *preprocessed MT input*. Even with R0 it
    is never relabelled as an original, independently archived source frame.
    Future Android source-frame v2 needs separate verification of the file
    hash, orientation and dimensions before a coverage metric can use it.
    """
    group = session.mt_invocations.get(invocation_id)
    if group is None:
        return dict(schema=AUDIT_SCHEMA, status="UNKNOWN_INVOCATION", invocation_id=invocation_id,
                    frame_key=None, roi_source_px=None, mt_input_evidence=False,
                    full_source_frame_verified=False, reasons=["UNKNOWN_INVOCATION"])
    records = list(group.records)
    reasons: list[str] = []
    if not records or not group.executed or group.cancelled or group.execution_failed:
        reasons.append("MT_INVOCATION_NOT_ELIGIBLE")
    frame_keys = set()
    roi_keys = set()
    roi_policies = set()
    for row in records:
        frame_key = tuple(str(row.get(field) or "").strip() for field in (
            "session_id", "source_sequence", "source_timestamp_domain",
            "source_timestamp_nanos", "visual_epoch", "camera_transform_generation"))
        frame_keys.add(frame_key)
        roi_keys.add(tuple(_num(row.get(field)) for field in (
            "roi_left", "roi_top", "roi_right", "roi_bottom")))
        roi_policies.add(str(row.get("roi_policy") or row.get("mt_roi_policy") or "").strip().lower())
    if len(frame_keys) != 1 or len(roi_keys) != 1 or len(roi_policies) != 1:
        reasons.append("INCONSISTENT_INVOCATION_GEOMETRY_OR_FRAME")
    frame_key = next(iter(frame_keys)) if len(frame_keys) == 1 else None
    if frame_key is None or not all(frame_key[i] for i in (0, 1, 2, 3)):
        reasons.append("FRAME_IDENTITY_INCOMPLETE")
    roi = next(iter(roi_keys)) if len(roi_keys) == 1 else None
    if roi is None or any(value is None for value in roi) or not (
        roi[0] >= 0 and roi[1] >= 0 and roi[2] > roi[0] and roi[3] > roi[1]):
        reasons.append("ROI_GEOMETRY_MISSING_OR_INVALID")
        roi = None
    policy = next(iter(roi_policies)) if len(roi_policies) == 1 else ""
    if not policy:
        reasons.append("ROI_POLICY_UNKNOWN")
    input_geometry = None
    if records:
        a = records[0]
        input_geometry = tuple(_num(a.get(field)) for field in (
            "input_width", "input_height", "input_scale", "input_pad_x", "input_pad_y"))
    if not input_geometry or any(v is None for v in input_geometry) or not (
        input_geometry[0] > 0 and input_geometry[1] > 0 and input_geometry[2] > 0
        and input_geometry[3] >= 0 and input_geometry[4] >= 0):
        reasons.append("INPUT_TRANSFORM_UNVERIFIED")
        input_geometry = None
    elif any(tuple(_num(row.get(field)) for field in (
            "input_width", "input_height", "input_scale", "input_pad_x", "input_pad_y")) != input_geometry
            for row in records):
        reasons.append("INCONSISTENT_INPUT_TRANSFORM")
    entry_names = set(getattr(session, "entry_names", ()))
    mt_input_ok = any(str(entry) in entry_names for entry in group.evidence_entries)
    if not mt_input_ok:
        reasons.append("MT_INPUT_EVIDENCE_MISSING")
    # Importantly: presence of a path is NOT image/hash/orientation verification.
    full_source_frame_verified = False
    if not any(str(row.get("source_frame_entry") or "").strip() for row in records):
        reasons.append("NO_FULL_SOURCE_FRAME_EVIDENCE")
    else:
        reasons.append("SOURCE_FRAME_V2_NOT_VALIDATED")
    status = "ROI_UNVERIFIED"
    if roi and input_geometry and policy and not any(k in reasons for k in (
        "ROI_GEOMETRY_MISSING_OR_INVALID", "ROI_POLICY_UNKNOWN", "INCONSISTENT_INVOCATION_GEOMETRY_OR_FRAME",
        "INPUT_TRANSFORM_UNVERIFIED", "INCONSISTENT_INPUT_TRANSFORM")):
        if policy == "r0_full_frame" and roi[0] == 0 and roi[1] == 0:
            status = "FULL_MT_INPUT_CONTEXT_ONLY"
        elif policy == "r0_full_frame":
            status = "ROI_UNVERIFIED"
            reasons.append("R0_ROI_ORIGIN_CONFLICT")
        else:
            status = "PARTIAL_OR_UNKNOWN_MT_ROI_CONTEXT_ONLY"
    return dict(schema=AUDIT_SCHEMA, status=status, invocation_id=invocation_id,
                frame_key=list(frame_key) if frame_key is not None else None,
                roi_policy=policy or None,
                roi_source_px=list(roi) if roi else None,
                input_transform=list(input_geometry) if input_geometry else None,
                mt_input_evidence=mt_input_ok,
                full_source_frame_verified=full_source_frame_verified,
                reasons=list(dict.fromkeys(reasons)))


def formal_multi_plate_gate(session, invocation_id: str, inv: dict | None) -> dict[str, Any]:
    """Explicit denial for legacy operator matched/assisted data.

    Upgrading this gate to MEASURED requires an independently verified v2
    full-frame GT contract, fixed IoU policy and geometric one-to-one match.
    A UI checkbox or an operator's `assign_attempt` can never unlock it.
    """
    audit = invocation_roi_audit(session, invocation_id)
    reasons = list(audit["reasons"])
    if not audit["full_source_frame_verified"]:
        reasons.append("FORMAL_FULL_FRAME_NOT_VERIFIED")
    if getattr(session.review, "review_mode", "") not in (
        "blinded", "independent_blinded", "blinded_manual_v1"):
        reasons.append("ASSISTED_OR_UNVERIFIED_GT_PROVENANCE")
    plates = inv.get("plates") if isinstance(inv, dict) else None
    if not isinstance(plates, list) or not plates:
        reasons.append("NO_GT_PLATE_INVENTORY")
    else:
        if any(not isinstance(p, dict) or not p.get("polygon_source_frame_px") for p in plates):
            reasons.append("NO_FULL_FRAME_GT_POLYGONS")
        if any(not isinstance(p, dict) or p.get("gt_review_mode") != "blinded_independent_frozen_v1"
               for p in plates):
            reasons.append("NO_FROZEN_INDEPENDENT_GT")
    if not (isinstance(inv, dict) and isinstance(inv.get("frozen_matching_protocol"), dict)):
        reasons.append("NO_FROZEN_IOU_MATCHING_PROTOCOL")
    # The old schema is always diagnostic: even forged extra keys cannot
    # turn it into a new protocol without an explicitly versioned validator.
    reasons.append("LEGACY_MULTI_PLATE_V1_OPERATOR_MAPPING_ONLY")
    return dict(schema=FORMAL_SCHEMA, status="NOT_EVALUABLE", tp=None, fp=None, fn=None,
                evaluable_gt_plates=None, iou_threshold=None, reasons=list(dict.fromkeys(reasons)),
                invocation=audit)
