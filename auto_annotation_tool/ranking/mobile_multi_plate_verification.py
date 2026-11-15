"""Z4MOBREV024 — one-to-one MT predictions and independently verified GT plates.

The data live ONLY in the existing SHA-bound human review sidecar under
invocation_annotations[*].multi_plate_v1. No source image, predictions or
historical GT are changed. Exact GT count is required for multiple plates;
visual status is never converted into ground-truth geometry.
"""
from __future__ import annotations

import copy
import math
from typing import Any

SCHEMA = "alpr.mobile.mt_multi_plate.v1"
MAX_PLATES = 32
DETECTION_STATUSES = frozenset({"VALID_QUAD", "DETECTION_INVALID_QUAD"})


def is_multi(session, invocation_id: str) -> bool:
    from .mobile_canvas_review import has_scene
    if has_scene(session, invocation_id):
        return False  # Active canvas data are reported separately; legacy inventory stays in history.
    group = session.mt_invocations.get(invocation_id)
    if group is None or not group.executed or group.cancelled or group.execution_failed:
        return False
    return session.invocation_annotation(invocation_id).get("visible_plate_count") == "multiple"


def _ann(session, invocation_id: str) -> dict:
    value = session.invocation_annotation(invocation_id)
    return value if isinstance(value, dict) else {}


def inventory(session, invocation_id: str) -> dict | None:
    if not is_multi(session, invocation_id):
        return None
    value = _ann(session, invocation_id).get("multi_plate_v1")
    return value if isinstance(value, dict) and value.get("schema") == SCHEMA else None


def _records(group) -> dict[str, dict]:
    return {str(row.get("id") or ""): row for row in group.records
            if str(row.get("mt_status") or "").upper() in DETECTION_STATUSES}


def _write(session, invocation_id: str, updated: dict) -> None:
    if not is_multi(session, invocation_id):
        raise ValueError("To wywołanie MT nie jest aktywną sceną z kilkoma tablicami")
    changed = copy.deepcopy(session.review)
    annotation = dict(changed.invocation_annotations.get(invocation_id, {}))
    if annotation.get("visible_plate_count") != "multiple":
        raise ValueError("Najpierw wskaż kilka prawdziwych tablic na zdjęciu")
    annotation["multi_plate_v1"] = updated
    changed.invocation_annotations[invocation_id] = annotation
    changed.review_status = "IN_PROGRESS"
    session._save_change(changed)


def set_count(session, invocation_id: str, count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= MAX_PLATES:
        raise ValueError(f"Podaj rzeczywistą liczbę tablic od 2 do {MAX_PLATES}")
    original = inventory(session, invocation_id)
    old_plates = list(original.get("plates") or ()) if original else []
    if count < len(old_plates):
        if any(p.get("attempt_id") or p.get("bbox_mt_px") or p.get("gt_text") or
               p.get("text_exclusion_reason") for p in old_plates[count:]):
            raise ValueError("Najpierw usuń przypisania lub GT ostatnich tablic; nie kasujemy ich automatycznie")
    plates = old_plates[:count]
    for i in range(len(plates), count):
        plates.append({"plate_id": f"GT{i + 1:02d}", "attempt_id": "", "gt_text": "",
                       "text_exclusion_reason": "", "bbox_mt_px": None,
                       "bbox_image_entry": "", "bbox_input_size_px": None})
    _write(session, invocation_id, {"schema": SCHEMA, "plate_count": count, "plates": plates})


def _edit_plate(session, invocation_id: str, plate_id: str, modify) -> None:
    inv = inventory(session, invocation_id)
    if inv is None:
        raise ValueError("Najpierw określ dokładną liczbę prawdziwych tablic")
    target = copy.deepcopy(inv)
    hits = [p for p in target["plates"] if p.get("plate_id") == plate_id]
    if len(hits) != 1:
        raise ValueError("Nieznany identyfikator tablicy GT")
    modify(hits[0], target)
    _write(session, invocation_id, target)


def assign_attempt(session, invocation_id: str, plate_id: str, attempt_id: str) -> None:
    """Operator pairs an individual predicted region to exactly one GT plate."""
    group = session.mt_invocations[invocation_id]
    candidates = _records(group)
    aid = str(attempt_id or "")
    if aid and aid not in candidates:
        raise ValueError("Wskazanie MT nie należy do tego wejścia lub nie jest detekcją")
    if aid:
        from .mobile_verification_policy import selected_visual_state
        state = selected_visual_state(session.review.attempt_annotations.get(aid, {}))
        if state not in ("accurate", "imprecise"):
            raise ValueError("Najpierw potwierdź, że wskazanie MT przedstawia prawdziwą tablicę")

    def change(plate, inv):
        if aid and any(p["plate_id"] != plate_id and p.get("attempt_id") == aid for p in inv["plates"]):
            raise ValueError("Jedna predykcja MT nie może należeć do dwóch tablic GT")
        plate["attempt_id"] = aid
    _edit_plate(session, invocation_id, plate_id, change)


def set_plate_text(session, invocation_id: str, plate_id: str, gt_text: str,
                   exclusion_reason: str = "") -> None:
    from .mobile_human_review import normalize_registration
    gt = normalize_registration(gt_text)
    reason = " ".join(str(exclusion_reason or "").split())
    if gt and reason:
        raise ValueError("GT numeru i przyczyna wykluczenia wzajemnie się wykluczają")
    if len(gt) > 256 or len(reason) > 240:
        raise ValueError("Zbyt długi numer GT lub powód wyłączenia")
    def change(plate, inv):
        plate["gt_text"] = gt
        plate["text_exclusion_reason"] = reason
    _edit_plate(session, invocation_id, plate_id, change)


def set_missed_bbox(session, invocation_id: str, plate_id: str, bbox: tuple,
                    image_entry: str, input_size: tuple) -> None:
    """Manual 2-corner *GT BBOX* only for an unmatched plate, never MT output."""
    group = session.mt_invocations[invocation_id]
    if image_entry not in group.evidence_entries or image_entry not in session.entry_names:
        raise ValueError("Niezweryfikowany obraz wejścia MT")
    try:
        w, h = map(float, input_size)
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        raise ValueError("Nieprawidłowe współrzędne ramki GT") from None
    if not (all(map(math.isfinite, (w, h, x1, y1, x2, y2))) and
            w > 0 and h > 0 and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and
            x2 - x1 >= 3 and y2 - y1 >= 3):
        raise ValueError("GT musi być rzeczywistym, niepustym prostokątem na wejściu MT")
    # Ensure every recorded member agrees on dimensions for this invocation.
    for attempt in group.records:
        if attempt.get("mt_input_evidence_entry") == image_entry:
            if (float(attempt["input_width"]),float(attempt["input_height"])) != (w,h):
                raise ValueError("Niezgodna geometria wejścia MT")

    def change(plate, inv):
        if plate.get("attempt_id"):
            raise ValueError("GT pominiętej tablicy nie może równocześnie wskazywać predykcji MT")
        plate["bbox_mt_px"] = [x1, y1, x2, y2]
        plate["bbox_image_entry"] = image_entry
        plate["bbox_input_size_px"] = [w, h]
    _edit_plate(session, invocation_id, plate_id, change)


def clear_missed_bbox(session, invocation_id: str, plate_id: str) -> None:
    def change(plate, inv):
        plate.update(bbox_mt_px=None, bbox_image_entry="", bbox_input_size_px=None)
    _edit_plate(session, invocation_id, plate_id, change)


def _valid_bbox_for_group(session, group, plate) -> bool:
    bbox = plate.get("bbox_mt_px")
    entry = plate.get("bbox_image_entry")
    size = plate.get("bbox_input_size_px")
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4 or not isinstance(size, (tuple,list)) or len(size)!=2:
        return False
    if entry not in session.entry_names or entry not in group.evidence_entries:
        return False
    try:
        w,h=map(float,size)
        x1,y1,x2,y2=map(float,bbox)
        return (all(map(math.isfinite, (w,h,x1,y1,x2,y2))) and w>0 and h>0 and
                0<=x1<x2<=w and 0<=y1<y2<=h and x2-x1>=3 and y2-y1>=3 and
                any(str(row.get("mt_input_evidence_entry") or "") == entry and
                    float(row.get("input_width")) == w and float(row.get("input_height")) == h
                    for row in group.records))
    except (ValueError, TypeError, OverflowError):
        return False


def _mz_ran(session, attempt_id: str) -> bool:
    attempt = session.attempts[attempt_id]
    status = str(attempt.get("mz_status") or "").upper()
    if status and status not in {"NOT_RUN", "UNKNOWN"}:
        return True
    for sid in session.sample_ids_by_attempt.get(attempt_id, ()):
        sample=session.samples.get(sid,{})
        status=str(sample.get("mz_status") or "").upper()
        if status and status not in {"NOT_RUN", "UNKNOWN"}:
            return True
    return False


def group_issues(session, invocation_id: str) -> list[str]:
    if not is_multi(session, invocation_id):
        return []
    group = session.mt_invocations[invocation_id]
    inv = inventory(session, invocation_id)
    prefix = f"MT {invocation_id}"
    if inv is None:
        return [f"{prefix}: podaj dokładną liczbę tablic GT (nie tylko 2+)"]
    n = inv.get("plate_count")
    plates = inv.get("plates")
    if (type(n) is not int or not 2 <= n <= MAX_PLATES or
            not isinstance(plates,list) or len(plates) != n or
            [p.get("plate_id") for p in plates if isinstance(p,dict)] !=
            [f"GT{i+1:02d}" for i in range(n)]):
        return [f"{prefix}: niespójny wykaz tablic GT"]
    from .mobile_verification_policy import selected_visual_state
    records = _records(group)
    pairs: set[str] = set()
    issues: list[str] = []
    for aid, row in records.items():
        state = selected_visual_state(session.review.attempt_annotations.get(aid, {}))
        if state not in ("accurate", "imprecise", "not_plate"):
            issues.append(f"MT {aid}: zweryfikuj, czy wskazanie jest tablicą (nie może pozostać niepewne)")
        # A correction selected in the visual editor must be truly persisted,
        # also for multiple GT plates. This guards BOTH form readiness and
        # scientific multi-plate aggregation via the same group_issues().
        from .mobile_verification_policy import _guided_correction_issue
        guided_issue = _guided_correction_issue(session, row)
        if guided_issue:
            issues.append(guided_issue)
    for plate in plates:
        pid = plate["plate_id"]
        aid = str(plate.get("attempt_id") or "")
        if aid:
            if aid not in records or aid in pairs:
                issues.append(f"{prefix}/{pid}: błędne lub podwójne powiązanie z MT")
            else:
                pairs.add(aid)
                state=selected_visual_state(session.review.attempt_annotations.get(aid, {}))
                if state not in ("accurate", "imprecise"):
                    issues.append(f"{prefix}/{pid}: przypisana detekcja nie jest potwierdzoną tablicą")
            if plate.get("bbox_mt_px") is not None:
                issues.append(f"{prefix}/{pid}: nie mieszaj ręcznego GT pominięcia z dopasowaną detekcją")
            if aid in records and _mz_ran(session, aid):
                if not str(plate.get("gt_text") or "").strip() and not str(plate.get("text_exclusion_reason") or "").strip():
                    issues.append(f"{prefix}/{pid}: brak niezależnego GT numeru lub przyczyny wykluczenia OCR")
        elif not _valid_bbox_for_group(session, group, plate):
            issues.append(f"{prefix}/{pid}: brak ręcznej ramki GT pominiętej tablicy")
        if plate.get("gt_text") and plate.get("text_exclusion_reason"):
            issues.append(f"{prefix}/{pid}: nie można jednocześnie wykluczyć i oceniać tego samego numeru")
    for aid in records:
        state=selected_visual_state(session.review.attempt_annotations.get(aid, {}))
        if state in ("accurate", "imprecise") and aid not in pairs:
            issues.append(f"MT {aid}: prawdziwa tablica wymaga wskazania odpowiedniego GT")
        if state == "not_plate" and aid in pairs:
            issues.append(f"MT {aid}: fałszywe wskazanie nie może być przypisane do GT")
    if not any(entry in session.entry_names for entry in group.evidence_entries):
        issues.append(f"{prefix}: brak zweryfikowanego zdjęcia wejścia MT")
    return issues


def multi_scientific_summary(session) -> dict[str, Any]:
    """Operator mappings are diagnostic; formal per-plate TP/FP/FN are gated.

    `multi_plate_v1` lacks full-frame independent GT and a frozen geometric
    matching protocol. A completed form is not a completed MT science result.
    Old sidecars are read-only here and retain their full operator inventory.
    """
    from .mobile_verification_policy import selected_visual_state
    from .mobile_roi_evaluation import formal_multi_plate_gate
    result = dict(schema=SCHEMA, total_multi_invocations=0, verified_multi_invocations=0,
                  gt_plates=None, tp=None, fp=None, fn=None, text_gt_reads=0,
                  text_exact=0, text_errors=0, text_gt_characters=0, rows=[],
                  formal_status="NOT_EVALUABLE", formal_tp=None, formal_fp=None, formal_fn=None,
                  formal_evaluable_invocations=0, formal_issues=[],
                  operator_diagnostic=dict(plate_count=0, linked_predictions=0,
                                           unmatched_plate_annotations=0,
                                           marked_not_plate=0),
                  status="NOT_RUN",
                  scope="mapowanie decyzji operatora per wywołanie MT (nie TP/FP/FN naukowe)")
    for gid, group in session.mt_invocations.items():
        if not is_multi(session, gid):
            continue
        result["total_multi_invocations"] += 1
        inv = inventory(session, gid)
        formal = formal_multi_plate_gate(session, gid, inv)
        result["formal_issues"].append(dict(mt_invocation_id=gid,
                                           status=formal["status"],
                                           reason_codes=formal["reasons"],
                                           roi=formal["invocation"]))
        issues = group_issues(session, gid)
        if inv is None or issues:
            continue
        plates = inv["plates"]
        map_aid = {p.get("attempt_id"): p["plate_id"] for p in plates if p.get("attempt_id")}
        diagnostic = result["operator_diagnostic"]
        diagnostic["plate_count"] += len(plates)
        diagnostic["linked_predictions"] += len(map_aid)
        diagnostic["unmatched_plate_annotations"] += len(plates) - len(map_aid)
        diagnostic["marked_not_plate"] += sum(
            selected_visual_state(session.review.attempt_annotations.get(aid, {})) == "not_plate"
            for aid in _records(group))
        result["verified_multi_invocations"] += 1
        for p in plates:
            aid = str(p.get("attempt_id") or "")
            row = dict(mt_invocation_id=gid, gt_plate_id=p["plate_id"], attempt_id=aid,
                       operator_mapping="linked_to_mt" if aid else "no_mt_link",
                       mt_outcome=None, formal_mt_outcome="NOT_EVALUABLE",
                       gt_text=p.get("gt_text") or "", bbox_gt_manual=p.get("bbox_mt_px"),
                       gt_geometry_source="MT_INPUT_LEGACY_ASSISTED")
            if aid and _mz_ran(session, aid) and p.get("gt_text"):
                from .mobile_human_review import align_plate_text
                sample_ids = session.sample_ids_by_attempt.get(aid, ())
                sample = next((session.samples[i] for i in sample_ids if i in session.samples), {})
                attempt = session.attempts[aid]
                prediction = str(sample.get("prediction", attempt.get("prediction", "")) or "")
                alignment = align_plate_text(str(p["gt_text"]), prediction)
                row.update(mz_prediction=prediction, mz_exact=alignment.exact_match,
                           mz_cer=alignment.cer, mz_comparison_provenance="ASSISTED_DIAGNOSTIC")
                result["text_gt_reads"] += 1
                result["text_exact"] += int(alignment.exact_match)
                result["text_gt_characters"] += len(alignment.ground_truth)
                result["text_errors"] += alignment.edit_distance
            result["rows"].append(row)
    if result["verified_multi_invocations"]:
        result["status"] = ("VERIFIED_SUBSET" if result["verified_multi_invocations"] <
                            result["total_multi_invocations"] else "OPERATOR_VERIFIED")
    elif result["total_multi_invocations"]:
        result["status"] = "PENDING"
    return result
