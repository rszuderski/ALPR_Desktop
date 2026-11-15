"GT-assisted correction layer for Z3/PZ2."

from __future__ import annotations

import copy
import hashlib
import json
import math
import tkinter as tk

from ..plate_ground_truth import normalize_plate_ground_truth_text
from .z3_gt_box_policy import limit_boxes_to_gt
from .z3_gt_contract import (
    canonical_raw_detection_hash,
    revision_ids_from_data,
)


GT_ASSIST_SCHEMA = "alpr.pz2.gt_assist.v1"
GT_ASSIST_SOURCE = "gt_assisted"
GT_ASSIST_CONFIRMED_SOURCE = "gt_assist_confirmed"


def apply_live_gt_assist(host, data: dict | None, *, prepare: bool = False) -> dict:
    """Label a working annotation; select geometry only at explicit preparation."""
    if not isinstance(data, dict):
        return {"changed": False, "reason": "no_plate"}
    state = data.get("review_state") or {}
    if state.get("status") != "in_progress":
        return {"changed": False, "reason": "not_editing"}
    gt = normalize_plate_ground_truth_text(host._get_preview_ground_truth_text(data))
    chars = data.get("characters") or []
    box_limit = None
    if prepare:
        chars, box_limit = limit_boxes_to_gt(host, data, chars)
    if box_limit:
        host._update_preview_plate_layout_metadata(data, chars)
        chars = host._annotate_preview_character_reading_positions(chars, data=data)
        data["characters"] = chars
        data["gt_box_limit"] = copy.deepcopy(box_limit)
    result = {"schema": "alpr.pz2.live_gt_assist.v1", "ground_truth_text": gt,
              "source_gt_revision_ids": revision_ids_from_data(data, "source_gt_revision_ids", "source_gt_revision_id"),
              "source_gt_hash": data.get("source_gt_hash"), "box_count": len(chars), "changed": bool(box_limit)}
    reason = "ready"
    if not gt:
        reason = "missing_gt"
    elif len(chars) != len(gt):
        reason = "box_count_mismatch"
    else:
        try:
            for rec in chars:
                x1, y1, x2, y2 = map(float, rec["bbox"][:4])
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                    raise ValueError("invalid box")
            if host._preview_layout_separator_conflicts_with_chars(data, chars):
                reason = "layout_conflict"
            elif str(data.get("plate_layout", "")) == "two_row_candidate" and not data.get("plate_layout_override"):
                reason = "layout_uncertain"
        except (TypeError, ValueError, KeyError):
            reason = "invalid_geometry"
    if reason == "ready":
        ordered = host._sort_character_records_by_x(chars, data=data)
        for rec, symbol in zip(ordered, gt):
            sign_source = str(rec.get("sign_source") or "").strip().lower().replace("-", "_")
            symbol_source = str(rec.get("symbol_source") or "").strip().lower().replace("-", "_")
            symbol_method = str(rec.get("symbol_method") or "").strip().lower().replace("-", "_")
            manual_sign = (
                sign_source in {"manual_sign", "manual", "local_manual", "cvat_manual", "preview_editor"}
                or (
                    not sign_source
                    and (
                        symbol_source in {"manual", "manual_sign", "local_manual", "cvat_manual", "preview_editor"}
                        or symbol_method in {"manual", "manual_sign", "local_manual", "cvat_manual", "preview_editor"}
                    )
                )
            )
            if manual_sign:
                continue
            if str(rec.get("character") or "") != symbol:
                rec.setdefault("gt_assist_original_character", rec.get("character", ""))
                rec.setdefault("gt_assist_original_sign_source", rec.get("sign_source", ""))
                rec["character"] = symbol
                rec["correction_source"] = GT_ASSIST_SOURCE
                rec["sign_source"] = GT_ASSIST_SOURCE
                result["changed"] = True
        if result["changed"]:
            data["correction_source"] = GT_ASSIST_SOURCE
    result["reason"] = reason
    data["live_gt_assist"] = {key: value for key, value in result.items() if key != "changed"}
    return result


def get_live_gt_assist_presentation(host, data: dict | None = None) -> dict:
    if data is None:
        data = host._get_preview_active_data(create=False)
    data = data if isinstance(data, dict) else {}
    gt = normalize_plate_ground_truth_text(data.get("ground_truth_text"))
    if not gt:
        getter = getattr(host, "_get_preview_ground_truth_text", None)
        gt = getter(data) if callable(getter) else ""
    chars = data.get("characters") or []
    editing = (data.get("review_state") or {}).get("status") == "in_progress"
    text, tone = "Asysta GT: aktywna podczas korekty", "info"
    if not gt:
        text, tone = "Numer tablicy: brak — powstanie po zatwierdzeniu znaków", "info"
    elif not editing:
        text = f"GT: {gt} · asysta aktywna podczas korekty"
    elif len(chars) != len(gt):
        action = f"Brakuje {len(gt) - len(chars)} ramek" if len(chars) < len(gt) else f"Usuń {len(chars) - len(gt)} nadmiarowych ramek"
        text, tone = f"GT: {gt} · ramki {len(chars)}/{len(gt)}. {action}.", "warning"
    elif (data.get("live_gt_assist") or {}).get("reason") in {"invalid_geometry", "layout_conflict", "layout_uncertain"}:
        text, tone = f"GT: {gt} · sprawdź geometrię i układ ramek", "warning"
    else:
        recovery = data.get("gt_geometry_recovery") or {}
        if recovery.get("generated_box_count"):
            text = f"GT: {gt} · ramki odzyskane z obrazu — sprawdź ich geometrię"
        else:
            text = f"GT: {gt} · asysta znaków aktywna"
    return {"text": text, "tone": tone, "status": "automatic", "can_accept": False, "can_reject": False}


def _canonical_raw_fingerprint(
    raw_detection: dict | None,
) -> str:
    # Backward-compatible private alias used by older callers/tests.
    return canonical_raw_detection_hash(raw_detection)


def _characters_text(host, records, data=None) -> str:
    try:
        return str(
            host._characters_to_text(records, data=data)
            or ""
        ).strip().upper()
    except TypeError:
        try:
            return str(
                host._characters_to_text(records) or ""
            ).strip().upper()
        except Exception:
            return ""
    except Exception:
        return ""


def _serialize(host, records, data=None) -> list[dict]:
    try:
        return list(
            host._serialize_character_records(
                records,
                fusion_strategy="gt_assist",
                fusion_details={
                    "source": GT_ASSIST_SOURCE,
                },
                data=data,
            )
            or []
        )
    except Exception:
        result = []
        for record in list(records or []):
            if isinstance(record, dict):
                result.append(copy.deepcopy(record))
        return result


def build_gt_assist_suggestion(host, data: dict | None) -> dict:
    source_data = data if isinstance(data, dict) else {}
    raw_detection = source_data.get("raw_detection")
    if not isinstance(raw_detection, dict):
        return {
            "schema": GT_ASSIST_SCHEMA,
            "status": "unavailable",
            "reason": "missing_raw_detection",
            "operations": [],
        }

    raw_contract = str(
        raw_detection.get("contract") or ""
    ).strip()
    if raw_contract != "gt_blind.v1":
        return {
            "schema": GT_ASSIST_SCHEMA,
            "status": "unavailable",
            "reason": "raw_not_gt_blind",
            "source_raw_contract": raw_contract,
            "operations": [],
        }

    raw_chars = raw_detection.get("characters", [])
    if not isinstance(raw_chars, list):
        raw_chars = []
    raw_chars = copy.deepcopy(raw_chars)

    gt_text = normalize_plate_ground_truth_text(
        source_data.get("ground_truth_text")
    )
    if not gt_text:
        try:
            gt_text = normalize_plate_ground_truth_text(
                host._get_preview_ground_truth_text(source_data)
            )
        except Exception:
            gt_text = ""

    raw_text = _characters_text(
        host,
        raw_chars,
        data=source_data,
    )
    raw_fingerprint = _canonical_raw_fingerprint(
        raw_detection
    )
    gt_revision_ids = revision_ids_from_data(
        source_data,
        "source_gt_revision_ids",
        "source_gt_revision_id",
    )
    gt_revision_id = (
        gt_revision_ids[0]
        if len(gt_revision_ids) == 1
        else None
    )

    base = {
        "schema": GT_ASSIST_SCHEMA,
        "source": GT_ASSIST_SOURCE,
        "source_raw_contract": raw_contract,
        "source_raw_fingerprint": raw_fingerprint,
        "source_gt_hash": str(
            source_data.get("source_gt_hash") or ""
        ).strip() or None,
        "source_gt_revision_id": gt_revision_id,
        "source_gt_revision_ids": list(gt_revision_ids),
        "source_raw_result_hash": str(
            raw_detection.get("result_hash") or raw_fingerprint
        ).strip(),
        "source_raw_prediction_text": raw_text,
        "ground_truth_text": gt_text or None,
        "operations": [],
        "accepted": False,
    }

    if not gt_text:
        return {
            **base,
            "status": "unavailable",
            "reason": "missing_ground_truth",
            "suggestion_text": raw_text,
            "characters": raw_chars,
        }

    if raw_text == gt_text and len(raw_chars) == len(gt_text):
        return {
            **base,
            "status": "not_needed",
            "reason": "raw_exact",
            "suggestion_text": raw_text,
            "characters": raw_chars,
            "exact_text_match": True,
            "exact_count_match": True,
        }

    current = copy.deepcopy(raw_chars)
    operations: list[dict] = []

    if len(current) > len(gt_text):
        try:
            fitted, details = host._fit_detection_count_to_truths(
                current,
                [gt_text],
            )
        except Exception:
            fitted, details = current, None
        if (
            isinstance(details, dict)
            and int(details.get("trimmed_extra_boxes", 0) or 0) > 0
        ):
            current = copy.deepcopy(list(fitted or []))
            operations.append(
                {
                    "type": "trim_extra_boxes",
                    "details": copy.deepcopy(details),
                }
            )

    yolo_candidates = source_data.get("yolo_box_detections") if "yolo_box_detections" in source_data else (
        source_data.get("yolo_nms_detections")
        or source_data.get("yolo_detections")
        or source_data.get("yolo_raw_detections")
        or []
    )
    if not isinstance(yolo_candidates, list):
        yolo_candidates = []

    current_text = _characters_text(
        host,
        current,
        data=source_data,
    )

    if (
        current_text != gt_text
        and yolo_candidates
        and len(yolo_candidates) >= len(gt_text)
    ):
        try:
            yolo_fitted, yolo_details = (
                host._fit_detection_count_to_truths(
                    yolo_candidates,
                    [gt_text],
                )
            )
        except Exception:
            yolo_fitted, yolo_details = yolo_candidates, None

        yolo_fitted = list(yolo_fitted or [])
        yolo_text = _characters_text(
            host,
            yolo_fitted,
            data=source_data,
        )
        if (
            len(yolo_fitted) == len(gt_text)
            and yolo_text == gt_text
        ):
            current = copy.deepcopy(yolo_fitted)
            operations.append(
                {
                    "type": "select_yolo_gt_match",
                    "details": copy.deepcopy(
                        yolo_details
                        if isinstance(yolo_details, dict)
                        else {}
                    ),
                }
            )
            current_text = yolo_text

    if (
        current_text != gt_text
        and len(current) == len(gt_text)
        and yolo_candidates
    ):
        try:
            repaired, repair_details = (
                host._repair_ocr_with_yolo_boxes(
                    current,
                    yolo_candidates,
                    [gt_text],
                    max_mismatch_count=len(gt_text),
                )
            )
        except Exception:
            repaired, repair_details = None, None

        if repaired:
            repaired_text = _characters_text(
                host,
                repaired,
                data=source_data,
            )
            if repaired_text == gt_text:
                current = list(repaired)
                operations.append(
                    {
                        "type": "repair_symbols_with_yolo",
                        "details": copy.deepcopy(
                            repair_details
                            if isinstance(repair_details, dict)
                            else {}
                        ),
                    }
                )
                current_text = repaired_text

    serialized = _serialize(
        host,
        current,
        data=source_data,
    )
    suggestion_text = _characters_text(
        host,
        serialized,
        data=source_data,
    )
    exact_text_match = bool(suggestion_text == gt_text)
    exact_count_match = bool(
        len(serialized) == len(gt_text)
    )

    if operations:
        status = "suggested"
        reason = (
            "exact_gt_assisted"
            if exact_text_match and exact_count_match
            else "partial_gt_assist"
        )
    else:
        status = "no_suggestion"
        reason = "no_safe_gt_assist"

    return {
        **base,
        "status": status,
        "reason": reason,
        "operations": operations,
        "suggestion_text": suggestion_text,
        "characters": serialized,
        "exact_text_match": exact_text_match,
        "exact_count_match": exact_count_match,
    }


def store_gt_assist_suggestion(
    host,
    data: dict | None,
) -> dict:
    if not isinstance(data, dict):
        return {
            "schema": GT_ASSIST_SCHEMA,
            "status": "unavailable",
            "reason": "invalid_metadata",
            "operations": [],
        }
    suggestion = build_gt_assist_suggestion(host, data)
    data["gt_assist"] = copy.deepcopy(suggestion)
    return suggestion


def gt_assist_is_current(
    host,
    data: dict | None,
) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "invalid_metadata"

    assist = data.get("gt_assist")
    raw = data.get("raw_detection")
    if not isinstance(assist, dict):
        return False, "missing_gt_assist"
    if not isinstance(raw, dict):
        return False, "missing_raw_detection"

    expected_raw = str(
        assist.get("source_raw_fingerprint") or ""
    ).strip()
    current_raw = _canonical_raw_fingerprint(raw)
    if not expected_raw or expected_raw != current_raw:
        return False, "stale_raw_detection"

    expected_gt = normalize_plate_ground_truth_text(
        assist.get("ground_truth_text")
    )
    current_gt = normalize_plate_ground_truth_text(
        data.get("ground_truth_text")
    )
    if not current_gt:
        try:
            current_gt = normalize_plate_ground_truth_text(
                host._get_preview_ground_truth_text(data)
            )
        except Exception:
            current_gt = ""

    if expected_gt != current_gt:
        return False, "stale_ground_truth"

    expected_gt_hash = str(
        assist.get("source_gt_hash") or ""
    ).strip()
    current_gt_hash = str(
        data.get("source_gt_hash") or ""
    ).strip()
    if (
        expected_gt_hash
        and current_gt_hash
        and expected_gt_hash != current_gt_hash
    ):
        return False, "stale_gt_hash"

    expected_revision_ids = revision_ids_from_data(
        assist,
        "source_gt_revision_ids",
        "source_gt_revision_id",
    )
    current_revision_ids = revision_ids_from_data(
        data,
        "source_gt_revision_ids",
        "source_gt_revision_id",
    )
    if (
        expected_revision_ids
        and expected_revision_ids != current_revision_ids
    ):
        return False, "stale_gt_revision"

    return True, "current"


def accept_gt_assist_suggestion(
    host,
    data: dict | None,
) -> dict:
    if not isinstance(data, dict):
        return {"ok": False, "reason": "invalid_metadata"}

    assist = data.get("gt_assist")
    if not isinstance(assist, dict):
        return {"ok": False, "reason": "missing_gt_assist"}
    if str(assist.get("status") or "") != "suggested":
        return {"ok": False, "reason": "no_pending_suggestion"}

    current, reason = gt_assist_is_current(host, data)
    if not current:
        return {"ok": False, "reason": reason}

    suggestion_chars = assist.get("characters", [])
    if not isinstance(suggestion_chars, list):
        return {"ok": False, "reason": "invalid_suggestion"}

    accepted = copy.deepcopy(suggestion_chars)
    for record in accepted:
        if isinstance(record, dict):
            record["correction_source"] = (
                GT_ASSIST_CONFIRMED_SOURCE
            )

    accepted = _serialize(
        host,
        accepted,
        data=data,
    )
    try:
        host._mark_review_edit_started(data)
    except Exception:
        pass
    data["characters"] = accepted
    data["correction_source"] = GT_ASSIST_CONFIRMED_SOURCE
    data["review_source"] = GT_ASSIST_CONFIRMED_SOURCE

    assist["accepted"] = True
    assist["status"] = "accepted"
    assist["accepted_text"] = _characters_text(
        host,
        accepted,
        data=data,
    )
    data["gt_assist"] = assist

    try:
        data["status"] = host._derive_preview_status_from_data(
            dict(data),
            accepted,
        )
    except Exception:
        pass

    return {
        "ok": True,
        "reason": "accepted",
        "characters": accepted,
        "text": assist.get("accepted_text", ""),
        "status": str(data.get("status") or ""),
    }


def reject_gt_assist_suggestion(
    host,
    data: dict | None,
) -> dict:
    if not isinstance(data, dict):
        return {"ok": False, "reason": "invalid_metadata"}
    assist = data.get("gt_assist")
    if not isinstance(assist, dict):
        return {"ok": False, "reason": "missing_gt_assist"}

    assist["accepted"] = False
    assist["status"] = "rejected"
    data["gt_assist"] = assist
    return {"ok": True, "reason": "rejected"}

def get_gt_assist_presentation(
    host,
    data: dict | None = None,
) -> dict:
    source_data = data
    if source_data is None:
        try:
            source_data = host._get_preview_active_data(
                create=False
            )
        except Exception:
            source_data = None

    if not isinstance(source_data, dict):
        return {
            "text": "GT Assist: brak aktywnej tablicy",
            "tone": "muted",
            "can_accept": False,
            "can_reject": False,
            "status": "none",
        }

    assist = source_data.get("gt_assist")
    if not isinstance(assist, dict):
        return {
            "text": "GT Assist: brak sugestii",
            "tone": "muted",
            "can_accept": False,
            "can_reject": False,
            "status": "none",
        }

    status = str(assist.get("status") or "").strip().lower()
    raw_text = str(
        assist.get("source_raw_prediction_text") or ""
    ).strip()
    suggestion_text = str(
        assist.get("suggestion_text") or ""
    ).strip()
    reason = str(assist.get("reason") or "").strip()

    if status == "suggested":
        current, current_reason = gt_assist_is_current(
            host,
            source_data,
        )
        if current:
            return {
                "text": (
                    f"GT Assist: {raw_text or '—'} → "
                    f"{suggestion_text or '—'}"
                ),
                "tone": "warning",
                "can_accept": True,
                "can_reject": True,
                "status": "suggested",
            }
        return {
            "text": (
                "GT Assist: sugestia nieaktualna "
                f"({current_reason})"
            ),
            "tone": "warning",
            "can_accept": False,
            "can_reject": True,
            "status": "stale",
        }

    if status == "accepted":
        return {
            "text": (
                "GT Assist: zastosowany"
                + (
                    f" → {assist.get('accepted_text')}"
                    if str(assist.get("accepted_text") or "").strip()
                    else ""
                )
            ),
            "tone": "success",
            "can_accept": False,
            "can_reject": False,
            "status": "accepted",
        }

    if status == "rejected":
        return {
            "text": "GT Assist: odrzucony",
            "tone": "muted",
            "can_accept": False,
            "can_reject": False,
            "status": "rejected",
        }

    if status == "not_needed":
        return {
            "text": "GT Assist: RAW zgodny z GT",
            "tone": "success",
            "can_accept": False,
            "can_reject": False,
            "status": "not_needed",
        }

    if status == "no_suggestion":
        return {
            "text": "GT Assist: brak bezpiecznej korekty",
            "tone": "muted",
            "can_accept": False,
            "can_reject": False,
            "status": "no_suggestion",
        }

    if status == "unavailable":
        unavailable_text = "GT Assist: niedostępny"
        if reason:
            unavailable_text += f" ({reason})"
        return {
            "text": unavailable_text,
            "tone": "muted",
            "can_accept": False,
            "can_reject": False,
            "status": "unavailable",
        }

    return {
        "text": f"GT Assist: {status or 'brak'}",
        "tone": "muted",
        "can_accept": False,
        "can_reject": False,
        "status": status or "none",
    }


def refresh_gt_assist_controls(host) -> dict:
    presentation = get_live_gt_assist_presentation(host)

    busy = bool(
        getattr(host, "fast_test_running", False)
        or getattr(host, "is_processing", False)
    )
    accept_enabled = bool(
        presentation.get("can_accept")
        and not busy
    )
    reject_enabled = bool(
        presentation.get("can_reject")
        and not busy
    )

    accept_btn = getattr(
        host,
        "btn_gt_assist_apply",
        None,
    )
    reject_btn = getattr(
        host,
        "btn_gt_assist_reject",
        None,
    )
    status_lbl = getattr(
        host,
        "gt_assist_status_lbl",
        None,
    )

    for widget, enabled in (
        (accept_btn, accept_enabled),
        (reject_btn, reject_enabled),
    ):
        if widget is None:
            continue
        try:
            widget.configure(
                state=(tk.NORMAL if enabled else tk.DISABLED)
            )
        except Exception:
            pass

    if status_lbl is not None:
        text = str(presentation.get("text") or "")
        tone = str(presentation.get("tone") or "muted")
        try:
            setter = getattr(
                host,
                "_set_inline_status_label_state",
                None,
            )
            if callable(setter):
                setter(
                    status_lbl,
                    text=text,
                    tone=tone,
                    emphasis=False,
                )
            else:
                status_lbl.configure(text=text)
        except Exception:
            pass

    return presentation


def _persist_gt_assist_review_change(
    host,
    *,
    plate_id: str,
    message: str,
    tone: str,
) -> None:
    try:
        host._persist_preview_metadata(
            success_message=None,
            refresh_list=False,
            sync_access=False,
        )
    except Exception:
        pass

    try:
        host._refresh_preview_listbox_row(plate_id)
    except Exception:
        pass

    try:
        host._on_preview_select(None)
    except Exception:
        pass

    try:
        host._update_preview_edit_status(
            message,
            tone=tone,
        )
    except TypeError:
        try:
            host._update_preview_edit_status(message)
        except Exception:
            pass
    except Exception:
        pass

    try:
        refresh_gt_assist_controls(host)
    except Exception:
        pass


def apply_active_gt_assist(host) -> dict:
    try:
        data = host._get_preview_active_data(
            create=False
        )
    except Exception:
        data = None

    plate_id = str(
        getattr(host, "_preview_active_pid", "")
        or ""
    ).strip()

    if not isinstance(data, dict) or not plate_id:
        return {
            "ok": False,
            "reason": "missing_active_plate",
        }

    try:
        host._push_preview_history_snapshot(plate_id)
    except Exception:
        pass

    result = accept_gt_assist_suggestion(
        host,
        data,
    )

    if not result.get("ok"):
        try:
            host._update_preview_edit_status(
                "Nie można zastosować GT Assist: "
                + str(result.get("reason") or "nieznany powód"),
                tone="warning",
            )
        except Exception:
            pass
        refresh_gt_assist_controls(host)
        return result

    _persist_gt_assist_review_change(
        host,
        plate_id=plate_id,
        message=(
            "Zastosowano GT Assist jako jawnie "
            "potwierdzoną korektę review."
        ),
        tone="success",
    )
    return result


def reject_active_gt_assist(host) -> dict:
    try:
        data = host._get_preview_active_data(
            create=False
        )
    except Exception:
        data = None

    plate_id = str(
        getattr(host, "_preview_active_pid", "")
        or ""
    ).strip()

    if not isinstance(data, dict) or not plate_id:
        return {
            "ok": False,
            "reason": "missing_active_plate",
        }

    try:
        host._push_preview_history_snapshot(plate_id)
    except Exception:
        pass

    result = reject_gt_assist_suggestion(
        host,
        data,
    )
    if not result.get("ok"):
        refresh_gt_assist_controls(host)
        return result

    _persist_gt_assist_review_change(
        host,
        plate_id=plate_id,
        message="Odrzucono sugestię GT Assist.",
        tone="muted",
    )
    return result
