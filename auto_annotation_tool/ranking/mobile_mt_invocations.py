"""Validated MT backend invocations, independent of subjects and OCR predictions."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math


DETECTION_STATUSES = {"VALID_QUAD", "DETECTION_INVALID_QUAD"}
EXECUTED_STATUSES = DETECTION_STATUSES | {"NO_DETECTION"}
ROI_FIELDS = ("roi_left", "roi_top", "roi_right", "roi_bottom", "roi_geometry")
INPUT_FIELDS = ("input_width", "input_height", "input_scale", "input_pad_x", "input_pad_y", "input_geometry")
IDENTITY_FIELDS = ("session_id", "scene_generation", "visual_epoch", "camera_transform_generation",
                   "source_sequence", "source_timestamp_nanos", "roi_policy") + ROI_FIELDS + INPUT_FIELDS


def true(value):
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _canonical(value):
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return tuple(sorted((key, _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    text = str(value).strip()
    if text.startswith(("[", "{")):
        return _canonical(json.loads(text))
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        raise ValueError("Nieprawidłowa wartość liczbowa w geometrii lub tożsamości MT.")
    return number


def _integer(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != int(number) or number < 0:
            raise ValueError
        return int(number)
    except (ValueError, InvalidOperation, OverflowError):
        raise ValueError(f"Nieprawidłowy indeks lub liczba detekcji MT: {value}") from None


def detection_box_on_mt_input(row) -> tuple[float, float, float, float] | None:
    """Map a source-frame box through the recorded ROI/letterbox, without guessing."""
    fields = ("plate_left", "plate_top", "plate_right", "plate_bottom", "roi_left", "roi_top",
              "input_width", "input_height", "input_scale", "input_pad_x", "input_pad_y")
    try:
        if any(isinstance(row[name], bool) for name in fields):
            return None
        left, top, right, bottom, roi_x, roi_y, width, height, scale, pad_x, pad_y = (
            float(row[name]) for name in fields)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom, roi_x, roi_y,
                                                  width, height, scale, pad_x, pad_y)):
        return None
    if width <= 0 or height <= 0 or scale <= 0:
        return None
    box = ((left - roi_x) * scale + pad_x, (top - roi_y) * scale + pad_y,
           (right - roi_x) * scale + pad_x, (bottom - roi_y) * scale + pad_y)
    if not all(math.isfinite(value) for value in box):
        return None
    left, top, right, bottom = (max(0.0, min(limit, value)) for value, limit in zip(box, (width, height, width, height)))
    return (left, top, right, bottom) if right > left and bottom > top else None


@dataclass(frozen=True)
class MtInvocationGroup:
    mt_invocation_id: str
    session_id: str
    scene_generation: object
    subject_keys: frozenset[str]
    source_sequence: object
    source_timestamp_nanos: object
    roi_geometry: dict
    input_geometry: dict
    evidence_entries: tuple[str, ...]
    records: tuple[dict, ...]
    detection_count: int
    cancelled: bool
    executed: bool
    legacy_identity: bool
    execution_failed: bool = False
    execution_errors: tuple[str, ...] = ()

    @property
    def evidence_entry(self):
        return self.evidence_entries[0] if self.evidence_entries else ""


def group_mt_invocations(attempts: dict[str, dict], session_id: str) -> dict[str, MtInvocationGroup]:
    buckets = {}
    for row in attempts.values():
        key = str(row.get("mt_invocation_id") or row["id"])
        buckets.setdefault(key, []).append(row)
    groups = {}
    for key, rows in buckets.items():
        for field in IDENTITY_FIELDS:
            signatures = {str(row.get(field) or "") if field == "session_id" else _canonical(row.get(field)) for row in rows}
            if len(signatures) != 1:
                raise ValueError(f"Sprzeczne {field} w wywołaniu MT {key}.")
        detected = [row for row in rows if row.get("mt_status") in DETECTION_STATUSES]
        counts = [row.get("mt_detection_count") for row in rows]
        indices = [row.get("mt_detection_index") for row in rows]
        has_counts = any(value not in (None, "") for value in counts)
        has_indices = any(value not in (None, "") for value in indices)
        if has_counts or has_indices:
            if any(value in (None, "") for value in counts):
                raise ValueError(f"Brak liczby detekcji w części wywołania MT {key}.")
            parsed_counts = {_integer(value) for value in counts}
            if len(parsed_counts) != 1:
                raise ValueError(f"Sprzeczne liczby detekcji MT {key}.")
            count = parsed_counts.pop()
            if count == 0:
                if detected or has_indices or len(rows) != 1:
                    raise ValueError(f"Nieprawidłowy zapis wywołania MT bez detekcji: {key}.")
            else:
                if len(detected) != count or len(rows) != count or any(value in (None, "") for value in indices):
                    raise ValueError(f"Niepełny zestaw detekcji wywołania MT {key}.")
                parsed_indices = [_integer(value) for value in indices]
                if sorted(parsed_indices) != list(range(count)):
                    raise ValueError(f"Indeksy detekcji MT {key} muszą być unikalne i należeć do 0..N-1.")
        if detected and any(row.get("mt_status") == "NO_DETECTION" for row in rows):
            raise ValueError(f"Wywołanie MT {key} jednocześnie zawiera detekcje i NO_DETECTION.")
        # Android may save the same input under different child-specific names.
        # Preserve every explicit input reference, never choose a child crop as
        # evidence for counting how many plates were visible in the full input.
        entries = []
        for row in rows:
            entry = row.get("mt_input_evidence_entry")
            if not entry and "mt_input_evidence_entry" not in row:
                candidate = str(row.get("evidence_entry") or "")
                entry = candidate if candidate.startswith("samples/evidence/") else ""
            if entry and entry not in entries:
                entries.append(str(entry))
        first = rows[0]  # All shared geometry/identity fields have been validated.
        errors = tuple(dict.fromkeys(str(row.get("execution_error") or "").strip() for row in rows
                                     if str(row.get("execution_error") or "").strip()))
        groups[key] = MtInvocationGroup(
            key, str(first.get("session_id") or session_id), first.get("scene_generation"),
            frozenset(row["subject_key"] for row in rows), first.get("source_sequence"), first.get("source_timestamp_nanos"),
            {name: first[name] for name in ROI_FIELDS if name in first},
            {name: first[name] for name in INPUT_FIELDS if name in first}, tuple(entries), tuple(rows), len(detected),
            any(true(row.get("stale_or_cancelled")) for row in rows),
            any(true(row.get("mt_executed")) if row.get("mt_executed") not in (None, "") else row.get("mt_status") in EXECUTED_STATUSES for row in rows),
            not any(row.get("mt_invocation_id") for row in rows), bool(errors), errors)
    return groups


def calculate_mt_invocations(session):
    names = ("evaluable_mt_invocations", "mt_successful_invocations", "mt_no_detection_invocations",
             "mt_invalid_quad_invocations", "mt_multi_plate_invocations", "mt_uncertain_invocations", "mt_false_detections",
             "mt_execution_error_invocations")
    counts = dict.fromkeys(names, 0)
    results = []
    false_sample_attempts = {session.samples[key].get("attempt_id") for key, decision in session.review.sample_annotations.items()
                             if decision.get("is_plate") is False}
    for key, group in session.mt_invocations.items():
        annotation = session.invocation_annotation(key)
        visible = annotation.get("visible_plate_count")
        evidence = any(entry in session.entry_names for entry in group.evidence_entries)
        active = group.executed and not group.cancelled and not group.execution_failed
        included = active and annotation.get("evaluable") is True and evidence and visible == "one"
        valid, invalid, false = False, False, 0
        for row in group.records:
            child = session.review.attempt_annotations.get(row["id"], {})
            is_plate = False if row["id"] in false_sample_attempts else child.get("is_plate")
            if visible == "none":
                is_plate = False
            elif (is_plate is None and session.review.mt_review_policy == "legacy_visibility_v1"
                  and key not in session.review.invocation_annotations and child.get("plate_visibility") == "visible"):
                is_plate = True
            if active and row.get("mt_status") in DETECTION_STATUSES and is_plate is False:
                false += 1
            if active and child.get("evaluable") is not False and is_plate is True:
                valid |= row.get("mt_status") == "VALID_QUAD"
                invalid |= row.get("mt_status") == "DETECTION_INVALID_QUAD"
        no_detection = not valid and any(row.get("mt_status") == "NO_DETECTION" for row in group.records)
        outcome = "success" if valid else "no_detection" if no_detection else "invalid_quad" if invalid else "no_valid_localization"
        if group.cancelled:
            outcome = "cancelled"
        elif not group.executed:
            outcome = "not_executed"
        elif group.execution_failed:
            outcome = "execution_error"
            counts["mt_execution_error_invocations"] += 1
        counts["mt_false_detections"] += false
        counts["mt_multi_plate_invocations"] += active and visible == "multiple"
        counts["mt_uncertain_invocations"] += active and visible == "uncertain"
        if included:
            counts["evaluable_mt_invocations"] += 1
            if valid:
                counts["mt_successful_invocations"] += 1
            elif no_detection:
                counts["mt_no_detection_invocations"] += 1
            elif invalid:
                counts["mt_invalid_quad_invocations"] += 1
        results.append({"mt_invocation_id": key, "subject_keys": sorted(group.subject_keys), "records": [row["id"] for row in group.records],
                        "detection_count": group.detection_count, "cancelled": group.cancelled, "evidence_available": evidence,
                        "executed": group.executed, "execution_failed": group.execution_failed,
                        "execution_errors": list(group.execution_errors),
                        "visible_plate_count": visible, "included": bool(included), "outcome": outcome,
                        "false_detections": false, "decision_source": annotation.get("decision_source", "operator")})
    denominator = counts["evaluable_mt_invocations"]
    counts["mt_localization_success_rate"] = counts["mt_successful_invocations"] / denominator if denominator else None
    if not session.attempts_available:
        counts = dict.fromkeys(counts, None)
    for old, new in (("evaluable_mt_attempts", "evaluable_mt_invocations"), ("mt_valid_localizations", "mt_successful_invocations"),
                     ("mt_no_detections", "mt_no_detection_invocations"), ("mt_invalid_quads", "mt_invalid_quad_invocations")):
        counts[old] = counts[new]
    return counts, results
