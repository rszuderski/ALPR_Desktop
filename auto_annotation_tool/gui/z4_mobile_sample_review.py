# Z4MOBREV032_FIX02_CURRENT_SHA_CANVAS_20261023
# Z4MOBREV007_POSE_QUALITY_WITHOUT_GT_CTA_20261016
# Z4MOBREV006_STABLE_VIEW_AUTO_GT_20261016
# Z4MOBREV005_POSE_POLYGON_OVERLAY_20261016
# Z4MOBREV004_OPERATOR_FIRST_UI_20261016
# Z4MOBREV003_VIEWPORT_ZOOM_PAN_20261016
# Z4MOBREV002_FIX01_DECISIONS_UI_20261016
# Z4MOBREV001_SIMPLE_REVIEW_20261016
"""Session → subject → evidence review. File IO and metrics run off the Tk thread."""
from __future__ import annotations

import csv
import json
import math
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image, ImageTk

from .mobile_science_summary_ui import MobileScienceSummaryPanel
from .mobile_canvas_editor import CanvasReviewEditor
from .mobile_review_export_layout import (
    validate_export_destination, next_export_path, reserve_export_path,
    finalize_export_manifest, mark_export_incomplete,
)

from ..rectification.polygon_validator import PolygonValidator

from ..ranking.mobile_human_review import MobileReviewSession, align_plate_text, normalize_registration
from ..ranking.mobile_mt_invocations import DETECTION_STATUSES, detection_box_on_mt_input
from ..ranking.mobile_pose_metrics import POSE_GT_SCHEMA, compare_mt_pose_to_gt, validate_mt_quad
from ..ranking.mobile_pose_correction import (
    CORRECTION_SCHEMA, correction_for_attempt, make_correction,
    hit_target, move_polygon, moved,
)
from ..ranking.mobile_package_experiments import MobilePackageExperimentStore, read_mobile_report_bundle
from ..ranking.mobile_source_frame_roi import (
    source_invocation_context, verified_invocation_detections,
    quad_contains_point, read_mobile_source_frame,
)
from ..ranking.mobile_thesis_report import export_thesis_bundle
from ..ranking.mobile_verification_policy import (
    selected_visual_state, visual_decision, verification_progress,
    subject_verification_issues, observation_verification_cells,
)
from ..ranking.mobile_multi_plate_verification import (
    is_multi, inventory, group_issues, set_count, assign_attempt,
    set_plate_text, set_missed_bbox, clear_missed_bbox, multi_scientific_summary,
    _records as multi_detection_records,
)
from ..ranking.mobile_scientific_review import (
    make_scientific_summary, display_scientific_summary,
    session_conditions_text, write_scientific_companion,
)

REVIEW_LABELS = {"NOT_STARTED": "Nie rozpoczęto", "IN_PROGRESS": "W toku", "COMPLETED": "Zatwierdzona"}
MODE_LABELS = {"blinded_gt_v1": "Zaślepiona GT", "assisted": "Asystowana (diagnostyczna)"}
VISIBILITY_LABELS = {"none": "0 — brak tablic", "one": "1 — jedna tablica",
                     "multiple": "2+ — kilka tablic", "uncertain": "? — nie wiem, ile tablic"}
HIDDEN_PREDICTION = "ukryta do czasu GT"
STAGE_LABELS = {"VALID_QUAD": "Poprawny obszar", "NO_DETECTION": "Brak detekcji", "DETECTION_INVALID_QUAD": "Błędna geometria",
                "NOT_RUN": "Nie uruchomiono", "NO_CHARACTERS": "Brak znaków", "READ": "Odczyt"}



def _observation_list_label(number, state, *, cancelled=False, execution_failed=False):
    """Transparent stage status; do not hide a verified MT decision."""
    prefix = f"{int(number):02d}"
    if state in {"MT zweryfikowane", "MT i MZ zweryfikowane"}:
        return (f"✓  {prefix}   {state}", "review_done")
    if state == "MT zweryfikowane · MZ w toku":
        return (f"◐  {prefix}   MT zweryfikowane • MZ w toku", "review_pending")
    if state in {"Poza weryfikacją MT", "Poza weryfikacją MT"}:
        label = ("Próba przerwana" if cancelled else "Błąd wykonania"
                 if execution_failed else "Poza weryfikacją MT")
        return (f"—  {prefix}   {label}", "review_excluded")
    if state.startswith("Decyzje:"):
        return (f"○  {prefix}   Dokończ weryfikację", "review_pending")
    if state == "Wykrycie zweryfikowane · obrys w toku":
        return (f"◐  {prefix}   Wykrycie OK • obrys w toku", "review_pending")
    return (f"○  {prefix}   Weryfikacja MT w toku", "review_pending")


def _cancelled_mt_explanation(group):
    """Explain Android attempt lifecycle separately from the MT model outcome."""
    reasons = {str(row.get("cancel_reason") or "").strip().lower()
               for row in (getattr(group, "records", ()) or ()) if isinstance(row, dict)}
    recorded = [row for row in (getattr(group, "records", ()) or ())
                if isinstance(row, dict) and str(row.get("mt_status") or "").upper()
                in DETECTION_STATUSES]
    if "scene_superseded" in reasons:
        reason = "Telefon zastąpił tę scenę nowszą podczas przetwarzania."
    else:
        reason = "Telefon zakończył tę próbę przed ukończeniem przetwarzania."
    if recorded:
        observed = ("MT zapisał wykrycie i jego obrys Pose." if any(
                    str(row.get("mt_pose_status") or "").upper() == "VALID_QUAD"
                    for row in recorded) else "MT zapisał wykrycie tablicy.")
    else:
        observed = "Próba nie dostarcza ukończonego wyniku wykrywania."
    return (f"{reason} {observed} Dane można obejrzeć, ale nie wliczamy "
            "tej próby do skuteczności MT ani do weryfikacji OCR.")


def _display(value, *, rate=False):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Tak" if value else "Nie"
    if rate:
        return f"{value * 100:.2f}%"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _review_image_sources(kind, row, attempt, group, samples, available_names):
    """Opisuj rodzaj dowodu, nie kolejny 'model MT'. Usun duplikaty sciezek."""
    seen, rows = set(), []
    available = set(available_names or ())

    def add(label, entry, payload):
        if not entry or entry in seen or entry not in available:
            return
        seen.add(entry)
        rows.append((label, payload))


    # Android P1: a pipeline Bitmap before ROI belongs to the entire MT
    # invocation, not just the currently chosen detection. Historic archives
    # without this evidence continue using their original MT-input images.
    invocation_id = str(attempt.get("mt_invocation_id") or "")
    source = source_invocation_context(
        getattr(group, "records", ()) or (), invocation_id, available)
    if source is not None:
        add("Pełna klatka + obszar ROI (podgląd)", source["entry"],
            {"image_entry": source["entry"], "source_frame_view": True,
             "source_invocation_id": invocation_id})
    full = [] if group is None else sorted(group.evidence_entries)
    preferred = attempt.get("mt_input_evidence_entry")
    ordered_full = ([preferred] if preferred in full else []) + [item for item in full if item != preferred]
    for i, entry in enumerate(ordered_full):
        label = ("Wejście MT — bieżąca próba" if entry == preferred else
                 f"Wejście MT — dodatkowy dowód {i + 1}")
        add(label, entry, {"image_entry": entry})

    if kind == "sample":
        add("Wycięta tablica (crop)", row.get("image_entry"), row)
    add("Zdjęcie źródłowe z telefonu", attempt.get("evidence_entry"), attempt)
    if kind == "attempt":
        related = [sample for sample in samples if sample.get("attempt_id") == row.get("id")]
        for i, sample in enumerate(related, 1):
            label = "Wycięta tablica (crop)" if len(related) == 1 else f"Wycięta tablica (crop {i})"
            add(label, sample.get("image_entry"), sample)
    return rows



def _ordered_operator_gt_quad(points, width, height):
    """Canonicalize NEW operator clicks using the project's polygon sorter.

    Only the manually collected GT is reordered. Recorded MT keypoints and
    historical/frozen GT are never reindexed or rewritten. Reject bad quads
    using the stricter, convex non-crossing GT validator.
    """
    try:
        w, h = float(width), float(height)
        if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
            raise ValueError("Nieprawidłowy rozmiar wejścia MT")
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            raise ValueError("Wskaż dokładnie cztery narożniki")
        raw = tuple((float(p[0]), float(p[1])) for p in points)
        if (any(not math.isfinite(v) for xy in raw for v in xy)
                or any(not (0 <= x <= w and 0 <= y <= h) for x, y in raw)):
            raise ValueError("Narożniki muszą być poprawne i znajdować się na wejściu MT")
        ordered = PolygonValidator.order_points(list(raw))
        return validate_mt_quad(ordered, w, h)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(str(exc) or "Nieprawidłowe narożniki GT") from exc


def _pose_quad_on_mt_input(attempt, sample):
    """Zapisany polygon Pose: źródłowa klatka -> ROI/letterbox -> wejście MT.

    NIGDY nie tworzymy polygonu z BBOX. Brak zweryfikowanych czterech punktów
    oznacza None. Nie zmieniamy narożników oryginału.
    """
    import math
    if not isinstance(attempt, dict) or not isinstance(sample, dict):
        return None
    attempt_id = str(attempt.get("id") or attempt.get("attempt_id") or "")
    if not attempt_id or str(sample.get("attempt_id") or "") != attempt_id:
        return None
    geometry = sample.get("plate_geometry")
    geometry = geometry if isinstance(geometry, dict) else {}
    if geometry.get("available") is False:
        return None
    raw = geometry.get("plate_corners_norm") or sample.get("plate_corners_norm")
    if isinstance(raw, str):
        try:
            raw = [list(map(float, pair.split(":"))) for pair in raw.split(";")]
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, (tuple, list)) or len(raw) != 4:
        return None
    try:
        points = [(float(pair[0]), float(pair[1])) for pair in raw
                  if isinstance(pair, (tuple, list)) and len(pair) == 2]
        source_w = float(geometry.get("source_width_px") or sample["source_width_px"])
        source_h = float(geometry.get("source_height_px") or sample["source_height_px"])
        roi_left = float(attempt["roi_left"])
        roi_top = float(attempt["roi_top"])
        roi_right = float(attempt["roi_right"])
        roi_bottom = float(attempt["roi_bottom"])
        scale = float(attempt["input_scale"])
        pad_x = float(attempt["input_pad_x"])
        pad_y = float(attempt["input_pad_y"])
        input_w = float(attempt["input_width"])
        input_h = float(attempt["input_height"])
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    numbers = [value for p in points for value in p] + [source_w, source_h,
               roi_left, roi_top, roi_right, roi_bottom, scale, pad_x,
               pad_y, input_w, input_h]
    if (len(points) != 4 or not all(math.isfinite(v) for v in numbers)
            or source_w <= 0 or source_h <= 0 or input_w <= 0 or input_h <= 0
            or scale <= 0 or roi_left < 0 or roi_top < 0
            or roi_right > source_w + 1 or roi_bottom > source_h + 1
            or roi_left >= roi_right or roi_top >= roi_bottom
            or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points)):
        return None
    # Odrzucamy czteropunktowy zapis zdegenerowany, ale nie poprawiamy
    # kolejności/współrzędnych modelu automatycznie.
    area2 = sum(points[i][0] * points[(i + 1) % 4][1]
                - points[(i + 1) % 4][0] * points[i][1] for i in range(4))
    if abs(area2) <= 1e-8:
        return None
    src = [(x * source_w, y * source_h) for x, y in points]
    # Nie łączymy narożników cropa z inną detekcją / inną klatką.
    try:
        bbox = [float(attempt[k]) for k in
                ("plate_left", "plate_top", "plate_right", "plate_bottom")]
    except (ValueError, KeyError, TypeError):
        return None
    if not all(math.isfinite(v) for v in bbox):
        return None
    envelope = [min(x for x, _ in src), min(y for _, y in src),
                max(x for x, _ in src), max(y for _, y in src)]
    tolerance = max(12.0, 0.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    if max(abs(a - b) for a, b in zip(bbox, envelope)) > tolerance:
        return None
    return tuple(((x - roi_left) * scale + pad_x,
                  (y - roi_top) * scale + pad_y) for x, y in src)


def _pose_quad_from_mt_attempt(attempt):
    """Read ONLY source-frame MT keypoints from this exact detection record.

    The explicit v1 contract prevents mixing another detection's crop/Pose,
    and forbids reconstructing an MT polygon from its BBOX. New-format records
    with missing or malformed source geometry MUST NOT fall back to crop data.
    """
    if not isinstance(attempt, dict) or attempt.get("mt_pose_status") != "VALID_QUAD":
        return None
    if (attempt.get("mt_pose_coordinate_space") != "source_frame_normalized_v1"
            or attempt.get("mt_pose_corner_order") != "TL_TR_BR_BL"):
        return None
    raw = attempt.get("mt_pose_corners_norm")
    try:
        if isinstance(raw, str):
            pairs = [part.split(":") for part in raw.split(";")]
            if len(pairs) != 4 or any(len(pair) != 2 for pair in pairs):
                return None
            points = [(float(x), float(y)) for x, y in pairs]
        elif isinstance(raw, (list, tuple)) and len(raw) == 4:
            points = [(float(pair[0]), float(pair[1])) for pair in raw]
        else:
            return None
        source_w = float(attempt["mt_pose_source_width_px"])
        source_h = float(attempt["mt_pose_source_height_px"])
        roi_left, roi_top, roi_right, roi_bottom = (
            float(attempt[k]) for k in ("roi_left", "roi_top", "roi_right", "roi_bottom"))
        scale, pad_x, pad_y, input_w, input_h = (
            float(attempt[k]) for k in ("input_scale", "input_pad_x", "input_pad_y",
                                      "input_width", "input_height"))
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        return None
    values = ([v for pair in points for v in pair]
              + [source_w, source_h, roi_left, roi_top, roi_right, roi_bottom,
                 scale, pad_x, pad_y, input_w, input_h])
    if (not all(math.isfinite(value) for value in values)
            or source_w <= 0 or source_h <= 0 or input_w <= 0 or input_h <= 0
            or scale <= 0 or roi_left < 0 or roi_top < 0
            or roi_right > source_w + 1 or roi_bottom > source_h + 1
            or roi_left >= roi_right or roi_top >= roi_bottom
            or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points)):
        return None
    source = [(x * source_w, y * source_h) for x, y in points]
    # The bbox is an independent MT output, NOT the source of these points.
    # A loose bbox must not invalidate a valid Pose recorded for this attempt.
    transformed = tuple(((x - roi_left) * scale + pad_x,
                         (y - roi_top) * scale + pad_y) for x, y in source)
    try:
        return validate_mt_quad(transformed, input_w, input_h)
    except (TypeError, ValueError):
        return None


def _mt_pose_contract_declared(session, attempt):
    """A new-format export must never be silently rescued by legacy MZ data."""
    if "mt_pose_status" in attempt:
        return True
    schema = getattr(getattr(session, "bundle", None), "sample_schema", {})
    capabilities = schema.get("capabilities", {}) if isinstance(schema, dict) else {}
    return bool(isinstance(capabilities, dict) and capabilities.get("mt_pose_per_detection") is True)


def _linked_pose_quad_on_mt_input(session, attempt, selected_sample=None):
    """Prefer recorded MT Pose; use a linked crop only for legacy archives."""
    if not isinstance(attempt, dict):
        return None
    # The existence of this field declares the per-detection contract.
    # A broken new record cannot be silently rescued with data from MZ.
    if _mt_pose_contract_declared(session, attempt):
        return _pose_quad_from_mt_attempt(attempt)
    attempt_id = str(attempt.get("id") or attempt.get("attempt_id") or "")
    if not attempt_id or not session:
        return None
    samples = getattr(session, "samples", {})
    candidates = [selected_sample] if isinstance(selected_sample, dict) else []
    index = getattr(session, "sample_ids_by_attempt", {}) or {}
    ids = index.get(attempt_id) or [k for k, value in samples.items()
                                    if value.get("attempt_id") == attempt_id]
    candidates.extend(samples[k] for k in ids if k in samples)
    for sample in candidates:
        quad = _pose_quad_on_mt_input(attempt, sample)
        if quad is not None:
            return quad
    return None

def _pose_source_status(session, attempt):
    """Why this exact MT attempt has or lacks an auditable four-corner polygon.

    A bounding box is NEVER promoted to a polygon. A missing linked crop may
    mean the export lacks corner evidence; it does NOT prove MT predicted a box.
    """
    if not attempt:
        return "no_attempt", "Brak powiązanej detekcji MT."
    status = str(attempt.get("mt_status") or "").upper()
    if status not in DETECTION_STATUSES:
        return "no_detection", "Ten rekord nie zawiera detekcji MT do weryfikacji geometrii."
    if _mt_pose_contract_declared(session, attempt):
        if _pose_quad_from_mt_attempt(attempt) is not None:
            return "mt_record_pose", "Dostępne są cztery rogi zapisane bezpośrednio z MT."
        if status == "DETECTION_INVALID_QUAD" or attempt.get("mt_pose_status") == "INVALID_QUAD":
            return "mt_invalid_quad", "Model nie podał prawidłowych czterech rogów."
        if attempt.get("mt_pose_status") == "VALID_QUAD":
            return "mt_pose_record_inconsistent", "Zapis Pose tej detekcji jest niekompletny lub nieprawidłowy."
        return "mt_pose_unavailable", "W tym rekordzie nie zapisano czterech rogów modelu."
    if _linked_pose_quad_on_mt_input(session, attempt) is not None:
        return "pose_available", "Historyczny Pose odczytany z powiązanej próbki MZ."
    aid = str(attempt.get("id") or attempt.get("attempt_id") or "")
    samples = getattr(session, "samples", {}) or {}
    linked_ids = (getattr(session, "sample_ids_by_attempt", {}) or {}).get(aid) or []
    linked = [samples[key] for key in linked_ids if key in samples]
    if not linked:
        linked = [value for value in samples.values() if str(value.get("attempt_id") or "") == aid]
    if status == "DETECTION_INVALID_QUAD":
        return "mt_invalid_quad", "MT nie podało prawidłowych czterech rogów."
    if not linked:
        return "no_linked_crop", (
            "W tej starszej sesji nie zapisano czterech rogów tej detekcji. "
            "Brak cropa nie oznacza, że model ich nie wyznaczył.")
    has_raw_corners = any(
        bool((sample.get("plate_geometry") if isinstance(sample.get("plate_geometry"), dict) else {}).get("plate_corners_norm")
             or sample.get("plate_corners_norm")) for sample in linked)
    if not has_raw_corners:
        return "no_saved_corners", "Powiązany crop istnieje, ale nie zapisano przy nim czterech narożników Pose."
    return "invalid_or_unlinked_corners", (
        "Narożniki istnieją w danych cropa, lecz nie przechodzą walidacji powiązania, "
        "współrzędnych lub transformacji do wejścia MT.")


def _pose_geometry_rows(session):
    """One audit row per MT attempt; numeric metrics only with independent manual GT.

    A crop or an incomplete GT never silently enters the denominator. No data is
    modified here and historical subjective pose_quality remains only in sidecar.
    """
    rows = []
    for attempt_id, row in session.attempts.items():
        decision = session.review.attempt_annotations.get(attempt_id, {})
        group = session.mt_invocations.get(str(row.get("mt_invocation_id") or ""))
        active = bool(group and group.executed and not group.cancelled and not group.execution_failed
                      and not session.cancelled(row))
        pred = (_linked_pose_quad_on_mt_input(session, row) if active and
                str(row.get("mt_status", "")).upper() in DETECTION_STATUSES else None)
        true_plate = decision.get("is_plate") is True and decision.get("evaluable") is not False
        image_entry = row.get("mt_input_evidence_entry")
        evidence_present = (image_entry in session.entry_names if hasattr(session, "entry_names") else bool(image_entry))
        pose_source_status, _ = _pose_source_status(session, row)
        unavailable_reason = str(decision.get("pose_gt_unavailable_reason") or "")
        eligible = bool(pred is not None and true_plate and evidence_present and not unavailable_reason)
        raw_gt = decision.get("pose_gt_quad_mt_input_px")
        status, metrics = "not_evaluable", None
        try:
            size = (float(row["input_width"]), float(row["input_height"]))
            pred_valid = validate_mt_quad(pred, *size) if pred is not None else None
        except (ValueError, TypeError, KeyError):
            pred_valid = None
        if not active:
            status = "source_not_evaluable"
        elif (str(row.get("mt_status") or "").upper() == "NO_DETECTION" and
              raw_gt and decision.get("pose_gt_schema") == POSE_GT_SCHEMA):
            try:
                invocation = session.invocation_annotation(str(row.get("mt_invocation_id") or ""))
                if (invocation.get("visible_plate_count") != "one" or
                        invocation.get("evaluable") is False):
                    status = "missed_detection_gt_cardinality_conflict"
                elif (size is None or decision.get("pose_gt_image_entry") != image_entry or
                        decision.get("pose_gt_input_size_px") != list(size)):
                    status = "missed_detection_gt_provenance_mismatch"
                else:
                    validate_mt_quad(raw_gt, *size)
                    status = "missed_detection_gt_saved"
            except (ValueError, TypeError, KeyError):
                status = "missed_detection_gt_invalid"
        elif pred_valid is None:
            status = "no_valid_pose"
        elif not evidence_present:
            status = "no_mt_input_evidence"
        elif not true_plate:
            status = "not_confirmed_plate"
        elif unavailable_reason and raw_gt:
            status = "gt_exclusion_conflict"
        elif unavailable_reason:
            status = "manual_gt_unavailable"
        elif not raw_gt:
            status = "missing_manual_gt"
        elif (decision.get("pose_gt_schema") != POSE_GT_SCHEMA or
              decision.get("pose_gt_image_entry") != row.get("mt_input_evidence_entry") or
              decision.get("pose_gt_input_size_px") != list(size)):
            status = "gt_provenance_mismatch"
        else:
            try:
                metrics = compare_mt_pose_to_gt(pred_valid, raw_gt, *size)
                status = "measured"
            except (ValueError, TypeError):
                status = "invalid_manual_gt"
        rows.append({
            "session_id": session.session_id,
            "source_archive_sha256": session.review.source_archive_sha256,
            "attempt_id": attempt_id,
            "mt_invocation_id": row.get("mt_invocation_id", ""),
            "mt_status": row.get("mt_status", ""),
            "mt_input_evidence_entry": row.get("mt_input_evidence_entry", ""),
            "pose_source_status": pose_source_status,
            "pose_gt_unavailable_reason": unavailable_reason,
            "eligible": eligible and pred_valid is not None,
            "metric_status": status,
            "gt_schema": decision.get("pose_gt_schema") or "",
            "pose_gt_review_mode": decision.get("pose_gt_review_mode") or "",
            "gt_quad_mt_input_px": json.dumps(raw_gt, ensure_ascii=False) if raw_gt else "",
            "pred_quad_mt_input_px": json.dumps(pred, ensure_ascii=False) if pred else "",
            "nmce_pct": metrics["nmce_pct"] if metrics else None,
            "polygon_iou": metrics["polygon_iou"] if metrics else None,
            "polygon_iou_pct": metrics["polygon_iou_pct"] if metrics else None,
            "mean_corner_error_px": metrics["mean_corner_error_px"] if metrics else None,
            "per_corner_errors_px": json.dumps(metrics["per_corner_errors_px"]) if metrics else "",
            "gt_envelope_diagonal_px": metrics["gt_envelope_diagonal_px"] if metrics else None,
        })
    return rows


def _pose_geometry_summary(session):
    rows = _pose_geometry_rows(session)
    measured = [row for row in rows if row["metric_status"] == "measured"]
    eligible = [row for row in rows if row["eligible"]]
    return {
        "pose_eligible": len(eligible),
        "pose_measured": len(measured),
        "pose_pending": len(eligible) - len(measured),
        "pose_manual_unavailable": sum(row["metric_status"] == "manual_gt_unavailable" for row in rows),
        "pose_nmce_mean_pct": sum(row["nmce_pct"] for row in measured) / len(measured) if measured else None,
        "pose_iou_mean_pct": sum(row["polygon_iou_pct"] for row in measured) / len(measured) if measured else None,
        "pose_no_geometry": sum(row["metric_status"] in {"no_valid_pose", "missed_detection_gt_saved",
                                  "missed_detection_gt_invalid", "missed_detection_gt_provenance_mismatch",
                                  "missed_detection_gt_cardinality_conflict"}
                                for row in rows),
        "missed_detection_manual_gt": sum(row["metric_status"] == "missed_detection_gt_saved" for row in rows),
    }


def _review_color_mix(base, foreground, amount=0.05):
    """Very subtle, theme-neutral Tk card tint; never hardcode a light-only palette."""
    try:
        a, b = (str(color).lstrip("#") for color in (base, foreground))
        if len(a) != 6 or len(b) != 6:
            return base
        x = [int(a[i:i + 2], 16) for i in (0, 2, 4)]
        y = [int(b[i:i + 2], 16) for i in (0, 2, 4)]
        return "#" + "".join(f"{round(p * (1 - amount) + q * amount):02x}" for p, q in zip(x, y))
    except (ValueError, TypeError):
        return base



def _primary_review_image_index(image_rows, attempt, group):
    """One unambiguous image for one attempt; never choose another MT record's frame."""
    if not isinstance(attempt, dict):
        return None
    entries = set(getattr(group, "evidence_entries", ()) or ())
    preferred = str(attempt.get("mt_input_evidence_entry") or "")
    if preferred and preferred in entries:
        for index, (_, item) in enumerate(image_rows or ()):
            if item.get("image_entry") == preferred:
                return index
    fallback = str(attempt.get("evidence_entry") or "")
    if fallback:
        for index, (_, item) in enumerate(image_rows or ()):
            if item.get("image_entry") == fallback or item.get("evidence_entry") == fallback:
                return index
    return None


class MobileSampleReviewWindow:
    def __init__(self, browser, report, bundle=None):
        self.browser, self.report = browser, report
        self.session = None
        self.stats = {}
        self.subject_key = ""
        self.current_record = None
        self.current_invocation = ""
        self._subject_ids = {}
        self._record_ids = {}
        self._multi_child_targets = {}
        self._multi_active_gt = ""
        self._multi_active_group = ""
        self._multi_ui_group = ""
        self._multi_drag_start = None
        self._multi_drag_point = None
        self._multi_bbox_edit = False
        self._multi_requested_attempt = ""
        self._closed = False
        self._saving = False
        self._close_requested = False
        self._pending_subject = ""
        self._retained_image_on_refresh = None
        self._rendering = False
        self._autosave_id = None
        self._image_token = 0
        self._image = None
        self._image_source_size = None
        self._image_entry = ""
        self._photo = None
        self._view_zoom = 1.0
        self._view_center = None
        self._view_fit_scale = 1.0
        self._view_pixel_scale = 1.0
        self._view_drag_at = None
        self._zoom_to_mt_pending = False
        self._pose_gt_cache_token = None
        self._pose_gt_points = []
        self._pose_gt_drag_index = None
        self._pose_gt_before_drag = None
        self._mt_corr_active = False
        self._mt_corr_attempt = ""
        self._mt_corr_points = []
        self._mt_corr_drag = None
        self._mt_corr_last_saved = []
        # Freehand 4-click redraw is a separate, explicitly assisted annotation.
        # Nothing is written until four valid points have been supplied.
        self._mt_redraw_active = False
        self._mt_redraw_attempt = ""
        self._mt_redraw_points = []
        self._status_pills = {}
        self._status_paint_scheduled = False
        self._mt_correction_ux_v026 = True
        # One guided entry per selected MT image. Cancelling must not reopen
        # the editor on the next asynchronous refresh/autosave.
        self._pose_auto_token = None
        self._results = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mobile-review")
        palette = browser.palette
        self.bg = palette.get("panel", "#162013")
        self.fg = palette.get("fg", "#e9efdf")
        self.muted = palette.get("muted", self.fg)
        self.accent = palette.get("accent", self.fg)
        self.review_card_bg = _review_color_mix(self.bg, self.fg, 0.035)
        self.review_border = _review_color_mix(self.bg, self.fg, 0.19)
        self.window = tk.Toplevel(browser.window)
        self.window._aat_skip_window_recovery = True
        self.window.title("Weryfikacja próbek z telefonu")
        width = min(1680, self.window.winfo_screenwidth() - 60)
        height = min(960, self.window.winfo_screenheight() - 85)
        self.window.geometry(f"{width}x{height}+25+25")
        self.window.minsize(min(1050, self.window.winfo_screenwidth() - 60), 670)
        self.window.configure(bg=self.bg)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", self._cancel_mt_correction, add="+")
        self.window.bind("<Destroy>", self._destroyed, add="+")
        self.actions = []
        self._build()
        self._poll_id = self.window.after(40, self._poll)
        source = Path(bundle.path if bundle and bundle.source_archive_sha256 == report.source_archive_sha256 else report.source_path)
        expected_hash = report.source_archive_sha256
        def load():
            # Revalidate on opening; the stored report may outlive its archive.
            fresh = read_mobile_report_bundle(source)
            if expected_hash and fresh.source_archive_sha256 != expected_hash:
                raise ValueError("Plik źródłowy nie odpowiada SHA-256 zapisanego raportu. Zaimportuj właściwą sesję.")
            session = MobileReviewSession(fresh)
            # Viewing must not migrate a saved review or create a sidecar.
            # Canvas provenance is ASSISTED; a fresh review adopts that mode
            # atomically with its first explicit annotation in core._store.
            converted = False
            from ..ranking.mobile_canvas_review import context as canvas_context
            session._history_inspection_only = not any(canvas_context(session,gid)
                for gid in (*session.mt_invocations, *session.scene_frames))
            return session, session.statistics(), converted
        self._submit(load, self._loaded)

    def _label(self, parent, text="", **kwargs):
        # Inherit the parent surface so labels do not punch rectangular holes
        # into the softly shaded review cards on Windows / both app themes.
        bg = kwargs.pop("bg", None)
        fg = kwargs.pop("fg", self.fg)
        if bg is None:
            try:
                bg = parent.cget("bg")
            except (tk.TclError, AttributeError, KeyError):
                bg = self.bg
        return tk.Label(parent, text=text, bg=bg, fg=fg,
                        anchor="w", justify="left", **kwargs)

    def _review_section(self, parent):
        return tk.Frame(parent, bg=self.review_card_bg, padx=13, pady=11,
                        highlightthickness=1, highlightbackground=self.review_border, bd=0)

    def _button(self, parent, text, command, *, primary=False, **grid):
        button = CanvasReviewEditor.make_action_button(parent, text, command, primary=primary)
        button.configure(state="disabled")
        button.grid(**grid, padx=3, pady=3)
        self.actions.append(button)
        return button

    def _table(self, parent, columns, widths, *, height=8):
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height,
                            selectmode="browse", style="MobileReview.Treeview")
        for name, width in zip(columns, widths):
            tree.heading(name, text=name)
            tree.column(name, width=width, minwidth=45, stretch=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        return frame, tree

    def _build(self):
        # Older GUI probes construct the window without calling __init__.
        # Resolve the same theme defaults here as well (no global style reset).
        self.accent = getattr(self, "accent", (getattr(self.browser, "palette", {}) or {}).get("accent", self.fg))
        self.review_card_bg = getattr(self, "review_card_bg", _review_color_mix(self.bg, self.fg, 0.035))
        self.review_border = getattr(self, "review_border", _review_color_mix(self.bg, self.fg, 0.19))
        style = None
        try:
            style = ttk.Style(self.window)
            style.configure("MobileReview.Treeview", rowheight=28, font=("Segoe UI", 9))
            style.configure("MobileReview.Treeview.Heading", font=("Segoe UI", 9, "bold"))
            style.configure("MobileReview.TRadiobutton", background=self.review_card_bg,
                            foreground=self.fg, font=("Segoe UI", 10))
            style.map("MobileReview.TRadiobutton", background=[("active", self.review_card_bg)])
            style.configure("MobileReview.TCheckbutton", background=self.review_card_bg,
                            foreground=self.fg, font=("Segoe UI", 10))
            style.map("MobileReview.TCheckbutton", background=[("active", self.review_card_bg)])
        except tk.TclError:
            pass
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        header = tk.Frame(self.window, bg=self.bg, padx=10, pady=4)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.session_label = self._label(header, "Wczytywanie sesji…", font=("Segoe UI", 11, "bold"), width=1)
        self.session_label.grid(row=0, column=0, sticky="ew")
        self.session_label.bind("<Configure>", lambda e:
                                self.session_label.configure(wraplength=max(1, e.width)))
        self.progress_label = self._label(header, font=("Segoe UI", 14, "bold"))
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        actions = tk.Frame(header, bg=self.bg)
        actions.grid(row=0, column=1, sticky="e")
        export_button = self._button(actions, "Eksport wyników", self.export, row=0, column=0)
        self.export_button = export_button
        export_button.bind('<ButtonPress-1>', self._export_press, add='+')
        self.form_status_label = self._label(actions, "W TOKU", font=("Segoe UI", 9, "bold"))
        self.form_status_label.grid(row=0, column=1, sticky="e", padx=(10, 7))
        self.complete_button = self._button(actions, "Zatwierdź weryfikację", self.complete,
                                            primary=True, row=0, column=2)
        for button in (export_button, self.complete_button):
            button.configure(font=("Segoe UI", 9, "bold"), padx=8, pady=4)

        tabs = ttk.Notebook(self.window)
        tabs.grid(row=2, column=0, sticky="nsew", padx=10)
        review = tk.Frame(tabs, bg=self.bg)
        review.columnconfigure(0, weight=0, minsize=440)
        review.columnconfigure(1, weight=1, minsize=360)
        review.columnconfigure(2, weight=0, minsize=415)
        review.rowconfigure(0, weight=1)
        tabs.add(review, text="Weryfikacja")

        left = tk.Frame(review, bg=self.review_card_bg, padx=8, pady=8,
                        highlightthickness=1, highlightbackground=self.review_border)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        self._label(left, "OBRAZY", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(3, 3))
        cols = ("Nr", "MT", "Wsk.", "GT", "Rogi", "Form.")
        widths = (36, 42, 49, 39, 46, 49)
        frame, self.subject_tree = self._table(left, cols, widths, height=18)
        frame.grid(row=2, column=0, sticky="nsew")
        for key, width in zip(cols, widths):
            self.subject_tree.heading(key, text=key, anchor="center")
            self.subject_tree.column(key, width=width, minwidth=max(32, width - 3),
                                     stretch=(key != "Nr"), anchor="center")
        try:
            if style is not None:
                style.configure("MobileReview.Observations.Treeview", rowheight=30,
                                font=("Segoe UI", 10))
                style.configure("MobileReview.Observations.Treeview.Heading",
                                font=("Segoe UI", 9, "bold"))
                style.map("MobileReview.Observations.Treeview",
                          background=[("selected", "#DDF2FF")],
                          foreground=[("selected", "#15364B")])
                self.subject_tree.configure(style="MobileReview.Observations.Treeview")
        except tk.TclError:
            pass
        self.subject_tree.configure(show="tree headings")
        self.subject_tree.heading("#0", text="Obserwacja / tablica GT", anchor="w")
        self.subject_tree.column("#0", width=124, minwidth=100, stretch=True, anchor="w")
        self.subject_tree.tag_configure("review_done", background="#EDF5EC", foreground="#223A2D")
        self.subject_tree.tag_configure("review_pending", background="#F6F8F8", foreground="#253942")
        self.subject_tree.tag_configure("review_excluded", background="#F0F2F3", foreground="#5B6B70")
        self.subject_tree.bind("<<TreeviewSelect>>", self._select_subject)
        self.subject_tree.bind("<<TreeviewSelect>>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<Configure>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<<TreeviewOpen>>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<<TreeviewClose>>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<MouseWheel>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<MouseWheel>", self._route_image_wheel, add="+")
        self.subject_tree.bind("<Button-4>", lambda _e: self._schedule_status_pills(), add="+")
        self.subject_tree.bind("<Button-5>", lambda _e: self._schedule_status_pills(), add="+")
        # Treeview itself cannot color individual cells. Render lightweight
        # visible-only status chips, retaining the native keyboard/scroll tree.
        self._status_scrollbar = next((w for w in frame.winfo_children()
                                       if isinstance(w, ttk.Scrollbar)), None)
        if self._status_scrollbar is not None:
            self.subject_tree.configure(yscrollcommand=self._status_yscroll)
        self._button(left, "Następna do weryfikacji", self.next_pending,
                     primary=True, row=3, column=0, sticky="ew")

        viewer = tk.Frame(review, bg=self.bg)
        viewer.grid(row=0, column=1, sticky="nsew", padx=(0, 3))
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(4, weight=1)
        image_bar = tk.Frame(viewer, bg=self.bg)
        image_bar.grid(row=0, column=0, sticky="ew")
        image_bar.columnconfigure(0, weight=1)
        self.record_label = self._label(image_bar, "Wybrany zapis: —", font=("Segoe UI", 9))
        self.record_label.grid(row=0, column=0, sticky="ew")
        # Internal source-index controller only. A technical list of evidence
        # entries is not an operator choice and must not appear in the UI.
        self.image_choice = ttk.Combobox(image_bar, state="readonly", width=24)
        self.target_nav = tk.Frame(image_bar, bg=self.bg)
        self.target_nav.grid(row=1, column=0, sticky="w", pady=(2, 1))
        self.target_previous = CanvasReviewEditor.make_action_button(self.target_nav,
            "← Poprzedni obszar", lambda: self._switch_detected_area(-1), compact=True)
        self.target_previous.pack(side="left")
        self.target_position = self._label(self.target_nav, "Obszar 1 z 1", font=("Segoe UI", 9, "bold"))
        self.target_position.pack(side="left", padx=10)
        self.target_next = CanvasReviewEditor.make_action_button(self.target_nav,
            "Następny obszar →", lambda: self._switch_detected_area(1), compact=True)
        self.target_next.pack(side="left")
        self.target_nav.grid_remove()
        self.image_help = self._label(viewer,
            "Zdjęcie przypisane do weryfikowanego wskazania MT. Niebieski obrys pochodzi z telefonu.",
            font=("Segoe UI", 9), wraplength=700)
        self.image_help.grid(row=1, column=0, sticky="ew", pady=(1, 3))
        zoom_bar = tk.Frame(viewer, bg=self.bg)
        zoom_bar.grid(row=2, column=0, sticky="ew", pady=(3, 1))
        CanvasReviewEditor.make_action_button(zoom_bar, "−", lambda: self.zoom_image(1 / 1.5),
            compact=True, width=3).pack(side="left", padx=2)
        CanvasReviewEditor.make_action_button(zoom_bar, "+", lambda: self.zoom_image(1.5),
            compact=True, width=3).pack(side="left", padx=2)
        CanvasReviewEditor.make_action_button(zoom_bar, "Dopasuj", self.fit_image,
            compact=True).pack(side="left", padx=(5, 3))
        self.plate_fit_button = CanvasReviewEditor.make_action_button(zoom_bar, "Następna tablica (A)",
            self.zoom_to_mt_detection, compact=True)
        self.plate_fit_button.pack(side="left", padx=(4, 8))

        # Source frame is a read-only inspection surface. Do not use its
        # coordinates in the MT-input/GT editing paths (separate Desktop P2B).
        self.source_frame_button = CanvasReviewEditor.make_action_button(zoom_bar,
            "Cała klatka + ROI", self._toggle_source_frame_view, compact=True)
        self.source_frame_button.pack(side="left", padx=(2, 8))
        self.source_frame_button.pack_forget()
        self.zoom_value = self._label(zoom_bar, "×1.0", font=("Segoe UI", 9, "bold"))
        self.zoom_value.pack(side="left")
        # Z2-style in-canvas editing: no mode CTA. Selected MT polygon is
        # directly draggable. D/Shift+click starts a replacement, Delete
        # removes only the operator correction. The original MT stays read-only.
        self.mt_quad_mode_bar = None  # deprecated widget name for older probes
        self.canvas = tk.Canvas(viewer, height=520, bg="#101820", highlightthickness=0, cursor="hand2")
        self.canvas.grid(row=4, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self._paint_image())
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", self._wheel_zoom)
        self.canvas.bind("<Button-5>", self._wheel_zoom)
        # Windows może kierować rolkę do widżetu z fokusem zamiast do obrazu.
        # Przechwycenie wyłącznie na Toplevel weryfikatora, nie bind_all:
        # położenie kursora rozstrzyga, czy zdarzenie dotyczy canvasu.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.window.bind(sequence, self._route_image_wheel, add="+")
        self.canvas.bind("<Motion>", self._update_mt_edit_cursor, add="+")
        self.canvas.bind("<Enter>", self._focus_image_for_wheel, add="+")
        self.canvas.bind("<ButtonPress-1>", self._pan_begin)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.canvas.bind("<Double-Button-1>", lambda event: "break" if self.pose_edit_var.get() or self._mt_corr_active or self._mt_redraw_active else self.fit_image())
        # Only on the photo canvas: never intercept Delete or D in a GT field.
        for sequence in ("<KeyPress-d>", "<KeyPress-D>"):
            self.canvas.bind(sequence, self._start_mt_redraw)
        for sequence in ("<Delete>", "<BackSpace>"):
            self.canvas.bind(sequence, self._delete_mt_operator_correction)
        self.canvas.bind("<ButtonPress-3>", lambda event: self._pan_begin(event, force_pan=True))
        self.canvas.bind("<Control-ButtonPress-3>", self._delete_mt_operator_correction, add="+")
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)
        self._label(viewer, "Rolka: zoom  •  przeciągnij róg/obrys: popraw  •  D / Shift+klik: nowy obrys  •  Delete / Ctrl+PPM: usuń korektę  •  Esc: anuluj  •  PPM: przesuń",
                    font=("Segoe UI", 9), fg=self.muted, wraplength=750).grid(row=5, column=0, sticky="w", pady=(1, 3))

        right_host = tk.Frame(review, bg=self.bg)
        right_host.grid(row=0, column=2, sticky="nsew")
        right_host.columnconfigure(0, weight=1)
        right_host.rowconfigure(0, weight=1)
        sidebar_scroll = tk.Canvas(right_host, bg=self.bg, bd=0, highlightthickness=0)
        sidebar_scroll.grid(row=0, column=0, sticky="nsew")
        self.review_sidebar_scroll = sidebar_scroll
        sidebar_bar = ttk.Scrollbar(right_host, orient="vertical", command=sidebar_scroll.yview)
        sidebar_bar.grid(row=0, column=1, sticky="ns")
        sidebar_scroll.configure(yscrollcommand=sidebar_bar.set)
        right = tk.Frame(sidebar_scroll, bg=self.bg, padx=5, pady=5)
        right.columnconfigure(0, weight=1)
        inside = sidebar_scroll.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda event: sidebar_scroll.configure(scrollregion=sidebar_scroll.bbox("all")))
        sidebar_scroll.bind("<Configure>", lambda event: sidebar_scroll.itemconfigure(inside, width=event.width))
        self.subject_label = self._label(right, font=("Segoe UI", 11, "bold"), wraplength=370)
        self.subject_label.grid(row=0, column=0, sticky="ew", pady=(4, 11), padx=(3, 0))

        # MT: zawsze te same dwie proste oceny, bez dropdownow nieznanej funkcji.
        mt = self._review_section(right)
        mt.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        mt.columnconfigure(0, weight=1)
        self._label(mt, "01  ·  WYKRYCIE TABLICY", font=("Segoe UI", 10, "bold"),
                    fg=self.accent).grid(row=0, column=0, sticky="ew")
        self.invocation_label = self._label(mt, "—", font=("Segoe UI", 9),
                                            fg=self.muted, wraplength=370)
        self.invocation_label.grid(row=1, column=0, sticky="ew", pady=(3, 4))
        self.show_input_button = self._button(mt, "Pokaż całe zdjęcie", self.show_invocation_input,
                                               row=2, column=0, sticky="w")
        self.show_input_button.grid_remove()
        self.mt_count_question = self._label(mt, "Ile tablic widzisz na zdjęciu?",
                    font=("Segoe UI", 10, "bold"), wraplength=370)
        self.mt_count_question.grid(row=3, column=0, sticky="ew", pady=(9, 2))
        self.mt_count_hint = self._label(mt, "Policz fizyczne tablice, NIE surowe detekcje MT.",
                    font=("Segoe UI", 9), fg=self.muted, wraplength=370)
        self.mt_count_hint.grid(row=4, column=0, sticky="ew")
        self.invocation_count_value = tk.StringVar(value="")
        self.invocation_count_radios = []
        count_group = tk.Frame(mt, bg=self.review_card_bg)
        count_group.grid(row=5, column=0, sticky="ew", pady=(3, 5))
        for index, (value, label) in enumerate(VISIBILITY_LABELS.items()):
            radio = ttk.Radiobutton(count_group, text=label, variable=self.invocation_count_value,
                                    value=value, command=self.choose_visible_plate_count,
                                    style="MobileReview.TRadiobutton")
            radio.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 12), pady=1)
            self.invocation_count_radios.append(radio)
        self.mt_count_group = count_group
        # Nie rysujemy starego dropdownu. Zostaje tylko jako zgodny odczyt dla diagnostyki/testow.
        self.invocation_choice = ttk.Combobox(mt, state="disabled", values=list(VISIBILITY_LABELS.values()))
        self.invocation_error_label = self._label(mt, font=("Segoe UI", 9), wraplength=350)
        self.invocation_error_label.grid(row=6, column=0, sticky="ew")
        self.invocation_error_label.grid_remove()
        self.mt_section_separator = ttk.Separator(mt, orient="horizontal")
        self.mt_section_separator.grid(row=7, column=0, sticky="ew", pady=(7, 9))
        self.detection_question = self._label(mt,
            "Czy zaznaczony obszar jest tablicą?",
            font=("Segoe UI", 10, "bold"), wraplength=370)
        self.detection_question.grid(row=8, column=0, sticky="ew", pady=(2, 2))
        self.detection_context = self._label(mt, "Zweryfikuj niebieski obrys na zdjęciu.",
                                              font=("Segoe UI", 9), fg=self.muted,
                                              wraplength=370)
        self.detection_context.grid(row=9, column=0, sticky="ew")
        self.detection_choice = tk.StringVar(value="")
        self.detection_radios = []
        decision_group = tk.Frame(mt, bg=self.review_card_bg)
        self.mt_detection_group = decision_group
        decision_group.grid(row=10, column=0, sticky="ew", pady=(3, 1))
        for index, (label, value) in enumerate((("Tak, to tablica", "yes"),
                ("Nie, to nie tablica", "no"), ("Nie mogę ustalić, czy to tablica", "skip"))):
            radio = ttk.Radiobutton(decision_group, text=label, variable=self.detection_choice,
                                     value=value, command=self.save_detection_choice,
                                     style="MobileReview.TRadiobutton")
            radio.grid(row=index, column=0, sticky="w", pady=1)
            self.detection_radios.append(radio)
        # Source-backed visual MT verification. Both surfaces share one variable;
        # this NEVER copies a predicted quad into quantitative Ground Truth.
        self.mt_visual_choice = tk.StringVar(value="")
        self.mt_visual_host = tk.Frame(mt, bg=self.review_card_bg)
        self.mt_visual_host.grid(row=11, column=0, sticky="ew", pady=(8, 2))
        self._label(self.mt_visual_host, "Szybka weryfikacja wskazania MT",
                    font=("Segoe UI", 9, "bold"), fg=self.fg).pack(anchor="w", pady=(0, 4))
        self.mt_visual_radios = []
        for value, label in (("not_plate", "Nie tablica"),
                ("imprecise", "Tablica — obrys niedokładny"),
                ("accurate", "Tablica — obrys dopasowany"),
                ("uncertain", "Nie mogę zweryfikować")):
            radio = tk.Checkbutton(self.mt_visual_host, variable=self.mt_visual_choice,
                onvalue=value, offvalue="", text=label,
                bg=self.review_card_bg, fg=self.fg, activebackground=self.review_card_bg,
                selectcolor=self.review_card_bg, anchor="w", command=self.save_mt_visual_choice)
            radio.pack(anchor="w")
            self.mt_visual_radios.append(radio)
        self.mt_visual_overlay = tk.Frame(self.canvas, bg="#10212b", padx=5, pady=4)
        self.mt_visual_overlay_buttons = []
        for value, label in (("not_plate", "✕ Nie tablica"),
                ("imprecise", "≈ Niedokładny"),
                ("accurate", "✓ Dobry"), ("uncertain", "? Nie wiem")):
            control = tk.Radiobutton(self.mt_visual_overlay, variable=self.mt_visual_choice,
                value=value, text=label, command=self.save_mt_visual_choice,
                indicatoron=False, relief=tk.FLAT, borderwidth=1,
                bg="#10212b", fg="#ffffff", activebackground="#24475E",
                activeforeground="#ffffff", selectcolor="#1686BD",
                font=("Segoe UI", 9, "bold"), padx=7, pady=4)
            control.pack(side="left", padx=(0, 4))
            # Canvas-mounted radios can keep keyboard focus after a choice;
            # the same Z2-style edit shortcuts must still work from them.
            for shortcut in ("<KeyPress-d>", "<KeyPress-D>"):
                control.bind(shortcut, self._start_mt_redraw, add="+")
            for shortcut in ("<Delete>", "<BackSpace>"):
                control.bind(shortcut, self._delete_mt_operator_correction, add="+")
            self.mt_visual_overlay_buttons.append(control)
        self.mt_visual_host.grid_remove()  # Decyzja tylko na obrazie.
        self.detection_question.grid_remove()
        self.detection_context.grid_remove()
        self.mt_detection_group.grid_remove()
        self.mt_section_separator.grid_remove()
        self.mt_visual_status = self._label(mt,
            "Wskazanie MT: wybierz odpowiedź na zdjęciu.",
            font=("Segoe UI", 9), fg=self.muted, wraplength=360)
        self.mt_visual_status.grid(row=8, column=0, sticky="ew", pady=(9, 3))
        self._mt_scored_controls = (
            self.mt_count_question, self.mt_count_hint, self.mt_count_group,
            self.mt_visual_status)

        # The second question is answered independently from the first.
        # Only an explicit positive answer here starts the one-shot drawing.
        # Absence of Pose permits detection assessment, but not corner metrics.
        self.pose_gt_host = self._review_section(right)
        self.pose_gt_host.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.pose_gt_host.columnconfigure(0, weight=1)
        self.pose_gt_title = self._label(self.pose_gt_host,
            "02  ·  POMIAR OBRYSU (OPCJONALNIE)", font=("Segoe UI", 10, "bold"),
            fg=self.accent, wraplength=370)
        self.pose_gt_title.grid(row=0, column=0, sticky="ew")
        self.pose_gt_question = self._label(self.pose_gt_host,
            "Chcesz wyznaczyć cztery rogi GT do pomiaru IoU/NMCE?",
            font=("Segoe UI", 10, "bold"), wraplength=370)
        self.pose_gt_question.grid(row=1, column=0, sticky="ew", pady=(8, 3))
        self.pose_gt_hint = self._label(self.pose_gt_host,
            "Opcjonalny pomiar ilościowy: zaznacz niezależny GT czterech narożników.",
            font=("Segoe UI", 9), fg=self.muted, wraplength=370)
        self.pose_gt_hint.grid(row=2, column=0, sticky="ew", pady=(0, 3))
        self.pose_edit_var = tk.BooleanVar(value=False)
        self.pose_gt_answer = tk.StringVar(value="")
        self.pose_gt_radios = []
        answer_group = tk.Frame(self.pose_gt_host, bg=self.review_card_bg)
        self.pose_answer_group = answer_group
        answer_group.grid(row=3, column=0, sticky="ew", pady=(2, 4))
        for value, label in (
                ("mark", "Tak, zaznaczę cztery rogi"),
                ("unavailable", "Nie mogę oznaczyć GT narożników")):
            radio = ttk.Radiobutton(answer_group, text=label,
                variable=self.pose_gt_answer, value=value, command=self._choose_pose_gt_answer,
                style="MobileReview.TRadiobutton")
            radio.pack(anchor="w", pady=2)
            self.pose_gt_radios.append(radio)
        # Compatibility: older synthetic GUI probes inspect this widget by name.
        self.pose_gt_unavailable_button = self.pose_gt_radios[1]
        # A button is useful only for correction of existing GT or cancellation
        # of an active draft. Initial drawing requires no additional CTA.
        self.pose_edit_button = ttk.Button(self.pose_gt_host,
            text="Popraw zaznaczenie", command=self._toggle_pose_gt_edit)
        self.pose_edit_button.grid(row=4, column=0, sticky="w", pady=(2, 3))
        self.pose_edit_button.grid_remove()
        self.pose_gt_reset = ttk.Button(self.pose_gt_host,
            text="Usuń zapisany obrys", command=self._clear_pose_gt)
        self.pose_gt_reset.grid(row=5, column=0, sticky="w", pady=(0, 3))
        self.pose_gt_reset.grid_remove()
        self.pose_gt_progress = self._label(self.pose_gt_host, "", font=("Segoe UI", 9), fg=self.muted)
        self.pose_gt_progress.grid(row=6, column=0, sticky="ew")
        self.pose_gt_metric = self._label(self.pose_gt_host, "",
                                          font=("Segoe UI", 10, "bold"), wraplength=370)
        self.pose_gt_metric.grid(row=7, column=0, sticky="ew", pady=(2, 0))
        self.pose_gt_instruction = self._label(self.pose_gt_host,
            "Kliknij cztery rogi w dowolnej kolejności. Program sam uporządkuje punkty "
            "i zapisze obrys po czwartym kliknięciu.",
            font=("Segoe UI", 9), fg=self.muted, wraplength=370)
        self.pose_gt_instruction.grid(row=8, column=0, sticky="ew", pady=(4, 0))
        self.pose_gt_instruction.grid_remove()
        self.pose_gt_host.grid_remove()  # 026: korekta jest bezpośrednio na zdjęciu.

        # MZ: zawsze widoczna predykcja (o ile mozna), GT i wyjasnienie.
        mz = self._review_section(right)
        mz.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        mz.columnconfigure(0, weight=1)
        self.mz_title = self._label(mz, "02 · SPRAWDZENIE NUMERU TABLICY",
                    font=("Segoe UI", 10, "bold"), fg=self.accent)
        self.mz_title.grid(row=0, column=0, sticky="ew")
        self.result_context = self._label(mz, "Numer odczytany przez aplikację:", font=("Segoe UI", 9), fg=self.muted)
        self.result_context.grid(row=1, column=0, sticky="ew")
        self.prediction_label = self._label(mz, "—", font=("Segoe UI", 16, "bold"))
        self.prediction_label.grid(row=2, column=0, sticky="ew", pady=(1, 5))
        self.crop_preview_button = ttk.Button(mz, text="Pokaż wyciętą tablicę",
                                               command=self._show_crop_preview)
        self.crop_preview_button.grid(row=3, column=0, sticky="w", pady=(0, 5))
        self.crop_preview_button.grid_remove()
        self._crop_preview_row = None
        self._crop_preview_window = None
        self.gt_prompt_label = self._label(mz,
            "Jaki numer widzisz na zdjęciu?", font=("Segoe UI", 10, "bold"))
        self.gt_prompt_label.grid(row=4, column=0, sticky="ew")
        self.gt_var = tk.StringVar()
        self.gt_entry = ttk.Entry(mz, textvariable=self.gt_var, font=("Segoe UI", 14), width=15)
        self.gt_entry.grid(row=5, column=0, sticky="ew")
        # Jedyna akcja operatora to wpisanie GT; zapis bez dodatkowego CTA.
        # Wynik MZ nie staje się GT bez niezależnego odczytania tablicy.
        self.gt_entry.bind("<Return>", lambda event: self.save_gt())
        self.gt_entry.bind("<FocusOut>", lambda event: self.save_gt())
        # One human-readable decision instead of two technical OCR radio modes.
        # Legacy 'assess' / 'exclude' persists unchanged in the review sidecar.
        self.subject_eval_choice = tk.StringVar(value="assess")
        self.subject_eval_radios = []  # compatibility with older lifecycle probes
        self.subject_unreadable_var = tk.BooleanVar(value=False)
        evaluable_group = tk.Frame(mz, bg=self.review_card_bg)
        self.mz_evaluable_group = evaluable_group
        evaluable_group.grid(row=6, column=0, sticky="ew", pady=(5, 3))
        self.subject_unreadable_button = ttk.Checkbutton(
            evaluable_group, text="Nie mogę odczytać numeru ze zdjęcia",
            variable=self.subject_unreadable_var,
            command=self._on_subject_unreadable_toggle,
            style="MobileReview.TCheckbutton")
        self.subject_unreadable_button.pack(anchor="w", pady=1)
        self.mz_exclusion_caption = self._label(mz,
                    "Przepisz numer ze zdjęcia, nie z wyniku aplikacji. Program porówna je automatycznie.",
                    font=("Segoe UI", 9), fg=self.muted, wraplength=370)
        self.mz_exclusion_caption.grid(row=7, column=0, sticky="ew", pady=(2, 6))
        self.mz_note_caption = self._label(mz,
                    "Dlaczego numer jest nieczytelny? (opcjonalnie)")
        self.mz_note_caption.grid(row=8, column=0, sticky="w")
        self.mz_note_caption.grid_remove()
        self.note_var = tk.StringVar()
        self.note_entry = ttk.Entry(mz, textvariable=self.note_var)
        self.note_entry.grid(row=9, column=0, sticky="ew", pady=(0, 3))
        self.note_entry.grid_remove()
        # This displays an older reference WITHOUT proposing to verify an
        # aborted or otherwise unexecuted inference. The stored GT stays intact.
        self.mz_archived_gt = self._label(mz, "", font=("Segoe UI", 10),
                                         fg=self.muted, wraplength=370)
        self.mz_archived_gt.grid(row=4, column=0, sticky="ew", pady=(3, 3))
        self.mz_archived_gt.grid_remove()
        self.gt_var.trace_add("write", self._schedule_autosave)
        self.note_var.trace_add("write", self._schedule_autosave)

        # Exact, operator-owned plate inventory for multi-plate MT invocations.
        # This replaces the *single-plate* GT fields only for that invocation.
        self._single_mz_host = mz
        multi = self._review_section(right)
        self._multi_host = multi
        multi.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        multi.columnconfigure(0, weight=1)
        self._label(multi, "02 · TABLICE GT NA ZDJĘCIU (KILKA)",
                    font=("Segoe UI",10,"bold"),fg=self.accent).grid(row=0,column=0,sticky="w")
        self._label(multi,"Podaj dokładną liczbę prawdziwych tablic. Każda ma własne GT, "
                    "powiązanie z MT i wynik MZ. Niewykrytą tablicę oznacz ręcznie na zdjęciu.",
                    font=("Segoe UI",9),fg=self.muted,wraplength=370).grid(row=1,column=0,sticky="ew")
        count_row = tk.Frame(multi,bg=self.review_card_bg)
        count_row.grid(row=2,column=0,sticky="ew",pady=(6,8))
        self._multi_count_var = tk.StringVar(value="")
        self._multi_count_spin = ttk.Spinbox(count_row, from_=2, to=32,
                  width=4, textvariable=self._multi_count_var)
        self._multi_count_spin.pack(side="left",padx=(0,6))
        ttk.Button(count_row,text="Zapisz liczbę tablic",command=self._multi_save_count).pack(side="left")
        self._multi_summary_var = tk.StringVar(value="Wymagany dokładny wykaz GT")
        self._label(multi, textvariable=self._multi_summary_var,
                    font=("Segoe UI",9,"bold"),wraplength=370).grid(row=3,column=0,sticky="ew",pady=(0,4))
        self._multi_plate_list = ttk.Treeview(multi,columns=("GT","MT","Numer"),
                  show="headings",height=4,selectmode="browse",style="MobileReview.Treeview")
        for col,width in (("GT",70),("MT",115),("Numer",120)):
            self._multi_plate_list.heading(col,text=col)
            self._multi_plate_list.column(col,width=width,stretch=True)
        self._multi_plate_list.grid(row=4,column=0,sticky="ew",pady=(0,6))
        self._multi_plate_list.bind("<<TreeviewSelect>>",self._multi_choose_plate)
        self._multi_plate_list.bind("<Double-1>", self._multi_open_selected_attempt)
        self._multi_selected_var = tk.StringVar(value="Wybierz tablicę GT")
        self._label(multi,textvariable=self._multi_selected_var,
                    font=("Segoe UI",9,"bold")).grid(row=5,column=0,sticky="ew",pady=(3,3))
        self._multi_match_values = {}
        self._multi_match_choice = ttk.Combobox(multi,state="readonly",width=32)
        self._multi_match_choice.grid(row=6,column=0,sticky="ew")
        actions = tk.Frame(multi, bg=self.review_card_bg)
        actions.grid(row=7, column=0, sticky="ew", pady=(3,9))
        ttk.Button(actions,text="Zapisz powiązanie GT ↔ MT",command=self._multi_save_match).pack(side="left")
        self._multi_open_button = ttk.Button(actions,text="Pokaż MT",command=self._multi_open_selected_attempt)
        self._multi_open_button.pack(side="left",padx=(6,0))
        self._label(multi,"Jaki numer jest na tej tablicy?",
                    font=("Segoe UI",9,"bold")).grid(row=8,column=0,sticky="w")
        self._multi_text_var=tk.StringVar(value="")
        ttk.Entry(multi,textvariable=self._multi_text_var).grid(row=9,column=0,sticky="ew")
        self._label(multi,"Jeśli numer jest nieczytelny, dlaczego?",
                    font=("Segoe UI",9)).grid(row=10,column=0,sticky="w")
        self._multi_reason_var=tk.StringVar(value="")
        ttk.Entry(multi,textvariable=self._multi_reason_var).grid(row=11,column=0,sticky="ew")
        ttk.Button(multi,text="Zapisz numer lub przyczynę nieczytelności",
                   command=self._multi_save_text).grid(row=12,column=0,sticky="ew",pady=(4,8))
        self._multi_bbox_button=ttk.Button(multi,text="Wskaż pominiętą tablicę na zdjęciu (GT)",
                                         command=self._multi_start_bbox)
        self._multi_bbox_button.grid(row=13,column=0,sticky="ew")
        self._multi_bbox_clear=ttk.Button(multi,text="Usuń ręczną ramkę GT (tylko tej tablicy)",
                                         command=self._multi_clear_bbox)
        self._multi_bbox_clear.grid(row=14,column=0,sticky="ew",pady=(3,2))
        self._label(multi,"Oznaczenie dwóch przeciwległych rogów tworzy ręczny bbox GT. "
                    "Nie zastępuje GT Pose i nie daje IoU/NMCE.",
                    fg=self.muted,font=("Segoe UI",9),wraplength=370).grid(row=15,column=0,sticky="ew",pady=4)
        multi.grid_remove()

        comparison = tk.Frame(right, bg=self.bg)
        comparison.grid(row=4, column=0, sticky="ew", pady=(1, 4))
        comparison.columnconfigure(0, weight=1)
        self.alignment_label = self._label(comparison, font=("Segoe UI", 9), wraplength=350)
        self.alignment_label.grid(row=0, column=0, sticky="ew")
        self.alignment_text = tk.Text(comparison, height=2, width=27, font=("Consolas", 10), bg=self.bg, fg=self.fg,
                                     bd=0, highlightthickness=0, wrap="none", state="disabled", takefocus=False)
        self.alignment_text.grid(row=1, column=0, sticky="ew")
        self.alignment_text.tag_configure("incorrect", foreground=self.browser.error)
        self.alignment_text.tag_configure("missing", foreground=self.browser.warning)
        self.alignment_text.tag_configure("extra", foreground=self.browser.warning)

        self._details_open = False
        self.record_details_toggle = ttk.Button(right, text="Pokaż zapisy techniczne (detekcje i cropy)",
                                                 command=self._toggle_record_details)
        self.record_details_toggle.grid(row=5, column=0, sticky="ew", pady=(3, 2))
        self.record_detail_host = tk.Frame(right, bg=self.bg)
        self.record_detail_host.grid(row=6, column=0, sticky="ew", pady=(0, 3))
        frame, self.record_tree = self._table(self.record_detail_host,
            ("Rodzaj", "Id", "MT / MZ", "Predykcja", "Weryfikacja"), (55, 62, 85, 98, 81), height=3)
        frame.grid(row=0, column=0, sticky="ew")
        self.record_detail_host.grid_remove()
        self.record_tree.bind("<<TreeviewSelect>>", self._select_record)
        self.record_tree.bind("<MouseWheel>", self._route_image_wheel, add="+")

        # Kazda kontrolka panelu odbiera rolke, niezaleznie od focusu; zoom obrazu osobno.
        self._bind_sidebar_wheel(sidebar_scroll)
        self._bind_sidebar_wheel(right)

        # Scientific results first; the legacy/raw tables remain available on demand.
        summary = tk.Frame(tabs, bg=self.bg)
        summary.rowconfigure(1, weight=1)
        summary.rowconfigure(2, weight=1)
        summary.columnconfigure(0, weight=1)
        tabs.add(summary, text="Analiza wyników")
        science_top = tk.Frame(summary, bg=self.bg)
        science_top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        science_top.columnconfigure(0, weight=1)
        self.science_caption = self._label(science_top, "Podsumowanie wyników sesji",
                                           font=("Segoe UI", 11, "bold"))
        self.science_caption.grid(row=0, column=0, sticky="ew")
        self.science_toggle = CanvasReviewEditor.make_action_button(
            science_top, "Tabela techniczna", self._toggle_scientific_details,
            compact=True)
        self.science_toggle.grid(row=0, column=1, sticky="e")
        # The canonical calculation stays in make_scientific_summary. This
        # dashboard is only a typed, read-only visualization of its result.
        summary.rowconfigure(2, weight=0)
        self.science_dashboard = MobileScienceSummaryPanel(
            summary, background=self.bg, surface=self.review_card_bg,
            foreground=self.fg, muted=self.muted, accent=self.accent,
            border=self.review_border, on_review=lambda: tabs.select(review))
        self.science_dashboard.frame.grid(row=1, column=0, rowspan=2,
                                          sticky="nsew", padx=(12, 12), pady=(3, 12))
        frame, self.stats_tree = self._table(summary, ("Zakres", "Miara", "Wynik"), (150, 480, 150), height=20)
        self._science_raw_host = frame
        self._science_raw_open = False
        # Only one of overview/raw technical tree is visible.
        details = tk.Frame(tabs, bg=self.bg)
        details.rowconfigure(1, weight=1)
        details.columnconfigure(0, weight=1)
        tabs.add(details, text="Warunki i źródła")
        reviewer = tk.Frame(details, bg=self.bg)
        reviewer.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self._label(reviewer, "Osoba weryfikująca (opcjonalny identyfikator)").grid(row=0, column=0, padx=5)
        self.reviewer_var = tk.StringVar()
        ttk.Entry(reviewer, textvariable=self.reviewer_var).grid(row=0, column=1)
        self._button(reviewer, "Zapisz", self.save_reviewer, row=0, column=2)
        self.mode_choice = ttk.Combobox(reviewer, state="disabled", values=list(MODE_LABELS.values()))
        details_top = tk.Frame(details, bg=self.bg)
        details_top.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        details_top.columnconfigure(0, weight=1)
        details_top.rowconfigure(1, weight=1)
        self._label(details_top, "Najważniejsze parametry eksperymentu",
                    font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="ew", pady=(4, 8))
        self.conditions_view = tk.Text(details_top, wrap="word", height=11,
                                        font=("Segoe UI", 10), bg=self.review_card_bg,
                                        fg=self.fg, relief="flat", padx=16, pady=13,
                                        highlightthickness=1, highlightbackground=self.review_border,
                                        state="disabled")
        self.conditions_view.grid(row=1, column=0, sticky="nsew")
        self.details_toggle = CanvasReviewEditor.make_action_button(details_top,
            "Pokaż pełną proweniencję techniczną", self._toggle_provenance_details,
            compact=True)
        self.details_toggle.grid(row=2, column=0, sticky="w", pady=(9, 5))
        # Source warnings and post-NMS diagnostics belong with the sources,
        # not in the two-row working header. Keep the existing update targets.
        self.notice_label = self._label(details_top, wraplength=1050, font=("Segoe UI", 9))
        self.notice_label.grid(row=3, column=0, sticky="ew", pady=3)
        self.notice_label.grid_remove()
        self.mt_postnms_info = self._label(details_top, "", font=("Segoe UI", 9),
                                         fg=self.muted, wraplength=1050)
        self.mt_postnms_info.grid(row=4, column=0, sticky="ew", pady=3)
        self.mt_postnms_info.grid_remove()
        frame, self.details_tree = self._table(details, ("Pole", "Wartość"), (340, 750), height=20)
        self._provenance_raw_host = frame
        self._provenance_raw_open = False
        self.status_var = tk.StringVar(value="Odczyt pełnego indeksu i kontrola SHA-256…")
        self._label(self.window, textvariable=self.status_var, font=("Segoe UI", 9), wraplength=1250).grid(
            row=3, column=0, sticky="ew", padx=12, pady=8)
        self._canvas_editor = CanvasReviewEditor(self, right, review, viewer)
        # Toolbar and shortcut A must call the SAME Canvas fit controller.
        # The original zoom_to_mt_detection remains for legacy report flows.
        self.plate_fit_button.configure(command=self._canvas_editor.fit_plate_toggle)
        self._canvas_editor.build_report_tab(tabs)
        from .mobile_f5a_assisted import install as install_f5a_assisted
        install_f5a_assisted(self, header, left)

    def _bind_sidebar_wheel(self, widget):
        """Lokalne zdarzenia potomkow: bez bind_all i bez przechwytywania zoomu obrazu."""
        widget.bind("<MouseWheel>", self._on_sidebar_wheel, add="+")
        widget.bind("<Button-4>", self._on_sidebar_wheel, add="+")
        widget.bind("<Button-5>", self._on_sidebar_wheel, add="+")
        for child in widget.winfo_children():
            self._bind_sidebar_wheel(child)

    def _on_sidebar_wheel(self, event):
        if getattr(event, "num", 0) in (4, 5):
            steps = -1 if event.num == 4 else 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return "break"
            steps = -max(1, round(delta / 120)) if delta > 0 else max(1, round(-delta / 120))
        self.review_sidebar_scroll.yview_scroll(steps, "units")
        return "break"

    def _toggle_record_details(self):
        self._details_open = not self._details_open
        if self._details_open:
            self.record_detail_host.grid()
        else:
            self.record_detail_host.grid_remove()
        self.record_details_toggle.configure(text=(
            "Ukryj zapisy techniczne" if self._details_open else
            "Pokaż zapisy techniczne (detekcje i cropy)"))

    def _submit(self, work, done, *, saving=False, refresh_controls=True):
        if self._closed:
            return
        if saving:
            self._saving = True
            self.gt_entry.configure(state="disabled")
            self.note_entry.configure(state="disabled")
            self.mode_choice.configure(state="disabled")
            self.invocation_choice.configure(state="disabled")
            self.pose_edit_button.configure(state="disabled")
            for radio in getattr(self, "pose_gt_radios", ()):
                radio.configure(state="disabled")
            self.pose_gt_reset.configure(state="disabled")
            for radio in self.subject_eval_radios + self.detection_radios + self.invocation_count_radios:
                radio.configure(state="disabled")
            for button in self.actions:
                button.configure(state="disabled")
        def task():
            try:
                self._results.put((done, work(), None, saving, refresh_controls))
            except Exception as error:
                self._results.put((done, None, str(error), saving, refresh_controls))
        self._executor.submit(task)

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                item = self._results.get_nowait()
                done, result, error, saving = item[:4]
                refresh_controls = item[4] if len(item)>4 else True
                if saving:
                    self._saving = False
                    self.gt_entry.configure(state="normal")
                    self.note_entry.configure(state="normal")
                    for button in self.actions:
                        button.configure(state="normal")
                if error:
                    if getattr(self, "_canvas_editor", None) is not None:
                        self._canvas_editor.in_refresh = False
                        self._canvas_editor._pending_observation_target = None
                    self.status_var.set(error)
                    if self._mt_corr_active and self._mt_corr_drag is None:
                        self._mt_corr_points = [list(p) for p in self._mt_corr_last_saved]
                        self._paint_image()
                    self._close_requested = False
                else:
                    done(result)
                    if saving and self._close_requested:
                        self.close()
                        return
                if self.session and refresh_controls:
                    self._sync_review_controls()
                if saving and not error and getattr(self, '_canvas_editor', None) is not None:
                    from .mobile_canvas_observation_blocks import resume
                    resume(self._canvas_editor)
        except queue.Empty:
            pass
        self._poll_id = self.window.after(40, self._poll)

    def _loaded(self, result):
        self.session, self.stats, self._review_was_converted = result
        for button in self.actions:
            button.configure(state="normal")
        if getattr(self.session,'_history_inspection_only',False):
            self.complete_button.configure(state='disabled')
        session = self.session
        self.reviewer_var.set(session.review.reviewer_id)
        self.session_label.configure(text=f"Sesja: {session.session_id}")
        context = dict(session.bundle.metadata, **session.bundle.report_payload)
        context.update(session.bundle.collection_session or context.get("research_collection", {}))
        state = str(context.get("state") or context.get("session_state") or context.get("status") or
                    session.bundle.experiment_session.get("status") or "").upper()
        notices = list(session.warnings)
        if state in {"PARTIAL", "ERROR"}:
            notices.insert(0, f"Sesja niekompletna: {state}. Wyniki obejmują wyłącznie zapisane dowody.")
        elif session.bundle.manifest.get("collection_complete") is False:
            notices.insert(0, "Sesja niekompletna. Wyniki obejmują wyłącznie zapisane dowody.")
        self._source_notices = notices
        self.notice_label.configure(
            text="\n".join(notices),
            fg=self.browser.warning if notices else self.muted)
        if notices:
            self.notice_label.grid()
        else:
            self.notice_label.grid_remove()
        self._subject_ids = {f"subject-{index}": key for index, key in enumerate(session.subjects)}
        self._refresh_subject_rows()
        self._refresh_stats()
        self._populate_details()
        self._sync_review_controls()
        self.status_var.set(f"Zapis automatyczny • rewizja {session.review.review_revision}"
                            + (" Przełączono wcześniejszą weryfikację na tryb asystowany; GT zachowano." if self._review_was_converted else ""))
        if getattr(session,'_history_inspection_only',False):
            self.status_var.set('Tylko inspekcja historii · bez zapisu danych')
        if session.scene_frames and getattr(self, '_canvas_editor', None):
            first = next(iter(self._canvas_editor.scene_tree_ids))
            self.subject_tree.selection_set(first)
            self.subject_tree.focus(first)
            self._canvas_editor.select_tree_row(first)
        elif self._subject_ids:
            first = next(iter(self._subject_ids))
            self.subject_tree.selection_set(first)
            self.subject_tree.focus(first)
        if getattr(self,'_f5a_a3',None):self._f5a_a3.loaded()

    def _multi_group(self):
        if not self.session or not self.current_invocation:
            return None
        return self.session.mt_invocations.get(self.current_invocation) if is_multi(self.session,self.current_invocation) else None

    def _sync_multi_controls(self):
        if getattr(self, "_canvas_editor", None) is not None:
            return
        if not hasattr(self,"_multi_host") or not self.session:
            return
        group=self._multi_group()
        if group is None:
            self._multi_host.grid_remove()
            self._single_mz_host.grid()
            self._multi_ui_group=""
            self._multi_active_gt=""
            self._multi_bbox_edit=False
            return
        self._single_mz_host.grid_remove()
        self._multi_host.grid()
        gid=self.current_invocation
        inv=inventory(self.session,gid)
        if self._multi_ui_group!=gid:
            self._multi_ui_group=gid
            if self._multi_active_group!=gid:
                self._multi_active_gt=""
            self._multi_bbox_edit=False
        if inv is not None:
            self._multi_count_var.set(str(inv["plate_count"]))
        elif not self._multi_count_var.get().strip().isdigit():
            self._multi_count_var.set("2")
        issues=group_issues(self.session,gid)
        n=inv["plate_count"] if inv else 0
        paired=sum(bool(p.get("attempt_id")) for p in inv["plates"]) if inv else 0
        self._multi_summary_var.set(f"GT: {n} • powiązania MT: {paired} • brakujące decyzje: {len(issues)}"
                                     if inv else "Podaj dokładną liczbę tablic GT (2–32)")
        selected=self._multi_active_gt
        self._multi_plate_list.delete(*self._multi_plate_list.get_children())
        if inv:
            for p in inv["plates"]:
                aid=p.get("attempt_id") or ""
                row=self.session.attempts.get(aid,{})
                index=row.get("mt_detection_index")
                target=f"MT {int(index)+1}" if aid and index not in (None,"") else "Pominięta"
                status=p.get("gt_text") or ("Nieczytelna" if p.get("text_exclusion_reason") else "—")
                self._multi_plate_list.insert("","end",iid=p["plate_id"],values=(p["plate_id"],target,status))
            selected=selected if selected and self._multi_plate_list.exists(selected) else inv["plates"][0]["plate_id"]
            self._multi_active_gt=selected
            self._multi_plate_list.selection_set(selected)
            self._multi_plate_list.focus(selected)
            self._multi_choose_plate()
        else:
            self._multi_selected_var.set("Utwórz wykaz tablic, następnie powiąż wskazania MT")
            self._multi_match_choice.configure(values=())
            self._multi_match_choice.set("")
            self._multi_open_button.configure(state="disabled")

    def _multi_open_selected_attempt(self, event=None):
        """Navigate to an already persisted GT/MT match; never save draft selections."""
        group = self._multi_group()
        inv = inventory(self.session, self.current_invocation) if group else None
        plate = next((p for p in inv["plates"] if p["plate_id"] == self._multi_active_gt), None) if inv else None
        aid = str(plate.get("attempt_id") or "") if plate else ""
        if not aid:
            self.status_var.set("Wybrana tablica GT nie ma zapisanego powiązania MT")
            return "break"
        self._multi_focus_attempt(aid)
        return "break"

    def _multi_choose_plate(self,event=None):
        group=self._multi_group()
        if group is None:
            return
        inv=inventory(self.session,self.current_invocation)
        selection=self._multi_plate_list.selection()
        if not inv or not selection:
            return
        pid=selection[0]
        plate=next((p for p in inv["plates"] if p["plate_id"]==pid),None)
        if plate is None:
            return
        self._multi_active_gt=pid
        self._multi_selected_var.set(f"Wybrana tablica: {pid} • jej własny numer GT i przypisanie")
        self._multi_text_var.set(str(plate.get("gt_text") or ""))
        self._multi_reason_var.set(str(plate.get("text_exclusion_reason") or ""))
        pairs={"Brak detekcji MT — pominięta tablica": ""}
        for aid,row in sorted(multi_detection_records(group).items(),key=lambda a:int(a[1].get("mt_detection_index") or 0)):
            index=int(row.get("mt_detection_index") or 0)+1
            pairs[f"Wskazanie MT #{index} ({aid[-8:]})"]=aid
        self._multi_match_values=pairs
        self._multi_match_choice.configure(values=tuple(pairs))
        current=str(plate.get("attempt_id") or "")
        self._multi_match_choice.set(next((name for name,aid in pairs.items() if aid==current),next(iter(pairs))))
        self._multi_open_button.configure(state="normal" if current and current in self.session.attempts else "disabled")
        self._multi_bbox_button.configure(state="normal" if not current else "disabled")
        if self._image is not None:
            self._paint_image()
        self._multi_bbox_clear.configure(state="normal" if plate.get("bbox_mt_px") else "disabled")
        if self._multi_bbox_edit and (not self._multi_active_gt or current):
            self._multi_bbox_edit=False

    def _multi_save_count(self):
        group=self._multi_group()
        if group is None or self._saving:
            return
        try:
            n=int(self._multi_count_var.get())
        except ValueError:
            self.status_var.set("Podaj liczbę całkowitą 2–32")
            return
        self._change(lambda:set_count(self.session,self.current_invocation,n),save_draft=False)

    def _multi_save_match(self):
        if self._multi_group() is None or self._saving or not self._multi_active_gt:
            return
        gid,pid=self.current_invocation,self._multi_active_gt
        aid=self._multi_match_values.get(self._multi_match_choice.get())
        if aid is None:
            self.status_var.set("Wybierz rzeczywiste wskazanie MT z listy")
            return
        self._change(lambda:assign_attempt(self.session,gid,pid,aid),save_draft=False)

    def _multi_save_text(self):
        if self._multi_group() is None or self._saving or not self._multi_active_gt:
            return
        gid,pid=self.current_invocation,self._multi_active_gt
        gt,reason=self._multi_text_var.get(),self._multi_reason_var.get()
        self._change(lambda:set_plate_text(self.session,gid,pid,gt,reason),save_draft=False)

    def _multi_start_bbox(self):
        group=self._multi_group()
        inv=inventory(self.session,self.current_invocation) if group else None
        plate=next((p for p in inv["plates"] if p["plate_id"]==self._multi_active_gt),None) if inv else None
        if not plate or plate.get("attempt_id") or self._saving:
            return
        row=self._current_mt_attempt()
        size=tuple(float(row[k]) for k in ("input_width","input_height")) if row else ()
        if (self._image is None or self._image_entry not in group.evidence_entries or
                self._image_entry != row.get("mt_input_evidence_entry") or self._image_source_size != size):
            self.status_var.set("Przed ręcznym zaznaczeniem pominiętej tablicy wybierz pełne wejście MT")
            self.show_invocation_input()
            return
        self._cancel_mt_correction()
        self._multi_bbox_edit=True
        self._leave_pose_gt_edit_on_navigation()
        self.canvas.configure(cursor="crosshair")
        self.status_var.set("Przeciągnij lewym przyciskiem od jednego rogu pominiętej tablicy do przeciwległego")

    def _multi_clear_bbox(self):
        if self._multi_group() is None or self._saving or not self._multi_active_gt:
            return
        gid,pid=self.current_invocation,self._multi_active_gt
        self._change(lambda:clear_missed_bbox(self.session,gid,pid),save_draft=False)

    def _multi_focus_attempt(self,attempt_id):
        if not attempt_id or not self.session:
            return
        row=self.session.attempts.get(attempt_id,{})
        if not row:
            return
        owning=row.get("subject_key")
        if owning!=self.subject_key:
            self.subject_key=owning
            self._multi_requested_attempt=attempt_id
            self._render_subject()
            return
        iid=next((iid for iid,r in self._record_ids.items() if r==("attempt",attempt_id)),None)
        if iid:
            self.record_tree.selection_set(iid)
            self.record_tree.focus(iid)
            self._select_record()

    def _multi_choose_child(self, child):
        subject_key,gid,pid,aid=child
        self._multi_active_group=gid
        self._multi_active_gt=pid
        self._multi_ui_group=gid
        if not aid:
            group=self.session.mt_invocations.get(gid)
            aid=str(group.records[0].get("id") or "") if group and group.records else ""
        if aid:
            self._multi_focus_attempt(aid)
        if self._multi_group() is not None:
            self._sync_multi_controls()

    def _focus_multi_gt(self, pid):
        if not self.session or not self.current_invocation:
            return
        cid = next((iid for iid, data in self._multi_child_targets.items()
                    if data[1] == self.current_invocation and data[2] == pid), None)
        if cid and self.subject_tree.exists(cid):
            self.subject_tree.selection_set(cid)
            self.subject_tree.focus(cid)
            self.subject_tree.see(cid)
            self._select_subject()

    def _status_yscroll(self, first, last):
        if getattr(self,"_status_scrollbar",None) is not None:
            self._status_scrollbar.set(first,last)
        self._schedule_status_pills()

    def _schedule_status_pills(self):
        if getattr(self,"_status_paint_scheduled",False):
            return
        self._status_paint_scheduled=True
        self._status_paint_after = self.subject_tree.after_idle(self._render_status_pills)

    def _render_status_pills(self):
        self._status_paint_after=None
        self._status_paint_scheduled=False
        if not self.subject_tree.winfo_exists():
            return
        columns=("MT", "Wsk.", "GT", "Rogi", "Form.")
        selected=set(self.subject_tree.selection())
        visible=[]
        def visit(parent):
            for iid in self.subject_tree.get_children(parent):
                if self.subject_tree.bbox(iid, "MT"):
                    visible.append(iid)
                if self.subject_tree.item(iid,"open"):
                    visit(iid)
        visit("")
        wanted={}
        for iid in visible:
            values=self.subject_tree.item(iid,"values")
            if len(values)<6:continue
            for col, value in zip(columns,values[1:6]):
                box=self.subject_tree.bbox(iid,col)
                if not box or box[2]<20:continue
                wanted[(iid,col)]=(str(value),box,iid in selected)
        spare=[]
        for key, widget in list(self._status_pills.items()):
            if key not in wanted:
                widget.place_forget()
                spare.append(widget)
                del self._status_pills[key]
        palette={"✓":("#E1F3E6","#206C41"),
                 "○":("#FFF1D4","#8C5C0C"),
                 "◐":("#FFF1D4","#8C5C0C"),
                 "—":("#E9EDF0","#657781"),
                 "×":("#FDECEF","#AB3545")}
        for key,(value,box,is_selected) in wanted.items():
            x,y,w,h=box
            bg,fg=palette.get(value,("#E9EDF0","#657781"))
            # Canvas status colors remain meaningful on selection. The legacy
            # standalone matrix retains its prior blue selection contract.
            if is_selected and (not getattr(self, "_canvas_editor", None) or value not in palette):
                bg,fg="#C2E9FD","#07557D"
            if key not in self._status_pills:
                widget = spare.pop() if spare else tk.Label(
                    self.subject_tree,font=("Segoe UI",11,"bold"),bd=0,relief="flat")
                # Rebinding is essential when a pooled badge serves another row.
                widget.bind("<Button-1>",lambda e, iid=key[0]: self._choose_status_row(iid))
                self._status_pills[key]=widget
            widget=self._status_pills[key]
            widget.configure(text=value,bg=bg,fg=fg)
            bw=min(26,max(18,w-6)); bh=max(16,min(22,h-5))
            widget.place(x=x+(w-bw)//2,y=y+(h-bh)//2,width=bw,height=bh)
        for widget in spare:
            widget.destroy()  # bounded by visible table rows, not session length

    def _choose_status_row(self,iid):
        if self.subject_tree.exists(iid):
            self.subject_tree.selection_set(iid)
            self.subject_tree.focus(iid)
            self._select_subject()

    def _refresh_subject_rows(self, prepared=None):
        self._subject_stats = {row["subject_key"]: row for row in self.stats["subjects"]}
        geometry = prepared['geometry'] if prepared is not None else _pose_geometry_rows(self.session)
        self._science_data = (prepared['science'] if prepared is not None else
                              make_scientific_summary(self.session, self.stats, geometry))
        pose_map = {row["attempt_id"]: row for row in geometry}
        if getattr(self, "_canvas_editor", None) is not None:
            self._canvas_editor.refresh_tree()
            return
        previous=self.subject_tree.selection()
        self._multi_child_targets={}
        for index, (iid, key) in enumerate(self._subject_ids.items(), 1):
            statuses = observation_verification_cells(self.session, key, pose_map)
            # The check marks mean that a decision/document exists, NOT that
            # the model recognized the number correctly. Human GT/corners are
            # separate columns and optional corners never block Form.
            active = [self.session.mt_invocations.get(self.session.invocation_by_attempt.get(aid))
                      for aid in self.session.subjects[key].get("attempts", ())]
            excluded = bool(active) and all(group is None or group.cancelled or
                                            not group.executed or group.execution_failed
                                            for group in active)
            # Cancelled and technically failed attempts are outside quality
            # denominators. A historical GT can exist, but do not present it
            # as a freshly verified MZ result in the status matrix.
            if excluded:
                statuses = tuple("—" if idx == 2 else cell for idx, cell in enumerate(statuses))
            from ..ranking.mobile_mt_postdedup import suppressed_subject
            raw_only_subject = suppressed_subject(self.session, self.session.subjects[key])
            values = (f"{index:02d}",) + statuses
            tag = ("review_excluded" if excluded else
                   "review_done" if statuses[-1] == "✓" else "review_pending")
            if self.subject_tree.exists(iid):
                self.subject_tree.item(iid, text=f"Obserwacja {index:02d}" +
                                       (" · przerwana" if excluded else " · RAW odrzucony przez NMS" if raw_only_subject else ""),
                                       values=values,tags=(tag,))
            else:
                self.subject_tree.insert("", "end", iid=iid,
                    text=f"Obserwacja {index:02d}" + (" · przerwana" if excluded else " · RAW odrzucony przez NMS" if raw_only_subject else ""),
                    values=values,tags=(tag,),open=True)
            # One GT plate = one child, even when multiple GT plates share one image.
            # For a shared invocation only its first subject owns the subrows.
            for child_iid in self.subject_tree.get_children(iid):
                self.subject_tree.delete(child_iid)
            groups=sorted({self.session.invocation_by_attempt.get(aid) for aid in self.session.subjects[key].get("attempts",())
                           if aid in self.session.invocation_by_attempt})
            for gid in groups:
                if not gid or not is_multi(self.session,gid):
                    continue
                group=self.session.mt_invocations[gid]
                if key!=sorted(group.subject_keys)[0]:
                    continue
                inv=inventory(self.session,gid)
                if inv is None:
                    cid=f"multi::{iid}::{gid}::setup"
                    self.subject_tree.insert(iid,"end",iid=cid,text="GT: podaj liczbę",values=("","○","○","—","○","○"))
                    self._multi_child_targets[cid]=(key,gid,"","")
                    continue
                linked=set()
                for plate in inv["plates"]:
                    pid=plate["plate_id"]
                    aid=plate.get("attempt_id") or ""
                    linked.add(aid)
                    verified=bool(aid or plate.get("bbox_mt_px"))
                    text_done=bool(plate.get("gt_text") or plate.get("text_exclusion_reason"))
                    idx=self.session.attempts.get(aid,{}).get("mt_detection_index")
                    name=f"{pid} ↔ MT {int(idx)+1}" if aid and idx not in (None,"") else f"{pid} · pominięta"
                    cid=f"multi::{iid}::{gid}::{pid}"
                    self.subject_tree.insert(iid,"end",iid=cid,text=name,
                           values=("", "✓", "✓" if verified else "○", "✓" if text_done else "—", "—", "✓" if verified else "○"))
                    self._multi_child_targets[cid]=(key,gid,pid,aid)
                for aid,row in sorted(multi_detection_records(group).items(),key=lambda z:int(z[1].get("mt_detection_index") or 0)):
                    if aid in linked:continue
                    from ..ranking.mobile_verification_policy import selected_visual_state
                    state=selected_visual_state(self.session.review.attempt_annotations.get(aid,{}))
                    idx=int(row.get("mt_detection_index") or 0)+1
                    cid=f"multi::{iid}::{gid}::mt{idx}"
                    self.subject_tree.insert(iid,"end",iid=cid,text=f"MT {idx} · {'fałszywe' if state=='not_plate' else 'bez GT'}",
                         values=("","✓","✓" if state=="not_plate" else "○","—","—","✓" if state=="not_plate" else "○"))
                    self._multi_child_targets[cid]=(key,gid,"",aid)
            self.subject_tree.item(iid,open=True)
        if previous and self.subject_tree.exists(previous[0]):
            self.subject_tree.selection_set(previous[0])
        self._schedule_status_pills()

    def _select_subject(self, event=None):
        selected = self.subject_tree.selection()
        clicked = selected[0] if selected else ""
        if (getattr(self, "_canvas_editor", None) is not None
                and self._canvas_editor.select_tree_row(clicked)):
            return
        child = self._multi_child_targets.get(clicked)
        key = self._subject_ids.get(clicked) or (child[0] if child else None)
        if not key:
            return
        if key == self.subject_key:
            if child:
                self._multi_choose_child(child)
            return
        if getattr(self, "_canvas_editor", None) is not None and not self._canvas_editor.prepare_navigation():
            return
        if self._saving:
            self._pending_subject = key
            return
        self.save_gt()
        if self._saving:
            self._pending_subject = key
            return
        self._cancel_mt_correction()
        self._leave_pose_gt_edit_on_navigation()
        self._multi_bbox_edit=False
        self._multi_drag_start=None
        self._close_crop_preview()
        self._pose_auto_token = None
        self.subject_key = key
        if child:
            self._multi_active_gt=child[2]
            self._multi_requested_attempt=child[3]
        self._render_subject()
        if child:
            self._multi_choose_child(child)
        # Selecting a DIFFERENT observation starts a new questionnaire at the
        # first question. Autosave/refresh of the same record keeps scroll.
        try:
            self.review_sidebar_scroll.yview_moveto(0)
        except (tk.TclError, AttributeError):
            pass

    def _render_subject(self):
        key = self.subject_key
        if not key:
            return
        subject = self.session.subjects[key]
        decision = self.session.review.subjects.get(key, {})
        self._rendering = True
        self.gt_var.set(self.session.subject_draft(key))
        self.note_var.set(decision.get("note", ""))
        self._rendering = False
        number = list(self.session.subjects).index(key) + 1
        legacy = " • tożsamość legacy" if subject["legacy_identity"] else ""
        stats = self._subject_stats.get(key, {})
        invocations = {self.session.invocation_by_attempt[item] for item in subject["attempts"]}
        invocation_count = sum(self.session.mt_invocations[item].executed for item in invocations) if self.session.attempts_available else None
        self.subject_label.configure(text=f"Obserwacja {number} z {len(self.session.subjects)} — wybrana tablica na zdjęciu")
        visible = self.session.predictions_visible(key)
        self._sync_review_controls()
        if not visible:
            self.alignment_label.configure(text="Weryfikacja zaślepiona: wynik MZ pozostaje ukryty. Aby przejść na prostą ocenę z predykcjami, użyj przycisku u góry.")
            self.alignment_text.configure(state="normal")
            self.alignment_text.delete("1.0", "end")
            self.alignment_text.configure(state="disabled")
        old = self.current_record
        # Odbudowa tabeli po radiobuttonie NIE moze resetowac kadru ani zoomu.
        self._retained_image_on_refresh = ((old, self._image_entry)
            if old and self._image is not None and self._image_entry else None)
        self.record_tree.delete(*self.record_tree.get_children())
        self._record_ids = {}
        for kind, keys, records in (("attempt", subject["attempts"], self.session.attempts), ("sample", subject["samples"], self.session.samples)):
            for record_id in keys:
                row = records[record_id]
                iid = f"record-{len(self._record_ids)}"
                self._record_ids[iid] = (kind, record_id)
                annotation = (self.session.review.attempt_annotations if kind == "attempt" else self.session.review.sample_annotations).get(record_id, {})
                state = "Nie tablica" if annotation.get("is_plate") is False else "Pominięta" if annotation.get("evaluable") is False else annotation.get("plate_visibility", "GT zapisane" if kind == "sample" and decision.get("ground_truth") else "Do weryfikacji")
                state = {"visible": "Widoczna", "invisible": "Niewidoczna", "uncertain": "Niejednoznaczna"}.get(state, state)
                if self.session.cancelled(row):
                    state = "Próba przerwana"
                invocation = self.session.invocation_by_attempt.get(record_id if kind == "attempt" else row.get("attempt_id"), "")
                group = self.session.mt_invocations.get(invocation)
                if group and group.execution_failed and not group.cancelled:
                    state = "Błąd wykonania MT"
                index = row.get("mt_detection_index")
                label = f"Detekcja {int(index) + 1}" if kind == "attempt" and index not in (None, "") else "Rekord MT" if kind == "attempt" else "Crop"
                identity = f"{invocation} / {record_id}" if invocation and invocation != record_id else record_id
                self.record_tree.insert("", "end", iid=iid, values=(label, identity,
                    self._record_stages(row, group), (row.get("prediction") or "brak odczytu") if visible else HIDDEN_PREDICTION, state))
        if self._record_ids:
            default = next((iid for iid, value in self._record_ids.items() if value[0] == "sample"), next(iter(self._record_ids)))
            iid = next((iid for iid, value in self._record_ids.items() if value == old), default)
            if self._record_ids[iid] != old:
                self._retained_image_on_refresh = None
            self.current_record = None
            requested=getattr(self,"_multi_requested_attempt","")
            if requested:
                iid=next((candidate for candidate, data in self._record_ids.items()
                          if data==("attempt",requested)),iid)
                self._multi_requested_attempt=""
            self.record_tree.selection_set(iid)
            self.record_tree.focus(iid)
        else:
            self.current_record = None
            self.prediction_label.configure(text="Brak zapisanego wyniku MZ")
            self._image = None
            self._paint_image()

    def _record_stages(self, row, group):
        mt = ("Próba przerwana" if group and group.cancelled else
              "Błąd wykonania MT" if group and group.execution_failed else STAGE_LABELS.get(row.get("mt_status"), ""))
        # MZ status itself can disclose a no-read before GT, even with OCR hidden.
        mz = STAGE_LABELS.get(row.get("mz_status"), "") if self.session.predictions_visible(self.subject_key) else ""
        return " / ".join(stage for stage in (mt, mz) if stage)

    def _select_record(self, event=None):
        # TreeviewSelect is queued by Tk. A full-scene selection deliberately
        # has no subject; an old MT/sample event must not leave that scene or
        # finish an edit before its record context has been validated.
        if self.session is None:
            return
        subject = self.session.subjects.get(self.subject_key)
        if subject is None:
            return
        selected = self.record_tree.selection()
        record = self._record_ids.get(selected[0]) if selected else None
        if not record or record == self.current_record:
            return
        kind, key = record
        members = subject.get({"attempt": "attempts", "sample": "samples"}.get(kind, ""), ())
        if key not in members:
            return
        if getattr(self, "_canvas_editor", None) is not None and not self._canvas_editor.prepare_navigation():
            return
        # Selecting another MT detection on a full source frame must not
        # unexpectedly leave the source image or reset its zoom/pan position.
        keep_source = str(getattr(self, "_source_frame_keep_on_selection", "") or "")
        self._source_frame_keep_on_selection = ""
        if (self._mt_corr_active and str(self.session.attempts.get(record[1] if record[0] == "attempt" else
                self.session.samples.get(record[1], {}).get("attempt_id"), {}).get("id") or "") != self._mt_corr_attempt):
            self._cancel_mt_correction()
        retained = self._retained_image_on_refresh
        self._retained_image_on_refresh = None
        # A real record change ends GT editing. A refresh of the SAME record
        # (after autosave) must preserve the edit state and Zoom/Pan.
        if not retained or retained[0] != record:
            self._cancel_mt_correction()
            self._leave_pose_gt_edit_on_navigation()
            self._close_crop_preview()
            self._pose_auto_token = None
        self.current_record = record
        kind, key = record
        row = (self.session.attempts if kind == "attempt" else self.session.samples)[key]
        self._image_rows = []
        attempt = row if kind == "attempt" else self.session.attempts.get(row.get("attempt_id"), {})
        self.current_invocation = self.session.invocation_by_attempt.get(attempt.get("id"), "")
        group = self.session.mt_invocations.get(self.current_invocation)
        self._image_rows = _review_image_sources(
            kind, row, attempt, group,
            [self.session.samples[sid] for sid in subject["samples"]],
            self.session.entry_names,
        )
        self.image_choice.configure(values=[title for title, _ in self._image_rows])
        self._sync_source_frame_button()
        source_index = _primary_review_image_index(self._image_rows, attempt, group)
        if getattr(self, "_canvas_editor", None) is not None:
            source_index = next((i for i, (_, candidate) in enumerate(self._image_rows)
                if candidate.get("source_frame_view")), source_index)
        if keep_source:
            saved_index = next((i for i, (_, item) in enumerate(self._image_rows)
                if item.get("source_frame_view") and item.get("image_entry") == keep_source), None)
            if saved_index is not None:
                source_index = saved_index
        if source_index is not None:
            self.image_choice.current(source_index)
        else:
            self.image_choice.set("Brak zdjęcia powiązanego z obszarem")
        self._sync_detected_area_navigation()
        self._sync_crop_preview_source(attempt, row if kind == "sample" else None)
        self.record_label.configure(text="Zdjęcie do weryfikacji")
        # Ocena odczytu MZ nie znika tylko dlatego, ze operator wybral rekord MT.
        # Przy probie MT pokazujemy powiazany crop, jesli istnieje.
        result_row = row
        if kind == "attempt":
            for candidate in (self.session.samples[sid]
                              for sid in subject["samples"]):
                if candidate.get("attempt_id") == key:
                    result_row = candidate
                    break
        self._result_row = result_row
        gt = self.session.review.subjects.get(self.subject_key, {}).get("ground_truth", "")
        prediction = str(result_row.get("prediction") or "").strip()
        mz_status = str(result_row.get("mz_status", "")).upper()
        self._result_prediction = prediction if mz_status != "NOT_RUN" else ""
        if not self.session.predictions_visible(self.subject_key):
            self.prediction_label.configure(text="Ukryta — weryfikacja zaślepiona")
        elif self.session.cancelled(result_row):
            self.prediction_label.configure(text="Próba przerwana")
        elif mz_status == "NOT_RUN":
            self.prediction_label.configure(text="MZ nie uruchomiono")
        else:
            self.prediction_label.configure(text=prediction or "Brak odczytu")
        self.result_context.configure(text=(
            "MZ — brak ukończonego pomiaru w przerwanej próbie" if self.session.cancelled(result_row) else
            "MZ nie uruchomiono — poniżej wpisz niezależny GT" if mz_status == "NOT_RUN" else
            "Wynik powiązanego cropa MZ" if result_row is not row else
            "Wynik wybranego zapisu MZ"))
        self.gt_prompt_label.configure(text=(
            "Numer z tablicy (GT operatora; MZ nie uruchomiono)" if mz_status == "NOT_RUN" else
            "Numer z tablicy (GT operatora, niezależny od MZ)"))
        self.alignment_text.configure(state="normal")
        self.alignment_text.delete("1.0", "end")
        if not self.session.predictions_visible(self.subject_key):
            self.alignment_label.configure(text="Wpisz GT z obrazu — zapis jest automatyczny.")
        elif self.session.cancelled(result_row):
            self.alignment_label.configure(text=(
                "Próba przerwana. Nie weryfikujemy tu wyniku OCR (CER/Exact nie dotyczy); "
                "ewentualny GT służy tylko dokumentacji."))
        elif mz_status == "NOT_RUN":
            self.alignment_label.configure(text="MZ nie został uruchomiony w tej próbie. Oceń widoczność tablicy dla MT.")
        elif self.session.review.subjects.get(self.subject_key, {}).get("evaluable") is False:
            self.alignment_label.configure(text=(
                "Znaki nieczytelne — odczyt MZ/CER i skuteczność ALPR poza oceną. "
                "Wykrycie i obrys MT wymagają osobnej weryfikacji."))
        elif gt:
            match = align_plate_text(gt, prediction)
            outcome = "Brak odczytu" if not match.prediction else "Poprawny odczyt" if match.exact_match else "Niepoprawny odczyt"
            self.alignment_label.configure(text=f"{outcome} • predykcja: {prediction or '—'}\n"
                f"Poprawnie rozpoznane: {match.correct_characters} • błędnie rozpoznane: {match.incorrect_characters} • "
                f"brakujące: {match.missing_characters} • nadmiarowe: {match.extra_characters} • CER: {_display(match.cer, rate=True)}")
            for prefix, field in (("GT    : ", "ground_truth"), ("Odczyt: ", "prediction")):
                self.alignment_text.insert("end", prefix)
                for item in match.alignment:
                    self.alignment_text.insert("end", (item[field] or "·") + " ", item["kind"])
                self.alignment_text.insert("end", "\n")
        else:
            self.alignment_label.configure(text=f"Predykcja: {prediction or 'brak odczytu'} • wpisz GT dla tej tablicy.")
        self.alignment_text.configure(state="disabled")
        self._sync_review_controls()
        if retained and retained[0] == record and self._image is not None:
            # Rewizja GT/MT nie zmienia obrazu ani transformacji Zoom/Pan.
            for index, (_, candidate) in enumerate(self._image_rows):
                entry = candidate.get("image_entry") or candidate.get("evidence_entry")
                if (entry == retained[1] == self._image_entry and
                        (index == _primary_review_image_index(self._image_rows, attempt, group) or
                         (getattr(self, "_canvas_editor", None) is not None and candidate.get("source_frame_view")))):
                    self.image_choice.current(index)
                    self._set_image_help()
                    self._paint_image()
                    return
        if (keep_source and self._image is not None and
                keep_source == self._image_entry and
                self._image_rows and 0 <= self.image_choice.current() < len(self._image_rows) and
                self._image_rows[self.image_choice.current()][1].get("source_frame_view")):
            self._paint_image()  # keep zoom/pan on this same immutable Bitmap
            self._sync_source_frame_button()
            return
        self._load_image()

    def _sync_review_controls(self):
        if not self.session:
            return
        if getattr(self, "_canvas_editor", None) is not None:
            self._canvas_editor.sync()
            return
        state = self._mz_review_state()
        if state == "cancelled":
            intro = ("Próba przerwana. Obejrzyj zapisane zdjęcie i dane, ale nie "
                     "weryfikuj wyniku MT ani MZ: przetwarzanie nie zostało ukończone.")
        elif state == "technical":
            intro = ("Brak ukończonego wykonania modelu. Dostępne dowody służą "
                     "diagnostyce, nie ocenie skuteczności MT lub MZ.")
        elif state == "not_run":
            intro = ("Zweryfikuj wynik MT. Model MZ nie uruchomił się dla tej próby; "
                     "GT tekstu można zapisać osobno, ale nie weryfikujemy OCR.")
        else:
            intro = ("Zweryfikuj wskazanie MT i odczyt MZ. GT narożników służy "
                     "wyłącznie do opcjonalnego pomiaru IoU/NMCE.")
        warnings = getattr(self, "_source_notices", [])
        self.notice_label.configure(text=intro + ("\n" + "\n".join(warnings) if warnings else ""),
                                    fg=self.browser.warning if warnings else self.fg)
        can_show = bool(self.subject_key and self.session.predictions_visible(self.subject_key))
        # Brak CTA kopiującego MZ do GT. Jedynym źródłem GT pozostaje operator.
        self._sync_subject_eval_choice()
        self.invocation_error_label.grid_remove()
        if self.current_invocation:
            group = self.session.mt_invocations[self.current_invocation]
            if group.cancelled:
                self.invocation_label.configure(text="Status próby: przerwana — poza oceną skuteczności MT")
                self.invocation_error_label.configure(
                    text=_cancelled_mt_explanation(group), fg=self.muted)
                self.invocation_error_label.grid()
            elif not group.executed:
                self.invocation_label.configure(text="MT nie uruchomiono — poza oceną skuteczności")
                self.invocation_error_label.configure(
                    text="Brak ukończonego wywołania modelu. Nie przypisuj mu błędu wykrywania.",
                    fg=self.muted)
                self.invocation_error_label.grid()
            elif group.execution_failed:
                self.invocation_label.configure(text="Błąd wykonania MT — poza oceną skuteczności")
                error = " ".join("; ".join(group.execution_errors).split())
                error = error[:157] + "…" if len(error) > 160 else error
                self.invocation_error_label.configure(
                    text=f"Wywołanie technicznie nieukończone; nie jest to wynik jakości modelu. {error}",
                    fg=self.browser.warning)
                self.invocation_error_label.grid()
            else:
                self.invocation_label.configure(
                    text=f"Surowe wskazania MT: {group.detection_count} "
                         "(liczba GT wymaga niezależnej decyzji)")
            annotation = self.session.invocation_annotation(self.current_invocation)
            self.invocation_choice.set("Brak wcześniejszej weryfikacji liczby tablic" if annotation.get("decision_source") == "legacy_cardinality_unknown"
                                       else VISIBILITY_LABELS.get(annotation.get("visible_plate_count"), "Oceń całe wejście MT"))
            self.invocation_count_value.set(annotation.get("visible_plate_count", ""))
            if group.execution_failed:
                self.invocation_choice.set("Poza weryfikacją MT")
            allowed = not (self._saving or group.cancelled or not group.executed or group.execution_failed)
            self.invocation_choice.configure(state="readonly" if allowed else "disabled")
            for radio in self.invocation_count_radios:
                radio.configure(state="normal" if allowed else "disabled")
            # No phantom questions that are disabled and can never be answered:
            # display a lifecycle explanation instead for excluded invocations.
            for widget in getattr(self, "_mt_scored_controls", ()):
                if allowed:
                    widget.grid()
                else:
                    widget.grid_remove()
        else:
            self.invocation_label.configure(text="MT: nie ma zapisu wywołania dla tego obrazu")
            self.invocation_choice.set("")
            self.invocation_choice.configure(state="disabled")
            self.invocation_count_value.set("")
            for radio in self.invocation_count_radios:
                radio.configure(state="disabled")
            for widget in getattr(self, "_mt_scored_controls", ()):
                widget.grid_remove()
        self._sync_detection_choice()
        self._sync_mt_visual_choice()
        self._sync_mt_postnms_info()
        self._sync_pose_gt_controls()
        self._sync_multi_controls()
        self._sync_mz_panel_state()

    def _sync_mt_postnms_info(self):
        label = getattr(self, "mt_postnms_info", None)
        if label is None or not self.session or not self.current_invocation:
            if label is not None:
                label.grid_remove()
            return
        from ..ranking.mobile_mt_postdedup import post_nms_proof, eligible_suppressed_id
        proof = post_nms_proof(self.session, self.current_invocation)
        if proof is None:
            label.grid_remove()
            return
        annotation = self.session.invocation_annotation(self.current_invocation)
        physical = annotation.get("visible_plate_count")
        suppressed = eligible_suppressed_id(self.session, self.current_invocation)
        suffix = ("Potwierdzono jedną tablicę w dotychczasowej weryfikacji jednoobiektowej; "
                  "odrzucona surowa ramka nie tworzy dodatkowego GT ani końcowego FP." if suppressed else
                  "Liczbę tablic GT musi ustalić operator na pełnym obrazie. "
                  "Dowód NMS nie modyfikuje istniejącego canvasu ani danych GT.")
        label.configure(text=(
            f"MT na tej klatce: {proof['raw']} surowe -> {proof['after_dedup']} po NMS "
            f"({proof['suppressed']} odrzucone); IoU dwóch ramek = {proof['bbox_iou']*100:.2f}%. "
            f"Liczba GT wskazana przez operatora: {physical or 'brak'}. {suffix}"))
        label.grid()

    def _mz_review_state(self):
        """One source of truth for operator actions in the MZ section.

        Cancelled/error/not-executed = read-only even if historical GT exists.
        MZ NOT_RUN in an active MT attempt may retain *independent* GT for
        diagnostics; that never becomes a measured OCR result.
        """
        if not self.session or not self.current_record:
            return "empty"
        attempt = self._current_mt_attempt()
        group = self.session.mt_invocations.get(self.current_invocation)
        row = getattr(self, "_result_row", {}) or attempt
        if ((group and group.cancelled) or self.session.cancelled(attempt)
                or self.session.cancelled(row)):
            return "cancelled"
        if group and (not group.executed or group.execution_failed):
            return "technical"
        mz_status = str(row.get("mz_status") or attempt.get("mz_status") or "").upper()
        if mz_status in {"", "NOT_RUN", "UNKNOWN"}:
            return "not_run"
        return "measured"

    def _sync_mz_panel_state(self):
        """Render lifecycle and editable actions together, not separately.

        This changes widgets only. Previously recorded GT and sidecars are
        never rewritten or cleared when a cancelled attempt is selected.
        """
        if not self.session or not self.current_record or not hasattr(self, "mz_title"):
            return
        if self._multi_group() is not None:
            return  # Per-plate GT panel owns the active multi-plate workflow.
        state = self._mz_review_state()
        offline = state in {"cancelled", "technical"}
        controls = (self.gt_prompt_label, self.gt_entry, self.mz_evaluable_group,
                    self.mz_exclusion_caption, self.mz_note_caption, self.note_entry)
        if offline:
            for widget in controls:
                widget.grid_remove()
            old = self.session.review.subjects.get(self.subject_key, {})
            gt = str(old.get("ground_truth") or "").strip()
            self.mz_archived_gt.configure(text=(
                f"Wcześniej zapisany numer GT: {gt}. Zachowany tylko w dokumentacji; "
                "nie jest wynikiem weryfikacji MZ tej próby." if gt else
                "Brak numeru GT. Nie trzeba go uzupełniać w tej próbie."))
            self.mz_archived_gt.grid()
            self.gt_entry.configure(state="disabled")
            self.note_entry.configure(state="disabled")
            self.subject_unreadable_button.configure(state="disabled")
            self.mz_title.configure(text="02 · MZ — POZA WERYFIKACJĄ")
            if state == "cancelled":
                self.result_context.configure(text="Aplikacja przerwała próbę przed ukończeniem odczytu.")
                self.prediction_label.configure(text="Próba przerwana")
            else:
                self.result_context.configure(text="Model nie zakończył wykonania; brak wyniku do weryfikacji.")
                self.prediction_label.configure(text="Brak wyniku MZ")
            self.alignment_label.configure(text=(
                "W tej próbie nie obliczamy poprawności OCR, Exact Match ani CER. "
                "Nie należy traktować braku pomiaru jako błędnego odczytu."))
            self.alignment_text.configure(state="normal")
            self.alignment_text.delete("1.0", "end")
            self.alignment_text.configure(state="disabled")
            self.crop_preview_button.configure(text="Pokaż zachowany obraz (bez weryfikacji)")
            return
        self.mz_archived_gt.grid_remove()
        for widget in controls:
            widget.grid()
        is_not_run = state in {"not_run", "empty"}
        self.mz_title.configure(text=("02 · NUMER TABLICY — MZ NIE URUCHOMIONO" if is_not_run
                                      else "02 · SPRAWDZENIE NUMERU TABLICY"))
        self.gt_prompt_label.configure(text="Jaki numer widzisz na zdjęciu?")
        self.subject_unreadable_button.configure(state="disabled" if self._saving else "normal")
        self._sync_mz_readability_details(is_not_run=is_not_run)
        if is_not_run:
            self.result_context.configure(text=(
                "MZ nie został uruchomiony. Nie ma odczytu znaków do weryfikacji."))
            self.prediction_label.configure(text="MZ nie uruchomiono")
            self.alignment_label.configure(text=(
                "Możesz zapisać numer widocznej tablicy jako niezależny GT do analizy "
                "pominięcia MT/ALPR. Nie powstaje przez to wynik MZ ani CER."))
            self.crop_preview_button.configure(text="Pokaż zachowany obraz (bez odczytu MZ)")
        else:
            self.crop_preview_button.configure(text="Pokaż obraz tablicy użyty przez MZ")

    def _visual_target(self):
        if not self.session or not self.current_record:
            return "", {}
        attempt = self._current_mt_attempt()
        aid = str(attempt.get("id") or "")
        group = self.session.mt_invocations.get(self.current_invocation)
        if not (aid and group and group.executed and not group.cancelled and
                not group.execution_failed and not self.session.cancelled(attempt) and
                str(attempt.get("mt_status") or "").upper() in DETECTION_STATUSES):
            return "", {}
        return aid, attempt

    def _sync_mt_visual_choice(self):
        if getattr(self, "_canvas_editor", None) is not None:
            return
        if not hasattr(self, "mt_visual_choice"):
            return
        aid, attempt = self._visual_target()
        annotation = self.session.review.attempt_annotations.get(aid, {}) if aid else {}
        self.mt_visual_choice.set(selected_visual_state(annotation))
        for control in getattr(self, "mt_visual_radios", ()):
            control.configure(state="normal" if aid and not self._saving else "disabled")
        if hasattr(self, "mt_visual_host"):
            self.mt_visual_host.grid_remove()
        if hasattr(self, "mt_visual_status"):
            statuses = {"not_plate": "Fałszywe wskazanie MT — obszar nie jest tablicą.",
                        "imprecise": "Tablica potwierdzona — korekta obrysu wspomagana predykcją MT.",
                        "accurate": "Tablica potwierdzona — obrys MT zaakceptowany wizualnie.",
                        "uncertain": "Brak możliwości weryfikacji tego obrysu."}
            self.mt_visual_status.configure(text=(statuses.get(self.mt_visual_choice.get()) or
                "Zweryfikuj wskazanie MT, wybierając opcję nad zdjęciem."))
        for control in getattr(self, "mt_visual_overlay_buttons", ()):
            try:
                selected = control.cget("value") == self.mt_visual_choice.get()
                control.configure(bg="#1686BD" if selected else "#10212b",
                                  relief=tk.SUNKEN if selected else tk.FLAT)
            except tk.TclError:
                pass
        # Direct manipulation is the default for an imprecise MT detection.
        # Re-enter on record/image selection (but do not discard an active
        # four-click draft, or restart while the asynchronous save is pending).
        if (aid and selected_visual_state(annotation) == "imprecise"
                and self._image is not None
                and self._image_entry == attempt.get("mt_input_evidence_entry")
                and not self._saving and not self._mt_redraw_active
                and not self.pose_edit_var.get()
                and (not self._mt_corr_active or self._mt_corr_attempt != aid)):
            self._start_mt_correction(aid)
        elif (selected_visual_state(annotation) != "imprecise"
                and self._mt_corr_active and not self._mt_redraw_active):
            self._cancel_mt_correction()

    def save_mt_visual_choice(self):
        aid, attempt = self._visual_target()
        if not aid or self._saving:
            self._sync_mt_visual_choice()
            return
        if self._image is None or self._image_entry != attempt.get("mt_input_evidence_entry"):
            self.status_var.set("Wybierz pełne wejście MT przed weryfikacją tego wskazania.")
            self._sync_mt_visual_choice()
            return
        state = self.mt_visual_choice.get()
        if not state:
            self._sync_mt_visual_choice()
            return
        original = _linked_pose_quad_on_mt_input(self.session, attempt)
        if state in {"accurate", "imprecise"} and original is None:
            self.status_var.set("MT nie zapisał poprawnych czterech rogów. Nie tworzymy obrysu z prostokąta bbox.")
            self._sync_mt_visual_choice()
            return
        old = self.session.review.attempt_annotations.get(aid, {})
        if state == "not_plate" and old.get("pose_gt_quad_mt_input_px"):
            self.status_var.set("Istnieje wcześniejszy GT narożników. Wyjaśnij konflikt przed oznaczeniem «Nie tablica».")
            self._sync_mt_visual_choice()
            return
        if (old.get("pose_mt_correction_v1") and state != "imprecise" and
                selected_visual_state(old) == "imprecise"):
            if not messagebox.askyesno("Zmiana weryfikacji wskazania MT",
                    "To wskazanie ma zapisaną korektę obrysu. Po zmianie odpowiedzi "
                    "korekta pozostanie w historii, ale nie będzie aktywnym wynikiem. "
                    "Czy kontynuować?", parent=self.window):
                self._sync_mt_visual_choice()
                return
        # An archived assisted correction is never erased by changing the choice.
        # Only an explicitly requested new correction requires completion;
        # legacy imprecise reviews remain historically valid.
        if state != "imprecise":
            self._cancel_mt_correction()
        decision = visual_decision(state)
        decision["pose_correction_required_v1"] = state == "imprecise"
        should_write = any(old.get(key) != value for key, value in decision.items())
        if should_write:
            self._change(lambda: self.session.annotate("attempt", aid, **decision),
                         save_draft=False,
                         after=(lambda aid=aid: self._start_mt_correction(aid)) if state == "imprecise" else None)
        elif state == "imprecise":
            self._start_mt_correction(aid)
        else:
            self._sync_mt_visual_choice()

    def _valid_mt_correction(self, aid, attempt, model):
        decision = self.session.review.attempt_annotations.get(aid, {})
        return correction_for_attempt(decision, attempt_id=aid,
             source_sha=self.session.review.source_archive_sha256,
             image_entry=attempt["mt_input_evidence_entry"],
             size=(float(attempt["input_width"]), float(attempt["input_height"])),
             prediction=model)

    def _start_mt_correction(self, aid):
        if getattr(self, "_mt_redraw_active", False):
            self._cancel_mt_correction()
        target, attempt = self._visual_target()
        if target != aid or self._saving:
            return
        if self._image is None or self._image_entry != attempt.get("mt_input_evidence_entry"):
            return
        model = _linked_pose_quad_on_mt_input(self.session, attempt)
        if model is None:
            self.status_var.set("Brak poprawnego Pose MT — nie można korygować nieistniejącego obrysu.")
            return
        try:
            saved = self._valid_mt_correction(aid, attempt, model)
        except (ValueError, KeyError, TypeError) as exc:
            self.status_var.set(f"Nie można otworzyć korekty: {exc}")
            return
        self._leave_pose_gt_edit_on_navigation()
        self._multi_bbox_edit = False
        self._mt_corr_active = True
        self._mt_corr_attempt = aid
        self._mt_corr_points = [list(p) for p in (saved or model)]
        self._mt_corr_last_saved = [list(p) for p in (saved or model)]
        self._mt_corr_drag = None
        self.canvas.configure(cursor="crosshair")
        self._paint_image()
        self.status_var.set("Przeciągnij bursztynowe uchwyty narożników albo wnętrze obrysu. Zapis po puszczeniu myszy. Esc: anuluj szkic.")

    def _resume_mt_correction(self):
        """Explicitly re-enter drag editing for a previously verified detection."""
        aid, _attempt = self._visual_target()
        if aid:
            self._start_mt_correction(aid)
            self._sync_mt_visual_choice()

    def _start_mt_redraw(self, event=None):
        if str(getattr(self, "_image_entry", "")).startswith("samples/frames/"):
            self.status_var.set("Pełna klatka jest na razie tylko do odczytu. Wróć do wejścia MT, aby korygować obrys.")
            return "break"
        """D or Shift+click begins replacement of the sole operator polygon.

        The operator already saw the MT prediction in this assisted review;
        the resulting annotation is NOT a blinded/independent scientific GT.
        """
        if self._mt_redraw_active:
            self._cancel_mt_correction()
            self.status_var.set("Anulowano niezapisany obrys. Weryfikacja modelu pozostała bez zmian.")
            self._sync_mt_visual_choice()
            return "break"
        aid, attempt = self._visual_target()
        if not aid or self._saving or self._image is None:
            return
        if (self._image_entry != attempt.get("mt_input_evidence_entry") or
                self._image_source_size != (float(attempt["input_width"]),
                                            float(attempt["input_height"]))):
            self.status_var.set("Wybierz oryginalny obraz wejścia MT tej detekcji.")
            return
        decision = self.session.review.attempt_annotations.get(aid, {})
        if selected_visual_state(decision) != "imprecise":
            self.status_var.set("Najpierw wybierz «Niedokładny» na zdjęciu.")
            return
        if _linked_pose_quad_on_mt_input(self.session, attempt) is None:
            self.status_var.set("MT nie zapisał czterech narożników; nie tworzymy ich z bbox.")
            return
        # Historical independent GT is immutable. This redraw creates a
        # SEPARATE assisted correction of this same detection, and only the
        # active operator polygon is rendered (no duplicate green GT frame).
        if decision.get("pose_mt_correction_v1") and not messagebox.askyesno(
                "Nowy obrys ręczny",
                "Istnieje już zapisana korekta obrysu. Nowy obrys zastąpi aktywną korektę, "
                "ale poprzednia zostanie zachowana w historii tej detekcji. Kontynuować?",
                parent=self.window):
            return
        self._cancel_mt_correction()  # cancel only unsaved drag, never stored review
        self._leave_pose_gt_edit_on_navigation()
        self._multi_bbox_edit = False
        self._mt_redraw_active = True
        self._mt_redraw_attempt = aid
        self._mt_redraw_points = []
        self.canvas.configure(cursor="crosshair")
        self._sync_mt_visual_choice()
        self._paint_image()
        self.status_var.set("Nowy obrys operatora: kliknij cztery rogi tablicy (0/4). "
                            "Po czwartej poprawnej pozycji zastąpi on poprzednią korektę; Esc anuluje szkic.")
        return "break"

    def _add_mt_redraw_point(self, event):
        """A click records one source-MT-space point; the fourth commits atomically."""
        if not self._mt_redraw_active or self._saving:
            return False
        aid, attempt = self._visual_target()
        if (not aid or aid != self._mt_redraw_attempt or self._image is None
                or self._image_entry != attempt.get("mt_input_evidence_entry")):
            self._cancel_mt_correction()
            return False
        point = self._pose_canvas_to_mt_input(event, aid)
        if point is None:
            self.status_var.set("Kliknij wewnątrz rzeczywistego obrazu wejścia MT.")
            return True
        # Avoid adding two corners when a double-click repeats the same point.
        previous = [self._pose_mt_to_canvas(p, aid) for p in self._mt_redraw_points]
        if any(math.dist(v, (event.x, event.y)) < 8 for v in previous):
            self.status_var.set("Róg jest zbyt blisko poprzedniego. Kliknij następny róg tablicy.")
            return True
        candidate = self._mt_redraw_points + [[float(point[0]), float(point[1])]]
        if len(candidate) < 4:
            self._mt_redraw_points = candidate
            self._paint_image()
            self.status_var.set(f"Nowy obrys: {len(candidate)}/4 rogów. Kliknij następny róg.")
            return True
        try:
            size = (float(attempt["input_width"]), float(attempt["input_height"]))
            quad = _ordered_operator_gt_quad(candidate, *size)
            model = _linked_pose_quad_on_mt_input(self.session, attempt)
            correction = make_correction(
                attempt_id=aid, source_sha=self.session.review.source_archive_sha256,
                image_entry=attempt["mt_input_evidence_entry"], size=size,
                prediction=model, corrected=quad)
            correction["drawing_method"] = "four_manual_corner_clicks_v1"
            old = self.session.review.attempt_annotations.get(aid, {})
            # No automatic promotion of a traced MT prediction to independent GT.
            # Existing independent and historic GT are never replaced here.
        except (ValueError, TypeError, KeyError) as exc:
            self.status_var.set(f"Nie zapisano obrysu: {exc}. Wskaż ponownie czwarty róg.")
            return True  # keep the first three valid clicks for correction
        payload = dict(pose_mt_correction_v1=correction,
                       pose_correction_required_v1=True)
        # Compatibility with 027/028: historical assisted GT was also stored
        # in the Pose GT slot. Keep its single historical representation in
        # sync; NEVER touch a GT whose provenance is independent/unknown.
        if old.get("pose_gt_review_mode") == "assisted_manual_redraw_from_mt_v1":
            previous_gt = old.get("pose_gt_quad_mt_input_px")
            previous_record = old.get("pose_mt_correction_v1")
            if previous_gt and isinstance(previous_record, dict):
                saved_polygon = previous_record.get("operator_quad_mt_input_px")
                if saved_polygon != previous_gt:
                    self.status_var.set("Konflikt historii GT: nie zastępujemy obrysu. Odśwież sesję.")
                    return True
            payload["pose_gt_quad_mt_input_px"] = [
                [round(float(x), 4), round(float(y), 4)] for x, y in quad]
        if old.get("pose_mt_correction_v1"):
            history = old.get("pose_mt_correction_history_v1")
            history = list(history) if isinstance(history, list) else []
            previous = old["pose_mt_correction_v1"]
            if previous not in history:
                history.append(previous)
            payload["pose_mt_correction_history_v1"] = history
        # One review revision, one exact attempt; never alter source MT or OCR.
        self._mt_redraw_active = False
        self._mt_redraw_attempt = ""
        self._mt_redraw_points = []
        self.canvas.configure(cursor="hand2")
        self._sync_mt_visual_choice()
        self._paint_image()
        self._change(lambda: self.session.annotate("attempt", aid, **payload),
                     save_draft=False,
                     after=self._resume_mt_correction)
        return True

    def _cancel_mt_correction(self, event=None):
        if getattr(self, "_mt_redraw_active", False):
            self._mt_redraw_active = False
            self._mt_redraw_attempt = ""
            self._mt_redraw_points = []
            if getattr(self, "canvas", None) is not None:
                self.canvas.configure(cursor="hand2")
                self._paint_image()
            # Cancel only the unsaved draft; direct editing resumes without
            # requiring the old artificial CTA.
            self._sync_mt_visual_choice()
            return "break"
        if not getattr(self, "_mt_corr_active", False):
            return None
        self._mt_corr_active = False
        self._mt_corr_attempt = ""
        self._mt_corr_points = []
        self._mt_corr_drag = None
        self._mt_corr_last_saved = []
        if getattr(self,"canvas",None) is not None:
            self.canvas.configure(cursor="hand2")
            self._paint_image()
        return "break"

    def _delete_mt_operator_correction(self, event=None):
        """Delete only the selected operator correction, never MT predictions/GT.

        Like Z2's delete action this targets only the selected object. The
        previous correction stays in audit history, and the form becomes
        incomplete until the imprecise outline is re-verified.
        """
        aid, attempt = self._visual_target()
        if (not aid or self._saving or self._image is None
                or self._image_entry != attempt.get("mt_input_evidence_entry")
                or selected_visual_state(self.session.review.attempt_annotations.get(aid, {})) != "imprecise"):
            return "break"
        current = self.session.review.attempt_annotations.get(aid, {})
        record = current.get("pose_mt_correction_v1")
        if not isinstance(record, dict):
            self.status_var.set("Nie ma zapisanej korekty do usunięcia. Obrys MT pozostaje bez zmian.")
            return "break"
        if not messagebox.askyesno("Usuń korektę operatora",
                "Usunąć wyłącznie aktywną korektę tej tablicy? "
                "Predykcja MT i niezależny GT pozostaną niezmienione; "
                "poprzednia korekta zostanie zachowana w historii.", parent=self.window):
            return "break"
        history = list(current.get("pose_mt_correction_history_v1") or [])
        if record not in history:
            history.append(record)
        payload = dict(pose_mt_correction_v1=None,
                       pose_mt_correction_history_v1=history,
                       pose_correction_required_v1=True)
        # Revisions 027/028 could mirror an assisted redraw into Pose GT.
        # Clear ONLY this mirror, never an independent/older ground truth.
        if current.get("pose_gt_review_mode") == "assisted_manual_redraw_from_mt_v1":
            mirror = current.get("pose_gt_quad_mt_input_px")
            if mirror is not None and mirror != record.get("operator_quad_mt_input_px"):
                self.status_var.set("Niezgodna geometria historycznego GT. Nie usunięto korekty.")
                return "break"
            payload.update(pose_gt_schema=None, pose_gt_image_entry=None,
                           pose_gt_input_size_px=None, pose_gt_quad_mt_input_px=None,
                           pose_gt_review_mode=None)
        self._cancel_mt_correction()
        self._change(lambda: self.session.annotate("attempt", aid, **payload),
                     save_draft=False)
        return "break"

    def _mt_corr_canvas_points(self, attempt):
        if not self._mt_corr_points or not self._image or not self._view_center:
            return []
        w, h = float(attempt["input_width"]), float(attempt["input_height"])
        cx, cy = self._view_center
        scale=self._view_pixel_scale
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        return [((x*self._image.width/w-cx)*scale+cw/2,
                 (y*self._image.height/h-cy)*scale+ch/2) for x,y in self._mt_corr_points]

    def _update_mt_edit_cursor(self, event):
        """Z2-like direct manipulation: hover handle vs whole-quad movement."""
        if not getattr(self, "_mt_corr_active", False) or self._saving:
            return
        if self._mt_corr_drag is not None:
            return
        aid, attempt = self._visual_target()
        if aid != self._mt_corr_attempt:
            return
        points = self._mt_corr_canvas_points(attempt)
        target = hit_target((event.x, event.y), points, radius=17)
        cursor = "hand2" if target and target[0] == "corner" else (
            "fleur" if target and target[0] == "body" else "crosshair")
        self.canvas.configure(cursor=cursor)

    def _mt_corr_begin_drag(self, event):
        aid, attempt = self._visual_target()
        if not (self._mt_corr_active and aid == self._mt_corr_attempt and not self._saving):
            return False
        xy = self._pose_canvas_to_mt_input(event, aid)
        if xy is None:
            return False
        target = hit_target((event.x,event.y), self._mt_corr_canvas_points(attempt), radius=17)
        if target is None:
            return False
        self._mt_corr_drag = (target, xy, [list(p) for p in self._mt_corr_points])
        self.canvas.configure(cursor="fleur")
        return True

    def _mt_corr_move_drag(self, event):
        if not self._mt_corr_drag:
            return False
        aid, attempt = self._visual_target()
        if aid != self._mt_corr_attempt:
            return False
        target, start, initial = self._mt_corr_drag
        end = self._pose_canvas_to_mt_input(event, aid)
        if end is None:
            return True
        try:
            self._mt_corr_points = move_polygon(initial, target, start, end,
                        (float(attempt["input_width"]), float(attempt["input_height"])))
        except (ValueError, TypeError):
            # Preserve last valid draft until cursor returns to valid geometry.
            return True
        self._paint_image()
        return True

    def _mt_corr_finish_drag(self):
        if not self._mt_corr_drag:
            return False
        target, start, initial = self._mt_corr_drag
        self._mt_corr_drag = None
        aid, attempt = self._visual_target()
        if aid != self._mt_corr_attempt or self._saving:
            return True
        if not moved(initial, self._mt_corr_points):
            self._paint_image()
            return True
        model = _linked_pose_quad_on_mt_input(self.session, attempt)
        try:
            record = make_correction(attempt_id=aid,
                source_sha=self.session.review.source_archive_sha256,
                image_entry=attempt["mt_input_evidence_entry"],
                size=(float(attempt["input_width"]),float(attempt["input_height"])),
                prediction=model,corrected=self._mt_corr_points)
        except (ValueError, TypeError, KeyError) as exc:
            self._mt_corr_points = [list(p) for p in self._mt_corr_last_saved]
            self.status_var.set(f"Nie zapisano nieprawidłowej korekty: {exc}")
            self._paint_image()
            return True
        committed = [list(p) for p in self._mt_corr_points]
        decision = self.session.review.attempt_annotations.get(aid, {})
        payload = dict(pose_mt_correction_v1=record, pose_correction_required_v1=True)
        previous = decision.get("pose_mt_correction_v1")
        if isinstance(previous, dict):
            history = list(decision.get("pose_mt_correction_history_v1") or [])
            if previous not in history:
                history.append(previous)
            payload["pose_mt_correction_history_v1"] = history
            if previous.get("drawing_method") in {
                    "four_manual_corner_clicks_v1", "four_manual_corner_clicks_then_drag_v1"}:
                record["drawing_method"] = "four_manual_corner_clicks_then_drag_v1"
        # W wersji 027 obrys narysowany czterema kliknięciami zapisano też
        # jako GT asystowany. Dalsze przeciąganie musi zachować ich tożsamość.
        # Niezależny lub starszy GT NIGDY nie jest tutaj modyfikowany.
        if decision.get("pose_gt_review_mode") == "assisted_manual_redraw_from_mt_v1":
            existing_gt = decision.get("pose_gt_quad_mt_input_px")
            expected = self._mt_corr_last_saved
            if (not isinstance(existing_gt, (list, tuple)) or
                    len(existing_gt) != 4 or
                    any(math.dist([float(v) for v in a], [float(v) for v in b]) > 0.0006
                        for a, b in zip(existing_gt, expected))):
                self._mt_corr_points = [list(p) for p in self._mt_corr_last_saved]
                self.status_var.set("Konflikt z zapisanym obrysem operatora: odśwież weryfikację przed edycją.")
                self._paint_image()
                return True
            payload["pose_gt_quad_mt_input_px"] = [
                [round(float(x), 4), round(float(y), 4)] for x, y in committed]
        self._change(lambda: self.session.annotate("attempt", aid, **payload),
                     save_draft=False,
                     after=lambda: setattr(self, "_mt_corr_last_saved", committed))
        self._paint_image()
        return True

    def _paint_mt_visual_overlay(self):
        if not hasattr(self, "mt_visual_overlay") or not self.session or self._image is None:
            return
        aid, attempt = self._visual_target()
        if (not aid or self._image_entry != attempt.get("mt_input_evidence_entry") or
                self.pose_edit_var.get()):
            return
        width = max(1, self.canvas.winfo_width())
        self.canvas.create_window(max(30, width - 12), 9, window=self.mt_visual_overlay,
                                  anchor="ne", tags=("mt_visual_quick_verify",))

    def save_detection_choice(self):
        """Zapis jednej czytelnej oceny operatora dla detekcji/cropa.
        Nie zastępuje ona pytania o liczbę tablic na całym wejściu MT.
        """
        if not self.session or not self.current_record or self._saving:
            return
        kind, record_id = self._active_detection_target()
        row = (self.session.attempts if kind == "attempt" else self.session.samples).get(record_id, {})
        if kind == "attempt" and str(row.get("mt_status", "")).upper() not in DETECTION_STATUSES:
            return  # Bez detekcji nie ma obszaru, który można uznać za tablicę.
        choices = {
            "yes": {"plate_visibility": "visible", "is_plate": True, "evaluable": True},
            "no": {"plate_visibility": "invisible", "is_plate": False, "evaluable": False},
            "skip": {"plate_visibility": "uncertain", "is_plate": None, "evaluable": False},
        }
        answer = self.detection_choice.get()
        decision = choices.get(answer)
        if decision is None:
            return
        # Answering the presence question merely unlocks the next question.
        # Never hide the predicted outline before the operator has *explicitly*
        # answered that the corners are identifiable on this image.
        def after_save():
            if answer == "yes" and kind == "attempt":
                self.status_var.set("Tablica potwierdzona. Zweryfikuj dopasowanie obrysu; ręczny GT narożników jest opcjonalny.")
        if answer != "yes":
            self._leave_pose_gt_edit_on_navigation()
        saved = (self.session.review.attempt_annotations if kind == "attempt"
                 else self.session.review.sample_annotations).get(record_id, {})
        if kind == "attempt":
            if answer == "no" and saved.get("pose_gt_quad_mt_input_px"):
                self.status_var.set("Zapisany GT narożników jest sprzeczny z decyzją «Nie tablica». Wyjaśnij rozbieżność przed zmianą.")
                self._sync_review_controls()
                return
            if answer in {"no", "skip"} or (answer == "yes" and saved.get("is_plate") is not True):
                decision = dict(decision, pose_quality=None)
        if all(saved.get(name) == value for name, value in decision.items()):
            after_save()
            self._sync_review_controls()
            return
        self._change(lambda: self.session.annotate(kind, record_id, **decision), after=after_save)

    def _active_detection_target(self):
        """Na obrazie MT oceń próbę MT, a na cropie — sam crop MZ."""
        kind, record_id = self.current_record or (None, None)
        if kind == "sample" and self.session:
            sample = self.session.samples.get(record_id, {})
            attempt_id = str(sample.get("attempt_id") or "")
            group = self.session.mt_invocations.get(self.current_invocation)
            if (attempt_id in self.session.attempts and group
                    and self._image_entry in group.evidence_entries):
                return "attempt", attempt_id
        return kind, record_id

    def _pose_attempt_context(self):
        """Ocena polygonu zawsze należy do attempt_id, nigdy do cropa MZ."""
        if not self.session or not self.current_record or not self.current_invocation:
            return None
        kind, record_id = self.current_record
        sample = self.session.samples.get(record_id, {}) if kind == "sample" else None
        attempt_id = (str(sample.get("attempt_id") or "") if sample is not None else record_id)
        attempt = self.session.attempts.get(attempt_id)
        group = self.session.mt_invocations.get(self.current_invocation)
        if not attempt or not group or self.session.cancelled(attempt):
            return None
        if group.cancelled or not group.executed or group.execution_failed:
            return None
        if self._image_entry not in group.evidence_entries:
            return None
        if self._image_entry not in {entry.get("mt_input_evidence_entry") for entry in group.records}:
            return None
        try:
            valid_size = self._image_source_size == (
                float(attempt["input_width"]), float(attempt["input_height"]))
        except (KeyError, ValueError, TypeError):
            return None
        if not valid_size or str(attempt.get("mt_status") or "").upper() not in DETECTION_STATUSES:
            return None
        # Manual GT is independent of a four-corner MT prediction. BBOX is
        # never used as a synthetic model polygon.
        return attempt_id

    def _missed_plate_attempt_context(self):
        """One real missed plate: exact MT input, not a fictitious detection.

        One no-detection invocation has exactly one record; multiple visible
        plates need a separate future multi-GT annotation model. We do not
        guess which plate an unindexed GT belongs to.
        """
        if not self.session or not self.current_record or not self.current_invocation:
            return None
        kind, attempt_id = self.current_record
        if kind != "attempt":
            return None
        attempt = self.session.attempts.get(attempt_id)
        group = self.session.mt_invocations.get(self.current_invocation)
        if not attempt or not group or str(attempt.get("mt_status") or "").upper() != "NO_DETECTION":
            return None
        if (group.cancelled or not group.executed or group.execution_failed or
                group.detection_count != 0 or len(group.records) != 1 or
                str(group.records[0].get("id")) != attempt_id or
                self.session.cancelled(attempt)):
            return None
        getter = getattr(self.session, "invocation_annotation", None)
        if not callable(getter):
            return None
        verdict = getter(self.current_invocation)
        if (verdict.get("visible_plate_count") != "one" or verdict.get("evaluable") is False):
            return None
        entry = str(attempt.get("mt_input_evidence_entry") or "")
        if not entry or entry != self._image_entry or entry not in group.evidence_entries:
            return None
        try:
            size = float(attempt["input_width"]), float(attempt["input_height"])
        except (ValueError, TypeError, KeyError):
            return None
        if size[0] <= 0 or size[1] <= 0 or self._image_source_size != size or self._image is None:
            return None
        return attempt_id

    def _gt_target_is_confirmed(self, attempt_id):
        """Positive detected region OR one operator-confirmed missed plate."""
        if not attempt_id or not self.session:
            return False
        if self._missed_plate_attempt_context() == attempt_id:
            return True
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        return decision.get("is_plate") is True and decision.get("evaluable") is not False

    def _pose_gt_attempt_context(self):
        """Exact original MT input for the *same* attempt; never annotate a crop.

        The parent helper validates link, 4 Pose corners, status and dimensions.
        Here we additionally require the attempt's own explicit input evidence,
        not a different frame from the same MT invocation.
        """
        try:
            attempt_id = self._missed_plate_attempt_context() or self._pose_attempt_context()
        except (AttributeError, KeyError, TypeError):
            return None  # Synthetic older GUI probes may not have a complete session.
        if not attempt_id:
            return None
        attempt = self.session.attempts.get(attempt_id, {})
        if not attempt.get("mt_input_evidence_entry") or self._image_entry != attempt["mt_input_evidence_entry"]:
            return None
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        if decision.get("is_plate") is False or decision.get("evaluable") is False:
            return None
        return attempt_id

    def _ensure_pose_gt_points(self, attempt_id):
        """Load only manually annotated GT bound to the exact source/input size."""
        if not attempt_id:
            self._pose_gt_cache_token = None
            self._pose_gt_points = []
            return
        attempt = self.session.attempts[attempt_id]
        token = (attempt_id, str(attempt.get("mt_input_evidence_entry")))
        if self._pose_gt_cache_token == token:
            return
        self._pose_gt_cache_token = token
        self._pose_gt_points = []
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        try:
            width, height = float(attempt["input_width"]), float(attempt["input_height"])
            if (decision.get("pose_gt_schema") == POSE_GT_SCHEMA
                    and decision.get("pose_gt_image_entry") == token[1]
                    and decision.get("pose_gt_input_size_px") == [width, height]):
                self._pose_gt_points = list(validate_mt_quad(
                    decision.get("pose_gt_quad_mt_input_px"), width, height))
        except (ValueError, TypeError, KeyError):
            self._pose_gt_points = []

    def _sync_pose_gt_controls(self):
        if getattr(self, "_canvas_editor", None) is not None:
            return
        if getattr(self, "_mt_correction_ux_v026", False):
            self.pose_gt_host.grid_remove()
            return
        if not self.session:
            return
        attempt_id = self._pose_gt_attempt_context()
        self._ensure_pose_gt_points(attempt_id)
        decision = self.session.review.attempt_annotations.get(attempt_id, {}) if attempt_id else {}
        # For an MT input with no Pose, the record decision is still relevant.
        if not attempt_id and self.current_record:
            kind, key = self._active_detection_target()
            if kind == "attempt":
                decision = self.session.review.attempt_annotations.get(key, {})
        confirmed = self._gt_target_is_confirmed(attempt_id)
        if not confirmed:
            if self.pose_edit_var.get():
                self._leave_pose_gt_edit_on_navigation()
                self._paint_image()
            # Keep step 02 visible so the questionnaire never jumps 01 -> 03.
            # This panel is informational only; cancellation is NOT a negative
            # model prediction or an invitation to fabricate matching GT.
            self.pose_gt_host.grid()
            self.pose_gt_title.configure(text="02  ·  POMIAR OBRYSU (OPCJONALNIE)")
            group = self.session.mt_invocations.get(self.current_invocation)
            if group and group.cancelled:
                question = "Obrys poza weryfikacją — próba przerwana"
                hint = ("Obrys zapisany przez MT możesz obejrzeć na zdjęciu. "
                        "Próba nie weszła do pomiaru jakości; GT, IoU i NMCE "
                        "nie są wymagane.")
            elif group and (not group.executed or group.execution_failed):
                question = "Obrys poza oceną — brak ukończonego MT"
                hint = "Nie ma wyniku MT, który można porównać z GT."
            else:
                question = "Najpierw potwierdź tablicę w kroku 01"
                hint = "Jeśli MT wskazał tablicę, po jej potwierdzeniu możesz zaznaczyć cztery rogi GT."
            self.pose_gt_question.configure(text=question)
            self.pose_gt_hint.configure(text=hint)
            for widget in (self.pose_answer_group, self.pose_edit_button, self.pose_gt_reset,
                           self.pose_gt_progress, self.pose_gt_metric,
                           self.pose_gt_instruction):
                widget.grid_remove()
            return
        self.pose_gt_host.grid()
        self.pose_answer_group.grid()
        self.pose_gt_progress.grid()
        self.pose_gt_metric.grid()
        missed = (str(self.session.attempts[attempt_id].get("mt_status") or "").upper()
                  == "NO_DETECTION")
        self.pose_gt_title.configure(text=("02  ·  OBRYS POMINIĘTEJ TABLICY" if missed
                                      else "02  ·  POMIAR OBRYSU (OPCJONALNIE)"))
        self.pose_gt_question.configure(text=(
            "Czy możesz zaznaczyć cztery rogi tablicy pominiętej przez MT?" if missed
            else "Chcesz wyznaczyć cztery rogi GT do pomiaru IoU/NMCE?"))
        unavailable = bool(decision.get("pose_gt_unavailable_reason"))
        can_edit = bool(attempt_id and self._image is not None and not self._saving and not unavailable)
        if self.pose_edit_var.get() and not can_edit:
            self._leave_pose_gt_edit_on_navigation()
            self._paint_image()
        count = len(self._pose_gt_points)
        self.pose_gt_answer.set("unavailable" if unavailable else
                                "mark" if attempt_id and (count == 4 or self.pose_edit_var.get()) else "")
        for radio in self.pose_gt_radios:
            radio.configure(state="normal" if attempt_id and not self._saving else "disabled")
        show_edit = bool(can_edit and (count == 4 or self.pose_edit_var.get()))
        self.pose_edit_button.configure(state="normal" if show_edit else "disabled",
            text="Anuluj oznaczanie" if self.pose_edit_var.get() else
                 "Popraw zaznaczenie")
        if show_edit:
            self.pose_edit_button.grid()
        else:
            self.pose_edit_button.grid_remove()
        if can_edit and count == 4:
            self.pose_gt_reset.configure(state="normal")
            self.pose_gt_reset.grid()
        else:
            self.pose_gt_reset.configure(state="disabled")
            self.pose_gt_reset.grid_remove()
        if not attempt_id:
            source_row = None
            if self.current_record:
                kind, record_id = self.current_record
                if kind == "attempt":
                    source_row = self.session.attempts.get(record_id)
                elif kind == "sample":
                    sample = self.session.samples.get(record_id, {})
                    source_row = self.session.attempts.get(sample.get("attempt_id"))
            if source_row and self._image_entry == source_row.get("mt_input_evidence_entry"):
                _, reason = _pose_source_status(self.session, source_row)
                self.pose_gt_hint.configure(text=("Porównanie obrysów jest niedostępne dla tego zapisu. "
                    "Nadal możesz zweryfikować wykrycie i odczyt numeru."))
            else:
                self.pose_gt_hint.configure(text=(
                    "Obrys zweryfikujesz na pełnym zdjęciu. Wybierz «Wejście MT — bieżąca próba»."))
            self.pose_gt_progress.configure(text="Brak danych do pomiaru obrysu")
            self.pose_gt_metric.configure(text="Porównanie obrysów niedostępne")
            return
        attempt = self.session.attempts[attempt_id]
        has_pose = _linked_pose_quad_on_mt_input(self.session, attempt) is not None
        if unavailable:
            self.pose_gt_hint.configure(text=(
                "Nie wykonano ręcznego GT narożników. Weryfikacja wykrycia MT pozostaje niezależna."))
        elif missed:
            self.pose_gt_hint.configure(text=(
                "Rzeczywisty obrys GT został zapisany. MT nie wykrył tej tablicy." if count == 4 else
                "Wskaż rogi tablicy na zdjęciu. MT jej nie wykrył; nie będzie porównania IoU/NMCE."))
        elif count == 4:
            self.pose_gt_hint.configure(text=(
                "Zapisano cztery rogi. Możesz poprawić zaznaczenie." if has_pose else
                "Zapisano cztery rogi GT. Brak obrysu modelu MT do porównania."))
        elif self.pose_edit_var.get():
            self.pose_gt_hint.configure(text=(
                "Zaznacz cztery rogi. Obrys modelu jest chwilowo ukryty, "
                "aby nie sugerować położenia punktów."))
        elif has_pose:
            self.pose_gt_hint.configure(text=(
                "Wskaż własne rogi GT, a porównamy je z obrysem MT."))
        else:
            self.pose_gt_hint.configure(text=(
                "Możesz zapisać prawdziwe rogi GT również bez obrysu modelu MT."))
        self.pose_gt_progress.configure(text=(
            "GT obrysu: niewyznaczony (decyzja operatora)" if unavailable else
            f"Zaznaczone rogi: {count}/4" + (
                " • przeciągnij punkt, aby poprawić" if count == 4 and self.pose_edit_var.get() else
                " • zapisano" if count == 4 else
                " • kliknij kolejny róg na zdjęciu" if self.pose_edit_var.get() else "")))
        instruction = getattr(self, "pose_gt_instruction", None)
        if instruction is not None:
            if self.pose_edit_var.get():
                instruction.grid()
            else:
                instruction.grid_remove()
        if hasattr(self, "pose_gt_title"):
            self.pose_gt_title.configure(fg=self.accent if self.pose_edit_var.get() else self.fg)
        if count == 4:
            if not has_pose:
                self.pose_gt_metric.configure(text=(
                    "GT zapisane • MT nie wykrył tablicy • IoU/NMCE: —" if missed else
                    "GT zapisane • brak obrysu Pose MT • IoU/NMCE: —"))
            else:
                try:
                    width, height = float(attempt["input_width"]), float(attempt["input_height"])
                    pred = _linked_pose_quad_on_mt_input(self.session, attempt)
                    measured = compare_mt_pose_to_gt(pred, self._pose_gt_points, width, height)
                    self.pose_gt_metric.configure(text=(
                        f"Błąd położenia rogów: {measured['nmce_pct']:.2f}%  •  "
                        f"Zgodność obrysu: {measured['polygon_iou_pct']:.2f}%"))
                except (ValueError, TypeError, KeyError):
                    self.pose_gt_metric.configure(text="Nie można porównać obrysów")
        else:
            self.pose_gt_metric.configure(text="")
        # Keep this card focused: show drawing help and results only when relevant.
        if unavailable or count == 4 or self.pose_edit_var.get():
            self.pose_gt_progress.grid()
        else:
            self.pose_gt_progress.grid_remove()
        if count == 4:
            self.pose_gt_metric.grid()
        else:
            self.pose_gt_metric.grid_remove()

    def _maybe_begin_guided_pose_gt(self):
        if getattr(self, "_canvas_editor", None) is not None:
            return
        """Enter drawing only after an explicit response to the corner question.

        This helper must never be invoked on load/refresh/selection; the saved
        presence decision alone is not permission to hide the model overlay.
        """
        if not self.session or self._saving or self.pose_edit_var.get():
            return False
        attempt_id = self._pose_gt_attempt_context()
        if not attempt_id or self._image is None:
            return False
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        if (not self._gt_target_is_confirmed(attempt_id)
                or decision.get("pose_gt_unavailable_reason")):
            return False
        self._ensure_pose_gt_points(attempt_id)
        if len(self._pose_gt_points) == 4:
            return False
        token = (attempt_id, self._image_entry)
        if getattr(self, "_pose_auto_token", None) == token:
            return False
        self._pose_auto_token = token
        self._toggle_pose_gt_edit()
        if self.pose_edit_var.get():
            self.status_var.set("Wskaż cztery rogi tablicy w dowolnej kolejności. Po czwartym punkcie zapis nastąpi sam.")
            return True
        return False

    def _choose_pose_gt_answer(self):
        """An operator *answers a question* instead of activating an unrelated CTA."""
        attempt_id = self._pose_gt_attempt_context()
        if not attempt_id or self._saving:
            self._sync_pose_gt_controls()
            return
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        if not self._gt_target_is_confirmed(attempt_id):
            self._sync_pose_gt_controls()
            return
        self._ensure_pose_gt_points(attempt_id)
        answer = self.pose_gt_answer.get()
        if answer == "unavailable":
            if len(self._pose_gt_points) == 4:
                self.pose_gt_answer.set("mark")
                self.status_var.set("Rogi zostały już zapisane. Najpierw usuń ten obrys, jeśli jest błędny.")
                return
            # An unfinished 1–3 point draft is not saved GT. The operator may
            # decide it is impossible to identify the remaining corners.
            if 0 < len(self._pose_gt_points) < 4:
                self._leave_pose_gt_edit_on_navigation()
                self._paint_image()
            if not decision.get("pose_gt_unavailable_reason"):
                self._set_pose_gt_unavailable()
        elif answer == "mark":
            if decision.get("pose_gt_unavailable_reason"):
                self._change(lambda: self.session.annotate("attempt", attempt_id,
                    pose_gt_unavailable_reason=None),
                    after=lambda: self._resume_pose_after_answer(attempt_id))
            else:
                self._resume_pose_after_answer(attempt_id)

    def _resume_pose_after_answer(self, attempt_id):
        if self._pose_gt_attempt_context() != attempt_id:
            return
        self._pose_auto_token = None  # explicit operator answer can resume after cancel
        self._maybe_begin_guided_pose_gt()

    def _leave_pose_gt_edit_on_navigation(self):
        """Never carry a GT edit latch onto a different observation/detection/image.

        Only an unfinished in-memory 1–3 point draft is discarded. Saved GT,
        manual decisions and underlying MT model predictions are unchanged.
        """
        control = getattr(self, "pose_edit_var", None)
        if control is None or not control.get():
            return
        control.set(False)
        self._pose_gt_drag_index = None
        self._pose_gt_before_drag = None
        self._view_drag_at = None
        if 0 < len(getattr(self, "_pose_gt_points", [])) < 4:
            self._pose_gt_points = []
            self._pose_gt_cache_token = None
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="hand2")

    def _toggle_pose_gt_edit(self):
        """One-shot action for this MT attempt, not a sticky checkbox."""
        if self.pose_edit_var.get():
            self._leave_pose_gt_edit_on_navigation()
        else:
            attempt_id = self._pose_gt_attempt_context()
            if not attempt_id or self._saving:
                return
            decision = self.session.review.attempt_annotations.get(attempt_id, {})
            if (not self._gt_target_is_confirmed(attempt_id)
                    or decision.get("pose_gt_unavailable_reason")):
                self.status_var.set("Najpierw oceń liczbę tablic albo cofnij pominięcie GT.")
                return
            self.pose_edit_var.set(True)
        self._pose_gt_drag_index = None
        self._pose_gt_before_drag = None
        self._view_drag_at = None
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="crosshair" if self.pose_edit_var.get() else "hand2")
            self._paint_image()
        self._sync_pose_gt_controls()

    def _set_pose_gt_unavailable(self):
        """Independent geometrical exclusion; never touches detection or MZ decisions."""
        attempt_id = self._pose_gt_attempt_context()
        if not attempt_id or self._saving:
            return
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        if not self._gt_target_is_confirmed(attempt_id):
            return
        self._ensure_pose_gt_points(attempt_id)
        if self._pose_gt_points:
            self.status_var.set("Najpierw usuń istniejący GT narożników, aby oznaczyć go jako nieoznaczalny.")
            return
        reason = (None if decision.get("pose_gt_unavailable_reason")
                  else "corners_not_distinguishable")
        self._leave_pose_gt_edit_on_navigation()
        self._change(lambda: self.session.annotate("attempt", attempt_id,
            pose_gt_unavailable_reason=reason))

    def _pose_canvas_to_mt_input(self, event, attempt_id):
        if self._image is None or self._image_source_size is None:
            return None
        row = self.session.attempts[attempt_id]
        w, h = float(row["input_width"]), float(row["input_height"])
        if self._image_source_size != (w, h) or not self._view_center:
            return None
        scale = max(1e-9, self._view_pixel_scale)
        cx, cy = self._view_center
        vw, vh = self.canvas.winfo_width(), self.canvas.winfo_height()
        # Exact inverse of _paint_image, including source->thumbnail scale.
        ix = cx + (event.x - vw / 2) / scale
        iy = cy + (event.y - vh / 2) / scale
        x = ix * w / self._image.width
        y = iy * h / self._image.height
        return (x, y) if 0 <= x <= w and 0 <= y <= h else None

    def _pose_mt_to_canvas(self, point, attempt_id):
        row = self.session.attempts[attempt_id]
        w, h = float(row["input_width"]), float(row["input_height"])
        cx, cy = self._view_center
        scale = self._view_pixel_scale
        vw, vh = self.canvas.winfo_width(), self.canvas.winfo_height()
        return ((point[0] * self._image.width / w - cx) * scale + vw / 2,
                (point[1] * self._image.height / h - cy) * scale + vh / 2)

    def _draw_pose_gt_overlay(self, attempt_id, to_canvas):
        self._ensure_pose_gt_points(attempt_id)
        points = self._pose_gt_points
        valid_polygon = False
        canonical_labels = False
        # In-progress clicks have no edge order. Connect only four valid
        # vertices. Saved GT is drawn AS RECORDED, never silently migrated.
        if len(points) == 4:
            try:
                attempt = self.session.attempts[attempt_id]
                w, h = attempt["input_width"], attempt["input_height"]
                if self.pose_edit_var.get():
                    points = _ordered_operator_gt_quad(points, w, h)
                else:
                    points = validate_mt_quad(points, w, h)
                canonical_labels = tuple(points) == _ordered_operator_gt_quad(points, w, h)
                valid_polygon = True
            except (ValueError, TypeError, KeyError):
                pass
        pts = [to_canvas(*xy) for xy in points]
        if valid_polygon:
            flat = [v for xy in pts for v in xy]
            self.canvas.create_polygon(*flat, fill="", outline="#40e7aa", width=1.8,
                                       tags=("pose_gt_overlay", "pose_gt_polygon"))
        for index, (x, y) in enumerate(pts):
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#40e7aa",
                                    outline="#103d32", width=1,
                                    tags=("pose_gt_overlay", "pose_gt_corner"))
            label = ("TL", "TR", "BR", "BL")[index] if canonical_labels else str(index + 1)
            self.canvas.create_text(x + 10, y - 10, text=label,
                                    fill="#40e7aa", anchor="sw", font=("Segoe UI", 10, "bold"),
                                    tags=("pose_gt_overlay", "pose_gt_corner_label"))

    def _finish_pose_gt_drag(self):
        if self._pose_gt_drag_index is None:
            return
        self._pose_gt_drag_index = None
        previous = self._pose_gt_before_drag or []
        self._pose_gt_before_drag = None
        attempt_id = self._pose_gt_attempt_context()
        if not attempt_id:
            self._pose_gt_points = previous
            return
        if len(self._pose_gt_points) < 4:
            self._sync_pose_gt_controls()
            return
        row = self.session.attempts[attempt_id]
        try:
            w, h = float(row["input_width"]), float(row["input_height"])
            quad = _ordered_operator_gt_quad(self._pose_gt_points, w, h)
            pred = _linked_pose_quad_on_mt_input(self.session, row)
            if pred is not None:
                compare_mt_pose_to_gt(pred, quad, w, h)
        except (ValueError, TypeError, KeyError) as exc:
            self._pose_gt_points = previous
            self.status_var.set(f"Nie zapisano GT: {exc}. Popraw układ narożników.")
            self._paint_image()
            self._sync_pose_gt_controls()
            return
        # The displayed edit handles and the persisted annotation now share
        # the canonical TL/TR/BR/BL order, irrespective of click sequence.
        self._pose_gt_points = list(quad)
        points = [[round(x, 4), round(y, 4)] for x, y in quad]
        current = self.session.review.attempt_annotations.get(attempt_id, {})
        # As soon as four points are valid, leave editing and show MT+GT again.
        # This is local to the current attempt and requires no extra approval.
        self.pose_edit_var.set(False)
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="hand2")
            self._paint_image()
        if (current.get("pose_gt_quad_mt_input_px") == points and
                current.get("pose_gt_schema") == POSE_GT_SCHEMA and
                current.get("pose_gt_image_entry") == self._image_entry):
            self._sync_pose_gt_controls()
            return
        is_missed = str(row.get("mt_status") or "").upper() == "NO_DETECTION"
        payload = dict(
            pose_gt_schema=POSE_GT_SCHEMA,
            pose_gt_image_entry=row["mt_input_evidence_entry"],
            pose_gt_input_size_px=[w, h],
            pose_gt_quad_mt_input_px=points,
            pose_gt_review_mode=("assisted_gt_missed_mt_detection_v1" if is_missed else
                                 "assisted_with_pred_preview_v1" if pred is not None else
                                 "assisted_gt_without_mt_pose_v1"),
            pose_gt_unavailable_reason=None)
        if not is_missed:
            payload.update(plate_visibility="visible", is_plate=True, evaluable=True)
        # A NO_DETECTION record is never promoted into a detected plate.
        # Its independent operator GT is tied to the exact attempt/MT image.
        self._change(lambda: self.session.annotate("attempt", attempt_id, **payload))
        self._sync_pose_gt_controls()

    def _clear_pose_gt(self):
        attempt_id = self._pose_gt_attempt_context()
        if not attempt_id or self._saving:
            return
        decision = self.session.review.attempt_annotations.get(attempt_id, {})
        self._pose_gt_points = []
        self._pose_gt_drag_index = None
        self._pose_gt_before_drag = None
        self.pose_edit_var.set(False)
        self.canvas.configure(cursor="hand2")
        self._paint_image()
        self._sync_pose_gt_controls()
        if decision.get("pose_gt_quad_mt_input_px") is not None:
            self._change(lambda: self.session.annotate("attempt", attempt_id,
                pose_gt_schema=None, pose_gt_image_entry=None,
                pose_gt_input_size_px=None, pose_gt_quad_mt_input_px=None,
                pose_gt_review_mode=None))

    def _sync_detection_choice(self):
        if not self.session or not hasattr(self, "detection_radios"):
            return
        current = self.current_record
        annotation = {}
        allowed = False
        if current:
            kind, record_id = self._active_detection_target()
            row = (self.session.attempts if kind == "attempt" else self.session.samples).get(record_id, {})
            cancelled = self.session.cancelled(row)
            allowed = not cancelled and (kind == "sample" or
                       str(row.get("mt_status", "")).upper() in DETECTION_STATUSES)
            annotation = (self.session.review.attempt_annotations if kind == "attempt"
                          else self.session.review.sample_annotations).get(record_id, {})
            if kind == "attempt" and not allowed:
                question = "Brak detekcji MT — oceń tablice na całym obrazie (powyżej)."
            elif kind == "sample":
                question = "Czy wycinek obrazu przedstawia tablicę?"
            else:
                question = "Oceń zaznaczenie na zdjęciu."
        else:
            question = "Weryfikacja operatora: wybierz wskazanie MT lub obraz tablicy."
        self.detection_context.configure(text=question)
        if annotation.get("is_plate") is True and annotation.get("evaluable") is not False:
            selected = "yes"
        elif annotation.get("is_plate") is False:
            selected = "no"
        elif annotation.get("evaluable") is False:
            selected = "skip"
        else:
            selected = ""  # Bez decyzji — nie zaznaczaj nic automatycznie.
        self.detection_choice.set(selected)
        for radio in self.detection_radios:
            radio.configure(state="normal" if allowed and not self._saving else "disabled")

    def show_invocation_input(self):
        attempt = self._current_mt_attempt()
        group = self.session.mt_invocations.get(self.current_invocation) if self.session else None
        index = _primary_review_image_index(getattr(self, "_image_rows", ()), attempt, group)
        if index is None:
            self.status_var.set("Brak zdjęcia przypisanego do tej próby. Nie wybieramy obrazu z innej detekcji.")
            return
        if self.image_choice.current() == index and self._image is not None:
            return  # Preserve the current zoom and GT context.
        self.image_choice.current(index)
        self._load_image()

    def choose_visible_plate_count(self):
        value = self.invocation_count_value.get()
        if value in VISIBILITY_LABELS:
            self.invocation_choice.set(VISIBILITY_LABELS[value])
            self.save_invocation_decision()

    def save_invocation_decision(self, event=None):
        key = self.current_invocation
        group = self.session.mt_invocations.get(key) if self.session else None
        if not group or group.cancelled or not group.executed or group.execution_failed:
            self._sync_review_controls()
            return
        value = next((key for key, label in VISIBILITY_LABELS.items() if label == self.invocation_choice.get()), None)
        if key and value:
            old=self.session.invocation_annotation(key)
            if old.get("visible_plate_count")==value:
                return  # reselect does not destroy the multi-plate inventory
            if old.get("multi_plate_v1") is not None:
                self.status_var.set("Najpierw zakończ lub skoryguj wykaz GT; zmiana liczby tablic usunęłaby zapisane powiązania.")
                self._sync_review_controls()
                return
            self._change(lambda: self.session.annotate_invocation(key, visible_plate_count=value, evaluable=value != "uncertain"))

    def reveal_predictions(self):
        """Jawny krok dla starych rozpoczętych ocen zaślepionych."""
        if not self.session or self._saving or self.session.review.review_mode == "assisted":
            return
        self._change(self.session.reveal_predictions_for_review,
                     after=lambda: self.status_var.set("Predykcje widoczne. Istniejące GT zachowane. Odtąd weryfikacja jest asystowana, nie zaślepiona."))

    def change_review_mode(self, event=None):
        mode = next((key for key, label in MODE_LABELS.items() if label == self.mode_choice.get()), None)
        if mode:
            key, text, note = self.subject_key, self.gt_var.get(), self.note_var.get()
            def change():
                self.session.set_review_mode(mode)
                if key and (text or note):
                    self.session.save_subject_draft(key, ground_truth=text, note=note)
            self._change(change, save_draft=False)

    def _current_mt_attempt(self):
        if not self.session or not self.current_record:
            return {}
        kind, key = self.current_record
        if kind == "attempt":
            return self.session.attempts.get(key, {})
        sample = self.session.samples.get(key, {})
        return self.session.attempts.get(str(sample.get("attempt_id") or ""), {})

    def _review_areas(self):
        """Detectable regions linked to this observation and MT invocation."""
        if not self.session or not self.subject_key or not self.current_invocation:
            return []
        subject = self.session.subjects.get(self.subject_key, {})
        return [aid for aid in subject.get("attempts", ())
                if (self.session.invocation_by_attempt.get(aid) == self.current_invocation and
                    self.session.attempts.get(aid, {}).get("mt_status") in DETECTION_STATUSES)]

    def _sync_detected_area_navigation(self):
        if not hasattr(self, "target_nav"):
            return
        ids = self._review_areas()
        current_id = str(self._current_mt_attempt().get("id") or "")
        if len(ids) <= 1 or current_id not in ids:
            self.target_nav.grid_remove()
            return
        pos = ids.index(current_id)
        self.target_position.configure(text=f"Obszar {pos + 1} z {len(ids)}")
        self.target_previous.configure(state="normal" if pos > 0 and not self._saving else "disabled")
        self.target_next.configure(state="normal" if pos + 1 < len(ids) and not self._saving else "disabled")
        self.target_nav.grid()

    def _switch_detected_area(self, direction):
        ids = self._review_areas()
        current_id = str(self._current_mt_attempt().get("id") or "")
        if self._saving or current_id not in ids:
            return
        target = ids.index(current_id) + int(direction)
        if target < 0 or target >= len(ids):
            return
        iid = next((iid for iid, info in self._record_ids.items()
                    if info == ("attempt", ids[target])), None)
        if iid is None:
            return
        self.record_tree.selection_set(iid)
        self.record_tree.focus(iid)
        self._select_record()

    def _sync_crop_preview_source(self, attempt, current_sample=None):
        self._crop_preview_row = None
        if not self.session:
            return
        attempt_id = str(attempt.get("id") or "")
        candidates = ([current_sample] if isinstance(current_sample, dict) else [])
        candidates += [self.session.samples[key] for key in
                       (getattr(self.session, "sample_ids_by_attempt", {}) or {}).get(attempt_id, ())
                       if key in self.session.samples]
        for sample in candidates:
            if (sample.get("attempt_id") == attempt_id and
                    sample.get("image_entry") in self.session.entry_names):
                self._crop_preview_row = sample
                break
        if self._crop_preview_row:
            self.crop_preview_button.configure(state="normal")
            self.crop_preview_button.grid()
        else:
            self.crop_preview_button.grid_remove()

    def _close_crop_preview(self):
        popup = getattr(self, "_crop_preview_window", None)
        self._crop_preview_window = None
        if popup is not None:
            try:
                if popup.winfo_exists():
                    popup.destroy()
            except tk.TclError:
                pass

    def _show_crop_preview(self):
        """Read-only MZ image in a separate window; never changes MT evidence or GT."""
        row = getattr(self, "_crop_preview_row", None)
        if not row or not self.session or self._saving:
            return
        self._close_crop_preview()
        popup = tk.Toplevel(self.window)
        self._crop_preview_window = popup
        popup.configure(bg=self.bg)
        popup.title("Wycięta tablica — podgląd numeru")
        popup.geometry("770x300")
        popup.minsize(340, 170)
        popup.protocol("WM_DELETE_WINDOW", self._close_crop_preview)
        caption = self._label(popup, "Wycięcie tablicy zapisane przez telefon. Podgląd pomaga odczytać numer.",
                              font=("Segoe UI", 10), wraplength=710)
        caption.pack(fill="x", padx=14, pady=(12, 5))
        picture = tk.Label(popup, text="Wczytywanie wycięcia…", bg=self.bg, fg=self.fg)
        picture.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        selected_row = dict(row)

        def work():
            img = self.session.image(selected_row)
            if img is None:
                raise ValueError("Nie znaleziono wyciętej tablicy w archiwum.")
            img = img.convert("RGB")
            max_w, max_h = 740, 480
            factor = min(5.0, max_w / max(1, img.width), max_h / max(1, img.height))
            if factor != 1.0:
                img = img.resize((max(1, round(img.width * factor)),
                                  max(1, round(img.height * factor))), Image.Resampling.LANCZOS)
            return img

        def done(image):
            if self._crop_preview_window is not popup or not popup.winfo_exists():
                return
            picture_image = ImageTk.PhotoImage(image, master=picture)
            picture.configure(image=picture_image, text="")
            picture.image = picture_image
            popup.geometry(f"{max(360, image.width + 32)}x{max(190, image.height + 80)}")

        # The standard background worker reports errors through the status bar.
        self._submit(work, done)


    def _source_frame_context(self):
        """Require one coherent source Bitmap for the selected MT invocation."""
        if not self.session or not self.current_invocation:
            return None
        group = self.session.mt_invocations.get(self.current_invocation)
        if group is None:
            return None
        return source_invocation_context(
            getattr(group, "records", ()) or (), self.current_invocation,
            self.session.entry_names,
            self._image_source_size if self._is_source_frame_view() and self._image is not None else None)

    def _is_source_frame_view(self):
        return bool(self._image_entry and self._image_entry.startswith("samples/frames/"))

    def _sync_source_frame_button(self):
        button = getattr(self, "source_frame_button", None)
        if button is None:
            return
        context = self._source_frame_context()
        if context is None:
            button.pack_forget()
            return
        button.configure(text="← Wejście MT" if self._is_source_frame_view()
                         else "Cała klatka + ROI")
        if not button.winfo_manager():
            button.pack(side="left", padx=(2, 8))

    def _toggle_source_frame_view(self):
        if getattr(self, "_canvas_editor", None) is not None and not self._canvas_editor.prepare_navigation():
            return
        context = self._source_frame_context()
        if context is None:
            self.status_var.set("Brak spójnego zdjęcia źródłowego dla tego wywołania MT.")
            return
        if self._is_source_frame_view():
            self.show_invocation_input()
            return
        if self._saving:
            return
        index = next((i for i, (_, row) in enumerate(self._image_rows)
                      if row.get("source_frame_view") and
                      row.get("image_entry") == context["entry"]), None)
        if index is None:
            self.status_var.set("Brak pełnej klatki w archiwum sesji.")
            return
        self.image_choice.current(index)
        self._load_image()

    def _paint_source_frame_overlays(self, width, height):
        """Read-only full-frame ROI and *every* original Pose detection.

        Do not paint assisted GT/corrections from MT-input coordinates; those
        have their own distinct review and provenance in the legacy view.
        """
        if not self._is_source_frame_view() or self._image is None:
            return
        context = self._source_frame_context()
        if context is None or self._image_entry != context["entry"]:
            self.image_help.configure(text=(
                "Brak zgodnego źródła/ROI dla wybranego wywołania. "
                "Nie rysujemy obrysów na podstawie niepewnych współrzędnych."))
            return
        iw, ih = context["size"]
        cx, cy = self._view_center
        zoom = self._view_pixel_scale
        def to_canvas(x, y):
            return ((x * self._image.width / iw - cx) * zoom + width / 2,
                    (y * self._image.height / ih - cy) * zoom + height / 2)
        left, top, right, bottom = context["roi"]
        x1, y1 = to_canvas(left, top)
        x2, y2 = to_canvas(right, bottom)
        bx1, by1 = to_canvas(0, 0)
        bx2, by2 = to_canvas(iw, ih)
        # Only obscure regions genuinely outside the analyzed ROI. No inferred
        # missing targets, FN or GT are displayed here.
        if (left, top, right, bottom) != (0, 0, iw, ih):
            for a, b, c, d in ((bx1, by1, bx2, y1), (bx1, y2, bx2, by2),
                               (bx1, y1, x1, y2), (x2, y1, bx2, y2)):
                if c > a and d > b:
                    self.canvas.create_rectangle(a, b, c, d, fill="#07111B",
                        outline="", stipple="gray50", tags=("source_outside_roi",))
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FFD16C",
            width=2, dash=(7, 4), tags=("mt_source_roi",))
        label = self.canvas.create_text(max(6, min(width - 150, x1 + 5)),
            max(6, min(height - 28, y1 + 5)), anchor="nw",
            text=f"ROI MT • {int(left)},{int(top)}–{int(right)},{int(bottom)}",
            fill="#FFF1CC", font=("Segoe UI", 9, "bold"), tags=("mt_source_roi",))
        self._canvas_text_background(label, "mt_source_roi")
        found = verified_invocation_detections(context)
        selected = str(self._current_mt_attempt().get("id") or "")
        for item in found:
            pts = [to_canvas(*p) for p in item["quad"]]
            is_selected = item["attempt_id"] == selected
            tag = "source_mt_det_" + str(item["detection_index"])
            self.canvas.create_polygon(*[v for xy in pts for v in xy], fill="",
                outline="#36DDFB" if is_selected else "#83C5D5",
                width=2.4 if is_selected else 1.5,
                tags=("source_mt_all_detections", tag))
            xx, yy = min(pts, key=lambda p: (p[1], p[0]))
            label = self.canvas.create_text(max(5, min(width-70, xx)),
                max(6, min(height-20, yy-18)), anchor="nw",
                text=f"MT {item['detection_index']+1}",
                fill="#F5FCFF", font=("Segoe UI", 9, "bold"),
                tags=("source_mt_all_detections", tag))
            self._canvas_text_background(label, tag)
        self._source_frame_detections = found
        self.image_help.configure(text=(
            f"Pełna klatka przed ROI • {len(found)} obrysów Pose MT z tego "
            f"wywołania (spośród {len(context['records'])} zapisów). "
            "Żółta linia: rzeczywisty ROI; błękit: oryginalne MT. "
            "Kliknij tablicę, aby wybrać detekcję. Widok tylko do odczytu — "
            "edycja GT pozostaje przy wejściu MT."))

    def _select_source_frame_detection(self, event):
        """Selection only; never change the original MT/GT or metrics."""
        if not self._is_source_frame_view() or self._image is None:
            return False
        context = self._source_frame_context()
        if context is None or self._image_entry != context["entry"]:
            return False
        cx, cy = self._view_center
        zoom = max(self._view_pixel_scale, 1e-12)
        source_x = (cx + (event.x - self.canvas.winfo_width()/2)/zoom) * (
            context["size"][0] / self._image.width)
        source_y = (cy + (event.y - self.canvas.winfo_height()/2)/zoom) * (
            context["size"][1] / self._image.height)
        candidates = verified_invocation_detections(context)
        # Test selected geometry first to prevent oscillation in overlapping
        # polygons; choosing another contour still navigates to its attempt.
        current = str(self._current_mt_attempt().get("id") or "")
        candidates.sort(key=lambda item: item["attempt_id"] != current)
        for item in candidates:
            if quad_contains_point(item["quad"], source_x, source_y):
                if item["attempt_id"] != current:
                    self._source_frame_keep_on_selection = context["entry"]
                    self._multi_focus_attempt(item["attempt_id"])
                return True
        return False

    def _set_image_help(self):
        index = self.image_choice.current()
        if index < 0 or not getattr(self, "_image_rows", None):
            return
        label, context_row = self._image_rows[index]
        if context_row.get('context_only'):
            self.image_help.configure(text='Wycinek TV — JPEG94, tylko kontekst. GT i weryfikacja pozostają na pełnej klatce kamery.')
            return
        if self._is_source_frame_view():
            self.image_help.configure(text="Klatka przed ROI — widok tylko do odczytu. Złota ramka oznacza wejście MT.")
            return
        missing_mt = str(self._current_mt_attempt().get("mt_status") or "").upper() == "NO_DETECTION"
        self.image_help.configure(text=(
            "MT nie wykrył tablicy. Jeśli widzisz jedną, wybierz «1 — jedna tablica» "
            "i zaznacz jej rzeczywiste rogi GT."
            if missing_mt and label.startswith("Wejście MT —") else
            "Niebieski obrys pokazuje obszar wskazany przez MT."
            if label.startswith("Wejście MT —") else
            "Zdjęcie z telefonu. Nie ma dla niego potwierdzonego obrysu bieżącej detekcji."))

    def _load_image(self):
        # Mode is local to the exact viewed evidence, never sticky across images.
        index = self.image_choice.current()
        if getattr(self, "_image_rows", None) and 0 <= index < len(self._image_rows):
            next_row = self._image_rows[index][1]
            next_entry = next_row.get("image_entry") or next_row.get("evidence_entry") or ""
            if self._image_entry and next_entry != self._image_entry:
                self._cancel_mt_correction()  # discard only local sketch on image change
                self._leave_pose_gt_edit_on_navigation()
        self._image_token += 1
        self._view_zoom = 1.0
        self._view_center = None
        self._view_drag_at = None
        token = self._image_token
        self._image = None
        self._image_source_size = None
        self._image_entry = ""
        self._sync_pose_gt_controls()
        self._paint_image("Wczytywanie obrazu…")
        index = self.image_choice.current()
        if not self._image_rows or index < 0:
            self.image_help.configure(text="Nie zapisano zdjęcia powiązanego z tym obszarem.")
            self._paint_image("Brak zdjęcia dla tej detekcji. Nie wykorzystujemy obrazu z innej próby.")
            return
        _label, row = self._image_rows[index]
        self._set_image_help()
        def work():
            try:
                if row.get("source_frame_view") or row.get('context_only'):
                    self.session.verify_source()
                    image = read_mobile_source_frame(self.session.bundle, row["image_entry"])
                    self.session.verify_source()
                else:
                    image = self.session.image(row)
                source_size = image.size
                image = image.convert("RGB")  # bezpieczny tryb transformacji viewportu
                image.thumbnail((3200, 2400), Image.Resampling.LANCZOS)
                return image, source_size, ""
            except Exception as error:
                return None, None, str(error)
        def done(result):
            if token == self._image_token:
                self._image, self._image_source_size, error = result
                self._image_entry = row.get("image_entry") or row.get("evidence_entry") or ""
                self._paint_image(error)
                self._sync_detection_choice()
                self._sync_pose_gt_controls()
                self._sync_mt_visual_choice()
                self._sync_source_frame_button()
                # Never start drawing just because a saved positive decision was restored.
                # The operator must explicitly choose that the corners can be marked.
                if self._zoom_to_mt_pending and self._image is not None:
                    self.zoom_to_mt_detection()
        self._submit(work, done)

    def _paint_image(self, message=""):
        """Jedna transformacja dla pikseli i overlay MT: zoom/pan nie rozjezdza ramki."""
        from .mobile_canvas_zoom import cancel_zoom
        if not getattr(self, '_viewport_paint', False):
            cancel_zoom(self)
            from .mobile_canvas_pan import cancel_pan
            cancel_pan(self)
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._image is None:
            self.canvas.delete("all")
            self._photo = None
            self._view_fit_scale = 1.0
            self._view_pixel_scale = 1.0
            self.canvas.create_text(width / 2, height / 2,
                text=message or "Wybierz obserwację i obraz", fill="#edf4f6",
                width=max(50, width - 30))
            if hasattr(self, "zoom_value"):
                self.zoom_value.configure(text="—")
            return
        image_items = self.canvas.find_withtag('evidence_image')
        image_item = image_items[0] if len(image_items) == 1 else None
        if image_item is None:
            self.canvas.delete('all')
        else:
            # Keep the bitmap item/PhotoImage alive across viewport frames.
            # Recreating its Tk image instance was the largest remaining cost.
            self.canvas.addtag_all('viewport_overlay_cleanup')
            self.canvas.dtag(image_item, 'viewport_overlay_cleanup')
            self.canvas.delete('viewport_overlay_cleanup')
        iw, ih = self._image.size
        self._view_fit_scale = max(0.0001, min((width - 16) / iw, (height - 16) / ih))
        self._view_pixel_scale = self._view_fit_scale * self._view_zoom
        scale = self._view_pixel_scale
        if self._view_center is None:
            self._view_center = (iw / 2.0, ih / 2.0)
        cx, cy = self._view_center
        half_x = width / (2.0 * scale)
        half_y = height / (2.0 * scale)
        a3 = getattr(self, '_f5a_a3', None)
        if not (a3 and a3.active):
            cx = iw / 2.0 if iw <= 2 * half_x else max(half_x, min(iw - half_x, cx))
            cy = ih / 2.0 if ih <= 2 * half_y else max(half_y, min(ih - half_y, cy))
        # A3 permits margins around the frame: clamping/recentring here would
        # undo the cursor anchor, especially while the frame is letterboxed.
        self._view_center = (cx, cy)
        # Renderowanie TYLKO widocznego obszaru, bez tworzenia olbrzymich PhotoImage przy zoomie.
        # Obraz PIL jest tylko do odczytu; piksele poza nim wypelnia ciemne tlo.
        inv = 1.0 / scale
        box = (cx-width*inv/2, cy-height*inv/2, cx+width*inv/2, cy+height*inv/2)
        preview = getattr(self, '_zoom_preview', False)
        if preview and 0 <= box[0] < box[2] <= iw and 0 <= box[1] < box[3] <= ih:
            # Axis-aligned viewport: separable resize is faster than affine.
            rendered = self._image.resize((width, height), Image.Resampling.BILINEAR, box=box)
        else:
            rendered = self._image.transform((width, height), Image.Transform.AFFINE,
                (inv, 0, box[0], 0, inv, box[1]),
                resample=Image.Resampling.BILINEAR if preview else Image.Resampling.BICUBIC,
                fillcolor=(16, 24, 32))
        if (image_item is not None and self._photo is not None
                and (self._photo.width(), self._photo.height()) == (width, height)):
            self._photo.paste(rendered)
        else:
            photo = ImageTk.PhotoImage(rendered, master=self.canvas)
            if image_item is None:
                self.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("evidence_image",))
            else:
                self.canvas.itemconfigure(image_item, image=photo)
            self._photo = photo
        if hasattr(self, "zoom_value"):
            self.zoom_value.configure(text=f"×{self._view_zoom:.1f}")
        if hasattr(self, "image_help"):
            self.image_help.configure(text=(
                "Błękit: oryginalny obrys MT. Pomarańcz: obrys operatora. "
                "Rolką przybliżaj obraz, narożniki przeciągaj myszą."))
        if getattr(self, "_canvas_editor", None) is not None:
            handled = self._canvas_editor.paint(width, height)
            a3=getattr(self,'_f5a_a3',None)
            if a3 and a3.active:a3.painted()
            if handled:
                return
        if self._is_source_frame_view():
            self._paint_source_frame_overlays(width, height)
            return
        self._paint_detection_overlay(width, height, 0, 0)
        # In the 4-click mode other detections must not obscure the truth image.
        if not self._mt_redraw_active:
            self._paint_multi_overlays(width, height)

    def _paint_detection_overlay(self, width, height, offset_x, offset_y):
        if not self.current_record or not self.current_invocation or self._image is None:
            return
        kind, key = self.current_record
        row = self.session.attempts.get(key if kind == "attempt" else
            self.session.samples[key].get("attempt_id"), {})
        group = self.session.mt_invocations[self.current_invocation]
        explicit_inputs = {child.get("mt_input_evidence_entry") for child in group.records}
        # Overlay only on the exact recorded MT input; never on an MZ crop.
        if (not self._image_entry or self._image_entry not in group.evidence_entries or
                self._image_entry.startswith("samples/crops/")):
            return
        exact_input = row.get("mt_input_evidence_entry")
        if str(row.get("mt_status") or "").upper() == "NO_DETECTION":
            gt_id = self._pose_gt_attempt_context()
            try:
                size = float(row["input_width"]), float(row["input_height"])
            except (TypeError, ValueError, KeyError):
                return
            if gt_id and self._image_entry == exact_input and self._image_source_size == size:
                cx, cy = self._view_center
                scale = self._view_pixel_scale
                xscale = self._image.width / size[0]
                yscale = self._image.height / size[1]
                def to_canvas(x, y):
                    return ((x * xscale - cx) * scale + width / 2,
                            (y * yscale - cy) * scale + height / 2)
                self._draw_pose_gt_overlay(gt_id, to_canvas)
            return
        if row.get("mt_status") not in DETECTION_STATUSES:
            return
        box = (detection_box_on_mt_input(row) if exact_input and
               self._image_entry == exact_input and exact_input in explicit_inputs else None)
        message = ("Brak obrazu wejścia MT dla tej detekcji. Nie pokazujemy ramki z innej próby."
                   if self._image_entry != exact_input else "Brak geometrii detekcji w tej sesji.")
        try:
            input_size = (float(row["input_width"]), float(row["input_height"]))
        except (ValueError, TypeError, KeyError):
            input_size = None
        if (self._image_entry != exact_input or exact_input not in explicit_inputs or
                input_size is None or self._image_source_size != input_size):
            if self._image_source_size != input_size:
                message = "Rozmiar obrazu nie odpowiada geometrii wejścia MT."
            text_id = self.canvas.create_text(12, 12, anchor="nw", text=message,
                fill="#ffffff", width=max(50, width - 24),
                tags=("detection_geometry_notice",))
            self._canvas_text_background(text_id, "detection_geometry_notice")
            return
        # Recorded Pose and independent GT can be rendered without a BBOX;
        # never infer any four corners from the BBOX.
        cx, cy = self._view_center
        scale = self._view_pixel_scale
        ratio_x = self._image.width / input_size[0]
        ratio_y = self._image.height / input_size[1]
        def to_canvas(x, y):
            return ((x * ratio_x - cx) * scale + width / 2,
                    (y * ratio_y - cy) * scale + height / 2)

        selected_sample = self.session.samples.get(key) if kind == "sample" else None
        quad = _linked_pose_quad_on_mt_input(self.session, row, selected_sample)
        gt_attempt_id = self._pose_gt_attempt_context()
        if gt_attempt_id:
            self._ensure_pose_gt_points(gt_attempt_id)
        editing_without_complete_gt = bool(gt_attempt_id and self.pose_edit_var.get() and
                                           len(self._pose_gt_points) < 4)
        selected_aid = str(row.get("id") or "")
        if self._mt_redraw_active and self._mt_redraw_attempt == selected_aid:
            # Source MT stays visible (thin dashed blue) while the operator
            # draws a separate, explicitly assisted contour in amber.
            if quad is not None:
                predicted = [to_canvas(*xy) for xy in quad]
                self.canvas.create_polygon(*[v for xy in predicted for v in xy],
                    fill="", outline="#40DCFF", width=1.5, dash=(5, 4),
                    tags=("source_mt_quad", "detection_overlay"))
            pts = [to_canvas(*xy) for xy in self._mt_redraw_points]
            if len(pts) >= 2:
                self.canvas.create_line(*[v for xy in pts for v in xy],
                    fill="#FFBD59", width=2, dash=(5, 3),
                    tags=("manual_gt_draft",))
            for idx, (px, py) in enumerate(pts, 1):
                self.canvas.create_oval(px-4, py-4, px+4, py+4,
                    fill="#FFBD59", outline="#152835", width=1,
                    tags=("manual_gt_draft",))
                self.canvas.create_text(px+8, py-6, text=str(idx),
                    fill="#FFE5A1", anchor="sw", tags=("manual_gt_draft",))
            if hasattr(self, "image_help"):
                self.image_help.configure(text=(
                    f"MT = błękitna przerywana linia; nowy obrys = pomarańczowy • "
                    f"{len(pts)}/4 rogów • Esc: anuluj szkic"))
            self._paint_mt_visual_overlay()
            return  # Both geometries are visualized; model data remain immutable.
        correction_in_progress = bool(self._mt_corr_active and self._mt_corr_attempt == selected_aid)
        corrected = None
        if quad is not None and not correction_in_progress:
            decision = self.session.review.attempt_annotations.get(selected_aid, {})
            if selected_visual_state(decision) == "imprecise":
                try:
                    corrected = self._valid_mt_correction(selected_aid, row, quad)
                except (ValueError, KeyError, TypeError):
                    corrected = None  # Bad provenance never changes source MT geometry.
        if correction_in_progress:
            corrected = self._mt_corr_points
        if quad is not None and not editing_without_complete_gt:
            model_pts = [to_canvas(x, y) for x, y in quad]
            model_flat = [v for xy in model_pts for v in xy]
            # Always preserve and show the untouched model result. Once a
            # correction exists, MT becomes a thin dashed cyan reference.
            self.canvas.create_polygon(*model_flat, fill="",
                outline="#40DCFF", width=1.5 if corrected is not None else 2,
                dash=(5, 4) if corrected is not None else (),
                tags=("detection_overlay", "source_mt_quad"))
            geometry = corrected if corrected is not None else quad
            pts = [to_canvas(x, y) for x, y in geometry]
            correction_has_moved = bool(corrected is not None and moved(quad, corrected))
            if correction_has_moved:
                self.canvas.create_polygon(*[v for xy in pts for v in xy],
                    fill="", outline="#FFBD59", width=2,
                    tags=("detection_overlay", "operator_quad"))
            if corrected is not None or correction_in_progress:
                for i, (x, y) in enumerate(pts):
                    radius = 5 if correction_in_progress else 3
                    self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                        fill="#FFBD59" if correction_has_moved else "#40DCFF",
                        outline="#172a34", width=1,
                        tags=("detection_overlay", "operator_quad", "pose_corner"))
            else:
                for x, y in model_pts:
                    self.canvas.create_oval(x-3, y-3, x+3, y+3,
                        fill="#40DCFF", outline="#172a34", width=1,
                        tags=("detection_overlay", "source_mt_quad"))
            anchor_x = min(x for x, _ in pts)
            anchor_y = min(y for _, y in pts)
            prefix = "KOREKTA OPERATORA" if corrected is not None else "POSE MT"
            if corrected is not None and hasattr(self, "image_help"):
                if moved(quad, corrected):
                    try:
                        measured = compare_mt_pose_to_gt(quad, corrected, *input_size)
                        note = (f"IoU MT ↔ korekta operatora: {measured['polygon_iou_pct']:.2f}%  •  "
                                f"NMCE: {measured['nmce_pct']:.2f}%")
                    except (ValueError, TypeError, KeyError):
                        note = "IoU/NMCE: brak prawidłowego porównania obrysów"
                else:
                    note = "Korekta jeszcze niezmieniona — przesuń róg lub całą ramkę"
                self.image_help.configure(text=(
                    "MT: błękitny obrys przerywany • operator: pomarańczowy obrys ciągły. "
                    + note + " (porównanie asystowane, nie niezależne GT)."))
        elif not editing_without_complete_gt:
            # BBOX is retained in the source for diagnostics/zoom, never drawn
            # as a substitute for MT Pose in the scientific review.
            _, reason = _pose_source_status(self.session, row)
            text_id = self.canvas.create_text(12, 12, anchor="nw",
                text=reason, fill="#ffffff", width=max(50, width - 24),
                tags=("detection_geometry_notice",))
            self._canvas_text_background(text_id, "detection_geometry_notice")
            if gt_attempt_id:
                self._draw_pose_gt_overlay(gt_attempt_id, to_canvas)
            self._paint_mt_visual_overlay()
            return
        if not editing_without_complete_gt:
            ordinal = row.get("mt_detection_index")
            title = (f"Detekcja {int(ordinal) + 1} / {group.detection_count}"
                     if ordinal not in (None, "") else "Detekcja")
            title = f"{prefix} • {title}"
            if row.get("mt_status") == "DETECTION_INVALID_QUAD":
                title += " • błędna geometria"
            title_x = max(4, min(width - 15, anchor_x + 5))
            title_y = max(3, min(height - 22, anchor_y - 24))
            text_id = self.canvas.create_text(title_x, title_y, anchor="nw", text=title,
                fill="#ffffff", font=("Segoe UI", 10, "bold"),
                tags=("detection_overlay", "detection_label"))
            bounds = self.canvas.bbox(text_id)
            if bounds is not None:
                self.canvas.move(text_id, min(0, width - 8 - bounds[2]), 0)
            self._canvas_text_background(text_id, "detection_overlay")
        if gt_attempt_id and corrected is None and not correction_in_progress:
            # Only one visible operator polygon per MT detection. Independent
            # GT remains in sidecar for audit/science, never a third overlaid
            # frame during a correction. Legacy assisted GT is mirrored in
            # 027/028 data and must not be redrawn as a separate green quad.
            decision = self.session.review.attempt_annotations.get(selected_aid, {})
            if decision.get("pose_gt_review_mode") != "assisted_manual_redraw_from_mt_v1":
                self._draw_pose_gt_overlay(gt_attempt_id, to_canvas)
        self._paint_mt_visual_overlay()

    def fit_image(self):
        """Przywroc caly obraz. Dziala rowniez po zmianie rozmiaru okna."""
        if self._image is None:
            return
        self._view_zoom = 1.0
        self._view_center = (self._image.width / 2.0, self._image.height / 2.0)
        self._paint_image()

    def zoom_image(self, factor: float, anchor=None):
        """Skalowanie wzgledem wskazanego miejsca (mysz) lub srodka canvasu."""
        if self._image is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ax, ay = anchor if anchor is not None else (width / 2, height / 2)
        if self._view_center is None:
            self._view_center = (self._image.width / 2, self._image.height / 2)
        prev = max(0.0001, self._view_fit_scale * self._view_zoom)
        img_x = self._view_center[0] + (ax - width / 2) / prev
        img_y = self._view_center[1] + (ay - height / 2) / prev
        from .mobile_canvas_zoom import cancel_zoom, paint_viewport, MAX_ZOOM
        cancel_zoom(self)
        newer = max(1.0, min(MAX_ZOOM, self._view_zoom * float(factor)))
        if abs(newer - self._view_zoom) < 0.0001:
            return
        self._view_zoom = newer
        next_scale = max(0.0001, self._view_fit_scale * self._view_zoom)
        self._view_center = (img_x - (ax - width / 2) / next_scale,
                             img_y - (ay - height / 2) / next_scale)
        paint_viewport(self)

    def _wheel_zoom(self, event):
        if (getattr(self, "_canvas_editor", None) is not None
                and self._canvas_editor.route_scroll(event) == "break"):
            return "break"
        if self._image is None:
            return "break"
        from .mobile_canvas_zoom import wheel_factor, request_zoom
        a3 = getattr(self, '_f5a_a3', None)
        factor = wheel_factor(event, step_factor=1.12 if a3 and a3.active else 1.25)
        if factor is not None:
            request_zoom(self, factor, (event.x, event.y))
        return "break"

    def _focus_image_for_wheel(self, _event=None):
        """Recover Canvas wheel focus from Treeview without blurring a GT entry."""
        try:
            focused = self.window.focus_get()
            if isinstance(focused, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
                return
            self.canvas.focus_set()
        except (AttributeError, tk.TclError):
            return

    def _route_image_wheel(self, event):
        """Roll zoom also when focus is in another widget of this review window.

        Event.x/y refer to the focused widget, NOT necessarily the canvas.
        Recalculate with the cursor's screen coordinates and limit to the
        image canvas. All other areas retain their ordinary scrolling.
        """
        if (getattr(self, "_canvas_editor", None) is not None
                and self._canvas_editor.route_scroll(event) == "break"):
            return "break"
        canvas = getattr(self, "canvas", None)
        if canvas is None or self._image is None or event.widget is canvas:
            return None  # canvas binding already handles its own event
        try:
            x, y = canvas.winfo_pointerxy()
            ix, iy = x - canvas.winfo_rootx(), y - canvas.winfo_rooty()
            if not (0 <= ix < canvas.winfo_width() and
                    0 <= iy < canvas.winfo_height()):
                return None
        except (AttributeError, TypeError):
            return None
        return self._wheel_zoom(type("CanvasWheelEvent", (), {
            "x": ix, "y": iy,
            "delta": getattr(event, "delta", 0),
            "num": getattr(event, "num", 0),
        })())

    def _multi_map_canvas(self, x, y):
        row=self._current_mt_attempt()
        if not row or self._image is None or self._image_source_size is None:
            return None
        try:
            iw,ih=float(row["input_width"]),float(row["input_height"])
        except (KeyError,TypeError,ValueError):
            return None
        if self._image_source_size!=(iw,ih) or self._image_entry!=row.get("mt_input_evidence_entry"):
            return None
        ww,hh=self.canvas.winfo_width(),self.canvas.winfo_height()
        scale=max(1e-9,self._view_pixel_scale)
        centerx,centery=self._view_center
        ix=centerx+(x-ww/2)/scale
        iy=centery+(y-hh/2)/scale
        return (ix*iw/self._image.width,iy*ih/self._image.height)

    def _paint_multi_overlays(self,width,height):
        group=self._multi_group()
        if group is None or self._image is None:
            return
        row=self._current_mt_attempt()
        if not row or self._image_entry!=row.get("mt_input_evidence_entry") or self._image_entry not in group.evidence_entries:
            return
        try:
            iw,ih=float(row["input_width"]),float(row["input_height"])
        except (KeyError,TypeError,ValueError):
            return
        if self._image_source_size!=(iw,ih):
            return
        cx,cy=self._view_center;scale=self._view_pixel_scale
        def pt(x,y):
            return ((x*self._image.width/iw-cx)*scale+width/2,
                    (y*self._image.height/ih-cy)*scale+height/2)
        chosen=str(row.get("id") or "")
        for aid,other in multi_detection_records(group).items():
            if aid==chosen or other.get("mt_input_evidence_entry")!=self._image_entry:
                continue
            corners=_linked_pose_quad_on_mt_input(self.session,other)
            if corners is None:
                bbox=detection_box_on_mt_input(other)
                if bbox is None:continue
                a,b,c,d=bbox
                corners=[(a,b),(c,b),(c,d),(a,d)]
            points=[pt(*corner) for corner in corners]
            tag="mtplate_"+aid.replace("-","_")
            inv_selected = inventory(self.session, self.current_invocation)
            selected_plate = next((p for p in inv_selected["plates"] if p["plate_id"] == self._multi_active_gt), None) if inv_selected else None
            highlighted = bool(selected_plate and selected_plate.get("attempt_id") == aid)
            self.canvas.create_polygon(*[v for xy in points for v in xy],fill="",
                              outline="#ffcf4a" if highlighted else "#7993A1",
                              width=4 if highlighted else 2, tags=("multi_region",tag))
            label_pos=min(points,key=lambda x:x[1])
            label=self.canvas.create_text(label_pos[0],max(4,label_pos[1]-10),text=f"MT {int(other.get('mt_detection_index') or 0)+1}",
                                          fill="#ffffff",font=("Segoe UI",10,"bold"),tags=("multi_region",tag))
            self._canvas_text_background(label,"multi_region")
            # The label background is a separate canvas item; clicking it must
            # choose the same numbered detection instead of swallowing input.
            backgrounds=self.canvas.find_withtag("multi_region")
            if backgrounds:
                self.canvas.addtag_withtag(tag,backgrounds[-1])
            self.canvas.tag_bind(tag,"<ButtonPress-1>",lambda e,item=aid:(self._multi_focus_attempt(item),"break")[-1])
        inv=inventory(self.session,self.current_invocation)
        if inv:
            for plate in inv["plates"]:
                bbox=plate.get("bbox_mt_px")
                if not bbox or plate.get("bbox_image_entry")!=self._image_entry:
                    continue
                a,b,c,d=bbox;x1,y1=pt(a,b);x2,y2=pt(c,d)
                tag="multi_gt_"+str(plate["plate_id"])
                active = plate["plate_id"] == self._multi_active_gt
                self.canvas.create_rectangle(x1,y1,x2,y2,
                        outline="#ffcf4a" if active else "#f0d578",
                        width=4 if active else 2,dash=(5,3),tags=("multi_gt",tag))
                self.canvas.create_text(x1,y1-12,text=plate["plate_id"]+" GT",fill="#ffef92",
                                        anchor="w",tags=("multi_gt",tag))
                self.canvas.tag_bind(tag,"<ButtonPress-1>",
                    lambda e,pid=plate["plate_id"]:(self._focus_multi_gt(pid),"break")[-1])
        if self._multi_bbox_edit and self._multi_drag_start and self._multi_drag_point:
            a,b=self._multi_drag_start;c,d=self._multi_drag_point
            x1,y1=pt(a,b);x2,y2=pt(c,d)
            self.canvas.create_rectangle(x1,y1,x2,y2,outline="#ffcc33",width=2,dash=(4,2),tags=("multi_bbox_draft",))

    def _pan_begin(self, event, *, force_pan=False):
        if self._image is None:
            return
        if not force_pan:
            try:
                self.canvas.focus_set()
            except (AttributeError, tk.TclError):
                pass
        if (not force_pan and not self._mt_redraw_active and
                bool(int(getattr(event, "state", 0) or 0) & 0x0001) and
                self._mt_corr_active and not self._saving):
            # Shift+click starts a four-corner replacement; the first click
            # is already corner 1, so the operator makes exactly four clicks.
            aid, attempt = self._visual_target()
            if (aid == self._mt_corr_attempt and
                    self._image_entry == attempt.get("mt_input_evidence_entry")):
                self._start_mt_redraw()
                if self._mt_redraw_active:
                    self._add_mt_redraw_point(event)
                return "break"
        if not force_pan and str(getattr(self, "_image_entry", "")).startswith("samples/frames/"):
            if self._select_source_frame_detection(event):
                return "break"
            # A source frame can be panned normally, but must never enter
            # an MT-input corner-editing, manual GT, or missed-plate mode.
            self._multi_bbox_edit = False
            self._leave_pose_gt_edit_on_navigation()
            self._cancel_mt_correction()
        if not force_pan and self._mt_redraw_active:
            self._add_mt_redraw_point(event)
            return "break"
        if not force_pan and self._multi_bbox_edit:
            point=self._multi_map_canvas(event.x,event.y)
            if point is None:
                self.status_var.set("Wybierz zweryfikowany obraz wejścia MT do oznaczenia ramki GT")
                return "break"
            self._multi_drag_start=point
            self._multi_drag_point=point
            return "break"
        if not force_pan and self._mt_corr_active:
            if self._mt_corr_begin_drag(event):
                return "break"
            # Left mouse outside editable outline remains normal image pan.
        if not force_pan and self.pose_edit_var.get():
            attempt_id = self._pose_gt_attempt_context()
            if not attempt_id or self._saving:
                self.status_var.set("Wybierz obraz wejścia MT bieżącej próby, aby zaznaczyć GT.")
                return "break"
            self._ensure_pose_gt_points(attempt_id)
            pos = self._pose_canvas_to_mt_input(event, attempt_id)
            if pos is None:
                self.status_var.set("Kliknij wewnątrz obrazu wejścia MT.")
                return "break"
            self._pose_gt_before_drag = list(self._pose_gt_points)
            nearest = [(i, math.dist(
                self._pose_mt_to_canvas(p, attempt_id), (event.x, event.y)))
                for i, p in enumerate(self._pose_gt_points)]
            if len(self._pose_gt_points) < 4:
                if nearest and min(distance for _, distance in nearest) <= 2:
                    self.status_var.set("Punkt pokrywa się z poprzednim narożnikiem. Powiększ obraz kółkiem myszy.")
                    return "break"
                self._pose_gt_points.append(pos)
                self._pose_gt_drag_index = len(self._pose_gt_points) - 1
            elif nearest and min(distance for _, distance in nearest) <= 13:
                self._pose_gt_drag_index = min(nearest, key=lambda item: item[1])[0]
            else:
                self.status_var.set("Aby skorygować GT, przeciągnij jeden z czterech zielonych punktów.")
                return "break"
            self._paint_image()
            self._sync_pose_gt_controls()
            return "break"
        if self._view_center is None:
            self._view_center = (self._image.width / 2, self._image.height / 2)
        self._view_drag_at = (event.x, event.y, *self._view_center)
        self.canvas.configure(cursor="fleur")

    def _pan_move(self, event):
        if self._multi_bbox_edit and self._multi_drag_start:
            self._multi_drag_point=self._multi_map_canvas(event.x,event.y)
            self._paint_image()
            return "break"
        if self._image is None:
            return
        if self._mt_corr_drag is not None:
            self._mt_corr_move_drag(event)
            return "break"
        if self._pose_gt_drag_index is not None:
            attempt_id = self._pose_gt_attempt_context()
            if attempt_id:
                point = self._pose_canvas_to_mt_input(event, attempt_id)
                if point is not None:
                    self._pose_gt_points[self._pose_gt_drag_index] = point
                    self._paint_image()
                    self._sync_pose_gt_controls()
            return "break"
        if self._view_drag_at is None:
            return
        mx, my, cx, cy = self._view_drag_at
        scale = max(0.0001, self._view_pixel_scale)
        self._view_center = (cx - (event.x - mx) / scale,
                             cy - (event.y - my) / scale)
        self._paint_image()

    def _pan_end(self, event=None):
        if self._mt_redraw_active and (event is None or getattr(event, "num", 1) == 1):
            return "break"  # right-button pan still works while drawing
        if self._multi_bbox_edit:
            start=self._multi_drag_start
            end=self._multi_drag_point
            self._multi_drag_start=None
            self._multi_drag_point=None
            self._multi_bbox_edit=False
            self.canvas.configure(cursor="hand2")
            if start and end and self._multi_group() is not None:
                a,b=start;c,d=end
                box=(min(a,c),min(b,d),max(a,c),max(b,d))
                row=self._current_mt_attempt()
                gid,pid=self.current_invocation,self._multi_active_gt
                entry=self._image_entry
                size=(float(row["input_width"]),float(row["input_height"]))
                self._change(lambda:set_missed_bbox(self.session,gid,pid,box,entry,size),save_draft=False)
            self._paint_image()
            return "break"
        if self._mt_corr_drag is not None:
            self._mt_corr_finish_drag()
            self._view_drag_at = None
            self.canvas.configure(cursor="crosshair" if self._mt_corr_active else "hand2")
            return "break"
        if self._pose_gt_drag_index is not None:
            self._finish_pose_gt_drag()
            self._paint_image()
        self._view_drag_at = None
        self.canvas.configure(cursor=("crosshair" if self._mt_corr_active or self.pose_edit_var.get() else "hand2")
                              if self._image is not None else "")

    def zoom_to_mt_detection(self):
        """Zblizenie wybranej detekcji; tylko gdy znana geometria jej wejscia MT."""
        if self._image is None or not self.current_record or not self.current_invocation:
            return
        kind, key = self.current_record
        row = self.session.attempts.get(key if kind == "attempt" else
            self.session.samples[key].get("attempt_id"), {})
        group = self.session.mt_invocations[self.current_invocation]
        exact_input = row.get("mt_input_evidence_entry")
        if not exact_input or exact_input not in group.evidence_entries:
            self._zoom_to_mt_pending = False
            self.status_var.set("Brak powiązanego zdjęcia wejściowego MT do zbliżenia.")
            return
        if self._image_entry != exact_input:
            if _primary_review_image_index(getattr(self, "_image_rows", ()), row, group) is None:
                self._zoom_to_mt_pending = False
                self.status_var.set("Brak zapisanego wejścia MT dla tej detekcji.")
                return
            if not self._zoom_to_mt_pending:
                self._zoom_to_mt_pending = True
                self.show_invocation_input()
            else:
                self._zoom_to_mt_pending = False
                self.status_var.set("Brak odpowiedniego obrazu wejścia MT dla zbliżenia.")
            return
        self._zoom_to_mt_pending = False
        box = detection_box_on_mt_input(row)
        selected_sample = self.session.samples.get(key) if kind == "sample" else None
        pose_quad = _linked_pose_quad_on_mt_input(self.session, row, selected_sample)
        if pose_quad is not None:
            # Fokus obejmuje rzeczywiste cztery narożniki Pose (nie otaczający je bbox).
            box = (min(x for x, _ in pose_quad), min(y for _, y in pose_quad),
                   max(x for x, _ in pose_quad), max(y for _, y in pose_quad))
        try:
            input_size = (float(row["input_width"]), float(row["input_height"]))
        except (ValueError, TypeError, KeyError):
            input_size = None
        if box is None or self._image_source_size != input_size:
            self.status_var.set("Zbliżenie wymaga kompletnej geometrii MT. Możesz użyć ręcznego zoomu.")
            return
        self._paint_image()
        sx = self._image.width / input_size[0]
        sy = self._image.height / input_size[1]
        x1, y1, x2, y2 = box
        box_w = max(1.0, abs(x2 - x1) * sx)
        box_h = max(1.0, abs(y2 - y1) * sy)
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        preferred_scale = min(cw * 0.60 / box_w, ch * 0.55 / box_h)
        self._view_zoom = max(1.0, min(24.0, preferred_scale / max(self._view_fit_scale, 0.0001)))
        self._view_center = ((x1 + x2) * 0.5 * sx, (y1 + y2) * 0.5 * sy)
        self._paint_image()
        self.status_var.set("Zbliżenie na wskazany obszar MT. Kółko: zoom, lewy przycisk: przesuń, Dopasuj: cały obraz.")

    def _canvas_text_background(self, text_id, tag):
        bounds = self.canvas.bbox(text_id)
        background = self.canvas.create_rectangle(bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2,
            fill="#10212b", outline="", tags=(tag,))
        self.canvas.tag_lower(background, text_id)

    def _sync_subject_eval_choice(self):
        if not self.session or not self.subject_key:
            return
        decision = self.session.review.subjects.get(self.subject_key, {})
        excluded = decision.get("evaluable") is False
        self.subject_eval_choice.set("exclude" if excluded else "assess")
        self.subject_unreadable_var.set(excluded)
        self.subject_unreadable_button.configure(state="disabled" if self._saving else "normal")

    def _sync_mz_readability_details(self, *, is_not_run=False):
        """Only the operator decides readability; the program decides OCR eligibility."""
        unreadable = bool(self.subject_unreadable_var.get())
        if unreadable:
            self.mz_exclusion_caption.configure(text=(
                "Zaznacz tylko wtedy, gdy numer jest nieczytelny nawet po powiększeniu. "
                "Nie porównamy wtedy odczytu aplikacji z GT. Wykrycie tablicy "
                "i jej obrys sprawdzamy osobno."
                if not is_not_run else
                "Zapisujemy, że numeru nie da się odczytać. MZ nie uruchomiono w tej próbie."))
            self.mz_note_caption.grid()
            self.note_entry.grid()
        else:
            self.mz_exclusion_caption.configure(text=(
                "Wpisany numer służy do opisu tablicy; MZ nie uruchomiono."
                if is_not_run else
                "Przepisz numer ze zdjęcia, nie z wyniku aplikacji. Program porówna je automatycznie."))
            self.mz_note_caption.grid_remove()
            self.note_entry.grid_remove()
        self.gt_entry.configure(state="disabled" if unreadable or self._saving else "normal")
        self.note_entry.configure(state="disabled" if self._saving else "normal")

    def _on_subject_unreadable_toggle(self):
        """Prevent an accidental exclusion while readable GT is present."""
        if not self.session or not self.subject_key or self._saving or (
                self._mz_review_state() in {"cancelled", "technical"}):
            self._sync_subject_eval_choice()
            return
        unreadable = bool(self.subject_unreadable_var.get())
        if unreadable and normalize_registration(self.gt_var.get()):
            self.subject_unreadable_var.set(False)
            messagebox.showwarning("Numer tablicy",
                "Dla tej tablicy wpisano już numer. Jeżeli nie możesz go odczytać "
                "ze zdjęcia, najpierw usuń wpisany numer, a następnie zaznacz tę opcję.",
                parent=self.window)
            return
        self.subject_eval_choice.set("exclude" if unreadable else "assess")
        self._sync_mz_readability_details(is_not_run=self._mz_review_state() in {"not_run", "empty"})
        self.save_subject_eval_choice()

    def save_subject_eval_choice(self):
        if (not self.session or not self.subject_key or self._saving
                or self._mz_review_state() in {"cancelled", "technical"}):
            return
        key, gt, note = self.subject_key, self.gt_var.get(), self.note_var.get()
        choice = self.subject_eval_choice.get()
        if choice not in {"assess", "exclude"}:
            return
        evaluable = False if choice == "exclude" else (True if normalize_registration(gt) else None)
        old = self.session.review.subjects.get(key, {})
        if (old.get("evaluable") is evaluable
                and normalize_registration(old.get("ground_truth")) == normalize_registration(gt)
                and old.get("note", "") == note):
            return
        self._change(lambda: self.session.set_subject(key, ground_truth=gt,
                       evaluable=evaluable, note=note), save_draft=False)

    def _schedule_autosave(self, *args):
        if (self._rendering or not self.subject_key
                or self._mz_review_state() in {"cancelled", "technical"}):
            return
        self.mode_choice.configure(state="disabled")
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
        self._autosave_id = self.window.after(750, self.save_gt)

    def _persist_subject_fields(self, key, gt, note):
        """Write only when this record is eligible for operator GT entry."""
        if (not key or self._mz_review_state() in {"cancelled", "technical"}):
            return
        old = self.session.review.subjects.get(key, {})
        normalized = normalize_registration(gt)
        if (normalize_registration(old.get("ground_truth")) == normalized
                and old.get("note", "") == note):
            return
        evaluable = (False if old.get("evaluable") is False else
                     True if normalized else None)
        self.session.set_subject(key, ground_truth=gt, note=note, evaluable=evaluable)

    def _draft_operation(self):
        if getattr(self, "_canvas_editor", None) is not None:
            return lambda: None  # Canvas GT is per plate, never a shared hidden subject field.
        key, gt, note = self.subject_key, self.gt_var.get(), self.note_var.get()
        return lambda: self._persist_subject_fields(key, gt, note)

    def _change(self, operation, *, save_draft=True, update_ranking=True, after=None):
        if not self.session or self._saving:
            return
        self.status_var.set("Zapisuję weryfikację…" if update_ranking else "Eksportuję zapisaną rewizję…")
        draft = self._draft_operation() if save_draft else None
        def work():
            if save_draft:
                draft()
            operation()
            stats = self.session.statistics()
            warning = ""
            updated_report = None
            if update_ranking:
                try:
                    store = MobilePackageExperimentStore(self.browser.store.root_dir)
                    if store.apply_human_review(self.session):
                        updated_report = next((report for report in store.reports if report.report_id == self.report.report_id
                                               and report.source_archive_sha256 == self.session.review.source_archive_sha256), None)
                except Exception as error:
                    warning = f"Decyzje zapisane; ranking wymaga odświeżenia: {error}"
            return stats, warning, updated_report
        def done(result):
            self._changed(result)
            if after:
                after()
        self._submit(work, done, saving=True)

    def _changed(self, result):
        from ..ranking.mobile_session_metrics import current, result_key
        snapshot=result[0].get('live_session_metrics_v2')
        if snapshot and not current(snapshot,result_key(self.session)):
            # A delayed worker must not replace a newer revision/session. The
            # existing debounce schedules the current saved state once idle.
            editor=getattr(self,'_canvas_editor',None)
            if editor is not None:editor.schedule_refresh()
            return
        self.stats, warning, updated_report = result
        self._refresh_subject_rows()
        self._refresh_stats()
        if self._pending_subject:
            self._leave_pose_gt_edit_on_navigation()
            self.subject_key, self._pending_subject = self._pending_subject, ""
        self._render_subject()
        self.status_var.set(warning or f"Zapisano automatycznie • rewizja {self.session.review.review_revision}")
        self._publish_review_report(updated_report)

    def _publish_review_report(self, updated_report):
        if updated_report is not None and self.browser.window.winfo_exists():
            self.browser.store.add_report(updated_report, save=False)
            for iid, report in self.browser.report_by_iid.items():
                if report.report_id == updated_report.report_id and report.source_archive_sha256 == updated_report.source_archive_sha256:
                    self.browser.report_by_iid[iid] = updated_report
            if (getattr(self.browser, "current_report", None) and self.browser.current_report.report_id == updated_report.report_id
                    and self.browser.current_report.source_archive_sha256 == updated_report.source_archive_sha256):
                self.browser._show_report(updated_report, self.browser.current_bundle)

    def save_gt(self, *, explicit=False):
        if not explicit and self._export_has_focus():
            self._prepare_export()
            return
        if getattr(self, "_canvas_editor", None) is not None:
            self._canvas_editor.finish_entry()
            return
        # No writing to a cancelled/technical evidence record; the old GT stays.
        if self._mz_review_state() in {"cancelled", "technical"}:
            if self._autosave_id:
                self.window.after_cancel(self._autosave_id)
                self._autosave_id = None
            return
        # explicit pozostaje dla istniejących wywołań, ale GT nie wymaga CTA.
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
            self._autosave_id = None
        if not self.session or not self.subject_key:
            return
        if self._saving:
            self._autosave_id = self.window.after(200, self.save_gt)
            return
        key, gt, note = self.subject_key, self.gt_var.get(), self.note_var.get()
        old = self.session.review.subjects.get(key, {})
        if (normalize_registration(old.get("ground_truth")) == normalize_registration(gt)
                and old.get("note", "") == note):
            return
        self._change(lambda: self._persist_subject_fields(key, gt, note), save_draft=False)

    def exclude_subject(self):
        """Kompatybilność z wywołaniami testów; w GUI służy temu radio."""
        self.subject_eval_choice.set("exclude")
        self.save_subject_eval_choice()

    def annotate(self, decision):
        if self.current_record:
            kind, key = self.current_record
            self._change(lambda: self.session.annotate(kind, key, **decision))

    def save_reviewer(self):
        value = self.reviewer_var.get()
        self._change(lambda: self.session.set_reviewer(value))


    def complete(self):
        if getattr(self.session,'_history_inspection_only',False):
            self.status_var.set('Wyłącznie historia — brak wywołań dopuszczonych do weryfikacji.')
            return
        if getattr(self, "_canvas_editor", None) is not None and not self._canvas_editor.prepare_navigation():
            return
        if not self.session or self._saving:
            return
        state = verification_progress(self.session)
        if state["status"] == "READY_FOR_EVALUATION":
            self.status_var.set(f"Zapisano automatycznie • rewizja {self.session.review.review_revision}")
            return
        if state["issues"]:
            self.status_var.set(f"Weryfikacja niekompletna: {len(state['issues'])} brakujących decyzji. "
                                + state["issues"][0])
            return
        # A separate operator confirmation is necessary even if all statuses
        # are green. Completeness is not a scientifically validated result.
        from tkinter import messagebox
        if not messagebox.askyesno("Zatwierdzenie weryfikacji",
                "Wszystkie wymagane decyzje są zapisane. Zatwierdzić formularz jako "
                "materiał gotowy do późniejszej ewaluacji? Pomiar ilościowy GT narożników "
                "pozostaje osobny. Utworzona wersja dokumentu będzie niezmienna.",
                parent=self.window):
            return
        self._change(self.session.complete,
                     after=lambda: self.status_var.set(
                         f"Zapisano automatycznie • rewizja {self.session.review.review_revision}"))


    def next_pending(self):
        a3=getattr(self,'_f5a_a3',None)
        if a3 and a3.active:return a3.next_pending()
        if getattr(self, "_canvas_editor", None) is not None:
            return self._canvas_editor.next_pending()
        # Stage MT takes priority; never send operator back to an already assessed
        # missed plate solely because OCR never ran in that attempt.
        stages = getattr(self, "_science_data", {}).get("subjects", {})
        for iid, key in self._subject_ids.items():
            if stages.get(key, {}).get("mt_pending"):
                self.subject_tree.selection_set(iid)
                self.subject_tree.see(iid)
                self._select_subject()
                return
        for iid, key in self._subject_ids.items():
            if stages.get(key, {}).get("mz_pending"):
                self.subject_tree.selection_set(iid)
                self.subject_tree.see(iid)
                self._select_subject()
                return
        self.status_var.set("Nie ma brakujących decyzji MT ani odczytów MZ do weryfikacji. "
                            "Zobacz «Analiza wyników»; brak uruchomienia MZ nie blokuje weryfikacji MT.")

    def _export_press(self, event=None):
        # Widget binding precedes Tk's Button class focus handling. Export
        # cannot finish a GTZ field merely by taking keyboard focus from it.
        if not self._prepare_export():
            return 'break'

    def _export_has_focus(self):
        return self.window.focus_get() is getattr(self,'export_button',None)

    def _prepare_export(self):
        if not self.session or self._saving:
            return False
        if self.reviewer_var.get().strip()!=self.session.review.reviewer_id:
            self.status_var.set('Zakończ lub anuluj edycję przed eksportem. Eksport odczytuje zapisany stan.')
            return False
        editor=getattr(self,'_canvas_editor',None)
        if editor is not None:
            return editor.prepare_export()
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id);self._autosave_id=None
        old=self.session.review.subjects.get(self.subject_key,{})
        dirty=(normalize_registration(old.get('ground_truth'))!=normalize_registration(self.gt_var.get())
               or old.get('note','')!=self.note_var.get()
               or self.reviewer_var.get().strip()!=self.session.review.reviewer_id)
        if dirty:
            self.status_var.set('Zakończ lub anuluj edycję przed eksportem. Eksport odczytuje zapisany stan.')
            return False
        return True

    def export(self):
        if not self.session or self._saving:
            return
        if not self._prepare_export():
            return
        directory = filedialog.askdirectory(
            parent=self.window,
            title="Wybierz główny folder 03_wyniki — sesja i rewizja powstaną automatycznie")
        if not directory:
            return
        try:
            base, _session_dir, _rev = validate_export_destination(
                Path(directory), self.session.session_id, self.session.review.review_revision)
            proposed = next_export_path(base, self.session.session_id,
                                        self.session.review.review_revision)
        except ValueError as error:
            self.status_var.set(str(error))
            messagebox.showerror("Nieprawidłowe miejsce eksportu", str(error), parent=self.window)
            return
        if not messagebox.askyesno(
                "Eksport jednej sesji",
                "Każda sesja i rewizja weryfikacji otrzyma własny folder.\n\n"
                f"Planowane miejsce:\n{proposed}\n\n"
                f"Stan weryfikacji: {self.session.review.review_status}; nie zmieni się przez eksport.\n"
                "Istniejące pliki w 03_wyniki i porownanie zostaną bez zmian.\n"
                "Eksport może mieć inny numer końcowy, jeżeli równolegle powstanie inny eksport.\n\n"
                "Rozpocząć eksport?",
                parent=self.window):
            return
        paths = []
        result_dir = {"path": None, "file_count": 0}
        def export_all():
            from ..ranking.mobile_mt_object_export import capture_review_state,_publish_guard
            review_state=capture_review_state(self.session)
            # The saved GT review (not a draft or earlier session) determines the revision.
            target = reserve_export_path(base, self.session.session_id,
                                         self.session.review.review_revision)
            result_dir["path"] = target
            try:
                paths.extend(self.session.export(target))
                from .mobile_bbox_spatial_review import export_addendum
                paths.extend(export_addendum(self.session,target))
                from .mobile_mz_quality_context import export as export_mz_context
                paths.extend(export_mz_context(self.session,target))
                export_path = target / "mt_pose_geometry.csv"
                temp = export_path.with_name(export_path.name + ".tmp")
                rows = _pose_geometry_rows(self.session)
                columns = tuple(rows[0]) if rows else (
                    "session_id", "source_archive_sha256", "attempt_id", "metric_status")
                try:
                    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(rows)
                    os.replace(temp, export_path)
                finally:
                    if temp.exists():
                        temp.unlink()
                paths.append(export_path)
                paths.extend(export_thesis_bundle(
                    self.session, self.session.statistics(), rows, tuple(paths), target))
                scientific = make_scientific_summary(self.session, self.session.statistics(), rows)
                paths.extend(write_scientific_companion(scientific, target))
                multi = multi_scientific_summary(self.session)
                multi_path = target / "mt_multi_plate_verified.json"
                temp_multi = multi_path.with_name(multi_path.name + ".tmp")
                try:
                    temp_multi.write_text(json.dumps(multi, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(temp_multi, multi_path)
                finally:
                    if temp_multi.exists():
                        temp_multi.unlink()
                paths.append(multi_path)
                # New diagnostic population, separate/versioned files. The
                # existing final MT+MZ report and its definitions are untouched.
                from ..ranking.mobile_candidate_stacks import export as export_candidates
                candidate_dir = export_candidates(self.session, target)
                paths.extend(sorted(p for p in candidate_dir.iterdir() if p.is_file()))
                from ..ranking.mobile_mt_object_evaluation import export as export_objects
                paths.extend(export_objects(self.session, target))
                _publish_guard(self.session,*review_state)
                from ..ranking.mobile_f4_variant import for_session
                paths.append(finalize_export_manifest(self.session, target, paths,
                    variant=for_session(self.session),
                    before_publish=lambda:_publish_guard(self.session,*review_state)))
                result_dir["file_count"] = sum(1 for path in target.rglob("*") if path.is_file())
            except Exception as error:
                mark_export_incomplete(target, error)
                raise RuntimeError(f"Eksport nieukończony w {target}: {error}") from error
        self._change(export_all,save_draft=False,update_ranking=False,
                     after=lambda: self.status_var.set(
                         f"Eksport jednej sesji: {result_dir['file_count']} plików do {result_dir['path']} "
                         "· poprzednie wyniki zachowane"))

    def _refresh_stats(self, prepared=None):
        form = prepared['form'] if prepared is not None else verification_progress(self.session)
        canvas_editor = getattr(self, "_canvas_editor", None)
        if hasattr(self, "form_status_label") and canvas_editor is None:
            titles = {"IN_PROGRESS": "Weryfikacja w toku", "READY_TO_APPROVE": "Obrazy sprawdzone — zatwierdź weryfikację",
                      "READY_FOR_EVALUATION": "Weryfikacja zatwierdzona — gotowa do ewaluacji",
                      "APPROVED_PENDING_SNAPSHOT": "Zatwierdzone decyzje — pakiet do weryfikacji",
                      "NOT_STARTED": "Weryfikacja nierozpoczęta"}
            self.form_status_label.configure(text=(
                f"{titles.get(form['status'], 'Weryfikacja w toku')} • "
                f"obserwacje z wymaganymi decyzjami: {form['verified']}/{form['total']} • "
                f"pozostałe decyzje: {len(form['issues'])} • oryginał MT i korekta są osobnymi ramkami"))
        summary = self.stats["summary"]
        progress = self._science_data["progress"]
        if canvas_editor is None:
            self.progress_label.configure(text=(
                f"Obserwacje: {progress['subjects_total']} • "
                f"Liczba tablic MT: {progress['mt_invocations_cardinality_reviewed']}/{progress['mt_executed_active']} "
                f"wywołań z decyzją operatora • "
                f"Weryfikacja: {REVIEW_LABELS[self.session.review.review_status]}"))
        self.stats_tree.delete(*self.stats_tree.get_children())
        sections = {
            "Tablice / ALPR": {"evaluable_subjects": "Tablice do weryfikacji", "subjects_with_exact_read": "Z poprawnym odczytem", "subjects_without_exact_read": "Bez poprawnego odczytu", "subject_success_rate": "Skuteczność systemu dla tablic"},
            "Odczyty MZ": {"evaluable_reads": "Zweryfikowane obrazy tablic", "exact_reads": "Poprawne odczyty", "incorrect_reads": "Niepoprawne odczyty", "no_reads": "Brak odczytu", "exact_read_rate": "Udział poprawnych odczytów", "no_read_rate": "Udział braku odczytu"},
            "Znaki MZ": {"gt_characters": "Znaki GT", "correct_characters": "Poprawnie rozpoznane znaki", "incorrect_characters": "Błędnie rozpoznane znaki", "missing_characters": "Brakujące znaki", "extra_characters": "Znaki nadmiarowe", "cer": "CER — suma błędów / suma znaków GT"},
            "Lokalizacja MT": {"evaluable_mt_invocations": "Zweryfikowane wywołania z jedną widoczną tablicą", "mt_successful_invocations": "Wywołania z poprawną lokalizacją", "mt_no_detection_invocations": "Wywołania bez detekcji", "mt_execution_error_invocations": "Wywołania z błędem wykonania", "mt_invalid_quad_invocations": "Wywołania z błędną geometrią", "mt_multi_plate_invocations": "Wywołania z wieloma tablicami", "mt_uncertain_invocations": "Wywołania niejednoznaczne", "mt_false_detections": "Fałszywe detekcje", "mt_localization_success_rate": "Skuteczność lokalizacji MT w wywołaniach z jedną tablicą"},
            "Czas i próby": {"median_time_to_first_exact_ms": "Mediana czasu do próby z poprawnym odczytem [ms]", "p90_time_to_first_exact_ms": "P90 czasu do próby z poprawnym odczytem [ms]", "median_attempts_to_first_exact": "Mediana liczby prób do poprawnego odczytu"},
        }
        for section, fields in sections.items():
            for key, label in fields.items():
                self.stats_tree.insert("", "end", values=(section, label, _display(summary.get(key), rate=key.endswith("rate") or key == "cer")))
        pose = prepared['pose'] if prepared is not None else _pose_geometry_summary(self.session)
        for key, label in (
            ("pose_eligible", "Prawdziwe tablice MT Pose do pomiaru"),
            ("pose_measured", "Polygony z ręcznie wyznaczonym GT — mianownik"),
            ("pose_pending", "Czeka na ręczne GT narożników"),
            ("pose_manual_unavailable", "GT rogów niewyznaczalne — wyłączone z pomiaru"),
            ("pose_no_geometry", "Brak zweryfikowanych 4 punktów MT w eksporcie"),
            ("missed_detection_manual_gt", "Pominięte tablice z zapisanymi rogami GT (bez IoU/NMCE)"),
            ("pose_nmce_mean_pct", "Średni znormalizowany błąd narożników NMCE [%]"),
            ("pose_iou_mean_pct", "Średnie Polygon IoU [%]"),
        ):
            value = pose[key]
            shown = f"{value:.2f}%" if key.endswith("_pct") and value is not None else _display(value)
            self.stats_tree.insert("", "end", values=("Geometria MT Pose / GT", label, shown))
        for row in self.stats["character_confusion"]:
            self.stats_tree.insert("", "end", values=("Pomyłki znaków", f"{row['ground_truth']} → {row['prediction']}", row["count"]))
        if hasattr(self, "science_dashboard"):
            source_sha = str(getattr(self.session.review, "source_archive_sha256", "") or "").lower()
            bundle_sha = str(getattr(self.session.bundle, "source_archive_sha256", "") or "").lower()
            summary_sha = str(self._science_data.get("source_archive_sha256") or "").lower()
            source_valid = bool(
                getattr(getattr(self.session.bundle, "validation", None), "ok", False)
                and source_sha and source_sha == bundle_sha == summary_sha
                and str(self.session.session_id) == str(self._science_data.get("session_id")))
            if source_valid:
                try:
                    self.session.verify_source()
                except (OSError, ValueError, RuntimeError):
                    source_valid = False
            self.science_dashboard.render(self._science_data, form, source_verified=source_valid)
            self.science_caption.configure(text="Analiza historyczna · wyniki i czasy telefonu")
        if hasattr(self, "conditions_view"):
            self.conditions_view.configure(state="normal")
            self.conditions_view.delete("1.0", "end")
            self.conditions_view.insert("end", self._conditions_text())
            self.conditions_view.configure(state="disabled")
        if getattr(self, "_canvas_editor", None) is not None:
            self._canvas_editor.update_report()

    def _toggle_scientific_details(self):
        self._science_raw_open = not self._science_raw_open
        if self._science_raw_open:
            self.science_dashboard.frame.grid_remove()
            self._science_raw_host.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=12, pady=(3, 12))
            self.science_toggle.configure(text="Wróć do podsumowania")
        else:
            self._science_raw_host.grid_remove()
            self.science_dashboard.frame.grid(row=1, column=0, rowspan=2,
                                              sticky="nsew", padx=(12, 12), pady=(3, 12))
            self.science_toggle.configure(text="Tabela techniczna")

    def _toggle_provenance_details(self):
        self._provenance_raw_open = not self._provenance_raw_open
        if self._provenance_raw_open:
            self._provenance_raw_host.grid(row=2, column=0, sticky="nsew", padx=12, pady=(2, 9))
            self.details_toggle.configure(text="Ukryj pełną proweniencję techniczną")
        else:
            self._provenance_raw_host.grid_remove()
            self.details_toggle.configure(text="Pokaż pełną proweniencję techniczną")

    def _populate_details(self):
        values = {"session_id": self.session.session_id, "źródło": str(self.session.path),
                  "SHA-256": self.session.review.source_archive_sha256, "weryfikacja": str(self.session.sidecar_path),
                  "schemat próbek": self.session.bundle.sample_schema,
                  "eksperyment": self.session.bundle.experiment_session,
                  "capture": self.session.bundle.report_payload.get("capture", {}),
                  "pochodzenie modeli i warianty": self.session.review.provenance,
                  "błędy wykonania MT": {key: list(group.execution_errors) for key, group in self.session.mt_invocations.items() if group.execution_failed}}
        values['dowód wejścia MZ (odczytowo)'] = self.session.mz_input_evidence
        if self.session.tv_bbox_evidence.get('available'):
            values['wejście TV BBOX (odczytowo)'] = self.session.tv_bbox_evidence
        def append(value, prefix=""):
            for key, item in value.items():
                name = f"{prefix} / {key}" if prefix else str(key)
                if isinstance(item, dict):
                    append(item, name)
                else:
                    self.details_tree.insert("", "end", values=(name, json.dumps(item, ensure_ascii=False) if isinstance(item, list) else str(item)))
        append(values)
        if hasattr(self, "conditions_view"):
            self.conditions_view.configure(state="normal")
            self.conditions_view.delete("1.0", "end")
            self.conditions_view.insert("end", self._conditions_text())
            self.conditions_view.configure(state="disabled")

    def _conditions_text(self):
        from ..ranking.mobile_mz_input_evidence import diagnostic_text
        from ..ranking.mobile_tv_bbox_evidence import diagnostic_text as tv_text
        return (session_conditions_text(self._science_data['conditions']) + '\n\n'
                + diagnostic_text(self.session.mz_input_evidence)
                + ('\n\n' + tv_text(self.session.tv_bbox_evidence) if self.session.tv_bbox_evidence.get('available') else ''))

    def close(self):
        if self._closed:
            return
        from .mobile_canvas_zoom import cancel_zoom
        cancel_zoom(self)
        if getattr(self, "_canvas_editor", None) is not None and not self._canvas_editor.prepare_navigation():
            return
        self.save_gt()
        if self._saving:
            self._close_requested = True
            self.status_var.set("Kończę zapis weryfikacji przed zamknięciem…")
            return
        if getattr(self, "_canvas_editor", None) is not None:
            self._canvas_editor.dispose()
        from .mobile_canvas_statistics_refresh import cancel
        cancel(self)
        self._closed = True
        self._close_crop_preview()
        if getattr(self, "_status_paint_after", None):
            self.subject_tree.after_cancel(self._status_paint_after)
            self._status_paint_after = None
        self.window.after_cancel(self._poll_id)
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()

    def _destroyed(self, event):
        if event.widget is self.window:
            self._closed = True
            from .mobile_canvas_statistics_refresh import cancel
            cancel(self)
            self._executor.shutdown(wait=False, cancel_futures=False)


def open_mobile_sample_review(browser, report, bundle=None):
    reviews = {key: panel for key, panel in getattr(browser, "_sample_reviews", {}).items() if not panel._closed}
    key = report.source_archive_sha256 or report.source_path
    current = reviews.get(key)
    if current is not None and not current._closed:
        browser._sample_review = current
        current.window.lift()
        current.window.focus_set()
        return current
    from .mobile_f5a_scene_gt import open_if_supported
    try:
        scene_review = open_if_supported(browser, report, bundle)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror('GT scen F5A', 'Nie otwarto GT scen: '+str(exc), parent=browser.window)
        return None
    browser._sample_review = scene_review or MobileSampleReviewWindow(browser, report, bundle)
    reviews[key] = browser._sample_review
    browser._sample_reviews = reviews
    return browser._sample_review
