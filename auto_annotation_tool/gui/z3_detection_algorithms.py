from __future__ import annotations

import copy

from ..character_recognition import DetectionMethod
from .z3_character_box_refiner import refine_perfect_character_box

DETECTION_CONF_MIN = 0.00001
DETECTION_CONF_DIGITS = 5


def _normalize_detection_conf(value, default: float = 0.25) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if number != number:
        number = float(default)
    return round(max(DETECTION_CONF_MIN, min(1.0, number)), DETECTION_CONF_DIGITS)


def levenshtein_distance(left: str, right: str) -> int:
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

    return previous[-1]


def best_text_distance(host, candidate_text: str, true_texts) -> int:
    normalized = [str(item or "").strip().upper() for item in (true_texts or []) if str(item or "").strip()]
    if not normalized:
        return 10 ** 9

    candidate = str(candidate_text or "").strip().upper()
    return min(host._levenshtein_distance(candidate, truth) for truth in normalized)


def fit_detection_count_to_truths(host, detections, true_texts):
    ordered = host._sort_character_records_by_x(list(detections or []))
    normalized_truths = [
        str(item or "").strip().upper()
        for item in (true_texts or [])
        if str(item or "").strip()
    ]
    if not ordered or not normalized_truths:
        return ordered, None

    detection_count = len(ordered)
    if detection_count <= 0:
        return ordered, None

    if any(len(truth) == detection_count for truth in normalized_truths):
        return ordered, None

    candidates = [truth for truth in normalized_truths if 0 < len(truth) <= detection_count]
    if not candidates:
        return ordered, None

    best_window = None
    best_details = None
    best_score = None

    for truth in candidates:
        expected_len = len(truth)
        for start_idx in range(0, detection_count - expected_len + 1):
            window = ordered[start_idx:start_idx + expected_len]
            candidate_text = host._characters_to_text(window)
            distance = host._levenshtein_distance(candidate_text, truth)
            avg_conf = (
                sum(host._char_record_confidence(rec) for rec in window) / float(len(window))
                if window else 0.0
            )
            dropped_left = int(start_idx)
            dropped_right = int(detection_count - (start_idx + expected_len))
            score = (distance, -avg_conf, dropped_left + dropped_right, dropped_left)
            if best_score is None or score < best_score:
                best_score = score
                best_window = window
                best_details = {
                    "expected_text": truth,
                    "window_text": candidate_text,
                    "original_box_count": int(detection_count),
                    "trimmed_box_count": int(expected_len),
                    "trimmed_extra_boxes": int(detection_count - expected_len),
                    "trimmed_left_boxes": dropped_left,
                    "trimmed_right_boxes": dropped_right,
                    "distance": int(distance),
                }

    if best_window is None or best_details is None:
        return ordered, None
    if int(best_details.get("trimmed_extra_boxes", 0) or 0) <= 0:
        return ordered, None
    return list(best_window), best_details


def apply_final_truth_count_guard(host, characters, true_texts, fusion_details: dict | None = None, data=None):
    ordered_chars = host._sort_character_records_by_x(list(characters or []), data=data)
    fitted_chars, trim_details = host._fit_detection_count_to_truths(ordered_chars, true_texts)
    if not isinstance(trim_details, dict) or not trim_details:
        return ordered_chars, fusion_details

    expected_len = int(trim_details.get("trimmed_box_count", 0) or len(fitted_chars or []) or 0)
    manual_ids = {
        id(rec)
        for rec in ordered_chars
        if isinstance(rec, dict) and host._is_manual_character_record(rec, data=data)
    }
    if manual_ids:
        manual_records = [
            rec for rec in ordered_chars
            if isinstance(rec, dict) and id(rec) in manual_ids
        ]
        fitted_ids = {id(rec) for rec in list(fitted_chars or [])}
        if not manual_ids.issubset(fitted_ids):
            selected_records = []
            selected_ids = set()

            def add_record(rec) -> None:
                if rec is None:
                    return
                rec_id = id(rec)
                if rec_id in selected_ids:
                    return
                selected_records.append(rec)
                selected_ids.add(rec_id)

            for rec in manual_records:
                add_record(rec)
                if expected_len > 0 and len(selected_records) >= expected_len:
                    break

            if expected_len <= 0 or len(manual_records) <= expected_len:
                for candidate_source in (list(fitted_chars or []), ordered_chars):
                    for rec in candidate_source:
                        if expected_len > 0 and len(selected_records) >= expected_len:
                            break
                        if not isinstance(rec, dict) or id(rec) in manual_ids:
                            continue
                        try:
                            if host._character_record_collides_with_manual(rec, manual_records):
                                continue
                        except Exception:
                            pass
                        add_record(rec)
                        if expected_len > 0 and len(selected_records) >= expected_len:
                            break
                    if expected_len > 0 and len(selected_records) >= expected_len:
                        break

            merged_details = dict(fusion_details or {})
            merged_details.update(trim_details)
            merged_details["gt_count_guard_applied"] = True
            merged_details["gt_count_guard_stage"] = "final_characters"
            merged_details["gt_count_guard_manual_protected"] = True
            merged_details["gt_count_guard_manual_count"] = int(len(manual_ids))
            merged_details["gt_count_guard_manual_safe_trim"] = True
            merged_details["gt_count_guard_manual_over_limit"] = bool(expected_len > 0 and len(manual_records) > expected_len)
            merged_details["original_box_count"] = int(len(ordered_chars))
            merged_details["trimmed_box_count"] = int(len(selected_records))
            merged_details["trimmed_extra_boxes"] = max(0, int(len(ordered_chars) - len(selected_records)))
            return host._sort_character_records_by_x(list(selected_records), data=data), merged_details

    merged_details = dict(fusion_details or {})
    merged_details.update(trim_details)
    merged_details["gt_count_guard_applied"] = True
    merged_details["gt_count_guard_stage"] = "final_characters"
    return host._sort_character_records_by_x(list(fitted_chars or []), data=data), merged_details


def pick_best_true_text(host, candidate_text: str, true_texts, same_length_only: bool = False) -> str:
    normalized = [str(item or "").strip().upper() for item in (true_texts or []) if str(item or "").strip()]
    if same_length_only:
        normalized = [item for item in normalized if len(item) == len(str(candidate_text or ""))]

    if not normalized:
        return ""

    candidate = str(candidate_text or "").strip().upper()
    return min(
        normalized,
        key=lambda item: (
            host._levenshtein_distance(candidate, item),
            abs(len(candidate) - len(item)),
            item,
        )
    )


def get_text_mismatch_positions(candidate_text: str, expected_text: str):
    candidate = str(candidate_text or "").strip().upper()
    expected = str(expected_text or "").strip().upper()

    if not candidate or not expected or len(candidate) != len(expected):
        return []

    return [idx for idx, (left, right) in enumerate(zip(candidate, expected)) if left != right]


def repair_ocr_with_yolo_boxes(host, ocr_detections, yolo_detections, true_texts, max_mismatch_count: int = 999):
    ordered_ocr = host._sort_character_records_by_x(list(ocr_detections or []))
    ordered_yolo = host._sort_character_records_by_x(list(yolo_detections or []))

    if not ordered_ocr or not ordered_yolo:
        return None, None

    try:
        requested_mismatch_limit = int(max_mismatch_count or 0)
    except Exception:
        requested_mismatch_limit = 0

    ocr_text = host._characters_to_text(ordered_ocr)
    expected_text = host._pick_best_true_text(ocr_text, true_texts, same_length_only=True)
    if not expected_text:
        return None, None

    mismatch_positions = host._get_text_mismatch_positions(ocr_text, expected_text)
    if not mismatch_positions:
        return None, None

    expected_box_count = max(1, len(expected_text))
    auto_limit = requested_mismatch_limit <= 0 or requested_mismatch_limit >= expected_box_count
    max_mismatch_count = expected_box_count if auto_limit else max(1, requested_mismatch_limit)
    if len(mismatch_positions) > max_mismatch_count:
        return None, None

    repaired = [host._clone_character_detection(det) for det in ordered_ocr]
    used_yolo_indices = set()

    for slot_index in mismatch_positions:
        rescue_index = host._find_best_yolo_rescue_index(
            slot_index,
            expected_text[slot_index],
            ordered_ocr,
            ordered_yolo,
            used_yolo_indices,
        )
        if rescue_index is None:
            return None, None

        used_yolo_indices.add(rescue_index)
        repaired[slot_index] = host._clone_character_detection(
            ordered_yolo[rescue_index],
            character=expected_text[slot_index],
            method="yolo",
        )

    repaired = host._sort_character_records_by_x(repaired)
    repaired_text = host._characters_to_text(repaired)
    if repaired_text != expected_text:
        return None, None

    details = {
        "ocr_text": ocr_text,
        "yolo_text": host._characters_to_text(ordered_yolo),
        "expected_text": expected_text,
        "mismatch_positions": list(mismatch_positions),
        "max_mismatch_count": int(max_mismatch_count),
        "rescue_limit_mode": "auto" if auto_limit else "manual",
        "expected_box_count": int(expected_box_count),
    }
    return repaired, details


def get_yolo_runtime_settings(host):
    fallback_conf = _normalize_detection_conf(host.yolo_conf_var.get())
    try:
        box_conf = _normalize_detection_conf(host.yolo_box_conf_var.get(), fallback_conf)
    except Exception:
        box_conf = fallback_conf
    try:
        symbol_conf = _normalize_detection_conf(host.yolo_symbol_conf_var.get(), fallback_conf)
    except Exception:
        symbol_conf = fallback_conf
    conf = _normalize_detection_conf(min(float(box_conf), float(symbol_conf)), fallback_conf)
    iou = float(host.yolo_iou_var.get())
    overlap = float(host.yolo_overlap_var.get())
    seq_center_y = float(host.yolo_seq_center_y_var.get())
    seq_min_h = float(host.yolo_seq_min_h_ratio_var.get())
    seq_max_h = float(host.yolo_seq_max_h_ratio_var.get())
    seq_max_w = float(host.yolo_seq_max_w_ratio_var.get())
    seq_soft_overlap = float(host.yolo_seq_soft_overlap_var.get())
    seq_hard_overlap = float(host.yolo_seq_hard_overlap_var.get())

    if not (DETECTION_CONF_MIN <= box_conf <= 1.0):
        raise ValueError("Prog confidence YB musi byc w zakresie 0.00001-1.00.")
    if not (DETECTION_CONF_MIN <= symbol_conf <= 1.0):
        raise ValueError("Prog confidence YS musi byc w zakresie 0.00001-1.00.")
    if not (DETECTION_CONF_MIN <= conf <= 1.0):
        raise ValueError("Prog confidence YOLO musi byc w zakresie 0.00001-1.00.")
    if not (0.01 <= iou <= 0.99):
        raise ValueError("Próg NMS IoU musi być w zakresie 0.01-0.99.")
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("Próg nakładania boxów musi być w zakresie 0.00-1.00.")
    if not (0.10 <= seq_center_y <= 1.50):
        raise ValueError("Tolerancja osi Y dla filtra sekwencji musi być w zakresie 0.10-1.50.")
    if not (0.20 <= seq_min_h <= 1.00):
        raise ValueError("Minimalna zgodność wysokości znaku musi być w zakresie 0.20-1.00.")
    if not (1.00 <= seq_max_h <= 3.50):
        raise ValueError("Maksymalna wysokość znaku względem mediany musi być w zakresie 1.00-3.50.")
    if seq_min_h >= seq_max_h:
        raise ValueError("Minimalna zgodność wysokości musi być mniejsza od maksymalnej wysokości względem mediany.")
    if not (1.00 <= seq_max_w <= 4.50):
        raise ValueError("Maksymalna szerokość znaku względem mediany musi być w zakresie 1.00-4.50.")
    if not (0.0 <= seq_soft_overlap <= 1.0):
        raise ValueError("Miękki próg konfliktu nakładania musi być w zakresie 0.00-1.00.")
    if not (0.0 <= seq_hard_overlap <= 1.0):
        raise ValueError("Twardy próg konfliktu nakładania musi być w zakresie 0.00-1.00.")
    if seq_soft_overlap > seq_hard_overlap:
        raise ValueError("Miękki próg konfliktu nie może być większy od twardego progu konfliktu.")

    return {
        "conf": conf,
        "box_conf": box_conf,
        "symbol_conf": symbol_conf,
        "iou": iou,
        "overlap": overlap,
        "agnostic_nms": bool(host.yolo_agnostic_nms_var.get()),
        "seq_center_y": seq_center_y,
        "seq_min_h": seq_min_h,
        "seq_max_h": seq_max_h,
        "seq_max_w": seq_max_w,
        "seq_soft_overlap": seq_soft_overlap,
        "seq_hard_overlap": seq_hard_overlap,
    }


def build_pz2_detection_guard_counts(host) -> dict:
    metadata = host.preview_metadata if isinstance(getattr(host, "preview_metadata", None), dict) else {}
    result = {
        "total": 0,
        "with_chars": 0,
        "perfect": 0,
        "perfect_ocr": 0,
        "manual": 0,
        "manual_boxes": 0,
        "perfect_with_manual": 0,
    }

    for _pid, data in metadata.items():
        if not isinstance(data, dict):
            continue
        result["total"] += 1

        chars = data.get("characters", [])
        if not isinstance(chars, list):
            chars = []
        if chars:
            result["with_chars"] += 1

        is_perfect = host._is_existing_plate_perfect(chars, data=data)
        if is_perfect:
            result["perfect"] += 1
            try:
                if host._get_perfect_strategy_bucket(data) == "ocr_exact":
                    result["perfect_ocr"] += 1
            except Exception:
                pass

        manual_count = 0
        for rec in chars:
            if isinstance(rec, dict) and host._is_manual_character_record(rec, data=data):
                manual_count += 1
        if manual_count > 0:
            result["manual"] += 1
            result["manual_boxes"] += int(manual_count)
            if is_perfect:
                result["perfect_with_manual"] += 1

    result["needs_detection"] = max(0, int(result["total"]) - int(result["perfect"]))
    return result


def find_best_yolo_rescue_index(host, slot_index: int, expected_char: str, ocr_detections, yolo_detections, used_indices, min_index: int | None = None):
    self = host
    if slot_index < 0 or slot_index >= len(ocr_detections):
        return None

    expected_char = str(expected_char or "").strip().upper()
    if not expected_char:
        return None

    slot_rec = ocr_detections[slot_index]
    _, slot_center_x = self._char_record_to_symbol_and_x(slot_rec, fallback_index=slot_index)
    slot_center_y = self._char_record_center_y(slot_rec)
    ocr_width = max(1.0, self._char_record_width(slot_rec))
    ocr_height = max(1.0, self._char_record_height(slot_rec))
    all_ocr_widths = [self._char_record_width(det) for det in ocr_detections if self._char_record_width(det) > 0.0]
    all_ocr_heights = [self._char_record_height(det) for det in ocr_detections if self._char_record_height(det) > 0.0]
    if all_ocr_widths:
        sorted_widths = sorted(all_ocr_widths)
        median_width = float(sorted_widths[len(sorted_widths) // 2])
    else:
        median_width = ocr_width
    if all_ocr_heights:
        sorted_heights = sorted(all_ocr_heights)
        median_height = float(sorted_heights[len(sorted_heights) // 2])
    else:
        median_height = ocr_height

    allowed_gap = max(6.0, ocr_width * 0.85, median_width * 0.75)
    allowed_y_gap = max(8.0, ocr_height * 0.85, median_height * 0.75)

    if slot_index < len(yolo_detections) and slot_index not in used_indices:
        aligned_det = yolo_detections[slot_index]
        aligned_char, aligned_center_x = self._char_record_to_symbol_and_x(aligned_det, fallback_index=slot_index)
        if (min_index is None or slot_index >= int(min_index)) and str(aligned_char or "").strip().upper() == expected_char:
            aligned_center_y = self._char_record_center_y(aligned_det)
            if (
                abs(aligned_center_x - slot_center_x) <= allowed_gap * 1.35
                and abs(aligned_center_y - slot_center_y) <= allowed_y_gap * 1.35
            ):
                return slot_index

    best_index = None
    best_score = None

    for idx, det in enumerate(yolo_detections):
        if idx in used_indices:
            continue
        if min_index is not None and idx < int(min_index):
            continue

        det_char, det_center_x = self._char_record_to_symbol_and_x(det, fallback_index=idx)
        if str(det_char or "").strip().upper() != expected_char:
            continue

        center_gap = abs(float(det_center_x) - float(slot_center_x))
        if center_gap > allowed_gap * 1.35:
            continue
        center_y_gap = abs(float(self._char_record_center_y(det)) - float(slot_center_y))
        if center_y_gap > allowed_y_gap * 1.35:
            continue

        score = (
            (center_gap / allowed_gap)
            + (0.55 * (center_y_gap / allowed_y_gap))
            + (0.12 * abs(idx - slot_index))
            - min(0.20, self._char_record_confidence(det) * 0.10)
        )

        if best_score is None or score < best_score:
            best_index = idx
            best_score = score

    return best_index


def find_best_yolo_box_backend_index(
    host,
    slot_index: int,
    base_detections,
    yolo_detections,
    used_indices,
    min_index: int | None = None,
    match_stats: dict | None = None,
    geometry_first: bool = False,
):
    self = host
    if slot_index < 0 or slot_index >= len(base_detections):
        return None

    def _record_symbol(record, fallback_index: int = 0) -> str:
        try:
            symbol, _center_x = self._char_record_to_symbol_and_x(record, fallback_index=fallback_index)
        except Exception:
            symbol = ""
        return str(symbol or "").strip().upper()

    slot_rec = base_detections[slot_index]
    slot_symbol = _record_symbol(slot_rec, fallback_index=slot_index)
    _, slot_center_x = self._char_record_to_symbol_and_x(slot_rec, fallback_index=slot_index)
    slot_center_y = self._char_record_center_y(slot_rec)
    base_width = max(1.0, self._char_record_width(slot_rec))
    base_height = max(1.0, self._char_record_height(slot_rec))
    base_widths = [self._char_record_width(det) for det in base_detections if self._char_record_width(det) > 0.0]
    base_heights = [self._char_record_height(det) for det in base_detections if self._char_record_height(det) > 0.0]
    base_symbols = [_record_symbol(det, fallback_index=idx) for idx, det in enumerate(base_detections)]
    if base_widths:
        sorted_widths = sorted(base_widths)
        median_width = float(sorted_widths[len(sorted_widths) // 2])
    else:
        median_width = base_width
    if base_heights:
        sorted_heights = sorted(base_heights)
        median_height = float(sorted_heights[len(sorted_heights) // 2])
    else:
        median_height = base_height

    allowed_gap = max(8.0, base_width * 1.25, median_width * 1.05)
    allowed_y_gap = max(8.0, base_height * 0.95, median_height * 0.85)
    cross_slot_tolerance = max(5.0, median_width * 0.45, base_width * 0.42)
    lane_left = 0.0
    lane_right = float("inf")
    try:
        base_centers_x = [
            float(self._char_record_to_symbol_and_x(rec, fallback_index=base_idx)[1])
            for base_idx, rec in enumerate(base_detections)
        ]
        if slot_index > 0:
            lane_left = (float(base_centers_x[slot_index - 1]) + float(base_centers_x[slot_index])) / 2.0
        if slot_index + 1 < len(base_centers_x):
            lane_right = (float(base_centers_x[slot_index]) + float(base_centers_x[slot_index + 1])) / 2.0
    except Exception:
        lane_left = 0.0
        lane_right = float("inf")
    lane_pad = max(2.0, median_width * 0.18, base_width * 0.16)

    same_symbol_available = False
    if slot_symbol:
        for candidate_idx, candidate_rec in enumerate(yolo_detections):
            if candidate_idx in used_indices:
                continue
            if min_index is not None and candidate_idx < int(min_index):
                continue
            if _record_symbol(candidate_rec, fallback_index=candidate_idx) == slot_symbol:
                same_symbol_available = True
                break

    def _nearest_base_slot_for_candidate(candidate_rec, *, prefer_matching_symbol: bool = False):
        candidate_symbol = _record_symbol(candidate_rec)
        _, candidate_center_x = self._char_record_to_symbol_and_x(candidate_rec)
        candidate_center_y = self._char_record_center_y(candidate_rec)
        best_slot = None
        best_slot_score = None
        candidate_slot_indexes = range(len(base_detections))
        if prefer_matching_symbol and candidate_symbol:
            matching_indexes = [
                base_idx
                for base_idx, base_symbol in enumerate(base_symbols)
                if base_symbol == candidate_symbol
            ]
            if matching_indexes:
                candidate_slot_indexes = matching_indexes

        for base_idx in candidate_slot_indexes:
            base_rec = base_detections[base_idx]
            base_bbox = self._char_record_bbox(base_rec)
            if not base_bbox:
                continue
            _, base_center_x = self._char_record_to_symbol_and_x(base_rec, fallback_index=base_idx)
            base_center_y = self._char_record_center_y(base_rec)
            base_rec_width = max(1.0, self._char_record_width(base_rec))
            base_rec_height = max(1.0, self._char_record_height(base_rec))
            candidate_allowed_gap = max(8.0, base_rec_width * 1.25, median_width * 1.05)
            candidate_allowed_y_gap = max(8.0, base_rec_height * 0.95, median_height * 0.85)
            candidate_score = (
                abs(float(candidate_center_x) - float(base_center_x)) / max(1.0, candidate_allowed_gap)
                + 0.55 * abs(float(candidate_center_y) - float(base_center_y)) / max(1.0, candidate_allowed_y_gap)
            )
            if best_slot_score is None or candidate_score < best_slot_score:
                best_slot = base_idx
                best_slot_score = candidate_score
        return best_slot

    best_index = None
    best_score = None
    for idx, det in enumerate(yolo_detections):
        if idx in used_indices:
            continue
        if min_index is not None and idx < int(min_index):
            continue

        det_symbol = _record_symbol(det, fallback_index=idx)
        if slot_symbol and not geometry_first:
            if det_symbol and det_symbol != slot_symbol:
                if isinstance(match_stats, dict):
                    match_stats["symbol_mismatch_skipped_count"] = int(match_stats.get("symbol_mismatch_skipped_count", 0) or 0) + 1
                continue
            if not det_symbol and same_symbol_available:
                if isinstance(match_stats, dict):
                    match_stats["symbol_unknown_skipped_count"] = int(match_stats.get("symbol_unknown_skipped_count", 0) or 0) + 1
                continue

        _, det_center_x = self._char_record_to_symbol_and_x(det, fallback_index=idx)
        if geometry_first and not (float(lane_left) - lane_pad <= float(det_center_x) <= float(lane_right) + lane_pad):
            if isinstance(match_stats, dict):
                match_stats["cross_slot_skipped_count"] = int(match_stats.get("cross_slot_skipped_count", 0) or 0) + 1
            continue
        center_gap = abs(float(det_center_x) - float(slot_center_x))
        if center_gap > allowed_gap * 1.35:
            continue
        center_y_gap = abs(float(self._char_record_center_y(det)) - float(slot_center_y))
        if center_y_gap > allowed_y_gap * 1.35:
            continue

        nearest_any_slot = _nearest_base_slot_for_candidate(det, prefer_matching_symbol=False)
        if geometry_first:
            if nearest_any_slot != slot_index:
                if isinstance(match_stats, dict):
                    match_stats["cross_slot_skipped_count"] = int(match_stats.get("cross_slot_skipped_count", 0) or 0) + 1
                continue
        else:
            nearest_symbol_slot = _nearest_base_slot_for_candidate(det, prefer_matching_symbol=True)
            if nearest_symbol_slot != slot_index:
                continue
        if nearest_any_slot != slot_index and center_gap > cross_slot_tolerance:
            if isinstance(match_stats, dict):
                match_stats["cross_slot_skipped_count"] = int(match_stats.get("cross_slot_skipped_count", 0) or 0) + 1
            continue

        if not det_symbol and isinstance(match_stats, dict):
            match_stats["symbol_unknown_used_count"] = int(match_stats.get("symbol_unknown_used_count", 0) or 0) + 1

        symbol_penalty = 0.0
        if geometry_first and slot_symbol:
            if det_symbol == slot_symbol:
                symbol_penalty = -0.08
            elif det_symbol:
                symbol_penalty = 0.18
                if isinstance(match_stats, dict):
                    match_stats["symbol_mismatch_soft_candidate_count"] = int(match_stats.get("symbol_mismatch_soft_candidate_count", 0) or 0) + 1

        score = (
            (center_gap / allowed_gap)
            + (0.55 * (center_y_gap / allowed_y_gap))
            + (0.16 * abs(idx - slot_index))
            - min(0.18, self._char_record_confidence(det) * 0.08)
            + symbol_penalty
        )
        if best_score is None or score < best_score:
            best_index = idx
            best_score = score

    return best_index


def apply_yolo_box_backend(
    host,
    base_detections,
    yolo_detections,
    *,
    preserve_base_record_metadata: bool = False,
    protect_manual_records: bool = False,
    base_data=None,
    plate_image=None,
    enable_perfect_refiner: bool = False,
    perfect_refiner_continuity_guard: bool = True,
):
    self = host
    ordered_base = self._sort_character_records_by_x(list(base_detections or []))
    ordered_yolo = self._sort_character_records_by_x(list(yolo_detections or []))

    if not ordered_base or not ordered_yolo:
        return ordered_base, None

    expected_text = self._characters_to_text(ordered_base)
    if not expected_text:
        return ordered_base, None

    yolo_stats = self._build_char_record_geometry_stats(ordered_yolo)
    yolo_median_width = max(1.0, float(yolo_stats.get("median_width", 0.0) or 0.0))
    yolo_median_height = max(1.0, float(yolo_stats.get("median_height", 0.0) or 0.0))
    yolo_median_center_y = float(yolo_stats.get("median_center_y", 0.0) or 0.0)
    base_centers_x = []
    for base_index, base_rec in enumerate(ordered_base):
        try:
            _base_symbol, base_center_x = self._char_record_to_symbol_and_x(base_rec, fallback_index=base_index)
            base_centers_x.append(float(base_center_x))
        except Exception:
            base_centers_x.append(float(base_index))
    base_pitches = [
        max(1.0, float(base_centers_x[index + 1]) - float(base_centers_x[index]))
        for index in range(max(0, len(base_centers_x) - 1))
        if float(base_centers_x[index + 1]) > float(base_centers_x[index])
    ]
    if base_pitches:
        sorted_pitches = sorted(base_pitches)
        median_slot_pitch = max(1.0, float(sorted_pitches[len(sorted_pitches) // 2]))
    else:
        median_slot_pitch = max(1.0, yolo_median_width * 1.12)

    used_yolo_indices = set()
    retired_yolo_indices = set()
    if preserve_base_record_metadata:
        replaced = [copy.deepcopy(det) for det in ordered_base]
    else:
        replaced = [self._clone_character_detection(det) for det in ordered_base]
    replaced_positions = []
    manual_protected_positions = []
    last_yolo_index = -1
    match_stats = {
        "symbol_guard_enabled": True,
        "symbol_mismatch_skipped_count": 0,
        "symbol_unknown_skipped_count": 0,
        "symbol_unknown_used_count": 0,
        "cross_slot_skipped_count": 0,
        "partial_overlap_skipped_count": 0,
        "perfect_refiner_count": 0,
        "perfect_refiner_failed_count": 0,
        "perfect_refiner_iterations": 0,
        "order_guard_retired_count": 0,
        "single_slot_guard_skipped_count": 0,
        "sequence_guard_rejected_count": 0,
        "sequence_guard_order_count": 0,
        "sequence_guard_overlap_count": 0,
        "sequence_guard_width_count": 0,
        "sequence_guard_lane_count": 0,
        "sequence_guard_neighbor_count": 0,
    }

    def _retire_yolo_candidate(candidate_index):
        nonlocal last_yolo_index
        try:
            candidate_index = int(candidate_index)
        except Exception:
            return
        retired_yolo_indices.add(candidate_index)
        last_yolo_index = max(int(last_yolo_index), candidate_index)
        match_stats["order_guard_retired_count"] = int(match_stats.get("order_guard_retired_count", 0) or 0) + 1

    def _candidate_slot_overlap_quality(base_bbox, candidate_bbox) -> dict:
        try:
            bx1, by1, bx2, by2 = [float(value) for value in base_bbox[:4]]
            cx1, cy1, cx2, cy2 = [float(value) for value in candidate_bbox[:4]]
        except Exception:
            return {"ok": False, "reason": "invalid_bbox"}

        base_w = max(1.0, bx2 - bx1)
        base_h = max(1.0, by2 - by1)
        candidate_w = max(1.0, cx2 - cx1)
        candidate_h = max(1.0, cy2 - cy1)
        inter_w = max(0.0, min(bx2, cx2) - max(bx1, cx1))
        inter_h = max(0.0, min(by2, cy2) - max(by1, cy1))
        horizontal_overlap = inter_w / max(1.0, min(base_w, candidate_w))
        vertical_overlap = inter_h / max(1.0, min(base_h, candidate_h))
        center_gap = abs(((cx1 + cx2) / 2.0) - ((bx1 + bx2) / 2.0))
        normalized_center_gap = center_gap / max(1.0, min(base_w, candidate_w))

        # YOLO ma prawo korygowac OCR, ale nie moze "poprawiac" slotu boxem,
        # ktory jedynie zahacza o znak. Taka poprawka wyglada jak pelny box
        # przypiety do sasiedniego znaku albo do fragmentu miedzy znakami.
        ok = True
        if horizontal_overlap < 0.12:
            ok = False
        elif horizontal_overlap < 0.32 and normalized_center_gap > 0.36:
            ok = False
        if vertical_overlap < 0.42:
            ok = False

        return {
            "ok": bool(ok),
            "horizontal_overlap": float(horizontal_overlap),
            "vertical_overlap": float(vertical_overlap),
            "normalized_center_gap": float(normalized_center_gap),
        }

    def _slot_lane_bounds(slot_index: int):
        try:
            slot_center = float(base_centers_x[slot_index])
        except Exception:
            slot_center = float(slot_index) * median_slot_pitch
        if slot_index > 0:
            left = (float(base_centers_x[slot_index - 1]) + slot_center) / 2.0
        else:
            left = slot_center - median_slot_pitch / 2.0
        if slot_index + 1 < len(base_centers_x):
            right = (slot_center + float(base_centers_x[slot_index + 1])) / 2.0
        else:
            right = slot_center + median_slot_pitch / 2.0
        if right <= left + 1.0:
            left = slot_center - median_slot_pitch / 2.0
            right = slot_center + median_slot_pitch / 2.0
        return float(left), float(right), float(slot_center)

    def _candidate_single_slot_quality(slot_index: int, candidate_bbox) -> dict:
        try:
            cx1, _cy1, cx2, _cy2 = [float(value) for value in candidate_bbox[:4]]
        except Exception:
            return {"ok": False, "reason": "invalid_bbox"}
        if cx2 <= cx1:
            return {"ok": False, "reason": "invalid_bbox"}

        lane_left, lane_right, slot_center = _slot_lane_bounds(slot_index)
        lane_width = max(1.0, lane_right - lane_left)
        candidate_width = max(1.0, cx2 - cx1)
        candidate_center = (cx1 + cx2) / 2.0
        lane_pad = max(1.0, min(3.0, median_slot_pitch * 0.08))
        edge_pad = max(1.0, min(2.5, median_slot_pitch * 0.055))

        # Box poprawki ma prawo delikatnie wyjsc poza pas slotu, ale nie moze
        # zawierac centrum sasiedniego pola wpisu. To pilnuje semantyki
        # "sparowania" boxa z aktywnym polem znaku.
        for neighbor_index in (slot_index - 1, slot_index + 1):
            if neighbor_index < 0 or neighbor_index >= len(base_centers_x):
                continue
            neighbor_center = float(base_centers_x[neighbor_index])
            if cx1 + edge_pad <= neighbor_center <= cx2 - edge_pad:
                return {"ok": False, "reason": "neighbor_center_inside"}

        if candidate_center < lane_left - lane_pad or candidate_center > lane_right + lane_pad:
            return {"ok": False, "reason": "center_outside_slot_lane"}
        if cx1 < lane_left - lane_pad or cx2 > lane_right + lane_pad:
            left_bleed = max(0.0, lane_left - cx1)
            right_bleed = max(0.0, cx2 - lane_right)
            bleed_ratio = (left_bleed + right_bleed) / candidate_width
            if bleed_ratio > 0.18:
                return {"ok": False, "reason": "slot_lane_bleed"}
        if candidate_width > lane_width * 1.22:
            return {"ok": False, "reason": "too_wide_for_slot_lane"}
        if candidate_width > median_slot_pitch * 1.28:
            return {"ok": False, "reason": "too_wide_for_pitch"}
        if abs(candidate_center - slot_center) > max(4.0, lane_width * 0.54):
            return {"ok": False, "reason": "center_too_far_from_slot"}
        return {"ok": True, "reason": "ok"}

    def _median_number(values, fallback=1.0) -> float:
        numbers = []
        for value in values or []:
            try:
                number = float(value)
            except Exception:
                continue
            if number > 0.0:
                numbers.append(number)
        if not numbers:
            return float(fallback)
        numbers.sort()
        return float(numbers[len(numbers) // 2])

    def _slot_symbol_width_ceiling(slot_index: int) -> float:
        try:
            symbol, _slot_x = self._char_record_to_symbol_and_x(ordered_base[slot_index], fallback_index=slot_index)
            symbol = str(symbol or "").strip().upper()
        except Exception:
            symbol = ""
        if symbol in {"M", "W"}:
            return 1.75
        if symbol in {"1", "I", "L"}:
            return 1.28
        return 1.52

    def _record_bbox_4(record):
        bbox = self._char_record_bbox(record)
        if not bbox:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _sequence_geometry_quality(records) -> dict:
        # Lokalny refiner moze trafic "prawie dobrze", ale przy powtarzajacych
        # sie znakach globalna sekwencja nadal bywa przestawiona. Ten guard
        # akceptuje poprawki dopiero wtedy, gdy cala geometria znakow jest spojna.
        if len(records or []) != len(ordered_base):
            return {"ok": False, "reason": "count"}

        bboxes = []
        centers = []
        widths = []
        for slot_index, record in enumerate(records or []):
            bbox = _record_bbox_4(record)
            if bbox is None:
                return {"ok": False, "reason": "invalid_bbox", "slot": slot_index}
            x1, _y1, x2, _y2 = bbox
            width = max(1.0, x2 - x1)
            center = (x1 + x2) / 2.0
            lane_left, lane_right, slot_center = _slot_lane_bounds(slot_index)
            lane_width = max(1.0, lane_right - lane_left)
            lane_pad = max(1.0, min(3.5, median_slot_pitch * 0.09))
            edge_pad = max(1.0, min(2.5, median_slot_pitch * 0.055))

            for neighbor_index in (slot_index - 1, slot_index + 1):
                if neighbor_index < 0 or neighbor_index >= len(base_centers_x):
                    continue
                neighbor_center = float(base_centers_x[neighbor_index])
                if x1 + edge_pad <= neighbor_center <= x2 - edge_pad:
                    return {"ok": False, "reason": "neighbor", "slot": slot_index}

            if center < lane_left - lane_pad or center > lane_right + lane_pad:
                return {"ok": False, "reason": "lane", "slot": slot_index}
            if abs(center - slot_center) > max(4.0, lane_width * 0.58, median_slot_pitch * 0.58):
                return {"ok": False, "reason": "lane", "slot": slot_index}
            if x1 < lane_left - lane_pad or x2 > lane_right + lane_pad:
                left_bleed = max(0.0, lane_left - x1)
                right_bleed = max(0.0, x2 - lane_right)
                bleed_ratio = (left_bleed + right_bleed) / width
                max_bleed = 0.13 if enable_perfect_refiner else 0.18
                if bleed_ratio > max_bleed:
                    return {"ok": False, "reason": "lane", "slot": slot_index}

            bboxes.append(bbox)
            centers.append(center)
            widths.append(width)

        median_width = _median_number(widths, fallback=max(1.0, median_slot_pitch * 0.72))
        sorted_widths = sorted(widths)
        if len(sorted_widths) >= 4:
            reference_width = max(1.0, float(sorted_widths[max(0, (len(sorted_widths) // 2) - 1)]))
        else:
            reference_width = median_width

        for slot_index, width in enumerate(widths):
            symbol_limit = reference_width * _slot_symbol_width_ceiling(slot_index)
            pitch_limit = median_slot_pitch * (1.22 if _slot_symbol_width_ceiling(slot_index) > 1.6 else 1.12)
            width_limit = max(symbol_limit, pitch_limit)
            if width > width_limit:
                return {"ok": False, "reason": "width", "slot": slot_index}

        max_overlap_ratio = 0.18 if enable_perfect_refiner else 0.26
        for slot_index in range(max(0, len(bboxes) - 1)):
            left_box = bboxes[slot_index]
            right_box = bboxes[slot_index + 1]
            if centers[slot_index] >= centers[slot_index + 1]:
                return {"ok": False, "reason": "order", "slot": slot_index}
            left_width = max(1.0, left_box[2] - left_box[0])
            right_width = max(1.0, right_box[2] - right_box[0])
            overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
            if overlap / max(1.0, min(left_width, right_width)) > max_overlap_ratio:
                return {"ok": False, "reason": "overlap", "slot": slot_index}

        return {"ok": True, "reason": "ok"}

    for slot_index, det in enumerate(ordered_base):
        if protect_manual_records:
            try:
                if self._is_manual_character_record(det, data=base_data):
                    manual_protected_positions.append(slot_index)
                    continue
            except Exception:
                pass

        unavailable_yolo_indices = set(used_yolo_indices) | set(retired_yolo_indices)
        rescue_index = self._find_best_yolo_box_backend_index(
            slot_index,
            ordered_base,
            ordered_yolo,
            unavailable_yolo_indices,
            min_index=last_yolo_index + 1,
            match_stats=match_stats,
            geometry_first=bool(enable_perfect_refiner),
        )
        if rescue_index is None:
            continue

        candidate = ordered_yolo[rescue_index]
        candidate_bbox = self._char_record_bbox(candidate)
        if not candidate_bbox:
            _retire_yolo_candidate(rescue_index)
            continue
        base_bbox = self._char_record_bbox(det)
        expected_symbol = ""
        try:
            expected_symbol, _slot_x = self._char_record_to_symbol_and_x(det, fallback_index=slot_index)
            expected_symbol = str(expected_symbol or "").strip().upper()
        except Exception:
            expected_symbol = ""
        refine_result = None
        if enable_perfect_refiner and plate_image is not None:
            refine_result = refine_perfect_character_box(
                self,
                plate_image=plate_image,
                base_detections=ordered_base,
                slot_index=slot_index,
                base_rec=det,
                candidate_rec=candidate,
                expected_symbol=expected_symbol,
                yolo_stats=yolo_stats,
                continuity_guard=bool(perfect_refiner_continuity_guard),
            )
        candidate_for_clone = candidate
        if refine_result is not None and bool(refine_result.accepted):
            refined_candidate_bbox = tuple(float(value) for value in refine_result.bbox[:4])
            candidate_for_clone = copy.deepcopy(candidate) if isinstance(candidate, dict) else self._clone_character_detection(candidate)
            if isinstance(candidate_for_clone, dict):
                candidate_for_clone["bbox"] = [float(value) for value in refined_candidate_bbox[:4]]
                candidate_for_clone["box_refined_by_image"] = True
                candidate_for_clone["box_refine_source"] = "perfect_character_refiner"
                candidate_for_clone["box_refine_score"] = float(refine_result.score)
                candidate_for_clone["box_refine_ink_coverage"] = float(refine_result.ink_coverage)
                candidate_for_clone["box_refine_ink_purity"] = float(refine_result.ink_purity)
            else:
                try:
                    candidate_for_clone.bbox = tuple(float(value) for value in refined_candidate_bbox[:4])
                except Exception:
                    candidate_for_clone = candidate
            candidate_bbox = refined_candidate_bbox
            match_stats["perfect_refiner_count"] = int(match_stats.get("perfect_refiner_count", 0) or 0) + 1
            match_stats["perfect_refiner_iterations"] = int(match_stats.get("perfect_refiner_iterations", 0) or 0) + int(refine_result.iterations)
        elif enable_perfect_refiner and plate_image is not None:
            match_stats["perfect_refiner_failed_count"] = int(match_stats.get("perfect_refiner_failed_count", 0) or 0) + 1
            _retire_yolo_candidate(rescue_index)
            continue

        if base_bbox and refine_result is None:
            overlap_quality = _candidate_slot_overlap_quality(base_bbox, candidate_bbox)
            if not bool(overlap_quality.get("ok", False)):
                match_stats["partial_overlap_skipped_count"] = int(match_stats.get("partial_overlap_skipped_count", 0) or 0) + 1
                _retire_yolo_candidate(rescue_index)
                continue

        single_slot_quality = _candidate_single_slot_quality(slot_index, candidate_bbox)
        if not bool(single_slot_quality.get("ok", False)):
            match_stats["single_slot_guard_skipped_count"] = int(match_stats.get("single_slot_guard_skipped_count", 0) or 0) + 1
            _retire_yolo_candidate(rescue_index)
            continue

        candidate_width = max(1.0, float(candidate_bbox[2]) - float(candidate_bbox[0]))
        candidate_height = max(1.0, float(candidate_bbox[3]) - float(candidate_bbox[1]))
        candidate_center_y = (float(candidate_bbox[1]) + float(candidate_bbox[3])) / 2.0

        width_ratio = candidate_width / yolo_median_width
        height_ratio = candidate_height / yolo_median_height
        median_center_y_gap = abs(candidate_center_y - yolo_median_center_y) if yolo_median_center_y > 0.0 else 0.0
        allowed_center_y_gap = max(8.0, yolo_median_height * 0.42)

        min_width_ratio = 0.08 if refine_result is not None else 0.28
        min_height_ratio = 0.35 if refine_result is not None else 0.55
        if width_ratio < min_width_ratio or width_ratio > 2.45:
            _retire_yolo_candidate(rescue_index)
            continue
        if height_ratio < min_height_ratio or height_ratio > 1.95:
            _retire_yolo_candidate(rescue_index)
            continue
        if yolo_median_center_y > 0.0 and median_center_y_gap > allowed_center_y_gap:
            _retire_yolo_candidate(rescue_index)
            continue
        if self._char_record_confidence(candidate) < 0.18:
            _retire_yolo_candidate(rescue_index)
            continue

        replaced[slot_index] = self._clone_base_record_with_candidate_bbox(det, candidate_for_clone)
        if refine_result is not None and isinstance(replaced[slot_index], dict):
            replaced[slot_index]["box_refined_by_image"] = True
            replaced[slot_index]["box_refine_source"] = "perfect_character_refiner"
            replaced[slot_index]["box_refine_score"] = float(refine_result.score)
            replaced[slot_index]["box_refine_ink_coverage"] = float(refine_result.ink_coverage)
            replaced[slot_index]["box_refine_ink_purity"] = float(refine_result.ink_purity)
        used_yolo_indices.add(rescue_index)
        replaced_positions.append(slot_index)
        last_yolo_index = rescue_index

    if not replaced_positions:
        return ordered_base, None

    sequence_quality = _sequence_geometry_quality(replaced)
    if not bool(sequence_quality.get("ok", False)):
        reason = str(sequence_quality.get("reason") or "unknown")
        reason_key = {
            "order": "sequence_guard_order_count",
            "overlap": "sequence_guard_overlap_count",
            "width": "sequence_guard_width_count",
            "lane": "sequence_guard_lane_count",
            "neighbor": "sequence_guard_neighbor_count",
        }.get(reason, "sequence_guard_rejected_count")
        match_stats["sequence_guard_rejected_count"] = int(match_stats.get("sequence_guard_rejected_count", 0) or 0) + 1
        match_stats[reason_key] = int(match_stats.get(reason_key, 0) or 0) + 1
        return ordered_base, None

    replaced = self._sort_character_records_by_x(replaced)
    if self._characters_to_text(replaced) != expected_text:
        return ordered_base, None

    details = {
        "box_backend": "yolo",
        "box_backend_match_mode": "symbol_geometry_slot",
        "box_backend_positions": list(replaced_positions),
        "box_backend_count": int(len(replaced_positions)),
        "box_backend_reference": "yolo_filtered",
        "box_backend_reference_count": int(yolo_stats.get("count", 0) or 0),
    }
    for key, value in match_stats.items():
        if key == "symbol_guard_enabled" or int(value or 0) > 0:
            details[key] = value
    if manual_protected_positions:
        details["manual_protected_positions"] = list(manual_protected_positions)
        details["manual_protected_count"] = int(len(manual_protected_positions))
    return replaced, details


def resolve_canonical_detections(
    host,
    method,
    combined_detections,
    ocr_detections,
    yolo_detections,
    true_texts,
    hybrid_rescue_max_chars: int = 999,
    prefer_yolo_box_positions: bool = False,
    yolo_box_backend_detections=None,
    plate_image=None,
):
    self = host
    ordered_combined = self._sort_character_records_by_x(list(combined_detections or []))
    ordered_ocr = self._sort_character_records_by_x(list(ocr_detections or []))
    ordered_yolo = self._sort_character_records_by_x(list(yolo_detections or []))
    ordered_yolo_box_backend = self._sort_character_records_by_x(
        list(yolo_box_backend_detections or ordered_yolo or [])
    )
    normalized_truths = [
        str(item or "").strip().upper()
        for item in (true_texts or [])
        if str(item or "").strip()
    ]

    def apply_truth_count_guard(source_records, strategy_name: str, details: dict | None = None):
        fitted_records, trim_details = self._fit_detection_count_to_truths(source_records, normalized_truths)
        if not isinstance(trim_details, dict) or not trim_details:
            return source_records, strategy_name, details
        merged_details = dict(details or {})
        merged_details.update(trim_details)
        return fitted_records, strategy_name, merged_details

    if not ordered_combined and not ordered_ocr and not ordered_yolo:
        if normalized_truths:
            return [], "no_detection", {"distance": self._best_text_distance("", normalized_truths)}
        return [], "no_detection_no_gt", {"distance": 0}

    if method == DetectionMethod.OCR:
        return apply_truth_count_guard(ordered_ocr or ordered_combined, "ocr_only", None)

    if method == DetectionMethod.YOLO:
        return apply_truth_count_guard(ordered_yolo or ordered_combined, "yolo_only", None)

    if method == DetectionMethod.YOLO_OCR:
        combined_text = self._characters_to_text(ordered_combined)
        details = {
            "box_source": "yolo_nms",
            "box_count": int(len(ordered_combined)),
            "ocr_filled_count": int(sum(1 for rec in ordered_combined if self._preview_record_has_symbol(rec))),
        }

        if normalized_truths:
            if combined_text and combined_text in normalized_truths:
                details.update({"final_text": combined_text})
                return apply_truth_count_guard(ordered_combined, "yolo_box_ocr", details)
            details.update({"distance": self._best_text_distance(combined_text, normalized_truths)})
            return apply_truth_count_guard(ordered_combined, "yolo_box_ocr_fallback", details)

        return ordered_combined, "yolo_box_ocr_no_gt", details

    if normalized_truths:
        ocr_text = self._characters_to_text(ordered_ocr)
        yolo_text = self._characters_to_text(ordered_yolo)

        if ocr_text and ocr_text in normalized_truths:
            details = {"final_text": ocr_text}
            if prefer_yolo_box_positions:
                rebuilt, backend_details = self._apply_yolo_box_backend(
                    ordered_ocr,
                    ordered_yolo_box_backend,
                    plate_image=plate_image,
                )
                if backend_details:
                    details.update(backend_details)
                    return apply_truth_count_guard(rebuilt, "ocr_exact", details)
            return apply_truth_count_guard(ordered_ocr, "ocr_exact", details)

        if yolo_text and yolo_text in normalized_truths:
            return apply_truth_count_guard(ordered_yolo, "yolo_exact", {"final_text": yolo_text})

        if int(hybrid_rescue_max_chars or 0) > 0:
            rescued, rescue_details = self._repair_ocr_with_yolo_boxes(
                ordered_ocr,
                ordered_yolo,
                normalized_truths,
                max_mismatch_count=hybrid_rescue_max_chars,
            )
            if rescued:
                if prefer_yolo_box_positions:
                    rebuilt, backend_details = self._apply_yolo_box_backend(
                        rescued,
                        ordered_yolo_box_backend,
                        plate_image=plate_image,
                    )
                    if backend_details:
                        if not isinstance(rescue_details, dict):
                            rescue_details = {}
                        rescue_details.update(backend_details)
                        return apply_truth_count_guard(rebuilt, "ocr_yolo_rescue", rescue_details)
                return apply_truth_count_guard(rescued, "ocr_yolo_rescue", rescue_details)

        best_source = ordered_ocr or ordered_yolo or ordered_combined
        best_strategy = "ocr_fallback" if ordered_ocr else ("yolo_fallback" if ordered_yolo else "both_combined")
        best_distance = self._best_text_distance(self._characters_to_text(best_source), normalized_truths)

        for candidate_source, candidate_strategy in (
            (ordered_yolo, "yolo_fallback"),
            (ordered_combined, "both_combined"),
        ):
            if not candidate_source:
                continue

            candidate_distance = self._best_text_distance(self._characters_to_text(candidate_source), normalized_truths)
            if candidate_distance < best_distance:
                best_source = candidate_source
                best_strategy = candidate_strategy
                best_distance = candidate_distance

        fallback_details = {"distance": best_distance}
        if prefer_yolo_box_positions and best_strategy == "ocr_fallback":
            rebuilt, backend_details = self._apply_yolo_box_backend(
                best_source,
                ordered_yolo_box_backend,
                plate_image=plate_image,
            )
            if backend_details:
                fallback_details.update(backend_details)
                best_source = rebuilt

        return apply_truth_count_guard(best_source, best_strategy, fallback_details)

    if prefer_yolo_box_positions and ordered_ocr:
        rebuilt, backend_details = self._apply_yolo_box_backend(
            ordered_ocr,
            ordered_yolo_box_backend,
            plate_image=plate_image,
        )
        if backend_details:
            return rebuilt, "both_combined_no_gt", backend_details

    return ordered_combined, "both_combined_no_gt", None
