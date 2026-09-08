from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharacterBoxRefineResult:
    bbox: tuple[float, float, float, float]
    score: float
    ink_coverage: float
    ink_purity: float
    iterations: int
    accepted: bool = True


def _safe_bbox(host, rec):
    try:
        bbox = host._char_record_bbox(rec)
    except Exception:
        bbox = None
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _rect_area(rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _clamp_rect(rect, *, width: float, height: float):
    x1, y1, x2, y2 = [float(value) for value in rect]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 + 1.0:
        return None
    if y2 <= y1 + 1.0:
        return None
    return (x1, y1, x2, y2)


def _intersect_rect(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    x1 = max(float(lx1), float(rx1))
    y1 = max(float(ly1), float(ry1))
    x2 = min(float(lx2), float(rx2))
    y2 = min(float(ly2), float(ry2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _rect_to_mask_slice(rect, *, sx1: int, sy1: int, roi_w: int, roi_h: int):
    x1, y1, x2, y2 = rect
    rx1 = max(0, min(int(roi_w), int(round(float(x1) - float(sx1)))))
    rx2 = max(0, min(int(roi_w), int(round(float(x2) - float(sx1)))))
    ry1 = max(0, min(int(roi_h), int(round(float(y1) - float(sy1)))))
    ry2 = max(0, min(int(roi_h), int(round(float(y2) - float(sy1)))))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return (rx1, ry1, rx2, ry2)


def _symbol_width_ratios(symbol: str) -> tuple[float, float]:
    """Return conservative width limits for a recognized character slot."""
    normalized = str(symbol or "").strip().upper()
    if normalized in {"I", "1", "L"}:
        return 0.36, 1.20
    if normalized in {"M", "W"}:
        return 0.58, 1.62
    return 0.64, 1.34


def _build_candidate_variants(seed_bbox, *, lane_rect, img_w: float, img_h: float, step_x: float, step_y: float):
    x1, y1, x2, y2 = seed_bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    variants = [seed_bbox]
    for scale_x in (0.82, 0.92, 1.0, 1.10, 1.22):
        for scale_y in (0.98, 1.06, 1.16):
            w = width * scale_x
            h = height * scale_y
            for dx in (-step_x, 0.0, step_x):
                for dy in (-step_y, 0.0, step_y):
                    candidate = (cx + dx - w / 2.0, cy + dy - h / 2.0, cx + dx + w / 2.0, cy + dy + h / 2.0)
                    lane_candidate = _intersect_rect(candidate, lane_rect)
                    if lane_candidate is None:
                        continue
                    clamped = _clamp_rect(lane_candidate, width=img_w, height=img_h)
                    if clamped is not None:
                        variants.append(clamped)
    return variants


def _active_column_groups(active_columns, *, max_gap: int):
    groups = []
    active_indices = [index for index, active in enumerate(active_columns) if bool(active)]
    if not active_indices:
        return groups
    start = active_indices[0]
    previous = start
    allowed_gap = max(0, int(max_gap))
    for index in active_indices[1:]:
        if int(index) - int(previous) > allowed_gap + 1:
            groups.append((int(start), int(previous) + 1))
            start = int(index)
        previous = int(index)
    groups.append((int(start), int(previous) + 1))
    return groups


def refine_perfect_character_box(
    host,
    *,
    plate_image,
    base_detections,
    slot_index: int,
    base_rec,
    candidate_rec,
    expected_symbol: str,
    yolo_stats: dict | None = None,
    continuity_guard: bool = True,
) -> CharacterBoxRefineResult | None:
    """Refine a YOLO geometry hypothesis for a known perfect-plate slot.

    This is intentionally conservative: if we cannot find a stable foreground
    component for the expected slot, we return None and the caller can keep the
    lightweight matcher result.
    """

    expected_symbol = str(expected_symbol or "").strip().upper()
    if not expected_symbol:
        return None
    # Przy tablicy perfect znak slotu znamy z zatwierdzonego odczytu.
    # Klasa YOLO bywa mylona na sąsiednich znakach, więc kandydat jest tu
    # hipotezą geometrii, a nie źródłem prawdy o symbolu.

    if plate_image is None or getattr(plate_image, "size", 0) == 0:
        return None

    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    try:
        img_h, img_w = plate_image.shape[:2]
    except Exception:
        return None
    if img_w <= 1 or img_h <= 1:
        return None

    base_bbox = _safe_bbox(host, base_rec)
    candidate_bbox = _safe_bbox(host, candidate_rec)
    if candidate_bbox is None:
        return None
    if base_bbox is None:
        base_bbox = candidate_bbox

    base_heights = []
    base_centers_y = []
    for rec in list(base_detections or []):
        rec_bbox = _safe_bbox(host, rec)
        if rec_bbox is None:
            continue
        rec_h = max(1.0, float(rec_bbox[3]) - float(rec_bbox[1]))
        base_heights.append(rec_h)
        base_centers_y.append((float(rec_bbox[1]) + float(rec_bbox[3])) / 2.0)
    base_median_height = float(sorted(base_heights)[len(base_heights) // 2]) if base_heights else max(1.0, base_bbox[3] - base_bbox[1])
    base_center_y = (float(base_bbox[1]) + float(base_bbox[3])) / 2.0

    centers_x = []
    base_symbols = []
    for base_index, rec in enumerate(list(base_detections or [])):
        try:
            symbol, center_x = host._char_record_to_symbol_and_x(rec, fallback_index=base_index)
            centers_x.append(float(center_x))
            base_symbols.append(str(symbol or "").strip().upper())
        except Exception:
            centers_x.append(float(base_index))
            base_symbols.append("")
    if not centers_x:
        return None

    stats = dict(yolo_stats or {})
    median_width = max(2.0, float(stats.get("median_width", 0.0) or host._char_record_width(candidate_rec) or 2.0))
    median_height = max(2.0, float(stats.get("median_height", 0.0) or host._char_record_height(candidate_rec) or 2.0))
    reference_height = max(
        2.0,
        min(float(img_h), base_median_height * 0.92),
        min(float(img_h), (float(base_bbox[3]) - float(base_bbox[1])) * 0.92),
        min(float(img_h), median_height * 0.82),
    )

    def _normalize_vertical_extent(rect, *, min_height: float = reference_height):
        x1, y1, x2, y2 = [float(value) for value in rect]
        current_height = max(1.0, y2 - y1)
        if current_height >= min_height:
            return _clamp_rect((x1, y1, x2, y2), width=float(img_w), height=float(img_h))
        center_y = base_center_y
        half_h = float(min_height) / 2.0
        normalized = (x1, center_y - half_h, x2, center_y + half_h)
        return _clamp_rect(normalized, width=float(img_w), height=float(img_h))

    lane_left = 0.0
    lane_right = float(img_w)
    try:
        if slot_index > 0:
            lane_left = (float(centers_x[slot_index - 1]) + float(centers_x[slot_index])) / 2.0
        if slot_index + 1 < len(centers_x):
            lane_right = (float(centers_x[slot_index]) + float(centers_x[slot_index + 1])) / 2.0
    except Exception:
        lane_left = 0.0
        lane_right = float(img_w)
    if lane_right <= lane_left + 2.0:
        lane_left = 0.0
        lane_right = float(img_w)

    min_width_ratio, max_width_ratio = _symbol_width_ratios(expected_symbol)
    local_lane_width = max(2.0, float(lane_right) - float(lane_left))
    width_reference = max(2.0, min(float(median_width), local_lane_width * 0.92))
    min_acceptable_width = max(2.0, width_reference * min_width_ratio)
    max_acceptable_width = max(min_acceptable_width + 1.0, width_reference * max_width_ratio)

    strict_lane = (
        max(0.0, lane_left),
        0.0,
        min(float(img_w), lane_right),
        float(img_h),
    )
    same_symbol_neighbor = False
    try:
        same_symbol_neighbor = (
            (slot_index > 0 and base_symbols[slot_index - 1] == expected_symbol)
            or (slot_index + 1 < len(base_symbols) and base_symbols[slot_index + 1] == expected_symbol)
        )
    except Exception:
        same_symbol_neighbor = False

    output_lane_pad = 0.0 if same_symbol_neighbor else max(0.4, min(2.0, median_width * 0.06))
    max_lane_bleed_ratio = 0.04 if same_symbol_neighbor else 0.16
    search_lane_pad = max(2.0, median_width * 0.42)
    output_lane = (
        max(0.0, lane_left - output_lane_pad),
        0.0,
        min(float(img_w), lane_right + output_lane_pad),
        float(img_h),
    )
    search_lane = (
        max(0.0, lane_left - search_lane_pad),
        0.0,
        min(float(img_w), lane_right + search_lane_pad),
        float(img_h),
    )

    cx1, cy1, cx2, cy2 = candidate_bbox
    bx1, by1, bx2, by2 = base_bbox
    pad_x = max(3.0, median_width * 0.55)
    pad_y = max(2.0, median_height * 0.28)
    search_rect = (
        min(cx1, bx1) - pad_x,
        min(cy1, by1) - pad_y,
        max(cx2, bx2) + pad_x,
        max(cy2, by2) + pad_y,
    )
    search_rect = _intersect_rect(search_rect, search_lane)
    if search_rect is None:
        return None
    search_rect = _clamp_rect(search_rect, width=float(img_w), height=float(img_h))
    if search_rect is None:
        return None

    sx1, sy1, sx2, sy2 = [int(round(value)) for value in search_rect]
    sx1 = max(0, min(int(img_w), sx1))
    sx2 = max(0, min(int(img_w), sx2))
    sy1 = max(0, min(int(img_h), sy1))
    sy2 = max(0, min(int(img_h), sy2))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = plate_image[sy1:sy2, sx1:sx2]
    if roi is None or getattr(roi, "size", 0) == 0:
        return None

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold, _binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border_parts = [blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]]
        border = np.concatenate([part.reshape(-1) for part in border_parts if getattr(part, "size", 0) > 0])
        border_median = float(np.median(border)) if getattr(border, "size", 0) > 0 else float(np.median(blurred))
        mask = blurred < float(threshold) if border_median >= float(threshold) else blurred > float(threshold)
        mask = cv2.medianBlur((mask.astype("uint8") * 255), 3)
        num_labels, labels, stats_arr, centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return None

    roi_h, roi_w = mask.shape[:2]
    if roi_w <= 1 or roi_h <= 1:
        return None

    candidate_roi = _rect_to_mask_slice(candidate_bbox, sx1=sx1, sy1=sy1, roi_w=roi_w, roi_h=roi_h)
    slot_center_x = float(centers_x[slot_index]) if 0 <= slot_index < len(centers_x) else ((cx1 + cx2) / 2.0)
    slot_center_roi_x = slot_center_x - float(sx1)
    roi_area = max(1.0, float(roi_w * roi_h))
    min_component_area = max(3.0, roi_area * 0.0010)
    max_component_area = roi_area * 0.68

    selected_label = None
    selected_score = None
    selected_rect_roi = None
    for label_idx in range(1, int(num_labels)):
        area = float(stats_arr[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_area or area > max_component_area:
            continue
        x = float(stats_arr[label_idx, cv2.CC_STAT_LEFT])
        y = float(stats_arr[label_idx, cv2.CC_STAT_TOP])
        w = float(stats_arr[label_idx, cv2.CC_STAT_WIDTH])
        h = float(stats_arr[label_idx, cv2.CC_STAT_HEIGHT])
        if w <= 0.0 or h <= 0.0:
            continue
        comp_rect_roi = (x, y, x + w, y + h)
        comp_center_x = float(centroids[label_idx][0])
        center_distance = abs(comp_center_x - slot_center_roi_x) / max(1.0, median_width)
        touch_bonus = 0.0
        if candidate_roi:
            overlap = _intersect_rect(comp_rect_roi, candidate_roi)
            if overlap is not None:
                touch_bonus = -0.75
        size_bonus = -min(0.35, area / max(1.0, median_width * median_height))
        score = center_distance + touch_bonus + size_bonus
        if selected_score is None or score < selected_score:
            selected_label = int(label_idx)
            selected_score = float(score)
            selected_rect_roi = comp_rect_roi

    if selected_label is None or selected_rect_roi is None:
        return None

    target_mask = labels == int(selected_label)
    target_total = float(target_mask.sum())
    if target_total <= 0.0:
        return None

    component_w = max(1.0, float(selected_rect_roi[2]) - float(selected_rect_roi[0]))
    component_h = max(1.0, float(selected_rect_roi[3]) - float(selected_rect_roi[1]))
    if continuity_guard:
        component_min_width = max(2.0, min_acceptable_width * 0.58)
        component_min_height = max(2.0, reference_height * 0.30)
        component_max_width = max_acceptable_width * 1.18
        if component_w < component_min_width:
            return None
        if component_h < component_min_height:
            return None
        if component_w > component_max_width:
            return None

    foreground_mask = mask > 0

    def _strict_lane_bleed_ratio(rect) -> float:
        try:
            x1, _y1, x2, _y2 = [float(value) for value in rect]
        except Exception:
            return 1.0
        width = max(1.0, x2 - x1)
        bleed = max(0.0, float(strict_lane[0]) - x1) + max(0.0, x2 - float(strict_lane[2]))
        return float(bleed / width)

    def _column_support_metrics(mask_slice, rect):
        rx1, ry1, rx2, ry2 = mask_slice
        width = max(1, int(rx2) - int(rx1))
        height = max(1, int(ry2) - int(ry1))
        if width <= 0 or height <= 0:
            return {
                "active_columns": 0.0,
                "active_span_width": 0.0,
                "foreign_column_ratio": 0.0,
                "span_over_width": 0.0,
                "group_count": 0,
            }

        column_counts = foreground_mask[ry1:ry2, rx1:rx2].sum(axis=0)
        min_hits = max(3, int(round(height * 0.055)))
        active_columns = column_counts >= int(min_hits)
        groups = _active_column_groups(active_columns, max_gap=max(1, int(round(median_width * 0.075))))
        if not groups:
            return {
                "active_columns": 0.0,
                "active_span_width": 0.0,
                "foreign_column_ratio": 0.0,
                "span_over_width": 0.0,
                "group_count": 0,
            }

        first_active = int(groups[0][0])
        last_active = int(groups[-1][1])
        active_span_width = max(1.0, float(last_active - first_active))
        active_total = float(column_counts[active_columns].sum())

        rect_x1 = float(rect[0])
        target_half_width = max(min_acceptable_width * 0.58, max_acceptable_width * 0.52)
        target_left = max(float(rect[0]), float(slot_center_x) - target_half_width)
        target_right = min(float(rect[2]), float(slot_center_x) + target_half_width)
        foreign_total = 0.0
        for local_x, count in enumerate(column_counts):
            if not bool(active_columns[local_x]):
                continue
            image_x = rect_x1 + float(local_x) + 0.5
            if image_x < target_left or image_x > target_right:
                foreign_total += float(count)

        return {
            "active_columns": float(active_columns.sum()),
            "active_span_width": float(active_span_width),
            "foreign_column_ratio": float(foreign_total / max(1.0, active_total - foreign_total)),
            "span_over_width": float(active_span_width / max(1.0, max_acceptable_width)),
            "group_count": int(len(groups)),
        }

    def score_bbox(rect):
        rect = _intersect_rect(rect, output_lane)
        if rect is None:
            return None
        rect = _normalize_vertical_extent(rect)
        if rect is None:
            return None
        rect = _clamp_rect(rect, width=float(img_w), height=float(img_h))
        if rect is None:
            return None
        mask_slice = _rect_to_mask_slice(rect, sx1=sx1, sy1=sy1, roi_w=roi_w, roi_h=roi_h)
        if mask_slice is None:
            return None
        rx1, ry1, rx2, ry2 = mask_slice
        column_support = _column_support_metrics(mask_slice, rect)
        target_inside = float(target_mask[ry1:ry2, rx1:rx2].sum())
        foreground_inside = float(foreground_mask[ry1:ry2, rx1:rx2].sum())
        coverage = target_inside / max(1.0, target_total)
        purity = target_inside / max(1.0, foreground_inside)
        foreign_inside = max(0.0, foreground_inside - target_inside)
        dominant_foreign_inside = 0.0
        if continuity_guard and foreign_inside > 0.0:
            try:
                rect_labels = labels[ry1:ry2, rx1:rx2]
                label_values, label_counts = np.unique(rect_labels[rect_labels > 0], return_counts=True)
                for label_value, label_count in zip(label_values, label_counts):
                    if int(label_value) == int(selected_label):
                        continue
                    dominant_foreign_inside = max(dominant_foreign_inside, float(label_count))
            except Exception:
                dominant_foreign_inside = foreign_inside
        foreign_ratio = foreign_inside / max(1.0, target_inside)
        dominant_foreign_ratio = dominant_foreign_inside / max(1.0, target_inside)
        rect_area = max(1.0, _rect_area(rect))
        target_density = target_inside / rect_area
        width = max(1.0, rect[2] - rect[0])
        height = max(1.0, rect[3] - rect[1])
        center_gap = abs(((rect[0] + rect[2]) / 2.0) - slot_center_x) / max(1.0, median_width)
        narrow_penalty = max(0.0, (min_acceptable_width - width) / max(1.0, min_acceptable_width))
        wide_penalty = max(0.0, (width - max_acceptable_width) / max(1.0, max_acceptable_width))
        lane_bleed_penalty = _strict_lane_bleed_ratio(rect)
        short_penalty = max(0.0, (reference_height * 0.88 - height) / max(1.0, reference_height))
        tall_penalty = max(0.0, (height / max(1.0, reference_height)) - 1.28)
        continuity_penalty = 0.0
        if continuity_guard:
            continuity_penalty = (
                0.95 * min(1.4, foreign_ratio)
                + 0.70 * min(1.2, dominant_foreign_ratio)
                + 0.68 * min(1.4, float(column_support.get("foreign_column_ratio", 0.0)))
                + 0.55 * max(0.0, float(column_support.get("span_over_width", 0.0)) - 1.04)
            )
        score = (
            (2.45 * coverage)
            + (1.15 * purity)
            + (0.45 * min(1.0, target_density * 8.0))
            - (0.28 * center_gap)
            - (0.85 * narrow_penalty)
            - (0.70 * wide_penalty)
            - (0.72 * lane_bleed_penalty)
            - (0.60 * short_penalty)
            - (0.20 * tall_penalty)
            - continuity_penalty
        )
        return {
            "bbox": rect,
            "score": float(score),
            "coverage": float(coverage),
            "purity": float(purity),
            "width": float(width),
            "lane_bleed_ratio": float(lane_bleed_penalty),
            "foreign_ratio": float(foreign_ratio),
            "dominant_foreign_ratio": float(dominant_foreign_ratio),
            "column_foreign_ratio": float(column_support.get("foreign_column_ratio", 0.0)),
            "column_span_over_width": float(column_support.get("span_over_width", 0.0)),
            "column_active_span_width": float(column_support.get("active_span_width", 0.0)),
            "column_group_count": int(column_support.get("group_count", 0) or 0),
        }

    x1, y1, x2, y2 = selected_rect_roi
    ink_bbox = (
        float(sx1) + x1 - 1.0,
        float(sy1) + y1 - 1.0,
        float(sx1) + x2 + 1.0,
        float(sy1) + y2 + 1.0,
    )
    seed = _intersect_rect(ink_bbox, output_lane)
    if seed is None:
        return None
    seed = _normalize_vertical_extent(seed)
    if seed is None:
        return None
    seed = _clamp_rect(seed, width=float(img_w), height=float(img_h))
    if seed is None:
        return None

    best = score_bbox(seed)
    original_score = score_bbox(candidate_bbox)
    if original_score and (best is None or original_score["score"] > best["score"]):
        best = original_score
    if best is None:
        return None

    iterations = 0
    step_x = max(1.0, median_width * 0.18)
    step_y = max(1.0, median_height * 0.08)
    for _iteration in range(5):
        iterations += 1
        improved = False
        variants = _build_candidate_variants(
            best["bbox"],
            lane_rect=output_lane,
            img_w=float(img_w),
            img_h=float(img_h),
            step_x=step_x,
            step_y=step_y,
        )
        for variant in variants:
            candidate_score = score_bbox(variant)
            if not candidate_score:
                continue
            if candidate_score["score"] > best["score"] + 0.012:
                best = candidate_score
                improved = True
        if not improved:
            break
        step_x *= 0.55
        step_y *= 0.55

    min_coverage = 0.78 if (continuity_guard and same_symbol_neighbor) else (0.72 if continuity_guard else 0.38)
    min_purity = 0.64 if continuity_guard else 0.38
    max_foreign_ratio = 0.26 if same_symbol_neighbor else 0.42
    max_dominant_foreign_ratio = 0.16 if same_symbol_neighbor else 0.28
    max_column_foreign_ratio = 0.20 if same_symbol_neighbor else 0.30
    max_column_span_over_width = 1.10 if same_symbol_neighbor else 1.18
    if best["coverage"] < min_coverage:
        return None
    if best["purity"] < min_purity:
        return None
    if best["width"] < min_acceptable_width:
        return None
    if best["width"] > max_acceptable_width * 1.10:
        return None
    if best.get("lane_bleed_ratio", 0.0) > max_lane_bleed_ratio:
        return None
    if continuity_guard:
        if best.get("foreign_ratio", 0.0) > max_foreign_ratio:
            return None
        if best.get("dominant_foreign_ratio", 0.0) > max_dominant_foreign_ratio:
            return None
        if best.get("column_foreign_ratio", 0.0) > max_column_foreign_ratio:
            return None
        if best.get("column_span_over_width", 0.0) > max_column_span_over_width:
            return None

    return CharacterBoxRefineResult(
        bbox=tuple(float(value) for value in best["bbox"]),
        score=float(best["score"]),
        ink_coverage=float(best["coverage"]),
        ink_purity=float(best["purity"]),
        iterations=int(iterations),
    )
