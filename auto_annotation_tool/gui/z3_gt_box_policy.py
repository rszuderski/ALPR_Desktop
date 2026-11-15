"""GT limits for the working result; detector evidence remains immutable."""
import math

from ..plate_ground_truth import normalize_plate_ground_truth_text


def plate_gt(data):
    data = data if isinstance(data, dict) else {}
    attrs = data.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    return normalize_plate_ground_truth_text(data.get("ground_truth_text") or attrs.get("ground_truth_text"))


def limit_boxes_to_gt(host, data, records):
    raw = (data or {}).get("raw_detection")
    frozen = isinstance(raw, dict) and raw.get("contract") == "gt_blind.v1" and raw.get("characters") is records
    if not frozen:
        return _select_limited_boxes(host, data, records)
    metadata = getattr(host, "preview_metadata", None)
    cached = getattr(host, "_preview_gt_box_cache", None)
    if cached is None or cached[0] != id(metadata):
        cached = host._preview_gt_box_cache = (id(metadata), {})
    key = (id(records), raw.get("result_hash"), plate_gt(data),
           data.get("plate_layout"), data.get("plate_layout_override"),
           repr(data.get("plate_layout_separator")), data.get("plate_image_width"), data.get("plate_image_height"))
    entry = cached[1].get(id(data))
    if entry is not None and entry[0] == key and entry[1] is records:
        return entry[2]
    result = _select_limited_boxes(host, data, records)
    cached[1][id(data)] = (key, records, result)
    return result


def working_characters(host, data):
    """Use the same stage for total counters as for the ordinary preview."""
    if not isinstance(data, dict):
        return []
    if data.get("characters") or (data.get("review_state") or {}).get("status"):
        return data.get("characters") or []
    raw = data.get("raw_detection")
    if isinstance(raw, dict):
        return raw.get("characters") or []
    return data.get("yolo_detections") or data.get("yolo_nms_detections") or data.get("yolo_raw_detections") or []


def _select_limited_boxes(host, data, records):
    """Choose an ordered subset, preferring manual boxes, GT matches and confidence."""
    ordered = host._sort_character_records_by_x(list(records or []), data=data)
    gt = plate_gt(data)
    limit = len(gt)
    if not limit or len(ordered) <= limit:
        return ordered, None

    # Unlike a contiguous crop, a subsequence can also discard an extra
    # prediction inside the registration (including a repeated character).
    best = [None] * (limit + 1)
    best[0] = ((0, 0, 0, 0.0), ())
    for index, rec in enumerate(ordered):
        if not isinstance(rec, dict):
            continue
        try:
            box = list(map(float, rec.get("bbox", [])[:4]))
            valid = len(box) == 4 and all(map(math.isfinite, box)) and box[2] > box[0] and box[3] > box[1]
        except (TypeError, ValueError):
            valid = False
        manual = str(rec.get("box_source") or "") == "manual_box" or str(rec.get("method") or "") in {"manual", "cvat_manual"}
        try:
            confidence = float(rec.get("confidence") or 0)
            confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        symbol = str(rec.get("character") or "").strip().upper()
        for count in range(min(index + 1, limit), 0, -1):
            previous = best[count - 1]
            if previous is None:
                continue
            score, indices = previous
            contribution = (int(valid), int(manual), int(symbol == gt[count - 1]), confidence)
            candidate = (tuple(a + b for a, b in zip(score, contribution)), indices + (index,))
            if best[count] is None or candidate[0] > best[count][0]:
                best[count] = candidate
    chosen = best[limit] or next(item for item in reversed(best) if item is not None)
    indices = chosen[1]
    selected = [ordered[index] for index in indices]
    selected_ids = set(indices)
    return selected, {
        "ground_truth_text": gt, "limit": limit, "original_count": len(ordered),
        "kept_count": len(selected), "removed_count": len(ordered) - len(selected),
        "discarded": [record for index, record in enumerate(ordered) if index not in selected_ids],
    }


def can_add_character_box(host, data, *, notify=True):
    """Manual PZ2 editing is never count-blocked by GT.

    GT count guards belong to automatic detection/post-processing. The operator
    must always be able to add the missing box, even when inherited Z2 GT later
    turns out to be wrong. Any mismatch is resolved at explicit R/OK review.
    """
    return True
