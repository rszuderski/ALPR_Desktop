"""Image-backed recovery of working character geometry; never model RAW."""

from __future__ import annotations

import copy
from pathlib import Path

from ..plate_ground_truth import normalize_plate_ground_truth_text

GT_GEOMETRY_PREPARATION_VERSION = "z2_gt_image_recovery.v1"


def _iou(a, b):
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return overlap / max(1.0, area_a + area_b - overlap)


def _load_crop(host, plate_id):
    try:
        import cv2
        import numpy as np
        directory = Path(host.preview_dir_var.get()) / "images"
        for suffix in (".jpg", ".png", ".jpeg", ".bmp", ".webp"):
            path = directory / f"{plate_id}{suffix}"
            if path.is_file():
                return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (AttributeError, OSError, ValueError, ImportError):
        pass
    return None


def recover_z2_gt_geometry(host, data, records, *, plate_id="", plate_image=None, settings=None):
    """Accept only a stable, complete, single-row segmentation anchored in YOLO.

    The known Z2 number supplies the expected count. Pixels supply geometry;
    no evenly spaced or guessed missing boxes are manufactured.
    """
    attrs = data.get("plate_attributes") or {}
    attrs = attrs if isinstance(attrs, dict) else {}
    source = data.get("ground_truth_source") or attrs.get("ground_truth_source")
    gt = normalize_plate_ground_truth_text(data.get("ground_truth_text") or attrs.get("ground_truth_text"))
    layout = data.get("plate_layout_override") or data.get("plate_layout")
    if source != "manual_z2" or not gt or layout not in ("single_row", "1R"):
        return records, None
    if any(str(r.get("box_source") or "") == "manual_box"
           or str(r.get("method") or "").startswith(("manual", "cvat")) for r in records):
        return records, None
    seeds = data.get("yolo_nms_detections") or []
    if len(seeds) < 3:
        return records, None
    if plate_image is None:
        plate_image = _load_crop(host, plate_id)
    if plate_image is None:
        return records, None
    try:
        import cv2
        import numpy as np
        height, width = plate_image.shape[:2]
        if min(height, width) < 8:
            return records, None
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if plate_image.ndim == 3 else plate_image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_text = float(np.median(gray)) > threshold
        blue_end = 0
        if plate_image.ndim == 3:
            hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, np.array([90, 55, 40]), np.array([140, 255, 255]))
            for x, coverage in enumerate(blue.mean(axis=0) / 255):
                if x >= width * 0.15 or coverage < 0.35:
                    break
                blue_end = x + 1
            if blue_end < max(3, width * 0.02):
                blue_end = 0
        runtime = dict(settings or {})
        if not runtime:
            try:
                runtime = host._get_yolo_runtime_settings()
            except (AttributeError, ValueError, TypeError):
                runtime = {}
        segmentations = []
        for level in (threshold - 10, threshold, threshold + 10):
            foreground = gray < level if dark_text else gray > level
            mask = foreground.astype(np.uint8) * 255
            mask[:, :blue_end] = 0
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            boxes = sorted([float(x), float(y), float(x + w), float(y + h)]
                           for x, y, w, h, area in stats[1:]
                           if 2 <= w <= width * 0.20 and h >= height * 0.40 and area >= height * 0.40)
            if len(boxes) != len(gt):
                continue
            widths = np.array([b[2] - b[0] for b in boxes])
            heights = np.array([b[3] - b[1] for b in boxes])
            centers = np.array([(b[1] + b[3]) / 2 for b in boxes])
            median_width = float(np.median(widths))
            median_height = float(np.median(heights))
            if (np.any(widths > median_width * float(runtime.get("seq_max_w", 2.60)))
                    or np.any(heights < median_height * float(runtime.get("seq_min_h", 0.55)))
                    or np.any(heights > median_height * float(runtime.get("seq_max_h", 1.80)))
                    or np.any(np.abs(centers - np.median(centers)) > median_height * float(runtime.get("seq_center_y", 0.60)))):
                continue
            # Every recovered component must overlap an actual YOLO candidate.
            if any(max((_iou(b, r.get("bbox") or [0, 0, 0, 0]) for r in seeds), default=0) < 0.25 for b in boxes):
                continue
            segmentations.append(boxes)
        if len(segmentations) < 2:
            return records, None
        boxes = segmentations[len(segmentations) // 2]
        if any(_iou(a, b) < 0.70 for other in segmentations for a, b in zip(boxes, other)):
            return records, None
    except (ImportError, ValueError, TypeError, IndexError):
        return records, None

    recovered = []
    reused = set()
    generated = 0
    reference_width = float(np.median([b[2] - b[0] for b in boxes]))
    reference_height = float(np.median([b[3] - b[1] for b in boxes]))
    for box in boxes:
        matches = [(i, _iou(box, r.get("bbox") or [0, 0, 0, 0])) for i, r in enumerate(records) if i not in reused]
        index, match = max(matches, key=lambda value: value[1], default=(-1, 0))
        original_height = (records[index]["bbox"][3] - records[index]["bbox"][1]) if index >= 0 else 0
        original_width = (records[index]["bbox"][2] - records[index]["bbox"][0]) if index >= 0 else 0
        if (match >= 0.65 and original_height >= (box[3] - box[1]) * 0.85
                and original_width <= reference_width * float(runtime.get("seq_max_w", 2.60))
                and reference_height * float(runtime.get("seq_min_h", 0.55)) <= original_height
                <= reference_height * float(runtime.get("seq_max_h", 1.80))):
            rec = copy.deepcopy(records[index])
            reused.add(index)
        else:
            generated += 1
            rec = {"character": "", "bbox": box, "confidence": 0.0,
                   "method": "segment", "source_tag": "generated_box", "source_kind": "generated_box",
                   "box_source": "generated_box", "sign_source": "", "geometry_source": "segment",
                   "geometry_method": "segment", "box_backend": "segment",
                   "box_refine_source": "z2_gt_image_recovery", "box_refined_by_image": True}
        recovered.append(rec)
    if len(recovered) == len(records) and not generated:
        return records, None
    return recovered, {"schema": "alpr.pz2.gt_geometry_recovery.v1", "source": "image_components",
                       "ground_truth_source": "manual_z2", "expected_count": len(gt),
                       "original_count": len(records), "recovered_count": len(recovered),
                       "generated_box_count": generated, "blue_strip_excluded": bool(blue_end),
                       "threshold_checks": len(segmentations), "requires_review": True}
