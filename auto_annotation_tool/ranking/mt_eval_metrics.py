"""Pure operational Box and raw indexed-corner metrics; no inference or I/O.

The caller supplies an approved geometry preflight and one model's persisted
prediction rows. Matching is confidence-ordered greedy, not maximum-cardinality
assignment. An excluded Pose GT still participates in the Box association.
Partial results use processed-scene denominators; pending scenes are not FN.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from math import floor, hypot, isfinite
from numbers import Real

from ..rectification.quad_geometry import analyze_quad


METRICS_SCHEMA = "alpr.mt_operational_metrics.v1"
METRIC_PROTOCOL = {
    "operational_conf": .25,
    "confidence_comparison": ">",
    "match_iou": .50,
    "matching": "confidence_desc_iou_desc_plate_id_v1",
    "percentile_method": "linear",
    "pose_normalization": "gt_bbox_diagonal",
    "plate_class_ids": [0],
    "keypoint_bounds_tolerance_px": 1.0,
    "keypoint_confidence_threshold": None,
    "corner_reordering": "forbidden",
    "reversed_raw_winding": "measure_original_indices_and_report_separately",
    "nonfinite_encoding": "__nonfinite_float__",
}
_FROZEN_PROTOCOL = deepcopy(METRIC_PROTOCOL)
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(v) for v in value):
        return None
    box = list(map(float, value))
    if box[2] <= box[0] or box[3] <= box[1] or not isfinite((box[2]-box[0])*(box[3]-box[1])):
        return None
    return box


def box_iou(first, second):
    """Continuous XYXY IoU; no pixel '+1', clipping, reordering or bbox repair."""
    a, b = _bbox(first), _bbox(second)
    if a is None or b is None:
        return 0.0
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 and isfinite(union) else 0.0


def distribution(values):
    """Linear order-statistic quantiles, with null values for an empty sample."""
    ordered = sorted(float(v) for v in values)
    _require(all(isfinite(v) for v in ordered), "NONFINITE_METRIC_SAMPLE")
    def quantile(probability):
        if not ordered:
            return None
        position = (len(ordered)-1)*probability
        lower = floor(position)
        upper = min(lower+1, len(ordered)-1)
        return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)
    return {"n": len(ordered), "mean": sum(ordered)/len(ordered) if ordered else None,
            "median": quantile(.5), "p90": quantile(.9), "p95": quantile(.95)}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _json_safe(value):
    if isinstance(value, float) and not isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _raw_pose(prediction, width, height, protocol):
    """Analyze raw order. A valid reverse cycle is measured without repair."""
    if prediction.get("bbox_fallback") or str(prediction.get("keypoint_source", "raw_model")).lower() not in {
            "raw_model", "raw_pose", "model", "raw"}:
        return {"status": "BBOX_FALLBACK", "quad_status": "BBOX_FALLBACK", "valid": False}
    keypoints = prediction.get("keypoints")
    if keypoints is None or keypoints == []:
        return {"status": "MISSING_KEYPOINTS", "quad_status": "KEYPOINTS_MISSING", "valid": False}
    if not isinstance(keypoints, (list, tuple)) or len(keypoints) != 4:
        return {"status": "INVALID_QUAD", "quad_status": "KEYPOINT_COUNT_NOT_FOUR", "valid": False}
    if (any(not isinstance(p, (list, tuple)) or len(p) not in (2, 3) for p in keypoints)
            or len({len(p) for p in keypoints}) != 1
            or any(not all(_number(v) for v in p) for p in keypoints)):
        return {"status": "INVALID_QUAD", "quad_status": "NONFINITE_OR_MALFORMED_KEYPOINTS", "valid": False}
    points = [[float(p[0]), float(p[1])] for p in keypoints]
    analysis = analyze_quad(points)
    if not analysis.is_geometrically_valid:
        return {"status": "INVALID_QUAD", "quad_status": analysis.status, "valid": False}
    tolerance = protocol["keypoint_bounds_tolerance_px"]
    if any(not (-tolerance <= x <= width+tolerance and -tolerance <= y <= height+tolerance) for x, y in points):
        return {"status": "INVALID_QUAD", "quad_status": "OUT_OF_BOUNDS", "valid": False}
    return {"status": "VALID", "quad_status": analysis.status, "valid": True, "points": points}


def _scene_result(row, scene, protocol):
    width, height = scene["image"]["oriented_size"]
    _require(row.get("domain") == scene["domain"], "PREDICTION_DOMAIN_MISMATCH")
    _require(type(row.get("width")) is int and type(row.get("height")) is int
             and [row.get("width"), row.get("height")] == [width, height], "PREDICTION_INPUT_DIMENSIONS_MISMATCH")
    raw = row.get("predictions")
    _require(isinstance(raw, list), "PREDICTIONS_MISSING")
    predictions, indices = [], set()
    for prediction in raw:
        _require(isinstance(prediction, dict), "INVALID_PREDICTION_RECORD")
        index, confidence, class_id = prediction.get("index"), prediction.get("confidence"), prediction.get("class_id")
        _require(type(index) is int and index >= 0 and index not in indices, "PREDICTION_INDEX_INVALID_OR_DUPLICATE")
        _require(_number(confidence) and 0 <= confidence <= 1, "PREDICTION_CONFIDENCE_INVALID")
        _require(type(class_id) is int and class_id >= 0, "PREDICTION_CLASS_INVALID")
        indices.add(index)
        record = deepcopy(prediction)
        record.update(box_status="UNMATCHED", pose_status="NOT_EVALUATED", raw_quad_status="NOT_EVALUATED",
                      matched_plate_id=None, iou=None)
        if class_id not in protocol["plate_class_ids"]:
            record["box_status"] = "FILTERED_CLASS"
        elif confidence <= protocol["operational_conf"]:
            record["box_status"] = "FILTERED_CONFIDENCE"
        else:
            pose = _raw_pose(prediction, width, height, protocol)
            record["raw_quad_status"] = pose["quad_status"]
            record["pose_status"] = pose["status"]
            record["bbox_valid"] = _bbox(prediction.get("bbox")) is not None
        predictions.append(record)
    accepted = sorted((p for p in predictions if not p["box_status"].startswith("FILTERED")),
                      key=lambda p: (-p["confidence"], p["index"]))
    available = {plate["plate_id"]: plate for plate in scene["plates"]}
    matched, unmatched = [], []
    for prediction in accepted:
        candidates = [(box_iou(prediction.get("bbox"), plate["bbox"]), plate_id, plate)
                      for plate_id, plate in available.items()]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates or candidates[0][0] < protocol["match_iou"]:
            prediction["box_status"] = "FP"
            prediction["pose_status"] = "NOT_EVALUATED_UNMATCHED"
            unmatched.append(prediction["index"])
            continue
        iou, plate_id, plate = candidates[0]
        del available[plate_id]
        prediction.update(box_status="TP", matched_plate_id=plate_id, iou=iou)
        status = prediction["pose_status"] if plate["pose_eligible"] else "NOT_EVALUATED"
        prediction["pose_status"] = status
        pair = {"plate_id": plate_id, "prediction_index": prediction["index"], "iou": iou,
                "pose_status": status, "pose_eligible": plate["pose_eligible"],
                "pose_exclusion": deepcopy(plate.get("pose_exclusion")),
                "raw_quad_status": prediction["raw_quad_status"],
                "gt_bbox": deepcopy(plate["bbox"]), "gt_quad": deepcopy(plate["quad"]),
                "predicted_bbox": deepcopy(prediction.get("bbox")),
                "raw_keypoints": deepcopy(prediction.get("keypoints")),
                "corner_errors_px": None, "corner_errors_normalized": None}
        if status == "VALID":
            diagonal = hypot(plate["bbox"][2]-plate["bbox"][0], plate["bbox"][3]-plate["bbox"][1])
            errors = [hypot(raw_point[0]-gt_point[0], raw_point[1]-gt_point[1])
                      for raw_point, gt_point in zip(prediction["keypoints"], plate["quad"])]
            pair.update(corner_errors_px=errors, corner_errors_normalized=[value/diagonal for value in errors])
        matched.append(pair)
    return {"scene_sha256": scene["sha256"], "domain": scene["domain"], "file": scene.get("file", ""),
            "width": width, "height": height, "matched": matched,
            "unmatched_gt_ids": sorted(available), "unmatched_prediction_indices": unmatched,
            "predictions": sorted(predictions, key=lambda p: p["index"])}


def _domain_result(domain, results, gt_scenes):
    cohort = [s for s in gt_scenes if domain == "ALL" or s["domain"] == domain]
    selected = [s for s in results if domain == "ALL" or s["domain"] == domain]
    processed = {s["scene_sha256"] for s in selected}
    cohort_plates = [p for s in cohort for p in s["plates"]]
    plates = [p for s in cohort if s["sha256"] in processed for p in s["plates"]]
    by_id = {p["plate_id"]: p for p in plates}
    matches = [p for s in selected for p in s["matched"]]
    accepted = [p for s in selected for p in s["predictions"] if p["box_status"] in {"TP", "FP"}]
    eligible_matches = [p for p in matches if p["pose_eligible"]]
    valid = [p for p in eligible_matches if p["pose_status"] == "VALID"]
    tp, fp, fn = len(matches), sum(p["box_status"] == "FP" for p in accepted), sum(len(s["unmatched_gt_ids"]) for s in selected)
    pose_gt = sum(p["pose_eligible"] for p in plates)
    raw_status = Counter(p["raw_quad_status"] for p in accepted)
    counts = Counter(p["pose_status"] for p in eligible_matches)
    raw_valid = raw_status["VALID"] + raw_status["REVERSED_WINDING"]
    return {"domain": domain, "processed_scenes": len(selected), "cohort_scenes": len(cohort),
        "pending_scenes": len(cohort)-len(selected), "cohort_box_gt": len(cohort_plates),
        "cohort_pose_gt": sum(p["pose_eligible"] for p in cohort_plates),
        "box_gt": len(plates), "box_tp": tp, "box_fp": fp, "box_fn": fn,
        "box_precision": _ratio(tp, tp+fp), "box_recall": _ratio(tp, tp+fn), "box_f1": _ratio(2*tp, 2*tp+fp+fn),
        "box_iou": distribution(p["iou"] for p in matches), "no_detection_scenes": sum(
            not any(p["box_status"] in {"TP", "FP"} for p in s["predictions"]) for s in selected),
        "pose_gt": pose_gt, "pose_matched_gt": len(eligible_matches),
        "pose_unmatched_gt": sum(by_id[pid]["pose_eligible"] for s in selected for pid in s["unmatched_gt_ids"]),
        "pose_valid_matched": len(valid), "pose_coverage": _ratio(len(valid), pose_gt),
        "pose_missing_keypoints": counts["MISSING_KEYPOINTS"], "pose_invalid_quad": counts["INVALID_QUAD"],
        "pose_bbox_fallback": counts["BBOX_FALLBACK"],
        "pose_reversed_winding": sum(p["raw_quad_status"] == "REVERSED_WINDING" for p in valid),
        "pose_excluded_gt": len(plates)-pose_gt, "pose_excluded_matched": len(matches)-len(eligible_matches),
        "pose_unmatched_predictions": fp,
        "unindexed_geometry_diagnostics": {"gt": len(plates), "matched": len(matches),
            "valid_matched": sum(p["raw_quad_status"] in {"VALID", "REVERSED_WINDING"} for p in matches),
            "coverage": _ratio(sum(p["raw_quad_status"] in {"VALID", "REVERSED_WINDING"} for p in matches), len(plates))},
        "corner_error_px": distribution(value for pair in valid for value in pair["corner_errors_px"]),
        "corner_error_normalized": distribution(value for pair in valid for value in pair["corner_errors_normalized"]),
        "per_corner": [{"index": index, "name": name,
            "error_px": distribution(pair["corner_errors_px"][index] for pair in valid),
            "error_normalized": distribution(pair["corner_errors_normalized"][index] for pair in valid)}
            for index, name in enumerate(CORNER_NAMES)],
        "prediction_diagnostics": {"operating_predictions": len(accepted),
            "filtered_confidence": sum(p["box_status"] == "FILTERED_CONFIDENCE" for s in selected for p in s["predictions"]),
            "filtered_class": sum(p["box_status"] == "FILTERED_CLASS" for s in selected for p in s["predictions"]),
            "invalid_bbox": sum(not p["bbox_valid"] for p in accepted),
            "raw_quad_status_counts": dict(sorted(raw_status.items())),
            "raw_valid_quad_predictions": raw_valid, "raw_valid_quad_rate": _ratio(raw_valid, len(accepted))}}


def aggregate(rows, gt_scenes, protocol=None):
    """Evaluate one model's rows against an already approved GT preflight.

    GT trust/metric eligibility is enforced by the runner's preflight gate.
    Here duplicate/unknown scenes, dimensions, IDs and malformed GT fail closed.
    Rows may be a prefix/subset after cancellation; no pending scene is scored.
    """
    protocol = deepcopy(_FROZEN_PROTOCOL if protocol is None else protocol)
    _require(protocol == _FROZEN_PROTOCOL, "MT_METRIC_PROTOCOL_CHANGED")
    _require(isinstance(rows, (list, tuple)) and isinstance(gt_scenes, (list, tuple)), "SCENE_LIST_REQUIRED")
    gt_by_sha, plate_ids = {}, set()
    for scene in gt_scenes:
        sha = scene.get("sha256")
        _require(isinstance(sha, str) and bool(sha) and sha not in gt_by_sha, "GT_SCENE_IDENTITY_INVALID_OR_DUPLICATE")
        _require(scene.get("domain") in {"DAY", "NIGHT"}, "GT_DOMAIN_INVALID")
        size = scene.get("image", {}).get("oriented_size")
        _require(isinstance(size, (list, tuple)) and len(size) == 2 and all(type(v) is int and v > 0 for v in size), "GT_INPUT_SIZE_INVALID")
        _require(isinstance(scene.get("plates"), list), "GT_PLATES_MISSING")
        for plate in scene["plates"]:
            plate_id = plate.get("plate_id")
            _require(isinstance(plate_id, str) and plate_id and plate_id not in plate_ids, "GT_PLATE_IDENTITY_INVALID_OR_DUPLICATE")
            plate_ids.add(plate_id)
            _require(plate.get("bbox_eligible") is True and _bbox(plate.get("bbox")) is not None, "GT_BBOX_NOT_ELIGIBLE")
            _require(type(plate.get("pose_eligible")) is bool, "GT_POSE_ELIGIBILITY_MISSING")
            if plate["pose_eligible"]:
                analysis = analyze_quad(plate.get("quad"))
                _require(analysis.status == "VALID", "ELIGIBLE_GT_QUAD_INVALID")
            else:
                _require(bool(plate.get("pose_exclusion")) and plate.get("pose_evaluation_status") == "EXCLUDED_BY_OPERATOR", "GT_POSE_EXCLUSION_UNEXPLAINED")
        gt_by_sha[sha] = scene
    by_sha = {}
    for row in rows:
        _require(isinstance(row, dict), "PREDICTION_SCENE_RECORD_INVALID")
        sha = row.get("scene_sha256")
        _require(isinstance(sha, str) and sha in gt_by_sha and sha not in by_sha, "PREDICTION_SCENE_UNKNOWN_OR_DUPLICATE")
        by_sha[sha] = row
    results = [_scene_result(by_sha[sha], gt_by_sha[sha], protocol) for sha in sorted(by_sha)]
    output = {"schema": METRICS_SCHEMA, "protocol": protocol,
        "status": "COMPLETE" if len(by_sha) == len(gt_by_sha) else "PARTIAL",
        "metric_kind": "operational_box_and_raw_indexed_corner_error",
        "domains": {domain: _domain_result(domain, results, gt_scenes) for domain in ("ALL", "DAY", "NIGHT")},
        "scenes": results, "official_metrics": {"box_ap": "NOT_COMPUTED", "pose_ap": "NOT_COMPUTED"}}
    return _json_safe(output)
