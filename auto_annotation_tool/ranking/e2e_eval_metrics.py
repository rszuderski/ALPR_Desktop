"""Pure E2E scoring of persisted predictions. No image/model/file operations."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from .eval396 import digest, require
from .mt_eval_metrics import aggregate as aggregate_mt
from .mz_whole_plate_evaluation import read_plate_records, aggregate_plate_metrics
from .mobile_human_review import align_plate_text
from ..registration_text import normalize_registration, NORMALIZATION_POLICY

SCHEMA = "alpr.e2e.metrics.v1"
CHAR_COUNTS = {"ALL": 3052, "DAY": 1522, "NIGHT": 1530}
SCORING_POLICY = {"id": "alpr.e2e.box479_text449_micro_cer.v1",
    "matching": "confidence_desc_iou_desc_plate_id_v1", "iou": .5,
    "normalization": NORMALIZATION_POLICY, "cer": "micro_edit_distance_over_gt_characters",
    "unmatched_readable_gt": "empty_prediction_all_deletions",
    "unreadable": "bbox_only", "fp_ocr": "separate_no_synthetic_gt",
    "conditional": "readable_bbox_matched_successfully_rectified_including_empty_mz"}


def validate_reference(scenes):
    ids, shas = set(), set()
    counts = {d: {"scenes": 0, "bbox": 0, "pose": 0, "ocr": 0, "characters": 0} for d in ("ALL", "DAY", "NIGHT")}
    for scene in scenes:
        require(scene["sha256"] not in shas and scene["domain"] in {"DAY", "NIGHT"}, "E2E_GT_SCENE_INVALID")
        shas.add(scene["sha256"])
        for d in ("ALL", scene["domain"]):
            counts[d]["scenes"] += 1
        for plate in scene["plates"]:
            pid = plate["plate_id"]
            require(pid not in ids and plate.get("bbox_eligible") is True, "E2E_GT_PLATE_INVALID")
            ids.add(pid)
            readable = plate.get("text_decision") == "approved"
            require(type(plate.get("ocr_eligible")) is bool and plate["ocr_eligible"] == readable,
                    "E2E_READABILITY_CHANGED")
            text = plate.get("ground_truth_text")
            if readable:
                require(isinstance(text, str), "E2E_GT_TEXT_MISSING")
                normalized = normalize_registration(text)
                require(bool(normalized), "E2E_APPROVED_GT_TEXT_EMPTY")
            else:
                require(plate.get("text_decision") == "excluded_unreadable", "E2E_GT_DECISION_INVALID")
                require(text is None, "E2E_UNREADABLE_GT_TEXT_INVALID")
                normalized = ""  # Character count only; preserve the reference null.
            for d in ("ALL", scene["domain"]):
                counts[d]["bbox"] += 1
                counts[d]["pose"] += bool(plate["pose_eligible"])
                counts[d]["ocr"] += readable
                counts[d]["characters"] += len(normalized) if readable else 0
    return counts


def score_pair(rows, reference):
    """Rows must have finished their complete 0..N prediction list before scoring.

    Uses exactly the existing MT association, including unreadable/Pose-excluded
    GT, and the existing text normalizer/alignment/micro aggregation.
    """
    cohort = validate_reference(reference)
    by_sha = {s["sha256"]: s for s in reference}
    seen = set()
    mt_rows, by_prediction, scenes = [], {}, []
    for row in rows:
        sha = row["scene_sha256"]
        require(sha in by_sha and sha not in seen and row.get("status") == "COMPLETE", "E2E_UNFINISHED_OR_UNKNOWN_SCENE")
        seen.add(sha)
        source = by_sha[sha]
        require([row["width"], row["height"]] == source["image"]["oriented_size"], "E2E_SCENE_DIMENSIONS_CHANGED")
        predictions = []
        for item in row["detections"]:
            mt = item["mt"]
            key = (sha, mt["index"])
            require(key not in by_prediction, "E2E_DUPLICATE_PREDICTION_INDEX")
            require(item["reading"] == read_plate_records(item["characters"]), "E2E_READING_CHANGED")
            require(item["status"] in {"READ", "NO_READ"}, "E2E_INFRASTRUCTURE_FAILURE_IS_NOT_NO_READ")
            require((item["status"] == "NO_READ") == item["reading"]["no_read"], "E2E_READ_STATUS_CHANGED")
            by_prediction[key] = item
            predictions.append(deepcopy(mt))
        mt_rows.append({"scene_sha256": sha, "domain": source["domain"],
                        "width": row["width"], "height": row["height"], "predictions": predictions})
    # Public MT aggregate provides the authoritative, regression-tested matching.
    association = aggregate_mt(mt_rows, reference)
    gt_rows, fp_rows = [], []
    for scene_result in association["scenes"]:
        sha = scene_result["scene_sha256"]
        source = by_sha[sha]
        matches = {m["plate_id"]: m for m in scene_result["matched"]}
        for plate in source["plates"]:
            match = matches.get(plate["plate_id"])
            detection = by_prediction[(sha, match["prediction_index"])] if match else None
            prediction = detection["reading"]["prediction"] if detection else ""
            readable = plate["ocr_eligible"]
            if not detection:
                failure = "MT_NO_DETECTION" if not scene_result["predictions"] else "MT_UNMATCHED_GT"
            else:
                failure = detection["failure"]
            alignment = None
            if readable:
                alignment = asdict(align_plate_text(plate["ground_truth_text"], prediction))
                alignment["alignment"] = list(alignment["alignment"])
                alignment["no_read"] = not prediction
                if prediction and not alignment["exact_match"]:
                    failure = "OCR_MISMATCH"
            gt_rows.append({"scene_sha256": sha, "plate_id": plate["plate_id"], "domain": source["domain"],
                "prediction_index": match["prediction_index"] if match else None,
                "iou": match["iou"] if match else None, "readable": readable,
                "conditional_eligible": bool(readable and detection and detection["rectification"]["status"] == "RECTIFIED"),
                "prediction": prediction, "failure": failure, "scoring": alignment})
        for index in scene_result["unmatched_prediction_indices"]:
            det = by_prediction[(sha, index)]
            fp_rows.append({"scene_sha256": sha, "prediction_index": index, "domain": source["domain"],
                "prediction": det["reading"]["prediction"], "failure": det["failure"],
                "fp_ocr": bool(det["reading"]["prediction"])})
        scenes.append({"scene_sha256": sha, "domain": source["domain"],
            "detections": len(scene_result["predictions"]), "matched": len(matches),
            "unmatched_gt": len(scene_result["unmatched_gt_ids"]), "fp": len(scene_result["unmatched_prediction_indices"])})
    domains = {}
    for domain in ("ALL", "DAY", "NIGHT"):
        gt = [r for r in gt_rows if domain == "ALL" or r["domain"] == domain]
        readable = [r for r in gt if r["readable"]]
        conditional = [r for r in readable if r["conditional_eligible"]]
        fp = [r for r in fp_rows if domain == "ALL" or r["domain"] == domain]
        text = aggregate_plate_metrics([r["scoring"] for r in readable])
        box = association["domains"][domain]
        conditional_exact = sum(r["scoring"]["exact_match"] for r in conditional)
        domains[domain] = {**text, "processed_scenes": box["processed_scenes"],
            "cohort": cohort[domain], "pending_scenes": box["pending_scenes"],
            **{k: box[k] for k in ("box_tp", "box_fp", "box_fn", "box_precision", "box_recall", "box_f1", "box_iou")},
            "conditional_plates": len(conditional), "conditional_exact_matches": conditional_exact,
            "conditional_exact_rate": conditional_exact/len(conditional) if conditional else None,
            "fp_ocr": sum(r["fp_ocr"] for r in fp),
            "failures_readable": dict(sorted(Counter(r["failure"] for r in readable if r["failure"]).items())),
            "failures_all_gt": dict(sorted(Counter(r["failure"] for r in gt if r["failure"]).items()))}
    return {"schema": SCHEMA, "status": association["status"], "policy": deepcopy(SCORING_POLICY),
            "domains": domains, "scenes": scenes, "gt_rows": gt_rows, "fp_rows": fp_rows,
            "official_metrics": {"pose_map": "NOT_COMPUTED", "box_map": "NOT_COMPUTED"}}
