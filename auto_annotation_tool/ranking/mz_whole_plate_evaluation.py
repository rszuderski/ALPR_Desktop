"""Char Detect test-crop evaluation using the shared reading order and CER."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from ..character_recognition.reading_order import annotate_records_reading_order
from ..registration_text import normalize_registration
from ..dataset_split_assignment import validate_group_assignment
from .mobile_human_review import align_plate_text

READING_ORDER_POLICY = "alpr.reading_order.geometry.v1"


def read_plate_records(predictions: list[dict]) -> dict:
    # Prediction reading order uses only predicted geometry, never GT text,
    # expected character count or the old GT pack.
    ordered = annotate_records_reading_order(predictions)
    prediction = normalize_registration("".join(str(row.get("character") or "") for row in ordered))
    return {"prediction": prediction, "no_read": not prediction,
            "reading_order_policy": READING_ORDER_POLICY, "ordered_predictions": ordered}


def evaluate_plate_records(ground_truth: str, predictions: list[dict]) -> dict:
    reading = read_plate_records(predictions)
    result = asdict(align_plate_text(ground_truth, reading["prediction"]))
    result.update(reading)
    return result


def aggregate_plate_metrics(records: list[dict]) -> dict:
    count = len(records)
    gt_characters = sum(len(row["ground_truth"]) for row in records)
    sums = {name:sum(row[name] for row in records) for name in ["correct_characters",
        "incorrect_characters","missing_characters","extra_characters","edit_distance"]}
    return {"schema":"alpr.mz_whole_plate_metrics.v1","plates":count,
        "exact_matches":sum(row["exact_match"] for row in records),
        "exact_match_rate":sum(row["exact_match"] for row in records)/count if count else None,
        "no_read":sum(row["no_read"] for row in records),"ground_truth_characters":gt_characters,
        "cer":sums["edit_distance"]/gt_characters if gt_characters else None,
        "cer_aggregation":"micro_edit_distance_over_gt_characters",**sums}


def approved_manifest_text(item: dict) -> str:
    chars = normalize_registration("".join(str(row.get("character") or "") for row in item.get("characters") or []))
    text = normalize_registration((item.get("layout") or {}).get("plate_text_reading_order") or chars)
    if not text or text!=chars:
        raise ValueError("Final PZ3 approved text is absent or inconsistent with characters")
    return text


def evaluate_mz_test_split(model, dataset_dir, inference: dict) -> tuple[dict,list[dict]]:
    """Run only test items from the immutable assignment, without GT assistance."""
    root = Path(dataset_dir)
    assignment = json.loads((root/"split_assignment_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((root/"metadata_manifest.json").read_text(encoding="utf-8"))
    check = validate_group_assignment(assignment,source_manifest=manifest)
    if not check["ok"]:
        raise ValueError("Invalid test assignment: "+str(check["errors"]))
    if str(getattr(model,"task","detect"))!="detect":
        raise ValueError("Whole-plate MZ evaluator requires Detect")
    names = getattr(model,"names",{})
    if isinstance(names,list):
        names = dict(enumerate(names))
    names = {int(key):str(value) for key,value in names.items()}
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if names!={i:ch for i,ch in enumerate(alphabet)}:
        raise ValueError("MZ checkpoint must have the final 36 character classes")
    items = {item["pid"]:item for item in manifest["items"]}
    rows = []
    for row in assignment["items"]:
        if row["split"]!="test":
            continue
        item = items[row["source_item_id"]]
        image = root/row["image_path"]
        results = model.predict(source=str(image),imgsz=int(inference["imgsz"]),
            conf=float(inference["confidence"]),iou=float(inference["iou"]),
            max_det=int(inference["max_det"]),device=inference["device"],
            end2end=bool(inference["end2end"]),augment=False,half=False,verbose=False)
        if len(results)!=1:
            raise ValueError("Inference must return exactly one result per test crop")
        boxes = results[0].boxes
        predictions = []
        if boxes is not None:
            for bbox,cls,confidence in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist()):
                predictions.append({"bbox":bbox,"character":names[int(cls)],"confidence":float(confidence)})
        evaluated = evaluate_plate_records(approved_manifest_text(item),predictions)
        evaluated.update(crop_identity=row["crop_identity"],source_pid=row["source_pid"],
            group_id=row["group_id"],image_path=row["image_path"])
        rows.append(evaluated)
    if not rows:
        raise ValueError("Final test split is empty")
    summary = aggregate_plate_metrics(rows)
    summary.update(assignment_sha256=assignment["assignment_sha256"],inference=dict(inference),
        reading_order_policy=READING_ORDER_POLICY,ground_truth_source="final_approved_PZ3_manifest")
    return summary,rows
