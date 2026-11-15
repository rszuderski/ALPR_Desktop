"""Phase 2A: read-only MT input/GT preflight and explicit operator attestation.

No model loader, predictor, dataset builder or inference entry point lives here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import cv2
import numpy as np
import PIL

from ..image_orientation import ORIENTATION_POLICY, DECODER, inspect_oriented_image
from .eval396 import (EvidenceReader, SELECTION_SHA, SELECTION_MANIFEST_SHA, FREEZE_SHA,
                      _read_selection, contained, digest, require)
from .mt_metric_policy import (METRIC_POLICY_ID, METRIC_POLICY_SHA, load_metric_policy,
                               bound_pose_exclusion, metric_population)

REVIEW_SCHEMA = "alpr.mt_geometry_preflight.v2"
APPROVAL_SCHEMA = "alpr.mt_geometry_operator_contract.v2"
CORNER_NAMES = ("TL", "TR", "BR", "BL")
POLICY = {"id": "alpr.mt.geometry_protocol.v2", "orientation_policy": ORIENTATION_POLICY,
          "metric_policy_id": METRIC_POLICY_ID, "metric_policy_sha256": METRIC_POLICY_SHA,
          "pixel_space": "EXIF-oriented original scene XY",
          "decoder": DECODER, "input_format": "BGR uint8", "gt_transform": "none",
          "gt_corner_order": list(CORNER_NAMES), "corner_reordering": "forbidden",
          "bounds_tolerance_px": 1.0, "pillow_comparison_mae_limit": 6.0, "pillow_comparison_p95_limit": 18.0}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("GT_NONFINITE_OR_NONNUMERIC_COORDINATE")
    return float(value)


def validate_gt_bbox(bbox, width, height):
    require(isinstance(bbox, (list, tuple)) and len(bbox) == 4, "GT_BBOX_MISSING_OR_INVALID")
    x1, y1, x2, y2 = box = [_number(value) for value in bbox]
    require(x1 < x2 and y1 < y2, "GT_BBOX_DEGENERATE")
    tolerance = POLICY["bounds_tolerance_px"]
    require(-tolerance <= x1 and x2 <= width + tolerance
            and -tolerance <= y1 and y2 <= height + tolerance, "GT_OUTSIDE_ORIENTED_SCENE")
    return box


def validate_gt_geometry(bbox, polygon, width, height):
    box = validate_gt_bbox(bbox, width, height)
    require(isinstance(polygon, (list, tuple)) and len(polygon) == 4, "GT_QUAD_MISSING_OR_NOT_FOUR_CORNERS")
    points = []
    for point in polygon:
        require(isinstance(point, (list, tuple)) and len(point) == 2, "GT_INVALID_CORNER")
        points.append([_number(point[0]), _number(point[1])])
    tolerance = POLICY["bounds_tolerance_px"]
    for x, y in points:
        require(-tolerance <= x <= width + tolerance and -tolerance <= y <= height + tolerance,
                "GT_OUTSIDE_ORIENTED_SCENE")
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    require(not all(value < -1e-6 for value in cross), "GT_CORNER_ORDER_REVERSED")
    require(all(value > 1e-6 for value in cross), "GT_QUAD_INVALID")
    require(points[0][0] + points[3][0] < points[1][0] + points[2][0]
            and points[0][1] + points[1][1] < points[2][1] + points[3][1], "GT_CORNER_ORDER_NOT_TL_TR_BR_BL")
    return box, points


def assess_metric_scope(scenes, metric_policy):
    """Derive flags from original XY; an exclusion never rewrites GT or a bbox."""
    require(digest(metric_policy) == METRIC_POLICY_SHA, "METRIC_POLICY_CHANGED")
    diagnostics, blocking, exclusions = [], [], []
    expected_ids = {row["plate_id"] for row in metric_policy["exclusions"]}
    plate_ids = []
    for scene in scenes:
        width, height = scene["image"]["oriented_size"]
        for plate in scene["plates"]:
            plate_ids.append(plate["plate_id"])
            bbox_issue = issue = ""
            try:
                validate_gt_bbox(plate["bbox"], width, height)
            except ValueError as exc:
                bbox_issue = str(exc)
            try:
                validate_gt_geometry(plate["bbox"], plate["quad"], width, height)
            except ValueError as exc:
                issue = str(exc)
            plate["geometry_issue"] = issue
            exclusion = (bound_pose_exclusion(scene, plate, issue, metric_policy) if not bbox_issue else None)
            plate.update(bbox_eligible=not bbox_issue, pose_eligible=not issue,
                         ocr_eligible=plate["text_decision"] == "approved", pose_exclusion=exclusion,
                         pose_evaluation_status="ELIGIBLE" if not issue else "BLOCKED")
            diagnostic = {"scene": scene["file"], "sha256": scene["sha256"],
                          "plate_id": plate["plate_id"], "reason": issue}
            if issue:
                diagnostics.append(diagnostic)
            if exclusion:
                plate["pose_evaluation_status"] = "EXCLUDED_BY_OPERATOR"
                exclusions.append(exclusion)
            elif issue:
                blocking.append(diagnostic)
            if plate["plate_id"] in expected_ids and not exclusion:
                blocking.append({**diagnostic, "reason": "POSE_EXCLUSION_ANCHOR_MISMATCH"})
    counts = metric_population(scenes)
    if len(plate_ids) != len(set(plate_ids)) or len(scenes) != len({s["sha256"] for s in scenes}):
        blocking.append({"reason": "METRIC_IDENTITY_NOT_UNIQUE", "plate_id": "", "scene": ""})
    if counts != metric_policy["expected_counts"]:
        blocking.append({"reason": "METRIC_POPULATION_CHANGED", "plate_id": "", "scene": ""})
    return {"geometry_issues": diagnostics, "blocking_geometry_issues": blocking,
            "metric_exclusions": exclusions, "metric_counts": counts}


def required_review_scenes(scenes, diagnostics):
    review = [s["sha256"] for s in scenes if s["image"]["exif_orientation"] != 1]
    for domain in ("DAY", "NIGHT"):
        control = next(s for s in scenes if s["domain"] == domain and s["image"]["exif_orientation"] == 1)
        review.append(control["sha256"])
    return list(dict.fromkeys(review + [issue["sha256"] for issue in diagnostics]))


def cohort_counts(scenes):
    return {"scenes": len(scenes), "plates": sum(len(s["plates"]) for s in scenes),
            "scene_domains": dict(Counter(s["domain"] for s in scenes)),
            "plate_domains": {domain: sum(len(s["plates"]) for s in scenes if s["domain"] == domain)
                              for domain in ("DAY", "NIGHT")},
            "unreadable_included": sum(p["text_decision"] == "excluded_unreadable" for s in scenes for p in s["plates"])}


def review_fingerprint(report):
    return digest({key: report[key] for key in ("schema", "selection_sha", "gt_freeze_sha", "policy",
                  "models", "scenes", "counts", "geometry_issues", "review_scene_hashes", "runtime", "code_sha256", "source_files",
                  "metric_policy", "metric_policy_sha256", "metric_counts", "metric_exclusions", "blocking_geometry_issues",
                  "repo_root", "source_scene_root", "selection_dir", "freeze_dir")})


def preflight_mt_geometry(selection_dir, *, repo_root=None, progress=None, cancelled=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    selection_dir = Path(selection_dir).resolve()
    if selection_dir.name == "selection_manifest.json":
        selection_dir = selection_dir.parent
    def notify(message):
        if cancelled and cancelled():
            raise InterruptedError("Sprawdzanie geometrii anulowane")
        if progress:
            progress(message)
    reader = EvidenceReader(notify)
    metric_policy = load_metric_policy(reader)
    notify("Weryfikuję selekcję, freeze, checkpointy i niezależność źródeł…")
    manifest, selection = _read_selection(reader, selection_dir, repo_root)
    references = manifest["references"]
    scene_root = contained(repo_root, references["source_scene_root"])
    freeze = contained(repo_root, references["source_snapshot_folder"])
    metadata = reader.json(freeze / "metadata.json", references["parent_metadata_sha256"])
    by_scene = {}
    for plate in selection["included_plates.json"]:
        by_scene.setdefault(plate["source_file_sha256"], []).append(plate)
    scenes = []
    for index, row in enumerate(selection["included_scenes.json"], 1):
        notify(f"Geometria i EXIF: scena {index}/396 — {row['file']}")
        path = contained(scene_root, row["file"])
        try:
            image, image_info = inspect_oriented_image(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{row['file']}: {exc}") from exc
        width, height = image_info["oriented_size"]
        plates = []
        for plate in by_scene[row["sha256"]]:
            original = metadata[plate["plate_id"]]
            require(original.get("source_file_sha256") == row["sha256"], "GT_SOURCE_IDENTITY_MISMATCH")
            plates.append({"plate_id": plate["plate_id"], "source_annotation_id": plate["source_annotation_id"],
                           "text_decision": plate["decision"], "bbox": original.get("source_bbox"),
                           "quad": original.get("source_polygon")})
        scenes.append({**row, "image": image_info, "plates": plates})
        del image
    counts = cohort_counts(scenes)
    require(counts == {"scenes": 396, "plates": 479, "scene_domains": {"DAY": 199, "NIGHT": 197},
                      "plate_domains": {"DAY": 235, "NIGHT": 244}, "unreadable_included": 30}, "MT_POPULATION_CHANGED")
    metric_scope = assess_metric_scope(scenes, metric_policy)
    review = required_review_scenes(scenes, metric_scope["geometry_issues"])
    code = {}
    for name in ("image_orientation.py", "annotators/base.py", "annotators/plate_annotator.py",
                 "ranking/mt_geometry_review.py", "ranking/mt_metric_policy.py"):
        code[name] = hashlib.sha256(reader.read(repo_root / "auto_annotation_tool" / name)).hexdigest()
    models = [m for m in selection["model_lineage.json"] if m["role"] == "plate"]
    require(sorted(m["label"] for m in models) == sorted(metric_policy["model_labels"]), "METRIC_POLICY_MODEL_SET_CHANGED")
    report = {"schema": REVIEW_SCHEMA, "status": "FAIL" if metric_scope["blocking_geometry_issues"] else "READY_FOR_REVIEW", "selection_sha": SELECTION_SHA,
        "gt_freeze_sha": FREEZE_SHA, "policy": deepcopy(POLICY), "source_scene_root": str(scene_root),
        "repo_root": str(repo_root),
        "selection_dir": str(selection_dir), "freeze_dir": str(freeze),
        "models": models, "scenes": scenes, "counts": counts, **metric_scope,
        "metric_policy": metric_policy, "metric_policy_sha256": METRIC_POLICY_SHA, "review_scene_hashes": review,
        "runtime": {"opencv": cv2.__version__, "pillow": PIL.__version__, "numpy": np.__version__},
        "code_sha256": code, "source_files": dict(reader.receipt),
        "checkpoint_tensor_contract": "NOT_INSPECTED_PHASE2A", "inference_performed": False,
        "visual_gt_approved": False, "created_at": datetime.now(timezone.utc).isoformat()}
    reader.finish()
    report["review_sha256"] = review_fingerprint(report)
    return report


def verify_report_source_binding(report, reader):
    """Bind in-memory review values to the independently pinned source files.

    A caller can recompute a report's integrity SHA. It cannot replace the
    trusted selection manifest or the metadata SHA recorded in that manifest.
    """
    root = Path(report["repo_root"]).resolve()
    selection_dir = Path(report["selection_dir"]).resolve()
    manifest = reader.json(selection_dir / "selection_manifest.json", SELECTION_MANIFEST_SHA)
    require(manifest["selection_fingerprint_sha256"] == SELECTION_SHA, "SELECTION_FINGERPRINT_MISMATCH")
    refs = manifest["references"]
    freeze = contained(root, refs["source_snapshot_folder"])
    scene_root = contained(root, refs["source_scene_root"])
    require(freeze.resolve() == Path(report["freeze_dir"]).resolve()
            and scene_root.resolve() == Path(report["source_scene_root"]).resolve(), "REPORT_SOURCE_PATHS_CHANGED")
    source_scenes = reader.json(selection_dir / "included_scenes.json", manifest["files"]["included_scenes.json"])
    source_plates = reader.json(selection_dir / "included_plates.json", manifest["files"]["included_plates.json"])
    metadata = reader.json(freeze / "metadata.json", refs["parent_metadata_sha256"])
    models = reader.json(selection_dir / "model_lineage.json", manifest["files"]["model_lineage.json"])
    require(report["models"] == [m for m in models if m["role"] == "plate"], "REPORT_MODELS_CHANGED")
    require(len(source_scenes) == len(report["scenes"]), "REPORT_SCENE_MEMBERSHIP_CHANGED")
    expected = {p["plate_id"]: p for p in source_plates}
    observed = set()
    for scene, source_scene in zip(report["scenes"], source_scenes):
        require(all(scene.get(key) == value for key, value in source_scene.items()), "REPORT_SCENE_IDENTITY_CHANGED")
        for plate in scene["plates"]:
            pid = plate["plate_id"]
            require(pid in expected and pid not in observed, "REPORT_PLATE_MEMBERSHIP_CHANGED")
            observed.add(pid)
            source, gt = expected[pid], metadata[pid]
            require(source["source_file_sha256"] == scene["sha256"]
                    and source["domain"] == scene["domain"]
                    and plate["source_annotation_id"] == source["source_annotation_id"]
                    and plate["text_decision"] == source["decision"]
                    and plate["bbox"] == gt["source_bbox"]
                    and plate["quad"] == gt["source_polygon"], f"REPORT_DIFFERS_FROM_FROZEN_GT: {pid}")
    require(observed == set(expected), "REPORT_PLATE_MEMBERSHIP_CHANGED")


def recheck_review(report, *, allow_geometry_failure=False):
    require(report.get("schema") == REVIEW_SCHEMA, "PREFLIGHT_SCHEMA_INVALID")
    require(all(key in report for key in ("repo_root", "source_scene_root", "selection_dir", "freeze_dir")),
            "PREFLIGHT_SOURCE_BINDING_MISSING")
    ready = report.get("status") == "READY_FOR_REVIEW" and not report.get("blocking_geometry_issues")
    require(ready or (allow_geometry_failure and report.get("status") == "FAIL"), "PREFLIGHT_NOT_READY")
    require(report.get("review_sha256") == review_fingerprint(report), "PREFLIGHT_CHANGED")
    require(report["selection_sha"] == SELECTION_SHA and report["gt_freeze_sha"] == FREEZE_SHA
            and report["policy"] == POLICY, "GEOMETRY_POLICY_CHANGED")
    require(report["metric_policy"] == load_metric_policy()
            and report["metric_policy_sha256"] == METRIC_POLICY_SHA, "METRIC_POLICY_CHANGED")
    derived_scenes = deepcopy(report["scenes"])
    derived = assess_metric_scope(derived_scenes, report["metric_policy"])
    require(derived_scenes == report["scenes"] and all(report[key] == value for key, value in derived.items()),
            "METRIC_ELIGIBILITY_CHANGED")
    require(report["counts"] == cohort_counts(derived_scenes), "COHORT_COUNTS_CHANGED")
    require(report["review_scene_hashes"] == required_review_scenes(derived_scenes, derived["geometry_issues"]),
            "REQUIRED_VISUAL_REVIEW_CHANGED")
    require(sorted(m["label"] for m in report["models"]) == sorted(report["metric_policy"]["model_labels"]),
            "METRIC_POLICY_MODEL_SET_CHANGED")
    reader = EvidenceReader()
    reader.receipt = dict(report["source_files"])
    verify_report_source_binding(report, reader)
    reader.finish()


def _review_output_root(report, output_root):
    output_root = Path(output_root).resolve()
    experiment_root = Path(report["selection_dir"]).parent.parent
    protected = [Path(report[key]).resolve() for key in ("selection_dir", "freeze_dir", "source_scene_root")]
    protected.append(experiment_root / "evaluation_runs/E-MZ-DN-01_EVAL396_MZ_v1")
    require(not any(output_root.is_relative_to(path) for path in protected), "PROTECTED_OUTPUT_DIRECTORY")
    return output_root


def prepare_geometry_protocol(report, output_root):
    """Persist the authorized metric decision and an unsigned review candidate.

    This is not an inference run and not an operator visual approval. The
    future signed contract refers to this same source/policy fingerprint.
    """
    recheck_review(report)
    output_root = _review_output_root(report, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    name = "EVAL396_MT_protocol_review_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    stage = output_root / (name + ".partial")
    stage.mkdir()
    payloads = {"preflight.json": report, "metric_eligibility_policy.json": report["metric_policy"],
                "status.json": {"schema": "alpr.mt_geometry_review_candidate.v1",
                    "status": "AWAITING_OPERATOR_GEOMETRY_REVIEW", "measurement_status": "NOT_RUN",
                    "metric_exclusion_authorization": "OPERATOR_MESSAGE_2026-10-09",
                    "metric_policy_sha256": METRIC_POLICY_SHA, "review_sha256": report["review_sha256"],
                    "metric_counts": report["metric_counts"], "visual_gt_approved": False}}
    for filename, payload in payloads.items():
        (stage / filename).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    recheck_review(report)
    destination = output_root / name
    stage.rename(destination)
    return destination


def save_operator_decision(report, output_root, *, operator, reviewed_scenes, accepted, reason=""):
    """Called only after the operator explicitly confirms/rejects in the GUI.

    The name and SHA are a local attestation/integrity seal, not a PKI signature.
    Neither outcome invokes a model or edits a frozen data artifact.
    """
    require(isinstance(operator, str) and operator.strip(), "OPERATOR_SIGNATURE_REQUIRED")
    require(type(accepted) is bool, "EXPLICIT_DECISION_REQUIRED")
    if accepted:
        require(set(reviewed_scenes) == set(report["review_scene_hashes"]), "ALL_REVIEW_SCENES_MUST_BE_CONFIRMED")
    else:
        require(bool(reason.strip()), "REJECTION_REASON_REQUIRED")
    recheck_review(report, allow_geometry_failure=not accepted)
    output_root = _review_output_root(report, output_root)
    decision = {"schema": APPROVAL_SCHEMA, "policy": deepcopy(POLICY), "selection_sha": SELECTION_SHA,
                "gt_freeze_sha": FREEZE_SHA, "review_sha256": report["review_sha256"],
                "decision": "APPROVED" if accepted else "REJECTED", "operator_signature": operator.strip(),
                "signature_kind": "local_operator_attestation_with_sha256_integrity",
                "reviewed_scene_hashes": sorted(set(reviewed_scenes)), "reason": reason.strip(),
                "metric_policy_sha256": METRIC_POLICY_SHA, "metric_counts": deepcopy(report["metric_counts"]),
                "metric_exclusions": deepcopy(report["metric_exclusions"]),
                "approval_scope": "oriented_pixels_and_GT_for_479_bbox_and_478_indexed_pose",
                "signed_at": datetime.now(timezone.utc).isoformat(), "measurement_status": "NOT_RUN"}
    decision["contract_sha256"] = digest(decision)
    output_root.mkdir(parents=True, exist_ok=True)
    name = "EVAL396_MT_geometry_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    stage = output_root / (name + ".partial")
    stage.mkdir()
    for filename, payload in (("preflight.json", report), ("geometry_policy.json", decision),
                              ("metric_eligibility_policy.json", report["metric_policy"])):
        (stage / filename).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    recheck_review(report, allow_geometry_failure=not accepted)
    (stage / "status.json").write_text(json.dumps({"status": "GEOMETRY_APPROVED" if accepted else "FAIL",
        "kind": "operator_geometry_review", "measurement_status": "NOT_RUN"}), encoding="utf-8")
    destination = output_root / name
    stage.rename(destination)
    return destination, decision


def require_current_approval(report, decision):
    """Future MT runner must call this gate after its fresh preflight."""
    recheck_review(report)
    require(decision.get("schema") == APPROVAL_SCHEMA and decision.get("decision") == "APPROVED"
            and bool(str(decision.get("operator_signature") or "").strip()), "EXIF_OPERATOR_APPROVAL_REQUIRED")
    require(decision.get("contract_sha256") == digest({k: v for k, v in decision.items() if k != "contract_sha256"}),
            "OPERATOR_CONTRACT_CHANGED")
    require(decision.get("policy") == POLICY and decision.get("review_sha256") == report["review_sha256"]
            and decision.get("selection_sha") == SELECTION_SHA and decision.get("gt_freeze_sha") == FREEZE_SHA
            and set(decision.get("reviewed_scene_hashes", [])) == set(report["review_scene_hashes"]), "STALE_GEOMETRY_APPROVAL")
    require(decision.get("metric_policy_sha256") == METRIC_POLICY_SHA
            and decision.get("metric_counts") == report["metric_counts"]
            and decision.get("metric_exclusions") == report["metric_exclusions"], "METRIC_POLICY_APPROVAL_MISMATCH")
    return True


def load_operator_decision(report, output_root):
    for path in sorted(Path(output_root).glob("EVAL396_MT_geometry_*/geometry_policy.json"), reverse=True):
        if path.parent.name.endswith(".partial"):
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("review_sha256") != report["review_sha256"]:
            continue
        require(decision.get("schema") == APPROVAL_SCHEMA, "SAVED_OPERATOR_CONTRACT_SCHEMA_CHANGED")
        require(decision.get("contract_sha256") == digest({k: v for k, v in decision.items() if k != "contract_sha256"}),
                "SAVED_OPERATOR_CONTRACT_CHANGED")
        if decision.get("decision") == "APPROVED":
            require_current_approval(report, decision)
        else:
            require(decision.get("decision") == "REJECTED", "UNKNOWN_OPERATOR_DECISION")
        return path.parent, decision
    return None
