#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only forensic bridge for an EXISTING MZ-DAY split RAW signed-INT8 TFLite.

Never exports a .alprmodel, never calls Ultralytics conversion, never changes the
existing MZ_RAW_INT8_CONTRACT_NOT_READY safety barrier.  The generated JSON is
research-only staging metadata for the eventual dual-output Android contract.

The model, its full-graph static report, the six-case CPU parity evidence, the
12 TRAIN-image report and source checkpoint MUST all agree under pinned SHA256.
No final_test or EVAL396 samples are inspected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from .tflite_precision import analyze_tflite_file

RESEARCH_SCHEMA = "alpr.mz_day_raw_int8_research_staging.v1"
MODEL_SHA = "a2f715399bdcb574da899e202388b5324d66f574b8519626cb611bf0e263e792"
CLEAN_EVIDENCE_SHA = "bdf93990c1f971f571b281b73c02882ebf84adf680050ec31278ef66130678c3"
TRAIN_PARITY_SHA = "f2f1b89ec18fd9d71c40c1dbd280bfcef9ea32f3c78d7faa4be9ba8d53976b45"
CHECKPOINT_SHA = "631bb1453554cdb6b15eca724baa392e2acffb350899107f892dc11f937dbbf8"
ORIGINAL_RAW_SHA = "2db549d7ff1e60b7448f66e354343f238476c609f5c51ad80c11165f10ce47ea"
REJECTED_MIXED_SHA = "3abac57a1549c969ca1a2b239eed5e0b945bf21dc9f32ae691f74d63b4be2c13"
BASE = Path("output/mz_day_int8_raw_probe")
MODEL_REL = BASE / "MZ_DAY_RAW_INT8_CLEAN_EXPLORATORY.tflite"
EVIDENCE_REL = BASE / "MZ_DAY_RAW_INT8_CLEAN_EVIDENCE.json"
TRAIN_REPORT_REL = BASE / "MZ_DAY_CLEAN_RAW_TRAIN_PARITY_VS_PT_20261013.json"
CHECKPOINT_REL = Path("Workspace/5_training_runs/chars/20261007_215302/train/weights/best.pt")
STAGED_REL = BASE / "MZ_DAY_RAW_INT8_T2B_CONTRACT_PROVISIONAL.json"


class ResearchContractNotReady(ValueError):
    """Research input mismatch. No conversion or packaging is attempted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _sha_guard(path: Path, expected_sha: str, label: str) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ResearchContractNotReady(f"{label}: missing file or symlink: {path}")
    found = sha256_file(path)
    if found != expected_sha:
        raise ResearchContractNotReady(f"{label}: SHA256 mismatch ({found} != {expected_sha})")


def _read_pinned_json(path: Path, expected_sha: str, label: str) -> dict:
    _sha_guard(path, expected_sha, label)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResearchContractNotReady(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchContractNotReady(f"{label}: JSON must be an object")
    return payload


def _expect_field(payload: dict, key: str, expected, source: str) -> None:
    if payload.get(key) != expected:
        raise ResearchContractNotReady(f"{source}: invalid {key}: {payload.get(key)!r} != {expected!r}")


def _validate_evidence(evidence: dict) -> None:
    for key, expected in {
        "source_sha256": ORIGINAL_RAW_SHA,
        "candidate_sha256": MODEL_SHA,
        "precision_after": "INT8",
        "precision_before": "UNVERIFIED",
        "full_integer_graph": True,
        "issues_after": [],
        "number_of_operators": 388,
        "number_of_tensors_after": 647,
        "number_of_tensors_before": 649,
        "model_scope": "EXPLORATORY_ONLY",
        "final_test": "NOT_VERIFIED",
        "android_execution": "UNVERIFIED",
        "mobile_decoder": "NOT_IMPLEMENTED",
        "new_io": {"input": [0], "outputs": [646, 634]},
    }.items():
        _expect_field(evidence, key, expected, "CLEAN_EVIDENCE")
    cpu = evidence.get("cpu_parity")
    if not isinstance(cpu, dict):
        raise ResearchContractNotReady("CLEAN_EVIDENCE: missing CPU parity")
    for key, expected in {
        "status": "CPU_EXACT_6_CASES",
        "values_compared": 504000,
        "android_runtime": "UNVERIFIED",
        "model_quality": "NOT_VERIFIED",
        "final_test": "NOT_RUN",
    }.items():
        _expect_field(cpu, key, expected, "CLEAN_EVIDENCE.cpu_parity")
    cases = cpu.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ResearchContractNotReady("CLEAN_EVIDENCE: expected six CPU inputs")
    for i, case in enumerate(cases):
        outputs = case.get("outputs") if isinstance(case, dict) else None
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise ResearchContractNotReady(f"CLEAN_EVIDENCE: invalid case {i} outputs")
        for out, kind, shape, count in zip(outputs, ("geometry", "scores"), ([1,4,2100], [1,36,2100]), (8400, 75600)):
            if (out.get("kind") != kind or out.get("shape") != shape or
                    out.get("mismatches") != 0 or out.get("values_compared") != count):
                raise ResearchContractNotReady(f"CLEAN_EVIDENCE: case {i} not exact")


def _validate_train_report(report: dict) -> dict:
    for key, expected in {
        "schema": "alpr.f4.mz_clean_signed_int8_vs_original_raw_and_pt_train.v1",
        "status": "TRAIN_CLEAN_VS_ORIGINAL_RAW_EXACT_AND_PT_DIAGNOSTIC_ONLY",
        "scope": "TRAIN_ONLY_DIAGNOSTIC_NOT_FINAL_TEST",
        "source_mixed_sha256": REJECTED_MIXED_SHA,
        "raw_original_sha256": ORIGINAL_RAW_SHA,
        "clean_int8_sha256": MODEL_SHA,
        "clean_evidence_sha256": CLEAN_EVIDENCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "images": 12,
        "raw_output_parity_exact_cases": 12,
        "raw_output_parity_exact_tensors": 24,
        "score_threshold": 0.25,
        "iou_match_threshold": 0.5,
        "execution_precision_android": "UNVERIFIED",
        "android_decoder": "NOT_IMPLEMENTED",
        "model_scope": "EXPLORATORY_ONLY",
        "final_test": "NOT_RUN",
        "quality": "NOT_CERTIFIED",
    }.items():
        _expect_field(report, key, expected, "TRAIN_REPORT")
    decoder = str(report.get("decoder", ""))
    preprocess = str(report.get("preprocess", ""))
    if "topk k=300" not in decoder or "no NMS" not in decoder or "LTRB" not in decoder:
        raise ResearchContractNotReady("TRAIN_REPORT: unexpected decode diagnostic")
    if "RGB letterbox 320" not in preprocess or "padding 114" not in preprocess:
        raise ResearchContractNotReady("TRAIN_REPORT: unexpected preprocessing")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ResearchContractNotReady("TRAIN_REPORT: need exactly 12 TRAIN cases")
    all_images = set()
    counts = {"pt_above_conf": 0, "raw_above_conf": 0, "matches_same_class_iou_ge_050": 0}
    differences: list[dict] = []
    ties = []
    weighted_iou_sum = 0.0
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ResearchContractNotReady(f"TRAIN_REPORT: invalid case {i}")
        image_path = str(case.get("image_train_split_relative") or "").replace("\\", "/")
        if not image_path.startswith("images/train/") or ".." in Path(image_path).parts or image_path in all_images:
            raise ResearchContractNotReady(f"TRAIN_REPORT: non-TRAIN or duplicated image {i}: {image_path}")
        all_images.add(image_path)
        if (case.get("two_output_exact_quantized_equality") is not True or
                case.get("clean_output_sha256") != case.get("original_raw_output_sha256") or
                case.get("output_mismatches") != [0,0]):
            raise ResearchContractNotReady(f"TRAIN_REPORT: RAW<->CLEAN parity not exact: {image_path}")
        item = case.get("matching")
        topk = case.get("raw_selection")
        if not isinstance(item, dict) or not isinstance(topk, dict):
            raise ResearchContractNotReady(f"TRAIN_REPORT: missing metrics {i}")
        vals = {}
        for k in counts:
            raw = item.get(k)
            if type(raw) is not int or raw < 0:
                raise ResearchContractNotReady(f"TRAIN_REPORT: invalid count {k}: {image_path}")
            counts[k] += raw
            vals[k] = raw
        if vals["matches_same_class_iou_ge_050"] > min(vals["pt_above_conf"], vals["raw_above_conf"]):
            raise ResearchContractNotReady(f"TRAIN_REPORT: impossible matches: {image_path}")
        iou = item.get("mean_iou_of_matched")
        if type(iou) not in (float, int) or not 0.0 <= iou <= 1.0:
            raise ResearchContractNotReady(f"TRAIN_REPORT: invalid IoU: {image_path}")
        weighted_iou_sum += float(iou) * vals["matches_same_class_iou_ge_050"]
        tie_count = topk.get("quantized_score_threshold_ties")
        if type(tie_count) is not int or tie_count < 0:
            raise ResearchContractNotReady(f"TRAIN_REPORT: invalid ties count: {image_path}")
        ties.append(tie_count)
        if topk.get("geometry") != "normalized_distance_ltrb_plus_analytic_grid_to_xyxy":
            raise ResearchContractNotReady(f"TRAIN_REPORT: unexpected LTRB grid: {image_path}")
        if vals["pt_above_conf"] != vals["raw_above_conf"]:
            differences.append({"image_train_relative": image_path,
                                "pytorch_count": vals["pt_above_conf"],
                                "clean_int8_count": vals["raw_above_conf"],
                                "matched_to_pt": vals["matches_same_class_iou_ge_050"]})
    if counts != {"pt_above_conf": 88, "raw_above_conf": 90, "matches_same_class_iou_ge_050": 87}:
        raise ResearchContractNotReady(f"TRAIN_REPORT: unsupported aggregate: {counts}")
    if report.get("totals") != counts or len(differences) != 4:
        raise ResearchContractNotReady("TRAIN_REPORT: unexpected totals or mismatch count")
    if not math.isclose(float(report.get("pt_recovery_fraction", 0)), 87 / 88, abs_tol=1e-12):
        raise ResearchContractNotReady("TRAIN_REPORT: inconsistent PT recovery fraction")
    return {"totals": counts, "difference_images": differences,
            "weighted_matched_iou": weighted_iou_sum / 87,
            "topk_score_ties_range": [min(ties), max(ties)]}


def _validate_static_graph(graph: dict) -> dict:
    if (graph.get("artifact_sha256") != MODEL_SHA or graph.get("classification") != "INT8"
            or graph.get("full_integer_graph") is not True or graph.get("issues")
            or len(graph.get("subgraphs") or []) != 1):
        raise ResearchContractNotReady("MZ_CLEAN: strict static full-graph INT8 proof failed")
    sg = graph["subgraphs"][0]
    if len(sg.get("operators") or []) != 388 or len(sg.get("tensors") or []) != 647:
        raise ResearchContractNotReady("MZ_CLEAN: unexpected operator/tensor counts")
    if graph.get("tensor_type_counts") != {"INT32": 144, "INT64": 10, "INT8": 493}:
        raise ResearchContractNotReady("MZ_CLEAN: unexpected tensor types after cleanup")
    ports = [sg.get("inputs") or [], sg.get("outputs") or []]
    if len(ports[0]) != 1 or len(ports[1]) != 2:
        raise ResearchContractNotReady("MZ_CLEAN: wrong number of I/O ports")
    specs = (
        (ports[0][0], 0, [1,320,320,3], 0.003921568859368563, -128),
        (ports[1][0], 646, [1,4,2100], 0.006233580876141787, -126),
        (ports[1][1], 634, [1,36,2100], 0.00390625, -128),
    )
    for tensor, index, shape, scale, zero in specs:
        if (tensor.get("index") != index or tensor.get("shape") != shape
                or tensor.get("dtype") != "INT8" or tensor.get("zero_point") != [zero]
                or len(tensor.get("scale") or []) != 1
                or not math.isclose(tensor["scale"][0], scale, rel_tol=1e-6, abs_tol=1e-12)):
            raise ResearchContractNotReady(f"MZ_CLEAN: port #{index} shape/dtype/quantizer mismatch")
    return {
        "input": {"port": 0, "tensor_index": 0, "shape": [1,320,320,3],
                  "data_type": "INT8", "quantization": {"scale": specs[0][3], "zero_point": -128}},
        "outputs": [
            {"port": 0, "tensor_index": 646, "role": "ltrb_normalized_distances_after_dfl",
             "shape": [1,4,2100], "data_type": "INT8", "quantization": {"scale": specs[1][3], "zero_point": -126}},
            {"port": 1, "tensor_index": 634, "role": "sigmoid_class_scores",
             "shape": [1,36,2100], "data_type": "INT8", "quantization": {"scale": specs[2][3], "zero_point": -128}},
        ]}


def inspect_mz_day_research_candidate(*, model: Path, evidence: Path,
                                      train_parity: Path, checkpoint: Path) -> dict:
    """Validate pinned files and strict static graph, returning research-only contract.

    This NEVER creates a model package, never re-exports YOLO and does not
    authorize MobileAlprPackageExporter.  Calibration must be revalidated
    separately before an Android-compatible export can ever be enabled.
    """
    _sha_guard(model, MODEL_SHA, "MZ_RAW_CLEAN")
    _sha_guard(checkpoint, CHECKPOINT_SHA, "MZ_CHECKPOINT")
    evidence_obj = _read_pinned_json(evidence, CLEAN_EVIDENCE_SHA, "CLEAN_EVIDENCE")
    report_obj = _read_pinned_json(train_parity, TRAIN_PARITY_SHA, "TRAIN_PARITY")
    _validate_evidence(evidence_obj)
    summary = _validate_train_report(report_obj)
    graph = analyze_tflite_file(Path(model))
    io = _validate_static_graph(graph)
    return {
        "schema": RESEARCH_SCHEMA,
        "status": "EXPLORATORY_ONLY",
        "contract_state": "PROVISIONAL_T2B_NO_ANDROID_DECODER",
        "full_integer_graph": True,
        "graph_static_precision": "INT8",
        "execution_precision_android": "UNVERIFIED",
        "android_decoder": "NOT_IMPLEMENTED",
        "android_import_approved": False,
        "alprmodel_export_approved": False,
        "final_test": "NOT_VERIFIED",
        "accuracy_vs_ground_truth": "NOT_MEASURED_HERE",
        "sampling_scope": "TRAIN_ONLY_DIAGNOSTIC_NOT_FINAL_TEST",
        "sources": {
            "clean_raw_tflite": {"path": MODEL_REL.as_posix(), "sha256": MODEL_SHA},
            "clean_evidence": {"path": EVIDENCE_REL.as_posix(), "sha256": CLEAN_EVIDENCE_SHA},
            "train_parity": {"path": TRAIN_REPORT_REL.as_posix(), "sha256": TRAIN_PARITY_SHA},
            "checkpoint": {"path": CHECKPOINT_REL.as_posix(), "sha256": CHECKPOINT_SHA},
            "original_raw_sha256": ORIGINAL_RAW_SHA,
            "rejected_mixed_sha256": REJECTED_MIXED_SHA,
        },
        "graph": {"operator_count": 388, "tensor_count": 647,
                  "analysis_sha256": graph.get("analysis_sha256"),
                  "tensor_type_counts": graph["tensor_type_counts"]},
        "draft_output_contract": {
            "format": "raw_split_ltrb_scores_v1",
            "decoder": "yolo26_detect_ltrb_scores_topk_v1",
            "approval": "DRAFT_NOT_ANDROID_COMPATIBLE",
            "io": io,
            "number_of_classes": 36,
            "anchor_grid": {"levels": [
                {"side": 40, "stride": 8, "count": 1600},
                {"side": 20, "stride": 16, "count": 400},
                {"side": 10, "stride": 32, "count": 100},
            ], "order": "P3_P4_P5_y_major_x_fastest", "center_formula": "(x+0.5)/side,(y+0.5)/side"},
            "boxes": "x1=cx-left;y1=cy-top;x2=cx+right;y2=cy+bottom;normalized;after_DFL",
            "scores": "sigmoid_completed;36_classes_per_anchor;no_objectness",
            "topk_reference": {"stage1_anchors": 300, "stage2_anchor_class_pairs": 300,
                               "ties": "numpy_argsort_desc_stable_in_train_diagnostic_ONLY",
                               "compatibility_with_pytorch": "NOT_FROZEN"},
            "nms_in_graph": False, "nms_policy_vs_fp32": "NOT_FROZEN",
            "letterbox_preprocess_train_diagnostic": "RGB_PIL_bilinear_320_pad114",
            "labels_class_order": "REQUIRES_CHECKPOINT_OR_APPROVED_MANIFEST",
            "calibration_manifest": "REQUIRES_LIVE_REVALIDATION_BEFORE_PACKAGE",
        },
        "train_evidence": {"raw_clean_exact": "24_OF_24_TENSORS",
                           "pt_reference": summary["totals"],
                           "difference_images": summary["difference_images"],
                           "matched_iou_mean_weighted": summary["weighted_matched_iou"],
                           "topk_score_ties_range": summary["topk_score_ties_range"]},
        "limitations": ["FINAL_TEST_NOT_VERIFIED", "ANDROID_RUNTIME_UNVERIFIED",
                        "RAW_DUAL_OUTPUT_DECODER_NOT_IMPLEMENTED", "TOPK_NMS_FP32_PARITY_NOT_FROZEN",
                        "CALIBRATION_MANIFEST_NOT_REVALIDATED_BY_STAGING"],
    }


def _contract_bytes(contract: dict) -> bytes:
    return (json.dumps(contract, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write_research_contract_only(destination: Path, contract: dict) -> str:
    """Write a plain JSON research sidecar only; fail-closed on nonmatching output."""
    dest = Path(destination)
    if dest.suffix.lower() != ".json" or dest.name != STAGED_REL.name:
        raise ResearchContractNotReady("Only the exact research JSON sidecar name is allowed")
    blob = _contract_bytes(contract)
    if dest.exists() or dest.is_symlink():
        if dest.is_file() and not dest.is_symlink() and dest.read_bytes() == blob:
            return "EXISTING_IDENTICAL_RESEARCH_JSON"
        raise ResearchContractNotReady(f"research contract already exists but differs: {dest}")
    if not dest.parent.is_dir() or dest.parent.is_symlink():
        raise ResearchContractNotReady(f"research output directory is missing or a symlink: {dest.parent}")
    temp = dest.parent / f".{dest.name}.writing-{os.getpid()}"
    if temp.exists() or temp.is_symlink():
        raise ResearchContractNotReady(f"staging path occupied: {temp}")
    try:
        with temp.open("xb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        if dest.exists() or dest.is_symlink():
            raise ResearchContractNotReady("output appeared during staging")
        temp.replace(dest)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
    return "RESEARCH_JSON_STAGED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only MZ RAW INT8 contract staging; no package export")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Read-only T2B inspection")
    group.add_argument("--stage", action="store_true", help="Write a JSON research contract only")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    contract = inspect_mz_day_research_candidate(
        model=root / MODEL_REL,
        evidence=root / EVIDENCE_REL,
        train_parity=root / TRAIN_REPORT_REL,
        checkpoint=root / CHECKPOINT_REL,
    )
    print("MZ_STATIC_FULL_INT8_PASS: 388 operators; two INT8 outputs")
    print("MZ_TRAIN_PARITY_PASS: 24/24 tensors exact; PT 87/88 matched; not GT accuracy")
    print("STATUS: EXPLORATORY_ONLY; ANDROID_UNVERIFIED; FINAL_TEST_NOT_VERIFIED")
    if args.check:
        print("T2B_RESEARCH_CONTRACT_CHECK_PASS_READ_ONLY; no files written")
        print("NEXT: --stage writes only a JSON sidecar; no .alprmodel")
    else:
        result = _write_research_contract_only(root / STAGED_REL, contract)
        print(result + ":", root / STAGED_REL)
        print("NOT_AN_ANDROID_PACKAGE; NO_CONVERSION_NO_TRAINING_NO_GIT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"T2B_STAGE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
