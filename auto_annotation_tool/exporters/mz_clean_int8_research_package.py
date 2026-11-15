#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2-I-10 Desktop: reuse ONE frozen MZ CLEAN INT8 into a research-only .alprmodel.

This is deliberately a separate, opt-in engineering step. It never calls
Ultralytics conversion, declares a calibration loader replay, exports other
models, modifies the Z4 artifact library, or approves Android release/final_test.

Usage:
    python -m auto_annotation_tool.exporters.mz_clean_int8_research_package --repo ROOT --assess
    python -m auto_annotation_tool.exporters.mz_clean_int8_research_package --repo ROOT --package
    python -m auto_annotation_tool.exporters.mz_clean_int8_research_package --repo ROOT --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

from .mz_raw_int8_package_readiness import SOURCES, STAGED_REL, STAGED_SHA256, _safe_managed_path
from .mz_raw_int8_live_evidence import inspect_mz_day_live_evidence, ordered_label_digest
from .mobile_model_exporter import (
    MobileExportError, MobileModelExporter, _precision_report_bytes,
    _require_tflite_precision, compact_evidence,
)
from .tflite_precision import analyze_tflite_file

SCHEMA = "alpr.mz_clean_int8_research_package.v1"
NAME = "MZ_DAY_CLEAN_INT8_RESEARCH_ONLY.alprmodel"
DEST_REL = Path("output/mz_day_int8_raw_probe") / NAME
MODEL_MEMBER = "variants/tflite_int8/model.tflite"
REPORT_MEMBER = "precision/tflite-int8.json"
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABELS_SHA256 = "f93925af23277c69c2bac02d0ac546c236a6a7c3a647169ad83b69c103c12e92"
AUDIT_SCHEMA = "alpr.f4.mz_day_clean_live_train_evidence_audit.v1"


class ResearchPackageBlocked(ValueError):
    """Nothing should be published if an evidence binding is incomplete."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blob(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _guard_audit(audit: dict) -> tuple[str, str]:
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ResearchPackageBlocked("LIVE_AUDIT_SCHEMA_MISMATCH")
    if (audit.get("status") != "RESEARCH_EVIDENCE_ONLY_NOT_DEPLOYABLE"
            or audit.get("model_sha256") != SOURCES["clean_raw_tflite"][1]
            or audit.get("checkpoint_sha256") != SOURCES["checkpoint"][1]
            or audit.get("android_import_approved") is not False
            or audit.get("alprmodel_export_approved") is not False
            or audit.get("final_test") != "NOT_VERIFIED"
            or audit.get("conversion_performed") is not False
            or audit.get("packaging_performed") is not False):
        raise ResearchPackageBlocked("LIVE_AUDIT_NOT_RESEARCH_ONLY")
    if (audit.get("class_order") or {}).get("status") != "CHECKPOINT_VERIFIED":
        raise ResearchPackageBlocked("CHECKPOINT_CLASS_ORDER_NOT_VERIFIED")
    actual_labels = (audit["class_order"] or {}).get("ordered_labels")
    if (actual_labels != LABELS or ordered_label_digest(actual_labels) != LABELS_SHA256
            or audit["class_order"].get("order_sha256") != LABELS_SHA256):
        raise ResearchPackageBlocked("CHECKPOINT_CLASS_ORDER_SHA_MISMATCH")
    calibration = audit.get("calibration") or {}
    if calibration.get("status") != "EXPLORATORY_ONLY_REVALIDATED":
        raise ResearchPackageBlocked("TRAIN_CALIBRATION_NOT_REVALIDATED")
    fingerprint = str(calibration.get("input_fingerprint") or "")
    digest = str(calibration.get("sha256") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", fingerprint) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ResearchPackageBlocked("TRAIN_CALIBRATION_INVALID_FINGERPRINT")
    if not calibration.get("manifest"):
        raise ResearchPackageBlocked("TRAIN_CALIBRATION_PATH_MISSING")
    scan = audit.get("manifest_discovery") or {}
    if scan.get("truncated") is True or (scan.get("selection") != "USER_EXPLICIT_LIVE_VALIDATION_REQUIRED"
            and scan.get("selection") != "UNIQUE_PREFILTER_LIVE_VALIDATION_REQUIRED"):
        raise ResearchPackageBlocked("TRAIN_CALIBRATION_BINDING_AMBIGUOUS")
    blockers = audit.get("remaining_blockers")
    if set(blockers or []) != {
        "ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED",
        "ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT",
        "FINAL_TEST_NOT_VERIFIED",
    }:
        raise ResearchPackageBlocked("UNEXPECTED_RESEARCH_READINESS_BLOCKERS")
    return digest, fingerprint


def _safe_calibration_path(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ResearchPackageBlocked("CALIBRATION_PATH_OUTSIDE_REPOSITORY") from exc
    else:
        relative = candidate
    return _safe_managed_path(root, relative)


def make_manifest(*, audit: dict, graph: dict, calibration_sha: str, calibration_fingerprint: str) -> dict:
    """Pure manifest assembly, requires a *live* audit from the caller."""
    _guard_audit(audit)
    if (not isinstance(graph, dict)
            or graph.get("classification") != "INT8"
            or graph.get("full_integer_graph") is not True
            or graph.get("artifact_sha256") != SOURCES["clean_raw_tflite"][1]):
        raise ResearchPackageBlocked("MZ_LIVE_GRAPH_NOT_FULL_INT8")
    input_spec = {
        "width": 320, "height": 320, "channels": 3, "layout": "NHWC",
        "color": "RGB", "data_type": "INT8", "scale": 1.0 / 255.0, "offset": 0.0,
        "quantization": {"scale": 0.003921568859368563, "zero_point": -128},
        "resize_mode": "letterbox", "padding_value": 114, "interpolation": "bilinear",
    }
    output_spec = {
        "decoder": "ultralytics_detect_split_ltrb_scores_int8_v1",
        "output_format": "raw_split_ltrb_scores_v1", "box_format": "ltrb",
        "tensor_layout": "channels_first", "class_count": 36,
        "keypoint_count": 0, "keypoint_dimensions": 0, "has_objectness": False,
        "normalized_coordinates": True, "nms_in_graph": False, "nms_required": False,
        "confidence_threshold": 0.25, "iou_threshold": 0.45,
    }
    _require_tflite_precision(graph, "int8", input_spec)
    report_blob = _precision_report_bytes(graph)
    evidence = dict(
        compact_evidence(graph),
        report_file=REPORT_MEMBER,
        report_sha256=_sha(report_blob),
    )
    refs = {
        "staged_contract": {"file": "research/staged_contract.json", "sha256": STAGED_SHA256},
        "clean_evidence": {"file": "research/clean_evidence.json", "sha256": SOURCES["clean_evidence"][1]},
        "train_parity": {"file": "research/train_parity.json", "sha256": SOURCES["train_parity"][1]},
        "train_calibration": {
            "file": "research/calibration_manifest.json",
            "sha256": calibration_sha,
            "input_fingerprint": calibration_fingerprint,
        },
    }
    return {
        "schema": "alpr.model.v1",
        "model_id": "mz-day-clean-int8-research-only",
        "name": "MZ-DAY CLEAN INT8 — research ONLY, no production import approval",
        "version": "T2-I-10",
        "role": "character", "task": "detect",
        "input": input_spec, "output": output_spec,
        "labels": list(LABELS),
        "variants": [{
            "id": "tflite-int8", "runtime": "tflite", "precision": "int8",
            "file": MODEL_MEMBER,
            "sha256": {MODEL_MEMBER: SOURCES["clean_raw_tflite"][1]},
            "graph_precision": evidence,
        }],
        "source": {
            "checkpoint": SOURCES["checkpoint"][0],
            "checkpoint_sha256": SOURCES["checkpoint"][1],
            "source_kind": "EXISTING_SHA_PINNED_MODEL_NO_CONVERSION",
        },
        "research_mz_int8": {
            "schema": SCHEMA,
            "status": "EXPLORATORY_ONLY",
            "android_import_approved": False,
            "production_deployment_approved": False,
            "final_test": "NOT_VERIFIED",
            "calibration_usage": "NOT_ATTESTED_DURING_REUSE",
            "class_order_sha256": LABELS_SHA256,
            "split_outputs": [[1, 4, 2100], [1, 36, 2100]],
            "backend_scope": "PINNED_SHA_DEBUG_CPU_RESEARCH_ONLY",
            "evidence": refs,
            "limitations": [
                "ANDROID_IMPORT_NOT_APPROVED",
                "FINAL_TEST_NOT_VERIFIED",
                "CALIBRATION_LOADER_NOT_REEXECUTED_DURING_PACKAGING",
                "MZ_SPLIT_FP32_PARITY_NOT_FROZEN",
            ],
        },
        "exporter": {
            "name": "auto_annotation_tool.mz_clean_int8_research_package",
            "provenance": "LIVE_REVALIDATED_TRAIN_ONLY",
            "conversion_performed": False,
        },
    }


def _prepare(repo_root: Path, *, calibration_manifest: Path | None = None, audit_override=None, graph_override=None):
    root = Path(repo_root).resolve()
    audit = (inspect_mz_day_live_evidence(root, calibration_manifest=calibration_manifest)
             if audit_override is None else audit_override)
    expected_cal_sha, fingerprint = _guard_audit(audit)
    source = {}
    for key in ("clean_raw_tflite", "clean_evidence", "train_parity"):
        relative, digest, _zip_name = SOURCES[key]
        path = _safe_managed_path(root, relative)
        if _file_sha(path) != digest:
            raise ResearchPackageBlocked("PINNED_SOURCE_SHA_CHANGED: " + key)
        source[key] = path
    staged = _safe_managed_path(root, STAGED_REL)
    if _file_sha(staged) != STAGED_SHA256:
        raise ResearchPackageBlocked("PINNED_STAGED_CONTRACT_CHANGED")
    cal_path = _safe_calibration_path(root, str(audit["calibration"]["manifest"]))
    if _file_sha(cal_path) != expected_cal_sha:
        raise ResearchPackageBlocked("TRAIN_CALIBRATION_SHA_CHANGED")
    report = (analyze_tflite_file(source["clean_raw_tflite"])
              if graph_override is None else graph_override)
    manifest = make_manifest(audit=audit, graph=report, calibration_sha=expected_cal_sha,
                             calibration_fingerprint=fingerprint)
    return manifest, {
        MODEL_MEMBER: source["clean_raw_tflite"],
        "research/staged_contract.json": staged,
        "research/clean_evidence.json": source["clean_evidence"],
        "research/train_parity.json": source["train_parity"],
        "research/calibration_manifest.json": cal_path,
    }, report


def inspect_read_only(repo_root: Path, *, calibration_manifest: Path | None = None) -> dict:
    manifest, members, _graph = _prepare(repo_root, calibration_manifest=calibration_manifest)
    return {
        "status": "RESEARCH_PACKAGE_PRECHECK_PASS_NOT_PRODUCTION_APPROVAL",
        "manifest_schema": manifest["schema"],
        "model_sha256": SOURCES["clean_raw_tflite"][1],
        "class_order_sha256": LABELS_SHA256,
        "calibration_sha256": manifest["research_mz_int8"]["evidence"]["train_calibration"]["sha256"],
        "sources": list(members),
        "output_destination": str(_safe_managed_path(Path(repo_root).resolve(), DEST_REL, exists=False)),
        "conversion_performed": False,
        "android_import_approved": False,
        "final_test": "NOT_VERIFIED",
    }


def package_existing_research_model(repo_root: Path, *, calibration_manifest: Path | None = None,
                                    validator=None, audit_override=None, graph_override=None) -> dict:
    """Explicit opt-in research publication. Never overwrite an existing archive."""
    root = Path(repo_root).resolve()
    destination = _safe_managed_path(root, DEST_REL, exists=False)
    if destination.exists() or destination.is_symlink():
        raise ResearchPackageBlocked("RESEARCH_ALPRMODEL_ALREADY_EXISTS_NO_OVERWRITE")
    if not destination.parent.is_dir():
        raise ResearchPackageBlocked("RESEARCH_ALPRMODEL_PARENT_MISSING")
    manifest, members, report = _prepare(root, calibration_manifest=calibration_manifest,
                                         audit_override=audit_override, graph_override=graph_override)
    exporter = MobileModelExporter() if validator is None else validator
    hashes = {name: _file_sha(path) for name, path in members.items()}
    with tempfile.TemporaryDirectory(prefix=".mz_day_clean_t2i10_", dir=destination.parent) as temporary:
        temp = Path(temporary) / NAME
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=False) as archive:
            archive.writestr("manifest.json", _blob(manifest))
            archive.writestr(REPORT_MEMBER, _precision_report_bytes(report))
            for name, source in sorted(members.items()):
                archive.write(source, name)
        # Reread all bytes, verify member set, and independently validate graph.
        with zipfile.ZipFile(temp, "r") as archive:
            expected = set(members) | {"manifest.json", REPORT_MEMBER}
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ResearchPackageBlocked("RESEARCH_PACKAGE_MEMBER_SET_INCOMPLETE")
            for name, digest in hashes.items():
                if _sha(archive.read(name)) != digest:
                    raise ResearchPackageBlocked("RESEARCH_PACKAGE_MEMBER_SHA_MISMATCH: " + name)
        exporter.validate_package(temp, require_precision=True)
        for name, source in members.items():
            if _file_sha(source) != hashes[name]:
                raise ResearchPackageBlocked("RESEARCH_SOURCE_CHANGED_DURING_PACKAGING: " + name)
        try:
            # Link published atomically with NO_REPLACE; never silently overwrite.
            os.link(temp, destination)
        except FileExistsError as exc:
            raise ResearchPackageBlocked("RESEARCH_PACKAGE_DESTINATION_RACE") from exc
        except OSError as exc:
            raise ResearchPackageBlocked("RESEARCH_PACKAGE_SAFE_PUBLICATION_FAILED: " + str(exc)) from exc
    return {
        "status": "RESEARCH_ALPRMODEL_PACKAGED_NOT_ANDROID_APPROVED",
        "path": str(destination), "sha256": _file_sha(destination),
        "source_tflite_sha256": hashes[MODEL_MEMBER],
        "calibration_usage": "NOT_ATTESTED_DURING_REUSE",
        "conversion_performed": False,
        "final_test": "NOT_VERIFIED",
        "android_import_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--assess", action="store_true", help="read only; revalidate pinned graph and TRAIN evidence")
    choice.add_argument("--package", action="store_true", help="explicit research-only .alprmodel creation")
    choice.add_argument("--verify", action="store_true", help="read-only package verification")
    args = parser.parse_args(argv)
    try:
        if args.assess:
            result = inspect_read_only(args.repo, calibration_manifest=args.calibration_manifest)
        elif args.package:
            result = package_existing_research_model(args.repo, calibration_manifest=args.calibration_manifest)
        else:
            dest = _safe_managed_path(Path(args.repo).resolve(), DEST_REL)
            MobileModelExporter().validate_package(dest, require_precision=True)
            result = {"status": "RESEARCH_ALPRMODEL_VERIFIED_NOT_ANDROID_APPROVED",
                      "path": str(dest), "sha256": _file_sha(dest),
                      "android_import_approved": False, "final_test": "NOT_VERIFIED"}
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, MobileExportError) as exc:
        print(f"T2I10_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
