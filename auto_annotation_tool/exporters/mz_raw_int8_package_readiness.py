#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB025: evidence-pinned MZ-DAY CLEAN INT8 research packaging gate.

This is intentionally *not* an alpr.model.v1 / Android deployment exporter.
A two-output split-RAW model cannot be labelled deployable while the Android
import/decoder contract and live calibration remain unapproved.  The ordinary
MZ_RAW_INT8_CONTRACT_NOT_READY barrier is never changed by this module.

The optional .alprresearch ZIP is an archive of already existing evidence, NOT
an .alprmodel, and never modifies/reconverts the embedded TFLite model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import uuid
import zipfile

# The contract JSON and source bytes supplied for the 2026-10-14 review.
STAGED_SHA256 = "610775c88d60e0c6798f49b9575ce75e320b02f48a2e34b8e2d23af5003cef56"
RESEARCH_SCHEMA = "alpr.mz_day_raw_int8_research_staging.v1"
STAGED_REL = Path("output/mz_day_int8_raw_probe/MZ_DAY_RAW_INT8_T2B_CONTRACT_PROVISIONAL.json")
BUNDLE_REL = Path("output/mz_day_int8_raw_probe/MZ_DAY_CLEAN_INT8_RESEARCH_ONLY.alprresearch")
SOURCES = {
    "clean_raw_tflite": (
        "output/mz_day_int8_raw_probe/MZ_DAY_RAW_INT8_CLEAN_EXPLORATORY.tflite",
        "a2f715399bdcb574da899e202388b5324d66f574b8519626cb611bf0e263e792",
        "model.tflite",
    ),
    "clean_evidence": (
        "output/mz_day_int8_raw_probe/MZ_DAY_RAW_INT8_CLEAN_EVIDENCE.json",
        "bdf93990c1f971f571b281b73c02882ebf84adf680050ec31278ef66130678c3",
        "clean_evidence.json",
    ),
    "train_parity": (
        "output/mz_day_int8_raw_probe/MZ_DAY_CLEAN_RAW_TRAIN_PARITY_VS_PT_20261013.json",
        "f2f1b89ec18fd9d71c40c1dbd280bfcef9ea32f3c78d7faa4be9ba8d53976b45",
        "train_parity.json",
    ),
    "checkpoint": (
        "Workspace/5_training_runs/chars/20261007_215302/train/weights/best.pt",
        "631bb1453554cdb6b15eca724baa392e2acffb350899107f892dc11f937dbbf8",
        None,
    ),
}


class MzPackagingNotReady(ValueError):
    """Strict research verification failed; never infer approval."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_managed_path(root: Path, relative: Path | str, *, exists: bool = True) -> Path:
    root = Path(root).absolute()
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in ("..", "") for part in path.parts):
        raise MzPackagingNotReady(f"UNSAFE_RELATIVE_PATH: {relative}")
    current = root
    for component in path.parts:
        current = current / component
        # Check before .resolve(), otherwise an on-disk link hides itself.
        junction = getattr(current, "is_junction", None)
        if current.is_symlink() or (callable(junction) and junction()):
            raise MzPackagingNotReady(f"REDIRECTED_PATH: {current}")
    if exists and (not current.is_file() or current.is_symlink()):
        raise MzPackagingNotReady(f"SOURCE_MISSING: {current}")
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise MzPackagingNotReady(f"SOURCE_OUTSIDE_REPOSITORY: {current}") from exc
    return current


def _verified_sources(root: Path, contract: dict) -> dict[str, Path]:
    if not isinstance(contract, dict) or contract.get("schema") != RESEARCH_SCHEMA:
        raise MzPackagingNotReady("MZ_STAGE_SCHEMA_MISMATCH")
    staged_sources = contract.get("sources")
    if not isinstance(staged_sources, dict):
        raise MzPackagingNotReady("MZ_STAGE_MISSING_SOURCES")
    paths: dict[str, Path] = {}
    for name, (expected_rel, expected_sha, _) in SOURCES.items():
        described = staged_sources.get(name)
        if not isinstance(described, dict) or described.get("path") != expected_rel or described.get("sha256") != expected_sha:
            raise MzPackagingNotReady(f"MZ_SOURCE_CONTRACT_MISMATCH: {name}")
        path = _safe_managed_path(root, expected_rel)
        actual = _hash_file(path)
        if actual != expected_sha:
            raise MzPackagingNotReady(f"MZ_SOURCE_SHA_MISMATCH: {name}: {actual}")
        paths[name] = path
    return paths


def _validate_staged_flags(contract: dict) -> None:
    required = {
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
    }
    for field, expected in required.items():
        value = contract.get(field)
        if type(value) is not type(expected) or value != expected:
            raise MzPackagingNotReady(f"MZ_STAGE_FIELD_MISMATCH: {field}")
    output = contract.get("draft_output_contract") or {}
    io = output.get("io") or {}
    ports = io.get("outputs") or []
    if (output.get("format") != "raw_split_ltrb_scores_v1"
            or output.get("number_of_classes") != 36
            or output.get("approval") != "DRAFT_NOT_ANDROID_COMPATIBLE"
            or output.get("decoder") != "yolo26_detect_ltrb_scores_topk_v1"
            or output.get("labels_class_order") != "REQUIRES_CHECKPOINT_OR_APPROVED_MANIFEST"
            or output.get("calibration_manifest") != "REQUIRES_LIVE_REVALIDATION_BEFORE_PACKAGE"
            or not isinstance(ports, list) or len(ports) != 2):
        raise MzPackagingNotReady("MZ_STAGE_DUAL_OUTPUT_CONTRACT_MISMATCH")
    expected_outputs = (
        (0, 646, [1, 4, 2100], "ltrb_normalized_distances_after_dfl", 0.006233580876141787, -126),
        (1, 634, [1, 36, 2100], "sigmoid_class_scores", 0.00390625, -128),
    )
    for value, (port, index, shape, role, scale, zero) in zip(ports, expected_outputs):
        if not isinstance(value, dict) or any((
            value.get("port") != port,
            value.get("tensor_index") != index,
            value.get("shape") != shape,
            value.get("role") != role,
            value.get("data_type") != "INT8",
            (value.get("quantization") or {}).get("zero_point") != zero,
        )):
            raise MzPackagingNotReady(f"MZ_STAGE_OUTPUT_MISMATCH: port {port}")
        actual_scale = (value.get("quantization") or {}).get("scale")
        if not isinstance(actual_scale, (int, float)) or abs(float(actual_scale) - scale) > 1e-10:
            raise MzPackagingNotReady(f"MZ_STAGE_QUANTIZATION_MISMATCH: port {port}")
    inp = io.get("input") or {}
    if (inp.get("shape") != [1, 320, 320, 3] or inp.get("data_type") != "INT8"
            or inp.get("tensor_index") != 0 or inp.get("port") != 0):
        raise MzPackagingNotReady("MZ_STAGE_INPUT_MISMATCH")
    if (contract.get("graph") or {}).get("operator_count") != 388:
        raise MzPackagingNotReady("MZ_STAGE_GRAPH_COUNT_MISMATCH")
    if (contract.get("graph") or {}).get("tensor_count") != 647:
        raise MzPackagingNotReady("MZ_STAGE_TENSOR_COUNT_MISMATCH")


def assess_research_mz_int8(
    repo_root: Path,
    *,
    calibration_manifest: Path | None = None,
    verify_checkpoint_labels: bool = False,
    _research_inspector=None,
    _calibration_validator=None,
    _labels_reader=None,
) -> dict:
    """Read-only pinned artifact audit, never an Android package approval.

    Optional collaborators are for isolated regression tests. The production
    default always runs mz_raw_int8_reuse.inspect_mz_day_research_candidate.
    """
    root = Path(repo_root).resolve()
    staged_path = _safe_managed_path(root, STAGED_REL)
    staged_sha = _hash_file(staged_path)
    if staged_sha != STAGED_SHA256:
        raise MzPackagingNotReady(f"MZ_STAGE_SHA_MISMATCH: {staged_sha}")
    if staged_path.stat().st_size > 2 * 1024 * 1024:
        raise MzPackagingNotReady("MZ_STAGE_OVERSIZED")
    try:
        staged = json.loads(staged_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MzPackagingNotReady(f"MZ_STAGE_INVALID_JSON: {exc}") from exc
    if not isinstance(staged, dict):
        raise MzPackagingNotReady("MZ_STAGE_NOT_OBJECT")
    _validate_staged_flags(staged)
    sources = _verified_sources(root, staged)

    if _research_inspector is None:
        from .mz_raw_int8_reuse import inspect_mz_day_research_candidate
        _research_inspector = inspect_mz_day_research_candidate
    live = _research_inspector(
        model=sources["clean_raw_tflite"],
        evidence=sources["clean_evidence"],
        train_parity=sources["train_parity"],
        checkpoint=sources["checkpoint"],
    )
    if not isinstance(live, dict) or live != staged:
        raise MzPackagingNotReady("MZ_LIVE_EVIDENCE_DIFFERS_FROM_PINNED_STAGE")

    blockers = [
        "ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED",
        "ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT",
        "FINAL_TEST_NOT_VERIFIED",
    ]
    calibration = {"status": "NOT_REVALIDATED"}
    if calibration_manifest is None:
        blockers.append("LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED")
    else:
        path = Path(calibration_manifest)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(root)
            except ValueError as exc:
                raise MzPackagingNotReady("CALIBRATION_OUTSIDE_REPOSITORY") from exc
        else:
            relative = path
        path = _safe_managed_path(root, relative)
        if _calibration_validator is None:
            from . import exploratory_calibration
            _calibration_validator = exploratory_calibration.validate_manifest
        try:
            validated = _calibration_validator(
                path, role="character", checkpoint=sources["checkpoint"],
                image_size=320, confidence_threshold=0.25, iou_threshold=0.45,
            )
        except Exception as exc:
            raise MzPackagingNotReady(f"LIVE_CALIBRATION_VALIDATION_FAILED: {exc}") from exc
        if not isinstance(validated, dict) or validated.get("status") != "EXPLORATORY_ONLY" or not re.fullmatch(
            r"[0-9a-f]{64}", str(validated.get("input_fingerprint") or "")
        ):
            raise MzPackagingNotReady("LIVE_CALIBRATION_EVIDENCE_INCOMPLETE")
        calibration = {
            "status": "EXPLORATORY_ONLY_REVALIDATED",
            "manifest": str(path),
            "sha256": _hash_file(path),
            "input_fingerprint": validated["input_fingerprint"],
        }

    labels = {"status": "NOT_VERIFIED"}
    if verify_checkpoint_labels:
        if _labels_reader is None:
            from .mobile_model_exporter import MobileModelExporter
            def _labels_reader(checkpoint):
                exporter = MobileModelExporter()
                model = exporter._load_yolo_model(checkpoint)
                return exporter._labels_from_model(model)
        try:
            names = _labels_reader(sources["checkpoint"])
        except Exception as exc:
            raise MzPackagingNotReady(f"CHECKPOINT_LABELS_UNREADABLE: {exc}") from exc
        if (not isinstance(names, (list, tuple)) or len(names) != 36
                or any(not isinstance(value, str) or not value.strip() for value in names)
                or len(set(names)) != 36):
            raise MzPackagingNotReady("CHECKPOINT_36_CLASS_ORDER_NOT_VERIFIED")
        labels = {"status": "CHECKPOINT_VERIFIED", "ordered_labels": list(names)}
    else:
        blockers.append("CHECKPOINT_36_CLASS_ORDER_NOT_VERIFIED")

    # Detect unexpected source modification during a potentially lengthy graph scan.
    if _hash_file(staged_path) != STAGED_SHA256:
        raise MzPackagingNotReady("MZ_STAGE_CHANGED_DURING_INSPECTION")
    for name, path in sources.items():
        if _hash_file(path) != SOURCES[name][1]:
            raise MzPackagingNotReady(f"MZ_SOURCE_CHANGED_DURING_INSPECTION: {name}")
    return {
        "schema": "alpr.f4.mz_raw_int8_packaging_readiness.v1",
        "status": "BLOCKED_NOT_ANDROID_DEPLOYABLE",
        "research_artifact_integrity": "PASS_PINNED_SHA_AND_LIVE_GRAPH_EVIDENCE",
        "model_sha256": SOURCES["clean_raw_tflite"][1],
        "checkpoint_sha256": SOURCES["checkpoint"][1],
        "split_output": "raw_split_ltrb_scores_v1",
        "input_shape": [1, 320, 320, 3],
        "output_shapes": [[1, 4, 2100], [1, 36, 2100]],
        "calibration": calibration,
        "class_order": labels,
        "blockers": blockers,
        "alprmodel_export_approved": False,
        "android_import_approved": False,
        "final_test": "NOT_VERIFIED",
        "conversion_performed": False,
        "packaging_performed": False,
    }


def stage_research_bundle(repo_root: Path, *, destination: Path | None = None) -> dict:
    """Explicit, opt-in archival packaging. Output is NEVER an .alprmodel.

    No Android import approval is implied by a successful .alprresearch archive.
    """
    root = Path(repo_root).resolve()
    audit = assess_research_mz_int8(root)
    dest = _safe_managed_path(root, BUNDLE_REL, exists=False) if destination is None else Path(destination)
    expected_dest = _safe_managed_path(root, BUNDLE_REL, exists=False)
    if dest.absolute() != expected_dest.absolute() or dest.suffix.lower() != ".alprresearch":
        raise MzPackagingNotReady("ONLY_EXACT_RESEARCH_ARCHIVE_DESTINATION_ALLOWED")
    if dest.exists() or dest.is_symlink():
        raise MzPackagingNotReady(f"RESEARCH_ARCHIVE_ALREADY_EXISTS: {dest}")
    if not dest.parent.is_dir():
        raise MzPackagingNotReady("RESEARCH_ARCHIVE_PARENT_MISSING")
    members = {"research_contract.json": _safe_managed_path(root, STAGED_REL)}
    members.update({zip_name: _safe_managed_path(root, rel) for rel, _sha, zip_name in SOURCES.values() if zip_name})
    snapshot = {arc: _hash_file(path) for arc, path in members.items()}
    temp = dest.parent / f".{dest.name}.building-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("xb") as output:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
                for arc, path in sorted(members.items()):
                    if PurePosixPath(arc).is_absolute() or ".." in PurePosixPath(arc).parts:
                        raise MzPackagingNotReady("RESEARCH_ARCHIVE_UNSAFE_MEMBER")
                    archive.write(path, arc)
            output.flush()
            os.fsync(output.fileno())
        with zipfile.ZipFile(temp, "r") as archive:
            if set(archive.namelist()) != set(members):
                raise MzPackagingNotReady("RESEARCH_ARCHIVE_MEMBER_MISMATCH")
            for arc, digest in snapshot.items():
                if hashlib.sha256(archive.read(arc)).hexdigest() != digest:
                    raise MzPackagingNotReady(f"RESEARCH_ARCHIVE_HASH_MISMATCH: {arc}")
        for arc, path in members.items():
            if _hash_file(path) != snapshot[arc]:
                raise MzPackagingNotReady(f"RESEARCH_SOURCE_CHANGED_DURING_BUNDLE: {arc}")
        # os.link creates a destination only if it doesn't exist (atomic NO_REPLACE).
        # Do not fallback to os.replace or a check-then-rename race.
        os.link(temp, dest)
    except FileExistsError as exc:
        raise MzPackagingNotReady("RESEARCH_ARCHIVE_DESTINATION_RACE") from exc
    except OSError as exc:
        raise MzPackagingNotReady(f"RESEARCH_ARCHIVE_NO_SAFE_PUBLICATION: {exc}") from exc
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "status": "RESEARCH_ARCHIVE_ONLY_NOT_ANDROID_PACKAGE",
        "path": str(dest),
        "sha256": _hash_file(dest),
        "source_model_sha256": audit["model_sha256"],
        "android_import_approved": False,
        "alprmodel_export_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MZ-DAY INT8 research packaging gate (no conversion or alprmodel)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--assess", action="store_true", help="Read-only audit and blocking reasons")
    group.add_argument("--stage-research", action="store_true", help="Explicit research-only .alprresearch ZIP")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--verify-checkpoint-labels", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage_research:
            if args.calibration_manifest or args.verify_checkpoint_labels:
                raise MzPackagingNotReady("RESEARCH_STAGING_DOES_NOT_GRANT_CALIBRATION_OR_ANDROID_APPROVAL")
            result = stage_research_bundle(args.repo)
        else:
            result = assess_research_mz_int8(
                args.repo, calibration_manifest=args.calibration_manifest,
                verify_checkpoint_labels=args.verify_checkpoint_labels,
            )
    except Exception as exc:
        print(f"MZ_RESEARCH_GATE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
