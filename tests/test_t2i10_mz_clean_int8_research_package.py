"""T2-I-10 research-only package negative gates (no real model conversion)."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from auto_annotation_tool.exporters import mz_clean_int8_research_package as mod
from auto_annotation_tool.exporters.mobile_model_exporter import (
    MobileModelExporter, MobileExportError, _validate_variant_output_contract,
)


def audit():
    return {
        "schema": mod.AUDIT_SCHEMA,
        "status": "RESEARCH_EVIDENCE_ONLY_NOT_DEPLOYABLE",
        "model_sha256": mod.SOURCES["clean_raw_tflite"][1],
        "checkpoint_sha256": mod.SOURCES["checkpoint"][1],
        "android_import_approved": False,
        "alprmodel_export_approved": False,
        "final_test": "NOT_VERIFIED",
        "conversion_performed": False,
        "packaging_performed": False,
        "class_order": {"status": "CHECKPOINT_VERIFIED", "ordered_labels": list(mod.LABELS),
                        "order_sha256": mod.LABELS_SHA256},
        "calibration": {"status": "EXPLORATORY_ONLY_REVALIDATED",
                        "manifest": "Workspace/calibration_character_EXPLORATORY_ONLY.json",
                        "sha256": "a"*64, "input_fingerprint": "b"*64},
        "manifest_discovery": {"truncated": False,
                               "selection": "UNIQUE_PREFILTER_LIVE_VALIDATION_REQUIRED"},
        "remaining_blockers": ["ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED",
                               "ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT",
                               "FINAL_TEST_NOT_VERIFIED"],
    }


def graph():
    return {
        "classification": "INT8", "full_integer_graph": True,
        "artifact_sha256": mod.SOURCES["clean_raw_tflite"][1],
        "analysis_sha256": "c"*64,
    }


def manifest(monkeypatch):
    monkeypatch.setattr(mod, "_require_tflite_precision", lambda *args: None)
    monkeypatch.setattr(mod, "compact_evidence", lambda report: {"artifact_sha256": report["artifact_sha256"],
                                                                 "classification": report["classification"]})
    return mod.make_manifest(audit=audit(), graph=graph(), calibration_sha="a"*64,
                             calibration_fingerprint="b"*64)


def test_assembled_manifest_carries_research_not_approval(monkeypatch):
    man = manifest(monkeypatch)
    MobileModelExporter()._validate_manifest_basic(man)
    assert man["role"] == "character"
    assert man["output"]["output_format"] == "raw_split_ltrb_scores_v1"
    assert man["labels"][26] == "Q"
    assert man["variants"][0]["sha256"][mod.MODEL_MEMBER] == mod.SOURCES["clean_raw_tflite"][1]
    assert man["research_mz_int8"]["android_import_approved"] is False
    assert man["research_mz_int8"]["production_deployment_approved"] is False
    assert man["research_mz_int8"]["calibration_usage"] == "NOT_ATTESTED_DURING_REUSE"
    assert "calibration_evidence" not in man


@pytest.mark.parametrize("change", [
    lambda m: m["labels"].reverse(),
    lambda m: m["variants"][0]["sha256"].update({mod.MODEL_MEMBER: "f" * 64}),
    lambda m: m["source"].update({"checkpoint_sha256": "f" * 64}),
    lambda m: m["research_mz_int8"].update({"android_import_approved": True}),
    lambda m: m["research_mz_int8"].update({"production_deployment_approved": True}),
    lambda m: m["research_mz_int8"].update({"calibration_usage": "CONSUMED_BY_VERIFIED_LOADER"}),
    lambda m: m["input"].update({"width": 512}),
    lambda m: m["input"]["quantization"].update({"zero_point": 0}),
    lambda m: m["variants"].append(deepcopy(m["variants"][0])),
    lambda m: m["variants"][0].update({"precision": "fp32"}),
    lambda m: m["variants"][0].update({"output": deepcopy(m["output"])}),
])
def test_wrong_frozen_package_contract_fails(monkeypatch, change):
    man = manifest(monkeypatch)
    change(man)
    with pytest.raises((MobileExportError, ValueError)):
        MobileModelExporter()._validate_manifest_basic(man)


@pytest.mark.parametrize("key,value", [
    ("decoder", "ultralytics_detect_raw_v1"),
    ("box_format", "xywh"),
    ("tensor_layout", "anchors_first"),
    ("class_count", 35),
    ("normalized_coordinates", False),
    ("nms_required", True),
    ("has_objectness", True),
    ("iou_threshold", float('nan')),
    ("confidence_threshold", -0.01),
])
def test_invalid_split_decoder_contract_fails(monkeypatch, key, value):
    spec = manifest(monkeypatch)["output"]
    spec[key] = value
    with pytest.raises(MobileExportError):
        _validate_variant_output_contract("tflite", spec, task="detect", class_count=36)


def test_audit_does_not_authorize_deployment():
    a = audit()
    assert mod._guard_audit(a) == ("a"*64, "b"*64)
    for name in ("android_import_approved", "alprmodel_export_approved", "conversion_performed"):
        changed = deepcopy(a)
        changed[name] = True
        with pytest.raises(mod.ResearchPackageBlocked):
            mod._guard_audit(changed)
    changed = deepcopy(a)
    changed["remaining_blockers"].remove("FINAL_TEST_NOT_VERIFIED")
    with pytest.raises(mod.ResearchPackageBlocked):
        mod._guard_audit(changed)


def test_unverified_calibration_and_ambiguous_binding_rejected():
    for mutation in ({"calibration": {"status": "NOT_REVALIDATED"}},
                     {"manifest_discovery": {"truncated": True, "selection": "INCOMPLETE_SCAN_NO_AUTOMATIC_BINDING"}}):
        a = audit()
        a.update(mutation)
        with pytest.raises(mod.ResearchPackageBlocked):
            mod._guard_audit(a)


def test_research_archive_calibration_usage_is_not_fabricated(monkeypatch):
    from auto_annotation_tool.exporters.mobile_model_exporter import _validate_mz_split_research_archive
    man = manifest(monkeypatch)
    with zipfile.ZipFile(io.BytesIO(), "w") as unused:
        pass
    with zipfile.ZipFile(io.BytesIO(), "w") as archive:
        # Even before reading the archive, a false consumed-calibration marker blocks.
        man["calibration_evidence"] = {"status": "EXPLORATORY_ONLY"}
        with pytest.raises(MobileExportError, match="CANNOT_ASSERT_CALIBRATION_LOADER_USAGE"):
            _validate_mz_split_research_archive(archive, man)


def test_existing_destination_not_overwritten(monkeypatch, tmp_path):
    root = tmp_path
    destination = root / mod.DEST_REL
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"do not overwrite me")
    with pytest.raises(mod.ResearchPackageBlocked, match="ALREADY_EXISTS"):
        mod.package_existing_research_model(root)
    assert destination.read_bytes() == b"do not overwrite me"


def test_reject_mismatching_graph_sha(monkeypatch):
    monkeypatch.setattr(mod, "_require_tflite_precision", lambda *args: None)
    bad = graph()
    bad["artifact_sha256"] = "e"*64
    with pytest.raises(mod.ResearchPackageBlocked, match="NOT_FULL_INT8"):
        mod.make_manifest(audit=audit(), graph=bad, calibration_sha="a"*64,
                          calibration_fingerprint="b"*64)
