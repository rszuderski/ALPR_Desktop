"""Focused Z4LIB006 tests: observed conversions, not reconstructed history."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from auto_annotation_tool.gui.z4_mobile_conversion_journal import (
    attest_conversion,
    build_conversion_plan,
    record_conversion_results,
    record_conversion_starts,
    record_conversion_unconfirmed,
)


def _new_request(tmp_path, role="character", *, fmt="litert", precision="int8"):
    weight = tmp_path / f"{role}_best.pt"
    weight.write_bytes(b"frozen-weights-" + role.encode())
    return SimpleNamespace(role=role, checkpoint=weight, formats=(fmt,),
                           quantizations=(precision,), format_quantizations={fmt: (precision,)})


def _manifest(request, *, precision="int8"):
    return {"schema": "alpr.model.v1", "role": request.role,
            "task": "pose" if request.role == "plate" else "detect",
            "model_id": "MODEL-TEST", "source": {"checkpoint_sha256": registry.sha256_file(request.checkpoint)},
            "variants": [{"runtime": "tflite", "precision": precision}]}


def _zip_package(path, manifest, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, value in (extra or {}).items():
            archive.writestr(name, value)
    return path


def _bundle(tmp_path, request):
    child_manifest = _manifest(request)
    child_file = tmp_path / "mz_child.alprmodel"
    _zip_package(child_file, child_manifest)
    sidecar_name = "models/character/manifest.json"
    child_name = "models/character/model.alprmodel"
    sidecar_raw = json.dumps(child_manifest).encode()
    child_raw = child_file.read_bytes()
    plate_manifest = {"schema": "alpr.model.v1", "role": "plate", "task": "pose",
                      "model_id": "REUSED-MT", "source": {"checkpoint_sha256": "b" * 64},
                      "variants": [{"runtime": "tflite", "precision": "int8"}]}
    plate_sidecar = "models/plate/manifest.json"
    plate_child = "models/plate/model.alprmodel"
    plate_sidecar_raw = json.dumps(plate_manifest).encode()
    plate_file = tmp_path / "mt_child.alprmodel"
    _zip_package(plate_file, plate_manifest)
    plate_raw = plate_file.read_bytes()
    root_manifest = {
        "schema": "alpr.package.v1", "models": {
            "plate": {"role": "plate", "manifest_file": plate_sidecar, "package_file": plate_child,
                      "sha256": {plate_sidecar: hashlib.sha256(plate_sidecar_raw).hexdigest(),
                                 plate_child: hashlib.sha256(plate_raw).hexdigest()}},
            "character": {"role": "character", "manifest_file": sidecar_name, "package_file": child_name,
                          "sha256": {sidecar_name: hashlib.sha256(sidecar_raw).hexdigest(),
                                     child_name: hashlib.sha256(child_raw).hexdigest()}}
        },
    }
    full = tmp_path / "complete.alprmodel"
    _zip_package(full, root_manifest, {sidecar_name: sidecar_raw, child_name: child_raw,
                                       plate_sidecar: plate_sidecar_raw, plate_child: plate_raw})
    return full


def _make_bundle_request(tmp_path, *, fresh_mz=True, fresh_mt=False):
    mz = _new_request(tmp_path, "character") if fresh_mz else None
    mt = _new_request(tmp_path, "plate") if fresh_mt else None
    return SimpleNamespace(
        destination=tmp_path / "complete.alprmodel", vehicle_request=None,
        plate_request=mt, character_request=mz, vehicle_package=None,
        plate_package="ready_mt.alprmodel" if not fresh_mt else None,
        character_package="ready_mz.alprmodel" if not fresh_mz else None,
    ), mz, mt


def test_reuse_only_means_zero_new_conversions(tmp_path):
    request, _, _ = _make_bundle_request(tmp_path, fresh_mz=False)
    assert build_conversion_plan(request, parent_operation_id="Z4_TEST_REUSE") == []
    assert not registry.history_file(tmp_path).exists()


def test_only_mz_is_recorded_for_reused_mt(tmp_path):
    bundle, mz, _ = _make_bundle_request(tmp_path)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_TEST_ONLY_MZ")
    assert len(plan) == 1
    assert plan[0]["role"] == "MZ"
    assert plan[0]["checkpoint_sha256"] == registry.sha256_file(mz.checkpoint)
    assert plan[0]["requested_variants"] == [{"runtime": "tflite", "precision": "int8"}]
    assert plan[0]["artifact_scope"] == "NESTED_CHILD_OF_COMPLETE_PACKAGE"


def test_archive_attests_real_mz_and_history_contains_shas(tmp_path):
    bundle, mz, _ = _make_bundle_request(tmp_path)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_REAL_MODEL_001")
    record_conversion_starts(tmp_path, plan, destination=bundle.destination)
    result = _bundle(tmp_path, mz)
    assert record_conversion_results(tmp_path, plan, artifact_path=result) == []
    events, _ = registry.load_history(tmp_path)
    assert [e["outcome"] for e in events] == ["STARTED", "SUCCEEDED"]
    assert all(e["operation"] == "MODEL_CONVERSION" for e in events)
    assert events[1]["conversion"]["actual_variants"] == [{"runtime": "tflite", "precision": "int8"}]
    assert events[1]["conversion"]["android_approved"] is False
    assert events[1]["artifact_sha256"] == registry.sha256_file(result)
    assert events[1]["conversion"]["embedded_package_sha256"]
    rows, all_events, _ = registry.library_snapshot(tmp_path, tmp_path)
    assert len(all_events) == 2
    complete = next(row for row in rows if Path(row.path) == result.resolve())
    assert complete.state == "REGISTERED_SHA_MATCH_NOT_VERIFIED"


def test_attestation_is_read_only(tmp_path):
    bundle, mz, _ = _make_bundle_request(tmp_path)
    result = _bundle(tmp_path, mz)
    before = result.read_bytes()
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_READ_ONLY_1")
    proof = attest_conversion(plan[0], result)
    assert proof["model_id"] == "MODEL-TEST"
    assert result.read_bytes() == before
    assert not registry.history_file(tmp_path).exists()


def test_wrong_checkpoint_sha_leads_to_unconfirmed(tmp_path):
    bundle, mz, _ = _make_bundle_request(tmp_path)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_WRONG_SHA_1")
    record_conversion_starts(tmp_path, plan, destination=bundle.destination)
    result = _bundle(tmp_path, mz)
    mz.checkpoint.write_bytes(b"mutated-by-external-operator")
    warnings = record_conversion_results(tmp_path, plan, artifact_path=result)
    assert warnings and "SHA" in warnings[0]
    events, _ = registry.load_history(tmp_path)
    assert [e["outcome"] for e in events] == ["STARTED", "UNCONFIRMED"]
    assert events[-1]["artifact_sha256"] is None


def test_bad_nested_sidecar_hash_does_not_claim_success(tmp_path):
    bundle, mz, _ = _make_bundle_request(tmp_path)
    result = _bundle(tmp_path, mz)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_BAD_SIDECAR_1")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(result, 'a') as archive:
            archive.writestr("models/character/manifest.json", "{}")
    assert record_conversion_results(tmp_path, plan, artifact_path=result)
    events, _ = registry.load_history(tmp_path)
    assert len(events) == 1 and events[0]["outcome"] == "UNCONFIRMED"


def test_root_single_model_success_and_variants(tmp_path):
    req = _new_request(tmp_path)
    req.destination = tmp_path / "single.alprmodel"
    plan = build_conversion_plan(req, parent_operation_id="Z4_SINGLE_MODEL_1")
    assert len(plan) == 1 and plan[0]["artifact_scope"] == "SINGLE_MODEL_PACKAGE"
    result = _zip_package(req.destination, _manifest(req))
    assert attest_conversion(plan[0], result)["embedded_package_sha256"] == ""
    record_conversion_starts(tmp_path, plan, destination=req.destination)
    assert record_conversion_results(tmp_path, plan, artifact_path=result) == []


def test_failed_bundle_marks_unconfirmed_not_failed_conversion(tmp_path):
    bundle, _, _ = _make_bundle_request(tmp_path)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_INTERRUPTED_1")
    record_conversion_starts(tmp_path, plan, destination=bundle.destination)
    assert record_conversion_unconfirmed(tmp_path, plan, reason="pakowanie failed") == []
    events, _ = registry.load_history(tmp_path)
    assert [e["outcome"] for e in events] == ["STARTED", "UNCONFIRMED"]


def test_mixed_fresh_and_reused_not_confused(tmp_path):
    bundle, mz, mt = _make_bundle_request(tmp_path, fresh_mz=True, fresh_mt=True)
    plan = build_conversion_plan(bundle, parent_operation_id="Z4_TWO_FRESH_1")
    assert [entry["role"] for entry in plan] == ["MT", "MZ"]
    bundle.plate_package = "already_exported_mt.alprmodel"
    with pytest.raises(ValueError, match="niejednoznaczny"):
        build_conversion_plan(bundle, parent_operation_id="Z4_TWO_FRESH_1")


def test_historical_legacy_events_unmodified(tmp_path):
    package = _zip_package(tmp_path / 'existing.alprmodel',
                           {"schema":"alpr.model.v1", "role":"plate", "variants":[]})
    before = registry.append_event(tmp_path, operation_id="Z4_LEGACY_001",
                                   operation="SINGLE_MODEL_EXPORT", outcome="SUCCEEDED",
                                   artifact_path=str(package))
    request = _new_request(tmp_path)
    request.destination = tmp_path / "next.alprmodel"
    plan = build_conversion_plan(request, parent_operation_id="Z4_NEXT_ONE_1")
    record_conversion_starts(tmp_path, plan, destination=request.destination)
    events, _ = registry.load_history(tmp_path)
    assert events[0] == before and len(events) == 2


def test_conversion_event_metadata_is_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        registry.append_event(tmp_path, operation_id="Z4_BAD_EVENT_1", operation="MODEL_CONVERSION",
                              outcome="SUCCEEDED", conversion={"role":"MZ", "checkpoint_sha256":"123",
                                                           "parent_export_operation_id":"Z4_BAD_EVENT_1"})
    with pytest.raises(ValueError, match="metadane|Metadane"):
        registry.append_event(tmp_path, operation_id="Z4_BAD_EVENT_1", operation="SINGLE_MODEL_EXPORT",
                              outcome="STARTED", conversion={"role":"MZ"})
    assert not registry.history_file(tmp_path).exists()


def test_source_integration_only_inside_gui_export_worker():
    from pathlib import Path
    gui = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_model_export.py"
    source = gui.read_text(encoding="utf-8-sig")
    for name in ("build_conversion_plan(", "record_conversion_starts(",
                 "record_conversion_results(", "record_conversion_unconfirmed("):
        assert name in source
    assert source.index("record_conversion_starts(") < source.index("package_exporter.export(")
