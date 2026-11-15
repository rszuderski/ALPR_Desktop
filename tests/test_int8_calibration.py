import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auto_annotation_tool.exporters import int8_calibration as cal
from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from int8_calibration_fixture import fixture, holdout, save_plan, write_json


@pytest.mark.parametrize("role", ["plate", "character"])
def test_deterministic_train_selection_sha_dedup_and_provenance(tmp_path, role):
    f = fixture(tmp_path, role, duplicate=True)
    a = cal.build_manifest(**f["config"])
    assert a == cal.build_manifest(**f["config"])
    assert a["status"] == "READY" and a["split"] == "train"
    assert a["counts"] == dict(train_candidates=4, cross_split_rejected=0, duplicates_removed=1,
                               eligible_unique=3, selected=2, selected_source_scenes=2)
    expected = sorted(cal.file_sha256(f["paths"]["train" + str(i)]) for i in range(3))[:2]
    assert [s["sha256"] for s in a["samples"]] == expected
    assert all(s["split"] == "train" and s["source_scene_sha256"] for s in a["samples"])
    assert a["execution_precision"] == "UNVERIFIED"


@pytest.mark.parametrize("field", ["final_inventory", "eval_inventory", "approved_by"])
def test_missing_proof_blocks_even_when_train_exists(tmp_path, field):
    f = fixture(tmp_path)
    f["config"][field] = ""
    p = tmp_path / "plan.json"
    m = save_plan(f, p)
    assert m["status"] == "BLOCKED" and m["samples"]
    with pytest.raises(cal.CalibrationError, match="BLOCKED"):
        cal.validate_manifest(p)


@pytest.mark.parametrize("purpose", ["EVAL396", "final_test"])
@pytest.mark.parametrize("role", ["plate", "character"])
def test_overlap_includes_mz_source_scene_not_only_crop(tmp_path, purpose, role):
    f = fixture(tmp_path, role)
    source = f["inv"]["source_scene_hashes"][0]["sha256"]
    holdout(f["eval_path"] if purpose == "EVAL396" else f["final_path"], purpose, [source])
    m = cal.build_manifest(**f["config"])
    assert m["status"] == "BLOCKED"
    assert any("Nakładanie SHA" in b and purpose in b for b in m["blockers"])


@pytest.mark.parametrize("role", ["plate", "character"])
def test_sources_shared_with_val_test_are_explicitly_rejected(tmp_path, role):
    f = fixture(tmp_path, role)
    f["inv"]["source_scene_hashes"][0]["splits"].append("val")
    write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    m = cal.build_manifest(**f["config"])
    assert m["counts"]["cross_split_rejected"] == 1
    assert f["inv"]["source_scene_hashes"][0]["sha256"] not in {r["source_scene_sha256"] for r in m["samples"]}


@pytest.mark.parametrize("change", ["sample", "metadata", "checkpoint", "history", "exclusion", "imgsz", "role", "fingerprint"])
def test_revalidation_rejects_stale_or_tampered_evidence(tmp_path, change):
    f = fixture(tmp_path, "character")
    p = tmp_path / "plan.json"
    m = save_plan(f, p)
    kwargs = dict(role="character", checkpoint=f["checkpoint"], image_size=32)
    if change == "sample":
        Path(m["samples"][0]["path"]).write_bytes(b"changed")
    elif change == "metadata":
        write_json(f["dataset"] / "metadata_manifest.json", {"items": []})
    elif change == "checkpoint":
        f["checkpoint"].write_bytes(b"changed checkpoint")
    elif change == "history":
        f["run"]["status"] = "running"
        write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    elif change == "exclusion":
        holdout(f["final_path"], "final_test", ["c" * 64])
    elif change == "imgsz":
        kwargs["image_size"] = 64
    elif change == "role":
        kwargs["role"] = "plate"
    else:
        m["counts"]["selected"] += 1
        write_json(p, m)
    with pytest.raises(cal.CalibrationError):
        cal.validate_manifest(p, **kwargs)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "bad_sha", "unfrozen", "tampered", "missing_approval"])
def test_invalid_holdout_never_certifies_independence(tmp_path, kind):
    p = tmp_path / "final.json"
    holdout(p, "final_test", ["a" * 64])
    d = json.loads(p.read_text())
    if kind == "empty": d.update(source_scene_hashes=[], scene_count=0)
    if kind == "duplicate": d.update(source_scene_hashes=["a" * 64] * 2, scene_count=2)
    if kind == "bad_sha": d["source_scene_hashes"] = ["invalid"]
    if kind == "unfrozen": d["status"] = "draft"
    if kind == "missing_approval": d["approved_by"] = ""
    if kind != "tampered": d["inventory_fingerprint"] = cal.fingerprint({k:v for k,v in d.items() if k != "inventory_fingerprint"})
    else: d["cohort_id"] = "changed"
    write_json(p, d)
    with pytest.raises(cal.CalibrationError): cal.exclusion_inventory(p, "final_test")


@pytest.mark.parametrize("train", ["images/val", "images/test", "missing", "../outsider"])
def test_yaml_cannot_redirect_train_to_another_split(tmp_path, train):
    f = fixture(tmp_path)
    p = f["dataset"] / "data.yaml"
    p.write_text(f"train: {train}\nval: images/val\nnames: [plate]\n")
    f["run"]["training_dataset_snapshot"]["data_yaml_sha256"] = cal.file_sha256(p)
    write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    with pytest.raises(cal.CalibrationError): cal.build_manifest(**f["config"])


@pytest.mark.parametrize("fault", ["missing", "conflict", "val"])
def test_mz_requires_frozen_exact_path_bindings_in_both_manifests(tmp_path, fault):
    f = fixture(tmp_path, "character")
    if fault == "missing": f["rows"].pop(0)
    if fault == "conflict": f["rows"][0]["provenance"]["source_image_id"] = "img-sha256-" + "e" * 64
    if fault == "val": f["rows"][0]["split"] = "val"
    p = write_json(f["dataset"] / "metadata_manifest.json", {"items": f["rows"]})
    f["run"]["training_dataset_snapshot"]["manifests"][0]["sha256"] = cal.file_sha256(p)
    write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    with pytest.raises(cal.CalibrationError): cal.build_manifest(**f["config"])


def test_real_readers_receive_only_selected_images_and_identical_rgb(tmp_path):
    f = fixture(tmp_path)
    m = cal.build_manifest(**f["config"])
    reader = cal.ApprovedOnnxReader(m, "images")
    batches = list(cal.ApprovedLiteRTLoader(m))
    for row, batch in zip(m["samples"], batches):
        tensor = reader.get_next()["images"]
        assert batch["im_file"] == [row["path"]]
        assert batch["img"].dtype.__str__() == "torch.uint8"
        np.testing.assert_array_equal(tensor, batch["img"].numpy().astype(np.float32) / 255)
        assert tensor.shape == (1, 3, 32, 32)
        np.testing.assert_array_equal(batch["img"][0, :, 0, 0].numpy(), [114, 114, 114])
        assert str(f["paths"]["val0"]) != row["path"] and str(f["paths"]["test0"]) != row["path"]
    assert reader.get_next() is None
    reader.assert_complete()
    reader.rewind()
    assert reader.get_next() is not None


def test_changed_image_is_rejected_at_decode_time(tmp_path):
    f = fixture(tmp_path)
    m = cal.build_manifest(**f["config"])
    Path(m["samples"][0]["path"]).write_bytes(b"changed after validation")
    with pytest.raises(cal.CalibrationError): cal.ApprovedOnnxReader(m, "images").get_next()
    with pytest.raises(cal.CalibrationError): next(iter(cal.ApprovedLiteRTLoader(m)))


def request_for(f, p):
    return mobile.MobileExportRequest(checkpoint=f["checkpoint"], destination=p.parent / "unused.alprmodel",
        role=f["config"]["role"], formats=("litert", "onnx"), quantizations=("int8",), image_size=32,
        calibration_manifest=p)


def test_native_onnx_path_passes_exact_reader_to_quantizer(tmp_path):
    f = fixture(tmp_path)
    p = tmp_path / "plan.json"
    m = save_plan(f, p)
    request = request_for(f, p)
    output = tmp_path / "synthetic.onnx"
    observed = []
    def fake_quantizer(source, target, reader, **kwargs):
        while True:
            value = reader.get_next()
            if value is None: break
            observed.append(value["images"])
        output.write_bytes(b"synthetic output; no conversion")
    with patch.object(mobile, "_onnx_input_name_and_size", return_value=("images", (32, 32))), \
         patch("onnx.load", return_value=SimpleNamespace(graph=SimpleNamespace(node=[]))), \
         patch("onnxruntime.quantization.quantize_static", side_effect=fake_quantizer):
        mobile.MobileModelExporter()._quantize_onnx_int8(tmp_path / "unused.onnx", request, output_path=output)
    assert len(observed) == m["counts"]["selected"]
    for row, tensor in zip(m["samples"], observed):
        np.testing.assert_array_equal(tensor[0, :, 16, 16], cal.sample_rgb(row, m["preprocessing"])[16, 16] / np.float32(255))


@pytest.mark.parametrize("split", ["val", "test", None])
def test_litert_binding_rejects_split_override_and_restores_callbacks(tmp_path, split):
    m = cal.build_manifest(**fixture(tmp_path)["config"])
    model = SimpleNamespace(callbacks={"on_export_start": []})
    original = model.callbacks
    with pytest.raises(cal.CalibrationError):
        with cal.bind_litert_loader(model, m):
            exporter = SimpleNamespace(args=SimpleNamespace(split=split, batch=1, int8=True, format="tflite", fraction=1.0))
            model.callbacks["on_export_start"][-1](exporter)
    assert model.callbacks is original


def test_litert_requires_loader_consumption_no_silent_default(tmp_path):
    m = cal.build_manifest(**fixture(tmp_path)["config"])
    model = SimpleNamespace(callbacks={})
    with pytest.raises(cal.CalibrationError):
        with cal.bind_litert_loader(model, m): pass


def test_actual_ultralytics_saved_model_bridge_receives_approved_pixels(tmp_path):
    # Execute installed Exporter.export_saved_model up to the converter boundary.
    # No model forward, tracing, export_onnx or converter is executed.
    import torch
    from ultralytics.engine import exporter as ul
    f = fixture(tmp_path)
    p = tmp_path / "plan.json"
    m = save_plan(f, p)
    captured = {}
    class SyntheticModel:
        callbacks = {"on_export_start": []}
        def export(self, **kwargs):
            captured["args"] = kwargs
            e = ul.Exporter.__new__(ul.Exporter)
            e.args = SimpleNamespace(**kwargs)
            e.file = tmp_path / "stub.pt"
            e.imgsz = [32, 32]
            e.metadata = {}
            e.model = SimpleNamespace(model=[object()])
            e.callbacks = self.callbacks
            e.export_onnx = Mock(return_value=tmp_path / "not_created.onnx")
            e.run_callbacks("on_export_start")
            return e.export_saved_model()[0]
    def fake_bridge(source, target, **kwargs):
        captured["pixels"] = kwargs["images"].copy()
        captured["int8"] = kwargs["int8"]
        target.mkdir()
        return object()
    model = SyntheticModel()
    original = model.callbacks
    with patch.dict(sys.modules, {"tensorflow": SimpleNamespace(__version__="2.19.0")}), \
         patch.object(ul, "check_requirements"), patch.object(ul, "check_version"), \
         patch.object(ul, "onnx2saved_model", side_effect=fake_bridge), \
         patch.object(ul.Exporter, "get_int8_calibration_dataloader", side_effect=AssertionError("default loader forbidden")):
        mobile.MobileModelExporter()._run_ultralytics_export(model, runtime="tflite", precision="int8", request=request_for(f, p))
    assert model.callbacks is original
    assert captured["args"]["split"] == "train" and captured["int8"]
    expected = np.stack([cal.sample_rgb(row, m["preprocessing"]) for row in m["samples"]]).astype(np.float32)
    np.testing.assert_array_equal(captured["pixels"], expected)
    assert not list(tmp_path.rglob("*.alprmodel")) and not list(tmp_path.rglob("*.tflite"))


def test_new_int8_request_requires_manifest_fp32_legacy_unchanged(tmp_path):
    f = fixture(tmp_path)
    p = tmp_path / "plan.json"
    request = replace(request_for(f, p), calibration_manifest=None, calibration_data=f["dataset"] / "data.yaml")
    e = mobile.MobileModelExporter()
    with patch.object(mobile, "mobile_export_required_specs", return_value=[]), \
         patch.object(mobile, "mobile_export_requirement_status", return_value=(False, "fixture")), \
         patch.object(mobile, "check_mobile_tflite_inspection_runtime", return_value=None), \
         patch.object(mobile, "check_mobile_onnx_int8_runtime", return_value=None):
        assert any("manifestu" in p for p in e.preflight(request))
        assert not e.preflight(replace(request, quantizations=("fp32",)))


def test_legacy_yaml_helper_has_no_val_test_or_directory_fallback(tmp_path):
    f = fixture(tmp_path)
    images = mobile._collect_calibration_images(f["dataset"] / "data.yaml")
    assert len(images) == 3 and all(p.parent.name == "train" for p in images)
    (f["dataset"] / "data.yaml").write_text("val: images/val\ntest: images/test\n")
    with pytest.raises(mobile.MobileExportError): mobile._collect_calibration_images(f["dataset"] / "data.yaml")


def test_manifest_cannot_change_between_export_variants(tmp_path):
    f = fixture(tmp_path)
    p = tmp_path / "plan.json"
    m = save_plan(f, p)
    request = replace(request_for(f, p), calibration_fingerprint=m["input_fingerprint"])
    f["config"]["limit"] = 3
    save_plan(f, p)
    with pytest.raises(mobile.MobileExportError, match="trakcie eksportu"):
        mobile.MobileModelExporter()._approved_calibration(request)


def test_installed_onnx2tf_adapter_receives_exact_pixels_and_normalization(tmp_path):
    from ultralytics.utils.export import tensorflow as bridge
    m = cal.build_manifest(**fixture(tmp_path / "inputs")["config"])
    pixels = np.stack([cal.sample_rgb(r, m["preprocessing"]) for r in m["samples"]]).astype(np.float32)
    captured = {}
    # Stub only the converter and legacy download; exercise the installed bridge.
    def convert(**kwargs):
        dataset = kwargs["custom_input_op_name_np_data_path"]
        captured["name"], path, captured["mean"], captured["std"] = dataset[0]
        captured["pixels"] = np.load(path)
        return object()
    with patch.dict(sys.modules, {"onnx2tf": SimpleNamespace(__version__="synthetic", convert=convert)}), \
         patch.object(bridge, "attempt_download_asset") as download:
        bridge.onnx2saved_model(tmp_path / "unused.onnx", tmp_path / "synthetic-output", int8=True, images=pixels)
    np.testing.assert_array_equal(captured["pixels"], pixels)
    assert captured["mean"] == [[[[0, 0, 0]]]] and captured["std"] == [[[[255, 255, 255]]]]
    assert captured["name"] == "images"
    assert not list(tmp_path.rglob("*.tflite"))


def test_optional_archive_calibration_evidence_sha_and_legacy_compatibility(tmp_path):
    import hashlib
    import zipfile
    from test_tflite_precision import write_package
    f = fixture(tmp_path / "inputs", "character")
    f["config"]["image_size"] = 2
    calibration = cal.build_manifest(**f["config"])
    # Archive fixture only; never call a production converter/exporter.
    package = write_package(tmp_path, "int8", "int8", evidence=True)
    e = mobile.MobileModelExporter()
    e.validate_package(package, require_precision=True)  # Legacy, no calibration proof.
    with zipfile.ZipFile(package) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["source"] = {"checkpoint_sha256": cal.file_sha256(f["checkpoint"])}
    data = json.dumps(calibration).encode()
    manifest["calibration_evidence"] = {"file": "calibration_manifest.json", "sha256": hashlib.sha256(data).hexdigest(),
        "input_fingerprint": calibration["input_fingerprint"], "split": "train", "sample_count": len(calibration["samples"])}
    entries["manifest.json"] = json.dumps(manifest).encode()
    entries["calibration_manifest.json"] = data
    for name, damaged in (("valid.alprmodel", False), ("damaged.alprmodel", True)):
        p = tmp_path / name
        with zipfile.ZipFile(p, "w") as z:
            for entry, raw in entries.items():
                z.writestr(entry, raw + b" " if damaged and entry == "calibration_manifest.json" else raw)
        if damaged:
            with pytest.raises(mobile.MobileExportError, match="kalibracji"): e.validate_package(p)
        else: e.validate_package(p, require_precision=True)


def test_native_onnx_quantizer_must_consume_entire_manifest(tmp_path):
    f = fixture(tmp_path)
    p = tmp_path / "plan.json"
    save_plan(f, p)
    with patch.object(mobile, "_onnx_input_name_and_size", return_value=("images", (32, 32))), \
         patch("onnx.load", return_value=SimpleNamespace(graph=SimpleNamespace(node=[]))), \
         patch("onnxruntime.quantization.quantize_static", return_value=None):
        with pytest.raises(mobile.MobileExportError, match="zużył"):
            mobile.MobileModelExporter()._quantize_onnx_int8(tmp_path / "unused.onnx", request_for(f, p), output_path=tmp_path / "unused-out.onnx")


def test_real_eval396_inventory_validates_declared_hash_and_count_without_images(tmp_path):
    scenes = [{"sha256": f"{n:064x}", "file": f"forbidden/{n}.jpg"} for n in range(396)]
    p = write_json(tmp_path / "included_scenes.json", scenes)
    manifest = {"schema": "emzdn01.eval_scene_selection.v1", "status": "EVALUATION_SELECTION_CREATED",
        "files": {p.name: cal.file_sha256(p)}, "counts": {"scenes": {"ALL": 396}}}
    target = write_json(tmp_path / "selection_manifest.json", manifest)
    hashes, proof = cal.exclusion_inventory(target, "EVAL396")
    assert len(hashes) == proof["scene_count"] == 396
    p.write_text("[]")
    with pytest.raises(cal.CalibrationError): cal.exclusion_inventory(target, "EVAL396")


@pytest.mark.parametrize("change", ["samples", "gate", "approval", "counts"])
def test_rehashed_inconsistent_manifest_still_fails(tmp_path, change):
    m = cal.build_manifest(**fixture(tmp_path)["config"])
    if change == "samples": m["samples"][0]["split"] = "val"
    if change == "gate": m["gates"]["final_test_inventory"] = "FINAL_TEST_INVENTORY_NOT_VERIFIED"
    if change == "approval": m["config"]["approved_by"] = ""
    if change == "counts": m["counts"]["selected"] = 999
    m["input_fingerprint"] = cal.fingerprint({k:v for k,v in m.items() if k != "input_fingerprint"})
    with pytest.raises(cal.CalibrationError): cal.validate_embedded_manifest(m)
