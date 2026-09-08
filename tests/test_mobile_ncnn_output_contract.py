import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import pytest

from auto_annotation_tool.exporters import mobile_model_exporter as mobile


def output_spec(*, raw=False, role="plate", dimensions=2, classes=1):
    task = "pose" if role == "plate" else "detect"
    keypoints = 4 if task == "pose" else 0
    return mobile._infer_output_spec(
        [1, 4 + classes + keypoints * dimensions, 5376] if raw else [1, 300, 6 + keypoints * dimensions],
        task=task, class_count=classes, keypoint_count=keypoints,
        keypoint_dimensions=dimensions if keypoints else 0, end2end_output=True,
        confidence_threshold=0.0, iou_threshold=0.37,
    )


def make_variant(tmp_path, runtime, spec):
    suffixes = (".param", ".bin") if runtime == "ncnn" else (".onnx" if runtime == "onnx" else ".tflite",)
    folder = tmp_path / runtime
    folder.mkdir(parents=True, exist_ok=True)
    files = tuple(folder / ("model" + suffix) for suffix in suffixes)
    for path in files:
        path.write_bytes(b"fixture model")
    return mobile.ExportedVariant(
        id=runtime + "-fp32", runtime=runtime, precision="fp32", files=files,
        relative_files=tuple("variants/" + runtime + "/" + path.name for path in files),
        input_spec={"width": 512, "height": 512, "channels": 3, "layout": "NCHW",
                    "color": "RGB", "data_type": "FLOAT32", "scale": 1 / 255, "offset": 0},
        output_spec=spec,
    )


@pytest.mark.parametrize("architecture", ["yolo26n-pose", "yolo26s-pose"])
@pytest.mark.parametrize("dimensions,expected", [(2, 13), (3, 17)])
def test_ncnn_export_uses_raw_artifact_contract(tmp_path, architecture, dimensions, expected):
    sources = tmp_path / "sources"
    sources.mkdir()
    for suffix in ("param", "bin"):
        (sources / ("model.ncnn." + suffix)).write_bytes(b"fixture")
    request = mobile.MobileExportRequest(
        checkpoint=tmp_path / (architecture + ".pt"), destination=tmp_path / "model.alprmodel",
        role="plate", formats=("ncnn",), image_size=512, confidence_threshold=0.0, iou_threshold=0.37,
    )
    model = SimpleNamespace(export=Mock(return_value=sources), end2end=True)
    variant = mobile.MobileModelExporter()._export_variant(
        model, runtime="ncnn", precision="fp32", request=request, package_root=tmp_path / "package",
        task="pose", class_count=1, keypoint_count=4, keypoint_dimensions=dimensions, end2end_output=True,
    )
    out = variant.output_spec
    assert out == {
        "decoder": "ultralytics_pose_raw_v1", "output_format": "raw_yolo",
        "class_count": 1, "keypoint_count": 4, "keypoint_dimensions": dimensions,
        "end2end_output": False, "has_objectness": False, "tensor_layout": "channels_first",
        "box_format": "xywh", "normalized_coordinates": False, "nms_in_graph": False,
        "nms_required": True, "confidence_threshold": 0.0, "iou_threshold": 0.37,
    }
    assert mobile.expected_yolo_output_attributes(**{
        key: out[key] for key in ("output_format", "class_count", "keypoint_count", "keypoint_dimensions", "has_objectness")
    }) == expected
    assert model.end2end is True


@pytest.mark.parametrize("runtime", ["onnx", "tflite", "ncnn"])
@pytest.mark.parametrize("role,task,classes", [("plate", "pose", 1), ("vehicle", "detect", 80), ("character", "detect", 36)])
def test_task_comes_from_role_even_if_decoder_disagrees(tmp_path, runtime, role, task, classes):
    spec = output_spec(role=role, classes=classes)
    spec["decoder"] = "ultralytics_detect_end2end_v1" if role == "plate" else "ultralytics_pose_end2end_v1"
    variant = make_variant(tmp_path, runtime, spec)
    exporter = mobile.MobileModelExporter()
    with patch.object(exporter, "_inspect_onnx_variant", return_value=(variant.input_spec, {})) as onnx, \
         patch.object(exporter, "_inspect_tflite_variant", return_value=(variant.input_spec, {})) as tflite:
        result = exporter.inspect_variant(variant, role=role)
    if runtime == "ncnn":
        assert result.output_spec["decoder"] == f"ultralytics_{task}_raw_v1"
        assert result.output_spec["class_count"] == classes
    else:
        call = (onnx if runtime == "onnx" else tflite).call_args
        assert call.kwargs["task"] == task
        assert call.kwargs["confidence_threshold"] == 0.0
        assert call.kwargs["iou_threshold"] == 0.37


@pytest.mark.parametrize("runtime", ["onnx", "tflite"])
@pytest.mark.parametrize("shape,raw", [([1, 13, 5376], True), ([1, 5376, 13], True), ([1, 300, 14], False)])
def test_artifact_shape_resolves_pose_contract(tmp_path, runtime, shape, raw):
    variant = make_variant(tmp_path, runtime, output_spec())
    exporter = mobile.MobileModelExporter()
    if runtime == "onnx":
        import onnx
        from onnx import TensorProto, helper
        graph = helper.make_graph([], "output-contract", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 512, 512])],
                                  [helper.make_tensor_value_info("output0", TensorProto.FLOAT, shape)])
        onnx.save(helper.make_model(graph), str(variant.files[0]))
        result = exporter.inspect_variant(variant, role="plate")
    else:
        interpreter = Mock()
        interpreter.get_input_details.return_value = [{"shape": np.array([1, 512, 512, 3]), "dtype": np.float32}]
        interpreter.get_output_details.return_value = [{"shape": np.array(shape), "dtype": np.float32}]
        with patch.object(mobile, "_tflite_interpreter_class", return_value=Mock(return_value=interpreter)):
            result = exporter.inspect_variant(variant, role="plate")
    out = result.output_spec
    assert out["output_format"] == ("raw_yolo" if raw else "end2end_detections")
    assert out["decoder"] == ("ultralytics_pose_raw_v1" if raw else "ultralytics_pose_end2end_v1")
    assert out["end2end_output"] is (not raw)
    assert out["nms_required"] is raw
    assert out["box_format"] == ("xywh" if raw else "xyxy")
    assert out["keypoint_dimensions"] == 2
    assert out["normalized_coordinates"] is (runtime == "tflite")


def write_package(tmp_path, spec, *, override=None, runtime="ncnn", role="plate", name="model"):
    variant = make_variant(tmp_path / name, runtime, spec)
    manifest = {
        "schema": "alpr.model.v1", "model_id": name, "role": role, "task": "pose" if role == "plate" else "detect",
        "input": variant.input_spec, "output": spec,
        "labels": [f"class{i}" for i in range(spec["class_count"])],
        "variants": [{"id": variant.id, "runtime": runtime, "precision": "fp32", "files": list(variant.relative_files),
                      "sha256": {relative: mobile.sha256_file(path) for path, relative in zip(variant.files, variant.relative_files)}}],
    }
    if override is not None:
        manifest["variants"][0]["output"] = override
    path = tmp_path / (name + ".alprmodel")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for source, relative in zip(variant.files, variant.relative_files):
            archive.write(source, relative)
    return path


@pytest.mark.parametrize("placement", ["top", "override", "partial_override"])
def test_zip_validator_rejects_ncnn_end2end_even_with_valid_checksums(tmp_path, placement):
    raw, end2end = output_spec(raw=True), output_spec()
    top = end2end if placement == "top" else raw
    override = None if placement == "top" else end2end
    if placement == "partial_override":
        override = {"output_format": "end2end_detections"}
    package = write_package(tmp_path, top, override=override)
    with pytest.raises(mobile.MobileExportError, match="ncnn"):
        mobile.MobileModelExporter().validate_package(package)


@pytest.mark.parametrize("field,value", [
    ("decoder", "ultralytics_detect_raw_v1"), ("box_format", "xyxy"), ("nms_required", False),
    ("nms_in_graph", True), ("end2end_output", True), ("tensor_layout", "detections_first"),
    ("has_objectness", True), ("normalized_coordinates", True), ("class_count", 2), ("keypoint_dimensions", 0),
])
def test_zip_validator_rejects_inconsistent_ncnn_override(tmp_path, field, value):
    raw = output_spec(raw=True)
    package = write_package(tmp_path, raw, override={**raw, field: value})
    with pytest.raises(mobile.MobileExportError, match="ncnn"):
        mobile.MobileModelExporter().validate_package(package)


def test_ncnn_raw_override_can_replace_end2end_top_level(tmp_path):
    package = write_package(tmp_path, output_spec(), override=output_spec(raw=True))
    mobile.MobileModelExporter().validate_package(package)


def test_incomplete_override_does_not_inherit_missing_dimensions(tmp_path):
    package = write_package(tmp_path, output_spec(raw=True), override={"output_format": "raw_yolo"})
    with pytest.raises(mobile.MobileExportError, match="ncnn"):
        mobile.MobileModelExporter().validate_package(package)


def test_mixed_runtime_manifest_preserves_outputs_and_provenance(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {"training": {"best_epoch": 135, "dataset_id": "dataset", "split_sha256": "split", "data_yaml_sha256": "yaml"},
                "lineage": {"parent": "original"}, "metrics": {"best_epoch": 135}}
    snapshot = copy.deepcopy(metadata)
    request = mobile.MobileExportRequest(checkpoint=checkpoint, destination=tmp_path / "mixed.alprmodel",
                                        role="plate", formats=("litert", "onnx", "ncnn"), image_size=512, metadata=metadata)
    variants = [make_variant(tmp_path, runtime, output_spec(raw=runtime == "ncnn")) for runtime in ("tflite", "onnx", "ncnn")]
    # TFLite uses normalized coordinates, while ONNX/NCNN use pixels.
    variants[0].output_spec["normalized_coordinates"] = True
    exporter = mobile.MobileModelExporter()
    manifest = exporter._build_manifest(request=request, model_id="mixed", role="plate", task="pose",
                                        width=512, height=512, labels=["plate"], variants=variants,
                                        checkpoint=checkpoint, model_info={"end2end_output": True})
    exporter._validate_manifest_basic(manifest)
    assert manifest["output"]["output_format"] == "end2end_detections"
    for variant, item in zip(variants, manifest["variants"]):
        assert item.get("output", manifest["output"]) == variant.output_spec
    assert manifest["training"] == snapshot["training"]
    assert manifest["lineage"] == snapshot["lineage"]
    assert manifest["metrics"] == snapshot["metrics"]
    assert manifest["source"]["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert metadata == snapshot


def test_full_s_package_and_separate_n_keep_mp_mz_contracts(tmp_path):
    exporter = mobile.MobileAlprPackageExporter()
    mp = write_package(tmp_path, output_spec(raw=True, role="vehicle", classes=80), role="vehicle", name="mp")
    mz = write_package(tmp_path, output_spec(raw=True, role="character", classes=36), role="character", name="mz")
    mt_s = write_package(tmp_path, output_spec(raw=True), name="mt_s")
    mt_n = write_package(tmp_path, output_spec(raw=True), name="mt_n")
    original = {"vehicle": mp.read_bytes(), "character": mz.read_bytes()}
    request = mobile.MobileAlprPackageRequest(destination=tmp_path / "full_s.alprmodel", vehicle_package=mp,
                                             plate_package=mt_s, character_package=mz, package_id="full_s")
    full_s = exporter.export(request)
    exporter.validate_package(full_s)
    mobile.MobileModelExporter().validate_package(mt_n)
    with zipfile.ZipFile(full_s) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for role, expected_bytes in original.items():
            assert archive.read(manifest["models"][role]["package_file"]) == expected_bytes


@pytest.mark.parametrize("has_objectness,expected", [(False, 13), (True, 14)])
def test_expected_dimensions_distinguish_raw_objectness_and_end2end(has_objectness, expected):
    assert mobile.expected_yolo_output_attributes(output_format="raw_yolo", class_count=1, keypoint_count=4,
                                                  keypoint_dimensions=2, has_objectness=has_objectness) == expected
    assert mobile.expected_yolo_output_attributes(output_format="end2end_detections", class_count=1,
                                                  keypoint_count=4, keypoint_dimensions=2) == 14
