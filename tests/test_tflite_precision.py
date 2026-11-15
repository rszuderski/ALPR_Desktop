import hashlib
import json
from unittest.mock import patch
import zipfile

import pytest

from auto_annotation_tool.exporters import tflite_precision as precision
from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from tflite_precision_fixture import graph_bytes


@pytest.mark.parametrize("kind,expected", [("fp32", "FP32"), ("int8", "INT8"), ("mixed", "MIXED")])
def test_actual_tensor_types_override_names_and_have_deterministic_sha(kind, expected, tmp_path):
    data = graph_bytes(kind)
    path = tmp_path / "misleading_int8_filename.tflite"
    path.write_bytes(data)
    result = precision.analyze_tflite_file(path)
    assert result["classification"] == expected, result["issues"]
    assert result["artifact_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["analysis_sha256"] == precision.report_digest(result)
    assert result == precision.analyze_tflite_bytes(data)
    assert result["execution_precision"] == "UNVERIFIED"
    assert result["full_integer_graph"] is (expected == "INT8")
    assert result["subgraphs"][0]["operators"][0]["weights"] == [1]


def test_integer_accumulators_per_axis_weights_and_bias_are_allowed():
    result = precision.analyze_tflite_bytes(graph_bytes("int8", per_axis=True))
    assert result["classification"] == "INT8", result["issues"]
    assert result["tensor_type_counts"] == {"INT32": 1, "INT8": 3}


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_invalid_quantization_never_proves_integer(index):
    r = precision.analyze_tflite_bytes(graph_bytes("int8", bad_quant=index))
    assert r["classification"] == "UNVERIFIED"
    assert r["full_integer_graph"] is False
    assert any("QUANTIZATION" in x or "BIAS" in x for x in r["issues"])
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("opcode,version", [("CUSTOM", 1), ("IF", 1), ("CONV_2D", 99)])
def test_unsupported_operator_or_version_is_unverified(opcode, version):
    r = precision.analyze_tflite_bytes(graph_bytes("int8", opcode=opcode, version=version))
    assert r["classification"] == "UNVERIFIED"
    assert any("UNSUPPORTED" in x for x in r["issues"])


def test_integer_core_with_float_boundary_is_mixed():
    r = precision.analyze_tflite_bytes(graph_bytes("int8", float_boundary=True))
    assert r["classification"] == "MIXED", r["issues"]


def test_all_subgraphs_are_checked():
    r = precision.analyze_tflite_bytes(graph_bytes("int8", extra_subgraph=True))
    assert r["classification"] == "MIXED", r["issues"]


@pytest.mark.parametrize("data", [b"", b"fake model", b"\xff\xff\xff\xffTFL3xxxx", graph_bytes()[:40]])
def test_malformed_flatbuffers_is_not_a_precision_pass(data):
    r = precision.analyze_tflite_bytes(data)
    assert r["classification"] == "UNVERIFIED"
    assert r["issues"]
    assert r["artifact_sha256"] == hashlib.sha256(data).hexdigest()


def test_missing_parser_is_unverified_without_importing_runtime():
    with patch.object(precision, "load_schema", side_effect=ImportError("schema absent")):
        r = precision.analyze_tflite_bytes(graph_bytes_cached())
    assert r["classification"] == "UNVERIFIED"
    assert "schema absent" in r["issues"][0]


_FP32 = graph_bytes()


def graph_bytes_cached():
    return _FP32


def write_package(tmp_path, kind="fp32", claimed="fp32", *, evidence=False, edit=None):
    data = graph_bytes(kind)
    graph = precision.analyze_tflite_bytes(data)
    relative = "variants/tflite/model.tflite"
    input_spec = {"width": 2, "height": 2, "channels": 3, "layout": "NHWC", "color": "RGB",
                  "data_type": "INT8" if kind == "int8" else "FLOAT32", "scale": 1/255, "offset": 0}
    output = mobile._infer_output_spec([1, 300, 6], task="detect", class_count=1, keypoint_count=0,
                                      end2end_output=True, normalized_coordinates=True,
                                      confidence_threshold=.25, iou_threshold=.45)
    variant = {"id": "tflite-"+claimed, "runtime": "tflite", "precision": claimed,
               "file": relative, "sha256": {relative: hashlib.sha256(data).hexdigest()}}
    manifest = {"schema": "alpr.model.v1", "model_id": "synthetic", "role": "character", "task": "detect",
                "input": input_spec, "output": output, "labels": ["A"], "variants": [variant]}
    entries = {relative: data}
    if evidence:
        report_file = "precision/tflite-"+claimed+".json"
        payload = mobile._precision_report_bytes(graph)
        entries[report_file] = payload
        variant["graph_precision"] = dict(precision.compact_evidence(graph), report_file=report_file,
                                           report_sha256=hashlib.sha256(payload).hexdigest())
    if edit:
        edit(manifest, entries)
    path = tmp_path / "synthetic.alprmodel"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for name, payload in entries.items():
            z.writestr(name, payload)
    return path


def test_legacy_mixed_package_remains_readable_but_not_certified(tmp_path):
    path = write_package(tmp_path, "mixed", "int8")
    before = path.read_bytes()
    assert mobile.MobileModelExporter().validate_package(path) is None
    r = mobile.inspect_mobile_precision(path)
    assert r["variants"][0]["graph"]["classification"] == "MIXED"
    assert r["variants"][0]["metadata_issues"]
    assert r["file_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    with pytest.raises(mobile.MobileExportError, match="PRECISION_NOT_PROVEN"):
        mobile.MobileModelExporter().validate_package(path, require_precision=True)


@pytest.mark.parametrize("kind", ["fp32", "int8"])
def test_additive_precision_evidence_revalidated_without_schema_change(tmp_path, kind):
    p = write_package(tmp_path, kind, kind, evidence=True)
    rows = []
    assert mobile.MobileModelExporter().validate_package(p, precision_reports=rows) is None
    assert rows[0]["graph"]["classification"] == kind.upper()
    assert rows[0]["graph"]["execution_precision"] == "UNVERIFIED"


@pytest.mark.parametrize("field,value", [("classification", "INT8"), ("artifact_sha256", "0"*64),
                                         ("execution_precision", "INT8"), ("report_sha256", "0"*64)])
def test_forged_precision_evidence_rejected(tmp_path, field, value):
    def edit(m, _):
        m["variants"][0]["graph_precision"][field] = value
    p = write_package(tmp_path, evidence=True, edit=edit)
    with pytest.raises(mobile.MobileExportError, match="dowód precyzji"):
        mobile.MobileModelExporter().validate_package(p)


def test_wrong_manifest_dtype_is_reported_and_strictly_blocked(tmp_path):
    p = write_package(tmp_path, "int8", "int8", edit=lambda m, _: m["input"].update(data_type="FLOAT32"))
    r = mobile.inspect_mobile_precision(p)
    assert r["variants"][0]["graph"]["classification"] == "INT8"
    assert "INPUT_DTYPE_MISMATCH" in r["variants"][0]["metadata_issues"]
    with pytest.raises(mobile.MobileExportError, match="INPUT_DTYPE_MISMATCH"):
        mobile.MobileModelExporter().validate_package(p, require_precision=True)


def test_wrong_manifest_quantization_rejected(tmp_path):
    p = write_package(tmp_path, "int8", "int8", edit=lambda m, _: m["input"].update(quantization={"scale": 8., "zero_point": 0}))
    with pytest.raises(mobile.MobileExportError, match="INPUT_QUANTIZATION_MISMATCH"):
        mobile.MobileModelExporter().validate_package(p, require_precision=True)


def test_file_selection_uses_graph_not_filename_and_fails_closed(tmp_path):
    fp = tmp_path / "misleading_int8.tflite"
    quant = tmp_path / "no_precision_hint.tflite"
    fp.write_bytes(graph_bytes("fp32")); quant.write_bytes(graph_bytes("int8"))
    e = mobile.MobileModelExporter()
    assert e._locate_export_outputs(tmp_path, runtime="tflite", precision="int8") == (quant,)
    assert e._locate_export_outputs(tmp_path, runtime="tflite", precision="fp32") == (fp,)
    with pytest.raises(mobile.MobileExportError, match="potwierdzoną precyzją"):
        e._locate_export_outputs(fp, runtime="tflite", precision="int8")


def test_no_interpreter_or_model_runtime_needed(tmp_path):
    p = tmp_path / "toy.tflite"
    p.write_bytes(graph_bytes("int8"))
    with patch.object(mobile, "_tflite_interpreter_class", side_effect=AssertionError("runtime forbidden")):
        assert mobile.inspect_mobile_precision(p)["variants"][0]["graph"]["classification"] == "INT8"


def test_corrupt_file_checksum_still_rejected(tmp_path):
    p = write_package(tmp_path, edit=lambda m, _: m["variants"][0]["sha256"].update({"variants/tflite/model.tflite": "f"*64}))
    with pytest.raises(mobile.MobileExportError, match="SHA-256"):
        mobile.inspect_mobile_precision(p)


def rewrite_graph(data, edit):
    import flatbuffers
    s = precision.load_schema()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(bytearray(data), 0))
    edit(model, s)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


@pytest.mark.parametrize("error", ["nan", "zero_point", "weight_zero", "bias_scale", "float16", "tensor_index", "no_ops"])
def test_bad_tensor_metadata_and_graph_references_fail_closed(error):
    def edit(m, s):
        t = m.subgraphs[0].tensors
        if error == "nan": t[0].quantization.scale[0] = float("nan")
        if error == "zero_point": t[0].quantization.zeroPoint[0] = 128
        if error == "weight_zero": t[1].quantization.zeroPoint[0] = 3
        if error == "bias_scale": t[2].quantization.scale[0] = 5.
        if error == "float16": t[0].type = s.TensorType.FLOAT16
        if error == "tensor_index": m.subgraphs[0].operators[0].inputs[0] = 999
        if error == "no_ops": m.subgraphs[0].operators = []
    r = precision.analyze_tflite_bytes(rewrite_graph(graph_bytes("int8"), edit))
    assert r["classification"] == "UNVERIFIED", r
    assert not r["full_integer_graph"]
    json.dumps(r, allow_nan=False)


def test_structural_int32_shape_tensor_does_not_turn_fp32_into_mixed():
    # A true integer shape operation coexists with float conv, without fake INT8 evidence.
    def edit(m, s):
        import numpy as np
        tensor = s.TensorT(); tensor.name = "shape"; tensor.shape = np.array([4], dtype=np.int32)
        tensor.type = s.TensorType.INT32; tensor.buffer = 0
        m.subgraphs[0].tensors.append(tensor)
        code = s.OperatorCodeT(); code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.SHAPE; code.version = 1
        m.operatorCodes.append(code)
        op = s.OperatorT(); op.opcodeIndex = 1; op.inputs = np.array([0], dtype=np.int32); op.outputs = np.array([4], dtype=np.int32)
        m.subgraphs[0].operators.append(op)
    r = precision.analyze_tflite_bytes(rewrite_graph(graph_bytes("fp32"), edit))
    assert r["classification"] == "FP32", r["issues"]


def test_stale_embedded_report_cannot_be_resigned_to_another_graph(tmp_path):
    def edit(m, entries):
        variant = m["variants"][0]
        variant["precision"] = "int8"
        data = graph_bytes("int8")
        entries[variant["file"]] = data
        variant["sha256"][variant["file"]] = hashlib.sha256(data).hexdigest()
        m["input"]["data_type"] = "INT8"
    p = write_package(tmp_path, evidence=True, edit=edit)
    with pytest.raises(mobile.MobileExportError, match="dowód precyzji"):
        mobile.inspect_mobile_precision(p)


def test_new_evidence_rejects_wrong_input_shape(tmp_path):
    p = write_package(tmp_path, evidence=True, edit=lambda m, _: m["input"].update(width=512))
    with pytest.raises(mobile.MobileExportError, match="INPUT_SHAPE_MISMATCH"):
        mobile.inspect_mobile_precision(p)


def test_invalid_per_axis_activation_is_not_integer_evidence():
    def edit(m, s):
        import numpy as np
        q = m.subgraphs[0].tensors[0].quantization
        q.scale = np.array([.5, .5, .5], dtype=np.float32)
        q.zeroPoint = np.array([0, 0, 0], dtype=np.int64)
        q.quantizedDimension = 3
    r = precision.analyze_tflite_bytes(rewrite_graph(graph_bytes("int8"), edit))
    assert r["classification"] == "UNVERIFIED"
    assert any("PER_TENSOR" in issue for issue in r["issues"])


def test_new_manifest_evidence_is_bound_to_actual_file_sha(tmp_path):
    path = tmp_path / "model.tflite"
    path.write_bytes(graph_bytes("int8"))
    report = precision.analyze_tflite_file(path)
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    request = mobile.MobileExportRequest(checkpoint=checkpoint, destination=tmp_path / "unused.alprmodel",
        role="character", formats=("litert",), image_size=2)
    variant = mobile.ExportedVariant(id="tflite-int8", runtime="tflite", precision="int8", files=(path,),
        relative_files=("variants/tflite/model.tflite",),
        input_spec={"width": 2, "height": 2, "channels": 3, "layout": "NHWC", "data_type": "INT8"},
        output_spec={}, precision_report=report)
    kwargs = dict(request=request, model_id="synthetic", role="character", task="detect", width=2, height=2,
                  labels=["A"], variants=[variant], checkpoint=checkpoint, model_info={})
    e = mobile.MobileModelExporter()
    manifest = e._build_manifest(**kwargs)
    evidence = manifest["variants"][0]["graph_precision"]
    assert evidence["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["report_sha256"] == hashlib.sha256(mobile._precision_report_bytes(report)).hexdigest()
    assert evidence["execution_precision"] == "UNVERIFIED"
    path.write_bytes(graph_bytes("fp32"))
    with pytest.raises(mobile.MobileExportError, match="zmienił się"):
        e._build_manifest(**kwargs)


@pytest.mark.parametrize("kind", ["dynamic_signature", "custom_quantization"])
def test_unsupported_tensor_extensions_remain_unverified(kind):
    def edit(m, s):
        import numpy as np
        t = m.subgraphs[0].tensors[0]
        if kind == "dynamic_signature":
            t.shapeSignature = np.array([-1, 2, 2, 3], dtype=np.int32)
        else:
            t.quantization.detailsType = s.QuantizationDetails.CustomQuantization
            t.quantization.details = s.CustomQuantizationT()
    r = precision.analyze_tflite_bytes(rewrite_graph(graph_bytes("int8"), edit))
    assert r["classification"] == "UNVERIFIED", r["issues"]


def test_nested_legacy_package_collects_per_role_graphs_without_rewriting(tmp_path):
    # Assemble synthetic archives directly, without calling any exporter.
    mz = write_package(tmp_path, "mixed", "int8")
    with zipfile.ZipFile(mz) as z:
        manifest = json.loads(z.read("manifest.json")); child = mz.read_bytes()
    mt_manifest = json.loads(json.dumps(manifest))
    mt_manifest.update(role="plate", task="pose", model_id="synthetic-mt")
    mt_manifest["output"] = mobile._infer_output_spec([1, 300, 14], task="pose", class_count=1,
        keypoint_count=4, keypoint_dimensions=2, end2end_output=True,
        normalized_coordinates=True, confidence_threshold=.25, iou_threshold=.45)
    import io
    b = io.BytesIO()
    with zipfile.ZipFile(mz) as old, zipfile.ZipFile(b, "w") as new:
        for n in old.namelist():
            new.writestr(n, json.dumps(mt_manifest) if n == "manifest.json" else old.read(n))
    entries = {}; models = {}
    for role, raw, man in [("plate", b.getvalue(), mt_manifest), ("character", child, manifest)]:
        file = f"models/{role}/model.alprmodel"; side = f"models/{role}/manifest.json"
        entries[file] = raw; entries[side] = json.dumps(man).encode()
        models[role] = {"role": role, "task": man["task"], "model_id": man["model_id"], "schema": "alpr.model.v1",
                        "package_file": file, "manifest_file": side,
                        "sha256": {k: hashlib.sha256(entries[k]).hexdigest() for k in [file, side]}}
    package = tmp_path / "nested.alprmodel"
    root = {"schema": "alpr.package.v1", "package_id": "synthetic-alpr", "version": "1", "name": "Synthetic", "models": models}
    root["pipeline"] = [{"stage": name} for name in
                        ["plate_detection", "plate_rectification", "character_detection", "sequence_assembly"]]
    with zipfile.ZipFile(package, "w") as z:
        z.writestr("manifest.json", json.dumps(root))
        for n, raw in entries.items(): z.writestr(n, raw)
    before = package.read_bytes()
    result = mobile.inspect_mobile_precision(package)
    assert {r["role"] for r in result["variants"]} == {"plate", "character"}
    assert all(r["graph"]["classification"] == "MIXED" for r in result["variants"])
    assert package.read_bytes() == before
    with pytest.raises(mobile.MobileExportError, match="PRECISION_NOT_PROVEN"):
        mobile.MobileAlprPackageExporter().validate_package(package, require_precision=True)
