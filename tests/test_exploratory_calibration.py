"""Synthetic-only exploratory authority, exact loaders, signed graph and archive regressions."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np
import pytest
from auto_annotation_tool.exporters import int8_calibration as strict
from auto_annotation_tool.exporters import exploratory_calibration as exp
from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from int8_calibration_fixture import fixture, holdout, write_json


def prepared(tmp_path, role="plate", **overrides):
    f = fixture(tmp_path / "inputs", role, duplicate=True)
    cfg = {k:v for k,v in f["config"].items() if k != "final_inventory"}
    cfg.update(operator_confirmation=True, checkpoint_sha256=strict.file_sha256(f["checkpoint"]))
    cfg.update(overrides)
    m = exp.build_manifest_v1(**cfg)
    p = write_json(tmp_path / "research.json", m)
    req = mobile.MobileExportRequest(checkpoint=f["checkpoint"], destination=tmp_path / "unused.alprmodel",
        role=role, formats=("litert",), image_size=32, quantizations=("int8",), calibration_manifest=p,
        calibration_mode="exploratory", exploratory_confirmation=True)
    return f, m, p, req


@pytest.mark.parametrize("role", ["plate", "character"])
def test_separate_authority_preserves_strict_blockade(tmp_path, role):
    f, m, p, req = prepared(tmp_path, role)
    assert m == exp.rebuild_manifest(m)
    assert exp.validate_manifest(p, role=role) == m
    assert m["status"] == "EXPLORATORY_ONLY" and m["source_audit"]["status"] == "BLOCKED"
    assert "FINAL_TEST_NOT_VERIFIED" in m["warnings"] and m["execution_precision"] == "UNVERIFIED"
    assert mobile.MobileModelExporter()._calibration_for_request(req) == m
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter()._approved_calibration(req)
    with pytest.raises(strict.CalibrationError):
        strict.validate_embedded_manifest(m)
    assert m["counts"]["duplicates_removed"] == 1


@pytest.mark.parametrize("change", ["schema", "status", "both"])
def test_renamed_strict_blocked_never_authorizes_research(tmp_path, change):
    f = fixture(tmp_path)
    m = strict.build_manifest(**{**f["config"], "final_inventory": ""})
    if change in ("schema", "both"): m["schema"] = exp.SCHEMA
    if change in ("status", "both"): m["status"] = "EXPLORATORY_ONLY"
    m["input_fingerprint"] = strict.fingerprint({k:v for k,v in m.items() if k != "input_fingerprint"})
    with pytest.raises((strict.CalibrationError, KeyError)):
        exp.validate_embedded_manifest(m)


@pytest.mark.parametrize("missing", ["operator_confirmation", "approved_by", "checkpoint_sha256", "eval_inventory"])
def test_missing_authority_or_eval_is_blocked(tmp_path, missing):
    _, m, p, req = prepared(tmp_path, **{missing: False if missing == "operator_confirmation" else ""})
    assert m["status"] == "BLOCKED"
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter()._calibration_for_request(req)


@pytest.mark.parametrize("role", ["plate", "character"])
def test_eval_scene_overlap_blocks_also_mz_crop(tmp_path, role):
    f,m,p,req = prepared(tmp_path, role)
    holdout(f["eval_path"], "EVAL396", [m["samples"][0]["source_scene_sha256"]])
    changed = exp.rebuild_manifest(m)
    assert changed["source_status"] == changed["status"] == "BLOCKED"
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter()._calibration_for_request(req)


@pytest.mark.parametrize("change", ["sample", "checkpoint", "run", "metadata", "inventory", "eval",
                                    "preprocessing", "fingerprint", "group", "versions"])
def test_changed_inputs_reject_live_validation(tmp_path, change):
    f,m,p,req = prepared(tmp_path, "character")
    if change == "sample": Path(m["samples"][0]["path"]).write_bytes(b"changed")
    elif change == "checkpoint": f["checkpoint"].write_bytes(b"changed")
    elif change in ("run", "inventory"):
        if change == "run": f["run"]["status"] = "running"
        else: f["inv"]["status"] = "partial"
        write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    elif change == "metadata": write_json(f["dataset"] / "metadata_manifest.json", {"items": []})
    elif change == "eval": write_json(f["eval_path"], {})
    else:
        if change == "preprocessing": m["preprocessing"]["color"] = "BGR"
        elif change == "group": m["samples"][0]["source_group_ids"] = ["forged"]
        elif change == "versions": m["runtime_versions"]["onnx2tf"] = "different"
        else: m["input_fingerprint"] = "0" * 64
        write_json(p,m)
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter()._calibration_for_request(req)


@pytest.mark.parametrize("field,value", [("image_size",64), ("confidence_threshold",.3), ("iou_threshold",.5),
                                        ("role","character"), ("calibration_fingerprint","a"*64),
                                        ("exploratory_confirmation",False), ("calibration_mode","strict"),
                                        ("calibration_mode","auto"), ("formats",("onnx",))])
def test_request_cannot_reuse_approval_for_different_input(tmp_path, field, value):
    _,_,_,req = prepared(tmp_path)
    if field == "exploratory_confirmation":
        # Historical v1 retains its original evidence, but export needs no second click.
        assert mobile.MobileModelExporter()._calibration_for_request(replace(req, **{field:value}))["status"] == "EXPLORATORY_ONLY"
    else:
        with pytest.raises(mobile.MobileExportError):
            mobile.MobileModelExporter()._calibration_for_request(replace(req, **{field:value}))


@pytest.mark.parametrize("role", ["plate", "character"])
def test_cross_split_removed_and_exact_pixels_reach_both_loaders(tmp_path, role):
    f,m,p,req = prepared(tmp_path, role)
    f["inv"]["source_scene_hashes"][0]["splits"].append("val")
    write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    m = exp.rebuild_manifest(m)
    exp.validate_embedded_manifest(m)
    assert m["counts"]["cross_split_rejected"] == 2  # source plus duplicate
    a,b = strict.ApprovedOnnxReader(m,"images"), strict.ApprovedLiteRTLoader(m)
    batches = list(b)
    for row,batch in zip(m["samples"],batches):
        np.testing.assert_array_equal(a.get_next()["images"],batch["img"].numpy().astype(np.float32)/255)
        assert batch["im_file"] == [row["path"]]
        assert row["source_scene_sha256"] != f["inv"]["source_scene_hashes"][0]["sha256"]
    assert a.get_next() is None
    a.assert_complete(); b.assert_complete()
    assert a.consumed == b.consumed == [r["sha256"] for r in m["samples"]]
    receipt = exp.usage_evidence(m)
    assert receipt["sample_sha256"] == b.consumed and receipt["source_scene_sha256"]
    if role == "character": assert receipt["source_group_ids"]


@pytest.mark.parametrize("split", ["val", "test"])
def test_yaml_cannot_redirect_research_to_holdout(tmp_path, split):
    f,m,_,_ = prepared(tmp_path)
    y = f["dataset"] / "data.yaml"
    y.write_text(f"train: images/{split}\n")
    f["run"]["training_dataset_snapshot"]["data_yaml_sha256"] = strict.file_sha256(y)
    write_json(f["history"], {"runs": {"synthetic": f["run"]}})
    with pytest.raises(strict.CalibrationError): exp.rebuild_manifest(m)


@pytest.mark.parametrize("kind", ["incomplete", "changed"])
def test_loaders_recheck_actual_bytes_and_complete_consumption(tmp_path, kind):
    _,m,_,_ = prepared(tmp_path)
    for cls,args in ((strict.ApprovedLiteRTLoader,()),(strict.ApprovedOnnxReader,("images",))):
        loader = cls(m,*args)
        if kind == "changed":
            Path(m["samples"][0]["path"]).write_bytes(b"modified before decoding")
            with pytest.raises(strict.CalibrationError):
                next(iter(loader)) if not args else loader.get_next()
        else:
            with pytest.raises(strict.CalibrationError): loader.assert_complete()


@pytest.mark.parametrize("kind", ["fp32", "int8", "mixed", "missing"])
def test_signed_artifact_selection_requires_actual_graph(tmp_path, kind, monkeypatch):
    monkeypatch.setattr(mobile.MobileModelExporter, "_precision_diagnostics_root", staticmethod(lambda: tmp_path / "diagnostics"))
    from tflite_precision_fixture import graph_bytes
    returned = tmp_path / "model_int8.tflite"
    returned.write_bytes(graph_bytes("mixed"))
    if kind != "missing":
        (tmp_path / "model_full_integer_quant.tflite").write_bytes(graph_bytes(kind))
    e = mobile.MobileModelExporter()
    if kind == "int8":
        p = e._exploratory_signed_output(returned)
        r = mobile.analyze_tflite_file(p)
        assert r["classification"] == "INT8" and r["execution_precision"] == "UNVERIFIED"
    else:
        with pytest.raises(mobile.MobileExportError): e._exploratory_signed_output(returned)


def test_installed_bridge_receives_research_pixels_without_default_loader(tmp_path):
    # Real installed Exporter method and instance callback; stub only converter/model.
    from ultralytics.engine import exporter as ul
    from tflite_precision_fixture import graph_bytes
    f,m,p,req = prepared(tmp_path)
    captured = {}
    class SyntheticModel:
        callbacks = {"on_export_start": []}
        def export(self, **kwargs):
            e = ul.Exporter.__new__(ul.Exporter)
            e.args = SimpleNamespace(**kwargs); e.file = tmp_path / "stub.pt"; e.imgsz = [32,32]
            e.metadata = {}; e.model = SimpleNamespace(model=[object()]); e.callbacks = self.callbacks
            e.export_onnx = Mock(return_value=tmp_path / "not_created.onnx")
            e.run_callbacks("on_export_start")
            result = e.export_saved_model()[0]
            return Path(result) / "stub_int8.tflite"
    def fake_bridge(source,target,**kwargs):
        captured["pixels"] = kwargs["images"].copy(); target.mkdir()
        (target / "stub_full_integer_quant.tflite").write_bytes(graph_bytes("int8"))
        return object()
    model = SyntheticModel(); original = model.callbacks
    with patch.dict(sys.modules,{"tensorflow":SimpleNamespace(__version__="2.19.0")}), \
         patch.object(ul,"check_requirements"), patch.object(ul,"check_version"), \
         patch.object(ul,"onnx2saved_model",side_effect=fake_bridge), \
         patch.object(ul.Exporter,"_add_tflite_metadata"), \
         patch.object(ul.Exporter,"get_int8_calibration_dataloader",side_effect=AssertionError("default loader forbidden")):
        result = mobile.MobileModelExporter()._run_ultralytics_export(model,runtime="tflite",precision="int8",request=req)
    assert model.callbacks is original and result.name.endswith("_full_integer_quant.tflite")
    expected = np.stack([strict.sample_rgb(r,m["preprocessing"]) for r in m["samples"]]).astype(np.float32)
    np.testing.assert_array_equal(captured["pixels"],expected)


def research_package(root,role,kind="int8"):
    import hashlib
    from test_tflite_precision import write_package
    root.mkdir(parents=True)
    f,m,p,req=prepared(root,role,image_size=2)
    # Consumption of synthetic images only, no converter/inference.
    loader=strict.ApprovedLiteRTLoader(m); list(loader); loader.assert_complete()
    def edit(man,entries):
        man.update(role=role,model_id="synthetic-"+role,task="pose" if role=="plate" else "detect",
                   source={"checkpoint_sha256":m["checkpoint"]["sha256"]})
        if role=="plate":
            man["output"]=mobile._infer_output_spec([1,300,14],task="pose",class_count=1,
                keypoint_count=4,keypoint_dimensions=2,end2end_output=True,normalized_coordinates=True,
                confidence_threshold=.25,iou_threshold=.45)
        raw=json.dumps(m).encode(); usage=json.dumps(exp.usage_evidence(m)).encode()
        entries.update({"calibration_manifest.json":raw,"calibration_usage.json":usage})
        man["calibration_evidence"]=dict(file="calibration_manifest.json",sha256=hashlib.sha256(raw).hexdigest(),
            input_fingerprint=m["input_fingerprint"],split="train",sample_count=len(m["samples"]),
            mode="exploratory",status="EXPLORATORY_ONLY",
            usage=dict(file="calibration_usage.json",sha256=hashlib.sha256(usage).hexdigest()))
        man["research_calibration"]=dict(status="EXPLORATORY_ONLY",final_test_status="NOT_VERIFIED",
                                         execution_precision="UNVERIFIED",limitations=m["limitations"])
    return write_package(root,kind,"int8",evidence=True,edit=edit)


def test_research_nested_package_preserves_role_proof_and_sha(tmp_path):
    import zipfile
    from auto_annotation_tool.ranking.mobile_package_experiments import read_alpr_package_manifest
    children={role:research_package(tmp_path/role,role) for role in ("plate","character")}
    e=mobile.MobileAlprPackageExporter()
    req=mobile.MobileAlprPackageRequest(destination=tmp_path/"nested.alprmodel",
        plate_package=children["plate"],character_package=children["character"])
    path=e.export(req); e.validate_package(path)
    root=read_alpr_package_manifest(path)
    assert root["schema"]=="alpr.package.v1"
    with zipfile.ZipFile(path) as z:
        for role,child in children.items():
            assert z.read(root["models"][role]["package_file"])==child.read_bytes()
            with zipfile.ZipFile(child) as c:
                manifest=json.loads(c.read("manifest.json"))
                assert manifest["research_calibration"]["status"]=="EXPLORATORY_ONLY"
                assert json.loads(c.read("calibration_usage.json"))["source_scene_sha256"]


@pytest.mark.parametrize("damage", ["mixed_graph", "usage_sha", "usage_rehashed", "missing_warning", "wrong_mode"])
def test_archive_cannot_claim_research_int8_with_bad_graph_or_proof(tmp_path,damage):
    import zipfile, hashlib
    p=research_package(tmp_path/"fixture","character","mixed" if damage=="mixed_graph" else "int8")
    with zipfile.ZipFile(p) as z: entries={n:z.read(n) for n in z.namelist()}
    man=json.loads(entries["manifest.json"])
    if damage in ("usage_sha","usage_rehashed"):
        entries["calibration_usage.json"]=b'{}'
        if damage=="usage_rehashed": man["calibration_evidence"]["usage"]["sha256"]=hashlib.sha256(b'{}').hexdigest()
    elif damage=="missing_warning": man.pop("research_calibration")
    elif damage=="wrong_mode": man["calibration_evidence"]["mode"]="strict"
    entries["manifest.json"]=json.dumps(man).encode()
    with zipfile.ZipFile(p,"w") as z:
        for n,b in entries.items(): z.writestr(n,b)
    with pytest.raises(mobile.MobileExportError):
        mobile.MobileModelExporter().validate_package(p,require_precision=True)


@pytest.mark.parametrize("fault", ["partial", "unfrozen", "missing_mz_binding", "unknown_augmentation"])
def test_incomplete_provenance_never_authorizes_research(tmp_path,fault):
    f,m,p,req=prepared(tmp_path,"character")
    if fault=="partial": f["inv"]["status"]="partial"
    elif fault=="unfrozen": f["run"]["training_dataset_snapshot"]["snapshot_source"]="reconstructed"
    elif fault=="missing_mz_binding": f["run"]["training_dataset_snapshot"]["manifests"].pop()
    else: write_json(f["dataset"]/"augmentation_manifest.json",{})
    write_json(f["history"],{"runs":{"synthetic":f["run"]}})
    with pytest.raises((strict.CalibrationError,ValueError)):
        exp.rebuild_manifest(m)


def test_native_export_writes_research_manifest_usage_and_both_variant_shas(tmp_path):
    import zipfile
    from tflite_precision_fixture import graph_bytes
    f,m,p,req=prepared(tmp_path,"character",image_size=2)
    req=replace(req,image_size=2,quantizations=("fp32","int8"))
    def yolo_stub(kind):
        # Five channels = 4 box coordinates + 1 class. The generic precision
        # fixture has two output channels and correctly fails YOLO inspection.
        import flatbuffers
        from auto_annotation_tool.exporters.tflite_precision import load_schema
        schema=load_schema()
        model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(graph_bytes(kind),0))
        tensors=model.subgraphs[0].tensors
        for index,shape in ((1,[5,1,1,3]),(2,[5]),(3,[1,2,2,5])):
            tensor=tensors[index]; tensor.shape=np.array(shape,dtype=np.int32)
            if index in (1,2):
                size=1 if tensor.type==schema.TensorType.INT8 else 4
                model.buffers[tensor.buffer].data=np.zeros(int(np.prod(shape))*size,dtype=np.uint8)
        builder=flatbuffers.Builder(1024); offset=model.Pack(builder)
        builder.Finish(offset,file_identifier=b'TFL3'); return bytes(builder.Output())
    class SyntheticModel:
        callbacks={"on_export_start":[]}
        def export(self,**kwargs):
            kind="int8" if kwargs.get("int8") else "fp32"
            if kind=="int8":
                exporter=SimpleNamespace(args=SimpleNamespace(**kwargs))
                for callback in self.callbacks["on_export_start"]: callback(exporter)
                list(exporter.get_int8_calibration_dataloader())
            out=tmp_path/("stub_int8.tflite" if kind=="int8" else "stub_float32.tflite")
            out.write_bytes(yolo_stub(kind))
            if kind=="int8": (tmp_path/"stub_full_integer_quant.tflite").write_bytes(yolo_stub("int8"))
            return out
    model=SyntheticModel()
    e=mobile.MobileModelExporter()
    info=dict(labels=["A"],class_count=1,keypoint_count=0,keypoint_dimensions=0)
    with patch.object(e,"preflight",return_value=[]), patch.object(e,"_load_yolo_model",return_value=model), \
         patch.object(e,"_inspect_yolo_checkpoint",return_value=info):
        result=e.export(req)
    e.validate_package(result,require_precision=True)
    with zipfile.ZipFile(result) as z:
        manifest=json.loads(z.read("manifest.json"))
        assert manifest["research_calibration"]["status"]=="EXPLORATORY_ONLY"
        assert manifest["source"]["checkpoint_sha256"]==strict.file_sha256(f["checkpoint"])
        usage=json.loads(z.read("calibration_usage.json"))
        assert usage["sample_sha256"]==[r["sha256"] for r in m["samples"]]
        assert {v["precision"] for v in manifest["variants"]}=={"fp32","int8"}
        assert all(v["graph_precision"]["execution_precision"]=="UNVERIFIED" for v in manifest["variants"])
