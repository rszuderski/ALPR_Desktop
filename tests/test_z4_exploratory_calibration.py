import threading
import tkinter as tk
from unittest.mock import patch
import pytest
from auto_annotation_tool.gui import z4_int8_calibration as ui
from auto_annotation_tool.exporters import exploratory_calibration as exp
from test_z4_int8_calibration import root, pump, fill, calibration_tk_runtime
from test_exploratory_calibration import prepared


def fill_research(d, f):
    d.mode.set(exp.LABEL)
    for key, value in f["config"].items():
        if key not in {"approved_by", "final_inventory"}:
            d.values[key].set(str(value) if key != "role" else ("MT" if value == "plate" else "MZ"))
    d.read_checkpoint_sha()
    pump(d, lambda: not d._busy)
    assert not d.approval.get() and not d.values["approved_by"].get()


@pytest.mark.parametrize("role", ["plate", "character"])
def test_native_mode_manifest_warning_and_defaults(root, tmp_path, role):
    f,_,_,_ = prepared(tmp_path,role)
    d = ui.Int8CalibrationDialog(root)
    d.values["role"].set("MT" if role == "plate" else "MZ")
    d.mode.set(exp.LABEL)
    assert d.values["image_size"].get() == ("512" if role == "plate" else "320")
    assert d.values["limit"].get() == "" and not hasattr(d, "research_check")
    assert "FINAL_TEST_NOT_VERIFIED" in d.status.get()
    fill_research(d,f)
    d.prepare_button.invoke(); pump(root,lambda:not d._busy)
    assert d.result["status"] == "EXPLORATORY_ONLY"
    assert "FINAL_TEST_NOT_VERIFIED" in d.status.get()
    assert d.result["source_audit"]["status"] == "BLOCKED"
    d.mode.set("Ścisła kalibracja INT8 — wymaga final_test")
    assert d.result is None and not hasattr(d, "research_check")
    d.destroy()


@pytest.mark.parametrize("key", ["history_path", "run_id", "role", "image_size", "limit", "eval_inventory", "final_inventory"])
def test_native_source_change_resets_all_authority(root,tmp_path,key):
    f,_,_,_ = prepared(tmp_path)
    d = ui.Int8CalibrationDialog(root); fill_research(d,f)
    d.values[key].set("changed")
    assert not d.approval.get() and not hasattr(d, "research_check") and d.result is None
    assert str(d.save_button["state"]) == "disabled"
    d.destroy()


@pytest.mark.parametrize("key", ["checkpoint_sha256", "confidence_threshold", "iou_threshold"])
def test_native_runtime_change_resets_all_authority(root,tmp_path,key):
    f,_,_,_ = prepared(tmp_path)
    d = ui.Int8CalibrationDialog(root); fill_research(d,f)
    d.research_values[key].set("changed")
    assert not d.approval.get() and not hasattr(d, "research_check")
    d.destroy()


def test_close_native_dialog_during_worker_does_not_call_destroyed_widgets(root,tmp_path):
    f,_,_,_ = prepared(tmp_path)
    started,release,finished = threading.Event(),threading.Event(),threading.Event()
    errors=[]; root.report_callback_exception=lambda *args:errors.append(args)
    original=exp.build_manifest
    def slow(**kwargs):
        started.set(); release.wait(5)
        try: return original(**kwargs)
        finally: finished.set()
    d=ui.Int8CalibrationDialog(root); fill_research(d,f)
    with patch.object(exp,"build_manifest",side_effect=slow):
        d.prepare_button.invoke(); assert started.wait(2)
        d.destroy(); release.set(); assert finished.wait(5)
        root.update()
    assert d._closed and d._poll_id is None and not errors


@pytest.mark.parametrize("key", ["calibration", "imgsz", "conf", "iou", "int8", "candidate_identity"])
def test_real_export_slot_has_no_confirmation_for_any_input(root,key):
    values={name:tk.StringVar(root,"initial") for name in ("calibration","imgsz","conf","iou","int8","candidate_identity")}
    control=ui.CalibrationModeControl(root,values)
    assert ui.calibration_request_options(values)==dict(calibration_mode="strict")
    control.choice.set(exp.LABEL)
    assert "EXPLORATORY_ONLY" in control.notice["text"]
    assert not hasattr(control, "check")
    assert ui.calibration_request_options(values)==dict(calibration_mode="exploratory")
    values[key].set("new")
    assert "exploratory_confirmation" not in values
    assert ui.calibration_request_options(values)==dict(calibration_mode="exploratory")
    control.destroy()


def test_export_slot_live_manifest_validation_requires_explicit_mode(root,tmp_path):
    from auto_annotation_tool.gui.z4_model_export import _mobile_export_calibration_problem
    f,m,p,_=prepared(tmp_path)
    candidate=dict(target="plate",best_weights=f["checkpoint"])
    options=dict(calibration_mode="exploratory",exploratory_confirmation=True,imgsz=32,conf=.25,iou=.45)
    assert not _mobile_export_calibration_problem(candidate,p,options=options)
    assert "BLOCKED" in _mobile_export_calibration_problem(candidate,p)
    assert not _mobile_export_calibration_problem(candidate,p,options={**options,"exploratory_confirmation":False})
    for change in (dict(imgsz=64),dict(conf=.4)):
        assert "BLOCKED" in _mobile_export_calibration_problem(candidate,p,options={**options,**change})
