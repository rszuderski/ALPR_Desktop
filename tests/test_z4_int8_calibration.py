import json
import gc
import threading
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z4_int8_calibration as ui
from int8_calibration_fixture import fixture

_calibration_runtime = None

@pytest.fixture(scope="session")
def calibration_tk_runtime():
    # Match the application: one Tcl/Tk runtime, separate Toplevel per case.
    # Imported fixtures are registered separately for each test module; share
    # the actual interpreter too (Tk variables otherwise bind the wrong root).
    global _calibration_runtime
    if _calibration_runtime is None:
        _calibration_runtime = tk.Tk()
    r = _calibration_runtime
    r.withdraw()
    yield r
    if _calibration_runtime is r:
        r.destroy()
        _calibration_runtime = None


@pytest.fixture
def root(calibration_tk_runtime):
    r = calibration_tk_runtime
    r.withdraw()
    callback = r.report_callback_exception
    yield r
    for child in list(r.winfo_children()):
        child.destroy()
    r.report_callback_exception = callback
    r.update_idletasks()
    gc.collect()  # Dispose Tk-owned Python objects on the UI thread.


def pump(root, done):
    end = time.monotonic() + 10
    while not done() and time.monotonic() < end:
        root.update()
        time.sleep(.01)
    assert done()


def fill(dialog, f):
    for key, value in f["config"].items():
        dialog.values[key].set(str(value) if key != "role" else ("MT — sceny" if value == "plate" else "MZ — cropy"))
    dialog.approval.set(True)


@pytest.mark.parametrize("role", ["plate", "character"])
def test_real_tk_build_save_and_invalidate_manifest(root, tmp_path, role):
    f = fixture(tmp_path / "inputs", role)
    d = ui.open_int8_calibration_dialog(root)
    errors = []
    root.report_callback_exception = lambda *a: errors.append(a)
    fill(d, f)
    d.prepare_button.invoke()
    pump(root, lambda: not d._busy)
    assert d.result["status"] == "READY"
    assert "wybrane: 2" in d.status.get()
    assert d.result["input_fingerprint"] in d.details.get("1.0", "end")
    target = tmp_path / "manifest.json"
    with patch.object(ui.filedialog, "asksaveasfilename", return_value=str(target)):
        d.save_button.invoke()
        pump(root, lambda: not d._busy)
    assert json.loads(target.read_text(encoding="utf-8"))["split"] == "train"
    before = target.read_bytes()
    with patch.object(ui.filedialog, "asksaveasfilename", return_value=str(target)), patch.object(ui.messagebox, "showerror") as error:
        d.save_button.invoke()
        pump(root, lambda: not d._busy)
        error.assert_called_once()
    assert target.read_bytes() == before
    d.values["image_size"].set("64")
    assert d.result is None and str(d.save_button["state"]) == "disabled"
    assert not d.approval.get() and not d.details.get("1.0", "end").strip()
    assert not errors
    d.destroy()


def test_missing_final_test_is_visible_blocked_and_not_approval(root, tmp_path):
    f = fixture(tmp_path / "inputs")
    f["config"]["final_inventory"] = ""
    d = ui.Int8CalibrationDialog(root)
    fill(d, f)
    d.prepare_button.invoke()
    pump(root, lambda: not d._busy)
    assert d.result["status"] == "BLOCKED"
    assert "FINAL_TEST_INVENTORY_NOT_VERIFIED" in d.status.get()
    target = tmp_path / "blocked.json"
    with patch.object(ui.filedialog, "asksaveasfilename", return_value=str(target)):
        d.save_button.invoke()
        pump(root, lambda: not d._busy)
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "BLOCKED"
    d.destroy()


def test_failed_or_pending_edits_do_not_reuse_stale_result(root, tmp_path):
    f = fixture(tmp_path)
    d = ui.Int8CalibrationDialog(root)
    fill(d, f)
    d.prepare_button.invoke()
    pump(root, lambda: not d._busy)
    assert d.result
    d.values["run_id"].set("nonexistent")
    d.prepare_button.invoke()
    pump(root, lambda: not d._busy)
    assert d.result is None and "BLOCKED" in d.status.get()
    d.destroy()


def test_change_settings_while_worker_runs_discards_result(root, tmp_path):
    f = fixture(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = ui.build_manifest
    def slow(**kwargs):
        started.set()
        release.wait(5)
        return original(**kwargs)
    with patch.object(ui, "build_manifest", side_effect=slow):
        d = ui.Int8CalibrationDialog(root)
        fill(d, f)
        d.prepare_button.invoke()
        assert started.wait(2)
        d.values["limit"].set("1")
        release.set()
        pump(root, lambda: not d._busy)
        assert d.result is None and str(d.save_button["state"]) == "disabled"
        d.destroy()


def test_export_modal_calibration_is_contextual_and_no_export(root, tmp_path):
    from auto_annotation_tool.gui import z4_model_export as export
    from test_z4_calibration_context import candidate_fixture
    _,candidate=candidate_fixture(tmp_path)
    host = SimpleNamespace(app=SimpleNamespace(root=root, palette={}, show_assistant_message=Mock()), frame=root,
                           _build_mobile_model_export_path=lambda *_: tmp_path / "unused.alprmodel")
    def walk(w):
        yield w
        for child in w.winfo_children(): yield from walk(child)
    with patch.object(export, "_collect_mobile_export_candidates", return_value=[candidate]), \
         patch.object(export, "open_int8_calibration_dialog") as open_dialog, \
         patch.object(export.MobileModelExporter, "export", side_effect=AssertionError("export forbidden")):
        center = export._open_mobile_model_export_center(host);center.deiconify();root.update()
        tree=next(w for w in walk(center) if isinstance(w,ttk.Treeview) and len(w.get_children())==1)
        tree.see(candidate['iid']);root.update();box=tree.bbox(candidate['iid'],'#1');assert box
        tree.event_generate('<Button-1>',x=box[0]+5,y=box[1]+box[3]//2);root.update()
        next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Eksportuj zaznaczone').invoke();root.update()
        from auto_annotation_tool.gui.z4_int8_calibration import CalibrationModeControl
        slot=next(w for w in walk(center) if isinstance(w,CalibrationModeControl))
        slot.variables['litert'].set(False);slot.variables['int8'].set(True)
        next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Przygotuj manifest INT8 — MT').invoke()
        assert open_dialog.call_count==1
        assert open_dialog.call_args.kwargs['context'].role=='plate'
        center.destroy()


def test_model_slot_validates_manifest_and_never_auto_selects_yaml(tmp_path):
    from auto_annotation_tool.gui import z4_model_export as export
    from int8_calibration_fixture import save_plan
    f = fixture(tmp_path / "inputs", "character")
    p = tmp_path / "manifest.json"
    save_plan(f, p)
    candidate = {"target": "char", "best_weights": f["checkpoint"], "dataset_path": f["dataset"]}
    assert export._mobile_export_default_calibration_path(candidate) == ""
    assert export._mobile_export_calibration_problem(candidate, p) == ""
    assert "BLOCKED" in export._mobile_export_calibration_problem({**candidate, "target": "plate"}, p)
    assert "BLOCKED" in export._mobile_export_calibration_problem(candidate, f["dataset"] / "data.yaml")
