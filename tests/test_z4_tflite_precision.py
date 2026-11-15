import json
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z4_tflite_precision as ui
from tflite_precision_fixture import graph_bytes
from test_z4_int8_calibration import root, calibration_tk_runtime


def pump(root, done, timeout=8):
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    assert done()


def test_real_tk_displays_mixed_sha_and_saves_only_separate_report(root, tmp_path):
    p = tmp_path / "int8_in_name.tflite"
    data = graph_bytes("mixed")
    p.write_bytes(data)
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    dialog = ui.open_tflite_precision_dialog(root)
    with patch.object(ui.filedialog, "askopenfilename", return_value=str(p)):
        dialog.choose_button.invoke()
    pump(root, lambda: dialog.result is not None)
    row = dialog.table.item("0", "values")
    assert row[3] == "MIXED"
    assert dialog.result["file_sha256"] in dialog.fingerprint.get()
    assert "UNVERIFIED" in dialog.details.get("1.0", "end")
    target = tmp_path / "precision.json"
    with patch.object(ui.filedialog, "asksaveasfilename", return_value=str(target)):
        dialog.save_button.invoke()
    assert json.loads(target.read_text(encoding="utf-8"))["variants"][0]["graph"]["classification"] == "MIXED"
    with patch.object(ui.filedialog, "asksaveasfilename", return_value=str(p)), \
         patch.object(ui.messagebox, "showerror") as error:
        dialog.save_button.invoke()
        error.assert_called_once()
    assert p.read_bytes() == data
    assert not errors
    dialog.destroy()


def test_failed_selection_clears_stale_evidence(root, tmp_path):
    p = tmp_path / "model.tflite"
    p.write_bytes(graph_bytes("fp32"))
    d = ui.TflitePrecisionDialog(root)
    d.inspect(p)
    pump(root, lambda: d.result is not None)
    d.inspect(tmp_path / "missing.tflite")
    pump(root, lambda: str(d.choose_button["state"]) == "normal")
    assert d.result is None
    assert str(d.save_button["state"]) == "disabled"
    assert not d.table.get_children()
    assert "Nie udało" in d.status.get()
    d.destroy()


def test_close_during_worker_has_no_tk_call_from_worker(root, tmp_path):
    import threading
    started, release = threading.Event(), threading.Event()
    def slow(_):
        started.set()
        release.wait(3)
        raise ValueError("closed")
    with patch.object(ui, "inspect_mobile_precision", side_effect=slow):
        d = ui.TflitePrecisionDialog(root)
        d.inspect(tmp_path / "unused.tflite")
        assert started.wait(2)
        d.destroy()
        release.set()
        root.update()


def test_existing_export_center_exposes_inspection_without_export(root, tmp_path):
    from auto_annotation_tool.gui import z4_model_export as export
    p = tmp_path / "synthetic.pt"
    p.write_bytes(b"synthetic checkpoint placeholder")
    candidate = dict(best_weights=p, model_label="Synthetic MT", role="plate", target="plate", task="pose",
                     model_version="YOLO26n-pose", training_provenance={"provenance_version": 1})
    host = SimpleNamespace(app=SimpleNamespace(root=root, palette={}, show_assistant_message=Mock()), frame=root,
                           _build_mobile_model_export_path=lambda *_: tmp_path / "unused.alprmodel")
    def walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from walk(child)
    with patch.object(export, "_collect_mobile_export_candidates", return_value=[candidate]), \
         patch.object(export, "open_tflite_precision_dialog") as open_dialog, \
         patch.object(export.MobileModelExporter, "export", side_effect=AssertionError("export forbidden")):
        dialog = export._open_mobile_model_export_center(host)
        try:
            root.update()
            button = next(w for w in walk(dialog) if isinstance(w, ttk.Button) and w["text"] == "Sprawdź precyzję TFLite")
            button.invoke()
            open_dialog.assert_called_once_with(dialog)
        finally:
            dialog.destroy()
