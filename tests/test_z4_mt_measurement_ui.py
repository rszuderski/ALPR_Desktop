"""Native MT executor controls use explicit clicks and immutable raw results."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image
import pytest

from auto_annotation_tool.gui import z4_mt_results as ui
from auto_annotation_tool.gui import z4_mt_geometry as geometry
from auto_annotation_tool.image_orientation import inspect_oriented_image
from auto_annotation_tool.ranking.mt_eval_metrics import aggregate


def spin(root, predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Native MT UI did not reach the expected state")


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def evidence(tmp_path):
    scenes, rows = [], []
    for index, domain in enumerate(("DAY", "NIGHT")):
        path = tmp_path / (domain + ".png")
        Image.new("RGB", (160, 100), (30+index, 50, 70)).save(path)
        _pixels, info = inspect_oriented_image(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        quad = [[20, 20], [120, 20], [120, 60], [20, 60]]
        plate = {"plate_id": f"fixture-{index}", "bbox": [20, 20, 120, 60], "quad": quad,
                 "bbox_eligible": True, "pose_eligible": True, "pose_evaluation_status": "ELIGIBLE", "pose_exclusion": None}
        scene = {"file": path.name, "sha256": sha, "domain": domain, "image": info, "plates": [plate]}
        scenes.append(scene)
        predictions = [{"index": 0, "bbox": plate["bbox"], "confidence": .9, "class_id": 0,
                        "keypoints": [[x, y, .6+i*.1] for i, (x, y) in enumerate(quad)]}] if index == 0 else []
        rows.append({"scene_sha256": sha, "file": path.name, "domain": domain, "width": 160, "height": 100,
                     "predictions": predictions})
    protocol = {"protocol_sha256": "p" * 64, "geometry_approval_contract_sha256": "d" * 64,
                "metric_counts": {"scenes": 396, "bbox": 479, "pose": 478, "ocr": 449},
                "inference": {"imgsz": 512, "conf": .25, "iou": .7, "max_det": 300, "device": "cpu",
                              "half": False, "augment": False, "batch": 1, "end2end": True},
                "models": [{"label": name} for name in ("MT-n", "MT-DAY", "MT-NIGHT")]}
    preflight = {"review_sha256": "r" * 64, "source_scene_root": str(tmp_path), "scenes": scenes,
                 "review_scene_hashes": [s["sha256"] for s in scenes], "status": "READY_FOR_REVIEW",
                 "geometry_issues": [], "blocking_geometry_issues": [], "metric_counts": protocol["metric_counts"]}
    prepared = {"protocol": protocol, "preflight": preflight,
                "decision": {"decision": "APPROVED", "contract_sha256": "d" * 64, "review_sha256": "r" * 64,
                             "reviewed_scene_hashes": preflight["review_scene_hashes"]},
                "approval_dir": str(tmp_path / "approval"), "output_root": str(tmp_path / "runs")}
    complete = {"run_dir": str(tmp_path / "runs" / "TEST_ONLY"), "protocol": protocol, "preflight": preflight,
                "status": "COMPLETE", "source_files": {},
                "models": [{"label": name, "status": "COMPLETE", "rows": deepcopy(rows), "metrics": aggregate(rows, scenes)}
                           for name in ("MT-n", "MT-DAY", "MT-NIGHT")]}
    partial = deepcopy(complete)
    partial["status"] = "PARTIAL"
    partial["models"][0].update(status="PARTIAL", rows=rows[:1], metrics=aggregate(rows[:1], scenes))
    for model in partial["models"][1:]:
        model.update(status="NOT_RUN", rows=[], metrics=None)
    return SimpleNamespace(prepared=prepared, complete=complete, partial=partial, rows=rows)


@pytest.fixture
def native(tk_root, tmp_path, evidence, monkeypatch):
    calls = []
    block_run = threading.Event()
    block_run.set()
    run_finished = threading.Event()

    def prepare(selection_dir, **kwargs):
        calls.append(("prepare", kwargs))
        kwargs["progress"]({"stage": "prepare", "message": "TEST: no inference"})
        return deepcopy(evidence.prepared)

    def run(prepared, **kwargs):
        calls.append(("run", kwargs))
        kwargs["progress"]({"stage": "inference", "message": "TEST scene", "model": "MT-n", "scene": 1,
                            "total_scenes": 396, "completed": 1, "total": 1188})
        deadline = time.monotonic() + 5
        while not block_run.is_set() and not kwargs["cancelled"]() and time.monotonic() < deadline:
            time.sleep(.005)
        output = deepcopy(evidence.partial if kwargs["cancelled"]() else evidence.complete)
        run_finished.set()
        return output

    fake = SimpleNamespace(prepare_campaign=prepare, run_campaign=run,
        list_campaigns=lambda root: [{"run_dir": evidence.partial["run_dir"], "status": "PARTIAL", "created_at": "TEST",
                                     "models_completed": 0, "scenes_completed": 1, "protocol_sha256": "p"*64}],
        load_campaign=lambda path: deepcopy(evidence.partial))
    monkeypatch.setattr(ui, "runner_api", lambda: fake)
    window = tk.Toplevel(tk_root)
    window.withdraw()
    errors = []
    tk_root.report_callback_exception = lambda *args: errors.append(str(args[1]))
    panel = ui.MTMeasurementPanel(window, tmp_path / "selection")
    panel.pack(fill="both", expand=True)
    tk_root.update()
    context = SimpleNamespace(root=tk_root, window=window, panel=panel, calls=calls, fake=fake,
                              block_run=block_run, run_finished=run_finished, errors=errors)
    try:
        yield context
    finally:
        block_run.set()
        if window.winfo_exists():
            window.destroy()
        tk_root.update()
        assert not errors, errors


def test_preparation_verifies_gate_but_never_launches_model(native):
    panel = native.panel
    assert panel.start_button.instate(["disabled"])
    assert not native.calls
    panel.prepare_button.invoke()
    spin(native.root, lambda: panel.prepared is not None and not panel.busy)
    assert [name for name, _ in native.calls] == ["prepare"]
    assert panel.can_start and not panel.start_button.instate(["disabled"])
    assert "imgsz=512" in panel.protocol_text.get() and "conf=0.25" in panel.protocol_text.get()
    assert "IoU NMS=0.7 nieaktywny" in panel.protocol_text.get()
    assert "MT-n → MT-DAY → MT-NIGHT" in panel.protocol_text.get()
    assert "479" in panel.protocol_text.get() and "478" in panel.protocol_text.get()
    panel.device.set("0")
    panel.device_choice.event_generate("<<ComboboxSelected>>")
    native.root.update()
    assert panel.prepared is None and panel.start_button.instate(["disabled"])
    assert len(native.calls) == 1


def test_missing_or_stale_approval_blocks_start(native):
    def rejected(*args, **kwargs):
        raise ValueError("EXIF_OPERATOR_APPROVAL_REQUIRED")
    native.fake.prepare_campaign = rejected
    native.panel.prepare_button.invoke()
    spin(native.root, lambda: not native.panel.busy)
    assert native.panel.start_button.instate(["disabled"])
    assert "EXIF_OPERATOR_APPROVAL_REQUIRED" in native.panel.status.get()
    assert not any(name == "run" for name, _ in native.calls)


def test_explicit_start_progress_cancel_and_verified_resume(native, evidence):
    panel = native.panel
    panel.prepare_button.invoke()
    spin(native.root, lambda: panel.prepared is not None and not panel.busy)
    native.block_run.clear()
    panel.start_button.invoke()
    spin(native.root, lambda: panel.running and "1/396" in panel.progress_text.get())
    assert panel.start_button.instate(["disabled"])
    assert float(panel.progress.cget("value")) == 1
    panel.cancel_button.invoke()
    spin(native.root, lambda: not panel.busy)
    assert panel.campaign["status"] == "PARTIAL"
    assert panel.can_resume
    native.block_run.set()
    panel.resume_button.invoke()
    spin(native.root, lambda: not panel.busy)
    runs = [options for name, options in native.calls if name == "run"]
    assert len(runs) == 2
    assert runs[1]["resume_dir"] == evidence.partial["run_dir"]
    assert panel.campaign["status"] == "COMPLETE"
    assert not panel.can_resume


def test_history_load_never_starts_or_prepares_inference(native):
    panel = native.panel
    panel.refresh_button.invoke()
    spin(native.root, lambda: panel.history_loaded and not panel.busy)
    panel.history_tree.selection_set("0")
    spin(native.root, lambda: panel.campaign is not None and not panel.busy)
    assert panel.campaign["status"] == "PARTIAL"
    assert not native.calls
    assert panel.start_button.instate(["disabled"])
    assert panel.resume_button.instate(["disabled"])


def test_native_results_keep_missing_metrics_null_and_filter_domains(native, evidence):
    panel = native.panel
    panel.campaign = deepcopy(evidence.partial)
    panel.refresh_results()
    assert len(panel.result_tree.get_children()) == 3
    not_run = panel.result_tree.item("1", "values")
    assert not_run[1] == "NOT_RUN" and all(value == "—" for value in not_run[2:])
    assert "1/2" in panel.result_tree.item("0", "values")[1]
    panel.domain.set("NIGHT")
    panel.domain_choice.event_generate("<<ComboboxSelected>>")
    native.root.update()
    assert panel.result_tree.item("0", "values")[2] == "—"
    panel.domain.set("DAY")
    panel.refresh_results()
    assert panel.result_tree.item("0", "values")[4] == "100,00%"
    assert panel.result_tree.item("0", "values")[-2:] == ("0,00", "0,0000")
    panel.chart_kind.set("Rozkład błędu rogu [px]")
    panel.draw_chart()
    labels = [panel.chart.itemcget(item, "text") for item in panel.chart.find_all() if panel.chart.type(item) == "text"]
    assert "MT-n (n=4)" in labels and "MT-DAY (n=0)" in labels
    assert "NIEPOLICZONE" in panel.metrics_note.get()


def test_result_scope_note_tracks_complete_partial_and_cleared_history(native, evidence):
    panel = native.panel
    panel.campaign = deepcopy(evidence.complete)
    panel.refresh_results()
    assert panel.metrics_note.get().startswith("COMPLETE: wszystkie sceny przetworzone.")
    assert "PARTIAL:" not in panel.metrics_note.get()
    panel.campaign = deepcopy(evidence.partial)
    panel.refresh_results()
    assert panel.metrics_note.get().startswith("PARTIAL:")
    assert "oczekujące nie są FN" in panel.metrics_note.get()
    panel.campaign = None
    panel.refresh_results()
    assert panel.metrics_note.get().startswith("Brak pomiaru.")
    assert not panel.result_tree.get_children()


def test_error_overlay_uses_original_keypoint_order_and_verified_pixels(native, evidence):
    panel = native.panel
    campaign = deepcopy(evidence.complete)
    row = campaign["models"][0]["rows"][0]
    raw = row["predictions"][0]["keypoints"]
    raw[1], raw[3] = raw[3], raw[1]
    original = deepcopy(raw)
    campaign["models"][0]["metrics"] = aggregate(campaign["models"][0]["rows"], campaign["preflight"]["scenes"])
    panel.campaign = campaign
    panel.refresh_results()
    assert panel.error_rows
    panel.error_tree.selection_set("0")
    spin(native.root, lambda: panel.active_error is not None)
    assert panel.active_error[1]["predictions"][0]["keypoints"] == original
    panel.error_canvas.delete("preview_overlay")
    panel.draw_error_overlay(panel.error_canvas)
    labels = [panel.error_canvas.itemcget(item, "text") for item in panel.error_canvas.find_withtag("preview_overlay")
              if panel.error_canvas.type(item) == "text"]
    assert all(label in labels for label in ("k0", "k1", "k2", "k3", "TL", "TR", "BR", "BL"))
    assert campaign["models"][0]["rows"][0]["predictions"][0]["keypoints"] == original
    assert not native.calls


def test_changed_error_image_is_not_displayed(native, evidence):
    panel = native.panel
    panel.campaign = deepcopy(evidence.complete)
    panel.only_errors.set(False)
    panel.refresh_results()
    scene = panel.campaign["preflight"]["scenes"][0]
    path = Path(panel.campaign["preflight"]["source_scene_root"]) / scene["file"]
    path.write_bytes(path.read_bytes() + b"TEST_CHANGED_SOURCE")
    panel.error_tree.selection_set("0")
    spin(native.root, lambda: panel.error_note.get().startswith("FAIL / STOP"))
    assert panel.active_error is None
    assert panel.error_canvas.original_image is None


def test_export_routes_to_native_report_adapter_without_model_calls(native, evidence, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ui.filedialog, "askdirectory", lambda **kw: str(tmp_path / "reports"))
    monkeypatch.setattr(ui, "report_api", lambda: SimpleNamespace(export_mt_report=lambda *args, **kwargs:
                        calls.append((args, kwargs)) or tmp_path / "reports" / "TEST"))
    panel = native.panel
    panel.campaign = deepcopy(evidence.partial)
    panel.update_controls()
    panel.export_button.invoke()
    spin(native.root, lambda: not panel.busy)
    assert calls[0][0] == (evidence.partial["run_dir"], tmp_path / "reports")
    assert "Raport zapisany" in panel.status.get()
    assert not native.calls


def test_close_requests_cancel_without_tk_calls_from_worker(native):
    panel = native.panel
    panel.prepare_button.invoke()
    spin(native.root, lambda: panel.can_start)
    native.block_run.clear()
    panel.start_button.invoke()
    spin(native.root, lambda: panel.running and any(name == "run" for name, _ in native.calls))
    native.window.destroy()
    assert panel.cancelled.is_set()
    assert native.run_finished.wait(2)
    assert not panel.alive


def test_measurement_tab_is_added_to_existing_notebook_without_auto_launch(tk_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "runner_api", lambda: SimpleNamespace(list_campaigns=lambda root: calls.append("history") or []))
    window = tk.Toplevel(tk_root)
    window.withdraw()
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)
    original = ttk.Frame(notebook)
    notebook.add(original, text="MZ")
    panel = geometry.GeometryReviewPanel(notebook, tmp_path)
    notebook.add(panel, text="MT geometria")
    notebook.select(panel)
    try:
        spin(tk_root, lambda: panel.measurement_panel is not None)
        assert notebook.select() == str(panel)
        assert panel.measurement_panel.master is notebook
        assert len(notebook.tabs()) == 3
        assert panel.measure_button.instate(["disabled"])
        assert not panel.measure_button.cget("command")
        assert not calls
    finally:
        window.destroy()
        tk_root.update()


def test_prepared_protocol_displays_existing_approval_in_geometry_tab(tk_root, tmp_path, evidence, monkeypatch):
    monkeypatch.setattr(ui, "runner_api", lambda: SimpleNamespace(
        list_campaigns=lambda root: [], prepare_campaign=lambda *a, **kw: deepcopy(evidence.prepared)))
    window = tk.Toplevel(tk_root)
    window.withdraw()
    notebook = ttk.Notebook(window)
    notebook.pack()
    host = geometry.GeometryReviewPanel(notebook, tmp_path)
    notebook.add(host, text="MT geometria")
    try:
        spin(tk_root, lambda: host.measurement_panel is not None)
        panel = host.measurement_panel
        panel.prepare_button.invoke()
        spin(tk_root, lambda: panel.can_start)
        assert host.report == evidence.prepared["preflight"]
        assert host.decision == evidence.prepared["decision"]
        assert host.reviewed == set(evidence.prepared["decision"]["reviewed_scene_hashes"])
        assert host.status.get().startswith("POLITYKA EXIF ZATWIERDZONA")
        assert not host.measure_button.instate(["disabled"])
        assert host.measure_button.cget("command")
        assert not list(tmp_path.glob("**/geometry_policy.json"))
        host.messages.put(("error", "TEST_SOURCE_CHANGED"))
        spin(tk_root, lambda: host.report is None)
        assert panel.prepared is None and panel.start_button.instate(["disabled"])
        assert host.measure_button.instate(["disabled"])
    finally:
        window.destroy()
        tk_root.update()
