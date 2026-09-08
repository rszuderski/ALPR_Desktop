"""D8 native Tk smoke: selection -> preflight -> save -> completion terminology.

Run from the repository root. Uses synthetic checkpoints, stubs conversion and
dependency checks, and confines all audit output to output/model_package_gui.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ctypes
import json
import os
import re
import time
import tkinter as tk
from tkinter import ttk
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from PIL import ImageGrab

from auto_annotation_tool.gui import z4_model_export as export
from auto_annotation_tool.gui.app_theme_definitions import THEME_DEFINITIONS
from test_mobile_export_project_sources import _ExportHost

OUT = ROOT / "output/model_package_gui"
OUT.mkdir(parents=True, exist_ok=True)


def guard(event, args):
    if event == "open":
        path, _, flags = args
        if isinstance(path, str) and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            if Path(path).resolve().is_relative_to(ROOT / "Workspace"):
                raise PermissionError("GUI audit must not change project files: " + path)


sys.addaudithook(guard)
root = tk.Tk()
root.withdraw()
errors, results = [], []
root.report_callback_exception = lambda *exc: errors.append(str(exc))
host = _ExportHost()
host.app = SimpleNamespace(root=root, palette=THEME_DEFINITIONS["dark_visual_cs"]["palette"])
host.frame = root
host._build_mobile_model_export_path = lambda *_a: OUT / "model.alprmodel"
host._build_mobile_alpr_package_export_path = lambda *_a, **_kw: OUT / "complete.alprmodel"
host._safe_model_export_slug = lambda value, fallback: re.sub(r"[^A-Za-z0-9._-]", "_", value) or fallback
host._json_safe_training_value = export._mobile_export_manifest_safe_value
host._build_mobile_export_metadata = lambda *_a: {}
host._append_train_log = lambda *_a: None


def pump(ms=150):
    done = tk.BooleanVar(root, False)
    root.after(ms, lambda: done.set(True))
    root.wait_variable(done)
    root.update_idletasks()


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def texts(widget):
    values = []
    for child in walk(widget):
        if "text" in child.keys():
            values.append(str(child.cget("text")))
        if isinstance(child, tk.Canvas):
            values.extend(child.itemcget(item, "text") for item in child.find_all() if child.type(item) == "text")
    return "\n".join(values)


def capture(widget, name):
    widget.lift()
    pump(300)
    get_ancestor = ctypes.windll.user32.GetAncestor
    get_ancestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    get_ancestor.restype = ctypes.c_void_p
    hwnd = get_ancestor(widget.winfo_id(), 2)
    redraw = ctypes.windll.user32.RedrawWindow
    redraw.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    redraw(hwnd, None, None, 0x181)
    pump(250)
    ImageGrab.grab(window=hwnd).save(OUT / f"{name}.png")


candidates = []
for role, marker, target in (("vehicle", "MP", "vehicle"), ("plate", "MT", "plate"), ("character", "MZ", "char")):
    path = OUT / f"{marker}.pt"
    path.write_bytes(b"GUI fixture, not a trained checkpoint")
    candidates.append(dict(target=target, role=role, target_label=marker, model_label=f"Test {marker}",
                           best_weights=path, task="pose" if role == "plate" else "detect", img_size=512,
                           model_version="yolo26n-pose" if role == "plate" else "yolo26n", file_size_mb=0.001,
                           project_name="Test kontraktu", project_root=str(OUT), source_label="Test"))

def run():
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(export, "_collect_mobile_export_candidates", return_value=candidates))
            stack.enter_context(patch.object(export, "mobile_export_requirement_status", return_value=(True, "fixture ready")))
            stack.enter_context(patch.object(export, "check_mobile_yolo_export_runtime", return_value=None))
            for cls in (export.MobileModelExporter, export.MobileAlprPackageExporter):
                stack.enter_context(patch.object(cls, "preflight", return_value=[]))
            singles = stack.enter_context(patch.object(export.MobileModelExporter, "export", side_effect=lambda request, **_: request.destination))
            packages = stack.enter_context(patch.object(export.MobileAlprPackageExporter, "export", side_effect=lambda request, **_: request.destination))
            save = stack.enter_context(patch.object(export.filedialog, "asksaveasfilename", return_value=str(OUT / "result.alprmodel")))
            info = stack.enter_context(patch.object(export.messagebox, "showinfo"))
            failure = stack.enter_context(patch.object(export.messagebox, "showerror"))
            stack.enter_context(patch.object(export.messagebox, "showwarning"))
            dialog = export._open_mobile_model_export_center(host)
            dialog.geometry("1340x850+20+20")
            pump(350)
            tree = next(w for w in walk(dialog) if isinstance(w, ttk.Treeview) and w.exists("mobile_export_candidate_0"))
            button = next(w for w in walk(dialog) if isinstance(w, ttk.Button) and w.cget("text") == "Eksportuj zaznaczone")
            selected = set()
            ids = {export._mobile_export_target_marker(candidate): candidate["iid"] for candidate in candidates}
            cases = [("MT",), ("MZ",), ("MP",), ("MT", "MZ"), ("MP", "MT", "MZ"), ("MP", "MT"), ("MP", "MZ")]
            for markers in cases:
                for marker in selected ^ set(markers):
                    tree.see(ids[marker])
                    pump()
                    x, y, width, height = tree.bbox(ids[marker], "#1")
                    tree.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
                    tree.event_generate("<ButtonRelease-1>", x=x + width // 2, y=y + height // 2)
                    pump()
                selected = set(markers)
                complete = {"MT", "MZ"} <= selected
                valid = len(selected) == 1 or complete
                assert bool(button.instate(["disabled"])) == (not valid), markers
                if not valid:
                    assert "Kompletny pakiet ALPR wymaga MT i MZ" in texts(dialog), markers
                    results.append({"selection": markers, "blocked": True})
                    continue
                schema = "alpr.package.v1" if complete else "alpr.model.v1"
                assert schema in texts(dialog), markers
                button.invoke()
                pump(250)
                modal = next(w for w in dialog.winfo_children() if isinstance(w, tk.Toplevel) and "Eksport" in w.title())
                assert modal.title() == ("Eksport kompletnego pakietu ALPR" if complete else "Eksport modelu mobilnego")
                assert schema in texts(modal)
                primary = next(w for w in walk(modal) if isinstance(w, ttk.Button) and w.cget("text") == "Sprawdź gotowość eksportu")
                primary.invoke()
                deadline = time.monotonic() + 20
                while not str(primary.cget("text")).startswith("Eksportuj") and time.monotonic() < deadline:
                    pump(200)
                noun = "pakiet ALPR" if complete else "model mobilny"
                assert primary.cget("text") == f"Eksportuj {noun} (.alprmodel)", (markers, texts(modal))
                if markers in (("MT",), ("MT", "MZ")):
                    capture(modal, "_".join(markers))
                singles.reset_mock()
                packages.reset_mock()
                info.reset_mock()
                primary.invoke()
                deadline = time.monotonic() + 20
                while not info.called and time.monotonic() < deadline:
                    pump(200)
                assert not failure.called, failure.call_args
                assert info.called, (markers, texts(modal))
                assert save.call_args.kwargs["title"] == f"Zapisz {noun} (.alprmodel)"
                assert info.call_args.args[0] == ("Pakiet ALPR gotowy" if complete else "Model mobilny gotowy")
                assert packages.call_count == int(complete) and singles.call_count == int(not complete)
                results.append({"selection": markers, "schema": schema, "button": primary.cget("text"), "save_title": save.call_args.kwargs["title"]})
                modal.destroy()
                pump()
            assert not errors, errors
            dialog.destroy()
    except BaseException:
        import traceback
        errors.append(traceback.format_exc())
    finally:
        (OUT / "results.json").write_text(json.dumps({"cases": results, "callback_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        root.destroy()
    print(json.dumps(results, ensure_ascii=True))

root.after(0, run)
root.mainloop()
if errors:
    raise SystemExit("\n".join(errors))
