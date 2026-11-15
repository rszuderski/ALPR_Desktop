import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui import z4_analysis_ranking as ranking
from auto_annotation_tool.gui import z4_ranking_identity as evidence


def spin(root, condition=lambda: True, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.01)
    raise AssertionError("Tk condition timed out")


@pytest.fixture(scope="module")
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def host(root, tmp_path, monkeypatch):
    window = tk.Toplevel(root)
    window.withdraw()
    models = []
    for index in range(3):
        p = tmp_path / f"model_{index}.pt"
        p.write_bytes(bytes([index])*20)
        models.append(p)
    tracks = []
    for index in range(3):
        path = tmp_path / f"z2_original_{index}"
        path.mkdir()
        (path / "annotations.xml").write_text("<annotations/>")
        tracks.append({"path": str(path), "id": f"Z2-{index}", "type": "tablice", "split": "run",
                       "counts": {"total": 720+index}, "ready": True, "status": "gotowy"})
    h = SimpleNamespace(frame=ttk.Frame(window), app=SimpleNamespace(palette={}), _ranking_workspace=tmp_path,
        rank_data_dir=tk.StringVar(value=tracks[0]["path"]), rank_models_dir=tk.StringVar(value=str(tmp_path)),
        rank_scope_var=tk.StringVar(value="Globalne"), rank_split_var=tk.StringVar(value="test"),
        _get_ranking_task_target=lambda: "plate", _get_ranking_task_label=lambda: "Tablice (Pose)",
        _get_ranking_split_name=lambda: "test", _refresh_ranking_reference_ui=lambda: None,
        _open_ranking_advanced_modal=lambda: None,
        _ranking_participant_enabled_by_key={evidence.path_key(models[1]): False})
    h.models, h.tracks = models, tracks
    monkeypatch.setattr(ranking, "_collect_ranking_participant_candidates", lambda *args, **kw: list(models))
    monkeypatch.setattr(ranking, "_collect_ranking_track_candidates", lambda *args, **kw: list(tracks))
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args[1]))
    try:
        yield h
    finally:
        window.destroy()
        root.update()
        assert not errors, errors


def test_models_filter_sort_refresh_preserves_start_and_current_track(root, host):
    ui = ranking._open_ranking_participants_modal(host).catalog
    spin(root)
    before = dict(host._ranking_participant_enabled_by_key)
    original_track = host.rank_data_dir.get()
    ui.tree.selection_set(evidence.path_key(host.models[1]))
    spin(root)
    ui.query.set("model_0.pt")
    spin(root)
    assert len(ui.tree.get_children()) == 1
    assert not ui.tree.selection()
    ui.sort("checkpoint")
    ui.refresh_button.invoke()
    ui.query.set("")
    spin(root)
    assert host._ranking_participant_enabled_by_key == before
    assert ui.tree.item(evidence.path_key(host.models[1]), "text") == "NIE"
    assert host.rank_data_dir.get() == original_track
    ui.tree.selection_set(evidence.path_key(host.models[1]))
    ui.toggle_start()
    assert ranking._is_ranking_participant_enabled(host, host.models[1])
    ui.refresh_button.invoke()
    assert ui.tree.item(evidence.path_key(host.models[1]), "text") == "TAK"


def test_model_details_actual_sha_copy_and_missing_metadata(root, host):
    ui = ranking._open_ranking_participants_modal(host).catalog
    key = evidence.path_key(host.models[0])
    ui.tree.selection_set(key)
    digest = hashlib.sha256(host.models[0].read_bytes()).hexdigest()
    spin(root, lambda: digest in ui.details.get("1.0", "end"))
    assert "nieustalone" in ui.details.get("1.0", "end")
    assert ui.path_value.get() == str(host.models[0].resolve())
    ui.copy_button.invoke()
    assert ui.dialog.clipboard_get() == str(host.models[0].resolve())


def test_track_search_sort_preview_do_not_apply(root, host):
    ui = ranking._open_ranking_track_modal(host).catalog
    spin(root)
    before = host.rank_data_dir.get()
    ui.query.set("z2_original_2")
    spin(root)
    item = ui.tree.get_children()[0]
    ui.tree.selection_set(item)
    spin(root)
    assert host.rank_data_dir.get() == before
    assert "annotations.xml" in ui.details.get("1.0", "end")
    assert "722" in ui.details.get("1.0", "end")
    ui.sort("source_count")
    ui.refresh_button.invoke()
    assert host.rank_data_dir.get() == before
    ui.double_click(SimpleNamespace(x=20, y=5))  # A double click on a sort heading cannot apply a track.
    assert host.rank_data_dir.get() == before and ui.dialog.winfo_exists()
    ui.copy_button.invoke()
    assert ui.dialog.clipboard_get() == host.tracks[2]["path"]
    ui.accept_button.invoke()
    assert host.rank_data_dir.get() == host.tracks[2]["path"]


def test_character_split_is_draft_until_explicit_accept(root, host):
    host._get_ranking_task_target = lambda: "char"
    ui = ranking._open_ranking_track_modal(host).catalog
    ui.draft_split.set("val")
    ui.reload()
    assert host.rank_split_var.get() == "test"
    ui.tree.selection_set(evidence.path_key(host.tracks[1]["path"]))
    ui.accept_button.invoke()
    assert host.rank_split_var.get() == "val"


@pytest.mark.parametrize("opener", ["_open_ranking_track_modal", "_open_ranking_participants_modal"])
def test_windows_native_controls_and_resize(root, host, opener):
    dialog = getattr(ranking, opener)(host)
    ui = dialog.catalog
    spin(root)
    assert not dialog.transient() and dialog.resizable() == (1, 1)
    normal_width = ui.tree.winfo_width()
    if root.tk.call("tk", "windowingsystem") == "win32":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("user32")
        api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        api.GetAncestor.restype = wintypes.HWND
        api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        api.GetWindowLongW.restype = ctypes.c_long
        style = api.GetWindowLongW(api.GetAncestor(dialog.winfo_id(), 2), -16)
        assert style & 0x20000 and style & 0x10000 and style & 0x80000
        dialog.iconify()
        spin(root, lambda: dialog.state() == "iconic")
        assert getattr(ranking, opener)(host) is dialog
        spin(root, lambda: dialog.state() == "normal")
        dialog.state("zoomed")
        spin(root, lambda: dialog.state() == "zoomed")
        assert ui.tree.winfo_width() >= normal_width
        assert ui.close_button.winfo_rooty()+ui.close_button.winfo_height() <= dialog.winfo_rooty()+dialog.winfo_height()
        assert ui.xs.winfo_ismapped()
        dialog.state("normal")
        spin(root, lambda: dialog.state() == "normal")
    dialog.geometry("940x600")
    spin(root)
    assert ui.close_button.winfo_rooty()+ui.close_button.winfo_height() <= dialog.winfo_rooty()+dialog.winfo_height()
