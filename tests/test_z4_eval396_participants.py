"""Horses and Results are separate native windows; viewing models is read-only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui import z4_eval396 as view
from auto_annotation_tool.gui import z4_analysis_ranking as ranking
from auto_annotation_tool.ranking import eval396 as contract


@pytest.fixture
def model_selection(tmp_path, monkeypatch):
    models = [{"label": label, "role": "plate" if label.startswith("MT-") else "character",
               "run_id": "test-"+label, "parent_run_id": "" if label in {"MT-n", "MZ-s"} else "test-base",
               "checkpoint_file": str(tmp_path / label / "best.pt"), "checkpoint_sha256": str(index)*64}
              for index, label in enumerate(contract.MODELS)]
    lineage = tmp_path / "model_lineage.json"
    lineage.write_bytes(contract.canonical(list(reversed(models))))
    manifest = {"schema": "emzdn01.eval_scene_selection.v1", "selection_fingerprint_sha256": contract.SELECTION_SHA,
                "files": {"model_lineage.json": hashlib.sha256(lineage.read_bytes()).hexdigest()}}
    file = tmp_path / "selection_manifest.json"
    file.write_bytes(contract.canonical(manifest))
    monkeypatch.setattr(contract, "SELECTION_MANIFEST_SHA", hashlib.sha256(file.read_bytes()).hexdigest())
    return tmp_path, models


def test_participants_are_pinned_ordered_and_do_not_need_checkpoints(model_selection):
    path, expected = model_selection
    before = {p: p.read_bytes() for p in path.iterdir()}
    assert view.participant_models(path) == expected
    assert view.participant_models(path / "selection_manifest.json") == expected
    assert {p: p.read_bytes() for p in path.iterdir()} == before
    assert not list(path.glob("**/*.pt"))  # No model loading, inference or downloads.


@pytest.mark.parametrize("filename", ["selection_manifest.json", "model_lineage.json"])
def test_changed_participant_sources_are_rejected(model_selection, filename):
    path, _ = model_selection
    with (path / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(contract.EvaluationImportError, match="SHA_MISMATCH"):
        view.participant_models(path)


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def native(tk_root, model_selection, monkeypatch):
    path, models = model_selection
    root = tk_root
    window = tk.Toplevel(root)
    window.withdraw()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args[1]))
    host = SimpleNamespace(frame=ttk.Frame(window), rank_data_dir=tk.StringVar(value=str(path)),
                           _get_ranking_task_target=lambda: "plate")
    def forbidden(*args, **kwargs):
        raise AssertionError("Viewing participants must not open/load results or geometry")
    monkeypatch.setattr(view, "open_results", forbidden)
    monkeypatch.setattr(view, "load_eval396", forbidden)
    try:
        yield root, host, models
    finally:
        window.destroy()
        root.update()
        assert not errors


def test_real_horses_button_opens_models_and_keeps_existing_results(native):
    root, host, models = native
    host._eval396_dialog = tk.Toplevel(host.frame)
    host._eval396_dialog.withdraw()
    original_results = host._eval396_dialog
    button = ttk.Button(host.frame, text="[ KONIE ] Uczestnicy",
                        command=lambda: ranking._open_ranking_participants_modal(host))
    button.invoke()
    root.update()
    dialog = host._eval396_participants_dialog
    assert dialog is not original_results and original_results.winfo_exists()
    assert "modele" in dialog.title() and "wyniki" not in dialog.title()
    assert dialog.participant_table.get_children() == tuple(contract.MODELS[:3])
    assert dialog.participant_weights.get() == models[0]["checkpoint_file"]
    assert dialog.participant_sha.get() == models[0]["checkpoint_sha256"]
    assert not hasattr(dialog, "evaluation_notebook")
    button.invoke()
    root.update()
    assert host._eval396_participants_dialog is dialog  # One participant window, separate from results.
    assert host._eval396_dialog is original_results


def test_real_model_filters_and_selection_preserve_lineage(native):
    root, host, models = native
    dialog = ranking._open_ranking_participants_modal(host)
    root.update()
    dialog.participant_group.set("MZ — znaki")
    dialog.participant_group.event_generate("<<ComboboxSelected>>")
    root.update()
    assert dialog.participant_table.get_children() == tuple(contract.MODELS[3:])
    dialog.participant_table.selection_set("MZ-NIGHT")
    root.update()
    assert dialog.participant_weights.get() == models[-1]["checkpoint_file"]
    assert dialog.participant_sha.get() == models[-1]["checkpoint_sha256"]
    dialog.participant_group.set("Wszystkie")
    dialog.participant_group.event_generate("<<ComboboxSelected>>")
    root.update()
    assert dialog.participant_table.get_children() == contract.MODELS
    assert "6 z 6" in dialog.participant_status.get()
    root.tk.call(dialog.protocol("WM_DELETE_WINDOW"))
    assert host._eval396_participants_dialog is None
    assert ranking._open_ranking_participants_modal(host) is not dialog


def test_char_target_opens_mz_group(native):
    root, host, _ = native
    host._get_ranking_task_target = lambda: "char"
    dialog = ranking._open_ranking_participants_modal(host)
    root.update()
    assert dialog.participant_table.get_children() == tuple(contract.MODELS[3:])


def test_failed_model_list_shows_error_instead_of_results(native, model_selection):
    root, host, _ = native
    path, _ = model_selection
    (path / "model_lineage.json").write_text("[]", encoding="utf-8")
    dialog = ranking._open_ranking_participants_modal(host)
    root.update()
    assert not dialog.participant_table.get_children()
    assert "Nie można" in dialog.participant_status.get()
    assert str(dialog.participant_group.cget("state")) == "disabled"


def test_real_local_participant_manifest_is_read_only():
    path = contract.default_selection(Path(__file__).resolve().parents[1])
    if not path.is_dir():
        pytest.skip("Local EVAL396 selection is not distributed")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    models = view.participant_models(path)
    assert tuple(m["label"] for m in models) == contract.MODELS
    assert all(Path(m["checkpoint_file"]).is_file() for m in models)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == before
