"""Z4LIB009: preserve library selections by identity and obey the active tab.

No exports, phone actions, package writes or JSONL changes are performed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Tree:
    def __init__(self):
        self.ids = []
        self.values = {}
        self.selected = []
        self.headings = {}

    def selection(self):
        return tuple(self.selected)

    def selection_set(self, iid):
        assert iid in self.values
        self.selected = [iid]

    def focus(self, iid):
        assert iid in self.values

    def get_children(self, _parent=""):
        return tuple(self.ids)

    def delete(self, iid):
        self.ids.remove(iid)
        self.values.pop(iid)
        if iid in self.selected:
            self.selected = []

    def insert(self, _parent, _position, *, iid, values):
        self.ids.append(iid)
        self.values[iid] = values

    def move(self, iid, _parent, pos):
        self.ids.remove(iid)
        self.ids.insert(pos, iid)

    def heading(self, column, **kwargs):
        self.headings[column] = kwargs


class Notebook:
    def __init__(self, active="packages"):
        self.active = active

    def select(self):
        return self.active


def artifact(name):
    return SimpleNamespace(
        role="MT", name=name, model_id=name, variants="TFLITE/INT8", state="DISCOVERED_NOT_VERIFIED",
        path=str(Path("C:/managed") / name), modified_utc="2026-10-14T13:00:00Z", size_bytes=100,
        schema="alpr.model.v1", error="", manifest={}, child_metadata={}, kind="MODEL",
    )


def event(op, *, time="2026-10-14T13:00:00+00:00", outcome="SUCCEEDED", ident=None):
    return {
        "operation_id": op, "operation": "SINGLE_MODEL_EXPORT", "outcome": outcome,
        "observed_utc": time, "event_id": ident or op+outcome, "artifact_path": f"C:/managed/{op}.alprmodel",
        "artifact_sha256": "a" * 64 if outcome == "SUCCEEDED" else None,
    }


def library():
    obj = object.__new__(MobileArtifactLibrary)
    obj.rows = []
    obj.history = []
    obj.art_tree = Tree()
    obj.event_tree = Tree()
    obj.by_iid = {}
    obj.events_by_iid = {}
    obj.filter_var = Var("")
    obj.show_raw_events = Var(False)
    obj._event_view_raw = False
    obj.event_view_status = Var("")
    obj.notebook = Notebook()
    obj.events_tab = "events"
    obj.packages_tab = "packages"
    obj.active_source = "artifact"
    obj._detail_sheet = None
    obj._detail_json = ""
    obj._detail_text = ""
    obj.sort_state = {"artifacts": None, "events": None}
    obj.sort_labels = {
        "artifacts": dict(zip(("role", "name", "variants", "size", "date", "state"),
                              ("Rola", "Nazwa", "Runtime / precyzja", "Rozmiar", "Data", "Stan"))),
        "events": dict(zip(("date", "operation", "outcome", "file", "sha"),
                       ("Czas", "Operacja", "Wynik", "Plik", "SHA"))),
    }
    return obj


def test_artifact_selection_survives_filter_and_scan_reordering():
    obj = library()
    a = artifact("mt_a.alprmodel")
    b = artifact("mt_b.alprmodel")
    obj.rows = [a, b]
    obj.refresh_views()
    obj.art_tree.selection_set("a1")
    assert obj.selected_artifact() is b
    obj.rows = [artifact("newer.alprmodel"), b, a]
    obj.refresh_views()
    assert obj.selected_artifact() is b
    assert obj.active_path() == Path(b.path)
    obj.filter_var.set("mt_b")
    obj.refresh_views()
    assert obj.selected_artifact() is b
    assert "mt_b" in obj._detail_text


def test_filter_removes_selection_and_clears_stale_details():
    obj = library()
    obj.rows = [artifact("mt_a.alprmodel"), artifact("mt_b.alprmodel")]
    obj.refresh_views()
    obj.art_tree.selection_set("a0")
    obj.filter_var.set("mt_b")
    obj.refresh_views()
    assert obj.selected_artifact() is None
    assert obj.active_path() is None
    assert "Wybierz paczkę" in obj._detail_text


def test_selected_history_operation_survives_insert_at_front():
    obj = library()
    a = event("OP-A-12345678")
    b = event("OP-B-12345678")
    obj.history = [a, b]
    obj.notebook.active = "events"
    obj.refresh_views()
    chosen_iid = next(i for i, r in obj.events_by_iid.items() if r["operation_id"] == "OP-B-12345678")
    obj.event_tree.selection_set(chosen_iid)
    obj.history.insert(0, event("OP-NEW-12345678", time="2026-10-14T13:05:00+00:00"))
    obj.refresh_views()
    assert obj.selected_event()["operation_id"] == "OP-B-12345678"
    assert obj.active_path() == Path(b["artifact_path"])


def test_history_audit_toggle_does_not_select_unrelated_event():
    obj = library()
    obj.notebook.active = "events"
    obj.history = [event("OP-A-12345678", time="2026-10-14T12:00:00+00:00", outcome="STARTED"),
                   event("OP-A-12345678", time="2026-10-14T12:01:00+00:00", outcome="SUCCEEDED")]
    obj.refresh_views()
    obj.event_tree.selection_set(obj.event_tree.ids[0])
    obj.show_raw_events.set(True)
    obj.refresh_views()
    assert obj.selected_event() is None
    assert "Wybierz wpis historii" in obj._detail_text


def test_history_raw_event_selection_pinned_to_event_id_not_index():
    obj = library()
    obj.notebook.active = "events"
    obj.show_raw_events.set(True)
    obj.history = [event("OP-A-12345678"), event("OP-B-12345678")]
    obj.refresh_views()
    obj.event_tree.selection_set("e1")
    obj.history.insert(0, event("OP-NEW-12345678"))
    obj.refresh_views()
    assert obj.selected_event()["event_id"] == "OP-B-12345678SUCCEEDED"


def test_tab_switch_controls_active_path_even_without_click():
    obj = library()
    obj.rows = [artifact("mt_a.alprmodel")]
    obj.history = [event("OP-A-12345678")]
    obj.refresh_views()
    obj.art_tree.selection_set("a0")
    obj.event_tree.selection_set("e0")
    obj.notebook.active = "events"
    assert obj.active_path() == Path(obj.history[0]["artifact_path"])
    obj._on_tab_changed()
    assert "OP-A-12345678" in obj._detail_text
    obj.notebook.active = "packages"
    obj._on_tab_changed()
    assert obj.active_path() == Path(obj.rows[0].path)
    assert "mt_a" in obj._detail_text


def test_hidden_tab_actions_cannot_validate_or_export(monkeypatch):
    obj = library()
    obj.rows = [artifact("mt_a.alprmodel")]
    obj.refresh_views()
    obj.art_tree.selection_set("a0")
    obj.notebook.active = "events"
    obj.choose_role = "MT"
    obj.on_choose = lambda *a: pytest.fail("Hidden-tab selection cannot export")
    obj.window = object()
    messages = []
    monkeypatch.setattr("auto_annotation_tool.gui.z4_mobile_artifact_library.messagebox.showwarning",
                        lambda title, message, **kw: messages.append(message))
    obj.choose_for_export()
    obj.validate_selected()
    assert len(messages) == 2
    assert all("Pakiety i modele" in message for message in messages)


def test_older_stub_without_selection_and_notebook_still_renders():
    obj = library()
    obj.rows = [artifact("mt_a.alprmodel")]
    del obj.notebook
    obj.refresh_views()
    assert obj.art_tree.ids == ["a0"]
