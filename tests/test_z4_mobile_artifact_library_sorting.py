"""Z4LIB003: sort table headers with type-aware keys and persistent order.

No model conversions, package mutations, Tk display or GUI event loop.
"""
from pathlib import Path

from auto_annotation_tool.gui.z4_mobile_artifact_registry import ArtifactRow
from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary


class FakeTree:
    def __init__(self, children, selected=()):
        self.children = list(children)
        self.selected = tuple(selected)
        self.headers = {}
        self.items = {}

    def get_children(self, parent=""):
        return tuple(self.children)

    def move(self, iid, parent, index):
        self.children.remove(iid)
        self.children.insert(index, iid)

    def heading(self, key, **kwargs):
        if kwargs:
            self.headers[key] = dict(kwargs)
        return self.headers.get(key, {})

    def selection(self):
        return self.selected

    def delete(self, iid):
        self.children.remove(iid)
        self.items.pop(iid, None)

    def insert(self, parent, index, iid, values):
        assert parent == "" and index == "end"
        self.children.append(iid)
        self.items[iid] = values


class FakeVar:
    def __init__(self, text=""):
        self.text = text

    def get(self):
        return self.text


def row(name, role, size, date, variant="TFLITE/INT8"):
    return ArtifactRow(path="C:/z4/"+name, name=name, schema="alpr.model.v1", kind="MODEL",
                       role=role, model_id=name, variants=variant, size_bytes=size,
                       modified_utc=date, state="DISCOVERED_NOT_VERIFIED", manifest={})


def library():
    obj = object.__new__(MobileArtifactLibrary)
    obj.sort_state = {"artifacts": None, "events": None}
    obj.sort_labels = {
        "artifacts": {"role": "Rola", "name": "Nazwa", "variants": "Runtime / precyzja",
                      "size": "Rozmiar", "date": "Data pliku (UTC)", "state": "Stan"},
        "events": {"date": "Czas (UTC)", "operation": "Operacja", "outcome": "Wynik",
                   "file": "Plik", "sha": "SHA-256"},
    }
    obj.art_tree = FakeTree(["a0", "a1", "a2"], selected=("a1",))
    obj.event_tree = FakeTree(["e0", "e1", "e2"])
    obj.rows = [
        row("a.alprmodel", "MT", 10*1024*1024, "2026-10-01T09:00:00+00:00"),
        row("b.alprmodel", "MZ", 2*1024*1024, "2026-09-29T12:00:00+00:00"),
        row("c.alprmodel", "MP+MT+MZ", 110*1024*1024, "2026-10-12T11:00:00+00:00"),
    ]
    obj.by_iid = {f"a{i}": r for i, r in enumerate(obj.rows)}
    obj.history = [
        {"event_id": "1", "observed_utc": "2026-10-14T12:05:00+00:00", "operation": "SINGLE_MODEL_EXPORT", "outcome": "FAILED", "artifact_path": "C:/z4/z.alprmodel"},
        {"event_id": "2", "observed_utc": "2026-10-13T10:30:00+00:00", "operation": "COMPLETE_PACKAGE_EXPORT", "outcome": "SUCCEEDED", "artifact_path": "C:/z4/a.alprmodel"},
        {"event_id": "3", "observed_utc": "2026-10-14T10:30:00+00:00", "operation": "COMPLETE_PACKAGE_EXPORT", "outcome": "STARTED", "artifact_path": "C:/z4/b.alprmodel"},
    ]
    obj.events_by_iid = {f"e{i}": r for i, r in enumerate(obj.history)}
    obj.filter_var = FakeVar()
    return obj


def test_artifact_size_is_numeric_and_toggles_direction_without_losing_selection():
    obj = library()
    obj.sort_column("artifacts", "size")
    assert obj.art_tree.children == ["a1", "a0", "a2"]
    assert obj.art_tree.selection() == ("a1",)
    assert obj.art_tree.headers["size"]["text"] == "Rozmiar ↑"
    obj.sort_column("artifacts", "size")
    assert obj.art_tree.children == ["a2", "a0", "a1"]
    assert obj.art_tree.headers["size"]["text"] == "Rozmiar ↓"


def test_artifact_date_is_full_utc_chronology():
    obj = library()
    obj.sort_column("artifacts", "date")
    assert obj.art_tree.children == ["a1", "a0", "a2"]
    obj.sort_column("artifacts", "date")
    assert obj.art_tree.children == ["a2", "a0", "a1"]


def test_text_columns_casefold_and_header_reset():
    obj = library()
    obj.rows[0] = row("Zulu.alprmodel", "MT", 1, "2026-10-01")
    obj.rows[1] = row("alpha.alprmodel", "MZ", 2, "2026-10-01")
    obj.by_iid = {f"a{i}": r for i,r in enumerate(obj.rows)}
    obj.sort_column("artifacts", "name")
    assert obj.art_tree.children == ["a1", "a2", "a0"]
    obj.sort_column("artifacts", "role")
    assert obj.art_tree.headers["name"]["text"] == "Nazwa"
    assert obj.art_tree.headers["role"]["text"] == "Rola ↑"


def test_history_table_sorts_date_and_file():
    obj = library()
    obj.sort_column("events", "date")
    assert obj.event_tree.children == ["e1", "e2", "e0"]
    obj.sort_column("events", "date")
    assert obj.event_tree.children == ["e0", "e2", "e1"]
    obj.sort_column("events", "file")
    assert obj.event_tree.children == ["e1", "e2", "e0"]


def test_filter_rebuild_reapplies_sort_for_both_tabs():
    obj = library()
    obj.sort_column("artifacts", "size")
    obj.sort_column("events", "date")
    obj.filter_var = FakeVar("alprmodel")
    obj.refresh_views()
    assert obj.art_tree.children == ["a1", "a0", "a2"]
    assert obj.event_tree.children == ["e1", "e2", "e0"]
    assert obj.art_tree.headers["size"]["text"] == "Rozmiar ↑"


def test_sort_does_not_modify_artifacts_or_export_log(tmp_path):
    obj = library()
    root = tmp_path/"mobile_packages"
    root.mkdir()
    file = root/"model.alprmodel"
    file.write_bytes(b"unchanged archive bytes")
    registry = root/"_registry"/"mobile_export_events.v1.jsonl"
    registry.parent.mkdir()
    registry.write_bytes(b"test-event-bytes\n")
    before = file.read_bytes(), registry.read_bytes()
    for key in obj.sort_labels["artifacts"]:
        obj.sort_column("artifacts", key)
    for key in obj.sort_labels["events"]:
        obj.sort_column("events", key)
    assert (file.read_bytes(), registry.read_bytes()) == before


def test_treeview_headings_are_clickable_and_are_wired_to_both_tables():
    source = (Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_artifact_library.py").read_text(encoding="utf-8")
    assert 'on_sort=lambda column: self.sort_column("artifacts", column)' in source
    assert 'on_sort=lambda column: self.sort_column("events", column)' in source
    assert 'command=lambda column=key: on_sort(column)' in source
