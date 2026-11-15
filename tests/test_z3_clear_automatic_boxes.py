from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk

import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_list_review as actions
from auto_annotation_tool.gui import z3_review_runtime as review


def box(source="yolo_box", *, sign="yolo_symbol", method="yolo", symbol="A"):
    return {"character": symbol, "bbox": [1.0, 2.0, 10.0, 30.0], "confidence": .8,
            "method": method, "box_source": source, "sign_source": sign}


def plate(records, *, approved=False, editing=True):
    row = {"characters": deepcopy(records), "ground_truth_text": "ABC123", "ground_truth_source": "manual_z2",
           "raw_detection": {"contract": "gt_blind.v1", "result_hash": "frozen", "characters": [box()]},
           "status": "perfect" if approved else "needs_fix",
           "gold_state": {"approved": approved, "candidate": approved},
           "plate_layout": "single_row", "yolo_detections": [box()], "gt_geometry_recovery": {"recovered_count": 6}}
    if editing:
        row["review_state"] = {"status": "approved" if approved else "in_progress", "approved_at": "now" if approved else None}
    return row


def owner(rows):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_metadata = rows
    host._preview_active_pid = next(iter(rows))
    host._get_preview_box_mode_key = lambda: "AUTO"
    host._refresh_preview_listbox_row = Mock()
    host._refresh_detection_review_controls = Mock()
    host._update_preview_info_label = Mock()
    host._sync_step3_access_from_preview_state = Mock()
    host._on_preview_select = Mock()
    host._update_preview_edit_status = Mock()
    host._schedule_preview_metadata_save = Mock()
    return host


def test_clear_preserves_manual_geometry_exactly_not_manual_symbols_on_auto_boxes():
    records = [box(), box("generated_box", sign="gt_assisted", method="segment"),
               box(sign="manual_sign"), box("manual_box", sign="manual_sign", method="manual"),
               box("manual_box", sign="gt_assisted", method="yolo")]
    legacy_manual = box(sign="", method="cvat_manual")
    legacy_manual.pop("box_source")
    records.append(legacy_manual)
    data = plate(records, approved=True)
    before = deepcopy(data)
    host = owner({"p": data})
    result = actions.clear_automatic_boxes(host, "p")
    assert result["removed_boxes"] == 3
    assert result["manual_boxes"] == 3
    assert data["characters"] == before["characters"][3:]
    assert data["raw_detection"] == before["raw_detection"]
    assert data["yolo_detections"] == before["yolo_detections"]
    assert data["ground_truth_text"] == before["ground_truth_text"]
    assert data["review_state"]["human_edited"]
    assert data["review_state"]["status"] == "in_progress"
    assert not data["gold_state"]["approved"]
    assert host._preview_history_undo["p"][-1] == before
    assert "gt_geometry_recovery" not in data


def test_empty_working_layer_stays_empty_and_does_not_show_raw_boxes_again():
    data = plate([box()])
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    assert data["characters"] == []
    assert host._get_preview_box_records(data)[0] == []
    assert not review.can_refresh_automatic_working_annotation(data)
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="p")
    assert actions.clear_automatic_boxes(host, "p")["unchanged"]


def test_raw_only_preview_can_be_cleared_without_changing_raw():
    data = plate([], editing=False)
    before = deepcopy(data["raw_detection"])
    host = owner({"p": data})
    assert actions.clear_automatic_boxes(host, "p")["removed_boxes"] == 1
    assert host._get_preview_box_records(data)[0] == []
    assert data["raw_detection"] == before


def test_manual_only_row_is_noop_and_keeps_approval():
    data = plate([box("manual_box", method="manual")], approved=True)
    before = deepcopy(data)
    host = owner({"p": data})
    assert actions.clear_automatic_boxes(host, "p")["unchanged"]
    assert data == before
    assert not hasattr(host, "_preview_history_undo")


@pytest.mark.parametrize("raw_hash", ["frozen", "new_detection"])
def test_explicit_detection_refills_cleared_boxes_even_when_raw_hash_is_identical(raw_hash):
    data = plate([box()])
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    fresh = [box(symbol="A"), box(symbol="B")]
    fresh[1]["bbox"] = [20.0, 2.0, 30.0, 30.0]
    data["raw_detection"] = {"contract": "gt_blind.v1", "result_hash": raw_hash, "characters": fresh}
    raw = deepcopy(data["raw_detection"])
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="p")
    assert data["characters"] == []
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert len(data["characters"]) == 2
    assert len(host._get_preview_box_records(data)[0]) == 2
    assert data["raw_detection"] == raw
    assert "automatic_boxes_cleared" not in data
    assert not data["gold_state"]["approved"]


def test_explicit_detection_preserves_manual_box_and_rejects_colliding_auto_candidate():
    manual = box("manual_box", sign="manual_sign", method="manual", symbol="M")
    data = plate([manual, box()])
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    automatic = box(symbol="B")
    automatic["bbox"] = [20.0, 2.0, 30.0, 30.0]
    data["raw_detection"] = {"contract": "gt_blind.v1", "result_hash": "new", "characters": [box(), automatic]}
    raw = deepcopy(data["raw_detection"])
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert len(data["characters"]) == 2
    retained = data["characters"][0]
    for key, value in manual.items():
        assert retained[key] == value
    assert data["characters"][1]["bbox"] == automatic["bbox"]
    assert data["raw_detection"] == raw
    assert data["review_state"]["human_edited"]
    assert "automatic_boxes_cleared" not in data


def test_detection_does_not_trim_manual_boxes_to_gt_length():
    manuals = [box("manual_box", sign="manual_sign", method="manual", symbol="M") for _ in range(3)]
    for index, rec in enumerate(manuals):
        rec["bbox"] = [index * 20.0, 2.0, index * 20.0 + 10.0, 30.0]
    data = plate(manuals + [box()])
    data["ground_truth_text"] = "AB"
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert len(data["characters"]) == 3
    assert [rec["bbox"] for rec in data["characters"]] == [rec["bbox"] for rec in manuals]
    assert all(rec["box_source"] == "manual_box" for rec in data["characters"])


def test_empty_detection_keeps_clear_marker_for_later_successful_rerun():
    data = plate([box()])
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    data["raw_detection"]["characters"] = []
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert data["characters"] == []
    assert data["automatic_boxes_cleared"]
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="p")
    data["raw_detection"]["characters"] = [box()]
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert len(data["characters"]) == 1
    assert "automatic_boxes_cleared" not in data


def test_detection_still_protects_other_human_edits_after_clear():
    data = plate([box()])
    host = owner({"p": data})
    actions.clear_automatic_boxes(host, "p")
    data["characters"] = [box(sign="manual_sign", symbol="X")]
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="p", from_detection=True)
    assert data == before


@pytest.fixture
def scene():
    root = tk.Tk()
    root.withdraw()
    host = owner({"a": plate([box(), box("manual_box", method="manual")]),
                  "b": plate([box()], approved=True), "c": plate([box("generated_box", method="segment")])})
    host.frame = tk.Frame(root)
    host.plates_listbox = tk.Listbox(host.frame, selectmode=tk.EXTENDED, exportselection=False)
    host._listbox_pid_by_index = ["c", "b", "a"]  # Display sorted differently from metadata.
    host.plates_listbox.insert(tk.END, *host._listbox_pid_by_index)
    host._preview_active_pid = "b"  # The active preview is deliberately outside the selection.
    host.preview_box_mode_var = tk.StringVar(root, "AUTO")
    yield host, root
    root.destroy()


def wait_batch(root, result):
    done = tk.BooleanVar(root, False)
    def poll():
        if result["done"]:
            done.set(True)
        else:
            root.after(5, poll)
    root.after(5, poll)
    timeout = root.after(2000, lambda: done.set(True))
    root.wait_variable(done)
    root.after_cancel(timeout)
    assert result["done"]


def test_context_menu_clears_only_selected_group_and_preserves_selection(scene, monkeypatch):
    host, root = scene
    host.plates_listbox.selection_set(0)
    host.plates_listbox.selection_set(2)
    before = deepcopy(host.preview_metadata)
    host.frame.pack(fill=tk.BOTH, expand=True)
    host.plates_listbox.pack(fill=tk.BOTH, expand=True)
    host._schedule_preview_select_render = Mock()
    root.deiconify()
    root.update_idletasks()
    monkeypatch.setattr(tk.Menu, "tk_popup", Mock())
    bounds = host.plates_listbox.bbox(0)
    event = SimpleNamespace(y=bounds[1] + bounds[3] // 2, x_root=10, y_root=10)
    assert actions.show_context_menu(host, event) == "break"
    menu = host._preview_list_context_menu
    assert menu.entrycget(2, "label") == "Usuń automatyczne ramki (2)"
    menu.invoke(2)
    result = host._preview_review_batch_result
    assert host._preview_review_batch_running
    wait_batch(root, result)
    assert result["changed"] == ["c", "a"]
    assert result["removed_boxes"] == 2
    assert result["manual_boxes"] == 1
    assert host.preview_metadata["b"] == before["b"]
    assert host.preview_metadata["c"]["characters"] == []
    assert host.preview_metadata["a"]["characters"] == [before["a"]["characters"][1]]
    assert host._preview_pending_save_pids == {"a", "c"}
    assert host.plates_listbox.curselection() == (0, 2)
    assert not host._preview_review_batch_running
    host._schedule_preview_metadata_save.assert_called_with(delay_ms=30)
    for pid in before:
        assert host.preview_metadata[pid]["raw_detection"] == before[pid]["raw_detection"]


def test_empty_selection_and_busy_editor_never_clear_active_or_all_rows(scene):
    host, _ = scene
    before = deepcopy(host.preview_metadata)
    assert actions.run_selected_clear_automatic_boxes(host) is None
    host.plates_listbox.selection_set(0, tk.END)
    host.fast_test_running = True
    assert actions.run_selected_clear_automatic_boxes(host) is None
    assert host.preview_metadata == before


def test_changing_selection_after_click_does_not_change_captured_scope(scene):
    host, root = scene
    host.plates_listbox.selection_set(0)
    result = actions.run_selected_clear_automatic_boxes(host)
    host.plates_listbox.selection_clear(0, tk.END)
    host.plates_listbox.selection_set(1)
    wait_batch(root, result)
    assert result["changed"] == ["c"]
    assert host.preview_metadata["b"]["gold_state"]["approved"]


def test_undo_restores_deleted_boxes(scene, monkeypatch):
    host, _ = scene
    host._preview_active_pid = "a"
    before = deepcopy(host.preview_metadata["a"])
    actions.clear_automatic_boxes(host, "a")
    from auto_annotation_tool.gui import z3_plate_gt_runtime
    monkeypatch.setattr(z3_plate_gt_runtime, "refresh_working_ground_truth", Mock())
    host._refresh_preview_live_metadata_ui = Mock()
    host._redraw_preview_character_overlays_light = Mock(return_value=True)
    host._persist_preview_metadata = Mock()
    assert host._undo_preview_edit() == "break"
    assert host.preview_metadata["a"]["characters"] == before["characters"]
    assert host.preview_metadata["a"]["raw_detection"] == before["raw_detection"]


def test_clear_action_is_available_only_in_context_menu():
    root = Path(__file__).resolve().parents[1]
    source = (root / "auto_annotation_tool/gui/z3_detection_tab_ui.py").read_text(encoding="utf-8")
    assert "btn_clear_automatic_boxes" not in source
    assert "run_selected_clear_automatic_boxes(self)" not in source
    source = (root / "auto_annotation_tool/gui/z3_list_review.py").read_text(encoding="utf-8")
    assert 'label=f"Usuń automatyczne ramki ({total})"' in source
    source = (root / "auto_annotation_tool/gui/z3_detection_runtime.py").read_text(encoding="utf-8")
    assert "geometry_settings=yolo_runtime, from_detection=True" in source
