from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk

import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_list_review as actions
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui import z3_preview_ui as preview
from auto_annotation_tool.gui import z3_preview_events as events
from auto_annotation_tool.gui.z3_gt_box_policy import limit_boxes_to_gt, can_add_character_box
from test_z3_raw_result_canvas import canvas_host


def chars(text):
    return [{"character": char, "bbox": [i*15, 0, i*15+12, 25], "confidence": .8,
             "method": "yolo", "box_source": "yolo_box", "sign_source": "yolo_symbol"}
            for i, char in enumerate(text)]


def data(text, gt, *, editing=False):
    record = {"characters": chars(text) if editing else [], "ground_truth_text": gt,
              "status": "needs_fix", "plate_layout": "single_row", "plate_layout_override": "single_row",
              "source_annotation_id": "annotation", "source_gt_hash": "gt-1",
              "source_info": {"bucket": "auto_preview", "origin": "pz2_detect"},
              "raw_detection": {"contract": "gt_blind.v1", "characters": chars(text)}}
    if editing:
        record["review_state"] = {"status": "in_progress"}
    return record


def test_gt_limit_discards_internal_extra_detections_without_mutating_raw():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = data("XABZCDY", "ABCD")
    before = deepcopy(record)
    limited, detail = limit_boxes_to_gt(host, record, record["raw_detection"]["characters"])
    assert "".join(rec["character"] for rec in limited) == "ABCD"
    assert detail["removed_count"] == 3
    assert record == before


def test_limit_prioritizes_manual_geometry_and_uses_confidence_to_break_ties():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = data("XXXX", "AB")
    raw = record["raw_detection"]["characters"]
    raw[0]["box_source"] = "manual_box"
    raw[0]["confidence"] = .01
    raw[3]["confidence"] = .99
    limited, _ = limit_boxes_to_gt(host, record, raw)
    assert limited == [raw[0], raw[3]]


def test_limit_cache_tracks_gt_changes_and_does_not_cache_live_review_edits():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = data("ABC", "AB", editing=True)
    raw = record["raw_detection"]["characters"]
    assert "".join(rec["character"] for rec in limit_boxes_to_gt(host, record, raw)[0]) == "AB"
    record["ground_truth_text"] = "AC"
    assert "".join(rec["character"] for rec in limit_boxes_to_gt(host, record, raw)[0]) == "AC"
    record["characters"][1]["box_source"] = "manual_box"
    selected, _ = limit_boxes_to_gt(host, record, record["characters"])
    assert record["characters"][1] in selected
    assert raw[1]["box_source"] == "yolo_box"


def test_gt_assisted_yolo_box_is_not_double_counted_as_two_yolo_boxes():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = chars("A")[0]
    record["sign_source"] = "gt_assisted"
    counts = host._count_character_sources([record])
    assert counts["yolo_box"] == 1


def test_auto_view_and_review_use_the_same_gt_limited_geometry():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    record = data("XABZCDY", "ABCD")
    host._get_preview_box_mode_key = lambda: "AUTO"
    displayed, _ = limit_boxes_to_gt(host, record, record["raw_detection"]["characters"])
    assert len(displayed) == 4
    host.preview_metadata = {"p": record}
    host._preview_active_pid = "p"
    host._ensure_plate_source_metadata = Mock()
    before = deepcopy(record["raw_detection"])
    assert review.start_review_from_raw(host, persist=False, quiet=True, refresh=False)["ok"]
    assert [rec["bbox"] for rec in record["characters"]] == [rec["bbox"] for rec in displayed]
    assert [rec["character"] for rec in record["characters"]] == list("ABCD")
    assert [rec["reading_index"] for rec in record["characters"]] == [1, 2, 3, 4]
    assert record["raw_detection"] == before
    assert len(record["gt_box_limit"]["discarded"]) == 3
    host._get_preview_box_mode_key = lambda: "RAW_RESULT"
    assert len(host._get_preview_box_records(record)[0]) == 7


def test_manual_addition_is_available_at_gt_count_and_after_delete():
    record = data("AB", "AB", editing=True)
    host = SimpleNamespace(_update_preview_edit_status=Mock())
    assert can_add_character_box(host, record)
    record["characters"].pop()
    assert can_add_character_box(host, record)


def test_d_and_completed_manual_drawing_can_exceed_gt_without_changing_raw(canvas_host, monkeypatch):
    host = canvas_host
    record = data("ABC", "ABC")
    host.preview_metadata = {"plate": record}
    host._preview_active_pid = None
    host.preview_box_mode_var.set("AUTO")
    host._preview_char_label_mode = False
    host._preview_char_label_active_index = None
    host._preview_char_add_click_armed = False
    host._preview_char_add_mode = False
    monkeypatch.setattr(host, "_schedule_preview_metadata_save", Mock())
    monkeypatch.setattr(host, "_refresh_preview_listbox_row", Mock())
    preview.on_preview_select(host)
    assert events.on_preview_canvas_keypress(host, SimpleNamespace(keysym="d", char="d", state=0)) == "break"
    assert host._preview_char_add_click_armed
    assert len(record["characters"]) == 3
    raw_before = deepcopy(record["raw_detection"])
    host._preview_char_add_state = {"bbox": [70, 2, 88, 25], "click_draw": True}
    assert events.finalize_preview_char_add_state(host) == "break"
    assert len(record["characters"]) == 4
    assert record["characters"][-1]["box_source"] == "manual_box"
    assert record["raw_detection"] == raw_before


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.geometry("640x650+20+20")
    yield window
    window.destroy()


@pytest.fixture
def scene(root, monkeypatch):
    monkeypatch.setattr(review, "_persist_review_az_revision_best_effort", Mock())
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.frame = tk.Frame(root)
    host.frame.pack(fill="both", expand=True)
    host.app = SimpleNamespace(palette={}, get_list_selection_colors=lambda: ("#336699", "#ffffff"))
    host.plates_listbox = tk.Listbox(host.frame, selectmode=tk.EXTENDED, exportselection=False, height=12)
    host.plates_listbox.pack(fill="x")
    host.preview_metadata = {"a": data("XABZCY", "ABC"), "b": data("DE", "DEF"),
                             "c": data("GH", "GH", editing=True), "d": data("JK", "JK")}
    host._listbox_pid_by_index = list(host.preview_metadata)
    host._preview_base_plate_ids = list(host.preview_metadata)
    host.preview_plate_ids = list(host.preview_metadata)
    host._preview_active_pid = "a"
    host._preview_metadata_revision = 0
    host._project_reset_token = 1
    host.preview_box_mode_var = tk.StringVar(root, "AUTO")
    host.preview_dir_var = tk.StringVar(root, "")
    host.plates_listbox.insert("end", *host._listbox_pid_by_index)
    host.plates_listbox.selection_set(0)
    host.plates_listbox.activate(0)
    for name in ("preview_perfect_count_lbl", "preview_error_count_lbl", "preview_unknown_count_lbl",
                 "preview_approved_char_count_lbl", "plates_legend_box_perfect_lbl"):
        widget = tk.Label(host.frame, text="0")
        widget.pack()
        setattr(host, name, widget)
    for name in ("_schedule_preview_select_render", "_schedule_preview_metadata_save", "_on_preview_select",
                 "_refresh_detection_review_controls", "_sync_step3_access_from_preview_state",
                 "_update_preview_repair_progress_ui", "_refresh_gold_export_filter_labels",
                 "_refresh_gold_export_source_labels", "_refresh_gold_export_scope_label",
                 "_refresh_preview_import_focus_ui", "_update_preview_edit_status", "_ensure_plate_source_metadata"):
        setattr(host, name, Mock())
    for pid, record in host.preview_metadata.items():
        review.prepare_working_annotation_from_raw(host, record, plate_id=pid)
    root.update()
    yield host
    host.frame.destroy()


def click(host, row, modifiers=0):
    bounds = host.plates_listbox.bbox(row)
    event = SimpleNamespace(x=10, y=bounds[1] + bounds[3]//2, state=modifiers)
    return events.on_preview_list_mouse_primary(host, event)


def wait_batch(host, result):
    done = tk.BooleanVar(host.frame, False)
    def poll():
        if result["done"]:
            done.set(True)
        else:
            host.frame.after(5, poll)
    host.frame.after(5, poll)
    timeout = host.frame.after(2000, lambda: done.set(True))
    host.frame.wait_variable(done)
    host.frame.after_cancel(timeout)
    assert result["done"]


def test_ctrl_shift_selection_and_active_preview(scene):
    click(scene, 0)
    click(scene, 2, 0x0004)
    assert scene.plates_listbox.curselection() == (0, 2)
    assert scene._get_current_preview_list_index() == 2
    click(scene, 3, 0x0001)
    assert scene.plates_listbox.curselection() == (2, 3)
    assert scene._get_current_preview_list_index() == 3
    click(scene, 0, 0x0005)
    assert scene.plates_listbox.curselection() == (0, 1, 2, 3)


def test_refresh_and_sort_preserve_selection_and_anchor(scene):
    click(scene, 0)
    click(scene, 2, 0x0004)
    scene._refresh_preview_listbox_row("a")
    assert scene.plates_listbox.curselection() == (0, 2)
    assert scene.plates_listbox.index(tk.ANCHOR) == 2
    assert scene.plates_listbox.index(tk.ACTIVE) == 2
    scene._get_sorted_preview_plate_ids = lambda ids: list(reversed(ids))
    preview.rebuild_preview_listbox(scene, preserve_selection=True, schedule_render=False)
    assert set(actions.selected_plate_ids(scene)) == {"a", "c"}
    assert scene._get_current_preview_list_index() == 1


def test_context_menu_retains_group_and_selects_outside_row(scene, monkeypatch):
    monkeypatch.setattr(tk.Menu, "tk_popup", lambda *args, **kwargs: None)
    click(scene, 0)
    click(scene, 2, 0x0004)
    def right_click(index):
        bounds = scene.plates_listbox.bbox(index)
        actions.show_context_menu(scene, SimpleNamespace(y=bounds[1]+2, x_root=10, y_root=10))
    right_click(2)
    assert actions.selected_plate_ids(scene) == ["a", "c"]
    assert "(2)" in scene._preview_list_context_menu.entrycget(0, "label")
    right_click(1)
    assert actions.selected_plate_ids(scene) == ["b"]
    assert "(1)" in scene._preview_list_context_menu.entrycget(0, "label")


def test_bulk_approval_skips_invalid_rows_and_refreshes_approved_frame_counter(scene):
    scene.plates_listbox.selection_set(0, 2)
    scene.plates_listbox.activate(2)
    before = deepcopy(scene.preview_metadata)
    # Prime the cache and actual labels with the old zero count.
    scene._update_preview_info_label()
    assert scene.preview_perfect_count_lbl.cget("text") == "0"
    # 3 + 2 + 2 + 2 visible working boxes, including results not opened for editing.
    assert scene._count_preview_statuses()["char_boxes"] == 9
    result = actions.run_selected_review_action(scene, True)
    wait_batch(scene, result)
    assert result["changed"] == ["a", "c"]
    assert [item["plate_id"] for item in result["failed"]] == ["b"]
    assert [call.args[1] for call in review._persist_review_az_revision_best_effort.call_args_list] == ["a", "c"]
    assert scene.preview_perfect_count_lbl.cget("text") == "2"
    assert scene.preview_approved_char_count_lbl.cget("text").endswith(": 5")
    assert scene.plates_legend_box_perfect_lbl.cget("text") == "5"
    assert scene.plates_listbox.curselection() == (0, 1, 2)
    assert scene.preview_metadata["d"] == before["d"]
    assert scene._preview_pending_save_pids == {"a", "b", "c"}
    for pid in before:
        assert scene.preview_metadata[pid]["raw_detection"] == before[pid]["raw_detection"]
    scene._schedule_preview_metadata_save.assert_called_with(delay_ms=30)
    result = actions.run_selected_review_action(scene, False)
    wait_batch(scene, result)
    assert scene.preview_perfect_count_lbl.cget("text") == "0"
    assert scene.plates_legend_box_perfect_lbl.cget("text") == "0"


def test_single_approval_refreshes_counts_without_waiting_for_idle_mouse(scene):
    review.start_review_from_raw(scene, "c", persist=False, quiet=True, refresh=False)
    scene._update_preview_info_label()
    scene._preview_last_char_edit_interaction_ts = 10**12
    assert review.confirm_review_gold(scene, "c", persist=False, quiet=True)["ok"]
    assert scene.preview_perfect_count_lbl.cget("text") == "1"
    assert scene.plates_legend_box_perfect_lbl.cget("text") == "2"


def test_context_menu_invokes_group_approval(scene, monkeypatch):
    monkeypatch.setattr(tk.Menu, "tk_popup", lambda *args, **kwargs: None)
    click(scene, 0)
    click(scene, 2, 0x0004)
    box = scene.plates_listbox.bbox(2)
    actions.show_context_menu(scene, SimpleNamespace(y=box[1]+2, x_root=10, y_root=10))
    scene._preview_list_context_menu.invoke(0)
    wait_batch(scene, scene._preview_review_batch_result)
    assert scene._preview_review_batch_result["changed"] == ["a", "c"]
    assert scene.preview_perfect_count_lbl.cget("text") == "2"


def test_batch_yields_and_project_change_cancels_without_touching_new_metadata(scene, monkeypatch):
    queue = []
    monkeypatch.setattr(scene.frame, "after", lambda delay, callback: queue.append(callback) or "queued")
    scene.preview_metadata = {str(i): data("ABC", "ABC") for i in range(60)}
    original = scene.preview_metadata
    scene._listbox_pid_by_index = list(original)
    scene.plates_listbox.delete(0, tk.END)
    scene.plates_listbox.insert(tk.END, *scene._listbox_pid_by_index)
    scene.plates_listbox.selection_set(0, tk.END)
    scene._refresh_preview_listbox_row = Mock()
    result = actions.run_selected_review_action(scene, True)
    queue.pop(0)()
    assert 0 < result["processed"] <= 25 and not result["done"]
    assert scene._preview_pending_save_pids
    scene._schedule_preview_metadata_save.assert_called_with(delay_ms=180)
    replacement = {"new": data("AB", "AB")}
    before = deepcopy(replacement)
    scene.preview_metadata = replacement
    queue.pop(0)()
    assert result["done"] and result["cancelled"]
    assert replacement == before
    assert not scene._preview_review_batch_running


def test_reopening_unapproved_selection_does_not_write_unchanged_dataset(scene):
    scene.plates_listbox.selection_set(0, tk.END)
    result = actions.run_selected_review_action(scene, False)
    wait_batch(scene, result)
    assert result["unchanged"] == 4 and not result["dirty"]
    scene._schedule_preview_metadata_save.assert_not_called()


def test_destroy_cancels_pending_batch_callback(scene):
    result = actions.run_selected_review_action(scene, True)
    job = result["after_id"]
    scene.frame.destroy()
    assert result["done"] and result["cancelled"]
    assert job not in scene.frame.tk.call("after", "info")
