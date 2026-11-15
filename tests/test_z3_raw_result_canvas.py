"""Exercise both real canvas render paths with frozen predictions and reviews."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk

from PIL import Image
import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_preview_ui as ui
from auto_annotation_tool.gui import z3_init_runtime as init
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


@pytest.fixture(scope="module")
def canvas_host(tmp_path_factory):
    directory = tmp_path_factory.mktemp("raw_canvas")
    (directory / "images").mkdir()
    Image.new("RGB", (240, 64), "white").save(directory / "images/plate.jpg")
    root = tk.Tk()
    root.geometry("900x650")
    # Build the actual tab state and canvas, isolating application navigation,
    # project/session writes and unrelated side panels.
    detached = [
        "_create_widgets", "reset_subtab_flow", "_bind_source_path_watchers",
        "_update_step3_source_path_lock", "_schedule_source_binding_refresh",
        "_clear_project_bound_session_values", "_on_app_close", "_save_local_setting",
        "_on_preview_dir_var_write", "_mark_startup_ui_ready",
        "_refresh_detection_review_controls", "_update_preview_record_source_label",
        "_update_preview_box_info_label", "_refresh_preview_editor_toolbar",
        "_place_preview_overlay_dock", "_place_preview_hint_overlay",
        "_refresh_preview_controls_legend", "_ensure_preview_mode_overlay_position",
        "_refresh_preview_typing_overlay_visibility", "_draw_preview_canvas_info_overlay",
        "_draw_preview_plate_status_frame",
    ]
    try:
        with ExitStack() as stack:
            for name in detached:
                stack.enter_context(patch.object(CharacterAnnotationTab, name, Mock()))
            stack.enter_context(patch.object(CharacterAnnotationTab, "_load_local_session", return_value={}))
            stack.enter_context(patch.object(init, "_ensure_ocr_presets_dir_with_legacy_copy",
                                            return_value=(directory, directory)))
            stack.enter_context(patch.object(ui, "_schedule_preview_neighbor_prefetch"))
            host = CharacterAnnotationTab(root, SimpleNamespace(root=root, palette=get_theme_palette()))
            host.frame.pack(fill="both", expand=True)
            host.preview_canvas = tk.Canvas(host.frame)
            host.preview_canvas.pack(fill="both", expand=True)
            host.plates_listbox = tk.Listbox(host.frame)
            host.plates_listbox.insert("end", "plate")
            host.plates_listbox.selection_set(0)
            host._listbox_pid_by_index = ["plate"]
            host.preview_dir_var.set(str(directory))
            root.update()
            yield host
    finally:
        root.destroy()


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("stage,source,expected", [
    ("raw", "RAW_RESULT", ["A", "B"]),
    ("review", "FINAL", ["R"]),
    ("empty_review", "FINAL", []),
    ("empty_raw", "RAW_RESULT", []),
    ("explicit_raw", "RAW_RESULT", ["A", "B"]),
])
def test_boxes_and_delayed_signatures_follow_the_same_stage(canvas_host, fast, stage, source, expected):
    host = canvas_host
    chars = [{"character": symbol, "bbox": [20 + i * 40, 4, 45 + i * 40, 60],
              "box_source": "yolo_box", "sign_source": "yolo_symbol", "confidence": .9}
             for i, symbol in enumerate("AB")]
    data = {"characters": [], "raw_detection": {"characters": chars, "contract": "gt_blind.v1"},
            "yolo_detections": [{"character": "X", "bbox": [140, 0, 160, 64]}]}
    if stage in {"review", "explicit_raw"}:
        data["characters"] = [{"character": "R", "bbox": [10, 2, 35, 60]}]
    if stage in {"review", "explicit_raw", "empty_review"}:
        data["review_state"] = {"status": "in_progress"}
    if stage == "empty_raw":
        data["raw_detection"]["characters"] = []
    raw_before, final_before = deepcopy(data["raw_detection"]), deepcopy(data["characters"])
    host.preview_metadata = {"plate": data}
    host.preview_box_mode_var.set("RAW_RESULT" if stage == "explicit_raw" else "AUTO")
    host._preview_fast_select_render = fast
    host._preview_char_selected_index = None

    with patch.object(ui.logger, "error") as errors:
        ui.on_preview_select(host)
        errors.assert_not_called()
    runtime = host._preview_char_runtime
    assert len(runtime) == len(expected)
    assert all(key.startswith(source + ":") for key in runtime)
    for value in runtime.values():
        assert host.preview_canvas.type(value["box_id"]) == "rectangle"
        x1, y1, x2, y2 = host.preview_canvas.coords(value["box_id"])
        assert x2 > x1 and y2 > y1

    if fast:
        assert ui.draw_preview_fast_render_details(host, "plate")
        for index, symbol in enumerate(expected):
            tag = host._get_preview_character_canvas_tag(source, index)
            labels = [host.preview_canvas.itemcget(item, "text")
                      for item in host.preview_canvas.find_withtag(tag)
                      if host.preview_canvas.type(item) == "text"]
            assert symbol in labels
    assert data["raw_detection"] == raw_before
    assert data["characters"] == final_before
