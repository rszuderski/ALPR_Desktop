from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z3_preview_ui as preview
from auto_annotation_tool.gui import z3_preview_dock_ui as dock_ui
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.section_header_label import SectionHeaderLabel
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui.web_slim_scrollbar import blend_hex_colors


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


@pytest.mark.parametrize("box,sign", [(True, False), (False, True), (True, True)])
def test_manual_edit_refreshes_list_even_when_quality_is_unchanged(box, sign):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    rec = {"character": "A", "bbox": [1, 2, 20, 40], "box_source": "yolo_box", "sign_source": "ocr_symbol"}
    data = {"characters": [rec], "status": "needs_fix"}
    host._preview_active_pid = "plate_1"
    host._preview_char_selected_index = 0
    host._get_preview_active_data = lambda **_kwargs: data
    host._get_preview_active_character_records = lambda **_kwargs: data["characters"]
    host._sort_character_records_by_x = lambda chars, **_kwargs: chars
    host._annotate_preview_character_reading_positions = lambda chars, **_kwargs: chars
    host._normalize_preview_char_bbox = lambda bbox, **_kwargs: bbox
    host._derive_preview_status_from_data = lambda *_args: "needs_fix"
    host._get_plate_source_bucket = lambda *_args, **_kwargs: "auto_preview"
    host._loaded_meta_path = None
    for name in ("_update_preview_plate_layout_metadata", "_ensure_plate_source_metadata",
                 "_draw_preview_plate_status_frame", "_refresh_preview_live_metadata_ui",
                 "_redraw_preview_character_overlays_light", "_schedule_preview_metadata_save",
                 "_schedule_preview_info_refresh", "_on_preview_select"):
        setattr(host, name, Mock())
    host._mark_preview_char_record_manual(rec, box=box, sign=sign)
    with patch.object(preview, "clear_detection_review_snapshot_after_manual_edit"), \
         patch.object(preview, "refresh_preview_canvas_info_overlay_only"), \
         patch.object(preview, "log_preview_edit_flow"):
        preview.persist_active_preview_characters(host, selected_record=rec, success_message="Zapisano",
                                                 render_preview=False, save_immediately=False, refresh_row=False)
    assert data["status"] == "needs_fix"
    assert host._refresh_preview_live_metadata_ui.call_args.kwargs["refresh_row"] is True
    flags = host._get_plate_listbox_source_flags(data)
    assert "MANUAL" in flags
    assert ("MB" in flags) == box
    assert ("MS" in flags) == sign
    if not box:
        assert rec["box_source"] == "yolo_box"
    if not sign:
        assert rec["sign_source"] == "ocr_symbol"
    host._on_preview_select.assert_not_called()


def test_z3_drawer_stays_hidden_after_resize_and_fullscreen_switch(root):
    canvas_host = tk.Frame(root, width=960, height=600)
    canvas_host.winfo_width = lambda: 960
    canvas_host.winfo_height = lambda: 600
    panel = tk.Frame(canvas_host)
    status = tk.Label(panel, text="Tablica: korekta")
    status.pack()
    host = SimpleNamespace(preview_canvas_host=canvas_host, preview_overlay_dock=panel,
                           preview_canvas=Mock(), _preview_render_state={"image": True},
                           app=SimpleNamespace(palette=get_theme_palette()))
    try:
        with patch.object(dock_ui, "render_preview_overlay_dock", return_value=(180, 200)):
            dock_ui.place_preview_overlay_dock(host)
            slide = host._preview_drawer_slide
            slide.toggle()
            slide._cancel()
            slide.clock = lambda: slide._started + 1
            slide._tick()
            assert panel.winfo_manager() == ""
            for fullscreen in (True, False, True):
                host._preview_fullscreen_active = fullscreen
                dock_ui.place_preview_overlay_dock(host, force_render=True)
                assert panel.winfo_manager() == ""
                assert slide.button.winfo_manager() == "place"
            slide.clock = lambda: 0
            slide.toggle()
            slide._cancel()
            slide.clock = lambda: slide._started + 1
            slide._tick()
            assert panel.winfo_manager() == "place"
            assert status["text"] == "Tablica: korekta"
    finally:
        canvas_host.destroy()


def test_resize_storm_schedules_one_render_after_fullscreen_geometry(root):
    canvas = tk.Canvas(root)
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_canvas = canvas
    host._preview_active_pid = "plate_1"
    host._preview_fullscreen_transition_active = True
    host._on_preview_select = Mock()
    host._ensure_preview_mode_overlay_position = Mock()
    host._refresh_preview_typing_overlay_visibility = Mock()
    try:
        for width in range(800, 840):
            host._on_preview_canvas_configure(SimpleNamespace(width=width, height=600))
        host._on_preview_select.assert_not_called()
        done = tk.BooleanVar(root, False)
        root.after(140, lambda: done.set(True))
        root.wait_variable(done)
        host._on_preview_select.assert_called_once_with(None)
        assert not host._preview_fullscreen_transition_active
        host._ensure_preview_mode_overlay_position.assert_not_called()
        host._refresh_preview_typing_overlay_visibility.assert_not_called()
    finally:
        canvas.destroy()


def test_new_box_has_two_contrasting_outlines_and_clear_interior(root):
    canvas = tk.Canvas(root)
    try:
        preview._draw_preview_new_character_box(canvas, 30, 40, 80, 120)
        outlines = [item for item in canvas.find_withtag("preview_char_add_preview")
                    if canvas.type(item) == "rectangle" and canvas.coords(item) == [30, 40, 80, 120]]
        assert len(outlines) == 2
        assert {canvas.itemcget(item, "outline") for item in outlines} == {"#071923", "#00e5ff"}
        assert all(canvas.itemcget(item, "fill") == "" for item in outlines)
    finally:
        canvas.destroy()


@pytest.mark.parametrize("theme", ["light_visual_cs", "dark_visual_cs"])
def test_header_gradient_retains_pixels_and_reuses_background_for_text(root, theme):
    header = SectionHeaderLabel(root, SimpleNamespace(palette=get_theme_palette(theme)), text="Nagłówek")
    header.winfo_width = lambda: 400
    try:
        header._render()
        panel, start, _ = header._get_colors()
        rgb = lambda color: tuple(value // 257 for value in root.winfo_rgb(color))
        assert header._background_image.get(0, 0) == rgb(start)
        assert header._background_image.get(399, 0) == rgb(panel)
        assert header._background_image.get(0, 1) == rgb(blend_hex_colors(start, "#ffffff", 0.24))
        with patch.object(header, "_paint_background", wraps=header._paint_background) as paint:
            header._text = "Drugi tekst"
            header._render()
            paint.assert_not_called()
            header.winfo_width = lambda: 300
            header._render()
            paint.assert_called_once()
        assert header._background_size[0] == 300
    finally:
        header._cancel_pending_render()
        header.destroy()


@pytest.mark.parametrize("bounds", [(12, 100, 370, 280), (12, 100, 550, 692)])
def test_assistant_avoids_compass_and_stays_inside_canvas(bounds):
    from auto_annotation_tool.gui import z3_preview_typing_runtime as typing
    host = SimpleNamespace(_get_preview_canvas_size=lambda: (1000, 700),
                           _get_preview_typing_overlay_bottom_offset=lambda: 8,
                           _get_preview_typing_overlay_anchor=lambda _height: (8, 100),
                           _preview_controls_legend_current_bounds=bounds)
    x, y = typing._resolve_preview_typing_overlay_anchor(host, 180, 70)
    assert x >= bounds[2] or y >= bounds[3]
    assert 0 <= x <= 820 and 0 <= y <= 630
