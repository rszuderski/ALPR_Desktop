from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z3_preview_ui as preview
from auto_annotation_tool.gui import z3_preview_dock_ui as dock_ui
from auto_annotation_tool.gui import z3_preview_events as preview_events
from auto_annotation_tool.gui import z3_review_runtime as review_runtime
from auto_annotation_tool.gui import z3_preview_list_ui as preview_list_ui
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


def _status_shortcut_host():
    return SimpleNamespace(
        _preview_review_batch_running=False,
        _preview_char_label_mode=False,
        _preview_char_label_active_index=None,
        _event_has_control_modifier=lambda _event: False,
        _sanitize_preview_char_symbol=lambda value: str(value or "").strip().upper(),
        _update_preview_edit_status=Mock(),
    )


def test_f_shortcut_toggles_plate_exclusion():
    host = _status_shortcut_host()
    host._toggle_review_excluded = Mock(return_value={"ok": True, "excluded": True})
    host._get_preview_active_data = Mock(return_value={})

    result = preview_events.on_preview_canvas_keypress(
        host,
        SimpleNamespace(keysym="f", char="f", state=0),
    )

    assert result == "break"
    host._toggle_review_excluded.assert_called_once_with(persist=True, refresh=False)
    assert "PZ3/Z4" in host._update_preview_edit_status.call_args.args[0]

def test_n_remains_character_when_symbol_field_is_active():
    host = _status_shortcut_host()
    host._preview_char_label_active_index = 0
    host._assign_character_to_active_preview_label = Mock(return_value=True)
    host._toggle_review_excluded = Mock()

    result = preview_events.on_preview_canvas_keypress(
        host,
        SimpleNamespace(keysym="n", char="n", state=0),
    )

    assert result == "break"
    host._assign_character_to_active_preview_label.assert_called_once_with("N")
    host._toggle_review_excluded.assert_not_called()


def test_r_shortcut_uses_existing_gold_confirmation():
    data = {
        "status": "needs_fix",
        "review_state": {"status": "in_progress"},
        "gold_state": {},
    }
    host = _status_shortcut_host()
    host._preview_active_pid = "plate_1"
    host._get_preview_active_data = lambda **_kwargs: data
    host._confirm_review_gold = Mock(
        return_value={"ok": True, "status": "perfect", "plate_id": "plate_1"}
    )
    host._refresh_preview_listbox_row = Mock()
    host._schedule_preview_metadata_save = Mock()
    host._schedule_preview_info_refresh = Mock()
    host._refresh_detection_review_controls = Mock()
    host.frame = SimpleNamespace(after_idle=lambda callback: callback())

    with patch.object(preview_events, "refresh_preview_canvas_info_overlay_only") as overlay:
        result = preview_events.on_preview_canvas_keypress(
            host,
            SimpleNamespace(keysym="r", char="r", state=0),
        )

    assert result == "break"
    host._confirm_review_gold.assert_called_once_with(
        quiet=True,
        persist=False,
        refresh=False,
    )
    host._refresh_preview_listbox_row.assert_called_once_with("plate_1")
    host._schedule_preview_metadata_save.assert_called_once_with(delay_ms=90)
    host._schedule_preview_info_refresh.assert_called_once_with(delay_ms=900)
    overlay.assert_called_once_with(host, data=data)

def test_toggle_review_excluded_preserves_gold_and_is_reversible():
    data = {
        "status": "perfect",
        "characters": [{"character": "A", "bbox": [0, 0, 10, 20]}],
        "gold_state": {"candidate": True, "approved": True},
    }
    host = SimpleNamespace(
        _preview_active_pid="plate_1",
        preview_metadata={"plate_1": data},
    )

    with patch.object(review_runtime, "_refresh_after_change") as refresh:
        first = review_runtime.toggle_review_excluded(host)
        assert first["ok"] is True
        assert first["excluded"] is True
        assert data["gold_state"]["excluded"] is True
        assert data["gold_state"]["excluded_reason"] == "unreadable"
        assert data["gold_state"]["approved"] is True
        assert data["gold_state"]["candidate"] is False

        second = review_runtime.toggle_review_excluded(host)
        assert second["ok"] is True
        assert second["excluded"] is False
        assert data["gold_state"]["excluded"] is False
        assert data["gold_state"]["approved"] is True
        assert data["gold_state"]["candidate"] is True

    assert refresh.call_count == 2


def _list_label_host(status: str = "needs_fix"):
    return SimpleNamespace(
        _get_preview_live_status=lambda data, chars=None, plate_id=None: status,
        _characters_to_display_text=lambda chars, data=None, separator=" / ": "ABC",
        _get_plate_listbox_ordinal=lambda _pid: 1,
        _get_plate_listbox_layout_flag=lambda _data: "",
        _get_plate_listbox_source_flags=lambda _data: [],
        _get_review_quality_status=lambda _data, _chars=None: "perfect",
    )


def test_pz2_n_overrides_ready_and_ok_list_flags():
    ready = {
        "status": "needs_fix",
        "characters": [{"character": "A", "bbox": [0, 0, 10, 20]}],
        "review_state": {"status": "in_progress"},
        "gold_state": {"excluded": True, "excluded_reason": "unreadable"},
    }
    label = preview_list_ui.format_plate_listbox_label(
        _list_label_host("needs_fix"), "p1", ready, ordinal=1
    )
    assert "N " in label
    assert "ZATWIERDŹ" not in label

    approved = {
        "status": "perfect",
        "characters": [{"character": "A", "bbox": [0, 0, 10, 20]}],
        "review_state": {"status": "approved"},
        "gold_state": {"approved": True, "excluded": True, "excluded_reason": "unreadable"},
    }
    label = preview_list_ui.format_plate_listbox_label(
        _list_label_host("perfect"), "p2", approved, ordinal=2
    )
    assert "N " in label
    assert "OK" not in label


def test_pz2_ready_row_is_blue_and_selected_row_keeps_status_colour():
    data = {
        "status": "needs_fix",
        "characters": [{"character": "A", "bbox": [0, 0, 10, 20]}],
        "review_state": {"status": "in_progress"},
        "gold_state": {"excluded": False},
    }
    listbox = Mock()
    host = SimpleNamespace(
        app=SimpleNamespace(
            palette={
                "model_role_vehicle": "#5aa8e8",
                "success": "#79bd73",
                "muted": "#8b949e",
                "error": "#cf7b6a",
            },
            get_list_selection_colors=lambda: ("#263c22", "#d7edcb"),
        ),
        plates_listbox=listbox,
        _listbox_pid_by_index=["p1"],
        preview_metadata={"p1": data},
        _get_review_quality_status=lambda _data: "perfect",
    )

    preview_list_ui.apply_plate_listbox_row_style(host, 0, "needs_fix")

    kwargs = listbox.itemconfig.call_args.kwargs
    assert kwargs["foreground"] == "#5aa8e8"
    assert kwargs["selectforeground"] == "#5aa8e8"


def test_pz2_selected_box_is_true_orange_not_theme_warning():
    host = SimpleNamespace(
        app=SimpleNamespace(
            palette={
                "panel": "#101010",
                "warning": "#7c651e",
                "info": "#9bc78a",
                "accent_alt": "#14b8a6",
            }
        ),
        _preview_char_geometry_inherit_down=False,
    )

    assert preview._get_preview_selected_character_box_color(host) == "#ff8a00"
    style = preview._get_preview_character_edit_grip_style(host)
    assert style["active_outline"] == "#ff8a00"


def test_inline_hud_status_is_n_ok_or_approve_without_mutating_shared_layout():
    from auto_annotation_tool.gui import z3_inline_hud

    palette = {
        "muted": "#8b949e",
        "success": "#79bd73",
        "warning": "#f59e0b",
        "model_role_vehicle": "#5aa8e8",
        "info": "#7ee7ff",
        "accent_alt": "#7ee7ff",
        "error": "#cf7b6a",
    }

    def host_for(status="needs_fix", ready=False, inherited_z2_number=""):
        return SimpleNamespace(
            app=SimpleNamespace(palette=palette),
            _preview_active_pid="p1",
            _get_preview_status_presentation=lambda **_kwargs: {
                "status": status,
                "ready_for_approval": ready,
                "inherited_z2_number": inherited_z2_number,
            },
        )

    excluded = {
        "plate_id": "p1",
        "status": "needs_fix",
        "characters": [],
        "gold_state": {"excluded": True},
    }
    assert z3_inline_hud.resolve_inline_hud_plate_status(
        host_for(), excluded
    ) == ("WYKLUCZONA", "#8b949e")

    approved = {
        "plate_id": "p1",
        "status": "perfect",
        "characters": [],
        "gold_state": {"excluded": False, "approved": True},
    }
    assert z3_inline_hud.resolve_inline_hud_plate_status(
        host_for(status="perfect"), approved
    ) == ("OK", "#79bd73")

    ready = {
        "plate_id": "p1",
        "status": "needs_fix",
        "characters": [],
        "gold_state": {"excluded": False},
    }
    assert z3_inline_hud.resolve_inline_hud_plate_status(
        host_for(status="needs_fix", ready=True), ready
    ) == ("DO KONTROLI", "#f59e0b")

    assert z3_inline_hud.resolve_inline_hud_plate_status(
        host_for(
            status="needs_fix",
            ready=True,
            inherited_z2_number="RCT962EE",
        ),
        ready,
    ) == ("ZATWIERDŹ", "#f59e0b")

    shared = [
        {
            "text": "Status tablicy: kompletne",
            "outline": "#79bd73",
        }
    ]
    hud_items = z3_inline_hud.apply_inline_hud_plate_status(
        shared,
        "OK",
        "#79bd73",
    )

    assert hud_items[0]["text"] == "Decyzja: OK"
    assert shared[0]["text"] == "Status tablicy: kompletne"
