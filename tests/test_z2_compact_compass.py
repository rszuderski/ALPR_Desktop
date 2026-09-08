from types import SimpleNamespace
import tkinter as tk

import pytest

from auto_annotation_tool.gui.tab_annotation import AnnotationTab
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui import z2_legend_ui as legend


@pytest.fixture
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def compass_owner(root, theme="light_visual_cs", fullscreen=False):
    owner = AnnotationTab.__new__(AnnotationTab)
    owner.frame = root
    owner.app = SimpleNamespace(palette=get_theme_palette(theme))
    owner._startup_ui_ready = True
    owner._preview_fullscreen_active = fullscreen
    owner.canvas_frame = tk.Frame(root, width=960, height=720)
    owner.canvas_frame.winfo_width = lambda: 960
    owner.canvas_frame.winfo_height = lambda: 720
    owner.preview_hint_frame = tk.Frame(owner.canvas_frame)
    owner.preview_controls_canvas = tk.Canvas(owner.preview_hint_frame, highlightthickness=0)
    owner.preview_controls_canvas.pack(fill="both", expand=True)
    owner.preview_canvas = SimpleNamespace(original_image=True)
    owner._should_freeze_preview_legend_updates = lambda: False
    context = {"filename": "ABC1234_0001.jpg", "image_text": "Zdjęcie: 1/9000",
               "plate_text": "Tablica: 1/3", "vehicle_text": "Pojazd: 0/1", "plate_total": 3, "pool_text": ""}
    owner._get_preview_legend_context = lambda: dict(context)
    return owner, context


@pytest.mark.parametrize("theme", ["light_visual_cs", "dark_visual_cs"])
@pytest.mark.parametrize("fullscreen", [False, True])
@pytest.mark.parametrize("scaling", [1.333, 2.0])
def test_compact_text_is_larger_and_stays_inside_shell(root, theme, fullscreen, scaling):
    root.tk.call("tk", "scaling", scaling)
    owner, context = compass_owner(root, theme, fullscreen)
    for filename in ("ABC1234_0001.jpg", "LONG_REGISTRATION_NAME_1234567890_ABCDEFG_123456789_001.jpg", "short.jpg"):
        context["filename"] = filename
        owner._place_preview_legend_overlay()
        canvas = owner.preview_controls_canvas
        bbox = canvas.bbox("preview_legend_context")
        assert bbox is not None
        assert float(owner.preview_hint_frame.place_info()["height"]) - bbox[3] >= 18
        assert bbox[2] <= float(owner.preview_hint_frame.place_info()["width"])
        file_item = owner._preview_controls_legend_context_item_ids["file"]
        assert canvas.itemcget(file_item, "text") == "Plik: " + filename
        assert legend._preview_controls_context_fonts(owner)[0].actual("size") == 11
        assert not owner._preview_controls_legend_scroll_enabled
        assert not canvas.find_overlapping(-10001, -10001, -9000, -9000)


def test_context_only_update_keeps_bigger_value_fonts(root):
    owner, context = compass_owner(root)
    owner._place_preview_legend_overlay()
    canvas = owner.preview_controls_canvas
    file_item = owner._preview_controls_legend_context_item_ids["file"]
    context["image_text"] = "Zdjęcie: 2/9000"
    owner._refresh_preview_controls_legend()
    assert owner._preview_controls_legend_context_item_ids["file"] == file_item
    row = owner._preview_controls_legend_context_item_ids["rows"][0]
    assert canvas.itemcget(row["value"], "text") == "2/9000"
    assert canvas.itemcget(row["value"], "font") == str(owner._get_preview_legend_font(10, "bold"))


def test_expanded_compass_keeps_original_fonts_and_scroll(root):
    owner, _ = compass_owner(root)
    owner._preview_controls_legend_inline_expanded = True
    owner.canvas_frame.winfo_height = lambda: 320
    owner._place_preview_legend_overlay()
    assert legend._preview_controls_context_fonts(owner)[0].actual("size") == 8
    assert owner._preview_controls_legend_scroll_enabled
    assert owner._preview_controls_legend_content_height > owner._preview_controls_legend_viewport_height
