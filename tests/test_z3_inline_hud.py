from types import SimpleNamespace
import tkinter as tk

import pytest

from auto_annotation_tool.gui.z3_inline_hud import plan_inline_hud, draw_inline_hud


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


@pytest.mark.parametrize("fullscreen", [False, True])
@pytest.mark.parametrize("width", [340, 600, 1200])
def test_hud_is_text_only_left_anchored_and_fits_narrow_and_fullscreen_canvas(root, width, fullscreen):
    host = SimpleNamespace(frame=root, _preview_fullscreen_active=fullscreen,
                           _get_current_preview_list_index=lambda: 9,
                           _listbox_pid_by_index=list(range(1171)))
    data = {"source_image": "ABC1234_very_long_source_image_name_with_extra_details.jpg"}
    statuses = {
        "neutral_badges": [{"text": "Ramki: 7", "outline": "#b6f5ce"}],
        "badges": [{"text": "Odczyt: [ABC1234]", "outline": "#2ecc71"},
                   {"text": "Układ: AUTO 1R", "outline": "#2ecc71",
                    "tags": ("preview_action::toggle_plate_layout",)},
                   {"text": "Status tablicy: do korekty", "outline": "#e74c3c"}],
    }
    canvas = tk.Canvas(root, width=width, height=600)
    try:
        plan = plan_inline_hud(host, width, data, statuses)
        draw_inline_hud(host, canvas, plan)
        items = canvas.find_withtag("preview_inline_hud")
        assert items and all(canvas.type(item) == "text" for item in items)
        for item in items:
            x1, y1, x2, y2 = canvas.bbox(item)
            assert x1 >= 9 and y1 >= 6 and x2 <= width - 8
            assert y2 <= plan["bar_height"]
        numbers = canvas.find_withtag("preview_inline_hud_number")
        assert canvas.itemcget(numbers[-1], "text") == "0010. "
        assert canvas.coords(numbers[-1]) == [12, 8]
        filenames = canvas.find_withtag("preview_inline_hud_filename")
        assert canvas.bbox(filenames[-1])[2] <= width - 80
        assert "preview_action::edit_source_filename" in canvas.gettags(filenames[-1])
        assert canvas.find_withtag("preview_action::toggle_plate_layout")
        counters = [canvas.itemcget(item, "text") for item in canvas.find_withtag("preview_inline_hud_counts")]
        assert "Tablica 10/1171" in counters and "Ramki: 7" in counters
    finally:
        canvas.destroy()
