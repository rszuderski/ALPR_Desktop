from types import SimpleNamespace
import tkinter as tk
import tkinter.font as tkfont

import pytest

from auto_annotation_tool.gui.z2_image_listbox import ImageListbox
from auto_annotation_tool.gui.z2_inline_hud import draw_inline_context
from auto_annotation_tool.gui import z2_preview_editor


@pytest.fixture
def root():
    window = tk.Tk()
    window.geometry("1100x500+40+40")
    yield window
    window.destroy()


def hud_owner(root):
    return SimpleNamespace(
        frame=root,
        _get_preview_annotation=lambda: SimpleNamespace(filename="DW009TR_DBA88AG_001.jpg"),
        _get_preview_legend_context=lambda: {
            "filename": "unused.jpg", "image_text": "Zdjęcie: 7/128",
            "vehicle_text": "Pojazd: 1/2", "pool_text": "Pula: 1000",
        },
        _get_preview_legend_theme=lambda: {},
        _preview_fullscreen_active=False,
    )


@pytest.mark.parametrize("width,scaling", [(1100, 1.333), (720, 2.0), (580, 1.333)])
def test_inline_hud_stays_left_and_does_not_collide_with_controls(root, width, scaling):
    root.tk.call("tk", "scaling", scaling)
    canvas = tk.Canvas(root, width=width, height=350)
    canvas.pack()
    root.update()
    canvas.winfo_width = lambda: width
    canvas.canvasx = lambda x: x + 120
    canvas.canvasy = lambda y: y + 70
    owner = hud_owner(root)
    z2_preview_editor._draw_preview_plate_combo_overlay(owner, canvas, [], None)
    items = canvas.find_withtag("preview_inline_hud")
    assert items and all(canvas.type(item) == "text" for item in items)
    assert all("preview_viewport_fixed" in canvas.gettags(item) for item in items)
    number = canvas.find_withtag("preview_inline_hud_number")[0]
    assert canvas.itemcget(number, "text") == "007. "
    assert canvas.coords(number) == [132.0, 80.0]
    filename = canvas.find_withtag("preview_inline_hud_filename")[0]
    assert canvas.itemcget(filename, "fill") == "#7ee7ff"
    assert tkfont.Font(root, font=canvas.itemcget(filename, "font")).actual("size") == 14
    hud_box = canvas.bbox("preview_inline_hud")
    fs_box = owner._preview_fullscreen_toggle_bbox
    # Controls share the top row on a wide viewport, wrap below on narrow ones.
    assert hud_box[2] < fs_box[0] or hud_box[3] < fs_box[1]
    assert hud_box[2] <= 120 + width
    assert canvas.find_withtag("preview_fullscreen_toggle")


def test_fullscreen_shows_the_same_left_anchored_hud(root):
    canvas = tk.Canvas(root, width=1000, height=350)
    canvas.pack()
    root.update()
    owner = hud_owner(root)
    owner._preview_fullscreen_active = True
    z2_preview_editor._draw_preview_plate_combo_overlay(owner, canvas, [], None)
    assert canvas.find_withtag("preview_inline_hud")
    number = canvas.find_withtag("preview_inline_hud_number")[0]
    assert canvas.itemcget(number, "text") == "007. "
    assert canvas.coords(number)[0] == canvas.canvasx(0) + 12
    assert canvas.find_withtag("preview_fullscreen_toggle")


def image_list(root, *, bg="#202830", fg="#ff6666", count=1000):
    annotations = [SimpleNamespace(filename=f"DW{index:05d}_001.jpg") for index in range(count)]
    owner = SimpleNamespace(current_annotations=annotations,
                            _preview_list_display_indices=list(reversed(range(count))))
    widget = ImageListbox(root, owner=owner, font=("Consolas", 10),
                          background=bg, foreground=fg, selectmode="extended",
                          exportselection=False, selectbackground="#345474", selectforeground="#ffffff")
    widget.pack(fill="both", expand=True)
    for i, actual in enumerate(owner._preview_list_display_indices):
        widget.insert("end", f"{i+1:04}. [M|BRAK GT: 1] D0.96 | {annotations[actual].filename}")
    root.update()
    return widget


@pytest.mark.parametrize("background", ["#202830", "#fafafa"])
def test_list_preserves_sorted_indices_and_separate_filename_color(root, background):
    widget = image_list(root, bg=background)
    surface = widget._surface
    files = surface.find_withtag("image_list_filename")
    statuses = surface.find_withtag("image_list_status")
    assert 0 < len(files) < 50  # 1000 rows do not create 1000 visual widgets.
    assert surface.itemcget(files[0], "text") == "DW00999_001.jpg"
    assert surface.itemcget(files[0], "fill") != surface.itemcget(statuses[0], "fill")
    widget.selection_set(0, "end")
    widget.activate(997)
    widget.see(997)
    root.update()
    assert len(widget.curselection()) == 1000
    assert widget.index("active") == 997
    # Fast Tcl path is used by the application when changing selection.
    widget.tk.call(widget._w, "selection", "clear", 0, "end")
    widget.selection_set(997)
    widget.itemconfig(997, foreground="#22bb66")
    root.update()
    assert widget.curselection() == (997,)
    row = surface.find_withtag("row_997")
    assert {surface.itemcget(item, "fill") for item in row} == {"#22bb66", "#ffffff"}
    assert widget._paint_job is None


def test_pointer_keyboard_context_and_wheel_use_native_list_behavior(root):
    widget = image_list(root, count=100)
    surface = widget._surface
    calls = []
    widget.bind("<Button-3>", lambda event: calls.append((event.widget, event.y)))
    surface.event_generate("<ButtonPress-1>", x=250, y=10)
    surface.event_generate("<ButtonRelease-1>", x=250, y=10)
    root.update()
    assert widget.curselection() == (0,)
    y = widget.bbox(2)[1] + 3
    surface.event_generate("<ButtonPress-1>", x=250, y=y, state=4)
    surface.event_generate("<ButtonRelease-1>", x=250, y=y, state=4)
    root.update()
    assert widget.curselection() == (0, 2)
    surface.event_generate("<ButtonPress-3>", x=250, y=y)
    assert calls == [(widget, y)]
    before = widget.yview()
    surface.event_generate("<MouseWheel>", delta=-120, x=250, y=y)
    root.update()
    assert widget.yview()[0] > before[0]
    widget.focus_force()
    widget.event_generate("<Down>")
    root.update()
    assert widget.index("active") == 3
    widget.delete(0, "end")
    root.update()
    assert not surface.find_all()
    widget.destroy()
    root.update()


def test_narrow_list_keeps_missing_gt_status_readable(root):
    root.geometry("350x400")
    widget = image_list(root, count=2)
    statuses = widget._surface.find_withtag("image_list_status")
    assert "BRAK GT: 1" in widget._surface.itemcget(statuses[0], "text")
    assert widget._surface.find_withtag("image_list_filename")
