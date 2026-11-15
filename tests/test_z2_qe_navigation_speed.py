import copy
import os
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from auto_annotation_tool.gui import z2_preview_images as images
from auto_annotation_tool.gui import z2_preview_state as state
from auto_annotation_tool.gui import z2_gt_pack_runtime as gt_runtime
from auto_annotation_tool.gui.z2_canvas_metrics_ui import render_preview_metrics_table
from auto_annotation_tool.gui.zoomable_canvas import ZoomableCanvas
from auto_annotation_tool.gt_pack import fingerprint_image
from test_z2_superzoom_delete import make_owner


def image_file(path, color="red", size=(10, 10)):
    Image.new("RGB", size, color).save(path)
    return path


def wait_prefetch(loader):
    with loader.lock:
        worker = loader.worker
    if worker is not None:
        worker.join(5)
        assert not worker.is_alive()


def test_prefetch_and_navigation_decode_a_file_only_once(tmp_path):
    path = image_file(tmp_path / "one.png")
    entered, release = threading.Event(), threading.Event()
    calls = []
    def decode(path):
        calls.append(path)
        entered.set()
        assert release.wait(5)
        return images.decode_image(path)
    loader = images.PreviewImages(decoder=decode)
    loader.prefetch([path])
    assert entered.wait(2)
    worker = loader.worker
    for _ in range(20):
        loader.prefetch([path])
        assert loader.worker is worker
    result = []
    foreground = threading.Thread(target=lambda: result.append(loader.load(path)))
    foreground.start()
    release.set()
    foreground.join(5)
    wait_prefetch(loader)
    assert calls == [path]
    assert result[0] is loader.load(path)
    loader.prefetch([path])
    assert loader.worker is None


def test_reversing_direction_discards_obsolete_queued_work(tmp_path):
    paths = [image_file(tmp_path / f"{i}.png") for i in range(4)]
    entered, release = threading.Event(), threading.Event()
    calls = []
    def decode(path):
        calls.append(path)
        if path == paths[0]:
            entered.set()
            assert release.wait(5)
        return images.decode_image(path)
    loader = images.PreviewImages(decoder=decode)
    loader.prefetch(paths[:3])
    assert entered.wait(2)
    loader.prefetch([paths[3]])
    release.set()
    wait_prefetch(loader)
    assert calls == [paths[0], paths[3]]


def test_lru_has_byte_limit_and_changed_files_reload(tmp_path):
    paths = [image_file(tmp_path / f"{i}.png") for i in range(3)]
    loader = images.PreviewImages(max_bytes=600, max_items=24)
    first = loader.load(paths[0])
    loader.load(paths[1])
    assert loader.load(paths[0]) is first
    loader.load(paths[2])
    assert {key[0] for key in loader.cache} == {images.path_key(paths[0]), images.path_key(paths[2])}
    old_stat = paths[0].stat()
    image_file(paths[0], "blue")
    os.utime(paths[0], ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 10_000_000))
    new = loader.load(paths[0])
    assert new is not first and new.getpixel((0, 0)) == (0, 0, 255)
    assert sum(loader._size(image) for image in loader.cache.values()) <= 600


def test_corrupt_prefetch_does_not_block_later_images(tmp_path):
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_text("not an image")
    valid = image_file(tmp_path / "valid.png", "blue")
    loader = images.PreviewImages()
    loader.prefetch([corrupt, valid])
    wait_prefetch(loader)
    assert loader.load(valid).getpixel((0, 0)) == (0, 0, 255)
    assert not loader.inflight


def test_prefetch_thread_start_failure_keeps_foreground_loading_available(tmp_path, monkeypatch):
    path = image_file(tmp_path / "one.png")
    loader = images.PreviewImages()
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("cannot start thread")))
    loader.prefetch([path])
    assert loader.worker is None
    assert loader.load(path).getpixel((0, 0)) == (255, 0, 0)


def test_unicode_paths_and_exif_orientation_agree_with_gt_identity(tmp_path, monkeypatch):
    path = tmp_path / "żółć_123.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40), "red").save(path, exif=exif)
    decoded = images.PreviewImages().load(path)
    assert decoded.size == (40, 80)
    expected = fingerprint_image(path)
    monkeypatch.setattr(Image.Image, "load", Mock(side_effect=AssertionError("GT restore must not decode pixels")))
    actual = gt_runtime.fingerprint_preview_image(SimpleNamespace(), path)
    assert (actual["width"], actual["height"]) == decoded.size
    assert actual["image_id"] == expected["image_id"]


def test_qe_keeps_sorted_plate_order_and_pixels_in_sync(tmp_path, monkeypatch):
    host = make_owner(2, 0)
    annotations = [make_owner(count, 0).current_annotations[0] for count in (2, 1, 2)]
    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    for i, ann in enumerate(annotations):
        ann.filename = f"image-{i}.png"
        image_file(tmp_path / ann.filename, colors[i], (600, 400))
    before_gt = [[copy.deepcopy(det.attributes) for det in ann.plates] for ann in annotations]
    host.current_annotations = annotations
    host.current_preview_index = 2
    nav = [2, 0, 1]
    selection = {"row": 0}
    host.preview_listbox = SimpleNamespace(
        curselection=lambda: (selection["row"],), index=lambda kind: selection["row"],
        selection_set=lambda i: selection.update(row=i), activate=lambda i: None,
        see=lambda i: None, size=lambda: 3,
    )
    host._get_preview_navigation_actual_indices = lambda: nav
    host._get_preview_actual_index_from_display = lambda i: nav[i]
    host._get_preview_display_index = lambda i: nav.index(i)
    host._clear_listbox_selection_fast = lambda box: None
    host._mark_preview_user_interaction = Mock()
    host._defer_preview_autosave_for_navigation = Mock()
    host._resolve_preview_image_path = lambda ann: tmp_path / ann.filename
    host._clear_preview_legend_image_cache = Mock()
    host._update_preview_toolbar_state = Mock()
    host._select_preview_index_for_super_correction = MethodType(state._select_preview_index_for_super_correction, host)
    host.preview_canvas.set_image_preserve_view.side_effect = lambda image, **kw: setattr(host.preview_canvas, "original_image", image)
    host.preview_canvas.original_image = images.decode_image(tmp_path / annotations[2].filename)
    captured = []
    def draw(**kwargs):
        captured.append((host.current_preview_index, host.preview_canvas.original_image.getpixel((0, 0)),
                         host.preview_canvas.view["polygon"]))
    host.preview_canvas._update_display.side_effect = draw
    monkeypatch.setattr(state, "_schedule_preview_neighbor_prefetch", Mock())
    expected = [(2, 1), (0, 0), (0, 1), (1, 0), (0, 1), (0, 0), (2, 1)]
    for step, (index, plate) in zip([1, 1, 1, 1, -1, -1, -1], expected):
        assert state._select_preview_global_plate_relative(host, step) == "break"
        assert captured[-1] == (index, colors[index], tuple(annotations[index].plates[plate].polygon))
    assert [[det.attributes for det in ann.plates] for ann in annotations] == before_gt
    assert not host._preview_dirty_images


@pytest.fixture(scope="module")
def root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    for after_id in root.tk.call("after", "info"):
        root.tk.call("after", "cancel", after_id)
    root.destroy()


def test_metrics_update_existing_widgets_and_remove_obsolete_rows(root):
    import tkinter as tk
    body = tk.Frame(root)
    tk.Label(body, text="old placeholder").pack()
    render_preview_metrics_table(body, [("Status", "OK", "success"), ("GT", "1/1", "success")], {})
    original = list(body.winfo_children())
    assert len(original) == 2
    for i in range(20):
        render_preview_metrics_table(body, [("Status", "OK", "success"), ("GT", f"{i}/20", "warning")], {})
    assert list(body.winfo_children()) == original
    assert body._z2_metric_rows[1][2].cget("text") == "19/20"
    render_preview_metrics_table(body, [("Status", "NOK", "error")], {})
    assert len(body.winfo_children()) == 1
    assert body._z2_metric_rows[0][2].cget("text") == "NOK"
    body.destroy()


def test_canvas_reuses_image_item_but_replaces_pixels_and_boxes(root):
    canvas = ZoomableCanvas(root, width=300, height=200)
    canvas.winfo_width = lambda: 300
    canvas.winfo_height = lambda: 200
    position = [10]
    canvas.set_overlay_renderer(lambda c: c.create_rectangle(position[0], 10, position[0] + 40, 50, tags="box"))
    canvas.set_image_fit_to_view(Image.new("RGB", (600, 400), "red"), interaction_fast=True)
    image_id = canvas.image_id
    first_box = canvas.find_withtag("box")[0]
    position[0] = 80
    canvas.set_image_fit_to_view(Image.new("RGB", (600, 400), "blue"), interaction_fast=True)
    assert canvas.image_id == image_id
    assert not canvas.type(first_box)
    assert len(canvas.find_withtag("box")) == 1
    assert canvas.coords(canvas.find_withtag("box")[0]) == [80, 10, 120, 50]
    assert tuple(canvas.tk.call(str(canvas.photo_image), "get", 5, 5)) == (0, 0, 255)
    canvas._cancel_final_quality_display()
    canvas.destroy()


def test_navigation_crop_covers_same_view_and_idle_restores_pan_buffer(root):
    canvas = ZoomableCanvas(root)
    canvas.winfo_width = lambda: 300
    canvas.winfo_height = lambda: 200
    canvas.original_image = Image.new("RGB", (1000, 800))
    canvas.zoom_level = 2.0
    canvas.pan_data["x"] = -500
    canvas.pan_data["y"] = -400
    canvas._navigation_rendering = True
    fast = canvas._get_visible_image_region(interaction_fast=True)
    canvas._navigation_rendering = False
    quality = canvas._get_visible_image_region(interaction_fast=False)
    assert fast["visible_bounds"] == quality["visible_bounds"]
    assert fast["draw_width"] < quality["draw_width"]
    assert fast["draw_height"] < quality["draw_height"]
    canvas.destroy()
