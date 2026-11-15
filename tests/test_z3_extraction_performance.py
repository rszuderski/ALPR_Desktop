from pathlib import Path
from threading import Thread, get_ident
from unittest.mock import Mock

from PIL import Image

from auto_annotation_tool.gt_pack import ALPRGTPack, fingerprint_image, fingerprint_image_identity
from auto_annotation_tool.gui.z3_extraction_progress import ExtractionProgress


def test_plate_resolution_reads_only_its_own_history_and_observes_other_writers(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (40, 20), (20, 30, 40)).save(image)
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    polygon = [(1, 1), (30, 1), (30, 15), (1, 15)]
    pack.upsert_z2_plate(image_path=image, polygon=polygon,
                         plate_annotation_id="target", ground_truth_text="ABC123")
    pack.upsert_z2_plate(image_path=image, polygon=polygon,
                         plate_annotation_id="other", ground_truth_text="DEF456")
    plate = pack.get_plate("target")
    expected = {
        "gt": {r["revision_id"]: r for r in pack.list_revisions() if r["revision_id"] in plate["revision_ids"]},
        "geometry": {r["geometry_id"]: r for r in pack.list_geometries() if r["geometry_id"] in plate["geometry_revision_ids"]},
        "layout": {r["layout_revision_id"]: r for r in pack.list_layout_revisions() if r["layout_revision_id"] in plate.get("layout_revision_ids", [])},
    }
    monkeypatch.setattr(pack, "_list_records", Mock(side_effect=AssertionError("full pack scan")))
    assert pack._revision_record_map(plate) == expected["gt"]
    assert pack._geometry_record_map(plate) == expected["geometry"]
    assert pack._layout_revision_record_map(plate) == expected["layout"]
    assert pack.resolve_ground_truth("target")["text"] == "ABC123"
    assert pack.resolve_plate_geometry("target")["resolved"]
    # No global cache: changes made through another pack instance are visible.
    second = ALPRGTPack.open(pack.root)
    second.set_ground_truth("target", "NEW123", source="manual_z2")
    assert pack.resolve_ground_truth("target")["text"] == "NEW123"
    for method in (pack._revision_record_map, pack._geometry_record_map, pack._layout_revision_record_map):
        assert method({}) == {}


def test_light_identity_keeps_exact_hash_and_exif_dimensions_without_pixel_decode(tmp_path, monkeypatch):
    path = tmp_path / "tablica_ąę.jpg"
    original = Image.new("RGB", (61, 29), (40, 80, 120))
    exif = Image.Exif()
    exif[274] = 6
    original.save(path, exif=exif)
    expected = fingerprint_image(path)
    monkeypatch.setattr(Image.Image, "load", Mock(side_effect=AssertionError("second pixel decode")))
    actual = fingerprint_image_identity(path)
    assert actual == {key: expected[key] for key in actual}
    assert (actual["width"], actual["height"]) == (29, 61)


class Frame:
    def __init__(self):
        self.thread = get_ident()
        self.jobs = {}
        self.serial = 0
        self.binding = None

    def assert_ui(self):
        assert get_ident() == self.thread, "Tk operation from worker"

    def bind(self, _event, callback, **_kwargs):
        self.assert_ui()
        self.binding = callback
        return "destroy"

    def unbind(self, *_args):
        self.assert_ui()
        self.binding = None

    def after(self, _delay, callback):
        self.assert_ui()
        self.serial += 1
        self.jobs[self.serial] = callback
        return self.serial

    def after_cancel(self, job):
        self.assert_ui()
        self.jobs.pop(job, None)

    def tick(self):
        jobs, self.jobs = self.jobs, {}
        for callback in jobs.values():
            callback()


def test_worker_progress_coalesces_thousands_of_images_without_calling_tk():
    frame = Frame()
    present = Mock()
    progress = ExtractionProgress(frame, present, lambda: True)
    def worker():
        for index in range(1000):
            progress.publish(index + 1, 1000, index + 1)
    thread = Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(frame.jobs) == 1
    frame.tick()
    present.assert_called_once_with(1000, 1000, 1000, "crop")
    closer = Thread(target=progress.close)
    closer.start()
    closer.join(timeout=2)
    frame.tick()
    assert not frame.jobs and frame.binding is None


def test_old_progress_cannot_overwrite_new_project_or_completion_status():
    frame = Frame()
    current = [True]
    present = Mock()
    progress = ExtractionProgress(frame, present, lambda: current[0])
    progress.publish(10, 1000, 11)
    current[0] = False
    frame.tick()
    present.assert_not_called()
    assert not frame.jobs
    assert frame.binding is None


def test_window_destruction_cancels_progress_timer():
    from types import SimpleNamespace
    frame = Frame()
    progress = ExtractionProgress(frame, Mock(), lambda: True)
    frame.binding(SimpleNamespace(widget=frame))
    assert not frame.jobs and frame.binding is None
    progress.publish(1, 10, 1)
    assert progress._latest is None
