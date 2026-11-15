from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gui import z2_plate_gt_runtime as runtime
from auto_annotation_tool.plate_ground_truth import (
    GROUND_TRUTH_SOURCE_ATTR,
    GROUND_TRUTH_TEXT_ATTR,
)


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeWidget:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


def plate(gt=""):
    attributes = {}
    if gt:
        attributes[GROUND_TRUTH_TEXT_ATTR] = gt
        attributes[GROUND_TRUTH_SOURCE_ATTR] = "manual_z2"
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(0.0, 0.0, 10.0, 5.0),
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        attributes=attributes,
    )


def host_for(plates, selected=0, editable=True, save_ok=True):
    ann = SimpleNamespace(filename="image.jpg", detections=plates)
    host = SimpleNamespace()
    host.plate_gt_var = FakeVar()
    host.plate_gt_context_var = FakeVar()
    host.plate_gt_entry = FakeWidget()
    host.plate_gt_save_btn = FakeWidget()
    host.plate_gt_clear_btn = FakeWidget()
    host._plate_gt_editor_syncing = False
    host._get_preview_annotation = lambda: ann
    host._get_plate_detections = lambda _ann: list(_ann.detections)
    host._get_selected_plate_index_for_ann = lambda _ann: selected
    host._preview_is_editable = lambda: editable
    host._mark_preview_image_dirty = Mock()
    host._save_preview_edits = Mock(return_value=save_ok)
    host._update_preview_edit_status = Mock()
    host._refresh_preview_canvas = Mock()
    return host, ann


def test_refresh_reads_gt_from_selected_plate_only():
    first = plate("AAA111")
    second = plate("BBB222")
    host, _ann = host_for([first, second], selected=1)

    runtime.refresh_plate_gt_editor(host)

    assert host.plate_gt_var.get() == "BBB222"
    assert host.plate_gt_context_var.get() == "Tablica 2/2 | GT: BBB222"
    assert host.plate_gt_entry.options["state"] == "normal"
    assert host.plate_gt_clear_btn.options["state"] == "normal"


def test_commit_normalizes_and_saves_gt_on_selected_plate():
    first = plate("AAA111")
    second = plate("")
    host, ann = host_for([first, second], selected=1)
    host.plate_gt_var.set(" wi 905-pw ")

    assert runtime.commit_plate_gt_editor(host) == "break"

    assert first.attributes[GROUND_TRUTH_TEXT_ATTR] == "AAA111"
    assert second.attributes[GROUND_TRUTH_TEXT_ATTR] == "WI905PW"
    assert second.attributes[GROUND_TRUTH_SOURCE_ATTR] == "manual_z2"
    host._mark_preview_image_dirty.assert_called_once_with(
        ann, refresh_list=False
    )
    host._save_preview_edits.assert_called_once()
    assert host.plate_gt_var.get() == "WI905PW"


def test_clear_removes_gt_from_selected_plate():
    selected = plate("ABC123")
    host, _ann = host_for([selected], selected=0)
    runtime.refresh_plate_gt_editor(host)

    runtime.clear_plate_gt_editor(host)

    assert GROUND_TRUTH_TEXT_ATTR not in selected.attributes
    assert GROUND_TRUTH_SOURCE_ATTR not in selected.attributes
    assert host.plate_gt_var.get() == ""


def test_failed_xml_save_restores_previous_gt():
    selected = plate("OLD123")
    host, _ann = host_for([selected], selected=0, save_ok=False)
    host.plate_gt_var.set("NEW456")

    runtime.commit_plate_gt_editor(host)

    assert selected.attributes[GROUND_TRUTH_TEXT_ATTR] == "OLD123"
    assert host.plate_gt_var.get() == "OLD123"
