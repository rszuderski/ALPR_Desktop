# Z4MOBREV010 -- MT geometry and MZ legibility are independent; GT editor is local.
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from auto_annotation_tool.gui import z4_mobile_sample_review as ui
from auto_annotation_tool.ranking.mobile_pose_metrics import POSE_GT_SCHEMA


class V:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class Canvas:
    def __init__(self):
        self.ops = []
        self.cursor = None
    def configure(self, **kwargs):
        self.cursor = kwargs.get("cursor", self.cursor)
    def create_polygon(self, *args, **kwargs):
        self.ops.append(("polygon", kwargs.get("tags", ())))
        return len(self.ops)
    def create_rectangle(self, *args, **kwargs):
        self.ops.append(("rectangle", kwargs.get("tags", ())))
        return len(self.ops)
    def create_oval(self, *args, **kwargs):
        return 4
    def create_text(self, *args, **kwargs):
        return 3
    def bbox(self, index):
        return (10, 10, 180, 30)
    def move(self, *args):
        pass
    def tag_lower(self, *args):
        pass


class FakeSession:
    def __init__(self):
        self.session_id = "test-exp"
        self.review = SimpleNamespace(
            source_archive_sha256="a" * 64,
            subjects={"s1": {"evaluable": True, "ground_truth": "ABC123"},
                      "s2": {"evaluable": False, "ground_truth": ""}},
            attempt_annotations={"a1": {"is_plate": True, "evaluable": True},
                                 "a2": {"is_plate": True, "evaluable": True}},
            sample_annotations={}, invocation_annotations={})
        self.attempts = {}
        self.samples = {}
        self.sample_ids_by_attempt = {}
        self.mt_invocations = {}
        self.entry_names = set()
        for number in (1, 2):
            a, c, i = f"a{number}", f"c{number}", f"i{number}"
            entry = f"samples/evidence/{a}.jpg"
            self.attempts[a] = dict(id=a, subject_key=f"s{number}", mt_status="VALID_QUAD",
                mt_invocation_id=i, mt_input_evidence_entry=entry,
                input_width=100, input_height=100, input_scale=1, input_pad_x=0,
                input_pad_y=0, roi_left=0, roi_top=0, roi_right=100, roi_bottom=100,
                plate_left=10, plate_top=10, plate_right=40, plate_bottom=20,
                mt_detection_index="0")
            self.samples[c] = dict(id=c, subject_key=f"s{number}", attempt_id=a,
                image_entry=f"samples/crops/{c}.jpg",
                plate_geometry={"available": True,"source_width_px":100,"source_height_px":100,
                    "plate_corners_norm": [(0.1,0.1),(0.4,0.1),(0.4,0.2),(0.1,0.2)]})
            self.sample_ids_by_attempt[a] = [c]
            self.mt_invocations[i] = SimpleNamespace(
                evidence_entries={entry}, records=[self.attempts[a]],
                cancelled=False, executed=True, execution_failed=False, detection_count=1)
            self.entry_names.add(entry)
        self.calls = []
    def cancelled(self, row):
        return False
    def set_subject(self, key, **data):
        self.review.subjects.setdefault(key, {}).update(data)
        self.calls.append(("subject", key, data))
    def annotate(self, kind, key, **data):
        self.calls.append((kind, key, data))
        self.review.attempt_annotations.setdefault(key, {}).update(data)


def make_view(session=None):
    session = session or FakeSession()
    view = ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    view.session = session
    view.pose_edit_var = V(True)
    view.canvas = Canvas()
    view._image = SimpleNamespace(width=100, height=100)
    view._image_entry = "samples/evidence/a1.jpg"
    view._image_source_size = (100, 100)
    view._view_center = (50, 50)
    view._view_pixel_scale = 1.0
    view._pose_gt_cache_token = None
    view._pose_gt_points = []
    view._pose_gt_drag_index = None
    view._pose_gt_before_drag = None
    view._view_drag_at = None
    view.current_record = ("sample", "c1")
    view.current_invocation = "i1"
    view._saving = False
    view.subject_key = "s1"
    return view


def test_editing_hides_pred_only_for_current_unfinished_gt():
    view = make_view()
    view._paint_detection_overlay(400, 400, 0, 0)
    assert not any("detection_polygon" in tags for op, tags in view.canvas.ops)
    # Change to a second observation which has unreadable *text* only.
    view._leave_pose_gt_edit_on_navigation()
    view.current_record, view.current_invocation = ("sample", "c2"), "i2"
    view._image_entry = "samples/evidence/a2.jpg"
    view._pose_gt_cache_token = None
    view._pose_gt_points = []
    view._paint_detection_overlay(400, 400, 0, 0)
    assert not view.pose_edit_var.get()
    assert any("detection_polygon" in tags for op, tags in view.canvas.ops)


def test_only_incomplete_uncommitted_draft_is_discarded_on_navigation():
    view = make_view()
    view._pose_gt_points = [(10, 10), (40, 10)]
    view._pose_gt_cache_token = ("a1", "samples/evidence/a1.jpg")
    view._pose_gt_drag_index = 1
    view._leave_pose_gt_edit_on_navigation()
    assert view._pose_gt_points == [] and view._pose_gt_cache_token is None
    assert view._pose_gt_drag_index is None and view.canvas.cursor == "hand2"
    assert not view.pose_edit_var.get()


def test_complete_gt_is_never_discarded_by_navigation():
    view = make_view()
    view._pose_gt_points = [(10, 10), (40, 10), (40, 20), (10, 20)]
    view._pose_gt_cache_token = ("a1", "samples/evidence/a1.jpg")
    view._leave_pose_gt_edit_on_navigation()
    assert len(view._pose_gt_points) == 4
    assert view._pose_gt_cache_token == ("a1", "samples/evidence/a1.jpg")
    assert not view.pose_edit_var.get()


def test_subject_navigation_exits_gt_edit():
    view = make_view()
    view._subject_ids = {"row2": "s2"}
    view.subject_tree = SimpleNamespace(selection=lambda: ("row2",))
    view.save_gt = lambda: None
    view._render_subject = lambda: None
    view._select_subject()
    assert view.subject_key == "s2" and view.pose_edit_var.get() is False


def test_switching_image_exits_gt_edit_without_modifying_session():
    view = make_view()
    view._image_rows = [("Inny dowod MT", {"image_entry": "samples/evidence/a2.jpg"})]
    view.image_choice = SimpleNamespace(current=lambda: 0)
    view._image_token = 0
    view._zoom_to_mt_pending = False
    view._paint_image = lambda *args: None
    view._sync_pose_gt_controls = lambda: None
    view._set_image_help = lambda: None
    view._sync_detection_choice = lambda: None
    view._submit = lambda work, done: done(work())
    from PIL import Image
    view.session.image = lambda row: Image.new("RGB", (100, 100))
    view._load_image()
    assert not view.pose_edit_var.get()
    assert view._image_entry == "samples/evidence/a2.jpg"
    assert view.session.calls == []


def test_refresh_same_record_does_not_disarm_operator_edit():
    source = inspect.getsource(ui.MobileSampleReviewWindow._select_record)
    assert "retained[0] != record" in source
    assert "_leave_pose_gt_edit_on_navigation()" in source
    assert "retained[0] == record" in source


def test_unreadable_MZ_subject_still_measures_MT_pose_geometry():
    session = FakeSession()
    session.review.attempt_annotations["a2"].update(
        pose_gt_schema=POSE_GT_SCHEMA, pose_gt_image_entry="samples/evidence/a2.jpg",
        pose_gt_input_size_px=[100.0, 100.0],
        pose_gt_quad_mt_input_px=[[10, 10], [40, 10], [40, 20], [10, 20]])
    geometry = {r["attempt_id"]: r for r in ui._pose_geometry_rows(session)}
    assert session.review.subjects["s2"]["evaluable"] is False
    assert geometry["a2"]["metric_status"] == "measured"
    assert geometry["a2"]["nmce_pct"] == pytest.approx(0)
    assert geometry["a2"]["polygon_iou_pct"] == pytest.approx(100)
    assert ui._pose_geometry_summary(session)["pose_measured"] == 1


def test_MZ_skip_does_not_edit_MT_invocation_or_attempt():
    session = FakeSession()
    view = make_view(session)
    view.subject_eval_choice = V("exclude")
    view.gt_var, view.note_var = V(""), V("nieczytelne znaki")
    view._change = lambda operation, **kwargs: operation()
    before_mt = dict(session.review.attempt_annotations["a1"])
    before_inv = dict(session.review.invocation_annotations)
    view.save_subject_eval_choice()
    assert session.review.subjects["s1"]["evaluable"] is False
    assert session.review.attempt_annotations["a1"] == before_mt
    assert session.review.invocation_annotations == before_inv


def test_skip_plate_truth_is_not_an_ocr_skip():
    # A different MT decision, not MZ legibility. Preserve this meaning.
    source = inspect.getsource(ui.MobileSampleReviewWindow.save_detection_choice)
    assert '"skip": {"plate_visibility": "uncertain", "is_plate": None, "evaluable": False}' in source
    build = inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert "Nie mogę ustalić, czy to tablica" in build
    assert "Pomiń ocenę odczytu numeru" in build


def test_gui_does_not_rewrite_old_sidecars_while_viewing():
    view = make_view()
    before = {k: dict(v) for k, v in view.session.review.attempt_annotations.items()}
    view._leave_pose_gt_edit_on_navigation()
    assert view.session.review.attempt_annotations == before
    assert view.session.calls == []
