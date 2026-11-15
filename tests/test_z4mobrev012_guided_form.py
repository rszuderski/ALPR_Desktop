"""Z4MOBREV012: contextual MT->Pose questionnaire, without hidden second CTA."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
import pytest
from auto_annotation_tool.gui import z4_mobile_sample_review as ui


class V:
    def __init__(self, value=None):
        self.value = value
    def set(self, value):
        self.value = value
    def get(self):
        return self.value


class Widget:
    def __init__(self):
        self.shown = False
        self.state = ''
        self.text = ''
    def configure(self, **kwargs):
        self.state = kwargs.get('state', self.state)
        self.text = kwargs.get('text', self.text)
    def grid(self):
        self.shown = True
    def grid_remove(self):
        self.shown = False


class Canvas:
    def __init__(self):
        self.cursor = 'hand2'
    def configure(self, **kwargs):
        self.cursor = kwargs.get('cursor', self.cursor)


class Session:
    def __init__(self):
        self.attempts = {'a1': {'id':'a1', 'mt_status':'VALID_QUAD',
                                'mt_input_evidence_entry':'frame1', 'input_width':100,
                                'input_height':100}}
        self.samples = {'c1': {'id':'c1', 'attempt_id':'a1', 'image_entry':'crop1'}}
        self.mt_invocations = {}
        self.review = SimpleNamespace(
            attempt_annotations={'a1': {}},
            sample_annotations={'c1': {}},
            subjects={'subject': {'evaluable': False, 'ground_truth':''}})
        self.calls = []
    def annotate(self, kind, key, **decision):
        self.calls.append((kind, key, decision))
        (self.review.attempt_annotations if kind=='attempt'
         else self.review.sample_annotations)[key].update(decision)
    def cancelled(self, row):
        return False


def view():
    w = ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    s = Session()
    w.session = s
    w.current_record = ('attempt','a1')
    w.current_invocation = 'inv1'
    w._image_entry = 'frame1'
    w._image = SimpleNamespace(width=100, height=100)
    w._image_source_size = (100.,100.)
    w._saving = False
    w.pose_edit_var = V(False)
    w.pose_gt_answer = V('')
    w.detection_choice = V('')
    w._pose_auto_token = None
    w._pose_gt_points = []
    w._pose_gt_cache_token = None
    w._pose_gt_drag_index = None
    w._pose_gt_before_drag = None
    w._view_drag_at = None
    w.status_var = V('')
    w.canvas = Canvas()
    w.pose_gt_host = Widget()
    w.pose_gt_hint = Widget()
    w.pose_gt_progress = Widget()
    w.pose_gt_metric = Widget()
    w.pose_edit_button = Widget()
    w.pose_gt_reset = Widget()
    w.pose_gt_radios = [Widget(), Widget()]
    w._paint_image = lambda *a, **k: None
    w._sync_review_controls = lambda: None
    w._ensure_pose_gt_points = lambda key: None
    w._pose_gt_attempt_context = lambda: (
        'a1' if w.current_record == ('attempt', 'a1')
        and w._image_entry == 'frame1' else None)
    w._change = lambda operation, *, save_draft=True, after=None: (operation(), after() if after else None)
    return w


def test_yes_only_unlocks_corners_without_hiding_model():
    w = view()
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert w.session.review.attempt_annotations['a1']['is_plate'] is True
    assert not w.pose_edit_var.get()
    assert w.canvas.cursor == 'hand2'
    assert len(w.session.calls) == 1
    assert 'pytanie 2' in w.status_var.get()


def test_image_loading_does_not_auto_start_drawing_for_saved_yes():
    w = view()
    w._image = None  # yes can be recorded before async source-image load completes
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert w.session.review.attempt_annotations['a1']['is_plate'] is True
    assert not w.pose_edit_var.get()
    w._image = SimpleNamespace(width=100, height=100)
    # Merely loading the picture must not invoke the editor; it remains a
    # readable model preview until an explicit answer in step 2.
    assert not w.pose_edit_var.get()
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    assert w.pose_edit_var.get()


def test_no_auto_repeat_on_save_refresh_or_operator_cancel():
    w = view()
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert not w.pose_edit_var.get()
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    assert w.pose_edit_var.get()
    w._toggle_pose_gt_edit()  # cancel
    assert not w.pose_edit_var.get()
    assert len(w.session.calls)==1


def test_previous_gt_not_rewritten_or_reopened():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True,
        plate_visibility='visible',pose_gt_quad_mt_input_px=[[1,1],[9,1],[9,4],[1,4]])
    w._pose_gt_points = [(1,1),(9,1),(9,4),(1,4)]
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert not w.pose_edit_var.get()
    assert w.session.calls==[]
    assert w.session.review.attempt_annotations['a1']['pose_gt_quad_mt_input_px'][0] == [1,1]


@pytest.mark.parametrize('answer,expected', [('no', False),('skip', None)])
def test_no_or_unknown_ends_pose_draft_but_preserves_saved_gt(answer,expected):
    w=view()
    w.session.review.attempt_annotations['a1']['pose_gt_quad_mt_input_px']=[[2,2],[5,2],[5,6],[2,6]]
    w.pose_edit_var.set(True)
    w._pose_gt_points=[(2,2),(5,2)]
    w.detection_choice.set(answer)
    w.save_detection_choice()
    assert w.session.review.attempt_annotations['a1']['is_plate'] is expected
    assert not w.pose_edit_var.get()
    assert w._pose_gt_points == []
    assert 'pose_gt_quad_mt_input_px' in w.session.review.attempt_annotations['a1']


def test_missing_mt_pose_or_different_image_never_invents_ground_truth():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._image_entry = 'crop1'
    assert not w._maybe_begin_guided_pose_gt()
    w._image_entry='frame1'
    w._pose_gt_attempt_context=lambda: None  # no verified Pose polygon in source
    assert not w._maybe_begin_guided_pose_gt()
    assert not w.pose_edit_var.get()
    assert w.session.calls==[]


def test_crop_yes_is_not_interpreted_as_full_mt_decision():
    w=view()
    w.current_record=('sample','c1')
    w._image_entry='crop1'
    w.current_invocation='inv1'
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert w.session.review.sample_annotations['c1']['is_plate'] is True
    assert w.session.review.attempt_annotations['a1']=={}
    assert not w.pose_edit_var.get()


def test_marking_unavailable_excludes_only_pose_but_not_mt_or_mz():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w.pose_gt_answer.set('unavailable')
    w._choose_pose_gt_answer()
    assert w.session.review.attempt_annotations['a1']['pose_gt_unavailable_reason']=='corners_not_distinguishable'
    assert w.session.review.attempt_annotations['a1']['is_plate'] is True
    assert w.session.review.subjects['subject']['evaluable'] is False
    assert w.session.review.sample_annotations['c1']=={}
    assert not w.pose_edit_var.get()


def test_reversing_unavailable_reenables_guided_drawing_without_new_cta():
    w=view()
    w.session.review.attempt_annotations['a1'].update(
        is_plate=True,evaluable=True,pose_gt_unavailable_reason='corners_not_distinguishable')
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    assert w.session.review.attempt_annotations['a1']['pose_gt_unavailable_reason'] is None
    assert w.pose_edit_var.get()
    assert w.session.review.subjects['subject']['evaluable'] is False


def test_unavailable_cannot_override_valid_existing_gt_silently():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._pose_gt_points=[(2,2),(5,2),(5,6),(2,6)]
    w.pose_gt_answer.set('unavailable')
    w._choose_pose_gt_answer()
    assert w.pose_gt_answer.get()=='mark'
    assert 'Najpierw usuń' in w.status_var.get()
    assert w.session.calls==[]


def test_geometry_hidden_for_no_or_pending_plate():
    w=view()
    w._sync_pose_gt_controls()
    assert not w.pose_gt_host.shown
    w.session.review.attempt_annotations['a1'].update(is_plate=False,evaluable=False)
    w._sync_pose_gt_controls()
    assert not w.pose_gt_host.shown


def test_geometry_question_visible_after_yes_without_original_draw_cta():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._sync_pose_gt_controls()
    assert w.pose_gt_host.shown
    assert w.pose_gt_answer.get()==''  # no preselected geometry answer
    assert not w.pose_edit_button.shown
    assert not w.pose_gt_reset.shown
    assert 'Wybierz odpowiedź poniżej' in w.pose_gt_hint.text


def test_started_editor_explicitly_explains_why_gt_is_needed():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    w._sync_pose_gt_controls()
    assert w.pose_edit_button.shown  # cancel, not initial draw
    assert 'obrys modelu' in w.pose_gt_hint.text.lower()
    assert 'Zaznacz' in w.pose_gt_hint.text


def test_correction_button_only_shown_for_existing_gt():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._pose_gt_points=[(2,2),(5,2),(5,6),(2,6)]
    # Stub metric computation is outside this test's target: deliberately
    # invalid pseudo-session lacks real Pose geometry and compare fails safely.
    w._sync_pose_gt_controls()
    assert w.pose_edit_button.shown and w.pose_gt_reset.shown
    assert 'Popraw zaznaczenie' in w.pose_edit_button.text


def test_no_stale_checkbox_in_async_save_lock():
    s=inspect.getsource(ui.MobileSampleReviewWindow._submit)
    assert 'pose_edit_checkbox' not in s
    assert 'pose_edit_button.configure' in s
    assert 'pose_gt_radios' in s


def test_questionnaire_has_no_initial_draw_cta_and_separate_mz():
    s=inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert 'Czy możesz wskazać cztery rogi tablicy?' in s
    assert 'self.pose_edit_button.grid_remove()' in s
    assert '03  ·  ODCZYT NUMERU' in s
    assert 'Nie mam pewności — pomiń ocenę obrysu' in s


def test_one_shot_autoguidance_only_for_exact_attempt():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    assert w._maybe_begin_guided_pose_gt()
    assert w._pose_auto_token==('a1','frame1')
    w._leave_pose_gt_edit_on_navigation()
    w._image_entry='crop1'
    assert not w._maybe_begin_guided_pose_gt()
    w._image_entry='frame1'
    assert not w._maybe_begin_guided_pose_gt()  # explicit operator cancellation respected
