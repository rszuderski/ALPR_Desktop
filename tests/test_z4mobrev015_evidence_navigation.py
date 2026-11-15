"""Z4MOBREV015: one MT image per attempt; no technical image selector in operator GUI."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import tkinter as tk
from PIL import Image

from auto_annotation_tool.gui import z4_mobile_sample_review as ui


@pytest.fixture(scope="module")
def tk_root():
    # Reuse one Tcl interpreter: repeated Tk() construction is fragile on Windows.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


class Widget:
    def __init__(self):
        self.shown = False
        self.state = None
        self.text = None
    def configure(self, **kwargs):
        self.state = kwargs.get('state', self.state)
        self.text = kwargs.get('text', self.text)
    def grid(self): self.shown = True
    def grid_remove(self): self.shown = False


def sources(rows):
    return [(str(i), {'image_entry': entry}) for i, entry in enumerate(rows)]


def test_exact_attempt_source_wins_over_other_group_evidence():
    # The same invocation contains two distinct image entries; index 0 belongs
    # to a different detected area. Picking index 0 would corrupt MT review.
    group=SimpleNamespace(evidence_entries={'other-frame.jpg','own-frame.jpg'})
    attempt={'mt_input_evidence_entry':'own-frame.jpg','evidence_entry':'source.jpg'}
    assert ui._primary_review_image_index(sources(['other-frame.jpg','own-frame.jpg','source.jpg']),attempt,group) == 1


def test_missing_exact_source_never_chooses_some_other_frame():
    group=SimpleNamespace(evidence_entries={'other-frame.jpg'})
    attempt={'mt_input_evidence_entry':'missing-frame.jpg'}
    assert ui._primary_review_image_index(sources(['other-frame.jpg','crop.jpg']),attempt,group) is None


def test_own_phone_photo_is_safe_fallback_when_mt_frame_missing():
    group=SimpleNamespace(evidence_entries={'other-frame.jpg'})
    attempt={'mt_input_evidence_entry':'missing-frame.jpg','evidence_entry':'own-phone.jpg'}
    assert ui._primary_review_image_index(sources(['other-frame.jpg','own-phone.jpg']),attempt,group)==1


def test_a_crop_is_never_the_default_input_of_mt_review():
    group=SimpleNamespace(evidence_entries=set())
    assert ui._primary_review_image_index(sources(['crop.jpg']), {'id':'a1'},group) is None


def test_crop_button_links_only_to_same_attempt():
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    w.crop_preview_button=Widget()
    w.session=SimpleNamespace(
        samples={'c2':{'id':'c2','attempt_id':'a2','image_entry':'crop2.jpg'},
                 'c1':{'id':'c1','attempt_id':'a1','image_entry':'crop1.jpg'}},
        sample_ids_by_attempt={'a1':['c1'],'a2':['c2']},
        entry_names={'crop1.jpg','crop2.jpg'})
    w._sync_crop_preview_source({'id':'a1'})
    assert w._crop_preview_row['id']=='c1'
    assert w.crop_preview_button.shown
    w._sync_crop_preview_source({'id':'a3'}, {'id':'bad','attempt_id':'a2','image_entry':'crop2.jpg'})
    assert w._crop_preview_row is None
    assert not w.crop_preview_button.shown


def test_old_session_without_crop_index_is_supported():
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    w.crop_preview_button=Widget()
    w.session=SimpleNamespace(samples={},entry_names=set())
    w._sync_crop_preview_source({'id':'a1'})
    assert w._crop_preview_row is None


def test_areas_are_from_same_observation_and_invocation_only():
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    w.subject_key='s1';w.current_invocation='mt-1';w.current_record=('attempt','a1')
    w.session=SimpleNamespace(
        subjects={'s1':{'attempts':['a1','a2','a3','a4']}},
        attempts={'a1':{'id':'a1','mt_status':'VALID_QUAD'},
                  'a2':{'id':'a2','mt_status':'VALID_QUAD'},
                  'a3':{'id':'a3','mt_status':'NO_DETECTION'},
                  'a4':{'id':'a4','mt_status':'VALID_QUAD'}},
        invocation_by_attempt={'a1':'mt-1','a2':'mt-1','a3':'mt-1','a4':'mt-2'})
    assert w._review_areas()==['a1','a2']


def test_technical_image_dropdown_not_rendered_in_tk_and_crop_not_shown_until_available(tk_root):
    root=tk_root
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    try:
        w.browser=SimpleNamespace(palette={'panel':'#f1f3e9','fg':'#1b241e','muted':'#5e6c60','accent':'#447c36'}, error='#aa2233',warning='#996600')
        w.bg='#f1f3e9';w.fg='#1b241e';w.muted='#5e6c60';w.accent='#447c36'
        w.window=tk.Toplevel(root);w.actions=[];w._image=None;w._photo=None
        w._build()
        root.update()
        assert not w.image_choice.winfo_manager()
        assert not w.crop_preview_button.winfo_ismapped()
        assert not w.target_nav.winfo_ismapped()
        assert not w.show_input_button.winfo_ismapped()
        assert w.record_label.cget('text') == 'Wybrany zapis: —'
    finally:
        w.window.destroy()


def test_crop_popup_does_not_switch_main_image_or_change_mt_state(tk_root):
    root=tk_root
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    try:
        w.window=tk.Toplevel(root); w.bg='#f3f3eb';w.fg='#191f18';w.muted='#555555'
        w._image_entry='mt-frame.jpg';w.current_record=('sample','c1');w._saving=False
        w._crop_preview_row={'id':'c1','attempt_id':'a1','image_entry':'crop.jpg'}
        w.session=SimpleNamespace(image=lambda row: Image.new('RGB',(90,25),'white'))
        w._submit=lambda work,done:done(work())
        w._show_crop_preview()
        root.update()
        assert w._crop_preview_window is not None
        assert w._crop_preview_window.winfo_exists()
        assert w._image_entry=='mt-frame.jpg'
        assert w.current_record==('sample','c1')
        w._close_crop_preview()
        assert w._crop_preview_window is None
    finally:
        w.window.destroy()


def test_current_mt_image_button_never_selects_unrelated_input():
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    w.session=SimpleNamespace(mt_invocations={'mt':SimpleNamespace(evidence_entries={'other.jpg'})}, attempts={'a':{'id':'a','mt_input_evidence_entry':'mine.jpg'}},samples={})
    w.current_record=('attempt','a');w.current_invocation='mt'
    w._image_rows=sources(['other.jpg'])
    w.status_var=SimpleNamespace(set=lambda text: setattr(w,'status',text))
    w.show_invocation_input()
    assert 'innej detekcji' in w.status


def test_overlay_checks_exact_attempt_frame_not_merely_group_membership():
    src=inspect.getsource(ui.MobileSampleReviewWindow._paint_detection_overlay)
    assert 'self._image_entry == exact_input' in src
    assert 'exact_input in explicit_inputs' in src


def test_real_tk_switching_areas_uses_each_attempts_own_mt_frame(tk_root):
    root=tk_root
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    try:
        w.window=tk.Toplevel(root);w.window.geometry('1150x760+15+15')
        w.browser=SimpleNamespace(palette={'panel':'#f1f3e9','fg':'#1b241e','muted':'#5e6c60','accent':'#447c36'}, error='#aa2233',warning='#996600')
        w.bg='#f1f3e9';w.fg='#1b241e';w.muted='#5e6c60';w.accent='#447c36';w.actions=[]
        w._image=None;w._photo=None;w._view_zoom=1.0;w._view_center=None
        w._image_token=0;w._image_source_size=None;w._image_entry=''
        w._saving=False;w._rendering=False;w._autosave_id=None;w._closed=False
        w._zoom_to_mt_pending=False;w._pose_auto_token=None
        w.current_record=None;w.current_invocation='';w.subject_key='s1'
        w._subject_stats={'s1': {'evaluable_reads':1,'mt_no_detection_invocations':0}}
        w._build()
        a1={'id':'a1','mt_status':'VALID_QUAD','mz_status':'READ','mt_detection_index':'0',
            'mt_input_evidence_entry':'e/a.jpg','evidence_entry':'e/a.jpg','input_width':640,'input_height':480}
        a2={'id':'a2','mt_status':'VALID_QUAD','mz_status':'NOT_RUN','mt_detection_index':'1',
            'mt_input_evidence_entry':'e/b.jpg','evidence_entry':'e/b.jpg','input_width':640,'input_height':480}
        c1={'id':'c1','attempt_id':'a1','image_entry':'e/crop.jpg','mz_status':'READ','prediction':'ABC1234'}
        group=SimpleNamespace(executed=True,cancelled=False,execution_failed=False,
            detection_count=2,execution_errors=[],evidence_entries={'e/a.jpg','e/b.jpg'},records=[a1,a2])
        class FakeSession:
            subjects={'s1':{'legacy_identity':False,'attempts':['a1','a2'],'samples':['c1']}}
            attempts={'a1':a1,'a2':a2};samples={'c1':c1}
            mt_invocations={'mt-1':group};invocation_by_attempt={'a1':'mt-1','a2':'mt-1'}
            entry_names={'e/a.jpg','e/b.jpg','e/crop.jpg'}
            sample_ids_by_attempt={'a1':['c1']};attempts_available=True
            review=SimpleNamespace(subjects={},attempt_annotations={},sample_annotations={},
                review_mode='assisted',review_mode_locked=False)
            @staticmethod
            def subject_draft(key): return ''
            @staticmethod
            def predictions_visible(key): return True
            @staticmethod
            def cancelled(row): return False
            @staticmethod
            def invocation_annotation(key): return {}
            @staticmethod
            def image(row): return Image.new('RGB',(640,480),'white')
        w.session=FakeSession()
        w._submit=lambda work,done,**kw:done(work())
        w._render_subject()
        root.update()
        assert w.current_record==('sample','c1')
        assert w._image_entry=='e/a.jpg'
        assert w.target_nav.winfo_ismapped()
        assert w.target_position.cget('text')=='Obszar 1 z 2'
        assert w.crop_preview_button.winfo_ismapped()
        w._switch_detected_area(+1)
        root.update()
        assert w.current_record==('attempt','a2')
        assert w._image_entry=='e/b.jpg'
        assert w.target_position.cget('text')=='Obszar 2 z 2'
        assert not w.crop_preview_button.winfo_ismapped()
        w._switch_detected_area(-1)
        root.update()
        assert w.current_record==('attempt','a1')
        assert w._image_entry=='e/a.jpg'
    finally:
        w.window.destroy()
