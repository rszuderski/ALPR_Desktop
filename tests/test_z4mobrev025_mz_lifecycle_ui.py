"""Z4MOBREV025: consistent read-only display for interrupted MT/MZ evidence.

The actual methods are extracted from the shipped GUI source, not reimplemented.
No real source archive, GT or human-review file is opened/written.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

GUI_PATH=Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_sample_review.py'
SOURCE=GUI_PATH.read_text(encoding='utf8')
METHODS={'_mz_review_state','_sync_mz_panel_state','_sync_subject_eval_choice',
         '_sync_mz_readability_details','_on_subject_unreadable_toggle',
         'save_subject_eval_choice','_schedule_autosave','_persist_subject_fields','save_gt'}


def load_real_methods():
    mod=ast.parse(SOURCE)
    cls=next(n for n in mod.body if isinstance(n,ast.ClassDef) and n.name=='MobileSampleReviewWindow')
    nodes=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in METHODS]
    assert {n.name for n in nodes}==METHODS
    isolated=ast.Module(body=[ast.ClassDef(name='StageHarness',bases=[],keywords=[],
        body=nodes,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace={'normalize_registration':lambda x:str(x or '').strip().upper()}
    exec(compile(isolated,str(GUI_PATH),'exec'),namespace)
    return namespace['StageHarness']

Harness=load_real_methods()

class FakeWidget:
    def __init__(self):
        self.visible=True
        self.options={}
        self.deleted=0
    def grid(self,**kwargs):self.visible=True
    def grid_remove(self):self.visible=False
    def configure(self,**kwargs):self.options.update(kwargs)
    config=configure
    def delete(self,*args):self.deleted+=1

class FakeVar:
    def __init__(self,value=''):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value

class FakeSession:
    def __init__(self):
        self.attempts={'a':{'id':'a','mz_status':'NOT_RUN','subject_key':'s'}}
        self.samples={}
        self.mt_invocations={'inv':SimpleNamespace(cancelled=True,executed=True,execution_failed=False)}
        self.review=SimpleNamespace(subjects={'s':{'ground_truth':'DW69374','evaluable':True,'note':'zapisane wcześniej'}})
        self.calls=[]
    def cancelled(self,row):return bool(row.get('stale_or_cancelled'))
    def set_subject(self,*a,**kw):self.calls.append((a,kw))


def window():
    h=Harness()
    h.session=FakeSession()
    h.current_record=('attempt','a')
    h.current_invocation='inv'
    h.subject_key='s'
    h._result_row=h.session.attempts['a']
    h._current_mt_attempt=lambda:h.session.attempts['a']
    h._multi_group=lambda:None
    h._saving=False
    h._rendering=False
    h._autosave_id=None
    h._change=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('unexpected _change'))
    h.window=SimpleNamespace(after_cancel=lambda _:None)
    h.gt_var=FakeVar('DW69374');h.note_var=FakeVar('zapisane wcześniej')
    h.subject_eval_choice=FakeVar('assess')
    h.subject_unreadable_var=FakeVar(False)
    h.subject_unreadable_button=FakeWidget()
    for name in ('mz_title','gt_prompt_label','gt_entry','mz_evaluable_group',
                 'mz_exclusion_caption','mz_note_caption','note_entry',
                 'mz_archived_gt','result_context','prediction_label','alignment_label',
                 'alignment_text','crop_preview_button','mode_choice'):
        setattr(h,name,FakeWidget())
    h.subject_eval_radios=[FakeWidget(),FakeWidget()]
    return h


def test_cancelled_mz_is_read_only_and_saved_gt_is_archival():
    h=window()
    assert h._mz_review_state()=='cancelled'
    h._sync_mz_panel_state()
    assert h.mz_title.options['text']=='02 · MZ — POZA WERYFIKACJĄ'
    for name in ('gt_prompt_label','gt_entry','mz_evaluable_group','mz_exclusion_caption',
                 'mz_note_caption','note_entry'):
        assert not getattr(h,name).visible, name
    assert h.mz_archived_gt.visible
    assert 'DW69374' in h.mz_archived_gt.options['text']
    assert 'nie jest wynikiem weryfikacji MZ' in h.mz_archived_gt.options['text']
    assert h.gt_entry.options['state']=='disabled'
    assert h.subject_unreadable_button.options['state']=='disabled'
    assert 'CER' in h.alignment_label.options['text']
    assert h.alignment_text.deleted>0
    assert h.session.review.subjects['s']['ground_truth']=='DW69374'


def test_cancelled_attempt_cannot_mutate_gt_even_by_direct_callbacks():
    h=window()
    h.gt_var.set('FAKE')
    h._persist_subject_fields('s','FAKE','')
    h._schedule_autosave()
    h.save_subject_eval_choice()
    h.save_gt()
    assert h.session.calls==[]
    assert h.session.review.subjects['s']['ground_truth']=='DW69374'


def test_any_cancelled_attempt_blocks_even_when_invocation_flag_is_false():
    h=window()
    h.session.mt_invocations['inv'].cancelled=False
    h.session.attempts['a']['stale_or_cancelled']=True
    assert h._mz_review_state()=='cancelled'


@pytest.mark.parametrize('executed,failed',[(False,False),(True,True)])
def test_not_executed_or_execution_failure_has_no_ocr_actions(executed,failed):
    h=window()
    g=h.session.mt_invocations['inv']
    g.cancelled=False;g.executed=executed;g.execution_failed=failed
    assert h._mz_review_state()=='technical'
    h._sync_mz_panel_state()
    assert not h.gt_entry.visible
    assert 'Brak wyniku MZ'==h.prediction_label.options['text']
    h.save_gt()
    assert not h.session.calls


def test_mz_not_run_active_keeps_independent_gt_but_not_ocr_claims():
    h=window();h.session.mt_invocations['inv'].cancelled=False
    assert h._mz_review_state()=='not_run'
    h._sync_mz_panel_state()
    assert h.gt_entry.visible and h.mz_evaluable_group.visible
    assert not h.mz_archived_gt.visible
    assert 'NUMER TABLICY — MZ NIE URUCHOMIONO' in h.mz_title.options['text']
    assert h.gt_prompt_label.options['text']=='Jaki numer widzisz na zdjęciu?'
    assert h.subject_unreadable_button.options['state']=='normal'
    assert not h.mz_note_caption.visible
    assert not h.note_entry.visible
    assert 'Nie powstaje przez to wynik MZ ani CER' in h.alignment_label.options['text']


def test_switch_cancelled_to_measured_restores_ui_in_same_window():
    h=window();h._sync_mz_panel_state()
    h.session.mt_invocations['inv'].cancelled=False
    h.session.attempts['a']['mz_status']='READ'
    assert h._mz_review_state()=='measured'
    h._sync_mz_panel_state()
    for name in ('gt_prompt_label','gt_entry','mz_evaluable_group'):
        assert getattr(h,name).visible, name
    assert not h.mz_note_caption.visible and not h.note_entry.visible
    assert not h.mz_archived_gt.visible
    assert 'SPRAWDZENIE NUMERU TABLICY' in h.mz_title.options['text']
    assert h.subject_unreadable_button.options['state']=='normal'
    assert h.gt_entry.options['state']=='normal'


def test_status_matrix_cannot_count_cancelled_archival_gt_as_measured():
    assert 'statuses = tuple("—" if idx == 2 else cell for idx, cell in enumerate(statuses))' in SOURCE
    assert '(" · przerwana" if excluded else "")' in SOURCE


def test_form_context_depends_on_lifecycle():
    assert 'intro = ("Próba przerwana.' in SOURCE
    assert 'Model MZ nie uruchomił się dla tej próby' in SOURCE
    assert 'GT narożników służy' in SOURCE


def test_unreadable_control_shows_reason_and_disables_gt_but_not_mt():
    h=window();h.session.mt_invocations['inv'].cancelled=False
    h.session.review.subjects['s']={'ground_truth':'','evaluable':False,'note':'rozmazany obraz'}
    h.gt_var.set('');h.note_var.set('rozmazany obraz')
    h._sync_subject_eval_choice();h._sync_mz_panel_state()
    assert h.subject_unreadable_var.get() is True
    assert h.gt_entry.options['state']=='disabled'
    assert h.mz_note_caption.visible and h.note_entry.visible
    assert 'CER' in h.mz_exclusion_caption.options['text'] or h._mz_review_state()=='not_run'

def test_blank_unchecked_gt_remains_pending_not_unreadable():
    h=window();h.session.mt_invocations['inv'].cancelled=False
    h.session.review.subjects['s']={'ground_truth':'','evaluable':None,'note':''}
    h.gt_var.set('');h.note_var.set('')
    h._sync_subject_eval_choice();h._sync_mz_panel_state()
    assert h.subject_unreadable_var.get() is False
    assert h.gt_entry.options['state']=='normal'
    assert not h.mz_note_caption.visible
