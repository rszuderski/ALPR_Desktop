"""Z4MOBREP023: unambiguous scientific cards, primary CTA, binary stage data."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from auto_annotation_tool.gui.mobile_report_science_dashboard import build_overview
from auto_annotation_tool.ranking.mobile_verification_policy import observation_verification_cells,visual_decision
from test_z4mobrev022_form_verification import FakeSession


def evidence():
 r=SimpleNamespace(report_id='r1',source_archive_sha256='a'*64,raw={},latency={
   'mt':{'count':7,'median_ms':665,'p95_ms':709},'mz':{'count':7,'median_ms':908,'p95_ms':941},
   'pipeline':{'count':7,'median_ms':2104,'p95_ms':2568}})
 b=SimpleNamespace(source_archive_sha256='a'*64,validation=SimpleNamespace(ok=True),
   report=SimpleNamespace(report_id='r1'),report_payload={'latency':r.latency,
   'frame_flow':{'frames_received':240,'frames_processed':7,'frames_skipped_frame_gate':233}})
 s={'source_archive_sha256':'a'*64,'scientific_state':'PRELIMINARY','review_mode':'assisted',
    'form_verification':{'status':'IN_PROGRESS','total':8,'verified':1,'issues':tuple(range(9))},
    'progress':{'mt_executed_active':7,'mt_invocations_cardinality_reviewed':5},
    'mt':{'single_plate_n':3,'localized_n':3},
    'pose':{'measured_n':2,'polygon_iou_mean_pct':77.6,'nmce_mean_pct':4.3},
    'mz':{'evaluable_reads':4,'exact_n':2,'cer':0.483},
    'alpr':{'evaluable_subjects':5,'exact_n':2}}
 return r,b,s


def test_8_observations_vs_7_invocations_are_not_conflated():
 out=build_overview(*evidence())
 assert out['progress']=='Kompletny formularz: 1 z 8 obserwacji'
 assert out['progress_ratio']==1/8
 assert '5/7 wywołaniach MT' in out['progress_subtitle']
 assert '9' in out['progress_subtitle']


def test_explicit_metrics_for_four_scientific_stages():
 out=build_overview(*evidence())
 a,b,c,d=out['cards']
 assert a['value']=='3/3 trafnych wskazań' and 'wywołania z jedną rzeczywistą tablicą' in a['note']
 assert b['value']=='IoU 77,6%' and '2 sparowane obrysy' in b['note']
 assert c['value']=='2/4 poprawnych' and 'cropów' in c['note'] and 'CER' in c['note']
 assert d['value']=='2/5 poprawnych' and 'obserwacji' in d['note']
 assert all('(telefon)' in card['title'] for card in out['cards'])


def test_frame_gate_is_not_model_false_negative():
 assert 'nie przez model MT' in build_overview(*evidence())['frames']


def test_missing_science_disables_all_quality_cards():
 r,b,_=evidence();out=build_overview(r,b,None)
 assert out['quality_verified'] is False
 assert all(card['value'].startswith('Brak ') for card in out['cards'])


def test_wrong_source_sha_disables_quality():
 r,b,s=evidence();r.source_archive_sha256='b'*64
 out=build_overview(r,b,s)
 assert not out['source_verified'] and not out['quality_verified']
 assert all(card['value'].startswith('Brak ') for card in out['cards'])


@pytest.mark.parametrize('status,text',[
 ('READY_TO_APPROVE','czeka na zatwierdzenie'),
 ('READY_FOR_EVALUATION','Weryfikacja zatwierdzona'),
])
def test_status_closes_form_not_scientific_result(status,text):
 r,b,s=evidence();s['form_verification']={'status':status,'total':8,'verified':8,'issues':()}
 out=build_overview(r,b,s)
 assert text in out['progress'] and out['progress_ratio']==1
 assert out['cards'][0]['value']=='3/3 trafnych wskazań'


def test_optional_polygon_does_not_block_form(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('accurate'))
 assert observation_verification_cells(s,'subject1',{'attempt1':{'eligible':True,'metric_status':'missing_manual_gt'}})==('✓','✓','✓','○','✓')


def test_measured_polygon_is_present(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('imprecise'))
 assert observation_verification_cells(s,'subject1',{'attempt1':{'eligible':True,'metric_status':'measured'}})==('✓','✓','✓','✓','✓')


def test_false_positive_choice_is_decision_not_detection_success(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('not_plate'))
 assert observation_verification_cells(s,'subject1')[1]=='✓'


def test_missing_cardinality_is_pending(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('accurate'))
 s.invocation_annotation=lambda _key:{}
 result=observation_verification_cells(s,'subject1')
 assert result[0]=='○' and result[-1]=='○'


def test_no_detection_and_no_mz_have_no_fake_source_metrics(tmp_path):
 s=FakeSession(tmp_path,mt='NO_DETECTION',mz='NOT_RUN',gt='')
 assert observation_verification_cells(s,'subject1')==('✓','—','—','—','✓')


def test_cancelled_invocation_is_not_quality_failure(tmp_path):
 s=FakeSession(tmp_path,mt='VALID_QUAD',mz='NOT_RUN',gt='')
 s.mt_invocations['invocation1'].cancelled=True
 assert observation_verification_cells(s,'subject1')==('—','—','—','—','✓')


def test_no_gt_remains_pending_when_mz_ran(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('accurate'),gt='')
 assert observation_verification_cells(s,'subject1')[2:]==('○','—','○')


def test_excluded_gt_with_reason_does_not_block_form(tmp_path):
 s=FakeSession(tmp_path,annotation=visual_decision('accurate'),gt='',excluded=True)
 assert observation_verification_cells(s,'subject1')[2:]==('—','—','✓')


def test_gui_list_is_compact_and_main_action_is_prominent():
 root=Path(__file__).resolve().parents[1]
 gui=(root/'auto_annotation_tool/gui/z4_mobile_sample_review.py').read_text(encoding='utf8')
 dash=(root/'auto_annotation_tool/gui/mobile_report_science_dashboard.py').read_text(encoding='utf8')
 ast.parse(gui);ast.parse(dash)
 assert 'cols = ("Nr", "MT", "Wsk.", "GT", "Rogi", "Form.")' in gui
 block=gui.split('    def _refresh_subject_rows(self):')[1].split('    def _select_subject')[0]
 assert 'observation_verification_cells(self.session, key' in block
 assert '_observation_list_label' not in block
 assert 'Przejdź do weryfikacji próbek' in dash and 'Oceń zdjęcia' not in dash
 assert 'command=on_review' in dash and 'action_button.grid(row=0, column=0, sticky="ew")' in dash
 # AKT89F: the 1180px Browser leaves about 560px for results. Verify the
 # visible two-column layout instead of requiring the obsolete 800px cutoff.
 import tkinter as tk
 from auto_annotation_tool.gui.mobile_report_science_dashboard import ScientificDashboard
 window=tk.Tk();window.geometry('560x680+0+0')
 panel=ScientificDashboard(window,palette={},on_review=lambda:None,
                           on_refresh=lambda:None,on_details=lambda:None)
 panel.pack(fill='both',expand=True)
 try:
  window.update()
  positions={(int(w.grid_info()['row']),int(w.grid_info()['column'])) for w in panel._card_frames}
  assert positions=={(0,0),(0,1),(1,0),(1,1)}
  assert all(w.winfo_ismapped() and w.winfo_rootx()>=panel.winfo_rootx()
             and w.winfo_rootx()+w.winfo_width()<=panel.winfo_rootx()+panel.winfo_width()
             for w in panel._card_frames)
 finally:
  window.destroy()
