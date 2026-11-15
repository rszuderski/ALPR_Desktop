"""Z4MOBREP016 — readable overview without invented scientific denominators."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui.mobile_report_science_dashboard import build_overview
from auto_annotation_tool.ranking.mobile_package_experiments import MobileBenchmarkReport

SHA='5'*64


def report(*, latency=None, raw=None):
    return MobileBenchmarkReport(report_id='r016',package_id='p',variant_id='fp32',
          measured_at='2026-10-17T20:00:00Z',source_archive_sha256=SHA,
          latency=latency or {},raw=raw or {})


def bundle(r=None, *, sha=SHA, valid=True, payload=None):
    return SimpleNamespace(report=r or report(),validation=SimpleNamespace(ok=valid),
       source_archive_sha256=sha,report_payload=payload or {},trace_rows=(),trace_total=0,
       frame_flow_rows=(),frame_flow_total=0)


def science(**changes):
    base={
        'source_archive_sha256':SHA,'scientific_state':'PRELIMINARY','review_mode':'assisted',
        'progress':{'subjects_total':8,'subjects_mt_done':1,'mt_executed_active':7,
                    'mt_invocations_cardinality_reviewed':1},
        'mt':{'single_plate_n':1,'localized_n':0,'no_detection_n':1},
        'pose':{'measured_n':0,'gt_missed_plate_n':1,'polygon_iou_mean_pct':None,'nmce_mean_pct':None},
        'mz':{'evaluable_reads':0,'exact_n':0,'cer':None},
        'alpr':{'evaluable_subjects':0,'exact_n':0}}
    base.update(changes)
    return base


def test_actual_case_mt_missing_pose_gt_not_fake_iou():
    r=report(latency={'pipeline':{'count':8,'median_ms':2396.05,'p95_ms':3654.64}})
    d=build_overview(r,bundle(r),science())
    assert d['quality_verified']
    assert d['cards'][0]['value']=='0 z 1'
    assert 'Wynik opisowy' in d['cards'][0]['note']
    assert d['cards'][1]['value']=='Brak oceny' and 'pominiętych' in d['cards'][1]['note']
    assert d['cards'][2]['value']=='Brak oceny'
    assert d['cards'][3]['value']=='Brak oceny'
    assert d['progress_ratio']==1/7
    assert '1 z 7' in d['progress']
    assert 'Ocena z widocznymi predykcjami' in ' '.join(d['warnings'])
    assert d['performance']['pipeline']['median']=='2,40 s'


def test_no_review_or_source_is_not_zero_quality():
    r=report()
    d=build_overview(r)
    assert d['status']=='Oryginalne archiwum niepodłączone'
    assert not d['quality_verified'] and not d['source_verified']
    assert all(row['value']=='Brak oceny' for row in d['cards'])
    assert d['progress_ratio'] is None


def test_wrong_sha_sidecar_ignored_but_verified_archive_visible():
    r=report()
    b=bundle(r)
    s=science(source_archive_sha256='6'*64,
              mt={'single_plate_n':12,'localized_n':12})
    d=build_overview(r,b,s)
    assert d['source_verified'] and not d['quality_verified']
    assert d['cards'][0]['value']=='Brak oceny'


def test_wrong_archive_sha_rejects_both_source_and_review():
    r=report()
    d=build_overview(r,bundle(r,sha='a'*64),science())
    assert not d['source_verified'] and not d['quality_verified']


def test_report_id_mismatch_discards_review():
    r=report()
    b=bundle(replace(r,report_id='another'))
    assert not build_overview(r,b,science())['quality_verified']


def test_valid_zero_is_shown_only_for_nonzero_denominator():
    r=report()
    s=science(mt={'single_plate_n':36,'localized_n':0},
              mz={'evaluable_reads':40,'exact_n':0,'cer':1.0},
              alpr={'evaluable_subjects':12,'exact_n':0})
    rows=build_overview(r,bundle(r),s)['cards']
    assert rows[0]['value']=='0 z 36'
    assert rows[2]['value']=='0 z 40' and '100,0%' in rows[2]['note']
    assert rows[3]['value']=='0 z 12'


def test_measured_pose_iou_and_nmce_independent_of_missed_gt():
    r=report()
    p={'measured_n':3,'gt_missed_plate_n':1,'polygon_iou_mean_pct':79.2,'nmce_mean_pct':3.3}
    d=build_overview(r,bundle(r),science(pose=p))
    assert d['cards'][1]['value']=='IoU 79,2%'
    assert 'NMCE' in d['cards'][1]['note'] and '3,3%' in d['cards'][1]['note']
    assert '3 par GT' in d['cards'][1]['note']


def test_missing_counts_not_promoted_to_accuracy():
    r=report()
    s=science(mt={'single_plate_n':9,'localized_n':None},mz={'evaluable_reads':5,'exact_n':None},
              alpr={'evaluable_subjects':3,'exact_n':None})
    assert all(row['value']=='Brak oceny' for row in (build_overview(r,bundle(r),s)['cards'][i] for i in (0,2,3)))


def test_completed_stage_not_confused_with_completed_full_session():
    r=report()
    s=science(scientific_state='MT_STAGE_COMPLETE_REVIEW_OPEN')
    assert 'sesja nadal otwarta' in build_overview(r,bundle(r),s)['status']
    assert 'zakończona' in build_overview(r,bundle(r),science(scientific_state='FINAL_FULL_REVIEW'))['status']


def test_native_time_not_confused_with_frame_gate_or_gt():
    r=report(latency={'pipeline':{'count':8,'median_ms':2396.05,'p95_ms':3654.64}},
             raw={'summary':{'frame_flow':{'frames_received':15,'frames_processed':8,'frames_skipped_frame_gate':7}}})
    d=build_overview(r,bundle(r),None)
    assert d['performance']['pipeline']['n']==8
    assert d['frames']=='Przetworzono 8 z 15 odebranych klatek · 7 pominięto przed analizą'
    assert 'P95 opiera się na 8 pomiarach' in ' '.join(d['warnings'])
    assert all(row['value']=='Brak oceny' for row in d['cards'])


def test_no_guess_on_negative_or_nan_latency():
    r=report(latency={'pipeline':{'count':8,'median_ms':float('nan'),'p95_ms':-1}})
    x=build_overview(r,bundle(r))['performance']['pipeline']
    assert x['median']=='—' and x['p95']=='—'


def test_all_technical_table_objects_still_in_browser_source():
    p=Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_report_browser.py'
    text=p.read_text(encoding='utf-8')
    for field in ('summary_table','comparison_table','config_table','latency_table','artifacts_table',
                  'quality_table','diagnostics_table','crops_table','raw_text','_draw_latency_chart',
                  '_on_crop_selected','read_mobile_sample_image','restore_source'):
        assert field in text
    assert 'self.technical_notebook = ttk.Notebook(details)' in text
    assert "name='alpr-scientific-overview'" in text
    assert 'def refresh_scientific_dashboard' in text
    assert 'self._science_sequence' in text


def test_only_four_operator_facing_tabs():
    text=(Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_report_browser.py').read_text('utf-8')
    assert tuple(label for label in ('Wyniki','Porównanie','Próbki','Szczegóły') if f'text="{label}"' in text)==('Wyniki','Porównanie','Próbki','Szczegóły')
    assert text.count('self.notebook.add(')==4
    assert text.count('self.technical_notebook.add(')==7
