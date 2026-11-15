"""Z4MOBREP019: prevent unsupported scientific comparison of Android sessions.

Synthetic fixtures only; no read or mutation of archived measurements or GT.
"""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'auto_annotation_tool/gui/mobile_report_scientific_comparability.py'
BROWSER=ROOT/'auto_annotation_tool/gui/z4_mobile_report_browser.py'
spec=importlib.util.spec_from_file_location('alpr_comparability019',HELPER)
module=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=module
spec.loader.exec_module(module)


def report(session='a',variant='B',series='F4',scenario='static',*,charging=False,thermal=1,
           received=15,processed=8,gate=7,apk='a'*64,mt='model-mt',mz='model-mz',
           device='SM-A125F',android='12',app_id=None,complete=True,independent=None):
    raw={
      'device':{'manufacturer':'samsung','model':device,'android_version':android,
                'charging':charging,'thermal_status':thermal},
      'capture':{'actual_width':480,'actual_height':640},
      'app_build':{'apk_identity':{'base_apk_sha256':apk}},
      'execution':{'vehicle':{'enabled':False}},
      'research_execution_config':{
        'analysis_mode':'static','scene_handling_mode':'strict_scene_boundary',
        'roi_budget_policy':'r0_full_frame','recognition_profile':'balanced',
        'feature_flags':{'adaptive_frame_gate':True},
        'stages':{
          'mp':{'enabled':False},
          'mt':{'enabled':True,'model_fingerprint':mt,'runtime':'tflite',
                'precision':'fp32' if variant=='B' else 'int8','delegate':'cpu',
                'cpu_threads':2,'input':{'width':512,'height':512}},
          'mz':{'enabled':True,'model_fingerprint':mz,'runtime':'tflite',
                'precision':'fp32' if variant=='B' else 'int8','delegate':'cpu',
                'cpu_threads':2,'input':{'width':320,'height':320}},
        },
      },
      'thermal':{'start':{'thermal_status':thermal,'charging':charging,
                          'battery_temperature_c':26.4 if thermal==0 else 29.5},
                 'end':{'battery_temperature_c':28.1 if thermal==0 else 30.6}},
      'frame_flow':{'frames_received':received,'frames_processed':processed,
                    'frames_skipped_frame_gate':gate},
      'summary':{'processed_frames':processed},'measured_runs':processed,
      'quality':{'ground_truth_samples':0,'reviewed_samples':0},
    }
    index={'session_id':'exp-'+session,'series_id':series,'scenario_id':scenario,
           'experiment_variant':variant, 'replicate_index':1 if session=='a' else 2}
    if independent is not None:index['independent_variable']=independent
    return SimpleNamespace(report_id='r-'+session,source_archive_sha256=str(session+'1'*64)[:64]
          if False else ('1'*64 if session=='a' else '2'*64),
          raw=raw,experiment_index=index,device=raw['device'],
          runtime='tflite',delegate='cpu',variant_id=variant,quality=raw['quality'])


def row(assessment,name):
    return next(item for item in assessment['rows'] if item[0]==name)


def test_same_scenario_and_device_do_not_certify_when_power_and_thermal_differ():
    a=report('a',charging=True,thermal=0,received=240,processed=7,gate=233)
    b=report('b',charging=False,thermal=1,received=15,processed=8,gate=7)
    result=module.assess_cohort(b,[a,b])
    assert result['cohort_size']==2
    assert result['status']=='Różne warunki — brak zgody na porównanie'
    assert set(result['issues']) >= {'Ładowanie telefonu','Stan termiczny na początku'}
    assert row(result,'Aplikacja (SHA APK)')[3]=='Zgodne (wstępnie)'
    assert row(result,'Model tablic (MT)')[3]=='Zgodne (wstępnie)'
    assert row(result,'Odebrane klatki')[3]=='Różne wartości'
    assert row(result,'Wniosek')[3]=='NIE ZATWIERDZONO'


def test_missing_evidence_never_default_matches():
    a=report('a');b=report('b');del a.raw['app_build'];del a.raw['capture']
    out=module.assess_cohort(b,[a,b])
    assert out['status']=='Niepełne dane — porównanie niepotwierdzone'
    assert 'Aplikacja (SHA APK)' in out['unknown']
    assert 'Rozmiar obrazu kamery' in out['unknown']
    assert row(out,'Aplikacja (SHA APK)')[3]=='Brak danych'


def test_declared_independent_model_change_never_auto_approves_study():
    a=report('a',variant='FP32',mt='FP32-sha',independent='precision')
    b=report('b',variant='INT8',mt='INT8-sha',independent='precision')
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Model tablic (MT)')[3]=='Planowana zmiana?'
    assert row(out,'Co celowo zmieniono w teście?')[3]=='Plan podany'
    assert out['status']!='Można porównywać'
    assert row(out,'Wniosek')[3]=='NIE ZATWIERDZONO'


def test_unknown_planned_factor_requires_design_when_variant_changes():
    a=report('a',variant='FP32',mt='FP32-sha')
    b=report('b',variant='INT8',mt='INT8-sha')
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Co celowo zmieniono w teście?')[3]=='WYMAGA PLANU'
    assert 'Brak deklaracji czynnika badanego' in out['notes']


def test_same_variant_different_model_is_uncontrolled_difference():
    a=report('a',mt='first')
    b=report('b',mt='second')
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Model tablic (MT)')[3]=='Różne modele'
    assert 'Model tablic (MT)' in out['issues']


def test_same_series_other_scenario_never_enter_comparison():
    a=report('a',scenario='day')
    b=report('b',scenario='night')
    out=module.assess_cohort(a,[a,b])
    assert out['cohort_size']==1 and out['status']=='Brak pary do porównania'
    assert row(out,'Wniosek')[3]=='WSTĘPNE'


def test_no_scenario_never_groups_unrelated_reports():
    a=report('a',scenario=None); b=report('b',scenario=None)
    out=module.assess_cohort(a,[a,b]); assert out['cohort_size']==1


def test_disabled_mp_never_becomes_fake_fingerprint():
    a=report('a');b=report('b')
    a.raw['runtime_composition']={'models':{'vehicle':{'fingerprint':'fake-fp'}}}
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Model pojazdów (MP)')[1]=='Wyłączony'
    assert row(out,'Model pojazdów (MP)')[3]=='Zgodne (wstępnie)'


def test_zero_gt_remains_observed_zero_not_no_source():
    a=report('a');b=report('b')
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Dostępność GT znaków w Androidzie')[1]=='0'
    assert 'GT w archiwum' in row(out,'Ocena jakości MT/MZ')[2]
    assert row(out,'Ocena jakości MT/MZ')[3]=='NIE POTWIERDZONO'


def test_stored_preview_telemetry_can_be_compared_without_rescanning_sources():
    a=report('a'); b=report('b',charging=True,thermal=0)
    a.raw['thermal']={'_kind':'object','_preview':a.raw['thermal']}
    a.raw['frame_flow']={'_kind':'object','_preview':a.raw['frame_flow']}
    out=module.assess_cohort(a,[a,b])
    assert row(out,'Ładowanie telefonu')[3]=='Różne warunki'
    assert row(out,'Odebrane klatki')[1]=='15'


def test_no_manifest_mutation_or_external_disk_access():
    a=report('a'); b=report('b')
    import copy
    before=(copy.deepcopy(a.raw),copy.deepcopy(b.raw))
    module.assess_cohort(a,[a,b])
    assert (a.raw,b.raw)==before


def test_browser_displays_new_scientific_assessment_instead_of_legacy_table():
    src=BROWSER.read_text('utf-8')
    assert 'Z4MOBREP019_NO_UNGROUNDED_SCIENTIFIC_COMPARABILITY_20261018' in src
    t=ast.parse(src)
    browser=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='MobileReportBrowser')
    show=next(n for n in browser.body if isinstance(n,ast.FunctionDef) and n.name=='_show_report')
    called=[ast.unparse(node.func) for node in ast.walk(show) if isinstance(node,ast.Call)]
    assert 'self._populate_scientific_comparison' in called
    assert 'self._populate_comparison_guard' not in called
    method=next(n for n in browser.body if isinstance(n,ast.FunctionDef) and n.name=='_populate_scientific_comparison')
    ns={'assess_cohort':module.assess_cohort,'logger':SimpleNamespace(exception=lambda *_:None),'MobileBenchmarkReport':AnyFake,'MobileReportBundle':AnyFake}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(BROWSER),'exec'),ns)
    collector=[];a=report('a',charging=True,thermal=0);b=report('b')
    host=SimpleNamespace(store=SimpleNamespace(reports=[a,b]),comparison_table=SimpleNamespace(set_rows=lambda rows:collector.extend(rows)))
    ns['_populate_scientific_comparison'](host,b,None)
    assert row({'rows':collector},'Ładowanie telefonu')[3]=='Różne warunki'


class AnyFake:pass


def test_repeated_archive_is_not_new_replicate_even_when_report_ids_differ():
    a=report('a'); b=report('b')
    b.source_archive_sha256=a.source_archive_sha256
    out=module.assess_cohort(a,[a,b])
    assert 'Powtórzone archiwum' in out['issues']
    assert row(out,'Niezależność źródeł')[3]=='NIEZALEŻNOŚĆ NIEPOTWIERDZONA'


def test_repeated_session_id_is_not_independent_replicate():
    a=report('a'); b=report('b')
    b.experiment_index['session_id']=a.experiment_index['session_id']
    out=module.assess_cohort(a,[a,b])
    assert 'Powtórzony identyfikator sesji' in out['issues']


def test_selected_verified_archive_cannot_carry_verification_to_peer():
    a=report('a'); b=report('b')
    bundle=SimpleNamespace(validation=SimpleNamespace(ok=True),
        source_archive_sha256=a.source_archive_sha256,
        report=SimpleNamespace(report_id=a.report_id),report_payload=a.raw,
        experiment_session=a.experiment_index)
    out=module.assess_cohort(a,[a,b],bundle=bundle)
    assert row(out,'Integralność źródeł')[1]=='Wybrane archiwum sprawdzone'
    assert row(out,'Integralność źródeł')[3]=='BRAK PEŁNEGO POTWIERDZENIA'
    assert row(out,'Identyczność materiału scen')[3]=='DO POTWIERDZENIA'
