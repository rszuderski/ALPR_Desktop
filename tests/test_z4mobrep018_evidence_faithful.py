"""Z4MOBREP018: evidence-faithful chart, model state, UTC and diagnostic semantics.

Tests use synthetic rows, do not read/alter GT, sidecars or research archives.
"""
from __future__ import annotations

import ast
from datetime import timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
HELPER = ROOT / 'auto_annotation_tool/gui/mobile_report_measurement_integrity.py'
DASHBOARD = ROOT / 'auto_annotation_tool/gui/mobile_report_science_dashboard.py'
spec = importlib.util.spec_from_file_location('z4mobrep018_pure_helpers', HELPER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_frame_ids_shared_for_missing_mz_without_independent_spacing():
    rows=[
        {'frame_id':'1','total_ms':'1100','plate_inference_ms':'600','character_inference_ms':'500'},
        {'frame_id':'2','total_ms':'750','plate_inference_ms':'650','character_inference_ms':''},
        {'frame_id':'3','total_ms':'1200','plate_inference_ms':'620','character_inference_ms':'580'},
    ]
    m=mod.timeline_stages(rows,('total_ms','plate_inference_ms','character_inference_ms'))
    assert m['x']==(1.0,2.0,3.0) and m['axis_label']=='ID klatki'
    assert m['stages']['character_inference_ms']==(500.0,None,580.0)
    assert mod.continuous_segments(m['stages']['character_inference_ms'])==()
    assert mod.continuous_segments(m['stages']['total_ms'])==((0,1),(1,2))


def test_zero_measured_is_distinct_from_missing_or_invalid():
    vals=['0','0.0','',None,'nan','inf','-1',False]
    result=mod.timeline_stages([{'frame_id':str(i+1),'mt':value} for i,value in enumerate(vals)],('mt',))['stages']['mt']
    assert result==(0.0,0.0,None,None,None,None,None,None)


def test_time_axis_when_frame_ids_missing_and_never_sort_rows():
    rows=[{'timestamp_ms':'1700','total_ms':'700'},
          {'timestamp_ms':'2000','total_ms':''},
          {'timestamp_ms':'3600','total_ms':'1000'}]
    m=mod.timeline_stages(rows,('total_ms',))
    assert m['x']==(0.0,0.3,1.9)
    assert m['axis_label']=='Czas od pierwszej klatki [s]'
    assert m['stages']['total_ms']==(700.0,None,1000.0)


def test_fallback_to_ordinal_is_explicit():
    m=mod.timeline_stages([{'frame_id':'4','timestamp_ms':'9'},
                           {'frame_id':'4','timestamp_ms':'8'}],('total_ms',))
    assert m['x']==(1.0,2.0)
    assert m['axis_label']=='Numer pomiaru (bez ID klatki)'


def test_stage_activation_not_inferred_from_pseudo_fingerprint():
    r=SimpleNamespace(raw={'research_execution_config':{'stages':{
        'mp':{'role':'vehicle','enabled':False}, 'mt':{'enabled':True}}},
       'runtime_composition':{'models':{'vehicle':{'fingerprint':'not_a_used_model'}}}})
    assert mod.stage_enabled(r,'mp') is False
    assert mod.stage_enabled(r,'mt') is True
    assert mod.stage_enabled(r,'mz') is None
    assert mod.stage_enabled(SimpleNamespace(raw={}), 'mp') is None


def test_model_inputs_are_not_camera_dimensions():
    r=SimpleNamespace(raw={'research_execution_config':{'stages':{
      'mp':{'enabled':False},'mt':{'enabled':True,'input':{'width':512,'height':512}},
      'mz':{'enabled':True,'input':{'width':320,'height':320}}}}})
    assert mod.model_input_sizes(r)=='MT 512×512; MZ 320×320'


def test_utc_to_polish_offset_is_explicit_and_source_untouched():
    source='2026-10-17T18:43:48.759Z'
    time=timezone(timedelta(hours=2))
    assert mod.local_measurement_time(source,zone=time)=='17.10.2026 20:43:48 (UTC+02:00)'
    assert mod.local_measurement_time(source,zone=time,compact=True,show_offset=False)=='17.10 20:43'
    assert 'brak strefy' in mod.local_measurement_time('2026-10-17T18:43:48')
    assert source=='2026-10-17T18:43:48.759Z'


def test_historical_crash_counter_is_not_current_session_crash():
    d={'crash_count':1,'crash_semantics':'previous_uncontrolled_measurement_sessions_recovered_on_next_process_start'}
    assert mod.crash_description(d)==('Odzyskane wcześniejsze sesje','1')
    assert mod.crash_description({'crash_count':1}) is None


def _extract_top_level(*names):
    tree=ast.parse(BROWSER.read_text(encoding='utf-8'))
    chosen=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(chosen)==len(names)
    namespace={'json':json, 'Any':object, 'local_measurement_time':mod.local_measurement_time}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(BROWSER),'exec'),namespace)
    return namespace


def _extract_browser_method(name, namespace):
    tree=ast.parse(BROWSER.read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MobileReportBrowser')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(BROWSER),'exec'),namespace)
    return namespace[name]


def test_diagnostic_zero_is_not_missing():
    n=_extract_top_level('_short_text','_flatten_rows','_nested_value')
    rows=n['_flatten_rows']({'pipeline_error_count':0,'runtime_failure_count':0,
       'crash_measurement_available':False},prefix='errors')
    assert ('errors.pipeline_error_count','0','') in rows
    assert ('errors.runtime_failure_count','0','') in rows
    assert ('errors.crash_measurement_available','False','') in rows


def test_gui_diagnostics_labels_previous_crash_and_keeps_zeros():
    n=_extract_top_level('_short_text','_flatten_rows','_nested_value')
    n.update({'crash_description':mod.crash_description})
    render=_extract_browser_method('_populate_diagnostics',n)
    collected=[]
    o=SimpleNamespace(diagnostics_table=SimpleNamespace(set_rows=lambda rows:collected.extend(rows)),
                      log_text=None, _replace_text=lambda *_:None)
    report=SimpleNamespace(errors={'crash_count':1,
        'crash_semantics':'previous_uncontrolled_measurement_sessions_recovered_on_next_process_start',
        'pipeline_error_count':0,'runtime_failure_count':0},raw={},memory={})
    render(o,report,None)
    assert any(x[0]=='Odzyskane wcześniejsze sesje' and x[1]=='1' for x in collected)
    assert any(x[0]=='errors.pipeline_error_count' and x[1]=='0' for x in collected)
    assert not any(x[0]=='errors.crash_count' for x in collected)


def test_disabled_mp_is_not_fake_fingerprint():
    n={'stage_enabled':mod.stage_enabled, '_merge_model_fingerprint_aliases': lambda *x:None}
    render=_extract_browser_method('_guard_model_fingerprint_value',n)
    r=SimpleNamespace(raw={'research_execution_config':{'stages':{'mp':{'enabled':False}}}})
    assert render(SimpleNamespace(),r,{},'mp')=='Wyłączony'
    assert 'MP fingerprint' not in BROWSER.read_text('utf8')


class FakeCanvas:
    def __init__(self):
        self.ovals=[]; self.rectangles=[]; self.polygons=[]; self.lines=[]; self.texts=[]
    def winfo_width(self):return 950
    def winfo_height(self):return 180
    def delete(self,*args):
        for key in ('ovals','rectangles','polygons','lines','texts'):
            getattr(self,key).clear()
    def create_rectangle(self,*args,**kwargs):self.rectangles.append((args,kwargs))
    def create_polygon(self,*args,**kwargs):self.polygons.append((args,kwargs))
    def create_text(self,*args,**kwargs):self.texts.append((args,kwargs))
    def create_line(self,*args,**kwargs):self.lines.append((args,kwargs))
    def create_oval(self,*args,**kwargs):self.ovals.append((args,kwargs))


def test_actual_chart_does_not_connect_across_missing_mz():
    # The chart method is extracted in isolation: provide its pure axis helpers
    # explicitly, just as the production module imports them.
    chart_helper = ROOT / 'auto_annotation_tool/gui/mobile_report_plot_axis.py'
    chart_spec = importlib.util.spec_from_file_location('z4mobrep_plot_axis', chart_helper)
    chart_mod = importlib.util.module_from_spec(chart_spec)
    chart_spec.loader.exec_module(chart_mod)
    style_path = ROOT / 'auto_annotation_tool/gui/mobile_report_chart_styles.py'
    style_spec = importlib.util.spec_from_file_location('z4mobrep_chart_styles', style_path)
    style_mod = importlib.util.module_from_spec(style_spec)
    style_spec.loader.exec_module(style_mod)
    n={'timeline_stages':mod.timeline_stages,'continuous_segments':mod.continuous_segments,
       'axis_ticks':chart_mod.axis_ticks,'pixel_column_extrema':chart_mod.pixel_column_extrema,
       'stage_enabled':mod.stage_enabled,'tk':tk,'blend_hex_colors':lambda a,b,t:'#777777',
       'LATENCY_SERIES_STYLES':style_mod.LATENCY_SERIES_STYLES,
       'draw_chart_marker':style_mod.draw_chart_marker,
       'logger':SimpleNamespace(exception=lambda *_:None)}
    render=_extract_browser_method('_draw_latency_chart',n)
    rows=[{'frame_id':'1','total_ms':'1200','plate_inference_ms':'100','character_inference_ms':'300'},
          {'frame_id':'2','total_ms':'1300','plate_inference_ms':'200','character_inference_ms':''},
          {'frame_id':'3','total_ms':'1400','plate_inference_ms':'300','character_inference_ms':'500'}]
    canvas=FakeCanvas()
    self=SimpleNamespace(latency_canvas=canvas,current_bundle=SimpleNamespace(trace_rows=rows),
         panel='#ffffff', border='#cccccc', muted='#333333',fg='#000000',accent='#111111',
         warning='#222222',success='#333333')
    render(self)
    # All series share the *source-frame* X positions. Colors and marker
    # shapes are independent of the window's accent/warning/success palette.
    colors = style_mod.LATENCY_SERIES_STYLES
    pipeline_color=colors['total_ms'].color
    mt_color=colors['plate_inference_ms'].color
    mz_color=colors['character_inference_ms'].color
    # Markers in the legend are at Y=11. Ignore them when comparing points.
    pipeline_x=[a[0] for a,k in canvas.polygons
                if k.get('fill')==pipeline_color and a[1]>=25]
    mt_x=[(a[0]+a[2])/2 for a,k in canvas.ovals
          if k.get('fill')==mt_color and a[1]>=25]
    mz_x=[(a[0]+a[2])/2 for a,k in canvas.rectangles
          if k.get('fill')==mz_color and a[1]>=25]
    assert len(pipeline_x)==len(mt_x)==3
    assert pipeline_x==mt_x
    assert mz_x==[pipeline_x[0],pipeline_x[2]]
    # No gap interpolation: the only MZ line is the legend key at Y=11.
    assert not any(k.get('fill')==mz_color and a[1]>20 for a,k in canvas.lines)
    assert not any(k.get('smooth') for a,k in canvas.lines)
    assert any('ID klatki' in str(k.get('text','')) for a,k in canvas.texts)


def test_semantics_are_only_in_view_and_export_source_is_unchanged():
    s=BROWSER.read_text('utf8')
    assert 'Z4MOBREP018_EVIDENCE_FAITHFUL_DISPLAY_20261018' in s
    assert 'Wspólna seria' in s
    assert 'not a cryptographic model ID' in s
    assert 'Data (lok.)' in s
    assert 'smooth=False' in s
    assert 'stage_enabled(report, role_key)' in s
    d=DASHBOARD.read_text('utf8')
    assert 'P95 = oszacowany 95. percentyl' in d


def test_nonconsecutive_source_frame_ids_never_connected_as_unbroken_measurement():
    rows=[{'frame_id':'1','total_ms':'1100','character_inference_ms':'200'},
          {'frame_id':'3','total_ms':'1250','character_inference_ms':'250'},
          {'frame_id':'4','total_ms':'950','character_inference_ms':''},
          {'frame_id':'6','total_ms':'1100','character_inference_ms':'220'}]
    t=mod.timeline_stages(rows,('total_ms','character_inference_ms'))
    assert t['x']==(1.,3.,4.,6.)
    assert mod.continuous_segments(t['stages']['total_ms'],t['x'],require_consecutive=True)==((1,2),)
    assert mod.continuous_segments(t['stages']['character_inference_ms'],t['x'],require_consecutive=True)==()


def test_model_description_without_identity_not_promoted_to_fingerprint():
    n={'stage_enabled':mod.stage_enabled, '_nested_value':_extract_top_level('_nested_value')['_nested_value'], '_merge_model_fingerprint_aliases':lambda out,value:out.update({'mt':value['plate']} if isinstance(value,dict) and 'plate' in value else {})}
    render=_extract_browser_method('_guard_model_fingerprint_value',n)
    r=SimpleNamespace(raw={'models':{'plate':{'enabled':True,'role':'plate'}}})
    assert render(SimpleNamespace(),r,{},'mt')=='-'
    real=SimpleNamespace(raw={'models':{'plate':{'enabled':True,'fingerprint':'real012345abcdef'}}})
    assert render(SimpleNamespace(),real,{},'mt')=='real012345abcdef'


def test_same_series_with_missing_model_evidence_not_marked_scientifically_comparable():
    n=_extract_top_level('_nested_value')
    n['_device_label']=lambda report:'Ten sam telefon'
    n['_safe_int']=lambda value,default=0:int(value) if value not in (None,'') else default
    func=_extract_browser_method('_populate_comparison_guard',n)
    reports=[SimpleNamespace(report_id='r1',identity=('r1',),experiment_index={'series_id':'F4', 'scenario_id':'scene'},
        runtime='tflite', delegate='cpu',raw={},quality={}),
             SimpleNamespace(report_id='r2',identity=('r2',),experiment_index={'series_id':'F4','scenario_id':'scene'},
        runtime='tflite',delegate='cpu',raw={},quality={})]
    # Environment and most values equal; but report 2 has no MT identity.
    values={'r1':{'mt':'hash-r1','mp':'Wyłączony'},'r2':{'mt':'-','mp':'-'}}
    collected=[]
    host=SimpleNamespace(
        store=SimpleNamespace(reports=reports),
        _report_experiment_index=lambda report,bundle=None:report.experiment_index,
        _guard_android_version_value=lambda report,index:'12',
        _guard_resolution_value=lambda report,index:'MT 512×512; MZ 320×320',
        _guard_app_git_sha_value=lambda report,index:'abcdef',
        _guard_app_version_value=lambda report,index:'1.0',
        _guard_profile_hash=lambda value:'balanced',
        _guard_model_fingerprint_value=lambda report,index,role:values[report.report_id].get(role,'mz-fp'),
        _guard_artifact_count=lambda index,key:8,
        comparison_table=SimpleNamespace(set_rows=lambda rows:collected.extend(rows)),
    )
    func(host,reports[0],None)
    scope=next(row for row in collected if row[0]=='Zakres porównania')
    mt=next(row for row in collected if row[0]=='MT fingerprint')
    mp=next(row for row in collected if row[0]=='Model pojazdów (MP)')
    assert scope[3]=='Wspólna seria'
    assert mt[3]=='Brak danych' and '1/2' in mt[2]
    assert mp[3]=='Brak danych', 'Unknown MP use in one source cannot be treated as disabled in both.'


def test_chart_ignores_unused_mp_even_when_legacy_trace_has_zero_placeholder():
    style_path = ROOT / 'auto_annotation_tool/gui/mobile_report_chart_styles.py'
    style_spec=importlib.util.spec_from_file_location('z4mobrep_mp_chart_styles',style_path)
    style_mod=importlib.util.module_from_spec(style_spec)
    style_spec.loader.exec_module(style_mod)
    n={'timeline_stages':mod.timeline_stages,'continuous_segments':mod.continuous_segments,
       'stage_enabled':mod.stage_enabled,'tk':tk,
       'LATENCY_SERIES_STYLES':style_mod.LATENCY_SERIES_STYLES,
       'draw_chart_marker':style_mod.draw_chart_marker,
       'axis_ticks':lambda xs,width,label:((min(xs),str(min(xs))),(max(xs),str(max(xs)))),
       'pixel_column_extrema':lambda *_:(),
       'blend_hex_colors':lambda a,b,t:'#777777',
       'logger':SimpleNamespace(exception=lambda *_:None)}
    render=_extract_browser_method('_draw_latency_chart',n)
    raw={'research_execution_config':{'stages':{'mp':{'enabled':False}}}}
    bundle=SimpleNamespace(trace_rows=[{'frame_id':'1','total_ms':'300','vehicle_inference_ms':'0.0'},
                                  {'frame_id':'2','total_ms':'350','vehicle_inference_ms':'0.0'}])
    canvas=FakeCanvas()
    host=SimpleNamespace(latency_canvas=canvas,current_bundle=bundle,current_report=SimpleNamespace(raw=raw),
       panel='#fff',border='#ddd',muted='#333',fg='#111',accent='#222',warning='#444',success='#666')
    render(host)
    assert all(k.get('fill')!=style_mod.LATENCY_SERIES_STYLES['vehicle_inference_ms'].color for a,k in (canvas.ovals+canvas.polygons+canvas.rectangles))
    assert not any('MP' in str(k.get('text','')) for a,k in canvas.texts)
