"""P0: approved review, measured timing, and full quality are distinct contracts."""
from pathlib import Path
from dataclasses import replace
import csv,hashlib,json,shutil
import pytest

OUT=Path(__file__).resolve().parents[1]/'output/Z4_PERFORMANCE_VS_QUALITY_20261030_001'
R02='exp-1792699303016-c8b05efb'
R01='exp-1792699088870-cc33764a'
R03='exp-1792699516035-d9e450f6'

@pytest.fixture
def actual(monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    from auto_annotation_tool.config import CONFIG
    from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
    from auto_annotation_tool.gui.mobile_import_comparison import record_from_import
    catalog=OUT/'private/catalog';root=OUT/'private/series/03_wyniki'
    monkeypatch.setattr(CONFIG,'DIR_7_RANKINGS_MOBILE_PACKAGES',catalog)
    inventory=json.loads((OUT/'source_inventory.json').read_text(encoding='utf-8'))
    result={}
    for sid in (R01,R02,R03):
        entry=next(r for r in inventory if r['session_id']==sid)
        bundle=read_mobile_report_bundle(entry['private_source'])
        record,traces,warnings=record_from_import(bundle.report,bundle,root)
        result[sid]=(bundle,record,traces)
    return result,root,catalog

def test_P19_performance_never_renders_assisted_quality_even_with_ready_member(actual):
    from auto_annotation_tool.gui.mobile_comparison_core import figures
    import matplotlib.pyplot as plt
    data,_,_=actual
    plots=list(figures([data[R01][1],data[R02][1]],comparison_scope='performance'))
    try:
        assert len(plots)==6
        assert all('assisted' not in name for name,_ in plots)
        assert all(fig.axes[0].get_xlabel()=='Czas [ms]' or fig.axes[0].get_ylabel()=='Czas [ms]' for _,fig in plots)
    finally:
        for _,fig in plots:plt.close(fig)

def test_P18_r02_counts_and_percentiles_are_from_android(actual):
    data,_,_=actual;s=data[R02][1]
    assert not s.canvas_ready and s.operator_approved
    assert {k:v.n for k,v in s.latency.items()}==dict(mt=21,mz=20,pipeline=21)
    assert s.latency['mt'].p50_ms==pytest.approx(470.958231)
    assert s.latency['mz'].p95_ms==pytest.approx(1244.08064965)
    assert s.latency['pipeline'].p95_ms==pytest.approx(2775.115539)

@pytest.fixture
def scoped(actual):
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import
    data,root,catalog=actual
    return {sid:assess_import(v[0].report,v[0].path,root,catalog) for sid,v in data.items()}

def clone(actual,tmp_path,sid=R02):
    data,root,catalog=actual;bundle,record,_=data[sid]
    source=tmp_path/'02_sesje'/Path(bundle.path).name;source.parent.mkdir(parents=True)
    shutil.copyfile(bundle.path,source)
    results=tmp_path/'03_wyniki';dest=results/sid/record.directory.name
    shutil.copytree(record.directory,dest);(results/'porownanie').mkdir()
    reviews=tmp_path/'catalog';(reviews/'human_reviews').mkdir(parents=True)
    side=reviews/'human_reviews'/(record.source_sha256+'.review.json')
    shutil.copyfile(catalog/'human_reviews'/side.name,side)
    return bundle.report,source,results,reviews,dest,side

def assess_copy(values):
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import
    return assess_import(*values[:4])

def test_P01_READY_qualified_for_both_scopes(scoped):
    a=scoped[R01]
    assert a.state=='READY' and a.allowed('performance') and a.allowed('quality')

def test_P02_limited_available_only_for_performance(scoped):
    a=scoped[R02]
    assert a.state=='APPROVED_LIMITED' and a.allowed('performance') and not a.allowed('quality')
    assert 'Obraz 07' in a.description and 'Obraz 08' in a.description

def test_P03_missing_quality_does_not_create_zero(scoped):
    a=scoped[R02];s=a.record
    assert s.canvas_exact_pct is None and s.canvas_cer_pct is None
    assert a.performance_eligible and all(v.n for v in s.latency.values())

@pytest.mark.parametrize('state',['IN_PROGRESS','INCOMPLETE','NONE','NEEDS_EXPORT'])
def test_P04_P05_unapproved_states_block_both(actual,tmp_path,state):
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import
    from test_z4_akt89f_results_identity import reseal
    args=clone(actual,tmp_path);report,source,root,catalog,export,side=args
    if state=='INCOMPLETE':
        for p in (export/'review.json',export/'export_manifest.json',next(export.glob('ocena_naukowa_*.json'))):
            d=json.loads(p.read_text(encoding='utf-8'));d['review_status']='IN_PROGRESS';p.write_text(json.dumps(d),encoding='utf-8')
        reseal(export);catalog=tmp_path/'no_live_review'
    else:
        root=tmp_path/'empty_results';root.mkdir()
        if state=='NONE':catalog=tmp_path/'no_live_review'
        else:
            d=json.loads(side.read_text(encoding='utf-8'));d['review_status']='IN_PROGRESS' if state=='IN_PROGRESS' else 'COMPLETED'
            side.write_text(json.dumps(d),encoding='utf-8')
    a=assess_import(report,source,root,catalog)
    assert a.state==state and not a.performance_eligible and not a.quality_ready

def test_P05_new_revision_revokes_eligibility(actual,tmp_path):
    args=clone(actual,tmp_path);side=args[-1];d=json.loads(side.read_text(encoding='utf-8'));d['review_revision']+=1
    side.write_text(json.dumps(d),encoding='utf-8');a=assess_copy(args)
    assert a.state=='STALE' and not a.allowed('performance') and not a.allowed('quality')

@pytest.mark.parametrize('fault',['report_id','source_sha','manifest_sha','unreadable'])
def test_P06_identity_and_read_failures_are_closed(actual,tmp_path,monkeypatch,fault):
    args=list(clone(actual,tmp_path))
    if fault=='report_id':args[0]=replace(args[0],report_id='wrong-report')
    elif fault=='source_sha':args[1].write_bytes(args[1].read_bytes()+b'changed')
    elif fault=='manifest_sha':(args[4]/'review.json').write_text('{}',encoding='utf-8')
    else:
        original=Path.open;blocked=args[4]/'export_manifest.json'
        def denied(p,*a,**k):
            if p==blocked:raise PermissionError(13,'Simulated WinError 5',str(p))
            return original(p,*a,**k)
        monkeypatch.setattr(Path,'open',denied)
    a=assess_copy(args)
    assert not a.allowed('performance') and not a.allowed('quality') and a.performance_reason

def test_P07_incomplete_trace_stream_is_not_zero(actual,monkeypatch):
    from auto_annotation_tool.gui import mobile_import_comparison as module
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import
    monkeypatch.setattr(module,'_read_all_traces',lambda b:([],'INCOMPLETE_TRACE_SOURCE: injected read failure'))
    data,root,catalog=actual;b=data[R02][0];a=assess_import(b.report,b.path,root,catalog)
    assert not a.performance_eligible
    assert all(v.n==0 and v.p50_ms is None and v.p95_ms is None for v in a.record.latency.values())

def test_P07_P28_unobserved_MZ_is_NA_without_blocking_measured_MT(actual,monkeypatch):
    from auto_annotation_tool.gui import mobile_import_comparison as module
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import,comparison_warnings
    original=module._read_all_traces
    # Inject an unavailable stage into the real trace structure; never invent a timing.
    def no_mz(bundle):
        rows,state=original(bundle)
        return [dict(row,character_inference_ms=None,mz_runs=0) for row in rows],state
    monkeypatch.setattr(module,'_read_all_traces',no_mz)
    data,root,catalog=actual;b=data[R02][0];a=assess_import(b.report,b.path,root,catalog)
    assert a.performance_eligible and a.record.latency['mt'].n==21
    mz=a.record.latency['mz'];assert mz.n==0 and mz.p50_ms is None and mz.p95_ms is None
    assert 'MZ N/A' in '\n'.join(comparison_warnings([a.record],'performance'))

def test_P08_newer_foreign_export_does_not_fall_back_to_older_READY(actual,tmp_path):
    args=clone(actual,tmp_path,R01)
    shutil.copytree(actual[0][R02][1].directory,args[2]/R01/'review_r999999_export_001')
    a=assess_copy(args)
    assert a.state=='INVALID' and not a.allowed('quality') and not a.allowed('performance')

def test_P08_session_identity_conflict_blocks_source(actual):
    from auto_annotation_tool.gui.mobile_comparison_scope import assess_import
    data,root,catalog=actual;b=data[R02][0]
    report=replace(b.report,experiment_index={'experiment_session_id':'another-session'})
    a=assess_import(report,b.path,root,catalog)
    assert not a.performance_eligible and not a.quality_ready

def test_P13_duplicate_source_and_single_comparison_blocked(scoped):
    from auto_annotation_tool.gui.mobile_comparison_scope import validate_comparison
    a=scoped[R02];s=a.record
    for rows in ([s],[s,s],[s,replace(s,session_id='duplicate-source')]):
        with pytest.raises(ValueError):validate_comparison(rows,comparison_scope='performance',require_sources=True)

def test_P17_performance_chart_values_match_original_traces(scoped):
    from auto_annotation_tool.gui.mobile_comparison_core import figures
    import matplotlib.pyplot as plt
    sessions=[scoped[R01].record,scoped[R02].record]
    plots=list(figures(sessions,comparison_scope='performance'))
    try:
        for name,fig in plots:
            stage,pct=next((s,p) for s in ('mt','mz','pipeline') for p in ('p50_ms','p95_ms') if name.endswith('_'+s+'_'+p))
            assert [p.get_width() for p in fig.axes[0].patches]==pytest.approx([getattr(r.latency[stage],pct) for r in sessions])
            assert all(f'n={r.latency[stage].n}' in ' '.join(t.get_text() for t in fig.axes[0].texts) for r in sessions)
    finally:
        for _,f in plots:plt.close(f)

def test_P20_quality_remains_strict_and_assisted(scoped):
    from auto_annotation_tool.gui.mobile_comparison_scope import validate_comparison
    from auto_annotation_tool.gui.mobile_comparison_core import figures
    import matplotlib.pyplot as plt
    with pytest.raises(ValueError,match='READY'):
        validate_comparison([scoped[R01].record,scoped[R02].record],comparison_scope='quality',require_sources=True)
    sessions=[scoped[R01].record,scoped[R03].record]
    validate_comparison(sessions,[list(scoped[s].traces) for s in (R01,R03)],comparison_scope='quality',require_sources=True)
    plots=list(figures(sessions,comparison_scope='quality'))
    try:
        assert any('assisted_cer' in n for n,_ in plots)
        assert any('assisted_exact' in n for n,_ in plots)
        assert sessions[0].canvas_exact_pct==pytest.approx(sessions[0].canvas['exact_reads']/sessions[0].canvas['evaluated_reads']*100)
    finally:
        for _,f in plots:plt.close(f)

def test_P21_P22_P26_performance_export_content_and_atomic_numbering(scoped,tmp_path):
    from auto_annotation_tool.gui.mobile_import_comparison import export_imported_charts
    root=tmp_path/'03_wyniki';(root/'porownanie').mkdir(parents=True)
    records=[scoped[s].record for s in (R01,R02)];rows=[list(scoped[s].traces) for s in (R01,R02)]
    first=export_imported_charts(records,rows,root,comparison_scope='performance')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in first.iterdir()}
    second=export_imported_charts(records,rows,root,comparison_scope='performance')
    assert first!=second and not list(root.rglob('.alpr_chart_stage_*'))
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in first.iterdir()}
    manifest=json.loads((first/'wykresy_manifest.json').read_text(encoding='utf-8'))
    assert manifest['comparison_scope']=='performance' and manifest['comparison_kind']=='descriptive'
    assert all(s['report_id'] and s['source_archive_sha256'] and s['trace_sha256'] and s['review_revision'] for s in manifest['sources'])
    assert len(list(first.glob('*.png')))==6 and len(list(first.glob('*.svg')))==6
    assert not (first/'geometria_MT_a_odczyt_MZ.csv').exists()
    for row in csv.DictReader((first/'dane_porownania.csv').open(encoding='utf-8-sig'),delimiter=';'):
        assert row['comparison_scope']=='performance'
        assert all(row[k]=='' for k in ('canvas_assisted','exact_pct','micro_cer_pct','geometry_n','polygon_iou_pct'))
        metric=scoped[row['session_id']].record.latency[row['stage']]
        assert int(row['n'])==metric.n and float(row['p95_ms'])==pytest.approx(metric.p95_ms)
    for item in manifest['files']:
        assert hashlib.sha256((first/item['name']).read_bytes()).hexdigest()==item['sha256']

@pytest.mark.parametrize('changed',['revision','source','trace_argument','export'])
def test_P24_export_revalidates_after_selection(actual,scoped,tmp_path,changed):
    from auto_annotation_tool.gui.mobile_import_comparison import export_imported_charts
    args=clone(actual,tmp_path/'copy');access=assess_copy(args)
    records=[scoped[R01].record,access.record];rows=[list(scoped[R01].traces),list(access.traces)]
    if changed=='revision':
        side=args[-1];d=json.loads(side.read_text(encoding='utf-8'));d['review_revision']+=1;side.write_text(json.dumps(d),encoding='utf-8')
    elif changed=='source':args[1].write_bytes(args[1].read_bytes()+b'changed')
    elif changed=='export':
        p=args[4]/'export_manifest.json';d=json.loads(p.read_text(encoding='utf-8'));d['note']='Changed after selection'
        p.write_text(json.dumps(d),encoding='utf-8')
    else:rows[1]=rows[1][:-1]
    with pytest.raises(ValueError):export_imported_charts(records,rows,args[2],comparison_scope='performance')
    assert not list((args[2]/'porownanie').iterdir())

def test_P24_P26_atomic_publish_revalidates_during_render(actual,scoped,tmp_path,monkeypatch):
    import auto_annotation_tool.gui.mobile_import_comparison as module
    args=clone(actual,tmp_path);a=assess_copy(args);original=module.figures
    def change_during_render(*args2,**kw):
        yield from original(*args2,**kw)
        side=args[-1];d=json.loads(side.read_text(encoding='utf-8'));d['review_revision']+=1;side.write_text(json.dumps(d),encoding='utf-8')
    monkeypatch.setattr(module,'figures',change_during_render)
    with pytest.raises(ValueError):module.export_imported_charts([scoped[R01].record,a.record],[list(scoped[R01].traces),list(a.traces)],args[2],comparison_scope='performance')
    assert not list((args[2]/'porownanie').iterdir())

def test_P25_single_session_export_is_independent_of_comparison_mode(actual,tmp_path):
    from auto_annotation_tool.gui.mobile_import_comparison import export_imported_charts
    root=tmp_path/'03_wyniki';root.mkdir();_,s,rows=actual[0][R02]
    dest=export_imported_charts([s],[rows],root,comparison_scope='single')
    assert list(dest.glob('*.png'))
    assert not any('assisted' in p.name for p in dest.glob('*.png'))
    manifest=json.loads((dest/'wykresy_manifest.json').read_text(encoding='utf-8'))
    assert manifest['scope']=='single_session' and not manifest['sources'][0]['canvas_assisted_ready']

def test_P27_frame_gate_is_not_a_model_FN_or_completed_trace(actual):
    b,s,rows=actual[0][R02]
    assert len(rows)==21 and s.latency['pipeline'].n==21
    stored=json.loads(next(s.directory.rglob('mt_detection_evaluation.json')).read_text(encoding='utf-8'))
    assert s.mt_evaluation==stored
    # 30 received frames are not a latency population; source evaluator untouched.
    assert len(rows)!=30
    base=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    path='auto_annotation_tool/ranking/mobile_mt_detection_evaluation.py'
    assert hashlib.sha256((OUT.parents[1]/path).read_bytes()).hexdigest()==base['sources'][path]

def test_P28_true_MZ_n_and_sum_per_trace_are_explicit(scoped):
    from auto_annotation_tool.gui.mobile_comparison_scope import comparison_warnings
    records=[scoped[R01].record,scoped[R02].record]
    text='\n'.join(comparison_warnings(records,'performance'))
    assert records[0].latency['mz'].n!=records[1].latency['mz'].n
    assert 'Różne liczby' in text and 'sumować wiele wywołań' in text

@pytest.mark.parametrize('field,value,fragment',[('scenario','PAIR_NIGHT_01','DAY'),('apk_sha256','x'*64,'APK'),('device','different','telefony'),('mt_checkpoint','other','checkpoint')])
def test_P29_P30_incompatible_groups_never_silently_merge(scoped,field,value,fragment):
    from auto_annotation_tool.gui.mobile_comparison_scope import validate_comparison
    records=[scoped[R01].record,replace(scoped[R02].record,**{field:value})]
    with pytest.raises(ValueError,match=fragment):validate_comparison(records,comparison_scope='performance')

def test_P30_P31_scientific_conditions_remain_independent_and_visible(scoped):
    from auto_annotation_tool.gui.mobile_comparison_scope import comparison_warnings
    text='\n'.join(comparison_warnings([scoped[R01].record,scoped[R02].record],'performance'))
    assert 'opisowe' in text and 'NIE ZATWIERDZONO' in text
    assert 'term' in text.lower() and 'APK' in text and 'identyczności' in text
    base=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    name='auto_annotation_tool/gui/mobile_report_scientific_comparability.py'
    assert hashlib.sha256((OUT.parents[1]/name).read_bytes()).hexdigest()==base['sources'][name]
