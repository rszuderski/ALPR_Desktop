"""Operator timing is a versioned presentation parameter, not an MT deadline."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest
from test_final180_desktop import frozen,scoring_state,plate,candidate
from auto_annotation_tool.gui import final180_tv as tv
from auto_annotation_tool.ranking import final180_reference as ref,final180_timeline as timeline
from auto_annotation_tool.ranking import final180_mobile as mobile,photo_reference as photo


@pytest.fixture(autouse=True)
def private_preferences(tmp_path,monkeypatch):
    from auto_annotation_tool import config
    from auto_annotation_tool.session import SessionManager
    session=SessionManager(tmp_path/'user_session.json');session.set('unrelated','keep','unchanged')
    monkeypatch.setattr(config,'SESSION',session)
    return session


@pytest.mark.parametrize('value,seconds',[('2,5',2.5),('2.5',2.5),('5',5),('7',7),('10',10),('30',30),('0.001',.001),('3600',3600)])
def test_supported_operator_seconds(value,seconds):
    assert ref.parse_dwell(value)==seconds


@pytest.mark.parametrize('value',['','text','0','-1','NaN','inf','-Infinity','1,2.5','3601','0.0000000001',True,None])
def test_invalid_values_are_not_substituted(value):
    with pytest.raises(ValueError,match='Czas'):ref.parse_dwell(value)


@pytest.mark.parametrize('count,seconds,total',[(176,5,972.5),(176,10,1852.5),(3,2.5,13.5),(1,7,12),(0,5,5)])
def test_plan_is_based_on_population_and_separate_phases(count,seconds,total):
    result=ref.plan_times(count,seconds)
    assert result['planned_total_seconds']==total
    assert result['planned_photo_seconds']==count*seconds
    assert result['countdown_seconds']==5 and result['transition_seconds']==.5


@pytest.mark.parametrize('seconds',[2.5,5,7,10,30])
def test_versioning_preserves_exact_sources_and_controller_deadline(tmp_path,seconds):
    parent=frozen(tmp_path/'material',n=2,negative=1,dwell_seconds=30)
    before={p.relative_to(parent).as_posix():photo.digest(p) for p in parent.rglob('*') if p.is_file()}
    target=ref.replan(parent,seconds);m,_,_=ref.load(target);old,_,_=ref.load(parent)
    assert m['exposures']==old['exposures'] and m['layout']==old['layout'] and m['policy']==old['policy']
    for name in ('material.json','reference.json','source_annotations.xml'):assert photo.digest(target/name)==before[name]
    if seconds!=30:
        assert photo.digest(target/'manifest.json')!=before['manifest.json']
        assert m['parent_plan']['manifest_sha256']==before['manifest.json']
    assert all(photo.digest(parent/name)==value for name,value in before.items())
    now=[0];run=tv.PresentationRun(target,m,clock=lambda:now[0]);run.present(0,{},lambda:None)
    assert run.deadline==round(seconds*1e9)
    m['dwell_seconds']=99  # External mutable GUI data cannot change admitted timing.
    now[0]=run.deadline-1;assert not run.due()
    now[0]+=1;run.end_photo();run.replace(lambda:None)
    run.present(1,{},lambda:None);assert run.deadline==round(seconds*2e9)
    run.stop(clear=lambda:None)
    row=timeline.project(run.manifest,run.events)[0]
    assert row['planned_duration_s']==seconds and row['duration_pc_s']==seconds
    assert photo.read_json(run.directory/'run.json')['dwell_seconds']==seconds
    request=next(e for e in run.events if e['event']=='PHOTO_REQUESTED')
    assert request['manifest_sha256']==photo.digest(target/'manifest.json') and request['dwell_seconds']==seconds


def test_delayed_io_and_render_do_not_consume_the_five_seconds(tmp_path,monkeypatch):
    d=frozen(tmp_path,dwell_seconds=5);m,_,_=ref.load(d);now=[0]
    r=tv.PresentationRun(d,m,clock=lambda:now[0]);fsync=tv.os.fsync
    def slow(fd):now[0]+=1_000_000_000;fsync(fd)
    monkeypatch.setattr(tv.os,'fsync',slow)
    r.present(0,{},lambda:now.__setitem__(0,now[0]+200_000_000))
    start=next(e['pc_perf_counter_ns'] for e in r.events if e['event']=='PHOTO_VIEW_UPDATED')
    assert r.deadline-start==5_000_000_000
    now[0]=r.deadline;r.end_photo();r.replace(lambda:None)
    assert timeline.project(m,r.events)[0]['duration_pc_ns']>=5_000_000_000
    r.stop()


def test_complete_176_five_second_software_schedule(tmp_path):
    d=frozen(tmp_path,n=166,negative=10,dwell_seconds=5);m,_,_=ref.load(d);now=[0]
    r=tv.PresentationRun(d,m,clock=lambda:now[0]);r.replace(lambda:None,phase='COUNTDOWN',reason='COUNTDOWN')
    r.plan_origin_ns=0;now[0]=5_000_000_000
    for i in range(176):
        r.present(i,{},lambda:None);now[0]+=5_000_000_000;r.end_photo()
        if i==175:r.complete(lambda:None)
        else:r.replace(lambda:None);now[0]+=500_000_000
    rows=timeline.project(m,r.events)
    assert len(rows)==176 and all(x['duration_pc_s']==5 and x['status']=='COMPLETED' for x in rows)
    assert now[0]==972_500_000_000 and sum(x['duration_pc_s'] for x in rows)==880
    assert len([e for e in r.events if e['event']=='REPLACEMENT_VIEW_UPDATED' and e['phase']=='T'])==175


def test_report_exposes_frozen_duration_and_measured_interval(tmp_path):
    d=frozen(tmp_path,dwell_seconds=2.5);m,_,_=ref.load(d);now=[0];r=tv.PresentationRun(d,m,clock=lambda:now[0])
    r.present(0,{},lambda:None);now[0]=1_000_000_000;r.stop(clear=lambda:None)
    report=mobile.analyze_archive(d,tmp_path/'absent.alprsession',r.directory/'run.json')
    assert report['presentation_plan']['dwell_seconds']==2.5
    assert report['presentation_timeline'][0]['duration_pc_s']==1
    target=mobile.export_report(report,tmp_path/'reports')
    assert '2.5 s/' in (target/'report.md').read_text('utf-8')


def test_new_preference_does_not_change_resume_and_tail_recovery(tmp_path):
    d=frozen(tmp_path,dwell_seconds=7);m,_,_=ref.load(d);now=[0];r=tv.PresentationRun(d,m,clock=lambda:now[0])
    r.present(0,{},lambda:None);r.journal.close()
    with r.journal_path.open('ab') as f:f.write(b'{"broken":')
    old=r.journal_path.read_bytes();tv.remember_selection(2.5)
    resumed=tv.PresentationRun(d,m,resume=r.directory,clock=lambda:now[0]);resumed.present(1,{},lambda:None)
    assert resumed.dwell_seconds==7 and resumed.next_index==2 and resumed.deadline==7_000_000_000
    resumed.stop(clear=lambda:None)
    assert r.journal_path.read_bytes()==old
    rows=timeline.project(m,resumed.events);assert rows[0]['status']=='UNKNOWN' and rows[2]['status']=='NOT_SHOWN'


def test_preference_uses_existing_session_and_default_is_five(private_preferences):
    assert tv.preferred_dwell()==5
    tv.remember_selection('2,5');assert tv.preferred_dwell()==2.5
    saved=photo.read_json(private_preferences.session_file)
    assert saved['unrelated']['keep']=='unchanged' and saved['final_presentation']['dwell_seconds']==2.5
    with pytest.raises(ValueError):tv.remember_selection('NaN')
    assert tv.preferred_dwell()==2.5


def test_cli_check_loaded_plan_and_override_are_truthful_readonly(tmp_path,capsys,private_preferences):
    d=frozen(tmp_path,dwell_seconds=30);before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    tv.remember_selection(5);prefs=private_preferences.session_file.read_bytes()
    tv.main(['--prepared',str(d),'--check'])
    report=json.loads(capsys.readouterr().out);assert report['plan']['dwell_seconds']==30 and not report['requires_new_plan']
    tv.main(['--prepared',str(d),'--dwell','2,5','--check'])
    report=json.loads(capsys.readouterr().out);assert report['frozen_dwell_seconds']==30 and report['plan']['dwell_seconds']==2.5 and report['requires_new_plan']
    assert all(photo.digest(p)==sha for p,sha in before.items()) and not (d/'runs').exists()
    assert private_preferences.session_file.read_bytes()==prefs
    assert '--dwell' not in tv.accepted_arguments(['--prepared',str(d)])


def test_five_second_source_identity_and_denominators_are_unchanged():
    # A result completing after the second display begins still belongs to e0001.
    e,s,f=scoring_state([plate('p')],[candidate('a',[10,10,40,30])]);f['mt_completed_elapsed_nanos']=9_000_000_000
    first,reason=mobile.select_first('e0001',[dict(f,exposure_id='e0002',captured_elapsed_nanos=2),f])
    assert first is f and not reason and mobile.score_frame(e,s,f)['tp']==1
    cancelled=dict(f,execution_status='CANCELLED');later=dict(f,captured_elapsed_nanos=3)
    assert mobile.select_first('e0001',[later,cancelled])[0] is cancelled
    # Per-plate full text count is 203, independently of the 176 exposure rows.
    rows=[]
    for i in range(176):
        objects=[plate(f'p{i}-{j}') for j in range(2 if i<37 else 1)] if i<166 else []
        if i<6:objects.append(plate('unreadable'+str(i),text=''))
        ex,state,frame=scoring_state(objects,[])
        rows.append(mobile.score_frame(ex,state,frame))
    summary=mobile.summarize(rows,[],dict(exposures=[dict(source_role='POSITIVE' if i<166 else 'NEGATIVE') for i in range(176)]))
    assert summary['system_full_read']['denominator']==203 and summary['planned_exposures']==176
    assert sum(len(r['plates']) for r in rows)==209


def test_native_gui_edit_prepare_start_lock_stop_resume(tmp_path,monkeypatch):
    if os.environ.get('F6_DWELL_ISOLATED_TK')!='1':
        p=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_gui_edit_prepare_start_lock_stop_resume','-q'],
            env=dict(os.environ,F6_DWELL_ISOLATED_TK='1'),capture_output=True,text=True,timeout=90)
        assert p.returncode==0,p.stdout+p.stderr
        return
    import tkinter as tk
    d=frozen(tmp_path/'material',dwell_seconds=30);legacy=tv.legacy_module(tv.LEGACY_DEFAULT)
    monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda: [dict(name='TEST_ONLY',w=1920,h=1080,x=0,y=0,verified=True,geometry_source='ISOLATED_TEST')])
    root=tk.Tk();root.withdraw()
    args=NS(xml=None,positives=None,negatives=None,output=tmp_path/'plans',dwell=5,negatives_declared=True,unreadable_ids=[])
    app=tv.make_app(legacy,args,parent=root);app.window.withdraw()
    # Real Tk widget/control flow, no physical monitor presentation or inference.
    def viewer(_):
        app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
    monkeypatch.setattr(app,'_make_viewer',viewer)
    monkeypatch.setattr(app,'_tick',lambda:None)
    try:
        app.loaded(d);assert app.dwell.get()=='30.0'
        app.dwell.set('NaN');app.begin_countdown();assert app.run is None
        app.dwell.set('2,5');assert '13,5' in app.plan.get() and '30 s' in app.plan.get()
        app.begin_countdown();assert app.mode=='PREPARING' and str(app.dwell_entry.cget('state'))=='disabled'
        deadline=time.monotonic()+20
        while app.mode=='PREPARING' and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert app.mode=='COUNTDOWN',app.status.get()
        assert app.run.dwell_seconds==2.5 and app.prepared!=d
        assert app.manifest['countdown_seconds']==5 and app.manifest['transition_seconds']==.5
        for mode in ('COUNTDOWN','RUNNING','TRANSITION'):
            app.mode=mode;app.dwell.set('7');assert float(app.dwell.get())==2.5
        previous=app.run.directory;app.abort();assert str(app.dwell_entry.cget('state'))=='normal'
        app.dwell.set('7');app.begin_countdown(resume=previous)
        assert float(app.dwell.get())==2.5 and app.run.dwell_seconds==2.5 and app.mode=='COUNTDOWN'
        app.abort();app.loaded(d);assert app.dwell.get()=='30.0' and app.manifest['dwell_seconds']==30
    finally:app._close();root.destroy()
