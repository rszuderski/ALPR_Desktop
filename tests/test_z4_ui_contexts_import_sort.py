"""Read-only UX projections; scientific fixtures retain their canonical results."""
from dataclasses import replace
from types import SimpleNamespace as NS
import copy
import pytest
from test_mobile_p2b2_canvas_review import source_session, text_all, MANUAL
from test_z4_akt89lm_resolution import invalid_source
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory, execution_state, focused_assessment, review_contexts
from auto_annotation_tool.gui.z4_mobile_report_browser_ux import report_sort_value, sorted_report_view, report_repeat, report_sort_preference, save_report_sort_preference


def contexts(s, oid='', gid='mt-one'):
    scene=core.scene(s,gid)
    model=focused_assessment(scene,oid,execution_state(s,gid),allow_auto=False)
    return review_contexts(s,scene,model,build_inventory(s)),model


def test_explicit_selection_fp_and_no_selection_never_mix_results(tmp_path):
    s=source_session(tmp_path);text_all(s);core.toggle_error(s,'mt-one','GT:a1');core.check_scene(s,'mt-one')
    before=copy.deepcopy(s.review.to_dict());summary=core.summary(s)
    a,_=contexts(s,'GT:a0');b,m=contexts(s,'GT:a1');empty,n=contexts(s)
    assert 'Wskazanie MT nr 1 — trafne' in a['object'] and 'WA01234' in a['object']
    assert 'Wskazanie MT nr 2 — błędne' in b['object']
    assert all(v not in b['object'] for v in ('WA01234','WA11234','CER','IoU','GT:'))
    assert a['image']==b['image']==empty['image'] and a['session']==b['session']==empty['session']
    assert 'GT fizyczne: 2' in a['image'] and 'Błędne: 1' in a['image'] and 'Pominięte: 1' in a['image']
    assert 'Wybierz ramkę' in empty['object'] and n['focus_id']=='' and n['focus_mode']=='NONE'
    assert m['cer'] is m['polygon_iou_pct'] is m['nmce_pct'] is None
    assert s.review.to_dict()==before and core.summary(s)==summary


@pytest.mark.parametrize('missing',[False,True])
def test_linked_invalid_pose_has_identity_and_reason_without_geometry(tmp_path,missing):
    s=invalid_source(tmp_path,missing_evidence=missing)
    pending,_=contexts(s,'GT:a0')
    assert ('brak danych narożników' if missing else 'niepoprawny wynik Pose') in pending['object']
    manual=core.add_plate(s,'mt-one',MANUAL);core.set_scene_review_state(s,'mt-one',checked=True)
    link_manual_to_mt(s,'mt-one',manual,'GT:a0')
    before=copy.deepcopy(s.review.to_dict());d,m=contexts(s,'GT:a0')
    assert 'Powiązanie tablicy potwierdzone' in d['object']
    assert ('brak danych narożników' if missing else 'niepoprawny wynik Pose') in d['object']
    assert m['polygon_iou_pct'] is m['nmce_pct'] is None
    assert 'ocena kompletna' in d['image'] and s.review.to_dict()==before


def test_two_confirmed_gt_ids_remain_distinct_in_presentation(tmp_path):
    s=source_session(tmp_path);text_all(s);core.check_scene(s,'mt-one')
    summary=core.summary(s);a,_=contexts(s,'GT:a0');b,_=contexts(s,'GT:a1')
    assert 'WA01234' in a['object'] and 'WA11234' not in a['object']
    assert 'WA11234' in b['object'] and 'WA01234' not in b['object']
    assert a['image']==b['image'] and 'GT fizyczne: 2' in a['image']
    assert len({p['gt_id'] for p in core.scene(s,'mt-one')['physical_gt_v1']['plates']})==2
    assert core.summary(s)==summary


def test_manual_fn_and_unknown_negative_are_not_invented(tmp_path):
    s=source_session(tmp_path,n=0,mz=False)
    initial,_=contexts(s);assert 'do ustalenia' in initial['image'] and 'Pominięte:' not in initial['image']
    oid=core.add_plate(s,'mt-one',MANUAL);core.set_text(s,'mt-one',oid,'MANUAL');core.check_scene(s,'mt-one')
    d,m=contexts(s,oid)
    assert 'bez wskazania MT' in d['object'] and 'Model odczytał: nie wykonano' in d['object']
    assert 'Pominięte: 1' in d['image'] and m['cer'] is None


@pytest.mark.parametrize('change',[dict(executed=False),dict(cancelled=True),dict(execution_failed=True)])
def test_non_execution_context_never_claims_quality_counts(tmp_path,change):
    s=source_session(tmp_path,n=0,mz=False);g=s.mt_invocations['mt-one']
    if 'executed' in change:
        for r in g.records:r.update(mt_status='NOT_RUN',mt_executed=False)
    s.mt_invocations['mt-one']=replace(g,**change)
    e=execution_state(s,'mt-one');m=focused_assessment(None,'',e,allow_auto=False)
    d=review_contexts(s,None,m,build_inventory(s))
    assert 'nie FN' in d['image'] and 'Pominięte:' not in d['image'] and 'TN' not in d['image']


def report(key, date=None, repeat=None, p95=None):
    return NS(report_id=key,source_archive_sha256=key*64,source_path='',package_id='p',variant_id='v',
              measured_at=date,experiment_index={'measured_at':date,'replicate_index':repeat},raw={},p95=p95)


def sort(rows,column,desc=False):
    return sorted_report_view(rows,lambda r:report_sort_value(r,column,{'variant':r.variant_id,'scenario':r.package_id},NS(label='W toku'),r.p95),desc)


def test_dates_absolute_time_year_zone_and_missing_last_both_directions():
    a=report('a','2025-12-31T23:30:00-02:00');b=report('b','2026-01-01T02:00:00+02:00')
    c=report('c');d=report('d','2026-01-01T01:30:00Z');e=report('e','bad')
    assert [r.report_id for r in sort([e,d,c,b,a],'Data')]==['b','a','d','c','e']
    assert [r.report_id for r in sort([e,d,c,b,a],'Data',True)]==['a','d','b','c','e']
    a.experiment_index={};a.raw={}  # importer may have filled measured_at with its clock
    assert report_sort_value(a,'Data',{},None,None) is None


@pytest.mark.parametrize('column',['Powt.','p95'])
def test_numeric_sort_missing_and_stable_ties(column):
    rows=[report('a',repeat=10,p95=100),report('b',repeat=2,p95=9),report('c'),report('d',repeat=2,p95=9)]
    assert [r.report_id for r in sort(rows,column)]==['b','d','a','c']
    assert [r.report_id for r in sort(rows,column,True)]==['a','b','d','c']
    rows[0].source_path='C:/x/02_sesje/DAY/scene/r02/file.alprsession'
    assert report_repeat(rows[0])==('r02',2)


def test_text_missing_stable_and_existing_session_preferences():
    rows=[report('b'),report('a'),report('c')];rows[2].variant_id=''
    assert [r.report_id for r in sort(rows,'Wariant')]==['a','b','c']
    assert [r.report_id for r in sort(rows,'Wariant',True)]==['a','b','c']
    class Settings:
        values={};saved=0
        def get(self,section,key,default=None):return self.values.get((section,key),default)
        def set(self,section,key,value):self.values[section,key]=value
        def save_session(self):self.saved+=1
    settings=Settings();assert report_sort_preference(settings)==('Data',True)
    save_report_sort_preference(settings,'Powt.',False)
    assert settings.saved==1 and report_sort_preference(settings)==('Powt.',False)


from test_mobile_p2b2_canvas_gui import gui, pump


def test_native_session_switch_same_gid_and_revision_clears_object_and_cache(gui,tmp_path):
    root,h,ed,original=gui
    ed._choose_plate('GT:a1');pump(root)
    old=ed._read();old_inv=ed.inventory()
    other_dir=tmp_path/'other';other_dir.mkdir()
    other=source_session(other_dir,n=1)
    assert other.review.review_revision==original.review.review_revision
    frozen=copy.deepcopy(other.review.to_dict())
    ed.pending_tree_target=(h.current_invocation,'GT:a0')  # queued by the old source
    h.session=other;ed.sync();ed.overlay();pump(root)
    assert ed.selected=='' and ed.assessment.model['focus_id']==''
    assert ed._read() is not old and ed.inventory() is not old_inv
    assert len(ed._read()['objects'])==1 and 'Wybierz ramkę' in ed.contexts.data['object']
    assert other.review.to_dict()==frozen


def test_sort_preference_survives_reopening_existing_session_manager(tmp_path):
    from auto_annotation_tool.session import SessionManager
    path=tmp_path/'isolated_ui_settings.json'
    settings=SessionManager(path)
    settings.set('mobile_report_import','last_successful_folder','UNCHANGED')
    save_report_sort_preference(settings,'p95',False)
    reopened=SessionManager(path)
    assert report_sort_preference(reopened)==('p95',False)
    assert reopened.get('mobile_report_import','last_successful_folder')=='UNCHANGED'
