"""GTR2: preserve ambiguous physical identity; count saved outlines read-only."""
from pathlib import Path
import copy,hashlib,json,shutil
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import settled
from test_z4_mt_geometry_layers import source_session
from auto_annotation_tool.ranking import mobile_human_review as review,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_object_evaluation as mt
from auto_annotation_tool.ranking.mobile_canvas_operator_view import physical_plate_ids,saved_reference_ids
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_review_tasks as tasks,mobile_canvas_review_feedback as feedback,mobile_canvas_navigator as nav

OUT=Path(__file__).resolve().parents[1]/'output/Z4_GHOST_GTR2_20261101_001'
SID='exp-1792702658896-47b4e60b';G=SID+'-a00000007';GOOD='GT:'+G;RAW='GT:'+SID+'-a00000008';BAD='GT:'+SID+'-a00000009'
P1='PGT:935580407c757a47393f51d2';P2='PGT:2092896f8c02ea9acf99642b'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
def frozen(s):return digest(s.review.to_dict()),hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()
def state(s,g=G):return core.scene(s,g),stacks.snapshot(s,g)

@pytest.fixture
def actual(tmp_path):
    src=OUT/'private'/SID
    shutil.copyfile(src/'source.alprsession',tmp_path/'source.alprsession');shutil.copyfile(src/'review.json',tmp_path/'review.json')
    return review.MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

@pytest.fixture
def gui(tmp_path,monkeypatch,actual):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    monkeypatch.setattr(native,'source_session',lambda _:actual)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify();root.update()
        iid=next(i for i,k in h._subject_ids.items() if G in subject_groups(s,k));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry']);settled(root,h,ed)
        yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def test_S01_S02_S04_exact_scene_and_three_source_roles(actual):
    s=actual;sc,v=state(s);before=frozen(s)
    assert s.session_id==SID and s.review.review_revision==102
    assert s.review.source_archive_sha256=='aab5df72a72ab422e8c7edcae4cc6b323d68599be87cb2f21ab01ff95fca562f'
    assert list(s.mt_invocations)[4]==G and sc['source_binding']==v['source_binding']
    assert [(c['object_id'],c['role']) for c in v['candidates']]==[(GOOD,'FINAL'),(RAW,'SUPPRESSED'),(BAD,'FINAL')]
    assert v['reference_numbers']=={P1:1,P2:2} and v['plates'][P2]['origin_object_id']==BAD
    assert frozen(s)==before

def test_S03_real_history_is_not_a_proven_physical_withdrawal(actual):
    s=actual;raw=core._legacy_scene(s,G);v=raw[stacks.KEY];h=raw[stacks.HISTORY]['entries'];p=v['plates'][P2]
    assert p['geometry_state']=='operator_removed' and p['gt_quad'] is None and p['confirmed'] and not p['text_confirmed']
    created=next(e for e in h if P2 not in e['before'].get(stacks.KEY,{}).get('plates',{}) and P2 in e['after'].get(stacks.KEY,{}).get('plates',{}))
    assert created['action']=='operator_toggle_frame_presence' and created['review_revision_before']==88
    assert p['origin_object_id'] not in v.get('removed_manual_ids',[])
    assert any(e['action']=='remove_operator_reference_geometry' and e['review_revision_before']==95 for e in h)
    assert partial.progress(core.scene(s,G))['pending_gtr_ids']==[P2]

def test_S11_S17_counter_shows_existing_GTR_before_completion(gui):
    root,h,ed,s=gui;before=frozen(s);science=digest(mt.evaluate_unit(s,G))
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['1','—','≥1','—']
    assert feedback.operator_inventory(*state(s))['counts']['gt']==1
    assert frozen(s)==before and digest(mt.evaluate_unit(s,G))==science

def test_S13_S20_S37_ambiguous_removed_identity_requires_explanation(actual):
    s=actual;before=frozen(s);t=tasks.for_scene(s,G)
    assert t['count']==1 and t['units'][0]['physical_gt_id']==P2
    assert t['units'][0]['kind']=='REFERENCE_CONFLICT'
    assert 'Wyjaśnij' in t['next_action'] and 'GTR 2' in t['next_action'] and '(D)' not in t['next_action']
    assert 'Narysuj' not in t['next_action'] and frozen(s)==before
    assert mt.evaluate_unit(s,G)['status']=='REVIEW_REQUIRED'

@pytest.mark.parametrize('reason',['_post_nms_unknown','migration_warnings','manual_bad'])
def test_S23_population_uncertainty_does_not_hide_saved_GTR(actual,reason):
    s=actual;sc,v=state(s);before=frozen(s)
    if reason=='manual_bad':sc['objects'].append(dict(object_id='legacy-bad',kind='manual',error=True,quad=None))
    else:sc[reason]=True if reason=='_post_nms_unknown' else ['Legacy source uncertainty']
    assert partial.progress(sc)['source_unknown']
    assert feedback.operator_inventory(sc,v)['counts']['gt']==1 and frozen(s)==before

@pytest.mark.parametrize('damage',['binding','quad','unconfirmed','missing_origin','bad_manual_owner'])
def test_S12_untrusted_GT_count_is_NA_with_reason(actual,damage):
    sc,v=state(actual)
    if damage=='binding':v['source_binding']='foreign'
    elif damage=='quad':v['plates'][P1]['gt_quad']=[[0,0]]*4
    elif damage=='unconfirmed':v['plates'][P1]['confirmed']=False
    elif damage=='missing_origin':v['plates'][P1]['origin_object_id']='missing-origin'
    else:
        v['plates'][P1]['origin_object_id']='bad-manual-owner'
        sc['objects'].append(dict(object_id='bad-manual-owner',kind='manual',error=True))
    value=feedback.operator_inventory(sc,v)
    assert value['counts']['gt'] is None and value['gt_reason']

def test_S05_S15_bad_seed_and_roundtrip_never_create_GTR(tmp_path):
    s=source_session(tmp_path,mz=False);g='mt-one';oid='GT:a0';before=copy.deepcopy(core.scene(s,g)['parents'])
    for _ in range(2):
        stacks.toggle_unassociated(s,g,oid,partial_review=True)
        assert not stacks.snapshot(s,g)['plates'] and not partial.progress(core.scene(s,g))['pending_gtr_ids']
    assert core.scene(s,g)['parents']==before

def test_S07_S08_S36_two_explicit_physical_GTs_preserve_missing_second_geometry(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);g='mt-one'
    for oid in ('GT:a0','GT:a1'):partial.confirm_stack(s,g,oid=oid)
    v=stacks.snapshot(s,g);ids=set(v['plates']);assert len(ids)==2
    assert feedback.operator_inventory(core.scene(s,g),v)['counts']['gt']==2
    assert (mt.evaluate_unit(s,g)['tp'],mt.evaluate_unit(s,g)['n_gt'])==(2,2)
    stacks.remove_linked_geometry(s,g,'GT:a1');sc,v=state(s,g);before=frozen(s)
    assert physical_plate_ids(sc,v)==ids and len(saved_reference_ids(sc,v))==1
    t=tasks.for_scene(s,g);assert t['count']==1 and 'Przywróć' in t['next_action'] and '(D)' in t['next_action']
    assert t['units'][0]['kind']!='REFERENCE_CONFLICT' and frozen(s)==before

def test_S06_explicit_manual_reference_removal_is_not_recreated(tmp_path):
    from test_mobile_p2b2_canvas_review import MANUAL
    s=source_session(tmp_path,mz=False);g='mt-one';oid=core.add_plate(s,g,MANUAL,partial_review=True)
    prior=stacks.snapshot(s,g);pid=prior['links'][oid]['plate_id'];core.remove_operator_geometry(s,g,oid)
    sc,v=state(s,g);assert pid not in physical_plate_ids(sc,v) and pid not in partial.progress(sc)['pending_gtr_ids']

def test_S09_S19_controlled_complete_pair_plus_BAD_v2(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);g='mt-one';partial.confirm_stack(s,g,oid='GT:a0');stacks.toggle_unassociated(s,g,'GT:a1',partial_review=True)
    result=mt.evaluate_unit(s,g);assert result['status']=='MEASURED' and (result['n_gt'],result['tp'],result['fp'],result['fn'])==(1,1,1,0)
    assert len(stacks.snapshot(s,g)['plates'])==1

def test_S10_RAW_never_pending_or_rejected(actual):
    sc,v=state(actual);t=tasks.for_scene(actual,G);r=mt.evaluate_unit(actual,G)
    assert RAW not in partial.progress(sc)['object_decisions']
    assert all(u.get('final_mt_id')!=RAW[3:] for u in t['units'])
    assert r['n_raw_suppressed']==1 and len(r['source_finals'])==2

def test_S13_no_history_is_not_permission_to_discard_removed_GTR(actual):
    sc,v=state(actual);sc.pop(stacks.HISTORY,None);sc['events']=[];before=digest((sc,v))
    t=tasks.pending_review_units(sc,v)
    assert t['units'][0]['kind']=='REFERENCE_CONFLICT' and t['count']==1 and digest((sc,v))==before

def test_S13_legacy_without_explicit_decisions_cannot_request_new_GTR(actual):
    from auto_annotation_tool.gui.mobile_canvas_reference_state import missing_outline_task
    sc,v=state(actual);sc.pop(partial.KEY,None);before=digest((sc,v))
    task=missing_outline_task(sc,v,P2,2)
    assert task['kind']=='REFERENCE_CONFLICT' and '(D)' not in task['text'] and digest((sc,v))==before

def test_S16_Undo_Redo_keeps_saved_geometry_identity(actual):
    s=actual;v=stacks.snapshot(s,G);before=copy.deepcopy(v['plates'][P1]);ids=set(v['plates']);q=copy.deepcopy(before['gt_quad']);q[0][0]+=1
    stacks.edit_gt(s,G,GOOD,quad=q);stacks.restore_history(s,G);assert stacks.snapshot(s,G)['plates'][P1]==before
    stacks.restore_history(s,G,redo=True);v=stacks.snapshot(s,G);assert set(v['plates'])==ids and v['plates'][P1]['gt_quad']==q

def test_S18_unchecked_complete_tasks_only_require_image_inspection(tmp_path):
    s=source_session(tmp_path,mz=False);partial.confirm_stack(s,'mt-one',oid='GT:a0');sc,v=state(s,'mt-one');sc['checked']=False
    t=tasks.pending_review_units(sc,v);assert t['count']==0 and t['next_action']=='Sprawdź cały obraz'

def test_S21_layers_Shift_zoom_are_read_only(gui):
    root,h,ed,s=gui;before=frozen(s);counts={k:w.cget('text') for k,w in ed.contexts.counters.items()}
    assert nav.select_layer(ed,'POSE',BAD);ready(root,h,ed)
    for key in ('z','x','Shift_L','s','a','Escape'):native.key(h,key);ready(root,h,ed)
    h._view_zoom*=1.1;h._paint_image();ready(root,h,ed)
    assert frozen(s)==before and {k:w.cget('text') for k,w in ed.contexts.counters.items()}==counts

def test_S22_PPM_and_Space_preserve_physical_reference_ids(gui):
    root,h,ed,s=gui;before=copy.deepcopy(stacks.snapshot(s,G)['plates'])
    nav.select_layer(ed,'POSE',BAD);ready(root,h,ed);clean.toggle_fp(ed);settled(root,h,ed);clean.toggle_fp(ed);settled(root,h,ed)
    assert partial.progress(core.scene(s,G))['object_decisions'][BAD]=='NOT_PLATE'
    nav.select_layer(ed,'GTR',GOOD);ready(root,h,ed);native.key(h,'space');settled(root,h,ed);native.key(h,'space');settled(root,h,ed)
    assert stacks.snapshot(s,G)['plates']==before

def test_S22b_S24_result_layout_converges_after_Space(gui):
    root,h,ed,s=gui;panel=ed.report_host;events=[];rescued=[]
    def observe(event):
        events.append((event.width,event.height))
        if len(events)==60:
            # End an unbounded G0 configure loop so the regression produces a
            # FAIL receipt instead of hanging CI. This changes only test UI.
            rescued.append(True)
            for w in (panel.progress,panel.notice,panel.mt_counts,panel.reads,panel.detection,*panel.limit_buttons.values()):w.configure(width=1)
    token=panel.body.bind('<Configure>',observe,add='+')
    try:
        nav.select_layer(ed,'GTR',GOOD);ready(root,h,ed);native.key(h,'space');settled(root,h,ed)
        assert not rescued,('Layout did not converge',len(events),events[-5:])
    finally:panel.body.unbind('<Configure>',token)

def test_S25_S26_S27_S28_metrics_MZ_source_and_policy_unchanged(actual):
    from auto_annotation_tool.gui.mobile_mz_quality_context import snapshot
    s=actual;before=frozen(s);sc,v=state(s);r=mt.evaluate_unit(s,G);source=digest(s.attempts)
    assert mt.POLICY_SHA256=='c5461594eae9a070d5af37d64a20ba23abd006c4a999a59e2f3d17e3fe4be3f0'
    assert r['status']=='REVIEW_REQUIRED' and r['partial_fp_lower_bound']==1 and all(r[k] is None for k in ('tp','fp','fn'))
    mz=snapshot(s);rows={r['attempt_id']:r for r in mz['rows']}
    assert rows[BAD[3:]]['quality_state']=='EXECUTED_EXCLUDED_MT_FP' and rows[BAD[3:]]['executed']
    assert rows[BAD[3:]]['timing']['character_inference_ms']>0 and not rows[BAD[3:]]['canonical_ocr_eligible']
    assert rows[GOOD[3:]]['quality_state']=='EXECUTED_UNASSESSED' and rows[GOOD[3:]]['operator_decision']=='PLATE'
    feedback.operator_inventory(sc,v);tasks.for_scene(s,G)
    assert mt.evaluate_unit(s,G)==r and digest(s.attempts)==source and frozen(s)==before

def test_S36_manual_physical_reference_survives_rejected_FINAL(tmp_path):
    from test_mobile_p2b2_canvas_review import MANUAL
    s=source_session(tmp_path,mz=False);g='mt-one';oid=core.add_plate(s,g,MANUAL,partial_review=True)
    before=copy.deepcopy(stacks.snapshot(s,g)['plates']);stacks.toggle_unassociated(s,g,'GT:a0',partial_review=True)
    sc,v=state(s,g);assert v['plates']==before and feedback.operator_inventory(sc,v)['counts']['gt']==1
    assert next(iter(v['plates'].values()))['origin_object_id']==oid

@pytest.mark.parametrize('damage',['missing_plate','unknown_role'])
def test_S35_missing_plate_cannot_crash_planner_or_pick_neighbor(actual,damage):
    sc,v=state(actual)
    if damage=='missing_plate':v['plates'].pop(P2)
    else:next(c for c in v['candidates'] if c['object_id']==BAD)['role']='UNKNOWN'
    t=tasks.pending_review_units(sc,v)
    assert t['count']==1 and t['units'][0]['kind']=='REFERENCE_CONFLICT' and t['units'][0]['physical_gt_id']==P2

def test_S38_source_active_saved_and_historical_identity_are_explicit(actual):
    sc,v=state(actual);assert len(physical_plate_ids(sc,v))==2
    assert len(partial.active_plate_ids(sc,v))==len(saved_reference_ids(sc,v))==1
    assert feedback.operator_inventory(sc,v)['counts']['gt']==1
    assert tasks.for_scene(actual,G)['count']==partial.progress(sc)['decisions']==1

def test_S39_verified_negative_and_S40_unreviewed_empty(tmp_path):
    s=source_session(tmp_path,mz=False);sc,v=state(s,'mt-one');sc['objects']=[];sc['parents']=[];sc[partial.KEY]=dict(decisions={},source_roles={},empty_confirmed=False)
    v.update(plates={},links={},candidates=[]);sc[stacks.KEY]=v
    t=tasks.pending_review_units(sc,v);assert not t['units'] and t['steps'][0]['kind']=='SCENE'
    sc[partial.KEY]['empty_confirmed']=True;sc['checked']=True
    assert feedback.operator_inventory(sc,v)['counts']['gt']==0 and tasks.pending_review_units(sc,v)['count']==0
