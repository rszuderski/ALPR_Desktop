"""PPM source/seed, active accent and stationary outside-click contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,shutil,time,hashlib,json
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking import mobile_human_review as review,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers

OUT=Path(__file__).resolve().parents[1]/'output/Z4_PPM_SEED_ACTIVE_CLICK_20261031_001'
SID='exp-1792699864138-add3e282';G=SID+'-a00000008';IDS=['GT:'+G,'GT:'+SID+'-a00000009']

def frozen(s):return hashlib.sha256(json.dumps(s.review.to_dict(),sort_keys=True).encode()).hexdigest(),hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()
def same_saved(before,after):
    assert set(before)==set(after)
    for pid,p in before.items():
        assert all(after[pid].get(k)==v for k,v in p.items())
        # The existing final-scene confirmation may add DISTINCT; it never
        # changes an existing identity, GTZ, quad or approval.
        assert set(after[pid])-set(p)<= {'scene_identity'}
        if 'scene_identity' not in p and 'scene_identity' in after[pid]:assert after[pid]['scene_identity']=='DISTINCT'
def settled(root,h,ed):
    native.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.refresh_after,seconds=30);ready(root,h,ed)

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
    h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify();root.update()
    iid=next(i for i,k in h._subject_ids.items() if G in subject_groups(s,k))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry']);settled(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def pick(gui,oid=IDS[1],kind='GTR'):
    root,h,ed,s=gui;assert nav.select_layer(ed,kind,oid);ready(root,h,ed)
    return tuple(sum(p[i] for p in nav.quad(layers.model(ed)))/4 for i in (0,1))
def ppm(gui,point):
    root,h,ed,s=gui;native.click(h,ed,point,button=3);settled(root,h,ed)

@pytest.mark.parametrize('oid',IDS)
@pytest.mark.parametrize('kind',['GTR','POSE','BBOX'])
def test_P01_P04_P09_P18_real_single_PPM_without_D(gui,oid,kind):
    root,h,ed,s=gui;point=pick(gui,oid,kind);row=clean.selected(ed);assert row['seed'] and not row['plate']
    before=copy.deepcopy(core.scene(s,G)['parents']);other=next(x for x in IDS if x!=oid)
    hud=ed.source_hud.cget('text');revision=s.review.review_revision
    ppm(gui,point)
    assert partial.progress(core.scene(s,G))['object_decisions'][oid]=='NOT_PLATE'
    assert partial.progress(core.scene(s,G))['object_decisions'][other]=='UNRESOLVED'
    assert not stacks.snapshot(s,G)['plates'] and not clean.selected(ed).get('seed')
    assert core.scene(s,G)['parents']==before and s.review.review_revision==revision+1
    assert ed.source_hud.cget('text')==hud and ed.c.find_withtag('fp_diagonal')

def test_P22_space_on_BAD_explicitly_accepts_same_source_without_full_image_check(gui):
    root,h,ed,s=gui
    # Secondary precondition only; P01 proves the real PPM event independently.
    stacks.toggle_unassociated(s,G,IDS[1],partial_review=True);settled(root,h,ed);pick(gui,IDS[1],'POSE')
    parents=copy.deepcopy(core.scene(s,G)['parents']);native.key(h,'space');settled(root,h,ed)
    sc=core.scene(s,G);v=stacks.snapshot(s,G)
    assert partial.progress(sc)['object_decisions'][IDS[1]]==partial.PLATE and len(v['plates'])==1
    assert sc['parents']==parents and not sc['checked'] and not partial.progress(sc)['pending_gtr_ids']
    assert ed.clean_gtr_confirm_button.cget('state')=='normal'

def test_P11_P12_P21_roundtrip_seed_and_Space(gui):
    root,h,ed,s=gui;point=pick(gui);before=copy.deepcopy(core.scene(s,G)['parents'])
    ppm(gui,point);ed.undo();settled(root,h,ed);assert clean.selected(ed)['seed']
    ed.undo(redo=True);settled(root,h,ed);assert not clean.selected(ed)['seed']
    ppm(gui,point);assert clean.selected(ed)['seed'] and not stacks.snapshot(s,G)['plates']
    native.key(h,'space');settled(root,h,ed);assert len(stacks.snapshot(s,G)['plates'])==1
    assert core.scene(s,G)['parents']==before

def test_P06_P07_P08_Shift_pan_ZX(gui):
    root,h,ed,s=gui;point=pick(gui);ed.fit_plate_toggle();ready(root,h,ed);ed.fit_plate_toggle();ready(root,h,ed)
    assert ed.selected==IDS[1]
    native.focus_canvas(h);ed.c.event_generate('<KeyPress-Shift_L>');ready(root,h,ed)
    ppm(gui,point);assert partial.progress(core.scene(s,G))['object_decisions'][IDS[1]]=='NOT_PLATE'
    ed.undo();settled(root,h,ed);nav.select_layer(ed,'BBOX');ready(root,h,ed)
    h._view_center=(h._view_center[0]+3,h._view_center[1]+2);h._paint_image();ready(root,h,ed)
    ppm(gui,point);assert partial.progress(core.scene(s,G))['object_decisions'][IDS[1]]=='NOT_PLATE'

def test_P13_P14_P15_saved_reference_protection(gui):
    root,h,ed,s=gui;partial.confirm_stack(s,G,oid=IDS[0]);stacks.edit_gt(s,G,IDS[0],text='CONTROLGT');settled(root,h,ed)
    saved=copy.deepcopy(stacks.snapshot(s,G)['plates']);point=pick(gui,IDS[0],'GTR');before=frozen(s)
    ppm(gui,point);assert frozen(s)==before
    pick(gui,IDS[0],'POSE');ppm(gui,point);assert stacks.snapshot(s,G)['plates']==saved
    point=pick(gui,IDS[1]);ppm(gui,point);same_saved(saved,stacks.snapshot(s,G)['plates'])

def empty(ed):
    return 12,ed.c.winfo_height()-12

def click_pixels(ed,point):
    x,y=point;ed.c.event_generate('<ButtonPress-1>',x=x,y=y);ed.c.event_generate('<ButtonRelease-1>',x=x,y=y)

def test_A01_active_accent_and_outside_click_read_only(gui):
    root,h,ed,s=gui;pick(gui);before=frozen(s)
    assert ed.c.find_withtag('active_frame_accent')
    click_pixels(ed,empty(ed));settled(root,h,ed)
    assert not ed.selected and not ed.gt_focus_id and ed.mode=='select'
    assert not ed.c.find_withtag('active_frame_accent') and frozen(s)==before

def test_A02_long_hold_does_not_deselect(gui):
    root,h,ed,s=gui;pick(gui);before=frozen(s);oid=ed.selected;x,y=empty(ed)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y)
    if hasattr(ed,'_outside_click') and ed._outside_click:ed._outside_click['start']-=1
    if hasattr(ed,'_left_press_time'):ed._left_press_time-=1
    ed.c.event_generate('<ButtonRelease-1>',x=x,y=y);settled(root,h,ed)
    assert ed.selected==oid and frozen(s)==before

@pytest.mark.parametrize('return_to_start',[False,True])
def test_A03_pan_never_deselects_even_returning_to_origin(gui,return_to_start):
    root,h,ed,s=gui;pick(gui);before=frozen(s);oid=ed.selected;x,y=empty(ed)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);ed.c.event_generate('<B1-Motion>',x=x+40,y=y-30)
    if return_to_start:ed.c.event_generate('<B1-Motion>',x=x,y=y)
    ed.c.event_generate('<ButtonRelease-1>',x=x if return_to_start else x+40,y=y if return_to_start else y-30);settled(root,h,ed)
    assert ed.selected==oid and frozen(s)==before

def test_A04_click_other_frame_selects_it_and_escape_matches_outside(gui):
    root,h,ed,s=gui;pick(gui);before=frozen(s)
    q=next(r['gt'] for r in clean.records(ed) if r['oid']==IDS[0]);point=tuple(sum(p[i] for p in q)/4 for i in (0,1))
    native.click(h,ed,point);ready(root,h,ed);assert ed.selected==IDS[0]
    assert ed.c.find_withtag('active_frame_accent');ed.escape();ready(root,h,ed)
    assert not ed.selected and not ed.c.find_withtag('active_frame_accent') and frozen(s)==before

def test_A05_click_draft_does_not_escape_and_corner_drag_persists(gui):
    root,h,ed,s=gui;pick(gui,IDS[0]);ed.escape();ready(root,h,ed);native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw';click_pixels(ed,empty(ed));ready(root,h,ed);assert ed.mode=='draw'
    ed.escape();pick(gui,IDS[1]);native.key(h,'d');ready(root,h,ed);q=copy.deepcopy(clean.selected(ed)['gt'])
    x,y=map(round,ed.to_canvas(q[0]));ed.c.event_generate('<ButtonPress-1>',x=x,y=y)
    ed.c.event_generate('<B1-Motion>',x=x+4,y=y+4);ed.c.event_generate('<ButtonRelease-1>',x=x+4,y=y+4);settled(root,h,ed)
    assert ed.selected==IDS[1] and len(stacks.snapshot(s,G)['plates'])==1

def test_P10_P19_reopen_single_event_and_pending(gui):
    root,h,ed,s=gui;point=pick(gui);revision=s.review.review_revision
    # Request pending paints before one complete native right-click.
    ed.request_overlay();ed.request_overlay();ppm(gui,point)
    assert s.review.review_revision==revision+1
    assert partial.progress(core.scene(s,G))['decisions']==1
    reopened=review.MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    from test_z4_auto_spatial_gtr import editor
    rows=clean.records(editor(reopened,G,IDS[1]));assert not next(r for r in rows if r['oid']==IDS[1])['seed']

def test_P16_RAW_Shift_label_never_BAD(gui):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    root,h,ed,s=gui;gid=SID+'-a00000005'
    iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);settled(root,h,ed)
    native.focus_canvas(h);ed.c.event_generate('<KeyPress-Shift_L>');ready(root,h,ed)
    label=next(r for r in ed.shift_preview_labels if r['frame']['role']=='SUPPRESSED');l,t,r,b=label['rect'];before=frozen(s)
    ed.c.event_generate('<ButtonPress-3>',x=round((l+r)/2),y=round((t+b)/2));ed.c.event_generate('<ButtonRelease-3>',x=round((l+r)/2),y=round((t+b)/2));settled(root,h,ed)
    assert frozen(s)==before

def test_S01_separator_has_only_text_halo(gui):
    root,h,ed,s=gui;pick(gui);before=frozen(s);items=ed.c.find_withtag('stack_position_separator')
    assert items and all(ed.c.type(i)=='text' for i in items)
    assert len(ed.c.find_withtag('stack_current_badge'))==2 and len(ed.c.find_withtag('stack_count_badge'))==2
    assert frozen(s)==before

def test_M01_M04_M07_M08_M09_real_MZ_parentage_and_quality(actual,tmp_path):
    from auto_annotation_tool.gui import mobile_mz_quality_context as mz
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    s=actual;source=copy.deepcopy((s.attempts,s.samples));hud=frame_execution(s,G)
    # Controlled sibling GTR is prepared only in this copy: none is saved on
    # original image07 revision71. Its prediction is kept as test GT text.
    partial.confirm_stack(s,G,oid=IDS[0]);stacks.edit_gt(s,G,IDS[0],text='WI6U1')
    saved=copy.deepcopy(stacks.snapshot(s,G)['plates']);stacks.toggle_unassociated(s,G,IDS[1],partial_review=True)
    doc=mz.snapshot(s);rows={r['attempt_id']:r for r in doc['rows'] if r['mt_invocation_id']==G}
    bad=rows[IDS[1][3:]];good=rows[IDS[0][3:]]
    assert bad['parent_proven'] and bad['mt_detection_index']==1 and bad['raw_prediction']=='W'
    assert bad['quality_state']=='EXECUTED_EXCLUDED' and bad['quality_reason']=='MZ_EXCLUDED_MT_FALSE_POSITIVE'
    assert bad['source_mz_status']=='READ' and bad['source_mz_ms']=='603.3909' and bad['timing']['frame_id']==1158
    assert good['quality_state']=='EXECUTED_ELIGIBLE' and good['raw_prediction']=='WI6U1'
    assert core.summary(s)['rows'][0] is not None
    result=next(r for r in core.summary(s)['rows'] if r['attempt_id']==IDS[1][3:]);assert result['cer'] is None and result['outcome']=='FALSE_POSITIVE_MT'
    same_saved(saved,stacks.snapshot(s,G)['plates'])
    assert (s.attempts,s.samples)==source and frame_execution(s,G)==hud
    paths=mz.export(s,tmp_path/'export');prior=[p.read_bytes() for p in paths]
    with pytest.raises(ValueError):mz.export(s,tmp_path/'export')
    assert [p.read_bytes() for p in paths]==prior
    stacks.restore_history(s,G);assert next(r for r in mz.snapshot(s)['rows'] if r['attempt_id']==IDS[1][3:])['quality_state']!='EXECUTED_EXCLUDED'
    stacks.restore_history(s,G,redo=True);assert next(r for r in mz.snapshot(s)['rows'] if r['attempt_id']==IDS[1][3:])['quality_state']=='EXECUTED_EXCLUDED'

def test_M05_M06_multiple_explicit_calls_and_unknown_parent(actual):
    from auto_annotation_tool.gui import mobile_mz_quality_context as mz
    s=actual;aid=IDS[1][3:];sample=copy.deepcopy(next(r for r in s.samples.values() if r['attempt_id']==aid))
    # Synthetic in-memory call records, explicitly parented; originals never change.
    for i in range(2):
        sample=copy.deepcopy(sample);sample['id']='qa-call-'+str(i);sample['mz_execution_id']='qa-call-'+str(i)
        s.samples[sample['id']]=sample
    unknown=copy.deepcopy(sample);unknown['id']='qa-unknown';unknown['attempt_id']='missing-parent';s.samples[unknown['id']]=unknown
    stacks.toggle_unassociated(s,G,IDS[1],partial_review=True)
    rows=mz.snapshot(s)['rows'];calls=[r for r in rows if r['source_execution_id'] in ('qa-call-0','qa-call-1') and r['parent_proven']]
    assert len(calls)==2 and all(r['quality_reason']=='MZ_EXCLUDED_MT_FALSE_POSITIVE' for r in calls)
    unknown=next(r for r in rows if r['source_sample_id']=='qa-unknown')
    assert not unknown['parent_proven'] and unknown['quality_reason']=='MZ_PARENT_UNPROVEN' and not unknown['canonical_ocr_eligible']

def test_P05_overlapping_unselected_seeds_require_source_selection(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    from test_z4_auto_spatial_gtr import editor
    from auto_annotation_tool.gui.mobile_canvas_reference_seed import right_target
    def identical(rows,trace,payload):
        rows[1]['mt_pose_corners_norm']=rows[0]['mt_pose_corners_norm']
        for r in rows:r.update(plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=2,mz=False,mutate=identical);s.set_review_mode('assisted')
    ed=editor(s,'mt-one');ed.selected='';ed.mode='select';ed.drag=None;ed.to_source=lambda x,y:(x,y);ed.to_canvas=lambda p:p
    before=frozen(s)
    with pytest.raises(ValueError,match='Nakładające'):right_target(ed,225,175)
    ed.selected='GT:a1';ed.mode='edit';assert right_target(ed,225,175)=='GT:a1'
    assert frozen(s)==before

def test_P17_zero_area_verdict_independent_from_PPM(gui):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    from auto_annotation_tool.gui import mobile_bbox_spatial_review as spatial
    root,h,ed,s=gui;gid=SID+'-a00000001';oid='GT:'+gid
    iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);settled(root,h,ed)
    spatial.set_verdict(s,gid,oid,'ON_PLATE');raw=spatial.path(s).read_bytes()
    assert nav.select_layer(ed,'BBOX',oid);ready(root,h,ed)
    q=nav.quad(layers.model(ed));point=tuple(sum(p[i] for p in q)/4 for i in (0,1))
    ppm(gui,point)
    assert spatial.path(s).read_bytes()==raw and partial.progress(core.scene(s,gid))['object_decisions'][oid]=='NOT_PLATE'

def test_A06_outside_gesture_cancelled_by_context_change(gui):
    root,h,ed,s=gui;pick(gui);x,y=empty(ed);before=frozen(s)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y)
    assert nav.select_layer(ed,'POSE',IDS[0]);ready(root,h,ed)
    ed.c.event_generate('<ButtonRelease-1>',x=x,y=y);ready(root,h,ed)
    assert ed.selected==IDS[0] and frozen(s)==before

def test_A07_background_double_binding_after_pan_uses_release_guard(gui):
    root,h,ed,s=gui;pick(gui);before=frozen(s);x,y=empty(ed)
    event=NS(widget=ed.c,x=x,y=y,state=0)
    ed.double_press(event);ed.motion(NS(x=x+25,y=y));ed.release(NS(x=x+25,y=y));ready(root,h,ed)
    assert ed.selected
    ed.double_press(event);ed.release(event);ready(root,h,ed)
    assert not ed.selected and frozen(s)==before

def test_P24_P25_panel_matches_canonical_scorer_after_false_seed(gui):
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    from auto_annotation_tool.gui.mobile_canvas_info_badge import badge_rows
    root,h,ed,s=gui;partial.confirm_stack(s,G,oid=IDS[0]);stacks.edit_gt(s,G,IDS[0],text='WI6U1');settled(root,h,ed)
    point=pick(gui);ppm(gui,point);score=evaluate_unit(s,G)
    assert score['tp'] is not None
    assert [ed.contexts.counters[k].cget('text') for k in ('tp','fp','fn')]==[str(score[k]) for k in ('tp','fp','fn')]
    assert partial.progress(core.scene(s,G))['decisions']==0
    assert next(value for name,value in badge_rows(ed) if name=='GTZ')=='nie dotyczy — odrzucone MT'
