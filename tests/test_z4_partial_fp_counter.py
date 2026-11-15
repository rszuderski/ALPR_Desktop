"""Known operator rejections must survive an unfinished image's UI counters."""
from pathlib import Path
import copy,hashlib,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import gui as previous_gui,settled
from auto_annotation_tool.ranking import mobile_human_review as review,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_mt_detection_evaluation as mt,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback,mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav

OUT=Path(__file__).resolve().parents[1]/'output/Z4_PARTIAL_FP_COUNTER_20261101_001'
SID='exp-1792699864138-add3e282';G=SID+'-a00000008';BAD='GT:'+SID+'-a00000009';GOOD='GT:'+G
def frozen(s):return hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest(),copy.deepcopy(mt.evaluate_session(s))

@pytest.fixture
def actual(tmp_path):
    src=OUT/'private'/SID
    shutil.copyfile(src/'source.alprsession',tmp_path/'source.alprsession');shutil.copyfile(src/'review.json',tmp_path/'review.json')
    return review.MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

@pytest.fixture
def gui(tmp_path,monkeypatch,actual):
    yield from previous_gui.__wrapped__(tmp_path,monkeypatch,actual)

def bound(s,gid=G):
    return feedback.partial_fp(core.scene(s,gid),stacks.snapshot(s,gid),s,mt.evaluate_unit(s,gid))

def test_real_partial_FP_and_zero_saved_GT_are_visible(gui):
    root,h,ed,s=gui;before=frozen(s)
    assert partial.progress(core.scene(s,G))['false_positive']==1
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==['0','—','≥1','—']
    assert 'Ocena częściowa' in ed.contexts.fields['image'].cget('text')
    assert 'MT 2' in ed.contexts.fields['image'].cget('text')
    assert clean.context(ed,{})['bbox_counts']==dict(tp='—',fp='—',fn='—')
    assert mt.evaluate_unit(s,G)['fp'] is None and mt.evaluate_unit(s,G)['n_qualified']==0
    ed.escape();ready(root,h,ed);assert ed.contexts.counters['fp'].cget('text')=='≥1'
    for ref,inf in ((False,True),(True,False),(False,False),(True,True)):
        ed.ref_visible.set(ref);ed.inf_visible.set(inf);ed.show_fp_var.set(False);ed.sync();ready(root,h,ed)
        assert ed.contexts.counters['fp'].cget('text')=='≥1'
    ed.contexts.counters['fp'].event_generate('<Button-1>');ready(root,h,ed)
    assert ed.selected==BAD and ed.show_fp_var.get() and ed.inf_visible.get()
    assert frozen(s)==before

def test_PPM_undo_redo_updates_partial_count_without_GT(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'BBOX',BAD);ready(root,h,ed)
    from auto_annotation_tool.gui.mobile_canvas_geometry_layers import model
    q=nav.quad(model(ed));point=tuple(sum(p[i] for p in q)/4 for i in (0,1))
    native.click(h,ed,point,button=3);settled(root,h,ed)
    assert ed.contexts.counters['fp'].cget('text')=='—' and not stacks.snapshot(s,G)['plates']
    ed.undo();settled(root,h,ed);assert ed.contexts.counters['fp'].cget('text')=='≥1'
    ed.undo(redo=True);settled(root,h,ed);assert ed.contexts.counters['fp'].cget('text')=='—'

def test_completed_image_switches_to_exact_canonical_counts(gui):
    root,h,ed,s=gui;partial.confirm_stack(s,G,oid=GOOD);settled(root,h,ed);value=mt.evaluate_unit(s,G)
    assert value['status']=='MEASURED'
    assert [ed.contexts.counters[k].cget('text') for k in ('tp','fp','fn')]==[str(value[k]) for k in ('tp','fp','fn')]
    assert 'Ocena częściowa' not in ed.contexts.fields['image'].cget('text')
    ed.undo();settled(root,h,ed);assert ed.contexts.counters['fp'].cget('text')=='≥1'

@pytest.mark.parametrize('box,confidence,expected',[
    ((150,140,300,210),.9,1),((150,140,300,210),0,1),
    ((150,170,300,170),.9,None),((150,140,300,210),'nan',None),
    ((-10,140,300,210),.9,None),((150,140,300,210),'',None)])
def test_invalid_or_missing_sources_do_not_invent_numeric_FP(tmp_path,box,confidence,expected):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def change(rows,trace,payload):
        for i,r in enumerate(rows):
            r.update(zip(('plate_left','plate_top','plate_right','plate_bottom'),box if i==0 else (370,140,520,210)))
            r['plate_confidence']=confidence;r['mt_confidence']=confidence
    s=fixture_archive(tmp_path,n=2,mz=False,mutate=change);s.set_review_mode('assisted')
    stacks.toggle_unassociated(s,'mt-one','GT:a0',partial_review=True);before=frozen(s)
    data=bound(s,'mt-one');assert data['lower_bound']==expected
    assert 'MT 1' in data['note'] and mt.evaluate_unit(s,'mt-one')['fp'] is None
    assert frozen(s)==before

def test_two_rejections_count_once_each_and_RAW_never_counts(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path,n=4,suppressed=3)
    for aid in ('a0','a1'):stacks.toggle_unassociated(s,'mt-one','GT:'+aid,partial_review=True)
    # This fixture's positive source metadata is inspected separately; its
    # legacy missing BBOX must not become a fabricated partial number.
    scene=core.scene(s,'mt-one');state=stacks.snapshot(s,'mt-one')
    inv=feedback.operator_inventory(scene,state);assert len(inv['rejected'])==2
    assert all(i['oid']!='GT:a3' for i in inv['rejected'])
    if bound(s,'mt-one')['lower_bound'] is not None:assert bound(s,'mt-one')['lower_bound']==2

def test_multiple_valid_rejections_have_numeric_lower_bound_and_undo(tmp_path):
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path,n=3,mz=False)
    for oid in ('GT:a0','GT:a1'):stacks.toggle_unassociated(s,'mt-one',oid,partial_review=True)
    assert bound(s,'mt-one')['lower_bound']==2 and len(bound(s,'mt-one')['items'])==2
    assert mt.evaluate_unit(s,'mt-one')['fp'] is None
    stacks.restore_history(s,'mt-one');assert bound(s,'mt-one')['lower_bound']==1

def test_spatial_zero_area_verdict_is_not_an_operator_MT_rejection(tmp_path):
    from test_z4_bbox_spatial_marker import source
    from auto_annotation_tool.gui import mobile_bbox_spatial_review as spatial
    s=source(tmp_path);spatial.set_verdict(s,'mt-one','GT:a0','OFF_PLATES_BAD')
    assert bound(s,'mt-one')['lower_bound'] is None and not bound(s,'mt-one')['items']

def test_exclusion_and_source_uncertainty_block_partial_numbers(actual):
    s=actual;scene=core.scene(s,G);state=stacks.snapshot(s,G);result=mt.evaluate_unit(s,G)
    for altered in (dict(scene,exclusion_reason='QA'),dict(scene,_post_nms_unknown=True)):
        assert feedback.partial_fp(altered,state,s,result)['lower_bound'] is None
    result=dict(result,status='NOT_EVALUABLE',unavailable_reasons=['SOURCE_NOT_VERIFIED'])
    value=feedback.partial_fp(scene,state,s,result)
    assert value['lower_bound'] is None and 'Odrzucone' in value['note']

def test_export_keeps_partial_unit_unqualified(actual,tmp_path):
    s=actual;before=frozen(s);paths=mt.export(s,tmp_path/'export')
    p=next(p for p in paths if p.name=='mt_detection_evaluation.json');data=json.loads(p.read_text(encoding='utf-8'))
    unit=next(r for r in data['units'] if r['mt_invocation_id']==G)
    assert unit['fp'] is None and unit['n_qualified']==0 and frozen(s)==before
