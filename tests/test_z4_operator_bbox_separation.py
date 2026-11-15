"""Operational Canvas inventory is separate from formal BBOX scoring."""
from pathlib import Path
import copy
import pytest
import test_z4_image09_feedback as case
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_gtr_reapproval import verified,MANUAL
from test_z4_mt_geometry_layers import source_session
from test_z4_p0_physical_gt import session_fixture
from auto_annotation_tool.gui import mobile_canvas_review_feedback as feedback,mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers,mobile_canvas_info_badge as badge
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_detection_evaluation as scoring
OUT=Path(__file__).resolve().parents[1]/'output/Z4_OPERATOR_VS_BBOX_20261030_001'
G=case.G

@pytest.fixture
def real_gui(tmp_path,monkeypatch):
    monkeypatch.setattr(case,'OUT',OUT)
    yield from case.gui.__wrapped__(tmp_path,monkeypatch)

def test_image09_main_counts_describe_operator_objects_not_matching_errors(real_gui):
    root,h,ed,s=real_gui;before=s.sidecar_path.read_bytes();value=clean.context(ed,{})
    assert value['image_counts']==dict(gt='1',mt='1',rejected='0',unlinked='0')
    assert value['bbox_counts']==dict(tp='0',fp='1',fn='1')
    assert 'Tablica wykryta' in value['bbox_note'] and 'nie spełnia kryterium lokalizacji' in value['bbox_note']
    assert s.sidecar_path.read_bytes()==before

def inventory(s,g='mt-one'):
    scene=core.scene(s,g);return feedback.operator_inventory(scene,stacks.snapshot(s,g))

def test_real_rejection_corresponds_to_PPM_cross_and_undo(real_gui):
    root,h,ed,s=real_gui;oid='GT:'+G;original=s.sidecar_path.read_bytes()
    stacks.toggle_unassociated(s,G,oid,partial_review=True);ready(root,h,ed)
    assert clean.context(ed,{})['image_counts']['rejected']=='1'
    assert ed.c.find_withtag('fp_diagonal')
    ed.show_fp_var.set(False);ed.overlay();assert not ed.c.find_withtag('fp_diagonal')
    assert clean.context(ed,{})['image_counts']['rejected']=='1'
    frozen=s.sidecar_path.read_bytes();ed.contexts.counters['rejected'].event_generate('<Button-1>');ready(root,h,ed)
    assert ed.show_fp_var.get() and ed.c.find_withtag('fp_diagonal') and s.sidecar_path.read_bytes()==frozen
    ed.contexts.counters['unlinked'].event_generate('<Button-1>');ready(root,h,ed)
    row=clean.selected(ed);assert row['gt'] and row['pid'] and row['candidate'] is None
    assert s.sidecar_path.read_bytes()==frozen and inventory(s,G)['counts']['rejected']==1
    ed.undo();ready(root,h,ed);assert clean.context(ed,{})['image_counts']['rejected']=='0'

def test_manual_GT_without_source_link_is_not_merged_with_detected_GT(tmp_path):
    s=verified(tmp_path);oid=core.add_plate(s,'mt-one',MANUAL,partial_review=True)
    state=stacks.snapshot(s,'mt-one');pid=next(pid for pid,p in state['plates'].items() if p['origin_object_id']==oid)
    partial.confirm_stack(s,'mt-one',plate_id=pid)
    before=s.sidecar_path.read_bytes();v=inventory(s)
    assert v['counts']==dict(gt=2,mt=1,rejected=0,unlinked=1)
    assert v['unlinked'][0]['gt_id']==pid and s.sidecar_path.read_bytes()==before

def test_unreviewed_source_is_not_synthetic_GT_or_TP(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);before=s.sidecar_path.read_bytes();v=inventory(s)
    assert v['counts']==dict(gt=0,mt=1,rejected=0,unlinked=0)
    assert scoring.evaluate_unit(s,'mt-one')['tp'] is None and s.sidecar_path.read_bytes()==before

def test_empty_inventory_does_not_auto_approve_negative_image(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);before=s.sidecar_path.read_bytes()
    assert inventory(s)['counts']==dict(gt=0,mt=0,rejected=0,unlinked=0)
    assert not partial.progress(core.scene(s,'mt-one'))['complete']
    assert scoring.evaluate_unit(s,'mt-one')['tp'] is None and s.sidecar_path.read_bytes()==before

def test_two_FINALs_same_GT_have_no_fake_operator_rejection(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,'mt-one',oid='GT:a0')
    stacks.associate(s,'mt-one','GT:a1','GT:a0');partial.confirm_stack(s,'mt-one',oid='GT:a0')
    assert inventory(s)['counts']==dict(gt=1,mt=2,rejected=0,unlinked=0)
    r=scoring.evaluate_unit(s,'mt-one');assert (r['tp'],r['fp'],r['fn'])==(1,1,0)

def test_RAW_never_enters_visible_FINAL_or_rejection_count(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=2);v=inventory(s)
    assert v['counts']['mt']==2 and all(i['oid']!='GT:a2' for i in v['rejected'])

def test_unknown_source_and_exclusion_cannot_report_success(tmp_path):
    s=verified(tmp_path);scene=core.scene(s,'mt-one');state=stacks.snapshot(s,'mt-one')
    for altered in (dict(scene,_post_nms_unknown=True),dict(scene,exclusion_reason='test exclusion')):
        assert all(v is None for v in feedback.operator_inventory(altered,state)['counts'].values())

def test_saved_GT_and_links_survive_undo_of_approval(real_gui):
    root,h,ed,s=real_gui;before=copy.deepcopy(stacks.snapshot(s,G)['plates']);ed.undo();ready(root,h,ed)
    value=clean.context(ed,{})
    assert value['image_counts']==dict(gt='1',mt='1',rejected='0',unlinked='0')
    assert all(v=='—' for v in value['bbox_counts'].values()) and '1 tablica do weryfikacji' in value['image_note']
    after=stacks.snapshot(s,G)['plates'];assert set(after)==set(before)
    assert all(after[pid][k]==p[k] for pid,p in before.items() for k in ('gt_quad','gt_text'))

def test_formal_buttons_locate_BBOX_and_GTR_without_writes(real_gui):
    root,h,ed,s=real_gui;before=s.sidecar_path.read_bytes()
    for kind,layer in (('fp','BBOX'),('fn','GTR')):
        ed.contexts.bbox_counters[kind].invoke();ready(root,h,ed)
        assert layers.model(ed)['active']==layer and s.sidecar_path.read_bytes()==before
    assert 'FP' not in badge.badge_rows(ed)[1][1]
    assert 'zatwierdzone' in badge.badge_rows(ed)[1][1]

def test_operator_cards_and_menu_are_explicitly_named(real_gui):
    root,h,ed,s=real_gui
    assert ed.contexts.counter_labels['rejected']['text']=='Odrzucone przez operatora'
    assert ed.contexts.counter_labels['unlinked']['text']=='GT bez powiązania MT'
    assert 'Ocena lokalizacji BBOX' in ed.contexts.bbox_title['text']
    labels=[ed.clean_menu.entrycget(i,'label') for i in range(ed.clean_menu.index('end')+1) if ed.clean_menu.type(i)!='separator']
    assert 'Pokaż odrzucone wskazania (PPM)' in labels
    assert not any('(FP)' in ed.contexts.counter_labels[k]['text'] or '(FN)' in ed.contexts.counter_labels[k]['text'] for k in ('gt','mt','rejected','unlinked'))

def test_inventory_and_formal_results_ignore_REF_INF_filters(real_gui):
    root,h,ed,s=real_gui;before=clean.context(ed,{})
    for ref,inf in ((False,False),(True,False),(False,True)):
        ed.ref_visible.set(ref);ed.inf_visible.set(inf);ed.show_fp_var.set(False);ed.overlay();after=clean.context(ed,{})
        assert after['image_counts']==before['image_counts'] and after['bbox_counts']==before['bbox_counts']

def test_scientific_JSON_CSV_and_denominators_do_not_follow_operator_cards(real_gui,tmp_path):
    root,h,ed,s=real_gui;before=s.sidecar_path.read_bytes();initial=scoring.evaluate_session(s)
    exported_before=scoring.export(s,tmp_path/'export_before')
    for kind in ('fp','fn'):feedback.inspect(ed,kind);ready(root,h,ed)
    exported_after=scoring.export(s,tmp_path/'export_after')
    assert {p.name:p.read_bytes() for p in exported_before}=={p.name:p.read_bytes() for p in exported_after}
    assert scoring.evaluate_session(s)==initial and s.sidecar_path.read_bytes()==before
    assert '1 GT' in ed.contexts.bbox_scope['text']

def test_true_missing_GT_counter_locates_saved_outline_without_writing(real_gui):
    root,h,ed,s=real_gui;oid=core.add_plate(s,G,[[40,50],[140,50],[140,90],[40,90]],partial_review=True)
    pid=next(pid for pid,p in stacks.snapshot(s,G)['plates'].items() if p['origin_object_id']==oid)
    partial.confirm_stack(s,G,plate_id=pid);ready(root,h,ed);before=s.sidecar_path.read_bytes()
    assert clean.context(ed,{})['image_counts']==dict(gt='2',mt='1',rejected='0',unlinked='1')
    ed.contexts.counters['unlinked'].event_generate('<Button-1>');ready(root,h,ed)
    assert clean.selected(ed)['pid']==pid and s.sidecar_path.read_bytes()==before
