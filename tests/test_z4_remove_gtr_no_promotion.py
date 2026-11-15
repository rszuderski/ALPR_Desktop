"""Deleting operator geometry must never install inferred corners as GT."""
import copy
from types import SimpleNamespace as NS
import pytest
from test_z4_gtr_reapproval import verified,edit,pid,G,OID,MANUAL
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_mt_geometry_layers import snapshot as layers,create_reference
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

def test_remove_operator_geometry_leaves_only_source_layers_and_retains_GT_identity(tmp_path):
    s=verified(tmp_path);edit(s);stacks.edit_gt(s,G,OID,text='KEEPGT');p=pid(s)
    before=copy.deepcopy(stacks.snapshot(s,G));parents=copy.deepcopy(core.scene(s,G)['parents']);revision=s.review.review_revision
    core.remove_operator_geometry(s,G,OID);v=stacks.snapshot(s,G);scene=core.scene(s,G)
    assert v['plates'][p]['gt_quad'] is None
    assert v['plates'][p]['geometry_state']=='operator_removed'
    assert v['plates'][p]['gt_text']=='KEEPGT' and v['links']==before['links']
    assert scene['parents']==parents and scene['objects'][0]['quad'] is None
    assert s.review.review_revision==revision+1 and not scene['checked'] and partial.progress(scene)['decisions']==1
    assert [x['kind'] for x in layers(s,G,OID)['layers']]==['BBOX','POSE']

def test_space_text_save_and_reopen_do_not_resurrect_removed_geometry(tmp_path):
    s=verified(tmp_path);core.remove_operator_geometry(s,G,OID);p=pid(s);before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):partial.toggle_stack(s,G,oid=OID)
    assert s.sidecar_path.read_bytes()==before
    stacks.edit_gt(s,G,OID,text='NEWTEXT');frozen=s.sidecar_path.read_bytes()
    s=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert s.sidecar_path.read_bytes()==frozen and stacks.snapshot(s,G)['plates'][p]['gt_quad'] is None
    assert not partial.stack_is_verified(core.scene(s,G),oid=OID)
    assert layers(s,G,OID)['gtr_count']==0

def test_undo_redo_restores_exact_operator_geometry_not_source_geometry(tmp_path):
    s=verified(tmp_path);q=edit(s);p=pid(s);core.remove_operator_geometry(s,G,OID)
    assert stacks.restore_history(s,G) and stacks.snapshot(s,G)['plates'][p]['gt_quad']==q
    assert stacks.restore_history(s,G,redo=True) and stacks.snapshot(s,G)['plates'][p]['gt_quad'] is None
    assert core.scene(s,G)['parents'][0]['quad']!=q

@pytest.mark.parametrize('pose',[True,False])
def test_explicit_redrawing_completes_same_identity_and_needs_approval(tmp_path,pose):
    if pose:s=verified(tmp_path);q=edit(s)
    else:
        def invalid(rows,trace,payload):rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
        s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
        q=[[160,150],[280,150],[280,190],[160,190]];create_reference(s,G,OID,q)
    p=pid(s);stacks.edit_gt(s,G,OID,text='KEEPGT');core.remove_operator_geometry(s,G,OID)
    q=copy.deepcopy(q);q[0][0]+=2
    stacks.complete_bbox_reference(s,G,OID,q,expected_plate_id=p,operator_draw=True)
    assert pid(s)==p and stacks.snapshot(s,G)['plates'][p]['gt_quad']==q
    assert stacks.snapshot(s,G)['plates'][p]['gt_text']=='KEEPGT' and not core.scene(s,G)['checked']
    partial.confirm_stack(s,G,oid=OID);assert core.scene(s,G)['checked']

def test_removal_preserves_other_physical_GT_and_shared_reference_guard(tmp_path):
    s=verified(tmp_path,n=2);other=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid(s,'GT:a1')])
    core.remove_operator_geometry(s,G,OID)
    assert stacks.snapshot(s,G)['plates'][pid(s,'GT:a1')]==other
    assert partial.progress(core.scene(s,G))['object_decisions']['GT:a1']==partial.PLATE

def test_shared_reference_is_not_deleted_for_one_member(tmp_path):
    s=verified(tmp_path,n=1)
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path/'shared',n=2,mz=False);partial.confirm_stack(s,G,oid=OID);stacks.associate(s,G,'GT:a1',OID)
    before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):core.remove_operator_geometry(s,G,OID)
    assert s.sidecar_path.read_bytes()==before


def test_exports_do_not_replace_removed_reference_with_original_MT(tmp_path):
    import json,csv
    s=verified(tmp_path);edit(s);core.remove_operator_geometry(s,G,OID)
    paths=core.export(s,tmp_path/'export');data=json.loads(paths[0].read_text(encoding='utf-8'))
    row=next(r for r in data['rows'] if r['object_id']==OID)
    assert row['operator_quad'] is None and row['original_mt_quad']
    with paths[1].open(encoding='utf-8-sig',newline='') as f:row=next(r for r in csv.DictReader(f) if r['object_id']==OID)
    assert row['operator_quad']=='' and row['original_mt_quad']
    d=stacks.diagnostics(s)['rows'][0]
    assert d['gt_quad'] is None and d['polygon_iou_pct'] is None and d['not_measurable_reason']=='GT_GEOMETRY_REMOVED'


@pytest.fixture
def gui(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    s=verified(tmp_path/'data');edit(s);stacks.edit_gt(s,G,OID,text='KEEPGT')
    monkeypatch.setattr(native,'source_session',lambda _:s);cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    # R is hover-scoped; external desktop cursor motion must not decide this
    # geometry test. Pointer-routing guards are covered separately.
    q=stacks.snapshot(s,G)['plates'][pid(s)]['gt_quad']
    monkeypatch.setattr(ed,'_canvas_hotkey_pointer',lambda event:ed.to_canvas(tuple(sum(p[i] for p in q)/4 for i in (0,1))))
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass


def removed_view(root,h,ed,s):
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_geometry_layers as visual
    ready(root,h,ed)
    assert layers(s,G,OID)['gtr_count']==0 and visual.model(ed)['active'] in ('BBOX','POSE')
    assert not ed.c.find_withtag('operator_quad') and not ed.c.find_withtag('reference_identity_text')
    assert not ed.c.find_withtag('corner_handle') and 'GTR · D' in ed.clean_gtr_confirm_button['text']
    assert 'Obrys operatora: brak referencji' in ed.navigator.hint['text']


@pytest.mark.parametrize('focus',['ordinary','reserved'])
def test_native_R_never_promotes_MT_and_history_preserves_operator_quad(gui,focus):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_navigator as nav
    root,h,ed,s=gui;q=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid(s)]['gt_quad']);nav.select_layer(ed,'GTR',OID);ready(root,h,ed)
    if focus=='reserved':native.key(h,'d');ready(root,h,ed)
    native.key(h,'r');removed_view(root,h,ed,s);before=s.sidecar_path.read_bytes()
    for kind in ('POSE','BBOX'):
        nav.select_layer(ed,kind);ready(root,h,ed);native.key(h,'space');removed_view(root,h,ed,s)
        native.key(h,'r');assert s.sidecar_path.read_bytes()==before
    ed.undo();ready(root,h,ed);assert stacks.snapshot(s,G)['plates'][pid(s)]['gt_quad']==q
    ed.undo(redo=True);removed_view(root,h,ed,s)


@pytest.mark.parametrize('start',['key','button'])
def test_native_D_redraws_same_GTR_from_source_without_changing_MT(gui,start):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_navigator as nav
    root,h,ed,s=gui;p=pid(s);parents=copy.deepcopy(core.scene(s,G)['parents']);q=copy.deepcopy(stacks.snapshot(s,G)['plates'][p]['gt_quad'])
    nav.select_layer(ed,'GTR',OID);ready(root,h,ed);native.key(h,'r');removed_view(root,h,ed,s)
    if start=='key':native.key(h,'d')
    else:ed.clean_gtr_confirm_button.invoke()
    ready(root,h,ed);assert ed.mode=='draw' and ed.draw_target['kind']=='complete'
    q[0][0]+=3
    for point in q:native.click(h,ed,point)
    ready(root,h,ed);state=stacks.snapshot(s,G)
    assert pid(s)==p and state['plates'][p]['gt_quad'] and state['plates'][p]['gt_text']=='KEEPGT'
    assert core.scene(s,G)['parents']==parents and not ed.checked_var.get()
    native.key(h,'space');ready(root,h,ed);assert ed.checked_var.get()
