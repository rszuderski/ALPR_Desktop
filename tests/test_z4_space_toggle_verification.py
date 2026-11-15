"""Space toggles one physical stack; data and other stacks are retained."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_gtr_reapproval import verified,edit,pid,MANUAL,G,OID
from test_z4_mt_geometry_layers import source_session
from test_z4_p0_physical_gt import session_fixture
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_approval_visuals as colors

def scene(s):return core.scene(s,G)
def stable(s):
    state=stacks.snapshot(s,G)
    return copy.deepcopy(dict(parents=scene(s)['parents'],links=state['links'],
        plates={p:{k:v[k] for k in ('gt_quad','gt_text','text_confirmed','text_exclusion_reason','origin_object_id')} for p,v in state['plates'].items()}))
def checked(s):return partial.progress(scene(s))['complete']

def test_stack_off_on_is_atomic_and_retains_GT_text_geometry_links(tmp_path):
    s=verified(tmp_path);partial.confirm_stack(s,G,oid=OID);stacks.edit_gt(s,G,OID,text='KEEPGT')
    before=stable(s);revision=s.review.review_revision;p=pid(s)
    for i,expected in enumerate((False,True,False,True),1):
        assert partial.toggle_stack(s,G,plate_id=p)
        assert checked(s) is expected and partial.stack_is_verified(scene(s),plate_id=p) is expected
        assert s.review.review_revision==revision+i and stable(s)==before
        assert (p in scene(s)[partial.KEY]['pending_gtr_ids']) is not expected

def test_shared_stack_switches_members_but_not_other_GT(tmp_path):
    s=source_session(tmp_path,n=3,mz=False);partial.confirm_stack(s,G,oid=OID);stacks.associate(s,G,'GT:a1',OID)
    partial.confirm_stack(s,G,oid=OID);partial.confirm_stack(s,G,oid='GT:a2');before=stable(s)
    partial.toggle_stack(s,G,oid='GT:a1');v=partial.progress(scene(s))
    assert v['object_decisions']=={OID:partial.UNRESOLVED,'GT:a1':partial.UNRESOLVED,'GT:a2':partial.PLATE}
    assert len(v['pending_gtr_ids'])==1 and stable(s)==before
    partial.toggle_stack(s,G,oid=OID);assert checked(s) and stable(s)==before

def test_another_pending_GT_is_not_reapproved(tmp_path):
    s=verified(tmp_path,n=2);edit(s,'GT:a1');other=pid(s,'GT:a1')
    partial.toggle_stack(s,G,plate_id=pid(s));partial.toggle_stack(s,G,plate_id=pid(s))
    assert not checked(s) and partial.progress(scene(s))['pending_gtr_ids']==[other]

def test_standalone_manual_GTR_toggles_without_creating_MT(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);oid=core.add_plate(s,G,MANUAL,partial_review=True);p=pid(s,oid)
    before=stable(s)
    for expected in (True,False,True):
        partial.toggle_stack(s,G,plate_id=p);assert checked(s) is expected and stable(s)==before
        assert not partial.progress(scene(s))['object_decisions']
    assert not core._object(scene(s),oid)['error']

def test_BBOX_without_Pose_never_acquires_fake_corners(tmp_path):
    def invalid(rows,trace,payload):rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    partial.toggle_stack(s,G,oid=OID);p=pid(s);before=stable(s)
    for expected in (False,True):
        partial.toggle_stack(s,G,oid=OID);assert checked(s) is expected and stable(s)==before
        assert stacks.snapshot(s,G)['plates'][p]['gt_quad'] is None
        assert not partial.progress(scene(s))['pending_gtr_ids']

def test_unlinked_BAD_is_preserved_and_explicit_source_can_restore_it(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,G,oid=OID);stacks.associate(s,G,'GT:a1',OID)
    stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True);bad=copy.deepcopy(stacks.snapshot(s,G)['links']['GT:a1'])
    partial.toggle_stack(s,G,plate_id=pid(s));partial.toggle_stack(s,G,plate_id=pid(s))
    assert stacks.snapshot(s,G)['links']['GT:a1']==bad and partial.progress(scene(s))['object_decisions']['GT:a1']==partial.NOT_PLATE
    partial.toggle_stack(s,G,oid='GT:a1');assert partial.progress(scene(s))['object_decisions']['GT:a1']==partial.PLATE

def test_history_and_reopen_restore_each_toggle_without_migration(tmp_path):
    s=verified(tmp_path);partial.confirm_stack(s,G,oid=OID);before=stable(s)
    partial.toggle_stack(s,G,oid=OID);assert not checked(s);revision=s.review.review_revision;frozen=s.sidecar_path.read_bytes()
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert not partial.stack_is_verified(scene(reopened),oid=OID) and reopened.review.review_revision==revision and s.sidecar_path.read_bytes()==frozen
    assert stacks.restore_history(reopened,G) and checked(reopened) and stable(reopened)==before
    assert stacks.restore_history(reopened,G,redo=True) and not checked(reopened) and stable(reopened)==before

def test_RAW_unknown_target_and_removed_reference_are_rejected_without_write(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=2);before=s.sidecar_path.read_bytes()
    for args in ({'oid':'GT:a2'},{'oid':'GT:missing'},{'plate_id':'PGT:missing'},{'oid':'GT:a0','plate_id':'PGT:missing'}):
        with pytest.raises(ValueError):partial.toggle_stack(s,G,**args)
        assert s.sidecar_path.read_bytes()==before

@pytest.fixture
def gui(tmp_path,monkeypatch):
    s=verified(tmp_path/'data');partial.confirm_stack(s,G,oid=OID)
    monkeypatch.setattr(native,'source_session',lambda _:s);cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_space_switches_verification_each_press_on_every_layer(gui,kind):
    root,h,ed,s=gui;nav.select_layer(ed,kind);ready(root,h,ed);before=stable(s);revision=s.review.review_revision
    for i,expected in enumerate((False,True,False,True),1):
        native.key(h,'space');ready(root,h,ed)
        assert checked(s) is expected and ed.checked_var.get() is expected
        assert layers.model(ed)['active']==kind and stable(s)==before and s.review.review_revision==revision+i
        assert ed.clean_gtr_confirm_button.winfo_manager()
        assert ('Cofnij' in ed.clean_gtr_confirm_button['text']) is expected
        item=next(i for i in ed.c.find_withtag('operator_quad') if pid(s) in ed.c.gettags(i))
        assert ed.c.itemcget(item,'outline')==(colors.GREEN if expected else colors.PENDING)

@pytest.mark.parametrize('focus',['canvas','navigator','image_list','button'])
def test_held_space_repeats_do_not_toggle_until_release(gui,focus):
    root,h,ed,s=gui;widget={'canvas':h.canvas,'navigator':ed.navigator.layer,'image_list':h.subject_tree,'button':ed.clean_gtr_confirm_button}[focus]
    widget.focus_force();native.pump(root,lambda:root.focus_get() is widget);before=s.review.review_revision
    for _ in range(8):widget.event_generate('<KeyPress-space>');root.update()
    ready(root,h,ed);assert not checked(s) and s.review.review_revision==before+1
    widget.event_generate('<KeyRelease-space>');widget.event_generate('<KeyPress-space>');widget.event_generate('<KeyRelease-space>')
    ready(root,h,ed);assert checked(s) and s.review.review_revision==before+2

def test_button_and_navigation_focus_use_identical_toggle(gui):
    root,h,ed,s=gui
    for expected in (False,True):ed.clean_gtr_confirm_button.invoke();ready(root,h,ed);assert checked(s) is expected
    ed.navigator.layer.focus_force();native.pump(root,lambda:root.focus_get() is ed.navigator.layer)
    ed.navigator.layer.event_generate('<KeyPress-space>');ed.navigator.layer.event_generate('<KeyRelease-space>');ready(root,h,ed);assert not checked(s)
    h.subject_tree.focus_force();native.pump(root,lambda:root.focus_get() is h.subject_tree)
    h.subject_tree.event_generate('<KeyPress-space>');h.subject_tree.event_generate('<KeyRelease-space>');ready(root,h,ed);assert checked(s)

def test_typing_ctrl_space_draft_and_other_window_cannot_toggle(gui):
    import tkinter as tk
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes()
    assert ed.confirm(NS(widget=h.canvas,keysym='space',state=4)) is None
    ed.start_entry(OID);native.pump(root)
    assert ed.confirm(NS(widget=ed.entry,keysym='space',state=0)) is None and s.sidecar_path.read_bytes()==before
    ed.escape();native.key(h,'d');native.key(h,'space');ready(root,h,ed)
    assert ed.mode=='draw' and s.sidecar_path.read_bytes()==before;ed.escape()
    other=tk.Toplevel(root)
    try:assert ed.confirm(NS(widget=other,keysym='space',state=0)) is None and s.sidecar_path.read_bytes()==before
    finally:other.destroy()

def test_hover_after_escape_toggles_without_reselecting(gui,monkeypatch):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);q=next(iter(stacks.snapshot(s,G)['plates'].values()))['gt_quad']
    xy=ed.to_canvas(tuple(sum(p[i] for p in q)/4 for i in (0,1)));monkeypatch.setattr(ed,'_canvas_hotkey_pointer',lambda event:xy)
    for expected in (False,True):native.key(h,'space');ready(root,h,ed);assert checked(s) is expected and not ed.selected


@pytest.fixture
def refreshing(gui,monkeypatch):
    import threading
    root,h,ed,s=gui;entered=threading.Event();release=threading.Event();original=s.statistics
    def slow():
        entered.set();assert release.wait(15),'Test did not release the statistics worker'
        return original()
    monkeypatch.setattr(s,'statistics',slow);ed.refresh_results()
    native.pump(root,lambda:entered.is_set() and h._saving and ed.in_refresh)
    try:yield root,h,ed,s,release
    finally:
        release.set();native.pump(root,lambda:not h._saving and not ed.in_refresh and not getattr(ed,'_pending_frame_toggles',[]))


@pytest.mark.parametrize('presses',[1,2,3])
def test_each_press_during_real_statistics_refresh_is_applied_once(refreshing,presses):
    root,h,ed,s,release=refreshing;before=s.sidecar_path.read_bytes();revision=s.review.review_revision;content=stable(s)
    for _ in range(presses):native.key(h,'space')
    assert s.sidecar_path.read_bytes()==before and len(ed._pending_frame_toggles)==presses
    release.set();native.pump(root,lambda:s.review.review_revision==revision+presses);ready(root,h,ed)
    assert checked(s) is (presses%2==0) and stable(s)==content and not ed._pending_frame_toggles


def test_refresh_queue_ignores_autorepeat_and_retains_explicit_target(refreshing):
    root,h,ed,s,release=refreshing;revision=s.review.review_revision
    native.focus_canvas(h)
    for _ in range(8):h.canvas.event_generate('<KeyPress-space>');root.update()
    assert len(ed._pending_frame_toggles)==1
    h.canvas.event_generate('<KeyRelease-space>');ed.escape()
    release.set();native.pump(root,lambda:s.review.review_revision==revision+1);ready(root,h,ed)
    assert not checked(s) and not ed.selected


@pytest.mark.parametrize('field',['session','current_invocation','_image_entry'])
def test_queued_toggle_cannot_cross_source_context(gui,field):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();original=getattr(h,field)
    ed._queue_frame_verification(dict(oid=OID,plate_id=None));h.canvas.after_cancel(ed._frame_toggle_after)
    try:
        setattr(h,field,object() if field=='session' else 'different-source')
        ed._drain_frame_verification()
        assert s.sidecar_path.read_bytes()==before and not ed._pending_frame_toggles
    finally:setattr(h,field,original)
